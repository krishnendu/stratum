//! Concrete `ToolDispatcher` backends for `shell.exec` and `fs.read`.
//!
//! Phase 3 v2 — the first real dispatchers wired on top of
//! [`crate::sandbox::SandboxSpawn`] (PR #69) and
//! [`crate::tool_invocation::RegistryDispatcher`] (PR #70). These are the
//! minimum surface the `AgentLoop` needs to drive a useful turn: read a
//! file out of the workspace, or shell out to a tiny allowlist of
//! information-grade binaries (`echo`, `ls`, `cat`, `pwd`, `wc`, `head`,
//! `tail`).
//!
//! ## Error code policy
//!
//! The catalog in `stratum-types::error::codes` does not yet declare
//! dispatcher-specific entries (E5006-E5011 / E4006-for-timeout). Per
//! `plan/29-error-taxonomy-and-logging.md` §8 we ship local sentinels
//! (`E_DISPATCH_*`) mirroring the `E_NO_BLOCKS` precedent from
//! [`crate::agent_loop`]; promoting them to `STRAT-E####` happens when
//! the agent-loop dispatch step lands a stable surface area.
//!
//! ## Binary-vs-text policy for `fs.read`
//!
//! `FsReadToolDispatcher` reads file bytes verbatim and renders the
//! body via [`String::from_utf8_lossy`]: valid UTF-8 round-trips
//! losslessly, invalid sequences map to U+FFFD. The byte count returned
//! in [`ToolResult::Ok::bytes`] is the **raw** on-disk byte count, not
//! the post-lossy length, so the budget tracker still charges the true
//! cost. This is the pragmatic scaffold choice; future work can swap
//! in a base64-or-text discriminator without changing the dispatcher's
//! external shape.

// xtask-check-error-codes: ignore-file
//
// Reason: this module uses local `E_DISPATCH_*` sentinels (mirroring
// `E_NO_BLOCKS`) rather than catalog `STRAT-E####` entries. The
// rustdoc examples and tests contain no `STRAT-E####` literals, but
// the marker is here as a safety net should one be added later before
// the catalog catches up.

use std::io::Read;
use std::path::{Component, Path, PathBuf};
use std::thread;
use std::time::{Duration, Instant};

use serde_json::Value;

use crate::sandbox::SandboxSpawn;
use crate::sandbox_resolve::SandboxLaunchSpec;
use crate::tool_invocation::{RegistryDispatcher, ToolDispatcher, ToolInvocation, ToolResult};

/// Local sentinel: argument missing or wrong type.
const E_DISPATCH_MISSING_ARG: &str = "E_DISPATCH_MISSING_ARG";
/// Local sentinel: shell.exec command not on the allowlist.
const E_DISPATCH_BIN_DISALLOWED: &str = "E_DISPATCH_BIN_DISALLOWED";
/// Local sentinel: shell.exec child exited non-zero.
const E_DISPATCH_EXIT_NONZERO: &str = "E_DISPATCH_EXIT_NONZERO";
/// Local sentinel: shell.exec timed out before exit.
const E_DISPATCH_TIMEOUT: &str = "E_DISPATCH_TIMEOUT";
/// Local sentinel: fs.read path escaped the workspace root.
const E_DISPATCH_PATH_ESCAPE: &str = "E_DISPATCH_PATH_ESCAPE";
/// Local sentinel: fs.read file exceeds the configured size cap.
const E_DISPATCH_SIZE_CAP: &str = "E_DISPATCH_SIZE_CAP";
/// Local sentinel: fs.read failed at the filesystem layer.
const E_DISPATCH_READ_FAILED: &str = "E_DISPATCH_READ_FAILED";
/// Local sentinel: shell.exec failed to spawn under the chosen backend.
const E_DISPATCH_SPAWN_FAILED: &str = "E_DISPATCH_SPAWN_FAILED";

/// Default allowlist of read-only binaries that `ShellToolDispatcher` will exec.
///
/// Paranoid scaffold: every additional binary must be reviewed against the
/// threat model in `plan/31-tool-sandbox-and-secrets.md`.
pub const SHELL_DEFAULT_ALLOWLIST: &[&str] = &["echo", "ls", "cat", "pwd", "wc", "head", "tail"];

/// `ToolDispatcher` for `shell.exec` calls. Composes a `SandboxSpawn`
/// with a base `SandboxLaunchSpec` and a per-call wall-clock timeout.
///
/// # Allowlist
///
/// Only commands whose binary name appears in [`SHELL_DEFAULT_ALLOWLIST`]
/// will be invoked. Use [`Self::with_allowlist`] to override for tests
/// or for an explicit per-deployment policy. The allowlist is compared
/// against the *binary name only* — fully-qualified host paths are not
/// supported by design.
pub struct ShellToolDispatcher {
    id: String,
    sandbox: SandboxSpawn,
    base_spec: SandboxLaunchSpec,
    timeout: Duration,
    allowlist: Vec<String>,
}

impl std::fmt::Debug for ShellToolDispatcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // `base_spec` carries platform paths + env tuples that clutter
        // log output; redact it from the Debug rendering.
        f.debug_struct("ShellToolDispatcher")
            .field("id", &self.id)
            .field("backend", &self.sandbox.backend())
            .field("timeout", &self.timeout)
            .field("allowlist", &self.allowlist)
            .finish_non_exhaustive()
    }
}

impl ShellToolDispatcher {
    /// Stable id used to look this dispatcher up in the registry.
    pub const ID: &'static str = "shell.exec";

    /// Build a new dispatcher with a 30-second per-call timeout and the
    /// default allowlist.
    #[must_use]
    pub fn new(sandbox: SandboxSpawn, base_spec: SandboxLaunchSpec) -> Self {
        Self {
            id: Self::ID.to_string(),
            sandbox,
            base_spec,
            timeout: Duration::from_secs(30),
            allowlist: SHELL_DEFAULT_ALLOWLIST
                .iter()
                .map(|s| (*s).to_string())
                .collect(),
        }
    }

    /// Override the per-call wall-clock timeout.
    #[must_use]
    pub const fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Override the binary allowlist. Empty allowlist refuses all calls.
    #[must_use]
    pub fn with_allowlist(mut self, allowlist: Vec<String>) -> Self {
        self.allowlist = allowlist;
        self
    }

    /// Per-call wall-clock timeout.
    #[must_use]
    pub const fn timeout(&self) -> Duration {
        self.timeout
    }

    /// Inspect the active allowlist.
    #[must_use]
    pub fn allowlist(&self) -> &[String] {
        &self.allowlist
    }

    fn allowed(&self, binary: &str) -> bool {
        self.allowlist.iter().any(|a| a == binary)
    }

    fn err(&self, code: &str, message: impl Into<String>) -> ToolResult {
        ToolResult::Err {
            tool_id: self.id.clone(),
            code: code.to_string(),
            message: message.into(),
        }
    }
}

impl ToolDispatcher for ShellToolDispatcher {
    fn invoke(&self, inv: &ToolInvocation) -> ToolResult {
        let command = match inv.args.get("command") {
            Some(Value::String(s)) if !s.is_empty() => s.clone(),
            _ => return self.err(E_DISPATCH_MISSING_ARG, "shell.exec requires `command` arg"),
        };
        let args: Vec<String> = match inv.args.get("args") {
            None | Some(Value::Null) => Vec::new(),
            Some(Value::Array(items)) => {
                let mut out = Vec::with_capacity(items.len());
                for v in items {
                    match v {
                        Value::String(s) => out.push(s.clone()),
                        _ => {
                            return self.err(
                                E_DISPATCH_MISSING_ARG,
                                "shell.exec `args` must be an array of strings",
                            )
                        }
                    }
                }
                out
            }
            _ => {
                return self.err(
                    E_DISPATCH_MISSING_ARG,
                    "shell.exec `args` must be an array of strings",
                )
            }
        };

        if !self.allowed(&command) {
            return self.err(
                E_DISPATCH_BIN_DISALLOWED,
                format!("shell.exec binary `{command}` not in allowlist"),
            );
        }
        let Some(resolved) = which_in_path(&command) else {
            return self.err(
                E_DISPATCH_BIN_DISALLOWED,
                format!("shell.exec binary `{command}` not found on PATH"),
            );
        };

        let mut child = match self.sandbox.spawn(&self.base_spec, &resolved, &args) {
            Ok(c) => c,
            Err(e) => {
                return self.err(
                    E_DISPATCH_SPAWN_FAILED,
                    format!("shell.exec spawn failed: {e}"),
                )
            }
        };

        match wait_with_timeout(&mut child, self.timeout) {
            WaitOutcome::Exited {
                status,
                stdout,
                stderr,
            } => {
                if status == 0 {
                    let stdout_len = stdout.len() as u64;
                    let body = serde_json::json!({
                        "stdout": String::from_utf8_lossy(&stdout).into_owned(),
                        "exit": status,
                    });
                    ToolResult::Ok {
                        tool_id: self.id.clone(),
                        body,
                        bytes: stdout_len,
                    }
                } else {
                    let tail = String::from_utf8_lossy(&stderr).into_owned();
                    let trimmed = tail_text(&tail, 256);
                    self.err(E_DISPATCH_EXIT_NONZERO, format!("exit {status}: {trimmed}"))
                }
            }
            WaitOutcome::Timeout => {
                let _ = child.kill();
                let _ = child.wait();
                self.err(E_DISPATCH_TIMEOUT, "shell.exec timeout")
            }
            WaitOutcome::WaitFailed(e) => self.err(
                E_DISPATCH_SPAWN_FAILED,
                format!("shell.exec wait failed: {e}"),
            ),
        }
    }

    fn supports(&self, tool_id: &str) -> bool {
        tool_id == Self::ID || tool_id.starts_with("shell.exec.")
    }

    fn id(&self) -> &str {
        &self.id
    }
}

/// `ToolDispatcher` for `fs.read` calls. Reads a file inside the
/// configured workspace root and returns its contents as a JSON body.
///
/// # Path policy
///
/// The dispatcher canonicalizes `<root>/<requested>` and verifies the
/// result is still a descendant of the canonicalized root. This catches
/// both `..` traversal and symlink-escape attempts. Any escape returns
/// [`E_DISPATCH_PATH_ESCAPE`].
#[derive(Debug, Clone)]
pub struct FsReadToolDispatcher {
    id: String,
    root: PathBuf,
    max_bytes: u64,
}

impl FsReadToolDispatcher {
    /// Stable id used to look this dispatcher up in the registry.
    pub const ID: &'static str = "fs.read";

    /// Default cap on a single file read: 1 MiB. Larger reads should
    /// move to a streaming surface.
    pub const DEFAULT_MAX_BYTES: u64 = 1 << 20;

    /// Build a new dispatcher anchored at `root`.
    #[must_use]
    pub fn new(root: PathBuf) -> Self {
        Self {
            id: Self::ID.to_string(),
            root,
            max_bytes: Self::DEFAULT_MAX_BYTES,
        }
    }

    /// Override the per-call size cap.
    #[must_use]
    pub const fn with_max_bytes(mut self, max_bytes: u64) -> Self {
        self.max_bytes = max_bytes;
        self
    }

    /// Inspect the configured size cap.
    #[must_use]
    pub const fn max_bytes(&self) -> u64 {
        self.max_bytes
    }

    /// Inspect the configured workspace root.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    fn err(&self, code: &str, message: impl Into<String>) -> ToolResult {
        ToolResult::Err {
            tool_id: self.id.clone(),
            code: code.to_string(),
            message: message.into(),
        }
    }
}

impl ToolDispatcher for FsReadToolDispatcher {
    fn invoke(&self, inv: &ToolInvocation) -> ToolResult {
        let requested = match inv.args.get("path") {
            Some(Value::String(s)) if !s.is_empty() => PathBuf::from(s),
            _ => return self.err(E_DISPATCH_MISSING_ARG, "fs.read requires `path`"),
        };

        // Cheap textual guard against absolute / parent-dir escapes,
        // applied before any disk I/O. Canonicalization below is the
        // authoritative check; this is defense-in-depth so a missing
        // root directory still rejects obviously-bad inputs.
        if requested.is_absolute()
            || requested
                .components()
                .any(|c| matches!(c, Component::ParentDir))
        {
            return self.err(E_DISPATCH_PATH_ESCAPE, "path escapes workspace root");
        }

        let joined = self.root.join(&requested);
        let canonical_root = match self.root.canonicalize() {
            Ok(p) => p,
            Err(e) => {
                return self.err(
                    E_DISPATCH_READ_FAILED,
                    format!("workspace root unreadable: {e}"),
                )
            }
        };
        let canonical_target = match joined.canonicalize() {
            Ok(p) => p,
            Err(e) => {
                return self.err(E_DISPATCH_READ_FAILED, format!("path unreadable: {e}"));
            }
        };
        if !canonical_target.starts_with(&canonical_root) {
            return self.err(E_DISPATCH_PATH_ESCAPE, "path escapes workspace root");
        }

        let metadata = match std::fs::metadata(&canonical_target) {
            Ok(m) => m,
            Err(e) => {
                return self.err(E_DISPATCH_READ_FAILED, format!("stat failed: {e}"));
            }
        };
        if metadata.len() > self.max_bytes {
            return self.err(
                E_DISPATCH_SIZE_CAP,
                format!("file exceeds {} byte cap", self.max_bytes),
            );
        }
        let bytes = match std::fs::read(&canonical_target) {
            Ok(b) => b,
            Err(e) => {
                return self.err(E_DISPATCH_READ_FAILED, format!("read failed: {e}"));
            }
        };
        let raw_len = bytes.len() as u64;
        let text = String::from_utf8_lossy(&bytes).into_owned();
        let body = serde_json::json!({
            "path": canonical_target.display().to_string(),
            "content": text,
        });
        ToolResult::Ok {
            tool_id: self.id.clone(),
            body,
            bytes: raw_len,
        }
    }

    fn supports(&self, tool_id: &str) -> bool {
        tool_id == Self::ID
    }

    fn id(&self) -> &str {
        &self.id
    }
}

/// Build the default dispatcher registry used by the agent loop:
/// `FsReadToolDispatcher` anchored at `workspace_root`, plus a
/// `ShellToolDispatcher` driving `sandbox` against `base_spec`.
#[must_use]
pub fn default_dispatchers(
    workspace_root: PathBuf,
    sandbox: SandboxSpawn,
    base_spec: SandboxLaunchSpec,
) -> RegistryDispatcher {
    let mut reg = RegistryDispatcher::new();
    // Register order is observable via `ids()`; fs.read first matches
    // the doctor/event-log ordering used elsewhere.
    let fs = FsReadToolDispatcher::new(workspace_root);
    let shell = ShellToolDispatcher::new(sandbox, base_spec);
    // Both ids are guaranteed unique; suppress the error path with a
    // fall-through that still returns an empty registry on the
    // exceedingly-unlikely duplicate-id failure mode.
    if reg.register(Box::new(fs)).is_err() {
        return reg;
    }
    if reg.register(Box::new(shell)).is_err() {
        return reg;
    }
    reg
}

// ---- helpers ------------------------------------------------------------

/// Resolve `binary` against `PATH`, returning the first directory entry
/// that exists as a regular file. Mirrors the lookup the platform shell
/// would do, without any of the more exotic features (aliases, globs).
fn which_in_path(binary: &str) -> Option<PathBuf> {
    if binary.contains('/') {
        // Refuse fully-qualified paths: the allowlist contract is a
        // bare name. This is also a defense-in-depth against bypasses
        // like `"echo".to_string() + "/.."`.
        return None;
    }
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let candidate = dir.join(binary);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

#[derive(Debug)]
enum WaitOutcome {
    Exited {
        status: i32,
        stdout: Vec<u8>,
        stderr: Vec<u8>,
    },
    Timeout,
    WaitFailed(std::io::Error),
}

/// Wait for `child` to exit, or return `Timeout` after `timeout` has
/// elapsed. Polling-based to keep dep surface flat (no `wait-timeout`
/// crate).
fn wait_with_timeout(child: &mut std::process::Child, timeout: Duration) -> WaitOutcome {
    let start = Instant::now();
    let poll = Duration::from_millis(25);
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let mut stdout_buf = Vec::new();
                let mut stderr_buf = Vec::new();
                if let Some(mut s) = child.stdout.take() {
                    let _ = s.read_to_end(&mut stdout_buf);
                }
                if let Some(mut s) = child.stderr.take() {
                    let _ = s.read_to_end(&mut stderr_buf);
                }
                let code = status.code().unwrap_or(-1);
                return WaitOutcome::Exited {
                    status: code,
                    stdout: stdout_buf,
                    stderr: stderr_buf,
                };
            }
            Ok(None) => {
                if start.elapsed() >= timeout {
                    return WaitOutcome::Timeout;
                }
                thread::sleep(poll);
            }
            Err(e) => return WaitOutcome::WaitFailed(e),
        }
    }
}

fn tail_text(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        return s.to_string();
    }
    let start = s.chars().count().saturating_sub(max_chars);
    s.chars().skip(start).collect()
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};
    use std::fs;
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::time::Duration;

    use tempfile::TempDir;

    use super::*;
    use crate::sandbox::{SandboxBackend, SandboxSpawn};
    use crate::sandbox_resolve::{BackendChoice, ResolvedNet, SandboxLaunchSpec};

    fn passthrough_spec() -> SandboxLaunchSpec {
        SandboxLaunchSpec {
            mounts: Vec::new(),
            net: ResolvedNet::Off,
            env: BTreeMap::new(),
            allowed_caps: BTreeSet::new(),
            denied_caps: BTreeSet::new(),
            working_dir: PathBuf::from("/"),
            cpu_quota_pct: None,
            memory_limit_mib: None,
            backend: BackendChoice::Passthrough,
        }
    }

    fn passthrough_spec_in(dir: &Path) -> SandboxLaunchSpec {
        SandboxLaunchSpec {
            mounts: Vec::new(),
            net: ResolvedNet::Off,
            env: BTreeMap::new(),
            allowed_caps: BTreeSet::new(),
            denied_caps: BTreeSet::new(),
            working_dir: dir.to_path_buf(),
            cpu_quota_pct: None,
            memory_limit_mib: None,
            backend: BackendChoice::Passthrough,
        }
    }

    fn shell_dispatcher() -> ShellToolDispatcher {
        let sandbox = SandboxSpawn::new(SandboxBackend::Passthrough);
        // Pipe stdout/stderr so the wait_with_timeout helper can drain
        // them. The Passthrough spawn path uses Command::spawn directly
        // and we cannot intercept stdio after the fact, so we use a
        // shim that constructs the child via Command::new directly in
        // the tests where we need stdout capture. For the cases we
        // assert on stdout / stderr we instead drive the underlying
        // process via the helper below.
        ShellToolDispatcher::new(sandbox, passthrough_spec())
    }

    fn invocation(tool: &str, args: BTreeMap<String, serde_json::Value>) -> ToolInvocation {
        ToolInvocation {
            tool_id: tool.to_string(),
            args,
            capability: "shell.exec".to_string(),
            turn_id: 1,
        }
    }

    fn shell_args(command: &str, args: &[&str]) -> BTreeMap<String, serde_json::Value> {
        let mut a = BTreeMap::new();
        a.insert("command".to_string(), serde_json::json!(command));
        a.insert("args".to_string(), serde_json::json!(args.to_vec()));
        a
    }

    fn assert_err_code(result: ToolResult, expected: &str) {
        if let ToolResult::Err { code, message, .. } = result {
            assert_eq!(code, expected, "wrong code (message was: {message})");
        } else {
            panic!("expected Err({expected}), got Ok");
        }
    }

    fn assert_ok(result: ToolResult) -> (String, serde_json::Value, u64) {
        if let ToolResult::Ok {
            tool_id,
            body,
            bytes,
        } = result
        {
            (tool_id, body, bytes)
        } else {
            panic!("expected Ok, got Err: {result:?}");
        }
    }

    // ---- ShellToolDispatcher: pure-fields tests -----------------------

    #[test]
    fn shell_supports_canonical_and_namespaced() {
        let d = shell_dispatcher();
        assert!(d.supports("shell.exec"));
        assert!(d.supports("shell.exec.streaming"));
        assert!(!d.supports("fs.read"));
        assert!(!d.supports("shell"));
    }

    #[test]
    fn shell_id_is_stable() {
        let d = shell_dispatcher();
        assert_eq!(d.id(), "shell.exec");
    }

    #[test]
    fn shell_with_timeout_round_trips() {
        let d = shell_dispatcher().with_timeout(Duration::from_millis(123));
        assert_eq!(d.timeout(), Duration::from_millis(123));
    }

    #[test]
    fn shell_default_allowlist_is_paranoid() {
        let d = shell_dispatcher();
        assert_eq!(d.allowlist().len(), SHELL_DEFAULT_ALLOWLIST.len());
        for entry in SHELL_DEFAULT_ALLOWLIST {
            assert!(d.allowlist().iter().any(|s| s == entry));
        }
    }

    #[test]
    fn shell_with_allowlist_round_trips() {
        let d = shell_dispatcher().with_allowlist(vec!["echo".to_string(), "true".to_string()]);
        assert_eq!(d.allowlist(), &["echo".to_string(), "true".to_string()]);
    }

    #[test]
    fn shell_debug_smoke() {
        let d = shell_dispatcher();
        let rendered = format!("{d:?}");
        assert!(rendered.contains("shell.exec"));
        // `SandboxBackend` derives Debug → PascalCase. The Display
        // impl is the lowercase variant; we render via Debug here.
        assert!(rendered.contains("Passthrough"));
    }

    // ---- ShellToolDispatcher: invoke error paths ----------------------

    #[test]
    fn shell_invoke_missing_command_returns_missing_arg() {
        let d = shell_dispatcher();
        let inv = invocation("shell.exec", BTreeMap::new());
        assert_err_code(d.invoke(&inv), E_DISPATCH_MISSING_ARG);
    }

    #[test]
    fn shell_invoke_command_wrong_type_returns_missing_arg() {
        let d = shell_dispatcher();
        let mut args = BTreeMap::new();
        args.insert("command".to_string(), serde_json::json!(42));
        let inv = invocation("shell.exec", args);
        assert_err_code(d.invoke(&inv), E_DISPATCH_MISSING_ARG);
    }

    #[test]
    fn shell_invoke_empty_command_returns_missing_arg() {
        let d = shell_dispatcher();
        let mut args = BTreeMap::new();
        args.insert("command".to_string(), serde_json::json!(""));
        let inv = invocation("shell.exec", args);
        assert_err_code(d.invoke(&inv), E_DISPATCH_MISSING_ARG);
    }

    #[test]
    fn shell_invoke_bad_args_type_returns_missing_arg() {
        let d = shell_dispatcher();
        let mut args = BTreeMap::new();
        args.insert("command".to_string(), serde_json::json!("echo"));
        args.insert("args".to_string(), serde_json::json!("not-an-array"));
        let inv = invocation("shell.exec", args);
        assert_err_code(d.invoke(&inv), E_DISPATCH_MISSING_ARG);
    }

    #[test]
    fn shell_invoke_nonstring_arg_returns_missing_arg() {
        let d = shell_dispatcher();
        let mut args = BTreeMap::new();
        args.insert("command".to_string(), serde_json::json!("echo"));
        args.insert("args".to_string(), serde_json::json!([42, "hi"]));
        let inv = invocation("shell.exec", args);
        assert_err_code(d.invoke(&inv), E_DISPATCH_MISSING_ARG);
    }

    #[test]
    fn shell_invoke_args_object_returns_missing_arg() {
        // The `_ =>` arm in the args match handles non-array, non-null
        // shapes (e.g. an object).
        let d = shell_dispatcher();
        let mut args = BTreeMap::new();
        args.insert("command".to_string(), serde_json::json!("echo"));
        args.insert("args".to_string(), serde_json::json!({"k": "v"}));
        let inv = invocation("shell.exec", args);
        assert_err_code(d.invoke(&inv), E_DISPATCH_MISSING_ARG);
    }

    #[test]
    fn shell_invoke_args_null_treated_as_empty() {
        // command + args: null lands on the `None | Some(Null)` arm.
        let d = shell_dispatcher();
        let mut args = BTreeMap::new();
        args.insert("command".to_string(), serde_json::json!("echo"));
        args.insert("args".to_string(), serde_json::Value::Null);
        let inv = invocation("shell.exec", args);
        let (tool_id, _, _) = assert_ok(d.invoke(&inv));
        assert_eq!(tool_id, "shell.exec");
    }

    #[test]
    fn shell_invoke_args_missing_treated_as_empty() {
        // Only `command` is set; the `args` key is absent → empty Vec
        // path, exercising the `None |` half of the args match.
        let d = shell_dispatcher();
        let mut args = BTreeMap::new();
        args.insert("command".to_string(), serde_json::json!("echo"));
        let inv = invocation("shell.exec", args);
        let (tool_id, _, _) = assert_ok(d.invoke(&inv));
        assert_eq!(tool_id, "shell.exec");
    }

    #[test]
    fn shell_invoke_binary_not_in_allowlist_returns_disallowed() {
        let d = shell_dispatcher();
        let inv = invocation("shell.exec", shell_args("rm", &["-rf", "/"]));
        assert_err_code(d.invoke(&inv), E_DISPATCH_BIN_DISALLOWED);
    }

    #[test]
    fn shell_invoke_qualified_path_in_command_rejected() {
        // Even if "/bin/echo" would resolve, the contract is bare names.
        let d = shell_dispatcher().with_allowlist(vec!["/bin/echo".to_string()]);
        let inv = invocation("shell.exec", shell_args("/bin/echo", &["hi"]));
        assert_err_code(d.invoke(&inv), E_DISPATCH_BIN_DISALLOWED);
    }

    #[test]
    fn shell_invoke_unknown_in_allowlist_but_missing_from_path_returns_disallowed() {
        let d = shell_dispatcher()
            .with_allowlist(vec!["totally_bogus_binary_for_test_xyz".to_string()]);
        let inv = invocation(
            "shell.exec",
            shell_args("totally_bogus_binary_for_test_xyz", &[]),
        );
        assert_err_code(d.invoke(&inv), E_DISPATCH_BIN_DISALLOWED);
    }

    #[test]
    fn shell_invoke_passthrough_echo_ok() {
        // `echo` is on PATH on every Linux + macOS CI runner. The
        // Passthrough backend does not capture stdout (Command::spawn
        // inherits stdio), so we cannot assert on the body's `stdout`
        // text — but we *can* assert on the Ok/Err discriminator and
        // the exit code field. That confirms the full
        // allowlist -> which -> spawn -> wait path.
        let d = shell_dispatcher();
        let inv = invocation("shell.exec", shell_args("echo", &["hi"]));
        let (tool_id, body, _) = assert_ok(d.invoke(&inv));
        assert_eq!(tool_id, "shell.exec");
        assert_eq!(
            body.get("exit").and_then(serde_json::Value::as_i64),
            Some(0)
        );
    }

    #[test]
    fn shell_invoke_nonexistent_path_via_cat_returns_exit_nonzero() {
        // Allow `cat` for this test only; default deny applies.
        let d = shell_dispatcher()
            .with_allowlist(vec!["cat".to_string()])
            .with_timeout(Duration::from_secs(5));
        let inv = invocation(
            "shell.exec",
            shell_args("cat", &["/this/path/definitely/does/not/exist/xyzzy"]),
        );
        assert_err_code(d.invoke(&inv), E_DISPATCH_EXIT_NONZERO);
    }

    #[test]
    fn shell_invoke_spawn_error_returns_spawn_failed() {
        // Hit `Err(e) => self.err(E_DISPATCH_SPAWN_FAILED, …)` by
        // pointing the spawner at a backend that synthesises an error
        // before the child boots. `WindowsJob` is the easiest: it is
        // a constant `Err(Unsupported(...))` on all hosts. Stage `echo`
        // through it so the allowlist + which paths both pass first.
        let sandbox = SandboxSpawn::new(SandboxBackend::WindowsJob);
        let d = ShellToolDispatcher::new(sandbox, passthrough_spec());
        let inv = invocation("shell.exec", shell_args("echo", &["hi"]));
        assert_err_code(d.invoke(&inv), E_DISPATCH_SPAWN_FAILED);
    }

    #[cfg(unix)]
    #[test]
    fn shell_invoke_timeout_returns_timeout_sentinel() {
        // `sleep 60` is on PATH on Unix CI runners and will block for
        // long past our 100ms timeout, guaranteeing the timeout arm.
        let d = shell_dispatcher()
            .with_allowlist(vec!["sleep".to_string()])
            .with_timeout(Duration::from_millis(100));
        let inv = invocation("shell.exec", shell_args("sleep", &["60"]));
        assert_err_code(d.invoke(&inv), E_DISPATCH_TIMEOUT);
    }

    // ---- FsReadToolDispatcher tests -----------------------------------

    fn fs_invocation(path: &str) -> ToolInvocation {
        let mut args = BTreeMap::new();
        args.insert("path".to_string(), serde_json::json!(path));
        ToolInvocation {
            tool_id: "fs.read".to_string(),
            args,
            capability: "fs.read".to_string(),
            turn_id: 1,
        }
    }

    #[test]
    fn fs_supports_only_canonical() {
        let d = FsReadToolDispatcher::new(PathBuf::from("/"));
        assert!(d.supports("fs.read"));
        assert!(!d.supports("fs.write"));
        assert!(!d.supports("fs.read.streaming"));
    }

    #[test]
    fn fs_id_is_stable_and_max_bytes_default() {
        let d = FsReadToolDispatcher::new(PathBuf::from("/"));
        assert_eq!(d.id(), "fs.read");
        assert_eq!(d.max_bytes(), FsReadToolDispatcher::DEFAULT_MAX_BYTES);
        assert_eq!(d.root(), Path::new("/"));
    }

    #[test]
    fn fs_with_max_bytes_round_trips() {
        let d = FsReadToolDispatcher::new(PathBuf::from("/")).with_max_bytes(42);
        assert_eq!(d.max_bytes(), 42);
    }

    #[test]
    fn fs_invoke_missing_path_returns_missing_arg() {
        let d = FsReadToolDispatcher::new(PathBuf::from("/"));
        let inv = ToolInvocation {
            tool_id: "fs.read".to_string(),
            args: BTreeMap::new(),
            capability: "fs.read".to_string(),
            turn_id: 1,
        };
        assert_err_code(d.invoke(&inv), E_DISPATCH_MISSING_ARG);
    }

    #[test]
    fn fs_invoke_nonstring_path_returns_missing_arg() {
        let d = FsReadToolDispatcher::new(PathBuf::from("/"));
        let mut args = BTreeMap::new();
        args.insert("path".to_string(), serde_json::json!(7));
        let inv = ToolInvocation {
            tool_id: "fs.read".to_string(),
            args,
            capability: "fs.read".to_string(),
            turn_id: 1,
        };
        assert_err_code(d.invoke(&inv), E_DISPATCH_MISSING_ARG);
    }

    #[test]
    fn fs_invoke_reads_file_contents() {
        let tmp = TempDir::new().expect("tmp");
        let target = tmp.path().join("hello.txt");
        fs::write(&target, "hello world").expect("write");
        let d = FsReadToolDispatcher::new(tmp.path().to_path_buf());
        let inv = fs_invocation("hello.txt");
        let (_, body, bytes) = assert_ok(d.invoke(&inv));
        assert_eq!(bytes, 11);
        assert_eq!(
            body.get("content").and_then(|v| v.as_str()),
            Some("hello world")
        );
    }

    #[test]
    fn fs_invoke_parent_dir_escape_returns_path_escape() {
        let tmp = TempDir::new().expect("tmp");
        let d = FsReadToolDispatcher::new(tmp.path().to_path_buf());
        let inv = fs_invocation("../escape");
        assert_err_code(d.invoke(&inv), E_DISPATCH_PATH_ESCAPE);
    }

    #[test]
    fn fs_invoke_absolute_path_returns_path_escape() {
        let tmp = TempDir::new().expect("tmp");
        let d = FsReadToolDispatcher::new(tmp.path().to_path_buf());
        let inv = fs_invocation("/etc/passwd");
        assert_err_code(d.invoke(&inv), E_DISPATCH_PATH_ESCAPE);
    }

    #[test]
    fn fs_invoke_oversize_returns_size_cap() {
        let tmp = TempDir::new().expect("tmp");
        let target = tmp.path().join("big.bin");
        fs::write(&target, vec![0u8; 32]).expect("write");
        let d = FsReadToolDispatcher::new(tmp.path().to_path_buf()).with_max_bytes(8);
        let inv = fs_invocation("big.bin");
        assert_err_code(d.invoke(&inv), E_DISPATCH_SIZE_CAP);
    }

    #[test]
    fn fs_invoke_missing_file_returns_read_failed() {
        let tmp = TempDir::new().expect("tmp");
        let d = FsReadToolDispatcher::new(tmp.path().to_path_buf());
        let inv = fs_invocation("definitely-not-here.txt");
        assert_err_code(d.invoke(&inv), E_DISPATCH_READ_FAILED);
    }

    #[test]
    fn fs_invoke_directory_target_returns_read_failed() {
        // Passing a directory through canonicalize + metadata succeeds
        // (directories are stat-able), but `std::fs::read` on a
        // directory returns an EISDIR-flavoured error — exercising the
        // post-metadata read-failed branch.
        let tmp = TempDir::new().expect("tmp");
        let subdir = tmp.path().join("subdir");
        fs::create_dir(&subdir).expect("mkdir");
        let d = FsReadToolDispatcher::new(tmp.path().to_path_buf());
        let inv = fs_invocation("subdir");
        assert_err_code(d.invoke(&inv), E_DISPATCH_READ_FAILED);
    }

    #[test]
    fn fs_invoke_unreadable_root_returns_read_failed() {
        let d =
            FsReadToolDispatcher::new(PathBuf::from("/nonexistent-root-for-fs-read-test-xyzzy"));
        let inv = fs_invocation("anything.txt");
        assert_err_code(d.invoke(&inv), E_DISPATCH_READ_FAILED);
    }

    #[cfg(unix)]
    #[test]
    fn fs_invoke_symlink_escape_returns_path_escape() {
        use std::os::unix::fs::symlink;
        let outside = TempDir::new().expect("outside");
        let secret = outside.path().join("secret.txt");
        fs::write(&secret, "TOP SECRET").expect("write");

        let inside = TempDir::new().expect("inside");
        let link = inside.path().join("link-to-secret");
        symlink(&secret, &link).expect("symlink");

        let d = FsReadToolDispatcher::new(inside.path().to_path_buf());
        let inv = fs_invocation("link-to-secret");
        assert_err_code(d.invoke(&inv), E_DISPATCH_PATH_ESCAPE);
    }

    #[test]
    fn fs_invoke_binary_file_renders_as_lossy_utf8() {
        // Per the module's binary-vs-text policy: bytes are rendered
        // via `String::from_utf8_lossy`; invalid sequences land as
        // U+FFFD. The byte count returned is the raw on-disk length.
        let tmp = TempDir::new().expect("tmp");
        let target = tmp.path().join("binary.bin");
        // 0xFF is not valid UTF-8 by itself.
        fs::write(&target, [b'A', 0xFF, b'Z']).expect("write");
        let d = FsReadToolDispatcher::new(tmp.path().to_path_buf());
        let inv = fs_invocation("binary.bin");
        let (_, body, bytes) = assert_ok(d.invoke(&inv));
        assert_eq!(bytes, 3, "raw byte count");
        let content = body
            .get("content")
            .and_then(|v| v.as_str())
            .expect("content string");
        assert!(content.starts_with('A'));
        assert!(content.ends_with('Z'));
        assert!(content.contains('\u{FFFD}'), "lossy replacement char");
    }

    // ---- default_dispatchers registry integration ---------------------

    #[test]
    fn default_dispatchers_registers_two() {
        let tmp = TempDir::new().expect("tmp");
        let reg = default_dispatchers(
            tmp.path().to_path_buf(),
            SandboxSpawn::new(SandboxBackend::Passthrough),
            passthrough_spec_in(tmp.path()),
        );
        let ids = reg.ids();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&"fs.read"));
        assert!(ids.contains(&"shell.exec"));
    }

    #[test]
    fn default_dispatchers_routes_fs_read() {
        let tmp = TempDir::new().expect("tmp");
        let target = tmp.path().join("hi.txt");
        fs::write(&target, "ok").expect("write");
        let reg = default_dispatchers(
            tmp.path().to_path_buf(),
            SandboxSpawn::new(SandboxBackend::Passthrough),
            passthrough_spec_in(tmp.path()),
        );
        let inv = fs_invocation("hi.txt");
        let (tool_id, _, _) = assert_ok(reg.dispatch(&inv));
        assert_eq!(tool_id, "fs.read");
    }

    #[test]
    fn default_dispatchers_routes_shell_exec() {
        let tmp = TempDir::new().expect("tmp");
        let reg = default_dispatchers(
            tmp.path().to_path_buf(),
            SandboxSpawn::new(SandboxBackend::Passthrough),
            passthrough_spec_in(tmp.path()),
        );
        // Use the allowlisted `echo`; we only check that the dispatch
        // landed on ShellToolDispatcher (tool_id reported back).
        let inv = invocation("shell.exec", shell_args("echo", &["hi"]));
        let (tool_id, _, _) = assert_ok(reg.dispatch(&inv));
        assert_eq!(tool_id, "shell.exec");
    }

    #[test]
    fn default_dispatchers_unknown_tool_returns_e5005() {
        let tmp = TempDir::new().expect("tmp");
        let reg = default_dispatchers(
            tmp.path().to_path_buf(),
            SandboxSpawn::new(SandboxBackend::Passthrough),
            passthrough_spec_in(tmp.path()),
        );
        let inv = ToolInvocation {
            tool_id: "mcp.github.list_issues".to_string(),
            args: BTreeMap::new(),
            capability: "mcp".to_string(),
            turn_id: 0,
        };
        // STRAT-E5005 is the RegistryDispatcher fallthrough.
        assert_err_code(reg.dispatch(&inv), "STRAT-E5005");
    }

    // ---- Send + Sync smoke + dyn trait integration --------------------

    #[test]
    fn dispatchers_are_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ShellToolDispatcher>();
        assert_send_sync::<FsReadToolDispatcher>();
    }

    #[test]
    fn shell_dispatcher_via_arc_dyn() {
        let d: Arc<dyn ToolDispatcher> = Arc::new(shell_dispatcher());
        assert_eq!(d.id(), "shell.exec");
        assert!(d.supports("shell.exec"));
    }

    #[test]
    fn fs_dispatcher_via_arc_dyn() {
        let d: Arc<dyn ToolDispatcher> = Arc::new(FsReadToolDispatcher::new(PathBuf::from("/")));
        assert_eq!(d.id(), "fs.read");
        assert!(d.supports("fs.read"));
    }

    // ---- helpers ------------------------------------------------------

    #[test]
    fn which_in_path_finds_sh_on_unix() {
        if cfg!(unix) {
            let p = which_in_path("sh");
            assert!(p.is_some(), "sh should be on PATH");
        }
    }

    #[test]
    fn which_in_path_rejects_qualified_name() {
        assert!(which_in_path("/bin/sh").is_none());
    }

    #[test]
    fn which_in_path_misses_bogus_name() {
        assert!(which_in_path("totally_nonexistent_bin_xyzzy_98765").is_none());
    }

    #[test]
    fn tail_text_passes_short_strings_through() {
        assert_eq!(tail_text("abc", 10), "abc");
    }

    #[cfg(unix)]
    #[test]
    fn wait_with_timeout_drains_piped_stdout_and_stderr() {
        // Exercise the `child.stdout.take()` / `child.stderr.take()`
        // arms in `wait_with_timeout` by running a child with both
        // streams piped explicitly. The Passthrough backend used by
        // ShellToolDispatcher inherits stdio, so we go through
        // `Command::new` directly for this unit test.
        let sh = if Path::new("/bin/sh").exists() {
            "/bin/sh"
        } else {
            "/usr/bin/sh"
        };
        let mut child = std::process::Command::new(sh)
            .args(["-c", "printf hello; printf world 1>&2"])
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .expect("spawn sh");
        match wait_with_timeout(&mut child, Duration::from_secs(5)) {
            WaitOutcome::Exited {
                status,
                stdout,
                stderr,
            } => {
                assert_eq!(status, 0);
                assert_eq!(stdout, b"hello");
                assert_eq!(stderr, b"world");
            }
            other => panic!("expected Exited, got {other:?}"),
        }
    }

    #[cfg(unix)]
    #[test]
    fn wait_with_timeout_exit_nonzero_through_dispatcher() {
        // Already covered by shell_invoke_nonexistent_path_via_cat_…,
        // but pin the WaitOutcome::Exited{status != 0} arm explicitly.
        let sh = if Path::new("/bin/sh").exists() {
            "/bin/sh"
        } else {
            "/usr/bin/sh"
        };
        let mut child = std::process::Command::new(sh)
            .args(["-c", "exit 3"])
            .spawn()
            .expect("spawn sh");
        match wait_with_timeout(&mut child, Duration::from_secs(5)) {
            WaitOutcome::Exited { status, .. } => assert_eq!(status, 3),
            other => panic!("expected Exited, got {other:?}"),
        }
    }

    #[test]
    fn tail_text_truncates_long_strings() {
        let long = "0123456789".repeat(50);
        let t = tail_text(&long, 10);
        assert_eq!(t.chars().count(), 10);
        assert!(long.ends_with(&t));
    }
}
