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

// Sandbox + git wrapper code uses short, conventional names (cmd, rel,
// b0/b1/b2, git_bin) that overlap with siblings; the clippy pedantic
// "similar_names" lint is more confusing than helpful here.
#![allow(
    clippy::similar_names,
    reason = "intentional short conventional names: cmd, rel, b0..b2, git_bin"
)]

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
/// Local sentinel: fs.write failed at the filesystem layer.
const E_DISPATCH_WRITE_FAILED: &str = "E_DISPATCH_WRITE_FAILED";
/// Local sentinel: fs.edit could not find the `old_string` to replace.
const E_DISPATCH_EDIT_NOT_FOUND: &str = "E_DISPATCH_EDIT_NOT_FOUND";
/// Local sentinel: fs.edit `old_string` appeared more than once.
const E_DISPATCH_EDIT_AMBIGUOUS: &str = "E_DISPATCH_EDIT_AMBIGUOUS";
/// Local sentinel: grep / glob pattern was invalid.
const E_DISPATCH_BAD_PATTERN: &str = "E_DISPATCH_BAD_PATTERN";

/// Default allowlist of read-only binaries that `ShellToolDispatcher` will exec.
///
/// Paranoid scaffold: every additional binary must be reviewed against the
/// threat model in `plan/31-tool-sandbox-and-secrets.md`.
pub const SHELL_DEFAULT_ALLOWLIST: &[&str] = &[
    "echo", "ls", "cat", "pwd", "wc", "head", "tail",
    // Read-only git operations. The shell dispatcher runs under a
    // passthrough sandbox today, so any git subcommand that *modifies*
    // history (commit / push / reset / rebase) is still implicitly
    // denied by the model's reluctance to call it — and explicitly
    // gateable via `with_allowlist` for hardened deployments.
    "git",
];

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

/// Write a file inside the workspace root.
#[derive(Debug, Clone)]
pub struct FsWriteToolDispatcher {
    id: String,
    root: PathBuf,
    max_bytes: u64,
}

impl FsWriteToolDispatcher {
    /// Stable id used to look this dispatcher up in the registry.
    pub const ID: &'static str = "fs.write";
    /// Default cap on a single write payload: 4 MiB.
    pub const DEFAULT_MAX_BYTES: u64 = 4 << 20;

    /// Build a new dispatcher anchored at `root`.
    #[must_use]
    pub fn new(root: PathBuf) -> Self {
        Self {
            id: Self::ID.to_string(),
            root,
            max_bytes: Self::DEFAULT_MAX_BYTES,
        }
    }

    fn err(&self, code: &str, message: impl Into<String>) -> ToolResult {
        ToolResult::Err {
            tool_id: self.id.clone(),
            code: code.to_string(),
            message: message.into(),
        }
    }
}

impl ToolDispatcher for FsWriteToolDispatcher {
    fn invoke(&self, inv: &ToolInvocation) -> ToolResult {
        let requested = match inv.args.get("path") {
            Some(Value::String(s)) if !s.is_empty() => PathBuf::from(s),
            _ => return self.err(E_DISPATCH_MISSING_ARG, "fs.write requires `path`"),
        };
        let content = match inv.args.get("content") {
            Some(Value::String(s)) => s.clone(),
            _ => return self.err(E_DISPATCH_MISSING_ARG, "fs.write requires `content`"),
        };
        if requested.is_absolute()
            || requested
                .components()
                .any(|c| matches!(c, Component::ParentDir))
        {
            return self.err(E_DISPATCH_PATH_ESCAPE, "path escapes workspace root");
        }
        let bytes_len = content.len() as u64;
        if bytes_len > self.max_bytes {
            return self.err(
                E_DISPATCH_SIZE_CAP,
                format!("content exceeds {} byte cap", self.max_bytes),
            );
        }
        let canonical_root = match self.root.canonicalize() {
            Ok(p) => p,
            Err(e) => {
                return self.err(
                    E_DISPATCH_WRITE_FAILED,
                    format!("workspace root unreadable: {e}"),
                )
            }
        };
        let target = canonical_root.join(&requested);
        if !target.starts_with(&canonical_root) {
            return self.err(E_DISPATCH_PATH_ESCAPE, "path escapes workspace root");
        }
        if let Some(parent) = target.parent() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                return self.err(E_DISPATCH_WRITE_FAILED, format!("mkdir failed: {e}"));
            }
        }
        if let Err(e) = std::fs::write(&target, content.as_bytes()) {
            return self.err(E_DISPATCH_WRITE_FAILED, format!("write failed: {e}"));
        }
        let body = serde_json::json!({
            "path": target.display().to_string(),
            "bytes": bytes_len,
        });
        ToolResult::Ok {
            tool_id: self.id.clone(),
            body,
            bytes: bytes_len,
        }
    }
    fn supports(&self, tool_id: &str) -> bool {
        tool_id == Self::ID
    }
    fn id(&self) -> &str {
        &self.id
    }
}

/// Single-occurrence string replace inside a workspace file.
#[derive(Debug, Clone)]
pub struct FsEditToolDispatcher {
    id: String,
    root: PathBuf,
    max_bytes: u64,
}

impl FsEditToolDispatcher {
    /// Stable id used to look this dispatcher up in the registry.
    pub const ID: &'static str = "fs.edit";
    /// Default cap on the post-edit file size: 4 MiB.
    pub const DEFAULT_MAX_BYTES: u64 = 4 << 20;

    /// Build a new dispatcher anchored at `root`.
    #[must_use]
    pub fn new(root: PathBuf) -> Self {
        Self {
            id: Self::ID.to_string(),
            root,
            max_bytes: Self::DEFAULT_MAX_BYTES,
        }
    }

    fn err(&self, code: &str, message: impl Into<String>) -> ToolResult {
        ToolResult::Err {
            tool_id: self.id.clone(),
            code: code.to_string(),
            message: message.into(),
        }
    }
}

impl ToolDispatcher for FsEditToolDispatcher {
    fn invoke(&self, inv: &ToolInvocation) -> ToolResult {
        let requested = match inv.args.get("path") {
            Some(Value::String(s)) if !s.is_empty() => PathBuf::from(s),
            _ => return self.err(E_DISPATCH_MISSING_ARG, "fs.edit requires `path`"),
        };
        let old_s = match inv.args.get("old_string") {
            Some(Value::String(s)) if !s.is_empty() => s.clone(),
            _ => return self.err(E_DISPATCH_MISSING_ARG, "fs.edit requires `old_string`"),
        };
        let new_s = match inv.args.get("new_string") {
            Some(Value::String(s)) => s.clone(),
            _ => return self.err(E_DISPATCH_MISSING_ARG, "fs.edit requires `new_string`"),
        };
        if requested.is_absolute()
            || requested
                .components()
                .any(|c| matches!(c, Component::ParentDir))
        {
            return self.err(E_DISPATCH_PATH_ESCAPE, "path escapes workspace root");
        }
        let canonical_root = match self.root.canonicalize() {
            Ok(p) => p,
            Err(e) => {
                return self.err(
                    E_DISPATCH_READ_FAILED,
                    format!("workspace root unreadable: {e}"),
                )
            }
        };
        let target = match canonical_root.join(&requested).canonicalize() {
            Ok(p) => p,
            Err(e) => return self.err(E_DISPATCH_READ_FAILED, format!("path unreadable: {e}")),
        };
        if !target.starts_with(&canonical_root) {
            return self.err(E_DISPATCH_PATH_ESCAPE, "path escapes workspace root");
        }
        let original = match std::fs::read_to_string(&target) {
            Ok(s) => s,
            Err(e) => return self.err(E_DISPATCH_READ_FAILED, format!("read failed: {e}")),
        };
        let count = original.matches(&old_s).count();
        if count == 0 {
            return self.err(E_DISPATCH_EDIT_NOT_FOUND, "old_string not found in file");
        }
        if count > 1 {
            return self.err(
                E_DISPATCH_EDIT_AMBIGUOUS,
                format!("old_string appears {count} times; widen the snippet"),
            );
        }
        let updated = original.replacen(&old_s, &new_s, 1);
        let bytes_len = updated.len() as u64;
        if bytes_len > self.max_bytes {
            return self.err(
                E_DISPATCH_SIZE_CAP,
                format!("result exceeds {} byte cap", self.max_bytes),
            );
        }
        if let Err(e) = std::fs::write(&target, updated.as_bytes()) {
            return self.err(E_DISPATCH_WRITE_FAILED, format!("write failed: {e}"));
        }
        let body = serde_json::json!({
            "path": target.display().to_string(),
            "bytes": bytes_len,
        });
        ToolResult::Ok {
            tool_id: self.id.clone(),
            body,
            bytes: bytes_len,
        }
    }
    fn supports(&self, tool_id: &str) -> bool {
        tool_id == Self::ID
    }
    fn id(&self) -> &str {
        &self.id
    }
}

/// Recursive regex search across the workspace root.
#[derive(Debug, Clone)]
pub struct GrepToolDispatcher {
    id: String,
    root: PathBuf,
    max_matches: usize,
}

impl GrepToolDispatcher {
    /// Stable id used to look this dispatcher up in the registry.
    pub const ID: &'static str = "grep";
    /// Default cap on returned matches.
    pub const DEFAULT_MAX_MATCHES: usize = 200;

    /// Build a new dispatcher anchored at `root`.
    #[must_use]
    pub fn new(root: PathBuf) -> Self {
        Self {
            id: Self::ID.to_string(),
            root,
            max_matches: Self::DEFAULT_MAX_MATCHES,
        }
    }

    fn err(&self, code: &str, message: impl Into<String>) -> ToolResult {
        ToolResult::Err {
            tool_id: self.id.clone(),
            code: code.to_string(),
            message: message.into(),
        }
    }
}

impl ToolDispatcher for GrepToolDispatcher {
    fn invoke(&self, inv: &ToolInvocation) -> ToolResult {
        let pattern = match inv.args.get("pattern") {
            Some(Value::String(s)) if !s.is_empty() => s.clone(),
            _ => return self.err(E_DISPATCH_MISSING_ARG, "grep requires `pattern`"),
        };
        let re = match regex::Regex::new(&pattern) {
            Ok(r) => r,
            Err(e) => return self.err(E_DISPATCH_BAD_PATTERN, format!("regex: {e}")),
        };
        let canonical_root = match self.root.canonicalize() {
            Ok(p) => p,
            Err(e) => {
                return self.err(
                    E_DISPATCH_READ_FAILED,
                    format!("workspace root unreadable: {e}"),
                )
            }
        };
        let mut matches: Vec<serde_json::Value> = Vec::new();
        let mut total_bytes: u64 = 0;
        let mut stack = vec![canonical_root.clone()];
        while let Some(dir) = stack.pop() {
            let Ok(read) = std::fs::read_dir(&dir) else {
                continue;
            };
            for entry in read.flatten() {
                let path = entry.path();
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    if name.starts_with('.') || name == "target" || name == "node_modules" {
                        continue;
                    }
                }
                let Ok(ft) = entry.file_type() else { continue };
                if ft.is_dir() {
                    stack.push(path);
                    continue;
                }
                if !ft.is_file() {
                    continue;
                }
                let Ok(content) = std::fs::read_to_string(&path) else {
                    continue;
                };
                for (i, line) in content.lines().enumerate() {
                    if re.is_match(line) {
                        let rel = path.strip_prefix(&canonical_root).unwrap_or(&path);
                        let entry = serde_json::json!({
                            "path": rel.display().to_string(),
                            "line": i + 1,
                            "text": line,
                        });
                        total_bytes += line.len() as u64;
                        matches.push(entry);
                        if matches.len() >= self.max_matches {
                            break;
                        }
                    }
                }
                if matches.len() >= self.max_matches {
                    break;
                }
            }
            if matches.len() >= self.max_matches {
                break;
            }
        }
        let body = serde_json::json!({
            "pattern": pattern,
            "matches": matches,
        });
        ToolResult::Ok {
            tool_id: self.id.clone(),
            body,
            bytes: total_bytes,
        }
    }
    fn supports(&self, tool_id: &str) -> bool {
        tool_id == Self::ID
    }
    fn id(&self) -> &str {
        &self.id
    }
}

/// Filename glob matching across the workspace root.
#[derive(Debug, Clone)]
pub struct GlobToolDispatcher {
    id: String,
    root: PathBuf,
    max_results: usize,
}

impl GlobToolDispatcher {
    /// Stable id used to look this dispatcher up in the registry.
    pub const ID: &'static str = "glob";
    /// Default cap on returned paths.
    pub const DEFAULT_MAX_RESULTS: usize = 500;

    /// Build a new dispatcher anchored at `root`.
    #[must_use]
    pub fn new(root: PathBuf) -> Self {
        Self {
            id: Self::ID.to_string(),
            root,
            max_results: Self::DEFAULT_MAX_RESULTS,
        }
    }

    fn err(&self, code: &str, message: impl Into<String>) -> ToolResult {
        ToolResult::Err {
            tool_id: self.id.clone(),
            code: code.to_string(),
            message: message.into(),
        }
    }
}

/// Translate a shell-style glob into an anchored regex.
fn glob_to_regex(pattern: &str) -> String {
    let mut out = String::with_capacity(pattern.len() + 4);
    out.push('^');
    let mut chars = pattern.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '*' => {
                if chars.peek() == Some(&'*') {
                    chars.next();
                    if chars.peek() == Some(&'/') {
                        chars.next();
                    }
                    out.push_str(".*");
                } else {
                    out.push_str("[^/]*");
                }
            }
            '?' => out.push_str("[^/]"),
            '.' | '+' | '(' | ')' | '|' | '^' | '$' | '{' | '}' | '\\' => {
                out.push('\\');
                out.push(c);
            }
            other => out.push(other),
        }
    }
    out.push('$');
    out
}

impl ToolDispatcher for GlobToolDispatcher {
    fn invoke(&self, inv: &ToolInvocation) -> ToolResult {
        let pattern = match inv.args.get("pattern") {
            Some(Value::String(s)) if !s.is_empty() => s.clone(),
            _ => return self.err(E_DISPATCH_MISSING_ARG, "glob requires `pattern`"),
        };
        let re_str = glob_to_regex(&pattern);
        let re = match regex::Regex::new(&re_str) {
            Ok(r) => r,
            Err(e) => return self.err(E_DISPATCH_BAD_PATTERN, format!("regex: {e}")),
        };
        let canonical_root = match self.root.canonicalize() {
            Ok(p) => p,
            Err(e) => {
                return self.err(
                    E_DISPATCH_READ_FAILED,
                    format!("workspace root unreadable: {e}"),
                )
            }
        };
        let mut results: Vec<String> = Vec::new();
        let mut stack = vec![canonical_root.clone()];
        while let Some(dir) = stack.pop() {
            let Ok(read) = std::fs::read_dir(&dir) else {
                continue;
            };
            for entry in read.flatten() {
                let path = entry.path();
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    if name.starts_with('.') || name == "target" || name == "node_modules" {
                        continue;
                    }
                }
                let Ok(ft) = entry.file_type() else { continue };
                if ft.is_dir() {
                    stack.push(path);
                    continue;
                }
                if !ft.is_file() {
                    continue;
                }
                let rel = path.strip_prefix(&canonical_root).unwrap_or(&path);
                let rel_str = rel.display().to_string();
                if re.is_match(&rel_str) {
                    results.push(rel_str);
                    if results.len() >= self.max_results {
                        break;
                    }
                }
            }
            if results.len() >= self.max_results {
                break;
            }
        }
        let total_bytes: u64 = results.iter().map(|s| s.len() as u64).sum();
        let body = serde_json::json!({
            "pattern": pattern,
            "matches": results,
        });
        ToolResult::Ok {
            tool_id: self.id.clone(),
            body,
            bytes: total_bytes,
        }
    }
    fn supports(&self, tool_id: &str) -> bool {
        tool_id == Self::ID
    }
    fn id(&self) -> &str {
        &self.id
    }
}

/// Dispatcher that delegates a single side-task to a registered
/// subagent. Args: `{"name":"<subagent>","task":"<prompt>"}`.
///
/// The dispatcher runs the subagent's prompt body as a system override on
/// the parent's `Provider` (single-shot, no nested tool dispatch — see
/// `plan/37` §2.3 "no nesting" rule). Returns the model's text reply as
/// the tool result.
pub struct SubagentToolDispatcher {
    id: String,
    registry: std::sync::Arc<crate::subagent::SubagentRegistry>,
    provider: std::sync::Arc<dyn crate::provider::Provider>,
    max_blocks: u32,
}

impl std::fmt::Debug for SubagentToolDispatcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SubagentToolDispatcher")
            .field("id", &self.id)
            .field("registry_len", &self.registry.len())
            .finish_non_exhaustive()
    }
}

impl SubagentToolDispatcher {
    /// Stable id used to look this dispatcher up in the registry.
    pub const ID: &'static str = "subagent.run";
    /// Default cap on subagent block emission.
    pub const DEFAULT_MAX_BLOCKS: u32 = 16;

    /// Build a new subagent dispatcher backed by `registry` and `provider`.
    #[must_use]
    pub fn new(
        registry: std::sync::Arc<crate::subagent::SubagentRegistry>,
        provider: std::sync::Arc<dyn crate::provider::Provider>,
    ) -> Self {
        Self {
            id: Self::ID.to_string(),
            registry,
            provider,
            max_blocks: Self::DEFAULT_MAX_BLOCKS,
        }
    }

    fn err(&self, code: &str, message: impl Into<String>) -> ToolResult {
        ToolResult::Err {
            tool_id: self.id.clone(),
            code: code.to_string(),
            message: message.into(),
        }
    }
}

impl ToolDispatcher for SubagentToolDispatcher {
    fn invoke(&self, inv: &ToolInvocation) -> ToolResult {
        let name = match inv.args.get("name") {
            Some(Value::String(s)) if !s.is_empty() => s.clone(),
            _ => return self.err(E_DISPATCH_MISSING_ARG, "subagent.run requires `name`"),
        };
        let task = match inv.args.get("task") {
            Some(Value::String(s)) if !s.is_empty() => s.clone(),
            _ => return self.err(E_DISPATCH_MISSING_ARG, "subagent.run requires `task`"),
        };
        let Some(sub) = self.registry.get(&name) else {
            return self.err(E_DISPATCH_MISSING_ARG, format!("unknown subagent: {name}"));
        };
        let req = crate::provider::GenerateRequest {
            model: stratum_types::ModelId::from("subagent"),
            prompt: task,
            max_blocks: self.max_blocks,
            system_override: Some(sub.prompt.clone()),
            history: Vec::new(),
            sampler: crate::provider::SamplerParams::default(),
        };
        let cancel = crate::cancel::CancelToken::new();
        let blocks = self.provider.generate(&req, &cancel);
        let text: String = blocks
            .iter()
            .filter_map(|b| match b {
                stratum_types::Block::Text { text } => Some(text.clone()),
                _ => None,
            })
            .collect::<String>();
        if text.is_empty() {
            return self.err(E_DISPATCH_READ_FAILED, "subagent returned no text blocks");
        }
        let body = serde_json::json!({
            "subagent": name,
            "answer": text,
        });
        let bytes = text.len() as u64;
        ToolResult::Ok {
            tool_id: self.id.clone(),
            body,
            bytes,
        }
    }

    fn supports(&self, tool_id: &str) -> bool {
        tool_id == Self::ID
    }

    fn id(&self) -> &str {
        &self.id
    }
}

/// Directory-tree listing of the workspace. Cheaper than `glob` for
/// "show me the layout" queries; depth-capped to keep output bounded.
#[derive(Debug, Clone)]
pub struct FsTreeToolDispatcher {
    id: String,
    root: PathBuf,
    /// Maximum tree depth from the workspace root. 0 = root listing only.
    max_depth: u32,
    /// Maximum number of entries returned across the entire walk.
    max_entries: usize,
}

impl FsTreeToolDispatcher {
    /// Stable id used to look this dispatcher up in the registry.
    pub const ID: &'static str = "fs.tree";
    /// Default recursion depth.
    pub const DEFAULT_DEPTH: u32 = 4;
    /// Default ceiling on returned entries.
    pub const DEFAULT_MAX_ENTRIES: usize = 1_000;

    /// Build a new dispatcher anchored at `root`.
    #[must_use]
    pub fn new(root: PathBuf) -> Self {
        Self {
            id: Self::ID.to_string(),
            root,
            max_depth: Self::DEFAULT_DEPTH,
            max_entries: Self::DEFAULT_MAX_ENTRIES,
        }
    }

    fn err(&self, code: &str, message: impl Into<String>) -> ToolResult {
        ToolResult::Err {
            tool_id: self.id.clone(),
            code: code.to_string(),
            message: message.into(),
        }
    }
}

impl ToolDispatcher for FsTreeToolDispatcher {
    fn invoke(&self, inv: &ToolInvocation) -> ToolResult {
        let depth = match inv.args.get("depth") {
            Some(Value::Number(n)) => n.as_u64().map_or(self.max_depth, |x| x.min(16) as u32),
            _ => self.max_depth,
        };
        let canonical_root = match self.root.canonicalize() {
            Ok(p) => p,
            Err(e) => {
                return self.err(
                    E_DISPATCH_READ_FAILED,
                    format!("workspace root unreadable: {e}"),
                )
            }
        };
        let mut entries: Vec<String> = Vec::new();
        let mut stack: Vec<(PathBuf, u32)> = vec![(canonical_root.clone(), 0)];
        let mut total_bytes: u64 = 0;
        while let Some((dir, level)) = stack.pop() {
            if level > depth {
                continue;
            }
            let Ok(read) = std::fs::read_dir(&dir) else {
                continue;
            };
            for entry in read.flatten() {
                if entries.len() >= self.max_entries {
                    break;
                }
                let path = entry.path();
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    if name.starts_with('.') || name == "target" || name == "node_modules" {
                        continue;
                    }
                }
                let rel = path.strip_prefix(&canonical_root).unwrap_or(&path);
                let kind = entry.file_type().ok().map_or(' ', |ft| {
                    if ft.is_dir() {
                        '/'
                    } else if ft.is_symlink() {
                        '@'
                    } else {
                        ' '
                    }
                });
                let line = format!("{}{}", rel.display(), kind);
                total_bytes += line.len() as u64;
                entries.push(line);
                if entry.file_type().is_ok_and(|ft| ft.is_dir()) {
                    stack.push((path, level + 1));
                }
            }
            if entries.len() >= self.max_entries {
                break;
            }
        }
        entries.sort();
        let body = serde_json::json!({
            "depth": depth,
            "entries": entries,
        });
        ToolResult::Ok {
            tool_id: self.id.clone(),
            body,
            bytes: total_bytes,
        }
    }
    fn supports(&self, tool_id: &str) -> bool {
        tool_id == Self::ID
    }
    fn id(&self) -> &str {
        &self.id
    }
}

/// `ToolDispatcher` for `git.diff`. Runs a read-only `git diff` rooted at
/// the workspace and returns the patch text.
///
/// # Arguments
///
/// - `path` (optional, string) — restrict the diff to a single file. Path
///   policy: must be relative, no `..`, canonicalized result must stay
///   inside the workspace root.
/// - `staged` (optional, bool) — true selects `--cached` (the staged
///   diff). Default false (working-tree diff).
/// - `since` (optional, string) — a ref to diff against (e.g. `main` or
///   a commit sha). Validated against `[A-Za-z0-9_./-]{1,80}` to refuse
///   shell metachars and unbounded input.
#[derive(Debug, Clone)]
pub struct GitDiffToolDispatcher {
    id: String,
    root: PathBuf,
    timeout: Duration,
    max_bytes: u64,
}

impl GitDiffToolDispatcher {
    /// Stable id used to look this dispatcher up in the registry.
    pub const ID: &'static str = "git.diff";
    /// Default per-call wall-clock timeout: 30 seconds.
    pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);
    /// Default cap on returned patch size: 2 MiB.
    pub const DEFAULT_MAX_BYTES: u64 = 2 << 20;

    /// Build a new dispatcher anchored at `root`.
    #[must_use]
    pub fn new(root: PathBuf) -> Self {
        Self {
            id: Self::ID.to_string(),
            root,
            timeout: Self::DEFAULT_TIMEOUT,
            max_bytes: Self::DEFAULT_MAX_BYTES,
        }
    }

    fn err(&self, code: &str, message: impl Into<String>) -> ToolResult {
        ToolResult::Err {
            tool_id: self.id.clone(),
            code: code.to_string(),
            message: message.into(),
        }
    }
}

impl ToolDispatcher for GitDiffToolDispatcher {
    fn invoke(&self, inv: &ToolInvocation) -> ToolResult {
        let staged = matches!(inv.args.get("staged"), Some(Value::Bool(true)));
        let since = match inv.args.get("since") {
            Some(Value::String(s)) if !s.is_empty() => Some(s.clone()),
            None | Some(Value::Null) => None,
            _ => return self.err(E_DISPATCH_MISSING_ARG, "git.diff `since` must be string"),
        };
        if let Some(ref s) = since {
            if !is_safe_ref(s) {
                return self.err(E_DISPATCH_MISSING_ARG, "git.diff `since` has unsafe chars");
            }
        }
        let path_arg = match inv.args.get("path") {
            Some(Value::String(s)) if !s.is_empty() => Some(PathBuf::from(s)),
            None | Some(Value::Null) => None,
            _ => return self.err(E_DISPATCH_MISSING_ARG, "git.diff `path` must be string"),
        };
        if let Some(ref p) = path_arg {
            if p.is_absolute() || p.components().any(|c| matches!(c, Component::ParentDir)) {
                return self.err(E_DISPATCH_PATH_ESCAPE, "path escapes workspace root");
            }
        }

        let mut args: Vec<String> = vec!["--no-pager".into(), "diff".into(), "--no-color".into()];
        if staged {
            args.push("--cached".into());
        }
        if let Some(s) = since {
            args.push(s);
        }
        if let Some(p) = path_arg {
            args.push("--".into());
            args.push(p.display().to_string());
        }

        run_git(&self.root, &args, self.timeout, self.max_bytes).map_or_else(
            |e| e.into_tool_result(&self.id),
            |out| ToolResult::Ok {
                tool_id: self.id.clone(),
                body: serde_json::json!({
                    "patch": String::from_utf8_lossy(&out.stdout).into_owned(),
                    "truncated": out.truncated,
                }),
                bytes: out.stdout.len() as u64,
            },
        )
    }

    fn supports(&self, tool_id: &str) -> bool {
        tool_id == Self::ID
    }

    fn id(&self) -> &str {
        &self.id
    }
}

/// `ToolDispatcher` for `git.log`. Returns recent commits in a stable
/// tab-separated shape.
///
/// # Arguments
///
/// - `path` (optional) — restrict the log to a single file
/// - `max` (optional, number) — entries to return; default 20, max 200
/// - `since` (optional, ref-shaped string) — show commits since `<ref>`
#[derive(Debug, Clone)]
pub struct GitLogToolDispatcher {
    id: String,
    root: PathBuf,
    timeout: Duration,
    max_bytes: u64,
}

impl GitLogToolDispatcher {
    /// Stable id used to look this dispatcher up in the registry.
    pub const ID: &'static str = "git.log";
    /// Default per-call wall-clock timeout: 15 seconds.
    pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(15);
    /// Default cap on returned text: 256 KiB.
    pub const DEFAULT_MAX_BYTES: u64 = 256 << 10;
    /// Default number of commits returned.
    pub const DEFAULT_MAX_ENTRIES: u64 = 20;
    /// Hard cap on the per-call `max` argument.
    pub const HARD_MAX_ENTRIES: u64 = 200;

    /// Build a new dispatcher anchored at `root`.
    #[must_use]
    pub fn new(root: PathBuf) -> Self {
        Self {
            id: Self::ID.to_string(),
            root,
            timeout: Self::DEFAULT_TIMEOUT,
            max_bytes: Self::DEFAULT_MAX_BYTES,
        }
    }

    fn err(&self, code: &str, message: impl Into<String>) -> ToolResult {
        ToolResult::Err {
            tool_id: self.id.clone(),
            code: code.to_string(),
            message: message.into(),
        }
    }
}

impl ToolDispatcher for GitLogToolDispatcher {
    fn invoke(&self, inv: &ToolInvocation) -> ToolResult {
        let max = match inv.args.get("max") {
            Some(Value::Number(n)) => n.as_u64().map_or(Self::DEFAULT_MAX_ENTRIES, |x| {
                x.clamp(1, Self::HARD_MAX_ENTRIES)
            }),
            _ => Self::DEFAULT_MAX_ENTRIES,
        };
        let since = match inv.args.get("since") {
            Some(Value::String(s)) if !s.is_empty() => Some(s.clone()),
            None | Some(Value::Null) => None,
            _ => return self.err(E_DISPATCH_MISSING_ARG, "git.log `since` must be string"),
        };
        if let Some(ref s) = since {
            if !is_safe_ref(s) {
                return self.err(E_DISPATCH_MISSING_ARG, "git.log `since` has unsafe chars");
            }
        }
        let path_arg = match inv.args.get("path") {
            Some(Value::String(s)) if !s.is_empty() => Some(PathBuf::from(s)),
            None | Some(Value::Null) => None,
            _ => return self.err(E_DISPATCH_MISSING_ARG, "git.log `path` must be string"),
        };
        if let Some(ref p) = path_arg {
            if p.is_absolute() || p.components().any(|c| matches!(c, Component::ParentDir)) {
                return self.err(E_DISPATCH_PATH_ESCAPE, "path escapes workspace root");
            }
        }

        let max_arg = format!("-n{max}");
        let mut args: Vec<String> = vec![
            "--no-pager".into(),
            "log".into(),
            "--no-color".into(),
            "--date=iso-strict".into(),
            // %x09 = TAB. Stable, machine-parseable shape:
            //   <short-hash>\t<iso-date>\t<author>\t<subject>
            "--pretty=format:%h%x09%ad%x09%an%x09%s".into(),
            max_arg,
        ];
        if let Some(s) = since {
            args.push(s);
        }
        if let Some(p) = path_arg {
            args.push("--".into());
            args.push(p.display().to_string());
        }

        run_git(&self.root, &args, self.timeout, self.max_bytes).map_or_else(
            |e| e.into_tool_result(&self.id),
            |out| {
                let text = String::from_utf8_lossy(&out.stdout).into_owned();
                let commits: Vec<serde_json::Value> = text
                    .lines()
                    .filter_map(|line| {
                        let mut parts = line.splitn(4, '\t');
                        let hash = parts.next()?;
                        let date = parts.next()?;
                        let author = parts.next()?;
                        let subject = parts.next()?;
                        Some(serde_json::json!({
                            "hash": hash,
                            "date": date,
                            "author": author,
                            "subject": subject,
                        }))
                    })
                    .collect();
                let bytes = out.stdout.len() as u64;
                ToolResult::Ok {
                    tool_id: self.id.clone(),
                    body: serde_json::json!({
                        "commits": commits,
                        "truncated": out.truncated,
                    }),
                    bytes,
                }
            },
        )
    }

    fn supports(&self, tool_id: &str) -> bool {
        tool_id == Self::ID
    }

    fn id(&self) -> &str {
        &self.id
    }
}

/// `ToolDispatcher` for `read_image`.
///
/// Returns base64-encoded image bytes plus a sniffed MIME type. The
/// agent loop forwards this through the multimodal provider path; this
/// dispatcher only handles the read.
#[derive(Debug, Clone)]
pub struct ReadImageToolDispatcher {
    id: String,
    root: PathBuf,
    max_bytes: u64,
}

impl ReadImageToolDispatcher {
    /// Stable id used to look this dispatcher up in the registry.
    pub const ID: &'static str = "read_image";
    /// Default cap on a single image read: 5 MiB.
    pub const DEFAULT_MAX_BYTES: u64 = 5 << 20;

    /// Build a new dispatcher anchored at `root`.
    #[must_use]
    pub fn new(root: PathBuf) -> Self {
        Self {
            id: Self::ID.to_string(),
            root,
            max_bytes: Self::DEFAULT_MAX_BYTES,
        }
    }

    fn err(&self, code: &str, message: impl Into<String>) -> ToolResult {
        ToolResult::Err {
            tool_id: self.id.clone(),
            code: code.to_string(),
            message: message.into(),
        }
    }
}

impl ToolDispatcher for ReadImageToolDispatcher {
    fn invoke(&self, inv: &ToolInvocation) -> ToolResult {
        let requested = match inv.args.get("path") {
            Some(Value::String(s)) if !s.is_empty() => PathBuf::from(s),
            _ => return self.err(E_DISPATCH_MISSING_ARG, "read_image requires `path`"),
        };
        if requested.is_absolute()
            || requested
                .components()
                .any(|c| matches!(c, Component::ParentDir))
        {
            return self.err(E_DISPATCH_PATH_ESCAPE, "path escapes workspace root");
        }

        let canonical_root = match self.root.canonicalize() {
            Ok(p) => p,
            Err(e) => {
                return self.err(
                    E_DISPATCH_READ_FAILED,
                    format!("workspace root unreadable: {e}"),
                )
            }
        };
        // Join against the *canonical* root, not the raw `self.root` —
        // matters if `self.root` is itself a symlink that resolves
        // elsewhere, so we evaluate the descent from a stable anchor.
        let joined = canonical_root.join(&requested);
        let canonical_target = match joined.canonicalize() {
            Ok(p) => p,
            Err(e) => return self.err(E_DISPATCH_READ_FAILED, format!("path unreadable: {e}")),
        };
        if !canonical_target.starts_with(&canonical_root) {
            return self.err(E_DISPATCH_PATH_ESCAPE, "path escapes workspace root");
        }

        let metadata = match std::fs::metadata(&canonical_target) {
            Ok(m) => m,
            Err(e) => return self.err(E_DISPATCH_READ_FAILED, format!("stat failed: {e}")),
        };
        if metadata.len() > self.max_bytes {
            return self.err(
                E_DISPATCH_SIZE_CAP,
                format!("image exceeds {} byte cap", self.max_bytes),
            );
        }
        let bytes = match std::fs::read(&canonical_target) {
            Ok(b) => b,
            Err(e) => return self.err(E_DISPATCH_READ_FAILED, format!("read failed: {e}")),
        };
        let mime =
            sniff_image_mime(&canonical_target, &bytes).unwrap_or("application/octet-stream");
        let raw_len = bytes.len() as u64;
        let encoded = base64_encode(&bytes);
        let body = serde_json::json!({
            "path": canonical_target.display().to_string(),
            "mime": mime,
            "data_base64": encoded,
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

/// Build the default dispatcher registry used by the agent loop.
/// Includes: `fs.read`, `fs.write`, `fs.edit`, `fs.tree`, `grep`,
/// `glob`, `shell.exec`, `git.diff`, `git.log`, `read_image`.
#[must_use]
pub fn default_dispatchers(
    workspace_root: PathBuf,
    sandbox: SandboxSpawn,
    base_spec: SandboxLaunchSpec,
) -> RegistryDispatcher {
    let mut reg = RegistryDispatcher::new();
    let fs_read = FsReadToolDispatcher::new(workspace_root.clone());
    let fs_write = FsWriteToolDispatcher::new(workspace_root.clone());
    let fs_edit = FsEditToolDispatcher::new(workspace_root.clone());
    let fs_tree = FsTreeToolDispatcher::new(workspace_root.clone());
    let grep = GrepToolDispatcher::new(workspace_root.clone());
    let glob = GlobToolDispatcher::new(workspace_root.clone());
    let shell = ShellToolDispatcher::new(sandbox, base_spec);
    let git_diff = GitDiffToolDispatcher::new(workspace_root.clone());
    let git_log = GitLogToolDispatcher::new(workspace_root.clone());
    let read_image = ReadImageToolDispatcher::new(workspace_root);
    let _ = reg.register(Box::new(fs_read));
    let _ = reg.register(Box::new(fs_write));
    let _ = reg.register(Box::new(fs_edit));
    let _ = reg.register(Box::new(fs_tree));
    let _ = reg.register(Box::new(grep));
    let _ = reg.register(Box::new(glob));
    let _ = reg.register(Box::new(shell));
    let _ = reg.register(Box::new(git_diff));
    let _ = reg.register(Box::new(git_log));
    let _ = reg.register(Box::new(read_image));
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

// ---- git helper ----------------------------------------------------------

/// Captured outcome of a successful `run_git` invocation.
struct GitOutput {
    stdout: Vec<u8>,
    truncated: bool,
}

enum GitRunError {
    Spawn(std::io::Error),
    Timeout,
    NonZero { status: i32, stderr_tail: String },
    Wait(std::io::Error),
}

impl GitRunError {
    fn into_tool_result(self, tool_id: &str) -> ToolResult {
        let (code, message) = match self {
            Self::Spawn(e) => (E_DISPATCH_SPAWN_FAILED, format!("git spawn failed: {e}")),
            Self::Timeout => (E_DISPATCH_TIMEOUT, "git timeout".to_string()),
            Self::NonZero {
                status,
                stderr_tail,
            } => (
                E_DISPATCH_EXIT_NONZERO,
                format!("git exit {status}: {stderr_tail}"),
            ),
            Self::Wait(e) => (E_DISPATCH_SPAWN_FAILED, format!("git wait failed: {e}")),
        };
        ToolResult::Err {
            tool_id: tool_id.to_string(),
            code: code.to_string(),
            message,
        }
    }
}

/// Run `git <args>` rooted at `cwd` with a wall-clock `timeout`. Truncates
/// captured stdout at `max_bytes` and reports it via [`GitOutput::truncated`].
///
/// Environment is reset to a minimal allowlist so the child's behavior is
/// reproducible and does not leak the caller's shell aliases. `GIT_PAGER`
/// is forced empty in case the caller forgot `--no-pager`.
fn run_git(
    cwd: &Path,
    args: &[String],
    timeout: Duration,
    max_bytes: u64,
) -> Result<GitOutput, GitRunError> {
    let Some(git_bin) = which_in_path("git") else {
        return Err(GitRunError::Spawn(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "git not on PATH",
        )));
    };
    // Neutralize hook-execution surfaces in ~/.gitconfig and
    // /etc/gitconfig — `diff.external`, custom `core.pager`,
    // credential helpers, etc. — even when HOME is honored. Without
    // these, env_clear is not enough: git still reads the user's
    // global config and would invoke any external command it names.
    let mut cmd = std::process::Command::new(&git_bin);
    cmd.args(args)
        .current_dir(cwd)
        .env_clear()
        .env("PATH", std::env::var_os("PATH").unwrap_or_default())
        .env("HOME", std::env::var_os("HOME").unwrap_or_default())
        .env("GIT_PAGER", "")
        .env("GIT_CONFIG_NOSYSTEM", "1")
        .env("GIT_CONFIG_GLOBAL", "/dev/null")
        .env("LANG", "C.UTF-8")
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    let mut child = cmd.spawn().map_err(GitRunError::Spawn)?;
    match wait_with_timeout(&mut child, timeout) {
        WaitOutcome::Exited {
            status,
            stdout,
            stderr,
        } => {
            if status == 0 {
                let truncated = (stdout.len() as u64) > max_bytes;
                let capped = if truncated {
                    stdout
                        .into_iter()
                        .take(usize::try_from(max_bytes).unwrap_or(usize::MAX))
                        .collect()
                } else {
                    stdout
                };
                Ok(GitOutput {
                    stdout: capped,
                    truncated,
                })
            } else {
                let tail = String::from_utf8_lossy(&stderr).into_owned();
                Err(GitRunError::NonZero {
                    status,
                    stderr_tail: tail_text(&tail, 256),
                })
            }
        }
        WaitOutcome::Timeout => {
            let _ = child.kill();
            let _ = child.wait();
            Err(GitRunError::Timeout)
        }
        WaitOutcome::WaitFailed(e) => Err(GitRunError::Wait(e)),
    }
}

/// Refs (sha, branch, tag) accept `[A-Za-z0-9._/-]` and must be 1..=80
/// chars. Refuses leading `-` so it can't be parsed as a flag, refuses
/// `..` which `git` treats as a range we cannot validate further, and
/// refuses anything starting or ending with `/` so a value like `/` or
/// `v1/` is not silently re-interpreted as a pathspec by git.
fn is_safe_ref(s: &str) -> bool {
    if s.is_empty()
        || s.len() > 80
        || s.starts_with('-')
        || s.starts_with('/')
        || s.ends_with('/')
        || s.contains("..")
    {
        return false;
    }
    s.chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '/' | '-'))
}

// ---- image helpers -------------------------------------------------------

fn sniff_image_mime(path: &Path, bytes: &[u8]) -> Option<&'static str> {
    // Magic-byte sniff first; fall back to extension if the bytes are
    // ambiguous. Order matters — JPEG has multiple framings.
    if bytes.starts_with(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]) {
        return Some("image/png");
    }
    if bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
        return Some("image/jpeg");
    }
    if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
        return Some("image/gif");
    }
    if bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
        return Some("image/webp");
    }
    if bytes.starts_with(b"BM") {
        return Some("image/bmp");
    }
    match path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("png") => Some("image/png"),
        Some("jpg" | "jpeg") => Some("image/jpeg"),
        Some("gif") => Some("image/gif"),
        Some("webp") => Some("image/webp"),
        Some("bmp") => Some("image/bmp"),
        Some("svg") => Some("image/svg+xml"),
        _ => None,
    }
}

/// Standard alphabet base64 with `=` padding. Kept inline to avoid pulling
/// the `base64` crate into the runtime dep graph for a single use site.
fn base64_encode(bytes: &[u8]) -> String {
    const ALPHA: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    let mut iter = bytes.chunks_exact(3);
    for chunk in &mut iter {
        let b0 = chunk[0];
        let b1 = chunk[1];
        let b2 = chunk[2];
        out.push(ALPHA[(b0 >> 2) as usize] as char);
        out.push(ALPHA[(((b0 & 0b11) << 4) | (b1 >> 4)) as usize] as char);
        out.push(ALPHA[(((b1 & 0b1111) << 2) | (b2 >> 6)) as usize] as char);
        out.push(ALPHA[(b2 & 0b11_1111) as usize] as char);
    }
    let rem = iter.remainder();
    match rem.len() {
        0 => {}
        1 => {
            let b0 = rem[0];
            out.push(ALPHA[(b0 >> 2) as usize] as char);
            out.push(ALPHA[((b0 & 0b11) << 4) as usize] as char);
            out.push('=');
            out.push('=');
        }
        2 => {
            let b0 = rem[0];
            let b1 = rem[1];
            out.push(ALPHA[(b0 >> 2) as usize] as char);
            out.push(ALPHA[(((b0 & 0b11) << 4) | (b1 >> 4)) as usize] as char);
            out.push(ALPHA[((b1 & 0b1111) << 2) as usize] as char);
            out.push('=');
        }
        _ => unreachable!(),
    }
    out
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
    fn default_dispatchers_registers_all() {
        let tmp = TempDir::new().expect("tmp");
        let reg = default_dispatchers(
            tmp.path().to_path_buf(),
            SandboxSpawn::new(SandboxBackend::Passthrough),
            passthrough_spec_in(tmp.path()),
        );
        let ids = reg.ids();
        assert_eq!(ids.len(), 10);
        assert!(ids.contains(&"fs.read"));
        assert!(ids.contains(&"fs.write"));
        assert!(ids.contains(&"fs.edit"));
        assert!(ids.contains(&"fs.tree"));
        assert!(ids.contains(&"grep"));
        assert!(ids.contains(&"glob"));
        assert!(ids.contains(&"shell.exec"));
        assert!(ids.contains(&"git.diff"));
        assert!(ids.contains(&"git.log"));
        assert!(ids.contains(&"read_image"));
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

    // ---- git.diff / git.log / read_image -----------------------------

    fn make_invocation(tool: &str, args: BTreeMap<String, serde_json::Value>) -> ToolInvocation {
        ToolInvocation {
            tool_id: tool.to_string(),
            args,
            capability: tool.to_string(),
            turn_id: 1,
        }
    }

    fn git_init_tmp() -> Option<TempDir> {
        // Skip when `git` is not installed (CI sandboxes without git).
        which_in_path("git")?;
        let tmp = TempDir::new().ok()?;
        let ok = std::process::Command::new("git")
            .args(["init", "-q", "-b", "main"])
            .current_dir(tmp.path())
            .status()
            .ok()?
            .success();
        if !ok {
            return None;
        }
        // Per-repo identity — never `git config --global`.
        for (k, v) in [
            ("user.email", "test@example.com"),
            ("user.name", "Test"),
            ("commit.gpgsign", "false"),
        ] {
            let _ = std::process::Command::new("git")
                .args(["config", k, v])
                .current_dir(tmp.path())
                .status();
        }
        Some(tmp)
    }

    fn git_commit_file(tmp: &Path, name: &str, body: &str, msg: &str) {
        fs::write(tmp.join(name), body).expect("write");
        assert!(std::process::Command::new("git")
            .args(["add", name])
            .current_dir(tmp)
            .status()
            .expect("git add")
            .success());
        assert!(std::process::Command::new("git")
            .args(["commit", "-q", "-m", msg])
            .current_dir(tmp)
            .status()
            .expect("git commit")
            .success());
    }

    #[test]
    fn is_safe_ref_accepts_typical_refs() {
        assert!(is_safe_ref("main"));
        assert!(is_safe_ref("origin/main"));
        assert!(is_safe_ref("v1.2.3"));
        assert!(is_safe_ref("feature/new-thing"));
        assert!(is_safe_ref("abc123"));
    }

    #[test]
    fn is_safe_ref_rejects_metachars_and_ranges() {
        assert!(!is_safe_ref(""));
        assert!(!is_safe_ref("-flag"));
        assert!(!is_safe_ref("a..b"));
        assert!(!is_safe_ref("HEAD;rm -rf /"));
        assert!(!is_safe_ref("--all"));
        assert!(!is_safe_ref(&"a".repeat(200)));
        // A leading or trailing slash trips git's pathspec heuristic
        // on some platforms — refuse so the parser stays in revision
        // mode unambiguously.
        assert!(!is_safe_ref("/"));
        assert!(!is_safe_ref("/main"));
        assert!(!is_safe_ref("v1.0/"));
    }

    #[test]
    fn git_log_returns_commit_rows() {
        let Some(tmp) = git_init_tmp() else { return };
        git_commit_file(tmp.path(), "a.txt", "one\n", "add a");
        git_commit_file(tmp.path(), "b.txt", "two\n", "add b");
        let dispatcher = GitLogToolDispatcher::new(tmp.path().to_path_buf());
        let inv = make_invocation("git.log", BTreeMap::new());
        match dispatcher.invoke(&inv) {
            ToolResult::Ok { body, .. } => {
                let commits = body
                    .get("commits")
                    .and_then(|v| v.as_array())
                    .expect("commits array");
                assert_eq!(commits.len(), 2);
                assert_eq!(
                    commits[0].get("subject").and_then(|v| v.as_str()),
                    Some("add b"),
                );
                assert_eq!(
                    commits[1].get("subject").and_then(|v| v.as_str()),
                    Some("add a"),
                );
            }
            ToolResult::Err { code, message, .. } => {
                panic!("expected Ok, got Err({code}): {message}")
            }
        }
    }

    #[test]
    fn git_log_rejects_unsafe_since() {
        let tmp = TempDir::new().expect("tmp");
        let dispatcher = GitLogToolDispatcher::new(tmp.path().to_path_buf());
        let mut args = BTreeMap::new();
        args.insert("since".to_string(), serde_json::json!("--all"));
        let inv = make_invocation("git.log", args);
        assert_err_code(dispatcher.invoke(&inv), E_DISPATCH_MISSING_ARG);
    }

    #[test]
    fn git_log_rejects_path_traversal() {
        let tmp = TempDir::new().expect("tmp");
        let dispatcher = GitLogToolDispatcher::new(tmp.path().to_path_buf());
        let mut args = BTreeMap::new();
        args.insert("path".to_string(), serde_json::json!("../etc/passwd"));
        let inv = make_invocation("git.log", args);
        assert_err_code(dispatcher.invoke(&inv), E_DISPATCH_PATH_ESCAPE);
    }

    #[test]
    fn git_diff_shows_unstaged_changes() {
        let Some(tmp) = git_init_tmp() else { return };
        git_commit_file(tmp.path(), "a.txt", "one\n", "add a");
        fs::write(tmp.path().join("a.txt"), "one\ntwo\n").expect("dirty write");
        let dispatcher = GitDiffToolDispatcher::new(tmp.path().to_path_buf());
        let inv = make_invocation("git.diff", BTreeMap::new());
        match dispatcher.invoke(&inv) {
            ToolResult::Ok { body, .. } => {
                let patch = body.get("patch").and_then(|v| v.as_str()).unwrap_or("");
                assert!(patch.contains("a.txt"));
                assert!(patch.contains("+two"));
            }
            ToolResult::Err { code, message, .. } => {
                panic!("expected Ok, got Err({code}): {message}")
            }
        }
    }

    #[test]
    fn git_diff_rejects_unsafe_since() {
        let tmp = TempDir::new().expect("tmp");
        let dispatcher = GitDiffToolDispatcher::new(tmp.path().to_path_buf());
        let mut args = BTreeMap::new();
        args.insert("since".to_string(), serde_json::json!("HEAD;rm -rf /"));
        let inv = make_invocation("git.diff", args);
        assert_err_code(dispatcher.invoke(&inv), E_DISPATCH_MISSING_ARG);
    }

    #[test]
    fn read_image_returns_base64_and_sniffs_png() {
        let tmp = TempDir::new().expect("tmp");
        // Minimal PNG header. 8-byte signature, IHDR, IDAT skipped — we
        // only assert on the MIME sniff path, not the round-trip.
        let png_bytes: [u8; 16] = [
            0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, b'I', b'H',
            b'D', b'R',
        ];
        fs::write(tmp.path().join("hi.png"), png_bytes).expect("write");
        let dispatcher = ReadImageToolDispatcher::new(tmp.path().to_path_buf());
        let mut args = BTreeMap::new();
        args.insert("path".to_string(), serde_json::json!("hi.png"));
        let inv = make_invocation("read_image", args);
        match dispatcher.invoke(&inv) {
            ToolResult::Ok { body, .. } => {
                assert_eq!(body.get("mime").and_then(|v| v.as_str()), Some("image/png"));
                let b64 = body
                    .get("data_base64")
                    .and_then(|v| v.as_str())
                    .expect("data_base64");
                // PNG signature in base64 is "iVBORw0KGgo".
                assert!(b64.starts_with("iVBORw0KGgo"));
            }
            ToolResult::Err { code, message, .. } => {
                panic!("expected Ok, got Err({code}): {message}")
            }
        }
    }

    #[test]
    fn read_image_rejects_oversize() {
        let tmp = TempDir::new().expect("tmp");
        let blob = vec![0u8; 2 * 1024 * 1024];
        fs::write(tmp.path().join("big.bin"), &blob).expect("write");
        let dispatcher = ReadImageToolDispatcher::new(tmp.path().to_path_buf());
        // Pull the cap down so we exercise the size-cap path
        // deterministically without writing megabytes.
        let dispatcher = ReadImageToolDispatcher {
            max_bytes: 1024,
            ..dispatcher
        };
        let mut args = BTreeMap::new();
        args.insert("path".to_string(), serde_json::json!("big.bin"));
        let inv = make_invocation("read_image", args);
        assert_err_code(dispatcher.invoke(&inv), E_DISPATCH_SIZE_CAP);
    }

    #[test]
    fn read_image_rejects_path_traversal() {
        let tmp = TempDir::new().expect("tmp");
        let dispatcher = ReadImageToolDispatcher::new(tmp.path().to_path_buf());
        let mut args = BTreeMap::new();
        args.insert("path".to_string(), serde_json::json!("../etc/passwd"));
        let inv = make_invocation("read_image", args);
        assert_err_code(dispatcher.invoke(&inv), E_DISPATCH_PATH_ESCAPE);
    }

    #[test]
    fn base64_encode_matches_known_vectors() {
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(base64_encode(b"foob"), "Zm9vYg==");
        assert_eq!(base64_encode(b"fooba"), "Zm9vYmE=");
        assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn sniff_image_mime_falls_back_to_extension() {
        let p = PathBuf::from("nope.svg");
        assert_eq!(sniff_image_mime(&p, b"<svg></svg>"), Some("image/svg+xml"));
        let p = PathBuf::from("unknown.bin");
        assert_eq!(sniff_image_mime(&p, b"random"), None);
    }
}
