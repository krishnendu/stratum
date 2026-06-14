//! Sandbox backend detection.
//!
//! Phase 2 v2 ships **detection only**; the concrete sandbox profile
//! bodies (`bwrap-strict`, `bwrap-net`, `macos-strict`, `passthrough`)
//! land with the tool registry in Phase 3 per
//! `plan/31-tool-sandbox-and-secrets.md`.
//!
//! The detector reports which backends are available on the host so
//! `stratum doctor` and the orchestrator can pick a profile at runtime.

use std::path::Path;
use std::process::{Command, Stdio};

use serde::{Deserialize, Serialize};

use crate::sandbox_resolve::{MountMode, ResolvedNet, SandboxLaunchSpec};

/// One sandbox backend Stratum can drive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SandboxBackend {
    /// `bubblewrap` on Linux user-namespace sandboxes.
    Bwrap,
    /// macOS `sandbox-exec` profile generator.
    SandboxExec,
    /// Windows Job Object + `AppContainer` (Phase 4+).
    WindowsJob,
    /// No isolation. Always available; the default fallback.
    Passthrough,
}

impl std::fmt::Display for SandboxBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let label = match self {
            Self::Bwrap => "bwrap",
            Self::SandboxExec => "sandbox-exec",
            Self::WindowsJob => "windows-job",
            Self::Passthrough => "passthrough",
        };
        f.write_str(label)
    }
}

/// Result of a sandbox detection run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SandboxReport {
    /// Every backend the host can drive, sorted from strongest to
    /// weakest (the orchestrator picks the head of the list by default).
    pub available: Vec<SandboxBackend>,
}

impl SandboxReport {
    /// Probe the live host: check for `bubblewrap` and `sandbox-exec`
    /// in `PATH`, plus the always-on `Passthrough` fallback.
    #[must_use]
    pub fn run() -> Self {
        let os = std::env::consts::OS;
        let path = std::env::var_os("PATH").unwrap_or_default();
        Self::evaluate(os, |cmd| has_in_path(cmd, &path))
    }

    /// Pure variant for testing: takes the target OS and a closure that
    /// reports whether a given binary is available, so every branch can
    /// be exercised without poking at the host.
    #[must_use]
    pub fn evaluate<F>(os: &str, has_bin: F) -> Self
    where
        F: Fn(&str) -> bool,
    {
        let mut out = Vec::with_capacity(3);
        match os {
            "linux" => {
                if has_bin("bwrap") {
                    out.push(SandboxBackend::Bwrap);
                }
            }
            "macos" => {
                if has_bin("sandbox-exec") {
                    out.push(SandboxBackend::SandboxExec);
                }
            }
            "windows" => {
                // Windows Job Object isolation is in-process; no binary
                // probe needed. Mark as available; profile body lands
                // with Phase 4+ work.
                out.push(SandboxBackend::WindowsJob);
            }
            _ => {}
        }
        out.push(SandboxBackend::Passthrough);
        Self { available: out }
    }

    /// Strongest backend on the host (head of `available`).
    #[must_use]
    pub fn preferred(&self) -> SandboxBackend {
        self.available
            .first()
            .copied()
            .unwrap_or(SandboxBackend::Passthrough)
    }
}

fn has_in_path(cmd: &str, path_env: &std::ffi::OsStr) -> bool {
    if path_env.is_empty() {
        return false;
    }
    std::env::split_paths(path_env).any(|dir| {
        let mut candidate = dir;
        candidate.push(cmd);
        candidate.is_file()
    })
}

/// Look up `helper` in the current `PATH`. Returns true if any
/// directory in `PATH` contains a regular file with that name.
#[must_use]
pub fn detect_helper_present(helper: &str) -> bool {
    let path = std::env::var_os("PATH").unwrap_or_default();
    has_in_path(helper, &path)
}

/// Errors emitted by [`SandboxSpawn::spawn`].
#[derive(Debug)]
pub enum SandboxSpawnError {
    /// `std::process::Command::spawn` itself failed.
    Spawn(std::io::Error),
    /// The requested backend is not implemented on this platform.
    Unsupported(&'static str),
    /// The synthesised sandbox profile could not be written or formed.
    BadProfile(String),
    /// The platform helper binary (e.g. `bwrap`, `sandbox-exec`) is
    /// missing from `PATH`.
    MissingHelper {
        /// Name of the missing helper.
        helper: &'static str,
    },
}

impl std::fmt::Display for SandboxSpawnError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Spawn(e) => write!(f, "sandbox spawn failed: {e}"),
            Self::Unsupported(b) => write!(f, "sandbox backend not supported: {b}"),
            Self::BadProfile(msg) => write!(f, "sandbox profile invalid: {msg}"),
            Self::MissingHelper { helper } => {
                write!(f, "sandbox helper missing from PATH: {helper}")
            }
        }
    }
}

impl std::error::Error for SandboxSpawnError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Spawn(e) => Some(e),
            _ => None,
        }
    }
}

/// Spawner that turns a [`SandboxLaunchSpec`] plus a concrete command
/// into a running child process under the chosen backend.
#[derive(Debug, Clone, Copy)]
pub struct SandboxSpawn {
    backend: SandboxBackend,
}

impl SandboxSpawn {
    /// Build a spawner pinned to `backend`.
    #[must_use]
    pub const fn new(backend: SandboxBackend) -> Self {
        Self { backend }
    }

    /// Backend this spawner is pinned to.
    #[must_use]
    pub const fn backend(self) -> SandboxBackend {
        self.backend
    }

    /// Spawn `command` with `args` under the configured backend.
    ///
    /// # Errors
    ///
    /// Returns [`SandboxSpawnError::MissingHelper`] when the platform
    /// helper binary is missing, [`SandboxSpawnError::Unsupported`] when
    /// the backend has no implementation on the current host (e.g.
    /// Windows-Job), [`SandboxSpawnError::BadProfile`] when the
    /// synthesised macOS profile cannot be persisted, and
    /// [`SandboxSpawnError::Spawn`] when the underlying
    /// `std::process::Command::spawn` call itself fails.
    pub fn spawn(
        self,
        spec: &SandboxLaunchSpec,
        command: &Path,
        args: &[String],
    ) -> Result<std::process::Child, SandboxSpawnError> {
        match self.backend {
            SandboxBackend::Bwrap => spawn_bwrap(spec, command, args),
            SandboxBackend::SandboxExec => spawn_sandbox_exec(spec, command, args),
            SandboxBackend::WindowsJob => spawn_windows_job(spec, command, args),
            SandboxBackend::Passthrough => spawn_passthrough(spec, command, args),
        }
    }
}

/// Build the bwrap argument vector for a given spec + command.
///
/// Pulled out so tests can assert on the assembled flags without
/// actually invoking `bwrap`.
fn build_bwrap_args(spec: &SandboxLaunchSpec, command: &Path, cmd_args: &[String]) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    // Baseline read-only binds for system libraries.
    for base in ["/usr", "/lib", "/lib64"] {
        out.push("--ro-bind".into());
        out.push(base.into());
        out.push(base.into());
    }
    // Per-mount entries.
    for m in &spec.mounts {
        match &m.mode {
            MountMode::ReadOnly => {
                out.push("--ro-bind".into());
                out.push(m.host.display().to_string());
                out.push(m.guest.display().to_string());
            }
            MountMode::ReadWrite => {
                out.push("--bind".into());
                out.push(m.host.display().to_string());
                out.push(m.guest.display().to_string());
            }
            MountMode::TempFs { .. } => {
                out.push("--tmpfs".into());
                out.push(m.guest.display().to_string());
            }
        }
    }
    // Net policy.
    match &spec.net {
        ResolvedNet::Off | ResolvedNet::Loopback => {
            out.push("--unshare-net".into());
        }
        ResolvedNet::Hosts { .. } => {
            // Hosts net keeps the host network namespace; deeper
            // filtering is deferred to a future net-policy layer.
        }
    }
    // Working directory.
    out.push("--chdir".into());
    out.push(spec.working_dir.display().to_string());
    // Env: clear then set each entry explicitly.
    out.push("--clearenv".into());
    for (k, v) in &spec.env {
        out.push("--setenv".into());
        out.push(k.clone());
        out.push(v.clone());
    }
    // Command terminator.
    out.push("--".into());
    out.push(command.display().to_string());
    for a in cmd_args {
        out.push(a.clone());
    }
    out
}

fn spawn_bwrap(
    spec: &SandboxLaunchSpec,
    command: &Path,
    cmd_args: &[String],
) -> Result<std::process::Child, SandboxSpawnError> {
    spawn_bwrap_with("bwrap", spec, command, cmd_args)
}

/// Inner bwrap spawn parameterised on the helper-binary name. Tests
/// inject a guaranteed-present binary so the post-presence-check spawn
/// path is exercised on hosts that do not ship `bwrap`.
fn spawn_bwrap_with(
    helper: &'static str,
    spec: &SandboxLaunchSpec,
    command: &Path,
    cmd_args: &[String],
) -> Result<std::process::Child, SandboxSpawnError> {
    if !detect_helper_present(helper) {
        return Err(SandboxSpawnError::MissingHelper { helper });
    }
    let bwrap_argv = build_bwrap_args(spec, command, cmd_args);
    spawn_program(helper, &bwrap_argv)
}

/// Run `Command::new(prog).args(argv).spawn()` and lift the IO error
/// into [`SandboxSpawnError::Spawn`]. Extracted so the post-presence-
/// check spawn path is exercised by tests that don't depend on bwrap
/// being installed (the same shim drives sandbox-exec, and the unit
/// test below feeds it a `/bin/true` / `/usr/bin/true` directly).
fn spawn_program(prog: &str, argv: &[String]) -> Result<std::process::Child, SandboxSpawnError> {
    Command::new(prog)
        .args(argv)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .stdin(Stdio::null())
        .spawn()
        .map_err(SandboxSpawnError::Spawn)
}

/// Render an SBPL-style profile body for `sandbox-exec`.
fn render_sandbox_exec_profile(spec: &SandboxLaunchSpec) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    out.push_str("(version 1)\n");
    out.push_str("(deny default)\n");
    // Baseline reads: dyld + libc need broad read access on macOS to
    // resolve frameworks, locale files, and timezone data living under
    // /usr, /System, /Library, /private, /var, /etc. Listing them all
    // is brittle (dyld closures, code-signing receipts, etc. move
    // around between point releases), so the scaffold keeps reads open
    // and locks down *writes* per-mount instead.
    out.push_str("(allow file-read*)\n");
    // Keep the per-root mentions in place as documentation anchors:
    // future tightening will swap the blanket allow above for these.
    out.push_str("(allow file-read* (subpath \"/usr\"))\n");
    out.push_str("(allow file-read* (subpath \"/System\"))\n");
    out.push_str("(allow file-read* (subpath \"/Library\"))\n");
    // Process control: needed to actually fork into the target binary.
    out.push_str("(allow process-fork)\n");
    out.push_str("(allow process-exec)\n");
    out.push_str("(allow signal (target self))\n");
    // Mach lookups, sysctl reads, and POSIX SHM are required for any
    // binary on macOS to find its libc / launch services; without them
    // even `/usr/bin/true` aborts with SIGABRT before main() runs.
    out.push_str("(allow mach-lookup)\n");
    out.push_str("(allow sysctl-read)\n");
    out.push_str("(allow ipc-posix-shm)\n");
    // Per-mount permissions. ReadOnly grants read-subpath, ReadWrite
    // grants read+write subpath, TempFs grants write under the guest.
    for m in &spec.mounts {
        let path = m.guest.display();
        match m.mode {
            MountMode::ReadOnly => {
                let _ = writeln!(out, "(allow file-read* (subpath \"{path}\"))");
            }
            MountMode::ReadWrite => {
                let _ = writeln!(out, "(allow file-read* (subpath \"{path}\"))");
                let _ = writeln!(out, "(allow file-write* (subpath \"{path}\"))");
            }
            MountMode::TempFs { .. } => {
                let _ = writeln!(out, "(allow file-write* (subpath \"{path}\"))");
            }
        }
    }
    // Network policy.
    match &spec.net {
        ResolvedNet::Off => {
            out.push_str("(deny network*)\n");
        }
        ResolvedNet::Loopback => {
            out.push_str("(deny network*)\n");
            out.push_str("(allow network* (local ip))\n");
            out.push_str("(allow network* (remote ip \"localhost:*\"))\n");
        }
        ResolvedNet::Hosts { .. } => {
            // Hosts-net allowlists are evaluated at the resolver/proxy
            // layer; the SBPL fragment leaves egress open.
            out.push_str("(allow network*)\n");
        }
    }
    out
}

fn spawn_sandbox_exec(
    spec: &SandboxLaunchSpec,
    command: &Path,
    cmd_args: &[String],
) -> Result<std::process::Child, SandboxSpawnError> {
    spawn_sandbox_exec_with("sandbox-exec", spec, command, cmd_args)
}

/// Inner sandbox-exec spawn parameterised on the helper-binary name.
/// Tests inject a guaranteed-present binary so the profile-writing and
/// post-presence-check spawn path is exercised on hosts that do not
/// ship `sandbox-exec`.
fn spawn_sandbox_exec_with(
    helper: &'static str,
    spec: &SandboxLaunchSpec,
    command: &Path,
    cmd_args: &[String],
) -> Result<std::process::Child, SandboxSpawnError> {
    if !detect_helper_present(helper) {
        return Err(SandboxSpawnError::MissingHelper { helper });
    }
    let body = render_sandbox_exec_profile(spec);
    let profile_path = unique_temp_profile_path();
    std::fs::write(&profile_path, body)
        .map_err(|e| SandboxSpawnError::BadProfile(format!("write profile: {e}")))?;

    let mut argv: Vec<String> = Vec::with_capacity(3 + cmd_args.len());
    argv.push("-f".into());
    argv.push(profile_path.display().to_string());
    argv.push(command.display().to_string());
    for a in cmd_args {
        argv.push(a.clone());
    }
    spawn_program(helper, &argv)
}

/// Produce a unique path under the system temp dir for a profile blob.
///
/// Stdlib-only: combines `temp_dir`, the process id, and a monotonic
/// counter to avoid collisions inside a single test binary run.
fn unique_temp_profile_path() -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    let mut p = std::env::temp_dir();
    p.push(format!("stratum-sandbox-{pid}-{n}.sb"));
    p
}

const fn spawn_windows_job(
    _spec: &SandboxLaunchSpec,
    _command: &Path,
    _args: &[String],
) -> Result<std::process::Child, SandboxSpawnError> {
    Err(SandboxSpawnError::Unsupported("windows-job"))
}

fn spawn_passthrough(
    spec: &SandboxLaunchSpec,
    command: &Path,
    cmd_args: &[String],
) -> Result<std::process::Child, SandboxSpawnError> {
    let mut cmd = Command::new(command);
    cmd.args(cmd_args);
    cmd.current_dir(&spec.working_dir);
    cmd.env_clear();
    for (k, v) in &spec.env {
        cmd.env(k, v);
    }
    // Pipe stdout / stderr so the dispatcher can capture them. Without
    // this the child inherits the parent's TTY and writes directly to
    // the screen — fatal under the TUI alternate-screen / raw-mode
    // session because the bytes bypass ratatui's render buffer and
    // corrupt the layout.
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());
    cmd.stdin(Stdio::null());
    cmd.spawn().map_err(SandboxSpawnError::Spawn)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_renders_lowercase() {
        assert_eq!(format!("{}", SandboxBackend::Bwrap), "bwrap");
        assert_eq!(format!("{}", SandboxBackend::SandboxExec), "sandbox-exec");
        assert_eq!(format!("{}", SandboxBackend::WindowsJob), "windows-job");
        assert_eq!(format!("{}", SandboxBackend::Passthrough), "passthrough");
    }

    #[test]
    fn backend_serde_roundtrip() {
        for b in [
            SandboxBackend::Bwrap,
            SandboxBackend::SandboxExec,
            SandboxBackend::WindowsJob,
            SandboxBackend::Passthrough,
        ] {
            let s = serde_json::to_string(&b).unwrap();
            let back: SandboxBackend = serde_json::from_str(&s).unwrap();
            assert_eq!(b, back);
        }
    }

    #[test]
    fn evaluate_linux_with_bwrap() {
        let r = SandboxReport::evaluate("linux", |c| c == "bwrap");
        assert_eq!(
            r.available,
            vec![SandboxBackend::Bwrap, SandboxBackend::Passthrough]
        );
        assert_eq!(r.preferred(), SandboxBackend::Bwrap);
    }

    #[test]
    fn evaluate_linux_without_bwrap() {
        let r = SandboxReport::evaluate("linux", |_| false);
        assert_eq!(r.available, vec![SandboxBackend::Passthrough]);
        assert_eq!(r.preferred(), SandboxBackend::Passthrough);
    }

    #[test]
    fn evaluate_macos_with_sandbox_exec() {
        let r = SandboxReport::evaluate("macos", |c| c == "sandbox-exec");
        assert_eq!(
            r.available,
            vec![SandboxBackend::SandboxExec, SandboxBackend::Passthrough]
        );
    }

    #[test]
    fn evaluate_macos_without_sandbox_exec() {
        let r = SandboxReport::evaluate("macos", |_| false);
        assert_eq!(r.available, vec![SandboxBackend::Passthrough]);
    }

    #[test]
    fn evaluate_windows_always_has_job() {
        let r = SandboxReport::evaluate("windows", |_| false);
        assert_eq!(
            r.available,
            vec![SandboxBackend::WindowsJob, SandboxBackend::Passthrough]
        );
    }

    #[test]
    fn evaluate_unknown_os_falls_through_to_passthrough_only() {
        let r = SandboxReport::evaluate("freebsd", |_| true);
        assert_eq!(r.available, vec![SandboxBackend::Passthrough]);
    }

    #[test]
    fn run_against_live_host_returns_at_least_passthrough() {
        let r = SandboxReport::run();
        assert!(r.available.contains(&SandboxBackend::Passthrough));
    }

    #[test]
    fn report_serde_roundtrip() {
        let r = SandboxReport::evaluate("linux", |c| c == "bwrap");
        let s = serde_json::to_string(&r).unwrap();
        let back: SandboxReport = serde_json::from_str(&s).unwrap();
        assert_eq!(r, back);
    }

    #[test]
    fn preferred_returns_passthrough_when_empty() {
        // The `available` list is never empty in `evaluate`, but defend
        // against an externally-constructed empty report.
        let r = SandboxReport { available: vec![] };
        assert_eq!(r.preferred(), SandboxBackend::Passthrough);
    }

    #[test]
    fn has_in_path_finds_sh_on_unix() {
        if cfg!(unix) {
            let path = std::env::var_os("PATH").unwrap_or_default();
            assert!(has_in_path("sh", &path));
        }
    }

    #[test]
    fn has_in_path_with_empty_path_returns_false() {
        let empty = std::ffi::OsString::new();
        assert!(!has_in_path("sh", &empty));
    }

    // ---- SandboxSpawn ------------------------------------------------

    use crate::sandbox_resolve::{MountMode, ResolvedMount, ResolvedNet, SandboxLaunchSpec};
    use std::collections::{BTreeMap, BTreeSet};
    use std::path::PathBuf;

    fn workspace_dir() -> PathBuf {
        // A directory we know exists on every CI runner (Linux + macOS).
        PathBuf::from("/")
    }

    fn empty_spec() -> SandboxLaunchSpec {
        SandboxLaunchSpec {
            mounts: Vec::new(),
            net: ResolvedNet::Off,
            env: BTreeMap::new(),
            allowed_caps: BTreeSet::new(),
            denied_caps: BTreeSet::new(),
            working_dir: workspace_dir(),
            cpu_quota_pct: None,
            memory_limit_mib: None,
            backend: crate::sandbox_resolve::BackendChoice::Passthrough,
        }
    }

    #[test]
    fn detect_helper_present_finds_nothing_for_bogus_name() {
        assert!(!detect_helper_present("nonexistent_binary_12345"));
    }

    #[test]
    fn detect_helper_present_finds_known_shell() {
        // CI runs on ubuntu-latest + macos-latest; `sh` is on `PATH` on
        // both. Windows is not a CI target.
        assert!(detect_helper_present("sh"));
    }

    #[test]
    fn windows_job_backend_returns_unsupported() {
        let spawner = SandboxSpawn::new(SandboxBackend::WindowsJob);
        let spec = empty_spec();
        let res = spawner.spawn(&spec, Path::new("nothing"), &[]);
        assert!(matches!(res, Err(SandboxSpawnError::Unsupported(_))));
    }

    #[test]
    fn sandbox_spawn_backend_accessor() {
        let s = SandboxSpawn::new(SandboxBackend::Passthrough);
        assert_eq!(s.backend(), SandboxBackend::Passthrough);
    }

    #[test]
    fn passthrough_spawn_runs_echo() {
        let echo = if Path::new("/bin/echo").exists() {
            Path::new("/bin/echo")
        } else {
            Path::new("/usr/bin/echo")
        };
        let spawner = SandboxSpawn::new(SandboxBackend::Passthrough);
        let spec = empty_spec();
        let mut child = spawner
            .spawn(&spec, echo, &["hi".to_string()])
            .expect("spawn echo");
        let status = child.wait().expect("wait");
        assert!(status.success());
    }

    #[test]
    fn spawn_error_display_smoke() {
        let v1 = SandboxSpawnError::Spawn(std::io::Error::other("boom"));
        let v2 = SandboxSpawnError::Unsupported("windows-job");
        let v3 = SandboxSpawnError::BadProfile("nope".into());
        let v4 = SandboxSpawnError::MissingHelper { helper: "bwrap" };
        for e in [&v1, &v2, &v3, &v4] {
            let s = format!("{e}");
            assert!(!s.is_empty());
        }
        // Spawn error exposes its source.
        let _src: Option<&(dyn std::error::Error + 'static)> = std::error::Error::source(&v1);
        let _none: Option<&(dyn std::error::Error + 'static)> = std::error::Error::source(&v2);
    }

    #[test]
    fn bwrap_spawn_either_runs_or_reports_missing_helper() {
        // Single test that exercises the backend dispatch for `Bwrap`.
        // When `bwrap` is on `PATH` (Linux CI with bubblewrap installed)
        // we spawn `/bin/true` end-to-end. When it is absent (default
        // Linux / macOS runners) we assert the `MissingHelper` path.
        // Either way the dispatcher arm at `SandboxSpawn::spawn` ->
        // `spawn_bwrap` is touched.
        let spawner = SandboxSpawn::new(SandboxBackend::Bwrap);
        let spec = empty_spec();
        let res = spawner.spawn(&spec, Path::new("/bin/true"), &[]);
        if detect_helper_present("bwrap") {
            let mut child = res.expect("spawn bwrap /bin/true");
            let _ = child.wait();
        } else {
            assert!(matches!(
                res,
                Err(SandboxSpawnError::MissingHelper { helper: "bwrap" })
            ));
        }
    }

    #[test]
    fn sandbox_exec_spawn_either_runs_or_reports_missing_helper() {
        // Same shape as the bwrap test: dispatches via the backend arm,
        // takes the live path on macOS and the missing-helper path
        // elsewhere.
        let spawner = SandboxSpawn::new(SandboxBackend::SandboxExec);
        let spec = empty_spec();
        let true_bin = if Path::new("/usr/bin/true").exists() {
            Path::new("/usr/bin/true")
        } else {
            Path::new("/bin/true")
        };
        let res = spawner.spawn(&spec, true_bin, &[]);
        if detect_helper_present("sandbox-exec") {
            let mut child = res.expect("spawn sandbox-exec true");
            let _ = child.wait();
        } else {
            assert!(matches!(
                res,
                Err(SandboxSpawnError::MissingHelper {
                    helper: "sandbox-exec"
                })
            ));
        }
    }

    #[test]
    fn bwrap_args_with_empty_mounts_carries_only_baseline_binds() {
        let spec = empty_spec();
        let argv = build_bwrap_args(&spec, Path::new("/bin/true"), &[]);
        // Baseline binds: /usr, /lib, /lib64 — three `--ro-bind` triples.
        let ro_count = argv.iter().filter(|a| a.as_str() == "--ro-bind").count();
        assert_eq!(ro_count, 3);
        // No `--bind` or `--tmpfs` user entries.
        assert!(!argv.iter().any(|a| a.as_str() == "--bind"));
        assert!(!argv.iter().any(|a| a.as_str() == "--tmpfs"));
    }

    #[test]
    fn bwrap_args_tempfs_emits_tmpfs_flag() {
        let mut spec = empty_spec();
        spec.mounts.push(ResolvedMount {
            host: PathBuf::from("/ignored"),
            guest: PathBuf::from("/scratch"),
            mode: MountMode::TempFs { size_mib: 16 },
        });
        let argv = build_bwrap_args(&spec, Path::new("/bin/true"), &[]);
        let idx = argv
            .iter()
            .position(|a| a == "--tmpfs")
            .expect("tmpfs flag");
        assert_eq!(argv[idx + 1], "/scratch");
    }

    #[test]
    fn bwrap_args_read_only_emits_ro_bind() {
        let mut spec = empty_spec();
        spec.mounts.push(ResolvedMount {
            host: PathBuf::from("/host/ro"),
            guest: PathBuf::from("/guest/ro"),
            mode: MountMode::ReadOnly,
        });
        let argv = build_bwrap_args(&spec, Path::new("/bin/true"), &[]);
        // Find the user-supplied --ro-bind triple after the baseline.
        let user = argv
            .windows(3)
            .find(|w| w[0] == "--ro-bind" && w[1] == "/host/ro" && w[2] == "/guest/ro");
        assert!(user.is_some());
    }

    #[test]
    fn bwrap_args_read_write_emits_bind() {
        let mut spec = empty_spec();
        spec.mounts.push(ResolvedMount {
            host: PathBuf::from("/host/rw"),
            guest: PathBuf::from("/guest/rw"),
            mode: MountMode::ReadWrite,
        });
        let argv = build_bwrap_args(&spec, Path::new("/bin/true"), &[]);
        let user = argv
            .windows(3)
            .find(|w| w[0] == "--bind" && w[1] == "/host/rw" && w[2] == "/guest/rw");
        assert!(user.is_some());
    }

    #[test]
    fn bwrap_args_loopback_net_still_unshares() {
        let mut spec = empty_spec();
        spec.net = ResolvedNet::Loopback;
        let argv = build_bwrap_args(&spec, Path::new("/bin/true"), &[]);
        assert!(argv.iter().any(|a| a == "--unshare-net"));
    }

    #[test]
    fn bwrap_args_hosts_net_does_not_unshare() {
        let mut spec = empty_spec();
        spec.net = ResolvedNet::Hosts {
            allow: BTreeSet::new(),
        };
        let argv = build_bwrap_args(&spec, Path::new("/bin/true"), &[]);
        assert!(!argv.iter().any(|a| a == "--unshare-net"));
    }

    #[test]
    fn bwrap_args_env_emits_setenv_pairs() {
        let mut spec = empty_spec();
        spec.env.insert("FOO".into(), "bar".into());
        let argv = build_bwrap_args(&spec, Path::new("/bin/true"), &[]);
        let setenv_idx = argv
            .iter()
            .position(|a| a == "--setenv")
            .expect("setenv flag");
        assert_eq!(argv[setenv_idx + 1], "FOO");
        assert_eq!(argv[setenv_idx + 2], "bar");
        assert!(argv.iter().any(|a| a == "--clearenv"));
    }

    #[test]
    fn bwrap_args_terminator_precedes_command() {
        let spec = empty_spec();
        let argv = build_bwrap_args(&spec, Path::new("/bin/true"), &["hi".into()]);
        let term = argv.iter().position(|a| a == "--").expect("terminator");
        assert_eq!(argv[term + 1], "/bin/true");
        assert_eq!(argv[term + 2], "hi");
    }

    #[test]
    fn sandbox_exec_profile_off_denies_network() {
        let spec = empty_spec();
        let body = render_sandbox_exec_profile(&spec);
        assert!(body.contains("(deny network*)"));
        assert!(body.contains("(deny default)"));
    }

    #[test]
    fn sandbox_exec_profile_loopback_allows_local_ip() {
        let mut spec = empty_spec();
        spec.net = ResolvedNet::Loopback;
        let body = render_sandbox_exec_profile(&spec);
        assert!(body.contains("(allow network* (local ip))"));
    }

    #[test]
    fn sandbox_exec_profile_hosts_allows_network() {
        let mut spec = empty_spec();
        spec.net = ResolvedNet::Hosts {
            allow: BTreeSet::new(),
        };
        let body = render_sandbox_exec_profile(&spec);
        assert!(body.contains("(allow network*)"));
    }

    #[test]
    fn sandbox_exec_profile_read_write_mount_grants_write() {
        let mut spec = empty_spec();
        spec.mounts.push(ResolvedMount {
            host: PathBuf::from("/h"),
            guest: PathBuf::from("/g"),
            mode: MountMode::ReadWrite,
        });
        let body = render_sandbox_exec_profile(&spec);
        assert!(body.contains("(allow file-write* (subpath \"/g\"))"));
    }

    #[test]
    fn sandbox_exec_profile_read_only_mount_grants_read_only() {
        let mut spec = empty_spec();
        spec.mounts.push(ResolvedMount {
            host: PathBuf::from("/h"),
            guest: PathBuf::from("/g"),
            mode: MountMode::ReadOnly,
        });
        let body = render_sandbox_exec_profile(&spec);
        assert!(body.contains("(allow file-read* (subpath \"/g\"))"));
        assert!(!body.contains("(allow file-write* (subpath \"/g\"))"));
    }

    #[test]
    fn sandbox_exec_profile_tempfs_grants_write_only() {
        let mut spec = empty_spec();
        spec.mounts.push(ResolvedMount {
            host: PathBuf::from("/h"),
            guest: PathBuf::from("/scratch"),
            mode: MountMode::TempFs { size_mib: 8 },
        });
        let body = render_sandbox_exec_profile(&spec);
        assert!(body.contains("(allow file-write* (subpath \"/scratch\"))"));
    }

    #[test]
    fn unique_temp_profile_path_is_unique_across_calls() {
        let a = unique_temp_profile_path();
        let b = unique_temp_profile_path();
        assert_ne!(a, b);
    }

    #[test]
    fn spawn_program_runs_an_existing_binary() {
        // Try /bin/true (Linux + macOS) and /usr/bin/true (macOS).
        let candidate = if Path::new("/bin/true").exists() {
            "/bin/true"
        } else {
            "/usr/bin/true"
        };
        let mut child = spawn_program(candidate, &[]).expect("spawn true");
        let status = child.wait().expect("wait");
        assert!(status.success());
    }

    #[test]
    fn spawn_program_reports_io_error_for_missing_binary() {
        let res = spawn_program("/nonexistent/path/to/nothing-12345", &[]);
        assert!(matches!(res, Err(SandboxSpawnError::Spawn(_))));
    }

    /// Pick a tiny helper present on every CI runner (ubuntu-latest +
    /// macos-latest). `echo` is guaranteed to be on `PATH` on both.
    fn fake_helper_name() -> &'static str {
        "echo"
    }

    #[test]
    fn spawn_bwrap_with_uses_provided_helper_when_present() {
        let helper = fake_helper_name();
        let spec = empty_spec();
        let mut child = spawn_bwrap_with(helper, &spec, Path::new("/bin/true"), &[])
            .expect("spawn echo via bwrap-with");
        let _ = child.wait();
    }

    #[test]
    fn spawn_bwrap_with_reports_missing_helper() {
        let spec = empty_spec();
        let res = spawn_bwrap_with(
            "nonexistent_helper_99999",
            &spec,
            Path::new("/bin/true"),
            &[],
        );
        assert!(matches!(res, Err(SandboxSpawnError::MissingHelper { .. })));
    }

    #[test]
    fn spawn_sandbox_exec_with_uses_provided_helper_when_present() {
        let helper = fake_helper_name();
        let spec = empty_spec();
        let mut child =
            spawn_sandbox_exec_with(helper, &spec, Path::new("/bin/true"), &["arg1".to_string()])
                .expect("spawn echo via sandbox-exec-with");
        let _ = child.wait();
    }

    #[test]
    fn spawn_sandbox_exec_with_reports_missing_helper() {
        let spec = empty_spec();
        let res = spawn_sandbox_exec_with(
            "nonexistent_helper_99998",
            &spec,
            Path::new("/bin/true"),
            &[],
        );
        assert!(matches!(res, Err(SandboxSpawnError::MissingHelper { .. })));
    }

    #[test]
    fn passthrough_honors_working_dir() {
        let pwd = if Path::new("/bin/pwd").exists() {
            Path::new("/bin/pwd")
        } else {
            Path::new("/usr/bin/pwd")
        };
        let tmp = std::env::temp_dir();
        let mut spec = empty_spec();
        spec.working_dir = tmp;
        let spawner = SandboxSpawn::new(SandboxBackend::Passthrough);
        let mut child = spawner.spawn(&spec, pwd, &[]).expect("spawn pwd");
        let status = child.wait().expect("wait");
        assert!(status.success());
    }

    #[test]
    fn passthrough_honors_env() {
        let env_bin = if Path::new("/usr/bin/env").exists() {
            Path::new("/usr/bin/env")
        } else {
            Path::new("/bin/env")
        };
        let mut spec = empty_spec();
        spec.env.insert("STRATUM_TEST_KEY".into(), "yes".into());
        let spawner = SandboxSpawn::new(SandboxBackend::Passthrough);
        let mut child = spawner.spawn(&spec, env_bin, &[]).expect("spawn env");
        let status = child.wait().expect("wait");
        assert!(status.success());
    }
}
