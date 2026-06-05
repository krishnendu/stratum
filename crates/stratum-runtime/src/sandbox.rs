//! Sandbox backend detection.
//!
//! Phase 2 v2 ships **detection only**; the concrete sandbox profile
//! bodies (`bwrap-strict`, `bwrap-net`, `macos-strict`, `passthrough`)
//! land with the tool registry in Phase 3 per
//! `plan/31-tool-sandbox-and-secrets.md`.
//!
//! The detector reports which backends are available on the host so
//! `stratum doctor` and the orchestrator can pick a profile at runtime.

use serde::{Deserialize, Serialize};

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
}
