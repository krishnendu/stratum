//! Plan-mode capability fence.
//!
//! Plan mode is a read-only sandbox the user can flip on (via the future
//! `/plan` TUI palette command) so the agent can think and draft but
//! cannot write to disk, run shell, hit the network, or call any
//! write-bearing tool. Per `plan/19-permissions-prompt.md` §3.
//!
//! The fence is composed of three pieces:
//!
//! 1. [`PlanMode`] — runtime flag tracking active / inactive + activation
//!    timestamp. Safe to share across threads.
//! 2. [`PLAN_MODE_DENIED_CAPS`] — the canonical authoritative list of
//!    capability identifiers that plan mode hard-denies. Wildcards are
//!    supported via a trailing `*` so future entries don't need code
//!    changes.
//! 3. [`enforce_plan_mode_on_request`] / [`filter_capability_matrix_for_plan`]
//!    — the two enforcement points: per-request deny and matrix
//!    pre-filter for surfaces (TUI palette, agent header) that want to
//!    show "what's still available".

use std::error::Error;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::SystemTime;

use crate::tools::{CapabilityEntry, CapabilityMatrix};

/// Canonical authoritative list of capabilities hard-denied while plan
/// mode is active.
///
/// Entries are matched exactly. A trailing `*` (e.g. `mcp.*`) is treated
/// as a prefix wildcard — the current list does not use it, but
/// [`is_capability_allowed_in_plan_mode`] honors the convention so a
/// future entry like `mcp.*` would deny `mcp.write`, `mcp.delete`, …
/// without code changes.
pub const PLAN_MODE_DENIED_CAPS: &[&str] = &[
    "fs.write",
    "fs.delete",
    "shell.exec",
    "net.fetch",
    "process.spawn",
    "mcp.write",
];

/// Runtime flag for plan mode. Shared across the turn driver, tool
/// registry, and the future TUI palette.
#[derive(Debug, Default)]
pub struct PlanMode {
    active: AtomicBool,
    since: Mutex<Option<SystemTime>>,
}

impl PlanMode {
    /// Build a fresh `PlanMode` in the inactive state.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            active: AtomicBool::new(false),
            since: Mutex::new(None),
        }
    }

    /// Flip plan mode on and record `now` as the activation timestamp.
    /// Idempotent — a second `activate` keeps the first timestamp so
    /// the UI can show "in plan mode for N seconds".
    pub fn activate(&self, now: SystemTime) {
        // Lock first so the timestamp write and the boolean flip cannot
        // observe each other half-applied from another thread.
        if let Ok(mut guard) = self.since.lock() {
            if guard.is_none() {
                *guard = Some(now);
            }
        }
        self.active.store(true, Ordering::SeqCst);
    }

    /// Flip plan mode off and clear the activation timestamp.
    pub fn deactivate(&self) {
        self.active.store(false, Ordering::SeqCst);
        if let Ok(mut guard) = self.since.lock() {
            *guard = None;
        }
    }

    /// Is plan mode currently active?
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.active.load(Ordering::SeqCst)
    }

    /// Timestamp of the most recent activation, if any.
    #[must_use]
    pub fn since(&self) -> Option<SystemTime> {
        self.since.lock().ok().and_then(|g| *g)
    }

    /// Activate plan mode and return an RAII guard that will deactivate
    /// on drop. Useful for scoped `/plan` blocks.
    pub fn activate_guard(&self, now: SystemTime) -> PlanModeGuard<'_> {
        self.activate(now);
        PlanModeGuard { plan: self }
    }
}

/// RAII guard returned by [`PlanMode::activate_guard`]. Calls
/// [`PlanMode::deactivate`] when dropped.
#[derive(Debug)]
pub struct PlanModeGuard<'a> {
    plan: &'a PlanMode,
}

impl Drop for PlanModeGuard<'_> {
    fn drop(&mut self) {
        self.plan.deactivate();
    }
}

/// Is `cap` allowed in plan mode?
///
/// Returns `false` when `cap` exactly matches an entry in
/// [`PLAN_MODE_DENIED_CAPS`] or when an entry ends with `*` and `cap`
/// shares the entry's prefix (the prefix excludes the trailing `*`).
#[must_use]
pub fn is_capability_allowed_in_plan_mode(cap: &str) -> bool {
    is_denied_with_list(cap, PLAN_MODE_DENIED_CAPS).is_none()
}

/// Shared deny-matcher used by [`is_capability_allowed_in_plan_mode`]
/// and the wildcard test. Returns the matched deny pattern (so callers
/// can render a friendly message) when `cap` is denied, else `None`.
fn is_denied_with_list<'a>(cap: &str, list: &'a [&'a str]) -> Option<&'a str> {
    list.iter().copied().find(|entry| {
        entry
            .strip_suffix('*')
            .map_or_else(|| *entry == cap, |prefix| cap.starts_with(prefix))
    })
}

/// Human-friendly reason for the deny, if `cap` is denied in plan mode.
/// Returns `None` for any allowed capability.
#[must_use]
pub fn explain_denied(cap: &str) -> Option<&'static str> {
    if is_capability_allowed_in_plan_mode(cap) {
        return None;
    }
    Some(reason_for(cap))
}

fn reason_for(cap: &str) -> &'static str {
    match cap {
        "fs.write" => "plan mode forbids file writes",
        "fs.delete" => "plan mode forbids file deletes",
        "shell.exec" => "plan mode forbids shell execution",
        "net.fetch" => "plan mode forbids network fetches",
        "process.spawn" => "plan mode forbids spawning processes",
        "mcp.write" => "plan mode forbids MCP write tools",
        _ => "plan mode forbids this capability",
    }
}

/// Build a new [`CapabilityMatrix`] that drops every entry whose verb
/// (the part before `:`) is denied in plan mode. The input is left
/// untouched.
#[must_use]
pub fn filter_capability_matrix_for_plan(matrix: &CapabilityMatrix) -> CapabilityMatrix {
    let kept: Vec<CapabilityEntry> = matrix
        .entries()
        .filter(|entry| {
            let (verb, _) = entry.parts();
            is_capability_allowed_in_plan_mode(verb)
        })
        .cloned()
        .collect();
    CapabilityMatrix::from_entries(kept)
}

/// Error returned when plan mode blocks a capability request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanModeError {
    /// `capability` was denied while plan mode is active.
    CapabilityDeniedInPlanMode {
        /// The capability the caller tried to use.
        capability: String,
        /// Friendly reason string (sourced from [`explain_denied`]).
        reason: String,
    },
}

impl fmt::Display for PlanModeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CapabilityDeniedInPlanMode { capability, reason } => {
                write!(f, "capability `{capability}` denied: {reason}")
            }
        }
    }
}

impl Error for PlanModeError {}

/// Top-level enforcement entry point. When `plan` is active and `cap`
/// is denied, returns [`PlanModeError::CapabilityDeniedInPlanMode`].
/// When plan mode is inactive, every cap is allowed.
///
/// # Errors
///
/// Returns [`PlanModeError::CapabilityDeniedInPlanMode`] when plan mode
/// is active and `cap` is on the deny list.
pub fn enforce_plan_mode_on_request(plan: &PlanMode, cap: &str) -> Result<(), PlanModeError> {
    if !plan.is_active() {
        return Ok(());
    }
    explain_denied(cap).map_or(Ok(()), |reason| {
        Err(PlanModeError::CapabilityDeniedInPlanMode {
            capability: cap.to_string(),
            reason: reason.to_string(),
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn new_is_inactive() {
        let p = PlanMode::new();
        assert!(!p.is_active());
        assert!(p.since().is_none());
    }

    #[test]
    fn activate_flips_active_and_records_timestamp() {
        let p = PlanMode::new();
        let now = SystemTime::UNIX_EPOCH + Duration::from_secs(100);
        p.activate(now);
        assert!(p.is_active());
        assert_eq!(p.since(), Some(now));
    }

    #[test]
    fn activate_is_idempotent_and_keeps_first_timestamp() {
        let p = PlanMode::new();
        let first = SystemTime::UNIX_EPOCH + Duration::from_secs(100);
        let second = SystemTime::UNIX_EPOCH + Duration::from_secs(200);
        p.activate(first);
        p.activate(second);
        assert!(p.is_active());
        assert_eq!(p.since(), Some(first));
    }

    #[test]
    fn deactivate_flips_off_and_clears_timestamp() {
        let p = PlanMode::new();
        p.activate(SystemTime::UNIX_EPOCH);
        p.deactivate();
        assert!(!p.is_active());
        assert!(p.since().is_none());
    }

    #[test]
    fn allowed_cap_is_allowed() {
        assert!(is_capability_allowed_in_plan_mode("fs.read"));
    }

    #[test]
    fn fs_write_denied() {
        assert!(!is_capability_allowed_in_plan_mode("fs.write"));
    }

    #[test]
    fn shell_exec_denied() {
        assert!(!is_capability_allowed_in_plan_mode("shell.exec"));
    }

    #[test]
    fn net_fetch_denied() {
        assert!(!is_capability_allowed_in_plan_mode("net.fetch"));
    }

    #[test]
    fn process_spawn_denied() {
        assert!(!is_capability_allowed_in_plan_mode("process.spawn"));
    }

    #[test]
    fn fs_delete_denied() {
        assert!(!is_capability_allowed_in_plan_mode("fs.delete"));
    }

    #[test]
    fn mcp_write_denied() {
        assert!(!is_capability_allowed_in_plan_mode("mcp.write"));
    }

    #[test]
    fn wildcard_entry_denies_prefix() {
        let list: &[&str] = &["mcp.*"];
        assert!(is_denied_with_list("mcp.write", list).is_some());
        assert!(is_denied_with_list("mcp.delete", list).is_some());
        assert!(is_denied_with_list("fs.read", list).is_none());
    }

    #[test]
    fn wildcard_entry_does_not_swallow_other_namespaces() {
        let list: &[&str] = &["mcp.*"];
        assert!(is_denied_with_list("net.fetch", list).is_none());
    }

    #[test]
    fn plan_mode_guard_drop_deactivates() {
        let p = PlanMode::new();
        {
            let _g = p.activate_guard(SystemTime::UNIX_EPOCH);
            assert!(p.is_active());
        }
        assert!(!p.is_active());
        assert!(p.since().is_none());
    }

    #[test]
    fn enforce_inactive_allows_denied_cap() {
        let p = PlanMode::new();
        assert!(enforce_plan_mode_on_request(&p, "fs.write").is_ok());
    }

    #[test]
    fn enforce_active_allows_allowed_cap() {
        let p = PlanMode::new();
        p.activate(SystemTime::UNIX_EPOCH);
        assert!(enforce_plan_mode_on_request(&p, "fs.read").is_ok());
    }

    #[test]
    fn enforce_active_blocks_denied_cap() {
        let p = PlanMode::new();
        p.activate(SystemTime::UNIX_EPOCH);
        let err = enforce_plan_mode_on_request(&p, "fs.write").unwrap_err();
        match err {
            PlanModeError::CapabilityDeniedInPlanMode { capability, reason } => {
                assert_eq!(capability, "fs.write");
                assert_eq!(reason, "plan mode forbids file writes");
            }
        }
    }

    #[test]
    fn filter_matrix_drops_denied_entries() {
        let m = CapabilityMatrix::from_entries(["fs.read", "fs.write:src/**", "shell.exec"]);
        let filtered = filter_capability_matrix_for_plan(&m);
        let names: Vec<&str> = filtered.entries().map(CapabilityEntry::as_str).collect();
        assert_eq!(names, vec!["fs.read"]);
    }

    #[test]
    fn filter_matrix_preserves_allowed_entries() {
        let m = CapabilityMatrix::from_entries(["fs.read", "git.diff"]);
        let filtered = filter_capability_matrix_for_plan(&m);
        assert_eq!(filtered.len(), 2);
        assert!(filtered.allows("fs.read", None));
        assert!(filtered.allows("git.diff", None));
    }

    #[test]
    fn filter_matrix_identity_when_nothing_denied() {
        let m = CapabilityMatrix::from_entries(["fs.read", "git.log"]);
        let filtered = filter_capability_matrix_for_plan(&m);
        assert_eq!(m, filtered);
    }

    #[test]
    fn explain_returns_some_for_every_denied_cap() {
        for cap in PLAN_MODE_DENIED_CAPS {
            assert!(
                explain_denied(cap).is_some(),
                "missing explanation for {cap}"
            );
        }
    }

    #[test]
    fn explain_returns_none_for_allowed_cap() {
        assert!(explain_denied("fs.read").is_none());
    }

    #[test]
    fn error_display_includes_capability_and_reason() {
        let err = PlanModeError::CapabilityDeniedInPlanMode {
            capability: "fs.write".to_string(),
            reason: "plan mode forbids file writes".to_string(),
        };
        let rendered = format!("{err}");
        assert!(rendered.contains("fs.write"));
        assert!(rendered.contains("plan mode forbids file writes"));
    }

    #[test]
    fn plan_mode_is_send_and_sync() {
        assert_send_sync::<PlanMode>();
    }

    #[test]
    fn since_is_none_when_inactive() {
        let p = PlanMode::new();
        assert!(p.since().is_none());
        p.activate(SystemTime::UNIX_EPOCH);
        p.deactivate();
        assert!(p.since().is_none());
    }

    #[test]
    fn reason_fallback_for_unknown_denied_cap() {
        // Direct coverage of the `_ => …` arm via the helper. The
        // `is_denied_with_list` matcher accepts a custom deny list so
        // we can route a non-canonical cap into `reason_for`.
        let list: &[&str] = &["custom.cap"];
        assert!(is_denied_with_list("custom.cap", list).is_some());
        assert_eq!(
            reason_for("custom.cap"),
            "plan mode forbids this capability"
        );
    }

    #[test]
    fn error_is_std_error() {
        let err = PlanModeError::CapabilityDeniedInPlanMode {
            capability: "fs.write".to_string(),
            reason: "plan mode forbids file writes".to_string(),
        };
        let as_err: &dyn Error = &err;
        assert!(as_err.source().is_none());
    }
}
