//! Stable error taxonomy and `StratumError`.
//!
//! Codes follow the `STRAT-Eaaaa` scheme grouped by category:
//!
//! | Prefix       | Category                            |
//! |--------------|-------------------------------------|
//! | `STRAT-E1xxx`| Config / install / first-run        |
//! | `STRAT-E2xxx`| Hardware / probe / tier             |
//! | `STRAT-E3xxx`| Model / provider / load             |
//! | `STRAT-E4xxx`| Agent / orchestrator                |
//! | `STRAT-E5xxx`| Tools / sandbox                     |
//! | `STRAT-E6xxx`| Network / update / telemetry        |
//! | `STRAT-E7xxx`| RAG / index                         |
//! | `STRAT-E8xxx`| Egress / `stratum serve`            |
//! | `STRAT-E9xxx`| Internal / panic / unexpected       |

use std::borrow::Cow;
use std::fmt;

use serde::{Deserialize, Serialize};

/// Stable error identifier surfaced to the user, the doctor JSON output, the
/// log ring buffer, and the crash report payload.
///
/// Constructed via [`ErrorCode::new_static`] for compile-time catalog entries
/// or [`ErrorCode::new`] for runtime values produced during deserialization.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ErrorCode(Cow<'static, str>);

impl ErrorCode {
    /// Build a code from a `'static` string. Used by the catalog constants in
    /// [`codes`].
    #[must_use]
    pub const fn new_static(code: &'static str) -> Self {
        Self(Cow::Borrowed(code))
    }

    /// Build a code from an owned string (typically the deserialization path).
    #[must_use]
    pub fn new(code: impl Into<String>) -> Self {
        Self(Cow::Owned(code.into()))
    }

    /// Borrow the underlying code string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Workspace-wide result type.
pub type StratumResult<T> = Result<T, StratumError>;

/// The structured error type carried across every Stratum crate boundary.
#[derive(Debug, thiserror::Error)]
#[error("[{code}] {message}{}", hint.map(|h| format!("\n  hint: {h}")).unwrap_or_default())]
pub struct StratumError {
    /// Stable code, e.g. `STRAT-E3007`.
    pub code: ErrorCode,
    /// Human-readable message; translated via `tr!()` at the boundary.
    pub message: String,
    /// Optional follow-up suggestion shown after the message.
    pub hint: Option<&'static str>,
    /// Optional wrapped cause for diagnostics; not user-visible by default.
    #[source]
    pub cause: Option<Box<dyn std::error::Error + Send + Sync>>,
}

impl StratumError {
    /// Build a new error.
    #[must_use]
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            hint: None,
            cause: None,
        }
    }

    /// Attach a hint shown below the message.
    #[must_use]
    pub const fn with_hint(mut self, hint: &'static str) -> Self {
        self.hint = Some(hint);
        self
    }

    /// Attach a cause for diagnostics.
    #[must_use]
    pub fn with_cause(mut self, cause: impl std::error::Error + Send + Sync + 'static) -> Self {
        self.cause = Some(Box::new(cause));
        self
    }

    /// Access the stable code.
    #[must_use]
    pub const fn code(&self) -> &ErrorCode {
        &self.code
    }
}

/// Catalog of known codes. Adding a new error MUST add a constant here so the
/// `xtask check-error-codes` lint can verify documentation coverage.
pub mod codes {
    use super::ErrorCode;

    // E1xxx — config / install / first-run
    /// installed.toml schema unreadable.
    pub const E1001_INSTALLED_SCHEMA_UNREADABLE: ErrorCode = ErrorCode::new_static("STRAT-E1001");
    /// Secret not set in OS keyring.
    pub const E1003_SECRET_UNSET: ErrorCode = ErrorCode::new_static("STRAT-E1003");

    // E2xxx — hardware / probe / tier
    /// Probe failed: not enough free RAM to start.
    pub const E2001_INSUFFICIENT_RAM: ErrorCode = ErrorCode::new_static("STRAT-E2001");
    /// Tier downgrade refused — manual doctor required.
    pub const E2003_TIER_DOWNGRADE_REFUSED: ErrorCode = ErrorCode::new_static("STRAT-E2003");

    // E3xxx — model / provider / load
    /// Model load refused by memory-safety gate.
    pub const E3007_MODEL_LOAD_REFUSED: ErrorCode = ErrorCode::new_static("STRAT-E3007");

    // E4xxx — agent / orchestrator
    /// User agent shadows built-in; using user version.
    pub const E4002_AGENT_SHADOW: ErrorCode = ErrorCode::new_static("STRAT-E4002");
    /// Token budget exceeded.
    pub const E4003_TOKEN_BUDGET: ErrorCode = ErrorCode::new_static("STRAT-E4003");
    /// Wall budget exceeded.
    pub const E4004_WALL_BUDGET: ErrorCode = ErrorCode::new_static("STRAT-E4004");
    /// Client disconnect mid-turn.
    pub const E4005_CLIENT_DISCONNECT: ErrorCode = ErrorCode::new_static("STRAT-E4005");
    /// Refused due to prompt-injection signal.
    pub const E4006_INJECTION_REFUSAL: ErrorCode = ErrorCode::new_static("STRAT-E4006");

    // E5xxx — tools / sandbox
    /// Tool call denied by capability matrix.
    pub const E5004_TOOL_DENIED: ErrorCode = ErrorCode::new_static("STRAT-E5004");
    /// Sandbox egress denied by net allowlist.
    pub const E5005_NET_DENIED: ErrorCode = ErrorCode::new_static("STRAT-E5005");
    /// Tool call rejected by the schema gate — required arg missing, or
    /// shell.exec command not on the allowlist.
    pub const E5006_TOOL_SCHEMA: ErrorCode = ErrorCode::new_static("STRAT-E5006");

    // E6xxx — network / update / telemetry
    /// Update signature verification failed.
    pub const E6001_UPDATE_SIG: ErrorCode = ErrorCode::new_static("STRAT-E6001");

    // E7xxx — RAG / index
    /// RAG index version older than runtime.
    pub const E7002_RAG_VERSION: ErrorCode = ErrorCode::new_static("STRAT-E7002");

    // E8xxx — egress / `stratum serve`
    /// Rate limit exceeded for client token.
    pub const E8001_RATE_LIMIT: ErrorCode = ErrorCode::new_static("STRAT-E8001");

    // E9xxx — internal
    /// Unexpected internal panic; report on the issue tracker.
    pub const E9001_INTERNAL_PANIC: ErrorCode = ErrorCode::new_static("STRAT-E9001");
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::io;

    use super::codes::*;
    use super::*;

    #[test]
    fn error_code_display_matches_inner() {
        assert_eq!(
            format!("{}", ErrorCode::new_static("STRAT-E3007")),
            "STRAT-E3007"
        );
    }

    #[test]
    fn error_code_as_str() {
        assert_eq!(ErrorCode::new_static("STRAT-E3007").as_str(), "STRAT-E3007");
    }

    #[test]
    fn error_code_new_owned() {
        let code = ErrorCode::new(String::from("STRAT-E9999"));
        assert_eq!(code.as_str(), "STRAT-E9999");
    }

    #[test]
    fn error_display_includes_code_and_message() {
        let err = StratumError::new(E3007_MODEL_LOAD_REFUSED, "free 0.4 GB, need 1 GB margin");
        let rendered = format!("{err}");
        assert!(rendered.contains("STRAT-E3007"));
        assert!(rendered.contains("free 0.4 GB"));
    }

    #[test]
    fn error_display_appends_hint() {
        let err = StratumError::new(E3007_MODEL_LOAD_REFUSED, "x").with_hint("free a slot");
        assert!(format!("{err}").contains("hint: free a slot"));
    }

    #[test]
    fn error_carries_cause() {
        let inner = io::Error::other("disk full");
        let err = StratumError::new(E1001_INSTALLED_SCHEMA_UNREADABLE, "x").with_cause(inner);
        assert!(std::error::Error::source(&err).is_some());
    }

    #[test]
    fn code_method_returns_stable_identifier() {
        let err = StratumError::new(E4002_AGENT_SHADOW, "x");
        assert_eq!(err.code(), &E4002_AGENT_SHADOW);
    }

    #[test]
    fn catalog_codes_are_unique() {
        let codes = [
            &E1001_INSTALLED_SCHEMA_UNREADABLE,
            &E1003_SECRET_UNSET,
            &E2001_INSUFFICIENT_RAM,
            &E2003_TIER_DOWNGRADE_REFUSED,
            &E3007_MODEL_LOAD_REFUSED,
            &E4002_AGENT_SHADOW,
            &E4003_TOKEN_BUDGET,
            &E4004_WALL_BUDGET,
            &E4005_CLIENT_DISCONNECT,
            &E4006_INJECTION_REFUSAL,
            &E5004_TOOL_DENIED,
            &E5005_NET_DENIED,
            &E6001_UPDATE_SIG,
            &E7002_RAG_VERSION,
            &E8001_RATE_LIMIT,
            &E9001_INTERNAL_PANIC,
        ];
        let set: HashSet<_> = codes.iter().copied().collect();
        assert_eq!(set.len(), codes.len());
    }

    #[test]
    fn code_serde_roundtrip() {
        let code = E3007_MODEL_LOAD_REFUSED;
        let s = serde_json::to_string(&code).unwrap();
        let back: ErrorCode = serde_json::from_str(&s).unwrap();
        assert_eq!(code, back);
    }
}
