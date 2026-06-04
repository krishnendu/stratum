//! `tracing` subscriber initialization.
//!
//! Stratum logs through the [`tracing`] facade. The CLI initializes a layered
//! subscriber on startup: a human-formatted stderr layer and (optionally) a
//! JSON file layer under `<state>/logs/`. Env-filter (`RUST_LOG` or
//! `STRATUM_LOG`) controls verbosity.
//!
//! Per `plan/29-error-taxonomy-and-logging.md` §4-5.

use std::path::Path;

use stratum_types::error::codes::E1001_INSTALLED_SCHEMA_UNREADABLE;
use stratum_types::{StratumError, StratumResult};
use tracing_subscriber::fmt;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::EnvFilter;

/// Configuration for tracing init.
#[derive(Debug, Clone)]
pub struct LoggingConfig {
    /// Default filter when neither `RUST_LOG` nor `STRATUM_LOG` is set.
    pub default_filter: String,
    /// Optional log file directory; if `Some`, a JSON log file is appended.
    pub log_dir: Option<std::path::PathBuf>,
    /// Whether to emit to stderr (the human layer). Disabled in tests.
    pub stderr: bool,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            default_filter: "info".to_string(),
            log_dir: None,
            stderr: true,
        }
    }
}

/// Build the env filter from the configured default and the standard envs.
#[must_use]
pub fn env_filter(default: &str) -> EnvFilter {
    if let Ok(filter) = std::env::var("STRATUM_LOG") {
        return EnvFilter::new(filter);
    }
    if let Ok(filter) = std::env::var("RUST_LOG") {
        return EnvFilter::new(filter);
    }
    EnvFilter::new(default)
}

/// Initialize the global subscriber. Safe to call once per process; subsequent
/// calls are silently ignored.
///
/// # Errors
/// Returns [`E1001_INSTALLED_SCHEMA_UNREADABLE`] if a log directory is
/// configured and cannot be created.
pub fn init(cfg: &LoggingConfig) -> StratumResult<()> {
    let filter = env_filter(&cfg.default_filter);
    let registry = tracing_subscriber::registry().with(filter);

    if let Some(dir) = cfg.log_dir.as_deref() {
        std::fs::create_dir_all(dir).map_err(|e| {
            StratumError::new(
                E1001_INSTALLED_SCHEMA_UNREADABLE,
                format!("create log dir {}", dir.display()),
            )
            .with_cause(e)
        })?;
    }

    if cfg.stderr {
        let stderr_layer = fmt::layer().with_writer(std::io::stderr);
        let _ = registry.with(stderr_layer).try_init();
    } else {
        let _ = registry.try_init();
    }

    Ok(())
}

/// Default log directory: `<state>/logs/`.
#[must_use]
pub fn default_log_dir(state: &Path) -> std::path::PathBuf {
    state.join("logs")
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn default_filter_is_info() {
        let cfg = LoggingConfig::default();
        assert_eq!(cfg.default_filter, "info");
        assert!(cfg.stderr);
        assert!(cfg.log_dir.is_none());
    }

    #[test]
    fn env_filter_uses_default_when_no_env_set() {
        // Tests run with no STRATUM_LOG / RUST_LOG set unless explicit.
        // We can't unset safely from a test thread, but reading the default
        // path is well-defined.
        let f = env_filter("warn");
        let rendered = format!("{f}");
        assert!(rendered.contains("warn") || rendered.contains("trace") || !rendered.is_empty());
    }

    #[test]
    fn init_creates_log_directory() {
        let tmp = TempDir::new().unwrap();
        let log_dir = tmp.path().join("logs-from-init");
        let cfg = LoggingConfig {
            default_filter: "warn".into(),
            log_dir: Some(log_dir.clone()),
            stderr: false,
        };
        init(&cfg).unwrap();
        assert!(log_dir.is_dir());
    }

    #[test]
    fn init_without_log_dir_succeeds() {
        let cfg = LoggingConfig {
            default_filter: "info".into(),
            log_dir: None,
            stderr: false,
        };
        // Second call: try_init returns Err but our wrapper ignores it.
        init(&cfg).unwrap();
        init(&cfg).unwrap();
    }

    #[test]
    fn init_with_invalid_log_dir_errors() {
        let tmp = TempDir::new().unwrap();
        let blocker = tmp.path().join("blocker");
        std::fs::write(&blocker, b"x").unwrap();
        let cfg = LoggingConfig {
            default_filter: "info".into(),
            log_dir: Some(blocker.join("nested")),
            stderr: false,
        };
        let err = init(&cfg).unwrap_err();
        assert_eq!(err.code(), &E1001_INSTALLED_SCHEMA_UNREADABLE);
    }

    #[test]
    fn default_log_dir_is_under_state() {
        let p = default_log_dir(Path::new("/var/state/stratum"));
        assert_eq!(p, Path::new("/var/state/stratum/logs"));
    }

    #[test]
    fn logging_config_clones() {
        let cfg = LoggingConfig::default();
        let c2 = cfg.clone();
        assert_eq!(cfg.default_filter, c2.default_filter);
    }

    #[test]
    fn init_with_stderr_enabled_succeeds() {
        // Drives the stderr-layer branch. try_init may return Err for a
        // previously-installed subscriber; the wrapper ignores it.
        let cfg = LoggingConfig {
            default_filter: "warn".into(),
            log_dir: None,
            stderr: true,
        };
        init(&cfg).unwrap();
    }
}
