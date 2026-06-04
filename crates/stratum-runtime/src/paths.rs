//! Filesystem path layout.
//!
//! Per `plan/18-first-run-and-system-tiers.md` §1, Stratum uses XDG-style
//! splits where the OS supports them and the platform's conventional folder
//! otherwise.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use stratum_types::error::codes::E1001_INSTALLED_SCHEMA_UNREADABLE;
use stratum_types::{StratumError, StratumResult};

/// The four resolved directories the runtime reads from and writes to.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Paths {
    /// Config directory (e.g. `~/.config/stratum/`).
    pub config: PathBuf,
    /// Data directory (model weights, embeddings, tts).
    pub data: PathBuf,
    /// Mutable state (scrollback, prompt cache, logs).
    pub state: PathBuf,
    /// Cache (partial downloads, capability probes, doctor reports).
    pub cache: PathBuf,
}

impl Paths {
    /// Resolve the canonical directories for the current OS via the [`dirs`]
    /// crate. Falls back to a sentinel error if the OS does not expose any of
    /// the required directories (rare; happens in stripped containers).
    ///
    /// # Errors
    /// Returns [`E1001_INSTALLED_SCHEMA_UNREADABLE`] if the OS does not expose
    /// a usable home directory.
    pub fn resolve() -> StratumResult<Self> {
        Self::resolve_from(
            dirs::config_dir(),
            dirs::data_dir(),
            dirs::state_dir(),
            dirs::cache_dir(),
        )
    }

    /// Pure variant of [`Self::resolve`] for testing: takes the four optional
    /// directories explicitly so each error branch can be exercised.
    ///
    /// # Errors
    /// Returns [`E1001_INSTALLED_SCHEMA_UNREADABLE`] when any of the required
    /// directories is `None`. The `state` argument is optional and falls back
    /// to the data directory.
    pub fn resolve_from(
        config: Option<std::path::PathBuf>,
        data: Option<std::path::PathBuf>,
        state: Option<std::path::PathBuf>,
        cache: Option<std::path::PathBuf>,
    ) -> StratumResult<Self> {
        let config = config
            .ok_or_else(|| {
                StratumError::new(
                    E1001_INSTALLED_SCHEMA_UNREADABLE,
                    "OS exposes no config dir",
                )
            })?
            .join("stratum");
        let data = data
            .ok_or_else(|| {
                StratumError::new(E1001_INSTALLED_SCHEMA_UNREADABLE, "OS exposes no data dir")
            })?
            .join("stratum");
        let state = state.unwrap_or_else(|| data.clone()).join("stratum");
        let cache = cache
            .ok_or_else(|| {
                StratumError::new(E1001_INSTALLED_SCHEMA_UNREADABLE, "OS exposes no cache dir")
            })?
            .join("stratum");
        Ok(Self {
            config,
            data,
            state,
            cache,
        })
    }

    /// Build a `Paths` rooted under an arbitrary base directory. Used by tests
    /// and by `--workspace <path>` overrides.
    #[must_use]
    pub fn under(base: &Path) -> Self {
        Self {
            config: base.join("config"),
            data: base.join("data"),
            state: base.join("state"),
            cache: base.join("cache"),
        }
    }

    /// Ensure all four directories exist on disk. Idempotent.
    ///
    /// # Errors
    /// Returns [`E1001_INSTALLED_SCHEMA_UNREADABLE`] wrapping the io error if
    /// any directory cannot be created.
    pub fn ensure_dirs(&self) -> StratumResult<()> {
        for dir in [&self.config, &self.data, &self.state, &self.cache] {
            std::fs::create_dir_all(dir).map_err(|e| {
                StratumError::new(
                    E1001_INSTALLED_SCHEMA_UNREADABLE,
                    format!("cannot create {}", dir.display()),
                )
                .with_cause(e)
            })?;
        }
        Ok(())
    }

    /// Path to the `installed.toml` marker file.
    #[must_use]
    pub fn installed_toml(&self) -> PathBuf {
        self.config.join("installed.toml")
    }
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::*;

    #[test]
    fn under_subdirectories() {
        let tmp = TempDir::new().unwrap();
        let p = Paths::under(tmp.path());
        assert_eq!(p.config, tmp.path().join("config"));
        assert_eq!(p.data, tmp.path().join("data"));
        assert_eq!(p.state, tmp.path().join("state"));
        assert_eq!(p.cache, tmp.path().join("cache"));
    }

    #[test]
    fn installed_toml_is_under_config() {
        let tmp = TempDir::new().unwrap();
        let p = Paths::under(tmp.path());
        assert_eq!(
            p.installed_toml(),
            tmp.path().join("config").join("installed.toml")
        );
    }

    #[test]
    fn ensure_dirs_creates_all_four() {
        let tmp = TempDir::new().unwrap();
        let p = Paths::under(tmp.path());
        p.ensure_dirs().unwrap();
        assert!(p.config.is_dir());
        assert!(p.data.is_dir());
        assert!(p.state.is_dir());
        assert!(p.cache.is_dir());
    }

    #[test]
    fn ensure_dirs_idempotent() {
        let tmp = TempDir::new().unwrap();
        let p = Paths::under(tmp.path());
        p.ensure_dirs().unwrap();
        // Second call must succeed.
        p.ensure_dirs().unwrap();
    }

    #[test]
    fn ensure_dirs_errors_on_unwritable_root() {
        // A path that includes a regular file as a parent component cannot be
        // created — exercises the io-error path.
        let tmp = TempDir::new().unwrap();
        let file_as_dir = tmp.path().join("not-a-dir");
        std::fs::write(&file_as_dir, b"oops").unwrap();
        let p = Paths::under(&file_as_dir);
        let err = p.ensure_dirs().unwrap_err();
        assert_eq!(err.code(), &E1001_INSTALLED_SCHEMA_UNREADABLE);
    }

    #[test]
    fn resolve_succeeds_on_supported_oses() {
        // macOS and Linux always expose the four dirs in test environments.
        let p = Paths::resolve().unwrap();
        assert!(p.config.ends_with("stratum"));
        assert!(p.data.ends_with("stratum"));
        assert!(p.state.ends_with("stratum"));
        assert!(p.cache.ends_with("stratum"));
    }

    #[test]
    fn serde_roundtrip() {
        let tmp = TempDir::new().unwrap();
        let p = Paths::under(tmp.path());
        let s = serde_json::to_string(&p).unwrap();
        let back: Paths = serde_json::from_str(&s).unwrap();
        assert_eq!(p, back);
    }

    #[test]
    fn resolve_from_happy_path() {
        let base = std::path::PathBuf::from("/srv/stratum-root");
        let p = Paths::resolve_from(
            Some(base.join("config")),
            Some(base.join("data")),
            Some(base.join("state")),
            Some(base.join("cache")),
        )
        .unwrap();
        assert!(p.config.ends_with("stratum"));
        assert!(p.data.ends_with("stratum"));
        assert!(p.state.ends_with("stratum"));
        assert!(p.cache.ends_with("stratum"));
    }

    #[test]
    fn resolve_from_state_falls_back_to_data() {
        let base = std::path::PathBuf::from("/srv/stratum-root");
        let p = Paths::resolve_from(
            Some(base.join("config")),
            Some(base.join("data")),
            None,
            Some(base.join("cache")),
        )
        .unwrap();
        // state defaults to (data/stratum)/stratum when not provided
        assert!(p.state.starts_with(base.join("data").join("stratum")));
    }

    #[test]
    fn resolve_from_missing_config_errors() {
        let err = Paths::resolve_from(None, Some("/x/data".into()), None, Some("/x/cache".into()))
            .unwrap_err();
        assert_eq!(err.code(), &E1001_INSTALLED_SCHEMA_UNREADABLE);
    }

    #[test]
    fn resolve_from_missing_data_errors() {
        let err = Paths::resolve_from(
            Some("/x/config".into()),
            None,
            None,
            Some("/x/cache".into()),
        )
        .unwrap_err();
        assert_eq!(err.code(), &E1001_INSTALLED_SCHEMA_UNREADABLE);
    }

    #[test]
    fn resolve_from_missing_cache_errors() {
        let err = Paths::resolve_from(Some("/x/config".into()), Some("/x/data".into()), None, None)
            .unwrap_err();
        assert_eq!(err.code(), &E1001_INSTALLED_SCHEMA_UNREADABLE);
    }
}
