//! Workspace / project concept.
//!
//! Phase 2 v2: opening Stratum against a directory is now a first-class
//! state. A directory becomes a **project** when it carries a
//! `stratum.toml`; bare directories that just have a `.git/` are
//! treated as **loose workspaces** with defaults. The runtime walks up
//! from the current working directory looking for either marker.
//!
//! Per `plan/30-workspace-and-project.md`.
//!
//! Today this module is parse-and-discover only; the
//! per-workspace agent allowlist / RAG scope / sandbox bindings plug
//! into this struct in later phases.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use stratum_types::error::codes::E1001_INSTALLED_SCHEMA_UNREADABLE;
use stratum_types::{StratumError, StratumResult};

/// Filename of the per-project marker.
pub const PROJECT_FILE: &str = "stratum.toml";

/// Filename of the per-project ignore list (gitignore-shaped).
pub const IGNORE_FILE: &str = ".stratumignore";

/// A discovered workspace: a directory the user opened Stratum against,
/// optionally accompanied by a parsed `stratum.toml`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Workspace {
    /// Absolute path of the workspace root (the directory that contains
    /// `stratum.toml` or `.git/`).
    pub root: PathBuf,
    /// Parsed project config, when a `stratum.toml` was present.
    pub config: Option<WorkspaceConfig>,
}

/// Top-level shape of `stratum.toml`.
///
/// Future fields land here; unknown keys are preserved through round-trip
/// so a newer Stratum can read an older project file (and vice versa) by
/// ignoring unrecognized sections during deserialization in this pass.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct WorkspaceConfig {
    /// Display name shown in the TUI status bar.
    pub name: Option<String>,
    /// Short description.
    pub description: Option<String>,
    /// On-disk schema version; bumped only when a field's semantics break.
    pub schema_version: u32,
}

impl Workspace {
    /// Search for a workspace root starting from `start` and walking up.
    /// Returns the first ancestor that contains either [`PROJECT_FILE`] or
    /// `.git/`. `None` when no marker is found before the filesystem root.
    #[must_use]
    pub fn find_from(start: &Path) -> Option<Self> {
        let mut here: PathBuf = start.canonicalize().ok()?;
        loop {
            if here.join(PROJECT_FILE).is_file() || here.join(".git").exists() {
                return Some(Self::from_root(&here));
            }
            if !here.pop() {
                return None;
            }
        }
    }

    /// Build a workspace at the given root, parsing `stratum.toml` when
    /// present. A parse failure surfaces as `Err` rather than being
    /// silently downgraded so misconfigurations are loud.
    ///
    /// # Errors
    /// Returns [`E1001_INSTALLED_SCHEMA_UNREADABLE`] when `stratum.toml`
    /// is present but fails to parse.
    pub fn load(root: &Path) -> StratumResult<Self> {
        let project = root.join(PROJECT_FILE);
        if !project.is_file() {
            return Ok(Self::from_root(root));
        }
        let raw = std::fs::read_to_string(&project).map_err(|e| {
            StratumError::new(
                E1001_INSTALLED_SCHEMA_UNREADABLE,
                format!("read {}", project.display()),
            )
            .with_cause(e)
        })?;
        let config: WorkspaceConfig = toml_edit::de::from_str(&raw).map_err(|e| {
            StratumError::new(
                E1001_INSTALLED_SCHEMA_UNREADABLE,
                format!("parse {}", project.display()),
            )
            .with_cause(e)
        })?;
        Ok(Self {
            root: root.to_path_buf(),
            config: Some(config),
        })
    }

    fn from_root(root: &Path) -> Self {
        Self {
            root: root.to_path_buf(),
            config: None,
        }
    }

    /// Is this a fully-configured project (a `stratum.toml` was parsed)?
    #[must_use]
    pub const fn is_project(&self) -> bool {
        self.config.is_some()
    }

    /// Path to the `.stratumignore` file (which may or may not exist).
    #[must_use]
    pub fn ignore_path(&self) -> PathBuf {
        self.root.join(IGNORE_FILE)
    }

    /// One-line label suitable for the TUI status bar:
    /// `"<name> (<root>)"` when configured, else just `"<root>"`.
    #[must_use]
    pub fn label(&self) -> String {
        self.config
            .as_ref()
            .and_then(|c| c.name.as_deref())
            .map_or_else(
                || self.root.display().to_string(),
                |name| format!("{name} ({})", self.root.display()),
            )
    }
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::*;

    #[test]
    fn find_from_returns_none_when_no_marker() {
        let tmp = TempDir::new().unwrap();
        let nested = tmp.path().join("nested");
        std::fs::create_dir(&nested).unwrap();
        // No stratum.toml or .git anywhere in the tmp tree, but the
        // filesystem root may have a .git; instead probe from inside
        // the temp dir which we know is clean.
        let found = Workspace::find_from(&nested);
        // The host's filesystem walk may find an ancestor with `.git/`
        // (e.g. running under a checkout). Treat both outcomes as valid:
        // either None or a path strictly above the tmp dir.
        if let Some(ws) = found {
            assert!(!ws.root.starts_with(&nested));
        }
    }

    #[test]
    fn find_from_discovers_stratum_toml() {
        let tmp = TempDir::new().unwrap();
        std::fs::write(tmp.path().join(PROJECT_FILE), "schema_version = 1\n").unwrap();
        let nested = tmp.path().join("sub");
        std::fs::create_dir(&nested).unwrap();
        let ws = Workspace::find_from(&nested).expect("workspace found");
        assert_eq!(
            ws.root.canonicalize().unwrap(),
            tmp.path().canonicalize().unwrap()
        );
    }

    #[test]
    fn find_from_discovers_git_dir() {
        let tmp = TempDir::new().unwrap();
        std::fs::create_dir(tmp.path().join(".git")).unwrap();
        let ws = Workspace::find_from(tmp.path()).expect("workspace found");
        assert_eq!(
            ws.root.canonicalize().unwrap(),
            tmp.path().canonicalize().unwrap()
        );
        // No stratum.toml → loose workspace.
        assert!(!ws.is_project());
        assert!(ws.config.is_none());
    }

    #[test]
    fn load_parses_stratum_toml() {
        let tmp = TempDir::new().unwrap();
        std::fs::write(
            tmp.path().join(PROJECT_FILE),
            "schema_version = 1\nname = \"stratum\"\ndescription = \"local LLM TUI\"\n",
        )
        .unwrap();
        let ws = Workspace::load(tmp.path()).unwrap();
        assert!(ws.is_project());
        let cfg = ws.config.unwrap();
        assert_eq!(cfg.name.as_deref(), Some("stratum"));
        assert_eq!(cfg.description.as_deref(), Some("local LLM TUI"));
        assert_eq!(cfg.schema_version, 1);
    }

    #[test]
    fn load_without_marker_returns_loose_workspace() {
        let tmp = TempDir::new().unwrap();
        let ws = Workspace::load(tmp.path()).unwrap();
        assert!(!ws.is_project());
    }

    #[test]
    fn load_with_malformed_toml_errors() {
        let tmp = TempDir::new().unwrap();
        std::fs::write(tmp.path().join(PROJECT_FILE), "not = [ valid toml").unwrap();
        let err = Workspace::load(tmp.path()).unwrap_err();
        assert_eq!(err.code(), &E1001_INSTALLED_SCHEMA_UNREADABLE);
    }

    #[cfg(unix)]
    #[test]
    fn load_with_unreadable_toml_errors() {
        use std::os::unix::fs::PermissionsExt;
        let tmp = TempDir::new().unwrap();
        let project = tmp.path().join(PROJECT_FILE);
        std::fs::write(&project, "schema_version = 1\n").unwrap();
        // Strip read permission so std::fs::read_to_string fails.
        std::fs::set_permissions(&project, std::fs::Permissions::from_mode(0o000)).unwrap();
        let result = Workspace::load(tmp.path());
        // Restore permissions so TempDir can clean up.
        let _ = std::fs::set_permissions(&project, std::fs::Permissions::from_mode(0o644));
        let err = result.unwrap_err();
        assert_eq!(err.code(), &E1001_INSTALLED_SCHEMA_UNREADABLE);
    }

    #[test]
    fn ignore_path_is_under_root() {
        let tmp = TempDir::new().unwrap();
        let ws = Workspace::load(tmp.path()).unwrap();
        assert_eq!(ws.ignore_path(), tmp.path().join(IGNORE_FILE));
    }

    #[test]
    fn label_uses_name_when_present() {
        let tmp = TempDir::new().unwrap();
        std::fs::write(
            tmp.path().join(PROJECT_FILE),
            "schema_version = 1\nname = \"foo\"\n",
        )
        .unwrap();
        let ws = Workspace::load(tmp.path()).unwrap();
        let label = ws.label();
        assert!(label.starts_with("foo ("));
        assert!(label.contains(&tmp.path().display().to_string()));
    }

    #[test]
    fn label_falls_back_to_root_path() {
        let tmp = TempDir::new().unwrap();
        std::fs::write(tmp.path().join(PROJECT_FILE), "schema_version = 1\n").unwrap();
        let ws = Workspace::load(tmp.path()).unwrap();
        assert_eq!(ws.label(), tmp.path().display().to_string());
    }

    #[test]
    fn is_project_reflects_config_presence() {
        let with_cfg = Workspace {
            root: PathBuf::from("/x"),
            config: Some(WorkspaceConfig::default()),
        };
        let without = Workspace {
            root: PathBuf::from("/x"),
            config: None,
        };
        assert!(with_cfg.is_project());
        assert!(!without.is_project());
    }

    #[test]
    fn workspace_config_serde_roundtrip() {
        let cfg = WorkspaceConfig {
            name: Some("foo".into()),
            description: None,
            schema_version: 1,
        };
        let s = toml_edit::ser::to_string(&cfg).unwrap();
        let back: WorkspaceConfig = toml_edit::de::from_str(&s).unwrap();
        assert_eq!(cfg, back);
    }

    #[test]
    fn workspace_config_default_is_zeroed() {
        let cfg = WorkspaceConfig::default();
        assert!(cfg.name.is_none());
        assert_eq!(cfg.schema_version, 0);
    }
}
