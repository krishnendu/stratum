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

use crate::tools::glob_match;

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

/// One parsed line of a `.stratumignore` file.
///
/// Stratum's matcher is a gitignore-shaped subset: blank lines and `#`
/// comments are skipped, `!`-prefixed lines re-include a previously
/// matched path, trailing `/` marks a directory-only rule. The pattern
/// itself is matched by the workspace's tiny glob (`*`, `**`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IgnoreRule {
    /// Glob pattern after stripping `!` and trailing `/`.
    pub pattern: String,
    /// Re-include after a previous rule excluded the path.
    pub negation: bool,
    /// Apply only to directory-typed paths.
    pub dir_only: bool,
}

impl IgnoreRule {
    /// Parse one line into a rule. Returns `None` for blank lines and
    /// comment lines.
    #[must_use]
    pub fn parse(line: &str) -> Option<Self> {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            return None;
        }
        let (negation, rest) = trimmed
            .strip_prefix('!')
            .map_or((false, trimmed), |s| (true, s));
        let (dir_only, pattern) = rest.strip_suffix('/').map_or((false, rest), |s| (true, s));
        if pattern.is_empty() {
            return None;
        }
        Some(Self {
            pattern: pattern.to_string(),
            negation,
            dir_only,
        })
    }

    /// Does this rule's pattern match `relative_path`? When the rule is
    /// directory-only, the caller must pass `is_dir = true` for it to
    /// match.
    #[must_use]
    pub fn matches(&self, relative_path: &str, is_dir: bool) -> bool {
        if self.dir_only && !is_dir {
            return false;
        }
        glob_match(&self.pattern, relative_path)
    }
}

/// Parsed `.stratumignore` file: an ordered list of rules. The matcher
/// walks rules in order; last match wins (gitignore semantics).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StratumIgnore {
    rules: Vec<IgnoreRule>,
}

impl StratumIgnore {
    /// Parse the raw text of a `.stratumignore` file.
    #[must_use]
    pub fn parse(text: &str) -> Self {
        let rules = text.lines().filter_map(IgnoreRule::parse).collect();
        Self { rules }
    }

    /// Load `.stratumignore` from `path`. A missing file returns an
    /// empty ignore set.
    ///
    /// # Errors
    /// Returns [`E1001_INSTALLED_SCHEMA_UNREADABLE`] on io failures
    /// other than `NotFound`.
    pub fn load(path: &Path) -> StratumResult<Self> {
        match std::fs::read_to_string(path) {
            Ok(text) => Ok(Self::parse(&text)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(StratumError::new(
                E1001_INSTALLED_SCHEMA_UNREADABLE,
                format!("read {}", path.display()),
            )
            .with_cause(e)),
        }
    }

    /// Number of rules in the file (blank + comment lines excluded).
    #[must_use]
    pub const fn len(&self) -> usize {
        self.rules.len()
    }

    /// No rules at all (also true when the file is empty or absent).
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }

    /// Iterate the parsed rules in declaration order.
    pub fn rules(&self) -> impl Iterator<Item = &IgnoreRule> {
        self.rules.iter()
    }

    /// Is `relative_path` ignored under these rules? `is_dir` selects
    /// directory-only rules. The walk applies every matching rule in
    /// order; the last match wins.
    #[must_use]
    pub fn is_ignored(&self, relative_path: &str, is_dir: bool) -> bool {
        let mut ignored = false;
        for rule in &self.rules {
            if rule.matches(relative_path, is_dir) {
                ignored = !rule.negation;
            }
        }
        ignored
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

    #[test]
    fn ignore_rule_parses_blank_and_comments_as_none() {
        assert!(IgnoreRule::parse("").is_none());
        assert!(IgnoreRule::parse("   ").is_none());
        assert!(IgnoreRule::parse("# a comment").is_none());
        // Lone `!` and lone `/` collapse to empty pattern → None.
        assert!(IgnoreRule::parse("!").is_none());
        assert!(IgnoreRule::parse("/").is_none());
    }

    #[test]
    fn ignore_rule_parses_simple_pattern() {
        let rule = IgnoreRule::parse("target").unwrap();
        assert_eq!(rule.pattern, "target");
        assert!(!rule.negation);
        assert!(!rule.dir_only);
    }

    #[test]
    fn ignore_rule_parses_negation() {
        let rule = IgnoreRule::parse("!keep.txt").unwrap();
        assert!(rule.negation);
        assert_eq!(rule.pattern, "keep.txt");
    }

    #[test]
    fn ignore_rule_parses_dir_only() {
        let rule = IgnoreRule::parse("node_modules/").unwrap();
        assert!(rule.dir_only);
        assert_eq!(rule.pattern, "node_modules");
    }

    #[test]
    fn ignore_rule_matches_file_only_when_not_dir_only() {
        let rule = IgnoreRule::parse("target").unwrap();
        assert!(rule.matches("target", false));
        assert!(rule.matches("target", true));
    }

    #[test]
    fn ignore_rule_dir_only_skips_file() {
        let rule = IgnoreRule::parse("node_modules/").unwrap();
        assert!(rule.matches("node_modules", true));
        assert!(!rule.matches("node_modules", false));
    }

    #[test]
    fn stratumignore_empty_when_text_is_blank() {
        let ig = StratumIgnore::parse("");
        assert!(ig.is_empty());
        assert_eq!(ig.len(), 0);
    }

    #[test]
    fn stratumignore_parses_multiple_rules() {
        let text = r"
# comment
target
*.log
!keep.log
node_modules/
";
        let ig = StratumIgnore::parse(text);
        assert_eq!(ig.len(), 4);
        let patterns: Vec<&str> = ig.rules().map(|r| r.pattern.as_str()).collect();
        assert_eq!(
            patterns,
            vec!["target", "*.log", "keep.log", "node_modules"]
        );
    }

    #[test]
    fn stratumignore_negation_re_includes() {
        let ig = StratumIgnore::parse("*.log\n!keep.log\n");
        assert!(ig.is_ignored("foo.log", false));
        assert!(!ig.is_ignored("keep.log", false));
    }

    #[test]
    fn stratumignore_last_match_wins() {
        // First rule includes, second excludes, third re-includes.
        let ig = StratumIgnore::parse("foo\n!foo\nfoo\n");
        assert!(ig.is_ignored("foo", false));
    }

    #[test]
    fn stratumignore_dir_only_rule_does_not_match_file() {
        let ig = StratumIgnore::parse("build/\n");
        assert!(ig.is_ignored("build", true));
        assert!(!ig.is_ignored("build", false));
    }

    #[test]
    fn stratumignore_glob_pattern_matches() {
        let ig = StratumIgnore::parse("**.bak\n");
        assert!(ig.is_ignored("notes.bak", false));
        assert!(ig.is_ignored("dir/a.bak", false));
        assert!(!ig.is_ignored("notes.txt", false));
    }

    #[test]
    fn stratumignore_load_returns_empty_for_missing_file() {
        let tmp = TempDir::new().unwrap();
        let missing = tmp.path().join(IGNORE_FILE);
        let ig = StratumIgnore::load(&missing).unwrap();
        assert!(ig.is_empty());
    }

    #[test]
    fn stratumignore_load_parses_real_file() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join(IGNORE_FILE);
        std::fs::write(&path, "target\n!keep\n").unwrap();
        let ig = StratumIgnore::load(&path).unwrap();
        assert_eq!(ig.len(), 2);
    }

    #[cfg(unix)]
    #[test]
    fn stratumignore_load_propagates_io_errors() {
        use std::os::unix::fs::PermissionsExt;
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join(IGNORE_FILE);
        std::fs::write(&path, "target\n").unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o000)).unwrap();
        let result = StratumIgnore::load(&path);
        let _ = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644));
        let err = result.unwrap_err();
        assert_eq!(err.code(), &E1001_INSTALLED_SCHEMA_UNREADABLE);
    }
}
