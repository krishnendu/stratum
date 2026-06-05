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
//!
//! ## Lock order
//!
//! [`Workspace`] currently has a single interior `Mutex` (`ignore`)
//! used to memoize the parsed `.stratumignore` file on first access.
//! When more locks are added in the future, acquire them in the order
//! they are declared on the struct; document any reverse acquisition
//! here.

use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

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
///
/// `Workspace` carries a lazily-populated cache of the
/// `.stratumignore` rules. The cache is wrapped in `Arc<Mutex<…>>` so
/// `Clone`d handles share the same memoized rules; the cache is excluded
/// from `PartialEq` / `Eq` since it is observable state, not identity.
#[derive(Debug)]
pub struct Workspace {
    /// Absolute path of the workspace root (the directory that contains
    /// `stratum.toml` or `.git/`).
    pub root: PathBuf,
    /// Parsed project config, when a `stratum.toml` was present.
    pub config: Option<WorkspaceConfig>,
    /// Memoized `.stratumignore` rules. `None` until first access;
    /// `Some(StratumIgnore::default())` when the file is absent or the
    /// workspace opted out via [`WorkspaceConfig::respect_gitignore`].
    ignore: Arc<Mutex<Option<StratumIgnore>>>,
}

impl Clone for Workspace {
    fn clone(&self) -> Self {
        Self {
            root: self.root.clone(),
            config: self.config.clone(),
            ignore: Arc::clone(&self.ignore),
        }
    }
}

impl PartialEq for Workspace {
    fn eq(&self, other: &Self) -> bool {
        self.root == other.root && self.config == other.config
    }
}

impl Eq for Workspace {}

/// Default for [`WorkspaceConfig::respect_gitignore`]. Lives at module
/// scope so serde's `default = "…"` attribute can name it.
const fn default_respect_gitignore() -> bool {
    true
}

/// Top-level shape of `stratum.toml`.
///
/// Future fields land here; unknown keys are preserved through round-trip
/// so a newer Stratum can read an older project file (and vice versa) by
/// ignoring unrecognized sections during deserialization in this pass.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct WorkspaceConfig {
    /// Display name shown in the TUI status bar.
    pub name: Option<String>,
    /// Short description.
    pub description: Option<String>,
    /// On-disk schema version; bumped only when a field's semantics break.
    pub schema_version: u32,
    /// Whether file-enumeration helpers consult `.stratumignore`.
    /// Defaults to `true`; missing field in TOML reads as `true`.
    #[serde(default = "default_respect_gitignore")]
    pub respect_gitignore: bool,
}

impl Default for WorkspaceConfig {
    fn default() -> Self {
        Self {
            name: None,
            description: None,
            schema_version: 0,
            respect_gitignore: default_respect_gitignore(),
        }
    }
}

/// Errors surfaced by the file-filtering surface of [`Workspace`].
///
/// These are distinct from [`StratumError`] because they describe
/// transient I/O and path-shape problems rather than installation
/// schema breakage; the catalog deliberately does not allocate
/// STRAT-E codes for them.
#[derive(Debug)]
pub enum WorkspaceError {
    /// Reading `.stratumignore` failed with an I/O error other than
    /// `NotFound`.
    IgnoreIo(std::io::Error),
    /// Parsing `.stratumignore` failed.
    IgnoreParse(String),
    /// The caller passed an absolute path that does not begin with the
    /// workspace root.
    PathOutsideWorkspace {
        /// The offending path, preserved for diagnostics.
        path: PathBuf,
    },
}

impl fmt::Display for WorkspaceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IgnoreIo(e) => write!(f, "failed to read .stratumignore: {e}"),
            Self::IgnoreParse(msg) => write!(f, "failed to parse .stratumignore: {msg}"),
            Self::PathOutsideWorkspace { path } => {
                write!(f, "path is outside the workspace root: {}", path.display())
            }
        }
    }
}

impl std::error::Error for WorkspaceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::IgnoreIo(e) => Some(e),
            Self::IgnoreParse(_) | Self::PathOutsideWorkspace { .. } => None,
        }
    }
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
            ignore: Arc::new(Mutex::new(None)),
        })
    }

    fn from_root(root: &Path) -> Self {
        Self {
            root: root.to_path_buf(),
            config: None,
            ignore: Arc::new(Mutex::new(None)),
        }
    }

    /// Should this workspace consult `.stratumignore` during
    /// file-enumeration? Defaults to `true`; controlled by
    /// [`WorkspaceConfig::respect_gitignore`].
    fn respect_gitignore(&self) -> bool {
        self.config.as_ref().is_none_or(|c| c.respect_gitignore)
    }

    /// Lazily load `.stratumignore` and invoke `f` with the cached
    /// rules. When `respect_gitignore` is false the cache is populated
    /// with an empty rule set, which makes [`Self::is_ignored`] cheap
    /// and side-effect-free in the opt-out path.
    fn with_ignore<F, T>(&self, f: F) -> Result<T, WorkspaceError>
    where
        F: FnOnce(&StratumIgnore) -> T,
    {
        let rules = {
            let mut guard = self
                .ignore
                .lock()
                .map_err(|e| WorkspaceError::IgnoreParse(format!("ignore cache poisoned: {e}")))?;
            if guard.is_none() {
                let loaded = if self.respect_gitignore() {
                    let path = self.root.join(IGNORE_FILE);
                    match std::fs::read_to_string(&path) {
                        Ok(text) => StratumIgnore::parse(&text),
                        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                            StratumIgnore::default()
                        }
                        Err(e) => return Err(WorkspaceError::IgnoreIo(e)),
                    }
                } else {
                    StratumIgnore::default()
                };
                *guard = Some(loaded);
            }
            guard
                .as_ref()
                .map_or_else(StratumIgnore::default, Clone::clone)
        };
        Ok(f(&rules))
    }

    /// Normalize `path` to a workspace-relative path string suitable
    /// for matching against ignore rules. Absolute paths must begin
    /// with the workspace root; relative paths are taken as-is.
    fn relativize(&self, path: &Path) -> Result<(String, bool), WorkspaceError> {
        let (rel, abs) = if path.is_absolute() {
            let stripped = path.strip_prefix(&self.root).map_err(|_| {
                WorkspaceError::PathOutsideWorkspace {
                    path: path.to_path_buf(),
                }
            })?;
            (stripped.to_path_buf(), path.to_path_buf())
        } else {
            (path.to_path_buf(), self.root.join(path))
        };
        let is_dir = abs.is_dir();
        let rel_str = rel.to_string_lossy().replace('\\', "/");
        Ok((rel_str, is_dir))
    }

    /// Is `path` covered by this workspace's `.stratumignore` rules?
    ///
    /// Accepts both relative and absolute paths; absolute paths must
    /// begin with the workspace root or [`WorkspaceError::PathOutsideWorkspace`]
    /// is returned. When the workspace opted out of gitignore via
    /// [`WorkspaceConfig::respect_gitignore`], or no `.stratumignore`
    /// exists, this always returns `Ok(false)`.
    ///
    /// # Errors
    /// - [`WorkspaceError::IgnoreIo`] when the ignore file exists but
    ///   cannot be read.
    /// - [`WorkspaceError::PathOutsideWorkspace`] for absolute paths
    ///   outside the workspace root.
    pub fn is_ignored(&self, path: &Path) -> Result<bool, WorkspaceError> {
        let (rel, is_dir) = self.relativize(path)?;
        // Empty relative path (the root itself) is never ignored.
        if rel.is_empty() || rel == "." {
            return Ok(false);
        }
        self.with_ignore(|rules| rules.is_ignored(&rel, is_dir))
    }

    /// Filter `paths`, returning the subset that is **not** ignored by
    /// `.stratumignore`. Input order is preserved.
    ///
    /// Mixed relative + absolute input is normalized against the
    /// workspace root before matching.
    ///
    /// # Errors
    /// Propagates the first error from [`Self::is_ignored`].
    pub fn filter_paths<I>(&self, paths: I) -> Result<Vec<PathBuf>, WorkspaceError>
    where
        I: IntoIterator<Item = PathBuf>,
    {
        let mut kept = Vec::new();
        for p in paths {
            if !self.is_ignored(&p)? {
                kept.push(p);
            }
        }
        Ok(kept)
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
            ignore: Arc::new(Mutex::new(None)),
        };
        let without = Workspace {
            root: PathBuf::from("/x"),
            config: None,
            ignore: Arc::new(Mutex::new(None)),
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
            respect_gitignore: true,
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
        // Default for the gitignore opt-in is "on".
        assert!(cfg.respect_gitignore);
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

    // ---- WorkspaceConfig::respect_gitignore + workspace ignore wiring ----

    #[test]
    fn workspace_config_default_respects_gitignore() {
        assert!(WorkspaceConfig::default().respect_gitignore);
    }

    #[test]
    fn workspace_config_missing_field_deserializes_to_true() {
        // Older `stratum.toml` files won't have `respect_gitignore`.
        let toml = "schema_version = 1\n";
        let cfg: WorkspaceConfig = toml_edit::de::from_str(toml).unwrap();
        assert!(cfg.respect_gitignore);
    }

    #[test]
    fn workspace_config_serde_roundtrip_with_opt_out() {
        let cfg = WorkspaceConfig {
            name: None,
            description: None,
            schema_version: 1,
            respect_gitignore: false,
        };
        let s = toml_edit::ser::to_string(&cfg).unwrap();
        let back: WorkspaceConfig = toml_edit::de::from_str(&s).unwrap();
        assert_eq!(cfg, back);
        assert!(!back.respect_gitignore);
    }

    fn workspace_with_ignore(tmp: &TempDir, ignore_contents: Option<&str>) -> Workspace {
        if let Some(text) = ignore_contents {
            std::fs::write(tmp.path().join(IGNORE_FILE), text).unwrap();
        }
        Workspace::load(tmp.path()).unwrap()
    }

    #[test]
    fn is_ignored_returns_true_for_matched_dir_rule() {
        let tmp = TempDir::new().unwrap();
        // `target/foo.txt` — the dir-only rule applies to the entire
        // `target` subtree via the `target/**` glob shape; the simple
        // matcher only knows leading-segment globs, so use a pattern
        // that matches the relative path the caller will pass.
        let ws = workspace_with_ignore(&tmp, Some("target/**\n"));
        let path = PathBuf::from("target/foo.txt");
        assert!(ws.is_ignored(&path).unwrap());
    }

    #[test]
    fn is_ignored_returns_false_for_unmatched_path() {
        let tmp = TempDir::new().unwrap();
        let ws = workspace_with_ignore(&tmp, Some("target/**\n"));
        let path = PathBuf::from("src/main.rs");
        assert!(!ws.is_ignored(&path).unwrap());
    }

    #[test]
    fn is_ignored_returns_false_when_respect_gitignore_disabled() {
        let tmp = TempDir::new().unwrap();
        std::fs::write(
            tmp.path().join(PROJECT_FILE),
            "schema_version = 1\nrespect_gitignore = false\n",
        )
        .unwrap();
        std::fs::write(tmp.path().join(IGNORE_FILE), "target/**\n").unwrap();
        let ws = Workspace::load(tmp.path()).unwrap();
        let path = PathBuf::from("target/foo.txt");
        assert!(!ws.is_ignored(&path).unwrap());
    }

    #[test]
    fn is_ignored_returns_false_when_no_ignore_file_present() {
        let tmp = TempDir::new().unwrap();
        let ws = Workspace::load(tmp.path()).unwrap();
        assert!(!ws.is_ignored(&PathBuf::from("target/foo.txt")).unwrap());
        assert!(!ws.is_ignored(&PathBuf::from("anything")).unwrap());
    }

    #[test]
    fn is_ignored_returns_false_for_workspace_root_itself() {
        let tmp = TempDir::new().unwrap();
        let ws = workspace_with_ignore(&tmp, Some("**\n"));
        // Empty relative path is treated as the root, never ignored.
        assert!(!ws.is_ignored(&PathBuf::from("")).unwrap());
        assert!(!ws.is_ignored(&PathBuf::from(".")).unwrap());
    }

    #[test]
    fn filter_paths_preserves_order() {
        let tmp = TempDir::new().unwrap();
        let ws = workspace_with_ignore(&tmp, Some("ignored.txt\n"));
        let inputs = vec![
            PathBuf::from("a.txt"),
            PathBuf::from("b.txt"),
            PathBuf::from("c.txt"),
        ];
        let kept = ws.filter_paths(inputs.clone()).unwrap();
        assert_eq!(kept, inputs);
    }

    #[test]
    fn filter_paths_drops_ignored_entries() {
        let tmp = TempDir::new().unwrap();
        let ws = workspace_with_ignore(&tmp, Some("ignored.txt\n"));
        let inputs = vec![
            PathBuf::from("keep1.txt"),
            PathBuf::from("ignored.txt"),
            PathBuf::from("keep2.txt"),
        ];
        let kept = ws.filter_paths(inputs).unwrap();
        assert_eq!(
            kept,
            vec![PathBuf::from("keep1.txt"), PathBuf::from("keep2.txt")]
        );
    }

    #[test]
    fn filter_paths_normalizes_relative_and_absolute() {
        let tmp = TempDir::new().unwrap();
        let ws = workspace_with_ignore(&tmp, Some("ignored.txt\n"));
        let abs_keep = tmp.path().join("keep.txt");
        let abs_drop = tmp.path().join("ignored.txt");
        let inputs = vec![
            abs_keep.clone(),
            PathBuf::from("ignored.txt"),
            PathBuf::from("keep_rel.txt"),
            abs_drop,
        ];
        let kept = ws.filter_paths(inputs).unwrap();
        assert_eq!(kept, vec![abs_keep, PathBuf::from("keep_rel.txt")]);
    }

    #[test]
    fn ignore_file_is_loaded_lazily_once() {
        // Build a workspace, observe that `.stratumignore` is not read
        // on construction, then trigger one read by calling
        // `is_ignored`. Subsequent calls reuse the cache: removing the
        // file does not change the result.
        let tmp = TempDir::new().unwrap();
        let ws = Workspace::load(tmp.path()).unwrap();
        // Write the file AFTER construction; the cache is empty so the
        // very first call will see it.
        std::fs::write(tmp.path().join(IGNORE_FILE), "target/**\n").unwrap();
        assert!(ws.is_ignored(&PathBuf::from("target/foo.txt")).unwrap());
        // Mutate the on-disk file; cached rules should be authoritative.
        std::fs::remove_file(tmp.path().join(IGNORE_FILE)).unwrap();
        assert!(ws.is_ignored(&PathBuf::from("target/foo.txt")).unwrap());
    }

    #[test]
    fn is_ignored_returns_path_outside_workspace_for_unrelated_abs_path() {
        let tmp = TempDir::new().unwrap();
        let other = TempDir::new().unwrap();
        let ws = Workspace::load(tmp.path()).unwrap();
        let err = ws.is_ignored(&other.path().join("foo.txt")).unwrap_err();
        match err {
            WorkspaceError::PathOutsideWorkspace { path } => {
                assert!(path.starts_with(other.path()));
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn workspace_error_display_covers_all_variants() {
        use std::error::Error as _;
        let io_err = WorkspaceError::IgnoreIo(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "boom",
        ));
        let parse_err = WorkspaceError::IgnoreParse("bad rule".into());
        let outside_err = WorkspaceError::PathOutsideWorkspace {
            path: PathBuf::from("/elsewhere"),
        };
        assert!(format!("{io_err}").contains(".stratumignore"));
        assert!(format!("{parse_err}").contains("bad rule"));
        assert!(format!("{outside_err}").contains("/elsewhere"));
        // `source()` returns the inner io::Error for IgnoreIo, None for the others.
        assert!(io_err.source().is_some());
        assert!(parse_err.source().is_none());
        assert!(outside_err.source().is_none());
    }

    #[cfg(unix)]
    #[test]
    fn is_ignored_propagates_io_error_when_ignore_unreadable() {
        use std::os::unix::fs::PermissionsExt;
        let tmp = TempDir::new().unwrap();
        let ignore_path = tmp.path().join(IGNORE_FILE);
        std::fs::write(&ignore_path, "target\n").unwrap();
        std::fs::set_permissions(&ignore_path, std::fs::Permissions::from_mode(0o000)).unwrap();
        let ws = Workspace::load(tmp.path()).unwrap();
        let result = ws.is_ignored(&PathBuf::from("target/foo.txt"));
        // Restore so TempDir can clean up.
        let _ = std::fs::set_permissions(&ignore_path, std::fs::Permissions::from_mode(0o644));
        match result {
            Err(WorkspaceError::IgnoreIo(_)) => {}
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn workspace_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Workspace>();
        assert_send_sync::<WorkspaceConfig>();
        assert_send_sync::<WorkspaceError>();
    }

    #[test]
    fn workspace_clone_shares_ignore_cache() {
        // Cloned workspaces share the same memoized rules (Arc<Mutex>).
        let tmp = TempDir::new().unwrap();
        let ws = workspace_with_ignore(&tmp, Some("target/**\n"));
        let clone = ws.clone();
        assert_eq!(ws, clone);
        assert!(clone.is_ignored(&PathBuf::from("target/x")).unwrap());
    }

    #[test]
    fn workspace_partial_eq_ignores_cache() {
        // Two workspaces with the same root + config compare equal even
        // when their ignore caches are in different states.
        let tmp = TempDir::new().unwrap();
        let ws1 = workspace_with_ignore(&tmp, Some("target/**\n"));
        let ws2 = Workspace::load(tmp.path()).unwrap();
        // Warm up ws1's cache.
        let _ = ws1.is_ignored(&PathBuf::from("target/x")).unwrap();
        assert_eq!(ws1, ws2);
    }
}
