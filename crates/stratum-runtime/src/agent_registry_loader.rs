//! `AgentRegistryLoader` — walks `<state>/agents/*.toml`, parses each
//! file via [`AgentLoader::load_file`], classifies the agent's role,
//! builds an [`AgentLoop`] for it via the shared [`AgentFactory`], and
//! registers the result into an [`AgentRegistry`].
//!
//! Where this fits
//! ----------------
//!
//! [`AgentLoader`] gives us *parsed [`AgentDef`]s* but nothing more — it
//! does not know how to map a role string onto a [`SuggestedRole`] enum,
//! how to build a working [`AgentLoop`], or how to keep going when one
//! file is broken. [`AgentHandoff`] consumes an [`AgentRegistry`] but has
//! no opinion on how that registry gets populated.
//!
//! `AgentRegistryLoader` bridges the two: one shot, "give me a directory
//! and a factory, and I'll give you a registry plus a report of what was
//! skipped or failed". This is what `stratum serve` / the CLI startup
//! path calls once during boot.
//!
//! Behaviour
//! ---------
//!
//! * Missing directory ⇒ `Ok((empty registry, empty report))`. We treat
//!   the absence of a per-user agents directory as "no custom agents,
//!   that's fine", not as a fatal config error.
//! * Each `*.toml` in the directory (depth 1, sorted by filename for
//!   determinism) is parsed in turn.
//! * Files with a TOML parse failure produce a [`LoadFailure`] entry in
//!   the report — the loader does not abort. Loud-but-not-fatal is the
//!   right posture for user-authored config.
//! * Files that parse but request an unknown role end up as
//!   [`SkipReason::UnknownRole`] in the report.
//! * Files with an *empty* `roles` array fall back to the file stem as
//!   the role name; we document and pin this in tests so the contract is
//!   stable.
//! * The first file to register a given role wins; subsequent files
//!   targeting the same role are recorded as
//!   [`SkipReason::DuplicateRole`].
//!
//! All of the above lives in the [`LoadReport`] — callers decide whether
//! to surface skips/errors to the user, write them to the event log, or
//! ignore them entirely.

use std::error::Error;
use std::fmt;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::agent_factory::{AgentFactory, AgentFactoryError};
use crate::agent_handoff::AgentRegistry;
use crate::agents::{AgentDef, AgentLoader};
use crate::intent_router::SuggestedRole;
use crate::provider::{EchoProvider, Provider};

const TOML_SUFFIX: &str = ".toml";

// ---------------------------------------------------------------------------
// Report types
// ---------------------------------------------------------------------------

/// Per-file skip reason — recorded when a TOML parses fine but cannot be
/// registered into the [`AgentRegistry`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SkipReason {
    /// The TOML declared a role string that does not map onto any known
    /// [`SuggestedRole`] variant.
    UnknownRole {
        /// File whose role was rejected.
        file: PathBuf,
        /// Raw role string from the TOML / file stem.
        role: String,
    },
    /// The TOML had an empty `roles` array *and* the file stem was empty
    /// (in practice only possible for pathological filenames).
    MissingRoleField {
        /// File whose role could not be determined.
        file: PathBuf,
    },
    /// A second file tried to register a role already taken by an
    /// earlier (alphabetically prior) file.
    DuplicateRole {
        /// Role that was already registered.
        role: SuggestedRole,
        /// Path of the file that won (registered first).
        existing_file: PathBuf,
        /// Path of the file that lost (skipped).
        new_file: PathBuf,
    },
}

/// Per-file load failure — recorded when [`AgentLoader::load_file`] or
/// the downstream [`AgentFactory::build`] returns an error.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LoadFailure {
    /// File the failure originated from.
    pub file: PathBuf,
    /// Stringified underlying error.
    pub error: String,
}

/// Aggregate result returned alongside the [`AgentRegistry`] from
/// [`AgentRegistryLoader::load`].
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct LoadReport {
    /// Roles that were successfully registered, in registration order.
    pub registered: Vec<SuggestedRole>,
    /// Files that parsed but were not registered.
    pub skipped: Vec<SkipReason>,
    /// Files that failed at parse or build time.
    pub errors: Vec<LoadFailure>,
}

impl LoadReport {
    /// `true` when no agents were registered, skipped, or errored — i.e.
    /// the directory was missing or empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.registered.is_empty() && self.skipped.is_empty() && self.errors.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Top-level error
// ---------------------------------------------------------------------------

/// Top-level [`AgentRegistryLoader::load`] errors.
///
/// Reserved for failures that prevent *any* agent from being considered
/// (e.g. the directory itself is unreadable for reasons other than
/// "missing"). Per-file problems live in [`LoadReport`] instead.
#[derive(Debug)]
pub enum AgentRegistryLoadError {
    /// I/O failure reading the agents directory.
    Io(io::Error),
    /// The shared [`AgentFactory`] has no provider configured, so we
    /// cannot build any loops. Carries the [`AgentFactoryError`] display
    /// text. Reserved for forward compatibility; the current loader hits
    /// this only when `AgentFactory::build` rejects every file with the
    /// same error.
    Factory(String),
}

impl fmt::Display for AgentRegistryLoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "AgentRegistryLoader: read agents dir: {e}"),
            Self::Factory(msg) => write!(f, "AgentRegistryLoader: factory error: {msg}"),
        }
    }
}

impl Error for AgentRegistryLoadError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Factory(_) => None,
        }
    }
}

impl From<AgentFactoryError> for AgentRegistryLoadError {
    fn from(e: AgentFactoryError) -> Self {
        Self::Factory(e.to_string())
    }
}

// ---------------------------------------------------------------------------
// Per-role provider resolution
// ---------------------------------------------------------------------------

/// Resolves an agent's declared model slug onto a concrete [`Provider`].
///
/// Pluggable so different roles can be backed by different GGUFs (e.g.
/// `cavemanish` → `qwen-0.5b`, `coder` → `phi`) without the loader
/// knowing anything about the model catalog.
///
/// Implementations are typically wired against a `ModelResolver` /
/// `ProviderCache` pair, but the loader does not assume that.
///
/// `model_slug` is `Some` when the TOML's `model` field is non-empty; the
/// loader currently always supplies the value, but the `Option` shape is
/// kept so future callers can ask for "give me the default" with `None`.
pub trait ProviderResolver: Send + Sync {
    /// Resolve `model_slug` to a concrete [`Provider`].
    ///
    /// # Errors
    ///
    /// * [`ProviderResolveError::UnknownSlug`] — the slug is not in this
    ///   resolver's catalog.
    /// * [`ProviderResolveError::Backend`] — the slug is known but the
    ///   backing model/provider could not be constructed.
    fn resolve(&self, model_slug: Option<&str>) -> Result<Arc<dyn Provider>, ProviderResolveError>;
}

/// Errors surfaced by [`ProviderResolver::resolve`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProviderResolveError {
    /// The supplied slug does not match any model the resolver knows
    /// about. Carries the offending slug for diagnostics.
    UnknownSlug(String),
    /// The slug resolved, but constructing the backing provider failed.
    /// Carries a stringified underlying cause.
    Backend(String),
}

impl fmt::Display for ProviderResolveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownSlug(s) => write!(f, "unknown model slug: {s}"),
            Self::Backend(msg) => write!(f, "provider backend error: {msg}"),
        }
    }
}

impl Error for ProviderResolveError {}

/// Default [`ProviderResolver`] used when the caller does not plug in a
/// real one. Always returns the same [`EchoProvider`] regardless of slug,
/// matching the runtime's "echo is the floor" posture.
#[derive(Debug, Clone)]
pub struct EchoProviderResolver {
    default: Arc<EchoProvider>,
}

impl EchoProviderResolver {
    /// Build a resolver backed by the supplied echo provider.
    #[must_use]
    pub const fn new(default: Arc<EchoProvider>) -> Self {
        Self { default }
    }
}

impl Default for EchoProviderResolver {
    fn default() -> Self {
        Self {
            default: Arc::new(EchoProvider::new("")),
        }
    }
}

impl ProviderResolver for EchoProviderResolver {
    fn resolve(
        &self,
        _model_slug: Option<&str>,
    ) -> Result<Arc<dyn Provider>, ProviderResolveError> {
        Ok(self.default.clone())
    }
}

// ---------------------------------------------------------------------------
// Loader
// ---------------------------------------------------------------------------

/// Walks an `<state>/agents/` directory and produces a populated
/// [`AgentRegistry`] together with a [`LoadReport`] describing every
/// per-file outcome.
///
/// One factory is shared across all agents — Phase 3 v2 is *not* the
/// place where per-agent provider/model bindings come from. The factory
/// supplies a single provider, capability matrix, etc., and every loop
/// in the registry inherits them. Per-agent overrides land alongside the
/// hot-reload work in a later PR.
#[derive(Clone)]
pub struct AgentRegistryLoader {
    dir: PathBuf,
    factory: Arc<AgentFactory>,
    provider_resolver: Option<Arc<dyn ProviderResolver>>,
}

impl fmt::Debug for AgentRegistryLoader {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AgentRegistryLoader")
            .field("dir", &self.dir)
            .field("factory", &self.factory)
            .field("provider_resolver_set", &self.provider_resolver.is_some())
            .finish()
    }
}

impl AgentRegistryLoader {
    /// Build a new loader over `dir`. Both arguments are stored as-is;
    /// no I/O is performed until [`Self::load`] is called.
    ///
    /// When no [`ProviderResolver`] is plugged in via
    /// [`Self::with_provider_resolver`], the loader defaults to an
    /// [`EchoProviderResolver`] internally, preserving the prior
    /// behaviour where every loop shared the factory's single provider.
    #[must_use]
    pub const fn new(dir: PathBuf, factory: Arc<AgentFactory>) -> Self {
        Self {
            dir,
            factory,
            provider_resolver: None,
        }
    }

    /// Plug in a [`ProviderResolver`] so each agent's TOML `model` field
    /// gets resolved to its own [`Provider`]. The resolved provider is
    /// layered onto the shared factory via [`AgentFactory::with_provider`]
    /// before [`AgentFactory::build`] is called — i.e. every loop ends up
    /// with its own backing provider while still inheriting every other
    /// dependency from the shared factory.
    #[must_use]
    pub fn with_provider_resolver(mut self, provider_resolver: Arc<dyn ProviderResolver>) -> Self {
        self.provider_resolver = Some(provider_resolver);
        self
    }

    /// The directory this loader scans.
    #[must_use]
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Walk `dir`, parsing every `*.toml` and registering each one.
    ///
    /// Returns `Ok((registry, report))` for both the missing-directory
    /// and "every file failed" cases — the caller inspects `report` to
    /// decide whether to act. The only `Err` variants are I/O failures
    /// reading the directory listing itself.
    ///
    /// # Errors
    ///
    /// * [`AgentRegistryLoadError::Io`] — the directory exists but
    ///   cannot be listed (permissions, etc.).
    pub fn load(&self) -> Result<(AgentRegistry, LoadReport), AgentRegistryLoadError> {
        let mut registry = AgentRegistry::new();
        let mut report = LoadReport::default();

        // Track which path won each role, so we can populate
        // SkipReason::DuplicateRole when a second file collides.
        let mut role_origin: Vec<(SuggestedRole, PathBuf)> = Vec::new();

        let paths = match list_toml_paths(&self.dir) {
            Ok(paths) => paths,
            Err(ListError::Missing) => return Ok((registry, report)),
            Err(ListError::Io(e)) => return Err(AgentRegistryLoadError::Io(e)),
        };

        // Default to EchoProviderResolver when no resolver was plugged
        // in — keeps backward compat with pre-resolver callers.
        let default_resolver: Arc<dyn ProviderResolver> = Arc::new(EchoProviderResolver::default());
        let resolver: &Arc<dyn ProviderResolver> =
            self.provider_resolver.as_ref().unwrap_or(&default_resolver);

        for path in paths {
            let def = match AgentLoader::load_file(&path) {
                Ok(d) => d,
                Err(e) => {
                    report.errors.push(LoadFailure {
                        file: path.clone(),
                        error: format!("{e}"),
                    });
                    continue;
                }
            };

            let Some(role_str) = extract_role_string(&def, &path) else {
                report
                    .skipped
                    .push(SkipReason::MissingRoleField { file: path.clone() });
                continue;
            };

            let Some(role) = parse_suggested_role(&role_str) else {
                report.skipped.push(SkipReason::UnknownRole {
                    file: path.clone(),
                    role: role_str,
                });
                continue;
            };

            if let Some((_, existing)) = role_origin.iter().find(|(r, _)| *r == role) {
                report.skipped.push(SkipReason::DuplicateRole {
                    role,
                    existing_file: existing.clone(),
                    new_file: path.clone(),
                });
                continue;
            }

            // Per-role provider resolution. `AgentDef::model` always carries
            // *some* slug (defaults to `"echo"` per `AgentDef::default`), so
            // we pass `Some(&str)` — the resolver decides what to do with
            // the default echo slug.
            let slug = def.model.as_str();
            let provider = match resolver.resolve(Some(slug)) {
                Ok(p) => p,
                Err(e) => {
                    report.errors.push(LoadFailure {
                        file: path.clone(),
                        error: format!("provider resolve: {e}"),
                    });
                    continue;
                }
            };

            // Clone the shared factory, layer the per-role provider on top,
            // then build. Each loop ends up with its own backing provider.
            let loop_ = match self
                .factory
                .as_ref()
                .clone()
                .with_provider(provider)
                .build()
            {
                Ok(l) => l,
                Err(e) => {
                    report.errors.push(LoadFailure {
                        file: path.clone(),
                        error: format!("{e}"),
                    });
                    continue;
                }
            };

            registry.register(role, Arc::new(loop_));
            report.registered.push(role);
            role_origin.push((role, path));
        }

        Ok((registry, report))
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

enum ListError {
    Missing,
    Io(io::Error),
}

fn list_toml_paths(dir: &Path) -> Result<Vec<PathBuf>, ListError> {
    let entries = match std::fs::read_dir(dir) {
        Ok(iter) => iter,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Err(ListError::Missing),
        Err(e) => return Err(ListError::Io(e)),
    };
    let mut paths: Vec<PathBuf> = entries
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| {
            p.is_file()
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.ends_with(TOML_SUFFIX))
        })
        .collect();
    paths.sort();
    Ok(paths)
}

/// Pull the first role string from the parsed [`AgentDef`]; fall back to
/// the file stem when `roles` is empty. Returns `None` only when both
/// sources yield an empty string (pathological filename).
fn extract_role_string(def: &AgentDef, path: &Path) -> Option<String> {
    if let Some(first) = def.roles.first() {
        let s = first.as_str();
        if !s.is_empty() {
            return Some(s.to_string());
        }
    }
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    if stem.is_empty() {
        None
    } else {
        Some(stem.to_string())
    }
}

/// Map a snake-case role string onto a [`SuggestedRole`]. Mirrors the
/// `#[serde(rename_all = "snake_case")]` derived parser.
fn parse_suggested_role(s: &str) -> Option<SuggestedRole> {
    match s {
        "default" => Some(SuggestedRole::Default),
        "cavemanish" => Some(SuggestedRole::Cavemanish),
        "polisher" => Some(SuggestedRole::Polisher),
        "coder" => Some(SuggestedRole::Coder),
        "researcher" => Some(SuggestedRole::Researcher),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    use crate::provider::EchoProvider;
    use tempfile::TempDir;

    fn factory() -> Arc<AgentFactory> {
        Arc::new(AgentFactory::new().with_provider(Arc::new(EchoProvider::new(""))))
    }

    fn write(dir: &Path, name: &str, body: &str) -> PathBuf {
        let path = dir.join(format!("{name}.toml"));
        std::fs::write(&path, body).unwrap();
        path
    }

    fn body_for(role: &str) -> String {
        // Minimal AgentDef body — the loader only cares about `roles`.
        format!(
            r#"
schema_version = 1
description = "x"
roles = ["{role}"]
model = "echo"
tools = []
sandbox = "passthrough"
"#
        )
    }

    fn body_no_role() -> &'static str {
        r#"
schema_version = 1
description = "y"
roles = []
model = "echo"
tools = []
sandbox = "passthrough"
"#
    }

    // -- 1. new constructor smoke --------------------------------------------

    #[test]
    fn new_stores_dir_and_factory() {
        let tmp = TempDir::new().unwrap();
        let loader = AgentRegistryLoader::new(tmp.path().to_path_buf(), factory());
        assert_eq!(loader.dir(), tmp.path());
        let rendered = format!("{loader:?}");
        assert!(rendered.contains("AgentRegistryLoader"));
    }

    // -- 2. missing directory -----------------------------------------------

    #[test]
    fn load_missing_dir_returns_empty() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path().join("does-not-exist");
        let loader = AgentRegistryLoader::new(dir, factory());
        let (reg, report) = loader.load().unwrap();
        assert!(reg.is_empty());
        assert!(report.is_empty());
    }

    // -- 3. empty directory --------------------------------------------------

    #[test]
    fn load_empty_dir_returns_empty() {
        let tmp = TempDir::new().unwrap();
        let loader = AgentRegistryLoader::new(tmp.path().to_path_buf(), factory());
        let (reg, report) = loader.load().unwrap();
        assert!(reg.is_empty());
        assert!(report.is_empty());
    }

    // -- 4. 1 valid agent ----------------------------------------------------

    #[test]
    fn load_single_valid_agent_registers_one() {
        let tmp = TempDir::new().unwrap();
        write(tmp.path(), "agent-a", &body_for("coder"));
        let loader = AgentRegistryLoader::new(tmp.path().to_path_buf(), factory());
        let (reg, report) = loader.load().unwrap();
        assert_eq!(reg.len(), 1);
        assert_eq!(report.registered, vec![SuggestedRole::Coder]);
        assert!(report.skipped.is_empty());
        assert!(report.errors.is_empty());
    }

    // -- 5. 2 valid agents ---------------------------------------------------

    #[test]
    fn load_two_valid_agents_registers_both() {
        let tmp = TempDir::new().unwrap();
        write(tmp.path(), "a-coder", &body_for("coder"));
        write(tmp.path(), "b-polisher", &body_for("polisher"));
        let loader = AgentRegistryLoader::new(tmp.path().to_path_buf(), factory());
        let (reg, report) = loader.load().unwrap();
        assert_eq!(reg.len(), 2);
        assert!(report.registered.contains(&SuggestedRole::Coder));
        assert!(report.registered.contains(&SuggestedRole::Polisher));
        assert_eq!(report.skipped.len(), 0);
        assert_eq!(report.errors.len(), 0);
    }

    // -- 6. unknown role -----------------------------------------------------

    #[test]
    fn load_unknown_role_is_skipped() {
        let tmp = TempDir::new().unwrap();
        write(tmp.path(), "weird", &body_for("not-a-real-role"));
        let loader = AgentRegistryLoader::new(tmp.path().to_path_buf(), factory());
        let (reg, report) = loader.load().unwrap();
        assert!(reg.is_empty());
        assert_eq!(report.skipped.len(), 1);
        match &report.skipped[0] {
            SkipReason::UnknownRole { role, .. } => assert_eq!(role, "not-a-real-role"),
            other => panic!("expected UnknownRole, got {other:?}"),
        }
    }

    // -- 7. missing roles field falls back to file stem ----------------------

    #[test]
    fn load_empty_roles_falls_back_to_file_stem() {
        // File stem `coder` is a real role, so the agent registers.
        let tmp = TempDir::new().unwrap();
        write(tmp.path(), "coder", body_no_role());
        let loader = AgentRegistryLoader::new(tmp.path().to_path_buf(), factory());
        let (reg, report) = loader.load().unwrap();
        assert_eq!(reg.len(), 1);
        assert_eq!(report.registered, vec![SuggestedRole::Coder]);
    }

    #[test]
    fn load_empty_roles_with_unknown_stem_skips_as_unknown_role() {
        // File stem `nonsense` is not a real role.
        let tmp = TempDir::new().unwrap();
        write(tmp.path(), "nonsense", body_no_role());
        let loader = AgentRegistryLoader::new(tmp.path().to_path_buf(), factory());
        let (_, report) = loader.load().unwrap();
        assert_eq!(report.skipped.len(), 1);
        match &report.skipped[0] {
            SkipReason::UnknownRole { role, .. } => assert_eq!(role, "nonsense"),
            other => panic!("expected UnknownRole, got {other:?}"),
        }
    }

    // -- 8. duplicate role: first wins --------------------------------------

    #[test]
    fn load_duplicate_role_first_wins() {
        let tmp = TempDir::new().unwrap();
        write(tmp.path(), "a-first", &body_for("coder"));
        write(tmp.path(), "b-second", &body_for("coder"));
        let loader = AgentRegistryLoader::new(tmp.path().to_path_buf(), factory());
        let (reg, report) = loader.load().unwrap();
        assert_eq!(reg.len(), 1);
        assert_eq!(report.registered, vec![SuggestedRole::Coder]);
        assert_eq!(report.skipped.len(), 1);
        match &report.skipped[0] {
            SkipReason::DuplicateRole {
                role,
                existing_file,
                new_file,
            } => {
                assert_eq!(*role, SuggestedRole::Coder);
                assert!(existing_file.ends_with("a-first.toml"));
                assert!(new_file.ends_with("b-second.toml"));
            }
            other => panic!("expected DuplicateRole, got {other:?}"),
        }
    }

    // -- 9. malformed TOML ---------------------------------------------------

    #[test]
    fn load_malformed_toml_records_load_failure() {
        let tmp = TempDir::new().unwrap();
        std::fs::write(tmp.path().join("broken.toml"), b"not = [ valid").unwrap();
        let loader = AgentRegistryLoader::new(tmp.path().to_path_buf(), factory());
        let (reg, report) = loader.load().unwrap();
        assert!(reg.is_empty());
        assert_eq!(report.errors.len(), 1);
        assert!(report.errors[0].file.ends_with("broken.toml"));
        assert!(!report.errors[0].error.is_empty());
    }

    #[test]
    fn load_malformed_toml_does_not_abort_other_files() {
        let tmp = TempDir::new().unwrap();
        // Alphabetically `bad` comes first; verify the loader still
        // continues to `good` afterwards.
        std::fs::write(tmp.path().join("bad.toml"), b"not = [ valid").unwrap();
        write(tmp.path(), "good", &body_for("coder"));
        let loader = AgentRegistryLoader::new(tmp.path().to_path_buf(), factory());
        let (reg, report) = loader.load().unwrap();
        assert_eq!(reg.len(), 1);
        assert_eq!(report.errors.len(), 1);
        assert_eq!(report.registered, vec![SuggestedRole::Coder]);
    }

    // -- 10. non-.toml files are ignored ------------------------------------

    #[test]
    fn load_ignores_non_toml_files() {
        let tmp = TempDir::new().unwrap();
        std::fs::write(tmp.path().join("notes.md"), b"# notes").unwrap();
        std::fs::write(tmp.path().join("readme.txt"), b"hi").unwrap();
        write(tmp.path(), "real", &body_for("polisher"));
        let loader = AgentRegistryLoader::new(tmp.path().to_path_buf(), factory());
        let (reg, report) = loader.load().unwrap();
        assert_eq!(reg.len(), 1);
        assert_eq!(report.registered, vec![SuggestedRole::Polisher]);
    }

    // -- 11. subdirectories are ignored -------------------------------------

    #[test]
    fn load_ignores_subdirectories() {
        let tmp = TempDir::new().unwrap();
        std::fs::create_dir(tmp.path().join("subdir.toml")).unwrap();
        // Even with a `.toml` suffix on the directory name, it's filtered out.
        write(tmp.path(), "real", &body_for("researcher"));
        let loader = AgentRegistryLoader::new(tmp.path().to_path_buf(), factory());
        let (reg, report) = loader.load().unwrap();
        assert_eq!(reg.len(), 1);
        assert_eq!(report.registered, vec![SuggestedRole::Researcher]);
    }

    // -- 12. LoadReport::is_empty -------------------------------------------

    #[test]
    fn load_report_is_empty_true_and_false() {
        let empty = LoadReport::default();
        assert!(empty.is_empty());
        let with_reg = LoadReport {
            registered: vec![SuggestedRole::Default],
            ..LoadReport::default()
        };
        assert!(!with_reg.is_empty());
        let with_skip = LoadReport {
            skipped: vec![SkipReason::MissingRoleField {
                file: PathBuf::from("x"),
            }],
            ..LoadReport::default()
        };
        assert!(!with_skip.is_empty());
        let with_err = LoadReport {
            errors: vec![LoadFailure {
                file: PathBuf::from("y"),
                error: "boom".into(),
            }],
            ..LoadReport::default()
        };
        assert!(!with_err.is_empty());
    }

    // -- 13. serde round-trip for each variant + LoadFailure ----------------

    #[test]
    fn skip_reason_unknown_role_serde_roundtrip() {
        let v = SkipReason::UnknownRole {
            file: PathBuf::from("/x/y.toml"),
            role: "weirdrole".into(),
        };
        let s = serde_json::to_string(&v).unwrap();
        let back: SkipReason = serde_json::from_str(&s).unwrap();
        assert_eq!(v, back);
    }

    #[test]
    fn skip_reason_missing_role_field_serde_roundtrip() {
        let v = SkipReason::MissingRoleField {
            file: PathBuf::from("/x/y.toml"),
        };
        let s = serde_json::to_string(&v).unwrap();
        let back: SkipReason = serde_json::from_str(&s).unwrap();
        assert_eq!(v, back);
    }

    #[test]
    fn skip_reason_duplicate_role_serde_roundtrip() {
        let v = SkipReason::DuplicateRole {
            role: SuggestedRole::Coder,
            existing_file: PathBuf::from("/x/a.toml"),
            new_file: PathBuf::from("/x/b.toml"),
        };
        let s = serde_json::to_string(&v).unwrap();
        let back: SkipReason = serde_json::from_str(&s).unwrap();
        assert_eq!(v, back);
    }

    #[test]
    fn load_failure_serde_roundtrip() {
        let v = LoadFailure {
            file: PathBuf::from("/p/q.toml"),
            error: "parse blew up".into(),
        };
        let s = serde_json::to_string(&v).unwrap();
        let back: LoadFailure = serde_json::from_str(&s).unwrap();
        assert_eq!(v, back);
    }

    #[test]
    fn load_report_serde_roundtrip() {
        let v = LoadReport {
            registered: vec![SuggestedRole::Coder, SuggestedRole::Polisher],
            skipped: vec![SkipReason::MissingRoleField {
                file: PathBuf::from("a"),
            }],
            errors: vec![LoadFailure {
                file: PathBuf::from("b"),
                error: "boom".into(),
            }],
        };
        let s = serde_json::to_string(&v).unwrap();
        let back: LoadReport = serde_json::from_str(&s).unwrap();
        assert_eq!(v, back);
    }

    // -- 14. AgentRegistryLoadError Display smoke ---------------------------

    #[test]
    fn registry_load_error_display_io() {
        let e = AgentRegistryLoadError::Io(io::Error::new(io::ErrorKind::PermissionDenied, "nope"));
        let s = format!("{e}");
        assert!(s.contains("read agents dir"));
        assert!(s.contains("nope"));
    }

    #[test]
    fn registry_load_error_display_factory() {
        let e = AgentRegistryLoadError::Factory("missing provider".into());
        let s = format!("{e}");
        assert!(s.contains("factory error"));
        assert!(s.contains("missing provider"));
    }

    #[test]
    fn registry_load_error_source_io_present_factory_none() {
        let e_io =
            AgentRegistryLoadError::Io(io::Error::new(io::ErrorKind::PermissionDenied, "nope"));
        assert!(e_io.source().is_some());
        let e_f = AgentRegistryLoadError::Factory("x".into());
        assert!(e_f.source().is_none());
    }

    #[test]
    fn registry_load_error_from_factory_error() {
        let f: AgentRegistryLoadError = AgentFactoryError::MissingProvider.into();
        match f {
            AgentRegistryLoadError::Factory(msg) => assert!(msg.contains("provider is required")),
            AgentRegistryLoadError::Io(e) => panic!("expected Factory, got Io({e:?})"),
        }
    }

    // -- 15. Sorted-by-filename traversal pin -------------------------------

    #[test]
    fn load_traverses_files_in_sorted_order() {
        // Write three valid files in non-sorted order; assert the
        // registration order matches alphabetical order, because that
        // is what `DuplicateRole` semantics rely on.
        let tmp = TempDir::new().unwrap();
        write(tmp.path(), "c-third", &body_for("coder"));
        write(tmp.path(), "a-first", &body_for("polisher"));
        write(tmp.path(), "b-second", &body_for("researcher"));
        let loader = AgentRegistryLoader::new(tmp.path().to_path_buf(), factory());
        let (_, report) = loader.load().unwrap();
        assert_eq!(
            report.registered,
            vec![
                SuggestedRole::Polisher,
                SuggestedRole::Researcher,
                SuggestedRole::Coder,
            ]
        );
    }

    // -- bonus coverage -----------------------------------------------------

    #[test]
    fn parse_suggested_role_covers_every_variant() {
        assert_eq!(
            parse_suggested_role("default"),
            Some(SuggestedRole::Default)
        );
        assert_eq!(
            parse_suggested_role("cavemanish"),
            Some(SuggestedRole::Cavemanish)
        );
        assert_eq!(
            parse_suggested_role("polisher"),
            Some(SuggestedRole::Polisher)
        );
        assert_eq!(parse_suggested_role("coder"), Some(SuggestedRole::Coder));
        assert_eq!(
            parse_suggested_role("researcher"),
            Some(SuggestedRole::Researcher)
        );
        assert_eq!(parse_suggested_role("unknown"), None);
    }

    #[test]
    fn extract_role_string_prefers_def_role_over_stem() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("polisher.toml");
        let def = AgentDef {
            roles: vec![stratum_types::RoleId::from("coder")],
            ..AgentDef::default()
        };
        let extracted = extract_role_string(&def, &path).unwrap();
        assert_eq!(extracted, "coder");
    }

    #[test]
    fn extract_role_string_skips_empty_first_role_uses_stem() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("coder.toml");
        let def = AgentDef {
            roles: vec![stratum_types::RoleId::from("")],
            ..AgentDef::default()
        };
        let extracted = extract_role_string(&def, &path).unwrap();
        assert_eq!(extracted, "coder");
    }

    // -- per-role provider resolution ---------------------------------------

    use crate::agent_loop::TurnContext;
    use crate::cancel::CancelToken;
    use crate::conversation::TurnOutcome;
    use crate::observability::TurnId;
    use std::collections::HashMap;
    use std::sync::Mutex;
    use std::time::{Duration, UNIX_EPOCH};
    use stratum_types::{Block, ModelId};

    fn body_with_model(role: &str, model: &str) -> String {
        format!(
            r#"
schema_version = 1
description = "x"
roles = ["{role}"]
model = "{model}"
tools = []
sandbox = "passthrough"
"#
        )
    }

    fn ctx_for(prompt: &str) -> TurnContext {
        TurnContext {
            user_prompt: prompt.into(),
            model: ModelId::from("echo"),
            turn_id: TurnId(1),
            started_at: UNIX_EPOCH + Duration::from_secs(1_700_000_000),
            history: Vec::new(),
        }
    }

    /// Test resolver returning a different `EchoProvider` (with a unique
    /// prefix) per slug. Records every slug it was asked about, so a test
    /// can assert the loader queried the resolver exactly once per file.
    #[derive(Debug)]
    struct ScriptedProviderResolver {
        by_slug: HashMap<String, Arc<EchoProvider>>,
        seen: Mutex<Vec<Option<String>>>,
    }

    impl ScriptedProviderResolver {
        fn new() -> Self {
            Self {
                by_slug: HashMap::new(),
                seen: Mutex::new(Vec::new()),
            }
        }

        fn insert(mut self, slug: &str, prefix: &str) -> Self {
            self.by_slug
                .insert(slug.to_string(), Arc::new(EchoProvider::new(prefix)));
            self
        }
    }

    impl ProviderResolver for ScriptedProviderResolver {
        fn resolve(
            &self,
            model_slug: Option<&str>,
        ) -> Result<Arc<dyn Provider>, ProviderResolveError> {
            self.seen
                .lock()
                .unwrap()
                .push(model_slug.map(str::to_string));
            let slug =
                model_slug.ok_or_else(|| ProviderResolveError::UnknownSlug("<none>".into()))?;
            self.by_slug
                .get(slug)
                .map(|p| Arc::clone(p) as Arc<dyn Provider>)
                .ok_or_else(|| ProviderResolveError::UnknownSlug(slug.to_string()))
        }
    }

    fn first_text(blocks: &[Block]) -> Option<&str> {
        blocks.iter().find_map(|b| match b {
            Block::Text { text } => Some(text.as_str()),
            _ => None,
        })
    }

    #[test]
    fn echo_provider_resolver_returns_same_arc_for_any_slug() {
        let inner = Arc::new(EchoProvider::new(""));
        let r = EchoProviderResolver::new(inner.clone());
        let a = r.resolve(Some("anything")).unwrap();
        let b = r.resolve(None).unwrap();
        let c = r.resolve(Some("other")).unwrap();
        // All three must be the same underlying allocation.
        assert!(Arc::ptr_eq(
            &(a.clone() as Arc<dyn Provider>),
            &(b.clone() as Arc<dyn Provider>),
        ));
        assert!(Arc::ptr_eq(&b, &c));
        // And it matches the configured default.
        let default_dyn: Arc<dyn Provider> = inner;
        assert!(Arc::ptr_eq(&a, &default_dyn));
    }

    #[test]
    fn echo_provider_resolver_default_is_usable() {
        let r = EchoProviderResolver::default();
        let p = r.resolve(Some("whatever")).unwrap();
        assert_eq!(p.id(), "echo");
    }

    #[test]
    fn with_provider_resolver_honors_per_role_binding() {
        // Two roles, two distinct providers — verify each loop returns
        // text prefixed with its own slug's marker.
        let tmp = TempDir::new().unwrap();
        write(
            tmp.path(),
            "cavemanish",
            &body_with_model("cavemanish", "qwen-0.5b"),
        );
        write(tmp.path(), "coder", &body_with_model("coder", "phi"));

        let resolver = Arc::new(
            ScriptedProviderResolver::new()
                .insert("qwen-0.5b", "alpha:")
                .insert("phi", "beta:"),
        );
        let loader = AgentRegistryLoader::new(tmp.path().to_path_buf(), factory())
            .with_provider_resolver(resolver.clone());
        let (registry, report) = loader.load().unwrap();

        assert_eq!(registry.len(), 2);
        assert!(report.errors.is_empty(), "errors: {:?}", report.errors);
        assert!(report.skipped.is_empty(), "skipped: {:?}", report.skipped);

        let cancel = CancelToken::new();
        let cavemanish_loop = registry.get(&SuggestedRole::Cavemanish).unwrap();
        let outcome_a = cavemanish_loop.run_turn(ctx_for("ping"), &cancel);
        assert!(matches!(outcome_a.outcome, TurnOutcome::Success));
        assert_eq!(first_text(&outcome_a.blocks), Some("alpha:ping"));

        let coder_loop = registry.get(&SuggestedRole::Coder).unwrap();
        let outcome_b = coder_loop.run_turn(ctx_for("ping"), &cancel);
        assert!(matches!(outcome_b.outcome, TurnOutcome::Success));
        assert_eq!(first_text(&outcome_b.blocks), Some("beta:ping"));

        // Resolver must have been asked exactly twice — once per file, in
        // file-stem alphabetical order (cavemanish, coder).
        let seen = resolver.seen.lock().unwrap().clone();
        assert_eq!(
            seen,
            vec![Some("qwen-0.5b".to_string()), Some("phi".to_string()),]
        );
    }

    #[test]
    fn unknown_slug_lands_as_load_failure_other_files_still_register() {
        let tmp = TempDir::new().unwrap();
        write(
            tmp.path(),
            "cavemanish",
            &body_with_model("cavemanish", "ghost"),
        );
        write(tmp.path(), "coder", &body_with_model("coder", "phi"));

        let resolver = Arc::new(ScriptedProviderResolver::new().insert("phi", "beta:"));
        let loader = AgentRegistryLoader::new(tmp.path().to_path_buf(), factory())
            .with_provider_resolver(resolver);
        let (registry, report) = loader.load().unwrap();

        // `coder` registers, `cavemanish` becomes a LoadFailure.
        assert_eq!(registry.len(), 1);
        assert_eq!(report.registered, vec![SuggestedRole::Coder]);
        assert_eq!(report.errors.len(), 1);
        assert!(report.errors[0].file.ends_with("cavemanish.toml"));
        assert!(
            report.errors[0].error.contains("provider resolve"),
            "error: {}",
            report.errors[0].error
        );
        assert!(report.errors[0].error.contains("ghost"));
    }

    #[test]
    fn no_resolver_falls_back_to_echo_resolver() {
        // Backward-compat: a loader constructed without a resolver still
        // produces a working echo-backed loop.
        let tmp = TempDir::new().unwrap();
        write(tmp.path(), "coder", &body_for("coder"));
        let loader = AgentRegistryLoader::new(tmp.path().to_path_buf(), factory());
        let (registry, report) = loader.load().unwrap();
        assert_eq!(registry.len(), 1);
        assert!(report.errors.is_empty());
        let cancel = CancelToken::new();
        let loop_ = registry.get(&SuggestedRole::Coder).unwrap();
        let outcome = loop_.run_turn(ctx_for("hi"), &cancel);
        assert!(matches!(outcome.outcome, TurnOutcome::Success));
    }

    #[test]
    fn factory_without_provider_plus_default_resolver_still_builds() {
        // Pin the "default resolver always supplies an echo provider"
        // contract: even if the shared factory has no provider set, the
        // default EchoProviderResolver fills it in per-agent.
        let tmp = TempDir::new().unwrap();
        write(tmp.path(), "coder", &body_for("coder"));
        let bare_factory = Arc::new(AgentFactory::new());
        let loader = AgentRegistryLoader::new(tmp.path().to_path_buf(), bare_factory);
        let (registry, report) = loader.load().unwrap();
        assert_eq!(registry.len(), 1, "report: {report:?}");
        assert!(report.errors.is_empty(), "errors: {:?}", report.errors);
    }

    #[test]
    fn provider_resolve_error_display_unknown_slug() {
        let s = ProviderResolveError::UnknownSlug("phi-xyz".into()).to_string();
        assert!(s.contains("unknown model slug"));
        assert!(s.contains("phi-xyz"));
    }

    #[test]
    fn provider_resolve_error_display_backend() {
        let s = ProviderResolveError::Backend("oom".into()).to_string();
        assert!(s.contains("backend"));
        assert!(s.contains("oom"));
    }

    #[test]
    fn provider_resolve_error_is_an_error_type() {
        fn assert_error<T: Error>(_: &T) {}
        assert_error(&ProviderResolveError::UnknownSlug("x".into()));
        assert_error(&ProviderResolveError::Backend("y".into()));
    }

    #[test]
    fn debug_includes_provider_resolver_flag() {
        let tmp = TempDir::new().unwrap();
        let loader = AgentRegistryLoader::new(tmp.path().to_path_buf(), factory());
        let no_res = format!("{loader:?}");
        assert!(no_res.contains("provider_resolver_set: false"));
        let loader = loader.with_provider_resolver(Arc::new(EchoProviderResolver::default()));
        let with_res = format!("{loader:?}");
        assert!(with_res.contains("provider_resolver_set: true"));
    }

    #[test]
    fn scripted_provider_resolver_returns_unknown_for_missing_slug() {
        // Cover the None branch + missing-key branch of the test helper so
        // the helper itself is fully exercised.
        let r = ScriptedProviderResolver::new();
        let err = r.resolve(None).unwrap_err();
        match err {
            ProviderResolveError::UnknownSlug(s) => assert_eq!(s, "<none>"),
            ProviderResolveError::Backend(msg) => {
                panic!("expected UnknownSlug, got Backend({msg})")
            }
        }
        let err = r.resolve(Some("missing")).unwrap_err();
        match err {
            ProviderResolveError::UnknownSlug(s) => assert_eq!(s, "missing"),
            ProviderResolveError::Backend(msg) => {
                panic!("expected UnknownSlug, got Backend({msg})")
            }
        }
    }
}
