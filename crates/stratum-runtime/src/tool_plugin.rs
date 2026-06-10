//! Third-party tool plugin SDK — manifest + filesystem registry + subprocess dispatcher.
//!
//! Phase 3 v2 scaffold. A *plugin* is an out-of-tree executable that
//! satisfies the [`ToolDispatcher`] surface for one or more `tool_id`s.
//! Each plugin is declared by a TOML manifest dropped into
//! `<state_root>/plugins/`. At startup the runtime walks that directory,
//! loads every manifest, and exposes a single
//! [`ToolPluginDispatcher`] that routes `tool_id` → plugin binary →
//! subprocess.
//!
//! ## Wire protocol
//!
//! For each invocation the dispatcher spawns the plugin binary with one
//! argument: `--invocation <json>`, where `<json>` is the
//! `serde_json` encoding of the [`ToolInvocation`]. Stdin is left empty.
//! The plugin MUST write **one line** to stdout containing a JSON-encoded
//! [`ToolResult`] and exit 0. Any other shape (non-zero exit, malformed
//! stdout, missing line) is normalised to
//! `ToolResult::Err { code: "E_PLUGIN_BAD_OUTPUT", … }` with a short
//! stderr tail in the message. A wall-clock timeout from
//! `manifest.timeout_ms` is enforced via the same polling helper as
//! [`crate::tool_dispatchers`]; on timeout the child is killed and an
//! `E_PLUGIN_TIMEOUT` is returned.
//!
//! ## Error code policy
//!
//! Like [`crate::tool_dispatchers`], this module uses local
//! `E_PLUGIN_*` sentinels rather than catalog `STRAT-E####` entries.
//! The sentinels are declared in `crates/xtask/src/check_sentinel_codes.rs`
//! in the same PR.

// xtask-check-error-codes: ignore-file
//
// Reason: this module uses local `E_PLUGIN_*` sentinels (mirroring the
// `E_DISPATCH_*` precedent in `tool_dispatchers.rs`) rather than catalog
// `STRAT-E####` entries. The sentinel-codes scanner is intentionally
// NOT opted out: each `E_PLUGIN_*` constant is declared in this file,
// and the scanner needs to see those declarations so the allowlist
// entries don't show up as orphans.

use std::error::Error;
use std::fmt;
use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::tool_invocation::{ToolDispatcher, ToolInvocation, ToolResult};

/// Local sentinel: plugin produced no/malformed stdout or exited non-zero.
const E_PLUGIN_BAD_OUTPUT: &str = "E_PLUGIN_BAD_OUTPUT";
/// Local sentinel: plugin exceeded its wall-clock timeout.
const E_PLUGIN_TIMEOUT: &str = "E_PLUGIN_TIMEOUT";
/// Local sentinel: dispatcher could not spawn the plugin binary.
const E_PLUGIN_SPAWN_FAILED: &str = "E_PLUGIN_SPAWN_FAILED";
/// Local sentinel: no plugin registered for the requested `tool_id`.
const E_PLUGIN_NO_MATCH: &str = "E_PLUGIN_NO_MATCH";
/// Local sentinel: dispatcher could not serialize the invocation for the
/// child process.
const E_PLUGIN_INVOCATION_ENCODING: &str = "E_PLUGIN_INVOCATION_ENCODING";

/// Default wall-clock timeout for a plugin invocation when the manifest
/// omits `timeout_ms`.
pub const DEFAULT_PLUGIN_TIMEOUT_MS: u64 = 30_000;

/// On-disk manifest describing a third-party tool plugin.
///
/// One TOML file per plugin lives in `<state_root>/plugins/`. The
/// `binary` field is resolved relative to the manifest's parent
/// directory if it is not absolute, mirroring the `agent_registry_loader`
/// behaviour for per-agent assets.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolPluginManifest {
    /// Human-friendly plugin name. MUST be unique across the
    /// registry — collisions surface as
    /// [`ToolPluginLoadError::DuplicateName`].
    pub name: String,
    /// Plugin version (free-form, e.g. `"0.1.0"`).
    pub version: String,
    /// Path to the executable invoked once per call.
    pub binary: PathBuf,
    /// Tool ids this plugin satisfies. Used by
    /// [`ToolPluginDispatcher::supports`] and
    /// [`ToolPluginRegistry::find_for`].
    pub supports: Vec<String>,
    /// Wall-clock timeout per invocation in milliseconds.
    #[serde(default = "default_timeout_ms")]
    pub timeout_ms: u64,
}

const fn default_timeout_ms() -> u64 {
    DEFAULT_PLUGIN_TIMEOUT_MS
}

impl Default for ToolPluginManifest {
    fn default() -> Self {
        Self {
            name: String::new(),
            version: String::new(),
            binary: PathBuf::new(),
            supports: Vec::new(),
            timeout_ms: DEFAULT_PLUGIN_TIMEOUT_MS,
        }
    }
}

/// A loaded plugin — the manifest plus the directory it was found in
/// (used to resolve relative `binary` paths).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolPlugin {
    manifest: ToolPluginManifest,
    dir: PathBuf,
}

impl ToolPlugin {
    /// Build a plugin from a manifest + the manifest's parent directory.
    #[must_use]
    pub const fn new(manifest: ToolPluginManifest, dir: PathBuf) -> Self {
        Self { manifest, dir }
    }

    /// Borrow the underlying manifest.
    #[must_use]
    pub const fn manifest(&self) -> &ToolPluginManifest {
        &self.manifest
    }

    /// Directory the manifest was loaded from.
    #[must_use]
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Resolved binary path. If [`ToolPluginManifest::binary`] is
    /// relative, it is joined to [`Self::dir`].
    #[must_use]
    pub fn resolved_binary(&self) -> PathBuf {
        if self.manifest.binary.is_absolute() {
            self.manifest.binary.clone()
        } else {
            self.dir.join(&self.manifest.binary)
        }
    }
}

/// Read-only registry of loaded plugins.
pub trait ToolPluginRegistry: Send + Sync {
    /// Snapshot of every registered plugin, in registration order.
    fn plugins(&self) -> Vec<Arc<ToolPlugin>>;
    /// Return the plugin that satisfies `tool_id`, if any.
    fn find_for(&self, tool_id: &str) -> Option<Arc<ToolPlugin>>;
}

/// Filesystem-backed plugin registry.
///
/// Walks a single directory of `*.toml` manifests at construction time
/// and freezes the result. Hot-reload is out of scope for the scaffold.
#[derive(Debug, Clone, Default)]
pub struct FileSystemPluginRegistry {
    plugins: Vec<Arc<ToolPlugin>>,
}

impl FileSystemPluginRegistry {
    /// Build an empty registry. Useful for tests.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct a registry from a pre-built list of plugins. Performs
    /// the same duplicate-name / duplicate-tool checks as
    /// [`Self::load_from_dir`].
    ///
    /// # Errors
    ///
    /// Returns [`ToolPluginLoadError::DuplicateName`] if two plugins
    /// share a name, or [`ToolPluginLoadError::DuplicateTool`] if two
    /// plugins claim the same `tool_id`.
    pub fn from_plugins(plugins: Vec<ToolPlugin>) -> Result<Self, ToolPluginLoadError> {
        validate_unique(&plugins)?;
        Ok(Self {
            plugins: plugins.into_iter().map(Arc::new).collect(),
        })
    }

    /// Walk `dir`, parse every `*.toml` file as a [`ToolPluginManifest`],
    /// and assemble a registry.
    ///
    /// A missing `dir` is treated as "no plugins installed" and returns
    /// an empty registry — this matches the scaffold guidance that
    /// plugins are opt-in.
    ///
    /// # Errors
    ///
    /// Returns [`ToolPluginLoadError::Io`] if `dir` exists but cannot be
    /// read, [`ToolPluginLoadError::Parse`] if a manifest is malformed,
    /// or [`ToolPluginLoadError::DuplicateName`] /
    /// [`ToolPluginLoadError::DuplicateTool`] on collisions.
    #[allow(clippy::needless_pass_by_value)]
    pub fn load_from_dir(dir: PathBuf) -> Result<Self, ToolPluginLoadError> {
        Self::load_from_path(&dir)
    }

    /// Borrowed variant of [`Self::load_from_dir`].
    ///
    /// # Errors
    ///
    /// See [`Self::load_from_dir`].
    pub fn load_from_path(dir: &Path) -> Result<Self, ToolPluginLoadError> {
        let entries = match fs::read_dir(dir) {
            Ok(it) => it,
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                return Ok(Self::default());
            }
            Err(e) => return Err(ToolPluginLoadError::Io(e)),
        };

        let mut paths: Vec<PathBuf> = Vec::new();
        for entry in entries {
            let entry = entry.map_err(ToolPluginLoadError::Io)?;
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "toml") && path.is_file() {
                paths.push(path);
            }
        }
        // Deterministic order regardless of the underlying readdir order.
        paths.sort();

        let mut plugins: Vec<ToolPlugin> = Vec::with_capacity(paths.len());
        for path in paths {
            let raw = fs::read_to_string(&path).map_err(ToolPluginLoadError::Io)?;
            let manifest: ToolPluginManifest = toml_edit::de::from_str(&raw)
                .map_err(|e| ToolPluginLoadError::Parse(format!("{}: {e}", path.display())))?;
            let parent = path
                .parent()
                .map_or_else(|| dir.to_path_buf(), Path::to_path_buf);
            plugins.push(ToolPlugin::new(manifest, parent));
        }

        Self::from_plugins(plugins)
    }
}

fn validate_unique(plugins: &[ToolPlugin]) -> Result<(), ToolPluginLoadError> {
    // Duplicate name?
    for (i, a) in plugins.iter().enumerate() {
        for b in plugins.iter().skip(i + 1) {
            if a.manifest.name == b.manifest.name {
                return Err(ToolPluginLoadError::DuplicateName(a.manifest.name.clone()));
            }
        }
    }
    // Duplicate tool_id?
    for (i, a) in plugins.iter().enumerate() {
        for tool_id in &a.manifest.supports {
            let mut owners: Vec<String> = vec![a.manifest.name.clone()];
            for b in plugins.iter().skip(i + 1) {
                if b.manifest.supports.iter().any(|t| t == tool_id) {
                    owners.push(b.manifest.name.clone());
                }
            }
            if owners.len() > 1 {
                return Err(ToolPluginLoadError::DuplicateTool {
                    tool_id: tool_id.clone(),
                    by_plugins: owners,
                });
            }
        }
    }
    Ok(())
}

impl ToolPluginRegistry for FileSystemPluginRegistry {
    fn plugins(&self) -> Vec<Arc<ToolPlugin>> {
        self.plugins.clone()
    }

    fn find_for(&self, tool_id: &str) -> Option<Arc<ToolPlugin>> {
        self.plugins
            .iter()
            .find(|p| p.manifest.supports.iter().any(|t| t == tool_id))
            .cloned()
    }
}

/// Dispatcher that routes calls to a [`ToolPluginRegistry`] and
/// satisfies them via subprocess invocation of the plugin binary.
pub struct ToolPluginDispatcher {
    id: String,
    registry: Arc<dyn ToolPluginRegistry>,
}

impl fmt::Debug for ToolPluginDispatcher {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ToolPluginDispatcher")
            .field("id", &self.id)
            .field("plugin_count", &self.registry.plugins().len())
            .finish()
    }
}

impl ToolPluginDispatcher {
    /// Stable id used for duplicate-registration detection.
    pub const ID: &'static str = "tool_plugin";

    /// Build a new dispatcher over `registry`.
    #[must_use]
    pub fn new(registry: Arc<dyn ToolPluginRegistry>) -> Self {
        Self {
            id: Self::ID.to_string(),
            registry,
        }
    }

    /// Override the registered id (mostly for tests that need to
    /// register multiple plugin dispatchers).
    #[must_use]
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = id.into();
        self
    }

    /// Snapshot of the underlying registry.
    #[must_use]
    pub fn registry(&self) -> Arc<dyn ToolPluginRegistry> {
        Arc::clone(&self.registry)
    }

    fn err(inv: &ToolInvocation, code: &str, message: impl Into<String>) -> ToolResult {
        ToolResult::Err {
            tool_id: inv.tool_id.clone(),
            code: code.to_string(),
            message: message.into(),
        }
    }
}

impl ToolDispatcher for ToolPluginDispatcher {
    fn invoke(&self, inv: &ToolInvocation) -> ToolResult {
        let Some(plugin) = self.registry.find_for(&inv.tool_id) else {
            return Self::err(
                inv,
                E_PLUGIN_NO_MATCH,
                format!("no plugin handles `{}`", inv.tool_id),
            );
        };

        let payload = match serde_json::to_string(inv) {
            Ok(s) => s,
            Err(e) => {
                return Self::err(
                    inv,
                    E_PLUGIN_INVOCATION_ENCODING,
                    format!("invocation encoding failed: {e}"),
                );
            }
        };

        let binary = plugin.resolved_binary();
        let timeout = Duration::from_millis(plugin.manifest.timeout_ms);

        let mut child = match Command::new(&binary)
            .arg("--invocation")
            .arg(&payload)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
        {
            Ok(c) => c,
            Err(e) => {
                return Self::err(
                    inv,
                    E_PLUGIN_SPAWN_FAILED,
                    format!("spawn `{}` failed: {e}", binary.display()),
                );
            }
        };

        match wait_with_timeout(&mut child, timeout) {
            WaitOutcome::Exited {
                status,
                stdout,
                stderr,
            } => {
                if status != 0 {
                    let tail = tail_text(&String::from_utf8_lossy(&stderr), 256);
                    return Self::err(
                        inv,
                        E_PLUGIN_BAD_OUTPUT,
                        format!("plugin exited {status}: {tail}"),
                    );
                }
                parse_plugin_stdout(inv, &stdout, &stderr)
            }
            WaitOutcome::Timeout => {
                let _ = child.kill();
                let _ = child.wait();
                Self::err(
                    inv,
                    E_PLUGIN_TIMEOUT,
                    format!("plugin timed out after {} ms", plugin.manifest.timeout_ms),
                )
            }
            WaitOutcome::WaitFailed(e) => {
                Self::err(inv, E_PLUGIN_SPAWN_FAILED, format!("wait failed: {e}"))
            }
        }
    }

    fn supports(&self, tool_id: &str) -> bool {
        self.registry.find_for(tool_id).is_some()
    }

    fn id(&self) -> &str {
        &self.id
    }
}

fn parse_plugin_stdout(inv: &ToolInvocation, stdout: &[u8], stderr: &[u8]) -> ToolResult {
    let Ok(text) = std::str::from_utf8(stdout) else {
        let tail = tail_text(&String::from_utf8_lossy(stderr), 256);
        return ToolResult::Err {
            tool_id: inv.tool_id.clone(),
            code: E_PLUGIN_BAD_OUTPUT.to_string(),
            message: format!("plugin stdout not utf-8: {tail}"),
        };
    };
    let line = text.lines().next().unwrap_or("").trim();
    if line.is_empty() {
        let tail = tail_text(&String::from_utf8_lossy(stderr), 256);
        return ToolResult::Err {
            tool_id: inv.tool_id.clone(),
            code: E_PLUGIN_BAD_OUTPUT.to_string(),
            message: format!("plugin stdout empty: {tail}"),
        };
    }
    match serde_json::from_str::<ToolResult>(line) {
        Ok(r) => r,
        Err(e) => {
            let tail = tail_text(&String::from_utf8_lossy(stderr), 256);
            ToolResult::Err {
                tool_id: inv.tool_id.clone(),
                code: E_PLUGIN_BAD_OUTPUT.to_string(),
                message: format!("plugin stdout not a ToolResult ({e}): {tail}"),
            }
        }
    }
}

#[derive(Debug)]
enum WaitOutcome {
    Exited {
        status: i32,
        stdout: Vec<u8>,
        stderr: Vec<u8>,
    },
    Timeout,
    WaitFailed(io::Error),
}

fn wait_with_timeout(child: &mut Child, timeout: Duration) -> WaitOutcome {
    let start = Instant::now();
    let poll = Duration::from_millis(20);
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let mut stdout_buf = Vec::new();
                let mut stderr_buf = Vec::new();
                if let Some(mut s) = child.stdout.take() {
                    let _ = s.read_to_end(&mut stdout_buf);
                }
                if let Some(mut s) = child.stderr.take() {
                    let _ = s.read_to_end(&mut stderr_buf);
                }
                let code = status.code().unwrap_or(-1);
                return WaitOutcome::Exited {
                    status: code,
                    stdout: stdout_buf,
                    stderr: stderr_buf,
                };
            }
            Ok(None) => {
                if start.elapsed() >= timeout {
                    return WaitOutcome::Timeout;
                }
                std::thread::sleep(poll);
            }
            Err(e) => return WaitOutcome::WaitFailed(e),
        }
    }
}

fn tail_text(s: &str, max_chars: usize) -> String {
    let trimmed = s.trim();
    if trimmed.chars().count() <= max_chars {
        trimmed.to_string()
    } else {
        let skip = trimmed.chars().count() - max_chars;
        trimmed.chars().skip(skip).collect()
    }
}

/// Errors returned by [`FileSystemPluginRegistry::load_from_dir`].
#[derive(Debug)]
pub enum ToolPluginLoadError {
    /// Filesystem error walking the plugin directory or reading a
    /// manifest file.
    Io(io::Error),
    /// A manifest could not be parsed as TOML. The string carries the
    /// file path and the underlying error message.
    Parse(String),
    /// Two manifests share a `name`.
    DuplicateName(String),
    /// Two manifests claim the same `tool_id`. Carries the offending
    /// tool id and the names of every plugin claiming it.
    DuplicateTool {
        /// The tool id that was claimed twice.
        tool_id: String,
        /// Names of every plugin claiming `tool_id`.
        by_plugins: Vec<String>,
    },
}

impl fmt::Display for ToolPluginLoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "plugin directory io error: {e}"),
            Self::Parse(msg) => write!(f, "plugin manifest parse error: {msg}"),
            Self::DuplicateName(name) => write!(f, "duplicate plugin name: {name}"),
            Self::DuplicateTool {
                tool_id,
                by_plugins,
            } => write!(
                f,
                "duplicate plugin tool_id `{tool_id}` claimed by: {}",
                by_plugins.join(", ")
            ),
        }
    }
}

impl Error for ToolPluginLoadError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use tempfile::TempDir;

    use super::*;

    fn sample_manifest(name: &str, binary: &str, supports: &[&str]) -> ToolPluginManifest {
        ToolPluginManifest {
            name: name.to_string(),
            version: "0.1.0".to_string(),
            binary: PathBuf::from(binary),
            supports: supports.iter().map(|s| (*s).to_string()).collect(),
            timeout_ms: 5_000,
        }
    }

    fn sample_invocation(tool: &str) -> ToolInvocation {
        let mut args = BTreeMap::new();
        args.insert("k".to_string(), serde_json::json!("v"));
        ToolInvocation {
            tool_id: tool.to_string(),
            args,
            capability: "plugin".to_string(),
            turn_id: 1,
        }
    }

    fn write_manifest(dir: &Path, file: &str, manifest: &ToolPluginManifest) {
        let s = toml_edit::ser::to_string(manifest).expect("serialize manifest fixture");
        fs::write(dir.join(file), s).expect("write manifest fixture");
    }

    #[test]
    fn manifest_serde_roundtrip() {
        let m = sample_manifest("alpha", "./alpha-bin", &["alpha.do"]);
        let s = toml_edit::ser::to_string(&m).expect("ser");
        let back: ToolPluginManifest = toml_edit::de::from_str(&s).expect("de");
        assert_eq!(m, back);
    }

    #[test]
    fn manifest_default_roundtrip() {
        let m = ToolPluginManifest::default();
        assert_eq!(m.timeout_ms, DEFAULT_PLUGIN_TIMEOUT_MS);
        let s = toml_edit::ser::to_string(&m).expect("ser");
        let back: ToolPluginManifest = toml_edit::de::from_str(&s).expect("de");
        assert_eq!(m, back);
    }

    #[test]
    fn manifest_default_timeout_when_missing() {
        let toml_src = r#"
            name = "alpha"
            version = "0.1.0"
            binary = "./alpha-bin"
            supports = ["alpha.do"]
        "#;
        let m: ToolPluginManifest = toml_edit::de::from_str(toml_src).expect("de");
        assert_eq!(m.timeout_ms, DEFAULT_PLUGIN_TIMEOUT_MS);
    }

    #[test]
    fn load_from_dir_empty_returns_empty_registry() {
        let dir = TempDir::new().expect("tmp");
        let reg =
            FileSystemPluginRegistry::load_from_dir(dir.path().to_path_buf()).expect("load empty");
        assert!(reg.plugins().is_empty());
    }

    #[test]
    fn load_from_dir_missing_dir_is_empty_not_error() {
        let dir = TempDir::new().expect("tmp");
        let missing = dir.path().join("does-not-exist");
        let reg = FileSystemPluginRegistry::load_from_dir(missing).expect("missing ok");
        assert!(reg.plugins().is_empty());
    }

    #[test]
    fn load_from_dir_stages_single_plugin() {
        let dir = TempDir::new().expect("tmp");
        let m = sample_manifest("alpha", "./alpha-bin", &["alpha.do"]);
        write_manifest(dir.path(), "alpha.toml", &m);
        let reg = FileSystemPluginRegistry::load_from_dir(dir.path().to_path_buf()).expect("load");
        assert_eq!(reg.plugins().len(), 1);
        let plugin = &reg.plugins()[0];
        assert_eq!(plugin.manifest().name, "alpha");
        assert_eq!(plugin.dir(), dir.path());
    }

    #[test]
    fn load_from_dir_ignores_non_toml() {
        let dir = TempDir::new().expect("tmp");
        let m = sample_manifest("alpha", "./alpha-bin", &["alpha.do"]);
        write_manifest(dir.path(), "alpha.toml", &m);
        fs::write(dir.path().join("README.md"), "ignored").expect("write readme");
        let reg = FileSystemPluginRegistry::load_from_dir(dir.path().to_path_buf()).expect("load");
        assert_eq!(reg.plugins().len(), 1);
    }

    #[test]
    fn load_from_dir_duplicate_name() {
        let dir = TempDir::new().expect("tmp");
        let m1 = sample_manifest("alpha", "./a", &["alpha.do"]);
        let m2 = sample_manifest("alpha", "./b", &["beta.do"]);
        write_manifest(dir.path(), "first.toml", &m1);
        write_manifest(dir.path(), "second.toml", &m2);
        let err = FileSystemPluginRegistry::load_from_dir(dir.path().to_path_buf())
            .expect_err("dup name");
        match err {
            ToolPluginLoadError::DuplicateName(name) => assert_eq!(name, "alpha"),
            other => panic!("wrong variant: {other:?}"),
        }
    }

    #[test]
    fn load_from_dir_duplicate_tool() {
        let dir = TempDir::new().expect("tmp");
        let m1 = sample_manifest("alpha", "./a", &["shared.tool"]);
        let m2 = sample_manifest("beta", "./b", &["shared.tool"]);
        write_manifest(dir.path(), "alpha.toml", &m1);
        write_manifest(dir.path(), "beta.toml", &m2);
        let err = FileSystemPluginRegistry::load_from_dir(dir.path().to_path_buf())
            .expect_err("dup tool");
        match err {
            ToolPluginLoadError::DuplicateTool {
                tool_id,
                by_plugins,
            } => {
                assert_eq!(tool_id, "shared.tool");
                assert!(by_plugins.contains(&"alpha".to_string()));
                assert!(by_plugins.contains(&"beta".to_string()));
            }
            other => panic!("wrong variant: {other:?}"),
        }
    }

    #[test]
    fn load_from_dir_path_is_file_returns_io_error() {
        let dir = TempDir::new().expect("tmp");
        let not_dir = dir.path().join("not-a-dir.toml");
        fs::write(&not_dir, "name = \"x\"").expect("write");
        let err = FileSystemPluginRegistry::load_from_dir(not_dir).expect_err("io");
        assert!(matches!(err, ToolPluginLoadError::Io(_)));
    }

    #[test]
    fn from_plugins_accepts_empty_list() {
        let reg = FileSystemPluginRegistry::from_plugins(vec![]).expect("empty ok");
        assert!(reg.plugins().is_empty());
    }

    #[test]
    fn from_plugins_accepts_multiple_unique() {
        let a = ToolPlugin::new(
            sample_manifest("alpha", "./a", &["alpha.do"]),
            PathBuf::from("/tmp"),
        );
        let b = ToolPlugin::new(
            sample_manifest("beta", "./b", &["beta.do"]),
            PathBuf::from("/tmp"),
        );
        let reg = FileSystemPluginRegistry::from_plugins(vec![a, b]).expect("ok");
        assert_eq!(reg.plugins().len(), 2);
        assert!(reg.find_for("alpha.do").is_some());
        assert!(reg.find_for("beta.do").is_some());
    }

    #[test]
    fn manifest_clone_eq() {
        let a = sample_manifest("alpha", "./a", &["alpha.do"]);
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn tool_plugin_clone_eq() {
        let a = ToolPlugin::new(
            sample_manifest("alpha", "./a", &["alpha.do"]),
            PathBuf::from("/tmp"),
        );
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn load_from_dir_malformed_toml_yields_parse() {
        let dir = TempDir::new().expect("tmp");
        fs::write(dir.path().join("broken.toml"), "this = is = not = toml\n").expect("write");
        let err =
            FileSystemPluginRegistry::load_from_dir(dir.path().to_path_buf()).expect_err("parse");
        assert!(matches!(err, ToolPluginLoadError::Parse(_)));
    }

    #[test]
    fn find_for_known_tool() {
        let m = sample_manifest("alpha", "./a", &["alpha.do"]);
        let reg =
            FileSystemPluginRegistry::from_plugins(vec![ToolPlugin::new(m, PathBuf::from("/tmp"))])
                .expect("build");
        let hit = reg.find_for("alpha.do").expect("hit");
        assert_eq!(hit.manifest().name, "alpha");
    }

    #[test]
    fn find_for_unknown_tool_returns_none() {
        let m = sample_manifest("alpha", "./a", &["alpha.do"]);
        let reg =
            FileSystemPluginRegistry::from_plugins(vec![ToolPlugin::new(m, PathBuf::from("/tmp"))])
                .expect("build");
        assert!(reg.find_for("nope").is_none());
    }

    #[test]
    fn resolved_binary_relative_joins_dir() {
        let m = sample_manifest("alpha", "./alpha-bin", &["alpha.do"]);
        let plugin = ToolPlugin::new(m, PathBuf::from("/var/lib/plugins"));
        assert_eq!(
            plugin.resolved_binary(),
            PathBuf::from("/var/lib/plugins/./alpha-bin")
        );
    }

    #[test]
    fn resolved_binary_absolute_passes_through() {
        let m = sample_manifest("alpha", "/usr/local/bin/alpha", &["alpha.do"]);
        let plugin = ToolPlugin::new(m, PathBuf::from("/var/lib/plugins"));
        assert_eq!(
            plugin.resolved_binary(),
            PathBuf::from("/usr/local/bin/alpha")
        );
    }

    #[test]
    fn dispatcher_supports_reflects_registry() {
        let m = sample_manifest("alpha", "./a", &["alpha.do"]);
        let reg = Arc::new(
            FileSystemPluginRegistry::from_plugins(vec![ToolPlugin::new(m, PathBuf::from("/tmp"))])
                .expect("build"),
        );
        let d = ToolPluginDispatcher::new(reg);
        assert!(d.supports("alpha.do"));
        assert!(!d.supports("beta.do"));
        assert_eq!(d.id(), ToolPluginDispatcher::ID);
    }

    #[test]
    fn dispatcher_no_match_returns_err() {
        let reg = Arc::new(FileSystemPluginRegistry::new());
        let d = ToolPluginDispatcher::new(reg);
        let inv = sample_invocation("missing");
        match d.invoke(&inv) {
            ToolResult::Err { code, .. } => assert_eq!(code, E_PLUGIN_NO_MATCH),
            ToolResult::Ok { .. } => panic!("expected Err"),
        }
    }

    #[test]
    fn dispatcher_spawn_failure_returns_err() {
        let m = sample_manifest("ghost", "/path/does/not/exist/ghost-plugin", &["ghost.do"]);
        let reg = Arc::new(
            FileSystemPluginRegistry::from_plugins(vec![ToolPlugin::new(m, PathBuf::from("/tmp"))])
                .expect("build"),
        );
        let d = ToolPluginDispatcher::new(reg);
        let inv = sample_invocation("ghost.do");
        match d.invoke(&inv) {
            ToolResult::Err { code, .. } => assert_eq!(code, E_PLUGIN_SPAWN_FAILED),
            ToolResult::Ok { .. } => panic!("expected spawn failure"),
        }
    }

    #[test]
    fn dispatcher_with_id_overrides_default() {
        let reg = Arc::new(FileSystemPluginRegistry::new());
        let d = ToolPluginDispatcher::new(reg).with_id("custom");
        assert_eq!(d.id(), "custom");
    }

    #[test]
    fn dispatcher_registry_handle_clones() {
        let m = sample_manifest("alpha", "./a", &["alpha.do"]);
        let reg = Arc::new(
            FileSystemPluginRegistry::from_plugins(vec![ToolPlugin::new(m, PathBuf::from("/tmp"))])
                .expect("build"),
        );
        let d = ToolPluginDispatcher::new(reg);
        assert_eq!(d.registry().plugins().len(), 1);
    }

    #[test]
    fn dispatcher_debug_contains_id() {
        let reg = Arc::new(FileSystemPluginRegistry::new());
        let d = ToolPluginDispatcher::new(reg);
        let s = format!("{d:?}");
        assert!(s.contains("ToolPluginDispatcher"));
        assert!(s.contains(ToolPluginDispatcher::ID));
    }

    #[test]
    fn dispatcher_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ToolPluginDispatcher>();
    }

    #[test]
    fn load_error_display_smoke() {
        let io_err = ToolPluginLoadError::Io(io::Error::other("broken"));
        assert!(format!("{io_err}").contains("io"));
        let parse = ToolPluginLoadError::Parse("path: oops".to_string());
        assert!(format!("{parse}").contains("oops"));
        let dup_name = ToolPluginLoadError::DuplicateName("alpha".to_string());
        assert!(format!("{dup_name}").contains("alpha"));
        let dup_tool = ToolPluginLoadError::DuplicateTool {
            tool_id: "t".to_string(),
            by_plugins: vec!["a".to_string(), "b".to_string()],
        };
        let s = format!("{dup_tool}");
        assert!(s.contains('a'));
        assert!(s.contains('b'));
    }

    #[test]
    fn load_error_source_only_for_io() {
        let io_err = ToolPluginLoadError::Io(io::Error::other("broken"));
        assert!(io_err.source().is_some());
        let parse = ToolPluginLoadError::Parse("oops".to_string());
        assert!(parse.source().is_none());
    }

    #[test]
    fn tail_text_short_passthrough() {
        assert_eq!(tail_text("hello", 256), "hello");
    }

    #[test]
    fn tail_text_truncates_long_strings() {
        let s = "a".repeat(300);
        let out = tail_text(&s, 256);
        assert_eq!(out.chars().count(), 256);
    }

    #[test]
    fn parse_plugin_stdout_empty_returns_bad_output() {
        let inv = sample_invocation("alpha.do");
        let r = parse_plugin_stdout(&inv, b"", b"err tail");
        match r {
            ToolResult::Err { code, message, .. } => {
                assert_eq!(code, E_PLUGIN_BAD_OUTPUT);
                assert!(message.contains("err tail"));
            }
            ToolResult::Ok { .. } => panic!("expected Err"),
        }
    }

    #[test]
    fn parse_plugin_stdout_non_utf8_returns_bad_output() {
        let inv = sample_invocation("alpha.do");
        let bad: &[u8] = &[0xff, 0xfe, 0xfd];
        let r = parse_plugin_stdout(&inv, bad, b"");
        match r {
            ToolResult::Err { code, .. } => assert_eq!(code, E_PLUGIN_BAD_OUTPUT),
            ToolResult::Ok { .. } => panic!("expected Err"),
        }
    }

    #[test]
    fn parse_plugin_stdout_invalid_json_returns_bad_output() {
        let inv = sample_invocation("alpha.do");
        let r = parse_plugin_stdout(&inv, b"not json\n", b"");
        match r {
            ToolResult::Err { code, .. } => assert_eq!(code, E_PLUGIN_BAD_OUTPUT),
            ToolResult::Ok { .. } => panic!("expected Err"),
        }
    }

    #[test]
    fn parse_plugin_stdout_valid_ok_passes_through() {
        let inv = sample_invocation("alpha.do");
        let payload = serde_json::json!({
            "status": "ok",
            "tool_id": "alpha.do",
            "body": {"hello": "world"},
            "bytes": 17_u64,
        })
        .to_string();
        let line = format!("{payload}\n");
        let r = parse_plugin_stdout(&inv, line.as_bytes(), b"");
        match r {
            ToolResult::Ok { tool_id, .. } => assert_eq!(tool_id, "alpha.do"),
            ToolResult::Err { .. } => panic!("expected Ok"),
        }
    }

    // ---- Unix-gated integration tests (real subprocess) -----------------
    //
    // These tests stage a tiny shell-script "plugin" in a TempDir and
    // exercise `ToolPluginDispatcher::invoke` end-to-end. They are skipped
    // on Windows where `#!/bin/sh` execution is not portable.

    #[cfg(unix)]
    mod unix {
        use std::os::unix::fs::PermissionsExt;

        use super::*;

        fn write_script(dir: &Path, name: &str, body: &str) -> PathBuf {
            let path = dir.join(name);
            fs::write(&path, body).expect("write script");
            let mut perms = fs::metadata(&path).expect("stat").permissions();
            perms.set_mode(0o755);
            fs::set_permissions(&path, perms).expect("chmod");
            path
        }

        fn registry_for(
            binary: PathBuf,
            supports: &[&str],
            timeout_ms: u64,
        ) -> Arc<dyn ToolPluginRegistry> {
            let m = ToolPluginManifest {
                name: "scripted".to_string(),
                version: "0.0.1".to_string(),
                binary,
                supports: supports.iter().map(|s| (*s).to_string()).collect(),
                timeout_ms,
            };
            Arc::new(
                FileSystemPluginRegistry::from_plugins(vec![ToolPlugin::new(
                    m,
                    PathBuf::from("/tmp"),
                )])
                .expect("build"),
            )
        }

        #[test]
        fn invoke_routes_to_registered_plugin() {
            let dir = TempDir::new().expect("tmp");
            let script = write_script(
                dir.path(),
                "ok.sh",
                "#!/bin/sh\nprintf '{\"status\":\"ok\",\"tool_id\":\"alpha.do\",\"body\":{\"hi\":1},\"bytes\":2}\\n'\n",
            );
            let reg = registry_for(script, &["alpha.do"], 5_000);
            let d = ToolPluginDispatcher::new(reg);
            let inv = sample_invocation("alpha.do");
            match d.invoke(&inv) {
                ToolResult::Ok { tool_id, bytes, .. } => {
                    assert_eq!(tool_id, "alpha.do");
                    assert_eq!(bytes, 2);
                }
                ToolResult::Err { code, message, .. } => {
                    panic!("expected Ok, got Err {code}: {message}")
                }
            }
        }

        #[test]
        fn invoke_malformed_json_yields_bad_output() {
            let dir = TempDir::new().expect("tmp");
            let script = write_script(dir.path(), "bad.sh", "#!/bin/sh\necho 'not json at all'\n");
            let reg = registry_for(script, &["alpha.do"], 5_000);
            let d = ToolPluginDispatcher::new(reg);
            let inv = sample_invocation("alpha.do");
            match d.invoke(&inv) {
                ToolResult::Err { code, .. } => assert_eq!(code, E_PLUGIN_BAD_OUTPUT),
                ToolResult::Ok { .. } => panic!("expected Err"),
            }
        }

        #[test]
        fn invoke_nonzero_exit_yields_bad_output_with_stderr_tail() {
            let dir = TempDir::new().expect("tmp");
            let script = write_script(
                dir.path(),
                "fail.sh",
                "#!/bin/sh\necho 'plugin angry' 1>&2\nexit 7\n",
            );
            let reg = registry_for(script, &["alpha.do"], 5_000);
            let d = ToolPluginDispatcher::new(reg);
            let inv = sample_invocation("alpha.do");
            match d.invoke(&inv) {
                ToolResult::Err { code, message, .. } => {
                    assert_eq!(code, E_PLUGIN_BAD_OUTPUT);
                    assert!(message.contains("plugin angry"));
                    assert!(message.contains('7'));
                }
                ToolResult::Ok { .. } => panic!("expected Err"),
            }
        }

        #[test]
        fn invoke_timeout_yields_timeout_code() {
            let dir = TempDir::new().expect("tmp");
            let script = write_script(dir.path(), "slow.sh", "#!/bin/sh\nsleep 5\n");
            let reg = registry_for(script, &["alpha.do"], 100);
            let d = ToolPluginDispatcher::new(reg);
            let inv = sample_invocation("alpha.do");
            match d.invoke(&inv) {
                ToolResult::Err { code, .. } => assert_eq!(code, E_PLUGIN_TIMEOUT),
                ToolResult::Ok { .. } => panic!("expected timeout"),
            }
        }
    }
}
