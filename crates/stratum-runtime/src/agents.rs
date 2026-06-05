//! User-agent loader.
//!
//! Phase 3 v2 prep — parses `<config>/stratum/agents/<name>.toml` files
//! into typed `AgentDef`s and validates each against the global
//! capability matrix. Hot-reload via `notify`, the `/agent` palette
//! surface, and the actual orchestrator binding (role → agent →
//! provider) land in Phase 3 alongside the tool registry impl.
//!
//! Per `plan/19-user-agents-and-plugins.md`.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use stratum_types::error::codes::E1001_INSTALLED_SCHEMA_UNREADABLE;
use stratum_types::{ModelId, RoleId, StratumError, StratumResult};

use crate::tools::CapabilityMatrix;

const SUFFIX: &str = ".toml";

/// Per-agent execution budget. Mirrors the TOML `[budget]` table.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
#[allow(
    clippy::struct_field_names,
    reason = "TOML schema dictates the field names; renaming would break the contract"
)]
pub struct AgentBudget {
    /// Maximum tokens per turn.
    pub max_tokens_per_turn: u32,
    /// Maximum wall-clock seconds per turn.
    pub max_wall_seconds: u32,
    /// Maximum simultaneous invocations of this agent.
    pub max_concurrent_invocations: u32,
}

impl Default for AgentBudget {
    fn default() -> Self {
        Self {
            max_tokens_per_turn: 4096,
            max_wall_seconds: 90,
            max_concurrent_invocations: 1,
        }
    }
}

/// One user-authored agent definition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct AgentDef {
    /// On-disk schema version; bumped only when a field's semantics break.
    pub schema_version: u32,
    /// Stable name; matches the TOML file stem.
    pub name: String,
    /// Free-form description shown in the `/agent` palette.
    pub description: String,
    /// Roles this agent can fulfill (`coder`, `critic`, …).
    pub roles: Vec<RoleId>,
    /// Primary model binding.
    pub model: ModelId,
    /// Optional fallback model when the primary fails the memory probe.
    pub fallback_model: Option<ModelId>,
    /// Tool allowlist (intersected with the global matrix at load time).
    pub tools: CapabilityMatrix,
    /// Sandbox profile name. Resolved against the installed profiles.
    pub sandbox: String,
    /// Whether internal I/O is routed through the caveman rewriter.
    pub caveman_io: bool,
    /// Whether the agent appears in the `/agent` palette picker.
    pub visible_in_palette: bool,
    /// Hard cap on tool calls per turn.
    pub max_tool_calls: u32,
    /// Budgets.
    pub budget: AgentBudget,
}

impl Default for AgentDef {
    fn default() -> Self {
        Self {
            schema_version: 1,
            name: String::new(),
            description: String::new(),
            roles: Vec::new(),
            model: ModelId::from("echo"),
            fallback_model: None,
            tools: CapabilityMatrix::new(),
            sandbox: "passthrough".to_string(),
            caveman_io: true,
            visible_in_palette: true,
            max_tool_calls: 24,
            budget: AgentBudget::default(),
        }
    }
}

impl AgentDef {
    /// Validate the definition against the global tool matrix. Returns
    /// `Ok(narrowed)` where `narrowed` is the agent's allowlist intersected
    /// with the global one; returns an error when the agent requests a
    /// capability the global matrix does not grant.
    ///
    /// # Errors
    /// Returns [`E1001_INSTALLED_SCHEMA_UNREADABLE`] when the agent's
    /// tool allowlist requests an entry outside the global matrix.
    pub fn validate_against(&self, global: &CapabilityMatrix) -> StratumResult<CapabilityMatrix> {
        for entry in self.tools.entries() {
            let (verb, arg) = entry.parts();
            if !global.allows(verb, arg) {
                return Err(StratumError::new(
                    E1001_INSTALLED_SCHEMA_UNREADABLE,
                    format!(
                        "agent {} requests `{}` which is outside the global capability matrix",
                        self.name,
                        entry.as_str()
                    ),
                ));
            }
        }
        Ok(self.tools.narrowed_by(global))
    }
}

/// Loader that walks `<config>/stratum/agents/` and parses each file.
#[derive(Debug, Clone, Default)]
pub struct AgentLoader;

impl AgentLoader {
    /// List every `*.toml` file directly under `dir` and parse it.
    /// Parse failures fail the load so misconfigurations are loud; a
    /// missing directory returns an empty list.
    ///
    /// # Errors
    /// Returns [`E1001_INSTALLED_SCHEMA_UNREADABLE`] on any io or parse
    /// failure inside the directory.
    pub fn load_dir(dir: &Path) -> StratumResult<Vec<AgentDef>> {
        let entries = match std::fs::read_dir(dir) {
            Ok(iter) => iter,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => {
                return Err(StratumError::new(
                    E1001_INSTALLED_SCHEMA_UNREADABLE,
                    format!("read agents dir {}", dir.display()),
                )
                .with_cause(e));
            }
        };
        let mut out = Vec::new();
        let mut paths: Vec<PathBuf> = entries
            .filter_map(Result::ok)
            .map(|e| e.path())
            .filter(|p| {
                p.is_file()
                    && p.file_name()
                        .and_then(|n| n.to_str())
                        .is_some_and(|n| n.ends_with(SUFFIX))
            })
            .collect();
        paths.sort();
        for path in paths {
            out.push(Self::load_file(&path)?);
        }
        Ok(out)
    }

    /// Parse a single agent TOML file.
    ///
    /// # Errors
    /// Returns [`E1001_INSTALLED_SCHEMA_UNREADABLE`] on read or parse
    /// failure.
    pub fn load_file(path: &Path) -> StratumResult<AgentDef> {
        let raw = std::fs::read_to_string(path).map_err(|e| {
            StratumError::new(
                E1001_INSTALLED_SCHEMA_UNREADABLE,
                format!("read {}", path.display()),
            )
            .with_cause(e)
        })?;
        let mut def: AgentDef = toml_edit::de::from_str(&raw).map_err(|e| {
            StratumError::new(
                E1001_INSTALLED_SCHEMA_UNREADABLE,
                format!("parse {}", path.display()),
            )
            .with_cause(e)
        })?;
        // Default the name from the file stem when the TOML omits it.
        if def.name.is_empty() {
            if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                def.name = stem.to_string();
            }
        }
        Ok(def)
    }
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::*;

    fn write_agent(dir: &Path, name: &str, body: &str) -> PathBuf {
        let path = dir.join(format!("{name}.toml"));
        std::fs::write(&path, body).unwrap();
        path
    }

    fn minimal_body() -> &'static str {
        r#"
schema_version = 1
name = "tight-reviewer"
description = "reviews diffs with a small reasoner"
roles = ["critic"]
model = "deepseek-r1-distill-qwen-1.5b-q4_k_m"
tools = ["fs.read", "git.diff"]
sandbox = "bwrap-strict"
"#
    }

    #[test]
    fn budget_default_is_phase19() {
        let b = AgentBudget::default();
        assert_eq!(b.max_tokens_per_turn, 4096);
        assert_eq!(b.max_wall_seconds, 90);
        assert_eq!(b.max_concurrent_invocations, 1);
    }

    #[test]
    fn agent_def_default_is_echo_passthrough() {
        let d = AgentDef::default();
        assert_eq!(d.schema_version, 1);
        assert_eq!(d.model, ModelId::from("echo"));
        assert_eq!(d.sandbox, "passthrough");
        assert!(d.caveman_io);
    }

    #[test]
    fn load_file_minimal_parses() {
        let tmp = TempDir::new().unwrap();
        let path = write_agent(tmp.path(), "tight-reviewer", minimal_body());
        let def = AgentLoader::load_file(&path).unwrap();
        assert_eq!(def.name, "tight-reviewer");
        assert_eq!(def.roles, vec![RoleId::from("critic")]);
        assert_eq!(def.sandbox, "bwrap-strict");
        assert_eq!(def.tools.len(), 2);
    }

    #[test]
    fn load_file_defaults_name_from_stem_when_missing() {
        let tmp = TempDir::new().unwrap();
        let body = r#"
schema_version = 1
description = "no name in TOML"
roles = []
model = "echo"
tools = []
sandbox = "passthrough"
"#;
        let path = write_agent(tmp.path(), "from-stem", body);
        let def = AgentLoader::load_file(&path).unwrap();
        assert_eq!(def.name, "from-stem");
    }

    #[test]
    fn load_file_propagates_parse_errors() {
        let tmp = TempDir::new().unwrap();
        let path = write_agent(tmp.path(), "broken", "not = [ valid");
        let err = AgentLoader::load_file(&path).unwrap_err();
        assert_eq!(err.code(), &E1001_INSTALLED_SCHEMA_UNREADABLE);
    }

    #[test]
    fn load_file_propagates_read_errors() {
        let tmp = TempDir::new().unwrap();
        let missing = tmp.path().join("missing.toml");
        let err = AgentLoader::load_file(&missing).unwrap_err();
        assert_eq!(err.code(), &E1001_INSTALLED_SCHEMA_UNREADABLE);
    }

    #[test]
    fn load_dir_returns_empty_when_dir_missing() {
        let tmp = TempDir::new().unwrap();
        let nonexistent = tmp.path().join("agents");
        let defs = AgentLoader::load_dir(&nonexistent).unwrap();
        assert!(defs.is_empty());
    }

    #[test]
    fn load_dir_collects_every_toml_alphabetically() {
        let tmp = TempDir::new().unwrap();
        write_agent(tmp.path(), "b-second", &minimal_body_named("b-second"));
        write_agent(tmp.path(), "a-first", &minimal_body_named("a-first"));
        // No .toml suffix → skipped by the filter.
        std::fs::write(tmp.path().join("notes.md"), b"# notes").unwrap();
        std::fs::create_dir(tmp.path().join("subdir")).unwrap();
        let defs = AgentLoader::load_dir(tmp.path()).unwrap();
        assert_eq!(defs.len(), 2);
        assert_eq!(defs[0].name, "a-first");
        assert_eq!(defs[1].name, "b-second");
    }

    fn minimal_body_named(name: &str) -> String {
        format!(
            r#"
schema_version = 1
name = "{name}"
description = "x"
roles = ["critic"]
model = "echo"
tools = []
sandbox = "passthrough"
"#
        )
    }

    #[test]
    fn load_dir_fails_when_any_toml_is_malformed() {
        let tmp = TempDir::new().unwrap();
        write_agent(tmp.path(), "good", minimal_body());
        write_agent(tmp.path(), "bad", "definitely = not valid");
        let err = AgentLoader::load_dir(tmp.path()).unwrap_err();
        assert_eq!(err.code(), &E1001_INSTALLED_SCHEMA_UNREADABLE);
    }

    #[test]
    fn validate_against_global_keeps_allowed_tools() {
        let tmp = TempDir::new().unwrap();
        let path = write_agent(tmp.path(), "ok", minimal_body());
        let def = AgentLoader::load_file(&path).unwrap();
        let global = CapabilityMatrix::from_entries(["fs.read", "git.*", "bash.run"]);
        let narrowed = def.validate_against(&global).unwrap();
        assert_eq!(narrowed.len(), 2);
    }

    #[test]
    fn validate_against_global_rejects_unknown_capability() {
        let tmp = TempDir::new().unwrap();
        let path = write_agent(tmp.path(), "naughty", minimal_body());
        let def = AgentLoader::load_file(&path).unwrap();
        // Global denies git.* entirely.
        let global = CapabilityMatrix::from_entries(["fs.read"]);
        let err = def.validate_against(&global).unwrap_err();
        assert_eq!(err.code(), &E1001_INSTALLED_SCHEMA_UNREADABLE);
        assert!(format!("{err}").contains("git.diff"));
    }

    #[test]
    fn agent_def_serde_roundtrip() {
        let tmp = TempDir::new().unwrap();
        let path = write_agent(tmp.path(), "rt", minimal_body());
        let def = AgentLoader::load_file(&path).unwrap();
        let s = serde_json::to_string(&def).unwrap();
        let back: AgentDef = serde_json::from_str(&s).unwrap();
        assert_eq!(def, back);
    }

    #[test]
    fn load_dir_skips_non_toml_files() {
        let tmp = TempDir::new().unwrap();
        std::fs::write(tmp.path().join("notes.md"), b"# notes").unwrap();
        write_agent(tmp.path(), "real", &minimal_body_named("real"));
        let defs = AgentLoader::load_dir(tmp.path()).unwrap();
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].name, "real");
    }

    #[cfg(unix)]
    #[test]
    fn load_dir_propagates_io_error_when_unreadable() {
        use std::os::unix::fs::PermissionsExt;
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path().join("locked");
        std::fs::create_dir(&dir).unwrap();
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o000)).unwrap();
        let result = AgentLoader::load_dir(&dir);
        // Restore so TempDir can clean up.
        let _ = std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o755));
        let err = result.unwrap_err();
        assert_eq!(err.code(), &E1001_INSTALLED_SCHEMA_UNREADABLE);
    }
}
