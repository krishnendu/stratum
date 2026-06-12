//! Subagent primitive per `plan/37-dynamic-agents-and-workflows.md`.
//!
//! A subagent is a specialized worker the parent LLM can delegate a
//! side task to. It carries:
//!
//! - A custom system prompt
//! - A restricted tool allowlist
//! - An optional model-tier override
//! - An optional permission-mode override
//!
//! Phase 3 v2 scaffold scope (this module): the data type + a loader
//! that scans `<config>/stratum/subagents/` plus a built-in seed
//! registry. Per-subagent context isolation, memory scopes, worktree
//! spawn, and hooks land in later passes; see `plan/37` §2.4 + §10.
//!
//! The schema is intentionally a strict subset of the Claude Code
//! frontmatter so that future imports (a `claude2stratum` migrator) are
//! mechanical.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// A registered subagent.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Subagent {
    /// Unique identifier using lowercase letters and hyphens.
    pub name: String,
    /// One-line description used by the parent LLM to decide when to
    /// delegate to this subagent.
    pub description: String,
    /// Tool ids this subagent may invoke. `None` = inherit parent.
    #[serde(default)]
    pub tools: Option<Vec<String>>,
    /// Tools to deny (subtracted from inherited list).
    #[serde(default)]
    pub denied_tools: Vec<String>,
    /// Model tier override (`low` / `medium` / `high` / `xl`). `None` =
    /// inherit session's tier.
    #[serde(default)]
    pub model_tier: Option<String>,
    /// Permission-mode override (`default` / `accept_edits` / `auto` /
    /// `dont_ask` / `bypass` / `plan`).
    #[serde(default)]
    pub permission_mode: Option<String>,
    /// Maximum agentic turns before the subagent stops.
    #[serde(default)]
    pub max_turns: Option<u32>,
    /// Persistent-memory scope (`user` / `project` / `local`). `None` =
    /// no memory.
    #[serde(default)]
    pub memory: Option<String>,
    /// Effort level (`low` / `medium` / `high` / `xhigh`).
    #[serde(default)]
    pub effort: Option<String>,
    /// `worktree` to run in a temporary `git worktree`; otherwise `none`.
    #[serde(default)]
    pub isolation: Option<String>,
    /// Display color (`red` / `blue` / ... / `cyan`).
    #[serde(default)]
    pub color: Option<String>,
    /// System prompt body.
    pub prompt: String,
}

/// In-memory registry of subagents keyed by `name`.
#[derive(Debug, Default, Clone)]
pub struct SubagentRegistry {
    by_name: BTreeMap<String, Subagent>,
}

impl SubagentRegistry {
    /// Empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Seeded with the v1 built-in subagents documented in
    /// `plan/37` §2.4: `explore`, `code-reviewer`, `code-architect`,
    /// `code-explorer`, `general-purpose`, `plan`.
    #[must_use]
    pub fn with_builtins() -> Self {
        let mut reg = Self::new();
        for s in builtin_seed() {
            reg.by_name.insert(s.name.clone(), s);
        }
        reg
    }

    /// Look up a subagent by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&Subagent> {
        self.by_name.get(name)
    }

    /// All registered subagents in name order.
    #[must_use]
    pub fn list(&self) -> Vec<&Subagent> {
        self.by_name.values().collect()
    }

    /// Insert / replace a subagent. Returns the previous entry if any.
    pub fn insert(&mut self, s: Subagent) -> Option<Subagent> {
        self.by_name.insert(s.name.clone(), s)
    }

    /// Number of registered subagents.
    #[must_use]
    pub fn len(&self) -> usize {
        self.by_name.len()
    }

    /// Whether the registry has any entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.by_name.is_empty()
    }
}

/// Load every `*.toml` in `dir` as a subagent. Returns the registry
/// plus a list of `(file, parse_error)` pairs for files that failed.
/// Missing dir → empty registry, no error.
///
/// # Errors
///
/// Surfaces `std::io::Error` on permission failures reading the dir
/// entry list itself; per-file parse errors are returned as the
/// second tuple element so a single bad file does not block the rest.
pub fn load_dir(dir: &Path) -> std::io::Result<(SubagentRegistry, Vec<(PathBuf, String)>)> {
    let mut reg = SubagentRegistry::new();
    let mut errors: Vec<(PathBuf, String)> = Vec::new();
    let entries = match std::fs::read_dir(dir) {
        Ok(it) => it,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok((reg, errors)),
        Err(e) => return Err(e),
    };
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("toml") {
            continue;
        }
        match std::fs::read_to_string(&path) {
            Ok(body) => match toml_edit::de::from_str::<Subagent>(&body) {
                Ok(s) => {
                    reg.insert(s);
                }
                Err(e) => errors.push((path, e.to_string())),
            },
            Err(e) => errors.push((path, e.to_string())),
        }
    }
    Ok((reg, errors))
}

fn builtin_seed() -> Vec<Subagent> {
    vec![
        Subagent {
            name: "explore".to_string(),
            description: "Fast read-only codebase search. Use to locate files, grep symbols, map a directory.".to_string(),
            tools: Some(vec!["fs.read".to_string(), "grep".to_string(), "glob".to_string()]),
            denied_tools: Vec::new(),
            model_tier: Some("low".to_string()),
            permission_mode: Some("auto".to_string()),
            max_turns: Some(8),
            memory: None,
            effort: Some("low".to_string()),
            isolation: None,
            color: Some("cyan".to_string()),
            prompt: "You are a read-only code explorer. Use grep / glob / fs.read to locate \
                     code. Return findings as a compact file:line table. Do not propose fixes."
                .to_string(),
        },
        Subagent {
            name: "code-reviewer".to_string(),
            description: "Reviews diffs for security + correctness. Use proactively after edits.".to_string(),
            tools: Some(vec![
                "fs.read".to_string(),
                "grep".to_string(),
                "shell.exec".to_string(),
            ]),
            denied_tools: vec!["fs.write".to_string(), "fs.edit".to_string()],
            model_tier: Some("high".to_string()),
            permission_mode: Some("default".to_string()),
            max_turns: Some(12),
            memory: Some("project".to_string()),
            effort: Some("high".to_string()),
            isolation: None,
            color: Some("yellow".to_string()),
            prompt: "You are a senior code reviewer. Focus on security vulnerabilities, \
                     correctness bugs, concurrency hazards. Ignore style nits. \
                     Report findings as one line per issue: `path:line: <severity>: <problem>. <fix>.`"
                .to_string(),
        },
        Subagent {
            name: "code-architect".to_string(),
            description: "Designs feature architectures by analyzing existing patterns. Returns implementation blueprints.".to_string(),
            tools: Some(vec![
                "fs.read".to_string(),
                "grep".to_string(),
                "glob".to_string(),
            ]),
            denied_tools: Vec::new(),
            model_tier: Some("high".to_string()),
            permission_mode: Some("default".to_string()),
            max_turns: Some(15),
            memory: Some("project".to_string()),
            effort: Some("high".to_string()),
            isolation: None,
            color: Some("blue".to_string()),
            prompt: "You are a software architect. Analyze the existing codebase patterns and \
                     produce a comprehensive implementation blueprint: files to create / modify, \
                     component designs, data flows, and a build sequence. Do not write code yet."
                .to_string(),
        },
        Subagent {
            name: "general-purpose".to_string(),
            description: "Capable agent for multi-step tasks needing both exploration and action.".to_string(),
            tools: None, // inherit all
            denied_tools: Vec::new(),
            model_tier: None, // inherit
            permission_mode: Some("default".to_string()),
            max_turns: Some(20),
            memory: None,
            effort: Some("medium".to_string()),
            isolation: None,
            color: Some("green".to_string()),
            prompt: "You are a capable assistant. Delegate exploration to grep / glob / fs.read; \
                     make edits with fs.edit / fs.write; verify with shell.exec. Return a concise \
                     summary of what changed."
                .to_string(),
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn builtins_present() {
        let reg = SubagentRegistry::with_builtins();
        assert!(reg.get("explore").is_some());
        assert!(reg.get("code-reviewer").is_some());
        assert!(reg.get("code-architect").is_some());
        assert!(reg.get("general-purpose").is_some());
    }

    #[test]
    fn list_returns_sorted_by_name() {
        let reg = SubagentRegistry::with_builtins();
        let names: Vec<&str> = reg.list().iter().map(|s| s.name.as_str()).collect();
        let mut sorted = names.clone();
        sorted.sort_unstable();
        assert_eq!(names, sorted);
    }

    #[test]
    fn load_dir_missing_returns_empty_no_error() {
        let tmp = TempDir::new().unwrap();
        let missing = tmp.path().join("nope");
        let (reg, errors) = load_dir(&missing).unwrap();
        assert!(reg.is_empty());
        assert!(errors.is_empty());
    }

    #[test]
    fn load_dir_parses_valid_toml() {
        let tmp = TempDir::new().unwrap();
        std::fs::write(
            tmp.path().join("custom.toml"),
            r#"
name = "custom"
description = "test agent"
tools = ["fs.read"]
prompt = "Be helpful."
"#,
        )
        .unwrap();
        let (reg, errors) = load_dir(tmp.path()).unwrap();
        assert!(errors.is_empty());
        let s = reg.get("custom").expect("custom registered");
        assert_eq!(s.description, "test agent");
        assert_eq!(s.tools.as_deref(), Some(&["fs.read".to_string()][..]));
    }

    #[test]
    fn load_dir_records_per_file_parse_errors() {
        let tmp = TempDir::new().unwrap();
        std::fs::write(tmp.path().join("bad.toml"), "not = valid = toml").unwrap();
        let (reg, errors) = load_dir(tmp.path()).unwrap();
        assert!(reg.is_empty());
        assert_eq!(errors.len(), 1);
    }

    #[test]
    fn load_dir_ignores_non_toml_files() {
        let tmp = TempDir::new().unwrap();
        std::fs::write(tmp.path().join("readme.md"), "ignore me").unwrap();
        let (reg, errors) = load_dir(tmp.path()).unwrap();
        assert!(reg.is_empty());
        assert!(errors.is_empty());
    }
}
