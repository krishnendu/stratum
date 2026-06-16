//! Auto-memory storage layer — `MEMORY.md` index + per-memory body files.
//!
//! Implements `plan/40 §2-§5`. The runtime provides:
//!
//! - **Repo-id derivation** from git remote / cwd path
//! - **Storage layout** under `<config>/stratum/projects/<repo-id>/memory/`
//! - **Index parsing + serialization** (MEMORY.md: one line per memory file)
//! - **Body CRUD** (load/save/forget)
//! - **Opt-out check** (`.stratum/config.toml [memory] auto = false` /
//!   `STRATUM_AUTO_MEMORY=0`)
//!
//! What this does NOT do (deferred):
//!
//! - The LLM-side decision of WHEN to save — that lives in the system
//!   prompt + the orchestrator. This module is the storage primitive.
//! - Read rules — orchestrator decides what to load into context.
//! - Crash-report redactor extension — `plan/40 §10`, Phase 5.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Four memory types per plan/40 §3.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MemoryType {
    /// Facts about the user (role, preferences, knowledge).
    User,
    /// Corrections / validated approaches from the user.
    Feedback,
    /// Ongoing project state, motivations, deadlines.
    Project,
    /// Pointers to external systems (Linear, Grafana, etc.).
    Reference,
}

/// One memory entry on disk.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryFrontmatter {
    /// Short kebab-case slug; becomes the filename `<name>.md`.
    pub name: String,
    /// One-line summary read by the index scanner.
    pub description: String,
    /// Memory taxonomy bucket.
    #[serde(rename = "type")]
    pub kind: MemoryType,
}

/// Parsed memory file: frontmatter + body.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Memory {
    /// Metadata; serialized as YAML frontmatter.
    pub frontmatter: MemoryFrontmatter,
    /// Markdown body (the rule / fact / pointer text).
    pub body: String,
}

/// Storage handle scoped to one repo's `memory/` directory.
#[derive(Debug, Clone)]
pub struct AutoMemoryStore {
    root: PathBuf,
}

impl AutoMemoryStore {
    /// Open (or create) the store at `<config>/stratum/projects/<repo-id>/memory/`.
    /// `config_root` is `<config>/stratum/`; `repo_id` comes from
    /// [`repo_id_for`].
    ///
    /// # Errors
    /// Returns `Err` when the directory cannot be created.
    pub fn open(config_root: &Path, repo_id: &str) -> std::io::Result<Self> {
        let root = config_root.join("projects").join(repo_id).join("memory");
        std::fs::create_dir_all(&root)?;
        Ok(Self { root })
    }

    /// Path to the index file (`MEMORY.md`).
    #[must_use]
    pub fn index_path(&self) -> PathBuf {
        self.root.join("MEMORY.md")
    }

    /// Load the index. Returns an empty vec when MEMORY.md doesn't exist.
    /// Lines that don't parse are silently dropped — the index is a
    /// human-edited file and should never error the runtime.
    #[must_use]
    pub fn load_index(&self) -> Vec<IndexEntry> {
        let Ok(raw) = std::fs::read_to_string(self.index_path()) else {
            return Vec::new();
        };
        raw.lines().filter_map(IndexEntry::parse).collect()
    }

    /// Save the index, atomically (write to a tmp file + rename).
    ///
    /// # Errors
    /// Returns `Err` on filesystem failure.
    pub fn save_index(&self, entries: &[IndexEntry]) -> std::io::Result<()> {
        let tmp = self.root.join("MEMORY.md.tmp");
        let body = entries
            .iter()
            .map(IndexEntry::render)
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(&tmp, body + "\n")?;
        std::fs::rename(tmp, self.index_path())
    }

    /// Load one memory by name.
    #[must_use]
    pub fn load(&self, name: &str) -> Option<Memory> {
        let raw = std::fs::read_to_string(self.body_path(name)).ok()?;
        parse_memory(&raw)
    }

    /// Save one memory (creates or replaces). Also updates the index.
    ///
    /// # Errors
    /// Returns `Err` on filesystem failure.
    pub fn save(&self, memory: &Memory) -> std::io::Result<()> {
        let body = render_memory(memory);
        let path = self.body_path(&memory.frontmatter.name);
        let tmp = path.with_extension("md.tmp");
        std::fs::write(&tmp, body)?;
        std::fs::rename(tmp, path)?;
        // Upsert into the index.
        let mut entries = self.load_index();
        if let Some(existing) = entries
            .iter_mut()
            .find(|e| e.name == memory.frontmatter.name)
        {
            existing
                .description
                .clone_from(&memory.frontmatter.description);
        } else {
            entries.push(IndexEntry {
                name: memory.frontmatter.name.clone(),
                description: memory.frontmatter.description.clone(),
            });
        }
        self.save_index(&entries)
    }

    /// Forget one memory (remove body + index entry).
    ///
    /// # Errors
    /// Returns `Err` on filesystem failure (other than the file
    /// already being absent — that's not an error).
    pub fn forget(&self, name: &str) -> std::io::Result<()> {
        let path = self.body_path(name);
        if path.exists() {
            std::fs::remove_file(&path)?;
        }
        let entries: Vec<IndexEntry> = self
            .load_index()
            .into_iter()
            .filter(|e| e.name != name)
            .collect();
        self.save_index(&entries)
    }

    /// All memories, ordered by `name`.
    #[must_use]
    pub fn list(&self) -> Vec<IndexEntry> {
        let mut entries = self.load_index();
        entries.sort_by(|a, b| a.name.cmp(&b.name));
        entries
    }

    /// Delete the whole `memory/` dir for this repo. Used by
    /// `/memory clear`. Caller is responsible for the user-confirm UI.
    ///
    /// # Errors
    /// Returns `Err` if the directory can't be removed.
    pub fn clear(self) -> std::io::Result<()> {
        if self.root.exists() {
            std::fs::remove_dir_all(self.root)?;
        }
        Ok(())
    }

    fn body_path(&self, name: &str) -> PathBuf {
        self.root.join(format!("{name}.md"))
    }
}

/// One row of the `MEMORY.md` index. Format:
///   - [Title](slug.md) — one-line hook
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexEntry {
    /// The memory's stable slug (matches `<slug>.md`).
    pub name: String,
    /// Description rendered after the em dash on the index line.
    pub description: String,
}

impl IndexEntry {
    fn parse(line: &str) -> Option<Self> {
        let line = line.trim();
        // Expect: `- [Title](<name>.md) — description`
        let inner = line.strip_prefix("- [")?;
        let close_bracket = inner.find("](")?;
        // We intentionally skip the human title; the slug after `](` is
        // the canonical key used to dedupe entries.
        let after = &inner[close_bracket + 2..];
        let close_paren = after.find(')')?;
        let path = &after[..close_paren];
        let name = path.strip_suffix(".md")?.to_string();
        let tail = &after[close_paren + 1..];
        // Allow either `— ` or `-- ` or just ` `.
        let description = tail
            .trim_start_matches(' ')
            .trim_start_matches('—')
            .trim_start_matches("--")
            .trim()
            .to_string();
        Some(Self { name, description })
    }

    fn render(&self) -> String {
        let title = title_case(&self.name);
        format!("- [{}]({}.md) — {}", title, self.name, self.description)
    }
}

fn title_case(slug: &str) -> String {
    slug.split('_')
        .map(|w| {
            let mut chars = w.chars();
            chars.next().map_or_else(String::new, |c| {
                c.to_ascii_uppercase().to_string() + chars.as_str()
            })
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Parse a `<name>.md` body — YAML frontmatter between `---` markers,
/// then body. Returns `None` when frontmatter is missing or malformed.
fn parse_memory(raw: &str) -> Option<Memory> {
    let stripped = raw.strip_prefix("---\n")?;
    let close_idx = stripped.find("\n---\n")?;
    let frontmatter_str = &stripped[..close_idx];
    let body = &stripped[close_idx + "\n---\n".len()..];
    let frontmatter: MemoryFrontmatter = serde_yaml_style_parse(frontmatter_str)?;
    Some(Memory {
        frontmatter,
        body: body.trim_end().to_string(),
    })
}

fn render_memory(memory: &Memory) -> String {
    let kind_str = match memory.frontmatter.kind {
        MemoryType::User => "user",
        MemoryType::Feedback => "feedback",
        MemoryType::Project => "project",
        MemoryType::Reference => "reference",
    };
    format!(
        "---\nname: {}\ndescription: {}\ntype: {}\n---\n\n{}\n",
        memory.frontmatter.name, memory.frontmatter.description, kind_str, memory.body
    )
}

/// Minimal YAML-style parser for the three known frontmatter fields.
/// We avoid pulling in a full YAML dep — these files are
/// runtime-authored and the shape is fixed.
fn serde_yaml_style_parse(s: &str) -> Option<MemoryFrontmatter> {
    let mut name = None;
    let mut description = None;
    let mut kind = None;
    for line in s.lines() {
        let line = line.trim_end();
        if let Some(rest) = line.strip_prefix("name:") {
            name = Some(rest.trim().to_string());
        } else if let Some(rest) = line.strip_prefix("description:") {
            description = Some(rest.trim().to_string());
        } else if let Some(rest) = line.strip_prefix("type:") {
            kind = match rest.trim() {
                "user" => Some(MemoryType::User),
                "feedback" => Some(MemoryType::Feedback),
                "project" => Some(MemoryType::Project),
                "reference" => Some(MemoryType::Reference),
                _ => return None,
            };
        }
    }
    Some(MemoryFrontmatter {
        name: name?,
        description: description?,
        kind: kind?,
    })
}

/// Derive a stable per-repo id from the workspace's git remote URL.
/// Falls back to the absolute cwd path's hash when no git remote.
///
/// Same repo on different machines → same id (via remote URL).
/// Different repos with the same path on different machines → different ids
/// (via cwd hash).
#[must_use]
pub fn repo_id_for(cwd: &Path) -> Option<String> {
    if let Some(url) = read_git_origin(cwd) {
        return Some(hash16(&url));
    }
    cwd.canonicalize().ok().and_then(|p| p.to_str().map(hash16))
}

fn read_git_origin(cwd: &Path) -> Option<String> {
    let mut cursor: Option<&Path> = Some(cwd);
    let mut depth = 0_usize;
    while let Some(p) = cursor {
        depth += 1;
        if depth > 16 {
            return None;
        }
        let cfg = p.join(".git").join("config");
        if cfg.is_file() {
            let raw = std::fs::read_to_string(cfg).ok()?;
            let mut in_remote = false;
            for line in raw.lines() {
                let line = line.trim();
                if line.starts_with("[remote ") {
                    in_remote = line.contains("\"origin\"");
                    continue;
                }
                if line.starts_with('[') {
                    in_remote = false;
                    continue;
                }
                if in_remote {
                    if let Some(rest) = line.strip_prefix("url") {
                        let v = rest.trim_start_matches('=').trim();
                        return Some(v.to_string());
                    }
                }
            }
            return None;
        }
        cursor = p.parent();
    }
    None
}

fn hash16(s: &str) -> String {
    use std::fmt::Write as _;
    let mut hasher = Sha256::new();
    hasher.update(s.as_bytes());
    let digest = hasher.finalize();
    let mut out = String::with_capacity(16);
    for byte in digest.iter().take(8) {
        let _ = write!(out, "{byte:02x}");
    }
    out
}

/// Per plan/40 §9. Reads `<project>/.stratum/config.toml` and the
/// `STRATUM_AUTO_MEMORY` env var. Returns `true` when auto-memory is
/// enabled; falls back to true (default-on) when neither is set.
#[must_use]
pub fn auto_memory_enabled(project_root: Option<&Path>) -> bool {
    if std::env::var("STRATUM_AUTO_MEMORY").as_deref() == Ok("0") {
        return false;
    }
    let Some(root) = project_root else {
        return true;
    };
    let cfg_path = root.join(".stratum").join("config.toml");
    let Ok(raw) = std::fs::read_to_string(cfg_path) else {
        return true;
    };
    // Minimal TOML parsing — look for [memory] section + auto = false.
    let mut in_memory = false;
    for line in raw.lines() {
        let line = line.trim();
        if line == "[memory]" {
            in_memory = true;
            continue;
        }
        if line.starts_with('[') {
            in_memory = false;
            continue;
        }
        if in_memory {
            let normalized = line.replace(' ', "");
            if normalized.starts_with("auto=false") {
                return false;
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn open_store() -> (TempDir, AutoMemoryStore) {
        let tmp = TempDir::new().unwrap();
        let store = AutoMemoryStore::open(tmp.path(), "test-repo").unwrap();
        (tmp, store)
    }

    #[test]
    fn empty_index_returns_empty_vec() {
        let (_tmp, s) = open_store();
        assert!(s.list().is_empty());
    }

    #[test]
    fn save_then_load_round_trips() {
        let (_tmp, s) = open_store();
        let m = Memory {
            frontmatter: MemoryFrontmatter {
                name: "user_role".to_string(),
                description: "Krishnendu — Rust + observability".to_string(),
                kind: MemoryType::User,
            },
            body: "user identifies as a senior dev focused on observability".to_string(),
        };
        s.save(&m).unwrap();
        let back = s.load("user_role").unwrap();
        assert_eq!(back.frontmatter.name, "user_role");
        assert_eq!(back.frontmatter.kind, MemoryType::User);
        assert!(back.body.contains("senior dev"));
    }

    #[test]
    fn save_updates_index() {
        let (_tmp, s) = open_store();
        let m = Memory {
            frontmatter: MemoryFrontmatter {
                name: "feedback_no_emoji".to_string(),
                description: "never add emoji to commits".to_string(),
                kind: MemoryType::Feedback,
            },
            body: "user explicitly asked: no emoji in commits".to_string(),
        };
        s.save(&m).unwrap();
        let idx = s.list();
        assert_eq!(idx.len(), 1);
        assert_eq!(idx[0].name, "feedback_no_emoji");
        assert!(idx[0].description.contains("never add"));
    }

    #[test]
    fn forget_removes_body_and_index_entry() {
        let (_tmp, s) = open_store();
        let m = Memory {
            frontmatter: MemoryFrontmatter {
                name: "drop_me".to_string(),
                description: "x".to_string(),
                kind: MemoryType::Project,
            },
            body: "x".to_string(),
        };
        s.save(&m).unwrap();
        s.forget("drop_me").unwrap();
        assert!(s.load("drop_me").is_none());
        assert!(s.list().is_empty());
    }

    #[test]
    fn forget_missing_is_not_error() {
        let (_tmp, s) = open_store();
        assert!(s.forget("never-existed").is_ok());
    }

    #[test]
    fn updating_an_existing_memory_keeps_one_index_row() {
        let (_tmp, s) = open_store();
        let mut m = Memory {
            frontmatter: MemoryFrontmatter {
                name: "user_role".to_string(),
                description: "old description".to_string(),
                kind: MemoryType::User,
            },
            body: "first body".to_string(),
        };
        s.save(&m).unwrap();
        m.frontmatter.description = "new description".to_string();
        m.body = "second body".to_string();
        s.save(&m).unwrap();
        let idx = s.list();
        assert_eq!(idx.len(), 1);
        assert!(idx[0].description.contains("new description"));
    }

    #[test]
    fn index_entry_parse_and_render_round_trip() {
        let line = "- [User Role](user_role.md) — Krishnendu / Rust / observability";
        let e = IndexEntry::parse(line).unwrap();
        assert_eq!(e.name, "user_role");
        assert_eq!(e.description, "Krishnendu / Rust / observability");
        let rendered = e.render();
        assert_eq!(rendered, line);
    }

    #[test]
    fn index_entry_parse_rejects_garbage() {
        assert!(IndexEntry::parse("just some prose").is_none());
        assert!(IndexEntry::parse("- broken").is_none());
        assert!(IndexEntry::parse("- [missing close").is_none());
    }

    #[test]
    fn repo_id_falls_back_to_cwd_hash_when_no_git() {
        let tmp = TempDir::new().unwrap();
        let a = repo_id_for(tmp.path()).unwrap();
        let b = repo_id_for(tmp.path()).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.len(), 16);
    }

    /// Process-wide guard for the three tests that read or mutate
    /// `STRATUM_AUTO_MEMORY`. `cargo test` runs in parallel, and the
    /// env-mutating test would race with the two that only read the
    /// var unless they all serialize through this mutex.
    fn env_test_guard() -> &'static std::sync::Mutex<()> {
        static GUARD: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
        GUARD.get_or_init(|| std::sync::Mutex::new(()))
    }

    #[test]
    fn auto_memory_enabled_default_is_true() {
        let _g = env_test_guard().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let prior = std::env::var("STRATUM_AUTO_MEMORY").ok();
        std::env::remove_var("STRATUM_AUTO_MEMORY");
        let tmp = TempDir::new().unwrap();
        let result = auto_memory_enabled(Some(tmp.path()));
        if let Some(v) = prior {
            std::env::set_var("STRATUM_AUTO_MEMORY", v);
        }
        assert!(result);
    }

    #[test]
    fn auto_memory_enabled_respects_config() {
        let _g = env_test_guard().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let prior = std::env::var("STRATUM_AUTO_MEMORY").ok();
        std::env::remove_var("STRATUM_AUTO_MEMORY");
        let tmp = TempDir::new().unwrap();
        std::fs::create_dir_all(tmp.path().join(".stratum")).unwrap();
        std::fs::write(
            tmp.path().join(".stratum").join("config.toml"),
            "[memory]\nauto = false\n",
        )
        .unwrap();
        let result = auto_memory_enabled(Some(tmp.path()));
        if let Some(v) = prior {
            std::env::set_var("STRATUM_AUTO_MEMORY", v);
        }
        assert!(!result);
    }

    #[test]
    fn auto_memory_disabled_via_env() {
        let _g = env_test_guard().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        // Save / restore so test order doesn't matter.
        let prior = std::env::var("STRATUM_AUTO_MEMORY").ok();
        std::env::set_var("STRATUM_AUTO_MEMORY", "0");
        let result = auto_memory_enabled(None);
        match prior {
            Some(v) => std::env::set_var("STRATUM_AUTO_MEMORY", v),
            None => std::env::remove_var("STRATUM_AUTO_MEMORY"),
        }
        assert!(!result);
    }

    #[test]
    fn read_git_origin_from_inline_config() {
        let tmp = TempDir::new().unwrap();
        let git_dir = tmp.path().join(".git");
        std::fs::create_dir(&git_dir).unwrap();
        std::fs::write(
            git_dir.join("config"),
            "[remote \"origin\"]\n\turl = git@example.com:foo/bar.git\n",
        )
        .unwrap();
        let url = read_git_origin(tmp.path()).unwrap();
        assert!(url.contains("foo/bar.git"));
    }

    #[test]
    fn clear_removes_root_dir() {
        let tmp = TempDir::new().unwrap();
        let s = AutoMemoryStore::open(tmp.path(), "to-clear").unwrap();
        let root = s.root.clone();
        assert!(root.exists());
        s.clear().unwrap();
        assert!(!root.exists());
    }

    #[test]
    fn clear_on_already_missing_dir_is_ok() {
        let tmp = TempDir::new().unwrap();
        let s = AutoMemoryStore::open(tmp.path(), "ghost").unwrap();
        std::fs::remove_dir_all(&s.root).unwrap();
        assert!(s.clear().is_ok());
    }

    #[test]
    fn list_sorts_by_name() {
        let (_tmp, s) = open_store();
        for name in ["zulu", "alpha", "mike"] {
            let m = Memory {
                frontmatter: MemoryFrontmatter {
                    name: name.to_string(),
                    description: "d".to_string(),
                    kind: MemoryType::Project,
                },
                body: "b".to_string(),
            };
            s.save(&m).unwrap();
        }
        let names: Vec<_> = s.list().into_iter().map(|e| e.name).collect();
        assert_eq!(names, vec!["alpha", "mike", "zulu"]);
    }

    #[test]
    fn save_renders_all_memory_kinds() {
        // Exercise the Reference / Feedback / Project arms of render_memory.
        for kind in [
            MemoryType::User,
            MemoryType::Feedback,
            MemoryType::Project,
            MemoryType::Reference,
        ] {
            let (_tmp, s) = open_store();
            let m = Memory {
                frontmatter: MemoryFrontmatter {
                    name: format!("{kind:?}").to_ascii_lowercase(),
                    description: "d".to_string(),
                    kind,
                },
                body: "b".to_string(),
            };
            s.save(&m).unwrap();
            let back = s.load(&format!("{kind:?}").to_ascii_lowercase()).unwrap();
            assert_eq!(back.frontmatter.kind, kind);
        }
    }

    #[test]
    fn parse_memory_rejects_missing_frontmatter() {
        assert!(parse_memory("just body, no frontmatter").is_none());
    }

    #[test]
    fn parse_memory_rejects_unclosed_frontmatter() {
        assert!(parse_memory("---\nname: x\nno closer here").is_none());
    }

    #[test]
    fn parse_memory_rejects_unknown_kind() {
        let raw = "---\nname: x\ndescription: d\ntype: bogus\n---\nbody\n";
        assert!(parse_memory(raw).is_none());
    }

    #[test]
    fn parse_memory_rejects_missing_required_field() {
        let raw = "---\nname: x\ndescription: d\n---\nbody\n";
        assert!(parse_memory(raw).is_none());
    }

    #[test]
    fn parse_memory_accepts_all_kinds() {
        for kind in ["user", "feedback", "project", "reference"] {
            let raw = format!("---\nname: x\ndescription: d\ntype: {kind}\n---\nbody\n");
            assert!(parse_memory(&raw).is_some(), "kind {kind} should parse");
        }
    }

    #[test]
    fn index_entry_parse_rejects_missing_md_suffix() {
        // Path lacks `.md` → strip_suffix returns None.
        assert!(IndexEntry::parse("- [Title](slug) — desc").is_none());
        // No `)` at all → close_paren find returns None.
        assert!(IndexEntry::parse("- [Title](slug.md").is_none());
    }

    #[test]
    fn read_git_origin_returns_some_when_remote_present() {
        // Direct coverage of `repo_id_for`'s git-remote branch.
        let tmp = TempDir::new().unwrap();
        let git = tmp.path().join(".git");
        std::fs::create_dir(&git).unwrap();
        std::fs::write(
            git.join("config"),
            "[remote \"origin\"]\nurl = https://example.com/x.git\n",
        )
        .unwrap();
        let id = repo_id_for(tmp.path()).unwrap();
        assert_eq!(id.len(), 16);
    }

    #[test]
    fn read_git_origin_skips_non_origin_remote_and_other_sections() {
        let tmp = TempDir::new().unwrap();
        let git = tmp.path().join(".git");
        std::fs::create_dir(&git).unwrap();
        // `upstream` remote (not origin), then a `[core]` section that flips
        // in_remote off, then no origin URL anywhere → None.
        std::fs::write(
            git.join("config"),
            "[remote \"upstream\"]\nurl = wrong\n[core]\nbare = false\n",
        )
        .unwrap();
        assert!(read_git_origin(tmp.path()).is_none());
    }

    #[test]
    fn read_git_origin_returns_none_when_no_git_anywhere() {
        let tmp = TempDir::new().unwrap();
        assert!(read_git_origin(tmp.path()).is_none());
    }

    #[test]
    fn auto_memory_enabled_ignores_unrelated_config_sections() {
        let tmp = TempDir::new().unwrap();
        std::fs::create_dir_all(tmp.path().join(".stratum")).unwrap();
        // `[other]` toggles in_memory off; the `auto = false` underneath is
        // outside [memory] and should be ignored.
        std::fs::write(
            tmp.path().join(".stratum").join("config.toml"),
            "[other]\nauto = false\n",
        )
        .unwrap();
        assert!(auto_memory_enabled(Some(tmp.path())));
    }
}
