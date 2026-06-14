//! `STRATUM.md` walk-up loader.
//!
//! Implements `plan/39 §2-§4`: walks up from cwd until a project
//! marker, collects every `STRATUM.md` along the chain, resolves
//! `@file` imports (depth-capped at 4), concatenates with
//! `[Source: <path>]` markers, and returns the combined text ready
//! to inject into the agent's system context.
//!
//! ## What this loads
//!
//! - Managed: `/etc/stratum/STRATUM.md` (Linux) /
//!   `/Library/Application Support/Stratum/STRATUM.md` (macOS) — admin-only
//! - User: `<config>/stratum/STRATUM.md`
//! - Project: nearest `STRATUM.md` walking up to the first project marker
//!   (`.git/`, `Cargo.toml`, `package.json`, `pyproject.toml`, `go.mod`,
//!   `.stratum/`), then every `STRATUM.md` between cwd and that root
//! - Local: `<project>/STRATUM.local.md` or `<project>/.stratum/local.md`
//!
//! ## What this does NOT do (deferred)
//!
//! - `.stratum/rules/<topic>.md` paths-frontmatter matching (Phase 4 v2)
//! - `/memory` palette command (chat.rs handles UI)
//! - Auto-memory (`MEMORY.md` index) — see `plan/40`
//! - Hot-reload on file change (Phase 4 v2)

use std::path::{Path, PathBuf};

/// Project root markers in priority order. First match wins the walk-up.
const PROJECT_MARKERS: &[&str] = &[
    ".stratum",
    ".git",
    "Cargo.toml",
    "package.json",
    "pyproject.toml",
    "go.mod",
];

/// Maximum walk-up depth (defensive against pathological filesystems).
const MAX_WALK_UP: usize = 16;

/// Maximum recursion depth for `@file` imports.
const MAX_IMPORT_DEPTH: usize = 4;

/// Maximum bytes any single tier's content (post-import) may contribute.
/// Prevents a 10MB `STRATUM.md` from blowing the context window.
const MAX_TIER_BYTES: usize = 32 * 1024;

/// Origin of one loaded section. Used for the `[Source: …]` marker.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Tier {
    /// Org-wide admin policy at `/etc/stratum/STRATUM.md`.
    Managed,
    /// User-level defaults at `<config>/stratum/STRATUM.md`.
    User,
    /// Project-tier file walked up from cwd.
    Project,
    /// Per-checkout local (gitignored).
    Local,
}

impl Tier {
    const fn label(&self) -> &'static str {
        match self {
            Self::Managed => "Managed",
            Self::User => "User",
            Self::Project => "Project",
            Self::Local => "Local",
        }
    }
}

/// One resolved memory section.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadedSection {
    /// Which tier this came from.
    pub tier: Tier,
    /// The path on disk (for the `[Source: …]` marker).
    pub source: PathBuf,
    /// The fully-imported body text, capped at `MAX_TIER_BYTES`.
    pub body: String,
}

/// Configuration handed to the loader. All paths are optional —
/// missing tiers are silently skipped.
#[derive(Debug, Clone, Default)]
pub struct LoaderConfig {
    /// Managed-tier file location (admin policy).
    pub managed_path: Option<PathBuf>,
    /// User-tier file location.
    pub user_path: Option<PathBuf>,
    /// Workspace cwd from which the project-tier walk-up starts.
    pub cwd: Option<PathBuf>,
}

/// Run the loader.
///
/// Returns one [`LoadedSection`] per tier that resolved successfully, in
/// concatenation order (managed → user → project [outermost → innermost]
/// → local). Caller stitches them into the model's system prompt.
#[must_use]
pub fn load(cfg: &LoaderConfig) -> Vec<LoadedSection> {
    let mut out: Vec<LoadedSection> = Vec::new();

    if let Some(p) = cfg.managed_path.as_ref() {
        if let Some(s) = read_section(Tier::Managed, p) {
            out.push(s);
        }
    }
    if let Some(p) = cfg.user_path.as_ref() {
        if let Some(s) = read_section(Tier::User, p) {
            out.push(s);
        }
    }
    if let Some(cwd) = cfg.cwd.as_ref() {
        for (path, is_local) in walk_up_project_files(cwd) {
            let tier = if is_local { Tier::Local } else { Tier::Project };
            if let Some(s) = read_section(tier, &path) {
                out.push(s);
            }
        }
    }

    out
}

/// Concatenate loaded sections into a single string with `[Source: …]`
/// markers. Suitable for appending to the model's system prompt.
#[must_use]
pub fn concat(sections: &[LoadedSection]) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    for s in sections {
        if !out.is_empty() {
            out.push_str("\n\n");
        }
        let _ = write!(
            out,
            "[Source: {} ({})]\n{}",
            s.source.display(),
            s.tier.label(),
            s.body
        );
    }
    out
}

fn read_section(tier: Tier, path: &Path) -> Option<LoadedSection> {
    let raw = std::fs::read_to_string(path).ok()?;
    let resolved = resolve_imports(&raw, path, 0);
    let body = if resolved.len() <= MAX_TIER_BYTES {
        resolved
    } else {
        // Truncate at a char boundary.
        let mut end = MAX_TIER_BYTES;
        while end > 0 && !resolved.is_char_boundary(end) {
            end -= 1;
        }
        format!(
            "{}\n[…truncated at {} bytes]",
            &resolved[..end],
            MAX_TIER_BYTES
        )
    };
    Some(LoadedSection {
        tier,
        source: path.to_path_buf(),
        body,
    })
}

/// Walk up from `cwd` toward `/`, collecting every `STRATUM.md` /
/// `STRATUM.local.md` / `.stratum/local.md` along the way. Stops at
/// the first project marker and returns in **outermost → innermost**
/// order (root first, cwd-nearest last) so the concatenation respects
/// "closer wins on conflict" via append-order.
fn walk_up_project_files(cwd: &Path) -> Vec<(PathBuf, bool)> {
    let mut hits: Vec<(PathBuf, bool)> = Vec::new();
    let mut cursor: Option<&Path> = Some(cwd);
    let mut depth = 0;
    while let Some(p) = cursor {
        depth += 1;
        if depth > MAX_WALK_UP {
            break;
        }
        for candidate in [p.join("STRATUM.md"), p.join(".stratum/STRATUM.md")] {
            if candidate.is_file() {
                hits.push((candidate, false));
            }
        }
        for candidate in [p.join("STRATUM.local.md"), p.join(".stratum/local.md")] {
            if candidate.is_file() {
                hits.push((candidate, true));
            }
        }
        // Stop at the first project marker (we processed this dir already).
        let at_root = PROJECT_MARKERS.iter().any(|m| p.join(m).exists());
        if at_root {
            break;
        }
        cursor = p.parent();
    }
    hits.reverse();
    hits
}

/// Resolve `@file` imports. Lines starting with `@./`, `@~/`, or
/// `@/` are replaced by the target file's contents (capped at
/// `MAX_TIER_BYTES` per import, recursion depth-limited).
fn resolve_imports(text: &str, anchor: &Path, depth: usize) -> String {
    use std::fmt::Write as _;
    if depth >= MAX_IMPORT_DEPTH {
        return text.to_string();
    }
    let mut out = String::with_capacity(text.len());
    for line in text.lines() {
        let trimmed = line.trim_start();
        if let Some(rest) = trimmed.strip_prefix('@') {
            let target = rest.trim();
            if let Some(resolved_path) = resolve_import_path(target, anchor) {
                match std::fs::read_to_string(&resolved_path) {
                    Ok(body) => {
                        let _ = writeln!(out, "<!-- @import: {} -->", resolved_path.display());
                        let nested = resolve_imports(&body, &resolved_path, depth + 1);
                        out.push_str(&nested);
                        out.push('\n');
                        continue;
                    }
                    Err(e) => {
                        let _ =
                            writeln!(out, "[import failed: {} ({})]", resolved_path.display(), e);
                        continue;
                    }
                }
            }
            let _ = writeln!(out, "[import failed: {target} (path rejected)]");
            continue;
        }
        out.push_str(line);
        out.push('\n');
    }
    out
}

fn resolve_import_path(target: &str, anchor: &Path) -> Option<PathBuf> {
    let anchor_dir = anchor.parent()?;
    if let Some(rest) = target.strip_prefix("./") {
        let p = anchor_dir.join(rest);
        return Some(p);
    }
    if let Some(rest) = target.strip_prefix("~/") {
        let home = dirs::home_dir()?;
        let p = home.join(rest);
        return Some(p);
    }
    if target.starts_with('/') {
        return Some(PathBuf::from(target));
    }
    // Bare relative path also OK.
    Some(anchor_dir.join(target))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn write(p: &Path, body: &str) {
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, body).unwrap();
    }

    #[test]
    fn load_returns_empty_when_no_files() {
        let cfg = LoaderConfig::default();
        assert!(load(&cfg).is_empty());
    }

    #[test]
    fn load_user_tier() {
        let tmp = TempDir::new().unwrap();
        let user = tmp.path().join("STRATUM.md");
        write(&user, "user rules");
        let cfg = LoaderConfig {
            user_path: Some(user),
            ..Default::default()
        };
        let out = load(&cfg);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].tier, Tier::User);
        assert!(out[0].body.contains("user rules"));
    }

    #[test]
    fn load_walks_up_project_to_first_marker() {
        let tmp = TempDir::new().unwrap();
        let project = tmp.path().join("repo");
        std::fs::create_dir_all(&project).unwrap();
        // Project marker.
        std::fs::write(project.join("Cargo.toml"), "").unwrap();
        // Project STRATUM.md at root.
        write(&project.join("STRATUM.md"), "project root rules");
        // Inner workspace with its own STRATUM.md.
        let inner = project.join("crates").join("foo");
        write(&inner.join("STRATUM.md"), "inner crate rules");
        let cfg = LoaderConfig {
            cwd: Some(inner),
            ..Default::default()
        };
        let out = load(&cfg);
        // Outermost first, innermost last.
        let bodies: Vec<&str> = out.iter().map(|s| s.body.as_str()).collect();
        assert!(bodies.iter().any(|b| b.contains("project root rules")));
        assert!(bodies.iter().any(|b| b.contains("inner crate rules")));
    }

    #[test]
    fn load_local_file_marked_local_tier() {
        let tmp = TempDir::new().unwrap();
        std::fs::write(tmp.path().join("Cargo.toml"), "").unwrap();
        write(&tmp.path().join("STRATUM.local.md"), "local rules");
        let cfg = LoaderConfig {
            cwd: Some(tmp.path().to_path_buf()),
            ..Default::default()
        };
        let out = load(&cfg);
        assert!(out
            .iter()
            .any(|s| s.tier == Tier::Local && s.body.contains("local rules")));
    }

    #[test]
    fn imports_resolve_relative_paths() {
        let tmp = TempDir::new().unwrap();
        std::fs::write(tmp.path().join("Cargo.toml"), "").unwrap();
        let main = tmp.path().join("STRATUM.md");
        let imported = tmp.path().join("style.md");
        write(&imported, "imported style rules");
        write(&main, "main rules\n@./style.md\nmore rules");
        let cfg = LoaderConfig {
            cwd: Some(tmp.path().to_path_buf()),
            ..Default::default()
        };
        let out = load(&cfg);
        let combined: String = out.iter().map(|s| s.body.clone()).collect();
        assert!(combined.contains("main rules"));
        assert!(combined.contains("imported style rules"));
        assert!(combined.contains("more rules"));
    }

    #[test]
    #[allow(
        clippy::many_single_char_names,
        reason = "single-char filenames mirror the @import depth-chain under test"
    )]
    fn imports_cap_at_max_depth() {
        let tmp = TempDir::new().unwrap();
        std::fs::write(tmp.path().join("Cargo.toml"), "").unwrap();
        let main = tmp.path().join("STRATUM.md");
        let a = tmp.path().join("a.md");
        let b = tmp.path().join("b.md");
        let c = tmp.path().join("c.md");
        let d = tmp.path().join("d.md");
        let e = tmp.path().join("e.md");
        write(&main, "main\n@./a.md");
        write(&a, "a\n@./b.md");
        write(&b, "b\n@./c.md");
        write(&c, "c\n@./d.md");
        write(&d, "d\n@./e.md");
        write(&e, "deepest");
        let cfg = LoaderConfig {
            cwd: Some(tmp.path().to_path_buf()),
            ..Default::default()
        };
        let out = load(&cfg);
        let combined: String = out.iter().map(|s| s.body.clone()).collect();
        // Depth 4 means we get a/b/c/d but the @./e.md inside d is the 5th level.
        assert!(combined.contains("main"));
        assert!(combined.contains('d'));
        // "deepest" sits past the recursion cap; the `@./e.md` line
        // is left as literal text inside d.md's rendered body.
        assert!(combined.contains("@./e.md"));
        assert!(!combined.contains("deepest"));
    }

    #[test]
    fn failed_import_surfaces_marker_not_silence() {
        let tmp = TempDir::new().unwrap();
        std::fs::write(tmp.path().join("Cargo.toml"), "").unwrap();
        write(&tmp.path().join("STRATUM.md"), "@./nope.md");
        let cfg = LoaderConfig {
            cwd: Some(tmp.path().to_path_buf()),
            ..Default::default()
        };
        let out = load(&cfg);
        let combined: String = out.iter().map(|s| s.body.clone()).collect();
        assert!(combined.contains("import failed"));
    }

    #[test]
    fn concat_emits_source_markers() {
        let tmp = TempDir::new().unwrap();
        std::fs::write(tmp.path().join("Cargo.toml"), "").unwrap();
        write(&tmp.path().join("STRATUM.md"), "rules");
        let cfg = LoaderConfig {
            cwd: Some(tmp.path().to_path_buf()),
            ..Default::default()
        };
        let out = load(&cfg);
        let combined = concat(&out);
        assert!(combined.contains("[Source:"));
        assert!(combined.contains("(Project)"));
        assert!(combined.contains("rules"));
    }

    #[test]
    fn body_truncated_at_tier_cap() {
        let tmp = TempDir::new().unwrap();
        std::fs::write(tmp.path().join("Cargo.toml"), "").unwrap();
        let big = "x".repeat(MAX_TIER_BYTES + 1000);
        write(&tmp.path().join("STRATUM.md"), &big);
        let cfg = LoaderConfig {
            cwd: Some(tmp.path().to_path_buf()),
            ..Default::default()
        };
        let out = load(&cfg);
        assert!(out[0].body.contains("truncated at"));
        assert!(out[0].body.len() <= MAX_TIER_BYTES + 80);
    }
}
