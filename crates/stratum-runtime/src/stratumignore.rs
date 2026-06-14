//! `.stratumignore` matcher — gitignore-syntax denylist applied
//! to `fs.read`, `fs.write`, `fs.edit`, `glob`, and `grep` so the
//! agent can't exfiltrate `.env`, secrets, build artifacts, etc.
//!
//! Per `plan/30 §3.1` and `plan/31 §5`. Pattern syntax is the
//! gitignore subset documented in `git help ignore`:
//!
//! - `*`  matches any run of non-`/` characters
//! - `**` matches any run including `/`
//! - `?`  matches one non-`/` character
//! - `!pattern` negates a prior match (un-ignore)
//! - `/pattern` is rooted at the file's directory
//! - `pattern/` matches only directories
//! - Blank lines + lines starting with `#` are skipped
//!
//! Implementation is intentionally minimal — we don't pull in a full
//! gitignore crate because most fs-tool denylists need only the basic
//! cases. Pathological globs degrade to "doesn't match" rather than
//! panic.

use std::path::Path;

/// One parsed `.stratumignore` rule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IgnoreRule {
    /// True if this rule starts with `!` (negation / un-ignore).
    pub negated: bool,
    /// True if the rule is anchored to the file's root (`/pattern`).
    pub anchored: bool,
    /// True if the rule only matches directories (trailing `/`).
    pub dir_only: bool,
    /// The compiled glob (pre-normalized).
    pub pattern: String,
}

impl IgnoreRule {
    /// Parse one line. Returns `None` for blank lines and comments.
    #[must_use]
    pub fn parse(line: &str) -> Option<Self> {
        let raw = line.trim();
        if raw.is_empty() || raw.starts_with('#') {
            return None;
        }
        let (negated, rest) = match raw.strip_prefix('!') {
            Some(r) => (true, r),
            None => (false, raw),
        };
        let (anchored, rest) = match rest.strip_prefix('/') {
            Some(r) => (true, r),
            None => (false, rest),
        };
        let (dir_only, rest) = match rest.strip_suffix('/') {
            Some(r) => (true, r),
            None => (false, rest),
        };
        if rest.is_empty() {
            return None;
        }
        Some(Self {
            negated,
            anchored,
            dir_only,
            pattern: rest.to_string(),
        })
    }
}

/// Compiled `.stratumignore` document. Match by walking rules in
/// source order — later matches override earlier ones, so `!secret/keep`
/// can carve out a single file from `secret/**`.
#[derive(Debug, Clone, Default)]
pub struct Ignore {
    rules: Vec<IgnoreRule>,
}

impl Ignore {
    /// Parse the entire file body. Bad lines are silently dropped.
    #[must_use]
    pub fn parse(body: &str) -> Self {
        let rules = body.lines().filter_map(IgnoreRule::parse).collect();
        Self { rules }
    }

    /// Load `.stratumignore` from the workspace root (the directory
    /// passed in). Returns an empty matcher if the file is absent.
    /// Per plan/30 §3.1 we ALSO honor `.gitignore` as a fallback
    /// when `.stratumignore` is missing — that matches user
    /// expectation for almost every Rust/Node/Python repo.
    #[must_use]
    pub fn from_workspace(root: &Path) -> Self {
        let primary = root.join(".stratumignore");
        if let Ok(raw) = std::fs::read_to_string(&primary) {
            return Self::parse(&raw);
        }
        let fallback = root.join(".gitignore");
        if let Ok(raw) = std::fs::read_to_string(&fallback) {
            return Self::parse(&raw);
        }
        Self::default()
    }

    /// Does this matcher have any rules? Used by callers to skip
    /// allocation on the hot path when there's no file at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }

    /// True when `path` (workspace-relative, forward-slash form) is
    /// ignored by the rule set. Walking rules in order; later
    /// negation matches can un-ignore.
    #[must_use]
    pub fn is_ignored(&self, path: &str, is_dir: bool) -> bool {
        let path = path.trim_start_matches('/');
        let mut ignored = false;
        for rule in &self.rules {
            if rule.dir_only && !is_dir {
                continue;
            }
            if rule_matches(rule, path) {
                ignored = !rule.negated;
            }
        }
        ignored
    }
}

fn rule_matches(rule: &IgnoreRule, path: &str) -> bool {
    if rule.anchored {
        glob_match(&rule.pattern, path)
    } else {
        // Unanchored: match the pattern against any suffix segment
        // boundary (any path component) — same as gitignore.
        if glob_match(&rule.pattern, path) {
            return true;
        }
        let mut cursor = 0;
        for (i, c) in path.char_indices() {
            if c == '/' && i > cursor {
                let segment = &path[i + 1..];
                if glob_match(&rule.pattern, segment) {
                    return true;
                }
                cursor = i + 1;
            }
        }
        false
    }
}

/// Glob matcher with `*` (non-slash), `**` (any), `?` (single non-slash).
fn glob_match(pattern: &str, s: &str) -> bool {
    let p: Vec<char> = pattern.chars().collect();
    let t: Vec<char> = s.chars().collect();
    glob_inner(&p, 0, &t, 0)
}

fn glob_inner(p: &[char], pi: usize, t: &[char], ti: usize) -> bool {
    if pi == p.len() {
        return ti == t.len();
    }
    // `**` — match any sequence including `/`.
    if pi + 1 < p.len() && p[pi] == '*' && p[pi + 1] == '*' {
        // Skip optional `/` after the `**`.
        let after = if pi + 2 < p.len() && p[pi + 2] == '/' {
            pi + 3
        } else {
            pi + 2
        };
        for k in ti..=t.len() {
            if glob_inner(p, after, t, k) {
                return true;
            }
        }
        return false;
    }
    if p[pi] == '*' {
        // Single-segment star.
        for k in ti..=t.len() {
            // `*` does NOT consume `/`.
            if t[ti..k].iter().any(|c| *c == '/') {
                break;
            }
            if glob_inner(p, pi + 1, t, k) {
                return true;
            }
        }
        return false;
    }
    if p[pi] == '?' {
        if ti < t.len() && t[ti] != '/' {
            return glob_inner(p, pi + 1, t, ti + 1);
        }
        return false;
    }
    if ti < t.len() && t[ti] == p[pi] {
        return glob_inner(p, pi + 1, t, ti + 1);
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_skips_blank_and_comment_lines() {
        let i = Ignore::parse("# header\n\n*.log\n# inline\n");
        assert_eq!(i.rules.len(), 1);
        assert_eq!(i.rules[0].pattern, "*.log");
    }

    #[test]
    fn parse_detects_negation_anchor_dir_only() {
        let i = Ignore::parse("!important.log\n/build\nnode_modules/\n");
        assert!(i.rules[0].negated);
        assert!(i.rules[1].anchored);
        assert!(i.rules[2].dir_only);
    }

    #[test]
    fn glob_star_matches_within_segment() {
        assert!(glob_match("*.log", "app.log"));
        assert!(!glob_match("*.log", "logs/app.log"));
        assert!(glob_match("logs/*.log", "logs/app.log"));
    }

    #[test]
    fn glob_double_star_crosses_segments() {
        assert!(glob_match("**/secrets/*.key", "deep/secrets/x.key"));
        assert!(glob_match("**/secrets/*.key", "secrets/x.key"));
    }

    #[test]
    fn unanchored_pattern_matches_any_segment() {
        let i = Ignore::parse(".env\n");
        assert!(i.is_ignored(".env", false));
        assert!(i.is_ignored("config/.env", false));
        assert!(i.is_ignored("deep/down/.env", false));
        assert!(!i.is_ignored(".envrc", false));
    }

    #[test]
    fn anchored_pattern_only_root() {
        let i = Ignore::parse("/build\n");
        assert!(i.is_ignored("build", false));
        assert!(!i.is_ignored("crates/foo/build", false));
    }

    #[test]
    fn dir_only_rule_skips_files() {
        let i = Ignore::parse("node_modules/\n");
        assert!(i.is_ignored("node_modules", true));
        assert!(!i.is_ignored("node_modules", false));
    }

    #[test]
    fn negation_un_ignores_a_specific_path() {
        let i = Ignore::parse("secrets/*\n!secrets/public.key\n");
        assert!(i.is_ignored("secrets/private.key", false));
        assert!(!i.is_ignored("secrets/public.key", false));
    }

    #[test]
    fn empty_matcher_ignores_nothing() {
        let i = Ignore::default();
        assert!(!i.is_ignored(".env", false));
        assert!(!i.is_ignored("anything", false));
    }

    #[test]
    fn typical_env_secrets_rule() {
        let i = Ignore::parse(".env\n.env.*\n*.key\n*.pem\nsecrets/\n");
        assert!(i.is_ignored(".env", false));
        assert!(i.is_ignored(".env.production", false));
        assert!(i.is_ignored("server.key", false));
        assert!(i.is_ignored("ca.pem", false));
        assert!(i.is_ignored("secrets", true));
        assert!(!i.is_ignored("src/main.rs", false));
    }

    #[test]
    fn from_workspace_falls_back_to_gitignore() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::write(tmp.path().join(".gitignore"), "target\n*.log\n").unwrap();
        let i = Ignore::from_workspace(tmp.path());
        assert!(i.is_ignored("target", false));
        assert!(i.is_ignored("debug.log", false));
    }

    #[test]
    fn from_workspace_prefers_stratumignore_over_gitignore() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::write(tmp.path().join(".gitignore"), "ignored_by_git\n").unwrap();
        std::fs::write(
            tmp.path().join(".stratumignore"),
            "ignored_by_stratum\n",
        )
        .unwrap();
        let i = Ignore::from_workspace(tmp.path());
        assert!(i.is_ignored("ignored_by_stratum", false));
        assert!(!i.is_ignored("ignored_by_git", false));
    }
}
