//! Permission DSL — `fs.write(*.rs)`, `shell.exec(npm *)`, `Bash(*)`.
//!
//! Parses rule strings into a [`Rule`] matcher and evaluates them
//! against `(tool_id, args_map)` pairs. Implements `plan/30 §10.1`.
//!
//! ## Grammar
//!
//! ```text
//! rule         := tool_id            // any args
//!               | tool_id "(" args ")"
//! tool_id      := alphanumeric, dots, underscores, dashes
//! args         := arg ("," arg)*
//! arg          := glob string
//! ```
//!
//! `Bash` is an alias for `shell.exec` (Claude Code parity).
//!
//! ## Tiers
//!
//! Three lists with explicit precedence: **deny wins**, then **allow**,
//! then **ask** (UI tier). [`RuleSet::evaluate`] returns a
//! [`Decision`] enum.

use serde::{Deserialize, Serialize};

/// Decision returned by the rule evaluator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Decision {
    /// Tool is explicitly denied — caller must NOT dispatch.
    Deny,
    /// Tool is explicitly allowed — caller MAY dispatch without prompting.
    Allow,
    /// Tool requires the user to confirm via the permission modal.
    Ask,
    /// No matching rule — caller's default policy applies (typically
    /// "ask for write-class tools, allow for read-class").
    Unspecified,
}

/// One compiled rule.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rule {
    /// Canonical tool id (post `Bash` → `shell.exec` aliasing).
    pub tool: String,
    /// Per-arg glob matchers in declaration order. `None` = no
    /// constraint on args.
    pub arg_globs: Option<Vec<String>>,
}

impl Rule {
    /// Parse a rule string. Returns `None` for syntax errors so a
    /// malformed rule in a user-written settings.json doesn't crash
    /// the runtime — the loader logs + drops it.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        let s = s.trim();
        if s.is_empty() {
            return None;
        }
        let (tool, args_part) = match s.find('(') {
            Some(open) => {
                let tool_part = s[..open].trim();
                let close = s.rfind(')')?;
                if close <= open {
                    return None;
                }
                let inner = s[open + 1..close].trim();
                let args: Vec<String> = inner
                    .split(',')
                    .map(|a| a.trim().to_string())
                    .filter(|a| !a.is_empty())
                    .collect();
                (tool_part, Some(args))
            }
            None => (s, None),
        };
        // Bash → shell.exec alias for Claude-Code parity.
        let tool = if tool.eq_ignore_ascii_case("Bash") {
            "shell.exec".to_string()
        } else {
            tool.to_string()
        };
        if tool.is_empty() {
            return None;
        }
        Some(Self {
            tool,
            arg_globs: args_part,
        })
    }

    /// Test whether this rule matches the given `(tool_id, args)`
    /// invocation. Args lookup is positional via convention: rule
    /// `fs.read(path)` checks the first string-valued arg in the
    /// JSON object (path / pattern / command / url / arg1).
    #[must_use]
    pub fn matches(&self, tool: &str, args: &[&str]) -> bool {
        if self.tool != tool {
            return false;
        }
        let Some(globs) = self.arg_globs.as_ref() else {
            return true;
        };
        if globs.is_empty() {
            return args.is_empty();
        }
        for (i, glob) in globs.iter().enumerate() {
            let Some(arg) = args.get(i) else {
                return false;
            };
            if !glob_match(glob, arg) {
                return false;
            }
        }
        true
    }
}

/// A set of rules grouped by tier.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RuleSet {
    /// Allow tier — first match short-circuits with `Allow` UNLESS a
    /// deny rule also matches (deny wins).
    pub allow: Vec<Rule>,
    /// Deny tier — any match short-circuits with `Deny`.
    pub deny: Vec<Rule>,
    /// Ask tier — match triggers the permission modal.
    pub ask: Vec<Rule>,
}

impl RuleSet {
    /// Build a rule set from three lists of rule strings (one per
    /// tier). Strings that fail to parse are silently dropped.
    #[must_use]
    pub fn from_strings(allow: &[String], deny: &[String], ask: &[String]) -> Self {
        Self {
            allow: allow.iter().filter_map(|s| Rule::parse(s)).collect(),
            deny: deny.iter().filter_map(|s| Rule::parse(s)).collect(),
            ask: ask.iter().filter_map(|s| Rule::parse(s)).collect(),
        }
    }

    /// Evaluate against an invocation. Precedence:
    /// 1. Any deny → Deny
    /// 2. Any allow → Allow
    /// 3. Any ask → Ask
    /// 4. otherwise → Unspecified
    #[must_use]
    pub fn evaluate(&self, tool: &str, args: &[&str]) -> Decision {
        for r in &self.deny {
            if r.matches(tool, args) {
                return Decision::Deny;
            }
        }
        for r in &self.allow {
            if r.matches(tool, args) {
                return Decision::Allow;
            }
        }
        for r in &self.ask {
            if r.matches(tool, args) {
                return Decision::Ask;
            }
        }
        Decision::Unspecified
    }
}

/// Glob matcher — `*` matches any run of characters (including
/// dots), `?` matches one. Simple implementation; no character
/// classes. Sufficient for the path/command patterns the DSL needs.
fn glob_match(pattern: &str, s: &str) -> bool {
    let p: Vec<char> = pattern.chars().collect();
    let t: Vec<char> = s.chars().collect();
    glob_inner(&p, 0, &t, 0)
}

fn glob_inner(p: &[char], pi: usize, t: &[char], ti: usize) -> bool {
    if pi == p.len() {
        return ti == t.len();
    }
    match p[pi] {
        '*' => {
            // Star with no remaining pattern → match anything left.
            for k in ti..=t.len() {
                if glob_inner(p, pi + 1, t, k) {
                    return true;
                }
            }
            false
        }
        '?' => {
            if ti < t.len() {
                glob_inner(p, pi + 1, t, ti + 1)
            } else {
                false
            }
        }
        c => {
            if ti < t.len() && t[ti] == c {
                glob_inner(p, pi + 1, t, ti + 1)
            } else {
                false
            }
        }
    }
}

/// Extract the canonical "args" sequence from a tool-call JSON args object.
///
/// Order: `path`, `command`, `pattern`, `query`, `url`, then remaining
/// string values alphabetically. Used by callers to build the `&[&str]`
/// slice passed to [`Rule::matches`] without every caller re-inventing
/// the convention.
#[must_use]
pub fn args_from_json(args_json: &str) -> Vec<String> {
    let Ok(v) = serde_json::from_str::<serde_json::Value>(args_json) else {
        return Vec::new();
    };
    let Some(obj) = v.as_object() else {
        return Vec::new();
    };
    let priority = ["path", "command", "pattern", "query", "url", "cmd"];
    let mut out: Vec<String> = Vec::new();
    let mut consumed = std::collections::BTreeSet::new();
    for key in &priority {
        if let Some(s) = obj.get(*key).and_then(|x| x.as_str()) {
            out.push(s.to_string());
            consumed.insert(*key);
        }
    }
    let mut remaining_keys: Vec<&str> = obj
        .keys()
        .map(String::as_str)
        .filter(|k| !consumed.contains(k))
        .collect();
    remaining_keys.sort_unstable();
    for k in remaining_keys {
        if let Some(s) = obj.get(k).and_then(|x| x.as_str()) {
            out.push(s.to_string());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_bare_tool_id() {
        let r = Rule::parse("fs.read").unwrap();
        assert_eq!(r.tool, "fs.read");
        assert!(r.arg_globs.is_none());
    }

    #[test]
    fn parse_with_args() {
        let r = Rule::parse("fs.write(*.rs)").unwrap();
        assert_eq!(r.tool, "fs.write");
        assert_eq!(r.arg_globs, Some(vec!["*.rs".to_string()]));
    }

    #[test]
    fn parse_bash_aliases_to_shell_exec() {
        let r = Rule::parse("Bash(npm *)").unwrap();
        assert_eq!(r.tool, "shell.exec");
        assert_eq!(r.arg_globs.as_deref().unwrap()[0], "npm *");
    }

    #[test]
    fn parse_multi_arg() {
        let r = Rule::parse("fs.edit(*.rs, *)").unwrap();
        assert_eq!(r.arg_globs.as_deref().unwrap().len(), 2);
    }

    #[test]
    fn parse_rejects_garbage() {
        assert!(Rule::parse("").is_none());
        assert!(Rule::parse("(").is_none());
        assert!(Rule::parse("foo(").is_none());
    }

    #[test]
    fn match_bare_rule_ignores_args() {
        let r = Rule::parse("fs.read").unwrap();
        assert!(r.matches("fs.read", &["any", "thing"]));
        assert!(r.matches("fs.read", &[]));
    }

    #[test]
    fn match_glob_args() {
        let r = Rule::parse("fs.write(*.rs)").unwrap();
        assert!(r.matches("fs.write", &["src/main.rs"]));
        assert!(!r.matches("fs.write", &["src/main.toml"]));
    }

    #[test]
    fn match_question_mark() {
        let r = Rule::parse("fs.read(a?c.txt)").unwrap();
        assert!(r.matches("fs.read", &["abc.txt"]));
        assert!(!r.matches("fs.read", &["ac.txt"]));
    }

    #[test]
    fn deny_wins_over_allow() {
        let rs = RuleSet::from_strings(
            &["fs.write(*)".to_string()],
            &["fs.write(.env*)".to_string()],
            &[],
        );
        assert_eq!(
            rs.evaluate("fs.write", &[".env.production"]),
            Decision::Deny
        );
        assert_eq!(rs.evaluate("fs.write", &["src/main.rs"]), Decision::Allow);
    }

    #[test]
    fn ask_only_fires_when_no_allow_or_deny() {
        let rs = RuleSet::from_strings(&[], &[], &["fs.edit(*)".to_string()]);
        assert_eq!(rs.evaluate("fs.edit", &["x.rs"]), Decision::Ask);
        assert_eq!(rs.evaluate("fs.read", &["x.rs"]), Decision::Unspecified);
    }

    #[test]
    fn shell_exec_first_word_glob() {
        let rs = RuleSet::from_strings(
            &["shell.exec(npm *)".to_string()],
            &["shell.exec(rm *)".to_string()],
            &[],
        );
        assert_eq!(rs.evaluate("shell.exec", &["npm install"]), Decision::Allow);
        assert_eq!(rs.evaluate("shell.exec", &["rm -rf /"]), Decision::Deny);
        assert_eq!(
            rs.evaluate("shell.exec", &["echo hi"]),
            Decision::Unspecified
        );
    }

    #[test]
    fn args_from_json_priority_order() {
        let args = args_from_json(r#"{"path":"a.rs","encoding":"utf-8"}"#);
        assert_eq!(args, vec!["a.rs".to_string(), "utf-8".to_string()]);
    }

    #[test]
    fn args_from_json_returns_empty_on_malformed() {
        assert!(args_from_json("not json").is_empty());
        assert!(args_from_json(r#"["array","not","object"]"#).is_empty());
    }

    #[test]
    fn deny_overrides_even_when_allow_is_more_specific() {
        // Allow rule is highly specific; deny is broad — deny still wins.
        let rs = RuleSet::from_strings(
            &["fs.write(src/x.rs)".to_string()],
            &["fs.write(*)".to_string()],
            &[],
        );
        assert_eq!(rs.evaluate("fs.write", &["src/x.rs"]), Decision::Deny);
    }

    #[test]
    fn unspecified_when_no_rules_match() {
        let rs = RuleSet::from_strings(
            &["fs.read(*)".to_string()],
            &["fs.write(*)".to_string()],
            &[],
        );
        assert_eq!(rs.evaluate("grep", &["TODO"]), Decision::Unspecified);
    }
}
