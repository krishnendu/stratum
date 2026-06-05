//! Tool registry and capability matrix.
//!
//! Phase 3 v2 — pure data + matcher. The concrete tool implementations
//! (fs.read, fs.write, bash.run, git.diff, git.log, read_image) plug in
//! later; this module pins the surface so agent TOMLs and workspace
//! `[tools]` blocks can be parsed and intersected today.
//!
//! Per `plan/19-user-agents-and-plugins.md` §7 and
//! `plan/31-tool-sandbox-and-secrets.md` §7.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

/// Tool entry in the capability matrix. The string follows the
/// `<namespace>.<verb>[:<arg-glob>]` shape:
///
/// - `fs.read` — every read.
/// - `fs.write:src/**.py` — write only under `src/` and only `.py`.
/// - `bash.run:py*` — bash commands starting with `py`.
/// - `git.*` — every git verb.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CapabilityEntry(String);

impl CapabilityEntry {
    /// Build a capability entry from any string-like value.
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// Borrow the underlying pattern.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Split into `(verb, optional arg pattern)` halves.
    #[must_use]
    pub fn parts(&self) -> (&str, Option<&str>) {
        match self.0.split_once(':') {
            Some((verb, arg)) => (verb, Some(arg)),
            None => (self.0.as_str(), None),
        }
    }

    /// Does this entry's verb match `target`? Handles `git.*`-style wildcards.
    #[must_use]
    pub fn verb_matches(&self, target: &str) -> bool {
        let (verb, _) = self.parts();
        glob_match(verb, target)
    }

    /// Does `target_arg` (caller's actual tool argument) satisfy this
    /// entry's arg pattern? Entries without a `:` impose no arg constraint.
    #[must_use]
    pub fn arg_matches(&self, target_arg: &str) -> bool {
        let (_, pattern) = self.parts();
        pattern.is_none_or(|p| glob_match(p, target_arg))
    }
}

impl From<&str> for CapabilityEntry {
    fn from(s: &str) -> Self {
        Self::new(s)
    }
}

impl From<String> for CapabilityEntry {
    fn from(s: String) -> Self {
        Self::new(s)
    }
}

/// Tool capability matrix.
///
/// A set of allow-entries; the narrowing operation produces the
/// intersection of two matrices, which the runtime uses to combine the
/// global matrix with each agent's allowlist and each workspace's
/// `[tools] allow`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CapabilityMatrix(BTreeSet<CapabilityEntry>);

impl CapabilityMatrix {
    /// Empty matrix.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Build a matrix from an iterator of entries.
    pub fn from_entries<I, S>(entries: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<CapabilityEntry>,
    {
        Self(entries.into_iter().map(Into::into).collect())
    }

    /// Iterate every allow-entry in sorted order.
    pub fn entries(&self) -> impl Iterator<Item = &CapabilityEntry> {
        self.0.iter()
    }

    /// Is this matrix empty (deny-everything)?
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Number of allow-entries in the matrix.
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Does the matrix allow `verb` (e.g. `"fs.read"`) with the given
    /// `arg` (e.g. `"src/main.rs"`)? `None` arg means "no specific arg
    /// requested"; matches entries without an arg pattern.
    #[must_use]
    pub fn allows(&self, verb: &str, arg: Option<&str>) -> bool {
        self.0.iter().any(|entry| {
            if !entry.verb_matches(verb) {
                return false;
            }
            arg.map_or_else(|| entry.parts().1.is_none(), |a| entry.arg_matches(a))
        })
    }

    /// Intersect with another matrix: an entry is kept only if **both**
    /// matrices would allow the same verb-and-arg combination. The
    /// algorithm copies entries from `self` and asks whether `other`
    /// would still allow them. The reverse direction is the caller's
    /// responsibility when meaningful.
    #[must_use]
    pub fn narrowed_by(&self, other: &Self) -> Self {
        let kept: BTreeSet<CapabilityEntry> = self
            .0
            .iter()
            .filter(|entry| {
                let (verb, arg) = entry.parts();
                other.allows(verb, arg)
            })
            .cloned()
            .collect();
        Self(kept)
    }
}

/// Tiny glob matcher supporting `*` (multi-char wildcard) and
/// `**` (path-aware wildcard treated like `*`). No `?`, no character
/// classes. Used by capability-entry pattern matching.
fn glob_match(pattern: &str, target: &str) -> bool {
    let mut pi = pattern.chars().peekable();
    let target_bytes = target.as_bytes();
    let mut ti = 0;
    let mut backtrack: Option<(std::iter::Peekable<std::str::Chars<'_>>, usize)> = None;
    loop {
        let pc = pi.peek().copied();
        match pc {
            Some('*') => {
                // Consume star (collapse `**` into a single wildcard).
                pi.next();
                if pi.peek() == Some(&'*') {
                    pi.next();
                }
                backtrack = Some((pi.clone(), ti));
                if pi.peek().is_none() {
                    return true;
                }
            }
            Some(c) => {
                if ti < target_bytes.len() && target_bytes[ti] == c as u8 {
                    pi.next();
                    ti += 1;
                } else if let Some((saved_pi, saved_ti)) = backtrack.as_ref() {
                    pi = saved_pi.clone();
                    ti = saved_ti + 1;
                    if ti > target_bytes.len() {
                        return false;
                    }
                    // Re-arm the backtrack point: still inside the wildcard.
                    backtrack = Some((pi.clone(), ti));
                } else {
                    return false;
                }
            }
            None => {
                if ti == target_bytes.len() {
                    return true;
                }
                if let Some((saved_pi, saved_ti)) = backtrack.as_ref() {
                    pi = saved_pi.clone();
                    ti = saved_ti + 1;
                    if ti > target_bytes.len() {
                        return false;
                    }
                    backtrack = Some((pi.clone(), ti));
                } else {
                    return false;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entry_parts_without_arg() {
        let e = CapabilityEntry::from("fs.read");
        assert_eq!(e.parts(), ("fs.read", None));
    }

    #[test]
    fn entry_parts_with_arg() {
        let e = CapabilityEntry::from("fs.write:src/**.py");
        assert_eq!(e.parts(), ("fs.write", Some("src/**.py")));
    }

    #[test]
    fn entry_verb_matches_exact() {
        let e = CapabilityEntry::from("fs.read");
        assert!(e.verb_matches("fs.read"));
        assert!(!e.verb_matches("fs.write"));
    }

    #[test]
    fn entry_verb_matches_wildcard() {
        let e = CapabilityEntry::from("git.*");
        assert!(e.verb_matches("git.diff"));
        assert!(e.verb_matches("git.log"));
        assert!(!e.verb_matches("fs.read"));
    }

    #[test]
    fn entry_arg_matches_without_constraint() {
        let e = CapabilityEntry::from("fs.read");
        assert!(e.arg_matches("anything"));
    }

    #[test]
    fn entry_arg_matches_glob() {
        let e = CapabilityEntry::from("bash.run:py*");
        assert!(e.arg_matches("python3"));
        assert!(e.arg_matches("pytest"));
        assert!(!e.arg_matches("rustc"));
    }

    #[test]
    fn matrix_starts_empty() {
        let m = CapabilityMatrix::new();
        assert!(m.is_empty());
        assert_eq!(m.len(), 0);
    }

    #[test]
    fn matrix_from_iter_collects_entries() {
        let m = CapabilityMatrix::from_entries(["fs.read", "fs.write:src/**"]);
        assert_eq!(m.len(), 2);
        assert!(!m.is_empty());
    }

    #[test]
    fn matrix_allows_without_arg() {
        let m = CapabilityMatrix::from_entries(["fs.read"]);
        assert!(m.allows("fs.read", None));
        assert!(!m.allows("fs.write", None));
    }

    #[test]
    fn matrix_allows_with_arg() {
        let m = CapabilityMatrix::from_entries(["fs.write:src/**.py"]);
        assert!(m.allows("fs.write", Some("src/main.py")));
        assert!(m.allows("fs.write", Some("src/nested/x.py")));
        assert!(!m.allows("fs.write", Some("docs/x.md")));
    }

    #[test]
    fn matrix_allows_wildcard_verb() {
        let m = CapabilityMatrix::from_entries(["git.*"]);
        assert!(m.allows("git.diff", None));
        assert!(m.allows("git.log", None));
        assert!(!m.allows("fs.read", None));
    }

    #[test]
    fn matrix_arg_required_when_entry_has_one() {
        let m = CapabilityMatrix::from_entries(["fs.write:src/**"]);
        // Caller asks "fs.write with no arg" — entry requires an arg → deny.
        assert!(!m.allows("fs.write", None));
    }

    #[test]
    fn narrow_keeps_intersection() {
        let global = CapabilityMatrix::from_entries(["fs.read", "fs.write", "bash.run"]);
        let agent = CapabilityMatrix::from_entries(["fs.read", "git.*"]);
        let narrowed = global.narrowed_by(&agent);
        let entries: Vec<&str> = narrowed.entries().map(CapabilityEntry::as_str).collect();
        assert_eq!(entries, vec!["fs.read"]);
    }

    #[test]
    fn narrow_empty_other_yields_empty() {
        let m = CapabilityMatrix::from_entries(["fs.read", "fs.write"]);
        let empty = CapabilityMatrix::new();
        assert!(m.narrowed_by(&empty).is_empty());
    }

    #[test]
    fn narrow_agent_unrestricted_keeps_global_arg_pattern() {
        let global = CapabilityMatrix::from_entries(["fs.write:src/**"]);
        let agent = CapabilityMatrix::from_entries(["fs.write"]);
        // Agent's bare `fs.write` allows the verb for any arg, so the
        // global's more-restrictive `fs.write:src/**` survives the
        // intersection unchanged.
        let narrowed = global.narrowed_by(&agent);
        let names: Vec<&str> = narrowed.entries().map(CapabilityEntry::as_str).collect();
        assert_eq!(names, vec!["fs.write:src/**"]);
    }

    #[test]
    fn narrow_drops_entries_when_agent_verb_not_allowed() {
        let global = CapabilityMatrix::from_entries(["fs.write:src/**", "fs.read"]);
        let agent = CapabilityMatrix::from_entries(["fs.read"]);
        let narrowed = global.narrowed_by(&agent);
        let names: Vec<&str> = narrowed.entries().map(CapabilityEntry::as_str).collect();
        assert_eq!(names, vec!["fs.read"]);
    }

    #[test]
    fn capability_entry_serde_roundtrip() {
        let e = CapabilityEntry::from("bash.run:py*");
        let s = serde_json::to_string(&e).unwrap();
        let back: CapabilityEntry = serde_json::from_str(&s).unwrap();
        assert_eq!(e, back);
    }

    #[test]
    fn matrix_serde_roundtrip() {
        let m = CapabilityMatrix::from_entries(["fs.read", "git.*"]);
        let s = serde_json::to_string(&m).unwrap();
        let back: CapabilityMatrix = serde_json::from_str(&s).unwrap();
        assert_eq!(m, back);
    }

    #[test]
    fn glob_exact_match() {
        assert!(glob_match("hello", "hello"));
        assert!(!glob_match("hello", "world"));
    }

    #[test]
    fn glob_empty_pattern_matches_empty_target() {
        assert!(glob_match("", ""));
        assert!(!glob_match("", "x"));
    }

    #[test]
    fn glob_wildcard_at_end() {
        assert!(glob_match("py*", "pytest"));
        assert!(glob_match("py*", "py"));
        assert!(!glob_match("py*", "rust"));
    }

    #[test]
    fn glob_wildcard_in_middle() {
        assert!(glob_match("src/*.py", "src/main.py"));
        assert!(!glob_match("src/*.py", "src/main.rs"));
    }

    #[test]
    fn glob_double_star_collapses_to_wildcard() {
        assert!(glob_match("src/**", "src/a/b/c.py"));
        assert!(glob_match("**.rs", "src/main.rs"));
    }

    #[test]
    fn glob_handles_unmatched_literal_after_wildcard() {
        assert!(glob_match("a*c", "abc"));
        assert!(glob_match("a*c", "axyzc"));
        assert!(!glob_match("a*c", "abx"));
    }

    #[test]
    fn capability_entry_as_str_returns_original() {
        let e = CapabilityEntry::from(String::from("fs.read"));
        assert_eq!(e.as_str(), "fs.read");
    }
}
