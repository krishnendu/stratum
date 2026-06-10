// xtask-check-sentinel-codes: ignore-file
//
// Reason: this module's tests fabricate `E_*` sentinel literals as fixtures
// and would otherwise be reported as unknown. The allowlist itself also
// declares each sentinel as a Rust string literal, which the scanner would
// pick up — but the file is the source of truth for the allowlist, so it
// must opt out of being scanned.

//! `check-sentinel-codes` validator.
//!
//! Complements [`check_error_codes`](super::check_error_codes) by auditing
//! the **local sentinel** namespace — short, all-caps identifiers like
//! `E_NO_BLOCKS`, `E_TOOL_PANIC`, `E_DISPATCH_TIMEOUT` that runtime modules
//! emit on [`stratum_runtime::Event::ProviderError`] /
//! [`stratum_runtime::TurnOutcome::ToolFailure`] paths.
//!
//! Sentinels are **not** the same as catalogued `STRAT-E####` codes:
//!
//! * Catalog codes live in `stratum_types::error::codes` and are validated by
//!   `check-error-codes`.
//! * Sentinels are runtime-local discriminants used for branching event
//!   handling in the CLI/TUI; they have no global registry, so this validator
//!   maintains a small hardcoded [`ALLOWLIST`] and reports drift.
//!
//! The scanner walks every workspace `.rs` file and looks for string literals
//! of the form `"E_[A-Z_]+"`. The walker skips:
//!
//! * `target/`, `.git/`, `.claude/`, `plan/` (matches `check-error-codes`).
//! * Files that contain the marker `xtask-check-sentinel-codes: ignore-file`.
//!   The catalog scanner's `xtask-check-error-codes: ignore-file` marker is
//!   **not** honoured here: several runtime modules (e.g.
//!   `tool_dispatchers.rs`, `tool_dispatcher_mcp.rs`) opt out of the catalog
//!   scan precisely because they declare `E_DISPATCH_*` sentinels, which
//!   means they are exactly the files this validator needs to read.
//! * Lines inside `mod tests` blocks, detected by a top-level
//!   `#[cfg(test)]` attribute and skipped until the next top-level `}`.

#![allow(unreachable_pub)]

use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// Errors raised by the validator.
#[derive(Debug, thiserror::Error)]
pub enum RunError {
    /// An I/O error occurred while reading a source file.
    #[error("io error reading {path}: {source}")]
    Io {
        /// File path that failed to read.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: io::Error,
    },
}

/// Hardcoded allowlist of sentinels that may legally appear in workspace
/// source. New sentinels MUST be added here in the same PR that introduces
/// them.
pub const ALLOWLIST: &[&str] = &[
    "E_NO_BLOCKS",
    "E_TOOL_PANIC",
    "E_PROVIDER_PANIC",
    "E_DISPATCH_TIMEOUT",
    "E_DISPATCH_BIN_DISALLOWED",
    "E_DISPATCH_EXIT_NONZERO",
    "E_DISPATCH_PATH_ESCAPE",
    "E_DISPATCH_SIZE_CAP",
    "E_DISPATCH_READ_FAILED",
    "E_DISPATCH_SPAWN_FAILED",
    "E_DISPATCH_MISSING_ARG",
    "E_DISPATCH_MCP_WRONG_SERVER",
    "E_DISPATCH_MCP_TOOL_DENIED",
    "E_DISPATCH_MCP_TOOL_ERROR",
    "E_DISPATCH_MCP_TRANSPORT",
    "E_PLUGIN_BAD_OUTPUT",
    "E_PLUGIN_TIMEOUT",
    "E_PLUGIN_SPAWN_FAILED",
    "E_PLUGIN_NO_MATCH",
    "E_PLUGIN_INVOCATION_ENCODING",
];

/// Outcome of a single `run()` invocation.
#[derive(Debug, Default, Clone)]
pub struct Report {
    /// Allowlisted sentinels seen in the codebase, mapped to the list of
    /// files that mention them. Sentinels in the allowlist but absent from
    /// the codebase appear in [`orphans`](Self::orphans) instead.
    pub allowlisted: BTreeMap<String, Vec<PathBuf>>,
    /// Sentinels declared in [`ALLOWLIST`] that are not referenced anywhere
    /// in workspace source.
    pub orphans: Vec<String>,
    /// `(path, sentinel)` pairs found in source that are not in
    /// [`ALLOWLIST`].
    pub unknown: Vec<(PathBuf, String)>,
}

/// Opt-out marker — placing this string anywhere in the file (typically
/// inside a top-of-file comment) excludes it from the sentinel scan.
const OPT_OUT_MARKER: &str = "xtask-check-sentinel-codes: ignore-file";

/// Run the sentinel validator across `workspace_root`.
pub fn run(workspace_root: &Path) -> Result<Report, RunError> {
    let mut references: BTreeMap<String, Vec<PathBuf>> = BTreeMap::new();
    walk(workspace_root, &mut references)?;
    for paths in references.values_mut() {
        paths.sort();
        paths.dedup();
    }

    let mut allowlisted: BTreeMap<String, Vec<PathBuf>> = BTreeMap::new();
    let mut unknown: Vec<(PathBuf, String)> = Vec::new();
    for (sentinel, paths) in &references {
        if ALLOWLIST.contains(&sentinel.as_str()) {
            allowlisted.insert(sentinel.clone(), paths.clone());
        } else {
            for path in paths {
                unknown.push((path.clone(), sentinel.clone()));
            }
        }
    }
    unknown.sort();

    let mut orphans: Vec<String> = ALLOWLIST
        .iter()
        .filter(|s| !references.contains_key(**s))
        .map(|s| (*s).to_string())
        .collect();
    orphans.sort();

    Ok(Report {
        allowlisted,
        orphans,
        unknown,
    })
}

fn walk(dir: &Path, by_code: &mut BTreeMap<String, Vec<PathBuf>>) -> Result<(), RunError> {
    let entries = fs::read_dir(dir).map_err(|source| RunError::Io {
        path: dir.to_path_buf(),
        source,
    })?;
    for entry in entries {
        let entry = entry.map_err(|source| RunError::Io {
            path: dir.to_path_buf(),
            source,
        })?;
        let path = entry.path();
        if is_skipped_dir(&path) {
            continue;
        }
        let file_type = entry.file_type().map_err(|source| RunError::Io {
            path: path.clone(),
            source,
        })?;
        if file_type.is_dir() {
            walk(&path, by_code)?;
            continue;
        }
        if !file_type.is_file() {
            continue;
        }
        if !is_rust_file(&path) {
            continue;
        }
        let body = fs::read_to_string(&path).map_err(|source| RunError::Io {
            path: path.clone(),
            source,
        })?;
        if is_opt_out(&body) {
            continue;
        }
        for sentinel in parse_sentinels_from_text(&body) {
            by_code.entry(sentinel).or_default().push(path.clone());
        }
    }
    Ok(())
}

fn is_skipped_dir(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
        return false;
    };
    matches!(name, "target" | ".git" | ".claude" | "plan")
}

fn is_rust_file(path: &Path) -> bool {
    path.extension().and_then(|e| e.to_str()) == Some("rs")
}

fn is_opt_out(body: &str) -> bool {
    body.contains(OPT_OUT_MARKER)
}

/// Scan `text` and return every `E_[A-Z_]+` string literal found, skipping
/// regions inside top-level `#[cfg(test)]` items.
///
/// The skip heuristic is intentionally simple: a line whose trimmed-left
/// content equals `#[cfg(test)]` and which has **no leading whitespace**
/// opens a test region, which closes at the next line that starts with `}`
/// at column 0. This catches the standard
///
/// ```text
/// #[cfg(test)]
/// mod tests {
///     ...
/// }
/// ```
///
/// pattern at the bottom of most source files while leaving in-impl
/// `#[cfg(test)] fn helper()` declarations alone.
#[must_use]
pub fn parse_sentinels_from_text(text: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut in_test_region = false;
    for line in text.lines() {
        if in_test_region {
            if line.starts_with('}') {
                in_test_region = false;
            }
            continue;
        }
        if line == "#[cfg(test)]" {
            in_test_region = true;
            continue;
        }
        out.extend(parse_sentinels_in_line(line));
    }
    out
}

/// Pull every `"E_[A-Z_]+"` quoted literal out of a single line.
fn parse_sentinels_in_line(line: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let bytes = line.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] != b'"' {
            i += 1;
            continue;
        }
        // Possible start of a string literal. Find the matching close quote.
        let content_start = i + 1;
        let mut j = content_start;
        while j < bytes.len() && bytes[j] != b'"' {
            // Honour backslash escapes so we don't terminate on \".
            if bytes[j] == b'\\' && j + 1 < bytes.len() {
                j += 2;
                continue;
            }
            j += 1;
        }
        if j >= bytes.len() {
            // Unterminated literal on this line — bail.
            break;
        }
        let literal = &line[content_start..j];
        if is_sentinel_literal(literal) {
            out.push(literal.to_string());
        }
        i = j + 1;
    }
    out
}

/// True iff `s` matches `^E_[A-Z_]+$`, where the suffix is at least one
/// character of `[A-Z_]`.
fn is_sentinel_literal(s: &str) -> bool {
    let bytes = s.as_bytes();
    if bytes.len() < 3 {
        return false;
    }
    if bytes[0] != b'E' || bytes[1] != b'_' {
        return false;
    }
    bytes[2..]
        .iter()
        .all(|b| b.is_ascii_uppercase() || *b == b'_')
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn write(path: &Path, body: &str) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(path, body).unwrap();
    }

    #[test]
    fn parse_sentinels_finds_literal_in_simple_line() {
        let text = "let code = \"E_NO_BLOCKS\";";
        assert_eq!(parse_sentinels_from_text(text), vec!["E_NO_BLOCKS"]);
    }

    #[test]
    fn parse_sentinels_finds_multiple_per_line() {
        let text = "match code { \"E_TOOL_PANIC\" | \"E_PROVIDER_PANIC\" => 1, _ => 0 }";
        assert_eq!(
            parse_sentinels_from_text(text),
            vec!["E_TOOL_PANIC", "E_PROVIDER_PANIC"],
        );
    }

    #[test]
    fn parse_sentinels_skips_top_level_cfg_test_block() {
        let text = "fn prod() -> &'static str { \"E_NO_BLOCKS\" }\n\
                    #[cfg(test)]\n\
                    mod tests {\n    \
                        fn t() { let _ = \"E_FABRICATED\"; }\n\
                    }\n\
                    fn after() -> &'static str { \"E_TOOL_PANIC\" }\n";
        let got = parse_sentinels_from_text(text);
        assert_eq!(got, vec!["E_NO_BLOCKS", "E_TOOL_PANIC"]);
    }

    #[test]
    fn parse_sentinels_keeps_indented_cfg_test() {
        // An indented `#[cfg(test)]` (e.g. inside an `impl` block on a
        // single helper fn) should NOT trigger the skip — the brief
        // restricts the heuristic to top-level test modules.
        let text = "impl Foo {\n    \
                        #[cfg(test)]\n    \
                        fn helper() -> &'static str { \"E_TOOL_TIMEOUT\" }\n\
                    }\n";
        assert_eq!(parse_sentinels_from_text(text), vec!["E_TOOL_TIMEOUT"]);
    }

    #[test]
    fn parse_sentinels_ignores_non_sentinel_strings() {
        let text = "let s = \"hello\"; let t = \"e_lower_case\"; let u = \"EX_NO\";";
        assert!(parse_sentinels_from_text(text).is_empty());
    }

    #[test]
    fn parse_sentinels_handles_escaped_quotes() {
        let text = "let s = \"he said \\\"hi\\\"\"; let t = \"E_TOOL_PANIC\";";
        assert_eq!(parse_sentinels_from_text(text), vec!["E_TOOL_PANIC"]);
    }

    #[test]
    fn parse_sentinels_requires_underscore_after_e() {
        // `"E1001_FOO"` (no underscore right after `E`) is not a sentinel.
        let text = "let s = \"E1001_FOO\"; let t = \"E_OK\";";
        assert_eq!(parse_sentinels_from_text(text), vec!["E_OK"]);
    }

    #[test]
    fn parse_sentinels_rejects_lowercase_in_suffix() {
        let text = "let s = \"E_Mixed\";";
        assert!(parse_sentinels_from_text(text).is_empty());
    }

    #[test]
    fn parse_sentinels_rejects_too_short() {
        let text = "let s = \"E_\"; let t = \"E\";";
        assert!(parse_sentinels_from_text(text).is_empty());
    }

    #[test]
    fn is_sentinel_literal_examples() {
        assert!(is_sentinel_literal("E_NO_BLOCKS"));
        assert!(is_sentinel_literal("E_X"));
        assert!(!is_sentinel_literal("E_"));
        assert!(!is_sentinel_literal("E"));
        assert!(!is_sentinel_literal(""));
        assert!(!is_sentinel_literal("e_lower"));
        assert!(!is_sentinel_literal("EX_NO"));
        assert!(!is_sentinel_literal("E_no"));
    }

    #[test]
    fn is_opt_out_recognises_sentinel_marker() {
        assert!(is_opt_out("// xtask-check-sentinel-codes: ignore-file\n"));
        assert!(!is_opt_out("// xtask-check-error-codes: ignore-file\n"));
        assert!(!is_opt_out("// some other comment\n"));
    }

    #[test]
    fn allowlist_lookup_is_exact() {
        assert!(ALLOWLIST.contains(&"E_NO_BLOCKS"));
        assert!(ALLOWLIST.contains(&"E_DISPATCH_MCP_TRANSPORT"));
        assert!(!ALLOWLIST.contains(&"E_NOT_A_REAL_SENTINEL"));
        // No leading/trailing whitespace allowed.
        assert!(!ALLOWLIST.contains(&" E_NO_BLOCKS"));
    }

    fn write_workspace_marker(root: &Path) {
        write(&root.join("Cargo.toml"), "[workspace]\nmembers=[]\n");
    }

    #[test]
    fn run_synthetic_three_file_fixture_classifies_correctly() {
        let tmp = TempDir::new().unwrap();
        write_workspace_marker(tmp.path());
        let allowed = tmp.path().join("crates").join("a").join("src").join("a.rs");
        let unknown_path = tmp.path().join("crates").join("b").join("src").join("b.rs");
        let neutral = tmp.path().join("crates").join("c").join("src").join("c.rs");
        write(
            &allowed,
            "fn a() { let _ = \"E_NO_BLOCKS\"; let _ = \"E_TOOL_PANIC\"; }\n",
        );
        write(&unknown_path, "fn b() { let _ = \"E_TOTALLY_NEW\"; }\n");
        write(&neutral, "fn c() { let _ = \"unrelated\"; }\n");

        let report = run(tmp.path()).unwrap();

        // Allowlisted: the two sentinels we planted.
        assert!(report.allowlisted.contains_key("E_NO_BLOCKS"));
        assert!(report.allowlisted.contains_key("E_TOOL_PANIC"));
        assert_eq!(report.allowlisted["E_NO_BLOCKS"], vec![allowed.clone()]);
        assert_eq!(report.allowlisted["E_TOOL_PANIC"], vec![allowed]);

        // Unknown: the synthetic literal not in the allowlist.
        assert!(report
            .unknown
            .iter()
            .any(|(_, c)| c.as_str() == "E_TOTALLY_NEW"),);

        // Orphans: every allowlist entry NOT planted above must show up.
        for entry in ALLOWLIST {
            if *entry == "E_NO_BLOCKS" || *entry == "E_TOOL_PANIC" {
                continue;
            }
            assert!(
                report.orphans.iter().any(|o| o == entry),
                "expected {entry} in orphans, got {:?}",
                report.orphans,
            );
        }
    }

    #[test]
    fn run_respects_sentinel_ignore_file_marker() {
        let tmp = TempDir::new().unwrap();
        write_workspace_marker(tmp.path());
        let opt_out = tmp.path().join("crates").join("a").join("src").join("a.rs");
        write(
            &opt_out,
            "// xtask-check-sentinel-codes: ignore-file\n\
             fn a() { let _ = \"E_TOTALLY_FAKE\"; let _ = \"E_NO_BLOCKS\"; }\n",
        );
        let report = run(tmp.path()).unwrap();
        // Neither the unknown nor the allowlisted sentinel should appear.
        assert!(
            report.unknown.iter().all(|(_, c)| c != "E_TOTALLY_FAKE"),
            "unknown should be empty for ignored file: {:?}",
            report.unknown,
        );
        assert!(
            !report.allowlisted.contains_key("E_NO_BLOCKS"),
            "ignored file should not feed the allowlisted set",
        );
    }

    #[test]
    fn run_does_not_honour_error_codes_marker_for_sentinel_scan() {
        // The catalog scanner's marker must NOT opt a file out of the
        // sentinel scan — runtime modules use that marker precisely because
        // they declare `E_DISPATCH_*` sentinels we want to track.
        let tmp = TempDir::new().unwrap();
        write_workspace_marker(tmp.path());
        let marked = tmp.path().join("crates").join("a").join("src").join("a.rs");
        write(
            &marked,
            "// xtask-check-error-codes: ignore-file\n\
             fn a() { let _ = \"E_DISPATCH_TIMEOUT\"; }\n",
        );
        let report = run(tmp.path()).unwrap();
        assert!(
            report.allowlisted.contains_key("E_DISPATCH_TIMEOUT"),
            "catalog marker should not hide allowlisted sentinels: {:?}",
            report.allowlisted,
        );
    }

    #[test]
    fn run_skips_top_level_cfg_test_modules() {
        let tmp = TempDir::new().unwrap();
        write_workspace_marker(tmp.path());
        let src = tmp.path().join("crates").join("a").join("src").join("a.rs");
        write(
            &src,
            "fn prod() { let _ = \"E_NO_BLOCKS\"; }\n\
             #[cfg(test)]\n\
             mod tests {\n    \
                 fn t() { let _ = \"E_NOT_REAL\"; }\n\
             }\n",
        );
        let report = run(tmp.path()).unwrap();
        assert!(
            report.unknown.iter().all(|(_, c)| c != "E_NOT_REAL"),
            "test-module literal should be skipped: {:?}",
            report.unknown,
        );
        assert!(report.allowlisted.contains_key("E_NO_BLOCKS"));
    }

    #[test]
    fn run_skips_standard_excluded_directories() {
        let tmp = TempDir::new().unwrap();
        write_workspace_marker(tmp.path());
        write(
            &tmp.path().join("target").join("a.rs"),
            "let _ = \"E_TOTALLY_FAKE\";\n",
        );
        write(
            &tmp.path().join(".git").join("a.rs"),
            "let _ = \"E_TOTALLY_FAKE\";\n",
        );
        write(
            &tmp.path().join(".claude").join("a.rs"),
            "let _ = \"E_TOTALLY_FAKE\";\n",
        );
        write(
            &tmp.path().join("plan").join("a.rs"),
            "let _ = \"E_TOTALLY_FAKE\";\n",
        );
        let report = run(tmp.path()).unwrap();
        assert!(
            report.unknown.iter().all(|(_, c)| c != "E_TOTALLY_FAKE"),
            "excluded dirs should be skipped: {:?}",
            report.unknown,
        );
    }

    #[test]
    fn run_ignores_non_rust_files() {
        let tmp = TempDir::new().unwrap();
        write_workspace_marker(tmp.path());
        write(
            &tmp.path().join("crates").join("a").join("README.md"),
            "Reference to \"E_TOTALLY_FAKE\".\n",
        );
        write(
            &tmp.path().join("crates").join("a").join("a.toml"),
            "value = \"E_TOTALLY_FAKE\"\n",
        );
        let report = run(tmp.path()).unwrap();
        assert!(
            report.unknown.iter().all(|(_, c)| c != "E_TOTALLY_FAKE"),
            "non-rust files should be ignored: {:?}",
            report.unknown,
        );
    }

    #[test]
    fn run_reports_all_orphans_for_empty_workspace() {
        let tmp = TempDir::new().unwrap();
        write_workspace_marker(tmp.path());
        let report = run(tmp.path()).unwrap();
        assert!(report.unknown.is_empty());
        assert!(report.allowlisted.is_empty());
        assert_eq!(report.orphans.len(), ALLOWLIST.len());
    }

    #[test]
    fn run_no_unknown_when_all_sentinels_present_and_in_allowlist() {
        let tmp = TempDir::new().unwrap();
        write_workspace_marker(tmp.path());
        let src = tmp.path().join("crates").join("a").join("src").join("a.rs");
        let mut body = String::new();
        for entry in ALLOWLIST {
            use std::fmt::Write as _;
            writeln!(body, "let _ = \"{entry}\";").unwrap();
        }
        write(&src, &body);
        let report = run(tmp.path()).unwrap();
        assert!(
            report.unknown.is_empty(),
            "unknown should be empty: {:?}",
            report.unknown,
        );
        assert!(
            report.orphans.is_empty(),
            "orphans should be empty: {:?}",
            report.orphans,
        );
        assert_eq!(report.allowlisted.len(), ALLOWLIST.len());
    }

    #[test]
    fn run_io_error_propagates_on_missing_root() {
        let tmp = TempDir::new().unwrap();
        let missing = tmp.path().join("does-not-exist");
        let err = run(&missing).unwrap_err();
        assert!(matches!(err, RunError::Io { .. }), "got {err:?}");
    }

    #[test]
    fn report_default_is_empty() {
        let r = Report::default();
        assert!(r.allowlisted.is_empty());
        assert!(r.orphans.is_empty());
        assert!(r.unknown.is_empty());
    }

    #[test]
    fn run_error_display_message_is_useful() {
        let e = RunError::Io {
            path: PathBuf::from("/x/y.rs"),
            source: io::Error::other("boom"),
        };
        assert!(format!("{e}").contains("io error reading"));
    }
}
