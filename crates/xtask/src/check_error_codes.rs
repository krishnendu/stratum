// xtask-check-error-codes: ignore-file
//
// Reason: this module's tests fabricate `STRAT-E####` literals as fixtures.
// They are not catalog usage and would otherwise be reported as unknown.

//! `check-error-codes` validator.
//!
//! Verifies the workspace invariants documented in
//! `plan/29-error-taxonomy-and-logging.md` §8 and
//! `plan/36-verification-gates.md` G5:
//!
//! 1. Every `STRAT-E####` literal that appears in workspace source
//!    (`*.rs`, `*.toml`, `*.yml` / `*.yaml`) outside of the catalog file
//!    `crates/stratum-types/src/error.rs` must be declared in
//!    `stratum_types::error::codes`.
//! 2. Every catalog constant declared in that module must be referenced from
//!    at least one workspace source file.
//!
//! The walker skips `target/`, `.git/`, `.claude/`, `plan/` (gitignored design
//! corpus), and Markdown files — those are doc surface, not code, and they
//! may legitimately mention codes without "using" them.
//!
//! Individual source files may opt out by placing the literal marker
//! `xtask-check-error-codes: ignore-file` somewhere in their content (typically
//! in a top-of-file comment). This is the escape hatch for test fixtures and
//! regex/parser tests that fabricate `STRAT-E####` strings without intending
//! them to be catalog entries.
//!
//! This module deliberately does **not** depend on `stratum-types` or use the
//! `STRAT-E####` catalog itself; doing so would create a chicken-and-egg
//! validation loop. Failures are surfaced via the local [`RunError`] enum.

// The `xtask` crate is a binary, so these `pub` items are only consumed by
// `main.rs` and the test module. We expose them as `pub` (rather than
// `pub(crate)`) because `plan/29-error-taxonomy-and-logging.md` §8 specifies
// the public API of this validator; that triggers `unreachable_pub` in a
// binary, so we explicitly allow it here.
#![allow(unreachable_pub)]

use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// Errors raised by the validator.
#[derive(Debug, thiserror::Error)]
pub enum RunError {
    /// The catalog source file was not where we expected it.
    #[error("catalog file not found at {0}")]
    MissingCatalog(PathBuf),
    /// An I/O error occurred while reading a source file.
    #[error("io error reading {path}: {source}")]
    Io {
        /// File path that failed to read.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: io::Error,
    },
    /// The catalog file was found but no constants could be parsed from it.
    #[error("failed to parse any catalog entries from {0}")]
    ParseEmpty(PathBuf),
}

/// Outcome of a single `run()` invocation.
#[derive(Debug, Default, Clone)]
pub struct Report {
    /// Catalog constant codes, in declaration order (e.g. `STRAT-E1001`).
    pub catalogued: Vec<String>,
    /// Map of `STRAT-E####` -> list of file paths referencing it (excluding
    /// the catalog file).
    pub references: BTreeMap<String, Vec<PathBuf>>,
    /// Catalogued codes with zero references anywhere in the workspace.
    pub orphans: Vec<String>,
    /// `(path, code)` pairs found in source that are not declared in the
    /// catalog.
    pub unknown: Vec<(PathBuf, String)>,
}

/// Run the validator across `workspace_root`.
///
/// `workspace_root` should be the directory that contains the top-level
/// `Cargo.toml`.
pub fn run(workspace_root: &Path) -> Result<Report, RunError> {
    let catalog_path = workspace_root
        .join("crates")
        .join("stratum-types")
        .join("src")
        .join("error.rs");
    if !catalog_path.is_file() {
        return Err(RunError::MissingCatalog(catalog_path));
    }
    let catalog = parse_catalog(&catalog_path)?;
    if catalog.is_empty() {
        return Err(RunError::ParseEmpty(catalog_path));
    }
    let catalogued_codes: Vec<String> = catalog.iter().map(|(_, code)| code.clone()).collect();
    let references = scan_workspace(workspace_root)?;
    let mut unknown: Vec<(PathBuf, String)> = Vec::new();
    for (code, paths) in &references {
        if !catalogued_codes.iter().any(|c| c == code) {
            for path in paths {
                unknown.push((path.clone(), code.clone()));
            }
        }
    }
    unknown.sort();
    let mut orphans: Vec<String> = catalogued_codes
        .iter()
        .filter(|code| !references.contains_key(*code))
        .cloned()
        .collect();
    orphans.sort();
    Ok(Report {
        catalogued: catalogued_codes,
        references,
        orphans,
        unknown,
    })
}

/// Extract every `pub const E####_FOO: ErrorCode = ErrorCode::new_static("STRAT-E####");`
/// declaration from the catalog file. Returns `(constant_name, code)` pairs.
pub fn parse_catalog(error_rs_path: &Path) -> Result<Vec<(String, String)>, RunError> {
    let body = fs::read_to_string(error_rs_path).map_err(|source| RunError::Io {
        path: error_rs_path.to_path_buf(),
        source,
    })?;
    let mut out: Vec<(String, String)> = Vec::new();
    for line in body.lines() {
        let trimmed = line.trim();
        if !trimmed.starts_with("pub const ") {
            continue;
        }
        // pub const E1001_FOO: ErrorCode = ErrorCode::new_static("STRAT-E1001");
        let name_segment = trimmed.trim_start_matches("pub const ");
        let Some(colon) = name_segment.find(':') else {
            continue;
        };
        let constant_name = name_segment[..colon].trim().to_string();
        let Some(code) = extract_code_literal(trimmed) else {
            continue;
        };
        out.push((constant_name, code));
    }
    Ok(out)
}

/// Walk the workspace and collect every `STRAT-E####` literal, excluding the
/// catalog file itself. Returns a map of `code -> distinct paths` (sorted,
/// de-duplicated). Skips `target/`, `.git/`, `.claude/`, `plan/`, and
/// Markdown files.
pub fn scan_workspace(root: &Path) -> Result<BTreeMap<String, Vec<PathBuf>>, RunError> {
    let mut by_code: BTreeMap<String, Vec<PathBuf>> = BTreeMap::new();
    let catalog_path = root
        .join("crates")
        .join("stratum-types")
        .join("src")
        .join("error.rs");
    walk(root, &catalog_path, &mut by_code)?;
    for paths in by_code.values_mut() {
        paths.sort();
        paths.dedup();
    }
    Ok(by_code)
}

fn walk(
    dir: &Path,
    catalog_path: &Path,
    by_code: &mut BTreeMap<String, Vec<PathBuf>>,
) -> Result<(), RunError> {
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
            walk(&path, catalog_path, by_code)?;
            continue;
        }
        if !file_type.is_file() {
            continue;
        }
        if path == catalog_path {
            continue;
        }
        if !is_code_bearing(&path) {
            continue;
        }
        let body = fs::read_to_string(&path).map_err(|source| RunError::Io {
            path: path.clone(),
            source,
        })?;
        if is_opt_out(&body) {
            continue;
        }
        for code in extract_codes(&body) {
            by_code.entry(code).or_default().push(path.clone());
        }
    }
    Ok(())
}

/// Magic marker that allows a file to opt out of the scan.
const OPT_OUT_MARKER: &str = "xtask-check-error-codes: ignore-file";

fn is_opt_out(body: &str) -> bool {
    body.contains(OPT_OUT_MARKER)
}

fn is_skipped_dir(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
        return false;
    };
    matches!(name, "target" | ".git" | ".claude" | "plan")
}

fn is_code_bearing(path: &Path) -> bool {
    let Some(ext) = path.extension().and_then(|e| e.to_str()) else {
        return false;
    };
    matches!(ext, "rs" | "toml" | "yml" | "yaml")
}

/// Extract every `STRAT-E####` substring from `text`.
fn extract_codes(text: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let bytes = text.as_bytes();
    let needle = b"STRAT-E";
    let mut i = 0;
    while i + needle.len() + 4 <= bytes.len() {
        if &bytes[i..i + needle.len()] == needle {
            let digits_start = i + needle.len();
            let digits_end = digits_start + 4;
            if digits_end <= bytes.len()
                && bytes[digits_start..digits_end]
                    .iter()
                    .all(u8::is_ascii_digit)
            {
                // Reject longer digit runs (e.g., STRAT-E12345).
                let next_byte_is_digit =
                    digits_end < bytes.len() && bytes[digits_end].is_ascii_digit();
                if !next_byte_is_digit {
                    let mut code = String::with_capacity(11);
                    code.push_str("STRAT-E");
                    // SAFETY: digits are ASCII by construction; build via push_char
                    // — but we have to avoid `unwrap`, so we re-read from the
                    // bytes slice as `str` instead.
                    if let Ok(suffix) = std::str::from_utf8(&bytes[digits_start..digits_end]) {
                        code.push_str(suffix);
                        out.push(code);
                    }
                }
                i = digits_end;
                continue;
            }
        }
        i += 1;
    }
    out
}

fn extract_code_literal(line: &str) -> Option<String> {
    let needle = "ErrorCode::new_static(\"";
    let start = line.find(needle)? + needle.len();
    let rest = line.get(start..)?;
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
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

    fn fixture_catalog(extra: &str) -> String {
        format!(
            r#"
use std::borrow::Cow;
#[derive(Debug, Clone)]
pub struct ErrorCode(Cow<'static, str>);
impl ErrorCode {{
    pub const fn new_static(code: &'static str) -> Self {{ Self(Cow::Borrowed(code)) }}
}}
pub mod codes {{
    use super::ErrorCode;
    /// docs
    pub const E1001_FOO: ErrorCode = ErrorCode::new_static("STRAT-E1001");
    /// docs
    pub const E2003_BAR: ErrorCode = ErrorCode::new_static("STRAT-E2003");
    {extra}
}}
"#
        )
    }

    fn write_workspace(root: &Path, catalog_extra: &str) {
        // Workspace marker.
        write(&root.join("Cargo.toml"), "[workspace]\nmembers = []\n");
        write(
            &root
                .join("crates")
                .join("stratum-types")
                .join("src")
                .join("error.rs"),
            &fixture_catalog(catalog_extra),
        );
    }

    #[test]
    fn parse_catalog_extracts_constants() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("error.rs");
        write(&path, &fixture_catalog(""));
        let parsed = parse_catalog(&path).unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].0, "E1001_FOO");
        assert_eq!(parsed[0].1, "STRAT-E1001");
        assert_eq!(parsed[1].0, "E2003_BAR");
        assert_eq!(parsed[1].1, "STRAT-E2003");
    }

    #[test]
    fn parse_catalog_missing_file_is_io_error() {
        let tmp = TempDir::new().unwrap();
        let missing = tmp.path().join("nope.rs");
        let err = parse_catalog(&missing).unwrap_err();
        assert!(matches!(err, RunError::Io { .. }), "got {err:?}");
    }

    #[test]
    fn parse_catalog_skips_non_const_lines() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("error.rs");
        write(
            &path,
            r#"
// pub const E9000_COMMENTED: ErrorCode = ErrorCode::new_static("STRAT-E9000");
pub const E1001_FOO: ErrorCode = ErrorCode::new_static("STRAT-E1001");
const PRIVATE_E1234: ErrorCode = ErrorCode::new_static("STRAT-E1234");
"#,
        );
        let parsed = parse_catalog(&path).unwrap();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].1, "STRAT-E1001");
    }

    #[test]
    fn scan_workspace_finds_planted_reference() {
        let tmp = TempDir::new().unwrap();
        write_workspace(tmp.path(), "");
        let user = tmp
            .path()
            .join("crates")
            .join("u")
            .join("src")
            .join("lib.rs");
        write(&user, "fn x() { let _ = \"STRAT-E1001\"; }\n");
        let refs = scan_workspace(tmp.path()).unwrap();
        assert!(refs.contains_key("STRAT-E1001"));
        assert_eq!(refs["STRAT-E1001"], vec![user]);
    }

    #[test]
    fn scan_workspace_finds_codes_in_toml_and_yaml() {
        let tmp = TempDir::new().unwrap();
        write_workspace(tmp.path(), "");
        let toml = tmp.path().join("a.toml");
        let yaml = tmp.path().join("a.yml");
        let yaml2 = tmp.path().join("b.yaml");
        write(&toml, "code = \"STRAT-E1001\"\n");
        write(&yaml, "code: STRAT-E2003\n");
        write(&yaml2, "code: STRAT-E2003\n");
        let refs = scan_workspace(tmp.path()).unwrap();
        assert!(refs.get("STRAT-E1001").unwrap().contains(&toml));
        let twos = refs.get("STRAT-E2003").unwrap();
        assert!(twos.contains(&yaml));
        assert!(twos.contains(&yaml2));
    }

    #[test]
    fn scan_workspace_skips_target_git_claude_plan_and_md() {
        let tmp = TempDir::new().unwrap();
        write_workspace(tmp.path(), "");
        // Each of these should be ignored.
        write(
            &tmp.path().join("target").join("foo.rs"),
            "// STRAT-E1001\n",
        );
        write(&tmp.path().join(".git").join("bar.rs"), "// STRAT-E1001\n");
        write(
            &tmp.path().join(".claude").join("baz.rs"),
            "// STRAT-E1001\n",
        );
        write(&tmp.path().join("plan").join("doc.rs"), "// STRAT-E1001\n");
        write(&tmp.path().join("docs.md"), "STRAT-E1001\n");
        let refs = scan_workspace(tmp.path()).unwrap();
        assert!(
            !refs.contains_key("STRAT-E1001"),
            "expected zero references but found {:?}",
            refs.get("STRAT-E1001"),
        );
    }

    #[test]
    fn scan_workspace_excludes_catalog_self_reference() {
        let tmp = TempDir::new().unwrap();
        write_workspace(tmp.path(), "");
        // Catalog file mentions STRAT-E1001 in its own source — that should
        // be ignored, but a reference in another file is kept.
        let other = tmp
            .path()
            .join("crates")
            .join("u")
            .join("src")
            .join("lib.rs");
        write(&other, "// STRAT-E1001\n");
        let refs = scan_workspace(tmp.path()).unwrap();
        let paths = refs.get("STRAT-E1001").unwrap();
        assert_eq!(paths, &vec![other]);
    }

    #[test]
    fn run_reports_clean_when_consistent() {
        let tmp = TempDir::new().unwrap();
        write_workspace(tmp.path(), "");
        let a = tmp
            .path()
            .join("crates")
            .join("a")
            .join("src")
            .join("lib.rs");
        let b = tmp
            .path()
            .join("crates")
            .join("b")
            .join("src")
            .join("lib.rs");
        write(&a, "// STRAT-E1001\n");
        write(&b, "// STRAT-E2003\n");
        let report = run(tmp.path()).unwrap();
        assert_eq!(report.catalogued, vec!["STRAT-E1001", "STRAT-E2003"]);
        assert!(report.unknown.is_empty(), "{:?}", report.unknown);
        assert!(report.orphans.is_empty(), "{:?}", report.orphans);
    }

    #[test]
    fn run_reports_unknown_code() {
        let tmp = TempDir::new().unwrap();
        write_workspace(tmp.path(), "");
        let a = tmp
            .path()
            .join("crates")
            .join("a")
            .join("src")
            .join("lib.rs");
        let b = tmp
            .path()
            .join("crates")
            .join("b")
            .join("src")
            .join("lib.rs");
        write(&a, "// STRAT-E1001\n// STRAT-E2003\n// STRAT-E9999\n");
        write(&b, "// STRAT-E1001\n// STRAT-E2003\n");
        let report = run(tmp.path()).unwrap();
        assert!(report.orphans.is_empty(), "{:?}", report.orphans);
        let unknown_codes: Vec<&str> = report.unknown.iter().map(|(_, c)| c.as_str()).collect();
        assert_eq!(unknown_codes, vec!["STRAT-E9999"]);
    }

    #[test]
    fn run_reports_orphan_code() {
        let tmp = TempDir::new().unwrap();
        write_workspace(
            tmp.path(),
            "pub const E5555_UNUSED: ErrorCode = ErrorCode::new_static(\"STRAT-E5555\");",
        );
        let a = tmp
            .path()
            .join("crates")
            .join("a")
            .join("src")
            .join("lib.rs");
        write(&a, "// STRAT-E1001\n// STRAT-E2003\n");
        let report = run(tmp.path()).unwrap();
        assert!(report.unknown.is_empty(), "{:?}", report.unknown);
        assert_eq!(report.orphans, vec!["STRAT-E5555"]);
    }

    #[test]
    fn run_missing_catalog_returns_error() {
        let tmp = TempDir::new().unwrap();
        write(&tmp.path().join("Cargo.toml"), "[workspace]\nmembers=[]\n");
        let err = run(tmp.path()).unwrap_err();
        assert!(matches!(err, RunError::MissingCatalog(_)), "{err:?}");
    }

    #[test]
    fn run_empty_catalog_returns_parse_empty() {
        let tmp = TempDir::new().unwrap();
        write(&tmp.path().join("Cargo.toml"), "[workspace]\n");
        // Valid file location, but no `pub const ... ErrorCode::new_static`
        // declarations.
        write(
            &tmp.path()
                .join("crates")
                .join("stratum-types")
                .join("src")
                .join("error.rs"),
            "// no catalog entries here\n",
        );
        let err = run(tmp.path()).unwrap_err();
        assert!(matches!(err, RunError::ParseEmpty(_)), "{err:?}");
    }

    #[test]
    fn extract_codes_skips_too_long_digit_runs() {
        let codes = extract_codes("STRAT-E12345 should not match four-digit code");
        assert!(codes.is_empty(), "got {codes:?}");
    }

    #[test]
    fn extract_codes_handles_multiple_in_one_file() {
        let codes = extract_codes("STRAT-E1001 and STRAT-E2003 and STRAT-E1001 again");
        assert_eq!(codes, vec!["STRAT-E1001", "STRAT-E2003", "STRAT-E1001"]);
    }

    #[test]
    fn extract_codes_rejects_too_short_digit_runs() {
        let codes = extract_codes("STRAT-E12 too short");
        assert!(codes.is_empty(), "got {codes:?}");
    }

    #[test]
    fn extract_code_literal_handles_typical_line() {
        let line = "pub const E1001_FOO: ErrorCode = ErrorCode::new_static(\"STRAT-E1001\");";
        assert_eq!(extract_code_literal(line), Some("STRAT-E1001".to_string()));
    }

    #[test]
    fn extract_code_literal_returns_none_for_unrelated_line() {
        assert_eq!(extract_code_literal("let x = 5;"), None);
    }

    #[test]
    fn scan_workspace_unreadable_root_returns_io_error() {
        let tmp = TempDir::new().unwrap();
        // Point at a path that does not exist to trigger read_dir failure.
        let missing = tmp.path().join("does-not-exist");
        let err = scan_workspace(&missing).unwrap_err();
        assert!(matches!(err, RunError::Io { .. }), "got {err:?}");
    }

    #[test]
    fn scan_workspace_respects_opt_out_marker() {
        let tmp = TempDir::new().unwrap();
        write_workspace(tmp.path(), "");
        let opt_out = tmp
            .path()
            .join("crates")
            .join("u")
            .join("src")
            .join("lib.rs");
        write(
            &opt_out,
            "// xtask-check-error-codes: ignore-file\n// STRAT-E9999\n",
        );
        let refs = scan_workspace(tmp.path()).unwrap();
        assert!(
            !refs.contains_key("STRAT-E9999"),
            "opt-out file should have been skipped"
        );
    }

    #[test]
    fn is_opt_out_detects_marker() {
        assert!(is_opt_out("// xtask-check-error-codes: ignore-file\n"));
        assert!(!is_opt_out("normal source file"));
    }

    #[test]
    fn report_default_is_empty() {
        let r = Report::default();
        assert!(r.catalogued.is_empty());
        assert!(r.references.is_empty());
        assert!(r.orphans.is_empty());
        assert!(r.unknown.is_empty());
    }

    #[test]
    fn run_error_display_messages_are_useful() {
        let missing = RunError::MissingCatalog(PathBuf::from("/x/error.rs"));
        let parse_empty = RunError::ParseEmpty(PathBuf::from("/y/error.rs"));
        assert!(format!("{missing}").contains("catalog file not found"));
        assert!(format!("{parse_empty}").contains("failed to parse"));
    }
}
