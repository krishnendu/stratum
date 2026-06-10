// xtask-check-error-codes: ignore-file
// xtask-check-sentinel-codes: ignore-file
//
// Reason: this binary's tests fabricate `STRAT-E####` and `E_*` literals
// in fixtures. They are not real catalog codes or sentinel usages.

//! `xtask` — workspace automation entry point.
//!
//! Subcommands:
//!
//! * `check-error-codes`: validate that every `STRAT-Exxxx` literal used in
//!   the workspace is declared in
//!   [`stratum_types::error::codes`](../stratum_types/error/codes/index.html)
//!   and that no catalog constant is orphaned.
//! * `check-sentinel-codes`: audit short local `E_*` sentinel literals
//!   (`E_NO_BLOCKS`, `E_TOOL_PANIC`, …) against a hardcoded allowlist.
//!
//! Running `xtask` with no subcommand executes **both** validators in order
//! and aggregates the exit status — `0` only if both succeed.
//!
//! This binary intentionally does **not** use the Stratum error catalog — it
//! is the tool that validates the catalog and so cannot depend on it without
//! creating a chicken-and-egg loop. Errors are surfaced as plain
//! `RunError` values printed to `stderr`, and the process exits non-zero on
//! any failure.

#![forbid(unsafe_code)]

use std::env;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

mod check_error_codes;
mod check_sentinel_codes;

const USAGE: &str = "\
xtask — Stratum workspace automation

USAGE:
    xtask [SUBCOMMAND]

SUBCOMMANDS:
    check-error-codes      Validate STRAT-E#### literals against the catalog
    check-sentinel-codes   Validate local E_* sentinel literals against the
                           hardcoded allowlist
    (no subcommand)        Run both validators in order; exit non-zero if
                           either fails
    help, --help, -h       Show this message

EXIT STATUS:
    0   success
    1   validation failure (unknown codes, orphans, or I/O error)
    2   bad usage
";

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    let stdout = io::stdout();
    let stderr = io::stderr();
    let mut out = stdout.lock();
    let mut err = stderr.lock();
    run(&args, &mut out, &mut err)
}

fn run<O: Write, E: Write>(args: &[String], out: &mut O, err: &mut E) -> ExitCode {
    match args.first().map(String::as_str) {
        None => dispatch_all(out, err),
        Some("check-error-codes") => match dispatch_check_error_codes(out, err) {
            Ok(()) => ExitCode::SUCCESS,
            Err(code) => code,
        },
        Some("check-sentinel-codes") => match dispatch_check_sentinel_codes(out, err) {
            Ok(()) => ExitCode::SUCCESS,
            Err(code) => code,
        },
        Some("help" | "--help" | "-h") => {
            let _ = writeln!(out, "{USAGE}");
            ExitCode::SUCCESS
        }
        Some(other) => {
            let _ = writeln!(err, "xtask: unknown subcommand '{other}'\n\n{USAGE}");
            ExitCode::from(2)
        }
    }
}

/// Run every validator in declaration order, surfacing the worst exit code.
fn dispatch_all<O: Write, E: Write>(out: &mut O, err: &mut E) -> ExitCode {
    // Run both validators unconditionally so the operator sees every report
    // even when an earlier one fails.
    let error_codes_failed = dispatch_check_error_codes(out, err).is_err();
    let sentinel_failed = dispatch_check_sentinel_codes(out, err).is_err();
    if error_codes_failed || sentinel_failed {
        ExitCode::from(1)
    } else {
        ExitCode::SUCCESS
    }
}

fn dispatch_check_error_codes<O: Write, E: Write>(
    out: &mut O,
    err: &mut E,
) -> Result<(), ExitCode> {
    let root = workspace_root();
    dispatch_check_error_codes_at(&root, out, err)
}

fn dispatch_check_error_codes_at<O: Write, E: Write>(
    root: &Path,
    out: &mut O,
    err: &mut E,
) -> Result<(), ExitCode> {
    match check_error_codes::run(root) {
        Ok(report) => {
            let _ = writeln!(
                out,
                "check-error-codes: scanned {} catalog entries, {} reference sites",
                report.catalogued.len(),
                report.references.values().map(Vec::len).sum::<usize>(),
            );
            if !report.unknown.is_empty() {
                let _ = writeln!(err, "error: undeclared STRAT-E#### codes in workspace:");
                for (path, code) in &report.unknown {
                    let _ = writeln!(err, "  {} -> {}", path.display(), code);
                }
            }
            if !report.orphans.is_empty() {
                let _ = writeln!(err, "error: catalog codes with no references:");
                for code in &report.orphans {
                    let _ = writeln!(err, "  {code}");
                }
            }
            if report.unknown.is_empty() && report.orphans.is_empty() {
                Ok(())
            } else {
                Err(ExitCode::from(1))
            }
        }
        Err(e) => {
            let _ = writeln!(err, "check-error-codes failed: {e}");
            Err(ExitCode::from(1))
        }
    }
}

fn dispatch_check_sentinel_codes<O: Write, E: Write>(
    out: &mut O,
    err: &mut E,
) -> Result<(), ExitCode> {
    let root = workspace_root();
    dispatch_check_sentinel_codes_at(&root, out, err)
}

fn dispatch_check_sentinel_codes_at<O: Write, E: Write>(
    root: &Path,
    out: &mut O,
    err: &mut E,
) -> Result<(), ExitCode> {
    match check_sentinel_codes::run(root) {
        Ok(report) => {
            let usage_sites: usize = report.allowlisted.values().map(Vec::len).sum();
            let _ = writeln!(
                out,
                "check-sentinel-codes: {} allowlisted sentinels in use ({} usage sites), {} orphans, {} unknown",
                report.allowlisted.len(),
                usage_sites,
                report.orphans.len(),
                report.unknown.len(),
            );
            for (sentinel, paths) in &report.allowlisted {
                let _ = writeln!(out, "  ok  {sentinel} ({} site(s))", paths.len());
            }
            if !report.unknown.is_empty() {
                let _ = writeln!(err, "error: sentinel literals not in the xtask allowlist:",);
                for (path, sentinel) in &report.unknown {
                    let _ = writeln!(err, "  {} -> {}", path.display(), sentinel);
                }
            }
            if !report.orphans.is_empty() {
                let _ = writeln!(
                    err,
                    "error: allowlisted sentinels with no references in source:",
                );
                for sentinel in &report.orphans {
                    let _ = writeln!(err, "  {sentinel}");
                }
            }
            if report.unknown.is_empty() && report.orphans.is_empty() {
                Ok(())
            } else {
                Err(ExitCode::from(1))
            }
        }
        Err(e) => {
            let _ = writeln!(err, "check-sentinel-codes failed: {e}");
            Err(ExitCode::from(1))
        }
    }
}

/// Walk up from `CARGO_MANIFEST_DIR` to find the workspace root (the directory
/// containing the workspace-level `Cargo.toml` with `[workspace]`).
fn workspace_root() -> PathBuf {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").map_or_else(
        |_| env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
        PathBuf::from,
    );
    let mut cursor: &Path = &manifest_dir;
    loop {
        let candidate = cursor.join("Cargo.toml");
        if candidate.is_file() {
            if let Ok(contents) = std::fs::read_to_string(&candidate) {
                if contents.contains("[workspace]") {
                    return cursor.to_path_buf();
                }
            }
        }
        match cursor.parent() {
            Some(parent) => cursor = parent,
            None => return manifest_dir,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_help_returns_success() {
        let mut out: Vec<u8> = Vec::new();
        let mut err: Vec<u8> = Vec::new();
        let code = run(&["help".to_string()], &mut out, &mut err);
        assert_eq!(code, ExitCode::SUCCESS);
        let text = String::from_utf8(out).unwrap_or_default();
        assert!(text.contains("xtask"));
    }

    #[test]
    fn run_long_help_returns_success() {
        let mut out: Vec<u8> = Vec::new();
        let mut err: Vec<u8> = Vec::new();
        let code = run(&["--help".to_string()], &mut out, &mut err);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn run_short_help_returns_success() {
        let mut out: Vec<u8> = Vec::new();
        let mut err: Vec<u8> = Vec::new();
        let code = run(&["-h".to_string()], &mut out, &mut err);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn run_unknown_subcommand_returns_usage_error() {
        let mut out: Vec<u8> = Vec::new();
        let mut err: Vec<u8> = Vec::new();
        let code = run(&["bogus".to_string()], &mut out, &mut err);
        assert_eq!(code, ExitCode::from(2));
        let text = String::from_utf8(err).unwrap_or_default();
        assert!(text.contains("unknown subcommand"));
    }

    #[test]
    fn workspace_root_finds_workspace_marker() {
        let root = workspace_root();
        let manifest = root.join("Cargo.toml");
        assert!(
            manifest.is_file(),
            "expected workspace Cargo.toml at {root:?}"
        );
        let body = std::fs::read_to_string(&manifest).unwrap_or_default();
        assert!(body.contains("[workspace]"));
    }

    #[test]
    fn run_default_subcommand_against_workspace_succeeds() {
        // Use the actual live workspace; should be clean.
        let mut out: Vec<u8> = Vec::new();
        let mut err: Vec<u8> = Vec::new();
        let code = run(&[], &mut out, &mut err);
        assert_eq!(
            code,
            ExitCode::SUCCESS,
            "stderr: {}",
            String::from_utf8(err).unwrap_or_default()
        );
    }

    #[test]
    fn run_explicit_subcommand_against_workspace_succeeds() {
        let mut out: Vec<u8> = Vec::new();
        let mut err: Vec<u8> = Vec::new();
        let code = run(&["check-error-codes".to_string()], &mut out, &mut err);
        assert_eq!(
            code,
            ExitCode::SUCCESS,
            "stderr: {}",
            String::from_utf8(err).unwrap_or_default()
        );
    }

    #[test]
    fn run_check_sentinel_codes_against_workspace_succeeds() {
        let mut out: Vec<u8> = Vec::new();
        let mut err: Vec<u8> = Vec::new();
        let code = run(&["check-sentinel-codes".to_string()], &mut out, &mut err);
        assert_eq!(
            code,
            ExitCode::SUCCESS,
            "stderr: {}",
            String::from_utf8(err).unwrap_or_default()
        );
        let out_text = String::from_utf8(out).unwrap_or_default();
        assert!(out_text.contains("check-sentinel-codes:"));
    }

    fn write_min_workspace(root: &Path) {
        std::fs::write(root.join("Cargo.toml"), "[workspace]\n").unwrap();
        let error_rs = root.join("crates").join("stratum-types").join("src");
        std::fs::create_dir_all(&error_rs).unwrap();
        std::fs::write(
            error_rs.join("error.rs"),
            "pub const E1001_F: ErrorCode = ErrorCode::new_static(\"STRAT-E1001\");\n",
        )
        .unwrap();
    }

    #[test]
    fn dispatch_at_reports_orphans_to_stderr() {
        // Orphan: catalog declares STRAT-E1001 but nothing else references it.
        let tmp = tempfile::TempDir::new().unwrap();
        write_min_workspace(tmp.path());
        let mut out: Vec<u8> = Vec::new();
        let mut err: Vec<u8> = Vec::new();
        let code = dispatch_check_error_codes_at(tmp.path(), &mut out, &mut err);
        assert!(code.is_err());
        let err_text = String::from_utf8(err).unwrap_or_default();
        assert!(
            err_text.contains("catalog codes with no references"),
            "stderr was: {err_text}",
        );
    }

    #[test]
    fn dispatch_at_reports_unknown_to_stderr() {
        let tmp = tempfile::TempDir::new().unwrap();
        write_min_workspace(tmp.path());
        // Plant a foreign code AND a reference to E1001 so the orphan path
        // is not also triggered.
        let user = tmp.path().join("crates").join("u").join("src");
        std::fs::create_dir_all(&user).unwrap();
        std::fs::write(user.join("lib.rs"), "// STRAT-E1001 STRAT-E9999\n").unwrap();
        let mut out: Vec<u8> = Vec::new();
        let mut err: Vec<u8> = Vec::new();
        let code = dispatch_check_error_codes_at(tmp.path(), &mut out, &mut err);
        assert!(code.is_err());
        let err_text = String::from_utf8(err).unwrap_or_default();
        assert!(
            err_text.contains("undeclared STRAT-E#### codes"),
            "stderr was: {err_text}",
        );
    }

    #[test]
    fn dispatch_at_reports_run_error_to_stderr() {
        // Workspace without the catalog file triggers MissingCatalog.
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::write(tmp.path().join("Cargo.toml"), "[workspace]\n").unwrap();
        let mut out: Vec<u8> = Vec::new();
        let mut err: Vec<u8> = Vec::new();
        let code = dispatch_check_error_codes_at(tmp.path(), &mut out, &mut err);
        assert!(code.is_err());
        let err_text = String::from_utf8(err).unwrap_or_default();
        assert!(
            err_text.contains("check-error-codes failed"),
            "stderr was: {err_text}",
        );
    }

    #[test]
    fn dispatch_sentinel_at_reports_unknown_to_stderr() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::write(tmp.path().join("Cargo.toml"), "[workspace]\n").unwrap();
        let src = tmp.path().join("crates").join("u").join("src");
        std::fs::create_dir_all(&src).unwrap();
        // Plant every allowlist entry so orphans are empty, plus an
        // unrecognised sentinel so unknown is non-empty.
        let mut body = String::from("fn main() {\n");
        for entry in check_sentinel_codes::ALLOWLIST {
            use std::fmt::Write as _;
            writeln!(body, "    let _ = \"{entry}\";").unwrap();
        }
        body.push_str("    let _ = \"E_FAKE_SENTINEL\";\n}\n");
        std::fs::write(src.join("lib.rs"), body).unwrap();
        let mut out: Vec<u8> = Vec::new();
        let mut err: Vec<u8> = Vec::new();
        let code = dispatch_check_sentinel_codes_at(tmp.path(), &mut out, &mut err);
        assert!(code.is_err());
        let err_text = String::from_utf8(err).unwrap_or_default();
        assert!(
            err_text.contains("sentinel literals not in the xtask allowlist"),
            "stderr was: {err_text}",
        );
    }

    #[test]
    fn dispatch_sentinel_at_reports_orphans_to_stderr() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::write(tmp.path().join("Cargo.toml"), "[workspace]\n").unwrap();
        // Empty crates dir: every allowlist entry is an orphan.
        let mut out: Vec<u8> = Vec::new();
        let mut err: Vec<u8> = Vec::new();
        let code = dispatch_check_sentinel_codes_at(tmp.path(), &mut out, &mut err);
        assert!(code.is_err());
        let err_text = String::from_utf8(err).unwrap_or_default();
        assert!(
            err_text.contains("allowlisted sentinels with no references"),
            "stderr was: {err_text}",
        );
    }

    #[test]
    fn dispatch_sentinel_at_succeeds_for_clean_fixture() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::write(tmp.path().join("Cargo.toml"), "[workspace]\n").unwrap();
        let src = tmp.path().join("crates").join("u").join("src");
        std::fs::create_dir_all(&src).unwrap();
        let mut body = String::from("fn main() {\n");
        for entry in check_sentinel_codes::ALLOWLIST {
            use std::fmt::Write as _;
            writeln!(body, "    let _ = \"{entry}\";").unwrap();
        }
        body.push_str("}\n");
        std::fs::write(src.join("lib.rs"), body).unwrap();
        let mut out: Vec<u8> = Vec::new();
        let mut err: Vec<u8> = Vec::new();
        let code = dispatch_check_sentinel_codes_at(tmp.path(), &mut out, &mut err);
        assert!(
            code.is_ok(),
            "stderr: {}",
            String::from_utf8(err).unwrap_or_default()
        );
        let out_text = String::from_utf8(out).unwrap_or_default();
        assert!(out_text.contains("check-sentinel-codes:"));
    }

    #[test]
    fn dispatch_at_succeeds_for_clean_fixture() {
        let tmp = tempfile::TempDir::new().unwrap();
        write_min_workspace(tmp.path());
        let user = tmp.path().join("crates").join("u").join("src");
        std::fs::create_dir_all(&user).unwrap();
        std::fs::write(user.join("lib.rs"), "// STRAT-E1001\n").unwrap();
        let mut out: Vec<u8> = Vec::new();
        let mut err: Vec<u8> = Vec::new();
        let code = dispatch_check_error_codes_at(tmp.path(), &mut out, &mut err);
        assert!(
            code.is_ok(),
            "stderr: {}",
            String::from_utf8(err).unwrap_or_default()
        );
        let out_text = String::from_utf8(out).unwrap_or_default();
        assert!(out_text.contains("check-error-codes:"));
    }
}
