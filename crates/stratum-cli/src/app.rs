//! The CLI behavior, factored out of `main` for testability.

use std::ffi::OsString;
use std::io::Write;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use serde::Serialize;
use stratum_types::ErrorCode;

/// Stratum CLI.
#[derive(Debug, Parser)]
#[command(name = "stratum", version, about = "Stratum local-LLM TUI agent")]
struct Cli {
    /// Emit machine-readable JSON instead of human prose where applicable.
    #[arg(long, global = true)]
    json: bool,

    #[command(subcommand)]
    command: Option<Command>,
}

/// Top-level subcommands. Phase 0 ships only `doctor` and the implicit
/// `hello` default; the rest land in later phases per `plan/07-…`.
#[derive(Debug, Subcommand)]
enum Command {
    /// Probe the host and print a tier report. Phase 0 stub.
    Doctor,
}

/// Run the CLI against the provided argv (excluding argv[0]).
/// Separated from `main` so integration tests can drive it without spawning.
#[must_use]
#[allow(
    clippy::redundant_pub_crate,
    reason = "intentional: visible to the bin crate root only"
)]
pub(super) fn run(argv: Vec<OsString>) -> ExitCode {
    run_with(argv, &mut std::io::stdout(), &mut std::io::stderr())
}

/// Drive the CLI with caller-supplied stdout/stderr handles.
#[must_use]
fn run_with(argv: Vec<OsString>, out: &mut dyn Write, err: &mut dyn Write) -> ExitCode {
    let mut full = vec![OsString::from("stratum")];
    full.extend(argv);
    let cli = match Cli::try_parse_from(full) {
        Ok(c) => c,
        Err(e) => {
            // clap's writer expects an io::Write; surface via the err handle.
            let _ = writeln!(err, "{e}");
            return ExitCode::from(64);
        }
    };

    match cli.command {
        None => {
            let _ = writeln!(out, "hello, tier=unknown");
            ExitCode::SUCCESS
        }
        Some(Command::Doctor) => doctor(cli.json, out),
    }
}

#[derive(Debug, Serialize)]
struct DoctorReport {
    schema_version: u32,
    stratum_version: &'static str,
    tier: &'static str,
    issues: Vec<DoctorIssue>,
}

#[derive(Debug, Serialize)]
struct DoctorIssue {
    code: ErrorCode,
    level: &'static str,
    message: &'static str,
}

fn doctor(json: bool, out: &mut dyn Write) -> ExitCode {
    let report = DoctorReport {
        schema_version: 1,
        stratum_version: env!("CARGO_PKG_VERSION"),
        tier: "unknown",
        issues: vec![DoctorIssue {
            code: stratum_types::error::codes::E2003_TIER_DOWNGRADE_REFUSED,
            level: "info",
            message: "phase 0 stub: hardware probe not yet implemented",
        }],
    };

    if json {
        // DoctorReport is plain owned data; `to_string_pretty` is infallible here.
        // `expect_used` is denied workspace-wide; carve-out documented in
        // `docs/coverage-exclusions.md`.
        #[allow(
            clippy::expect_used,
            reason = "DoctorReport serialization is infallible (primitives only)"
        )]
        let rendered = serde_json::to_string_pretty(&report)
            .expect("DoctorReport serialization is infallible");
        if writeln!(out, "{rendered}").is_err() {
            return ExitCode::from(74);
        }
    } else if writeln!(
        out,
        "stratum {} · tier=unknown · phase 0 stub",
        report.stratum_version
    )
    .is_err()
    {
        return ExitCode::from(74);
    }
    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;

    fn drive(cli_args: &[&str]) -> (ExitCode, String, String) {
        let mut out = Vec::new();
        let mut err = Vec::new();
        let argv: Vec<OsString> = cli_args.iter().map(OsString::from).collect();
        let code = run_with(argv, &mut out, &mut err);
        (
            code,
            String::from_utf8(out).unwrap(),
            String::from_utf8(err).unwrap(),
        )
    }

    #[test]
    fn default_prints_hello() {
        let (code, out, err) = drive(&[]);
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::SUCCESS));
        assert!(out.contains("hello, tier=unknown"));
        assert!(err.is_empty());
    }

    #[test]
    fn doctor_prose() {
        let (_code, out, _err) = drive(&["doctor"]);
        assert!(out.contains("phase 0 stub"));
        assert!(out.contains("tier=unknown"));
    }

    #[test]
    fn doctor_json_is_valid() {
        let (_code, out, _err) = drive(&["--json", "doctor"]);
        let parsed: serde_json::Value = serde_json::from_str(out.trim()).unwrap();
        assert_eq!(parsed["schema_version"], 1);
        assert_eq!(parsed["tier"], "unknown");
        assert!(parsed["issues"].is_array());
    }

    #[test]
    fn unknown_subcommand_exits_64() {
        let (code, _out, err) = drive(&["wat"]);
        assert!(!err.is_empty(), "clap should write to err");
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::from(64)));
    }

    #[test]
    fn help_flag_exits_64_with_message() {
        // clap's `--help` is reported as an Error in `try_parse_from`; surfaced as exit 64.
        let (_code, _out, err) = drive(&["--help"]);
        assert!(err.to_lowercase().contains("usage") || err.to_lowercase().contains("stratum"));
    }

    /// Writer that always returns an error. Used to exercise the IO-failure
    /// branches of `doctor()`.
    struct FailingWriter;

    impl Write for FailingWriter {
        fn write(&mut self, _buf: &[u8]) -> std::io::Result<usize> {
            Err(std::io::Error::other("forced failure for coverage test"))
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Err(std::io::Error::other("forced failure for coverage test"))
        }
    }

    #[test]
    fn doctor_prose_io_failure_returns_74() {
        let mut fail = FailingWriter;
        let mut err = Vec::new();
        let argv: Vec<OsString> = [OsString::from("doctor")].to_vec();
        let code = run_with(argv, &mut fail, &mut err);
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::from(74)));
    }

    #[test]
    fn doctor_json_io_failure_returns_74() {
        let mut fail = FailingWriter;
        let mut err = Vec::new();
        let argv: Vec<OsString> = [OsString::from("--json"), OsString::from("doctor")].to_vec();
        let code = run_with(argv, &mut fail, &mut err);
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::from(74)));
    }

    #[test]
    fn failing_writer_flush_errors() {
        // Covers the `flush` impl of FailingWriter for full line coverage.
        let mut fail = FailingWriter;
        assert!(fail.flush().is_err());
    }
}
