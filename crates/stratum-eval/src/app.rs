//! Eval-harness CLI scaffold.
//!
//! Phase 2 v2 surface is `stratum-eval list` (empty task list) and
//! `stratum-eval run` (no-op success). The structured exit code +
//! output channels are pinned now so Phase 7 can fill in the runner
//! without churning every test fixture.

use std::ffi::OsString;
use std::io::Write;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use serde::Serialize;

/// Stratum evaluation harness (scaffold).
#[derive(Debug, Parser)]
#[command(
    name = "stratum-eval",
    version,
    about = "Stratum eval harness scaffold (Phase 2)"
)]
struct Cli {
    /// Emit machine-readable JSON instead of human prose where applicable.
    #[arg(long, global = true)]
    json: bool,

    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Print the catalog of evaluation tasks (empty in the Phase 2 scaffold).
    List,
    /// Run the evaluation suite (no-op success in the Phase 2 scaffold).
    Run,
}

#[derive(Debug, Serialize)]
struct ListReport<'a> {
    schema_version: u32,
    tasks: &'a [&'static str],
}

#[derive(Debug, Serialize)]
struct RunReport {
    schema_version: u32,
    status: &'static str,
    completed: u32,
    total: u32,
}

/// Run the CLI against the provided argv (excluding argv[0]).
#[must_use]
#[allow(
    clippy::redundant_pub_crate,
    reason = "intentional: visible to the bin crate root only"
)]
pub(super) fn run(argv: Vec<OsString>) -> ExitCode {
    run_with(argv, &mut std::io::stdout(), &mut std::io::stderr())
}

#[must_use]
fn run_with(argv: Vec<OsString>, out: &mut dyn Write, err: &mut dyn Write) -> ExitCode {
    let mut full = vec![OsString::from("stratum-eval")];
    full.extend(argv);
    let cli = match Cli::try_parse_from(full) {
        Ok(c) => c,
        Err(e) => {
            let _ = writeln!(err, "{e}");
            return ExitCode::from(64);
        }
    };
    match cli.command.unwrap_or(Command::List) {
        Command::List => list_tasks(cli.json, out),
        Command::Run => run_tasks(cli.json, out),
    }
}

fn list_tasks(json: bool, out: &mut dyn Write) -> ExitCode {
    let report = ListReport {
        schema_version: 1,
        tasks: &[],
    };
    if json {
        #[allow(
            clippy::expect_used,
            reason = "ListReport serialization is infallible (primitives only)"
        )]
        let rendered =
            serde_json::to_string_pretty(&report).expect("ListReport serialization is infallible");
        if writeln!(out, "{rendered}").is_err() {
            return ExitCode::from(74);
        }
    } else if writeln!(
        out,
        "(no eval tasks defined yet; Phase 7 populates the suite)"
    )
    .is_err()
    {
        return ExitCode::from(74);
    }
    ExitCode::SUCCESS
}

fn run_tasks(json: bool, out: &mut dyn Write) -> ExitCode {
    let report = RunReport {
        schema_version: 1,
        status: "scaffold",
        completed: 0,
        total: 0,
    };
    if json {
        #[allow(
            clippy::expect_used,
            reason = "RunReport serialization is infallible (primitives only)"
        )]
        let rendered =
            serde_json::to_string_pretty(&report).expect("RunReport serialization is infallible");
        if writeln!(out, "{rendered}").is_err() {
            return ExitCode::from(74);
        }
    } else if writeln!(out, "stratum-eval: scaffold run, 0/0 tasks executed").is_err() {
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
    fn default_command_is_list_prose() {
        let (code, out, _err) = drive(&[]);
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::SUCCESS));
        assert!(out.contains("no eval tasks"));
    }

    #[test]
    fn list_json_emits_empty_array() {
        let (_code, out, _err) = drive(&["--json", "list"]);
        let v: serde_json::Value = serde_json::from_str(out.trim()).unwrap();
        assert_eq!(v["schema_version"], 1);
        assert_eq!(v["tasks"].as_array().unwrap().len(), 0);
    }

    #[test]
    fn run_prose_reports_scaffold() {
        let (code, out, _err) = drive(&["run"]);
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::SUCCESS));
        assert!(out.contains("scaffold run"));
    }

    #[test]
    fn run_json_reports_scaffold() {
        let (_code, out, _err) = drive(&["--json", "run"]);
        let v: serde_json::Value = serde_json::from_str(out.trim()).unwrap();
        assert_eq!(v["status"], "scaffold");
        assert_eq!(v["completed"], 0);
        assert_eq!(v["total"], 0);
    }

    #[test]
    fn unknown_subcommand_exits_64() {
        let (code, _out, err) = drive(&["wat"]);
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::from(64)));
        assert!(!err.is_empty());
    }

    #[test]
    fn help_flag_exits_64() {
        let (_code, _out, err) = drive(&["--help"]);
        assert!(err.to_lowercase().contains("stratum-eval"));
    }

    /// Writer that always errors — covers the IO-failure branches.
    struct FailingWriter;
    impl Write for FailingWriter {
        fn write(&mut self, _buf: &[u8]) -> std::io::Result<usize> {
            Err(std::io::Error::other("forced failure for coverage test"))
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Err(std::io::Error::other("forced failure for coverage test"))
        }
    }

    fn drive_failing(cli_args: &[&str]) -> ExitCode {
        let mut fail = FailingWriter;
        let mut err = Vec::new();
        let argv: Vec<OsString> = cli_args.iter().map(OsString::from).collect();
        run_with(argv, &mut fail, &mut err)
    }

    #[test]
    fn list_prose_io_failure_returns_74() {
        assert_eq!(
            format!("{:?}", drive_failing(&[])),
            format!("{:?}", ExitCode::from(74))
        );
    }

    #[test]
    fn list_json_io_failure_returns_74() {
        assert_eq!(
            format!("{:?}", drive_failing(&["--json", "list"])),
            format!("{:?}", ExitCode::from(74))
        );
    }

    #[test]
    fn run_prose_io_failure_returns_74() {
        assert_eq!(
            format!("{:?}", drive_failing(&["run"])),
            format!("{:?}", ExitCode::from(74))
        );
    }

    #[test]
    fn run_json_io_failure_returns_74() {
        assert_eq!(
            format!("{:?}", drive_failing(&["--json", "run"])),
            format!("{:?}", ExitCode::from(74))
        );
    }

    #[test]
    fn failing_writer_flush_errors() {
        let mut fail = FailingWriter;
        assert!(fail.flush().is_err());
    }
}
