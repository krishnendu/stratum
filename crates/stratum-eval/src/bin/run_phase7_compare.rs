//! `run-phase7-compare` — Phase 7 comparison runner.
//!
//! CLI:
//!
//! ```text
//! run-phase7-compare [--target stratum] [--model echo] [--out result.json] [--markdown]
//! ```
//!
//! Runs the canonical 5-task Phase 7 set against a stratum
//! `AgentLoop`. The default build wires the deterministic
//! `EchoProvider`, which is enough to smoke-test the harness end-to-end
//! and is the path the unit tests cover. A follow-up patch swaps the
//! Echo loop for a real provider (e.g. `stratum-llama-cpp`) without
//! touching the runner.
//!
//! Output:
//!
//! * stdout: JSON (default) or Markdown table (`--markdown`).
//! * `--out <path>`: also write the JSON report atomically to disk.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![allow(
    clippy::print_stdout,
    clippy::print_stderr,
    reason = "this is a CLI binary; user-facing output is allowed"
)]

use std::io::{self, Write};
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

use clap::Parser;
use stratum_eval::phase7_compare::{
    phase7_tasks, render_markdown_table, run_compare, write_report, Phase7CompareError,
    Phase7CompareReport,
};
use stratum_runtime::agent_factory::AgentFactory;
use stratum_types::ModelId;

#[derive(Debug, Parser)]
#[command(
    name = "run-phase7-compare",
    about = "Run the canonical Phase 7 task set and print a comparison report."
)]
struct Args {
    /// Target label to embed in the report (e.g. `stratum`, `ollama`,
    /// `claude-code`).
    #[arg(long, default_value = "stratum")]
    target: String,
    /// Model id to embed in the report; defaults to `echo` for the
    /// wiring smoke-test build.
    #[arg(long, default_value = "echo")]
    model: String,
    /// Optional path to write the JSON report to (atomic `.tmp` +
    /// rename). When omitted, only stdout is written.
    #[arg(long)]
    out: Option<PathBuf>,
    /// Print the Markdown table instead of JSON.
    #[arg(long)]
    markdown: bool,
}

fn main() -> ExitCode {
    let args = Args::parse();
    run(&args, &mut io::stdout(), &mut io::stderr())
}

fn run<O: Write, E: Write>(args: &Args, stdout: &mut O, stderr: &mut E) -> ExitCode {
    let report = match build_and_run(&args.target, &args.model) {
        Ok(r) => r,
        Err(e) => {
            let _ = writeln!(stderr, "run-phase7-compare: {e}");
            return ExitCode::from(70);
        }
    };

    if let Some(path) = &args.out {
        if let Err(e) = write_report(&report, path) {
            let _ = writeln!(stderr, "run-phase7-compare: write failed: {e}");
            return ExitCode::from(74);
        }
    }

    if args.markdown {
        let md = render_markdown_table(&report);
        if let Err(e) = stdout.write_all(md.as_bytes()) {
            let _ = writeln!(stderr, "run-phase7-compare: write failed: {e}");
            return ExitCode::from(74);
        }
    } else {
        let rendered = match serde_json::to_string_pretty(&report) {
            Ok(s) => s,
            Err(e) => {
                let _ = writeln!(stderr, "run-phase7-compare: serialize failed: {e}");
                return ExitCode::from(74);
            }
        };
        if let Err(e) = writeln!(stdout, "{rendered}") {
            let _ = writeln!(stderr, "run-phase7-compare: write failed: {e}");
            return ExitCode::from(74);
        }
    }
    ExitCode::SUCCESS
}

fn build_and_run(target: &str, model: &str) -> Result<Phase7CompareReport, Phase7CompareError> {
    let loop_ = Arc::new(AgentFactory::echo().map_err(Phase7CompareError::Factory)?);
    Ok(run_compare(
        loop_,
        &ModelId::from(model),
        target,
        &phase7_tasks(),
    ))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    fn default_args() -> Args {
        Args {
            target: "stratum".into(),
            model: "echo".into(),
            out: None,
            markdown: false,
        }
    }

    #[test]
    fn build_and_run_echo_returns_five_tasks() {
        let r = build_and_run("stratum", "echo").unwrap();
        assert_eq!(r.tasks.len(), 5);
        assert_eq!(r.target, "stratum");
        assert_eq!(r.model, "echo");
    }

    #[test]
    fn run_json_prints_to_stdout_and_returns_success() {
        let args = default_args();
        let mut out: Vec<u8> = Vec::new();
        let mut err: Vec<u8> = Vec::new();
        let code = run(&args, &mut out, &mut err);
        assert_eq!(code, ExitCode::SUCCESS);
        let s = std::str::from_utf8(&out).unwrap();
        assert!(s.contains("\"schema_version\""));
        assert!(s.contains("\"target\": \"stratum\""));
        assert!(err.is_empty(), "stderr should be empty on success");
    }

    #[test]
    fn run_markdown_prints_table_to_stdout() {
        let args = Args {
            markdown: true,
            ..default_args()
        };
        let mut out: Vec<u8> = Vec::new();
        let mut err: Vec<u8> = Vec::new();
        let code = run(&args, &mut out, &mut err);
        assert_eq!(code, ExitCode::SUCCESS);
        let s = std::str::from_utf8(&out).unwrap();
        assert!(s.contains("Summary:"));
        assert!(s.contains("| task |"));
    }

    #[test]
    fn run_writes_json_when_out_is_set() {
        let dir = tempfile::tempdir().unwrap();
        let out_path = dir.path().join("report.json");
        let args = Args {
            out: Some(out_path.clone()),
            ..default_args()
        };
        let mut out: Vec<u8> = Vec::new();
        let mut err: Vec<u8> = Vec::new();
        let code = run(&args, &mut out, &mut err);
        assert_eq!(code, ExitCode::SUCCESS);
        let raw = std::fs::read(&out_path).unwrap();
        let back: Phase7CompareReport = serde_json::from_slice(&raw).unwrap();
        assert_eq!(back.tasks.len(), 5);
    }

    #[test]
    fn run_returns_74_when_out_path_is_invalid() {
        // Path with no file_name component (`..`) is rejected by write_report.
        let dir = tempfile::tempdir().unwrap();
        let bad = dir.path().join("..");
        let args = Args {
            out: Some(bad),
            ..default_args()
        };
        let mut out: Vec<u8> = Vec::new();
        let mut err: Vec<u8> = Vec::new();
        let code = run(&args, &mut out, &mut err);
        assert_eq!(code, ExitCode::from(74));
        let e = std::str::from_utf8(&err).unwrap();
        assert!(e.contains("write failed"));
    }

    #[test]
    fn args_parse_uses_defaults() {
        let parsed = Args::try_parse_from(["run-phase7-compare"]).unwrap();
        assert_eq!(parsed.target, "stratum");
        assert_eq!(parsed.model, "echo");
        assert!(parsed.out.is_none());
        assert!(!parsed.markdown);
    }
}
