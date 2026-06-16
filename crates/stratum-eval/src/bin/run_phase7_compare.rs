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
    let report = match build_and_run(&args.target, &args.model) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("run-phase7-compare: {e}");
            return ExitCode::from(70);
        }
    };

    if let Some(path) = &args.out {
        if let Err(e) = write_report(&report, path) {
            eprintln!("run-phase7-compare: write failed: {e}");
            return ExitCode::from(74);
        }
    }

    if args.markdown {
        let md = render_markdown_table(&report);
        print!("{md}");
    } else {
        let rendered = match serde_json::to_string_pretty(&report) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("run-phase7-compare: serialize failed: {e}");
                return ExitCode::from(74);
            }
        };
        println!("{rendered}");
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
