//! `bench-floor` — deterministic nightly regression bench bin.
//!
//! CLI:
//!
//! ```text
//! bench-floor --evals-dir <dir> --out <result.json>
//! ```
//!
//! Loads every `*.json` / `*.toml` suite under `--evals-dir`, runs each
//! through the `EchoProvider` floor, and writes a single result JSON.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![allow(
    clippy::print_stdout,
    clippy::print_stderr,
    reason = "this is a CLI binary; user-facing output is allowed"
)]

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use stratum_eval::bench_floor::{discover_suites, run_bench, write_result};

#[derive(Debug, Parser)]
#[command(
    name = "bench-floor",
    about = "Run the deterministic Echo-backed eval suites and write a result JSON."
)]
struct Args {
    /// Directory containing `evals/<suite>.json` / `.toml` files.
    #[arg(long, default_value = "evals")]
    evals_dir: PathBuf,
    /// Output path for the bench result JSON.
    #[arg(long)]
    out: PathBuf,
}

fn main() -> ExitCode {
    let args = Args::parse();
    let suites = match discover_suites(&args.evals_dir) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("bench-floor: discover failed: {e}");
            return ExitCode::from(74);
        }
    };
    if suites.is_empty() {
        eprintln!(
            "bench-floor: no suites found under {}",
            args.evals_dir.display()
        );
        return ExitCode::from(66);
    }
    let result = match run_bench(&suites) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("bench-floor: run failed: {e}");
            return ExitCode::from(70);
        }
    };
    if let Err(e) = write_result(&result, &args.out) {
        eprintln!("bench-floor: write failed: {e}");
        return ExitCode::from(74);
    }
    println!(
        "bench-floor: wrote {} suite(s) to {}",
        result.suites.len(),
        args.out.display()
    );
    ExitCode::SUCCESS
}
