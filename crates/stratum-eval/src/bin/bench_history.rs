//! `bench-history` — append a bench-floor result to a versioned JSONL
//! time-series and refresh `latest.json`.
//!
//! CLI:
//!
//! ```text
//! bench-history --result <result.json> --history-dir <dir> [--timestamp-unix N]
//! ```
//!
//! The timestamp on the appended row defaults to the result file's
//! mtime — never `SystemTime::now()` — so re-runs against the same
//! artifact are reproducible.
//!
//! The bin does NOT git-commit the history change. That is the
//! workflow's responsibility (see `.github/workflows/bench-floor.yml`).

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![allow(
    clippy::print_stdout,
    clippy::print_stderr,
    reason = "this is a CLI binary; user-facing output is allowed"
)]

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::{Duration, SystemTime};

use clap::Parser;
use stratum_eval::bench_history::{append_history, mtime_of};

#[derive(Debug, Parser)]
#[command(
    name = "bench-history",
    about = "Append a bench-floor result JSON to a versioned history dir."
)]
struct Args {
    /// Path to the result JSON produced by `bench-floor`.
    #[arg(long)]
    result: PathBuf,
    /// History directory; defaults to `.github/bench-history`.
    #[arg(long, default_value = ".github/bench-history")]
    history_dir: PathBuf,
    /// Override the timestamp used to bucket the row. Defaults to the
    /// result file's mtime — pass this only for reproducible test runs.
    #[arg(long)]
    timestamp_unix: Option<u64>,
}

fn main() -> ExitCode {
    let args = Args::parse();

    let recorded_at = if let Some(unix) = args.timestamp_unix {
        SystemTime::UNIX_EPOCH + Duration::from_secs(unix)
    } else {
        match mtime_of(&args.result) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("bench-history: mtime failed: {e}");
                return ExitCode::from(74);
            }
        }
    };

    match append_history(&args.result, &args.history_dir, recorded_at) {
        Ok(entry) => {
            println!(
                "bench-history: appended row for {} to {}",
                entry.recorded_at_date,
                args.history_dir.display()
            );
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("bench-history: append failed: {e}");
            ExitCode::from(70)
        }
    }
}
