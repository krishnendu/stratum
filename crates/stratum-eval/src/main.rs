//! Stratum evaluation harness — Phase 2 scaffold.
//!
//! Phase 2 v2 ships an empty driver that proves the workspace structure;
//! the actual 50-task suite, judge transport (`claude -p` subprocess
//! per `plan/10-eval-and-bench.md`), and tier-aware pass thresholds land
//! in Phase 7.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod app;

use std::process::ExitCode;

fn main() -> ExitCode {
    app::run(std::env::args_os().skip(1).collect())
}
