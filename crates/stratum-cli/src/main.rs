//! Stratum CLI entry point.
//!
//! Phase 0 surface: a smoke binary that proves the workspace compiles and
//! the typed error / tracing primitives are wired up. Real subcommands land
//! in Phase 1.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod app;

use std::process::ExitCode;

fn main() -> ExitCode {
    app::run(std::env::args_os().skip(1).collect())
}
