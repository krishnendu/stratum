//! Stratum CLI entry point.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod app;
mod chat;

use std::process::ExitCode;

fn main() -> ExitCode {
    app::run(std::env::args_os().skip(1).collect())
}
