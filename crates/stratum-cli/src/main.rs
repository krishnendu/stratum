//! Stratum CLI entry point.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod app;
mod chat;

use std::process::ExitCode;

use stratum_runtime::logging::{self, LoggingConfig};

fn main() -> ExitCode {
    // Best-effort: a logging-init failure should not block the CLI; the
    // chosen log dir is `None` by default so the only failure mode is a
    // doubly-initialized subscriber, which the wrapper silently ignores.
    let _ = logging::init(&LoggingConfig::default());
    app::run(std::env::args_os().skip(1).collect())
}
