//! Build script that runs uniffi's scaffolding generator against
//! `src/stratum.udl`. The generated Rust shims are then pulled into
//! `src/lib.rs` via `uniffi::include_scaffolding!("stratum")`.
//!
//! Errors are reported to the cargo log via stderr writes (not panics):
//! a build-script panic produces a noisy backtrace that obscures the actual
//! UDL parse error, and the workspace lints already deny `panic!` / `unwrap`
//! / `expect` outside `#[cfg(test)]`. We use `cargo:warning=` so the message
//! surfaces in the normal cargo output even when stderr is captured.

use std::io::Write as _;
use std::process::ExitCode;

fn main() -> ExitCode {
    println!("cargo:rerun-if-changed=src/stratum.udl");
    println!("cargo:rerun-if-changed=build.rs");

    match uniffi::generate_scaffolding("src/stratum.udl") {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            // `cargo:warning=` makes the message visible in `cargo build`
            // output; the explicit stderr write is the fallback channel in
            // case cargo's warning capture is disabled (rare, but happens
            // under some IDE integrations).
            println!("cargo:warning=uniffi scaffolding generation failed: {err}");
            let mut stderr = std::io::stderr().lock();
            let _ = writeln!(
                stderr,
                "stratum-mobile-core build.rs: uniffi scaffolding generation failed: {err}"
            );
            ExitCode::FAILURE
        }
    }
}
