//! Build script for `stratum-mobile-core`.
//!
//! Two responsibilities:
//!
//! 1. **uniffi-rs scaffolding (Android JNI path).** Run uniffi's scaffolding
//!    generator against `src/stratum.udl` so `uniffi::include_scaffolding!`
//!    in `src/lib.rs` resolves. Runs unconditionally — the generated Rust
//!    shims are needed for the in-workspace `cargo test` build too.
//!
//! 2. **cbindgen header regeneration (iOS Swift Package path).** On iOS
//!    targets (or when `STRATUM_FORCE_CBINDGEN=1`), regenerate
//!    `include/stratum_mobile.h` from the crate's public C-ABI surface.
//!    The header is also checked into the repo so a Swift Package consumer
//!    does not have to run this build script first to get a usable header.
//!
//! Errors are surfaced via `cargo:warning=` (visible in normal cargo
//! output) and an explicit stderr write (fallback when warnings are
//! captured). Workspace lints deny `panic` / `unwrap` / `expect` outside
//! `#[cfg(test)]`, so build-script panics are avoided.

use std::env;
use std::io::Write as _;
use std::path::PathBuf;
use std::process::ExitCode;

fn main() -> ExitCode {
    println!("cargo:rerun-if-changed=src/stratum.udl");
    println!("cargo:rerun-if-changed=cbindgen.toml");
    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=STRATUM_FORCE_CBINDGEN");

    // (1) uniffi — always runs. Failure is fatal: the generated scaffolding
    // is `include!`'d from `src/lib.rs` and a missing file would surface as
    // a cryptic compile error later.
    if let Err(err) = uniffi::generate_scaffolding("src/stratum.udl") {
        println!("cargo:warning=uniffi scaffolding generation failed: {err}");
        let mut stderr = std::io::stderr().lock();
        let _ = writeln!(
            stderr,
            "stratum-mobile-core build.rs: uniffi scaffolding generation failed: {err}"
        );
        return ExitCode::FAILURE;
    }

    // (2) cbindgen — iOS-only, or on demand. Header missing on host builds
    // is recoverable (the in-tree checked-in header satisfies the Swift
    // Package); a regen failure here is therefore a warning, not a hard
    // error.
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let forced = env::var("STRATUM_FORCE_CBINDGEN").is_ok();
    if target_os == "ios" || forced {
        let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
        let crate_dir_path = PathBuf::from(&crate_dir);
        let out_header = crate_dir_path.join("include").join("stratum_mobile.h");

        let config = cbindgen::Config::from_file(crate_dir_path.join("cbindgen.toml"))
            .unwrap_or_else(|err| {
                println!("cargo:warning=cbindgen config error: {err}");
                cbindgen::Config::default()
            });

        match cbindgen::Builder::new()
            .with_crate(&crate_dir)
            .with_config(config)
            .generate()
        {
            Ok(bindings) => {
                if !bindings.write_to_file(&out_header) {
                    println!(
                        "cargo:warning=cbindgen did not rewrite {} (no changes)",
                        out_header.display()
                    );
                }
            }
            Err(err) => {
                println!("cargo:warning=cbindgen failed to generate header: {err}");
            }
        }
    }

    ExitCode::SUCCESS
}
