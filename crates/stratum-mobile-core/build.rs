//! Build script for `stratum-mobile-core`.
//!
//! On iOS targets, regenerate `include/stratum_mobile.h` from the crate's
//! public C-ABI surface using `cbindgen`. On every other target (host
//! builds, Android, tests in CI) the build script is a no-op so a normal
//! `cargo test --workspace` doesn't pay the price of bindgen.
//!
//! The generated header is checked into the repo so that the Swift
//! Package's `publicHeadersPath` resolves without anyone having to run
//! the build script first.

use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=cbindgen.toml");
    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-env-changed=STRATUM_FORCE_CBINDGEN");

    // Only regenerate on iOS targets, or when explicitly forced. This
    // keeps host / Android / CI builds fast and self-contained.
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let forced = env::var("STRATUM_FORCE_CBINDGEN").is_ok();
    if target_os != "ios" && !forced {
        return;
    }

    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    let crate_dir_path = PathBuf::from(&crate_dir);
    let out_header = crate_dir_path.join("include").join("stratum_mobile.h");

    let config =
        cbindgen::Config::from_file(crate_dir_path.join("cbindgen.toml")).unwrap_or_else(|err| {
            // cbindgen returns a typed error; surface it but keep building
            // — a missing header on host builds is recoverable.
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
