//! Mobile FFI surface for Stratum (Phase 8).
//!
//! This crate is the boundary between the Rust runtime and the host platform
//! (Android / iOS). It exposes a deliberately tiny C-ABI surface, generated
//! via [`uniffi`], so the Kotlin / Swift side can `loadLibrary` and call in
//! without hand-written JNI.
//!
//! The full runtime wiring lands in `feat/phase-8-mobile-core`; this crate
//! currently only ships two stub entry points so the bindings generator has
//! a non-empty surface to introspect against.

#![allow(
    missing_docs,
    reason = "uniffi's `include_scaffolding!` expands to extern `C` shims \
              and helper types whose names we don't author and can't doc."
)]
#![allow(
    unsafe_code,
    reason = "uniffi's generated scaffolding emits `#[no_mangle]` extern \
              functions for the C ABI surface the host (Android / iOS) calls. \
              No hand-written `unsafe` blocks live in this crate."
)]
#![allow(
    clippy::missing_const_for_fn,
    clippy::empty_line_after_doc_comments,
    reason = "uniffi's `include_scaffolding!`-generated code (e.g. the \
              `uniffi_*_checksum_func_*` shims) triggers these lints. The \
              generated file is in `target/`, so we cannot annotate it \
              locally — the allow has to live at the crate root."
)]

/// Initialize the mobile runtime.
///
/// Returns `0` on success. Real initialization (logging, config dir
/// resolution, backend probe) is implemented in `feat/phase-8-mobile-core`.
#[must_use]
pub fn mobile_init() -> u32 {
    0
}

/// Returns the crate version string.
///
/// This is the workspace version (currently `"1.0.1"`) and is used by the
/// host app to sanity-check that the bundled `lib*.so` matches the Kotlin
/// bindings it was generated against.
#[must_use]
pub fn mobile_version() -> String {
    env!("CARGO_PKG_VERSION").to_owned()
}

uniffi::include_scaffolding!("stratum");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_returns_zero() {
        assert_eq!(mobile_init(), 0);
    }

    #[test]
    fn version_is_workspace_version() {
        // Tracks the workspace `version` in the root `Cargo.toml`.
        assert_eq!(mobile_version(), env!("CARGO_PKG_VERSION"));
    }
}
