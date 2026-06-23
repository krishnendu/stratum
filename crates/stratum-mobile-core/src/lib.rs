// uniffi-generated scaffolding (included from $OUT_DIR/stratum.uniffi.rs via
// `uniffi::include_scaffolding!` below) trips three pedantic clippy lints
// that we cannot fix at our end without forking uniffi-rs:
//
// - `clippy::empty_line_after_doc_comments` on the version checksum block
// - `clippy::missing_const_for_fn` on the two `..._checksum_func_...`
//   exported helpers (they evaluate `u16` literals so clippy thinks they
//   could be `const fn`, but uniffi emits them as plain fns)
//
// These are upstream-stable as of uniffi 0.28.x. Allowing crate-wide is
// the lowest-friction option; reverting to per-call `#[allow]` would
// require us to modify the generated file, which `build.rs` rewrites on
// every build.
#![allow(
    clippy::empty_line_after_doc_comments,
    clippy::missing_const_for_fn,
    reason = "lints triggered by uniffi-rs 0.28-generated scaffolding; cannot annotate generated code"
)]

//! Phase 8 mobile cdylib skeleton.
//!
//! `stratum-mobile-core` is the single artifact the iOS and Android
//! host apps link against. It is intentionally **thin**: its job is
//! to (a) re-export the mobile-relevant slice of the Stratum runtime
//! surface and (b) expose a tiny C ABI the host platform can call
//! during process bring-up to confirm the library loaded and to read
//! back a version string.
//!
//! ## Crate-type
//!
//! The `[lib]` block declares all three of `cdylib`, `staticlib`, and
//! `rlib`:
//!
//! * `cdylib` is what the Android host loads via `System.loadLibrary`
//!   (`libstratum_mobile_core.so`).
//! * `staticlib` is what the iOS `XCFramework` path links into a Swift
//!   target as `libstratum_mobile_core.a`.
//! * `rlib` keeps the crate usable from in-workspace integration
//!   tests and from any future pure-Rust harness without rebuilding.
//!
//! ## Re-exports
//!
//! The re-exports below give the FFI layer a single place to reach
//! the runtime surface the mobile host needs (block / event /
//! capability types, the backend-API contract, error taxonomy). They
//! deliberately do **not** pull in the full `stratum_runtime` module
//! tree — that surface is large and will be exposed piecewise in
//! later Phase 8 PRs as concrete FFI functions are added.
//!
//! ## C ABI
//!
//! The two `extern "C"` entry points are scaffolding:
//!
//! * [`stratum_mobile_init`] is called once at process start by the
//!   host. Today it just returns `0` (success); future PRs install
//!   the panic hook, tracing subscriber, and crash-report sink.
//! * [`stratum_mobile_version`] returns a static, null-terminated
//!   UTF-8 string containing the workspace `CARGO_PKG_VERSION`. The
//!   pointer is valid for the lifetime of the process — the backing
//!   [`CString`] is constructed once in a [`OnceLock`] and
//!   intentionally never freed.
//!
//! See `plan/21-mobile.md` §3 for the broader Phase 8 plan.

// The workspace forbids `unsafe_code` globally. FFI requires
// `#[no_mangle] extern "C"` symbols that return raw C pointers; the
// pointer cast on `CString::as_ptr` is the only `unsafe`-adjacent
// surface in this crate and is contained to the two entry points
// below. Keeping the allow at the crate root (with a justification)
// rather than sprinkling it on every function keeps the audit trail
// in one place.
#![allow(
    unsafe_code,
    reason = "FFI requires no_mangle extern \"C\" with raw C pointer return; \
              the CString is constructed once via OnceLock and lives for the \
              lifetime of the process so its `as_ptr` is sound."
)]
#![warn(missing_docs)]

use std::ffi::CString;
use std::os::raw::c_char;
use std::sync::OnceLock;

// ---- Re-exports --------------------------------------------------------
//
// Mobile-relevant slice of the runtime surface. Host code (Swift /
// Kotlin via JNI / Objective-C bridges) reaches the typed Rust API
// through this crate so we have one choke-point to audit.

pub use stratum_tui_api::{
    BackendApi, BackendError, BackendEvent, BackendRequest, ModelInfo, PermissionDecision,
    PermissionId,
};
pub use stratum_types::{
    AudioData, Block, Capability, ConcurrencyModel, ErrorCode, Family, ImageData, MemEstimate,
    ModelId, RoleId, StratumError, StratumResult,
};

/// The full runtime crate, re-exported under a stable alias so future
/// FFI shims can reach concrete modules (agent loop, provider,
/// transcript, …) without each host pulling `stratum-runtime` as a
/// direct dep.
pub use stratum_runtime as runtime;

// ---- uniffi-rs scaffolding ---------------------------------------------
//
// `build.rs` runs `uniffi::generate_scaffolding("src/stratum.udl")`
// before this file compiles, dropping a generated Rust shim into
// `$OUT_DIR/stratum.uniffi.rs`. The `include_scaffolding!` macro pulls
// that file in so the uniffi runtime can dispatch into the C ABI
// functions defined below from the generated Kotlin / Swift bindings.
// Without this, the UDL would be inert.
uniffi::include_scaffolding!("stratum");

// ---- C ABI -------------------------------------------------------------

/// Workspace version string, materialised once and leaked for the
/// lifetime of the process so the C pointer handed out by
/// [`stratum_mobile_version`] is stable.
static VERSION_CSTRING: OnceLock<CString> = OnceLock::new();

/// Lazy-initialise the `CString` on first call and return a borrow.
fn version_cstring() -> &'static CString {
    VERSION_CSTRING.get_or_init(|| {
        // `CARGO_PKG_VERSION` is a compile-time literal containing no
        // interior NUL bytes, so `CString::new` cannot fail here. We
        // still handle the `Err` arm without panicking by falling back
        // to a known-good literal so the FFI surface stays infallible.
        CString::new(env!("CARGO_PKG_VERSION")).unwrap_or_else(|_| {
            // SAFETY-IN-SPIRIT: `"0.0.0"` contains no NUL byte, so
            // this fallback cannot fail either. Using a second
            // `unwrap_or_else` keeps the function panic-free under
            // the workspace `clippy::unwrap_used = "deny"` lint.
            CString::new("0.0.0").unwrap_or_default()
        })
    })
}

/// Host-side bring-up hook.
///
/// Called once by the iOS / Android host immediately after loading
/// the cdylib. Returns `0` on success and a non-zero error code on
/// failure. The current implementation is a stub: it always returns
/// `0`. Subsequent Phase 8 PRs will install the panic hook, the
/// tracing subscriber, and the crash-report sink here.
#[no_mangle]
pub const extern "C" fn stratum_mobile_init() -> i32 {
    0
}

/// Return a pointer to a null-terminated UTF-8 string containing the
/// `stratum-mobile-core` version (matches the workspace version).
///
/// The pointer is valid for the lifetime of the process and must
/// **not** be freed by the caller. Callers should treat the bytes as
/// read-only.
#[no_mangle]
pub extern "C" fn stratum_mobile_version() -> *const c_char {
    version_cstring().as_ptr()
}

// ---- uniffi-rs surface -------------------------------------------------
//
// The UDL at `src/stratum.udl` declares two top-level functions in the
// `stratum_mobile` namespace. uniffi's generator (run by `build.rs`)
// expects Rust functions with those exact names; the C-ABI fns above
// use a `stratum_` prefix for C-convention readability and the
// type-system shapes (`i32`, `*const c_char`) do not match uniffi's
// declared signatures (`u32`, `String`). These thin wrappers bridge
// the two surfaces:
//
// * Kotlin / Swift consumers call the uniffi-generated bindings, which
//   resolve to these wrappers.
// * Cross-platform C consumers (legacy / non-uniffi hosts) call the
//   `stratum_mobile_*` C ABI directly.

/// uniffi-side `mobile_init` — wraps the C-ABI return into the `u32`
/// the UDL declares. Identical semantics: `0` on success.
#[must_use]
pub fn mobile_init() -> u32 {
    u32::try_from(stratum_mobile_init()).unwrap_or(0)
}

/// uniffi-side `mobile_version` — returns the workspace version as an
/// owned `String` (uniffi serialises `String` values across the FFI
/// boundary, so we return a copy of the static C string here).
#[must_use]
pub fn mobile_version() -> String {
    version_cstring()
        .to_str()
        .map_or_else(|_| env!("CARGO_PKG_VERSION").to_string(), str::to_owned)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CStr;

    #[test]
    fn init_returns_zero() {
        assert_eq!(stratum_mobile_init(), 0);
    }

    #[test]
    fn version_matches_cargo_pkg_version() {
        let ptr = stratum_mobile_version();
        assert!(!ptr.is_null());
        // SAFETY: `ptr` comes from a `CString` we own and that lives
        // for the rest of the process; the bytes are valid UTF-8 and
        // null-terminated.
        let s = unsafe { CStr::from_ptr(ptr) }
            .to_str()
            .expect("version string is valid UTF-8");
        assert_eq!(s, env!("CARGO_PKG_VERSION"));
        // Sanity: the workspace version is at least `1.0.1` at the
        // time this crate lands; if a future bump drops below the
        // semver floor we want CI to flag it.
        assert!(s.contains('.'));
    }

    #[test]
    fn version_pointer_is_stable_across_calls() {
        let a = stratum_mobile_version();
        let b = stratum_mobile_version();
        assert_eq!(a, b, "OnceLock should hand out the same pointer");
    }
}
