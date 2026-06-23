# Phase 8 — iOS build flow (cbindgen + Swift Package)

This document describes how the `stratum-mobile-core` crate is built and
consumed as a Swift Package on iOS. It is the cbindgen-based sibling of
the uniffi scaffold (see `feat/phase-8-mobile-core`).

## Layout

```
crates/stratum-mobile-core/
  Cargo.toml          # cdylib + staticlib + lib, cbindgen build-dep
  build.rs            # iOS-gated cbindgen header generation
  cbindgen.toml       # C, include guard STRATUM_MOBILE_H, pragma once
  src/lib.rs          # safe `api::*` + tiny `extern "C"` shim
  include/
    stratum_mobile.h  # cbindgen-generated header (checked in)

platforms/ios/StratumMobileCore/
  Package.swift                            # SwiftPM manifest, iOS 18+
  Sources/StratumMobileCore/
    StratumMobileCore.swift                # public Swift surface
    include/
      module.modulemap                     # bridges C symbols → Swift
      stratum_mobile.h                     # copy of the cbindgen header
```

The header lives in **two** places on purpose: the canonical copy is in
`crates/stratum-mobile-core/include/`, where the build script writes it;
the Swift Package keeps its own copy under `Sources/.../include/` so the
package resolves with no prior Rust build (and because SwiftPM doesn't
follow symlinks portably).

A future PR will replace the duplicate with a single source of truth
(either a symlink set up by `xtask`, or `cp` triggered by the iOS build
job in CI). Until then: regenerate via the build script and copy.

## Building the dylib for iOS

```bash
# One-time toolchain setup.
rustup target add aarch64-apple-ios aarch64-apple-ios-sim x86_64-apple-ios

# Device build (real iPhone, aarch64).
cargo build -p stratum-mobile-core --release --target aarch64-apple-ios

# Simulator build (Apple Silicon Mac).
cargo build -p stratum-mobile-core --release --target aarch64-apple-ios-sim
```

The cdylib will land at
`target/aarch64-apple-ios/release/libstratum_mobile_core.dylib`, which
is exactly the path `Package.swift` points its `binaryTarget` at.

## Regenerating the C header

The build script regenerates `crates/stratum-mobile-core/include/stratum_mobile.h`
whenever `CARGO_CFG_TARGET_OS == "ios"`. To force regeneration on a host
build (e.g. when reviewing a PR locally), set:

```bash
STRATUM_FORCE_CBINDGEN=1 cargo build -p stratum-mobile-core
```

After regeneration, copy the canonical header into the Swift Package:

```bash
cp crates/stratum-mobile-core/include/stratum_mobile.h \
   platforms/ios/StratumMobileCore/Sources/StratumMobileCore/include/
```

## Consuming the Swift Package

In Xcode, **File → Add Packages… → Add Local…** and pick the
`platforms/ios/StratumMobileCore` directory. Then:

```swift
import StratumMobileCore

print(StratumMobile.version)  // → "1.0.1"
```

## Why a separate crate from `stratum-runtime`?

`stratum-runtime` transitively depends on `ring` via `rustls`/`ureq`,
which requires the iPhone SDK to cross-compile (see
`docs/phase-8-readiness.md`). Keeping the FFI surface in its own crate
lets the iOS scaffold compile on a bare host and keeps the C ABI
intentionally narrow.

## Why cbindgen *and* uniffi?

Two scaffolds, two purposes:

* **uniffi** (`feat/phase-8-mobile-core`) — owns the *typed* Swift /
  Kotlin surface where we want async, errors, records, enums, etc.
  generated for us.
* **cbindgen** (this PR) — owns the *raw C ABI* used by tooling and
  shared-library consumers (Android NDK, third-party Swift packages,
  any future C/C++ embed). It is the lowest-common-denominator surface.

The two are not redundant — they sit on opposite sides of the FFI
ergonomics tradeoff and we will keep both.

## Conflict note

This PR and the sibling `feat/phase-8-mobile-core` (uniffi scaffold) both
create `crates/stratum-mobile-core/` and both register it in the workspace
root `Cargo.toml`. Whichever lands first wins; the second PR will need a
merge that unifies `Cargo.toml`'s `[dependencies]` (add `uniffi` and
`cbindgen` as build-deps), keeps the `crate-type = ["lib", "cdylib",
"staticlib"]`, and combines both `build.rs` flows.
