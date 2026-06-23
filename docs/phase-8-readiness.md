# Phase 8 readiness audits

Per `plan/21` §12, the cleanest entry into Phase 8 (mobile: iOS + Android)
requires:

  1. The runtime crates compile for `aarch64-apple-ios` and
     `aarch64-linux-android`.
  2. The `Paths` abstraction is the sole gateway for platform path lookups.

This document records the results of both audits on `main` as of
the audit run, the small fixes applied in the same PR, and the
remaining Phase 8 entry checklist.

## Audit 1 — Cross-compile

Both targets installed cleanly via `rustup target add`:

```
aarch64-apple-darwin
aarch64-apple-ios          <-- newly installed
aarch64-linux-android      <-- newly installed
aarch64-unknown-linux-gnu
x86_64-apple-darwin
x86_64-unknown-linux-gnu
```

### `stratum-types`

| Target                  | Result | Notes                                       |
| ----------------------- | ------ | ------------------------------------------- |
| `aarch64-apple-ios`     | clean  | 4.50s, no warnings.                         |
| `aarch64-linux-android` | clean  | 1.29s, no warnings.                         |

`stratum-types` is pure Rust (serde, thiserror, zmij). It is mobile-ready
as-is — both targets pass `cargo check` with zero warnings on a host with
no Xcode or NDK installed.

### `stratum-runtime`

| Target                  | Result        | Blocker                                  |
| ----------------------- | ------------- | ---------------------------------------- |
| `aarch64-apple-ios`     | hard failure  | `ring` native build needs iPhoneOS SDK   |
| `aarch64-linux-android` | hard failure  | `ring` native build needs Android NDK    |

Both failures bottom out at the same transitive native build:

```
ring v0.17.14
  └── rustls v0.23.40 (--features tls)
      └── ureq v2.12.1
          └── stratum-runtime
```

The Rust source compiles fine. The build script for `ring` invokes the
platform C toolchain (`xcrun --sdk iphoneos` on iOS,
`aarch64-linux-android-clang` on Android) and aborts when those aren't
present.

#### What's needed (per target)

**iOS (`aarch64-apple-ios`):**
- Install full Xcode (not just Command Line Tools).
- `sudo xcode-select -s /Applications/Xcode.app/Contents/Developer`.
- Verify with `xcrun --sdk iphoneos --show-sdk-path` — must return a
  real path under `Platforms/iPhoneOS.platform/Developer/SDKs/`.
- After that, `cargo check -p stratum-runtime --target aarch64-apple-ios`
  is expected to succeed (no Rust source changes needed).

**Android (`aarch64-linux-android`):**
- Install the Android NDK (r26+ recommended for 16 KB page-size on
  Android 15/16).
- Set `ANDROID_NDK_HOME=$NDK_ROOT`.
- Add to `PATH`: `$NDK_ROOT/toolchains/llvm/prebuilt/<host>/bin`.
- Add to `~/.cargo/config.toml`:
  ```toml
  [target.aarch64-linux-android]
  linker = "aarch64-linux-android21-clang"
  ar     = "llvm-ar"
  ```
  with matching `CC_aarch64_linux_android` env var (or use `cargo-ndk`).
- After that, `cargo check -p stratum-runtime --target aarch64-linux-android`
  is expected to succeed.

#### Should we drop `ring` from the runtime?

Considered and rejected for this PR:

- `ring` is only pulled in because `ureq` is configured with the `tls`
  feature in the root `Cargo.toml`. Switching to `native-tls` would
  replace `ring` with the platform TLS stack (SecureTransport on iOS,
  BoringSSL via NDK on Android), which also requires those same
  toolchains — no net win for "build without Xcode / NDK".
- Removing TLS from `ureq` entirely is a non-starter — model downloads
  and catalog sync require HTTPS.
- The right fix is to install the toolchains. The cross-compile *will*
  work once they're present; the Rust source is mobile-clean.

### `stratum-eval`

| Target                  | Result        | Blocker                                  |
| ----------------------- | ------------- | ---------------------------------------- |
| `aarch64-apple-ios`     | hard failure  | same `ring` build (transitive via runtime) |
| `aarch64-linux-android` | hard failure  | same `ring` build (transitive via runtime) |

Same root cause as `stratum-runtime`. No additional iOS/Android-specific
deps. Same fix applies — install the platform C toolchains.

## Audit 2 — Paths abstraction completeness

Grepped the workspace for any path lookup that bypasses `Paths`:

```
$ grep -rn 'std::env::var.*"HOME"\|dirs::home_dir\|dirs::cache_dir\|\
            dirs::config_dir\|dirs::data_dir' crates/ --include='*.rs'
$ grep -rn 'std::env::temp_dir' crates/ --include='*.rs'
```

### Legitimate uses

- `crates/stratum-runtime/src/paths.rs:36-39` — the `Paths` struct itself
  calling `dirs::{config,data,state,cache}_dir()`. By design.
- `crates/stratum-runtime/src/memory_loader.rs:550` — comment, not code.
- `crates/stratum-runtime/src/sandbox.rs:936` — inside a
  `#[cfg(test)] mod tests {}` block.
- `crates/stratum-runtime/src/tool_dispatchers.rs:1764` — reads
  `std::env::var_os("HOME")` to **forward** to a child `git` process'
  environment, not to resolve a path. Intentional: with `env_clear()`,
  git would otherwise lose `HOME` and break credential helpers, SSH
  agent, signed commits. Production-correct; leave as-is. On iOS/Android
  this code path runs only inside an exec sandbox that's already off
  for mobile (git is desktop-only).

### Production bypasses (real smells)

| Location | What it does | Recommended Phase 8 fix |
| -------- | ------------ | ----------------------- |
| `stratum-runtime/src/memory_loader.rs:251` | Expands `~/` in user-authored STRATUM.md import paths via `dirs::home_dir()`. | Phase 8: route through `Paths` (mobile has no `HOME` in the desktop sense; map to the app's data dir or refuse `~/` imports). Single call site; easy switch when `Paths` gains a `home()` method. |
| `stratum-runtime/src/sandbox.rs:411` (`unique_temp_profile_path`) | Writes a macOS Seatbelt profile blob under `std::env::temp_dir()`. | macOS-only path; on iOS/Android the Seatbelt backend is unused. Still cleaner to take `&Paths` and write under `paths.cache.join("sandbox")`. Small refactor; defer to Phase 8.0. |
| `stratum-runtime/src/whisper.rs:180` | Uses `std::env::temp_dir()` for whisper.cpp transcription output. | Whisper isn't on the v1 mobile path. Either gate it `#[cfg(not(any(target_os="ios", target_os="android")))]` or thread `Paths`. Defer. |
| `stratum-tui/src/chat.rs:2659` (`/export` default) | Default export path under `std::env::temp_dir()`. | TUI doesn't ship on mobile in Phase 8 (mobile is a separate frontend). Acceptable to leave in `stratum-tui` for now. |
| `stratum-tui/src/chat.rs:5073` (`/edit` external editor) | Tmp file for `$EDITOR` round-trip. | Same as above — TUI-only, mobile irrelevant. |

### Summary

The runtime has **two** real bypasses to clean before mobile:
`memory_loader.rs:251` and `sandbox.rs:411` (the second is macOS-only
so functionally a no-op on mobile, but worth migrating for hygiene).
TUI bypasses don't affect mobile because the TUI crate isn't part of
the mobile frontend.

No bypasses were fixed in this PR — each needs a small `Paths` API
extension (a `home()` accessor and/or a `sandbox_dir()` accessor) and
is best done together at the top of Phase 8 once the mobile `Paths`
implementation is decided.

## What landed in the Phase 8 groundwork sweep

The following scaffolding PRs were opened together on 2026-06-23/24 to
make the Phase 8 entry tractable. They are scaffolds — real wiring lands
in follow-ups gated by toolchain availability and tracked in the issue
linked at the bottom of this document.

- **PR #212 — `feat(phase-8): consolidated mobile-core + uniffi + cbindgen scaffolding`** —
  Single consolidation PR that supersedes the three originally-spawned
  workflow PRs (#205, #207, #208), which all created
  `crates/stratum-mobile-core/` in isolated worktrees and so could not
  merge independently. #212 contains:
  - the canonical `stratum-mobile-core` cdylib + staticlib + rlib
    skeleton with `mobile_init` / `mobile_version` FFI stubs and the
    four `mobile-ultra-low` / `mobile-low` / `mobile-medium` /
    `mobile-high` tier feature flags,
  - the cbindgen iOS path (cbindgen config + `build.rs` emitting
    `include/stratum_mobile.h`, checked-in header, and the
    `platforms/ios/StratumMobileCore` SwiftPM package with module map),
  - the uniffi-rs Android path (`src/stratum.udl` + `build.rs` calling
    `uniffi::generate_scaffolding`, plus thin Rust wrappers bridging
    the UDL bare names to the `stratum_mobile_*` C-ABI symbols).
- **PR #209 — `feat(phase-8): mobile tier ladder + bundle TOMLs`** —
  Extends `Tier` with `MobileUltraLow` / `MobileLow` / `MobileMedium` /
  `MobileHigh`, adds an `is_mobile()` helper, makes `classify` default
  to `MobileMedium` on `target_os = "ios" | "android"`, and ships the
  four matching `assets/bundles/mobile-*-v1.toml` files.

## Phase 8 entry checklist

Before a mobile build can start, this is the remaining work:

- [ ] **Toolchains installed locally / in CI**
  - [ ] Full Xcode with iPhoneOS SDK on the macOS CI runner.
  - [ ] Android NDK r26+ on Linux + macOS CI runners.
  - [ ] `cargo-ndk` (or equivalent linker wiring in `.cargo/config.toml`).
- [ ] **Cross-compile gates**
  - [ ] `cargo check -p stratum-runtime --target aarch64-apple-ios`
        added to a `mobile-cross-compile` CI job (allowed to be
        opt-in / nightly, not per-PR).
  - [ ] Same for `aarch64-linux-android`.
  - [ ] `stratum-types` cross-compile already green; add a smoke job.
- [ ] **`Paths` API for mobile**
  - [ ] Decide the mobile path layout (iOS: `Documents/`,
        `Library/Caches/`; Android: `getFilesDir`, `getCacheDir`).
  - [ ] Add `Paths::resolve_mobile(...)` constructor that takes the
        roots from JNI / FFI (the dirs crate's iOS/Android defaults
        are not always what an app wants).
  - [ ] Add `Paths::home()` accessor (returns `Option<PathBuf>`,
        `None` on mobile).
- [ ] **Clean up the two production bypasses**
  - [ ] `memory_loader.rs:251` → use `Paths::home()`, gracefully
        refuse `~/` imports when `None`.
  - [ ] `sandbox.rs:411` → take `&Paths`, write under
        `paths.cache.join("sandbox")` (macOS-only code, but
        consistency).
- [ ] **Gate desktop-only features**
  - [ ] Audit `stratum-runtime` for `tokio` / `notify` / `keyring` /
        `tiny_http` / `sysinfo` deps that may not link on iOS without
        feature work. (Not done in this PR; they currently compile
        as far as the `ring` failure, so this audit was inconclusive.)
  - [ ] Re-run Audit 1 after the toolchains are present; resolve any
        new failures that surface past the `ring` build.
- [x] **TUI mobile decision** — confirmed in the groundwork sweep:
      `stratum-tui` is desktop-only and is not part of the Phase 8
      mobile frontend. The mobile frontend goes through
      `stratum-mobile-core` (scaffolded — PR #212).
- [x] **`stratum-mobile-core` cdylib carved out** — scaffolded —
      PR #212. The crate is in the workspace with the right
      crate-types and tier feature flags; real runtime wiring is the
      follow-up.
- [x] **FFI bindings generators chosen and scaffolded** — scaffolded —
      PR #212 (cbindgen → iOS Swift Package + uniffi-rs → Android JNI).
      UDL is the source of truth for Kotlin / Swift; cbindgen is kept
      for the raw-C Swift bridge.
- [x] **Mobile tier ladder defined** — scaffolded — PR #209.
      `Tier::Mobile{UltraLow,Low,Medium,High}` plus the four bundle
      TOMLs are in tree. Device-class detection and bundle-resolver
      wiring are the follow-up.

## Follow-up tracking

Phase 8's remaining work needs hardware + IDE (Xcode, Android Studio,
real iPhone / Pixel / Samsung / Xiaomi devices, Apple Developer Program
enrollment, store submission flows) that can't be driven from a headless
shell. That work is tracked in the GitHub issue
**"Phase 8 (mobile clients) — work that requires real hardware"** —
maintainers with the right setup pick items off of it as they become
unblocked.

## Fixes applied in this PR

None — both audits surfaced issues that need decisions (toolchain
provisioning, `Paths` API shape) rather than mechanical edits. The
deliverable is the inventory and the checklist; the source tree is
unchanged.
