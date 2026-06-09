# Verification Gates

Stratum's CI enforces 12 per-PR gates. Each is also documented in the project's private design corpus; this is the public-facing summary.

## The 12 gates

| # | Gate | Mechanism |
|---|---|---|
| **G1** | Builds clean on all CI targets | `cargo build --workspace --all-targets` on macOS arm64 + Linux x86_64 (Windows lands in Phase 4). The off-by-default `provider-llama-cpp` feature runs on nightly cron + release tag + manual dispatch via [`.github/workflows/provider-llama-cpp.yml`](../.github/workflows/provider-llama-cpp.yml); not gated on per-PR CI. See [`llama-cpp-build.md`](llama-cpp-build.md). |
| **G2** | Tests pass | `cargo test --workspace --all-features`. Sandbox tests: bwrap (Linux) is installed in CI via `apt-get install bubblewrap` in the `test` job so sandbox-spawn tests run live instead of skip-on-no-helper; sandbox-exec (macOS) is part of the OS. |
| **G2.1** | **≥ 98% line coverage** | `cargo llvm-cov --workspace --all-features --fail-under-lines 98`. The 100% target lives in [`coverage-exclusions.md`](coverage-exclusions.md); the gap is a small set of documented carve-outs (infallible-by-our-shape serialization, TTY-bound TUI init paths, platform-conditional CPU-feature branches). **Enforcement note:** the coverage gate is enforced via a dedicated `cargo llvm-cov --workspace --fail-under-lines 98` invocation (no `--lcov`), with the LCOV artifact emitted in a follow-up `cargo llvm-cov --workspace --lcov --output-path lcov.info` step. Some versions of `cargo-llvm-cov` have silently skipped `--fail-under-lines` enforcement when combined with `--lcov --output-path`; splitting the two invocations and guarding with `set -euxo pipefail` makes the gate strict. See the `coverage` job in [`.github/workflows/ci.yml`](../.github/workflows/ci.yml). |
| **G2.2** | **Integration tests for every public surface** | `tests/` cover each CLI subcommand, each TUI flow (via `ratatui::backend::TestBackend`), each provider method, each tool, each sandbox profile, each `STRAT-Exxxx` error path. |
| **G3** | Lints clean | `cargo fmt --check` + `cargo clippy --workspace --all-targets --all-features -- -D warnings` (pedantic + nursery + cargo groups enabled). |
| **G4** | No `unsafe`, no `unwrap`/`expect`/`panic` outside tests | Workspace lint `unsafe_code = "deny"`; clippy denies `unwrap_used` / `expect_used` / `panic` / `dbg_macro` / `print_stdout` / `print_stderr` outside `#[cfg(test)]`. Documented carve-outs only. |
| **G5** | Errors carry a `STRAT-Exxxx` code | Every new `Err` path uses a constant from [`crates/stratum-types/src/error.rs::codes`](https://github.com/krishnendu/stratum/blob/main/crates/stratum-types/src/error.rs). Adding a new code requires a doc entry in the same PR. |
| **G6** | Supply chain audit | `cargo deny check`: license allow-list, RustSec advisory DB, no unknown registries / git sources. |
| **G7** | Public docs updated when applicable | New public API → entry in `docs/` or a `TODO(docs)` annotation for the next phase-close graduation pass. |
| **G8** | Conventional commit + DCO sign-off | Subject matches `^(feat\|fix\|chore\|docs\|refactor\|test\|perf\|build\|ci\|revert)(\(.+\))?: `, body ends with `Signed-off-by:`. **No AI co-author trailers.** |
| **G9** | Security review when applicable | Tool / sandbox / secret / network / FFI / `unsafe` surfaces → reviewer ticks the checklist. |
| **G10** | TUI snapshot tests on UI changes | `ratatui::backend::TestBackend` snapshots for any rendering path. |
| **G11** | Schema migration tested | On-disk schema bumps ship with a migration step + an N-1 fixture in CI. The first such target is [`docs/schemas/doctor.v1.json`](schemas/doctor.v1.json), validated against the live `stratum doctor --json` output by the `doctor-schema-check` CI job (see [`docs/schemas/README.md`](schemas/README.md)). |
| **G12** | Network / telemetry impact disclosed | If a change introduces an outbound socket the PR description names the endpoint and the flag that gates it. |

## What CI blocks

All of G1, G2, G2.1, G2.2, G3, G4, G5, G6 are blocking on `main`. G7 / G10 / G11 are blocking conditionally (only when the relevant surface changes). G9 / G12 are reviewer-checklist items.

## Local pre-flight

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
cargo llvm-cov --workspace --all-features --fail-under-lines 98
cargo deny check
```

## PR checklist

Every PR description ends with the verification block from
[`.github/PULL_REQUEST_TEMPLATE.md`](../.github/PULL_REQUEST_TEMPLATE.md). Each gate is either ticked or marked `N/A: <reason>`. Reviewer enforces.
