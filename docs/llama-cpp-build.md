# `provider-llama-cpp` build

Stratum's llama.cpp-backed provider lives behind an off-by-default workspace
feature flag, `provider-llama-cpp`. The feature pulls in
[`llama-cpp-2`](https://crates.io/crates/llama-cpp-2) (pinned at `0.1` in
`Cargo.toml`'s `[workspace.dependencies]`) which in turn compiles the
upstream `llama.cpp` C++ tree via `cmake`. On a cold cache that build runs
for multiple minutes per CI runner.

To keep per-PR CI fast we keep this build OFF the critical path:

> **Default CI does not build `llama-cpp`.** Release tags, the nightly cron,
> and manual dispatch run it via `.github/workflows/provider-llama-cpp.yml`
> instead.

## When the workflow runs

The `.github/workflows/provider-llama-cpp.yml` job triggers on:

- `workflow_dispatch` — a maintainer asks for it from the Actions tab or the
  `gh` CLI.
- `schedule` — every day at 06:00 UTC, so regressions surface within 24 h.
- `release` (`types: [created]`) — every tagged release builds with the
  feature on before the artifact is published.

It runs a `build-and-test` matrix on `ubuntu-latest` + `macos-latest` with
a 60-minute timeout, installs `cmake` + `build-essential` (Ubuntu) /
`cmake` (macOS), then runs `cargo build` and `cargo test` with
`--features provider-llama-cpp`.

## Manual dispatch

```bash
gh workflow run provider-llama-cpp.yml
gh run watch
```

You can also pick a non-default ref:

```bash
gh workflow run provider-llama-cpp.yml --ref my-feature-branch
```

## What the feature flag activates

- Adds the `llama-cpp-2` crate (optional dep on `stratum-runtime`) to the
  build graph.
- Compiles `crates/stratum-runtime/src/llama_provider.rs`, currently a
  compile-only stub exposing `LlamaCppProvider`. The real provider lands
  in a follow-up PR.
- Re-exports `LlamaCppProvider` from `stratum_runtime`'s prelude.
- Mirrors the feature on `stratum-cli` so `cargo build -p stratum-cli
  --features provider-llama-cpp` works end-to-end.

## Why it is OFF by default

- `llama-cpp-2` invokes `cmake` to build `llama.cpp` from source. Cold
  build is multi-minute and the toolchain pulls in C/C++ build deps that
  per-PR CI runners would otherwise not need.
- The provider has no consumers yet — flipping it on would slow every PR
  for a code path nobody runs.
- Branch-protection status checks intentionally do not include this
  workflow; promoting it would force every contributor to wait on a slow
  build for a feature they did not touch.

When the provider graduates from scaffold to real implementation we will
revisit this decision and consider gating release artifacts on a green
`provider-llama-cpp` run.
