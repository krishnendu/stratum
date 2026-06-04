# Stratum

[![CI](https://github.com/krishnendu/stratum/actions/workflows/ci.yml/badge.svg)](https://github.com/krishnendu/stratum/actions/workflows/ci.yml)
[![License: Apache-2.0 OR MIT](https://img.shields.io/badge/license-Apache--2.0%20OR%20MIT-blue.svg)](#license)

> **Status:** pre-alpha. Phase 1 of the build is in progress — the binary boots, probes hardware, classifies into a tier, installs model files (local + HTTP, SHA-256 verified), and drives an echo-backed TUI chat loop. Real LLM inference (`LlamaCppProvider`) is the next deliverable. The runtime is not yet useful for daily work; the API surface is unstable.

Stratum is a local, multi-LLM agentic TUI agent written in **Rust**, designed for the **8-16 GB laptop class** (recommended 12 GB+, optimal 16 GB+). The goal is **best-in-class local agentic outcomes** — entirely offline — by composing a crew of small/selective-load models instead of a single monolith.

It is explicitly **not** a Claude Code clone, not a single-model wrapper, and not GPU-required.

## What works today

```bash
$ cargo run --quiet --bin stratum -- doctor
stratum 0.0.0 · tier=high · gpu=metal · ram=24576 MiB · cores=10 · installed=false

$ cargo run --quiet --bin stratum -- init
installed · tier=high · gpu=metal · wrote ~/Library/.../installed.toml

$ cargo run --quiet --bin stratum -- echo "hello stratum"
echo: hello
echo: stratum
(usage: prompt=2 completion=2)
(done)

$ cargo run --quiet --bin stratum -- models add \
    --from-url https://example.com/model.gguf \
    --sha256 <hex>
installed · …/model.gguf · N bytes · verified=true

$ cargo run --quiet --bin stratum -- chat       # ratatui TUI; Esc/Ctrl-C exit
```

`stratum doctor`, `stratum init`, `stratum echo`, `stratum chat`, and `stratum models {list, add}` are wired and tested. See [`docs/`](docs/) for surface documentation and [`plan/07-implementation-phases.md`](plan/07-implementation-phases.md) (private, gitignored) for the full roadmap.

## Architecture at a glance

- **Resident tier (~6.7 GB hot):** Gemma 4 E4B + E2B (spec-dec draft), Qwen3-0.6B (caveman comms + polisher), Arctic-Embed-L MRL (router + RAG embeddings).
- **Swap tier (one at a time, dense 7B):** Qwen2.5-Coder-7B (coder) and DeepSeek-R1-Distill-Qwen-7B (thinker) on the high tier; DeepSeek-R1-Distill-Qwen-1.5B on medium.
- **Backend:** single embedded `llama-cpp-2` for LLM inference; `candle` for the embedder; subprocess `whisper.cpp` / `piper` for voice.
- **TUI:** `ratatui` + `crossterm`.
- **GPU detection:** Metal → CUDA → Vulkan → CPU.
- **Memory-safety gate:** every load is sized against available RAM before it runs.

The 30B-A3B MoE family (Qwen3-Coder-30B-A3B, Qwen3-30B-A3B-Thinking) is supported as an **opt-in tier-`xl`** profile for ≥24 GB boxes, **not** the v1 default.

## Build

```bash
cargo build --workspace
cargo test --workspace
cargo llvm-cov --workspace --fail-under-lines 98   # coverage gate
```

Toolchain pinned at `1.90.0` via `rust-toolchain.toml`.

## Status of the gates

Stratum's CI runs 12 verification gates per PR (build, test, 100% line-coverage target with documented carve-outs, fmt, clippy `-D warnings`, no `unsafe`/`unwrap`/`expect`/`panic` outside `#[cfg(test)]`, structured error codes, supply-chain audit, etc.). See [`docs/verification-gates.md`](docs/verification-gates.md).

Current coverage: **98.30%** lines, gate `--fail-under-lines 98`.

## Roadmap (public)

| Phase | Status | Deliverable |
|---|---|---|
| 0 | done | Repo skeleton, types, testkit, CLI scaffold |
| 1 | in progress | Single-model TUI MVP (this branch ships pass 5 of 7+) |
| 2 | next | `Provider` trait extraction + router + embedder + workspace concept |
| 3 | later | Agent loop + dense 7B swap + sandbox profiles + user agents |
| 4 | later | Compression + speculative decoding + OSS hardening |
| 5 | later | Multimodal (vision + voice) |
| 6 | later | OpenAI-compatible egress, MCP client/server, IDE integrations |
| 7 | later | Eval suite + bench |
| 8 | later | iOS + Android |

## License

Dual-licensed under either of:

- Apache License, Version 2.0 ([`LICENSE-APACHE`](LICENSE-APACHE))
- MIT license ([`LICENSE-MIT`](LICENSE-MIT))

at your option. Third-party attributions in [`NOTICE`](NOTICE).

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md). All commits go through PRs; `main` is protected with required review and green CI.

By contributing you agree your work is dual-licensed under Apache-2.0 OR MIT.

## Code of Conduct

This project follows the Contributor Covenant v2.1. See [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).

## Security

See [`SECURITY.md`](SECURITY.md) for the vulnerability disclosure policy. Do not file public issues for security problems.
