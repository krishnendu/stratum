# Contributing to Stratum

## Workflow

1. Branch off `main`: `git checkout -B feat/<slug> origin/main`.
2. Open a PR.
3. Wait for 5 CI checks: `fmt`, `clippy` (macOS + Linux), `test` (macOS + Linux).
4. Local review by a maintainer.
5. Squash merge via `gh pr merge <N> --squash --auto --delete-branch`.

## Branch protection

`main` is protected: no force push, linear history, PR required, 5 status checks required. **Never use `gh pr merge --admin`** — every merge is gated by local review. Auto-merge via `--auto --squash --delete-branch` is fine and preferred.

## Commits

- [Conventional Commits](https://www.conventionalcommits.org/) subject line, e.g. `feat(runtime): add http install resume`.
- DCO sign-off required: `git commit --signoff`.
- **No AI co-author trailers.** Human authors only.

## Pre-flight checks

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
cargo llvm-cov --workspace --exclude xtask --fail-under-lines 95
cargo run -p xtask -- check-error-codes
```

## Lints

Workspace-wide: `clippy::pedantic`, `clippy::nursery`, `clippy::cargo`. `unwrap_used`, `expect_used`, and `panic` are denied outside `#[cfg(test)]`. Lint config lives in `clippy.toml` and `Cargo.toml` `[workspace.lints]`.

## Coverage

Gate: `cargo llvm-cov --workspace --exclude xtask --fail-under-lines 95`. Exclusions tracked in `docs/coverage-exclusions.md` — append in the same PR that adds them.

## xtask error-code validator

`cargo run -p xtask -- check-error-codes` enforces that every `StratumError` variant has a stable error code and a row in the error-code registry. Must pass on every PR.

## Test conventions

- Unit tests live in-module under `#[cfg(test)] mod tests { … }`.
- Integration tests live in `crates/<name>/tests/`.
- Prefer `stratum-testkit` for shared fixtures (fake providers, fake clocks, temp dirs).

## Authoring tools

- Local tools: implement the dispatch path in [`crates/stratum-runtime/src/tool_invocation.rs`](crates/stratum-runtime/src/tool_invocation.rs) and register a dispatcher in [`crates/stratum-runtime/src/tool_dispatchers.rs`](crates/stratum-runtime/src/tool_dispatchers.rs).
- MCP-backed tools: see [`crates/stratum-runtime/src/mcp_jsonrpc.rs`](crates/stratum-runtime/src/mcp_jsonrpc.rs) (stdio JSON-RPC client) and [`crates/stratum-runtime/src/tool_dispatcher_mcp.rs`](crates/stratum-runtime/src/tool_dispatcher_mcp.rs) (bridge to the dispatcher trait).

## Codebase tour

Major modules under `crates/stratum-runtime/src/`:

- Agent runtime: `agent_loop`, `agent_factory`, `agent_session`, `agent_handoff`, `agents`, `agent_registry_loader`, `conversation`, `intent_router`, `plan_mode`.
- Providers + inference: `provider`, `llama_provider`, `provider_cache`, `prompt_template`, `prompt_cache`, `embedder`.
- Tools + sandbox: `tool_invocation`, `tool_dispatchers`, `tool_dispatcher_mcp`, `mcp`, `mcp_jsonrpc`, `sandbox`, `sandbox_profile`, `sandbox_resolve`, `permission_prompt`.
- Models + install: `model_catalog`, `model_resolver`, `catalog_sync`, `install`, `download`, `registry`, `probe`, `tier`, `gate`.
- Resilience: `retry`, `rate_limit`, `budget`, `budget_meter`, `cancel`, `cancel_cascade`.
- Observability: `observability`, `event_log`, `logging`, `telemetry`, `crash_report`, `panic`.
- RAG + eval: `rag`, `rag_index_builder`, `rag_query`, `eval_runner`, `claude_cli_judge`.
- Daemon: `serve_protocol`, `serve_server`, `serve_handler_agent`, `serve_middleware`.
- Misc: `secrets`, `i18n`, `paths`, `prompts`, `injection`.

## License

By contributing you agree your work is dual-licensed under Apache-2.0 OR MIT.
