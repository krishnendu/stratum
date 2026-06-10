# Stratum Eval Suites

This directory contains eval suites for `stratum eval run`. Each suite is JSON matching `stratum_runtime::EvalSuite`.

## Suites

| File              | Cases | Purpose                                                            |
|---|---|---|
| `baseline.json`   | 20    | Smoke-test the chat plumbing against `EchoProvider` (no LLM)      |
| `coder.json`      | 10    | Code-shaped prompts (`fn`/`struct`/`impl`/`->`/`=>`/`;`/`{}`)     |
| `polisher.json`   | 10    | Prose-shaped prompts (sentences, punctuation, lists, contractions) |
| `llm-baseline.json` | 8   | Real-LLM smoke suite against `stratum-llama-cpp` + Qwen 0.5B GGUF  |
| `llm-coder.json`  | 6     | Real-LLM code-shaped prompts against `stratum-llama-cpp` + Qwen 0.5B |

## Running locally

```bash
cargo run -- eval run --suite evals/baseline.json
cargo run -- eval run --suite evals/coder.json
cargo run -- eval run --suite evals/polisher.json

# Real-LLM suites (require `provider-llama-cpp` feature + a registered GGUF model):
cargo run --features provider-llama-cpp -- eval run --suite evals/llm-baseline.json
cargo run --features provider-llama-cpp -- eval run --suite evals/llm-coder.json
```

## Adding a new suite

1. Use the existing `EvalSuite` JSON shape — see `crates/stratum-runtime/src/eval_runner.rs`.
2. Drop the file in `evals/<name>.json`.
3. Add a row to the table above.
4. Optional: add a CI job entry in `.github/workflows/eval-baseline.yml`.

## Why echo-only (`baseline` / `coder` / `polisher`)

The echo suites are the regression net for chat plumbing (turn lifecycle, transcript persistence, palette, event log, etc.) — they run against `EchoProvider` and need no model.

## Real-LLM suites (`llm-baseline` / `llm-coder`)

Run against `stratum-llama-cpp` with a Qwen 0.5B Instruct GGUF. They run on demand and nightly via `.github/workflows/eval-llm.yml` — not in PR CI, because they need a ~470 MB model download and a feature build.
