# Stratum Eval Suites

This directory contains eval suites for `stratum eval run`. Each suite is JSON matching `stratum_runtime::EvalSuite`.

## Suites

| File              | Cases | Purpose                                                            |
|---|---|---|
| `baseline.json`   | 10    | Smoke-test the chat plumbing against `EchoProvider` (no LLM)      |

## Running locally

```bash
cargo run -- eval run --suite evals/baseline.json
```

## Adding a new suite

1. Use the existing `EvalSuite` JSON shape — see `crates/stratum-runtime/src/eval_runner.rs`.
2. Drop the file in `evals/<name>.json`.
3. Add a row to the table above.
4. Optional: add a CI job entry in `.github/workflows/eval-baseline.yml`.

## Why echo-only

Quality evaluation against real models lands in a follow-up — needs llama-cpp feature + a GGUF download. The echo suite is the regression net for chat plumbing (turn lifecycle, transcript persistence, palette, event log, etc.).
