# Changelog

All notable changes to Stratum. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versioning: [SemVer 2.0](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-06-11

Initial public release.

### Added
- Phase 0-7 runtime scaffolding (~124 PRs landed).
- CLI subcommands: `doctor`, `models {list|add|remove|recommend|sync|validate}`, `chat`, `serve`, `client`, `self-update`, `mem-check`, `events tail`, `sessions`, `agents`, `eval run`, `mcp list`, `completions`.
- Real `LlamaCppProvider` behind `--features provider-llama-cpp` (Mac local-build verified).
- Multi-agent handoff via `--agents-dir` + TUI status-line role indicator.
- TUI palette commands: `/plan /cancel /clear /quit /help /agents`.
- TUI permission prompter modal (a/s/f/d/F key bindings).
- TUI status bar: in-flight `[generating…]` + token-rate.
- JSON-RPC 2.0 daemon (`stratum serve`) with rate-limit + auth + logging middleware; Unix-socket + TCP-loopback transports.
- Sandbox spawn via `bwrap` (Linux) + `sandbox-exec` (macOS) + passthrough fallback.
- MCP stdio client + real JSON-RPC handshake (initialize → tools/list → tools/call).
- Eval pipeline: `EvalRunner` + `ClaudeCliJudge` + `evals/baseline.json` (10 cases) + nightly CI cron.
- Permission flow with TUI prompter and remembered decisions (Session/Forever buckets).
- Transcript persistence at `<state>/transcripts/<id>.json` + `stratum chat --resume <id>`.
- Catalog HTTPS sync (`stratum models sync`) + atomic self-update (`--check` + `--apply` with `.bak` rollback).
- Shell completions: bash/zsh/fish/powershell/elvish via `stratum completions <shell>`.
- Release workflow + Homebrew formula scaffold + release-process docs.

## [0.0.0-dev] - 2026-06-10

Placeholder marker — pre-release development snapshot. Not published; no artifacts.
