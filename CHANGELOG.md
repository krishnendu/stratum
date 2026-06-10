# Changelog

All notable changes to Stratum. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versioning: [SemVer 2.0](https://semver.org/).

## [Unreleased]

### Added
- Phase 0-7 runtime scaffolding (~117 PRs landed).
- CLI subcommands: `doctor`, `models {list|add|remove|recommend|sync|validate}`, `chat`, `serve`, `client`, `self-update`, `mem-check`, `events tail`, `sessions`, `agents`, `eval run`, `mcp list`.
- Real `LlamaCppProvider` behind `--features provider-llama-cpp` (Mac local-build verified).
- Multi-agent handoff via `--agents-dir`.
- TUI palette commands: `/plan /cancel /clear /quit /help /agents`.
- JSON-RPC 2.0 daemon (`stratum serve`) with rate-limit + auth + logging middleware.
- Sandbox spawn via `bwrap` (Linux) + `sandbox-exec` (macOS).
- MCP stdio client + JSON-RPC handshake.
- Eval pipeline: `EvalRunner` + `ClaudeCliJudge`.
- Permission flow with TUI prompter and remembered decisions.
- Transcript persistence + `--resume`.
- Catalog HTTPS sync + atomic self-update with `.bak` rollback.

## [0.1.0] - TBD

Initial public release. Scope: everything in [Unreleased] above, pending release infrastructure (workflow, docs, signing).

## [0.0.0-dev] - 2026-06-10

Placeholder marker — pre-release development snapshot. Not published; no artifacts.
