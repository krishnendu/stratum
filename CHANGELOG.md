# Changelog

All notable changes to Stratum. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versioning: [SemVer 2.0](https://semver.org/).

## [Unreleased]

## [0.2.1] - 2026-06-11

User-visible plumbing for the v0.2.0 runtime capabilities.

### Fixed
- Brew install conflict: `stratum` and `stratum-llama-cpp` formulas now declare `conflicts_with` each other (both ship a `stratum` binary). Users must `brew uninstall` one before installing the other (#146).
- `stratum models sync` no longer 404s — release workflow now publishes `catalog-stable.json` with an opinionated seed entry (Qwen2.5 0.5B Q4_K_M pointed at HuggingFace) (#146).

### Added
- `stratum chat --agents-dir <dir>` now resolves each agent's `model = "<slug>"` TOML field through `<state>/models.json` + `ModelInstaller`. Without `--features provider-llama-cpp` the resolver errors clearly per role; with the feature each role loads its own GGUF (#147).
- `stratum chat --parallel <role1,role2,...> --prompt "..."` fans the prompt out to N agents concurrently via `AgentHandoff::run_turn_parallel` and renders per-role results. JSON form available via `--json` (#148).
- `/parallel <roles>` palette command in the chat TUI dispatches the same flow live (#148).

## [0.2.0] - 2026-06-11

First minor bump. Bundles UX polish + secondary brew formula for real-LLM build.

### Added
- `stratum doctor --strict` validates `--json` output against `docs/schemas/doctor.v1.json` (#141).
- `stratum config get|set|list|unset` for `<state>/config.toml` (#134).
- `stratum --version` long form now includes the build git SHA + build date (#137).
- `dist/homebrew/stratum-llama-cpp.rb` — second Homebrew formula that builds-from-source with `--features provider-llama-cpp` for real local LLM inference (#139).
- Eval suites expanded: `baseline.json` 10 → 20 cases + new `coder.json` (10) + `polisher.json` (10), all green against EchoProvider (#141).
- `/budget` palette command shows current `TurnMetrics` (#141).
- `dist/systemd/stratum.service` + `dist/launchd/dev.stratum.serve.plist` + `docs/service-install.md` (#131).
- `docs/architecture.md` — Mermaid diagrams + module map (#132).
- `InstalledMigrator` trait + chain dispatch in `install.rs` (#138).

### Fixed
- Default `--manifest-url` / catalog URLs no longer point at unregistered domains; now resolve to GitHub Releases artifacts (#135).

## [0.1.2] - 2026-06-11

### Fixed
- Default `--manifest-url` and catalog URLs pointed at `updates.stratum.dev` and `catalog.stratum.dev`, which are not registered. Switch to the GitHub Releases artifacts the release workflow already publishes:
  `https://github.com/krishnendu/stratum/releases/latest/download/<channel>.json`
  `https://github.com/krishnendu/stratum/releases/latest/download/catalog-<channel>.json` (#135)

### Added
- `stratum config get|set|list|unset` for `<state>/config.toml`. Supports `string`/`bool`/`int`/`float` types. (#134)
- `dist/systemd/stratum.service` + `dist/launchd/dev.stratum.serve.plist` + `docs/service-install.md` for running `stratum serve` as a background daemon. (#131)
- `docs/architecture.md` — high-level Mermaid diagram + per-turn sequence + module map. (#132)

## [0.1.1] - 2026-06-11

### Fixed
- `print_greeting` hardcoded `tier=unknown`; now reads the real tier from `installed.toml` (#127).
- `PlatformTag` serialized as `mac_os_aarch64` (auto-snake_case of `MacOs`); release manifest emits the conventional `macos_aarch64`. `stratum self-update --check` against the v0.1.0 release errored (#128). Per-variant `#[serde(rename)]` fixes the wire form.
- `Paths::resolve_from` double-nested the state dir (`<base>/stratum/stratum/`). State files (`telemetry.json` opt-out, `models.json`, `transcripts/`, `anon_install_id`) lived at the wrong path and were silently ignored by the CLI (#129). Migration: v0.1.0 users with state files at the double-nested path should `mv` them up one level.

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
