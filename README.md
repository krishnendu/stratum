# Stratum

> Local-LLM agentic TUI agent for laptop-class hardware.

[![CI](https://github.com/krishnendu/stratum/actions/workflows/ci.yml/badge.svg)](https://github.com/krishnendu/stratum/actions/workflows/ci.yml)
[![License: Apache-2.0 OR MIT](https://img.shields.io/badge/license-Apache--2.0%20OR%20MIT-blue.svg)](#license)

**Status:** Phase 0-7 functionally complete (~117 PRs). v0.1.0 release pending.

Stratum is a local, multi-LLM agentic TUI written in Rust, targeting the 8-16 GB laptop class. Entirely offline by default; composes a crew of small models instead of a single monolith.

## Quick start (Echo provider)

```bash
git clone https://github.com/krishnendu/stratum && cd stratum
cargo build --workspace
cargo run -- doctor
cargo run -- chat --prompt hi
```

## Real LLM (llama.cpp backend)

```bash
cargo build --features provider-llama-cpp
cargo run -- models list
cargo run -- models add --from-url https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf \
    --sha256 <hex>
cargo run --features provider-llama-cpp -- chat --model qwen-0.5b --prompt "hello"
```

## TUI palette

Run `stratum chat` (no `--prompt`) for the interactive TUI:

| Command | Effect |
|---|---|
| `/plan` | Toggle plan mode (read-only, no tool side-effects). |
| `/cancel` | Cancel the in-flight turn. |
| `/clear` | Clear the visible conversation buffer. |
| `/quit` | Exit the TUI. |
| `/help` | Show palette help. |
| `/agents` | List available agents in the registry. |

## CLI surface

```text
stratum doctor                          # probe host: CPU, RAM, GPU, tier
stratum models list                     # show installed + catalog models
stratum models add --from-url <U> --sha256 <H>
stratum models remove <id>
stratum models recommend                # suggest models for current tier
stratum models sync                     # refresh signed catalog over HTTPS
stratum models validate                 # re-verify installed SHA-256s
stratum chat [--model <id>] [--prompt <P>] [--resume <session>]
stratum serve [--tcp-port N] [--socket P] [--json]   # JSON-RPC 2.0 daemon
stratum client --method <m> [--tcp <addr> | --socket <p>]
stratum self-update --check             # check for newer signed release
stratum self-update --apply             # atomic swap, .bak rollback
stratum mem-check                       # dry-run RAM gate for a model
stratum events tail                     # follow structured event log
stratum sessions list
stratum sessions show <id>
stratum sessions delete <id>
stratum agents list
stratum agents show <name>
stratum eval run [--suite <s>]          # run eval pipeline (claude-cli judge)
stratum mcp list                        # show MCP servers in config
```

## Daemon mode

```bash
stratum serve --tcp-port 0 --json &
PORT=$(stratum serve --tcp-port 0 --json | jq -r .tcp_port)   # from stderr/stdout banner
stratum client --method ping --tcp 127.0.0.1:$PORT
```

## Shell completions

`stratum completions <shell>` prints a tab-completion script to stdout. Redirect the
output into your shell's completion directory:

```bash
# Bash (system-wide)
sudo stratum completions bash > /etc/bash_completion.d/stratum

# Zsh
mkdir -p ~/.zsh/completions
stratum completions zsh > ~/.zsh/completions/_stratum
# Add `fpath=(~/.zsh/completions $fpath)` and `autoload -Uz compinit && compinit` to ~/.zshrc

# Fish
stratum completions fish > ~/.config/fish/completions/stratum.fish
```

Supported shells: `bash`, `zsh`, `fish`, `powershell`, `elvish`.

## Architecture

The runtime is built around a small set of core abstractions in `crates/stratum-runtime/`: `AgentLoop` drives the turn cycle; `Provider` is the inference trait (Echo, llama-cpp); `ToolDispatcher` invokes tools per-call with capability gating; `Sandbox` resolves and enforces filesystem/network profiles (`bwrap` on Linux, `sandbox-exec` on macOS); `EventEmitter` produces a structured, JSON-line audit stream consumed by the TUI and `events tail`.

Deeper design notes live in `plan/` (gitignored, not published). Architecture docs intended for public consumption land under [`docs/`](docs/).

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md). Branch protection on `main`: PR required, 5 CI checks, local review gate, squash merge, linear history.

## License

Dual-licensed under either of:

- Apache License, Version 2.0 ([`LICENSE-APACHE`](LICENSE-APACHE))
- MIT license ([`LICENSE-MIT`](LICENSE-MIT))

at your option. Third-party attributions in [`NOTICE`](NOTICE).
