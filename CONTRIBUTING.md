# Contributing to Stratum

> Status: pre-alpha. Public contribution flow is being shaken down. PRs welcome; expect strict gates.

## Quick start

```bash
git clone https://github.com/krishnendu/stratum
cd stratum
cargo build --workspace
cargo test --workspace
```

The toolchain is pinned via `rust-toolchain.toml` (currently 1.90.0). `rustup` picks it up automatically.

## Verification gates

Every PR must pass the 12 gates from `docs/verification-gates.md` (mirrored from the private design corpus). CI runs them; locally you can pre-flight with:

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
cargo llvm-cov --workspace --all-features --fail-under-lines 98
cargo deny check
```

Coverage exclusions are listed in `docs/coverage-exclusions.md` and must be appended in the same PR that introduces them.

## Branch & PR flow

- `main` is protected. Direct pushes are rejected.
- Branch naming: `feat/<short-slug>`, `fix/<short-slug>`, `docs/<short-slug>`, `chore/<short-slug>`.
- Conventional Commits subject line, e.g. `feat(runtime): add HTTP install resume`.
- Squash-merge is the default. Linear history on `main`.
- Every commit signed off (DCO): `git commit --signoff`.
- **No AI co-author trailers.** Stratum commits show the human author only.

## PR description template

The repo ships a PR template under `.github/PULL_REQUEST_TEMPLATE.md` with the verification checklist. Fill it in.

## Issues

Use the structured templates at `.github/ISSUE_TEMPLATE/`. Bug reports require a reproducer; feature requests must motivate the change against the project goals (8-16 GB laptop class, local agentic outcomes).

## Security disclosure

Vulnerabilities go private. See `SECURITY.md`.

## Code of Conduct

This project follows the Contributor Covenant v2.1. See `CODE_OF_CONDUCT.md`. Reports to `conduct@stratum.dev` (alias pending domain registration; until then, email the maintainer privately via GitHub).

## License

By contributing you agree your work is dual-licensed under Apache-2.0 OR MIT (the project license).
