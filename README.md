# Stratum

> Status: **pre-alpha, Phase 0** — repo skeleton only. No runnable agent yet.

Stratum is a local, multi-LLM agentic TUI agent written in Rust, designed for the 8-16 GB laptop class. It embeds and orchestrates a crew of small/selective-load models to deliver best-in-class **local** agentic outcomes — entirely offline.

This README is the public face; the working-design corpus lives privately at `plan/` (gitignored) and graduates into `docs/` at the close of each phase.

## Workspace layout

```
crates/
  stratum-types/     core enums, error taxonomy
  stratum-testkit/   golden-transcript test harness
  stratum-cli/       binary entry point (`stratum`)
```

More crates land in Phase 1 (`stratum-providers`, `stratum-runtime`) and Phase 3 (`stratum-tools`, `stratum-agents`).

## Build

```
cargo build --workspace
cargo test --workspace
```

Coverage:

```
cargo llvm-cov --workspace --fail-under-lines 100
```

## License

Dual-licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

(License files land in Phase 4 ahead of the public push.)

## Contributing

The public repo opens for contributions in Phase 4. Until then, all commits and design discussion are private.
