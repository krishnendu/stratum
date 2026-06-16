# Coverage Exclusions

Stratum's CI gate (`G2.1` in `plan/36-verification-gates.md`) requires line coverage **≥ 96%** (`cargo llvm-cov --fail-under-lines 96`). The plan v2 target is 100%; the gap is the documented carve-outs below.

When a new carve-out is added it MUST be appended here in the same PR. The PR description's `G2.1` checkbox cannot be ticked otherwise.

## Documented carve-outs

| File | Construct | Reason |
|---|---|---|
| `crates/stratum-cli/src/app.rs` | `serde_json::to_string_pretty(&report).expect(...)` (inside `doctor` JSON branch) | `DoctorReport` carries only primitive types and a `Vec<DoctorIssue>` of primitives; `to_string_pretty` cannot fail for this input. The `expect_used` lint is denied workspace-wide so the path is explicitly `#[allow(clippy::expect_used, reason = "…")]`. |
| `crates/stratum-cli/src/app.rs` | `serde_json::to_string_pretty(&record).expect(...)` (inside `init` JSON branch) | Same reasoning for `InstalledToml`. |
| `crates/stratum-runtime/src/install.rs` | `toml_edit::ser::to_string(self).expect(...)` (inside `write_atomic`) | `InstalledToml` shape: primitive types + `Vec<String>`. `toml_edit::ser` is infallible for this. |
| `crates/stratum-runtime/src/install.rs` | `now.format(&Rfc3339).expect(...)` (inside `InstalledToml::new`) | `OffsetDateTime` always formats successfully under Rfc3339. |
| `crates/stratum-runtime/src/probe.rs` | `if std::arch::is_aarch64_feature_detected!("neon")` (inside `detect_cpu_features`) | On aarch64 hosts NEON is always present, so the false-arm of this branch is unreachable on the supported CI runners. Region-coverage artifact; the line itself is hit. |
| Coverage region artifacts in `#[cfg(test)]` test bodies (`matches!` second arms, `||` short-circuits) | various | These are llvm-cov's region tracking inside test assertions. They affect the regions percentage but not the line percentage. CI gate uses `--fail-under-lines`, not regions. |
| `crates/stratum-cli/src/chat.rs` | `run()` and `event_loop()` (lines ~218-244) | TTY-bound: requires real terminal raw-mode + alternate-screen + event poll. Not driveable from `cargo test`; covered manually by `stratum chat` and a Phase 7 expectrl-driven end-to-end. |
| `crates/stratum-cli/src/app.rs` | `chat_command` function | Forwards to `chat::run` whose body is TTY-bound (see above). The branch arms are mechanically obvious. |
| `crates/xtask/**` | Entire crate | Dev tooling (workspace lint validator, error-code scanner). Not shipped to users; verified at PR time by the `xtask check-error-codes` CI job, not by coverage. Excluded via `cargo llvm-cov --workspace --exclude xtask` in both gate + lcov-artifact steps. |
| `crates/stratum-tui/**` | Entire crate | Workspace-internal crate (`publish = false`) housing the chat renderer + palette + theme + brand that moved out of `stratum-cli`. The previous CLI-resident `chat::run` and `chat::event_loop` were already carved out as TTY-bound (see two rows above); the same paths move with the file, so the exclusion moves too. Other items inside the crate are exercised through the CLI's integration tests. Excluded via `cargo llvm-cov --workspace --exclude stratum-tui`. |
| `crates/stratum-backend-local/**` | Entire crate | Workspace-internal adapter crate. The translation layer between `AgentLoop` blocks and `BackendApi::BackendEvent` is exercised by integration tests against the CLI; covering it as its own crate boundary would require a duplicate test surface that's not yet built. Excluded via `cargo llvm-cov --workspace --exclude stratum-backend-local`. |

## Why 96 instead of 100

Two classes of carve-out:

1. **Infallible-by-our-shape error closures** (Rfc3339 format, toml/serde serialization on primitive-only structs) — the `expect_used` calls plus their wrapped `# Panics` doc carve-outs.
2. **TTY-bound TUI initialization paths** (`stratum chat`'s `run` + `event_loop`) which cannot be driven from `cargo test`. These will be exercised by an end-to-end harness (`expectrl`-based) in Phase 7 once the eval suite scaffolding lands.

The gate is set just above the current measured value to fail PRs that meaningfully regress coverage while tolerating the documented artifacts.

When a future change makes 100% achievable (e.g. nightly `#[coverage(off)]` stabilizes, or a more targeted llvm-cov region-exclusion mechanism lands), the gate moves back to 100 and this document shrinks accordingly.
