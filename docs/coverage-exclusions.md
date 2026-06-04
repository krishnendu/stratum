# Coverage Exclusions

Stratum's CI gate (`G2.1` in `plan/36-verification-gates.md`) requires line coverage **≥ 99%** (`cargo llvm-cov --fail-under-lines 99`). The plan v2 target is 100%; the gap is the small set of carve-outs listed below. Each carve-out is a path that is unreachable on the testable host without contorting the production code.

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

## Why 99 instead of 100

The remaining sub-percent is regional bookkeeping inside test-only code that cannot be removed without sacrificing test clarity. The gate is set just above the current measured value (99.77% lines) to fail PRs that meaningfully regress coverage while tolerating the documented artifacts.

When a future change makes 100% achievable (e.g. nightly `#[coverage(off)]` stabilizes, or a more targeted llvm-cov region-exclusion mechanism lands), the gate moves back to 100 and this document shrinks accordingly.
