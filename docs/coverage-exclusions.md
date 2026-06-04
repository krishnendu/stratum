# Coverage Exclusions

Stratum's CI gate (`G2.1` in `plan/36-verification-gates.md`) requires **100% line coverage** across the workspace. The only carve-outs allowed are paths that are infeasible to exercise from tests without contorting the production code; each carve-out must be listed below with reason.

| File | Line | Construct | Reason |
|---|---|---|---|
| `crates/stratum-cli/src/app.rs` | `serde_json::to_string_pretty(&report).expect(...)` (inside `doctor` JSON branch) | `expect` on infallible serialization | `DoctorReport` carries only primitive types and a `Vec<DoctorIssue>` of primitives; `serde_json::to_string_pretty` cannot fail for this input. The `expect_used` lint is denied workspace-wide so the path is explicitly `#[allow(clippy::expect_used, reason = "…")]`. |

If a new exclusion is added, it MUST be appended here in the same PR as the source change. The PR description's `G2.1` checkbox cannot be ticked otherwise.
