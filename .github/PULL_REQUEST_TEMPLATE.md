## What & why

<!-- 1-3 bullets. Link the plan doc or issue if relevant. -->

## How

<!-- Describe the approach. Call out non-obvious choices. -->

## Test plan

- [ ] Local `cargo test --workspace --all-features`
- [ ] Manual smoke if behavior changed (paste the command + output)
- [ ] CI green on this PR

## Verification gates

- [ ] G1 build (CI)
- [ ] G2 tests pass (CI)
- [ ] G2.1 line coverage ≥ 98% (CI)
- [ ] G2.2 integration tests cover the public surface touched here
- [ ] G3 fmt + clippy `-D warnings` clean (CI)
- [ ] G4 no `unsafe`; no `unwrap`/`expect`/`panic` outside `#[cfg(test)]` except documented carve-outs
- [ ] G5 every new `Err` path uses a catalog `ErrorCode` (`STRAT-Exxxx`)
- [ ] G6 `cargo deny check` clean (CI)
- [ ] G7 docs entry / `TODO(docs)` if a new public surface ships
- [ ] G8 conventional commit + DCO sign-off; **no AI co-author trailers**
- [ ] G9 security checklist (sandbox / secrets / FFI / network / `unsafe`)  or  N/A: <reason>
- [ ] G10 snapshot test for UI changes  or  N/A: <reason>
- [ ] G11 schema migration test for on-disk schema bumps  or  N/A: <reason>
- [ ] G12 network / telemetry impact disclosed  or  N/A: no network change
