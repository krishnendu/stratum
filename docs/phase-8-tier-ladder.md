# Phase 8 — Mobile tier ladder and bundles

This document summarizes the four mobile tiers added to `Tier` in
`crates/stratum-runtime/src/tier.rs` and the matching bundle TOMLs under
`assets/bundles/`. Source of truth for the ladder is
`plan/21-mobile-clients.md` §2 (tiers) and §5 (bundles).

## Tiers

The `Tier` enum gains four mobile variants alongside the existing
`Low` / `Medium` / `High` desktop ladder. Each mobile rung serializes as a
stable `snake_case` identifier (e.g. `mobile_ultra_low`) so on-disk records
written by `stratum doctor` and the installer roundtrip cleanly.

| Variant            | Wire form           | Target hardware                                          | Resident budget |
| ------------------ | ------------------- | -------------------------------------------------------- | --------------- |
| `MobileUltraLow`   | `mobile_ultra_low`  | Mid-range phones with < 6 GB total RAM                   | ~0.5-1 GB       |
| `MobileLow`        | `mobile_low`        | Flagship phones with ~8 GB total RAM                     | ~1.5-2 GB       |
| `MobileMedium`     | `mobile_medium`     | iPhone Pro / Pixel Pro / Samsung Ultra (12-16 GB RAM)    | ~3-4 GB         |
| `MobileHigh`       | `mobile_high`       | Foldables / tablets with 16 GB+ RAM                      | ~5-6 GB         |

`Tier::classify` on iOS / Android currently stubs to `MobileMedium`. Real
device-class detection (chip generation, thermal headroom, available unified
memory) is tracked as a Phase 8 follow-up; the stub keeps the rest of the
runtime compile-clean on mobile targets today.

The desktop bump rule (`bumped`) is unchanged — `Low → Medium → High` and
saturates at `High`. The mobile ladder has its own bump path
(`MobileUltraLow → MobileLow → MobileMedium → MobileHigh`) so future
heuristics (e.g. discovered NNAPI / CoreML accelerators) can promote a
device without crossing into the desktop world.

## Bundles

One TOML per tier under `assets/bundles/`, matching the schema sketched in
`plan/21` §5. Each file defines the four pipeline roles
(`role_main`, `role_router`, `role_polisher`, optional `role_embedder`).

### `mobile-ultra-low-v1.toml`

- `role_main`     = `qwen3-0.6b-instruct-q4_k_m`
- `role_router`   = `rule-based` (no neural router on this rung)
- `role_polisher` = `qwen3-0.6b-instruct-q4_k_m`

### `mobile-low-v1.toml`

- `role_main`     = `qwen3-1.7b-instruct-q4_k_m`
- `role_router`   = `qwen3-0.6b-instruct-q4_k_m`
- `role_polisher` = `qwen3-0.6b-instruct-q4_k_m`
- `role_embedder` = `arctic-embed-s-v2-mrl-256`

### `mobile-medium-v1.toml`

- `role_main`     = `gemma-4-e4b-q4_k_m`
- `role_router`   = `gemma-4-e2b-q4_k_m`
- `role_polisher` = `qwen3-0.6b-instruct-q4_k_m`
- `role_embedder` = `arctic-embed-l-v2-mrl-256`

### `mobile-high-v1.toml`

- `role_main`     = `qwen3-coder-7b-instruct-q4_k_m`
- `role_router`   = `gemma-4-e2b-q4_k_m`
- `role_polisher` = `qwen3-0.6b-instruct-q4_k_m`
- `role_embedder` = `arctic-embed-l-v2-mrl-256`

## Not in scope (deferred)

- A bundle loader / resolver that picks the right TOML at runtime — bundles
  are static assets today and will be wired through `model_resolver`
  alongside the mobile core crate.
- Device-class probing on iOS / Android (chipset → tier mapping).
- 30B MoE on mobile — explicitly out of scope per `plan/21` §2.
- Voice bundles — handled separately under the OS ASR / TTS path.
