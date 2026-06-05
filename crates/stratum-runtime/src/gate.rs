//! Memory-safety gate that refuses model loads when free RAM minus the
//! would-be hot footprint drops below the configured margin.
//!
//! See `plan/14-memory-safety-gate.md`. The gate is a pure predicate over a
//! [`HardwareProbe`] and a [`MemEstimate`]; it does no IO.

use stratum_types::error::codes::E3007_MODEL_LOAD_REFUSED;
use stratum_types::{MemEstimate, ModelId, StratumError, StratumResult};

use crate::probe::HardwareProbe;

/// A currently-resident model the gate may suggest unloading to free room for a
/// candidate load. Carried by the runtime's "active set" bookkeeping; the gate
/// itself never mutates it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadedModel {
    /// Stable identifier of the resident model.
    pub id: ModelId,
    /// Memory estimate of the resident model.
    pub estimate: MemEstimate,
    /// Optional role hint (e.g. `"router"`, `"planner"`) the caller may render
    /// alongside the unload suggestion. Not consumed by the gate itself.
    pub role_hint: Option<String>,
}

/// Default safety margin in mebibytes (1 GiB).
pub const DEFAULT_MARGIN_MIB: u32 = 1024;

/// The memory-safety gate.
///
/// Configured with a single `margin_mib`: the minimum amount of RAM, in
/// mebibytes, that must remain free **after** the hot footprint of the model
/// is subtracted from the currently-available RAM.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryGate {
    /// Required margin in mebibytes between free RAM and the model's hot
    /// footprint. Defaults to [`DEFAULT_MARGIN_MIB`] (1 GiB).
    pub margin_mib: u32,
}

impl Default for MemoryGate {
    fn default() -> Self {
        Self {
            margin_mib: DEFAULT_MARGIN_MIB,
        }
    }
}

impl MemoryGate {
    /// Construct a gate with an explicit margin.
    #[must_use]
    pub const fn new(margin_mib: u32) -> Self {
        Self { margin_mib }
    }

    /// Boolean form of the gate predicate. Returns `true` when the load would
    /// fit (i.e. `available - needed >= margin`), `false` otherwise. This is
    /// the form the TUI status bar consumes; it never constructs an error.
    #[must_use]
    #[allow(
        clippy::trivially_copy_pass_by_ref,
        reason = "signature pinned by plan/14-memory-safety-gate.md so the TUI status bar holds a &MemoryGate"
    )]
    pub fn would_fit(
        &self,
        probe: &HardwareProbe,
        estimate: &MemEstimate,
        context_tokens: u32,
    ) -> bool {
        let needed = estimate.hot_ram_mib(context_tokens);
        let available = probe.ram_available_mib;
        // available - needed >= margin, expressed without underflow.
        available >= needed.saturating_add(self.margin_mib)
    }

    /// Result-returning form. On refusal, builds a
    /// [`StratumError`] with [`E3007_MODEL_LOAD_REFUSED`] whose message
    /// quotes both the free GB and the would-be hot footprint, rounded to
    /// one decimal.
    ///
    /// # Errors
    /// Returns `Err(STRAT-E3007)` when `available - needed < margin`.
    #[allow(
        clippy::trivially_copy_pass_by_ref,
        reason = "signature pinned by plan/14-memory-safety-gate.md so callers hold a &MemoryGate"
    )]
    pub fn check(
        &self,
        probe: &HardwareProbe,
        estimate: &MemEstimate,
        context_tokens: u32,
    ) -> StratumResult<()> {
        if self.would_fit(probe, estimate, context_tokens) {
            return Ok(());
        }
        let needed_mib = estimate.hot_ram_mib(context_tokens);
        let free_gb = mib_to_gb_one_decimal(probe.ram_available_mib);
        let needed_gb = mib_to_gb_one_decimal(needed_mib);
        let margin_gb = mib_to_gb_one_decimal(self.margin_mib);
        let message = format!(
            "free {free_gb} GB, would need {needed_gb} GB hot, {margin_gb} GB margin required"
        );
        Err(StratumError::new(E3007_MODEL_LOAD_REFUSED, message)
            .with_hint("free RAM or pick a smaller model"))
    }

    /// Suggest the smallest set of currently-loaded model ids whose unload
    /// would free enough RAM for `candidate` (at `context_tokens` planned
    /// length) plus the gate's margin.
    ///
    /// Strategy: sort `loaded` by `estimate.hot_ram_mib(0)` descending and
    /// take from the head until the cumulative freed bytes plus the currently
    /// available bytes meet `needed + margin`. The greedy traversal stops as
    /// soon as the condition is met.
    ///
    /// Returns an empty `Vec` when no subset (including the full list) can
    /// satisfy the requirement — callers must not interpret an empty result
    /// as a partial suggestion.
    #[must_use]
    #[allow(
        clippy::trivially_copy_pass_by_ref,
        reason = "matches `check`/`would_fit` so callers hold a single &MemoryGate"
    )]
    pub fn suggest_unloads(
        &self,
        probe: &HardwareProbe,
        candidate: &MemEstimate,
        context_tokens: u32,
        loaded: &[LoadedModel],
    ) -> Vec<ModelId> {
        let needed_mib = candidate.hot_ram_mib(context_tokens);
        let required_mib = u64::from(needed_mib).saturating_add(u64::from(self.margin_mib));
        let available_mib = u64::from(probe.ram_available_mib);
        // Already fits — nothing to suggest.
        if available_mib >= required_mib {
            return Vec::new();
        }
        let deficit_mib = required_mib.saturating_sub(available_mib);

        // Sort by hot footprint (at zero context) descending.
        let mut ordered: Vec<&LoadedModel> = loaded.iter().collect();
        ordered.sort_by(|a, b| {
            b.estimate
                .hot_ram_mib(0)
                .cmp(&a.estimate.hot_ram_mib(0))
                .then_with(|| a.id.as_str().cmp(b.id.as_str()))
        });

        let mut freed: u64 = 0;
        let mut picks: Vec<ModelId> = Vec::new();
        for entry in ordered {
            freed = freed.saturating_add(u64::from(entry.estimate.hot_ram_mib(0)));
            picks.push(entry.id.clone());
            if freed >= deficit_mib {
                return picks;
            }
        }
        // Even unloading everything isn't enough.
        Vec::new()
    }

    /// Like [`Self::check`], but consults the active set so the refusal error
    /// can include an "unload X, Y" hint when a feasible suggestion exists.
    ///
    /// When `suggest_unloads` returns a non-empty list, the message ends with
    /// a `hint: unload <ids>` line embedded inline. When it returns empty, the
    /// error carries the same `"free RAM or pick a smaller model"` static hint
    /// as [`Self::check`].
    ///
    /// # Errors
    /// Returns `Err(STRAT-E3007)` when `available - needed < margin`.
    #[allow(
        clippy::trivially_copy_pass_by_ref,
        reason = "matches `check`/`would_fit` so callers hold a single &MemoryGate"
    )]
    pub fn check_with(
        &self,
        probe: &HardwareProbe,
        candidate: &MemEstimate,
        context_tokens: u32,
        loaded: &[LoadedModel],
    ) -> StratumResult<()> {
        if self.would_fit(probe, candidate, context_tokens) {
            return Ok(());
        }
        let needed_mib = candidate.hot_ram_mib(context_tokens);
        let free_gb = mib_to_gb_one_decimal(probe.ram_available_mib);
        let needed_gb = mib_to_gb_one_decimal(needed_mib);
        let margin_gb = mib_to_gb_one_decimal(self.margin_mib);
        let base = format!(
            "free {free_gb} GB, would need {needed_gb} GB hot, {margin_gb} GB margin required"
        );
        let suggestion = self.suggest_unloads(probe, candidate, context_tokens, loaded);
        if suggestion.is_empty() {
            return Err(StratumError::new(E3007_MODEL_LOAD_REFUSED, base)
                .with_hint("free RAM or pick a smaller model"));
        }
        let ids: Vec<&str> = suggestion.iter().map(ModelId::as_str).collect();
        let message = format!("{base}\nhint: unload {}", ids.join(", "));
        Err(StratumError::new(E3007_MODEL_LOAD_REFUSED, message))
    }
}

/// Mebibytes → gigabytes rounded to one decimal place, formatted as e.g.
/// `0.4`. The conversion uses base-10 GB so users see familiar numbers.
fn mib_to_gb_one_decimal(mib: u32) -> String {
    // 1 GB = 1000 MB; 1 MiB ≈ 1.048576 MB. We compute in fixed point:
    //   gb_x10 = round(mib * 1.048576 / 100)
    //         = round(mib * 1_048_576 / 100_000_000)
    let scaled = u64::from(mib) * 1_048_576;
    let gb_x10 = (scaled + 50_000_000) / 100_000_000;
    let whole = gb_x10 / 10;
    let frac = gb_x10 % 10;
    format!("{whole}.{frac}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::probe::GpuBackend;

    fn probe_with(avail_mib: u32) -> HardwareProbe {
        HardwareProbe::synthetic(
            16_384,
            avail_mib,
            "aarch64",
            vec!["neon"],
            8,
            GpuBackend::Metal,
            "macos",
        )
    }

    fn estimate(weight: u32, kv_per_token: u32, mmproj: u32) -> MemEstimate {
        MemEstimate {
            weight_rss_mib: weight,
            kv_per_token_bytes: kv_per_token,
            mmproj_mib: mmproj,
            vram_mib: 0,
        }
    }

    #[test]
    fn default_margin_is_one_gib() {
        assert_eq!(MemoryGate::default().margin_mib, 1024);
        assert_eq!(MemoryGate::default(), MemoryGate::new(1024));
    }

    #[test]
    fn allow_path_when_lots_of_free_ram() {
        let gate = MemoryGate::default();
        let probe = probe_with(8_000);
        let est = estimate(3_600, 0, 0);
        assert!(gate.would_fit(&probe, &est, 0));
        assert!(gate.check(&probe, &est, 0).is_ok());
    }

    #[test]
    fn refuse_path_when_not_enough_ram() {
        let gate = MemoryGate::default();
        // 1.5 GiB free, need 3.6 GiB → far below the 1 GiB margin.
        let probe = probe_with(1_536);
        let est = estimate(3_600, 0, 0);
        assert!(!gate.would_fit(&probe, &est, 0));
        let err = gate.check(&probe, &est, 0).unwrap_err();
        assert_eq!(err.code().as_str(), "STRAT-E3007");
        let msg = err.message.as_str();
        assert!(msg.contains("free "), "msg was: {msg}");
        assert!(msg.contains(" GB"), "msg was: {msg}");
        assert!(msg.contains("hot"), "msg was: {msg}");
        // 1536 MiB ≈ 1.6 GB.
        assert!(msg.contains("1.6"), "expected free 1.6 GB; msg was: {msg}");
        // 3600 MiB ≈ 3.8 GB.
        assert!(msg.contains("3.8"), "expected hot 3.8 GB; msg was: {msg}");
    }

    #[test]
    fn boundary_available_minus_needed_equals_margin_is_ok() {
        // needed = 2048, margin = 1024, so available exactly 2048 + 1024 = 3072.
        let gate = MemoryGate::new(1024);
        let probe = probe_with(3072);
        let est = estimate(2048, 0, 0);
        assert!(gate.would_fit(&probe, &est, 0));
        assert!(gate.check(&probe, &est, 0).is_ok());
    }

    #[test]
    fn boundary_one_mib_below_margin_is_refused() {
        let gate = MemoryGate::new(1024);
        let probe = probe_with(3071);
        let est = estimate(2048, 0, 0);
        assert!(!gate.would_fit(&probe, &est, 0));
        assert!(gate.check(&probe, &est, 0).is_err());
    }

    #[test]
    fn zero_context_uses_only_weights_and_mmproj() {
        let gate = MemoryGate::default();
        let probe = probe_with(5_000);
        let est = estimate(3_000, 4096, 256);
        // hot @ 0 ctx = 3000 + 256 = 3256 MiB. available - needed = 1744 >= 1024.
        assert!(gate.would_fit(&probe, &est, 0));
    }

    #[test]
    fn nonzero_context_adds_kv_pressure_and_can_flip_decision() {
        let gate = MemoryGate::default();
        let probe = probe_with(5_000);
        let est = estimate(3_000, 4096, 0);
        // @ 8192 tokens: KV = 32 MiB, hot = 3032, leaves 1968 → ok.
        assert!(gate.would_fit(&probe, &est, 8192));
        // @ 1_000_000 tokens: KV ≈ 3906 MiB, hot ≈ 6906 → refused.
        assert!(!gate.would_fit(&probe, &est, 1_000_000));
    }

    #[test]
    fn custom_margin_is_respected() {
        let strict = MemoryGate::new(4096);
        let lax = MemoryGate::new(0);
        let probe = probe_with(4_500);
        let est = estimate(3_000, 0, 0);
        // available 4500 - needed 3000 = 1500 MiB margin.
        assert!(!strict.would_fit(&probe, &est, 0));
        assert!(lax.would_fit(&probe, &est, 0));
    }

    #[test]
    fn check_error_has_hint() {
        let gate = MemoryGate::default();
        let probe = probe_with(400);
        let est = estimate(3000, 0, 0);
        let err = gate.check(&probe, &est, 0).unwrap_err();
        assert!(err.hint.is_some());
        let rendered = format!("{err}");
        assert!(rendered.contains("STRAT-E3007"));
        assert!(rendered.contains("hint:"));
    }

    #[test]
    fn saturating_add_protects_against_overflow() {
        // margin and needed both near u32::MAX must not panic.
        let gate = MemoryGate::new(u32::MAX);
        let probe = probe_with(u32::MAX);
        let est = estimate(u32::MAX, 0, 0);
        // would_fit must return false because needed + margin saturates to
        // u32::MAX, and available (u32::MAX) >= u32::MAX is true — but the
        // estimate's hot_ram_mib also saturates. Either way, no panic.
        let _ = gate.would_fit(&probe, &est, 0);
        let _ = gate.check(&probe, &est, 0);
    }

    #[test]
    fn mib_to_gb_one_decimal_known_values() {
        assert_eq!(mib_to_gb_one_decimal(0), "0.0");
        assert_eq!(mib_to_gb_one_decimal(1024), "1.1"); // 1.0737 → 1.1
        assert_eq!(mib_to_gb_one_decimal(953), "1.0"); // 0.9996 → 1.0
        assert_eq!(mib_to_gb_one_decimal(400), "0.4"); // 0.4194 → 0.4
        assert_eq!(mib_to_gb_one_decimal(3600), "3.8"); // 3.7748 → 3.8
    }

    #[test]
    fn memory_gate_is_copy_and_eq() {
        let a = MemoryGate::new(512);
        let b = a;
        assert_eq!(a, b);
    }

    fn loaded(id: &str, weight: u32) -> LoadedModel {
        LoadedModel {
            id: ModelId::from(id),
            estimate: estimate(weight, 0, 0),
            role_hint: None,
        }
    }

    #[test]
    fn suggest_unloads_empty_when_already_fits() {
        // Already fits — suggestion is empty regardless of loaded set.
        let gate = MemoryGate::default();
        let probe = probe_with(8_000);
        let est = estimate(3_000, 0, 0);
        let resident = vec![loaded("router", 1000), loaded("polisher", 500)];
        let picks = gate.suggest_unloads(&probe, &est, 0, &resident);
        assert!(picks.is_empty());
    }

    #[test]
    fn suggest_unloads_empty_loaded_returns_empty() {
        // No room and nothing to free → empty (don't return a partial).
        let gate = MemoryGate::default();
        let probe = probe_with(500);
        let est = estimate(3_000, 0, 0);
        let picks = gate.suggest_unloads(&probe, &est, 0, &[]);
        assert!(picks.is_empty());
    }

    #[test]
    fn suggest_unloads_single_large_model_suffices() {
        // free 1.5 GiB, need 2 GiB + 1 GiB margin = 3 GiB → deficit 1.5 GiB.
        // The 2 GiB model alone covers it.
        let gate = MemoryGate::default();
        let probe = probe_with(1_536);
        let est = estimate(2_048, 0, 0);
        let resident = vec![loaded("planner", 2_048), loaded("router", 256)];
        let picks = gate.suggest_unloads(&probe, &est, 0, &resident);
        assert_eq!(picks, vec![ModelId::from("planner")]);
    }

    #[test]
    fn suggest_unloads_two_small_models_needed() {
        // free 0, need 2 GiB + 1 GiB margin = 3 GiB → deficit 3 GiB.
        // Largest is 2 GiB, then 1 GiB → both required.
        let gate = MemoryGate::default();
        let probe = probe_with(0);
        let est = estimate(2_048, 0, 0);
        let resident = vec![
            loaded("planner", 1_024),
            loaded("polisher", 2_048),
            loaded("router", 256),
        ];
        let picks = gate.suggest_unloads(&probe, &est, 0, &resident);
        // Largest first: polisher (2048), then planner (1024) — cumulative 3072 >= 3072.
        assert_eq!(
            picks,
            vec![ModelId::from("polisher"), ModelId::from("planner")]
        );
    }

    #[test]
    fn suggest_unloads_no_combination_possible_returns_empty() {
        // free 0, need 8 GiB + 1 GiB margin = 9 GiB. Resident pool is 1 GiB total.
        let gate = MemoryGate::default();
        let probe = probe_with(0);
        let est = estimate(8_192, 0, 0);
        let resident = vec![loaded("planner", 512), loaded("router", 512)];
        let picks = gate.suggest_unloads(&probe, &est, 0, &resident);
        assert!(picks.is_empty(), "got: {picks:?}");
    }

    #[test]
    fn suggest_unloads_orders_largest_first() {
        // Three residents; only the largest is needed.
        let gate = MemoryGate::default();
        let probe = probe_with(0);
        let est = estimate(2_000, 0, 0);
        let resident = vec![
            loaded("small", 100),
            loaded("medium", 1_000),
            loaded("big", 3_500),
        ];
        let picks = gate.suggest_unloads(&probe, &est, 0, &resident);
        assert_eq!(picks, vec![ModelId::from("big")]);
    }

    #[test]
    fn suggest_unloads_context_tokens_increase_pressure() {
        // At ctx=0 the candidate fits; at huge ctx it doesn't and we need
        // to suggest an unload.
        let gate = MemoryGate::default();
        let probe = probe_with(4_000);
        let est = estimate(2_000, 4096, 0);
        // Big enough resident that unloading it always satisfies the deficit.
        let resident = vec![loaded("planner", 8_192)];
        // At ctx=0: needed=2000, +margin 1024 → 3024 < 4000 avail → fits → no suggestion.
        assert!(gate.suggest_unloads(&probe, &est, 0, &resident).is_empty());
        // At 1M tokens, KV ≈ 3906 MiB → needed ≈ 5906 → deficit ≈ 2930 → planner covers.
        let picks = gate.suggest_unloads(&probe, &est, 1_000_000, &resident);
        assert_eq!(picks, vec![ModelId::from("planner")]);
    }

    #[test]
    fn suggest_unloads_respects_custom_margin() {
        // Lax margin → no unload needed; strict margin → unload needed.
        let lax = MemoryGate::new(0);
        let strict = MemoryGate::new(4_096);
        let probe = probe_with(4_500);
        let est = estimate(3_000, 0, 0);
        // Strict: required = 3000 + 4096 = 7096, avail 4500 → deficit 2596.
        // A 3000 MiB resident covers it; lax case fits outright.
        let resident = vec![loaded("router", 3_000)];
        assert!(lax.suggest_unloads(&probe, &est, 0, &resident).is_empty());
        let picks = strict.suggest_unloads(&probe, &est, 0, &resident);
        assert_eq!(picks, vec![ModelId::from("router")]);
    }

    #[test]
    fn check_with_ok_path_when_fits() {
        let gate = MemoryGate::default();
        let probe = probe_with(8_000);
        let est = estimate(2_000, 0, 0);
        let resident = vec![loaded("router", 500)];
        assert!(gate.check_with(&probe, &est, 0, &resident).is_ok());
    }

    #[test]
    fn check_with_refused_includes_unload_hint() {
        let gate = MemoryGate::default();
        let probe = probe_with(1_536);
        let est = estimate(2_048, 0, 0);
        let resident = vec![loaded("planner", 2_048)];
        let err = gate.check_with(&probe, &est, 0, &resident).unwrap_err();
        assert_eq!(err.code().as_str(), "STRAT-E3007");
        let rendered = format!("{err}");
        assert!(rendered.contains("hint: unload planner"), "{rendered}");
    }

    #[test]
    fn check_with_refused_falls_back_to_static_hint_when_no_combination() {
        let gate = MemoryGate::default();
        let probe = probe_with(0);
        let est = estimate(8_192, 0, 0);
        let resident = vec![loaded("router", 64)];
        let err = gate.check_with(&probe, &est, 0, &resident).unwrap_err();
        let rendered = format!("{err}");
        assert!(
            rendered.contains("free RAM or pick a smaller model"),
            "{rendered}"
        );
        assert!(!rendered.contains("hint: unload"), "{rendered}");
    }

    #[test]
    fn check_with_refused_empty_loaded_uses_static_hint() {
        let gate = MemoryGate::default();
        let probe = probe_with(400);
        let est = estimate(3_000, 0, 0);
        let err = gate.check_with(&probe, &est, 0, &[]).unwrap_err();
        let rendered = format!("{err}");
        assert!(
            rendered.contains("free RAM or pick a smaller model"),
            "{rendered}"
        );
    }

    #[test]
    fn loaded_model_round_trip_fields() {
        let lm = LoadedModel {
            id: ModelId::from("planner"),
            estimate: estimate(2_048, 0, 0),
            role_hint: Some("planner".into()),
        };
        assert_eq!(lm.id.as_str(), "planner");
        assert_eq!(lm.role_hint.as_deref(), Some("planner"));
        let cloned = lm.clone();
        assert_eq!(lm, cloned);
    }
}
