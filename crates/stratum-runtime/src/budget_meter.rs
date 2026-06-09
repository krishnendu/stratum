//! Session-level cumulative budget meter.
//!
//! Complements [`crate::budget::BudgetTracker`] — that one enforces a single
//! turn against an [`crate::agents::AgentBudget`]. This one accumulates per-turn
//! metrics into per-session totals + per-role breakdowns, with an optional
//! dollar-equivalent hard cap. Designed for the future TUI status bar and the
//! eval-report renderer.
//!
//! All counters are `u64`. Money is tracked in micro-USD so $0.001 = 1_000
//! micro-USD; tokens-per-million pricing maps cleanly via
//! [`estimate_cost_micro_usd`].

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;
use std::sync::Mutex;
use std::time::SystemTime;

use serde::{Deserialize, Serialize};

/// Cumulative usage counters.
///
/// Every field is monotone non-decreasing for a given meter until [`BudgetMeter::reset`]
/// is called. Serializable so the TUI / eval reporter can stash snapshots in JSON.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BudgetTotals {
    /// Prompt-side tokens billed so far.
    pub prompt_tokens: u64,
    /// Completion-side tokens billed so far.
    pub completion_tokens: u64,
    /// Number of tool calls executed.
    pub tool_calls: u64,
    /// Wall-clock spent across all recorded turns, in milliseconds.
    pub wall_ms: u64,
    /// Dollar-equivalent cost, in micro-USD (`$1` = `1_000_000`).
    pub cost_micro_usd: u64,
}

impl BudgetTotals {
    /// Add `other`'s fields into `self`, checking for u64 overflow on each lane.
    fn checked_add_assign(&mut self, other: &Self) -> Result<(), BudgetMeterError> {
        let prompt = self
            .prompt_tokens
            .checked_add(other.prompt_tokens)
            .ok_or(BudgetMeterError::ArithmeticOverflow)?;
        let completion = self
            .completion_tokens
            .checked_add(other.completion_tokens)
            .ok_or(BudgetMeterError::ArithmeticOverflow)?;
        let tool_calls = self
            .tool_calls
            .checked_add(other.tool_calls)
            .ok_or(BudgetMeterError::ArithmeticOverflow)?;
        let wall = self
            .wall_ms
            .checked_add(other.wall_ms)
            .ok_or(BudgetMeterError::ArithmeticOverflow)?;
        let cost = self
            .cost_micro_usd
            .checked_add(other.cost_micro_usd)
            .ok_or(BudgetMeterError::ArithmeticOverflow)?;
        self.prompt_tokens = prompt;
        self.completion_tokens = completion;
        self.tool_calls = tool_calls;
        self.wall_ms = wall;
        self.cost_micro_usd = cost;
        Ok(())
    }
}

/// Session-level cumulative meter.
///
/// Aggregates [`BudgetTotals`] across many turns and many agent roles. Holds
/// two mutexes (totals + per-role map) and the session start time. An optional
/// `hard_cap_micro_usd` lets callers refuse further work once spend crosses a
/// threshold — the breaching turn is still recorded (atomic add then check)
/// so totals stay truthful.
#[derive(Debug)]
pub struct BudgetMeter {
    totals: Mutex<BudgetTotals>,
    per_role: Mutex<BTreeMap<String, BudgetTotals>>,
    started_at: SystemTime,
    hard_cap_micro_usd: Option<u64>,
}

impl BudgetMeter {
    /// Build a fresh meter with all counters at zero and no hard cap.
    #[must_use]
    pub fn new() -> Self {
        Self {
            totals: Mutex::new(BudgetTotals::default()),
            per_role: Mutex::new(BTreeMap::new()),
            started_at: SystemTime::now(),
            hard_cap_micro_usd: None,
        }
    }

    /// Attach a hard spend ceiling, in micro-USD. Once the running total
    /// exceeds `micro_usd`, [`Self::record_turn`] returns
    /// [`BudgetMeterError::HardCapExceeded`] (and still records the turn).
    #[must_use]
    pub const fn with_hard_cap(mut self, micro_usd: u64) -> Self {
        self.hard_cap_micro_usd = Some(micro_usd);
        self
    }

    /// Record a completed turn's metrics under `role`.
    ///
    /// Adds to both the session-wide totals and the per-role bucket. The
    /// transaction is atomic — if any lane would wrap u64 the meter is left
    /// untouched (and [`BudgetMeterError::ArithmeticOverflow`] is returned).
    /// If a configured hard cap is crossed the turn is still committed and
    /// [`BudgetMeterError::HardCapExceeded`] is returned.
    ///
    /// # Errors
    ///
    /// - [`BudgetMeterError::ArithmeticOverflow`] when a u64 counter lane
    ///   would wrap. State is left unchanged in this case.
    /// - [`BudgetMeterError::HardCapExceeded`] when the post-record cost
    ///   crosses the configured hard cap. The turn is still recorded.
    pub fn record_turn(
        &self,
        role: &str,
        prompt_tokens: u64,
        completion_tokens: u64,
        tool_calls: u64,
        wall_ms: u64,
        cost_micro_usd: u64,
    ) -> Result<(), BudgetMeterError> {
        let delta = BudgetTotals {
            prompt_tokens,
            completion_tokens,
            tool_calls,
            wall_ms,
            cost_micro_usd,
        };

        // Lock both maps before touching either, so an overflow on the per-role
        // lane doesn't leave totals advanced (or vice versa). The locks are
        // scoped tightly so we don't hold them across the hard-cap check.
        let mut totals = self.lock_totals();
        let mut per_role = self.lock_per_role();

        // Probe both lanes for overflow first; only mutate state once both
        // add cleanly.
        let mut next_totals = *totals;
        next_totals.checked_add_assign(&delta)?;
        let existing = per_role.get(role).copied().unwrap_or_default();
        let mut next_role = existing;
        next_role.checked_add_assign(&delta)?;

        *totals = next_totals;
        per_role.insert(role.to_string(), next_role);
        let next_cost = next_totals.cost_micro_usd;
        drop(per_role);
        drop(totals);

        if let Some(cap) = self.hard_cap_micro_usd {
            if next_cost > cap {
                return Err(BudgetMeterError::HardCapExceeded {
                    current: next_cost,
                    cap,
                });
            }
        }
        Ok(())
    }

    /// Snapshot the session-wide totals.
    #[must_use]
    pub fn totals(&self) -> BudgetTotals {
        *self.lock_totals()
    }

    /// Snapshot the per-role breakdown, sorted by role (`BTreeMap` iteration order).
    #[must_use]
    pub fn per_role_breakdown(&self) -> BTreeMap<String, BudgetTotals> {
        self.lock_per_role().clone()
    }

    /// Wall-clock the meter was constructed.
    #[must_use]
    pub const fn started_at(&self) -> SystemTime {
        self.started_at
    }

    /// Seconds elapsed since [`Self::new`]. Saturates to 0 if the clock has
    /// jumped backwards.
    #[must_use]
    pub fn elapsed_secs(&self) -> u64 {
        self.started_at.elapsed().map(|d| d.as_secs()).unwrap_or(0)
    }

    /// Zero out totals and per-role breakdown. Does not reset `started_at`.
    pub fn reset(&self) {
        *self.lock_totals() = BudgetTotals::default();
        self.lock_per_role().clear();
    }

    /// Acquire the totals lock; on a poisoned mutex, take the inner value.
    fn lock_totals(&self) -> std::sync::MutexGuard<'_, BudgetTotals> {
        match self.totals.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        }
    }

    /// Acquire the per-role lock; on a poisoned mutex, take the inner value.
    fn lock_per_role(&self) -> std::sync::MutexGuard<'_, BTreeMap<String, BudgetTotals>> {
        match self.per_role.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        }
    }

    /// Test-only: pre-seed the totals so we can drive the overflow path.
    #[cfg(test)]
    pub(crate) fn set_totals_for_test(&self, t: BudgetTotals) {
        *self.lock_totals() = t;
    }
}

impl Default for BudgetMeter {
    fn default() -> Self {
        Self::new()
    }
}

/// Errors surfaced by [`BudgetMeter::record_turn`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BudgetMeterError {
    /// The configured hard-cap (in micro-USD) was crossed. The crossing turn
    /// is already recorded — `current` is the post-add running total.
    HardCapExceeded {
        /// Post-record cumulative cost in micro-USD.
        current: u64,
        /// Configured ceiling in micro-USD.
        cap: u64,
    },
    /// A counter lane would have wrapped around `u64::MAX`. The meter is left
    /// untouched in this case.
    ArithmeticOverflow,
}

impl fmt::Display for BudgetMeterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HardCapExceeded { current, cap } => write!(
                f,
                "hard cap exceeded: cost {current} micro-USD > cap {cap} micro-USD"
            ),
            Self::ArithmeticOverflow => f.write_str("budget meter counter overflow (u64)"),
        }
    }
}

impl Error for BudgetMeterError {}

/// Estimate the dollar-equivalent cost of a turn, in micro-USD.
///
/// `per_million_prompt_micro` / `per_million_completion_micro` are the
/// per-million-token rates expressed in micro-USD (so `$3` per million prompt
/// tokens = `3_000_000`). All arithmetic is saturating, so extreme inputs cap
/// at `u64::MAX` rather than wrapping.
#[must_use]
pub const fn estimate_cost_micro_usd(
    prompt_tokens: u64,
    completion_tokens: u64,
    per_million_prompt_micro: u64,
    per_million_completion_micro: u64,
) -> u64 {
    let prompt_cost = prompt_tokens.saturating_mul(per_million_prompt_micro);
    let completion_cost = completion_tokens.saturating_mul(per_million_completion_micro);
    let total = prompt_cost.saturating_add(completion_cost);
    total / 1_000_000
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    use super::*;

    #[test]
    fn budget_totals_default_is_all_zero() {
        let t = BudgetTotals::default();
        assert_eq!(t.prompt_tokens, 0);
        assert_eq!(t.completion_tokens, 0);
        assert_eq!(t.tool_calls, 0);
        assert_eq!(t.wall_ms, 0);
        assert_eq!(t.cost_micro_usd, 0);
    }

    #[test]
    fn budget_meter_new_starts_at_zero() {
        let m = BudgetMeter::new();
        assert_eq!(m.totals(), BudgetTotals::default());
        assert!(m.per_role_breakdown().is_empty());
    }

    #[test]
    fn record_turn_accumulates_into_totals() {
        let m = BudgetMeter::new();
        m.record_turn("planner", 10, 5, 1, 100, 42).expect("ok");
        m.record_turn("planner", 4, 6, 2, 50, 8).expect("ok");
        let t = m.totals();
        assert_eq!(t.prompt_tokens, 14);
        assert_eq!(t.completion_tokens, 11);
        assert_eq!(t.tool_calls, 3);
        assert_eq!(t.wall_ms, 150);
        assert_eq!(t.cost_micro_usd, 50);
    }

    #[test]
    fn record_turn_accumulates_into_per_role() {
        let m = BudgetMeter::new();
        m.record_turn("planner", 10, 5, 1, 100, 42).expect("ok");
        m.record_turn("planner", 4, 6, 2, 50, 8).expect("ok");
        let by_role = m.per_role_breakdown();
        let p = by_role.get("planner").copied().expect("planner present");
        assert_eq!(p.prompt_tokens, 14);
        assert_eq!(p.completion_tokens, 11);
        assert_eq!(p.tool_calls, 3);
        assert_eq!(p.wall_ms, 150);
        assert_eq!(p.cost_micro_usd, 50);
    }

    #[test]
    fn multiple_roles_each_get_their_own_bucket() {
        let m = BudgetMeter::new();
        m.record_turn("planner", 10, 5, 0, 100, 1).expect("ok");
        m.record_turn("worker", 20, 7, 1, 200, 2).expect("ok");
        m.record_turn("polisher", 1, 1, 0, 10, 0).expect("ok");
        let by_role = m.per_role_breakdown();
        assert_eq!(by_role.len(), 3);
        assert_eq!(
            by_role
                .get("planner")
                .copied()
                .expect("planner")
                .prompt_tokens,
            10
        );
        assert_eq!(
            by_role
                .get("worker")
                .copied()
                .expect("worker")
                .prompt_tokens,
            20
        );
        assert_eq!(
            by_role
                .get("polisher")
                .copied()
                .expect("polisher")
                .prompt_tokens,
            1
        );
        // Totals fan-in correctly.
        assert_eq!(m.totals().prompt_tokens, 31);
    }

    #[test]
    fn totals_returns_independent_snapshot() {
        let m = BudgetMeter::new();
        m.record_turn("planner", 5, 5, 0, 0, 0).expect("ok");
        let snap_a = m.totals();
        m.record_turn("planner", 5, 5, 0, 0, 0).expect("ok");
        let snap_b = m.totals();
        // The old snapshot must not be mutated by later record_turn calls.
        assert_eq!(snap_a.prompt_tokens, 5);
        assert_eq!(snap_b.prompt_tokens, 10);
    }

    #[test]
    fn per_role_breakdown_is_sorted_by_role() {
        let m = BudgetMeter::new();
        // Insert in non-alphabetic order.
        m.record_turn("zeta", 1, 0, 0, 0, 0).expect("ok");
        m.record_turn("alpha", 1, 0, 0, 0, 0).expect("ok");
        m.record_turn("mu", 1, 0, 0, 0, 0).expect("ok");
        let by_role = m.per_role_breakdown();
        let keys: Vec<&String> = by_role.keys().collect();
        assert_eq!(
            keys,
            vec![&"alpha".to_string(), &"mu".to_string(), &"zeta".to_string()]
        );
    }

    #[test]
    fn reset_zeroes_everything() {
        let m = BudgetMeter::new();
        m.record_turn("planner", 10, 5, 1, 100, 42).expect("ok");
        m.record_turn("worker", 1, 1, 1, 1, 1).expect("ok");
        m.reset();
        assert_eq!(m.totals(), BudgetTotals::default());
        assert!(m.per_role_breakdown().is_empty());
    }

    #[test]
    fn no_hard_cap_means_never_errors() {
        let m = BudgetMeter::new();
        // A turn that would massively exceed any sane spend should still be Ok.
        m.record_turn("planner", 0, 0, 0, 0, 999_999_999_999)
            .expect("ok");
        assert_eq!(m.totals().cost_micro_usd, 999_999_999_999);
    }

    #[test]
    fn hard_cap_under_threshold_returns_ok() {
        let m = BudgetMeter::new().with_hard_cap(100_000);
        m.record_turn("planner", 0, 0, 0, 0, 50_000).expect("ok");
        m.record_turn("planner", 0, 0, 0, 0, 50_000).expect("ok");
        // Exactly at the cap is not over — must still be Ok.
        assert_eq!(m.totals().cost_micro_usd, 100_000);
    }

    #[test]
    fn hard_cap_breach_returns_hardcap_with_correct_values_and_still_records() {
        let m = BudgetMeter::new().with_hard_cap(100_000);
        m.record_turn("planner", 0, 0, 0, 0, 90_000).expect("ok");
        let err = m
            .record_turn("planner", 0, 0, 0, 0, 20_000)
            .expect_err("expected HardCapExceeded");
        match err {
            BudgetMeterError::HardCapExceeded { current, cap } => {
                assert_eq!(current, 110_000);
                assert_eq!(cap, 100_000);
            }
            other @ BudgetMeterError::ArithmeticOverflow => {
                panic!("expected HardCapExceeded, got {other:?}")
            }
        }
        // Even on the breach, the turn is recorded.
        assert_eq!(m.totals().cost_micro_usd, 110_000);
    }

    #[test]
    fn elapsed_secs_is_nonneg_immediately_after_new() {
        let m = BudgetMeter::new();
        // u64 is unsigned so the only check that makes sense: it's at most a
        // tiny number right after construction.
        let s = m.elapsed_secs();
        assert!(s < 60);
    }

    #[test]
    fn elapsed_secs_can_advance_after_sleep() {
        let m = BudgetMeter::new();
        let before = m.elapsed_secs();
        thread::sleep(Duration::from_millis(1_100));
        let after = m.elapsed_secs();
        assert!(after >= before);
        // We slept ~1.1s so the elapsed must have moved forward by at least 1s.
        assert!(after >= 1);
    }

    #[test]
    fn arithmetic_overflow_returns_error_and_leaves_state_untouched() {
        let m = BudgetMeter::new();
        m.set_totals_for_test(BudgetTotals {
            prompt_tokens: u64::MAX - 1,
            completion_tokens: 0,
            tool_calls: 0,
            wall_ms: 0,
            cost_micro_usd: 0,
        });
        let err = m
            .record_turn("planner", 2, 0, 0, 0, 0)
            .expect_err("expected ArithmeticOverflow");
        assert_eq!(err, BudgetMeterError::ArithmeticOverflow);
        // State unchanged.
        assert_eq!(m.totals().prompt_tokens, u64::MAX - 1);
        assert!(m.per_role_breakdown().is_empty());
    }

    #[test]
    fn estimate_cost_prompt_only() {
        // 1M prompt tokens @ 100 micro-USD per M = 100 micro-USD.
        let cost = estimate_cost_micro_usd(1_000_000, 0, 100, 0);
        assert_eq!(cost, 100);
    }

    #[test]
    fn estimate_cost_completion_only() {
        let cost = estimate_cost_micro_usd(0, 1_000_000, 0, 200);
        assert_eq!(cost, 200);
    }

    #[test]
    fn estimate_cost_mixed_rate() {
        // 0.5M prompt @ 100, 0.5M completion @ 200 → (50M + 100M)/1M = 150
        let cost = estimate_cost_micro_usd(500_000, 500_000, 100, 200);
        assert_eq!(cost, 150);
    }

    #[test]
    fn estimate_cost_saturates_on_extreme_inputs() {
        // u64::MAX tokens times any non-zero rate would overflow; we want a
        // capped, finite, never-panicking number.
        let cost = estimate_cost_micro_usd(u64::MAX, u64::MAX, u64::MAX, u64::MAX);
        // Worst case: prompt = u64::MAX, completion saturating-added to u64::MAX,
        // sum saturates to u64::MAX, divided by 1_000_000.
        assert_eq!(cost, u64::MAX / 1_000_000);
    }

    #[test]
    fn budget_meter_error_display_smoke_each_variant() {
        let cap = BudgetMeterError::HardCapExceeded {
            current: 200,
            cap: 100,
        };
        let s = format!("{cap}");
        assert!(s.contains("200"));
        assert!(s.contains("100"));
        assert!(s.contains("cap"));

        let ovf = BudgetMeterError::ArithmeticOverflow;
        let s = format!("{ovf}");
        assert!(s.contains("overflow"));
    }

    #[test]
    fn started_at_round_trip_is_stable() {
        let m = BudgetMeter::new();
        let a = m.started_at();
        let b = m.started_at();
        assert_eq!(a, b);
    }

    #[test]
    fn concurrent_record_turn_is_consistent() {
        let m = Arc::new(BudgetMeter::new());
        let mut handles = Vec::new();
        for _ in 0..4 {
            let m2 = Arc::clone(&m);
            handles.push(thread::spawn(move || {
                for _ in 0..25 {
                    m2.record_turn("planner", 2, 3, 1, 5, 7).expect("ok");
                }
            }));
        }
        for h in handles {
            h.join().expect("join");
        }
        let t = m.totals();
        // 4 threads * 25 turns = 100 turns.
        assert_eq!(t.prompt_tokens, 100 * 2);
        assert_eq!(t.completion_tokens, 100 * 3);
        assert_eq!(t.tool_calls, 100);
        assert_eq!(t.wall_ms, 100 * 5);
        assert_eq!(t.cost_micro_usd, 100 * 7);
        let by_role = m.per_role_breakdown();
        let p = by_role.get("planner").copied().expect("planner present");
        assert_eq!(p.prompt_tokens, 100 * 2);
    }

    #[test]
    fn default_impl_matches_new() {
        let a = BudgetMeter::default();
        let b = BudgetMeter::new();
        assert_eq!(a.totals(), b.totals());
        assert!(a.per_role_breakdown().is_empty());
        assert!(b.per_role_breakdown().is_empty());
    }

    #[test]
    fn error_trait_source_is_none() {
        let e: &dyn Error = &BudgetMeterError::ArithmeticOverflow;
        assert!(e.source().is_none());
    }

    #[test]
    fn budget_totals_serde_roundtrip() {
        let t = BudgetTotals {
            prompt_tokens: 10,
            completion_tokens: 20,
            tool_calls: 3,
            wall_ms: 400,
            cost_micro_usd: 5_000,
        };
        let s = serde_json::to_string(&t).expect("serialize");
        let back: BudgetTotals = serde_json::from_str(&s).expect("deserialize");
        assert_eq!(back, t);
    }
}
