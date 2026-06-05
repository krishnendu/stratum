//! Turn-level observability primitives.
//!
//! Data side of `plan/34-observability-ux.md`. The chat TUI's Phase 3 panel
//! (token meter, per-role latency breakdown, live tok/s gauge) needs a stable
//! shape it can render; this module is that shape. Nothing here touches the
//! terminal — it is pure accumulation over the `Block` stream the runtime
//! already emits.
//!
//! Three small pieces:
//!
//! * [`TurnId`] and [`TurnIdGen`] — monotonic identifiers for a single
//!   user→assistant turn within a session.
//! * [`RoleTimer`] — a thin wrapper around [`std::time::Instant`] for timing
//!   one role-step (planner, coder, critic, polisher, …).
//! * [`TurnRecorder`] — accumulates [`Block`]s into a [`TurnMetrics`] record
//!   the TUI can render.
//!
//! Token-rate utility [`format_tokens_per_second`] handles the divide-by-zero
//! edge case the meter would otherwise have to guard against.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use serde::{Deserialize, Serialize};
use stratum_types::Block;
use time::format_description::well_known::Rfc3339;
use time::OffsetDateTime;

/// Monotonic identifier for a single turn within a session.
///
/// Issued by [`TurnIdGen::next`]; opaque to consumers — only equality and
/// serialization matter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct TurnId(pub u64);

/// Atomic counter that hands out fresh [`TurnId`]s.
///
/// Cheap to share across orchestrator tasks; `next` is wait-free.
#[derive(Debug, Default)]
pub struct TurnIdGen {
    counter: AtomicU64,
}

impl TurnIdGen {
    /// Build a generator that starts at zero.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            counter: AtomicU64::new(0),
        }
    }

    /// Issue the next id. Distinct from every previous id from this
    /// generator. Wait-free.
    pub fn next(&self) -> TurnId {
        TurnId(self.counter.fetch_add(1, Ordering::Relaxed))
    }
}

/// One role-step entry in the per-turn waterfall.
///
/// Matches the rows the Phase 3 latency panel (`plan/34-observability-ux.md`
/// §5) renders.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoleStep {
    /// Role label (e.g. `"planner"`, `"coder"`, `"critic"`, `"polisher"`).
    pub name: String,
    /// Wall-clock duration of the step in milliseconds.
    pub duration_ms: u32,
}

/// Immutable record of one turn, ready for the TUI to render.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TurnMetrics {
    /// Identifier of the turn.
    pub turn_id: TurnId,
    /// Latest reported cumulative prompt tokens.
    pub prompt_tokens: u32,
    /// Latest reported cumulative completion tokens.
    pub completion_tokens: u32,
    /// Number of `Block`s the recorder observed for this turn.
    pub total_blocks: u32,
    /// RFC3339 timestamp when the turn started.
    pub started_at: String,
    /// RFC3339 timestamp when the turn finished. Empty until [`TurnRecorder::finish`].
    pub completed_at: String,
    /// Per-role waterfall, preserved in insertion order.
    pub role_steps: Vec<RoleStep>,
}

/// Wrapper around [`Instant`] for timing one role-step.
///
/// Constructed at the start of the step; [`RoleTimer::stop_ms`] returns the
/// elapsed milliseconds (clamped to `u32::MAX`). Pure — no global clock, no
/// observable side effects beyond reading `Instant::now()`.
#[derive(Debug, Clone, Copy)]
pub struct RoleTimer {
    started: Instant,
}

impl RoleTimer {
    /// Start a new timer at `Instant::now()`.
    #[must_use]
    pub fn start() -> Self {
        Self {
            started: Instant::now(),
        }
    }

    /// Build a timer that started at the given `Instant`. Useful for tests
    /// that pin both ends.
    #[must_use]
    pub const fn started_at(started: Instant) -> Self {
        Self { started }
    }

    /// Elapsed milliseconds since the timer started, saturating at
    /// `u32::MAX`.
    #[must_use]
    pub fn stop_ms(self) -> u32 {
        let elapsed = self.started.elapsed().as_millis();
        u32::try_from(elapsed).unwrap_or(u32::MAX)
    }
}

impl Default for RoleTimer {
    fn default() -> Self {
        Self::start()
    }
}

/// Mutable accumulator that turns a stream of [`Block`]s into an immutable
/// [`TurnMetrics`] record.
///
/// Lifecycle:
/// 1. [`TurnRecorder::new`] — stamps `started_at` and begins counting.
/// 2. [`TurnRecorder::record_block`] — call once per emitted `Block`.
/// 3. [`TurnRecorder::record_step`] — call once per finished role-step.
/// 4. [`TurnRecorder::finish`] — stamps `completed_at` and yields the record.
#[derive(Debug)]
pub struct TurnRecorder {
    metrics: TurnMetrics,
}

impl TurnRecorder {
    /// Start a recorder for the given turn at the current wall-clock time.
    ///
    /// The wall-clock stamp uses [`OffsetDateTime::now_utc`] formatted as
    /// RFC3339 (mirroring the on-disk format used by `installed.toml`).
    #[must_use]
    pub fn new(turn_id: TurnId) -> Self {
        Self {
            metrics: TurnMetrics {
                turn_id,
                prompt_tokens: 0,
                completion_tokens: 0,
                total_blocks: 0,
                started_at: format_now_rfc3339(OffsetDateTime::now_utc()),
                completed_at: String::new(),
                role_steps: Vec::new(),
            },
        }
    }

    /// Build a recorder that pins the start instant explicitly. Used by
    /// tests so the RFC3339 stamp is reproducible; production callers
    /// should prefer [`TurnRecorder::new`].
    #[must_use]
    pub fn new_at(turn_id: TurnId, started_at: OffsetDateTime) -> Self {
        Self {
            metrics: TurnMetrics {
                turn_id,
                prompt_tokens: 0,
                completion_tokens: 0,
                total_blocks: 0,
                started_at: format_now_rfc3339(started_at),
                completed_at: String::new(),
                role_steps: Vec::new(),
            },
        }
    }

    /// Fold one [`Block`] into the running totals.
    ///
    /// * Every block bumps `total_blocks` by one.
    /// * Only [`Block::Usage`] updates token counters. The latest values
    ///   overwrite — `prompt`/`completion` on the wire are already cumulative
    ///   (`crates/stratum-types/src/block.rs`), so summing would double-count.
    pub const fn record_block(&mut self, block: &Block) {
        self.metrics.total_blocks = self.metrics.total_blocks.saturating_add(1);
        if let Block::Usage { prompt, completion } = block {
            self.metrics.prompt_tokens = *prompt;
            self.metrics.completion_tokens = *completion;
        }
    }

    /// Append a finished role-step to the waterfall.
    ///
    /// Insertion order is preserved; the TUI renders top-to-bottom.
    pub fn record_step(&mut self, name: impl Into<String>, duration_ms: u32) {
        self.metrics.role_steps.push(RoleStep {
            name: name.into(),
            duration_ms,
        });
    }

    /// Borrow the in-progress metrics. Cheap; useful for the live status bar
    /// that wants tok/s mid-turn without consuming the recorder.
    #[must_use]
    pub const fn snapshot(&self) -> &TurnMetrics {
        &self.metrics
    }

    /// Stamp `completed_at` with the current wall clock and return the
    /// immutable record.
    #[must_use]
    pub fn finish(self) -> TurnMetrics {
        self.finish_at(OffsetDateTime::now_utc())
    }

    /// Stamp `completed_at` with the given timestamp and return the
    /// immutable record. Tests pin the clock here; production callers should
    /// prefer [`TurnRecorder::finish`].
    #[must_use]
    pub fn finish_at(mut self, now: OffsetDateTime) -> TurnMetrics {
        self.metrics.completed_at = format_now_rfc3339(now);
        self.metrics
    }
}

/// Compute tokens-per-second from a completion count and an elapsed window.
///
/// Returns `0.0` when `elapsed_ms` is zero, side-stepping the divide-by-zero
/// the TUI gauge would otherwise have to guard. Saturates at `f32::INFINITY`
/// only via the underlying floating-point division (input `u32`s cannot
/// reach that on their own).
///
/// # Precision
/// Both inputs are `u32`; casting to `f32` loses precision above 2^24. For
/// realistic per-turn token counts and elapsed millisecond windows this is
/// well within the precision envelope, so the lossy cast is documented and
/// allowed.
#[must_use]
#[allow(
    clippy::cast_precision_loss,
    reason = "u32 → f32 precision is sufficient for token-rate display"
)]
pub fn format_tokens_per_second(completion_tokens: u32, elapsed_ms: u32) -> f32 {
    if elapsed_ms == 0 {
        return 0.0;
    }
    (completion_tokens as f32) * 1000.0 / (elapsed_ms as f32)
}

/// Internal RFC3339 formatter.
///
/// Mirrors the pattern in [`crate::install::InstalledToml::new`]:
/// `OffsetDateTime::format` with the [`Rfc3339`] description is infallible
/// for any well-formed `OffsetDateTime`, so the `expect` is carved out and
/// tracked in `docs/coverage-exclusions.md`.
#[allow(
    clippy::expect_used,
    reason = "OffsetDateTime::format with Rfc3339 is infallible"
)]
fn format_now_rfc3339(now: OffsetDateTime) -> String {
    now.format(&Rfc3339)
        .expect("Rfc3339 formatting of OffsetDateTime is infallible")
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    fn fixed_time() -> OffsetDateTime {
        OffsetDateTime::from_unix_timestamp(1_700_000_000).unwrap()
    }

    fn later_time() -> OffsetDateTime {
        OffsetDateTime::from_unix_timestamp(1_700_000_042).unwrap()
    }

    #[test]
    fn turn_id_generator_is_monotonic_and_distinct() {
        let gen = TurnIdGen::new();
        let a = gen.next();
        let b = gen.next();
        let c = gen.next();
        assert_ne!(a, b);
        assert_ne!(b, c);
        assert!(a < b && b < c);
        assert_eq!(a, TurnId(0));
        assert_eq!(c, TurnId(2));
    }

    #[test]
    fn turn_id_generator_default_starts_at_zero() {
        let gen = TurnIdGen::default();
        assert_eq!(gen.next(), TurnId(0));
    }

    #[test]
    fn record_block_accumulates_text_and_usage() {
        let mut rec = TurnRecorder::new_at(TurnId(7), fixed_time());
        rec.record_block(&Block::Text { text: "hi".into() });
        rec.record_block(&Block::Text {
            text: "there".into(),
        });
        rec.record_block(&Block::Usage {
            prompt: 4,
            completion: 2,
        });
        let snap = rec.snapshot();
        assert_eq!(snap.total_blocks, 3);
        assert_eq!(snap.prompt_tokens, 4);
        assert_eq!(snap.completion_tokens, 2);
    }

    #[test]
    fn record_block_usage_latest_overwrites_previous() {
        let mut rec = TurnRecorder::new_at(TurnId(0), fixed_time());
        rec.record_block(&Block::Usage {
            prompt: 10,
            completion: 5,
        });
        rec.record_block(&Block::Usage {
            prompt: 12,
            completion: 9,
        });
        rec.record_block(&Block::Usage {
            prompt: 20,
            completion: 17,
        });
        let snap = rec.snapshot();
        // Latest wins; would be 42/31 if summed.
        assert_eq!(snap.prompt_tokens, 20);
        assert_eq!(snap.completion_tokens, 17);
        assert_eq!(snap.total_blocks, 3);
    }

    #[test]
    fn record_block_done_and_cancelled_count_but_do_not_touch_tokens() {
        let mut rec = TurnRecorder::new_at(TurnId(0), fixed_time());
        rec.record_block(&Block::Usage {
            prompt: 3,
            completion: 7,
        });
        rec.record_block(&Block::Cancelled {
            reason: "STRAT-E4002 cancelled by user".into(),
        });
        rec.record_block(&Block::Done);
        let snap = rec.snapshot();
        assert_eq!(snap.total_blocks, 3);
        assert_eq!(snap.prompt_tokens, 3);
        assert_eq!(snap.completion_tokens, 7);
    }

    #[test]
    fn record_block_tool_blocks_count_but_do_not_touch_tokens() {
        let mut rec = TurnRecorder::new_at(TurnId(0), fixed_time());
        rec.record_block(&Block::ToolCall {
            id: "t1".into(),
            tool: "fs.read".into(),
            args: "{}".into(),
        });
        rec.record_block(&Block::ToolResult {
            id: "t1".into(),
            output: "ok".into(),
        });
        let snap = rec.snapshot();
        assert_eq!(snap.total_blocks, 2);
        assert_eq!(snap.prompt_tokens, 0);
        assert_eq!(snap.completion_tokens, 0);
    }

    #[test]
    fn record_step_preserves_insertion_order() {
        let mut rec = TurnRecorder::new_at(TurnId(0), fixed_time());
        rec.record_step("planner", 120);
        rec.record_step("coder", 3400);
        rec.record_step("critic", 1500);
        let snap = rec.snapshot();
        assert_eq!(snap.role_steps.len(), 3);
        assert_eq!(snap.role_steps[0].name, "planner");
        assert_eq!(snap.role_steps[1].name, "coder");
        assert_eq!(snap.role_steps[2].name, "critic");
    }

    #[test]
    fn record_step_accumulates_wall_time() {
        let mut rec = TurnRecorder::new_at(TurnId(0), fixed_time());
        rec.record_step("planner", 120);
        rec.record_step("coder", 3400);
        rec.record_step("critic", 1500);
        let total: u32 = rec
            .snapshot()
            .role_steps
            .iter()
            .map(|s| s.duration_ms)
            .sum();
        assert_eq!(total, 5020);
    }

    #[test]
    fn finish_at_stamps_completion_timestamp() {
        let rec = TurnRecorder::new_at(TurnId(1), fixed_time());
        let m = rec.finish_at(later_time());
        assert!(!m.completed_at.is_empty());
        assert_ne!(m.completed_at, m.started_at);
    }

    #[test]
    fn finish_uses_current_wall_clock() {
        // finish() (no _at) should still produce a non-empty RFC3339 stamp.
        let rec = TurnRecorder::new(TurnId(99));
        let m = rec.finish();
        assert!(!m.completed_at.is_empty());
        assert_eq!(m.turn_id, TurnId(99));
    }

    #[test]
    fn snapshot_borrows_without_consuming() {
        let mut rec = TurnRecorder::new_at(TurnId(0), fixed_time());
        rec.record_block(&Block::Text { text: "x".into() });
        let _first = rec.snapshot();
        rec.record_block(&Block::Text { text: "y".into() });
        assert_eq!(rec.snapshot().total_blocks, 2);
    }

    #[test]
    fn format_tokens_per_second_happy_path() {
        // 412 tok in 1200 ms ≈ 343 tok/s (matches the example in
        // `plan/34-observability-ux.md` §4).
        let rate = format_tokens_per_second(412, 1200);
        assert!((rate - 343.333_3).abs() < 0.01);
    }

    #[test]
    fn format_tokens_per_second_zero_elapsed_returns_zero() {
        assert!((format_tokens_per_second(100, 0)).abs() < f32::EPSILON);
        assert!((format_tokens_per_second(0, 0)).abs() < f32::EPSILON);
    }

    #[test]
    fn format_tokens_per_second_zero_completion_returns_zero() {
        assert!((format_tokens_per_second(0, 1000)).abs() < f32::EPSILON);
    }

    #[test]
    fn format_tokens_per_second_handles_large_inputs_without_panic() {
        // Saturating overflow edge: largest possible inputs must not panic
        // and must produce a finite, non-negative result.
        let rate = format_tokens_per_second(u32::MAX, u32::MAX);
        assert!(rate.is_finite());
        assert!(rate >= 0.0);
    }

    #[test]
    fn format_tokens_per_second_small_window_does_not_panic() {
        // 1-ms windows are the smallest non-zero case the live gauge will
        // hit; verify it remains finite.
        let rate = format_tokens_per_second(50, 1);
        assert!((rate - 50_000.0).abs() < 1.0);
    }

    #[test]
    fn role_timer_records_positive_duration() {
        let t = RoleTimer::start();
        std::thread::sleep(Duration::from_millis(2));
        let ms = t.stop_ms();
        assert!(ms >= 1, "expected at least 1ms, got {ms}");
    }

    #[test]
    fn role_timer_started_at_uses_supplied_instant() {
        let pinned = Instant::now();
        let t = RoleTimer::started_at(pinned);
        // Cannot beat Instant monotonicity: elapsed must be >= 0.
        let _ms = t.stop_ms();
    }

    #[test]
    fn role_timer_default_starts_immediately() {
        let t = RoleTimer::default();
        // No panic, monotonic.
        let _ = t.stop_ms();
    }

    #[test]
    fn turn_metrics_serde_roundtrip() {
        let mut rec = TurnRecorder::new_at(TurnId(13), fixed_time());
        rec.record_block(&Block::Usage {
            prompt: 5,
            completion: 9,
        });
        rec.record_step("planner", 100);
        let m = rec.finish_at(later_time());
        let s = serde_json::to_string(&m).unwrap();
        let back: TurnMetrics = serde_json::from_str(&s).unwrap();
        assert_eq!(m, back);
    }

    #[test]
    fn role_step_serde_roundtrip() {
        let step = RoleStep {
            name: "coder".into(),
            duration_ms: 3400,
        };
        let s = serde_json::to_string(&step).unwrap();
        let back: RoleStep = serde_json::from_str(&s).unwrap();
        assert_eq!(step, back);
    }

    #[test]
    fn turn_id_serde_roundtrip() {
        let id = TurnId(42);
        let s = serde_json::to_string(&id).unwrap();
        let back: TurnId = serde_json::from_str(&s).unwrap();
        assert_eq!(id, back);
    }

    #[test]
    fn turn_recorder_new_records_started_at() {
        let rec = TurnRecorder::new(TurnId(0));
        let snap = rec.snapshot();
        assert!(!snap.started_at.is_empty());
        assert!(snap.completed_at.is_empty());
        assert_eq!(snap.total_blocks, 0);
        assert!(snap.role_steps.is_empty());
    }
}
