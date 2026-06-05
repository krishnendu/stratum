//! Per-tool-call timeout enforcement primitives.
//!
//! Composes with [`crate::cancel_cascade::CascadeToken`]: the cascade owns
//! cooperative cancellation; this module owns the per-call deadline policy
//! and the after-the-fact "this tool exceeded its budget" event surface.
//!
//! Three layers:
//!
//! * [`ToolTimeoutPolicy`] — pure data; default 30 s per tool, hard ceiling
//!   of 5 min. Per-tool overrides clamp to the ceiling on read.
//! * [`ToolTimeoutGuard`] — RAII timer thread that flips an
//!   [`AtomicBool`] and invokes a user callback if `disarm` isn't called
//!   before the deadline elapses.
//! * [`run_with_timeout`] / [`record_outcome`] — synchronous helpers that
//!   classify the result of an already-completed operation as `Ok`,
//!   `TimedOut`, or `Panicked`.
//!
//! **Important**: a userland timer cannot interrupt a blocking call. The
//! timer can only *flag* that the deadline has passed. Cooperative tool
//! implementations must periodically check
//! [`CascadeToken::is_cancelled`](crate::cancel_cascade::CascadeToken::is_cancelled)
//! and abort their own work. This module exists to (a) drive that flag,
//! (b) record the timeout event after the tool returns, and (c) provide a
//! deterministic test surface that does not depend on an async runtime.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

/// Default per-tool deadline: 30 seconds.
const DEFAULT_TOOL_DEADLINE: Duration = Duration::from_secs(30);

/// Hard upper bound on any per-tool deadline: 5 minutes.
const DEFAULT_MAX_OVERALL: Duration = Duration::from_secs(300);

/// Per-call timeout policy.
///
/// A `default` deadline applies to every tool unless an entry in
/// `overrides` says otherwise. Any deadline (default or override) is
/// clamped to `max_overall` at read time, so callers cannot accidentally
/// hand a tool a 30-minute budget by editing a config file.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ToolTimeoutPolicy {
    /// Deadline applied when a tool id is not in `overrides`.
    pub default: Duration,
    /// Tool-id-specific deadlines. Read-time clamped to `max_overall`.
    pub overrides: BTreeMap<String, Duration>,
    /// Hard ceiling. Both `default` and any `override` are clamped to this.
    pub max_overall: Duration,
}

impl Default for ToolTimeoutPolicy {
    fn default() -> Self {
        Self {
            default: DEFAULT_TOOL_DEADLINE,
            overrides: BTreeMap::new(),
            max_overall: DEFAULT_MAX_OVERALL,
        }
    }
}

impl ToolTimeoutPolicy {
    /// Resolve the deadline for `tool_id`.
    ///
    /// Returns the override if present, otherwise [`Self::default`].
    /// The result is always clamped to [`Self::max_overall`].
    #[must_use]
    pub fn deadline_for(&self, tool_id: &str) -> Duration {
        let raw = self.overrides.get(tool_id).copied().unwrap_or(self.default);
        raw.min(self.max_overall)
    }
}

/// RAII timer that fires `on_fire` if `disarm` isn't called in time.
///
/// `arm` spawns a sleeping thread. `disarm` signals that thread and joins
/// it, returning whether the deadline fired. Dropping the guard without
/// calling `disarm` signals the thread to exit but does *not* join — so
/// drop is non-blocking.
pub struct ToolTimeoutGuard {
    handle: Option<thread::JoinHandle<()>>,
    signal: Arc<AtomicBool>,
    deadline_hit: Arc<AtomicBool>,
    started_at: Instant,
    tool_id: String,
    deadline: Duration,
}

impl fmt::Debug for ToolTimeoutGuard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ToolTimeoutGuard")
            .field("tool_id", &self.tool_id)
            .field("deadline", &self.deadline)
            .field("deadline_hit", &self.deadline_hit.load(Ordering::SeqCst))
            .finish_non_exhaustive()
    }
}

impl ToolTimeoutGuard {
    /// Arm the guard. Spawns a timer thread that fires `on_fire` after
    /// `deadline`, unless [`Self::disarm`] is called first.
    ///
    /// `on_fire` runs on the timer thread.
    pub fn arm<F>(tool_id: String, deadline: Duration, on_fire: F) -> Self
    where
        F: FnOnce() + Send + 'static,
    {
        let signal = Arc::new(AtomicBool::new(false));
        let deadline_hit = Arc::new(AtomicBool::new(false));
        let started_at = Instant::now();

        let thread_signal = Arc::clone(&signal);
        let thread_hit = Arc::clone(&deadline_hit);
        let handle = thread::spawn(move || {
            // Poll the signal in small slices so disarm doesn't have to wait
            // out the full deadline. 10 ms is fine: the timer is for
            // multi-second tool budgets, not microsecond precision.
            let slice = Duration::from_millis(10);
            let mut remaining = deadline;
            while remaining > Duration::ZERO {
                if thread_signal.load(Ordering::SeqCst) {
                    return;
                }
                let sleep = remaining.min(slice);
                thread::sleep(sleep);
                remaining = remaining.saturating_sub(sleep);
            }
            if !thread_signal.load(Ordering::SeqCst) {
                thread_hit.store(true, Ordering::SeqCst);
                on_fire();
            }
        });

        Self {
            handle: Some(handle),
            signal,
            deadline_hit,
            started_at,
            tool_id,
            deadline,
        }
    }

    /// Signal the timer thread to exit and join it.
    ///
    /// Returns `true` if the deadline fired before disarm, `false`
    /// otherwise.
    #[must_use = "the boolean indicates whether the deadline fired; ignoring it loses that signal"]
    pub fn disarm(mut self) -> bool {
        self.signal.store(true, Ordering::SeqCst);
        if let Some(handle) = self.handle.take() {
            // Join errors here mean the timer thread panicked. We can't
            // do anything useful with that — the deadline_hit flag is
            // still authoritative for whether the deadline fired.
            let _ = handle.join();
        }
        self.deadline_hit.load(Ordering::SeqCst)
    }

    /// Wall-clock time since [`Self::arm`].
    #[must_use]
    pub fn elapsed(&self) -> Duration {
        self.started_at.elapsed()
    }

    /// Time remaining until the deadline.
    ///
    /// `None` once `elapsed >= deadline`.
    #[must_use]
    pub fn time_remaining(&self) -> Option<Duration> {
        self.deadline.checked_sub(self.elapsed())
    }

    /// The tool id this guard was armed for.
    #[must_use]
    pub fn tool_id(&self) -> &str {
        &self.tool_id
    }

    /// The deadline this guard was armed with.
    #[must_use]
    pub const fn deadline(&self) -> Duration {
        self.deadline
    }
}

impl Drop for ToolTimeoutGuard {
    fn drop(&mut self) {
        // Signal the timer thread to exit. We do NOT join — drop should
        // not block, and the thread will exit on its own within one slice
        // (~10 ms) once it sees the signal.
        self.signal.store(true, Ordering::SeqCst);
    }
}

/// Run `op` on the calling thread under a timer.
///
/// The timer cannot interrupt `op`. It flips a shared flag after
/// `deadline` elapses; once `op` returns we classify the result:
///
/// * if the flag fired and `op` ran longer than `deadline`, returns
///   [`ToolTimeoutError::TimedOut`];
/// * otherwise returns `Ok(op_result)`.
///
/// Cooperative `op` implementations should periodically poll
/// [`CascadeToken::is_cancelled`](crate::cancel_cascade::CascadeToken::is_cancelled)
/// — this helper alone cannot stop a runaway blocking call.
///
/// # Errors
///
/// Returns [`ToolTimeoutError::TimedOut`] if `op` ran past `deadline`.
pub fn run_with_timeout<T, F>(
    tool_id: &str,
    deadline: Duration,
    op: F,
) -> Result<T, ToolTimeoutError>
where
    F: FnOnce() -> T + Send,
    T: Send,
{
    let flag = Arc::new(AtomicBool::new(false));
    let stop = Arc::new(AtomicBool::new(false));

    let timer_flag = Arc::clone(&flag);
    let timer_stop = Arc::clone(&stop);
    let timer = thread::spawn(move || {
        let slice = Duration::from_millis(10);
        let mut remaining = deadline;
        loop {
            if timer_stop.load(Ordering::SeqCst) {
                return;
            }
            if remaining == Duration::ZERO {
                timer_flag.store(true, Ordering::SeqCst);
                return;
            }
            let sleep = remaining.min(slice);
            thread::sleep(sleep);
            remaining = remaining.saturating_sub(sleep);
        }
    });

    let started = Instant::now();
    let value = op();
    let elapsed = started.elapsed();

    stop.store(true, Ordering::SeqCst);
    let _ = timer.join();

    if flag.load(Ordering::SeqCst) && elapsed > deadline {
        return Err(ToolTimeoutError::TimedOut {
            tool_id: tool_id.to_string(),
            deadline,
            elapsed,
        });
    }
    Ok(value)
}

/// Classification error for a completed tool call.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ToolTimeoutError {
    /// The tool ran past its deadline.
    TimedOut {
        /// Tool id reported by the caller.
        tool_id: String,
        /// Deadline that was exceeded.
        deadline: Duration,
        /// Actual wall-clock elapsed time.
        elapsed: Duration,
    },
    /// The tool returned an in-band error (panic-equivalent bucket).
    Panicked {
        /// Tool id reported by the caller.
        tool_id: String,
    },
}

impl fmt::Display for ToolTimeoutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TimedOut {
                tool_id,
                deadline,
                elapsed,
            } => write!(
                f,
                "tool `{tool_id}` exceeded its {deadline:?} deadline (elapsed: {elapsed:?})"
            ),
            Self::Panicked { tool_id } => write!(f, "tool `{tool_id}` failed during execution"),
        }
    }
}

impl Error for ToolTimeoutError {}

/// Pure helper that classifies an already-completed tool call.
///
/// * `op_result = Err(_)` → [`ToolTimeoutError::Panicked`] (the catch-all
///   bucket for any in-op failure; the runtime treats it the same as a
///   panic for budget accounting).
/// * `elapsed > deadline` → [`ToolTimeoutError::TimedOut`].
/// * otherwise → `Ok(())`.
///
/// Provided as a separate function so the event-log integration can be
/// tested without spawning threads.
///
/// # Errors
///
/// Returns [`ToolTimeoutError::Panicked`] when `op_result` is `Err`, and
/// [`ToolTimeoutError::TimedOut`] when `elapsed` exceeds `deadline`. The
/// `Panicked` branch wins if both apply (an in-op failure is the more
/// actionable signal).
// `op_result` is taken by value: callers typically have ownership of the
// boxed error string and forwarding it lets a future variant carry the
// payload into `Panicked` without a clone.
#[allow(clippy::needless_pass_by_value)]
pub fn record_outcome(
    tool_id: &str,
    deadline: Duration,
    op_result: Result<(), String>,
    elapsed: Duration,
) -> Result<(), ToolTimeoutError> {
    if op_result.is_err() {
        return Err(ToolTimeoutError::Panicked {
            tool_id: tool_id.to_string(),
        });
    }
    if elapsed > deadline {
        return Err(ToolTimeoutError::TimedOut {
            tool_id: tool_id.to_string(),
            deadline,
            elapsed,
        });
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn policy_default_values() {
        let p = ToolTimeoutPolicy::default();
        assert_eq!(p.default, Duration::from_secs(30));
        assert_eq!(p.max_overall, Duration::from_secs(300));
        assert!(p.overrides.is_empty());
    }

    #[test]
    fn deadline_for_no_override_returns_default() {
        let p = ToolTimeoutPolicy::default();
        assert_eq!(p.deadline_for("shell"), Duration::from_secs(30));
    }

    #[test]
    fn deadline_for_matching_override_returns_override() {
        let mut p = ToolTimeoutPolicy::default();
        p.overrides
            .insert("read_file".to_string(), Duration::from_secs(5));
        assert_eq!(p.deadline_for("read_file"), Duration::from_secs(5));
    }

    #[test]
    fn deadline_for_clamps_override_to_max_overall() {
        let mut p = ToolTimeoutPolicy::default();
        p.overrides
            .insert("slow".to_string(), Duration::from_secs(9999));
        assert_eq!(p.deadline_for("slow"), Duration::from_secs(300));
    }

    #[test]
    fn deadline_for_clamps_default_to_max_overall() {
        let p = ToolTimeoutPolicy {
            default: Duration::from_secs(9999),
            overrides: BTreeMap::new(),
            max_overall: Duration::from_secs(60),
        };
        assert_eq!(p.deadline_for("anything"), Duration::from_secs(60));
    }

    #[test]
    fn guard_disarm_before_deadline_returns_false() {
        let g = ToolTimeoutGuard::arm("t".to_string(), Duration::from_secs(60), || {});
        assert!(!g.disarm());
    }

    #[test]
    fn guard_past_deadline_returns_true_and_fires_callback() {
        let fired = Arc::new(AtomicBool::new(false));
        let fired_cb = Arc::clone(&fired);
        let g = ToolTimeoutGuard::arm("t".to_string(), Duration::from_millis(20), move || {
            fired_cb.store(true, Ordering::SeqCst);
        });
        thread::sleep(Duration::from_millis(120));
        assert!(g.disarm());
        assert!(fired.load(Ordering::SeqCst));
    }

    #[test]
    fn guard_elapsed_increases_monotonically() {
        let g = ToolTimeoutGuard::arm("t".to_string(), Duration::from_secs(60), || {});
        let a = g.elapsed();
        thread::sleep(Duration::from_millis(20));
        let b = g.elapsed();
        assert!(b >= a);
        let _ = g.disarm();
    }

    #[test]
    fn guard_time_remaining_saturates_to_none_after_deadline() {
        let g = ToolTimeoutGuard::arm("t".to_string(), Duration::from_millis(10), || {});
        thread::sleep(Duration::from_millis(60));
        assert!(g.time_remaining().is_none());
        let _ = g.disarm();
    }

    #[test]
    fn guard_time_remaining_some_before_deadline() {
        let g = ToolTimeoutGuard::arm("t".to_string(), Duration::from_secs(60), || {});
        assert!(g.time_remaining().is_some());
        assert_eq!(g.tool_id(), "t");
        assert_eq!(g.deadline(), Duration::from_secs(60));
        let _ = g.disarm();
    }

    #[test]
    fn run_with_timeout_quick_op_returns_ok() {
        let r: Result<u32, _> = run_with_timeout("t", Duration::from_secs(60), || 42);
        assert_eq!(r.unwrap(), 42);
    }

    #[test]
    fn run_with_timeout_slow_op_returns_timed_out() {
        let r: Result<(), _> = run_with_timeout("slow", Duration::from_millis(10), || {
            thread::sleep(Duration::from_millis(80));
        });
        assert!(matches!(
            r,
            Err(ToolTimeoutError::TimedOut {
                ref tool_id,
                deadline,
                ..
            }) if tool_id == "slow" && deadline == Duration::from_millis(10)
        ));
    }

    #[test]
    fn run_with_timeout_returning_a_value_works() {
        let r: Result<String, _> =
            run_with_timeout("t", Duration::from_secs(60), || "hello".to_string());
        assert_eq!(r.unwrap(), "hello");
    }

    #[test]
    fn record_outcome_ok_happy() {
        let r = record_outcome("t", Duration::from_secs(10), Ok(()), Duration::from_secs(1));
        assert!(r.is_ok());
    }

    #[test]
    fn record_outcome_timed_out_when_elapsed_gt_deadline() {
        let r = record_outcome("t", Duration::from_secs(1), Ok(()), Duration::from_secs(5));
        assert!(matches!(r, Err(ToolTimeoutError::TimedOut { .. })));
    }

    #[test]
    fn record_outcome_panicked_when_op_result_is_err() {
        let r = record_outcome(
            "t",
            Duration::from_secs(10),
            Err("boom".to_string()),
            Duration::from_secs(1),
        );
        assert!(matches!(r, Err(ToolTimeoutError::Panicked { .. })));
    }

    #[test]
    fn record_outcome_panicked_wins_over_timed_out() {
        let r = record_outcome(
            "t",
            Duration::from_secs(1),
            Err("boom".to_string()),
            Duration::from_secs(5),
        );
        assert!(matches!(r, Err(ToolTimeoutError::Panicked { .. })));
    }

    #[test]
    fn error_display_smoke_both_variants() {
        let to = ToolTimeoutError::TimedOut {
            tool_id: "shell".to_string(),
            deadline: Duration::from_secs(1),
            elapsed: Duration::from_secs(2),
        };
        let pn = ToolTimeoutError::Panicked {
            tool_id: "shell".to_string(),
        };
        let to_s = format!("{to}");
        let pn_s = format!("{pn}");
        assert!(to_s.contains("shell"));
        assert!(to_s.contains("deadline"));
        assert!(pn_s.contains("shell"));
        assert!(pn_s.contains("failed"));
        // Error trait
        let _: &dyn Error = &to;
        let _: &dyn Error = &pn;
    }

    #[test]
    fn policy_serde_round_trip() {
        let mut p = ToolTimeoutPolicy::default();
        p.overrides
            .insert("read_file".to_string(), Duration::from_secs(5));
        let j = serde_json::to_string(&p).unwrap();
        let back: ToolTimeoutPolicy = serde_json::from_str(&j).unwrap();
        assert_eq!(p, back);
    }

    #[test]
    fn guard_is_send_and_sync_smoke() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}
        assert_send::<ToolTimeoutGuard>();
        assert_sync::<ToolTimeoutGuard>();
    }

    #[test]
    fn guard_drop_without_disarm_does_not_panic() {
        {
            let _g = ToolTimeoutGuard::arm("t".to_string(), Duration::from_secs(60), || {});
            // dropped here
        }
        // give the timer thread a moment to observe the signal
        thread::sleep(Duration::from_millis(30));
    }

    #[test]
    fn callback_fires_exactly_once_across_n_rapid_disarms() {
        // We can only `disarm` once (it takes self by value), but the
        // invariant we care about is that the callback runs at most once.
        // Spin up N guards back-to-back, each with a very short deadline,
        // and verify each fires its own callback exactly once.
        let counter = Arc::new(AtomicUsize::new(0));
        let n = 5;
        let mut guards = Vec::with_capacity(n);
        for _ in 0..n {
            let c = Arc::clone(&counter);
            guards.push(ToolTimeoutGuard::arm(
                "rapid".to_string(),
                Duration::from_millis(10),
                move || {
                    c.fetch_add(1, Ordering::SeqCst);
                },
            ));
        }
        thread::sleep(Duration::from_millis(80));
        for g in guards {
            assert!(g.disarm());
        }
        assert_eq!(counter.load(Ordering::SeqCst), n);
    }

    #[test]
    fn deadline_zero_fires_immediately_on_any_real_sleep() {
        let fired = Arc::new(AtomicBool::new(false));
        let cb = Arc::clone(&fired);
        let g = ToolTimeoutGuard::arm("zero".to_string(), Duration::ZERO, move || {
            cb.store(true, Ordering::SeqCst);
        });
        thread::sleep(Duration::from_millis(30));
        assert!(g.disarm());
        assert!(fired.load(Ordering::SeqCst));
    }

    #[test]
    fn debug_impl_smoke() {
        let g = ToolTimeoutGuard::arm("dbg".to_string(), Duration::from_secs(60), || {});
        let s = format!("{g:?}");
        assert!(s.contains("dbg"));
        let _ = g.disarm();
    }
}
