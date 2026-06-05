//! Per-turn budget enforcement.
//!
//! Phase 3 plumbing — wraps an [`AgentBudget`] with a wall-clock start, an
//! atomic token counter, and a shared [`CancelToken`]. The orchestrator
//! polls a [`BudgetTracker`] between provider stream chunks; any
//! non-`Ok` check flips the cancel token so subagents and tool processes
//! observe the breach on their next poll.
//!
//! Per `plan/32-cancellation-and-budgets.md`.

use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Instant;

use serde::{Deserialize, Serialize};
use stratum_types::error::codes::{E4003_TOKEN_BUDGET, E4004_WALL_BUDGET};
use stratum_types::StratumError;

use crate::agents::AgentBudget;
use crate::cancel::CancelToken;

/// Outcome of a single budget poll.
///
/// Tagged on the wire so each variant round-trips through serde with its
/// numeric payload intact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum BudgetCheck {
    /// Budget intact; the turn may continue.
    Ok,
    /// Cumulative token count crossed [`AgentBudget::max_tokens_per_turn`].
    TokenBudgetExceeded {
        /// Total tokens accumulated so far (post-add, saturating).
        used: u32,
        /// The configured per-turn limit.
        limit: u32,
    },
    /// Wall clock crossed [`AgentBudget::max_wall_seconds`].
    WallBudgetExceeded {
        /// Milliseconds elapsed since the tracker started.
        elapsed_ms: u64,
        /// The configured per-turn limit, in milliseconds.
        limit_ms: u64,
    },
}

impl BudgetCheck {
    /// Lift a non-`Ok` check into a [`StratumError`] tagged with the canonical
    /// code (`STRAT-E4003` for tokens, `STRAT-E4004` for wall). Returns
    /// `None` for the `Ok` variant.
    #[must_use]
    pub fn stratum_error(&self) -> Option<StratumError> {
        match self {
            Self::Ok => None,
            Self::TokenBudgetExceeded { used, limit } => Some(StratumError::new(
                E4003_TOKEN_BUDGET,
                format!("token budget exceeded: used {used} of {limit}"),
            )),
            Self::WallBudgetExceeded {
                elapsed_ms,
                limit_ms,
            } => Some(StratumError::new(
                E4004_WALL_BUDGET,
                format!("wall budget exceeded: elapsed {elapsed_ms}ms of {limit_ms}ms"),
            )),
        }
    }
}

/// Runtime enforcer for an [`AgentBudget`].
///
/// Construct one per turn, clone the embedded [`CancelToken`] into every
/// subagent / tool subprocess, then call [`Self::record_tokens`] for each
/// streamed delta and [`Self::check_wall`] from any timer-driven poll.
#[derive(Debug)]
pub struct BudgetTracker {
    budget: AgentBudget,
    started_at: Instant,
    tokens_used: AtomicU32,
    cancel: CancelToken,
}

impl BudgetTracker {
    /// Build a tracker, snapping the wall-clock start to *now* and adopting
    /// the supplied cancel token. Cloning the token (via
    /// [`CancelToken::child`] or `.clone()`) into descendants is the
    /// caller's responsibility.
    #[must_use]
    pub fn new(budget: AgentBudget, cancel: CancelToken) -> Self {
        Self {
            budget,
            started_at: Instant::now(),
            tokens_used: AtomicU32::new(0),
            cancel,
        }
    }

    /// Borrow the cancel token. Clone it (or call `.child()`) to hand to
    /// subagents and tool processes.
    #[must_use]
    pub const fn cancel_token(&self) -> &CancelToken {
        &self.cancel
    }

    /// Total tokens streamed so far.
    #[must_use]
    pub fn tokens_used(&self) -> u32 {
        self.tokens_used.load(Ordering::Acquire)
    }

    /// Milliseconds since the tracker was constructed.
    #[must_use]
    pub fn elapsed_ms(&self) -> u64 {
        let ms = self.started_at.elapsed().as_millis();
        u64::try_from(ms).unwrap_or(u64::MAX)
    }

    /// Tokens still available before the limit is breached. Saturates to
    /// zero once the cap is hit.
    #[must_use]
    pub fn remaining_tokens(&self) -> u32 {
        self.budget
            .max_tokens_per_turn
            .saturating_sub(self.tokens_used())
    }

    /// Add `delta` tokens to the counter (saturating at `u32::MAX`) and
    /// report the post-add state. On any non-`Ok` outcome the cancel token
    /// is flipped; the wall budget is checked first so a turn that has
    /// already run past the deadline reports the wall breach even when the
    /// fresh delta would also exceed the token cap.
    pub fn record_tokens(&self, delta: u32) -> BudgetCheck {
        // Wall first: a stale, over-deadline turn should surface the wall
        // breach rather than masquerade as a token overflow.
        let wall = self.check_wall();
        if !matches!(wall, BudgetCheck::Ok) {
            return wall;
        }
        // Saturating fetch_add — Ordering::AcqRel so the post-add view is
        // observed in full by the next poller.
        let prev = self.tokens_used.load(Ordering::Acquire);
        let next = prev.saturating_add(delta);
        self.tokens_used.store(next, Ordering::Release);
        if next > self.budget.max_tokens_per_turn {
            self.cancel.cancel();
            return BudgetCheck::TokenBudgetExceeded {
                used: next,
                limit: self.budget.max_tokens_per_turn,
            };
        }
        BudgetCheck::Ok
    }

    /// Poll the wall budget without mutating the token counter. Flips the
    /// cancel token if the deadline has passed.
    pub fn check_wall(&self) -> BudgetCheck {
        let elapsed_ms = self.elapsed_ms();
        let limit_ms = u64::from(self.budget.max_wall_seconds).saturating_mul(1000);
        if elapsed_ms > limit_ms {
            self.cancel.cancel();
            return BudgetCheck::WallBudgetExceeded {
                elapsed_ms,
                limit_ms,
            };
        }
        BudgetCheck::Ok
    }
}

#[cfg(test)]
mod tests {
    use std::thread;
    use std::time::Duration;

    use super::*;

    fn budget(tokens: u32, wall_seconds: u32) -> AgentBudget {
        AgentBudget {
            max_tokens_per_turn: tokens,
            max_wall_seconds: wall_seconds,
            max_concurrent_invocations: 1,
        }
    }

    #[test]
    fn record_tokens_accumulates_then_trips_token_budget() {
        let cancel = CancelToken::new();
        let tracker = BudgetTracker::new(budget(10, 60), cancel.clone());
        assert_eq!(tracker.record_tokens(4), BudgetCheck::Ok);
        assert_eq!(tracker.record_tokens(4), BudgetCheck::Ok);
        assert!(!cancel.is_cancelled());
        let breach = tracker.record_tokens(5);
        assert_eq!(
            breach,
            BudgetCheck::TokenBudgetExceeded {
                used: 13,
                limit: 10
            }
        );
        assert!(cancel.is_cancelled());
        assert_eq!(tracker.tokens_used(), 13);
    }

    #[test]
    fn record_tokens_saturates_on_overflow() {
        let cancel = CancelToken::new();
        let tracker = BudgetTracker::new(budget(u32::MAX, 60), cancel);
        // Push counter near the ceiling, then overflow it.
        assert_eq!(tracker.record_tokens(u32::MAX - 5), BudgetCheck::Ok);
        // Adding more saturates at u32::MAX rather than panicking.
        let next = tracker.record_tokens(u32::MAX);
        assert_eq!(next, BudgetCheck::Ok);
        assert_eq!(tracker.tokens_used(), u32::MAX);
    }

    #[test]
    fn check_wall_is_ok_within_budget() {
        let cancel = CancelToken::new();
        let tracker = BudgetTracker::new(budget(1024, 60), cancel.clone());
        assert_eq!(tracker.check_wall(), BudgetCheck::Ok);
        assert!(!cancel.is_cancelled());
    }

    #[test]
    fn check_wall_zero_seconds_trips_on_first_call() {
        let cancel = CancelToken::new();
        let tracker = BudgetTracker::new(budget(1024, 0), cancel.clone());
        // Sleep a beat so elapsed_ms > 0 deterministically.
        thread::sleep(Duration::from_millis(2));
        let check = tracker.check_wall();
        match check {
            BudgetCheck::WallBudgetExceeded {
                elapsed_ms,
                limit_ms,
            } => {
                assert_eq!(limit_ms, 0);
                assert!(elapsed_ms > 0);
            }
            other => panic!("expected WallBudgetExceeded, got {other:?}"),
        }
        assert!(cancel.is_cancelled());
    }

    #[test]
    fn token_breach_yields_e4003_with_numbers() {
        let breach = BudgetCheck::TokenBudgetExceeded {
            used: 17,
            limit: 10,
        };
        let err = breach.stratum_error().expect("error expected");
        assert_eq!(err.code(), &E4003_TOKEN_BUDGET);
        let rendered = format!("{err}");
        assert!(rendered.contains("17"));
        assert!(rendered.contains("10"));
    }

    #[test]
    fn wall_breach_yields_e4004_with_numbers() {
        let breach = BudgetCheck::WallBudgetExceeded {
            elapsed_ms: 1234,
            limit_ms: 1000,
        };
        let err = breach.stratum_error().expect("error expected");
        assert_eq!(err.code(), &E4004_WALL_BUDGET);
        let rendered = format!("{err}");
        assert!(rendered.contains("1234"));
        assert!(rendered.contains("1000"));
    }

    #[test]
    fn ok_has_no_stratum_error() {
        assert!(BudgetCheck::Ok.stratum_error().is_none());
    }

    #[test]
    fn fresh_tracker_does_not_cancel_token() {
        let cancel = CancelToken::new();
        let _tracker = BudgetTracker::new(budget(1024, 60), cancel.clone());
        assert!(!cancel.is_cancelled());
    }

    #[test]
    fn token_breach_flips_supplied_cancel_token() {
        let cancel = CancelToken::new();
        let tracker = BudgetTracker::new(budget(2, 60), cancel.clone());
        let _ = tracker.record_tokens(5);
        assert!(cancel.is_cancelled());
    }

    #[test]
    fn remaining_tokens_full_partial_and_overbudget() {
        let cancel = CancelToken::new();
        let tracker = BudgetTracker::new(budget(10, 60), cancel);
        assert_eq!(tracker.remaining_tokens(), 10);
        assert_eq!(tracker.record_tokens(3), BudgetCheck::Ok);
        assert_eq!(tracker.remaining_tokens(), 7);
        // Push past the limit; remaining saturates to zero.
        let _ = tracker.record_tokens(50);
        assert_eq!(tracker.remaining_tokens(), 0);
    }

    #[test]
    fn budget_check_serde_roundtrip_all_variants() {
        let cases = [
            BudgetCheck::Ok,
            BudgetCheck::TokenBudgetExceeded {
                used: 42,
                limit: 32,
            },
            BudgetCheck::WallBudgetExceeded {
                elapsed_ms: 999,
                limit_ms: 500,
            },
        ];
        for original in cases {
            let s = serde_json::to_string(&original).expect("serialize");
            assert!(s.contains("kind"));
            let back: BudgetCheck = serde_json::from_str(&s).expect("deserialize");
            assert_eq!(back, original);
        }
    }

    #[test]
    fn cancel_token_accessor_is_the_supplied_one() {
        let cancel = CancelToken::new();
        let tracker = BudgetTracker::new(budget(10, 60), cancel.clone());
        tracker.cancel_token().cancel();
        assert!(cancel.is_cancelled());
    }

    #[test]
    fn elapsed_ms_is_monotonic_nondecreasing() {
        let cancel = CancelToken::new();
        let tracker = BudgetTracker::new(budget(10, 60), cancel);
        let a = tracker.elapsed_ms();
        thread::sleep(Duration::from_millis(2));
        let b = tracker.elapsed_ms();
        assert!(b >= a);
    }

    #[test]
    fn wall_breach_preempts_token_check_on_record_tokens() {
        let cancel = CancelToken::new();
        let tracker = BudgetTracker::new(budget(10, 0), cancel.clone());
        thread::sleep(Duration::from_millis(2));
        let check = tracker.record_tokens(100);
        assert!(matches!(check, BudgetCheck::WallBudgetExceeded { .. }));
        // Token counter should NOT have been incremented since the wall
        // gate fires first.
        assert_eq!(tracker.tokens_used(), 0);
        assert!(cancel.is_cancelled());
    }
}
