//! Per-turn conversation state machine.
//!
//! Phase 3 (data + transitions only) — the future agentic loop will pump
//! [`TurnEvent`]s through [`TurnDriver`] to drive a single user-turn from
//! "user pressed enter" through model generation, optional tool calls, and
//! into a terminal `Done` or `Cancelled` state. This module is pure: no I/O,
//! no spawning, no logging side effects. The driver only computes
//! deterministic state transitions and records a `Vec<TurnTransition>`
//! history so callers (TUI, recorder, replay tools) can audit the lifecycle.
//!
//! # FSM
//!
//! ```text
//!   Idle ─StartGenerate──▶ Generating ─RequestToolUse──▶ AwaitingToolApproval
//!                              │                              │
//!                              │                              ├─ApproveTool──▶ RunningTool
//!                              │                              └─DenyTool────▶ Generating
//!                              ├─StartSummarize─▶ Summarizing ─Finish─▶ Done
//!                              └─Finish─▶ Done
//!
//!   RunningTool ─ToolCompleted(ok=true)──▶ Generating
//!   RunningTool ─ToolCompleted(ok=false)─▶ Generating   (recoverable; bubble via Finish)
//!
//!   <any non-terminal> ─Cancel──▶ Cancelled
//! ```
//!
//! # Tool-failure policy
//!
//! On `ToolCompleted { ok: false, code }` the FSM returns to `Generating`
//! so the model can react to the failure (retry with different args, ask
//! the user, give up gracefully). The driver does NOT auto-finish into
//! `Done { ToolFailure }` — that's a policy decision the caller pins by
//! emitting `Finish { outcome: ToolFailure { .. } }` itself. Pinned by
//! `running_tool_completed_failure_returns_to_generating`.
//!
//! See `plan/15-agentic-loop.md`.

use std::error::Error;
use std::fmt::{self, Display, Formatter};
use std::time::SystemTime;

use serde::{Deserialize, Serialize};

/// State of a single user turn.
///
/// Terminal: [`TurnState::Done`], [`TurnState::Cancelled`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum TurnState {
    /// No work has started — the driver is freshly constructed and
    /// awaiting [`TurnEvent::StartGenerate`].
    Idle,
    /// The model is producing tokens.
    Generating,
    /// The model proposed a tool call; waiting for user / policy approval.
    AwaitingToolApproval {
        /// Opaque tool-call request payload (serialized JSON or similar).
        /// The FSM doesn't introspect it; callers do.
        request: String,
    },
    /// An approved tool invocation is currently executing.
    RunningTool {
        /// Stable identifier for the tool call (e.g. `read_file:42`).
        tool_id: String,
        /// When the tool began running. Recorded by the driver for
        /// downstream latency accounting.
        started_at: SystemTime,
    },
    /// The model is producing the final summary turn after tools have
    /// settled.
    Summarizing,
    /// Terminal success / structured failure.
    Done {
        /// What ended the turn.
        outcome: TurnOutcome,
    },
    /// Terminal cancellation (user abort, deadline, parent cascade).
    Cancelled {
        /// Human-readable reason surfaced in logs / TUI.
        reason: String,
    },
}

/// Why a turn reached [`TurnState::Done`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum TurnOutcome {
    /// Clean completion.
    Success,
    /// A tool errored and the caller chose to bubble the failure.
    ToolFailure {
        /// Tool that failed.
        tool_id: String,
        /// Stable failure code (e.g. `E_TOOL_TIMEOUT`).
        code: String,
    },
    /// A per-turn budget (tokens, wall-clock, dollars) was exhausted.
    BudgetExceeded {
        /// Which budget tripped (e.g. `tokens`, `wall_clock_ms`).
        kind: String,
    },
    /// The model backend errored before producing a usable response.
    ModelError {
        /// Stable error code from the provider layer.
        code: String,
    },
    /// The user aborted but the caller wanted to record it under `Done`
    /// rather than `Cancelled` (rare — usually `Cancel` is preferred).
    UserAbort,
}

/// Event consumed by [`TurnDriver::apply`] / [`next_state`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum TurnEvent {
    /// Move `Idle` → `Generating`.
    StartGenerate,
    /// Model proposed a tool call.
    RequestToolUse {
        /// Opaque request payload, forwarded into
        /// [`TurnState::AwaitingToolApproval::request`].
        request: String,
    },
    /// Operator / policy approved the pending tool call.
    ApproveTool {
        /// Stable identifier the runtime will use to track the call.
        tool_id: String,
    },
    /// Operator / policy denied the pending tool call; the model is
    /// pushed back into `Generating` so it can try a different path.
    DenyTool,
    /// A running tool finished.
    ToolCompleted {
        /// Tool that completed.
        tool_id: String,
        /// Whether the tool succeeded.
        ok: bool,
        /// On failure, a stable code (`E_TOOL_TIMEOUT` etc.). Always
        /// present on `ok = false` in practice; the FSM does not enforce
        /// it because the field is informational.
        code: Option<String>,
    },
    /// Model is now producing the wrap-up turn.
    StartSummarize,
    /// Terminate the turn with `outcome`.
    Finish {
        /// What to record in [`TurnState::Done`].
        outcome: TurnOutcome,
    },
    /// Cancel from any non-terminal state.
    Cancel {
        /// Human-readable reason.
        reason: String,
    },
}

/// Errors returned by [`next_state`] and [`TurnDriver::apply`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TurnFsmError {
    /// The supplied `event` is not legal from `from`.
    InvalidTransition {
        /// Current state.
        from: TurnState,
        /// Event that was rejected.
        event: TurnEvent,
    },
    /// The driver is already in a terminal state (`Done` / `Cancelled`).
    Terminal,
}

impl Display for TurnFsmError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTransition { from, event } => write!(
                f,
                "invalid turn-fsm transition: event {event:?} not legal from state {from:?}"
            ),
            Self::Terminal => f.write_str("turn-fsm is terminal; no further events accepted"),
        }
    }
}

impl Error for TurnFsmError {}

/// One recorded `(from, event, to)` triple along with its timestamp.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TurnTransition {
    /// Wall-clock time supplied by the caller (`now`).
    pub at: SystemTime,
    /// Source state.
    pub from: TurnState,
    /// Event that drove the transition.
    pub event: TurnEvent,
    /// Resulting state.
    pub to: TurnState,
}

/// Pure transition function.
///
/// # Errors
///
/// - [`TurnFsmError::Terminal`] if `current` is `Done` / `Cancelled`.
/// - [`TurnFsmError::InvalidTransition`] for any non-allowed `(state,
///   event)` pair.
pub fn next_state(
    current: &TurnState,
    event: TurnEvent,
    now: SystemTime,
) -> Result<TurnState, TurnFsmError> {
    // Terminal short-circuit. `Cancel` is NOT accepted from terminal
    // states either — the contract says terminal states reject all events.
    if matches!(
        current,
        TurnState::Done { .. } | TurnState::Cancelled { .. }
    ) {
        return Err(TurnFsmError::Terminal);
    }

    // Cancel from any non-terminal state.
    if let TurnEvent::Cancel { reason } = &event {
        return Ok(TurnState::Cancelled {
            reason: reason.clone(),
        });
    }

    // Identical bodies below (e.g. multiple arms returning `Generating`,
    // `Done { outcome }`) are kept as distinct semantic transitions; do
    // not collapse via `|` because the source state is part of the
    // contract and we want stack traces / coverage to point at the right
    // arm. Hence the `match_same_arms` allow.
    #[allow(clippy::match_same_arms)]
    match (current, event) {
        (TurnState::Idle, TurnEvent::StartGenerate) => Ok(TurnState::Generating),

        (TurnState::Generating, TurnEvent::RequestToolUse { request }) => {
            Ok(TurnState::AwaitingToolApproval { request })
        }
        (TurnState::Generating, TurnEvent::StartSummarize) => Ok(TurnState::Summarizing),
        (TurnState::Generating, TurnEvent::Finish { outcome }) => Ok(TurnState::Done { outcome }),

        (TurnState::AwaitingToolApproval { .. }, TurnEvent::ApproveTool { tool_id }) => {
            Ok(TurnState::RunningTool {
                tool_id,
                started_at: now,
            })
        }
        (TurnState::AwaitingToolApproval { .. }, TurnEvent::DenyTool) => Ok(TurnState::Generating),

        (TurnState::RunningTool { .. }, TurnEvent::ToolCompleted { .. }) => {
            // Per module rustdoc: both success and recoverable failure
            // return to `Generating`. The caller bubbles a hard failure
            // by then emitting `Finish { outcome: ToolFailure }`.
            Ok(TurnState::Generating)
        }

        (TurnState::Summarizing, TurnEvent::Finish { outcome }) => Ok(TurnState::Done { outcome }),

        (from, event) => Err(TurnFsmError::InvalidTransition {
            from: from.clone(),
            event,
        }),
    }
}

/// Mutable wrapper around [`next_state`] that records every transition.
#[derive(Debug, Clone)]
pub struct TurnDriver {
    state: TurnState,
    history: Vec<TurnTransition>,
}

impl TurnDriver {
    /// Construct a driver in [`TurnState::Idle`] with empty history.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            state: TurnState::Idle,
            history: Vec::new(),
        }
    }

    /// Current state.
    #[must_use]
    pub const fn state(&self) -> &TurnState {
        &self.state
    }

    /// Apply `event` at `now`. On success, the new state is recorded in
    /// history and returned; on error the driver is unchanged.
    ///
    /// # Errors
    ///
    /// See [`next_state`].
    pub fn apply(&mut self, event: TurnEvent, now: SystemTime) -> Result<&TurnState, TurnFsmError> {
        let from = self.state.clone();
        let to = next_state(&from, event.clone(), now)?;
        self.history.push(TurnTransition {
            at: now,
            from,
            event,
            to: to.clone(),
        });
        self.state = to;
        Ok(&self.state)
    }

    /// Recorded transitions in arrival order.
    #[must_use]
    pub fn history(&self) -> &[TurnTransition] {
        &self.history
    }

    /// `true` once the FSM is in `Done` / `Cancelled`.
    #[must_use]
    pub const fn is_terminal(&self) -> bool {
        matches!(
            self.state,
            TurnState::Done { .. } | TurnState::Cancelled { .. }
        )
    }
}

impl Default for TurnDriver {
    fn default() -> Self {
        Self::new()
    }
}

/// Walk `history` and verify it represents a legal trajectory.
///
/// - The first entry's `from` must be [`TurnState::Idle`].
/// - Each subsequent entry's `from` must equal the previous entry's `to`.
/// - For every entry, `next_state(from, event, at)` must equal the
///   recorded `to`.
///
/// # Errors
///
/// Returns [`TurnFsmError::InvalidTransition`] on the first mismatch.
/// An empty `history` is accepted.
pub fn validate_history(history: &[TurnTransition]) -> Result<(), TurnFsmError> {
    let mut prev_to = TurnState::Idle;
    for (i, entry) in history.iter().enumerate() {
        let expected_from = if i == 0 {
            TurnState::Idle
        } else {
            prev_to.clone()
        };
        if entry.from != expected_from {
            return Err(TurnFsmError::InvalidTransition {
                from: entry.from.clone(),
                event: entry.event.clone(),
            });
        }
        let computed = next_state(&entry.from, entry.event.clone(), entry.at)?;
        if computed != entry.to {
            return Err(TurnFsmError::InvalidTransition {
                from: entry.from.clone(),
                event: entry.event.clone(),
            });
        }
        prev_to = entry.to.clone();
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn t0() -> SystemTime {
        SystemTime::UNIX_EPOCH + Duration::from_secs(1_700_000_000)
    }

    fn t1() -> SystemTime {
        t0() + Duration::from_millis(50)
    }

    fn t2() -> SystemTime {
        t0() + Duration::from_millis(120)
    }

    // -- next_state happy paths -------------------------------------------------

    #[test]
    fn idle_start_generate_goes_to_generating() {
        let s = next_state(&TurnState::Idle, TurnEvent::StartGenerate, t0()).unwrap();
        assert_eq!(s, TurnState::Generating);
    }

    #[test]
    fn idle_start_summarize_is_invalid() {
        let err = next_state(&TurnState::Idle, TurnEvent::StartSummarize, t0()).unwrap_err();
        assert!(matches!(err, TurnFsmError::InvalidTransition { .. }));
    }

    #[test]
    fn generating_request_tool_use_goes_to_awaiting() {
        let s = next_state(
            &TurnState::Generating,
            TurnEvent::RequestToolUse {
                request: "read_file foo.rs".into(),
            },
            t0(),
        )
        .unwrap();
        assert_eq!(
            s,
            TurnState::AwaitingToolApproval {
                request: "read_file foo.rs".into(),
            }
        );
    }

    #[test]
    fn generating_finish_goes_to_done() {
        let s = next_state(
            &TurnState::Generating,
            TurnEvent::Finish {
                outcome: TurnOutcome::Success,
            },
            t0(),
        )
        .unwrap();
        assert_eq!(
            s,
            TurnState::Done {
                outcome: TurnOutcome::Success
            }
        );
    }

    #[test]
    fn awaiting_approve_goes_to_running_tool() {
        let s = next_state(
            &TurnState::AwaitingToolApproval {
                request: "fs.read".into(),
            },
            TurnEvent::ApproveTool {
                tool_id: "fs.read#1".into(),
            },
            t1(),
        )
        .unwrap();
        assert_eq!(
            s,
            TurnState::RunningTool {
                tool_id: "fs.read#1".into(),
                started_at: t1(),
            }
        );
    }

    #[test]
    fn awaiting_deny_goes_back_to_generating() {
        let s = next_state(
            &TurnState::AwaitingToolApproval {
                request: "fs.write".into(),
            },
            TurnEvent::DenyTool,
            t0(),
        )
        .unwrap();
        assert_eq!(s, TurnState::Generating);
    }

    #[test]
    fn running_tool_completed_ok_returns_to_generating() {
        let s = next_state(
            &TurnState::RunningTool {
                tool_id: "fs.read#1".into(),
                started_at: t0(),
            },
            TurnEvent::ToolCompleted {
                tool_id: "fs.read#1".into(),
                ok: true,
                code: None,
            },
            t1(),
        )
        .unwrap();
        assert_eq!(s, TurnState::Generating);
    }

    /// Pins the recoverable-failure policy: a failed tool returns to
    /// `Generating`. To bubble the failure as terminal, the caller must
    /// emit `Finish { outcome: ToolFailure }`.
    #[test]
    fn running_tool_completed_failure_returns_to_generating() {
        let s = next_state(
            &TurnState::RunningTool {
                tool_id: "fs.read#1".into(),
                started_at: t0(),
            },
            TurnEvent::ToolCompleted {
                tool_id: "fs.read#1".into(),
                ok: false,
                code: Some("E_TOOL_TIMEOUT".into()),
            },
            t1(),
        )
        .unwrap();
        assert_eq!(s, TurnState::Generating);
    }

    #[test]
    fn generating_cancel_goes_to_cancelled() {
        let s = next_state(
            &TurnState::Generating,
            TurnEvent::Cancel {
                reason: "user_abort".into(),
            },
            t0(),
        )
        .unwrap();
        assert_eq!(
            s,
            TurnState::Cancelled {
                reason: "user_abort".into(),
            }
        );
    }

    #[test]
    fn done_start_generate_is_terminal() {
        let err = next_state(
            &TurnState::Done {
                outcome: TurnOutcome::Success,
            },
            TurnEvent::StartGenerate,
            t0(),
        )
        .unwrap_err();
        assert_eq!(err, TurnFsmError::Terminal);
    }

    #[test]
    fn cancelled_start_generate_is_terminal() {
        let err = next_state(
            &TurnState::Cancelled {
                reason: "deadline".into(),
            },
            TurnEvent::StartGenerate,
            t0(),
        )
        .unwrap_err();
        assert_eq!(err, TurnFsmError::Terminal);
    }

    // -- TurnDriver -------------------------------------------------------------

    #[test]
    fn driver_walks_idle_generating_done() {
        let mut d = TurnDriver::new();
        assert_eq!(d.state(), &TurnState::Idle);
        d.apply(TurnEvent::StartGenerate, t0()).unwrap();
        assert_eq!(d.state(), &TurnState::Generating);
        d.apply(
            TurnEvent::Finish {
                outcome: TurnOutcome::Success,
            },
            t1(),
        )
        .unwrap();
        assert_eq!(
            d.state(),
            &TurnState::Done {
                outcome: TurnOutcome::Success
            }
        );
    }

    #[test]
    fn driver_history_records_each_transition() {
        let mut d = TurnDriver::new();
        d.apply(TurnEvent::StartGenerate, t0()).unwrap();
        d.apply(
            TurnEvent::RequestToolUse {
                request: "fs.read".into(),
            },
            t1(),
        )
        .unwrap();
        d.apply(TurnEvent::DenyTool, t2()).unwrap();

        let h = d.history();
        assert_eq!(h.len(), 3);
        assert_eq!(h[0].at, t0());
        assert_eq!(h[0].from, TurnState::Idle);
        assert_eq!(h[0].to, TurnState::Generating);
        assert_eq!(h[1].at, t1());
        assert_eq!(
            h[1].to,
            TurnState::AwaitingToolApproval {
                request: "fs.read".into(),
            }
        );
        assert_eq!(h[2].at, t2());
        assert_eq!(h[2].to, TurnState::Generating);
    }

    #[test]
    fn driver_is_terminal_after_done() {
        let mut d = TurnDriver::new();
        d.apply(TurnEvent::StartGenerate, t0()).unwrap();
        assert!(!d.is_terminal());
        d.apply(
            TurnEvent::Finish {
                outcome: TurnOutcome::Success,
            },
            t1(),
        )
        .unwrap();
        assert!(d.is_terminal());
    }

    #[test]
    fn driver_is_terminal_after_cancel() {
        let mut d = TurnDriver::new();
        d.apply(TurnEvent::StartGenerate, t0()).unwrap();
        d.apply(
            TurnEvent::Cancel {
                reason: "ctrl-c".into(),
            },
            t1(),
        )
        .unwrap();
        assert!(d.is_terminal());
    }

    #[test]
    fn driver_default_matches_new() {
        let a: TurnDriver = TurnDriver::default();
        let b = TurnDriver::new();
        assert_eq!(a.state(), b.state());
        assert!(a.history().is_empty());
    }

    #[test]
    fn driver_apply_does_not_mutate_on_invalid_transition() {
        let mut d = TurnDriver::new();
        let before = d.state().clone();
        let err = d.apply(TurnEvent::StartSummarize, t0()).unwrap_err();
        assert!(matches!(err, TurnFsmError::InvalidTransition { .. }));
        assert_eq!(d.state(), &before);
        assert!(d.history().is_empty());
    }

    // -- serde round-trips ------------------------------------------------------

    #[test]
    fn turn_state_serde_round_trip() {
        for s in [
            TurnState::Idle,
            TurnState::Generating,
            TurnState::AwaitingToolApproval {
                request: "x".into(),
            },
            TurnState::RunningTool {
                tool_id: "t".into(),
                started_at: t0(),
            },
            TurnState::Summarizing,
            TurnState::Done {
                outcome: TurnOutcome::Success,
            },
            TurnState::Cancelled { reason: "r".into() },
        ] {
            let j = serde_json::to_string(&s).unwrap();
            let back: TurnState = serde_json::from_str(&j).unwrap();
            assert_eq!(s, back);
        }
    }

    #[test]
    fn turn_event_serde_round_trip() {
        for e in [
            TurnEvent::StartGenerate,
            TurnEvent::RequestToolUse {
                request: "q".into(),
            },
            TurnEvent::ApproveTool {
                tool_id: "id".into(),
            },
            TurnEvent::DenyTool,
            TurnEvent::ToolCompleted {
                tool_id: "id".into(),
                ok: false,
                code: Some("E".into()),
            },
            TurnEvent::StartSummarize,
            TurnEvent::Finish {
                outcome: TurnOutcome::Success,
            },
            TurnEvent::Cancel { reason: "r".into() },
        ] {
            let j = serde_json::to_string(&e).unwrap();
            let back: TurnEvent = serde_json::from_str(&j).unwrap();
            assert_eq!(e, back);
        }
    }

    #[test]
    fn turn_outcome_serde_round_trip() {
        for o in [
            TurnOutcome::Success,
            TurnOutcome::ToolFailure {
                tool_id: "t".into(),
                code: "E_TOOL_TIMEOUT".into(),
            },
            TurnOutcome::BudgetExceeded {
                kind: "tokens".into(),
            },
            TurnOutcome::ModelError {
                code: "E_MODEL".into(),
            },
            TurnOutcome::UserAbort,
        ] {
            let j = serde_json::to_string(&o).unwrap();
            let back: TurnOutcome = serde_json::from_str(&j).unwrap();
            assert_eq!(o, back);
        }
    }

    #[test]
    fn turn_transition_serde_round_trip() {
        let tr = TurnTransition {
            at: t0(),
            from: TurnState::Idle,
            event: TurnEvent::StartGenerate,
            to: TurnState::Generating,
        };
        let j = serde_json::to_string(&tr).unwrap();
        let back: TurnTransition = serde_json::from_str(&j).unwrap();
        assert_eq!(tr, back);
    }

    // -- validate_history -------------------------------------------------------

    #[test]
    fn validate_history_accepts_clean_walk() {
        let mut d = TurnDriver::new();
        d.apply(TurnEvent::StartGenerate, t0()).unwrap();
        d.apply(
            TurnEvent::Finish {
                outcome: TurnOutcome::Success,
            },
            t1(),
        )
        .unwrap();
        validate_history(d.history()).unwrap();
    }

    #[test]
    fn validate_history_accepts_empty() {
        validate_history(&[]).unwrap();
    }

    #[test]
    fn validate_history_rejects_broken_chain() {
        // Two entries where the second `from` does not match the first
        // `to`. Each entry, taken alone, is a legal `next_state` call.
        let history = vec![
            TurnTransition {
                at: t0(),
                from: TurnState::Idle,
                event: TurnEvent::StartGenerate,
                to: TurnState::Generating,
            },
            TurnTransition {
                at: t1(),
                from: TurnState::Idle, // BUG: should be Generating
                event: TurnEvent::StartGenerate,
                to: TurnState::Generating,
            },
        ];
        let err = validate_history(&history).unwrap_err();
        assert!(matches!(err, TurnFsmError::InvalidTransition { .. }));
    }

    #[test]
    fn validate_history_rejects_wrong_recorded_to() {
        // `from` and `event` are consistent with each other, but the
        // recorded `to` lies about the result.
        let history = vec![TurnTransition {
            at: t0(),
            from: TurnState::Idle,
            event: TurnEvent::StartGenerate,
            to: TurnState::Summarizing, // lies; should be Generating
        }];
        let err = validate_history(&history).unwrap_err();
        assert!(matches!(err, TurnFsmError::InvalidTransition { .. }));
    }

    // -- TurnFsmError Display ---------------------------------------------------

    #[test]
    fn turn_fsm_error_display_smoke() {
        let a = TurnFsmError::InvalidTransition {
            from: TurnState::Idle,
            event: TurnEvent::StartSummarize,
        };
        let s = format!("{a}");
        assert!(s.contains("invalid turn-fsm transition"));

        let b = TurnFsmError::Terminal;
        let s = format!("{b}");
        assert!(s.contains("terminal"));
    }
}
