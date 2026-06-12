//! [`TuiPromptResponder`] — a [`PromptResponder`] that hands the prompt to
//! the TUI event loop and blocks the calling thread on a per-prompt
//! condvar until the user (or a test) supplies a decision.
//!
//! # Architecture
//!
//! The agent loop runs on a worker thread inside [`crate::chat::ChatState::submit`].
//! When it hits a `Block::ToolCall` it calls
//! [`PromptResponder::ask`] synchronously. The TUI event loop, however,
//! drives `step` on the main thread and owns the terminal. We bridge the
//! two with a shared queue + decision map under a single `Mutex` +
//! [`Condvar`]:
//!
//! 1. Worker pushes a [`PendingPrompt`] into `requests` and `notify_all`s.
//! 2. Worker blocks in `wait_timeout` polling `decisions[prompt.id]`.
//! 3. The TUI event loop calls [`TuiPromptResponder::pending_request`] to
//!    pop the next request, renders the modal, reads a key, and calls
//!    [`TuiPromptResponder::submit_decision`] with the user's pick.
//! 4. `submit_decision` inserts into `decisions` and `notify_all`s, which
//!    wakes the worker; it removes the entry and returns the decision.
//!
//! Timeouts default to 60 s; tests use shorter ones via
//! [`TuiPromptResponder::new`]. A timeout returns
//! [`PermissionDecision::Deny`] (fail-closed).

#![allow(
    unreachable_pub,
    reason = "private module by design; pub kept for readability"
)]
#![allow(
    clippy::redundant_pub_crate,
    reason = "internal API kept pub for documentation; module itself is private"
)]

use std::collections::{BTreeMap, VecDeque};
use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant};

use stratum_runtime::{PendingPrompt, PermissionDecision, PromptId, PromptResponder};

/// Default wait window for an unanswered prompt. After this the responder
/// fails closed (returns [`PermissionDecision::Deny`]).
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(60);

/// Mutex-guarded shared state. Kept under a single lock so the `Condvar`
/// has a stable predicate to wait on.
#[derive(Debug, Default)]
struct Inner {
    requests: VecDeque<PendingPrompt>,
    decisions: BTreeMap<PromptId, PermissionDecision>,
}

/// TUI-driven [`PromptResponder`]: queue + condvar + per-prompt decision map.
#[derive(Debug)]
pub struct TuiPromptResponder {
    inner: Mutex<Inner>,
    condvar: Condvar,
    timeout: Duration,
}

impl Default for TuiPromptResponder {
    fn default() -> Self {
        Self::new(DEFAULT_TIMEOUT)
    }
}

impl TuiPromptResponder {
    /// Build a responder that waits up to `timeout` for a decision before
    /// failing closed.
    #[must_use]
    pub fn new(timeout: Duration) -> Self {
        Self {
            inner: Mutex::new(Inner::default()),
            condvar: Condvar::new(),
            timeout,
        }
    }

    /// Pop the next queued prompt, if any. The TUI event loop calls this
    /// once per render to discover work; `None` means "nothing pending".
    #[must_use]
    pub fn pending_request(&self) -> Option<PendingPrompt> {
        let mut guard = match self.inner.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        guard.requests.pop_front()
    }

    /// Push `prompt` back to the front of the queue without notifying
    /// waiters. Used by the TUI when a popped request was not actually
    /// answered (e.g. the user pressed an unrecognised key) — the next
    /// render tick will see it again.
    pub fn requeue_for_redisplay(&self, prompt: PendingPrompt) {
        let mut guard = match self.inner.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        guard.requests.push_front(prompt);
    }

    /// Clone the next queued prompt without removing it. Lets the TUI render
    /// the modal repeatedly without disturbing the queue.
    #[must_use]
    pub fn peek_request(&self) -> Option<PendingPrompt> {
        let guard = match self.inner.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        guard.requests.front().cloned()
    }

    /// Record a decision for prompt `id` and wake any waiter.
    pub fn submit_decision(&self, id: PromptId, decision: PermissionDecision) {
        {
            let mut guard = match self.inner.lock() {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };
            guard.decisions.insert(id, decision);
        }
        self.condvar.notify_all();
    }
}

impl PromptResponder for TuiPromptResponder {
    fn ask(&self, prompt: &PendingPrompt) -> PermissionDecision {
        // Phase 1: publish the request.
        {
            let mut guard = match self.inner.lock() {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };
            guard.requests.push_back(prompt.clone());
        }
        self.condvar.notify_all();

        // Phase 2: wait for a matching decision or timeout.
        let deadline = Instant::now() + self.timeout;
        let mut guard = match self.inner.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        loop {
            if let Some(decision) = guard.decisions.remove(&prompt.id) {
                return decision;
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                // Fail-closed on timeout; the unanswered request is left in
                // the queue so the TUI can still surface it if it catches up.
                return PermissionDecision::Deny;
            }
            let (next, wait) = match self.condvar.wait_timeout(guard, remaining) {
                Ok(pair) => pair,
                Err(p) => p.into_inner(),
            };
            guard = next;
            if wait.timed_out() && !guard.decisions.contains_key(&prompt.id) {
                return PermissionDecision::Deny;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::thread;
    use std::time::{Duration, SystemTime};

    use stratum_runtime::{PermissionRequest, PromptId};

    use super::*;

    fn prompt(id: u64) -> PendingPrompt {
        PendingPrompt {
            id: PromptId(id),
            request: PermissionRequest::ToolUse { args: String::new(),
                tool_id: "fs.write".into(),
            },
            issued_at: SystemTime::UNIX_EPOCH,
        }
    }

    #[test]
    fn pending_request_initially_none() {
        let r = TuiPromptResponder::new(Duration::from_millis(50));
        assert!(r.pending_request().is_none());
    }

    #[test]
    fn submit_then_pending_returns_request_then_none() {
        let r = Arc::new(TuiPromptResponder::new(Duration::from_secs(1)));
        let r_clone = Arc::clone(&r);
        let handle = thread::spawn(move || r_clone.ask(&prompt(7)));
        // Spin briefly until the worker has enqueued its request.
        let pending = loop {
            if let Some(p) = r.pending_request() {
                break p;
            }
            thread::sleep(Duration::from_millis(5));
        };
        assert_eq!(pending.id, PromptId(7));
        // Queue must now be empty.
        assert!(r.pending_request().is_none());
        r.submit_decision(pending.id, PermissionDecision::AllowOnce);
        let decision = handle.join().expect("worker joins");
        assert_eq!(decision, PermissionDecision::AllowOnce);
    }

    #[test]
    fn ask_times_out_to_deny_when_no_decision() {
        let r = TuiPromptResponder::new(Duration::from_millis(50));
        let decision = r.ask(&prompt(1));
        assert_eq!(decision, PermissionDecision::Deny);
    }

    #[test]
    fn submit_decision_before_pending_request_pop_still_drains() {
        // Race-style: TUI somehow submits the decision *before* the worker
        // even reaches the wait. Decision should be picked up immediately.
        let r = Arc::new(TuiPromptResponder::new(Duration::from_secs(1)));
        r.submit_decision(PromptId(42), PermissionDecision::AllowForever);
        let decision = r.ask(&prompt(42));
        assert_eq!(decision, PermissionDecision::AllowForever);
        // The request was pushed before the decision was consumed but
        // ask() drains the decision side first; the request remains
        // queued for the TUI to discover.
        let pending = r.pending_request();
        assert!(pending.is_some());
        assert_eq!(pending.expect("pending").id, PromptId(42));
    }

    #[test]
    fn default_timeout_is_sixty_seconds() {
        let r = TuiPromptResponder::default();
        assert_eq!(r.timeout, Duration::from_secs(60));
    }

    #[test]
    fn multiple_requests_round_trip_independently() {
        let r = Arc::new(TuiPromptResponder::new(Duration::from_secs(2)));
        let r_a = Arc::clone(&r);
        let r_b = Arc::clone(&r);
        let h_a = thread::spawn(move || r_a.ask(&prompt(10)));
        let h_b = thread::spawn(move || r_b.ask(&prompt(11)));
        // Drain both pending requests, then answer in reverse order.
        let mut ids = Vec::new();
        while ids.len() < 2 {
            if let Some(p) = r.pending_request() {
                ids.push(p.id);
            } else {
                thread::sleep(Duration::from_millis(5));
            }
        }
        r.submit_decision(PromptId(11), PermissionDecision::Deny);
        r.submit_decision(PromptId(10), PermissionDecision::AllowOnce);
        assert_eq!(h_a.join().expect("a"), PermissionDecision::AllowOnce);
        assert_eq!(h_b.join().expect("b"), PermissionDecision::Deny);
        assert!(ids.contains(&PromptId(10)));
        assert!(ids.contains(&PromptId(11)));
    }
}
