//! Hierarchical cancellation cascade layered atop [`crate::cancel::CancelToken`].
//!
//! The flat `CancelToken` covers the simple "one shared flag" case. This module
//! adds a *child-aware* layer used by long-running agent/tool hierarchies per
//! `plan/32-cancellation-and-timeouts.md`:
//!
//! * Each [`CascadeToken`] is either a root or a child of another cascade.
//! * Cancelling a node cancels every still-live descendant with the same
//!   [`CancelReason::ParentCancelled`] (unless the descendant already has a
//!   reason recorded — first-write-wins).
//! * Children are held weakly so a dropped child does not pin its parent's
//!   tracking vector.
//! * A [`DeadlineGuard`] is an RAII timer that cancels the token with
//!   [`CancelReason::Timeout`] unless disarmed before drop.
//!
//! This module deliberately uses only `std::sync` + `std::thread` so it stays
//! usable from the sync runtime layer (no `tokio` dependency).

use std::error::Error;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex, Weak};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use serde::{Deserialize, Serialize};

/// Why a [`CascadeToken`] was cancelled.
///
/// `Eq` + `Hash` so callers can key reason-specific metrics off it; `serde` so
/// it can be embedded in turn telemetry / crash reports.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "reason", rename_all = "snake_case")]
pub enum CancelReason {
    /// User pressed Ctrl-C / clicked Stop.
    UserAbort,
    /// Deadline elapsed.
    Timeout {
        /// Configured deadline in milliseconds.
        after_ms: u64,
    },
    /// Cascaded from an ancestor that was cancelled.
    ParentCancelled,
    /// A tool invocation failed in a way that should abort the turn.
    ToolFailure {
        /// Tool that failed.
        tool: String,
        /// Catalog error code.
        code: String,
    },
    /// Token, time, or money budget exceeded.
    BudgetExceeded {
        /// Which budget tripped (`tokens`, `wall_ms`, `usd`).
        kind: String,
    },
}

impl fmt::Display for CancelReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UserAbort => f.write_str("user abort"),
            Self::Timeout { after_ms } => write!(f, "timeout after {after_ms}ms"),
            Self::ParentCancelled => f.write_str("parent cancelled"),
            Self::ToolFailure { tool, code } => {
                write!(f, "tool `{tool}` failed ({code})")
            }
            Self::BudgetExceeded { kind } => write!(f, "{kind} budget exceeded"),
        }
    }
}

/// Returned by [`CascadeToken::try_check`] when the token is cancelled.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CancelError {
    /// The reason recorded on the token at the time of the check.
    pub reason: CancelReason,
}

impl fmt::Display for CancelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "cancelled: {}", self.reason)
    }
}

impl Error for CancelError {}

/// Inner shared state of a cascade node. Held by `Arc` from the owning token
/// and by `Weak` from the parent's child list.
#[derive(Debug)]
struct Inner {
    cancelled: AtomicBool,
    reason: Mutex<Option<CancelReason>>,
    children: Mutex<Vec<Weak<Inner>>>,
}

impl Inner {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            cancelled: AtomicBool::new(false),
            reason: Mutex::new(None),
            children: Mutex::new(Vec::new()),
        })
    }

    /// Apply a reason if the node has no reason yet, then cascade to children.
    ///
    /// First-write-wins: if a reason is already set, neither this node's reason
    /// nor its cancelled flag are revisited, but the cascade *still* walks
    /// children so a late `cancel` on the parent reaches newly-added children.
    fn cancel(self: &Arc<Self>, reason: CancelReason) {
        // Lock-then-store: holding the reason lock across the atomic store keeps
        // `reason()` consistent with `is_cancelled()` for observers.
        let mut slot = match self.reason.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        if slot.is_none() {
            *slot = Some(reason);
            self.cancelled.store(true, Ordering::Release);
        }
        drop(slot);

        // Cascade. Prune dead weaks while we walk.
        let mut children = match self.children.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        children.retain(|w| w.strong_count() > 0);
        let live: Vec<Arc<Self>> = children.iter().filter_map(Weak::upgrade).collect();
        drop(children);

        for child in live {
            child.cancel(CancelReason::ParentCancelled);
        }
    }

    fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }

    fn reason(&self) -> Option<CancelReason> {
        match self.reason.lock() {
            Ok(g) => g.clone(),
            Err(poisoned) => poisoned.into_inner().clone(),
        }
    }

    fn register_child(self: &Arc<Self>, child: &Arc<Self>) {
        let mut children = match self.children.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        children.retain(|w| w.strong_count() > 0);
        children.push(Arc::downgrade(child));
    }
}

/// Hierarchical cancellation token.
///
/// Construct a root via [`Self::root`] and spawn children via [`Self::child`].
/// Clones share the same node; calling [`Self::cancel`] on any clone cancels
/// the node and all live descendants.
#[derive(Debug, Clone)]
pub struct CascadeToken {
    inner: Arc<Inner>,
}

impl CascadeToken {
    /// Build a fresh, uncancelled root token.
    #[must_use]
    pub fn root() -> Self {
        Self {
            inner: Inner::new(),
        }
    }

    /// Spawn a child of this node.
    ///
    /// If `self` is already cancelled, the returned child starts cancelled
    /// with [`CancelReason::ParentCancelled`]. The child is registered weakly
    /// so dropping it does not keep its slot in the parent's vector alive.
    #[must_use]
    pub fn child(&self) -> Self {
        let child = Self {
            inner: Inner::new(),
        };
        self.inner.register_child(&child.inner);

        // If the parent is already cancelled, propagate immediately so the
        // child cannot escape cancellation by being registered after the fact.
        if self.inner.is_cancelled() {
            child.inner.cancel(CancelReason::ParentCancelled);
        }
        child
    }

    /// Cancel this node (and cascade to descendants). First call wins:
    /// subsequent calls do not overwrite the recorded reason.
    pub fn cancel(&self, reason: CancelReason) {
        self.inner.cancel(reason);
    }

    /// Has this token been cancelled?
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.inner.is_cancelled()
    }

    /// The cancellation reason, if any.
    #[must_use]
    pub fn reason(&self) -> Option<CancelReason> {
        self.inner.reason()
    }

    /// Polling helper for long-running loops.
    ///
    /// # Errors
    /// Returns [`CancelError`] if the token is cancelled, carrying the
    /// recorded [`CancelReason`].
    pub fn try_check(&self) -> Result<(), CancelError> {
        if self.is_cancelled() {
            // Fall back to `ParentCancelled` if a racing cancel set the flag
            // before the reason slot — the public contract guarantees a reason
            // is observable whenever `is_cancelled()` is true, but we keep this
            // branch defensive rather than `unwrap`.
            let reason = self.reason().unwrap_or(CancelReason::ParentCancelled);
            Err(CancelError { reason })
        } else {
            Ok(())
        }
    }

    /// Arm a deadline that will cancel this token after `dur` unless disarmed.
    ///
    /// The returned [`DeadlineGuard`] must be kept alive for the duration of
    /// the operation it protects. Drop or [`DeadlineGuard::disarm`] before the
    /// deadline elapses to cancel the timer.
    #[must_use]
    pub fn with_deadline(&self, dur: Duration) -> DeadlineGuard {
        DeadlineGuard::new(self.clone(), dur)
    }
}

/// RAII deadline timer attached to a [`CascadeToken`].
///
/// Spawns a background thread that sleeps for `dur` and then cancels the
/// token with [`CancelReason::Timeout`]. Call [`Self::disarm`] (or drop) to
/// stop the timer; if disarmed before the deadline, no cancellation happens.
#[derive(Debug)]
pub struct DeadlineGuard {
    tx: Option<Sender<()>>,
    join: Option<JoinHandle<()>>,
}

impl DeadlineGuard {
    fn new(token: CascadeToken, dur: Duration) -> Self {
        let (tx, rx) = mpsc::channel::<()>();
        let after_ms = u64::try_from(dur.as_millis()).unwrap_or(u64::MAX);
        let join = thread::spawn(move || match rx.recv_timeout(dur) {
            Err(RecvTimeoutError::Timeout) => {
                token.cancel(CancelReason::Timeout { after_ms });
            }
            Ok(()) | Err(RecvTimeoutError::Disconnected) => {
                // Disarmed (either via explicit signal or by Sender drop).
            }
        });
        Self {
            tx: Some(tx),
            join: Some(join),
        }
    }

    /// Disarm the timer. Safe to call multiple times; subsequent calls are
    /// no-ops. After disarm the protected token will not be cancelled by this
    /// guard, even if the deadline has not yet elapsed.
    pub fn disarm(mut self) {
        self.disarm_in_place();
    }

    fn disarm_in_place(&mut self) {
        if let Some(tx) = self.tx.take() {
            // Best-effort: if the worker already fired, send will fail and we
            // simply join below.
            let _ = tx.send(());
        }
        if let Some(handle) = self.join.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for DeadlineGuard {
    fn drop(&mut self) {
        self.disarm_in_place();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Instant;

    #[test]
    fn root_starts_uncancelled() {
        let t = CascadeToken::root();
        assert!(!t.is_cancelled());
        assert!(t.reason().is_none());
        assert!(t.try_check().is_ok());
    }

    #[test]
    fn cancel_sets_state_and_reason() {
        let t = CascadeToken::root();
        t.cancel(CancelReason::UserAbort);
        assert!(t.is_cancelled());
        assert_eq!(t.reason(), Some(CancelReason::UserAbort));
    }

    #[test]
    fn first_write_wins() {
        let t = CascadeToken::root();
        t.cancel(CancelReason::UserAbort);
        t.cancel(CancelReason::BudgetExceeded {
            kind: "tokens".into(),
        });
        assert_eq!(t.reason(), Some(CancelReason::UserAbort));
    }

    #[test]
    fn parent_cancels_single_child() {
        let parent = CascadeToken::root();
        let child = parent.child();
        parent.cancel(CancelReason::UserAbort);
        assert!(child.is_cancelled());
        assert_eq!(child.reason(), Some(CancelReason::ParentCancelled));
    }

    #[test]
    fn grandparent_cascades_to_grandchild() {
        let gp = CascadeToken::root();
        let parent = gp.child();
        let gc = parent.child();
        gp.cancel(CancelReason::UserAbort);
        assert!(parent.is_cancelled());
        assert!(gc.is_cancelled());
        assert_eq!(gc.reason(), Some(CancelReason::ParentCancelled));
    }

    #[test]
    fn child_cancel_does_not_propagate_to_parent() {
        let parent = CascadeToken::root();
        let child = parent.child();
        child.cancel(CancelReason::UserAbort);
        assert!(child.is_cancelled());
        assert!(!parent.is_cancelled());
        assert!(parent.reason().is_none());
    }

    #[test]
    fn sibling_isolation() {
        let parent = CascadeToken::root();
        let a = parent.child();
        let b = parent.child();
        a.cancel(CancelReason::UserAbort);
        assert!(a.is_cancelled());
        assert!(!b.is_cancelled());
        assert!(!parent.is_cancelled());
    }

    #[test]
    fn dropping_child_leaves_parent_intact() {
        let parent = CascadeToken::root();
        {
            let _child = parent.child();
        }
        // Force a walk of the children list; the dead weak should be pruned.
        parent.cancel(CancelReason::UserAbort);
        assert!(parent.is_cancelled());
    }

    #[test]
    fn deadline_fires_within_window() {
        let t = CascadeToken::root();
        let start = Instant::now();
        let _guard = t.with_deadline(Duration::from_millis(50));
        // Poll up to 250ms for the timer to fire.
        while start.elapsed() < Duration::from_millis(250) && !t.is_cancelled() {
            thread::sleep(Duration::from_millis(5));
        }
        assert!(t.is_cancelled(), "deadline guard never fired");
        assert_eq!(t.reason(), Some(CancelReason::Timeout { after_ms: 50 }));
    }

    #[test]
    fn deadline_disarm_prevents_cancel() {
        let t = CascadeToken::root();
        let guard = t.with_deadline(Duration::from_millis(50));
        guard.disarm();
        thread::sleep(Duration::from_millis(120));
        assert!(!t.is_cancelled());
        assert!(t.reason().is_none());
    }

    #[test]
    fn deadline_drop_disarms_when_short_circuit() {
        let t = CascadeToken::root();
        {
            let _guard = t.with_deadline(Duration::from_secs(60));
            // Drop the guard immediately — the worker should observe the
            // Sender drop and exit without cancelling.
        }
        thread::sleep(Duration::from_millis(40));
        assert!(!t.is_cancelled());
    }

    #[test]
    fn try_check_reports_reason() {
        let t = CascadeToken::root();
        assert!(t.try_check().is_ok());
        t.cancel(CancelReason::BudgetExceeded {
            kind: "wall_ms".into(),
        });
        let err = match t.try_check() {
            Ok(()) => unreachable!("token was cancelled"),
            Err(e) => e,
        };
        assert_eq!(
            err.reason,
            CancelReason::BudgetExceeded {
                kind: "wall_ms".into()
            }
        );
        assert!(err.to_string().contains("wall_ms"));
    }

    #[test]
    fn cancel_reason_serde_roundtrip() {
        let variants = [
            CancelReason::UserAbort,
            CancelReason::Timeout { after_ms: 1234 },
            CancelReason::ParentCancelled,
            CancelReason::ToolFailure {
                tool: "fs.read".into(),
                code: "test-code".into(),
            },
            CancelReason::BudgetExceeded { kind: "usd".into() },
        ];
        for v in variants {
            let s = serde_json::to_string(&v).expect("serialize");
            let back: CancelReason = serde_json::from_str(&s).expect("deserialize");
            assert_eq!(v, back);
        }
    }

    #[test]
    fn cancel_error_display_includes_reason() {
        let err = CancelError {
            reason: CancelReason::UserAbort,
        };
        let s = err.to_string();
        assert!(s.contains("user abort"), "got: {s}");
    }

    #[test]
    fn cancel_reason_display_covers_all_variants() {
        assert_eq!(CancelReason::UserAbort.to_string(), "user abort");
        assert_eq!(
            CancelReason::Timeout { after_ms: 9 }.to_string(),
            "timeout after 9ms"
        );
        assert_eq!(
            CancelReason::ParentCancelled.to_string(),
            "parent cancelled"
        );
        assert_eq!(
            CancelReason::ToolFailure {
                tool: "t".into(),
                code: "c".into()
            }
            .to_string(),
            "tool `t` failed (c)"
        );
        assert_eq!(
            CancelReason::BudgetExceeded { kind: "k".into() }.to_string(),
            "k budget exceeded"
        );
    }

    #[test]
    fn cross_thread_cancel_is_visible() {
        let t = CascadeToken::root();
        let t2 = t.clone();
        let handle = thread::spawn(move || {
            thread::sleep(Duration::from_millis(10));
            t2.cancel(CancelReason::UserAbort);
        });
        // Spin a tiny bit for the cancel to land.
        let start = Instant::now();
        while !t.is_cancelled() && start.elapsed() < Duration::from_millis(500) {
            thread::sleep(Duration::from_millis(2));
        }
        handle.join().expect("thread join");
        assert!(t.is_cancelled());
        assert_eq!(t.reason(), Some(CancelReason::UserAbort));
    }

    #[test]
    fn weak_child_does_not_panic_after_drop() {
        let parent = CascadeToken::root();
        for _ in 0..4 {
            let _short = parent.child();
        }
        // All children dropped; the parent's vec holds only dead Weaks.
        parent.cancel(CancelReason::UserAbort);
        assert!(parent.is_cancelled());
    }

    #[test]
    fn child_of_cancelled_parent_starts_cancelled() {
        let parent = CascadeToken::root();
        parent.cancel(CancelReason::UserAbort);
        let child = parent.child();
        assert!(child.is_cancelled());
        assert_eq!(child.reason(), Some(CancelReason::ParentCancelled));
    }

    #[test]
    fn clone_shares_state() {
        let t = CascadeToken::root();
        let c = t.clone();
        t.cancel(CancelReason::UserAbort);
        assert!(c.is_cancelled());
        assert_eq!(c.reason(), Some(CancelReason::UserAbort));
    }

    #[test]
    fn debug_renders() {
        let t = CascadeToken::root();
        assert!(format!("{t:?}").contains("CascadeToken"));
        let err = CancelError {
            reason: CancelReason::UserAbort,
        };
        assert!(format!("{err:?}").contains("CancelError"));
    }
}
