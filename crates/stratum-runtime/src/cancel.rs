//! Cooperative cancellation primitive.
//!
//! Polled by Provider implementations between tokens and by tool processes
//! around their long-running calls. The same token is shared across an
//! agent and its subagents per `plan/32-cancellation-and-budgets.md`.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Lightweight, cloneable cancellation flag.
///
/// All clones of a token observe the same cancellation state; calling
/// [`Self::cancel`] on any clone flips every observer.
#[derive(Debug, Clone, Default)]
pub struct CancelToken {
    flag: Arc<AtomicBool>,
}

impl CancelToken {
    /// Build a fresh, uncancelled token.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Flip the token to the cancelled state. Idempotent.
    pub fn cancel(&self) {
        self.flag.store(true, Ordering::Release);
    }

    /// Has the token been cancelled?
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.flag.load(Ordering::Acquire)
    }

    /// Spawn a child token that follows the parent. Cancelling the parent
    /// flips the child; cancelling the child does **not** flip the parent.
    ///
    /// Both tokens share the same atomic, so cancellation propagates from
    /// parent to all descendants by construction.
    #[must_use]
    pub fn child(&self) -> Self {
        Self {
            flag: Arc::clone(&self.flag),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_token_is_not_cancelled() {
        assert!(!CancelToken::new().is_cancelled());
    }

    #[test]
    fn cancel_flips_state() {
        let t = CancelToken::new();
        t.cancel();
        assert!(t.is_cancelled());
    }

    #[test]
    fn cancel_is_idempotent() {
        let t = CancelToken::new();
        t.cancel();
        t.cancel();
        assert!(t.is_cancelled());
    }

    #[test]
    fn clones_share_state() {
        let parent = CancelToken::new();
        let clone = parent.clone();
        parent.cancel();
        assert!(clone.is_cancelled());
    }

    #[test]
    fn child_inherits_parent_cancel() {
        let parent = CancelToken::new();
        let child = parent.child();
        parent.cancel();
        assert!(child.is_cancelled());
    }

    #[test]
    fn cancel_propagates_from_clone() {
        let parent = CancelToken::new();
        let clone = parent.clone();
        clone.cancel();
        assert!(parent.is_cancelled());
    }

    #[test]
    fn default_constructor_matches_new() {
        let a = CancelToken::new();
        let b = CancelToken::default();
        assert_eq!(a.is_cancelled(), b.is_cancelled());
    }

    #[test]
    fn debug_renders() {
        let t = CancelToken::new();
        assert!(format!("{t:?}").contains("CancelToken"));
    }
}
