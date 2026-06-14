//! Embedded system prompts for the caveman rewriter and polisher.
//!
//! Stratum's internal agents exchange messages in **caveman** — a
//! compressed dialect that saves tokens, latency, and context budget. The
//! [`CAVEMAN_REWRITER`] prompt rewrites a user's normal-English input into
//! caveman for downstream agents; the [`POLISHER`] prompt rewrites
//! caveman agent output back into user-facing English before the TUI
//! renders it.
//!
//! The prompt text lives under `crates/stratum-runtime/prompts/` and is
//! pulled in at compile time via `include_str!` so the runtime binary
//! ships with the prompts baked in — no filesystem lookup, no install
//! path to misconfigure. See `plan/28-finalization-v2.md`.

/// System prompt for the caveman rewriter agent.
///
/// Takes user input in normal English and rewrites it in caveman style.
pub const CAVEMAN_REWRITER: &str = include_str!("../prompts/caveman_rewriter.md");

/// System prompt for the polisher agent.
///
/// Takes caveman agent output and rewrites it back into user-facing English.
pub const POLISHER: &str = include_str!("../prompts/polisher.md");

/// System prompt for the reviewer agent.
///
/// Scores an assistant draft against a fixed checklist and returns a
/// JSON `{"verdict":"clean|fix","issues":[…],"severity":"low|medium|high"}`.
/// Run as a separate pass against a second provider so the model
/// isn't grading itself (plan/17 §Critic — anti-self-bias).
pub const REVIEWER: &str = include_str!("../prompts/reviewer.md");

/// Identifies which embedded system prompt to load.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PromptRole {
    /// The caveman rewriter that compresses user input.
    CavemanRewriter,
    /// The polisher that expands caveman output back to English.
    Polisher,
    /// The reviewer that scores assistant drafts against a checklist.
    Reviewer,
}

/// Look up the embedded system prompt for a given role.
///
/// Returns a `'static` slice into the binary; no allocation.
#[must_use]
pub const fn system_prompt(role: PromptRole) -> &'static str {
    match role {
        PromptRole::CavemanRewriter => CAVEMAN_REWRITER,
        PromptRole::Polisher => POLISHER,
        PromptRole::Reviewer => REVIEWER,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Indirection so the `const_is_empty` lint cannot see the constant.
    fn len_of(s: &str) -> usize {
        s.len()
    }

    #[test]
    fn caveman_rewriter_non_empty() {
        assert!(len_of(CAVEMAN_REWRITER) > 0);
    }

    #[test]
    fn polisher_non_empty() {
        assert!(len_of(POLISHER) > 0);
    }

    #[test]
    fn system_prompt_caveman_rewriter_matches_const() {
        assert_eq!(system_prompt(PromptRole::CavemanRewriter), CAVEMAN_REWRITER);
    }

    #[test]
    fn system_prompt_polisher_matches_const() {
        assert_eq!(system_prompt(PromptRole::Polisher), POLISHER);
    }

    #[test]
    fn caveman_rewriter_mentions_caveman_and_stratum() {
        assert!(CAVEMAN_REWRITER.contains("caveman"));
        assert!(CAVEMAN_REWRITER.contains("Stratum"));
    }

    #[test]
    fn polisher_mentions_caveman_and_stratum() {
        assert!(POLISHER.contains("caveman"));
        assert!(POLISHER.contains("Stratum"));
    }

    #[test]
    fn caveman_rewriter_preserves_error_code_contract() {
        assert!(CAVEMAN_REWRITER.contains("STRAT-E"));
    }

    #[test]
    fn polisher_preserves_error_code_contract() {
        assert!(POLISHER.contains("STRAT-E"));
    }

    #[test]
    fn prompt_role_debug_renders() {
        assert!(format!("{:?}", PromptRole::CavemanRewriter).contains("CavemanRewriter"));
        assert!(format!("{:?}", PromptRole::Polisher).contains("Polisher"));
    }

    #[test]
    fn prompt_role_equality_and_clone() {
        let a = PromptRole::CavemanRewriter;
        let b = a;
        assert_eq!(a, b);
        assert_ne!(PromptRole::CavemanRewriter, PromptRole::Polisher);
    }
}
