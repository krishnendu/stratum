//! Role-based model swap controller.
//!
//! Per `plan/02-model-roster.md`: different roles run on different
//! model slugs. Code tasks want a dense 7B; chat tasks want a small
//! 4B IT; research wants a long-context model.
//!
//! This module is the **policy layer** — it maps a `RoutedIntent`
//! to a preferred slug. The **mechanism layer** (Arc-swap of
//! `Provider`) already exists in chat.rs as the model_switcher
//! closure; this module produces the slug it should swap to.
//!
//! ## What this is NOT
//!
//! - Not the model loader. The runtime already loads models lazily
//!   via the catalog + `LlamaCppProvider::open`.
//! - Not the hot-swap mechanism. That lives in `chat::ChatState`
//!   already (the `with_model_switcher` builder takes a closure).
//! - Not the eval suite. Eval runs through the orchestrator using
//!   the slug this router picks.

use serde::{Deserialize, Serialize};

use crate::intent_router::{RoutedIntent, SuggestedRole};

/// One row in the roster: maps a role → preferred model slug.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoleAssignment {
    /// Role label (matches `SuggestedRole` variant in lower-snake).
    pub role: String,
    /// Catalog slug — what `--model` would accept.
    pub slug: String,
}

/// Roster of role → slug assignments. Built from settings.json or
/// from a hardcoded default per `plan/02 §Default roster`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Roster {
    /// Default model when no role-specific rule matches.
    pub default_slug: String,
    /// One assignment per known role.
    pub assignments: Vec<RoleAssignment>,
}

impl Roster {
    /// Build the Phase 3 v2 default roster per `plan/02 §Roster`:
    ///
    /// | Role | Slug |
    /// |------|------|
    /// | Coder | qwen-coder-7b |
    /// | Polisher | gemma-4-e2b |
    /// | Researcher | deepseek-r1-7b |
    /// | Cavemanish | qwen3-0.6b |
    /// | Default | gemma-4-e4b |
    #[must_use]
    pub fn default_v2() -> Self {
        Self {
            default_slug: "gemma-4-e4b".to_string(),
            assignments: vec![
                RoleAssignment {
                    role: "coder".to_string(),
                    slug: "qwen-coder-7b".to_string(),
                },
                RoleAssignment {
                    role: "polisher".to_string(),
                    slug: "gemma-4-e2b".to_string(),
                },
                RoleAssignment {
                    role: "researcher".to_string(),
                    slug: "deepseek-r1-7b".to_string(),
                },
                RoleAssignment {
                    role: "cavemanish".to_string(),
                    slug: "qwen3-0.6b".to_string(),
                },
            ],
        }
    }

    /// Resolve a role to a slug. Returns `default_slug` when the role
    /// isn't in the roster.
    #[must_use]
    pub fn slug_for(&self, role: SuggestedRole) -> &str {
        let role_label = match role {
            SuggestedRole::Coder => "coder",
            SuggestedRole::Polisher => "polisher",
            SuggestedRole::Researcher => "researcher",
            SuggestedRole::Cavemanish => "cavemanish",
            SuggestedRole::Default => return &self.default_slug,
        };
        self.assignments
            .iter()
            .find(|a| a.role == role_label)
            .map_or(self.default_slug.as_str(), |a| a.slug.as_str())
    }
}

/// What the swap controller decided for one turn.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SwapDecision {
    /// Slug the controller recommends running this turn against.
    pub target_slug: String,
    /// True iff `target_slug` differs from the currently-active slug
    /// (the caller may skip the swap when it's a no-op).
    pub should_swap: bool,
    /// Reason for telemetry / debug.
    pub reason: String,
}

/// Controller that maps `(intent, current_slug)` → `SwapDecision`.
/// Holds the active roster; cheap to clone.
#[derive(Debug, Clone)]
pub struct RoleSwapController {
    roster: Roster,
}

impl RoleSwapController {
    /// Build from a roster.
    #[must_use]
    pub const fn new(roster: Roster) -> Self {
        Self { roster }
    }

    /// Use the v2 default roster.
    #[must_use]
    pub fn default_v2() -> Self {
        Self::new(Roster::default_v2())
    }

    /// Decide which slug to run this turn against. `current_slug` may
    /// be empty (no active model); in that case the recommended slug
    /// is returned with `should_swap = true`.
    #[must_use]
    pub fn decide(&self, intent: &RoutedIntent, current_slug: &str) -> SwapDecision {
        let target = self.roster.slug_for(intent.suggested_role).to_string();
        // Confidence floor: only swap when we're at least 0.4 confident.
        // Low-confidence routes shouldn't churn the active model.
        if intent.confidence < 0.4 {
            return SwapDecision {
                target_slug: current_slug.to_string(),
                should_swap: false,
                reason: format!(
                    "confidence {:.2} below 0.4 floor; keeping {current_slug}",
                    intent.confidence
                ),
            };
        }
        let should_swap = target != current_slug;
        let reason = if should_swap {
            format!(
                "role {:?} (conf {:.2}) → {target} (was {current_slug})",
                intent.suggested_role, intent.confidence
            )
        } else {
            format!(
                "role {:?} (conf {:.2}) keeps {target}",
                intent.suggested_role, intent.confidence
            )
        };
        SwapDecision {
            target_slug: target,
            should_swap,
            reason,
        }
    }

    /// Borrow the active roster (read-only).
    #[must_use]
    pub const fn roster(&self) -> &Roster {
        &self.roster
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intent_router::Intent;
    use crate::model_catalog::ModelTier;
    use std::collections::BTreeSet;

    fn intent(role: SuggestedRole, confidence: f32) -> RoutedIntent {
        RoutedIntent {
            intent: Intent::Chat,
            confidence,
            required_tier: ModelTier::Medium,
            suggested_role: role,
            hinted_capabilities: BTreeSet::new(),
        }
    }

    #[test]
    fn default_v2_roster_assigns_known_slugs() {
        let r = Roster::default_v2();
        assert_eq!(r.slug_for(SuggestedRole::Coder), "qwen-coder-7b");
        assert_eq!(r.slug_for(SuggestedRole::Polisher), "gemma-4-e2b");
        assert_eq!(r.slug_for(SuggestedRole::Researcher), "deepseek-r1-7b");
        assert_eq!(r.slug_for(SuggestedRole::Cavemanish), "qwen3-0.6b");
        assert_eq!(r.slug_for(SuggestedRole::Default), "gemma-4-e4b");
    }

    #[test]
    fn controller_decides_swap_when_target_differs() {
        let c = RoleSwapController::default_v2();
        let d = c.decide(&intent(SuggestedRole::Coder, 0.9), "gemma-4-e4b");
        assert!(d.should_swap);
        assert_eq!(d.target_slug, "qwen-coder-7b");
    }

    #[test]
    fn controller_no_swap_when_already_on_target() {
        let c = RoleSwapController::default_v2();
        let d = c.decide(&intent(SuggestedRole::Coder, 0.9), "qwen-coder-7b");
        assert!(!d.should_swap);
        assert_eq!(d.target_slug, "qwen-coder-7b");
    }

    #[test]
    fn low_confidence_keeps_current_model() {
        let c = RoleSwapController::default_v2();
        let d = c.decide(&intent(SuggestedRole::Coder, 0.2), "gemma-4-e4b");
        assert!(!d.should_swap);
        assert_eq!(d.target_slug, "gemma-4-e4b");
        assert!(d.reason.contains("below 0.4 floor"));
    }

    #[test]
    fn unknown_role_falls_back_to_default() {
        let r = Roster {
            default_slug: "fallback-slug".to_string(),
            assignments: Vec::new(),
        };
        assert_eq!(r.slug_for(SuggestedRole::Coder), "fallback-slug");
    }

    #[test]
    fn empty_current_slug_triggers_swap() {
        let c = RoleSwapController::default_v2();
        let d = c.decide(&intent(SuggestedRole::Coder, 0.9), "");
        assert!(d.should_swap);
        assert!(!d.target_slug.is_empty());
    }

    #[test]
    fn reason_string_includes_role_and_confidence() {
        let c = RoleSwapController::default_v2();
        let d = c.decide(&intent(SuggestedRole::Coder, 0.85), "gemma-4-e4b");
        assert!(d.reason.contains("Coder"));
        assert!(d.reason.contains("0.85"));
    }

    #[test]
    fn roster_serde_roundtrip() {
        let r = Roster::default_v2();
        let s = serde_json::to_string(&r).unwrap();
        let back: Roster = serde_json::from_str(&s).unwrap();
        assert_eq!(r.default_slug, back.default_slug);
        assert_eq!(r.assignments.len(), back.assignments.len());
    }
}
