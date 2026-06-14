//! Canonical multi-role orchestrator.
//!
//! Wires the four canonical roles into one entry point:
//!
//! 1. **Router** — `IntentRouter` classifies the user prompt.
//! 2. **Loop** — `AgentLoop` runs the actual turn (tool dispatch +
//!    agentic continuation).
//! 3. **Reviewer** — `ReviewerPass` scores the draft using a SECOND
//!    provider (anti-self-bias per plan/17).
//! 4. **Polisher** — already wired as a continuation-prompt hint in
//!    `agent_loop::build_continuation_prompt`; this orchestrator
//!    surfaces the verdict back to the caller.
//!
//! ## Why this is opt-in
//!
//! Tests + low-effort callers want the single-loop semantics: send a
//! prompt, get a TurnResult, done. Production CLI flips
//! [`OrchestratorConfig::multi_role`] to `true` so users get the full
//! pipeline. Every existing call site keeps working.
//!
//! ## Per plan/03 + plan/17
//!
//! The role roster + tier targets live in `plan/02-model-roster.md`
//! and `plan/17-agent-roles.md`. This module is the **runtime glue**
//! that wires the existing role modules — it does NOT redefine roles.

use std::sync::Arc;

use crate::agent_loop::{AgentLoop, TurnContext, TurnResult};
use crate::cancel::CancelToken;
use crate::intent_router::{IntentRouter, RoutedIntent};
use crate::provider::Provider;
use crate::reviewer::{ReviewVerdict, ReviewerPass};

/// Configuration for the canonical orchestrator.
#[derive(Debug, Clone, Default)]
pub struct OrchestratorConfig {
    /// When `true`, run the full Router → Loop → Reviewer pipeline.
    /// When `false`, fall through to the bare [`AgentLoop`] for
    /// backward compatibility.
    pub multi_role: bool,
    /// When `multi_role` is on AND this is `Some`, run the reviewer
    /// pass after the main turn. Wired with a SECOND provider so the
    /// model isn't grading itself.
    pub reviewer: Option<Arc<ReviewerPass>>,
    /// Optional intent router. When `None`, the default
    /// `IntentRouter::default()` is used.
    pub router: Option<Arc<IntentRouter>>,
}

/// One turn's orchestrated result. Surfaces the routing decision +
/// optional reviewer verdict alongside the bare turn result.
#[derive(Debug, Clone)]
pub struct OrchestratedTurn {
    /// What the intent router classified the prompt as.
    pub intent: RoutedIntent,
    /// The agent-loop result.
    pub result: TurnResult,
    /// Reviewer verdict — `None` when reviewer is disabled.
    pub verdict: Option<ReviewVerdict>,
}

/// Canonical orchestrator. Holds a shared `AgentLoop` reference; the
/// orchestrator is cheap to clone and safe to share across threads.
#[derive(Debug, Clone)]
pub struct Orchestrator {
    loop_: Arc<AgentLoop>,
    config: OrchestratorConfig,
}

impl Orchestrator {
    /// Build an orchestrator wrapping an existing AgentLoop.
    #[must_use]
    pub fn new(loop_: Arc<AgentLoop>, config: OrchestratorConfig) -> Self {
        Self { loop_, config }
    }

    /// Run one turn through the canonical pipeline. When
    /// `config.multi_role` is `false`, this is identical to
    /// `loop_.run_turn(ctx, cancel)` + a default-routed envelope so
    /// callers always get the same return shape.
    pub fn run_turn(
        &self,
        ctx: TurnContext,
        cancel: &CancelToken,
    ) -> OrchestratedTurn {
        let router = self
            .config
            .router
            .as_deref()
            .cloned()
            .unwrap_or_default();
        let intent = router.classify(&ctx.user_prompt);

        if !self.config.multi_role {
            // Bare path: just run the loop. No reviewer, no extra hops.
            let result = self.loop_.run_turn(ctx, cancel);
            return OrchestratedTurn {
                intent,
                result,
                verdict: None,
            };
        }

        // Multi-role path: run the loop, then optionally review the
        // assistant draft. We do NOT (yet) hot-swap the producer
        // model based on intent — that lands when the dense-7B swap
        // controller is wired (see plan/02 §Roster).
        let user_prompt = ctx.user_prompt.clone();
        let result = self.loop_.run_turn(ctx, cancel);

        let verdict = self.config.reviewer.as_ref().and_then(|r| {
            let draft = extract_draft_text(&result);
            if draft.is_empty() {
                return None;
            }
            r.review(&user_prompt, &draft, cancel)
        });

        OrchestratedTurn {
            intent,
            result,
            verdict,
        }
    }
}

/// Concatenate the assistant draft from a [`TurnResult`]. Returns
/// the empty string when no `Block::Text` appears (e.g. tool-only
/// turn).
fn extract_draft_text(result: &TurnResult) -> String {
    use stratum_types::Block;
    result
        .blocks
        .iter()
        .filter_map(|b| match b {
            Block::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Convenience: a config the CLI typically wants — multi-role on,
/// using a reviewer if a second provider is available.
#[must_use]
pub fn cli_default_config(reviewer: Option<Arc<ReviewerPass>>) -> OrchestratorConfig {
    OrchestratorConfig {
        multi_role: true,
        reviewer,
        router: None,
    }
}

/// Backward-compatible default config. Used by tests and the
/// minimum-viable AgentLoop call path.
#[must_use]
pub fn legacy_single_loop_config() -> OrchestratorConfig {
    OrchestratorConfig::default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent_loop::{AgentLoopBuilder, AgentLoopConfig};
    use crate::event_log::{EventEmitter, MemoryEventSink};
    use crate::intent_router::IntentRouter;
    use crate::permission_prompt::{AllowAllResponder, PermissionStore, PromptIdGen};
    use crate::plan_mode::PlanMode;
    use crate::provider::EchoProvider;
    use crate::tool_invocation::RegistryDispatcher;
    use std::time::SystemTime;
    use stratum_types::ModelId;

    fn build_echo_loop() -> Arc<AgentLoop> {
        let provider: Arc<dyn Provider> = Arc::new(EchoProvider::new("echo: "));
        let events = Arc::new(EventEmitter::new(Arc::new(MemoryEventSink::new())));
        Arc::new(
            AgentLoopBuilder::default()
                .with_provider(provider)
                .with_router(IntentRouter::default())
                .with_permission_store(Arc::new(PermissionStore::new()))
                .with_prompt_gen(Arc::new(PromptIdGen::new()))
                .with_responder(Arc::new(AllowAllResponder))
                .with_events(events)
                .with_capability_matrix(Arc::new(crate::CapabilityMatrix::new()))
                .with_plan_mode(Arc::new(PlanMode::new()))
                .with_dispatcher(Arc::new(RegistryDispatcher::new()))
                .with_config(AgentLoopConfig::default())
                .build()
                .expect("build echo loop"),
        )
    }

    fn ctx_with(prompt: &str) -> TurnContext {
        TurnContext {
            user_prompt: prompt.into(),
            model: ModelId::from("echo"),
            turn_id: crate::observability::TurnId(1),
            started_at: SystemTime::UNIX_EPOCH,
            history: Vec::new(),
        }
    }

    #[test]
    fn legacy_config_runs_single_loop_and_no_reviewer() {
        let orch = Orchestrator::new(build_echo_loop(), legacy_single_loop_config());
        let cancel = CancelToken::new();
        let out = orch.run_turn(ctx_with("hi there"), &cancel);
        assert!(out.verdict.is_none(), "legacy path emits no reviewer verdict");
        // Result blocks come from the bare loop.
        assert!(!out.result.blocks.is_empty());
    }

    #[test]
    fn legacy_config_still_classifies_intent() {
        let orch = Orchestrator::new(build_echo_loop(), legacy_single_loop_config());
        let cancel = CancelToken::new();
        let out = orch.run_turn(ctx_with("read README.md"), &cancel);
        // Whatever the router decided, it must be one of the documented
        // RoutedIntent variants. Smoke-level — we just want the field
        // to populate every turn.
        let _ = out.intent;
    }

    #[test]
    fn multi_role_without_reviewer_skips_review() {
        let cfg = OrchestratorConfig {
            multi_role: true,
            reviewer: None,
            router: None,
        };
        let orch = Orchestrator::new(build_echo_loop(), cfg);
        let cancel = CancelToken::new();
        let out = orch.run_turn(ctx_with("hi"), &cancel);
        assert!(out.verdict.is_none());
    }

    #[test]
    fn multi_role_with_echo_reviewer_returns_none_verdict() {
        // EchoProvider isn't a real reviewer — its output won't parse
        // as JSON, so the verdict resolves to None. This documents
        // the graceful-degradation contract: bad reviewer ≠ broken turn.
        let provider: Arc<dyn Provider> = Arc::new(EchoProvider::new("echo: "));
        let reviewer = Arc::new(ReviewerPass::new(provider, ModelId::from("echo")));
        let cfg = OrchestratorConfig {
            multi_role: true,
            reviewer: Some(reviewer),
            router: None,
        };
        let orch = Orchestrator::new(build_echo_loop(), cfg);
        let cancel = CancelToken::new();
        let out = orch.run_turn(ctx_with("hi"), &cancel);
        assert!(out.verdict.is_none());
    }

    #[test]
    fn extract_draft_text_skips_non_text_blocks() {
        use stratum_types::Block;
        let result = TurnResult {
            turn_id: crate::observability::TurnId(1),
            outcome: crate::conversation::TurnOutcome::Success,
            blocks: vec![
                Block::Text { text: "hello".into() },
                Block::ToolCall {
                    id: "x".into(),
                    tool: "fs.read".into(),
                    args: "{}".into(),
                },
                Block::Text { text: "world".into() },
            ],
            transitions: Vec::new(),
            events_emitted: Vec::new(),
        };
        let draft = extract_draft_text(&result);
        assert_eq!(draft, "hello\nworld");
    }

    #[test]
    fn extract_draft_text_returns_empty_when_tool_only() {
        use stratum_types::Block;
        let result = TurnResult {
            turn_id: crate::observability::TurnId(1),
            outcome: crate::conversation::TurnOutcome::Success,
            blocks: vec![Block::ToolCall {
                id: "x".into(),
                tool: "fs.read".into(),
                args: "{}".into(),
            }],
            transitions: Vec::new(),
            events_emitted: Vec::new(),
        };
        assert!(extract_draft_text(&result).is_empty());
    }

    #[test]
    fn cli_default_enables_multi_role() {
        let cfg = cli_default_config(None);
        assert!(cfg.multi_role);
        assert!(cfg.reviewer.is_none());
    }
}
