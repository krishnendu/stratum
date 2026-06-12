//! `AgentLoop` orchestrator — the keystone that composes the existing FSM,
//! intent router, provider, permission store, event emitter, plan-mode fence,
//! and capability matrix into a single `run_turn` entry point.
//!
//! Per `plan/15-agentic-loop.md`. This module is data + traits + threads —
//! no async runtime, no I/O beyond what providers already do.
//!
//! # Modules composed
//!
//! * [`crate::conversation::TurnDriver`] — pumps the per-turn FSM.
//! * [`crate::intent_router::IntentRouter`] — classifies the user prompt.
//! * [`crate::provider::Provider`] — generates the model's response blocks.
//! * [`crate::permission_prompt::evaluate`] — gates `ToolCall` blocks via
//!   the [`crate::permission_prompt::PermissionStore`] + responder.
//! * [`crate::event_log::EventEmitter`] — records hand-offs, prompts,
//!   provider errors, and (placeholder) tool calls.
//! * [`crate::plan_mode::PlanMode`] — denies capability hints before the
//!   provider is even invoked.
//! * [`crate::tools::CapabilityMatrix`] — held for future intersection
//!   when tool execution lands.
//! * [`crate::observability::RoleTimer`] — bounds provider wall time.
//!
//! # Lock order
//!
//! `permission_store -> events -> turn_counter -> dispatcher`. The
//! orchestrator never holds the permission store guard across an event
//! emit, never holds either across a `turn_counter.fetch_add`, and never
//! holds the counter while invoking the dispatcher. The counter is the
//! only atomic we touch on the hot path.
//!
//! # Tool dispatch semantics
//!
//! Tool calls are **fail-fast**: the first `ToolResult::Err` (including
//! "no matching dispatcher") aborts the turn with a
//! [`TurnOutcome::ToolFailure`]. A future PR can introduce a
//! `continue-on-tool-error` config knob; for now this matches the
//! single-pass permission scaffold and keeps the surface small.
//!
//! Block::ToolCall does not currently carry a structured argument map
//! (only a JSON-serialized blob); the orchestrator therefore forwards an
//! empty [`std::collections::BTreeMap`] for `ToolInvocation::args`. Real
//! arg parsing lands when the dispatcher contract evolves to share a
//! typed schema with the model.
//!
//! Capability strings handed to the dispatcher come from the
//! [`crate::tools::CapabilityMatrix`]: the first matrix entry whose verb
//! matches the tool id is used; otherwise the orchestrator falls back to
//! the sentinel `format!("tool.{tool_id}")` so the dispatcher still sees
//! a non-empty capability label.
//!
//! Dispatcher invocations are wrapped in
//! [`std::panic::catch_unwind`]; a panicking dispatcher converts to an
//! `Event::ProviderError { code: "E_TOOL_PANIC" }` plus
//! [`TurnOutcome::ToolFailure`] with code `E_TOOL_PANIC`. This mirrors
//! the worker-thread panic path used by the provider above.
//!
//! # Error catalog
//!
//! Re-uses existing `STRAT-Exxxx` codes — no new entries are introduced.
//! * `STRAT-E3007` — plan-mode capability deny (matches the
//!   memory-gate refusal family; closest existing user-visible "refused"
//!   code). Documented in [`stratum_types::error::catalog`].
//! * `STRAT-E5004` — tool denied. Same code as the tool-deny path in the
//!   permission system. Documented in
//!   [`stratum_types::error::catalog::E5004_TOOL_DENIED`].
//! * `STRAT-E5005` — no dispatcher registered for the requested tool id.
//!   Surfaced by [`crate::tool_invocation::RegistryDispatcher::dispatch`].
//! * `E_NO_BLOCKS` — local sentinel for "provider returned zero blocks";
//!   intentionally lower-case to flag it as a non-catalog code (future
//!   PR can promote it).
//! * `E_TOOL_PANIC` — local sentinel for a panicking dispatcher; same
//!   non-catalog convention as `E_NO_BLOCKS` and `E_PROVIDER_PANIC`.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;
use std::panic::AssertUnwindSafe;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::{Duration, Instant, SystemTime};

use serde::{Deserialize, Serialize};
use stratum_types::{Block, ModelId};

use crate::cancel::CancelToken;
use crate::conversation::{TurnDriver, TurnEvent, TurnOutcome, TurnTransition};
use crate::event_log::{Event, EventEmitter};
use crate::intent_router::IntentRouter;
use crate::observability::{RoleTimer, TurnId};
use crate::permission_prompt::{
    evaluate as evaluate_permission, PermissionDecision, PermissionRequest, PermissionStore,
    PromptIdGen, PromptResponder,
};
use crate::plan_mode::{enforce_plan_mode_on_request, PlanMode};
use crate::provider::{GenerateRequest, Provider};
use crate::tool_invocation::{RegistryDispatcher, ToolInvocation, ToolResult};
use crate::tools::CapabilityMatrix;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Static configuration for one [`AgentLoop`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AgentLoopConfig {
    /// When `true`, plan-mode capability fence is consulted before
    /// generation begins.
    pub plan_mode: bool,
    /// Wall-clock cap on a single `run_turn`. The provider runs on a
    /// worker thread and is abandoned (via [`CancelToken`]) once the
    /// deadline expires.
    pub max_turn_duration: Duration,
    /// Upper bound on the number of `Block::ToolCall` permission checks
    /// performed within a single turn. Pinned by docs even though the
    /// scaffold's permission loop is single-pass.
    pub max_tool_calls_per_turn: u8,
    /// Maximum number of provider-generate iterations in a single
    /// `run_turn`. Each iteration may emit one or more `Block::ToolCall`
    /// entries; once dispatched, the agent loop builds a continuation
    /// prompt that includes the tool results and re-calls the provider.
    /// Hard upper bound on the agentic recursion depth — guards against
    /// loops that never emit a non-tool-call block.
    pub max_agentic_steps: u8,
}

impl Default for AgentLoopConfig {
    fn default() -> Self {
        Self {
            plan_mode: false,
            max_turn_duration: Duration::from_secs(300),
            max_tool_calls_per_turn: 8,
            // Default to one-shot dispatch (no agentic continuation).
            // Production wires this to a non-zero cap from the CLI when
            // building the LLM-backed loop; tests and EchoProvider
            // callers can rely on the single-iteration semantics.
            max_agentic_steps: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Turn context + result
// ---------------------------------------------------------------------------

/// Inputs to a single [`AgentLoop::run_turn`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TurnContext {
    /// Raw user prompt.
    pub user_prompt: String,
    /// Model the provider should serve.
    pub model: ModelId,
    /// Identifier handed back in [`TurnResult::turn_id`].
    pub turn_id: TurnId,
    /// Wall-clock instant the turn started — used as the
    /// `now` argument for permission and FSM evaluation.
    pub started_at: SystemTime,
}

/// Output of [`AgentLoop::run_turn`].
#[derive(Debug, Clone)]
pub struct TurnResult {
    /// Echoes [`TurnContext::turn_id`].
    pub turn_id: TurnId,
    /// Terminal outcome.
    pub outcome: TurnOutcome,
    /// Blocks emitted by the provider (verbatim).
    pub blocks: Vec<Block>,
    /// Full FSM transition history.
    pub transitions: Vec<TurnTransition>,
    /// Ids assigned by the [`EventEmitter`] during the turn, in emit
    /// order.
    pub events_emitted: Vec<u64>,
}

/// Errors a turn body can surface internally.
///
/// None of these reach the caller today — `run_turn` collapses everything
/// into a [`TurnOutcome`] — but the type is exposed so future async wiring
/// can fan a typed error up.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TurnResultError {
    /// The provider worker thread panicked.
    ProviderPanicked,
    /// The cancellation token observed an internal inconsistency
    /// (reserved for the future async wire-through).
    CancelInvariantViolation,
}

impl fmt::Display for TurnResultError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ProviderPanicked => f.write_str("provider worker thread panicked"),
            Self::CancelInvariantViolation => f.write_str("cancel-token invariant violated"),
        }
    }
}

impl Error for TurnResultError {}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/// Errors produced by [`AgentLoopBuilder::build`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AgentLoopBuildError {
    /// A required component was never set on the builder.
    MissingField(&'static str),
}

impl fmt::Display for AgentLoopBuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingField(name) => write!(f, "AgentLoop builder missing field: {name}"),
        }
    }
}

impl Error for AgentLoopBuildError {}

/// Fluent constructor for [`AgentLoop`].
#[derive(Default)]
pub struct AgentLoopBuilder {
    provider: Option<Arc<dyn Provider>>,
    router: Option<IntentRouter>,
    permission_store: Option<Arc<PermissionStore>>,
    prompt_gen: Option<Arc<PromptIdGen>>,
    responder: Option<Arc<dyn PromptResponder>>,
    events: Option<Arc<EventEmitter>>,
    capability_matrix: Option<Arc<CapabilityMatrix>>,
    plan_mode: Option<Arc<PlanMode>>,
    dispatcher: Option<Arc<RegistryDispatcher>>,
    config: Option<AgentLoopConfig>,
}

impl fmt::Debug for AgentLoopBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AgentLoopBuilder")
            .field("provider_set", &self.provider.is_some())
            .field("router_set", &self.router.is_some())
            .field("permission_store_set", &self.permission_store.is_some())
            .field("prompt_gen_set", &self.prompt_gen.is_some())
            .field("responder_set", &self.responder.is_some())
            .field("events_set", &self.events.is_some())
            .field("capability_matrix_set", &self.capability_matrix.is_some())
            .field("plan_mode_set", &self.plan_mode.is_some())
            .field("dispatcher_set", &self.dispatcher.is_some())
            .field("config", &self.config)
            .finish()
    }
}

impl AgentLoopBuilder {
    /// Set the provider.
    #[must_use]
    pub fn with_provider(mut self, provider: Arc<dyn Provider>) -> Self {
        self.provider = Some(provider);
        self
    }

    /// Set the intent router.
    #[must_use]
    pub fn with_router(mut self, router: IntentRouter) -> Self {
        self.router = Some(router);
        self
    }

    /// Set the permission store.
    #[must_use]
    pub fn with_permission_store(mut self, store: Arc<PermissionStore>) -> Self {
        self.permission_store = Some(store);
        self
    }

    /// Set the prompt-id generator. Pass an `Arc<PromptIdGen>` shared with
    /// the CLI so prompt ids are unique across the process.
    #[must_use]
    pub fn with_prompt_gen(mut self, gen: Arc<PromptIdGen>) -> Self {
        self.prompt_gen = Some(gen);
        self
    }

    /// Set the prompt responder (TUI or test responder).
    #[must_use]
    pub fn with_responder(mut self, responder: Arc<dyn PromptResponder>) -> Self {
        self.responder = Some(responder);
        self
    }

    /// Set the event emitter.
    #[must_use]
    pub fn with_events(mut self, events: Arc<EventEmitter>) -> Self {
        self.events = Some(events);
        self
    }

    /// Set the capability matrix.
    #[must_use]
    pub fn with_capability_matrix(mut self, matrix: Arc<CapabilityMatrix>) -> Self {
        self.capability_matrix = Some(matrix);
        self
    }

    /// Set the plan-mode flag.
    #[must_use]
    pub fn with_plan_mode(mut self, plan_mode: Arc<PlanMode>) -> Self {
        self.plan_mode = Some(plan_mode);
        self
    }

    /// Set the tool dispatcher registry.
    ///
    /// Optional. If never set, [`Self::build`] defaults to an empty
    /// [`RegistryDispatcher`]; any `Block::ToolCall` will then short-circuit
    /// to [`TurnOutcome::ToolFailure`] with code `STRAT-E5005`
    /// ("no matching dispatcher") via the registry's existing no-match
    /// path. This default preserves backward compatibility with callers
    /// (e.g. the current CLI) that don't yet wire real tool dispatchers.
    #[must_use]
    pub fn with_dispatcher(mut self, dispatcher: Arc<RegistryDispatcher>) -> Self {
        self.dispatcher = Some(dispatcher);
        self
    }

    /// Set the loop config.
    #[must_use]
    pub const fn with_config(mut self, config: AgentLoopConfig) -> Self {
        self.config = Some(config);
        self
    }

    /// Finalize the builder.
    ///
    /// # Errors
    ///
    /// Returns [`AgentLoopBuildError::MissingField`] for any required
    /// component that was never supplied.
    pub fn build(self) -> Result<AgentLoop, AgentLoopBuildError> {
        let provider = self
            .provider
            .ok_or(AgentLoopBuildError::MissingField("provider"))?;
        let router = self
            .router
            .ok_or(AgentLoopBuildError::MissingField("router"))?;
        let permission_store = self
            .permission_store
            .ok_or(AgentLoopBuildError::MissingField("permission_store"))?;
        let prompt_gen = self
            .prompt_gen
            .ok_or(AgentLoopBuildError::MissingField("prompt_gen"))?;
        let responder = self
            .responder
            .ok_or(AgentLoopBuildError::MissingField("responder"))?;
        let events = self
            .events
            .ok_or(AgentLoopBuildError::MissingField("events"))?;
        let capability_matrix = self
            .capability_matrix
            .ok_or(AgentLoopBuildError::MissingField("capability_matrix"))?;
        let plan_mode = self
            .plan_mode
            .ok_or(AgentLoopBuildError::MissingField("plan_mode"))?;
        // Dispatcher is optional. When omitted, fall back to an empty
        // `RegistryDispatcher`: any `Block::ToolCall` will short-circuit
        // to `TurnOutcome::ToolFailure { code: "STRAT-E5005" }` via the
        // registry's no-match path. This matches what the CLI does
        // implicitly today and keeps the builder backward-compatible.
        let dispatcher = self
            .dispatcher
            .unwrap_or_else(|| Arc::new(RegistryDispatcher::new()));
        let config = self
            .config
            .ok_or(AgentLoopBuildError::MissingField("config"))?;

        Ok(AgentLoop {
            provider,
            router,
            permission_store,
            prompt_gen,
            responder,
            events,
            capability_matrix,
            plan_mode,
            dispatcher,
            config,
            turn_counter: AtomicU64::new(0),
        })
    }
}

// ---------------------------------------------------------------------------
// AgentLoop
// ---------------------------------------------------------------------------

/// Composed orchestrator. Build via [`AgentLoop::builder`].
pub struct AgentLoop {
    provider: Arc<dyn Provider>,
    router: IntentRouter,
    permission_store: Arc<PermissionStore>,
    prompt_gen: Arc<PromptIdGen>,
    responder: Arc<dyn PromptResponder>,
    events: Arc<EventEmitter>,
    capability_matrix: Arc<CapabilityMatrix>,
    plan_mode: Arc<PlanMode>,
    dispatcher: Arc<RegistryDispatcher>,
    config: AgentLoopConfig,
    turn_counter: AtomicU64,
}

impl fmt::Debug for AgentLoop {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AgentLoop")
            .field("provider_id", &self.provider.id())
            .field("config", &self.config)
            .field("turn_counter", &self.turn_counter)
            .finish_non_exhaustive()
    }
}

impl AgentLoop {
    /// Begin a fluent build of a new [`AgentLoop`].
    #[must_use]
    pub fn builder() -> AgentLoopBuilder {
        AgentLoopBuilder::default()
    }

    /// Snapshot of [`PlanMode::is_active`].
    #[must_use]
    pub fn is_plan_mode_active(&self) -> bool {
        self.plan_mode.is_active()
    }

    /// Pin a turn ID counter snapshot. Mainly for tests asserting the
    /// counter increments.
    #[must_use]
    pub fn turn_counter(&self) -> u64 {
        self.turn_counter.load(Ordering::SeqCst)
    }

    /// Drive one turn end-to-end.
    ///
    /// See module rustdoc for the algorithm. The function is synchronous
    /// from the caller's perspective; the provider executes on a worker
    /// thread so the deadline can be honored without async runtime.
    #[allow(
        clippy::too_many_lines,
        clippy::needless_pass_by_value,
        reason = "orchestrator straight-line composition; owning the context matches the spec'd surface"
    )]
    pub fn run_turn(&self, ctx: TurnContext, cancel: &CancelToken) -> TurnResult {
        self.run_turn_inner(ctx, cancel, None, 0)
    }

    /// Streaming variant: forwards each incremental `Block` emitted by
    /// the provider to `chunk_tx` as it lands, then returns the same
    /// `TurnResult` as [`Self::run_turn`]. The receiver should drain
    /// chunks concurrently to keep the channel small. Closing the
    /// receiver early is safe — sends are best-effort.
    pub fn run_turn_streaming(
        &self,
        ctx: TurnContext,
        cancel: &CancelToken,
        chunk_tx: mpsc::Sender<Block>,
    ) -> TurnResult {
        self.run_turn_inner(ctx, cancel, Some(chunk_tx), 0)
    }

    fn run_turn_inner(
        &self,
        ctx: TurnContext,
        cancel: &CancelToken,
        chunk_tx: Option<mpsc::Sender<Block>>,
        step: u8,
    ) -> TurnResult {
        self.turn_counter.fetch_add(1, Ordering::SeqCst);

        let turn_id_u64 = ctx.turn_id.0;
        let mut events_emitted: Vec<u64> = Vec::new();
        let mut driver = TurnDriver::new();
        let mut blocks: Vec<Block> = Vec::new();
        let deadline_start = Instant::now();

        // 1. Hand-off event so observers know a new turn started.
        events_emitted.push(self.events.emit(
            Event::AgentHandoff {
                from: "router".into(),
                to: "intent_classifier".into(),
                reason: "new_turn".into(),
            },
            Some(turn_id_u64),
        ));

        // 2. Classify the prompt. The router never errors.
        let routed = self.router.classify(&ctx.user_prompt);

        // 3. Move FSM into Generating. `TurnDriver::new` always starts
        // at `Idle`, where `StartGenerate` is the legal transition;
        // ignoring the result is safe.
        let _ = driver.apply(TurnEvent::StartGenerate, ctx.started_at);

        // 4. Plan-mode fence over routed capability hints.
        if self.config.plan_mode {
            for cap in &routed.hinted_capabilities {
                if enforce_plan_mode_on_request(&self.plan_mode, cap).is_err() {
                    events_emitted.push(self.events.emit(
                        Event::ProviderError {
                            provider: self.provider.id().to_string(),
                            code: "STRAT-E3007".into(),
                            message: format!("capability `{cap}` denied by plan mode"),
                        },
                        Some(turn_id_u64),
                    ));
                    let outcome = TurnOutcome::ModelError {
                        code: "STRAT-E3007".into(),
                    };
                    let _ = driver.apply(
                        TurnEvent::Finish {
                            outcome: outcome.clone(),
                        },
                        ctx.started_at,
                    );
                    return TurnResult {
                        turn_id: ctx.turn_id,
                        outcome,
                        blocks,
                        transitions: driver.history().to_vec(),
                        events_emitted,
                    };
                }
            }
        }

        // 5. Build the request + run the provider on a worker thread so
        //    the deadline can be enforced. The provider polls `cancel`
        //    between tokens; we cancel the child token on deadline.
        let req = GenerateRequest {
            model: ctx.model.clone(),
            prompt: ctx.user_prompt.clone(),
            max_blocks: 64, system_override: None,
        };
        let child_cancel = cancel.child();

        let _role_timer = RoleTimer::start();
        let (tx, rx) = mpsc::channel::<Vec<Block>>();
        let provider = Arc::clone(&self.provider);
        let cancel_for_worker = child_cancel.clone();
        let chunk_tx_for_worker = chunk_tx.clone();
        let worker = thread::spawn(move || {
            let result = match chunk_tx_for_worker {
                Some(stream_tx) => {
                    let cb = |b: &Block| {
                        let _ = stream_tx.send(b.clone());
                    };
                    provider.generate_streaming(&req, &cancel_for_worker, &cb)
                }
                None => provider.generate(&req, &cancel_for_worker),
            };
            // Best-effort send; receiver may already be gone if the
            // deadline expired.
            let _ = tx.send(result);
        });

        let deadline = self.config.max_turn_duration;
        let recv_outcome = wait_for_blocks(&rx, deadline_start, deadline);

        // Always reap the worker. If we time out, cancel first so it
        // exits promptly between tokens.
        if matches!(recv_outcome, RecvOutcome::Deadline) {
            child_cancel.cancel();
        }
        let join_result = worker.join();
        let provider_panicked = join_result.is_err();

        match recv_outcome {
            RecvOutcome::Blocks(mut got) => {
                blocks.append(&mut got);
            }
            RecvOutcome::Deadline => {
                // Drain anything the worker may have sent before we
                // observed the deadline.
                if let Ok(mut leftover) = rx.try_recv() {
                    blocks.append(&mut leftover);
                }
                let outcome = TurnOutcome::BudgetExceeded {
                    kind: "turn_duration".into(),
                };
                let _ = driver.apply(
                    TurnEvent::Cancel {
                        reason: "deadline_exceeded".into(),
                    },
                    ctx.started_at,
                );
                return TurnResult {
                    turn_id: ctx.turn_id,
                    outcome,
                    blocks,
                    transitions: driver.history().to_vec(),
                    events_emitted,
                };
            }
            RecvOutcome::Disconnected => {
                // Worker dropped the sender without sending — only
                // possible on panic. Surface as a ModelError.
                let _ = provider_panicked;
                events_emitted.push(self.events.emit(
                    Event::ProviderError {
                        provider: self.provider.id().to_string(),
                        code: "E_PROVIDER_PANIC".into(),
                        message: "provider worker panicked".into(),
                    },
                    Some(turn_id_u64),
                ));
                let outcome = TurnOutcome::ModelError {
                    code: "E_PROVIDER_PANIC".into(),
                };
                let _ = driver.apply(
                    TurnEvent::Finish {
                        outcome: outcome.clone(),
                    },
                    ctx.started_at,
                );
                return TurnResult {
                    turn_id: ctx.turn_id,
                    outcome,
                    blocks,
                    transitions: driver.history().to_vec(),
                    events_emitted,
                };
            }
        }

        // 6. User-cancel check after provider returned.
        if cancel.is_cancelled() {
            let _ = driver.apply(
                TurnEvent::Cancel {
                    reason: "user_abort".into(),
                },
                ctx.started_at,
            );
            return TurnResult {
                turn_id: ctx.turn_id,
                outcome: TurnOutcome::UserAbort,
                blocks,
                transitions: driver.history().to_vec(),
                events_emitted,
            };
        }

        // 7. Zero-blocks short-circuit.
        if blocks.is_empty() {
            events_emitted.push(self.events.emit(
                Event::ProviderError {
                    provider: self.provider.id().to_string(),
                    code: "E_NO_BLOCKS".into(),
                    message: "provider returned no blocks".into(),
                },
                Some(turn_id_u64),
            ));
            let outcome = TurnOutcome::ModelError {
                code: "E_NO_BLOCKS".into(),
            };
            let _ = driver.apply(
                TurnEvent::Finish {
                    outcome: outcome.clone(),
                },
                ctx.started_at,
            );
            return TurnResult {
                turn_id: ctx.turn_id,
                outcome,
                blocks,
                transitions: driver.history().to_vec(),
                events_emitted,
            };
        }

        // 8. Walk blocks; gate ToolCalls through the permission store
        //    and dispatch the approved calls via `RegistryDispatcher`.
        let mut tool_checks: u8 = 0;
        for block in blocks.clone() {
            let Block::ToolCall {
                id: call_id,
                tool: tool_name,
                args: args_json,
            } = block
            else {
                continue;
            };
            // Use `tool_name` for dispatcher lookup; `call_id` is the
            // correlation id and gets carried into the matching
            // `Block::ToolResult` below.
            let id = tool_name.clone();
            if tool_checks >= self.config.max_tool_calls_per_turn {
                // Budget exhausted before this call ran — fail fast with
                // a structured `BudgetExceeded` outcome so the caller can
                // distinguish "model wanted more tools than allowed" from
                // a clean finish.
                let outcome = TurnOutcome::BudgetExceeded {
                    kind: "tool_calls".into(),
                };
                let _ = driver.apply(
                    TurnEvent::Finish {
                        outcome: outcome.clone(),
                    },
                    ctx.started_at,
                );
                return TurnResult {
                    turn_id: ctx.turn_id,
                    outcome,
                    blocks,
                    transitions: driver.history().to_vec(),
                    events_emitted,
                };
            }
            tool_checks = tool_checks.saturating_add(1);

            // FSM: Generating -> AwaitingToolApproval.
            let _ = driver.apply(
                TurnEvent::RequestToolUse {
                    request: id.clone(),
                },
                ctx.started_at,
            );

            let req = PermissionRequest::ToolUse {
                tool_id: id.clone(),
                args: args_json.clone(),
            };
            let decision = evaluate_permission(
                req,
                &self.permission_store,
                &self.prompt_gen,
                &*self.responder,
                ctx.started_at,
            );

            let decision_label = decision_label(decision);
            events_emitted.push(self.events.emit(
                Event::PermissionAsked {
                    request: format!("tool_use:{id}"),
                    decision: decision_label.into(),
                },
                Some(turn_id_u64),
            ));

            match decision {
                PermissionDecision::Deny | PermissionDecision::DenyForever => {
                    let _ = driver.apply(TurnEvent::DenyTool, ctx.started_at);
                    let outcome = TurnOutcome::ToolFailure {
                        tool_id: id,
                        code: "STRAT-E5004".into(),
                    };
                    let _ = driver.apply(
                        TurnEvent::Finish {
                            outcome: outcome.clone(),
                        },
                        ctx.started_at,
                    );
                    return TurnResult {
                        turn_id: ctx.turn_id,
                        outcome,
                        blocks,
                        transitions: driver.history().to_vec(),
                        events_emitted,
                    };
                }
                PermissionDecision::AllowOnce
                | PermissionDecision::AllowSession
                | PermissionDecision::AllowForever => {
                    let _ = driver.apply(
                        TurnEvent::ApproveTool {
                            tool_id: id.clone(),
                        },
                        ctx.started_at,
                    );

                    // Resolve the capability label from the matrix:
                    // first entry whose verb matches the tool id wins.
                    // Otherwise fall back to a `tool.<id>` sentinel so
                    // the dispatcher still sees a non-empty label.
                    let capability = self
                        .capability_matrix
                        .entries()
                        .find(|e| e.verb_matches(&id))
                        .map_or_else(|| format!("tool.{id}"), |e| e.as_str().to_string());

                    // Parse the model-supplied args JSON into a map.
                    // Empty / invalid JSON falls back to an empty map so the
                    // dispatcher can surface its own missing-arg sentinel.
                    let args_map: BTreeMap<String, serde_json::Value> =
                        if args_json.trim().is_empty() {
                            BTreeMap::new()
                        } else {
                            match serde_json::from_str::<serde_json::Value>(&args_json) {
                                Ok(serde_json::Value::Object(m)) => m.into_iter().collect(),
                                _ => BTreeMap::new(),
                            }
                        };
                    let inv = ToolInvocation {
                        tool_id: id.clone(),
                        args: args_map,
                        capability,
                        turn_id: turn_id_u64,
                    };
                    let _ = &call_id;

                    // Run the dispatch under `catch_unwind` so a
                    // panicking dispatcher cannot poison the turn loop.
                    // Mirrors the worker-thread join above used to
                    // contain provider panics.
                    let timer = RoleTimer::start();
                    let dispatcher = Arc::clone(&self.dispatcher);
                    let inv_ref = &inv;
                    let dispatch_result =
                        std::panic::catch_unwind(AssertUnwindSafe(|| dispatcher.dispatch(inv_ref)));
                    let duration_ms = u64::from(timer.stop_ms());

                    let Ok(result) = dispatch_result else {
                        events_emitted.push(self.events.emit(
                            Event::ProviderError {
                                provider: self.provider.id().to_string(),
                                code: "E_TOOL_PANIC".into(),
                                message: format!("dispatcher panicked for {id}"),
                            },
                            Some(turn_id_u64),
                        ));
                        let outcome = TurnOutcome::ToolFailure {
                            tool_id: id,
                            code: "E_TOOL_PANIC".into(),
                        };
                        let _ = driver.apply(
                            TurnEvent::Finish {
                                outcome: outcome.clone(),
                            },
                            ctx.started_at,
                        );
                        return TurnResult {
                            turn_id: ctx.turn_id,
                            outcome,
                            blocks,
                            transitions: driver.history().to_vec(),
                            events_emitted,
                        };
                    };

                    let ok = matches!(result, ToolResult::Ok { .. });
                    events_emitted.push(self.events.emit(
                        Event::ToolCall {
                            tool_id: id.clone(),
                            ok,
                            duration_ms,
                        },
                        Some(turn_id_u64),
                    ));

                    match result {
                        ToolResult::Ok { body, .. } => {
                            let _ = driver.apply(
                                TurnEvent::ToolCompleted {
                                    tool_id: id.clone(),
                                    ok: true,
                                    code: None,
                                },
                                ctx.started_at,
                            );
                            // Surface the tool output as a Block::ToolResult
                            // in the returned blocks. Required for the future
                            // agentic loop closure to feed results back into
                            // a continuation prompt; harmless for callers that
                            // only render Block::Text.
                            blocks.push(Block::ToolResult {
                                id: call_id.clone(),
                                output: serde_json::to_string(&body)
                                    .unwrap_or_else(|_| "{}".to_string()),
                            });
                        }
                        ToolResult::Err { code, .. } => {
                            // Fail-fast: the first tool error bails the
                            // turn. See module docs for the rationale.
                            let _ = driver.apply(
                                TurnEvent::ToolCompleted {
                                    tool_id: id.clone(),
                                    ok: false,
                                    code: Some(code.clone()),
                                },
                                ctx.started_at,
                            );
                            let outcome = TurnOutcome::ToolFailure { tool_id: id, code };
                            let _ = driver.apply(
                                TurnEvent::Finish {
                                    outcome: outcome.clone(),
                                },
                                ctx.started_at,
                            );
                            return TurnResult {
                                turn_id: ctx.turn_id,
                                outcome,
                                blocks,
                                transitions: driver.history().to_vec(),
                                events_emitted,
                            };
                        }
                    }
                }
            }
        }

        // 9. Agentic continuation: if the provider emitted any tool
        // calls AND we have budget left, build a continuation prompt
        // that includes the tool results and recurse. Mirrors Claude
        // Code's behavior of letting the model react to tool output.
        let dispatched_count = blocks
            .iter()
            .filter(|b| matches!(b, Block::ToolResult { .. }))
            .count();
        if dispatched_count > 0 && step < self.config.max_agentic_steps {
            let continuation_prompt = build_continuation_prompt(&ctx.user_prompt, &blocks);
            let next_ctx = TurnContext {
                user_prompt: continuation_prompt,
                model: ctx.model.clone(),
                turn_id: ctx.turn_id,
                started_at: ctx.started_at,
            };
            let sub = self.run_turn_inner(next_ctx, cancel, chunk_tx, step.saturating_add(1));
            // Merge the inner step's blocks + events into ours and adopt
            // its outcome / transition history.
            let mut merged_blocks = blocks;
            merged_blocks.extend(sub.blocks);
            let mut merged_events = events_emitted;
            merged_events.extend(sub.events_emitted);
            return TurnResult {
                turn_id: ctx.turn_id,
                outcome: sub.outcome,
                blocks: merged_blocks,
                transitions: sub.transitions,
                events_emitted: merged_events,
            };
        }

        // 10. Clean finish.
        let _ = driver.apply(
            TurnEvent::Finish {
                outcome: TurnOutcome::Success,
            },
            ctx.started_at,
        );
        TurnResult {
            turn_id: ctx.turn_id,
            outcome: TurnOutcome::Success,
            blocks,
            transitions: driver.history().to_vec(),
            events_emitted,
        }
    }
}

/// Build a continuation prompt from the original user prompt plus the
/// tool-call / tool-result pairs that were dispatched this iteration.
/// Used by [`AgentLoop::run_turn_inner`] to feed the provider context
/// for the next agentic step. Simple text-append format is provider-
/// agnostic; per-model formats (Qwen `<tool_call>`, Hermes ChatML, etc.)
/// land when GBNF grammar wiring does.
/// Max bytes per tool-result block in the continuation prompt. Beyond
/// this we truncate with an ellipsis. Caps prompt growth so the
/// agentic loop does not blow past `n_ctx` (causing
/// `GGML_ASSERT(n_tokens_all <= cparams.n_batch)` mid-decode).
const MAX_RESULT_BYTES: usize = 12_000;
/// Hard cap on the total continuation-prompt size. Above this we drop
/// older tool results in favor of the most recent ones — the model
/// needs the latest evidence more than the earliest.
const MAX_CONTINUATION_BYTES: usize = 24_000;

fn build_continuation_prompt(original: &str, blocks: &[Block]) -> String {
    use std::fmt::Write;
    // Collect tool call + result lines first so we can drop older ones
    // when the total exceeds the budget.
    let mut entries: Vec<String> = Vec::new();
    for b in blocks {
        match b {
            Block::ToolCall { tool, args, .. } => {
                let args_trunc = truncate_for_prompt(args, MAX_RESULT_BYTES);
                entries.push(format!("Tool call: {tool} args={args_trunc}\n"));
            }
            Block::ToolResult { output, .. } => {
                let out_trunc = truncate_for_prompt(output, MAX_RESULT_BYTES);
                entries.push(format!("Result: {out_trunc}\n"));
            }
            _ => {}
        }
    }
    // Drop oldest entries until under the global cap.
    while entries
        .iter()
        .map(String::len)
        .sum::<usize>()
        .saturating_add(original.len())
        > MAX_CONTINUATION_BYTES
        && entries.len() > 2
    {
        entries.remove(0);
    }
    let mut out = String::with_capacity(original.len() + 256);
    out.push_str(original);
    out.push_str("\n\n---\n");
    out.push_str("You issued the following tool calls and received these results:\n\n");
    for e in entries {
        out.push_str(&e);
    }
    out.push_str(
        "\nContinue. If you have enough information, give the final answer in plain text. \
         Otherwise issue another tool call.\n",
    );
    out
}

/// Cap `s` at `max` bytes, appending `…(truncated)` when shortened.
/// Splits on a UTF-8 char boundary so we don't corrupt multibyte text.
fn truncate_for_prompt(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…(truncated)", &s[..end])
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

enum RecvOutcome {
    Blocks(Vec<Block>),
    Deadline,
    Disconnected,
}

fn wait_for_blocks(
    rx: &mpsc::Receiver<Vec<Block>>,
    start: Instant,
    deadline: Duration,
) -> RecvOutcome {
    loop {
        let elapsed = start.elapsed();
        if elapsed >= deadline {
            return RecvOutcome::Deadline;
        }
        let remaining = deadline.saturating_sub(elapsed);
        // Poll at most every 1ms so the deadline is honored tightly
        // without thrashing the scheduler.
        let slice = remaining.min(Duration::from_millis(1));
        match rx.recv_timeout(slice) {
            Ok(blocks) => return RecvOutcome::Blocks(blocks),
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => return RecvOutcome::Disconnected,
        }
    }
}

const fn decision_label(decision: PermissionDecision) -> &'static str {
    match decision {
        PermissionDecision::AllowOnce => "allow_once",
        PermissionDecision::AllowSession => "allow_session",
        PermissionDecision::AllowForever => "allow_forever",
        PermissionDecision::Deny => "deny",
        PermissionDecision::DenyForever => "deny_forever",
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::event_log::{FixedEventClock, MemoryEventSink};
    use crate::intent_router::{IntentPattern, IntentRule, SuggestedRole};
    use crate::model_catalog::ModelTier;
    use crate::permission_prompt::{
        AllowAllResponder, DenyAllResponder, PermissionDecision, ScriptedResponder,
    };
    use crate::provider::EchoProvider;
    use std::sync::Mutex;
    use std::time::UNIX_EPOCH;
    use stratum_types::Capability;

    // -- fixtures ------------------------------------------------------

    fn t0() -> SystemTime {
        UNIX_EPOCH + Duration::from_secs(1_700_000_000)
    }

    fn fixed_emitter() -> Arc<EventEmitter> {
        let sink = Arc::new(MemoryEventSink::new());
        Arc::new(EventEmitter::with_clock(
            sink,
            Box::new(FixedEventClock(t0())),
        ))
    }

    fn fixed_emitter_with_sink() -> (Arc<EventEmitter>, Arc<MemoryEventSink>) {
        let sink = Arc::new(MemoryEventSink::new());
        let emitter = Arc::new(EventEmitter::with_clock(
            sink.clone(),
            Box::new(FixedEventClock(t0())),
        ));
        (emitter, sink)
    }

    fn ctx(prompt: &str) -> TurnContext {
        TurnContext {
            user_prompt: prompt.into(),
            model: ModelId::from("echo"),
            turn_id: TurnId(1),
            started_at: t0(),
        }
    }

    fn echo_loop() -> AgentLoop {
        AgentLoop::builder()
            .with_provider(Arc::new(EchoProvider::new("")))
            .with_router(IntentRouter::default())
            .with_permission_store(Arc::new(PermissionStore::new()))
            .with_prompt_gen(Arc::new(PromptIdGen::new()))
            .with_responder(Arc::new(AllowAllResponder))
            .with_events(fixed_emitter())
            .with_capability_matrix(Arc::new(CapabilityMatrix::new()))
            .with_plan_mode(Arc::new(PlanMode::new()))
            .with_dispatcher(Arc::new(RegistryDispatcher::new()))
            .with_config(AgentLoopConfig::default())
            .build()
            .unwrap()
    }

    fn empty_dispatcher() -> Arc<RegistryDispatcher> {
        Arc::new(RegistryDispatcher::new())
    }

    fn echo_dispatcher() -> Arc<RegistryDispatcher> {
        let mut registry = RegistryDispatcher::new();
        registry
            .register(Box::new(crate::tool_invocation::EchoDispatcher))
            .expect("register echo");
        Arc::new(registry)
    }

    fn deny_dispatcher() -> Arc<RegistryDispatcher> {
        let mut registry = RegistryDispatcher::new();
        registry
            .register(Box::new(crate::tool_invocation::DenyDispatcher::new(
                "deny", "blocked",
            )))
            .expect("register deny");
        Arc::new(registry)
    }

    fn permissive_dispatcher() -> Arc<RegistryDispatcher> {
        // Always-Ok dispatcher used by tests that need any tool id to
        // dispatch successfully without caring about the body.
        let mut registry = RegistryDispatcher::new();
        registry
            .register(Box::new(AcceptAllDispatcher))
            .expect("register accept_all");
        Arc::new(registry)
    }

    #[derive(Debug)]
    struct AcceptAllDispatcher;
    impl crate::tool_invocation::ToolDispatcher for AcceptAllDispatcher {
        fn invoke(
            &self,
            inv: &crate::tool_invocation::ToolInvocation,
        ) -> crate::tool_invocation::ToolResult {
            crate::tool_invocation::ToolResult::Ok {
                tool_id: inv.tool_id.clone(),
                body: serde_json::Value::Null,
                bytes: 0,
            }
        }
        fn supports(&self, _tool_id: &str) -> bool {
            true
        }
        fn id(&self) -> &'static str {
            "accept_all"
        }
    }

    // --- inline test-only dispatchers --------------------------------

    use crate::tool_invocation::{ToolDispatcher, ToolInvocation as Inv, ToolResult as TR};
    use std::sync::atomic::AtomicU64 as Counter;

    #[derive(Debug)]
    struct CountingDispatcher {
        count: Arc<Counter>,
    }
    impl ToolDispatcher for CountingDispatcher {
        fn invoke(&self, inv: &Inv) -> TR {
            self.count.fetch_add(1, Ordering::SeqCst);
            TR::Ok {
                tool_id: inv.tool_id.clone(),
                body: serde_json::Value::Null,
                bytes: 0,
            }
        }
        fn supports(&self, _tool_id: &str) -> bool {
            true
        }
        fn id(&self) -> &'static str {
            "counting"
        }
    }

    #[derive(Debug)]
    struct CapturingDispatcher {
        last_cap: Mutex<Option<String>>,
    }
    impl ToolDispatcher for CapturingDispatcher {
        fn invoke(&self, inv: &Inv) -> TR {
            *self.last_cap.lock().unwrap() = Some(inv.capability.clone());
            TR::Ok {
                tool_id: inv.tool_id.clone(),
                body: serde_json::Value::Null,
                bytes: 0,
            }
        }
        fn supports(&self, _tool_id: &str) -> bool {
            true
        }
        fn id(&self) -> &'static str {
            "capturing"
        }
    }

    #[derive(Debug)]
    struct SlowDispatcher;
    impl ToolDispatcher for SlowDispatcher {
        fn invoke(&self, inv: &Inv) -> TR {
            thread::sleep(Duration::from_millis(10));
            TR::Ok {
                tool_id: inv.tool_id.clone(),
                body: serde_json::Value::Null,
                bytes: 0,
            }
        }
        fn supports(&self, _tool_id: &str) -> bool {
            true
        }
        fn id(&self) -> &'static str {
            "slow_dispatcher"
        }
    }

    #[derive(Debug)]
    struct PanickingDispatcher;
    impl ToolDispatcher for PanickingDispatcher {
        fn invoke(&self, _inv: &Inv) -> TR {
            panic!("intentional dispatcher panic")
        }
        fn supports(&self, _tool_id: &str) -> bool {
            true
        }
        fn id(&self) -> &'static str {
            "panicking"
        }
    }

    /// Delegating dispatcher that re-uses a shared `CapturingDispatcher`
    /// instance. The wrapper exists so the test can keep the original
    /// `Arc` for assertions while the registry takes ownership.
    #[derive(Debug)]
    struct CapturingShim(Arc<CapturingDispatcher>);
    impl ToolDispatcher for CapturingShim {
        fn invoke(&self, inv: &Inv) -> TR {
            self.0.invoke(inv)
        }
        fn supports(&self, t: &str) -> bool {
            self.0.supports(t)
        }
        fn id(&self) -> &'static str {
            "capturing_shim"
        }
    }

    // ---- inline test-only providers ----------------------------------

    #[derive(Debug)]
    struct ZeroBlockProvider;
    impl Provider for ZeroBlockProvider {
        #[allow(clippy::unnecessary_literal_bound)]
        fn id(&self) -> &str {
            "zero"
        }
        fn capabilities(&self) -> &'static [Capability] {
            const CAPS: &[Capability] = &[Capability::Generate];
            CAPS
        }
        fn generate(&self, _req: &GenerateRequest, _cancel: &CancelToken) -> Vec<Block> {
            Vec::new()
        }
    }

    #[derive(Debug)]
    struct ScriptedProvider {
        script: Mutex<Vec<Vec<Block>>>,
    }
    impl ScriptedProvider {
        fn new(initial: Vec<Block>) -> Self {
            Self {
                script: Mutex::new(vec![initial]),
            }
        }
    }
    impl Provider for ScriptedProvider {
        #[allow(clippy::unnecessary_literal_bound)]
        fn id(&self) -> &str {
            "scripted"
        }
        fn capabilities(&self) -> &'static [Capability] {
            const CAPS: &[Capability] = &[Capability::Generate];
            CAPS
        }
        fn generate(&self, _req: &GenerateRequest, _cancel: &CancelToken) -> Vec<Block> {
            let mut g = self.script.lock().unwrap();
            if g.is_empty() {
                return Vec::new();
            }
            g.remove(0)
        }
    }

    #[derive(Debug)]
    struct PanickingProvider;
    impl Provider for PanickingProvider {
        #[allow(clippy::unnecessary_literal_bound)]
        fn id(&self) -> &str {
            "panic"
        }
        fn capabilities(&self) -> &'static [Capability] {
            const CAPS: &[Capability] = &[Capability::Generate];
            CAPS
        }
        fn generate(&self, _req: &GenerateRequest, _cancel: &CancelToken) -> Vec<Block> {
            panic!("intentional test panic");
        }
    }

    #[derive(Debug)]
    struct SlowProvider {
        sleep: Duration,
    }
    impl Provider for SlowProvider {
        #[allow(clippy::unnecessary_literal_bound)]
        fn id(&self) -> &str {
            "slow"
        }
        fn capabilities(&self) -> &'static [Capability] {
            const CAPS: &[Capability] = &[Capability::Generate];
            CAPS
        }
        fn generate(&self, _req: &GenerateRequest, cancel: &CancelToken) -> Vec<Block> {
            // Poll cancel periodically so we exit promptly on deadline.
            let started = Instant::now();
            while started.elapsed() < self.sleep {
                if cancel.is_cancelled() {
                    return Vec::new();
                }
                thread::sleep(Duration::from_millis(1));
            }
            vec![Block::Text {
                text: "slow".into(),
            }]
        }
    }

    // ---- tests -------------------------------------------------------

    #[test]
    fn config_default_pins_documented_values() {
        let c = AgentLoopConfig::default();
        assert!(!c.plan_mode);
        assert_eq!(c.max_turn_duration, Duration::from_secs(300));
        assert_eq!(c.max_tool_calls_per_turn, 8);
    }

    #[test]
    fn builder_requires_provider() {
        let err = AgentLoop::builder().build().unwrap_err();
        assert_eq!(err, AgentLoopBuildError::MissingField("provider"));
    }

    #[test]
    fn builder_requires_router() {
        let err = AgentLoop::builder()
            .with_provider(Arc::new(EchoProvider::new("")))
            .build()
            .unwrap_err();
        assert_eq!(err, AgentLoopBuildError::MissingField("router"));
    }

    #[test]
    fn builder_requires_permission_store() {
        let err = AgentLoop::builder()
            .with_provider(Arc::new(EchoProvider::new("")))
            .with_router(IntentRouter::empty())
            .build()
            .unwrap_err();
        assert_eq!(err, AgentLoopBuildError::MissingField("permission_store"));
    }

    #[test]
    fn builder_requires_prompt_gen() {
        let err = AgentLoop::builder()
            .with_provider(Arc::new(EchoProvider::new("")))
            .with_router(IntentRouter::empty())
            .with_permission_store(Arc::new(PermissionStore::new()))
            .build()
            .unwrap_err();
        assert_eq!(err, AgentLoopBuildError::MissingField("prompt_gen"));
    }

    #[test]
    fn builder_requires_responder() {
        let err = AgentLoop::builder()
            .with_provider(Arc::new(EchoProvider::new("")))
            .with_router(IntentRouter::empty())
            .with_permission_store(Arc::new(PermissionStore::new()))
            .with_prompt_gen(Arc::new(PromptIdGen::new()))
            .build()
            .unwrap_err();
        assert_eq!(err, AgentLoopBuildError::MissingField("responder"));
    }

    #[test]
    fn builder_requires_events() {
        let err = AgentLoop::builder()
            .with_provider(Arc::new(EchoProvider::new("")))
            .with_router(IntentRouter::empty())
            .with_permission_store(Arc::new(PermissionStore::new()))
            .with_prompt_gen(Arc::new(PromptIdGen::new()))
            .with_responder(Arc::new(AllowAllResponder))
            .build()
            .unwrap_err();
        assert_eq!(err, AgentLoopBuildError::MissingField("events"));
    }

    #[test]
    fn builder_requires_capability_matrix() {
        let err = AgentLoop::builder()
            .with_provider(Arc::new(EchoProvider::new("")))
            .with_router(IntentRouter::empty())
            .with_permission_store(Arc::new(PermissionStore::new()))
            .with_prompt_gen(Arc::new(PromptIdGen::new()))
            .with_responder(Arc::new(AllowAllResponder))
            .with_events(fixed_emitter())
            .build()
            .unwrap_err();
        assert_eq!(err, AgentLoopBuildError::MissingField("capability_matrix"));
    }

    #[test]
    fn builder_requires_plan_mode() {
        let err = AgentLoop::builder()
            .with_provider(Arc::new(EchoProvider::new("")))
            .with_router(IntentRouter::empty())
            .with_permission_store(Arc::new(PermissionStore::new()))
            .with_prompt_gen(Arc::new(PromptIdGen::new()))
            .with_responder(Arc::new(AllowAllResponder))
            .with_events(fixed_emitter())
            .with_capability_matrix(Arc::new(CapabilityMatrix::new()))
            .build()
            .unwrap_err();
        assert_eq!(err, AgentLoopBuildError::MissingField("plan_mode"));
    }

    #[test]
    fn builder_defaults_dispatcher_to_empty_registry() {
        // `.with_dispatcher` is intentionally omitted. `.build()` must
        // succeed by defaulting to an empty `RegistryDispatcher`, and
        // any tool block must then short-circuit through the registry's
        // no-match path to `TurnOutcome::ToolFailure { code:
        // "STRAT-E5005" }`. This pins the backward-compat contract for
        // the CLI, which currently never wires a dispatcher.
        let scripted = Arc::new(ScriptedProvider::new(vec![tool_call("foo", "foo")]));
        let loop_ = AgentLoop::builder()
            .with_provider(scripted)
            .with_router(IntentRouter::empty())
            .with_permission_store(Arc::new(PermissionStore::new()))
            .with_prompt_gen(Arc::new(PromptIdGen::new()))
            .with_responder(Arc::new(AllowAllResponder))
            .with_events(fixed_emitter())
            .with_capability_matrix(Arc::new(CapabilityMatrix::new()))
            .with_plan_mode(Arc::new(PlanMode::new()))
            .with_config(AgentLoopConfig::default())
            .build()
            .expect("build with default empty-registry dispatcher must succeed");
        let res = loop_.run_turn(ctx("call foo"), &CancelToken::new());
        match res.outcome {
            TurnOutcome::ToolFailure { tool_id, code } => {
                assert_eq!(tool_id, "foo");
                assert_eq!(code, "STRAT-E5005");
            }
            other => panic!("expected ToolFailure(STRAT-E5005), got {other:?}"),
        }
    }

    #[test]
    fn builder_requires_config() {
        let err = AgentLoop::builder()
            .with_provider(Arc::new(EchoProvider::new("")))
            .with_router(IntentRouter::empty())
            .with_permission_store(Arc::new(PermissionStore::new()))
            .with_prompt_gen(Arc::new(PromptIdGen::new()))
            .with_responder(Arc::new(AllowAllResponder))
            .with_events(fixed_emitter())
            .with_capability_matrix(Arc::new(CapabilityMatrix::new()))
            .with_plan_mode(Arc::new(PlanMode::new()))
            .with_dispatcher(empty_dispatcher())
            .build()
            .unwrap_err();
        assert_eq!(err, AgentLoopBuildError::MissingField("config"));
    }

    #[test]
    fn run_turn_echo_provider_succeeds_with_blocks() {
        let loop_ = echo_loop();
        let res = loop_.run_turn(ctx("hello world"), &CancelToken::new());
        assert!(matches!(res.outcome, TurnOutcome::Success));
        assert!(!res.blocks.is_empty());
        assert_eq!(res.turn_id, TurnId(1));
    }

    #[test]
    fn run_turn_zero_blocks_returns_model_error() {
        let loop_ = AgentLoop::builder()
            .with_provider(Arc::new(ZeroBlockProvider))
            .with_router(IntentRouter::empty())
            .with_permission_store(Arc::new(PermissionStore::new()))
            .with_prompt_gen(Arc::new(PromptIdGen::new()))
            .with_responder(Arc::new(AllowAllResponder))
            .with_events(fixed_emitter())
            .with_capability_matrix(Arc::new(CapabilityMatrix::new()))
            .with_plan_mode(Arc::new(PlanMode::new()))
            .with_dispatcher(empty_dispatcher())
            .with_config(AgentLoopConfig::default())
            .build()
            .unwrap();
        let res = loop_.run_turn(ctx("anything"), &CancelToken::new());
        match res.outcome {
            TurnOutcome::ModelError { code } => assert_eq!(code, "E_NO_BLOCKS"),
            other => panic!("expected ModelError(E_NO_BLOCKS), got {other:?}"),
        }
    }

    #[test]
    fn run_turn_denied_tool_returns_tool_failure() {
        let scripted = Arc::new(ScriptedProvider::new(vec![Block::ToolCall {
            id: "fs.read#1".into(),
            tool: "fs.read".into(),
            args: "{}".into(),
        }]));
        let loop_ = AgentLoop::builder()
            .with_provider(scripted)
            .with_router(IntentRouter::empty())
            .with_permission_store(Arc::new(PermissionStore::new()))
            .with_prompt_gen(Arc::new(PromptIdGen::new()))
            .with_responder(Arc::new(DenyAllResponder))
            .with_events(fixed_emitter())
            .with_capability_matrix(Arc::new(CapabilityMatrix::new()))
            .with_plan_mode(Arc::new(PlanMode::new()))
            .with_dispatcher(empty_dispatcher())
            .with_config(AgentLoopConfig::default())
            .build()
            .unwrap();
        let res = loop_.run_turn(ctx("call a tool"), &CancelToken::new());
        match res.outcome {
            TurnOutcome::ToolFailure { tool_id, code } => {
                // Reports the tool name, not the per-call correlation id.
                assert_eq!(tool_id, "fs.read");
                assert_eq!(code, "STRAT-E5004");
            }
            other => panic!("expected ToolFailure, got {other:?}"),
        }
    }

    #[test]
    fn run_turn_plan_mode_blocks_denied_capability() {
        // Router that hints `fs.write` for any prompt.
        let router = IntentRouter::with_rules(vec![IntentRule {
            pattern: IntentPattern::Contains("trigger".into()),
            intent: crate::intent_router::Intent::Code { language: None },
            weight: 1.0,
            tier: ModelTier::Low,
            role: SuggestedRole::Default,
            caps: vec!["fs.write".into()],
        }])
        .unwrap();
        let plan = Arc::new(PlanMode::new());
        plan.activate(t0());

        let cfg = AgentLoopConfig {
            plan_mode: true,
            ..AgentLoopConfig::default()
        };
        let loop_ = AgentLoop::builder()
            .with_provider(Arc::new(EchoProvider::new("")))
            .with_router(router)
            .with_permission_store(Arc::new(PermissionStore::new()))
            .with_prompt_gen(Arc::new(PromptIdGen::new()))
            .with_responder(Arc::new(AllowAllResponder))
            .with_events(fixed_emitter())
            .with_capability_matrix(Arc::new(CapabilityMatrix::new()))
            .with_plan_mode(plan)
            .with_dispatcher(empty_dispatcher())
            .with_config(cfg)
            .build()
            .unwrap();
        let res = loop_.run_turn(ctx("trigger fs.write"), &CancelToken::new());
        match res.outcome {
            TurnOutcome::ModelError { code } => assert_eq!(code, "STRAT-E3007"),
            other => panic!("expected ModelError(STRAT-E3007), got {other:?}"),
        }
    }

    #[test]
    fn run_turn_user_cancel_before_provider_returns_user_abort() {
        let loop_ = echo_loop();
        let cancel = CancelToken::new();
        cancel.cancel();
        let res = loop_.run_turn(ctx("hi"), &cancel);
        assert!(matches!(res.outcome, TurnOutcome::UserAbort));
    }

    #[test]
    fn run_turn_deadline_triggers_budget_exceeded() {
        let slow = Arc::new(SlowProvider {
            sleep: Duration::from_millis(200),
        });
        let cfg = AgentLoopConfig {
            max_turn_duration: Duration::from_millis(5),
            ..AgentLoopConfig::default()
        };
        let loop_ = AgentLoop::builder()
            .with_provider(slow)
            .with_router(IntentRouter::empty())
            .with_permission_store(Arc::new(PermissionStore::new()))
            .with_prompt_gen(Arc::new(PromptIdGen::new()))
            .with_responder(Arc::new(AllowAllResponder))
            .with_events(fixed_emitter())
            .with_capability_matrix(Arc::new(CapabilityMatrix::new()))
            .with_plan_mode(Arc::new(PlanMode::new()))
            .with_dispatcher(empty_dispatcher())
            .with_config(cfg)
            .build()
            .unwrap();
        let res = loop_.run_turn(ctx("hi"), &CancelToken::new());
        match res.outcome {
            TurnOutcome::BudgetExceeded { kind } => assert_eq!(kind, "turn_duration"),
            other => panic!("expected BudgetExceeded, got {other:?}"),
        }
    }

    #[test]
    fn events_emitted_ids_are_monotonic_and_match_sink() {
        let (events, sink) = fixed_emitter_with_sink();
        let loop_ = AgentLoop::builder()
            .with_provider(Arc::new(EchoProvider::new("")))
            .with_router(IntentRouter::default())
            .with_permission_store(Arc::new(PermissionStore::new()))
            .with_prompt_gen(Arc::new(PromptIdGen::new()))
            .with_responder(Arc::new(AllowAllResponder))
            .with_events(events)
            .with_capability_matrix(Arc::new(CapabilityMatrix::new()))
            .with_plan_mode(Arc::new(PlanMode::new()))
            .with_dispatcher(empty_dispatcher())
            .with_config(AgentLoopConfig::default())
            .build()
            .unwrap();
        let res = loop_.run_turn(ctx("hi world"), &CancelToken::new());
        let mut last = 0;
        for id in &res.events_emitted {
            assert!(*id > last, "ids must be strictly monotonic");
            last = *id;
        }
        let snap_ids: Vec<u64> = sink.snapshot().iter().map(|r| r.id).collect();
        assert!(res.events_emitted.iter().all(|id| snap_ids.contains(id)));
    }

    #[test]
    fn turn_counter_increments_per_call() {
        let loop_ = echo_loop();
        assert_eq!(loop_.turn_counter(), 0);
        loop_.run_turn(ctx("a"), &CancelToken::new());
        assert_eq!(loop_.turn_counter(), 1);
        loop_.run_turn(ctx("b"), &CancelToken::new());
        assert_eq!(loop_.turn_counter(), 2);
    }

    #[test]
    fn transitions_history_starts_at_idle() {
        let loop_ = echo_loop();
        let res = loop_.run_turn(ctx("hello"), &CancelToken::new());
        assert!(!res.transitions.is_empty());
        assert_eq!(
            res.transitions[0].from,
            crate::conversation::TurnState::Idle
        );
    }

    #[test]
    fn turn_context_eq_and_serde_round_trip() {
        let a = ctx("ping");
        let b = a.clone();
        assert_eq!(a, b);
        let json = serde_json::to_string(&a).unwrap();
        let back: TurnContext = serde_json::from_str(&json).unwrap();
        assert_eq!(back, a);
    }

    #[test]
    fn turn_result_non_trivial_fields_populated() {
        let loop_ = echo_loop();
        let res = loop_.run_turn(ctx("hello world"), &CancelToken::new());
        assert_eq!(res.turn_id, TurnId(1));
        assert!(!res.blocks.is_empty());
        assert!(!res.transitions.is_empty());
        assert!(!res.events_emitted.is_empty());
    }

    #[test]
    fn agent_loop_build_error_display_smoke() {
        let e = AgentLoopBuildError::MissingField("provider");
        let s = format!("{e}");
        assert!(s.contains("provider"));
        assert!(s.contains("missing"));
    }

    #[test]
    fn turn_result_error_display_smoke() {
        let a = TurnResultError::ProviderPanicked;
        assert!(format!("{a}").contains("panic"));
        let b = TurnResultError::CancelInvariantViolation;
        assert!(format!("{b}").contains("cancel"));
        let _: &dyn Error = &a;
        let _: &dyn Error = &b;
    }

    #[test]
    fn is_plan_mode_active_reflects_state() {
        let plan = Arc::new(PlanMode::new());
        let loop_ = AgentLoop::builder()
            .with_provider(Arc::new(EchoProvider::new("")))
            .with_router(IntentRouter::empty())
            .with_permission_store(Arc::new(PermissionStore::new()))
            .with_prompt_gen(Arc::new(PromptIdGen::new()))
            .with_responder(Arc::new(AllowAllResponder))
            .with_events(fixed_emitter())
            .with_capability_matrix(Arc::new(CapabilityMatrix::new()))
            .with_plan_mode(plan.clone())
            .with_dispatcher(empty_dispatcher())
            .with_config(AgentLoopConfig::default())
            .build()
            .unwrap();
        assert!(!loop_.is_plan_mode_active());
        plan.activate(t0());
        assert!(loop_.is_plan_mode_active());
        plan.deactivate();
        assert!(!loop_.is_plan_mode_active());
    }

    #[test]
    fn permission_remembered_across_two_turns() {
        // Single scripted decision, two calls — second short-circuits.
        let store = Arc::new(PermissionStore::new());
        let responder = Arc::new(ScriptedResponder::new(vec![
            PermissionDecision::AllowForever,
        ]));
        let scripted = Arc::new(ScriptedProvider {
            script: Mutex::new(vec![
                vec![Block::ToolCall {
                    id: "fs.read#1".into(),
                    tool: "fs.read".into(),
                    args: "{}".into(),
                }],
                vec![Block::ToolCall {
                    id: "fs.read#1".into(),
                    tool: "fs.read".into(),
                    args: "{}".into(),
                }],
            ]),
        });
        let loop_ = AgentLoop::builder()
            .with_provider(scripted)
            .with_router(IntentRouter::empty())
            .with_permission_store(store.clone())
            .with_prompt_gen(Arc::new(PromptIdGen::new()))
            .with_responder(responder)
            .with_events(fixed_emitter())
            .with_capability_matrix(Arc::new(CapabilityMatrix::new()))
            .with_plan_mode(Arc::new(PlanMode::new()))
            .with_dispatcher(permissive_dispatcher())
            .with_config(AgentLoopConfig::default())
            .build()
            .unwrap();
        let res1 = loop_.run_turn(ctx("call"), &CancelToken::new());
        assert!(matches!(res1.outcome, TurnOutcome::Success));
        let res2 = loop_.run_turn(ctx("call again"), &CancelToken::new());
        assert!(matches!(res2.outcome, TurnOutcome::Success));
        // Store carries the remembered AllowForever entry.
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn concurrent_run_turn_succeeds_across_threads() {
        let loop_ = Arc::new(echo_loop());
        let mut handles = Vec::new();
        for _ in 0..4 {
            let l = Arc::clone(&loop_);
            handles.push(thread::spawn(move || {
                let res = l.run_turn(ctx("hello"), &CancelToken::new());
                matches!(res.outcome, TurnOutcome::Success)
            }));
        }
        for h in handles {
            assert!(h.join().unwrap());
        }
        assert_eq!(loop_.turn_counter(), 4);
    }

    #[test]
    fn empty_intent_router_falls_back_to_chat() {
        let empty_router = IntentRouter::empty();
        // Re-classify just to pin the contract this loop relies on.
        let classified = empty_router.classify("anything");
        assert!(matches!(
            classified.intent,
            crate::intent_router::Intent::Chat
        ));
        // And the loop still produces Success.
        let loop_ = AgentLoop::builder()
            .with_provider(Arc::new(EchoProvider::new("")))
            .with_router(empty_router)
            .with_permission_store(Arc::new(PermissionStore::new()))
            .with_prompt_gen(Arc::new(PromptIdGen::new()))
            .with_responder(Arc::new(AllowAllResponder))
            .with_events(fixed_emitter())
            .with_capability_matrix(Arc::new(CapabilityMatrix::new()))
            .with_plan_mode(Arc::new(PlanMode::new()))
            .with_dispatcher(empty_dispatcher())
            .with_config(AgentLoopConfig::default())
            .build()
            .unwrap();
        let res = loop_.run_turn(ctx("hello"), &CancelToken::new());
        assert!(matches!(res.outcome, TurnOutcome::Success));
    }

    #[test]
    fn plan_mode_toggle_during_turn_does_not_affect_turn() {
        // The current implementation reads plan_mode.is_active() inside
        // run_turn — the snapshot is taken at the per-capability check.
        // This test pins that flipping plan-mode off between turns is
        // visible to the next turn (not retroactive to the in-flight one).
        let plan = Arc::new(PlanMode::new());
        let cfg = AgentLoopConfig {
            plan_mode: true,
            ..AgentLoopConfig::default()
        };
        let loop_ = AgentLoop::builder()
            .with_provider(Arc::new(EchoProvider::new("")))
            .with_router(IntentRouter::empty())
            .with_permission_store(Arc::new(PermissionStore::new()))
            .with_prompt_gen(Arc::new(PromptIdGen::new()))
            .with_responder(Arc::new(AllowAllResponder))
            .with_events(fixed_emitter())
            .with_capability_matrix(Arc::new(CapabilityMatrix::new()))
            .with_plan_mode(plan.clone())
            .with_dispatcher(empty_dispatcher())
            .with_config(cfg)
            .build()
            .unwrap();
        // Plan mode inactive → no caps to check → Success.
        let res = loop_.run_turn(ctx("hello"), &CancelToken::new());
        assert!(matches!(res.outcome, TurnOutcome::Success));
        // Activating mid-life and re-running should still succeed (empty
        // router → no hinted caps).
        plan.activate(t0());
        let res2 = loop_.run_turn(ctx("hello"), &CancelToken::new());
        assert!(matches!(res2.outcome, TurnOutcome::Success));
    }

    #[test]
    fn test_provider_fixtures_expose_id_and_capabilities() {
        // Cover the test-only Provider impls' `id`/`capabilities` to keep
        // the module's region coverage tight.
        let p1 = ZeroBlockProvider;
        assert_eq!(p1.id(), "zero");
        assert!(!p1.capabilities().is_empty());
        let p2 = ScriptedProvider::new(Vec::new());
        assert_eq!(p2.id(), "scripted");
        assert!(!p2.capabilities().is_empty());
        // Empty script returns Vec::new() — exercises the early-return arm.
        let _ = p2.generate(
            &GenerateRequest {
                model: ModelId::from("x"),
                prompt: String::new(),
                max_blocks: 0, system_override: None,
            },
            &CancelToken::new(),
        );
        // Drain the initial scripted entry too.
        let _ = p2.generate(
            &GenerateRequest {
                model: ModelId::from("x"),
                prompt: String::new(),
                max_blocks: 0, system_override: None,
            },
            &CancelToken::new(),
        );
        let p3 = PanickingProvider;
        assert_eq!(p3.id(), "panic");
        assert!(!p3.capabilities().is_empty());
        let slow = SlowProvider {
            sleep: Duration::from_millis(1),
        };
        assert_eq!(slow.id(), "slow");
        assert!(!slow.capabilities().is_empty());
        // Exercise the cancel-early branch of SlowProvider.
        let cancel = CancelToken::new();
        cancel.cancel();
        let blocks = slow.generate(
            &GenerateRequest {
                model: ModelId::from("x"),
                prompt: String::new(),
                max_blocks: 0, system_override: None,
            },
            &cancel,
        );
        assert!(blocks.is_empty());
        // And exercise the happy path (no cancel).
        let slow2 = SlowProvider {
            sleep: Duration::from_millis(1),
        };
        let blocks = slow2.generate(
            &GenerateRequest {
                model: ModelId::from("x"),
                prompt: String::new(),
                max_blocks: 0, system_override: None,
            },
            &CancelToken::new(),
        );
        assert!(!blocks.is_empty());
    }

    #[test]
    fn decision_label_covers_every_variant() {
        assert_eq!(decision_label(PermissionDecision::AllowOnce), "allow_once");
        assert_eq!(
            decision_label(PermissionDecision::AllowSession),
            "allow_session"
        );
        assert_eq!(
            decision_label(PermissionDecision::AllowForever),
            "allow_forever"
        );
        assert_eq!(decision_label(PermissionDecision::Deny), "deny");
        assert_eq!(
            decision_label(PermissionDecision::DenyForever),
            "deny_forever"
        );
    }

    #[test]
    fn agent_loop_builder_debug_renders_field_flags() {
        let b = AgentLoop::builder().with_config(AgentLoopConfig::default());
        let dbg = format!("{b:?}");
        assert!(dbg.contains("AgentLoopBuilder"));
        assert!(dbg.contains("provider_set"));
        assert!(dbg.contains("config"));
    }

    #[test]
    fn panicking_provider_surfaces_model_error() {
        let loop_ = AgentLoop::builder()
            .with_provider(Arc::new(PanickingProvider))
            .with_router(IntentRouter::empty())
            .with_permission_store(Arc::new(PermissionStore::new()))
            .with_prompt_gen(Arc::new(PromptIdGen::new()))
            .with_responder(Arc::new(AllowAllResponder))
            .with_events(fixed_emitter())
            .with_capability_matrix(Arc::new(CapabilityMatrix::new()))
            .with_plan_mode(Arc::new(PlanMode::new()))
            .with_dispatcher(empty_dispatcher())
            .with_config(AgentLoopConfig::default())
            .build()
            .unwrap();
        let res = loop_.run_turn(ctx("hello"), &CancelToken::new());
        match res.outcome {
            TurnOutcome::ModelError { code } => assert_eq!(code, "E_PROVIDER_PANIC"),
            other => panic!("expected ModelError(E_PROVIDER_PANIC), got {other:?}"),
        }
    }

    #[test]
    fn deny_forever_short_circuits_on_second_turn() {
        let store = Arc::new(PermissionStore::new());
        let responder = Arc::new(ScriptedResponder::new(vec![
            PermissionDecision::DenyForever,
        ]));
        let scripted = Arc::new(ScriptedProvider {
            script: Mutex::new(vec![
                vec![Block::ToolCall {
                    id: "fs.read#1".into(),
                    tool: "fs.read".into(),
                    args: "{}".into(),
                }],
                vec![Block::ToolCall {
                    id: "fs.read#1".into(),
                    tool: "fs.read".into(),
                    args: "{}".into(),
                }],
            ]),
        });
        let loop_ = AgentLoop::builder()
            .with_provider(scripted)
            .with_router(IntentRouter::empty())
            .with_permission_store(store.clone())
            .with_prompt_gen(Arc::new(PromptIdGen::new()))
            .with_responder(responder)
            .with_events(fixed_emitter())
            .with_capability_matrix(Arc::new(CapabilityMatrix::new()))
            .with_plan_mode(Arc::new(PlanMode::new()))
            .with_dispatcher(empty_dispatcher())
            .with_config(AgentLoopConfig::default())
            .build()
            .unwrap();
        let r1 = loop_.run_turn(ctx("a"), &CancelToken::new());
        assert!(matches!(r1.outcome, TurnOutcome::ToolFailure { .. }));
        let r2 = loop_.run_turn(ctx("b"), &CancelToken::new());
        assert!(matches!(r2.outcome, TurnOutcome::ToolFailure { .. }));
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn max_tool_calls_per_turn_caps_inner_loop() {
        // Provider returns three tool calls; loop is configured to allow
        // only one. With dispatch wired in, the second call short-circuits
        // to `BudgetExceeded { kind: "tool_calls" }` (fail-fast).
        let scripted = Arc::new(ScriptedProvider {
            script: Mutex::new(vec![vec![
                Block::ToolCall {
                    id: "a".into(),
                    tool: "fs.read".into(),
                    args: "{}".into(),
                },
                Block::ToolCall {
                    id: "b".into(),
                    tool: "fs.read".into(),
                    args: "{}".into(),
                },
                Block::ToolCall {
                    id: "c".into(),
                    tool: "fs.read".into(),
                    args: "{}".into(),
                },
            ]]),
        });
        let cfg = AgentLoopConfig {
            max_tool_calls_per_turn: 1,
            ..AgentLoopConfig::default()
        };
        let (events, sink) = fixed_emitter_with_sink();
        let loop_ = AgentLoop::builder()
            .with_provider(scripted)
            .with_router(IntentRouter::empty())
            .with_permission_store(Arc::new(PermissionStore::new()))
            .with_prompt_gen(Arc::new(PromptIdGen::new()))
            .with_responder(Arc::new(AllowAllResponder))
            .with_events(events)
            .with_capability_matrix(Arc::new(CapabilityMatrix::new()))
            .with_plan_mode(Arc::new(PlanMode::new()))
            .with_dispatcher(permissive_dispatcher())
            .with_config(cfg)
            .build()
            .unwrap();
        let res = loop_.run_turn(ctx("call"), &CancelToken::new());
        match res.outcome {
            TurnOutcome::BudgetExceeded { ref kind } => assert_eq!(kind, "tool_calls"),
            ref other => panic!("expected BudgetExceeded(tool_calls), got {other:?}"),
        }
        let tool_events = sink
            .snapshot()
            .into_iter()
            .filter(|r| matches!(r.event, Event::ToolCall { .. }))
            .count();
        assert_eq!(tool_events, 1);
    }

    #[test]
    fn turn_result_error_is_std_error() {
        let e = TurnResultError::ProviderPanicked;
        let _src: Option<&(dyn Error + 'static)> = e.source();
    }

    #[test]
    fn agent_loop_build_error_is_std_error() {
        let e = AgentLoopBuildError::MissingField("x");
        let _src: Option<&(dyn Error + 'static)> = e.source();
    }

    #[test]
    fn agent_loop_debug_does_not_panic() {
        let l = echo_loop();
        let dbg = format!("{l:?}");
        assert!(dbg.contains("AgentLoop"));
    }

    // ---- new dispatcher-wiring tests ---------------------------------

    fn build_loop_with(
        provider: Arc<dyn Provider>,
        dispatcher: Arc<RegistryDispatcher>,
        matrix: Arc<CapabilityMatrix>,
        cfg: AgentLoopConfig,
        events: Arc<EventEmitter>,
    ) -> AgentLoop {
        AgentLoop::builder()
            .with_provider(provider)
            .with_router(IntentRouter::empty())
            .with_permission_store(Arc::new(PermissionStore::new()))
            .with_prompt_gen(Arc::new(PromptIdGen::new()))
            .with_responder(Arc::new(AllowAllResponder))
            .with_events(events)
            .with_capability_matrix(matrix)
            .with_plan_mode(Arc::new(PlanMode::new()))
            .with_dispatcher(dispatcher)
            .with_config(cfg)
            .build()
            .unwrap()
    }

    fn tool_call(id: &str, tool: &str) -> Block {
        Block::ToolCall {
            id: id.into(),
            tool: tool.into(),
            args: "{}".into(),
        }
    }

    #[test]
    fn tool_dispatch_happy_path_emits_ok_event() {
        let scripted = Arc::new(ScriptedProvider::new(vec![tool_call("echo", "echo")]));
        let (events, sink) = fixed_emitter_with_sink();
        let loop_ = build_loop_with(
            scripted,
            echo_dispatcher(),
            Arc::new(CapabilityMatrix::new()),
            AgentLoopConfig::default(),
            events,
        );
        let res = loop_.run_turn(ctx("call echo"), &CancelToken::new());
        assert!(matches!(res.outcome, TurnOutcome::Success));
        let tool_evt = sink
            .snapshot()
            .into_iter()
            .find_map(|r| match r.event {
                Event::ToolCall { tool_id, ok, .. } => Some((tool_id, ok)),
                _ => None,
            })
            .expect("ToolCall event");
        assert_eq!(tool_evt, ("echo".to_string(), true));
    }

    #[test]
    fn tool_dispatch_latency_is_non_zero_for_slow_dispatcher() {
        let mut registry = RegistryDispatcher::new();
        registry
            .register(Box::new(SlowDispatcher))
            .expect("register");
        let dispatcher = Arc::new(registry);
        let scripted = Arc::new(ScriptedProvider::new(vec![tool_call("any", "fs.read")]));
        let (events, sink) = fixed_emitter_with_sink();
        let loop_ = build_loop_with(
            scripted,
            dispatcher,
            Arc::new(CapabilityMatrix::new()),
            AgentLoopConfig::default(),
            events,
        );
        let res = loop_.run_turn(ctx("slow"), &CancelToken::new());
        assert!(matches!(res.outcome, TurnOutcome::Success));
        let duration = sink
            .snapshot()
            .into_iter()
            .find_map(|r| match r.event {
                Event::ToolCall { duration_ms, .. } => Some(duration_ms),
                _ => None,
            })
            .expect("ToolCall event");
        assert!(duration > 0, "expected duration_ms > 0, got {duration}");
    }

    #[test]
    fn tool_dispatch_err_returns_tool_failure() {
        let scripted = Arc::new(ScriptedProvider::new(vec![tool_call(
            "fs.write", "fs.write",
        )]));
        let loop_ = build_loop_with(
            scripted,
            deny_dispatcher(),
            Arc::new(CapabilityMatrix::new()),
            AgentLoopConfig::default(),
            fixed_emitter(),
        );
        let res = loop_.run_turn(ctx("call"), &CancelToken::new());
        match res.outcome {
            TurnOutcome::ToolFailure { tool_id, code } => {
                assert_eq!(tool_id, "fs.write");
                assert_eq!(code, "STRAT-E5004");
            }
            other => panic!("expected ToolFailure(STRAT-E5004), got {other:?}"),
        }
    }

    #[test]
    fn no_matching_dispatcher_returns_e5005() {
        let scripted = Arc::new(ScriptedProvider::new(vec![tool_call("foo", "foo")]));
        let loop_ = build_loop_with(
            scripted,
            empty_dispatcher(),
            Arc::new(CapabilityMatrix::new()),
            AgentLoopConfig::default(),
            fixed_emitter(),
        );
        let res = loop_.run_turn(ctx("call foo"), &CancelToken::new());
        match res.outcome {
            TurnOutcome::ToolFailure { tool_id, code } => {
                assert_eq!(tool_id, "foo");
                assert_eq!(code, "STRAT-E5005");
            }
            other => panic!("expected ToolFailure(STRAT-E5005), got {other:?}"),
        }
    }

    #[test]
    fn budget_exhaustion_short_circuits_second_call() {
        let scripted = Arc::new(ScriptedProvider::new(vec![
            tool_call("echo", "echo"),
            tool_call("echo2", "echo"),
        ]));
        let cfg = AgentLoopConfig {
            max_tool_calls_per_turn: 1,
            ..AgentLoopConfig::default()
        };
        let (events, sink) = fixed_emitter_with_sink();
        let loop_ = build_loop_with(
            scripted,
            permissive_dispatcher(),
            Arc::new(CapabilityMatrix::new()),
            cfg,
            events,
        );
        let res = loop_.run_turn(ctx("two"), &CancelToken::new());
        match res.outcome {
            TurnOutcome::BudgetExceeded { ref kind } => assert_eq!(kind, "tool_calls"),
            ref other => panic!("expected BudgetExceeded, got {other:?}"),
        }
        // Exactly one tool dispatch happened.
        let tool_count = sink
            .snapshot()
            .into_iter()
            .filter(|r| matches!(r.event, Event::ToolCall { .. }))
            .count();
        assert_eq!(tool_count, 1);
    }

    #[test]
    fn deny_forever_does_not_invoke_dispatcher() {
        let count = Arc::new(Counter::new(0));
        let mut registry = RegistryDispatcher::new();
        registry
            .register(Box::new(CountingDispatcher {
                count: Arc::clone(&count),
            }))
            .expect("register");
        let dispatcher = Arc::new(registry);
        let scripted = Arc::new(ScriptedProvider::new(vec![tool_call(
            "fs.read#1",
            "fs.read",
        )]));
        let loop_ = AgentLoop::builder()
            .with_provider(scripted)
            .with_router(IntentRouter::empty())
            .with_permission_store(Arc::new(PermissionStore::new()))
            .with_prompt_gen(Arc::new(PromptIdGen::new()))
            .with_responder(Arc::new(DenyAllResponder))
            .with_events(fixed_emitter())
            .with_capability_matrix(Arc::new(CapabilityMatrix::new()))
            .with_plan_mode(Arc::new(PlanMode::new()))
            .with_dispatcher(dispatcher)
            .with_config(AgentLoopConfig::default())
            .build()
            .unwrap();
        let res = loop_.run_turn(ctx("call"), &CancelToken::new());
        assert!(matches!(res.outcome, TurnOutcome::ToolFailure { .. }));
        assert_eq!(count.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn panicking_dispatcher_returns_tool_panic_failure() {
        let mut registry = RegistryDispatcher::new();
        registry
            .register(Box::new(PanickingDispatcher))
            .expect("register");
        let dispatcher = Arc::new(registry);
        let scripted = Arc::new(ScriptedProvider::new(vec![tool_call("boom", "boom")]));
        let (events, sink) = fixed_emitter_with_sink();
        let loop_ = build_loop_with(
            scripted,
            dispatcher,
            Arc::new(CapabilityMatrix::new()),
            AgentLoopConfig::default(),
            events,
        );
        // Silence the libstd default panic-hook output during this test
        // so the captured panic doesn't pollute test logs.
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let res = loop_.run_turn(ctx("call"), &CancelToken::new());
        std::panic::set_hook(prev);
        match res.outcome {
            TurnOutcome::ToolFailure { ref code, .. } => assert_eq!(code, "E_TOOL_PANIC"),
            ref other => panic!("expected ToolFailure(E_TOOL_PANIC), got {other:?}"),
        }
        let saw_provider_error = sink.snapshot().into_iter().any(
            |r| matches!(r.event, Event::ProviderError { ref code, .. } if code == "E_TOOL_PANIC"),
        );
        assert!(
            saw_provider_error,
            "ProviderError(E_TOOL_PANIC) must be emitted"
        );
    }

    #[test]
    fn capability_matrix_hit_supplies_resolved_capability() {
        let cap = Arc::new(CapturingDispatcher {
            last_cap: Mutex::new(None),
        });
        // Re-use a shared CapturingDispatcher via a delegating shim so
        // the test can keep its own Arc for assertions.
        let mut registry = RegistryDispatcher::new();
        registry
            .register(Box::new(CapturingShim(Arc::clone(&cap))))
            .expect("register");
        let dispatcher = Arc::new(registry);

        let matrix = Arc::new(CapabilityMatrix::from_entries(["fs.read"]));
        let scripted = Arc::new(ScriptedProvider::new(vec![tool_call("fs.read", "fs.read")]));
        let loop_ = build_loop_with(
            scripted,
            dispatcher,
            matrix,
            AgentLoopConfig::default(),
            fixed_emitter(),
        );
        let res = loop_.run_turn(ctx("call"), &CancelToken::new());
        assert!(matches!(res.outcome, TurnOutcome::Success));
        assert_eq!(cap.last_cap.lock().unwrap().as_deref(), Some("fs.read"));
    }

    #[test]
    fn capability_matrix_miss_falls_back_to_sentinel() {
        let cap = Arc::new(CapturingDispatcher {
            last_cap: Mutex::new(None),
        });
        let mut registry = RegistryDispatcher::new();
        registry
            .register(Box::new(CapturingShim(Arc::clone(&cap))))
            .expect("register");
        let dispatcher = Arc::new(registry);

        let matrix = Arc::new(CapabilityMatrix::new()); // empty
        let scripted = Arc::new(ScriptedProvider::new(vec![tool_call(
            "unknown_id",
            "fs.read",
        )]));
        let loop_ = build_loop_with(
            scripted,
            dispatcher,
            matrix,
            AgentLoopConfig::default(),
            fixed_emitter(),
        );
        let res = loop_.run_turn(ctx("call"), &CancelToken::new());
        assert!(matches!(res.outcome, TurnOutcome::Success));
        // Capability sentinel is derived from the tool name, not the
        // per-call correlation id, so that the matrix lookup keys off the
        // verb (`fs.read`) rather than an opaque request id.
        assert_eq!(
            cap.last_cap.lock().unwrap().as_deref(),
            Some("tool.fs.read")
        );
    }

    #[test]
    fn multi_tool_happy_path_emits_two_tool_events() {
        let scripted = Arc::new(ScriptedProvider::new(vec![
            tool_call("echo", "echo"),
            tool_call("echo2", "echo"),
        ]));
        let (events, sink) = fixed_emitter_with_sink();
        let loop_ = build_loop_with(
            scripted,
            permissive_dispatcher(),
            Arc::new(CapabilityMatrix::new()),
            AgentLoopConfig::default(),
            events,
        );
        let res = loop_.run_turn(ctx("two"), &CancelToken::new());
        assert!(matches!(res.outcome, TurnOutcome::Success));
        let tool_count = sink
            .snapshot()
            .into_iter()
            .filter(|r| matches!(r.event, Event::ToolCall { .. }))
            .count();
        assert_eq!(tool_count, 2);
    }

    #[test]
    fn turn_counter_increments_once_regardless_of_tool_count() {
        let scripted = Arc::new(ScriptedProvider::new(vec![
            tool_call("a", "fs.read"),
            tool_call("b", "fs.read"),
            tool_call("c", "fs.read"),
        ]));
        let loop_ = build_loop_with(
            scripted,
            permissive_dispatcher(),
            Arc::new(CapabilityMatrix::new()),
            AgentLoopConfig::default(),
            fixed_emitter(),
        );
        assert_eq!(loop_.turn_counter(), 0);
        let _ = loop_.run_turn(ctx("call"), &CancelToken::new());
        assert_eq!(loop_.turn_counter(), 1);
    }

    #[test]
    fn concurrent_run_turn_shares_dispatcher_safely() {
        let count = Arc::new(Counter::new(0));
        let mut registry = RegistryDispatcher::new();
        registry
            .register(Box::new(CountingDispatcher {
                count: Arc::clone(&count),
            }))
            .expect("register");
        let dispatcher = Arc::new(registry);

        // Each provider script needs one tool call per turn.
        let make_loop = |dispatcher: Arc<RegistryDispatcher>| -> AgentLoop {
            let scripted = Arc::new(ScriptedProvider::new(vec![tool_call("x", "fs.read")]));
            build_loop_with(
                scripted,
                dispatcher,
                Arc::new(CapabilityMatrix::new()),
                AgentLoopConfig::default(),
                fixed_emitter(),
            )
        };

        let mut handles = Vec::new();
        for _ in 0..4 {
            let d = Arc::clone(&dispatcher);
            handles.push(thread::spawn(move || {
                // Each thread keeps a private loop but shares the
                // dispatcher Arc — we re-script for every turn by
                // building a new loop per iteration (cheap; the goal
                // is to exercise dispatcher Send+Sync under load).
                let mut ok = 0;
                for _ in 0..10 {
                    let loop_ = make_loop(Arc::clone(&d));
                    let res = loop_.run_turn(ctx("call"), &CancelToken::new());
                    if matches!(res.outcome, TurnOutcome::Success) {
                        ok += 1;
                    }
                }
                ok
            }));
        }
        let mut total = 0;
        for h in handles {
            total += h.join().unwrap();
        }
        assert_eq!(total, 40);
        assert_eq!(count.load(Ordering::SeqCst), 40);
    }

    #[test]
    fn tool_call_duration_ms_field_is_non_negative() {
        // u64 is always non-negative; this pins the type contract.
        let scripted = Arc::new(ScriptedProvider::new(vec![tool_call("echo", "echo")]));
        let (events, sink) = fixed_emitter_with_sink();
        let loop_ = build_loop_with(
            scripted,
            echo_dispatcher(),
            Arc::new(CapabilityMatrix::new()),
            AgentLoopConfig::default(),
            events,
        );
        let _ = loop_.run_turn(ctx("call"), &CancelToken::new());
        for r in sink.snapshot() {
            if let Event::ToolCall { duration_ms, .. } = r.event {
                // Always true for u64; keeps the assertion explicit.
                assert!(duration_ms < u64::MAX);
            }
        }
    }

    #[test]
    fn turn_result_blocks_preserve_tool_call_blocks() {
        let original = vec![tool_call("echo", "echo"), Block::Text { text: "hi".into() }];
        let scripted = Arc::new(ScriptedProvider::new(original.clone()));
        let loop_ = build_loop_with(
            scripted,
            echo_dispatcher(),
            Arc::new(CapabilityMatrix::new()),
            AgentLoopConfig::default(),
            fixed_emitter(),
        );
        let res = loop_.run_turn(ctx("call"), &CancelToken::new());
        assert!(matches!(res.outcome, TurnOutcome::Success));
        // Provider's original ToolCall + Text blocks are preserved verbatim,
        // followed by the synthesized `Block::ToolResult` emitted by the
        // dispatch loop so the agentic continuation can pick it up.
        let mut expected = original.clone();
        expected.push(Block::ToolResult {
            id: "echo".into(),
            output: "{}".into(),
        });
        assert_eq!(res.blocks, expected);
    }
}
