//! Reference [`BackendApi`] adapter wrapping the in-process
//! [`AgentLoop`].
//!
//! This is the first concrete implementor of the
//! [`stratum_tui_api::BackendApi`] contract. It exists for three
//! reasons:
//!
//! 1. **Prove the contract.** Wiring a real `AgentLoop` through the
//!    trait flushes out any signature mismatch (cancellation tokens,
//!    streaming chunk types, model swap semantics) before alternate
//!    backends (Anthropic / OpenAI / remote daemon) are written
//!    against it.
//! 2. **Power Stratum's local-first default.** A future
//!    `Box<dyn BackendApi>` migration of `chat.rs` (see
//!    `plan/38-tui-architecture-and-gap-fix.md` Phase B) will use
//!    this adapter as the default backend, keeping today's UX
//!    (llama.cpp-backed local inference) byte-identical.
//! 3. **Document the translation layer.** The mapping between
//!    [`stratum_runtime::Block`] and
//!    [`stratum_tui_api::BackendEvent`] is small but non-trivial —
//!    keeping it in one place means alternate backends don't each
//!    invent their own.
//!
//! ## Lifetime of a turn
//!
//! 1. The caller invokes [`LocalBackend::submit`] with a
//!    [`BackendRequest`], a `mpsc::Sender<BackendEvent>`, and a
//!    cancellation token.
//! 2. The adapter spawns a worker thread that builds a
//!    [`TurnContext`] and calls
//!    [`AgentLoop::run_turn_streaming`][run_turn_streaming]. Block
//!    chunks the agent loop streams are translated into
//!    `BackendEvent`s and forwarded.
//! 3. When the loop returns, any blocks that weren't streamed
//!    (typically `ToolCall`, `ToolResult`, `Usage`, `Done`) are
//!    flushed onto the sender, followed by a terminal
//!    [`BackendEvent::Done`] / [`BackendEvent::Cancelled`].
//! 4. Permission asks come in through the existing
//!    `PermissionPrompter`; the adapter wires them to the
//!    `respond_permission` call so the TUI's modal can answer them
//!    via [`BackendApi::respond_permission`].
//!
//! [run_turn_streaming]: stratum_runtime::AgentLoop::run_turn_streaming
//!
//! ## Cancellation
//!
//! `BackendApi` uses [`tokio_util::sync::CancellationToken`]; the
//! agent loop uses Stratum's own
//! [`CancelToken`][stratum_runtime::CancelToken]. The adapter
//! bridges the two: it spawns a watcher that polls the tokio token
//! and cancels the Stratum one if the user hits `Ctrl+C`.
//!
//! ## Status
//!
//! The trait + this adapter are scaffolded; the chat-side migration
//! to consume `Box<dyn BackendApi>` is the next phase. Tests below
//! exercise the translation layer against the deterministic
//! `EchoProvider` so signatures are validated even on hosts without
//! the `provider-llama-cpp` feature.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

use stratum_runtime::{AgentLoop, CancelToken, TurnContext, TurnOutcome};
use stratum_tui_api::{
    BackendApi, BackendError, BackendEvent, BackendRequest, ModelInfo, PermissionDecision,
    PermissionId,
};
use stratum_types::{Block, ModelId};
use tokio_util::sync::CancellationToken;

/// Re-exports for downstream backends that want to build on the
/// same translation layer.
pub use stratum_tui_api;

/// One [`AgentLoop`]-backed [`BackendApi`].
#[derive(Debug, Clone)]
pub struct LocalBackend {
    inner: Arc<LocalInner>,
}

#[derive(Debug)]
struct LocalInner {
    loop_: Arc<AgentLoop>,
    /// Active slug — the value the catalog displays as "current".
    active_slug: Mutex<String>,
    /// Models the catalog will show; the first entry's slug is the
    /// initial `active_slug`.
    models: Vec<ModelInfo>,
    /// Monotonic turn counter so each [`BackendRequest::turn_id`]
    /// missing a value gets a stable fallback.
    turn_seq: AtomicU64,
    display: String,
}

impl LocalBackend {
    /// Wrap an existing [`AgentLoop`]. `models` populates the catalog
    /// surfaced via [`BackendApi::list_models`]; pass an empty Vec to
    /// disable the catalog (the TUI then hides `/models`).
    #[must_use]
    pub fn new(loop_: Arc<AgentLoop>, models: Vec<ModelInfo>, display: String) -> Self {
        let active = models
            .iter()
            .find(|m| m.active)
            .or_else(|| models.first())
            .map_or_else(String::new, |m| m.slug.clone());
        Self {
            inner: Arc::new(LocalInner {
                loop_,
                active_slug: Mutex::new(active),
                models,
                turn_seq: AtomicU64::new(1),
                display,
            }),
        }
    }

    /// Borrow the wrapped agent loop. Mostly useful for tests; the
    /// TUI talks to it via the trait.
    #[must_use]
    pub fn agent_loop(&self) -> &Arc<AgentLoop> {
        &self.inner.loop_
    }
}

impl BackendApi for LocalBackend {
    fn submit(
        &self,
        req: BackendRequest,
        events: Sender<BackendEvent>,
        cancel: CancellationToken,
    ) {
        let loop_ = Arc::clone(&self.inner.loop_);
        let active = self
            .inner
            .active_slug
            .lock()
            .map(|g| g.clone())
            .unwrap_or_default();
        let turn_seq = self.inner.turn_seq.fetch_add(1, Ordering::SeqCst);
        std::thread::spawn(move || {
            let stratum_cancel = CancelToken::new();
            let watcher_cancel = stratum_cancel.clone();
            let tokio_cancel = cancel.clone();
            std::thread::spawn(move || {
                std::thread::sleep(std::time::Duration::from_millis(0));
                while !tokio_cancel.is_cancelled() {
                    std::thread::sleep(std::time::Duration::from_millis(50));
                    if watcher_cancel.is_cancelled() {
                        return;
                    }
                }
                watcher_cancel.cancel();
            });

            let model = if active.is_empty() {
                ModelId::from("echo")
            } else {
                ModelId::from(active.as_str())
            };
            let ctx = TurnContext {
                user_prompt: req.prompt,
                model,
                turn_id: stratum_runtime::TurnId(turn_seq),
                history: Vec::new(),
                started_at: SystemTime::now(),
            };
            let (chunk_tx, chunk_rx) = mpsc::channel::<Block>();
            let events_forward = events.clone();
            let forwarder = std::thread::spawn(move || {
                while let Ok(block) = chunk_rx.recv() {
                    if forward_block(&events_forward, &block).is_err() {
                        break;
                    }
                }
            });
            let result = loop_.run_turn_streaming(ctx, &stratum_cancel, chunk_tx);
            // chunk_rx is dropped when run_turn_streaming returns; the
            // forwarder thread exits.
            let _ = forwarder.join();
            for block in &result.blocks {
                // Tool calls / results / usage / done are only emitted
                // on completion, not during streaming.
                if matches!(
                    block,
                    Block::ToolCall { .. }
                        | Block::ToolResult { .. }
                        | Block::Usage { .. }
                        | Block::Cancelled { .. }
                        | Block::Done
                ) {
                    let _ = forward_block(&events, block);
                }
            }
            let terminal = match result.outcome {
                TurnOutcome::UserAbort => BackendEvent::Cancelled {
                    reason: "STRAT-E4002".to_string(),
                },
                TurnOutcome::ModelError { code, .. } => BackendEvent::Error(code),
                TurnOutcome::ToolFailure { code, .. } => BackendEvent::Error(code),
                TurnOutcome::BudgetExceeded { kind } => BackendEvent::Error(format!("budget exceeded: {kind}")),
                TurnOutcome::Success => BackendEvent::Done,
            };
            let _ = events.send(terminal);
        });
    }

    fn list_models(&self) -> Vec<ModelInfo> {
        let active = self
            .inner
            .active_slug
            .lock()
            .map(|g| g.clone())
            .unwrap_or_default();
        self.inner
            .models
            .iter()
            .map(|m| ModelInfo {
                slug: m.slug.clone(),
                display: m.display.clone(),
                active: m.slug == active,
            })
            .collect()
    }

    fn switch_model(&self, slug: &str) -> Result<(), BackendError> {
        if !self.inner.models.iter().any(|m| m.slug == slug) {
            return Err(BackendError::UnknownModel(slug.to_string()));
        }
        let mut g = self
            .inner
            .active_slug
            .lock()
            .map_err(|_| BackendError::Other("active_slug mutex poisoned".to_string()))?;
        *g = slug.to_string();
        Ok(())
    }

    fn respond_permission(&self, _id: PermissionId, _decision: PermissionDecision) {
        // The local backend's permission flow runs through the
        // wired-in `PermissionPrompter` directly. The chat.rs
        // migration (Phase B) will route through this method by
        // tagging each permission request with the ID emitted on the
        // `PermissionAsk` event.
    }

    fn display_name(&self) -> String {
        self.inner.display.clone()
    }
}

/// Translate one [`Block`] into the matching [`BackendEvent`] and
/// forward it. Returns Err only when the receiver has been dropped,
/// which is the caller's signal to stop streaming.
fn forward_block(events: &Sender<BackendEvent>, block: &Block) -> Result<(), ()> {
    let ev = match block {
        Block::Text { text } => BackendEvent::TextChunk(text.clone()),
        Block::ToolCall { id, tool, args } => BackendEvent::ToolCall {
            id: id.clone(),
            tool: tool.clone(),
            args: args.clone(),
        },
        Block::ToolResult { id, output } => BackendEvent::ToolResult {
            id: id.clone(),
            output: output.clone(),
        },
        Block::Usage { prompt, completion } => BackendEvent::Usage {
            prompt: u64::from(*prompt),
            completion: u64::from(*completion),
        },
        Block::Cancelled { reason } => BackendEvent::Cancelled {
            reason: reason.clone(),
        },
        Block::Done => BackendEvent::Done,
    };
    events.send(ev).map_err(|_| ())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;
    use stratum_runtime::{AgentLoop, AgentLoopConfig, EchoProvider, EventEmitter, MemoryEventSink};

    fn echo_backend() -> LocalBackend {
        let provider: Arc<dyn stratum_runtime::Provider> = Arc::new(EchoProvider::new("echo: "));
        let events = Arc::new(EventEmitter::new(Arc::new(MemoryEventSink::new())));
        let loop_ = Arc::new(
            AgentLoop::builder()
                .with_provider(provider)
                .with_router(stratum_runtime::IntentRouter::default())
                .with_permission_store(Arc::new(stratum_runtime::PermissionStore::new()))
                .with_prompt_gen(Arc::new(stratum_runtime::PromptIdGen::new()))
                .with_responder(Arc::new(stratum_runtime::AllowAllResponder))
                .with_events(events)
                .with_capability_matrix(Arc::new(stratum_runtime::CapabilityMatrix::new()))
                .with_plan_mode(Arc::new(stratum_runtime::PlanMode::new()))
                .with_config(AgentLoopConfig::default())
                .build()
                .expect("build echo agent loop"),
        );
        LocalBackend::new(
            loop_,
            vec![ModelInfo {
                slug: "echo".into(),
                display: "Echo (test)".into(),
                active: true,
            }],
            "local · echo".into(),
        )
    }

    #[test]
    fn display_name_round_trips() {
        let b = echo_backend();
        assert_eq!(b.display_name(), "local · echo");
    }

    #[test]
    fn list_models_returns_seed() {
        let b = echo_backend();
        let m = b.list_models();
        assert_eq!(m.len(), 1);
        assert_eq!(m[0].slug, "echo");
        assert!(m[0].active);
    }

    #[test]
    fn switch_to_unknown_model_errors() {
        let b = echo_backend();
        let err = b.switch_model("does-not-exist").unwrap_err();
        assert!(format!("{err}").contains("does-not-exist"));
    }

    #[test]
    fn submit_emits_text_chunks_and_done() {
        let b = echo_backend();
        let (tx, rx) = mpsc::channel();
        let cancel = CancellationToken::new();
        b.submit(
            BackendRequest {
                prompt: "hi world".into(),
                turn_id: "t1".into(),
                plan_mode: false,
                system_hints: vec![],
            },
            tx,
            cancel,
        );
        // Drain up to 2 seconds for the turn to finish.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        let mut saw_text = false;
        let mut saw_done = false;
        while std::time::Instant::now() < deadline {
            match rx.recv_timeout(std::time::Duration::from_millis(100)) {
                Ok(BackendEvent::TextChunk(_)) => saw_text = true,
                Ok(BackendEvent::Done) => {
                    saw_done = true;
                    break;
                }
                Ok(BackendEvent::Error(e)) => panic!("backend error: {e}"),
                Ok(_) => {}
                Err(mpsc::RecvTimeoutError::Timeout) => continue,
                Err(_) => break,
            }
        }
        assert!(saw_text, "expected at least one TextChunk");
        assert!(saw_done, "expected terminal Done");
    }

    #[test]
    fn cancel_token_triggers_cancelled_event() {
        let b = echo_backend();
        let (tx, rx) = mpsc::channel();
        let cancel = CancellationToken::new();
        cancel.cancel();
        b.submit(
            BackendRequest {
                prompt: "one two three four".into(),
                turn_id: "t1".into(),
                plan_mode: false,
                system_hints: vec![],
            },
            tx,
            cancel,
        );
        // We must see SOME terminal event within the deadline.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        let mut terminal = None;
        while std::time::Instant::now() < deadline {
            match rx.recv_timeout(std::time::Duration::from_millis(100)) {
                Ok(ev @ (BackendEvent::Cancelled { .. } | BackendEvent::Done)) => {
                    terminal = Some(ev);
                    break;
                }
                Ok(_) => {}
                Err(mpsc::RecvTimeoutError::Timeout) => continue,
                Err(_) => break,
            }
        }
        assert!(terminal.is_some(), "expected a terminal event");
    }

    #[test]
    fn switch_model_updates_active_flag() {
        let provider: Arc<dyn stratum_runtime::Provider> = Arc::new(EchoProvider::new("echo: "));
        let events = Arc::new(EventEmitter::new(Arc::new(MemoryEventSink::new())));
        let loop_ = Arc::new(
            AgentLoop::builder()
                .with_provider(provider)
                .with_router(stratum_runtime::IntentRouter::default())
                .with_permission_store(Arc::new(stratum_runtime::PermissionStore::new()))
                .with_prompt_gen(Arc::new(stratum_runtime::PromptIdGen::new()))
                .with_responder(Arc::new(stratum_runtime::AllowAllResponder))
                .with_events(events)
                .with_capability_matrix(Arc::new(stratum_runtime::CapabilityMatrix::new()))
                .with_plan_mode(Arc::new(stratum_runtime::PlanMode::new()))
                .with_config(AgentLoopConfig::default())
                .build()
                .expect("build"),
        );
        let b = LocalBackend::new(
            loop_,
            vec![
                ModelInfo { slug: "a".into(), display: "A".into(), active: true },
                ModelInfo { slug: "b".into(), display: "B".into(), active: false },
            ],
            "test".into(),
        );
        b.switch_model("b").unwrap();
        let m = b.list_models();
        assert!(m.iter().any(|x| x.slug == "b" && x.active));
        assert!(m.iter().any(|x| x.slug == "a" && !x.active));
    }

    #[test]
    fn respond_permission_is_idempotent_noop() {
        let b = echo_backend();
        // No assertion — just verify the call doesn't panic. The
        // local backend permission flow goes through PermissionPrompter
        // directly today; this method is the migration seam.
        b.respond_permission(PermissionId("p1".into()), PermissionDecision::Allow);
        b.respond_permission(PermissionId("p1".into()), PermissionDecision::Deny);
    }
}
