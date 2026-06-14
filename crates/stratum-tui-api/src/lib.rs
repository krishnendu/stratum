//! Backend-agnostic contract the Stratum TUI calls into.
//!
//! The chat renderer holds a `Box<dyn BackendApi>` and never names a
//! concrete agent loop. That keeps the TUI portable across:
//!
//! * **Local** — the in-process [`stratum_runtime::AgentLoop`] backed
//!   by `LlamaCppProvider`. The default Stratum binary uses this.
//! * **Daemon** — a long-running local agent process the TUI speaks
//!   to over a unix socket (planned: Phase 8 mobile path so the
//!   phone UI can drive a desktop agent).
//! * **Hosted** — a remote provider (Anthropic, `OpenAI`, `LiteLLM`)
//!   accessed over HTTP. Each is a thin [`BackendApi`] impl in its
//!   own crate (`stratum-backend-anthropic`, …) so users opt in via
//!   feature flag or runtime config without forcing the dependency
//!   on everyone else.
//!
//! ## Lifecycle
//!
//! 1. The TUI constructs a [`BackendRequest`] from the typed prompt
//!    and any pending workspace context.
//! 2. It calls [`BackendApi::submit`] with a [`Sender<BackendEvent>`]
//!    and a cancellation token. The backend takes ownership of both
//!    and runs the turn off the render thread.
//! 3. The TUI drains events as they arrive. Each [`BackendEvent`]
//!    maps 1:1 to a transcript block, a streaming text update, a
//!    permission prompt, or a terminal marker (cancelled / done /
//!    error).
//! 4. On [`BackendEvent::PermissionAsk`], the TUI renders the modal
//!    and calls [`BackendApi::respond_permission`] with the user's
//!    decision.
//! 5. The turn ends when a terminal event arrives.
//!
//! ## Why a trait, not just a function pointer
//!
//! Backends carry stateful resources (HTTP clients, sockets, model
//! caches) that need to survive across turns. A trait object owns
//! that state inside the binary and exposes a uniform call signature.
//! It also lets us model two cheap, very different operations —
//! [`BackendApi::list_models`] and [`BackendApi::switch_model`] —
//! without cluttering the per-turn fast path.
//!
//! ## Status
//!
//! This crate ships the contract only. The first concrete impl
//! (`stratum-backend-local`) and the chat-side migration land in
//! follow-up phases tracked in `plan/38-tui-architecture-and-gap-fix.md`.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use std::sync::mpsc::Sender;

use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

/// One model the user can select via `/models` / `/switch`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Stable identifier the user types into `/switch <slug>`.
    pub slug: String,
    /// Human-friendly label shown in the catalog.
    pub display: String,
    /// True if this is the currently-active model.
    pub active: bool,
}

/// What the user typed plus any context the backend needs to start a turn.
///
/// Backends receive this once per [`BackendApi::submit`] call. Free-form
/// fields stay in [`BackendRequest::system_hints`] so adding new metadata
/// doesn't require a contract bump.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendRequest {
    /// What the user typed for this turn.
    pub prompt: String,
    /// Stable turn ID assigned by the TUI. Backends echo it back on
    /// every event so racy turns can be matched without ambiguity.
    pub turn_id: String,
    /// Plan-mode flag — backends that support plan-vs-exec separation
    /// (everything backed by `AgentLoop`) honor this; others may
    /// ignore it.
    pub plan_mode: bool,
    /// Free-form key/value pairs the TUI passes through to the
    /// backend. Used today for the workspace path and the resumed
    /// session ID; future fields land here without breaking older
    /// backends.
    pub system_hints: Vec<(String, String)>,
}

/// One event flowing back from the backend to the TUI. Maps onto
/// the existing `chat::Block` types so the TUI can fold each event
/// into the transcript without a translation step.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BackendEvent {
    /// One chunk of streaming assistant text. May be a single
    /// character or a whole paragraph — backends choose. The TUI
    /// concatenates chunks until a terminal event arrives.
    TextChunk(String),
    /// Structured tool invocation. Mirrors the JSON the local
    /// `AgentLoop` already emits.
    ToolCall {
        /// Per-call ID so [`BackendEvent::ToolResult`] can be matched.
        id: String,
        /// Tool name (e.g. `fs.read`).
        tool: String,
        /// JSON-encoded args.
        args: String,
    },
    /// Result of a tool call.
    ToolResult {
        /// Matches the [`BackendEvent::ToolCall`] `id`.
        id: String,
        /// Tool output, already trimmed for display.
        output: String,
    },
    /// Backend wants the user to authorise something. The TUI
    /// renders a modal and replies via
    /// [`BackendApi::respond_permission`].
    PermissionAsk {
        /// Permission request ID to echo back on the answer.
        id: PermissionId,
        /// Human-readable summary the modal renders.
        summary: String,
        /// Tool being authorised (e.g. `fs.write`).
        tool: String,
    },
    /// Token usage for this turn. Often emitted once at the end.
    Usage {
        /// Prompt tokens billed.
        prompt: u64,
        /// Completion tokens billed.
        completion: u64,
    },
    /// Terminal: the user cancelled mid-stream.
    Cancelled {
        /// Backend-specific reason code (`STRAT-E4002`, …).
        reason: String,
    },
    /// Terminal: turn completed successfully.
    Done,
    /// Terminal: backend hit an unrecoverable error.
    Error(String),
}

/// Opaque ID for a [`BackendEvent::PermissionAsk`].
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PermissionId(pub String);

/// The TUI's answer to a [`BackendEvent::PermissionAsk`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PermissionDecision {
    /// Run this exact tool call once.
    Allow,
    /// Run this tool call AND remember the decision for this tool
    /// for the rest of the session.
    AllowAlways,
    /// Reject this tool call.
    Deny,
}

/// Stable error type backends return when a sync call fails.
#[derive(Debug, thiserror::Error)]
pub enum BackendError {
    /// Tried to switch to a slug the backend does not recognise.
    #[error("unknown model slug: {0}")]
    UnknownModel(String),
    /// Backend was asked to do something it does not support.
    /// Used by hosted backends when a feature (e.g. plan mode) only
    /// the local agent has gets requested.
    #[error("unsupported operation: {0}")]
    Unsupported(String),
    /// Catch-all for transport / configuration failures.
    #[error("backend error: {0}")]
    Other(String),
}

/// The contract every backend implements.
pub trait BackendApi: Send + Sync {
    /// Kick off a turn. The backend takes ownership of `events` and
    /// `cancel`; it MUST emit a terminal event (`Done`, `Cancelled`,
    /// or `Error`) before dropping the sender.
    ///
    /// Implementations spawn whatever worker fits — a thread, a
    /// tokio task, a remote call — and return immediately. The TUI
    /// stays responsive.
    fn submit(&self, req: BackendRequest, events: Sender<BackendEvent>, cancel: CancellationToken);

    /// Models the user can pick from. Cheap; called when the user
    /// opens `/models` or `/switch`.
    fn list_models(&self) -> Vec<ModelInfo>;

    /// Switch the active model. Returns `Ok` on success; backends
    /// that can't fulfil the request return [`BackendError::UnknownModel`]
    /// or [`BackendError::Unsupported`].
    ///
    /// # Errors
    /// Returns [`BackendError::UnknownModel`] for an unknown slug or
    /// [`BackendError::Unsupported`] when the backend pins one model
    /// (typical for hosted single-model deployments).
    fn switch_model(&self, slug: &str) -> Result<(), BackendError>;

    /// Reply to a [`BackendEvent::PermissionAsk`]. Idempotent — a
    /// duplicate answer for the same ID is silently dropped.
    fn respond_permission(&self, id: PermissionId, decision: PermissionDecision);

    /// Human-friendly name for the status bar (e.g. "local · gemma-4-e4b"
    /// or "anthropic · claude-opus-4-7").
    fn display_name(&self) -> String;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn permission_decision_round_trips_via_serde() {
        let raw = serde_json::to_string(&PermissionDecision::AllowAlways).unwrap();
        let back: PermissionDecision = serde_json::from_str(&raw).unwrap();
        assert_eq!(back, PermissionDecision::AllowAlways);
    }

    #[test]
    fn backend_event_text_chunk_round_trips() {
        let ev = BackendEvent::TextChunk("hi".into());
        let raw = serde_json::to_string(&ev).unwrap();
        let back: BackendEvent = serde_json::from_str(&raw).unwrap();
        assert_eq!(back, ev);
    }

    #[test]
    fn backend_event_tool_call_round_trips() {
        let ev = BackendEvent::ToolCall {
            id: "1".into(),
            tool: "fs.read".into(),
            args: "{}".into(),
        };
        let raw = serde_json::to_string(&ev).unwrap();
        let back: BackendEvent = serde_json::from_str(&raw).unwrap();
        assert_eq!(back, ev);
    }

    #[test]
    fn backend_event_permission_ask_carries_id() {
        let ev = BackendEvent::PermissionAsk {
            id: PermissionId("p1".into()),
            summary: "fs.write to README.md".into(),
            tool: "fs.write".into(),
        };
        let raw = serde_json::to_string(&ev).unwrap();
        let back: BackendEvent = serde_json::from_str(&raw).unwrap();
        assert_eq!(back, ev);
    }

    #[test]
    fn backend_event_done_round_trips() {
        let raw = serde_json::to_string(&BackendEvent::Done).unwrap();
        let back: BackendEvent = serde_json::from_str(&raw).unwrap();
        assert_eq!(back, BackendEvent::Done);
    }

    #[test]
    fn backend_event_cancelled_round_trips() {
        let ev = BackendEvent::Cancelled {
            reason: "STRAT-E4002".into(),
        };
        let raw = serde_json::to_string(&ev).unwrap();
        let back: BackendEvent = serde_json::from_str(&raw).unwrap();
        assert_eq!(back, ev);
    }

    #[test]
    fn backend_event_usage_round_trips() {
        let ev = BackendEvent::Usage {
            prompt: 12,
            completion: 5,
        };
        let raw = serde_json::to_string(&ev).unwrap();
        let back: BackendEvent = serde_json::from_str(&raw).unwrap();
        assert_eq!(back, ev);
    }

    #[test]
    fn backend_error_display_unknown_model() {
        let err = BackendError::UnknownModel("gpt-9".into());
        assert!(format!("{err}").contains("gpt-9"));
    }

    #[test]
    fn backend_error_display_unsupported() {
        let err = BackendError::Unsupported("plan-mode".into());
        assert!(format!("{err}").contains("unsupported"));
    }

    #[test]
    fn backend_error_display_other() {
        let err = BackendError::Other("network".into());
        assert!(format!("{err}").contains("network"));
    }

    #[test]
    fn backend_request_serialises() {
        let req = BackendRequest {
            prompt: "hi".into(),
            turn_id: "t1".into(),
            plan_mode: true,
            system_hints: vec![("workspace".into(), "/tmp".into())],
        };
        let raw = serde_json::to_string(&req).unwrap();
        let back: BackendRequest = serde_json::from_str(&raw).unwrap();
        assert_eq!(back, req);
    }

    #[test]
    fn model_info_serialises() {
        let m = ModelInfo {
            slug: "gemma-4-e4b".into(),
            display: "Gemma 4 E4B".into(),
            active: true,
        };
        let raw = serde_json::to_string(&m).unwrap();
        let back: ModelInfo = serde_json::from_str(&raw).unwrap();
        assert_eq!(back, m);
    }
}
