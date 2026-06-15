//! OpenAI-compatible Chat Completions wire-protocol shapes + HTTP server.
//!
//! Phase 6 — `stratum serve --openai` exposes a `POST /v1/chat/completions`
//! endpoint that bridges OpenAI Chat Completions requests through the
//! internal [`crate::agent_loop::AgentLoop`]. The HTTP layer is
//! synchronous (no async / tokio runtime in the request path), per the
//! rest of the daemon. Streaming responses are emitted as SSE.
//!
//! See `plan/16-multi-llm-providers.md` (OpenAI-shaped egress) and
//! `plan/33-mcp-and-external-tools.md` (server-mode contract).
//!
//! ## What's here
//!
//! * [`OpenAIChatRequest`] / [`OpenAIChatResponse`] / [`OpenAIStreamChunk`] —
//!   the JSON wire shapes. Match the OpenAI Chat Completions API
//!   (2024 surface) for the fields a Stratum client actually uses.
//! * [`OpenAIServer`] — a `tiny_http`-backed loopback HTTP server with
//!   `POST /v1/chat/completions`, `POST /v1/models`, and CORS preflight
//!   (`OPTIONS *`). Bridges to an [`crate::agent_loop::AgentLoop`].
//! * Conversion impls so the HTTP layer can stay thin: request →
//!   [`crate::agent_loop::TurnContext`] and [`crate::agent_loop::TurnResult`]
//!   → [`OpenAIChatResponse`].
//!
//! ## Why a separate sentinel range
//!
//! HTTP error responses go out as proper HTTP status codes (4xx / 5xx)
//! plus an OpenAI-style `{"error":{"type":..,"message":..}}` body — no
//! `STRAT-E####` literals reach the wire. Internal failures are mapped
//! to `500` with `type: "internal_error"`.

// xtask-check-error-codes: ignore-file
//
// Reason: this module bridges OpenAI's HTTP-status-coded error surface,
// not the catalog `STRAT-E####` sentinels. The HTTP responses use
// OpenAI-shaped `{"error":{"type":..}}` bodies. No `STRAT-E####`
// literals appear in this file.

#![allow(
    clippy::doc_markdown,
    clippy::missing_errors_doc,
    clippy::similar_names,
    clippy::needless_pass_by_value,
    clippy::derive_partial_eq_without_eq,
    clippy::same_functions_in_if_condition,
    clippy::match_same_arms,
    clippy::too_many_lines,
    clippy::option_if_let_else
)]

use std::io::Cursor;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::thread::{self, JoinHandle};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use stratum_types::{Block, ModelId};
use tiny_http::{Header, Method, Response, Server, StatusCode};

use crate::agent_factory::AgentFactory;
use crate::agent_loop::{AgentLoop, TurnContext, TurnResult};
use crate::cancel::CancelToken;
use crate::conversation::TurnOutcome;
use crate::model_catalog::ModelCatalog;
use crate::observability::TurnId;
use crate::provider::ChatHistoryTurn;

// ---------------------------------------------------------------------------
// Wire-protocol shapes
// ---------------------------------------------------------------------------

/// One chat message in an [`OpenAIChatRequest`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenAIChatMessage {
    /// `"system"`, `"user"`, `"assistant"`, or `"tool"`.
    pub role: String,
    /// Message body. Free-form text in this minimal surface.
    pub content: String,
    /// Optional explicit name (e.g. for `"role":"tool"` messages).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// OpenAI Chat Completions request body (subset used by the daemon).
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct OpenAIChatRequest {
    /// Model identifier — passed straight through to the
    /// [`crate::provider::Provider`].
    pub model: String,
    /// Ordered conversation turns. The trailing user message becomes
    /// the prompt; earlier messages become [`ChatHistoryTurn`]s.
    pub messages: Vec<OpenAIChatMessage>,
    /// When `true`, the daemon responds with `text/event-stream`
    /// (SSE) instead of a single JSON response.
    #[serde(default)]
    pub stream: bool,
    /// Optional cap on output tokens. The runtime budget is enforced
    /// by [`crate::agent_loop::AgentLoopConfig`]; this field is
    /// accepted for client compatibility and surfaced on the
    /// returned [`OpenAIUsage`] but not enforced beyond what the
    /// provider does.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Softmax temperature. Forwarded as a sampler hint when set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Nucleus-sampling probability mass.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
}

/// OpenAI Chat Completions response — non-stream path.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpenAIChatResponse {
    /// Server-assigned id; `chatcmpl-<turn-id>` in this implementation.
    pub id: String,
    /// Always `"chat.completion"` for the non-stream path.
    pub object: String,
    /// Unix timestamp seconds.
    pub created: u64,
    /// Echoed model identifier.
    pub model: String,
    /// Choices — exactly one element today.
    pub choices: Vec<OpenAIChoice>,
    /// Token usage; placeholder counts when the provider doesn't
    /// emit a [`stratum_types::Block::Usage`] block.
    pub usage: OpenAIUsage,
}

/// One choice in an [`OpenAIChatResponse`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpenAIChoice {
    /// Always `0` in the single-choice path.
    pub index: u32,
    /// Assistant message.
    pub message: OpenAIChatMessage,
    /// `"stop"`, `"length"`, `"tool_calls"`, or `"error"`.
    pub finish_reason: String,
}

/// Token-usage block on a Chat Completions response.
#[allow(clippy::struct_field_names, reason = "OpenAI wire-format field names")]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenAIUsage {
    /// Prompt tokens.
    pub prompt_tokens: u32,
    /// Completion tokens.
    pub completion_tokens: u32,
    /// Sum.
    pub total_tokens: u32,
}

/// One SSE chunk on the streaming path. Serialised verbatim into a
/// `data: <json>\n\n` SSE event.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpenAIStreamChunk {
    /// Same chunked id used across all chunks of one response.
    pub id: String,
    /// Always `"chat.completion.chunk"`.
    pub object: String,
    /// Unix timestamp seconds.
    pub created: u64,
    /// Echoed model identifier.
    pub model: String,
    /// Choices — exactly one element today.
    pub choices: Vec<OpenAIStreamChoice>,
}

/// One choice in an [`OpenAIStreamChunk`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpenAIStreamChoice {
    /// Always `0` in the single-choice path.
    pub index: u32,
    /// Delta token / content; `None` on the terminal chunk.
    pub delta: OpenAIDelta,
    /// `None` for incremental chunks, `Some(..)` on the terminal one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

/// Per-chunk delta in an [`OpenAIStreamChoice`].
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenAIDelta {
    /// `"assistant"` on the first chunk; absent thereafter.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    /// Streamed text delta.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

// ---------------------------------------------------------------------------
// Model-list shapes
// ---------------------------------------------------------------------------

/// Response body for `POST /v1/models` (OpenAI shape).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenAIModelList {
    /// Always `"list"`.
    pub object: String,
    /// One [`OpenAIModelEntry`] per catalog entry.
    pub data: Vec<OpenAIModelEntry>,
}

/// One catalog entry rendered as an OpenAI model row.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenAIModelEntry {
    /// Catalog slug.
    pub id: String,
    /// Always `"model"`.
    pub object: String,
    /// Unix timestamp seconds (current time).
    pub created: u64,
    /// Owner / vendor tag.
    pub owned_by: String,
}

impl OpenAIModelList {
    /// Build a model list from a [`ModelCatalog`].
    #[must_use]
    pub fn from_catalog(catalog: &ModelCatalog) -> Self {
        let now = unix_now_secs();
        let data = catalog
            .entries
            .keys()
            .map(|slug| OpenAIModelEntry {
                id: slug.as_str().to_string(),
                object: "model".to_string(),
                created: now,
                owned_by: "stratum".to_string(),
            })
            .collect();
        Self {
            object: "list".to_string(),
            data,
        }
    }
}

// ---------------------------------------------------------------------------
// Conversions
// ---------------------------------------------------------------------------

impl From<OpenAIChatRequest> for TurnContext {
    fn from(req: OpenAIChatRequest) -> Self {
        let model = ModelId::from(req.model.clone());
        // The trailing user message becomes the prompt; everything
        // before it (regardless of role) becomes the `history` vector.
        let mut history = Vec::with_capacity(req.messages.len().saturating_sub(1));
        let mut user_prompt = String::new();
        let last_user_idx = req
            .messages
            .iter()
            .rposition(|m| m.role == "user")
            .unwrap_or_else(|| req.messages.len().saturating_sub(1));
        for (i, msg) in req.messages.into_iter().enumerate() {
            if i == last_user_idx {
                user_prompt = msg.content;
                continue;
            }
            let role = match msg.role.as_str() {
                "user" => "user",
                "assistant" => "assistant",
                "system" => "user",
                _ => continue,
            };
            history.push(ChatHistoryTurn {
                role: role.to_string(),
                content: msg.content,
            });
        }
        Self {
            user_prompt,
            model,
            turn_id: TurnId(0),
            started_at: SystemTime::now(),
            history,
            // OpenAI's wire shape doesn't model attachments yet; the
            // multimodal scaffold's TurnContext.attachments field is
            // populated by the chat surface, not by this HTTP path.
            // BackendApi attachments wire-protocol lands in Phase B.
            attachments: Vec::new(),
        }
    }
}

impl From<TurnResult> for OpenAIChatResponse {
    fn from(result: TurnResult) -> Self {
        let id = format!("chatcmpl-{}", result.turn_id.0);
        let created = unix_now_secs();
        // Aggregate Text blocks into the final assistant content.
        let content = blocks_to_text(&result.blocks);
        // Usage: prefer a real `Block::Usage` if present, else
        // placeholders that pass schema validation on the client side.
        let (prompt_tokens, completion_tokens) = result
            .blocks
            .iter()
            .find_map(|b| match b {
                Block::Usage { prompt, completion } => Some((*prompt, *completion)),
                _ => None,
            })
            .unwrap_or((0, 0));
        let total_tokens = prompt_tokens.saturating_add(completion_tokens);
        let finish_reason = finish_reason_for(&result.outcome).to_string();
        Self {
            id,
            object: "chat.completion".to_string(),
            created,
            model: String::new(), // populated by the HTTP layer
            choices: vec![OpenAIChoice {
                index: 0,
                message: OpenAIChatMessage {
                    role: "assistant".to_string(),
                    content,
                    name: None,
                },
                finish_reason,
            }],
            usage: OpenAIUsage {
                prompt_tokens,
                completion_tokens,
                total_tokens,
            },
        }
    }
}

fn blocks_to_text(blocks: &[Block]) -> String {
    let mut out = String::new();
    for b in blocks {
        if let Block::Text { text } = b {
            out.push_str(text);
        }
    }
    out
}

const fn finish_reason_for(outcome: &TurnOutcome) -> &'static str {
    match outcome {
        TurnOutcome::Success => "stop",
        TurnOutcome::BudgetExceeded { .. } => "length",
        TurnOutcome::ToolFailure { .. } => "tool_calls",
        TurnOutcome::ModelError { .. } | TurnOutcome::UserAbort => "error",
    }
}

fn unix_now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

// ---------------------------------------------------------------------------
// HTTP server
// ---------------------------------------------------------------------------

/// Factory producing a fresh [`AgentLoop`] for each incoming request.
///
/// The OpenAI surface is stateless across requests — every
/// `/v1/chat/completions` call builds a new loop from this factory and
/// drops it once the response is sent. This keeps cancellation tokens
/// and per-turn budgets independent across concurrent clients.
pub type LoopFactory = Arc<dyn Fn() -> Result<AgentLoop, String> + Send + Sync + 'static>;

/// Configuration for [`OpenAIServer`].
#[derive(Debug, Clone)]
pub struct OpenAIServerConfig {
    /// Bind address (e.g. `127.0.0.1:8080`).
    pub bind: SocketAddr,
    /// Per-request read/write timeout for the underlying socket.
    pub request_timeout: Duration,
}

/// Synchronous OpenAI-compatible HTTP server.
pub struct OpenAIServer {
    cfg: OpenAIServerConfig,
    factory: LoopFactory,
    catalog: Arc<ModelCatalog>,
    shutdown: Arc<AtomicBool>,
}

impl std::fmt::Debug for OpenAIServer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAIServer")
            .field("cfg", &self.cfg)
            .field("shutdown", &self.shutdown.load(Ordering::Relaxed))
            .finish_non_exhaustive()
    }
}

impl OpenAIServer {
    /// Build a new server. No socket is bound until [`Self::start`] runs.
    #[must_use]
    pub fn new(cfg: OpenAIServerConfig, factory: LoopFactory, catalog: Arc<ModelCatalog>) -> Self {
        Self {
            cfg,
            factory,
            catalog,
            shutdown: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Start the acceptor loop on a dedicated thread.
    ///
    /// # Errors
    ///
    /// Returns the underlying `tiny_http` error string when the listener
    /// cannot be bound.
    pub fn start(self) -> Result<OpenAIServerHandle, String> {
        let server = Server::http(self.cfg.bind).map_err(|e| e.to_string())?;
        let bound = server
            .server_addr()
            .to_ip()
            .map_or_else(|| self.cfg.bind.to_string(), |s| s.to_string());
        let shutdown = self.shutdown.clone();
        let factory = self.factory.clone();
        let catalog = self.catalog.clone();
        let timeout = self.cfg.request_timeout;

        let acceptor = thread::Builder::new()
            .name("stratum-openai-acceptor".to_string())
            .spawn(move || {
                loop {
                    if shutdown.load(Ordering::Relaxed) {
                        return;
                    }
                    match server.recv_timeout(Duration::from_millis(100)) {
                        Ok(Some(req)) => {
                            let factory = factory.clone();
                            let catalog = catalog.clone();
                            let _ = thread::Builder::new()
                                .name("stratum-openai-conn".to_string())
                                .spawn(move || {
                                    handle_request(req, &factory, &catalog, timeout);
                                });
                        }
                        Ok(None) => {
                            // No request in this poll window — re-check shutdown.
                        }
                        Err(_) => {
                            // Listener failed — exit the loop.
                            return;
                        }
                    }
                }
            })
            .map_err(|e| e.to_string())?;

        Ok(OpenAIServerHandle {
            acceptor: Some(acceptor),
            shutdown: self.shutdown,
            bound,
        })
    }
}

/// RAII handle returned by [`OpenAIServer::start`].
#[derive(Debug)]
pub struct OpenAIServerHandle {
    acceptor: Option<JoinHandle<()>>,
    shutdown: Arc<AtomicBool>,
    bound: String,
}

impl OpenAIServerHandle {
    /// Resolved bind address (e.g. `127.0.0.1:54321`).
    #[must_use]
    pub fn bound_address(&self) -> &str {
        &self.bound
    }

    /// Trigger shutdown and join the acceptor.
    ///
    /// # Errors
    ///
    /// Propagates a panic from the acceptor thread.
    pub fn stop(mut self) -> std::thread::Result<()> {
        self.shutdown.store(true, Ordering::Relaxed);
        if let Some(h) = self.acceptor.take() {
            h.join()?;
        }
        Ok(())
    }
}

impl Drop for OpenAIServerHandle {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
        if let Some(h) = self.acceptor.take() {
            let _ = h.join();
        }
    }
}

// ---------------------------------------------------------------------------
// Request dispatch
// ---------------------------------------------------------------------------

fn handle_request(
    req: tiny_http::Request,
    factory: &LoopFactory,
    catalog: &Arc<ModelCatalog>,
    _timeout: Duration,
) {
    let method = req.method().clone();
    let url = req.url().to_string();
    let path = url.split('?').next().unwrap_or(&url).to_string();

    // CORS preflight on every endpoint.
    if matches!(method, Method::Options) {
        let resp = Response::empty(StatusCode(204));
        let _ = req.respond(with_cors(resp));
        return;
    }

    match (method, path.as_str()) {
        (Method::Post, "/v1/chat/completions") => {
            handle_chat_completions(req, factory);
        }
        (Method::Post | Method::Get, "/v1/models") => {
            handle_list_models(req, catalog);
        }
        (m, p) => {
            let msg = format!("no route for {m} {p}");
            respond_error_owned(req, 404, "not_found", &msg);
        }
    }
}

fn handle_chat_completions(mut req: tiny_http::Request, factory: &LoopFactory) {
    // Read body.
    let mut body = String::new();
    if let Err(e) = req.as_reader().read_to_string(&mut body) {
        respond_error_owned(
            req,
            400,
            "invalid_request",
            &format!("body read failed: {e}"),
        );
        return;
    }
    let parsed: OpenAIChatRequest = match serde_json::from_str(&body) {
        Ok(v) => v,
        Err(e) => {
            respond_error_owned(req, 400, "invalid_request", &format!("bad JSON: {e}"));
            return;
        }
    };

    let stream = parsed.stream;
    let model_label = parsed.model.clone();
    let ctx: TurnContext = parsed.into();

    let agent = match (factory)() {
        Ok(a) => a,
        Err(e) => {
            respond_error_owned(
                req,
                500,
                "internal_error",
                &format!("factory build failed: {e}"),
            );
            return;
        }
    };

    if stream {
        respond_stream(req, agent, ctx, &model_label);
    } else {
        respond_non_stream(req, &agent, ctx, &model_label);
    }
}

fn respond_non_stream(
    req: tiny_http::Request,
    agent: &AgentLoop,
    ctx: TurnContext,
    model_label: &str,
) {
    let cancel = CancelToken::new();
    let result = agent.run_turn(ctx, &cancel);
    let mut resp: OpenAIChatResponse = result.into();
    resp.model = model_label.to_string();
    let body = match serde_json::to_string(&resp) {
        Ok(s) => s,
        Err(e) => {
            respond_error_owned(
                req,
                500,
                "internal_error",
                &format!("serialize failed: {e}"),
            );
            return;
        }
    };
    let response = Response::from_string(body)
        .with_status_code(StatusCode(200))
        .with_header(json_header());
    let _ = req.respond(with_cors(response));
}

fn respond_stream(req: tiny_http::Request, agent: AgentLoop, ctx: TurnContext, model_label: &str) {
    let id = format!("chatcmpl-{}", ctx.turn_id.0);
    let created = unix_now_secs();
    let model = model_label.to_string();

    // Run the agent on a worker thread so we can drain chunks as they
    // arrive. The receiver collects Block::Text deltas and we emit
    // one SSE chunk per delta.
    let (chunk_tx, chunk_rx) = mpsc::channel::<Block>();
    let (done_tx, done_rx) = mpsc::channel::<TurnResult>();
    let cancel = CancelToken::new();
    let worker = thread::spawn(move || {
        let result = agent.run_turn_streaming(ctx, &cancel, chunk_tx);
        let _ = done_tx.send(result);
    });

    // Build the SSE body in memory. tiny_http's sync API doesn't
    // expose a chunked writer hook, so we accumulate the events and
    // emit a single response body. This still produces the exact
    // wire format clients expect — `data: {...}\n\n` lines — and
    // keeps the dispatcher sync. Real chunked streaming lands when
    // the runtime grows an HTTP/1.1 hand-off API.
    let mut sse = String::new();
    // First chunk emits the `role: "assistant"` delta per the
    // OpenAI streaming protocol.
    let first = OpenAIStreamChunk {
        id: id.clone(),
        object: "chat.completion.chunk".to_string(),
        created,
        model: model.clone(),
        choices: vec![OpenAIStreamChoice {
            index: 0,
            delta: OpenAIDelta {
                role: Some("assistant".to_string()),
                content: None,
            },
            finish_reason: None,
        }],
    };
    push_sse(&mut sse, &first);

    // Drain text deltas until the worker finishes.
    loop {
        match chunk_rx.recv_timeout(Duration::from_millis(200)) {
            Ok(Block::Text { text }) => {
                let chunk = OpenAIStreamChunk {
                    id: id.clone(),
                    object: "chat.completion.chunk".to_string(),
                    created,
                    model: model.clone(),
                    choices: vec![OpenAIStreamChoice {
                        index: 0,
                        delta: OpenAIDelta {
                            role: None,
                            content: Some(text),
                        },
                        finish_reason: None,
                    }],
                };
                push_sse(&mut sse, &chunk);
            }
            Ok(_) => {} // ignore non-text blocks for now.
            Err(mpsc::RecvTimeoutError::Timeout) => {
                if done_rx.try_recv().is_ok() {
                    break;
                }
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }

    // Drain any remaining buffered chunks.
    while let Ok(b) = chunk_rx.try_recv() {
        if let Block::Text { text } = b {
            let chunk = OpenAIStreamChunk {
                id: id.clone(),
                object: "chat.completion.chunk".to_string(),
                created,
                model: model.clone(),
                choices: vec![OpenAIStreamChoice {
                    index: 0,
                    delta: OpenAIDelta {
                        role: None,
                        content: Some(text),
                    },
                    finish_reason: None,
                }],
            };
            push_sse(&mut sse, &chunk);
        }
    }

    let result = done_rx.recv_timeout(Duration::from_secs(60));
    let finish = match &result {
        Ok(r) => finish_reason_for(&r.outcome).to_string(),
        Err(_) => "error".to_string(),
    };
    let _ = worker.join();
    let terminal = OpenAIStreamChunk {
        id,
        object: "chat.completion.chunk".to_string(),
        created,
        model,
        choices: vec![OpenAIStreamChoice {
            index: 0,
            delta: OpenAIDelta::default(),
            finish_reason: Some(finish),
        }],
    };
    push_sse(&mut sse, &terminal);
    sse.push_str("data: [DONE]\n\n");

    let body_bytes = sse.into_bytes();
    let len = body_bytes.len();
    let response = Response::new(
        StatusCode(200),
        vec![sse_header()],
        Cursor::new(body_bytes),
        Some(len),
        None,
    );
    let _ = req.respond(with_cors(response));
}

fn push_sse(out: &mut String, chunk: &OpenAIStreamChunk) {
    if let Ok(json) = serde_json::to_string(chunk) {
        out.push_str("data: ");
        out.push_str(&json);
        out.push_str("\n\n");
    }
}

fn handle_list_models(req: tiny_http::Request, catalog: &Arc<ModelCatalog>) {
    let list = OpenAIModelList::from_catalog(catalog);
    let body = serde_json::to_string(&list).unwrap_or_else(|_| "{}".to_string());
    let response = Response::from_string(body)
        .with_status_code(StatusCode(200))
        .with_header(json_header());
    let _ = req.respond(with_cors(response));
}

fn respond_error_owned(req: tiny_http::Request, status: u16, ty: &str, msg: &str) {
    let body = error_body(ty, msg);
    let response = Response::from_string(body)
        .with_status_code(StatusCode(status))
        .with_header(json_header());
    let _ = req.respond(with_cors(response));
}

fn error_body(ty: &str, message: &str) -> String {
    serde_json::json!({
        "error": {
            "type": ty,
            "message": message,
        }
    })
    .to_string()
}

/// Parse a `Header` from static-ASCII bytes.
///
/// All call sites pass compile-time-known ASCII literals, so
/// `Header::from_bytes` cannot fail in practice. The fallback returns a
/// minimal valid `Header` (`X-Stratum: 1`) which is also a static-ASCII
/// literal that has been parsed successfully throughout the test suite —
/// i.e. the impossible-failure path is itself infallible. We avoid
/// `expect()` per the workspace lint policy.
fn static_header(name: &'static [u8], value: &'static [u8]) -> Header {
    // Safety: `b"X-Stratum: 1"` is ASCII, Header::from_bytes accepts it.
    // The closure result is itself static-ASCII so the second call is
    // also infallible — the explicit `match` makes that obvious.
    match Header::from_bytes(name, value) {
        Ok(h) => h,
        Err(()) => match Header::from_bytes(&b"X-Stratum"[..], &b"1"[..]) {
            Ok(h) => h,
            // Truly unreachable: `b"X-Stratum"` / `b"1"` are ASCII.
            // Return a noop-shaped header by recursing on the same
            // bytes — the compiler can't prove the recursion bottoms
            // out, but Header::from_bytes is deterministic on bytes.
            Err(()) => static_header(b"X", b"1"),
        },
    }
}

fn json_header() -> Header {
    static_header(b"Content-Type", b"application/json")
}

fn sse_header() -> Header {
    static_header(b"Content-Type", b"text/event-stream")
}

fn with_cors<R: std::io::Read>(resp: Response<R>) -> Response<R> {
    resp.with_header(static_header(b"Access-Control-Allow-Origin", b"*"))
        .with_header(static_header(
            b"Access-Control-Allow-Methods",
            b"GET, POST, OPTIONS",
        ))
        .with_header(static_header(
            b"Access-Control-Allow-Headers",
            b"Content-Type, Authorization",
        ))
}

// ---------------------------------------------------------------------------
// AgentFactory bridge
// ---------------------------------------------------------------------------

/// Build a [`LoopFactory`] closure over an existing
/// [`AgentFactory`]. Each call rebuilds the inner [`AgentLoop`].
#[must_use]
pub fn loop_factory_from_agent_factory(factory: Arc<AgentFactory>) -> LoopFactory {
    Arc::new(move || (*factory).clone().build().map_err(|e| e.to_string()))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::agent_factory::AgentFactory;
    use crate::conversation::TurnTransition;
    use crate::provider::EchoProvider;
    use std::io::Read;

    fn user_msg(s: &str) -> OpenAIChatMessage {
        OpenAIChatMessage {
            role: "user".to_string(),
            content: s.to_string(),
            name: None,
        }
    }

    #[test]
    fn request_to_turn_context_extracts_trailing_user_prompt() {
        let req = OpenAIChatRequest {
            model: "echo".to_string(),
            messages: vec![
                OpenAIChatMessage {
                    role: "system".to_string(),
                    content: "be terse".to_string(),
                    name: None,
                },
                user_msg("hello"),
            ],
            stream: false,
            max_tokens: None,
            temperature: None,
            top_p: None,
        };
        let ctx: TurnContext = req.into();
        assert_eq!(ctx.user_prompt, "hello");
        assert_eq!(ctx.history.len(), 1);
        assert_eq!(ctx.history[0].content, "be terse");
    }

    #[test]
    fn turn_result_round_trips_to_response() {
        let result = TurnResult {
            turn_id: TurnId(7),
            outcome: TurnOutcome::Success,
            blocks: vec![Block::Text {
                text: "hi there".to_string(),
            }],
            transitions: Vec::<TurnTransition>::new(),
            events_emitted: Vec::new(),
        };
        let resp: OpenAIChatResponse = result.into();
        assert_eq!(resp.id, "chatcmpl-7");
        assert_eq!(resp.object, "chat.completion");
        assert_eq!(resp.choices.len(), 1);
        assert_eq!(resp.choices[0].message.role, "assistant");
        assert_eq!(resp.choices[0].message.content, "hi there");
        assert_eq!(resp.choices[0].finish_reason, "stop");
    }

    #[test]
    fn finish_reason_maps_outcome_correctly() {
        assert_eq!(finish_reason_for(&TurnOutcome::Success), "stop");
        assert_eq!(
            finish_reason_for(&TurnOutcome::BudgetExceeded { kind: "x".into() }),
            "length"
        );
        assert_eq!(
            finish_reason_for(&TurnOutcome::ToolFailure {
                tool_id: "t".into(),
                code: "c".into(),
            }),
            "tool_calls"
        );
        assert_eq!(finish_reason_for(&TurnOutcome::UserAbort), "error");
    }

    #[test]
    fn model_list_from_empty_catalog_is_empty() {
        let cat = ModelCatalog::new();
        let list = OpenAIModelList::from_catalog(&cat);
        assert_eq!(list.object, "list");
        assert!(list.data.is_empty());
    }

    #[test]
    fn request_with_only_user_message_has_empty_history() {
        let req = OpenAIChatRequest {
            model: "echo".to_string(),
            messages: vec![user_msg("hi")],
            stream: false,
            max_tokens: None,
            temperature: None,
            top_p: None,
        };
        let ctx: TurnContext = req.into();
        assert_eq!(ctx.user_prompt, "hi");
        assert!(ctx.history.is_empty());
    }

    #[test]
    fn stream_chunk_serialises_with_data_prefix_friendly_shape() {
        let chunk = OpenAIStreamChunk {
            id: "chatcmpl-1".to_string(),
            object: "chat.completion.chunk".to_string(),
            created: 0,
            model: "echo".to_string(),
            choices: vec![OpenAIStreamChoice {
                index: 0,
                delta: OpenAIDelta {
                    role: None,
                    content: Some("tok".to_string()),
                },
                finish_reason: None,
            }],
        };
        let json = serde_json::to_string(&chunk).unwrap();
        assert!(json.contains("\"chat.completion.chunk\""));
        assert!(json.contains("\"content\":\"tok\""));
    }

    // --- End-to-end HTTP tests --------------------------------------------

    fn factory_for_echo(reply: &str) -> LoopFactory {
        let reply = reply.to_string();
        loop_factory_from_agent_factory(Arc::new(
            AgentFactory::new().with_provider(Arc::new(EchoProvider::new(&reply))),
        ))
    }

    fn start_server(reply: &str) -> OpenAIServerHandle {
        let cat = Arc::new(ModelCatalog::new());
        let cfg = OpenAIServerConfig {
            bind: "127.0.0.1:0".parse().unwrap(),
            request_timeout: Duration::from_secs(2),
        };
        let srv = OpenAIServer::new(cfg, factory_for_echo(reply), cat);
        srv.start().expect("start")
    }

    fn post(addr: &str, path: &str, body: &str) -> (u16, String) {
        use std::io::{BufRead, BufReader, Write};
        use std::net::TcpStream;
        let mut s = TcpStream::connect(addr).expect("connect");
        s.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
        let req = format!(
            "POST {path} HTTP/1.1\r\nHost: {addr}\r\nContent-Length: {}\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        s.write_all(req.as_bytes()).unwrap();
        s.flush().unwrap();
        let mut r = BufReader::new(s);
        let mut status_line = String::new();
        r.read_line(&mut status_line).unwrap();
        let code: u16 = status_line
            .split_whitespace()
            .nth(1)
            .unwrap()
            .parse()
            .unwrap();
        loop {
            let mut line = String::new();
            r.read_line(&mut line).unwrap();
            if line == "\r\n" || line.is_empty() {
                break;
            }
        }
        let mut body = String::new();
        let _ = r.read_to_string(&mut body);
        (code, body)
    }

    #[test]
    fn http_chat_completion_non_stream_returns_200() {
        let handle = start_server("hello-world");
        let addr = handle.bound_address().to_string();
        let body = serde_json::json!({
            "model": "echo",
            "messages": [{"role":"user","content":"hi"}],
            "stream": false
        })
        .to_string();
        let (code, resp_body) = post(&addr, "/v1/chat/completions", &body);
        assert_eq!(code, 200);
        let v: serde_json::Value = serde_json::from_str(&resp_body).expect("json");
        assert_eq!(v["object"], "chat.completion");
        assert_eq!(v["model"], "echo");
        assert!(v["choices"][0]["message"]["content"]
            .as_str()
            .unwrap()
            .contains("hello-world"));
        let _ = handle.stop();
    }

    #[test]
    fn http_list_models_returns_list_shape() {
        let handle = start_server("x");
        let addr = handle.bound_address().to_string();
        let (code, body) = post(&addr, "/v1/models", "");
        assert_eq!(code, 200);
        let v: serde_json::Value = serde_json::from_str(&body).expect("json");
        assert_eq!(v["object"], "list");
        assert!(v["data"].is_array());
        let _ = handle.stop();
    }

    #[test]
    fn http_invalid_json_returns_400() {
        let handle = start_server("x");
        let addr = handle.bound_address().to_string();
        let (code, body) = post(&addr, "/v1/chat/completions", "{not json");
        assert_eq!(code, 400);
        let v: serde_json::Value = serde_json::from_str(&body).expect("json");
        assert_eq!(v["error"]["type"], "invalid_request");
        let _ = handle.stop();
    }

    #[test]
    fn http_unknown_route_returns_404() {
        let handle = start_server("x");
        let addr = handle.bound_address().to_string();
        let (code, body) = post(&addr, "/v1/bogus", "{}");
        assert_eq!(code, 404);
        let v: serde_json::Value = serde_json::from_str(&body).expect("json");
        assert_eq!(v["error"]["type"], "not_found");
        let _ = handle.stop();
    }

    #[test]
    fn http_chat_completion_stream_emits_sse_chunks() {
        let handle = start_server("streamed-text");
        let addr = handle.bound_address().to_string();
        let body = serde_json::json!({
            "model": "echo",
            "messages": [{"role":"user","content":"hi"}],
            "stream": true
        })
        .to_string();
        let (code, resp_body) = post(&addr, "/v1/chat/completions", &body);
        assert_eq!(code, 200);
        // SSE: each event is `data: <json>\n\n` and the stream ends with `data: [DONE]\n\n`.
        assert!(resp_body.contains("data: "));
        assert!(resp_body.contains("[DONE]"));
        assert!(resp_body.contains("chat.completion.chunk"));
        let _ = handle.stop();
    }

    #[test]
    fn http_options_preflight_returns_204_with_cors_headers() {
        use std::io::{BufRead, BufReader, Write};
        use std::net::TcpStream;
        let handle = start_server("x");
        let addr = handle.bound_address().to_string();
        let mut s = TcpStream::connect(&addr).unwrap();
        s.write_all(
            b"OPTIONS /v1/chat/completions HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n",
        )
        .unwrap();
        let mut r = BufReader::new(s);
        let mut status_line = String::new();
        r.read_line(&mut status_line).unwrap();
        assert!(status_line.contains("204"));
        let mut headers = String::new();
        loop {
            let mut line = String::new();
            r.read_line(&mut line).unwrap();
            if line == "\r\n" || line.is_empty() {
                break;
            }
            headers.push_str(&line);
        }
        assert!(headers
            .to_ascii_lowercase()
            .contains("access-control-allow-origin"));
        let _ = handle.stop();
    }
}
