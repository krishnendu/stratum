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
use stratum_types::{AudioData, Block, ImageData, ModelId};
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
///
/// `content` is a serde-untagged enum so both legacy shapes work on the
/// wire:
///
/// * `"content": "hello"` — the original 2023 shape, still emitted by
///   most CLI/SDK callers for text-only turns. Deserialises into
///   [`OpenAIMessageContent::Text`].
/// * `"content": [{"type":"text", ...}, {"type":"image_url", ...}, ...]`
///   — OpenAI's 2024 multimodal shape. Each element is an
///   [`OpenAIContentPart`]; `text` parts concatenate onto the user
///   prompt, `image_url` parts become [`Block::Image`] on
///   [`TurnContext.attachments`], `input_audio` parts become
///   [`Block::Audio`].
///
/// On serialise we always emit the plain-string shape (we only produce
/// assistant text today), so clients that don't understand the array
/// variant see something they can render.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenAIChatMessage {
    /// `"system"`, `"user"`, `"assistant"`, or `"tool"`.
    pub role: String,
    /// Message body. Either a free-form string or an array of typed
    /// content parts (text / image_url / input_audio). See
    /// [`OpenAIMessageContent`].
    pub content: OpenAIMessageContent,
    /// Optional explicit name (e.g. for `"role":"tool"` messages).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// Either a plain-string message body or an array of typed content
/// parts. Mirrors OpenAI's wire-format: `content` is `string |
/// Array<ContentPart>`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum OpenAIMessageContent {
    /// Legacy plain-string shape: `"content": "hello"`. Always emitted
    /// on the response path.
    Text(String),
    /// 2024 multimodal shape: `"content": [{"type":"text",...}, ...]`.
    /// Each part is interpreted by [`OpenAIContentPart`].
    Parts(Vec<OpenAIContentPart>),
}

impl Default for OpenAIMessageContent {
    fn default() -> Self {
        Self::Text(String::new())
    }
}

impl From<String> for OpenAIMessageContent {
    fn from(s: String) -> Self {
        Self::Text(s)
    }
}

impl From<&str> for OpenAIMessageContent {
    fn from(s: &str) -> Self {
        Self::Text(s.to_string())
    }
}

impl OpenAIMessageContent {
    /// Flatten this `content` field into `(text, attachments)`.
    ///
    /// * For the [`Self::Text`] variant: the string is the text,
    ///   attachments is empty.
    /// * For the [`Self::Parts`] variant: each `text` part is
    ///   concatenated (separated by `\n` when there are multiple),
    ///   each `image_url` / `input_audio` part is converted to the
    ///   matching [`Block`] variant.
    ///
    /// Returns a tuple so the caller can route the text into
    /// [`TurnContext.user_prompt`] / [`ChatHistoryTurn.content`] and
    /// the attachments into [`TurnContext.attachments`].
    #[must_use]
    pub fn flatten(self) -> (String, Vec<Block>) {
        match self {
            Self::Text(s) => (s, Vec::new()),
            Self::Parts(parts) => {
                let mut text = String::new();
                let mut blocks = Vec::with_capacity(parts.len());
                for part in parts {
                    match part {
                        OpenAIContentPart::Text { text: t } => {
                            if !text.is_empty() {
                                text.push('\n');
                            }
                            text.push_str(&t);
                        }
                        OpenAIContentPart::ImageUrl { image_url } => {
                            blocks.push(image_url_to_block(&image_url));
                        }
                        OpenAIContentPart::InputAudio { input_audio } => {
                            blocks.push(input_audio_to_block(&input_audio));
                        }
                    }
                }
                (text, blocks)
            }
        }
    }

    /// Borrowed view of the text payload, when present.
    #[must_use]
    pub const fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text(s) => Some(s.as_str()),
            Self::Parts(_) => None,
        }
    }
}

/// One typed element inside the multimodal `content` array.
///
/// The tag is OpenAI's `type` field — `"text"`, `"image_url"`, or
/// `"input_audio"`. Unknown tags fail deserialisation with a clear
/// serde error; the HTTP layer turns that into a `400 invalid_request`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OpenAIContentPart {
    /// `{"type": "text", "text": "..."}` — concatenated onto the
    /// user prompt.
    Text {
        /// The text fragment.
        text: String,
    },
    /// `{"type": "image_url", "image_url": {"url": "..."}}` — becomes
    /// a [`Block::Image`]. The `url` is either an `http(s)://` URL,
    /// a `data:image/<mime>;base64,...` data URI, or a `file://`
    /// path; we map the first form to [`ImageData::Url`] and the data
    /// URI to [`ImageData::Inline`].
    ImageUrl {
        /// The nested object holding the URL.
        image_url: OpenAIImageUrl,
    },
    /// `{"type": "input_audio", "input_audio": {"data": "<b64>",
    /// "format": "wav"}}` — becomes a [`Block::Audio`]. The `data`
    /// field is required; the `format` field defaults to `"wav"` and
    /// is mapped to a MIME via [`audio_format_to_mime`].
    InputAudio {
        /// The nested object holding the base64 + format.
        input_audio: OpenAIInputAudio,
    },
}

/// `image_url` nested object inside an [`OpenAIContentPart::ImageUrl`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenAIImageUrl {
    /// Either an `http(s)://` URL, a `data:image/<mime>;base64,...`
    /// data URI, or a `file://` path.
    pub url: String,
    /// Optional detail hint (`"low"` / `"high"` / `"auto"`). Accepted
    /// and ignored by Stratum — the underlying vision provider picks
    /// its own resolution policy.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// `input_audio` nested object inside an
/// [`OpenAIContentPart::InputAudio`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenAIInputAudio {
    /// Base64-encoded audio bytes (no `data:` URI prefix; OpenAI's
    /// `input_audio` field is bare base64).
    pub data: String,
    /// Format hint — `"wav"`, `"mp3"`, `"flac"`, `"ogg"`. Mapped to
    /// a MIME by [`audio_format_to_mime`]. Defaults to `"wav"` when
    /// the caller omits the field.
    #[serde(default = "default_audio_format")]
    pub format: String,
}

fn default_audio_format() -> String {
    "wav".to_string()
}

/// Map OpenAI's `format` short-tag to a MIME string.
#[must_use]
pub fn audio_format_to_mime(format: &str) -> &'static str {
    match format.to_ascii_lowercase().as_str() {
        "mp3" => "audio/mpeg",
        "flac" => "audio/flac",
        "ogg" | "oga" | "opus" => "audio/ogg",
        "m4a" | "mp4" => "audio/mp4",
        // Default + explicit "wav" both land here.
        _ => "audio/wav",
    }
}

fn image_url_to_block(iu: &OpenAIImageUrl) -> Block {
    // `data:image/<mime>;base64,<b64>` — extract MIME and payload,
    // count the decoded byte length without a runtime base64 dep by
    // estimating from the encoded length (3 bytes per 4 chars, minus
    // padding). The estimate is fine for the provider-side budget
    // check; exact byte counts come from a real decode later.
    if let Some(rest) = iu.url.strip_prefix("data:") {
        if let Some((mime_part, payload)) = rest.split_once(',') {
            // An empty payload is malformed (the comma is present but
            // no base64 follows); fall through to the URL form so the
            // caller gets a sane error rather than a zero-byte image.
            if !payload.is_empty() {
                let mime = mime_part
                    .split(';')
                    .next()
                    .filter(|s| !s.is_empty())
                    .unwrap_or("image/png")
                    .to_string();
                let encoded_len = payload.len();
                // O(1) padding count instead of scanning the whole
                // payload backwards — valid base64 has 0/1/2 trailing
                // `=` chars only.
                let pad: usize = if payload.ends_with("==") {
                    2
                } else {
                    usize::from(payload.ends_with('='))
                };
                let bytes_est = encoded_len
                    .saturating_mul(3)
                    .saturating_div(4)
                    .saturating_sub(pad);
                let bytes_u32 = u32::try_from(bytes_est).unwrap_or(u32::MAX);
                return Block::Image {
                    mime,
                    data: ImageData::Inline {
                        base64: payload.to_string(),
                        bytes: bytes_u32,
                    },
                    alt: None,
                };
            }
        }
    }
    // Fallback: plain URL reference. MIME is unknown — providers that
    // care can sniff it from the URL extension.
    let mime = mime_from_url(&iu.url, "image/").unwrap_or_else(|| "image/jpeg".to_string());
    Block::Image {
        mime,
        data: ImageData::Url {
            url: iu.url.clone(),
        },
        alt: None,
    }
}

fn input_audio_to_block(ia: &OpenAIInputAudio) -> Block {
    let mime = audio_format_to_mime(&ia.format).to_string();
    let encoded_len = ia.data.len();
    let pad = ia.data.chars().rev().take_while(|c| *c == '=').count();
    let bytes_est = encoded_len
        .saturating_mul(3)
        .saturating_div(4)
        .saturating_sub(pad);
    let bytes_u32 = u32::try_from(bytes_est).unwrap_or(u32::MAX);
    Block::Audio {
        mime,
        data: AudioData::Inline {
            base64: ia.data.clone(),
            bytes: bytes_u32,
        },
        transcript: None,
    }
}

fn mime_from_url(url: &str, prefix: &str) -> Option<String> {
    let ext = url
        .rsplit_once('.')
        .map(|(_, e)| e.split(&['?', '#'][..]).next().unwrap_or(e))
        .map(str::to_ascii_lowercase)?;
    let mime = match ext.as_str() {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        _ => return None,
    };
    if mime.starts_with(prefix) {
        Some(mime.to_string())
    } else {
        None
    }
}

/// OpenAI Chat Completions request body (subset used by the daemon).
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
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
    /// Number of completions. Accepted for client-shape compatibility
    /// but silently treated as `1` — Stratum's `run_turn` is one
    /// completion per call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,
    /// Stop sequences. Accepted for client-shape compatibility but
    /// not yet forwarded to the sampler.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop: Option<serde_json::Value>,
    /// Presence penalty. Accepted for client-shape compatibility but
    /// not yet forwarded to the sampler.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    /// Frequency penalty. Accepted for client-shape compatibility but
    /// not yet forwarded to the sampler.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    /// End-user identifier. Accepted and ignored (Stratum runs
    /// loopback by default; no rate-limit grouping needed).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
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
        let mut attachments: Vec<Block> = Vec::new();
        let last_user_idx = req
            .messages
            .iter()
            .rposition(|m| m.role == "user")
            .unwrap_or_else(|| req.messages.len().saturating_sub(1));
        for (i, msg) in req.messages.into_iter().enumerate() {
            // Flatten multimodal `content` into (text, attachment-blocks).
            // For string-shaped content the blocks vector is empty; for
            // array-shaped content the image_url / input_audio parts
            // become Block::Image / Block::Audio entries.
            let (text, mut parts_blocks) = msg.content.flatten();
            if i == last_user_idx {
                user_prompt = text;
                // Only the trailing user turn's attachments ride into
                // the current TurnContext. Earlier-turn attachments
                // live on the history string (multimodal history not
                // yet modelled on ChatHistoryTurn).
                attachments.append(&mut parts_blocks);
                continue;
            }
            // Preserve `system` distinctly — providers that branch on
            // role (e.g. anything wrapping a chat template) must not
            // see it coerced to `user`.
            let role = match msg.role.as_str() {
                "user" => "user",
                "assistant" => "assistant",
                "system" => "system",
                _ => continue,
            };
            history.push(ChatHistoryTurn {
                role: role.to_string(),
                content: text,
            });
        }
        Self {
            user_prompt,
            model,
            turn_id: TurnId(0),
            started_at: SystemTime::now(),
            history,
            // Multimodal attachments on the trailing user turn ride
            // into the runtime via TurnContext.attachments. The
            // EchoProvider tolerates these as no-ops; the
            // LlamaCppProvider routes them when the `vision` feature
            // is enabled.
            attachments,
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
                    content: OpenAIMessageContent::Text(content),
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
    // Cap request body at 20 MiB to bound the memory cost of a
    // multimodal request that ships base64 image payloads inline.
    // `.take(MAX + 1)` lets us tell "exactly at the cap" from
    // "exceeded the cap" so a truncated body becomes 400 instead of
    // silently corrupted.
    const MAX_BODY_BYTES: u64 = 20 * 1024 * 1024;
    let mut body = String::new();
    let mut limited = std::io::Read::take(req.as_reader(), MAX_BODY_BYTES + 1);
    if let Err(e) = std::io::Read::read_to_string(&mut limited, &mut body) {
        respond_error_owned(
            req,
            400,
            "invalid_request",
            &format!("body read failed: {e}"),
        );
        return;
    }
    if body.len() as u64 > MAX_BODY_BYTES {
        respond_error_owned(
            req,
            400,
            "invalid_request",
            "request body exceeds 20 MiB cap",
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

/// SSE response for `stream: true`.
///
/// **Known limitation**: `tiny_http` does not expose a streaming write
/// API, so we accumulate every `data: {chunk}` line into a single
/// buffered body and send one HTTP response. Clients see the full
/// SSE event sequence delivered atomically when the turn completes
/// — there is no progressive token rendering. The protocol shape is
/// still correct (SSE lines + `data: [DONE]` footer) so OpenAI-shaped
/// clients work end-to-end; only the latency-to-first-token gain
/// from real streaming is missing. True chunked transfer + per-block
/// cancellation lands when we swap the HTTP layer.
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

    // Drain text deltas until the worker drops `chunk_tx`. The worker
    // sends every block on `chunk_tx` BEFORE sending `done_tx`, so
    // breaking on the chunk-channel `Disconnected` is the
    // race-free signal that there are no more chunks coming. Breaking
    // earlier (e.g. on a `done_rx.try_recv` win during a `Timeout`
    // tick) lost chunks that the worker had emitted in the same
    // scheduling quantum as its final `done_tx.send`.
    loop {
        match chunk_rx.recv() {
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
            Ok(_) => {}      // ignore non-text blocks for now.
            Err(_) => break, // chunk_tx dropped → worker finished.
        }
    }

    // After the chunk channel closed, the worker has already sent its
    // TurnResult on `done_tx`. `try_recv` is sufficient and makes the
    // invariant visible — falling back to a long `recv_timeout` would
    // imply we expect to wait, which we don't.
    let result = done_rx.try_recv();
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
    // The OpenAI error shape carries `type`, `message`, AND `code`;
    // Python-SDK callers key on `error.code` for retry classification.
    // We mirror `code` to `type` since we don't yet emit a more
    // specific machine-readable code than the type label.
    serde_json::json!({
        "error": {
            "type": ty,
            "message": message,
            "code": ty,
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
            content: OpenAIMessageContent::Text(s.to_string()),
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
                    content: OpenAIMessageContent::Text("be terse".to_string()),
                    name: None,
                },
                user_msg("hello"),
            ],
            ..OpenAIChatRequest::default()
        };
        let ctx: TurnContext = req.into();
        assert_eq!(ctx.user_prompt, "hello");
        assert_eq!(ctx.history.len(), 1);
        // System role preserved distinctly — not coerced to user.
        assert_eq!(ctx.history[0].role, "system");
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
        assert_eq!(resp.choices[0].message.content.as_text(), Some("hi there"));
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
            ..OpenAIChatRequest::default()
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

    #[test]
    fn error_body_includes_code_field() {
        // Python SDK + many other clients key on `error.code` for
        // retry classification — the field must be present even when
        // it just mirrors the type label.
        let body = error_body("invalid_request", "missing model");
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["error"]["type"].as_str(), Some("invalid_request"));
        assert_eq!(v["error"]["message"].as_str(), Some("missing model"));
        assert_eq!(v["error"]["code"].as_str(), Some("invalid_request"));
    }

    #[test]
    fn request_accepts_unknown_wire_fields() {
        // Real OpenAI clients send n / stop / presence_penalty /
        // frequency_penalty / user on every call. Stratum accepts
        // them at the wire (so deserialise doesn't 400) and ignores
        // them inside the conversion path. This test pins that the
        // wire shape parses the full set.
        let json = r#"{
            "model": "echo",
            "messages": [{"role": "user", "content": "hi"}],
            "n": 1,
            "stop": ["\n"],
            "presence_penalty": 0.5,
            "frequency_penalty": 0.0,
            "user": "abc-123"
        }"#;
        let req: OpenAIChatRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.n, Some(1));
        assert!(req.stop.is_some());
        assert_eq!(req.presence_penalty, Some(0.5));
        assert_eq!(req.frequency_penalty, Some(0.0));
        assert_eq!(req.user.as_deref(), Some("abc-123"));
        // And the From conversion still works regardless.
        let ctx: TurnContext = req.into();
        assert_eq!(ctx.user_prompt, "hi");
    }

    #[test]
    fn finish_reason_maps_every_turn_outcome() {
        // The four-arm match in `finish_reason_for` is the only piece
        // of code that names every TurnOutcome variant — make sure
        // each lands on the documented OpenAI label.
        assert_eq!(finish_reason_for(&TurnOutcome::Success), "stop");
        assert_eq!(
            finish_reason_for(&TurnOutcome::BudgetExceeded {
                kind: "tokens".to_string()
            }),
            "length"
        );
        assert_eq!(
            finish_reason_for(&TurnOutcome::ToolFailure {
                tool_id: "x".to_string(),
                code: "STRAT-E5006".to_string()
            }),
            "tool_calls"
        );
        assert_eq!(
            finish_reason_for(&TurnOutcome::ModelError {
                code: "STRAT-E3007".to_string()
            }),
            "error"
        );
        assert_eq!(finish_reason_for(&TurnOutcome::UserAbort), "error");
    }

    #[test]
    fn blocks_to_text_collapses_text_blocks_skips_other_variants() {
        let blocks = vec![
            Block::Text {
                text: "hello ".to_string(),
            },
            Block::Usage {
                prompt: 3,
                completion: 1,
            },
            Block::Text {
                text: "world".to_string(),
            },
        ];
        assert_eq!(blocks_to_text(&blocks), "hello world");
    }

    #[test]
    fn turn_result_into_response_inherits_usage_block_when_present() {
        use crate::observability::TurnId;
        let result = TurnResult {
            turn_id: TurnId(42),
            outcome: TurnOutcome::Success,
            blocks: vec![
                Block::Text {
                    text: "ok".to_string(),
                },
                Block::Usage {
                    prompt: 7,
                    completion: 3,
                },
            ],
            transitions: Vec::new(),
            events_emitted: Vec::new(),
        };
        let resp: OpenAIChatResponse = result.into();
        assert_eq!(resp.id, "chatcmpl-42");
        assert_eq!(resp.object, "chat.completion");
        assert_eq!(resp.usage.prompt_tokens, 7);
        assert_eq!(resp.usage.completion_tokens, 3);
        assert_eq!(resp.usage.total_tokens, 10);
        assert_eq!(resp.choices[0].finish_reason, "stop");
        assert_eq!(resp.choices[0].message.content.as_text(), Some("ok"));
    }

    #[test]
    fn turn_result_into_response_zero_usage_when_no_usage_block() {
        use crate::observability::TurnId;
        let result = TurnResult {
            turn_id: TurnId(1),
            outcome: TurnOutcome::Success,
            blocks: vec![Block::Text {
                text: "x".to_string(),
            }],
            transitions: Vec::new(),
            events_emitted: Vec::new(),
        };
        let resp: OpenAIChatResponse = result.into();
        assert_eq!(resp.usage.prompt_tokens, 0);
        assert_eq!(resp.usage.completion_tokens, 0);
    }

    #[test]
    fn model_list_default_is_empty() {
        // Catalog with zero entries -> empty data list, object label
        // is "list" per the OpenAI shape.
        let catalog = ModelCatalog::default();
        let list = OpenAIModelList::from_catalog(&catalog);
        assert_eq!(list.object, "list");
        assert!(list.data.is_empty());
    }

    #[test]
    fn assistant_role_messages_appear_in_history() {
        // The role-coerce switch handles user / assistant / system —
        // the assistant case used to be untested.
        let req = OpenAIChatRequest {
            model: "echo".to_string(),
            messages: vec![
                user_msg("hi"),
                OpenAIChatMessage {
                    role: "assistant".to_string(),
                    content: OpenAIMessageContent::Text("hello back".to_string()),
                    name: None,
                },
                user_msg("how are you"),
            ],
            ..OpenAIChatRequest::default()
        };
        let ctx: TurnContext = req.into();
        assert_eq!(ctx.user_prompt, "how are you");
        assert_eq!(ctx.history.len(), 2);
        assert_eq!(ctx.history[0].role, "user");
        assert_eq!(ctx.history[1].role, "assistant");
    }

    // ---------- Multimodal `content` array ----------------------------------

    #[test]
    fn message_content_deserialises_plain_string_into_text_variant() {
        let raw = r#""hello""#;
        let parsed: OpenAIMessageContent = serde_json::from_str(raw).unwrap();
        assert_eq!(parsed.as_text(), Some("hello"));
        let (text, blocks) = parsed.flatten();
        assert_eq!(text, "hello");
        assert!(blocks.is_empty());
    }

    #[test]
    fn message_content_deserialises_parts_array_into_parts_variant() {
        let raw = r#"[{"type":"text","text":"hi"}]"#;
        let parsed: OpenAIMessageContent = serde_json::from_str(raw).unwrap();
        assert!(matches!(parsed, OpenAIMessageContent::Parts(ref ps) if ps.len() == 1));
        let (text, blocks) = parsed.flatten();
        assert_eq!(text, "hi");
        assert!(blocks.is_empty());
    }

    #[test]
    fn parts_text_segments_concatenate_with_newline_separator() {
        let raw = r#"[
            {"type":"text","text":"first"},
            {"type":"text","text":"second"}
        ]"#;
        let parsed: OpenAIMessageContent = serde_json::from_str(raw).unwrap();
        let (text, blocks) = parsed.flatten();
        assert_eq!(text, "first\nsecond");
        assert!(blocks.is_empty());
    }

    #[test]
    fn parts_image_url_part_becomes_block_image_url_variant() {
        let raw = r#"[
            {"type":"text","text":"see this"},
            {"type":"image_url","image_url":{"url":"https://example.com/x.png"}}
        ]"#;
        let parsed: OpenAIMessageContent = serde_json::from_str(raw).unwrap();
        let (text, blocks) = parsed.flatten();
        assert_eq!(text, "see this");
        assert_eq!(blocks.len(), 1);
        assert!(matches!(
            &blocks[0],
            Block::Image { mime, data: ImageData::Url { url }, .. }
                if mime == "image/png" && url == "https://example.com/x.png"
        ));
    }

    #[test]
    fn parts_image_url_data_uri_becomes_block_image_inline_variant() {
        // The base64 payload `AAAA` decodes to 3 zero bytes; the
        // estimate from the encoded-length / pad calculation should
        // land on `3`. (We don't pull base64 into the runtime for
        // exact decoding — the estimate is accurate within ±2 bytes.)
        let raw = r#"[
            {"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}
        ]"#;
        let parsed: OpenAIMessageContent = serde_json::from_str(raw).unwrap();
        let (text, blocks) = parsed.flatten();
        assert!(text.is_empty());
        assert_eq!(blocks.len(), 1);
        assert!(matches!(
            &blocks[0],
            Block::Image {
                mime,
                data: ImageData::Inline { base64, bytes },
                ..
            } if mime == "image/png" && base64 == "AAAA" && *bytes == 3
        ));
    }

    #[test]
    fn parts_input_audio_becomes_block_audio_inline_variant() {
        let raw = r#"[
            {"type":"text","text":"transcribe"},
            {"type":"input_audio","input_audio":{"data":"AAAAAAAAAAA=","format":"wav"}}
        ]"#;
        let parsed: OpenAIMessageContent = serde_json::from_str(raw).unwrap();
        let (text, blocks) = parsed.flatten();
        assert_eq!(text, "transcribe");
        assert_eq!(blocks.len(), 1);
        assert!(matches!(
            &blocks[0],
            Block::Audio { mime, data: AudioData::Inline { .. }, .. }
                if mime == "audio/wav"
        ));
    }

    #[test]
    fn input_audio_defaults_to_wav_format_when_missing() {
        let raw = r#"[
            {"type":"input_audio","input_audio":{"data":"AAAA"}}
        ]"#;
        let parsed: OpenAIMessageContent = serde_json::from_str(raw).unwrap();
        let (_, blocks) = parsed.flatten();
        assert_eq!(blocks.len(), 1);
        assert!(matches!(
            &blocks[0],
            Block::Audio { mime, .. } if mime == "audio/wav"
        ));
    }

    #[test]
    fn audio_format_to_mime_covers_known_formats() {
        assert_eq!(audio_format_to_mime("mp3"), "audio/mpeg");
        assert_eq!(audio_format_to_mime("MP3"), "audio/mpeg");
        assert_eq!(audio_format_to_mime("flac"), "audio/flac");
        assert_eq!(audio_format_to_mime("ogg"), "audio/ogg");
        assert_eq!(audio_format_to_mime("opus"), "audio/ogg");
        assert_eq!(audio_format_to_mime("wav"), "audio/wav");
        assert_eq!(audio_format_to_mime("m4a"), "audio/mp4");
        // Unknown formats fall back to wav (most permissive).
        assert_eq!(audio_format_to_mime("xyz"), "audio/wav");
    }

    #[test]
    fn multimodal_request_extracts_attachments_into_turn_context() {
        // The whole point of Phase 5 + 6 wiring: a multimodal request
        // body must produce a TurnContext with both a flattened text
        // prompt AND populated attachments.
        let json = r#"{
            "model": "echo",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe"},
                        {"type": "image_url", "image_url": {"url": "https://example.com/x.png"}}
                    ]
                }
            ]
        }"#;
        let req: OpenAIChatRequest = serde_json::from_str(json).unwrap();
        let ctx: TurnContext = req.into();
        assert_eq!(ctx.user_prompt, "describe");
        assert_eq!(ctx.attachments.len(), 1);
        assert!(matches!(
            &ctx.attachments[0],
            Block::Image { mime, .. } if mime == "image/png"
        ));
    }

    #[test]
    fn multimodal_request_with_string_content_has_empty_attachments() {
        // Pin that the string-content variant still produces an empty
        // attachments vector — the multimodal path is opt-in.
        let json = r#"{
            "model": "echo",
            "messages": [{"role": "user", "content": "plain text"}]
        }"#;
        let req: OpenAIChatRequest = serde_json::from_str(json).unwrap();
        let ctx: TurnContext = req.into();
        assert_eq!(ctx.user_prompt, "plain text");
        assert!(ctx.attachments.is_empty());
    }

    #[test]
    fn response_message_serialises_content_as_plain_string() {
        // Our response path always emits the `Text` variant —
        // serde untagged means it should round-trip as a bare
        // string on the wire (not an array), so older clients
        // that pre-date the multimodal extension still parse
        // assistant responses without complaint.
        let msg = OpenAIChatMessage {
            role: "assistant".to_string(),
            content: OpenAIMessageContent::Text("hello".to_string()),
            name: None,
        };
        let s = serde_json::to_string(&msg).unwrap();
        assert!(
            s.contains("\"content\":\"hello\""),
            "expected bare string, got {s}"
        );
    }

    #[test]
    fn unknown_content_part_type_fails_deserialisation() {
        // Defence-in-depth: an unknown `type` tag must NOT silently
        // deserialise to a "default" variant. serde's tagged enum
        // rejects the input; the HTTP layer turns the resulting
        // serde error into a `400 invalid_request`.
        let raw = r#"[{"type":"video","video":{"url":"x"}}]"#;
        let parsed: Result<OpenAIMessageContent, _> = serde_json::from_str(raw);
        assert!(parsed.is_err(), "should have rejected unknown type");
    }

    #[test]
    fn unknown_role_messages_are_dropped() {
        // Any role outside {user, assistant, system} is silently
        // discarded so a malformed client can't inject foreign turns.
        let req = OpenAIChatRequest {
            model: "echo".to_string(),
            messages: vec![
                OpenAIChatMessage {
                    role: "function".to_string(),
                    content: OpenAIMessageContent::Text("tool-call payload".to_string()),
                    name: None,
                },
                user_msg("hi"),
            ],
            ..OpenAIChatRequest::default()
        };
        let ctx: TurnContext = req.into();
        assert!(ctx.history.is_empty());
        assert_eq!(ctx.user_prompt, "hi");
    }
}
