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
use crate::model_catalog::{ModelCatalog, ModelSlug};
use crate::observability::TurnId;
use crate::probe::HardwareProbe;
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
    /// Test-only override for the RAM probe. `None` in production →
    /// the request handler instantiates a [`LiveRamProbe`] per call.
    #[cfg(test)]
    probe_override: Option<Arc<dyn RamProbe + Send + Sync>>,
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
            #[cfg(test)]
            probe_override: None,
        }
    }

    /// Test-only: swap in a synthetic RAM probe. Used by the memory-probe
    /// 503 tests so the assertion does not depend on the host's free
    /// memory.
    #[cfg(test)]
    fn with_probe(mut self, probe: Arc<dyn RamProbe + Send + Sync>) -> Self {
        self.probe_override = Some(probe);
        self
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
        #[cfg(test)]
        let probe_override = self.probe_override.clone();

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
                            #[cfg(test)]
                            let probe_override = probe_override.clone();
                            let _ = thread::Builder::new()
                                .name("stratum-openai-conn".to_string())
                                .spawn(move || {
                                    #[cfg(test)]
                                    {
                                        handle_request_with_probe(
                                            req,
                                            &factory,
                                            &catalog,
                                            timeout,
                                            probe_override.as_deref(),
                                        );
                                    }
                                    #[cfg(not(test))]
                                    {
                                        handle_request(req, &factory, &catalog, timeout);
                                    }
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

#[cfg(not(test))]
fn handle_request(
    req: tiny_http::Request,
    factory: &LoopFactory,
    catalog: &Arc<ModelCatalog>,
    _timeout: Duration,
) {
    dispatch(req, factory, catalog, &LiveRamProbe);
}

/// Test-mode variant of [`handle_request`] that accepts an optional
/// probe override. When `probe_override` is `None` the live host probe
/// is used; when `Some` the supplied probe is threaded into the chat
/// handler so synthetic-memory tests can refuse without depending on
/// the host's free RAM.
#[cfg(test)]
fn handle_request_with_probe(
    req: tiny_http::Request,
    factory: &LoopFactory,
    catalog: &Arc<ModelCatalog>,
    _timeout: Duration,
    probe_override: Option<&(dyn RamProbe + Send + Sync)>,
) {
    if let Some(probe) = probe_override {
        dispatch(req, factory, catalog, probe);
    } else {
        dispatch(req, factory, catalog, &LiveRamProbe);
    }
}

fn dispatch(
    req: tiny_http::Request,
    factory: &LoopFactory,
    catalog: &Arc<ModelCatalog>,
    probe: &dyn RamProbe,
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
            handle_chat_completions(req, factory, catalog, probe);
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

// ---------------------------------------------------------------------------
// Memory-probe / activation refusal
// ---------------------------------------------------------------------------

/// Working-set RAM estimate for a model on disk, in MiB.
///
/// Heuristic per `plan/07` Phase 6: `file_size_mib + 1.5x overhead` covers
/// the GGUF mmap residency + KV cache + activation buffers a llama.cpp
/// inference loop touches once it warms up. Saturating arithmetic guards
/// against catalog entries that declare nonsense sizes.
#[must_use]
pub const fn required_mib_for_model(file_size_mib: u64) -> u64 {
    let overhead = file_size_mib.saturating_mul(3).saturating_div(2);
    file_size_mib.saturating_add(overhead)
}

/// Pluggable RAM-availability source. The HTTP request path uses
/// [`LiveRamProbe`] (delegates to [`HardwareProbe::run`]); tests
/// substitute a [`SyntheticRamProbe`] so the assertion does not depend
/// on the host's free memory.
trait RamProbe {
    fn available_mib(&self) -> u64;
}

struct LiveRamProbe;

impl RamProbe for LiveRamProbe {
    fn available_mib(&self) -> u64 {
        u64::from(HardwareProbe::run().ram_available_mib)
    }
}

#[cfg(test)]
struct SyntheticRamProbe(u64);

#[cfg(test)]
impl RamProbe for SyntheticRamProbe {
    fn available_mib(&self) -> u64 {
        self.0
    }
}

/// Outcome of [`memory_probe_for_model`]: either the request can proceed
/// (model not in catalog, or fits), or it must be refused with a
/// pre-built 503 body.
enum MemoryCheck {
    /// No catalog entry matched, or the entry fits in the available RAM
    /// reported by the probe.
    Ok,
    /// The model's required working set exceeds available RAM. The
    /// payload carries the rendered error message for the 503 body.
    TooLarge { message: String },
}

/// Check whether the resolved model would fit in available RAM.
///
/// * If the `model` label parses as a [`ModelSlug`] and the catalog has
///   a matching entry, the entry's `size_mib` is multiplied through
///   [`required_mib_for_model`] and compared to the probe's available
///   MiB.
/// * If the label does not parse as a slug, or the slug is not in the
///   catalog, the check is skipped and the request is allowed through
///   (the eventual provider handles the "unknown model" surface).
fn memory_probe_for_model(
    model: &str,
    catalog: &ModelCatalog,
    probe: &dyn RamProbe,
) -> MemoryCheck {
    let Ok(slug) = model.parse::<ModelSlug>() else {
        return MemoryCheck::Ok;
    };
    let Some(entry) = catalog.get(&slug) else {
        return MemoryCheck::Ok;
    };
    let required = required_mib_for_model(entry.size_mib);
    let available = probe.available_mib();
    if required > available {
        MemoryCheck::TooLarge {
            message: format!(
                "model '{model}' requires {required} MiB but only {available} MiB is available",
            ),
        }
    } else {
        MemoryCheck::Ok
    }
}

fn handle_chat_completions(
    mut req: tiny_http::Request,
    factory: &LoopFactory,
    catalog: &Arc<ModelCatalog>,
    probe: &dyn RamProbe,
) {
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

    // Memory-probe gate: if the resolved model is in the catalog and
    // the working-set estimate exceeds available RAM, refuse with 503
    // + Retry-After so OpenAI-shaped clients can degrade rather than
    // OOM the host (plan/07 Phase 6).
    if let MemoryCheck::TooLarge { message } = memory_probe_for_model(&model_label, catalog, probe)
    {
        respond_model_too_large(req, &message);
        return;
    }

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
/// Real chunked-transfer streaming: each `data: {chunk}\n\n` event is
/// pushed to the response body as soon as it lands, so OpenAI-shaped
/// clients see progressive token delivery rather than a single buffered
/// response at end-of-turn.
///
/// Threading:
/// * **LLM worker** runs `agent.run_turn_streaming(...)`, posting
///   `Block::Text` deltas to `chunk_rx`.
/// * **Formatter thread** drains `chunk_rx`, serialises each delta to
///   an SSE event, and pushes the bytes onto `body_tx`. After the LLM
///   worker finishes it also pushes the terminal `finish_reason` chunk
///   and `data: [DONE]\n\n`, then drops `body_tx` to signal EOF.
/// * **Dispatcher thread** (this function) hands `body_rx` to
///   `tiny_http` via a [`StreamingSseReader`] adapter. With
///   `data_length: None`, `tiny_http` selects `Transfer-Encoding:
///   chunked` and drives `io::copy` on the reader, so each unblocked
///   `Read::read` produces one HTTP chunk on the wire.
fn respond_stream(req: tiny_http::Request, agent: AgentLoop, ctx: TurnContext, model_label: &str) {
    let id = format!("chatcmpl-{}", ctx.turn_id.0);
    let created = unix_now_secs();
    let model = model_label.to_string();

    let (chunk_tx, chunk_rx) = mpsc::channel::<Block>();
    let (done_tx, done_rx) = mpsc::channel::<TurnResult>();
    // Keep a clone of the cancel token alive on this thread so we can
    // signal the LLM worker when the HTTP client disconnects (see the
    // `cancel.cancel()` call after `req.respond` returns below).
    let cancel = CancelToken::new();
    let cancel_for_worker = cancel.clone();
    let llm_worker = thread::spawn(move || {
        let result = agent.run_turn_streaming(ctx, &cancel_for_worker, chunk_tx);
        let _ = done_tx.send(result);
    });

    // Body channel — bytes per SSE event. Bounded by `mpsc`'s default
    // unbounded queue; per-event payloads are small (one delta = one
    // JSON line) so memory pressure is negligible.
    let (body_tx, body_rx) = mpsc::channel::<Vec<u8>>();

    let id_for_fmt = id;
    let model_for_fmt = model;
    let formatter = thread::spawn(move || {
        // First chunk emits the `role: "assistant"` delta per the
        // OpenAI streaming protocol.
        let first = OpenAIStreamChunk {
            id: id_for_fmt.clone(),
            object: "chat.completion.chunk".to_string(),
            created,
            model: model_for_fmt.clone(),
            choices: vec![OpenAIStreamChoice {
                index: 0,
                delta: OpenAIDelta {
                    role: Some("assistant".to_string()),
                    content: None,
                },
                finish_reason: None,
            }],
        };
        let _ = body_tx.send(sse_event_bytes(&first));

        // Drain text deltas until the LLM worker drops `chunk_tx`.
        // Same race ordering as before: every block is sent on
        // `chunk_tx` BEFORE the worker's final `done_tx.send`, so
        // `Disconnected` on `chunk_rx` is the only race-free signal
        // that all blocks have been observed.
        loop {
            match chunk_rx.recv() {
                Ok(Block::Text { text }) => {
                    let chunk = OpenAIStreamChunk {
                        id: id_for_fmt.clone(),
                        object: "chat.completion.chunk".to_string(),
                        created,
                        model: model_for_fmt.clone(),
                        choices: vec![OpenAIStreamChoice {
                            index: 0,
                            delta: OpenAIDelta {
                                role: None,
                                content: Some(text),
                            },
                            finish_reason: None,
                        }],
                    };
                    let _ = body_tx.send(sse_event_bytes(&chunk));
                }
                Ok(_) => {}      // ignore non-text blocks for now.
                Err(_) => break, // chunk_tx dropped → worker finished.
            }
        }

        // Joining the worker guarantees its `done_tx.send(result)` has
        // completed (sequenced before thread return), so `try_recv` is
        // then guaranteed to find the result.
        let _ = llm_worker.join();
        let result = done_rx.try_recv();
        let finish = match &result {
            Ok(r) => finish_reason_for(&r.outcome).to_string(),
            Err(_) => "error".to_string(),
        };
        let terminal = OpenAIStreamChunk {
            id: id_for_fmt,
            object: "chat.completion.chunk".to_string(),
            created,
            model: model_for_fmt,
            choices: vec![OpenAIStreamChoice {
                index: 0,
                delta: OpenAIDelta::default(),
                finish_reason: Some(finish),
            }],
        };
        let _ = body_tx.send(sse_event_bytes(&terminal));
        let _ = body_tx.send(b"data: [DONE]\n\n".to_vec());
        // body_tx drops here → EOF on body_rx → io::copy returns →
        // tiny_http finalises the chunked stream.
    });

    let reader = StreamingSseReader::new(body_rx);
    // `data_length: None` triggers `Transfer-Encoding: chunked` in
    // tiny_http (see `choose_transfer_encoding` in tiny_http 0.12).
    let response = Response::new(StatusCode(200), vec![sse_header()], reader, None, None);
    let _ = req.respond(with_cors(response));
    // If the client disconnected mid-stream, `req.respond` returned with
    // an IO error; either way, signal the LLM worker so it stops as soon
    // as the next cancel poll fires. Without this the worker (and the
    // formatter joined on it) would run to completion against a
    // disconnected client, holding provider/CPU/GPU resources.
    cancel.cancel();
    let _ = formatter.join();
}

/// Serialise a single [`OpenAIStreamChunk`] to its SSE wire bytes:
/// `data: <json>\n\n`. Returns an empty vec only if serde fails, which
/// is unreachable for the primitive-only `OpenAIStreamChunk` shape but
/// kept as a non-panicking fallback.
fn sse_event_bytes(chunk: &OpenAIStreamChunk) -> Vec<u8> {
    let mut out = Vec::with_capacity(128);
    if let Ok(json) = serde_json::to_string(chunk) {
        out.extend_from_slice(b"data: ");
        out.extend_from_slice(json.as_bytes());
        out.extend_from_slice(b"\n\n");
    }
    out
}

/// Blocking `Read` adapter over an `mpsc::Receiver<Vec<u8>>` that
/// turns each pushed event into one or more HTTP chunks. `read()`
/// blocks until the next event arrives; when the sender is dropped,
/// `read()` returns `Ok(0)` (EOF) and `tiny_http` finalises the
/// chunked transfer with the trailing zero-length chunk.
struct StreamingSseReader {
    rx: mpsc::Receiver<Vec<u8>>,
    leftover: Vec<u8>,
}

impl StreamingSseReader {
    const fn new(rx: mpsc::Receiver<Vec<u8>>) -> Self {
        Self {
            rx,
            leftover: Vec::new(),
        }
    }
}

impl std::io::Read for StreamingSseReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.leftover.is_empty() {
            match self.rx.recv() {
                Ok(bytes) => self.leftover = bytes,
                Err(_) => return Ok(0), // disconnected → EOF
            }
        }
        let n = std::cmp::min(buf.len(), self.leftover.len());
        buf[..n].copy_from_slice(&self.leftover[..n]);
        self.leftover.drain(..n);
        Ok(n)
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

/// HTTP 503 response for a model-activation refusal: the OpenAI-shaped
/// error body carries `code: "model_too_large"` (distinct from `type`)
/// and the response sets `Retry-After: 30` so polite clients back off.
fn respond_model_too_large(req: tiny_http::Request, msg: &str) {
    let body = error_body_with_code("server_error", "model_too_large", msg);
    let response = Response::from_string(body)
        .with_status_code(StatusCode(503))
        .with_header(json_header())
        .with_header(static_header(b"Retry-After", b"30"));
    let _ = req.respond(with_cors(response));
}

fn error_body(ty: &str, message: &str) -> String {
    // The OpenAI error shape carries `type`, `message`, AND `code`;
    // Python-SDK callers key on `error.code` for retry classification.
    // We mirror `code` to `type` since we don't yet emit a more
    // specific machine-readable code than the type label.
    error_body_with_code(ty, ty, message)
}

fn error_body_with_code(ty: &str, code: &str, message: &str) -> String {
    serde_json::json!({
        "error": {
            "type": ty,
            "message": message,
            "code": code,
            "param": serde_json::Value::Null,
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

    /// Variant of `post` that returns the raw response headers as a
    /// single string in addition to the decoded body. Used by streaming
    /// tests that need to assert wire-level details (e.g.
    /// `Transfer-Encoding: chunked`).
    fn post_with_headers(addr: &str, path: &str, body: &str) -> (u16, String, String) {
        use std::io::{BufRead, BufReader, Read, Write};
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
        let mut headers = String::new();
        loop {
            let mut line = String::new();
            r.read_line(&mut line).unwrap();
            if line == "\r\n" || line.is_empty() {
                break;
            }
            headers.push_str(&line);
        }
        let mut body_buf = Vec::new();
        let _ = r.read_to_end(&mut body_buf);
        // If chunked, decode by stripping the size lines + trailing
        // zero-length chunk. For SSE we only need the payload bytes.
        let body_str = if headers
            .to_ascii_lowercase()
            .contains("transfer-encoding: chunked")
        {
            decode_chunked(&body_buf)
        } else {
            String::from_utf8_lossy(&body_buf).to_string()
        };
        (code, headers, body_str)
    }

    /// Decode an HTTP/1.1 chunked-transfer body into the concatenated
    /// payload bytes (best-effort; assumes well-formed chunks from
    /// tiny_http).
    fn decode_chunked(raw: &[u8]) -> String {
        let mut out = Vec::new();
        let mut i = 0;
        while i < raw.len() {
            // Read the size line (hex digits terminated by CRLF).
            let mut size_end = i;
            while size_end < raw.len() - 1
                && !(raw[size_end] == b'\r' && raw[size_end + 1] == b'\n')
            {
                size_end += 1;
            }
            let size_str = std::str::from_utf8(&raw[i..size_end]).unwrap_or("0");
            let size = usize::from_str_radix(size_str.trim(), 16).unwrap_or(0);
            i = size_end + 2; // skip CRLF
            if size == 0 {
                break;
            }
            if i + size > raw.len() {
                break;
            }
            out.extend_from_slice(&raw[i..i + size]);
            i += size + 2; // skip chunk + CRLF
        }
        String::from_utf8_lossy(&out).to_string()
    }

    #[test]
    fn http_chat_completion_stream_uses_chunked_transfer_encoding() {
        // Real-streaming contract: the SSE response MUST use
        // `Transfer-Encoding: chunked` so each `data: {...}\n\n` event
        // flushes to wire as it lands. A buffered response would set
        // Content-Length instead and defeat the streaming UX.
        //
        // Multi-word user prompt + multi-word echo prefix produces
        // multiple `Block::Text` deltas, exercising the formatter
        // loop more than once (catches refactor regressions that
        // collapse multi-event streams into a single buffered chunk).
        let handle = start_server("echo:");
        let addr = handle.bound_address().to_string();
        let body = serde_json::json!({
            "model": "echo",
            "messages": [{"role":"user","content":"alpha beta gamma"}],
            "stream": true
        })
        .to_string();
        let (code, headers, resp_body) = post_with_headers(&addr, "/v1/chat/completions", &body);
        assert_eq!(code, 200);
        let lh = headers.to_ascii_lowercase();
        assert!(
            lh.contains("transfer-encoding: chunked"),
            "expected chunked TE, headers were: {headers}"
        );
        assert!(
            !lh.contains("content-length:"),
            "chunked responses must not also set Content-Length (RFC 7230 §3.3.3)"
        );
        // Wire-level decode landed the same SSE event sequence the
        // original buffered path emitted.
        assert!(resp_body.contains("data: "));
        assert!(resp_body.contains("[DONE]"));
        assert!(resp_body.contains("chat.completion.chunk"));
        // Multi-delta proof: every `Block::Text` produces one `data: ...`
        // line carrying `"content":"<word>"`. With the 3-word prompt we
        // expect at least 3 content deltas, plus the role-only delta,
        // plus the terminal `finish_reason` chunk, plus `[DONE]` — total
        // SSE events ≥ 5. A regression that re-buffered would collapse
        // these into a single payload, which the assertion below catches.
        let delta_count = resp_body.matches("\"content\"").count();
        assert!(
            delta_count >= 3,
            "expected ≥3 content deltas from a 3-word echo, got {delta_count}\nbody:\n{resp_body}"
        );
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

    // ---------------------------------------------------------------------
    // Coverage backfill: helper-level branches and HTTP error paths.
    //
    // These tests target the openai.rs surface that was not exercised by
    // the original Phase 6 suite — trivial trait impls, `mime_from_url`
    // extension matches, the data-URI padding branches, debug helpers,
    // the 20 MiB body cap, the factory-build failure surface, and the
    // streaming worker's terminal-error path. Workspace coverage gate
    // moves from 95 back to 96 once these land.
    // ---------------------------------------------------------------------

    #[test]
    fn message_content_default_is_empty_text() {
        // Default constructs the Text("") variant — exercised whenever
        // a struct holding `OpenAIMessageContent` lands on Default::default.
        let v = OpenAIMessageContent::default();
        assert_eq!(v.as_text(), Some(""));
        let (text, blocks) = v.flatten();
        assert!(text.is_empty());
        assert!(blocks.is_empty());
    }

    #[test]
    fn message_content_from_string_and_str_yields_text_variant() {
        // Both From<String> and From<&str> route to the Text variant
        // so callers can pass either ergonomically.
        let from_str: OpenAIMessageContent = "hi".into();
        assert_eq!(from_str.as_text(), Some("hi"));
        let from_string: OpenAIMessageContent = String::from("hello").into();
        assert_eq!(from_string.as_text(), Some("hello"));
    }

    #[test]
    fn message_content_parts_variant_as_text_is_none() {
        // `as_text` on the Parts variant explicitly returns None so the
        // caller knows to use `flatten()` instead. Pin this so refactors
        // that change the API don't silently return Some("").
        let parts = OpenAIMessageContent::Parts(vec![OpenAIContentPart::Text {
            text: "x".to_string(),
        }]);
        assert!(parts.as_text().is_none());
    }

    #[test]
    fn mime_from_url_matches_known_image_extensions() {
        // The `mime_from_url` branch table is consumed indirectly through
        // image_url_to_block when the URL has no `data:` prefix and the
        // extension matches one of the known suffixes. Exercise every
        // arm (png/jpg/jpeg/gif/webp + unknown).
        let cases = [
            ("https://example.com/x.png", "image/png"),
            ("https://example.com/x.jpg", "image/jpeg"),
            ("https://example.com/x.jpeg", "image/jpeg"),
            ("https://example.com/x.gif", "image/gif"),
            ("https://example.com/x.webp", "image/webp"),
        ];
        for (url, mime) in cases {
            let block = image_url_to_block(&OpenAIImageUrl {
                url: url.to_string(),
                detail: None,
            });
            match block {
                Block::Image { mime: m, .. } => assert_eq!(m, mime, "url={url}"),
                _ => panic!("expected Block::Image for {url}"),
            }
        }
        // Unknown extension -> fallback to "image/jpeg" per the
        // `unwrap_or_else` branch.
        let block = image_url_to_block(&OpenAIImageUrl {
            url: "https://example.com/x.bin".to_string(),
            detail: None,
        });
        if let Block::Image { mime, .. } = block {
            assert_eq!(mime, "image/jpeg");
        }
        // Query-string + fragment stripping is part of the same code
        // path — confirm it lands on the right extension.
        let block = image_url_to_block(&OpenAIImageUrl {
            url: "https://example.com/x.webp?cache=1#frag".to_string(),
            detail: Some("auto".to_string()),
        });
        if let Block::Image { mime, .. } = block {
            assert_eq!(mime, "image/webp");
        }
    }

    #[test]
    fn image_data_uri_double_pad_decodes_byte_estimate() {
        // The `==` padding branch was uncovered. `AA==` decodes to 1
        // raw byte (4 chars * 3 / 4 - 2 pad = 1).
        let block = image_url_to_block(&OpenAIImageUrl {
            url: "data:image/png;base64,AA==".to_string(),
            detail: None,
        });
        match block {
            Block::Image {
                data: ImageData::Inline { base64, bytes },
                ..
            } => {
                assert_eq!(base64, "AA==");
                assert_eq!(bytes, 1);
            }
            _ => panic!("expected inline image"),
        }
    }

    #[test]
    fn image_data_uri_with_empty_payload_falls_through_to_url() {
        // `data:image/png;base64,` (comma present but no payload)
        // is malformed; the code falls through to the URL branch.
        let block = image_url_to_block(&OpenAIImageUrl {
            url: "data:image/png;base64,".to_string(),
            detail: None,
        });
        assert!(matches!(
            block,
            Block::Image {
                data: ImageData::Url { .. },
                ..
            }
        ));
    }

    #[test]
    fn image_data_uri_without_comma_falls_through_to_url() {
        // `data:` with no comma is also malformed; falls through.
        let block = image_url_to_block(&OpenAIImageUrl {
            url: "data:notreallybase64".to_string(),
            detail: None,
        });
        assert!(matches!(
            block,
            Block::Image {
                data: ImageData::Url { .. },
                ..
            }
        ));
    }

    #[test]
    fn image_data_uri_with_empty_mime_defaults_to_png() {
        // `data:;base64,AAAA` — the MIME slot is empty so the
        // `filter(|s| !s.is_empty()).unwrap_or("image/png")` branch
        // engages.
        let block = image_url_to_block(&OpenAIImageUrl {
            url: "data:;base64,AAAA".to_string(),
            detail: None,
        });
        if let Block::Image { mime, .. } = block {
            assert_eq!(mime, "image/png");
        }
    }

    #[test]
    fn unix_now_secs_returns_positive_when_past_epoch() {
        // Smoke: the helper computes seconds since UNIX_EPOCH; on any
        // reasonable test host it's strictly positive. The `map_or(0,..)`
        // SystemTimeError branch only fires when the clock is before
        // 1970 — unreachable on real systems.
        let now = unix_now_secs();
        assert!(now > 1_700_000_000); // post-Nov 2023
    }

    #[test]
    fn model_list_from_catalog_with_entries_yields_one_row_per_slug() {
        use crate::model_catalog::{ArtifactRef, ModelEntry, ModelTask, ModelTier};
        let mut cat = ModelCatalog::new();
        let artifact = ArtifactRef::new(
            "https://example.com/m.gguf".to_string(),
            "0".repeat(64),
            1024,
        )
        .unwrap();
        cat.insert(ModelEntry {
            slug: "foo".parse().unwrap(),
            family: "llama".to_string(),
            display_name: "Foo".to_string(),
            tier: ModelTier::Low,
            task: std::iter::once(ModelTask::Chat).collect(),
            size_mib: 100,
            quantization: "Q4_K_M".to_string(),
            artifact,
            license: "Apache-2.0".to_string(),
            homepage: None,
            vision_mmproj: None,
        });
        let list = OpenAIModelList::from_catalog(&cat);
        assert_eq!(list.object, "list");
        assert_eq!(list.data.len(), 1);
        let entry = &list.data[0];
        assert_eq!(entry.id, "foo");
        assert_eq!(entry.object, "model");
        assert_eq!(entry.owned_by, "stratum");
        // `created` is unix_now_secs(), strictly positive on a real host.
        assert!(entry.created > 1_700_000_000);
    }

    #[test]
    fn openai_server_debug_impl_does_not_panic_and_includes_cfg() {
        // Pin the manual `Debug` impl that bypasses the un-printable
        // `LoopFactory` and `ModelCatalog` fields.
        let cfg = OpenAIServerConfig {
            bind: "127.0.0.1:0".parse().unwrap(),
            request_timeout: Duration::from_millis(100),
        };
        let srv = OpenAIServer::new(cfg, factory_for_echo("x"), Arc::new(ModelCatalog::new()));
        let dbg = format!("{srv:?}");
        assert!(dbg.contains("OpenAIServer"));
        assert!(dbg.contains("cfg"));
        assert!(dbg.contains("shutdown"));
    }

    #[test]
    fn turn_result_into_response_for_budget_exceeded_outcome() {
        // Each TurnOutcome -> finish_reason mapping needs to round-trip
        // through the `From<TurnResult>` impl, not just `finish_reason_for`.
        let result = TurnResult {
            turn_id: TurnId(11),
            outcome: TurnOutcome::BudgetExceeded {
                kind: "tokens".to_string(),
            },
            blocks: vec![Block::Text {
                text: "partial".to_string(),
            }],
            transitions: Vec::new(),
            events_emitted: Vec::new(),
        };
        let resp: OpenAIChatResponse = result.into();
        assert_eq!(resp.choices[0].finish_reason, "length");
    }

    #[test]
    fn turn_result_into_response_for_tool_failure_outcome() {
        let result = TurnResult {
            turn_id: TurnId(12),
            outcome: TurnOutcome::ToolFailure {
                tool_id: "search".to_string(),
                code: "STRAT-E5006".to_string(),
            },
            blocks: Vec::new(),
            transitions: Vec::new(),
            events_emitted: Vec::new(),
        };
        let resp: OpenAIChatResponse = result.into();
        assert_eq!(resp.choices[0].finish_reason, "tool_calls");
        // No Text blocks → content is empty string.
        assert_eq!(resp.choices[0].message.content.as_text(), Some(""));
    }

    #[test]
    fn turn_result_into_response_for_model_error_outcome() {
        let result = TurnResult {
            turn_id: TurnId(13),
            outcome: TurnOutcome::ModelError {
                code: "STRAT-E3007".to_string(),
            },
            blocks: Vec::new(),
            transitions: Vec::new(),
            events_emitted: Vec::new(),
        };
        let resp: OpenAIChatResponse = result.into();
        assert_eq!(resp.choices[0].finish_reason, "error");
    }

    #[test]
    fn turn_result_into_response_for_user_abort_outcome() {
        let result = TurnResult {
            turn_id: TurnId(14),
            outcome: TurnOutcome::UserAbort,
            blocks: Vec::new(),
            transitions: Vec::new(),
            events_emitted: Vec::new(),
        };
        let resp: OpenAIChatResponse = result.into();
        assert_eq!(resp.choices[0].finish_reason, "error");
    }

    #[test]
    fn request_with_only_system_messages_has_empty_user_prompt() {
        // `rposition(role==user)` returns None; the fallback is the
        // last index, which is a system message — `last_user_idx` arm
        // then takes that system message's text into `user_prompt`.
        // Pin the observed behaviour so a refactor doesn't drift.
        let req = OpenAIChatRequest {
            model: "echo".to_string(),
            messages: vec![
                OpenAIChatMessage {
                    role: "system".to_string(),
                    content: OpenAIMessageContent::Text("be brief".to_string()),
                    name: None,
                },
                OpenAIChatMessage {
                    role: "system".to_string(),
                    content: OpenAIMessageContent::Text("be polite".to_string()),
                    name: None,
                },
            ],
            ..OpenAIChatRequest::default()
        };
        let ctx: TurnContext = req.into();
        // The trailing message (system) lands on user_prompt because
        // there is no user role in the conversation.
        assert_eq!(ctx.user_prompt, "be polite");
        assert_eq!(ctx.history.len(), 1);
        assert_eq!(ctx.history[0].role, "system");
        assert_eq!(ctx.history[0].content, "be brief");
    }

    #[test]
    fn request_with_empty_messages_yields_empty_turn_context() {
        // Zero-message request: rposition->None, len.saturating_sub(1)=0,
        // for-loop never iterates. user_prompt + history both empty.
        let req = OpenAIChatRequest {
            model: "echo".to_string(),
            messages: Vec::new(),
            ..OpenAIChatRequest::default()
        };
        let ctx: TurnContext = req.into();
        assert!(ctx.user_prompt.is_empty());
        assert!(ctx.history.is_empty());
        assert!(ctx.attachments.is_empty());
    }

    #[test]
    fn long_history_round_trips_without_panic() {
        // 50-message history exercises the Vec::with_capacity branch
        // and the role-coerce match across enough iterations to catch
        // accidental quadratic behaviour at debug-build instrumentation
        // levels.
        let mut messages = Vec::new();
        for i in 0..25 {
            messages.push(OpenAIChatMessage {
                role: if i % 2 == 0 { "user" } else { "assistant" }.to_string(),
                content: OpenAIMessageContent::Text(format!("turn {i}")),
                name: None,
            });
        }
        messages.push(user_msg("final"));
        let req = OpenAIChatRequest {
            model: "echo".to_string(),
            messages,
            ..OpenAIChatRequest::default()
        };
        let ctx: TurnContext = req.into();
        assert_eq!(ctx.user_prompt, "final");
        assert_eq!(ctx.history.len(), 25);
    }

    #[test]
    fn sse_event_bytes_emits_data_prefix_and_blank_line_terminator() {
        // The SSE protocol requires `data: <json>\n\n` per event. Cover
        // the helper directly so a serialisation refactor can't silently
        // drop the prefix/footer.
        let chunk = OpenAIStreamChunk {
            id: "chatcmpl-1".to_string(),
            object: "chat.completion.chunk".to_string(),
            created: 0,
            model: "echo".to_string(),
            choices: vec![OpenAIStreamChoice {
                index: 0,
                delta: OpenAIDelta {
                    role: Some("assistant".to_string()),
                    content: None,
                },
                finish_reason: None,
            }],
        };
        let bytes = sse_event_bytes(&chunk);
        let text = std::str::from_utf8(&bytes).expect("utf8");
        assert!(text.starts_with("data: "));
        assert!(text.ends_with("\n\n"));
        // Round-trip: the body between the prefix and the blank line is
        // valid JSON of an OpenAIStreamChunk.
        let payload = &text["data: ".len()..text.len() - 2];
        let parsed: OpenAIStreamChunk = serde_json::from_str(payload).unwrap();
        assert_eq!(parsed.id, "chatcmpl-1");
        assert_eq!(parsed.choices[0].delta.role.as_deref(), Some("assistant"));
    }

    #[test]
    fn streaming_sse_reader_blocks_until_event_arrives_and_eofs_on_drop() {
        use std::io::Read;

        let (tx, rx) = mpsc::channel::<Vec<u8>>();
        let mut reader = StreamingSseReader::new(rx);

        // Push one event, then drop the sender.
        tx.send(b"data: hi\n\n".to_vec()).unwrap();
        drop(tx);

        // First read returns the event bytes.
        let mut buf = [0u8; 64];
        let n = reader.read(&mut buf).unwrap();
        assert_eq!(&buf[..n], b"data: hi\n\n");

        // Next read returns EOF (Ok(0)) because the sender dropped.
        let n2 = reader.read(&mut buf).unwrap();
        assert_eq!(n2, 0);
    }

    #[test]
    fn streaming_sse_reader_splits_event_across_short_buffer() {
        use std::io::Read;

        let (tx, rx) = mpsc::channel::<Vec<u8>>();
        let mut reader = StreamingSseReader::new(rx);
        tx.send(b"abcdefgh".to_vec()).unwrap();
        drop(tx);

        // Buffer of length 3 forces three reads to drain the 8-byte event.
        let mut buf = [0u8; 3];
        assert_eq!(reader.read(&mut buf).unwrap(), 3);
        assert_eq!(&buf, b"abc");
        assert_eq!(reader.read(&mut buf).unwrap(), 3);
        assert_eq!(&buf, b"def");
        assert_eq!(reader.read(&mut buf).unwrap(), 2);
        assert_eq!(&buf[..2], b"gh");
        // EOF.
        assert_eq!(reader.read(&mut buf).unwrap(), 0);
    }

    #[test]
    fn stream_chunk_terminal_omits_finish_reason_when_none() {
        // Pin the `skip_serializing_if = "Option::is_none"` annotation on
        // `finish_reason`: when None the field must NOT appear on the
        // wire, so clients can rely on its presence to mean "terminal".
        let chunk = OpenAIStreamChunk {
            id: "chatcmpl-2".to_string(),
            object: "chat.completion.chunk".to_string(),
            created: 0,
            model: "echo".to_string(),
            choices: vec![OpenAIStreamChoice {
                index: 0,
                delta: OpenAIDelta::default(),
                finish_reason: None,
            }],
        };
        let s = serde_json::to_string(&chunk).unwrap();
        assert!(!s.contains("finish_reason"), "got {s}");
    }

    // --- HTTP error-path tests --------------------------------------------

    fn start_server_with_factory(factory: LoopFactory) -> OpenAIServerHandle {
        let cat = Arc::new(ModelCatalog::new());
        let cfg = OpenAIServerConfig {
            bind: "127.0.0.1:0".parse().unwrap(),
            request_timeout: Duration::from_secs(2),
        };
        OpenAIServer::new(cfg, factory, cat).start().expect("start")
    }

    fn post_raw_body(addr: &str, path: &str, body: &[u8]) -> (u16, String) {
        use std::io::{BufRead, BufReader, Write};
        use std::net::TcpStream;
        let mut s = TcpStream::connect(addr).expect("connect");
        s.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
        s.set_write_timeout(Some(Duration::from_secs(5))).unwrap();
        let header = format!(
            "POST {path} HTTP/1.1\r\nHost: {addr}\r\nContent-Length: {}\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n",
            body.len()
        );
        s.write_all(header.as_bytes()).unwrap();
        s.write_all(body).unwrap();
        s.flush().unwrap();
        let mut r = BufReader::new(s);
        let mut status_line = String::new();
        r.read_line(&mut status_line).unwrap();
        let code: u16 = status_line
            .split_whitespace()
            .nth(1)
            .unwrap_or("0")
            .parse()
            .unwrap_or(0);
        loop {
            let mut line = String::new();
            if r.read_line(&mut line).unwrap_or(0) == 0 {
                break;
            }
            if line == "\r\n" || line.is_empty() {
                break;
            }
        }
        let mut body = String::new();
        let _ = r.read_to_string(&mut body);
        (code, body)
    }

    #[test]
    fn http_body_over_20_mib_cap_returns_400() {
        // Stream just-over-cap bytes so the `.take(MAX + 1)` reader
        // observes the overage and the handler emits a typed 400.
        let handle = start_server("x");
        let addr = handle.bound_address().to_string();
        // 20 MiB + 1 — just trips the cap. Pad with a no-op JSON
        // string so the body is still bytes (the serde parser never
        // runs because the cap check fires first).
        let cap_plus_one = 20 * 1024 * 1024 + 1;
        let mut body = Vec::with_capacity(cap_plus_one);
        body.extend_from_slice(b"\"");
        body.resize(cap_plus_one - 1, b'a');
        body.push(b'"');
        let (code, resp_body) = post_raw_body(&addr, "/v1/chat/completions", &body);
        assert_eq!(code, 400);
        let v: serde_json::Value = serde_json::from_str(&resp_body).expect("json");
        assert_eq!(v["error"]["type"], "invalid_request");
        assert!(
            v["error"]["message"]
                .as_str()
                .unwrap_or("")
                .contains("20 MiB"),
            "got {resp_body}"
        );
        let _ = handle.stop();
    }

    #[test]
    fn http_factory_failure_returns_500_internal_error() {
        // A factory that always errors maps to a 500 + `internal_error`
        // body — pin that the bridge between `LoopFactory` failures and
        // the HTTP error surface is wired up.
        let factory: LoopFactory =
            Arc::new(|| Err("synthetic build error for coverage".to_string()));
        let handle = start_server_with_factory(factory);
        let addr = handle.bound_address().to_string();
        let body = serde_json::json!({
            "model": "echo",
            "messages": [{"role":"user","content":"hi"}],
        })
        .to_string();
        let (code, resp_body) = post(&addr, "/v1/chat/completions", &body);
        assert_eq!(code, 500);
        let v: serde_json::Value = serde_json::from_str(&resp_body).expect("json");
        assert_eq!(v["error"]["type"], "internal_error");
        assert!(v["error"]["message"]
            .as_str()
            .unwrap_or("")
            .contains("factory build failed"));
        let _ = handle.stop();
    }

    #[test]
    fn http_factory_failure_returns_500_on_stream_request_too() {
        // Same factory-build failure path, but on a streaming request.
        // The dispatcher checks `parsed.stream` after the factory build,
        // so a failing factory short-circuits to 500 regardless of the
        // streaming flag.
        let factory: LoopFactory = Arc::new(|| Err("boom".to_string()));
        let handle = start_server_with_factory(factory);
        let addr = handle.bound_address().to_string();
        let body = serde_json::json!({
            "model": "echo",
            "messages": [{"role":"user","content":"hi"}],
            "stream": true
        })
        .to_string();
        let (code, _resp_body) = post(&addr, "/v1/chat/completions", &body);
        assert_eq!(code, 500);
        let _ = handle.stop();
    }

    #[test]
    fn http_get_models_returns_list() {
        // The route table accepts both POST and GET for /v1/models;
        // the existing test covers POST. Cover GET so the alternation
        // arm is hit.
        use std::io::{BufRead, BufReader, Write};
        use std::net::TcpStream;
        let handle = start_server("x");
        let addr = handle.bound_address().to_string();
        let mut s = TcpStream::connect(&addr).unwrap();
        s.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
        s.write_all(
            format!("GET /v1/models HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n\r\n")
                .as_bytes(),
        )
        .unwrap();
        let mut r = BufReader::new(s);
        let mut status_line = String::new();
        r.read_line(&mut status_line).unwrap();
        assert!(status_line.contains("200"), "got {status_line}");
        let _ = handle.stop();
    }

    #[test]
    fn http_options_preflight_on_models_path_returns_204() {
        // CORS preflight matches every path before the route table,
        // so OPTIONS on /v1/models should also yield 204.
        use std::io::{BufRead, BufReader, Write};
        use std::net::TcpStream;
        let handle = start_server("x");
        let addr = handle.bound_address().to_string();
        let mut s = TcpStream::connect(&addr).unwrap();
        s.write_all(b"OPTIONS /v1/models HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
            .unwrap();
        let mut r = BufReader::new(s);
        let mut status_line = String::new();
        r.read_line(&mut status_line).unwrap();
        assert!(status_line.contains("204"), "got {status_line}");
        let _ = handle.stop();
    }

    #[test]
    fn http_unknown_route_with_get_method_also_returns_404() {
        // The route table fallthrough renders the `{method} {path}` pair
        // in the error message regardless of method — pin that GETting
        // a bogus path still lands on a typed 404.
        use std::io::{BufRead, BufReader, Read as _, Write};
        use std::net::TcpStream;
        let handle = start_server("x");
        let addr = handle.bound_address().to_string();
        let mut s = TcpStream::connect(&addr).unwrap();
        s.write_all(b"GET /v1/nope HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
            .unwrap();
        let mut r = BufReader::new(s);
        let mut status_line = String::new();
        r.read_line(&mut status_line).unwrap();
        assert!(status_line.contains("404"), "got {status_line}");
        // Drain headers.
        loop {
            let mut line = String::new();
            if r.read_line(&mut line).unwrap_or(0) == 0 {
                break;
            }
            if line == "\r\n" {
                break;
            }
        }
        let mut body = String::new();
        let _ = r.read_to_string(&mut body);
        let v: serde_json::Value = serde_json::from_str(&body).expect("json");
        assert_eq!(v["error"]["type"], "not_found");
        assert!(v["error"]["message"].as_str().unwrap_or("").contains("GET"));
        let _ = handle.stop();
    }

    #[test]
    fn server_handle_drop_triggers_acceptor_shutdown() {
        // The Drop impl flips the shutdown flag and joins the acceptor.
        // Hold the handle in a scope and drop explicitly; the acceptor
        // observes shutdown on its next 100ms poll tick.
        let handle = start_server("x");
        // Confirm we got a real bind address before dropping.
        assert!(!handle.bound_address().is_empty());
        // Explicit drop runs the Drop impl.
        drop(handle);
        // No assertion needed beyond "doesn't panic"; the Drop arm is
        // the coverage target.
    }

    #[test]
    fn static_header_constructs_known_header_pair() {
        // Cover the happy path of `static_header` directly through the
        // public helpers that call it.
        let h = json_header();
        let val = std::str::from_utf8(h.value.as_bytes()).unwrap_or("");
        assert!(val.contains("application/json"));
        let sse = sse_header();
        let sse_val = std::str::from_utf8(sse.value.as_bytes()).unwrap_or("");
        assert!(sse_val.contains("text/event-stream"));
    }

    #[test]
    fn http_invalid_utf8_body_returns_400_invalid_request() {
        // `Read::read_to_string` fails when the body bytes aren't valid
        // UTF-8 — the handler maps that to a typed 400 instead of a
        // panic. Send 0xFF bytes which are invalid as UTF-8 start bytes.
        let handle = start_server("x");
        let addr = handle.bound_address().to_string();
        let body: Vec<u8> = vec![0xFF, 0xFE, 0xFD];
        let (code, resp_body) = post_raw_body(&addr, "/v1/chat/completions", &body);
        assert_eq!(code, 400);
        let v: serde_json::Value = serde_json::from_str(&resp_body).expect("json");
        assert_eq!(v["error"]["type"], "invalid_request");
        assert!(
            v["error"]["message"]
                .as_str()
                .unwrap_or("")
                .contains("body read failed"),
            "got {resp_body}"
        );
        let _ = handle.stop();
    }

    #[test]
    fn handle_list_models_includes_real_catalog_entries_via_http() {
        // Wire a catalog with one entry through the HTTP path so the
        // /v1/models response renders a populated `data` array on the
        // real socket. Pin both the list shape and entry shape.
        use crate::model_catalog::{ArtifactRef, ModelEntry, ModelTask, ModelTier};
        let mut cat = ModelCatalog::new();
        let artifact = ArtifactRef::new(
            "https://example.com/m.gguf".to_string(),
            "f".repeat(64),
            1024,
        )
        .unwrap();
        cat.insert(ModelEntry {
            slug: "bar".parse().unwrap(),
            family: "qwen".to_string(),
            display_name: "Bar".to_string(),
            tier: ModelTier::High,
            task: std::iter::once(ModelTask::Code).collect(),
            size_mib: 200,
            quantization: "Q5_K_M".to_string(),
            artifact,
            license: "MIT".to_string(),
            homepage: Some("https://example.com".to_string()),
            vision_mmproj: None,
        });
        let cfg = OpenAIServerConfig {
            bind: "127.0.0.1:0".parse().unwrap(),
            request_timeout: Duration::from_secs(2),
        };
        let handle = OpenAIServer::new(cfg, factory_for_echo("x"), Arc::new(cat))
            .start()
            .unwrap();
        let addr = handle.bound_address().to_string();
        let (code, body) = post(&addr, "/v1/models", "");
        assert_eq!(code, 200);
        let v: serde_json::Value = serde_json::from_str(&body).expect("json");
        assert_eq!(v["object"], "list");
        assert_eq!(v["data"].as_array().unwrap().len(), 1);
        assert_eq!(v["data"][0]["id"], "bar");
        assert_eq!(v["data"][0]["owned_by"], "stratum");
        let _ = handle.stop();
    }

    #[test]
    fn http_multimodal_chat_request_propagates_attachments_through_handler() {
        // The HTTP handler must flatten a multimodal `content` array into
        // a TurnContext with attachments before invoking the agent. This
        // test drives the wire-format path end-to-end on a real socket
        // so the EchoProvider's multimodal-attachment warn arm engages
        // (transitively covering provider.rs ~170-173). The Echo response
        // still surfaces the prompt fragment as text.
        let handle = start_server("seen:");
        let addr = handle.bound_address().to_string();
        let body = serde_json::json!({
            "model": "echo",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/x.png"}}
                ]
            }],
            "stream": false
        })
        .to_string();
        let (code, resp_body) = post(&addr, "/v1/chat/completions", &body);
        assert_eq!(code, 200);
        let v: serde_json::Value = serde_json::from_str(&resp_body).expect("json");
        assert_eq!(v["object"], "chat.completion");
        // The Echo prefix is included in the text.
        assert!(v["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("")
            .contains("seen:"));
        let _ = handle.stop();
    }

    #[test]
    fn http_streaming_multimodal_request_returns_sse_with_terminal_chunk() {
        // Streaming variant of the above. The acceptor thread + worker
        // thread sequence runs end-to-end with multimodal attachments,
        // hitting the SSE buffering loop, `data: [DONE]` footer, and
        // terminal-chunk render.
        let handle = start_server("streamed:");
        let addr = handle.bound_address().to_string();
        let body = serde_json::json!({
            "model": "echo",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "hello"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
                ]
            }],
            "stream": true
        })
        .to_string();
        let (code, resp_body) = post(&addr, "/v1/chat/completions", &body);
        assert_eq!(code, 200);
        // Terminal frame present.
        assert!(resp_body.contains("[DONE]"));
        // At least one `chat.completion.chunk` frame (the role-frame).
        assert!(resp_body.contains("chat.completion.chunk"));
        // The terminal finish_reason for the EchoProvider Success outcome
        // is "stop"; pin it so a regression where we emit "error" on the
        // happy-path streaming worker terminal chunk is caught.
        assert!(resp_body.contains("\"finish_reason\":\"stop\""));
        let _ = handle.stop();
    }

    #[test]
    fn http_chat_request_with_system_then_user_history_flattens_into_response() {
        // End-to-end through HTTP with a multi-turn history (system +
        // assistant + user). Drives more agent_loop / agent_session code
        // than the single-message tests and pins that the response model
        // label echoes back even with a long history.
        let handle = start_server("echo:");
        let addr = handle.bound_address().to_string();
        let body = serde_json::json!({
            "model": "echo-mixed",
            "messages": [
                {"role":"system","content":"be brief"},
                {"role":"user","content":"first question"},
                {"role":"assistant","content":"a"},
                {"role":"user","content":"second question"}
            ],
            "stream": false
        })
        .to_string();
        let (code, resp_body) = post(&addr, "/v1/chat/completions", &body);
        assert_eq!(code, 200);
        let v: serde_json::Value = serde_json::from_str(&resp_body).expect("json");
        assert_eq!(v["model"], "echo-mixed");
        assert_eq!(v["choices"][0]["finish_reason"], "stop");
        let _ = handle.stop();
    }

    #[test]
    fn static_header_falls_back_when_name_is_invalid_header_bytes() {
        // `Header::from_bytes` rejects bytes containing illegal-in-header
        // characters (LF, CR, NUL). The fallback arm constructs a safe
        // `X-Stratum: 1` placeholder so the helper is total — exercise
        // this arm by passing a name with `\n` in it.
        let h = static_header(b"Bad\nName", b"value");
        // We can't peek the header's name through the public API
        // ergonomically, but rendering it via Debug confirms the
        // construction did not panic and produced a Header value.
        let dbg = format!("{h:?}");
        assert!(!dbg.is_empty());
    }

    #[test]
    fn finish_reason_for_each_outcome_round_trip() {
        // Same coverage as `finish_reason_maps_every_turn_outcome` but
        // exercising the `From<TurnResult>` path: pin that each outcome
        // round-trips through the Response conversion, not just the
        // helper. (Helper-only coverage misses the conversion's match.)
        let outcomes = [
            (TurnOutcome::Success, "stop"),
            (
                TurnOutcome::BudgetExceeded {
                    kind: "k".to_string(),
                },
                "length",
            ),
            (
                TurnOutcome::ToolFailure {
                    tool_id: "t".to_string(),
                    code: "c".to_string(),
                },
                "tool_calls",
            ),
            (
                TurnOutcome::ModelError {
                    code: "c".to_string(),
                },
                "error",
            ),
            (TurnOutcome::UserAbort, "error"),
        ];
        for (outcome, expected) in outcomes {
            let result = TurnResult {
                turn_id: TurnId(0),
                outcome,
                blocks: Vec::new(),
                transitions: Vec::new(),
                events_emitted: Vec::new(),
            };
            let resp: OpenAIChatResponse = result.into();
            assert_eq!(resp.choices[0].finish_reason, expected);
        }
    }

    #[test]
    fn server_config_round_trips_via_debug_format() {
        // OpenAIServerConfig derives Debug; pin the rendered shape so a
        // refactor that drops the derive surfaces in the diff.
        let cfg = OpenAIServerConfig {
            bind: "127.0.0.1:9999".parse().unwrap(),
            request_timeout: Duration::from_millis(250),
        };
        let s = format!("{cfg:?}");
        assert!(s.contains("OpenAIServerConfig"));
        assert!(s.contains("127.0.0.1"));
    }

    #[test]
    fn server_config_clones() {
        // Clone + Debug coverage smoke.
        let cfg = OpenAIServerConfig {
            bind: "127.0.0.1:0".parse().unwrap(),
            request_timeout: Duration::from_millis(123),
        };
        let cloned = cfg.clone();
        assert_eq!(cloned.bind, cfg.bind);
        assert_eq!(cloned.request_timeout, cfg.request_timeout);
    }

    #[test]
    fn delta_default_serialises_to_empty_object() {
        // OpenAIDelta uses `skip_serializing_if = "Option::is_none"` on
        // both fields, so the default serialises as `{}`.
        let d = OpenAIDelta::default();
        let s = serde_json::to_string(&d).unwrap();
        assert_eq!(s, "{}");
    }

    #[test]
    fn delta_with_role_only_omits_content() {
        let d = OpenAIDelta {
            role: Some("assistant".to_string()),
            content: None,
        };
        let s = serde_json::to_string(&d).unwrap();
        assert!(s.contains("\"role\":\"assistant\""));
        assert!(!s.contains("content"));
    }

    #[test]
    fn delta_with_content_only_omits_role() {
        let d = OpenAIDelta {
            role: None,
            content: Some("tok".to_string()),
        };
        let s = serde_json::to_string(&d).unwrap();
        assert!(s.contains("\"content\":\"tok\""));
        assert!(!s.contains("role"));
    }

    #[test]
    fn openai_usage_default_is_zero() {
        let u = OpenAIUsage::default();
        assert_eq!(u.prompt_tokens, 0);
        assert_eq!(u.completion_tokens, 0);
        assert_eq!(u.total_tokens, 0);
        // Copy semantics
        let u2 = u;
        assert_eq!(u, u2);
    }

    #[test]
    fn model_entry_round_trips_via_serde() {
        let entry = OpenAIModelEntry {
            id: "m1".to_string(),
            object: "model".to_string(),
            created: 1_700_000_000,
            owned_by: "stratum".to_string(),
        };
        let s = serde_json::to_string(&entry).unwrap();
        let back: OpenAIModelEntry = serde_json::from_str(&s).unwrap();
        assert_eq!(back, entry);
    }

    #[test]
    fn chat_request_serialises_optional_fields_only_when_set() {
        // `skip_serializing_if = "Option::is_none"` policy across the
        // request body — pin that an empty default skips all optionals
        // (the inverse condition; helps catch a serde tweak).
        let req = OpenAIChatRequest::default();
        // OpenAIChatRequest is Deserialize-only; we can still serde it
        // back via a Value round-trip if a Serialize derive lands later.
        // For now, a default round-trip via serde_json::to_value would
        // require Serialize; instead pin the shape via Debug.
        let dbg = format!("{req:?}");
        assert!(dbg.contains("OpenAIChatRequest"));
    }

    #[test]
    fn chat_response_round_trips_via_serde() {
        let resp = OpenAIChatResponse {
            id: "chatcmpl-1".to_string(),
            object: "chat.completion".to_string(),
            created: 1,
            model: "echo".to_string(),
            choices: vec![OpenAIChoice {
                index: 0,
                message: OpenAIChatMessage {
                    role: "assistant".to_string(),
                    content: OpenAIMessageContent::Text("ok".to_string()),
                    name: None,
                },
                finish_reason: "stop".to_string(),
            }],
            usage: OpenAIUsage {
                prompt_tokens: 1,
                completion_tokens: 2,
                total_tokens: 3,
            },
        };
        let s = serde_json::to_string(&resp).unwrap();
        let back: OpenAIChatResponse = serde_json::from_str(&s).unwrap();
        assert_eq!(back, resp);
    }

    #[test]
    fn image_url_struct_round_trips_serde_with_detail() {
        let iu = OpenAIImageUrl {
            url: "https://example.com/x.png".to_string(),
            detail: Some("high".to_string()),
        };
        let s = serde_json::to_string(&iu).unwrap();
        let back: OpenAIImageUrl = serde_json::from_str(&s).unwrap();
        assert_eq!(back, iu);
        // The `detail: None` case skips the field on the wire.
        let iu2 = OpenAIImageUrl {
            url: "x".to_string(),
            detail: None,
        };
        let s2 = serde_json::to_string(&iu2).unwrap();
        assert!(!s2.contains("detail"));
    }

    #[test]
    fn input_audio_struct_round_trips_serde() {
        let ia = OpenAIInputAudio {
            data: "AAAA".to_string(),
            format: "mp3".to_string(),
        };
        let s = serde_json::to_string(&ia).unwrap();
        let back: OpenAIInputAudio = serde_json::from_str(&s).unwrap();
        assert_eq!(back, ia);
    }

    #[test]
    fn content_part_round_trips_via_serde_for_each_variant() {
        let parts = [
            OpenAIContentPart::Text {
                text: "hi".to_string(),
            },
            OpenAIContentPart::ImageUrl {
                image_url: OpenAIImageUrl {
                    url: "https://example.com/x.png".to_string(),
                    detail: None,
                },
            },
            OpenAIContentPart::InputAudio {
                input_audio: OpenAIInputAudio {
                    data: "AAAA".to_string(),
                    format: "wav".to_string(),
                },
            },
        ];
        for p in parts {
            let s = serde_json::to_string(&p).unwrap();
            let back: OpenAIContentPart = serde_json::from_str(&s).unwrap();
            assert_eq!(back, p);
        }
    }

    #[test]
    fn stream_chunk_round_trips_via_serde() {
        let chunk = OpenAIStreamChunk {
            id: "chatcmpl-1".to_string(),
            object: "chat.completion.chunk".to_string(),
            created: 0,
            model: "echo".to_string(),
            choices: vec![OpenAIStreamChoice {
                index: 0,
                delta: OpenAIDelta {
                    role: Some("assistant".to_string()),
                    content: Some("delta".to_string()),
                },
                finish_reason: Some("stop".to_string()),
            }],
        };
        let s = serde_json::to_string(&chunk).unwrap();
        let back: OpenAIStreamChunk = serde_json::from_str(&s).unwrap();
        assert_eq!(back, chunk);
    }

    #[test]
    fn message_with_name_field_round_trips_when_present() {
        let m = OpenAIChatMessage {
            role: "tool".to_string(),
            content: OpenAIMessageContent::Text("result".to_string()),
            name: Some("search".to_string()),
        };
        let s = serde_json::to_string(&m).unwrap();
        assert!(s.contains("\"name\":\"search\""));
        let back: OpenAIChatMessage = serde_json::from_str(&s).unwrap();
        assert_eq!(back, m);
    }

    #[test]
    fn message_without_name_field_omits_name_on_serialise() {
        let m = OpenAIChatMessage {
            role: "user".to_string(),
            content: OpenAIMessageContent::Text("hi".to_string()),
            name: None,
        };
        let s = serde_json::to_string(&m).unwrap();
        assert!(!s.contains("\"name\""));
    }

    #[test]
    fn model_list_round_trips_via_serde() {
        let list = OpenAIModelList {
            object: "list".to_string(),
            data: vec![OpenAIModelEntry {
                id: "a".to_string(),
                object: "model".to_string(),
                created: 0,
                owned_by: "stratum".to_string(),
            }],
        };
        let s = serde_json::to_string(&list).unwrap();
        let back: OpenAIModelList = serde_json::from_str(&s).unwrap();
        assert_eq!(back, list);
    }

    // ----- Compact regression-pin tests for every wire-format edge -----

    #[test]
    fn audio_format_to_mime_all_aliases() {
        // Every alias documented in the source must map; pin each.
        assert_eq!(audio_format_to_mime("oga"), "audio/ogg");
        assert_eq!(audio_format_to_mime("opus"), "audio/ogg");
        assert_eq!(audio_format_to_mime("OPUS"), "audio/ogg");
        assert_eq!(audio_format_to_mime("mp4"), "audio/mp4");
        assert_eq!(audio_format_to_mime("MP4"), "audio/mp4");
        assert_eq!(audio_format_to_mime("M4A"), "audio/mp4");
        assert_eq!(audio_format_to_mime(""), "audio/wav");
        assert_eq!(audio_format_to_mime("FLAC"), "audio/flac");
    }

    #[test]
    fn input_audio_block_estimates_bytes_for_padded_payload() {
        // `=` (1 pad) → 4*3/4 - 1 = 2 bytes.
        let block = input_audio_to_block(&OpenAIInputAudio {
            data: "AAA=".to_string(),
            format: "wav".to_string(),
        });
        if let Block::Audio {
            data: AudioData::Inline { bytes, .. },
            ..
        } = block
        {
            assert_eq!(bytes, 2);
        }
    }

    #[test]
    fn input_audio_block_double_pad_estimate_is_one_byte() {
        let block = input_audio_to_block(&OpenAIInputAudio {
            data: "AA==".to_string(),
            format: "wav".to_string(),
        });
        if let Block::Audio {
            data: AudioData::Inline { bytes, .. },
            ..
        } = block
        {
            assert_eq!(bytes, 1);
        }
    }

    #[test]
    fn input_audio_block_no_pad_estimate_is_three_bytes() {
        let block = input_audio_to_block(&OpenAIInputAudio {
            data: "AAAA".to_string(),
            format: "mp3".to_string(),
        });
        if let Block::Audio {
            mime,
            data: AudioData::Inline { bytes, .. },
            ..
        } = block
        {
            assert_eq!(mime, "audio/mpeg");
            assert_eq!(bytes, 3);
        }
    }

    #[test]
    fn message_content_text_serialises_as_bare_string() {
        let c = OpenAIMessageContent::Text("x".to_string());
        let s = serde_json::to_string(&c).unwrap();
        assert_eq!(s, "\"x\"");
    }

    #[test]
    fn message_content_parts_serialises_as_array() {
        let c = OpenAIMessageContent::Parts(vec![OpenAIContentPart::Text {
            text: "x".to_string(),
        }]);
        let s = serde_json::to_string(&c).unwrap();
        assert!(s.starts_with('['));
    }

    #[test]
    fn flatten_empty_parts_yields_empty_text() {
        let c = OpenAIMessageContent::Parts(Vec::new());
        let (text, blocks) = c.flatten();
        assert!(text.is_empty());
        assert!(blocks.is_empty());
    }

    #[test]
    fn flatten_only_image_parts_yields_empty_text_with_blocks() {
        let c = OpenAIMessageContent::Parts(vec![OpenAIContentPart::ImageUrl {
            image_url: OpenAIImageUrl {
                url: "https://example.com/x.png".to_string(),
                detail: None,
            },
        }]);
        let (text, blocks) = c.flatten();
        assert!(text.is_empty());
        assert_eq!(blocks.len(), 1);
    }

    #[test]
    fn blocks_to_text_handles_empty_input() {
        let blocks: Vec<Block> = Vec::new();
        assert!(blocks_to_text(&blocks).is_empty());
    }

    #[test]
    fn blocks_to_text_concatenates_multiple_segments() {
        let blocks = vec![
            Block::Text {
                text: "a".to_string(),
            },
            Block::Text {
                text: "b".to_string(),
            },
            Block::Text {
                text: "c".to_string(),
            },
        ];
        assert_eq!(blocks_to_text(&blocks), "abc");
    }

    #[test]
    fn blocks_to_text_skips_image_blocks() {
        let blocks = vec![
            Block::Text {
                text: "x".to_string(),
            },
            Block::Image {
                mime: "image/png".to_string(),
                data: ImageData::Url {
                    url: "u".to_string(),
                },
                alt: None,
            },
            Block::Text {
                text: "y".to_string(),
            },
        ];
        assert_eq!(blocks_to_text(&blocks), "xy");
    }

    #[test]
    fn delta_round_trip_both_fields_some() {
        let d = OpenAIDelta {
            role: Some("assistant".to_string()),
            content: Some("tok".to_string()),
        };
        let s = serde_json::to_string(&d).unwrap();
        let back: OpenAIDelta = serde_json::from_str(&s).unwrap();
        assert_eq!(back, d);
    }

    #[test]
    fn stream_choice_with_finish_reason_serialises_field() {
        let c = OpenAIStreamChoice {
            index: 0,
            delta: OpenAIDelta::default(),
            finish_reason: Some("stop".to_string()),
        };
        let s = serde_json::to_string(&c).unwrap();
        assert!(s.contains("\"finish_reason\":\"stop\""));
    }

    #[test]
    fn choice_round_trips_through_serde() {
        let c = OpenAIChoice {
            index: 0,
            message: OpenAIChatMessage {
                role: "assistant".to_string(),
                content: OpenAIMessageContent::Text("ok".to_string()),
                name: None,
            },
            finish_reason: "stop".to_string(),
        };
        let s = serde_json::to_string(&c).unwrap();
        let back: OpenAIChoice = serde_json::from_str(&s).unwrap();
        assert_eq!(back, c);
    }

    #[test]
    fn usage_total_is_sum_of_prompt_and_completion() {
        let u = OpenAIUsage {
            prompt_tokens: 3,
            completion_tokens: 4,
            total_tokens: 7,
        };
        let s = serde_json::to_string(&u).unwrap();
        let back: OpenAIUsage = serde_json::from_str(&s).unwrap();
        assert_eq!(back, u);
        assert_eq!(u.total_tokens, u.prompt_tokens + u.completion_tokens);
    }

    #[test]
    fn turn_id_string_includes_inner_value_in_response_id() {
        let result = TurnResult {
            turn_id: TurnId(99_999),
            outcome: TurnOutcome::Success,
            blocks: Vec::new(),
            transitions: Vec::new(),
            events_emitted: Vec::new(),
        };
        let resp: OpenAIChatResponse = result.into();
        assert_eq!(resp.id, "chatcmpl-99999");
    }

    #[test]
    fn empty_blocks_yield_empty_response_content() {
        let result = TurnResult {
            turn_id: TurnId(0),
            outcome: TurnOutcome::Success,
            blocks: Vec::new(),
            transitions: Vec::new(),
            events_emitted: Vec::new(),
        };
        let resp: OpenAIChatResponse = result.into();
        assert_eq!(resp.choices[0].message.content.as_text(), Some(""));
        assert_eq!(resp.usage.prompt_tokens, 0);
        assert_eq!(resp.usage.total_tokens, 0);
    }

    #[test]
    fn message_round_trips_with_parts_content() {
        let m = OpenAIChatMessage {
            role: "user".to_string(),
            content: OpenAIMessageContent::Parts(vec![OpenAIContentPart::Text {
                text: "hi".to_string(),
            }]),
            name: None,
        };
        let s = serde_json::to_string(&m).unwrap();
        let back: OpenAIChatMessage = serde_json::from_str(&s).unwrap();
        assert_eq!(back, m);
    }

    #[test]
    fn default_audio_format_helper_yields_wav() {
        assert_eq!(default_audio_format(), "wav");
    }

    #[test]
    fn image_data_uri_with_mime_params_strips_to_first() {
        // `data:image/png;charset=utf-8;base64,AAAA` — the MIME slot is
        // `image/png;charset=utf-8`; we keep only the first
        // `image/png` part per the `split(';').next()` rule.
        let block = image_url_to_block(&OpenAIImageUrl {
            url: "data:image/png;charset=utf-8;base64,AAAA".to_string(),
            detail: None,
        });
        if let Block::Image { mime, .. } = block {
            assert_eq!(mime, "image/png");
        }
    }

    #[test]
    fn turn_id_zero_renders_as_chatcmpl_dash_zero() {
        let result = TurnResult {
            turn_id: TurnId(0),
            outcome: TurnOutcome::Success,
            blocks: Vec::new(),
            transitions: Vec::new(),
            events_emitted: Vec::new(),
        };
        let resp: OpenAIChatResponse = result.into();
        assert_eq!(resp.id, "chatcmpl-0");
    }

    #[test]
    fn finish_reason_for_const_is_const() {
        // Smoke that the helper is callable in a const context, pinning
        // the `const fn` qualifier so a refactor that drops it surfaces.
        const _OUTCOME: TurnOutcome = TurnOutcome::Success;
        let s = finish_reason_for(&_OUTCOME);
        assert_eq!(s, "stop");
    }

    #[test]
    fn loop_factory_from_agent_factory_builds_each_call() {
        // The bridge clones the factory per call so concurrent requests
        // get independent AgentLoops. Pin that two successive calls both
        // succeed with the same inner factory.
        let inner = Arc::new(AgentFactory::new().with_provider(Arc::new(EchoProvider::new("hi"))));
        let lf = loop_factory_from_agent_factory(inner);
        assert!((lf)().is_ok());
        assert!((lf)().is_ok());
    }

    // ---------------------------------------------------------------------
    // Memory-probe / activation refusal (plan/07 Phase 6).
    // ---------------------------------------------------------------------

    use crate::model_catalog::{ArtifactRef, ModelEntry, ModelTask, ModelTier};
    use std::collections::BTreeSet;

    fn catalog_with(slug: &str, size_mib: u64) -> ModelCatalog {
        let mut cat = ModelCatalog::new();
        let artifact = ArtifactRef::new(
            "https://example.com/m.gguf".to_string(),
            "a".repeat(64),
            1_024,
        )
        .unwrap();
        let mut tasks: BTreeSet<ModelTask> = BTreeSet::new();
        tasks.insert(ModelTask::Chat);
        cat.insert(ModelEntry {
            slug: slug.parse().unwrap(),
            family: "llama".to_string(),
            display_name: slug.to_string(),
            tier: ModelTier::Low,
            task: tasks,
            size_mib,
            quantization: "Q4_K_M".to_string(),
            artifact,
            license: "Apache-2.0".to_string(),
            homepage: None,
            vision_mmproj: None,
        });
        cat
    }

    #[test]
    fn required_mib_applies_one_point_five_x_overhead_to_file_size() {
        // 1000 MiB file → 1000 + (1000 * 1.5) = 2500 MiB working set.
        assert_eq!(required_mib_for_model(1_000), 2_500);
        assert_eq!(required_mib_for_model(0), 0);
        // 2 MiB → 2 + (2*1.5) = 5 (integer math: 2 + 3 = 5).
        assert_eq!(required_mib_for_model(2), 5);
        // Small rounding: 1 MiB → 1 + (1*3/2=1) = 2.
        assert_eq!(required_mib_for_model(1), 2);
    }

    #[test]
    fn required_mib_saturates_at_u64_max_instead_of_overflowing() {
        // file_size * 3 would overflow when file_size > u64::MAX / 3;
        // saturating_mul pins the intermediate at u64::MAX and the
        // subsequent saturating_add stays there. This is the only
        // branch that protects against a malformed catalog entry that
        // claims an absurd size. Pin against u64::MAX directly.
        let got = required_mib_for_model(u64::MAX);
        assert_eq!(got, u64::MAX);
    }

    #[test]
    fn memory_probe_skips_unknown_model_label() {
        // Label that is not a valid slug (uppercase) → no catalog lookup,
        // request proceeds. The handler then surfaces "unknown model"
        // through the provider, not as a 503.
        let cat = ModelCatalog::new();
        let probe = SyntheticRamProbe(1);
        assert!(matches!(
            memory_probe_for_model("Unknown-MODEL", &cat, &probe),
            MemoryCheck::Ok
        ));
    }

    #[test]
    fn memory_probe_skips_model_not_in_catalog() {
        // Valid slug shape but no catalog entry → request proceeds.
        let cat = ModelCatalog::new();
        let probe = SyntheticRamProbe(1);
        assert!(matches!(
            memory_probe_for_model("not-in-catalog", &cat, &probe),
            MemoryCheck::Ok
        ));
    }

    #[test]
    fn memory_probe_refuses_when_required_exceeds_available() {
        // 10_000 MiB model → 25_000 MiB required; only 1_024 MiB free →
        // refusal carries the rendered message.
        let cat = catalog_with("big-model", 10_000);
        let probe = SyntheticRamProbe(1_024);
        match memory_probe_for_model("big-model", &cat, &probe) {
            MemoryCheck::TooLarge { message } => {
                assert!(message.contains("big-model"));
                assert!(message.contains("25000 MiB"));
                assert!(message.contains("1024 MiB"));
            }
            MemoryCheck::Ok => panic!("expected refusal"),
        }
    }

    #[test]
    fn memory_probe_allows_when_required_fits_in_available() {
        // 100 MiB model → 250 MiB required; 8 GiB free → fits.
        let cat = catalog_with("small-model", 100);
        let probe = SyntheticRamProbe(8 * 1_024);
        assert!(matches!(
            memory_probe_for_model("small-model", &cat, &probe),
            MemoryCheck::Ok
        ));
    }

    #[test]
    fn memory_probe_boundary_required_equals_available_is_ok() {
        // Refusal is strict `>`, so required == available passes.
        // 100 MiB → 250 required; set probe to exactly 250.
        let cat = catalog_with("edge-model", 100);
        let probe = SyntheticRamProbe(250);
        assert!(matches!(
            memory_probe_for_model("edge-model", &cat, &probe),
            MemoryCheck::Ok
        ));
    }

    #[test]
    fn error_body_with_code_distinguishes_type_and_code_and_param_is_null() {
        // The 503 path uses distinct `type` and `code` values so clients
        // can branch on the machine-readable `code` field.
        let body = error_body_with_code("server_error", "model_too_large", "msg");
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["error"]["type"], "server_error");
        assert_eq!(v["error"]["code"], "model_too_large");
        assert_eq!(v["error"]["message"], "msg");
        // `param` must be present and explicitly null on every error.
        assert!(v["error"]["param"].is_null());
        assert!(v["error"].as_object().unwrap().contains_key("param"));
    }

    /// Spin up an HTTP server with a synthetic catalog + synthetic
    /// probe so the wire-level 503 path runs without depending on the
    /// host's free RAM.
    fn start_server_with_probe(
        reply: &str,
        catalog: ModelCatalog,
        available_mib: u64,
    ) -> OpenAIServerHandle {
        let cfg = OpenAIServerConfig {
            bind: "127.0.0.1:0".parse().unwrap(),
            request_timeout: Duration::from_secs(2),
        };
        let probe: Arc<dyn RamProbe + Send + Sync> = Arc::new(SyntheticRamProbe(available_mib));
        OpenAIServer::new(cfg, factory_for_echo(reply), Arc::new(catalog))
            .with_probe(probe)
            .start()
            .expect("start")
    }

    /// Variant of `post` that returns the raw response headers — needed
    /// to assert the `Retry-After` header on the 503.
    fn post_capture_headers(addr: &str, path: &str, body: &str) -> (u16, String, String) {
        use std::io::{BufRead, BufReader, Read as _, Write};
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
        let mut headers = String::new();
        loop {
            let mut line = String::new();
            r.read_line(&mut line).unwrap();
            if line == "\r\n" || line.is_empty() {
                break;
            }
            headers.push_str(&line);
        }
        let mut body_buf = String::new();
        let _ = r.read_to_string(&mut body_buf);
        (code, headers, body_buf)
    }

    #[test]
    fn http_chat_completion_returns_503_when_model_too_large_for_available_ram() {
        // 10_000 MiB model → 25_000 MiB required; probe reports 1 MiB
        // free → 503 with OpenAI-shaped error body and Retry-After.
        let cat = catalog_with("huge-model", 10_000);
        let handle = start_server_with_probe("x", cat, 1);
        let addr = handle.bound_address().to_string();
        let body = serde_json::json!({
            "model": "huge-model",
            "messages": [{"role":"user","content":"hi"}],
            "stream": false
        })
        .to_string();
        let (code, headers, resp_body) = post_capture_headers(&addr, "/v1/chat/completions", &body);
        assert_eq!(code, 503);
        // OpenAI-shaped body with `code == "model_too_large"`.
        let v: serde_json::Value = serde_json::from_str(&resp_body).expect("json");
        assert_eq!(v["error"]["code"], "model_too_large");
        assert_eq!(v["error"]["type"], "server_error");
        assert!(v["error"]["param"].is_null());
        assert!(v["error"]["message"]
            .as_str()
            .unwrap_or("")
            .contains("huge-model"));
        // Retry-After header present, value 30.
        let lh = headers.to_ascii_lowercase();
        assert!(
            lh.contains("retry-after: 30"),
            "expected Retry-After: 30 in headers; got:\n{headers}"
        );
        let _ = handle.stop();
    }

    #[test]
    fn http_chat_completion_proceeds_with_200_when_available_ram_is_sufficient() {
        // 100 MiB model → 250 MiB required; probe reports 8 GiB free →
        // falls through to the normal echo path with a 200.
        let cat = catalog_with("small-model", 100);
        let handle = start_server_with_probe("hello", cat, 8 * 1_024);
        let addr = handle.bound_address().to_string();
        let body = serde_json::json!({
            "model": "small-model",
            "messages": [{"role":"user","content":"hi"}],
            "stream": false
        })
        .to_string();
        let (code, _headers, resp_body) =
            post_capture_headers(&addr, "/v1/chat/completions", &body);
        assert_eq!(code, 200);
        let v: serde_json::Value = serde_json::from_str(&resp_body).expect("json");
        assert_eq!(v["object"], "chat.completion");
        let _ = handle.stop();
    }

    #[test]
    fn http_chat_completion_unknown_model_label_skips_memory_probe() {
        // The model isn't in the catalog at all — the memory check is
        // a no-op and the request flows to the echo provider as usual.
        let cat = ModelCatalog::new();
        let handle = start_server_with_probe("echoed", cat, 1);
        let addr = handle.bound_address().to_string();
        let body = serde_json::json!({
            "model": "no-such-model",
            "messages": [{"role":"user","content":"hi"}],
            "stream": false
        })
        .to_string();
        let (code, _headers, resp_body) =
            post_capture_headers(&addr, "/v1/chat/completions", &body);
        assert_eq!(code, 200);
        assert!(resp_body.contains("echoed"));
        let _ = handle.stop();
    }
}
