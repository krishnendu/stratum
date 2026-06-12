//! Llama.cpp-backed [`Provider`](crate::provider::Provider) implementation.
//!
//! Feature-gated behind `provider-llama-cpp`; the heavy `llama-cpp-2`
//! native build is **off** for default per-PR CI and only exercised by
//! `.github/workflows/provider-llama-cpp.yml` (nightly cron, release tag,
//! manual dispatch).
//!
//! The implementation wraps a single `LlamaBackend` (process-wide,
//! lazily initialized) plus a per-provider `LlamaModel` loaded from a
//! GGUF file. Each [`Provider::generate`] call spins a fresh
//! `LlamaContext` so requests do not share KV cache state with each
//! other — heavier, but matches the Phase 2 single-shot semantics.

use std::num::NonZeroU32;
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaChatMessage, LlamaChatTemplate, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;
use stratum_types::{Block, Capability};

use crate::cancel::CancelToken;
use crate::provider::{GenerateRequest, Provider};

/// Provider identifier surfaced through the registry.
pub const PROVIDER_ID: &str = "llama-cpp-2";

/// Default sampling temperature used when the request does not override it.
const DEFAULT_TEMPERATURE: f32 = 0.7;
/// Default nucleus-sampling cutoff.
const DEFAULT_TOP_P: f32 = 0.95;
/// Default repeat penalty applied across the last `REPEAT_PENALTY_LAST_N` tokens.
const DEFAULT_REPEAT_PENALTY: f32 = 1.1;
/// Window (in tokens) inspected by the repeat-penalty sampler.
const REPEAT_PENALTY_LAST_N: i32 = 64;
/// One `Block` ≈ this many tokens for the placeholder block-budget math.
const TOKENS_PER_BLOCK: u32 = 64;

/// GBNF grammar that constrains tool-call emission to exactly one JSON
/// object of shape `{"tool":"<id>","args":{...}}`. Wired in as a LAZY
/// sampler: it is only enforced once the model emits a leading `{`,
/// leaving plain-text replies entirely unconstrained.
const TOOL_CALL_GBNF: &str = r#"root        ::= "{" ws "\"tool\"" ws ":" ws tool-id ws "," ws "\"args\"" ws ":" ws object ws "}"
tool-id     ::= "\"" tool-name "\""
tool-name   ::= "fs.read" | "fs.write" | "fs.edit" | "grep" | "glob" | "shell.exec" | "subagent.run"
value       ::= object | array | string | number | "true" | "false" | "null"
object      ::= "{" ws ( member ( ws "," ws member )* )? ws "}"
member      ::= string ws ":" ws value
array       ::= "[" ws ( value ( ws "," ws value )* )? ws "]"
string      ::= "\"" char* "\""
char        ::= [^"\\] | "\\" ( ["\\/bfnrt] | "u" hex hex hex hex )
hex         ::= [0-9a-fA-F]
number      ::= "-"? ( "0" | [1-9] [0-9]* ) ( "." [0-9]+ )? ( [eE] [-+]? [0-9]+ )?
ws          ::= [ \t\n]*
"#;

/// Process-wide handle to the llama.cpp backend.
///
/// The backend must be initialized exactly once per process; doing it
/// lazily keeps the cost off the default-feature build entirely.
fn shared_backend() -> Result<Arc<LlamaBackend>, LlamaProviderError> {
    static BACKEND: OnceLock<Result<Arc<LlamaBackend>, String>> = OnceLock::new();
    let cached = BACKEND.get_or_init(|| {
        LlamaBackend::init()
            .map(|mut b| {
                if std::env::var_os("STRATUM_LLAMA_VERBOSE").is_none() {
                    b.void_logs();
                }
                Arc::new(b)
            })
            .map_err(|e| e.to_string())
    });
    match cached {
        Ok(b) => Ok(Arc::clone(b)),
        Err(msg) => Err(LlamaProviderError::Backend(msg.clone())),
    }
}

/// Configuration for opening a [`LlamaCppProvider`].
#[derive(Debug, Clone)]
pub struct LlamaCppProviderConfig {
    /// Path to the GGUF model file on disk.
    pub model_path: PathBuf,
    /// Logical context window in tokens.
    pub n_ctx: u32,
    /// Worker-thread count. `None` lets llama.cpp pick its own default.
    pub n_threads: Option<i32>,
    /// Number of layers offloaded to GPU. `0` keeps the model fully on CPU.
    pub n_gpu_layers: i32,
    /// Seed used for the sampling chain.
    pub seed: u64,
}

impl LlamaCppProviderConfig {
    /// Build a config, rejecting obviously invalid combinations early.
    ///
    /// `n_ctx == 0` is rejected because llama.cpp's `n_ctx` is a
    /// [`NonZeroU32`] under the hood; surfacing the error up here keeps
    /// callers from hitting an FFI panic deep in the open path.
    ///
    /// # Errors
    ///
    /// Returns [`LlamaProviderError::Backend`] when `n_ctx == 0`.
    pub fn new(
        model_path: PathBuf,
        n_ctx: u32,
        n_threads: Option<i32>,
        n_gpu_layers: i32,
        seed: u64,
    ) -> Result<Self, LlamaProviderError> {
        if n_ctx == 0 {
            return Err(LlamaProviderError::Backend(
                "n_ctx must be greater than zero".to_string(),
            ));
        }
        Ok(Self {
            model_path,
            n_ctx,
            n_threads,
            n_gpu_layers,
            seed,
        })
    }
}

/// Real llama.cpp-backed [`Provider`].
///
/// One instance owns one loaded model; cloning is intentionally not
/// supported because the underlying `LlamaModel` is FFI-owned and the
/// process is expected to hold one provider per model.
#[derive(Debug)]
pub struct LlamaCppProvider {
    /// Backend handle kept alive for the lifetime of this provider.
    backend: Arc<LlamaBackend>,
    /// Loaded GGUF model.
    model: LlamaModel,
    /// Context window in tokens, copied from the config.
    n_ctx: u32,
    /// Worker-thread count; `0` => library default.
    n_threads: i32,
    /// Sampling seed copied from the config.
    seed: u64,
    /// Tool names + one-line descriptions surfaced to the model in the
    /// system prompt so it knows what verbs to emit. `(id, description)`.
    /// Empty list = no tool guidance injected.
    tools: Vec<(String, String)>,
}

impl LlamaCppProvider {
    /// Load a model and prepare the provider for generation.
    ///
    /// # Errors
    ///
    /// - [`LlamaProviderError::Backend`] if the global backend fails to
    ///   initialize.
    /// - [`LlamaProviderError::ModelLoad`] if the GGUF file cannot be
    ///   loaded.
    pub fn open(cfg: &LlamaCppProviderConfig) -> Result<Self, LlamaProviderError> {
        if cfg.n_ctx == 0 {
            return Err(LlamaProviderError::Backend(
                "n_ctx must be greater than zero".to_string(),
            ));
        }
        let backend = shared_backend()?;
        let n_gpu_layers = u32::try_from(cfg.n_gpu_layers.max(0)).unwrap_or(0);
        let model_params = LlamaModelParams::default().with_n_gpu_layers(n_gpu_layers);
        let model = LlamaModel::load_from_file(&backend, &cfg.model_path, &model_params)
            .map_err(|e| LlamaProviderError::ModelLoad(e.to_string()))?;
        Ok(Self {
            backend,
            model,
            n_ctx: cfg.n_ctx,
            n_threads: cfg.n_threads.unwrap_or(0),
            seed: cfg.seed,
            tools: Vec::new(),
        })
    }

    /// Install the tool catalog the model is told about in its system
    /// prompt. Empty list disables tool guidance.
    #[must_use]
    pub fn with_tools(mut self, tools: Vec<(String, String)>) -> Self {
        self.tools = tools;
        self
    }

    /// Sampling temperature actually used for a request.
    const fn temperature(_req: &GenerateRequest) -> f32 {
        DEFAULT_TEMPERATURE
    }

    /// Nucleus-sampling cutoff actually used for a request.
    const fn top_p(_req: &GenerateRequest) -> f32 {
        DEFAULT_TOP_P
    }

    /// Repeat penalty actually used for a request.
    const fn repeat_penalty(_req: &GenerateRequest) -> f32 {
        DEFAULT_REPEAT_PENALTY
    }

    /// Build the sampler chain for one request.
    ///
    /// Opt-in GBNF grammar sampler: set `STRATUM_GBNF=1` to enable.
    /// When enabled we prepend a LAZY grammar keyed on the `^\{`
    /// trigger so the model can emit free-form text OR a single
    /// well-formed `{"tool":"…","args":{…}}` object — never a malformed
    /// hybrid. Off by default until the per-model grammar interaction
    /// is hardened (some llama.cpp 0.1.146 builds assert mid-decode
    /// when the lazy trigger fires on certain tokenizers).
    fn build_sampler(&self, req: &GenerateRequest) -> LlamaSampler {
        let seed = u32::try_from(self.seed & u64::from(u32::MAX)).unwrap_or(u32::MAX);
        let mut samplers: Vec<LlamaSampler> = Vec::with_capacity(6);
        if std::env::var_os("STRATUM_GBNF").as_deref() == Some(std::ffi::OsStr::new("1")) {
            if let Ok(grammar) = LlamaSampler::grammar_lazy_patterns(
                &self.model,
                TOOL_CALL_GBNF,
                "root",
                &["^\\{".to_string()],
                &[],
            ) {
                samplers.push(grammar);
            }
        }
        samplers.push(LlamaSampler::penalties(
            REPEAT_PENALTY_LAST_N,
            Self::repeat_penalty(req),
            0.0,
            0.0,
        ));
        samplers.push(LlamaSampler::top_k(40));
        samplers.push(LlamaSampler::top_p(Self::top_p(req), 1));
        samplers.push(LlamaSampler::temp(Self::temperature(req)));
        samplers.push(LlamaSampler::dist(seed));
        LlamaSampler::chain_simple(samplers)
    }

    /// Build the system message that tells the model which tools are
    /// available and how to call them. Empty `tools` list returns the
    /// generic helpful-assistant prompt.
    fn system_prompt(&self) -> String {
        if self.tools.is_empty() {
            return "You are Stratum, a friendly local coding assistant.".to_string();
        }
        use std::fmt::Write;
        let mut out = String::with_capacity(1024);

        // -- 1. IDENTITY (one sentence, top of prompt). --
        out.push_str(
            "You are Stratum, a friendly local coding assistant running on the user's \
             machine.\n\n\
             CRITICAL: when you call a tool, the JSON object MUST be your entire reply. \
             NO preamble. NO 'here is how'. NO surrounding prose. NO markdown fences. \
             Just the JSON, alone on one line.\n\n",
        );

        // -- 2. MODES. Strict default = chat; tool use is the exception. --
        out.push_str(
            "## MODES\n\
             You have two modes. By default you are in CHAT mode and you reply in plain \
             English. You only switch to TOOL mode when the user explicitly asks for a \
             workspace action.\n\n\
             CHAT mode applies to: greetings, questions about your capabilities, opinions, \
             explanations, summaries of prior tool output, code discussion, debugging \
             advice, planning, anything conversational.\n\n\
             TOOL mode applies to: \"list files\", \"open <file>\", \"read <file>\", \
             \"search for <pattern>\", \"find <regex>\", \"edit <file>\", \"write <file>\", \
             \"run <allowlisted command>\". The user names a concrete workspace operation.\n\n",
        );

        // -- 3. RULES — named bans (Cline-style). --
        out.push_str(
            "## RULES\n\
             1. You are STRICTLY FORBIDDEN from emitting JSON, code fences containing JSON, \
                or any tool-call payload in CHAT mode. A user typing \"hi\", \"hello\", \
                \"what can you do\", or any greeting/question/discussion gets a plain \
                English reply only.\n\
             2. You are STRICTLY FORBIDDEN from inventing filenames or paths. Use only \
                paths the user typed in this conversation or paths returned by a prior \
                `glob` / `grep` call.\n\
             3. You are STRICTLY FORBIDDEN from using absolute paths (`/`, `/tmp`, \
                `/workspace`, `/etc`, `~`, `~/.ssh`) or parent-dir escapes (`..`). \
                Paths are always workspace-relative.\n\
             4. You are STRICTLY FORBIDDEN from chaining tool calls in one reply. Emit \
                exactly one JSON object per turn, then wait for the result.\n\
             5. You are STRICTLY FORBIDDEN from wrapping JSON in markdown code fences \
                (` ```json `, ` ``` `). The JSON stands alone on one line.\n\n",
        );

        // -- 4. TOOL FORMAT (over-specified, one shape only). --
        out.push_str(
            "## TOOL FORMAT\n\
             When (and only when) you are in TOOL mode, your entire reply is exactly one \
             line of JSON, this shape:\n\
             {\"tool\":\"<tool_id>\",\"args\":{...}}\n\n",
        );

        // -- 5. AVAILABLE TOOLS (compact name list; details in examples). --
        out.push_str("## AVAILABLE TOOLS\n");
        for (id, desc) in &self.tools {
            let _ = writeln!(out, "- `{id}` — {desc}");
        }
        out.push('\n');

        // -- 6. FEW-SHOT (form-teaching, not content-teaching). --
        out.push_str(
            "## EXAMPLES\n\
             user: hi\n\
             assistant: Hello! What can I help with today?\n\n\
             user: what tools do you have\n\
             assistant: I can list files (glob), search code (grep), read/edit/write files, \
             and run a few shell commands. Tell me what you'd like and I'll do it.\n\n\
             user: what can you do\n\
             assistant: I'm a local coding assistant. I can chat, plan, explain code, and \
             when you point me at the workspace I can list files, search, read, edit, write, \
             and run a few small shell commands. What are you working on?\n\n\
             user: list rust files\n\
             assistant: {\"tool\":\"glob\",\"args\":{\"pattern\":\"**/*.rs\"}}\n\n\
             user: show me Cargo.toml\n\
             assistant: {\"tool\":\"fs.read\",\"args\":{\"path\":\"Cargo.toml\"}}\n\n\
             user: find usages of `foo`\n\
             assistant: {\"tool\":\"grep\",\"args\":{\"pattern\":\"foo\"}}\n\n",
        );

        // -- 7. REMINDER. --
        out.push_str(
            "When in doubt: CHAT mode. Plain English. No JSON.\n",
        );
        out
    }

    fn format_chat_prompt_with(&self, prompt: &str, system_override: Option<&str>) -> String {
        let system = system_override
            .map(str::to_owned)
            .unwrap_or_else(|| self.system_prompt());
        let chatml_fallback = || {
            format!(
                "<|im_start|>system\n{system}<|im_end|>\n\
                 <|im_start|>user\n{prompt}<|im_end|>\n\
                 <|im_start|>assistant\n"
            )
        };
        let tmpl = match self.model.chat_template(None) {
            Ok(t) => t,
            Err(_) => match LlamaChatTemplate::new("chatml") {
                Ok(t) => t,
                Err(_) => return chatml_fallback(),
            },
        };
        let system_msg = LlamaChatMessage::new("system".to_string(), system.clone()).ok();
        let user_msg = match LlamaChatMessage::new("user".to_string(), prompt.to_string()) {
            Ok(m) => m,
            Err(_) => return chatml_fallback(),
        };
        let msgs: Vec<LlamaChatMessage> = match system_msg {
            Some(s) => vec![s, user_msg],
            None => vec![user_msg],
        };
        self.model
            .apply_chat_template(&tmpl, &msgs, true)
            .unwrap_or_else(|_| chatml_fallback())
    }

    /// Run the core generate loop, returning aggregated text.
    fn generate_text(
        &self,
        req: &GenerateRequest,
        cancel: &CancelToken,
    ) -> Result<Option<String>, LlamaProviderError> {
        self.generate_text_streaming(req, cancel, &|_| {})
    }

    fn generate_text_streaming(
        &self,
        req: &GenerateRequest,
        cancel: &CancelToken,
        on_piece: &dyn Fn(&str),
    ) -> Result<Option<String>, LlamaProviderError> {
        let n_ctx_nz = NonZeroU32::new(self.n_ctx)
            .ok_or_else(|| LlamaProviderError::Backend("n_ctx must be non-zero".to_string()))?;
        // `n_batch` must be >= prompt token count or llama.cpp asserts
        // `GGML_ASSERT(n_tokens_all <= cparams.n_batch)`. Set it equal
        // to `n_ctx` so the largest single decode fits. The KV slot
        // allocator concern that motivated the old 512 cap is addressed
        // by `with_kv_unified` + the `next_pos < n_ctx` guard in the
        // generation loop below.
        let mut ctx_params = LlamaContextParams::default()
            .with_n_ctx(Some(n_ctx_nz))
            .with_n_batch(self.n_ctx)
            .with_n_seq_max(1)
            .with_kv_unified(true);
        if self.n_threads > 0 {
            ctx_params = ctx_params
                .with_n_threads(self.n_threads)
                .with_n_threads_batch(self.n_threads);
        }

        let mut ctx = self
            .model
            .new_context(&self.backend, ctx_params)
            .map_err(|e| LlamaProviderError::Backend(e.to_string()))?;

        let formatted_prompt =
            self.format_chat_prompt_with(&req.prompt, req.system_override.as_deref());
        let prompt_tokens = self
            .model
            .str_to_token(&formatted_prompt, AddBos::Never)
            .map_err(|e| LlamaProviderError::Tokenize(e.to_string()))?;
        if prompt_tokens.is_empty() {
            return Ok(None);
        }

        let max_new_tokens = req.max_blocks.saturating_mul(TOKENS_PER_BLOCK);
        if max_new_tokens == 0 {
            return Ok(None);
        }

        // Submit the full prompt as a single batch and decode it. The
        // last prompt token is what we sample from next.
        let batch_capacity = usize::try_from(self.n_ctx).unwrap_or(usize::MAX);
        let mut batch = LlamaBatch::new(batch_capacity.max(prompt_tokens.len()), 1);
        let last_prompt_index = prompt_tokens.len().saturating_sub(1);
        for (i, token) in prompt_tokens.iter().enumerate() {
            let pos = i32::try_from(i).map_err(|e| LlamaProviderError::Tokenize(e.to_string()))?;
            batch
                .add(*token, pos, &[0], i == last_prompt_index)
                .map_err(|e| LlamaProviderError::Tokenize(e.to_string()))?;
        }
        ctx.decode(&mut batch)
            .map_err(|e| LlamaProviderError::Sampling(e.to_string()))?;

        let mut sampler = self.build_sampler(req);
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut output = String::new();
        let mut produced: u32 = 0;
        let mut next_pos = i32::try_from(prompt_tokens.len())
            .map_err(|e| LlamaProviderError::Tokenize(e.to_string()))?;

        let n_ctx_i32 = i32::try_from(self.n_ctx).unwrap_or(i32::MAX);
        while produced < max_new_tokens {
            if cancel.is_cancelled() {
                return Ok(None);
            }
            if next_pos >= n_ctx_i32 {
                break;
            }
            let token: LlamaToken = sampler.sample(&ctx, batch.n_tokens() - 1);
            sampler.accept(token);
            if self.model.is_eog_token(token) {
                break;
            }
            let piece = self
                .model
                .token_to_piece(token, &mut decoder, false, None)
                .map_err(|e| LlamaProviderError::DecodeUtf8(e.to_string()))?;
            output.push_str(&piece);
            on_piece(&piece);

            batch.clear();
            batch
                .add(token, next_pos, &[0], true)
                .map_err(|e| LlamaProviderError::Sampling(e.to_string()))?;
            ctx.decode(&mut batch)
                .map_err(|e| LlamaProviderError::Sampling(e.to_string()))?;
            next_pos = next_pos.saturating_add(1);
            produced = produced.saturating_add(1);
        }

        // Strip chat-template sentinels that some GGUFs emit verbatim
        // when llama.cpp's EOG token list does not include them (notably
        // Gemma 4 IT files that ship a ChatML-flavored template).
        let cleaned = strip_chat_sentinels(&output);
        if cleaned.is_empty() {
            Ok(None)
        } else {
            Ok(Some(cleaned))
        }
    }
}

/// Remove trailing chat-template marker tokens that occasionally leak
/// into the rendered output. Covers both ChatML (`<|im_end|>` etc.) and
/// Gemma-style (`<end_of_turn>`, `<start_of_turn>`) sentinels so the
/// user never sees them in a reply.
fn strip_chat_sentinels(s: &str) -> String {
    const SENTINELS: &[&str] = &[
        "<|im_end|>",
        "<|im_start|>",
        "<|endoftext|>",
        "<end_of_turn>",
        "<start_of_turn>",
        "<eos>",
        "<bos>",
        // Qwen3 / DeepSeek-R1 thinking-mode tags. The whole `<think>...
        // </think>` chain-of-thought block is internal scratch; we strip
        // it so the user sees only the final answer.
        "<think>",
        "</think>",
    ];
    let mut out = s.to_string();
    loop {
        let trimmed_end = out.trim_end().to_string();
        let mut stripped = false;
        for sentinel in SENTINELS {
            if let Some(rest) = trimmed_end.strip_suffix(sentinel) {
                out = rest.trim_end().to_string();
                stripped = true;
                break;
            }
        }
        if !stripped {
            out = trimmed_end;
            break;
        }
    }
    // Strip entire `<think>…</think>` chain-of-thought blocks before
    // the per-sentinel pass — these are model scratch and should never
    // surface in the final answer.
    while let Some(start) = out.find("<think>") {
        if let Some(rel_end) = out[start..].find("</think>") {
            let end = start + rel_end + "</think>".len();
            out.replace_range(start..end, "");
        } else {
            // Unterminated — drop from <think> onwards.
            out.truncate(start);
            break;
        }
    }
    // Also drop any sentinel appearing inside the text (rare but
    // observed on long generations); replace with empty so prose
    // around it stays.
    for sentinel in SENTINELS {
        out = out.replace(sentinel, "");
    }
    // After sentinel removal, the leading user-turn block and any
    // role labels ("user", "assistant", "model") may show up as
    // orphan lines because the GGUF's chat template wrapped the
    // prompt with ChatML headers. Keep only what comes after the
    // LAST `assistant\n` or `model\n` role label, if any.
    for marker in ["assistant\n", "model\n", "assistant\r\n", "model\r\n"] {
        if let Some(idx) = out.rfind(marker) {
            out = out[idx + marker.len()..].to_string();
        }
    }
    out.trim().to_string()
}

impl Provider for LlamaCppProvider {
    fn id(&self) -> &str {
        PROVIDER_ID
    }

    fn capabilities(&self) -> &'static [Capability] {
        const CAPS: &[Capability] = &[Capability::Generate];
        CAPS
    }

    fn generate(&self, req: &GenerateRequest, cancel: &CancelToken) -> Vec<Block> {
        match self.generate_text(req, cancel) {
            Ok(Some(text)) => text_to_blocks(text),
            Ok(None) => Vec::new(),
            Err(e) => vec![Block::Cancelled {
                // No dedicated `STRAT-E####` is assigned yet — the CLI
                // surface that maps provider failures to taxonomy codes
                // lands in a follow-up PR. Until then we emit a plain
                // reason string and let the orchestrator log it.
                reason: format!("llama-cpp provider failure: {e}"),
            }],
        }
    }

    fn generate_streaming(
        &self,
        req: &GenerateRequest,
        cancel: &CancelToken,
        on_chunk: &dyn Fn(&Block),
    ) -> Vec<Block> {
        // Emit a `Block::Text` per llama.cpp piece as the model decodes.
        // The final returned `Vec<Block>` still holds the consolidated
        // text — possibly converted into a `Block::ToolCall` if the
        // model emitted a JSON tool call — so existing callers see
        // the same shape.
        let result = self.generate_text_streaming(req, cancel, &|piece| {
            on_chunk(&Block::Text {
                text: piece.to_string(),
            });
        });
        match result {
            Ok(Some(text)) => text_to_blocks(text),
            Ok(None) => Vec::new(),
            Err(e) => vec![Block::Cancelled {
                reason: format!("llama-cpp provider failure: {e}"),
            }],
        }
    }
}

/// Convert a raw model output string into one or more `Block`s.
///
/// If the output is a JSON object of shape `{"tool":"...","args":{...}}`
/// (optionally wrapped in surrounding whitespace), emit a
/// `Block::ToolCall` the dispatcher loop can consume. Otherwise emit a
/// plain `Block::Text`. The detection is lenient — we accept whitespace
/// before the opening brace and ignore trailing text after the
/// matching close — so partial-stream artifacts and trailing newlines
/// don't break detection.
fn text_to_blocks(text: String) -> Vec<Block> {
    let trimmed = strip_code_fence(text.trim());
    // Try to find an embedded `{"tool":...,"args":...}` JSON object
    // anywhere in the output — model often wraps it in markdown prose
    // ("Here is how: { ... }") despite system-prompt instructions.
    if let Some((start, end)) = find_tool_call_span(trimmed) {
        let candidate = &trimmed[start..=end];
        if let Ok(serde_json::Value::Object(map)) = serde_json::from_str::<serde_json::Value>(candidate) {
            if let (Some(serde_json::Value::String(tool)), Some(args_val)) =
                (map.get("tool"), map.get("args"))
            {
                let args_str =
                    serde_json::to_string(args_val).unwrap_or_else(|_| "{}".to_string());
                return vec![Block::ToolCall {
                    id: format!("call-{tool}"),
                    tool: tool.clone(),
                    args: args_str,
                }];
            }
        }
    }
    // Did the model attempt a tool call but emit malformed JSON?
    let looks_like_tool_attempt = trimmed.contains("\"tool\"") && trimmed.contains('{');
    if looks_like_tool_attempt {
        return vec![Block::Text {
            text: "(model tried to call a tool but emitted a malformed payload; ignoring)"
                .to_string(),
        }];
    }
    vec![Block::Text { text }]
}

/// Locate the byte indices `(start, end)` of the first balanced JSON
/// object in `s` that contains both `"tool"` and `"args"` keys. Returns
/// `None` when no such span exists. Scan-anywhere; we don't require the
/// JSON to be at position 0 because models routinely wrap it in prose.
fn find_tool_call_span(s: &str) -> Option<(usize, usize)> {
    let bytes = s.as_bytes();
    for start in 0..bytes.len() {
        if bytes[start] != b'{' {
            continue;
        }
        let rest = &s[start + 1..];
        if let Some(end_off) = find_balanced_close(rest) {
            let end = start + 1 + end_off;
            let candidate = &s[start..=end];
            // Cheap filter: only consider candidates that look like tool
            // calls. Avoids parsing every brace pair as JSON.
            if candidate.contains("\"tool\"") && candidate.contains("\"args\"") {
                return Some((start, end));
            }
        }
    }
    None
}

/// Strip a leading / trailing markdown code fence (```json, ```text,
/// ```, etc.) so the JSON-tool-call detector sees the raw object even
/// when the model wraps it for "presentation". Idempotent: if no fence
/// is present, returns the input unchanged.
fn strip_code_fence(s: &str) -> &str {
    let mut out = s;
    if let Some(rest) = out.strip_prefix("```") {
        // Skip the optional language tag up to the first newline.
        if let Some(nl) = rest.find('\n') {
            out = &rest[nl + 1..];
        } else {
            out = rest;
        }
    }
    if let Some(rest) = out.strip_suffix("```") {
        out = rest.trim_end();
    }
    out.trim()
}

/// Find the index (relative to `s`) of the `}` that closes the
/// opening `{` immediately before `s`. Assumes `s` starts inside
/// the object body. Returns `None` if no balanced close is found.
fn find_balanced_close(s: &str) -> Option<usize> {
    let bytes = s.as_bytes();
    let mut depth = 1i32;
    let mut in_string = false;
    let mut escape = false;
    for (i, &b) in bytes.iter().enumerate() {
        if escape {
            escape = false;
            continue;
        }
        match b {
            b'\\' if in_string => escape = true,
            b'"' => in_string = !in_string,
            b'{' if !in_string => depth += 1,
            b'}' if !in_string => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
    }
    None
}

/// Five-token deterministic smoke check used by the on-demand workflow.
///
/// Runs a fixed seed against the loaded model so CI can assert basic
/// plumbing without committing to any particular generated text.
///
/// # Errors
///
/// Surfaces any [`LlamaProviderError`] raised during generation.
pub fn echoey_smoke_text(provider: &LlamaCppProvider) -> Result<String, LlamaProviderError> {
    let req = GenerateRequest {
        model: stratum_types::ModelId::from(PROVIDER_ID),
        prompt: "Hello".to_string(),
        // 5 tokens budget — TOKENS_PER_BLOCK math rounds up, but the
        // EOG handler / sampler will short-circuit well before then on
        // a healthy model.
        max_blocks: 1, system_override: None,
    };
    let cancel = CancelToken::new();
    provider
        .generate_text(&req, &cancel)
        .map(Option::unwrap_or_default)
}

/// Errors raised by [`LlamaCppProvider`].
#[derive(Debug)]
pub enum LlamaProviderError {
    /// llama.cpp backend / context initialization failed.
    Backend(String),
    /// Loading the GGUF model from disk failed.
    ModelLoad(String),
    /// Tokenizing the prompt failed.
    Tokenize(String),
    /// Sampling or decoding a token failed.
    Sampling(String),
    /// Converting a token piece back into UTF-8 failed.
    DecodeUtf8(String),
}

impl std::fmt::Display for LlamaProviderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Backend(m) => write!(f, "llama backend error: {m}"),
            Self::ModelLoad(m) => write!(f, "llama model load error: {m}"),
            Self::Tokenize(m) => write!(f, "llama tokenize error: {m}"),
            Self::Sampling(m) => write!(f, "llama sampling error: {m}"),
            Self::DecodeUtf8(m) => write!(f, "llama decode utf-8 error: {m}"),
        }
    }
}

impl std::error::Error for LlamaProviderError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_rejects_zero_n_ctx() {
        let err = LlamaCppProviderConfig::new(PathBuf::from("/nonexistent"), 0, None, 0, 1)
            .expect_err("zero n_ctx must be rejected");
        assert!(matches!(err, LlamaProviderError::Backend(_)));
    }

    #[test]
    fn config_accepts_minimal_inputs() {
        let cfg = LlamaCppProviderConfig::new(PathBuf::from("/some/path"), 512, None, 0, 42)
            .expect("valid config");
        assert_eq!(cfg.n_ctx, 512);
        assert_eq!(cfg.seed, 42);
        assert_eq!(cfg.n_gpu_layers, 0);
        assert!(cfg.n_threads.is_none());
    }

    #[test]
    fn error_display_covers_each_variant() {
        let cases = [
            LlamaProviderError::Backend("b".into()),
            LlamaProviderError::ModelLoad("m".into()),
            LlamaProviderError::Tokenize("t".into()),
            LlamaProviderError::Sampling("s".into()),
            LlamaProviderError::DecodeUtf8("d".into()),
        ];
        for err in &cases {
            let rendered = format!("{err}");
            assert!(!rendered.is_empty(), "Display must emit something");
        }
        assert!(format!("{}", cases[0]).contains("backend"));
        assert!(format!("{}", cases[1]).contains("model load"));
        assert!(format!("{}", cases[2]).contains("tokenize"));
        assert!(format!("{}", cases[3]).contains("sampling"));
        assert!(format!("{}", cases[4]).contains("decode utf-8"));
    }

    #[test]
    fn error_is_std_error() {
        fn assert_error<T: std::error::Error>(_: &T) {}
        assert_error(&LlamaProviderError::Backend("x".into()));
    }

    #[test]
    fn provider_id_constant_is_stable() {
        assert_eq!(PROVIDER_ID, "llama-cpp-2");
    }

    #[test]
    fn backend_init_is_optional_no_op_without_model() {
        // When STRATUM_LLAMA_GGUF_PATH is unset this test is a no-op,
        // so the default-features build (which never sees this code)
        // and the on-demand workflow with no GGUF still pass.
        let Ok(path) = std::env::var("STRATUM_LLAMA_GGUF_PATH") else {
            return;
        };
        let Ok(cfg) = LlamaCppProviderConfig::new(PathBuf::from(path), 512, Some(2), 0, 0) else {
            return;
        };
        // A missing or malformed model file should not fail the unit
        // suite — that's the on-demand workflow's job to catch via the
        // sha256 verify step.
        let Ok(provider) = LlamaCppProvider::open(&cfg) else {
            return;
        };
        let _ = echoey_smoke_text(&provider);
    }
}
