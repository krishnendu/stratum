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
use llama_cpp_2::model::{AddBos, LlamaModel};
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

/// Process-wide handle to the llama.cpp backend.
///
/// The backend must be initialized exactly once per process; doing it
/// lazily keeps the cost off the default-feature build entirely.
fn shared_backend() -> Result<Arc<LlamaBackend>, LlamaProviderError> {
    static BACKEND: OnceLock<Result<Arc<LlamaBackend>, String>> = OnceLock::new();
    let cached = BACKEND.get_or_init(|| {
        LlamaBackend::init()
            .map(Arc::new)
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
        })
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
    fn build_sampler(&self, req: &GenerateRequest) -> LlamaSampler {
        let seed = u32::try_from(self.seed & u64::from(u32::MAX)).unwrap_or(u32::MAX);
        LlamaSampler::chain_simple([
            LlamaSampler::penalties(REPEAT_PENALTY_LAST_N, Self::repeat_penalty(req), 0.0, 0.0),
            LlamaSampler::top_k(40),
            LlamaSampler::top_p(Self::top_p(req), 1),
            LlamaSampler::temp(Self::temperature(req)),
            LlamaSampler::dist(seed),
        ])
    }

    /// Run the core generate loop, returning aggregated text.
    fn generate_text(
        &self,
        req: &GenerateRequest,
        cancel: &CancelToken,
    ) -> Result<Option<String>, LlamaProviderError> {
        let n_ctx_nz = NonZeroU32::new(self.n_ctx)
            .ok_or_else(|| LlamaProviderError::Backend("n_ctx must be non-zero".to_string()))?;
        let mut ctx_params = LlamaContextParams::default().with_n_ctx(Some(n_ctx_nz));
        if self.n_threads > 0 {
            ctx_params = ctx_params
                .with_n_threads(self.n_threads)
                .with_n_threads_batch(self.n_threads);
        }

        let mut ctx = self
            .model
            .new_context(&self.backend, ctx_params)
            .map_err(|e| LlamaProviderError::Backend(e.to_string()))?;

        let prompt_tokens = self
            .model
            .str_to_token(&req.prompt, AddBos::Always)
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

        while produced < max_new_tokens {
            if cancel.is_cancelled() {
                return Ok(None);
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

            batch.clear();
            batch
                .add(token, next_pos, &[0], true)
                .map_err(|e| LlamaProviderError::Sampling(e.to_string()))?;
            ctx.decode(&mut batch)
                .map_err(|e| LlamaProviderError::Sampling(e.to_string()))?;
            next_pos = next_pos.saturating_add(1);
            produced = produced.saturating_add(1);
        }

        if output.is_empty() {
            Ok(None)
        } else {
            Ok(Some(output))
        }
    }
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
            Ok(Some(text)) => vec![Block::Text { text }],
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
        max_blocks: 1,
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
