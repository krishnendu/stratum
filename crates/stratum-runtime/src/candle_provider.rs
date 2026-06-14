//! `CandleProvider` — `Provider` trait impl backed by Hugging Face's
//! Candle inference engine.
//!
//! ## Status
//!
//! **Scaffold.** Implements the [`Provider`] trait shape so callers can
//! wire `Arc<dyn Provider>` against Candle without changing call
//! sites. The actual model load + inference loop is a TODO — current
//! impl returns a single canned `Block::Text` so tests + plumbing
//! work end-to-end.
//!
//! ## Why a scaffold
//!
//! The Candle dep is heavy (pulls in tokenizers + safetensors) and
//! its CPU-only feature set is fine for embeddings but slow for chat
//! completions. Phase 2 v2 wants Candle primarily for **embeddings**
//! (Arctic-Embed-L per `plan/02 §Embeddings`) — that's the first
//! real use case to land. Chat-completion via Candle stays opt-in via
//! a future feature gate.
//!
//! Feature-gated behind `provider-candle` so the dep isn't pulled in
//! by default. Until the feature lands, this module compiles as a
//! pure trait surface — useful for documentation + as a TODO marker.

use std::sync::Arc;

use stratum_types::{Block, Capability, ModelId};

use crate::cancel::CancelToken;
use crate::provider::{GenerateRequest, Provider};

/// One Candle-backed provider.
#[derive(Debug, Clone)]
pub struct CandleProvider {
    model_path: Arc<std::path::PathBuf>,
    n_ctx: u32,
}

impl CandleProvider {
    /// Build a provider against a GGUF / safetensors file on disk.
    /// `n_ctx` is the requested context window — Candle may honor a
    /// smaller value if the model has a hard limit.
    ///
    /// # Errors
    /// Returns `Err` when `n_ctx == 0` (matches the `LlamaCppProvider`
    /// contract).
    pub fn open(model_path: std::path::PathBuf, n_ctx: u32) -> Result<Self, String> {
        if n_ctx == 0 {
            return Err("n_ctx must be > 0".into());
        }
        Ok(Self {
            model_path: Arc::new(model_path),
            n_ctx,
        })
    }

    /// Model file this provider was opened against.
    #[must_use]
    pub fn model_path(&self) -> &std::path::Path {
        &self.model_path
    }

    /// Effective context window.
    #[must_use]
    pub const fn n_ctx(&self) -> u32 {
        self.n_ctx
    }
}

impl Provider for CandleProvider {
    fn id(&self) -> &'static str {
        "candle"
    }

    fn capabilities(&self) -> &'static [Capability] {
        // Same surface as LlamaCppProvider — Phase 2 v2 adds the
        // `embed` capability once embedding mode is wired.
        const CAPS: &[Capability] = &[Capability::Generate];
        CAPS
    }

    fn generate(&self, _request: &GenerateRequest, _cancel: &CancelToken) -> Vec<Block> {
        // Scaffold: emit a single canned text block + a Done marker.
        // Real impl lands behind `provider-candle` once the Candle dep
        // is approved.
        vec![
            Block::Text {
                text: "(candle provider scaffold — model load not yet implemented)".to_string(),
            },
            Block::Done,
        ]
    }
}

/// Re-export of the model id constant Stratum tags Candle outputs with.
#[must_use]
pub fn candle_model_id() -> ModelId {
    ModelId::from("candle")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_rejects_zero_n_ctx() {
        let err = CandleProvider::open("/x".into(), 0).unwrap_err();
        assert!(err.contains("n_ctx"));
    }

    #[test]
    fn open_accepts_valid_inputs() {
        let p = CandleProvider::open("/x".into(), 4096).unwrap();
        assert_eq!(p.n_ctx(), 4096);
        assert_eq!(p.model_path().to_str().unwrap(), "/x");
    }

    #[test]
    fn provider_id_is_candle() {
        let p = CandleProvider::open("/x".into(), 1024).unwrap();
        assert_eq!(<CandleProvider as Provider>::id(&p), "candle");
    }

    #[test]
    fn provider_capabilities_include_generate() {
        let p = CandleProvider::open("/x".into(), 1024).unwrap();
        let caps = <CandleProvider as Provider>::capabilities(&p);
        assert!(caps.contains(&Capability::Generate));
    }

    #[test]
    fn generate_returns_scaffold_block_and_done() {
        let p = CandleProvider::open("/x".into(), 1024).unwrap();
        let req = GenerateRequest {
            model: candle_model_id(),
            prompt: "hi".into(),
            max_blocks: 4,
            system_override: None,
            history: Vec::new(),
            sampler: crate::provider::SamplerParams::default(),
        };
        let blocks = <CandleProvider as Provider>::generate(&p, &req, &CancelToken::new());
        assert_eq!(blocks.len(), 2);
        assert!(matches!(blocks[0], Block::Text { .. }));
        assert!(matches!(blocks[1], Block::Done));
    }

    #[test]
    fn candle_model_id_is_stable() {
        assert_eq!(candle_model_id(), ModelId::from("candle"));
    }
}
