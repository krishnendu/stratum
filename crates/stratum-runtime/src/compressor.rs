//! Context compressor — abstraction over "compress this text down
//! to roughly N tokens" so the runtime doesn't bake in a single
//! backend.
//!
//! ## Why a trait
//!
//! Phase 4 v2 (`plan/04 §LLMLingua-2`) plans an LLM-side compressor
//! that uses a small encoder to rate token importance and drop
//! low-value spans. That's research-heavy and pulls in a Python
//! dependency. v1 ships the deterministic [`crate::caveman`]
//! compressor as a `CompressorBackend` impl; LLMLingua-2 plugs in
//! later without touching callers.
//!
//! ## Token-budget contract
//!
//! `target_tokens` is the **rough** target. A compressor is allowed
//! to overshoot by ~10% — the runtime applies the cap downstream
//! via `truncate_for_prompt`. Compressors should never panic on
//! adversarial input; they degrade to "no change" + a warning.

use std::sync::Arc;

/// Per-call compression hint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompressionBudget {
    /// Approximate target token count. 0 means "compress as much as
    /// safely possible" (used for tool-output triage).
    pub target_tokens: usize,
}

/// Backend that compresses prose under a token budget.
pub trait CompressorBackend: Send + Sync + std::fmt::Debug {
    /// Compress `input` toward `budget`. Implementations MAY return
    /// the input unchanged if compression isn't applicable (e.g. a
    /// code block, a tool-call JSON).
    fn compress(&self, input: &str, budget: CompressionBudget) -> String;

    /// Stable identifier used in telemetry + crash reports.
    fn id(&self) -> &'static str;
}

/// No-op compressor — returns input unchanged. Useful for tests + as
/// the fallback when configuration disables compression.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoopCompressor;

impl CompressorBackend for NoopCompressor {
    fn compress(&self, input: &str, _budget: CompressionBudget) -> String {
        input.to_string()
    }
    fn id(&self) -> &'static str {
        "noop"
    }
}

/// Caveman compressor — heuristic prose → fragment (drops filler,
/// preserves paths/code/JSON). Stratum's v1 backend.
#[derive(Debug, Clone, Copy, Default)]
pub struct CavemanCompressor;

impl CompressorBackend for CavemanCompressor {
    fn compress(&self, input: &str, _budget: CompressionBudget) -> String {
        crate::caveman::compress(input)
    }
    fn id(&self) -> &'static str {
        "caveman"
    }
}

/// Toolbox front: picks the right backend by name + caches the
/// allocation.
#[derive(Debug, Clone)]
pub struct Compressor {
    backend: Arc<dyn CompressorBackend>,
}

impl Compressor {
    /// Build a compressor wrapping a backend.
    #[must_use]
    pub fn new(backend: Arc<dyn CompressorBackend>) -> Self {
        Self { backend }
    }

    /// Compress under a budget.
    #[must_use]
    pub fn compress(&self, input: &str, budget: CompressionBudget) -> String {
        self.backend.compress(input, budget)
    }

    /// Backend id (for telemetry).
    #[must_use]
    pub fn backend_id(&self) -> &'static str {
        self.backend.id()
    }

    /// Build the default v1 stack — Caveman.
    #[must_use]
    pub fn caveman() -> Self {
        Self::new(Arc::new(CavemanCompressor))
    }

    /// No-op (debug / disabled path).
    #[must_use]
    pub fn noop() -> Self {
        Self::new(Arc::new(NoopCompressor))
    }

    /// Resolve a backend by name. Falls back to `Caveman` for unknown
    /// values rather than erroring — the runtime never refuses to run
    /// a turn because of a typo in `compressor: "..."`.
    #[must_use]
    pub fn by_name(name: &str) -> Self {
        match name.trim().to_ascii_lowercase().as_str() {
            "noop" | "off" | "none" | "disabled" => Self::noop(),
            // `llmlingua` resolves to caveman until the real backend
            // lands; the typo-tolerant fallback is below.
            _ => Self::caveman(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noop_returns_input_unchanged() {
        let c = Compressor::noop();
        let out = c.compress(
            "the user is asking about the file system",
            CompressionBudget { target_tokens: 5 },
        );
        assert_eq!(out, "the user is asking about the file system");
    }

    #[test]
    fn caveman_shortens_filler_text() {
        let c = Compressor::caveman();
        let verbose = "The user is just simply asking about a file in the workspace";
        let out = c.compress(verbose, CompressionBudget { target_tokens: 8 });
        assert!(out.len() < verbose.len());
    }

    #[test]
    fn caveman_preserves_paths() {
        let c = Compressor::caveman();
        let out = c.compress(
            "look at the file src/main.rs in the workspace",
            CompressionBudget { target_tokens: 0 },
        );
        assert!(out.contains("src/main.rs"));
    }

    #[test]
    fn backend_id_is_stable() {
        assert_eq!(Compressor::noop().backend_id(), "noop");
        assert_eq!(Compressor::caveman().backend_id(), "caveman");
    }

    #[test]
    fn by_name_resolves_known_aliases() {
        assert_eq!(Compressor::by_name("noop").backend_id(), "noop");
        assert_eq!(Compressor::by_name("OFF").backend_id(), "noop");
        assert_eq!(Compressor::by_name("disabled").backend_id(), "noop");
        assert_eq!(Compressor::by_name("caveman").backend_id(), "caveman");
        // Unknown → typo-tolerant fallback to caveman.
        assert_eq!(Compressor::by_name("xyz").backend_id(), "caveman");
        // LLMLingua resolves to caveman until the real backend lands.
        assert_eq!(Compressor::by_name("llmlingua").backend_id(), "caveman");
    }

    #[test]
    fn compressor_is_clone_and_share_safe() {
        let c = Compressor::caveman();
        let c2 = c;
        let _: Arc<dyn CompressorBackend> = c2.backend.clone();
    }
}
