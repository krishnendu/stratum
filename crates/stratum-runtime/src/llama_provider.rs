//! Placeholder module — real `LlamaCppProvider` lands in a follow-up PR.
//!
//! Currently exposes an empty [`LlamaCppProvider`] type so the
//! `provider-llama-cpp` feature compiles end-to-end and the on-demand CI
//! workflow (`.github/workflows/provider-llama-cpp.yml`) has something to
//! build. It does **not** yet implement the [`crate::Provider`] trait.

/// Stub for the upcoming llama.cpp-backed [`crate::Provider`].
///
/// The real implementation will load a GGUF model via `llama-cpp-2` and
/// stream `Block`s back to the orchestrator. For now this type just
/// exists so the feature flag has a compile target.
#[derive(Debug, Default)]
pub struct LlamaCppProvider {
    _private: (),
}

impl LlamaCppProvider {
    /// Construct a new stub provider.
    ///
    /// Does no real work; the follow-up PR will accept a model path and
    /// runtime config here.
    #[must_use]
    pub const fn new() -> Self {
        Self { _private: () }
    }
}

#[cfg(test)]
mod tests {
    use super::LlamaCppProvider;

    #[test]
    fn stub_constructor_runs() {
        let _provider = LlamaCppProvider::new();
    }
}
