//! Capability, model family, and concurrency-model enums.

use serde::{Deserialize, Serialize};

/// Functional capability exposed by a Provider implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Capability {
    /// Streaming text generation.
    Generate,
    /// Vector embedding.
    Embed,
    /// Audio -> text.
    Transcribe,
    /// Text -> audio.
    Synthesize,
    /// Image / vision understanding.
    Vision,
}

/// Model family. Used to disambiguate prompts, tokenizers, and spec-dec pairings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Family {
    /// Google Gemma family.
    Gemma,
    /// Alibaba Qwen family.
    Qwen,
    /// `DeepSeek` family.
    DeepSeek,
    /// Mistral family.
    Mistral,
    /// Snowflake / Arctic.
    Arctic,
    /// Whisper (audio).
    Whisper,
    /// Piper (audio).
    Piper,
    /// Anything else.
    Other,
}

/// How a Provider handles concurrent inference requests against the same model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConcurrencyModel {
    /// One inference at a time per loaded model (mutex).
    Exclusive,
    /// One KV cache per session, pool-limited.
    KvPerSession,
    /// Stateless batched (typical for embedders).
    BatchedStateless,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capability_serde_roundtrip() {
        for c in [
            Capability::Generate,
            Capability::Embed,
            Capability::Transcribe,
            Capability::Synthesize,
            Capability::Vision,
        ] {
            let s = serde_json::to_string(&c).unwrap();
            let back: Capability = serde_json::from_str(&s).unwrap();
            assert_eq!(c, back);
        }
    }

    #[test]
    fn family_serde_roundtrip() {
        for f in [
            Family::Gemma,
            Family::Qwen,
            Family::DeepSeek,
            Family::Mistral,
            Family::Arctic,
            Family::Whisper,
            Family::Piper,
            Family::Other,
        ] {
            let s = serde_json::to_string(&f).unwrap();
            let back: Family = serde_json::from_str(&s).unwrap();
            assert_eq!(f, back);
        }
    }

    #[test]
    fn concurrency_serde_roundtrip() {
        for c in [
            ConcurrencyModel::Exclusive,
            ConcurrencyModel::KvPerSession,
            ConcurrencyModel::BatchedStateless,
        ] {
            let s = serde_json::to_string(&c).unwrap();
            let back: ConcurrencyModel = serde_json::from_str(&s).unwrap();
            assert_eq!(c, back);
        }
    }

    #[test]
    fn capability_distinct_variants() {
        use std::collections::HashSet;
        let set: HashSet<_> = [
            Capability::Generate,
            Capability::Embed,
            Capability::Transcribe,
            Capability::Synthesize,
            Capability::Vision,
        ]
        .iter()
        .collect();
        assert_eq!(set.len(), 5);
    }

    #[test]
    fn family_distinct_variants() {
        use std::collections::HashSet;
        let set: HashSet<_> = [
            Family::Gemma,
            Family::Qwen,
            Family::DeepSeek,
            Family::Mistral,
            Family::Arctic,
            Family::Whisper,
            Family::Piper,
            Family::Other,
        ]
        .iter()
        .collect();
        assert_eq!(set.len(), 8);
    }

    #[test]
    fn concurrency_distinct_variants() {
        use std::collections::HashSet;
        let set: HashSet<_> = [
            ConcurrencyModel::Exclusive,
            ConcurrencyModel::KvPerSession,
            ConcurrencyModel::BatchedStateless,
        ]
        .iter()
        .collect();
        assert_eq!(set.len(), 3);
    }
}
