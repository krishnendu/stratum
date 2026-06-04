//! Provider primitives.
//!
//! Phase 1 v2 ships **concrete** providers only; the `Provider` trait
//! abstraction lands in Phase 2 once a second consumer (the embedder)
//! makes the seams visible. For now we have `EchoProvider`, a deterministic
//! provider used by the chat-loop tests and by `stratum echo`.

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use stratum_types::{Block, ModelId};

use crate::cancel::CancelToken;

/// Generation request handed to a provider.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenerateRequest {
    /// Which model to use; concrete providers may ignore this.
    pub model: ModelId,
    /// User prompt.
    pub prompt: String,
    /// Maximum number of `Block`s to emit (excluding `Done`/`Cancelled`).
    pub max_blocks: u32,
}

/// Deterministic echo provider for end-to-end loop tests.
///
/// Splits the prompt on whitespace and emits one `Block::Text` per word,
/// followed by a `Block::Usage` summary and `Block::Done`.
#[derive(Debug, Clone, Default)]
pub struct EchoProvider {
    /// Prefix prepended to every emitted word, e.g. `"echo: "`.
    pub prefix: Arc<String>,
}

impl EchoProvider {
    /// Build a fresh provider with the given prefix.
    #[must_use]
    pub fn new(prefix: impl Into<String>) -> Self {
        Self {
            prefix: Arc::new(prefix.into()),
        }
    }

    /// Run the request synchronously, returning the captured stream.
    ///
    /// The provider polls `cancel` between words and emits `Block::Cancelled`
    /// when the token fires.
    #[must_use]
    pub fn generate(&self, request: &GenerateRequest, cancel: &CancelToken) -> Vec<Block> {
        let mut out = Vec::new();
        let mut emitted = 0_u32;
        let mut prompt_tokens = 0_u32;
        for word in request.prompt.split_whitespace() {
            prompt_tokens = prompt_tokens.saturating_add(1);
            if cancel.is_cancelled() {
                out.push(Block::Cancelled {
                    reason: "STRAT-E4002 cancelled by user".to_string(),
                });
                return out;
            }
            if emitted >= request.max_blocks {
                break;
            }
            out.push(Block::Text {
                text: format!("{}{word}", self.prefix),
            });
            emitted = emitted.saturating_add(1);
        }
        out.push(Block::Usage {
            prompt: prompt_tokens,
            completion: emitted,
        });
        out.push(Block::Done);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(prompt: &str, max_blocks: u32) -> GenerateRequest {
        GenerateRequest {
            model: ModelId::from("echo"),
            prompt: prompt.to_string(),
            max_blocks,
        }
    }

    #[test]
    fn echo_emits_text_per_word() {
        let p = EchoProvider::new("");
        let blocks = p.generate(&req("hello world", 10), &CancelToken::new());
        assert_eq!(
            blocks[0],
            Block::Text {
                text: "hello".into()
            }
        );
        assert_eq!(
            blocks[1],
            Block::Text {
                text: "world".into()
            }
        );
    }

    #[test]
    fn echo_ends_with_usage_and_done() {
        let p = EchoProvider::new("");
        let blocks = p.generate(&req("a b", 10), &CancelToken::new());
        assert_eq!(
            blocks[blocks.len() - 2],
            Block::Usage {
                prompt: 2,
                completion: 2
            }
        );
        assert_eq!(*blocks.last().unwrap(), Block::Done);
    }

    #[test]
    fn echo_respects_max_blocks() {
        let p = EchoProvider::new("");
        let blocks = p.generate(&req("a b c d", 2), &CancelToken::new());
        let text_count = blocks
            .iter()
            .filter(|b| matches!(b, Block::Text { .. }))
            .count();
        assert_eq!(text_count, 2);
    }

    #[test]
    fn echo_prefixes_each_word() {
        let p = EchoProvider::new("echo: ");
        let blocks = p.generate(&req("ping", 10), &CancelToken::new());
        assert_eq!(
            blocks[0],
            Block::Text {
                text: "echo: ping".into()
            }
        );
    }

    #[test]
    fn echo_emits_cancelled_when_token_fires() {
        let p = EchoProvider::new("");
        let cancel = CancelToken::new();
        cancel.cancel();
        let blocks = p.generate(&req("ping pong", 10), &cancel);
        assert!(matches!(blocks[0], Block::Cancelled { .. }));
    }

    #[test]
    fn echo_emits_only_usage_and_done_for_empty_prompt() {
        let p = EchoProvider::new("");
        let blocks = p.generate(&req("", 10), &CancelToken::new());
        assert_eq!(blocks.len(), 2);
        assert_eq!(
            blocks[0],
            Block::Usage {
                prompt: 0,
                completion: 0
            }
        );
        assert_eq!(blocks[1], Block::Done);
    }

    #[test]
    fn echo_default_constructor_uses_empty_prefix() {
        let p = EchoProvider::default();
        let blocks = p.generate(&req("hi", 10), &CancelToken::new());
        assert_eq!(blocks[0], Block::Text { text: "hi".into() });
    }

    #[test]
    fn generate_request_serde_roundtrip() {
        let r = req("hi", 5);
        let s = serde_json::to_string(&r).unwrap();
        let back: GenerateRequest = serde_json::from_str(&s).unwrap();
        assert_eq!(r, back);
    }

    #[test]
    fn echo_provider_clone_independent_prefix() {
        let p = EchoProvider::new("a:");
        let q = p.clone();
        assert_eq!(*p.prefix, *q.prefix);
    }
}
