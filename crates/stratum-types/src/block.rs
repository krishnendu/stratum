//! Streaming unit emitted by a Provider during generation.

use serde::{Deserialize, Serialize};

/// One chunk of output produced during a model turn. The Provider trait
/// streams `Block`s; the orchestrator and TUI render them.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Block {
    /// Plain assistant text.
    Text {
        /// Delta text for this chunk.
        text: String,
    },
    /// A tool call request issued by the model.
    ToolCall {
        /// Stable id for correlating with the matching result.
        id: String,
        /// Tool name as registered in the capability matrix.
        tool: String,
        /// Argument blob (JSON-serialized).
        args: String,
    },
    /// Result of a tool execution returned to the model.
    ToolResult {
        /// Matching `ToolCall::id`.
        id: String,
        /// Raw tool output (untrusted; fenced before re-entry into context).
        output: String,
    },
    /// Token usage at this point in the stream.
    Usage {
        /// Cumulative prompt tokens.
        prompt: u32,
        /// Cumulative completion tokens.
        completion: u32,
    },
    /// The stream ended normally.
    Done,
    /// The stream was cancelled.
    Cancelled {
        /// Human-readable reason; a `STRAT-Exxxx` code is included.
        reason: String,
    },
}

impl Block {
    /// If this block is a `ToolCall`, return its args parsed into a
    /// JSON object map. Returns `None` for non-ToolCall variants OR
    /// when the args string isn't a JSON object (an array, a number,
    /// or malformed JSON).
    ///
    /// Centralises the JSON-string-to-typed-map conversion that
    /// every dispatcher would otherwise re-implement. Cheap on the
    /// happy path — serde_json is forgiving and tail-recursive over
    /// short objects.
    #[must_use]
    pub fn tool_args(&self) -> Option<serde_json::Map<String, serde_json::Value>> {
        if let Block::ToolCall { args, .. } = self {
            serde_json::from_str::<serde_json::Value>(args)
                .ok()
                .and_then(|v| match v {
                    serde_json::Value::Object(m) => Some(m),
                    _ => None,
                })
        } else {
            None
        }
    }

    /// Extract a single string-valued arg from a `ToolCall`. Returns
    /// `None` for non-ToolCall variants, missing keys, or non-string
    /// values. The common dispatcher pattern is "fetch path / command
    /// / pattern" — this saves the parse + downcast boilerplate.
    #[must_use]
    pub fn tool_arg_str(&self, key: &str) -> Option<String> {
        self.tool_args()
            .as_ref()?
            .get(key)?
            .as_str()
            .map(str::to_string)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_block_roundtrip() {
        let b = Block::Text { text: "hi".into() };
        let s = serde_json::to_string(&b).unwrap();
        let back: Block = serde_json::from_str(&s).unwrap();
        assert_eq!(b, back);
    }

    #[test]
    fn tool_call_roundtrip() {
        let b = Block::ToolCall {
            id: "t1".into(),
            tool: "fs.read".into(),
            args: "{}".into(),
        };
        let s = serde_json::to_string(&b).unwrap();
        let back: Block = serde_json::from_str(&s).unwrap();
        assert_eq!(b, back);
    }

    #[test]
    fn tool_result_roundtrip() {
        let b = Block::ToolResult {
            id: "t1".into(),
            output: "ok".into(),
        };
        let s = serde_json::to_string(&b).unwrap();
        let back: Block = serde_json::from_str(&s).unwrap();
        assert_eq!(b, back);
    }

    #[test]
    fn usage_roundtrip() {
        let b = Block::Usage {
            prompt: 12,
            completion: 34,
        };
        let s = serde_json::to_string(&b).unwrap();
        let back: Block = serde_json::from_str(&s).unwrap();
        assert_eq!(b, back);
    }

    #[test]
    fn done_roundtrip() {
        let b = Block::Done;
        let s = serde_json::to_string(&b).unwrap();
        let back: Block = serde_json::from_str(&s).unwrap();
        assert_eq!(b, back);
    }

    #[test]
    fn tool_args_returns_map_on_well_formed_args() {
        let b = Block::ToolCall {
            id: "t1".into(),
            tool: "fs.read".into(),
            args: r#"{"path":"README.md","encoding":"utf-8"}"#.into(),
        };
        let map = b.tool_args().unwrap();
        assert_eq!(map.get("path").unwrap().as_str(), Some("README.md"));
        assert_eq!(map.get("encoding").unwrap().as_str(), Some("utf-8"));
    }

    #[test]
    fn tool_args_returns_none_on_malformed_json() {
        let b = Block::ToolCall {
            id: "t1".into(),
            tool: "x".into(),
            args: "not json".into(),
        };
        assert!(b.tool_args().is_none());
    }

    #[test]
    fn tool_args_returns_none_when_not_a_toolcall() {
        let b = Block::Text { text: "hi".into() };
        assert!(b.tool_args().is_none());
    }

    #[test]
    fn tool_args_returns_none_on_array_root() {
        // `{}` is the contract; arrays / numbers don't satisfy it.
        let b = Block::ToolCall {
            id: "t1".into(),
            tool: "x".into(),
            args: "[1,2,3]".into(),
        };
        assert!(b.tool_args().is_none());
    }

    #[test]
    fn tool_arg_str_extracts_known_key() {
        let b = Block::ToolCall {
            id: "t1".into(),
            tool: "fs.read".into(),
            args: r#"{"path":"a.txt"}"#.into(),
        };
        assert_eq!(b.tool_arg_str("path").as_deref(), Some("a.txt"));
        assert!(b.tool_arg_str("missing").is_none());
    }

    #[test]
    fn tool_arg_str_returns_none_for_non_string_value() {
        let b = Block::ToolCall {
            id: "t1".into(),
            tool: "x".into(),
            args: r#"{"count":42}"#.into(),
        };
        assert!(b.tool_arg_str("count").is_none());
    }

    #[test]
    fn cancelled_roundtrip() {
        let b = Block::Cancelled {
            reason: "STRAT-E4002 cancelled by user".into(),
        };
        let s = serde_json::to_string(&b).unwrap();
        let back: Block = serde_json::from_str(&s).unwrap();
        assert_eq!(b, back);
    }
}
