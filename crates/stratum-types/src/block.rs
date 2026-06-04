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
    fn cancelled_roundtrip() {
        let b = Block::Cancelled {
            reason: "STRAT-E4002 cancelled by user".into(),
        };
        let s = serde_json::to_string(&b).unwrap();
        let back: Block = serde_json::from_str(&s).unwrap();
        assert_eq!(b, back);
    }
}
