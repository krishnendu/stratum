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
    /// An image block — multi-modal data shape. No provider produces these yet.
    Image {
        /// MIME type (e.g. `image/png`, `image/jpeg`).
        mime: String,
        /// Inline base64 payload or URL reference.
        data: ImageData,
        /// Optional alt text for accessibility / fallback rendering.
        alt: Option<String>,
    },
    /// An audio block — multi-modal data shape. No provider produces these yet.
    Audio {
        /// MIME type (e.g. `audio/mpeg`, `audio/wav`).
        mime: String,
        /// Inline base64 payload or URL reference.
        data: AudioData,
        /// Optional textual transcript.
        transcript: Option<String>,
    },
}

/// Image payload — either inline base64 bytes or an out-of-band URL.
///
/// Serialized with an internal `kind` tag: `{"kind":"inline",...}` or
/// `{"kind":"url","url":"..."}`. Both variants are struct-shaped so the
/// internal tag works uniformly under serde.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ImageData {
    /// Inline base64-encoded image bytes.
    Inline {
        /// Base64-encoded image bytes.
        base64: String,
        /// Decoded byte length (for budgeting / display).
        bytes: u32,
    },
    /// Out-of-band URL reference (http/https/file).
    Url {
        /// The URL string.
        url: String,
    },
}

/// Audio payload — either inline base64 bytes or an out-of-band URL.
///
/// See [`ImageData`] for the tagging convention.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AudioData {
    /// Inline base64-encoded audio bytes.
    Inline {
        /// Base64-encoded audio bytes.
        base64: String,
        /// Decoded byte length (for budgeting / display).
        bytes: u32,
    },
    /// Out-of-band URL reference (http/https/file).
    Url {
        /// The URL string.
        url: String,
    },
}

impl Block {
    /// Construct an image block backed by a URL reference.
    pub fn image_url(mime: impl Into<String>, url: impl Into<String>) -> Self {
        Self::Image {
            mime: mime.into(),
            data: ImageData::Url { url: url.into() },
            alt: None,
        }
    }

    /// Construct an image block backed by an inline base64 payload.
    ///
    /// `bytes` is the decoded byte length of the payload, computed by the
    /// caller (we deliberately avoid pulling in a base64 decoder here).
    pub fn image_inline_b64(mime: impl Into<String>, b64: impl Into<String>, bytes: u32) -> Self {
        Self::Image {
            mime: mime.into(),
            data: ImageData::Inline {
                base64: b64.into(),
                bytes,
            },
            alt: None,
        }
    }

    /// Construct an audio block backed by a URL reference.
    pub fn audio_url(mime: impl Into<String>, url: impl Into<String>) -> Self {
        Self::Audio {
            mime: mime.into(),
            data: AudioData::Url { url: url.into() },
            transcript: None,
        }
    }

    /// Construct an audio block backed by an inline base64 payload.
    ///
    /// `bytes` is the decoded byte length of the payload, computed by the
    /// caller.
    pub fn audio_inline_b64(mime: impl Into<String>, b64: impl Into<String>, bytes: u32) -> Self {
        Self::Audio {
            mime: mime.into(),
            data: AudioData::Inline {
                base64: b64.into(),
                bytes,
            },
            transcript: None,
        }
    }

    /// Returns true if this block is an `Image` variant.
    #[must_use]
    pub const fn is_image(&self) -> bool {
        matches!(self, Self::Image { .. })
    }

    /// Returns true if this block is an `Audio` variant.
    #[must_use]
    pub const fn is_audio(&self) -> bool {
        matches!(self, Self::Audio { .. })
    }
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

    // --- Existing wire-shape regression: kind tag stays exactly what it was.

    #[test]
    fn text_wire_shape_unchanged() {
        let b = Block::Text { text: "hi".into() };
        let s = serde_json::to_string(&b).unwrap();
        assert_eq!(s, r#"{"kind":"text","text":"hi"}"#);
    }

    #[test]
    fn tool_call_wire_shape_unchanged() {
        let b = Block::ToolCall {
            id: "t1".into(),
            tool: "fs.read".into(),
            args: "{}".into(),
        };
        let s = serde_json::to_string(&b).unwrap();
        assert_eq!(
            s,
            r#"{"kind":"tool_call","id":"t1","tool":"fs.read","args":"{}"}"#
        );
    }

    // --- Image variant tests.

    #[test]
    fn image_inline_roundtrip() {
        let b = Block::Image {
            mime: "image/png".into(),
            data: ImageData::Inline {
                base64: "AAAA".into(),
                bytes: 3,
            },
            alt: Some("a square".into()),
        };
        let s = serde_json::to_string(&b).unwrap();
        let back: Block = serde_json::from_str(&s).unwrap();
        assert_eq!(b, back);
    }

    #[test]
    fn image_url_roundtrip() {
        let b = Block::Image {
            mime: "image/jpeg".into(),
            data: ImageData::Url {
                url: "https://example.com/cat.jpg".into(),
            },
            alt: None,
        };
        let s = serde_json::to_string(&b).unwrap();
        let back: Block = serde_json::from_str(&s).unwrap();
        assert_eq!(b, back);
    }

    #[test]
    fn image_url_constructor() {
        let b = Block::image_url("image/png", "https://example.com/x.png");
        assert!(matches!(
            &b,
            Block::Image { mime, data: ImageData::Url { url }, alt: None }
                if mime == "image/png" && url == "https://example.com/x.png"
        ));
    }

    #[test]
    fn image_inline_b64_constructor() {
        let b = Block::image_inline_b64("image/png", "AAAA", 3);
        assert!(matches!(
            &b,
            Block::Image {
                mime,
                data: ImageData::Inline { base64, bytes: 3 },
                alt: None,
            } if mime == "image/png" && base64 == "AAAA"
        ));
    }

    // --- Audio variant tests.

    #[test]
    fn audio_inline_roundtrip() {
        let b = Block::Audio {
            mime: "audio/wav".into(),
            data: AudioData::Inline {
                base64: "BBBB".into(),
                bytes: 3,
            },
            transcript: Some("hello".into()),
        };
        let s = serde_json::to_string(&b).unwrap();
        let back: Block = serde_json::from_str(&s).unwrap();
        assert_eq!(b, back);
    }

    #[test]
    fn audio_url_roundtrip() {
        let b = Block::Audio {
            mime: "audio/mpeg".into(),
            data: AudioData::Url {
                url: "https://example.com/clip.mp3".into(),
            },
            transcript: None,
        };
        let s = serde_json::to_string(&b).unwrap();
        let back: Block = serde_json::from_str(&s).unwrap();
        assert_eq!(b, back);
    }

    #[test]
    fn audio_url_constructor() {
        let b = Block::audio_url("audio/mpeg", "https://example.com/c.mp3");
        assert!(matches!(
            &b,
            Block::Audio { mime, data: AudioData::Url { url }, transcript: None }
                if mime == "audio/mpeg" && url == "https://example.com/c.mp3"
        ));
    }

    #[test]
    fn audio_inline_b64_constructor() {
        let b = Block::audio_inline_b64("audio/wav", "BBBB", 3);
        assert!(matches!(
            &b,
            Block::Audio {
                mime,
                data: AudioData::Inline { base64, bytes: 3 },
                transcript: None,
            } if mime == "audio/wav" && base64 == "BBBB"
        ));
    }

    // --- Accessor smoke.

    #[test]
    fn is_image_true_for_image_only() {
        assert!(Block::image_url("image/png", "x").is_image());
        assert!(!Block::Text { text: "x".into() }.is_image());
        assert!(!Block::audio_url("audio/wav", "x").is_image());
    }

    #[test]
    fn is_audio_true_for_audio_only() {
        assert!(Block::audio_url("audio/wav", "x").is_audio());
        assert!(!Block::Text { text: "x".into() }.is_audio());
        assert!(!Block::image_url("image/png", "x").is_audio());
    }

    // --- Cross-variant mixed serde.

    #[test]
    fn mixed_block_vec_roundtrip() {
        let blocks = vec![
            Block::Text { text: "hi".into() },
            Block::image_url("image/png", "https://example.com/x.png"),
            Block::audio_inline_b64("audio/wav", "BBBB", 3),
            Block::ToolCall {
                id: "t1".into(),
                tool: "fs.read".into(),
                args: "{}".into(),
            },
        ];
        let s = serde_json::to_string(&blocks).unwrap();
        let back: Vec<Block> = serde_json::from_str(&s).unwrap();
        assert_eq!(blocks, back);
    }

    // --- ImageData / AudioData tagging contract.

    #[test]
    fn image_data_inline_tagged_kind() {
        let d = ImageData::Inline {
            base64: "AAAA".into(),
            bytes: 3,
        };
        let s = serde_json::to_string(&d).unwrap();
        assert_eq!(s, r#"{"kind":"inline","base64":"AAAA","bytes":3}"#);
        let back: ImageData = serde_json::from_str(&s).unwrap();
        assert_eq!(d, back);
    }

    #[test]
    fn audio_data_url_tagged_kind() {
        let d = AudioData::Url {
            url: "https://example.com/c.mp3".into(),
        };
        let s = serde_json::to_string(&d).unwrap();
        assert_eq!(s, r#"{"kind":"url","url":"https://example.com/c.mp3"}"#);
        let back: AudioData = serde_json::from_str(&s).unwrap();
        assert_eq!(d, back);
    }
}
