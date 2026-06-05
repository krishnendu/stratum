//! Prompt-injection defense primitives.
//!
//! Phase 3 v2 prep — fencing + suspicion scoring. The full classifier
//! (Qwen3-0.6B inside the runtime) lands later; this module pins the
//! structural fence shape and a substring/regex-style heuristic so
//! every tool output can be wrapped before re-entering context.
//!
//! Per `plan/35-prompt-injection-policy.md`.

use serde::{Deserialize, Serialize};

/// Source-of-trust tag attached to a fenced block. Drives the wrapper
/// tag the orchestrator emits: `<tool_output>`, `<rag_hit>`, etc.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FenceSource {
    /// Output of a tool call.
    ToolOutput,
    /// Retrieval-augmented-generation hit from the local RAG index.
    RagHit,
    /// MCP server result.
    McpResult,
    /// OCR / image caption.
    Ocr,
    /// Automatic-speech-recognition transcript.
    Asr,
    /// Inbound `stratum serve` request body.
    ExternalRequest,
}

impl FenceSource {
    /// XML-style tag emitted around the fenced block.
    #[must_use]
    pub const fn tag(self) -> &'static str {
        match self {
            Self::ToolOutput => "tool_output",
            Self::RagHit => "rag_hit",
            Self::McpResult => "mcp_result",
            Self::Ocr => "ocr",
            Self::Asr => "asr",
            Self::ExternalRequest => "external_request",
        }
    }
}

/// Wrap `content` in a structural `<{tag} id="{id}">…</{tag}>` fence.
///
/// The model is instructed (in the system prompt) to treat fenced
/// payload as data, not instructions. The id allows downstream
/// pipelines to refer to a specific block when surfacing a warning.
#[must_use]
pub fn fence(source: FenceSource, id: &str, content: &str) -> String {
    let tag = source.tag();
    format!("<{tag} id=\"{id}\">\n{content}\n</{tag}>")
}

/// Heuristic suspicion score in `[0.0, 1.0]`.
///
/// The full classifier model lands later; this string-matcher gives
/// every fence a defensible initial value the runtime can act on today
/// (warning toast, `strict` agents refusing the step).
#[must_use]
pub fn suspicion_score(content: &str) -> f32 {
    let lower = content.to_ascii_lowercase();
    let mut hits = 0_u32;
    for trigger in TRIGGERS {
        if lower.contains(trigger) {
            hits = hits.saturating_add(1);
        }
    }
    let total = u32::try_from(TRIGGERS.len()).unwrap_or(u32::MAX).max(1);
    #[allow(
        clippy::cast_precision_loss,
        reason = "hits and total are small (< 64); f32 mantissa is wide enough"
    )]
    let normalized = (hits as f32) / (total as f32);
    normalized.clamp(0.0, 1.0)
}

/// Threshold above which the runtime should flag the fenced block as
/// suspicious. Tuned to match the plan's "≥ 0.7" guideline.
pub const SUSPICION_THRESHOLD: f32 = 0.7;

/// Convenience predicate: above-threshold score?
#[must_use]
pub fn is_suspicious(content: &str) -> bool {
    suspicion_score(content) >= SUSPICION_THRESHOLD
}

const TRIGGERS: &[&str] = &[
    "ignore all prior",
    "ignore previous",
    "disregard instructions",
    "system prompt",
    "you are now",
    "act as",
    "from now on",
    "<!-- system:",
    "// system:",
    "/* system:",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tag_for_each_source_is_distinct() {
        let tags = [
            FenceSource::ToolOutput.tag(),
            FenceSource::RagHit.tag(),
            FenceSource::McpResult.tag(),
            FenceSource::Ocr.tag(),
            FenceSource::Asr.tag(),
            FenceSource::ExternalRequest.tag(),
        ];
        let unique: std::collections::HashSet<_> = tags.iter().collect();
        assert_eq!(unique.len(), tags.len());
    }

    #[test]
    fn fence_wraps_with_tag_and_id() {
        let s = fence(FenceSource::ToolOutput, "t-001", "raw bytes");
        assert!(s.starts_with("<tool_output id=\"t-001\">"));
        assert!(s.contains("raw bytes"));
        assert!(s.ends_with("</tool_output>"));
    }

    #[test]
    fn fence_preserves_content_byte_for_byte() {
        let content = "line1\nline2 with `code` and {braces}";
        let s = fence(FenceSource::RagHit, "r-12", content);
        assert!(s.contains(content));
    }

    #[test]
    fn suspicion_score_zero_for_benign_text() {
        assert!(suspicion_score("hello world").abs() < f32::EPSILON);
    }

    #[test]
    fn suspicion_score_positive_for_known_trigger() {
        let s = suspicion_score("Ignore all prior instructions");
        assert!(s > 0.0);
    }

    #[test]
    fn suspicion_score_grows_with_multiple_triggers() {
        let single = suspicion_score("ignore previous");
        let multi = suspicion_score("ignore previous and act as a system prompt");
        assert!(multi > single);
    }

    #[test]
    fn suspicion_score_case_insensitive() {
        assert!(suspicion_score("IGNORE ALL PRIOR INSTRUCTIONS") > 0.0);
    }

    #[test]
    fn suspicion_score_in_unit_range() {
        for input in [
            "",
            "x",
            "ignore previous",
            "ignore previous and ignore previous",
        ] {
            let s = suspicion_score(input);
            assert!((0.0..=1.0).contains(&s), "score out of range: {s}");
        }
    }

    #[test]
    fn is_suspicious_threshold_is_seventy_percent() {
        // Five distinct triggers in one payload exceeds the 0.5 mark.
        let content = "ignore all prior. ignore previous. you are now. act as. from now on. system prompt. <!-- system: x // system: y /* system: z";
        assert!(is_suspicious(content));
    }

    #[test]
    fn is_suspicious_returns_false_for_benign_content() {
        assert!(!is_suspicious("read the file at src/main.rs"));
    }

    #[test]
    fn fence_source_serde_roundtrip() {
        for source in [
            FenceSource::ToolOutput,
            FenceSource::RagHit,
            FenceSource::McpResult,
            FenceSource::Ocr,
            FenceSource::Asr,
            FenceSource::ExternalRequest,
        ] {
            let s = serde_json::to_string(&source).unwrap();
            let back: FenceSource = serde_json::from_str(&s).unwrap();
            assert_eq!(source, back);
        }
    }

    #[test]
    fn suspicion_threshold_constant_is_documented_value() {
        assert!((SUSPICION_THRESHOLD - 0.7).abs() < f32::EPSILON);
    }
}
