//! Reviewer pass — scores an assistant draft against a fixed
//! checklist using a SEPARATE provider so the same model doesn't
//! grade itself.
//!
//! Plan/17 §Critic explicitly calls out anti-self-bias:
//!
//! > **Use any reasoner other than the producer model**, so the
//! > critic doesn't echo the producer's blind spots.
//!
//! Default config: reviewer **off**. When enabled, the caller wires
//! a second [`Arc<dyn Provider>`] (often a smaller distilled model
//! whose system prompt is [`crate::prompts::REVIEWER`]). The
//! reviewer's verdict surfaces as an event the TUI can render —
//! the agent loop itself doesn't retry or modify the draft based
//! on it (Phase 1 scope: signal, not gate).
//!
//! ## Output shape
//!
//! The reviewer system prompt forces a one-line JSON object. We
//! parse it lazily — a malformed reply is treated as "no opinion"
//! rather than a critical failure. The reviewer's role is advisory.

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use stratum_types::{Block, ModelId};

use crate::cancel::CancelToken;
use crate::provider::{ChatHistoryTurn, GenerateRequest, Provider, SamplerParams};

/// Verdict severity returned by the reviewer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    /// Cosmetic or wording nit.
    Low,
    /// Factual error or missed step; recoverable on next turn.
    Medium,
    /// Dangerous action, security violation, would corrupt data.
    High,
}

/// One issue flagged by the reviewer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReviewIssue {
    /// Short, user-readable description.
    pub message: String,
}

/// Parsed reviewer reply.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReviewVerdict {
    /// Either `clean` or `fix`.
    pub verdict: String,
    /// One entry per issue, in checklist-emission order.
    pub issues: Vec<ReviewIssue>,
    /// Worst severity across the issues.
    pub severity: Severity,
}

impl ReviewVerdict {
    /// True when the reviewer said "clean" with no issues.
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.verdict == "clean" && self.issues.is_empty()
    }
}

/// One reviewer pass driven by a dedicated provider.
#[derive(Debug, Clone)]
pub struct ReviewerPass {
    provider: Arc<dyn Provider>,
    model: ModelId,
}

impl ReviewerPass {
    /// Build a reviewer pass over a given provider + model id.
    /// Caller is responsible for ensuring the provider runs a
    /// different model than the producer (anti-self-bias).
    #[must_use]
    pub fn new(provider: Arc<dyn Provider>, model: ModelId) -> Self {
        Self { provider, model }
    }

    /// Score a draft assistant reply. The user prompt and the draft
    /// are sent verbatim; the reviewer's system prompt is forced via
    /// `system_override` so the producer's system prompt doesn't
    /// confuse the reviewer's role.
    ///
    /// Returns `None` when the reviewer reply isn't parseable —
    /// that's intentional, the reviewer is advisory, not load-bearing.
    pub fn review(
        &self,
        user_prompt: &str,
        draft: &str,
        cancel: &CancelToken,
    ) -> Option<ReviewVerdict> {
        let history = vec![
            ChatHistoryTurn {
                role: "user".to_string(),
                content: format!(
                    "User asked: {user_prompt}\n\nAssistant draft to review:\n{draft}"
                ),
            },
        ];
        let req = GenerateRequest {
            model: self.model.clone(),
            prompt: "Score the draft above.".to_string(),
            max_blocks: 2,
            system_override: Some(crate::prompts::REVIEWER.to_string()),
            history,
            // Deterministic-ish: low temperature, narrow top_p so the
            // verdict is stable across calls on the same draft.
            sampler: SamplerParams {
                temperature: Some(0.2),
                top_p: Some(0.85),
                repeat_penalty: None,
            },
        };
        let blocks = self.provider.generate(&req, cancel);
        let text = blocks
            .iter()
            .filter_map(|b| match b {
                Block::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("");
        parse_reviewer_reply(&text)
    }
}

/// Parse the reviewer's reply. Lenient: extracts the first JSON
/// object that contains a `verdict` key. Returns `None` for malformed
/// or empty replies.
fn parse_reviewer_reply(reply: &str) -> Option<ReviewVerdict> {
    let trimmed = reply.trim();
    // Find the first `{` and its balanced close.
    let start = trimmed.find('{')?;
    let body = &trimmed[start..];
    let v: serde_json::Value = serde_json::from_str(body).ok().or_else(|| {
        // Try parsing up to the matching close.
        find_balanced_close(&body[1..]).and_then(|end| {
            serde_json::from_str::<serde_json::Value>(&body[..=end + 1]).ok()
        })
    })?;
    let obj = v.as_object()?;
    let verdict = obj.get("verdict")?.as_str()?.to_string();
    let severity = match obj.get("severity").and_then(|x| x.as_str()) {
        Some("high") => Severity::High,
        Some("medium") => Severity::Medium,
        _ => Severity::Low,
    };
    let issues = obj
        .get("issues")
        .and_then(|x| x.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|x| x.as_str())
                .map(|s| ReviewIssue {
                    message: s.to_string(),
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    Some(ReviewVerdict {
        verdict,
        issues,
        severity,
    })
}

fn find_balanced_close(s: &str) -> Option<usize> {
    let mut depth = 1_i32;
    let mut in_string = false;
    let mut escape = false;
    for (i, b) in s.bytes().enumerate() {
        if escape {
            escape = false;
            continue;
        }
        match b {
            b'\\' if in_string => escape = true,
            b'"' => in_string = !in_string,
            b'{' if !in_string => depth += 1,
            b'}' if !in_string => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::EchoProvider;

    #[test]
    fn parse_reviewer_reply_clean_verdict() {
        let r = parse_reviewer_reply(
            r#"{"verdict":"clean","issues":[],"severity":"low"}"#,
        )
        .unwrap();
        assert_eq!(r.verdict, "clean");
        assert!(r.is_clean());
        assert_eq!(r.severity, Severity::Low);
    }

    #[test]
    fn parse_reviewer_reply_fix_with_issues() {
        let r = parse_reviewer_reply(
            r#"{"verdict":"fix","issues":["hallucinated path src/missing.rs","missed tool"],"severity":"medium"}"#,
        )
        .unwrap();
        assert_eq!(r.verdict, "fix");
        assert_eq!(r.issues.len(), 2);
        assert!(r.issues[0].message.contains("hallucinated"));
        assert_eq!(r.severity, Severity::Medium);
        assert!(!r.is_clean());
    }

    #[test]
    fn parse_reviewer_reply_high_severity() {
        let r = parse_reviewer_reply(
            r#"{"verdict":"fix","issues":["dangerous shell command"],"severity":"high"}"#,
        )
        .unwrap();
        assert_eq!(r.severity, Severity::High);
    }

    #[test]
    fn parse_reviewer_reply_malformed_returns_none() {
        assert!(parse_reviewer_reply("not json at all").is_none());
        assert!(parse_reviewer_reply("{ broken").is_none());
        assert!(parse_reviewer_reply("").is_none());
    }

    #[test]
    fn parse_reviewer_reply_extracts_from_surrounding_prose() {
        // Reviewer ignores the system prompt rule and adds prose:
        let r = parse_reviewer_reply(
            r#"Here's my assessment: {"verdict":"clean","issues":[],"severity":"low"}. Cheers!"#,
        )
        .unwrap();
        assert_eq!(r.verdict, "clean");
    }

    #[test]
    fn parse_reviewer_reply_missing_verdict_field_returns_none() {
        // No verdict key → can't grade.
        assert!(parse_reviewer_reply(r#"{"issues":[]}"#).is_none());
    }

    #[test]
    fn parse_reviewer_reply_unknown_severity_defaults_to_low() {
        let r = parse_reviewer_reply(
            r#"{"verdict":"fix","issues":["x"],"severity":"galactic"}"#,
        )
        .unwrap();
        assert_eq!(r.severity, Severity::Low);
    }

    #[test]
    fn severity_serializes_lowercase() {
        let s = serde_json::to_string(&Severity::High).unwrap();
        assert_eq!(s, "\"high\"");
    }

    #[test]
    fn review_verdict_clean_helper_requires_no_issues() {
        let v = ReviewVerdict {
            verdict: "clean".to_string(),
            issues: vec![ReviewIssue {
                message: "x".to_string(),
            }],
            severity: Severity::Low,
        };
        assert!(!v.is_clean(), "clean verdict with issues is not actually clean");
    }

    #[test]
    fn reviewer_with_echo_provider_returns_none() {
        // EchoProvider just echoes the prompt — that's not valid
        // reviewer JSON, so review() should return None gracefully.
        let provider: Arc<dyn Provider> = Arc::new(EchoProvider::new("echo: "));
        let r = ReviewerPass::new(provider, ModelId::from("echo"));
        let cancel = CancelToken::new();
        let verdict = r.review("hi", "Hello!", &cancel);
        assert!(verdict.is_none());
    }
}
