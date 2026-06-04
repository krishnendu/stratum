//! Shared test utilities for the Stratum workspace.
//!
//! The golden-transcript format is documented in `plan/12-tui-rust-design.md`
//! and elaborated by tests under `tests/integration/`.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use serde::{Deserialize, Serialize};
use stratum_types::Block;

/// One step in a golden transcript: an input message followed by the
/// sequence of `Block`s the runtime is expected to emit.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TranscriptStep {
    /// User-supplied input for this turn.
    pub input: String,
    /// Expected ordered output stream.
    pub expect: Vec<Block>,
}

/// A complete fixture: a named transcript with one or more steps.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GoldenTranscript {
    /// Stable name; matches the fixture directory.
    pub name: String,
    /// Steps in order.
    pub steps: Vec<TranscriptStep>,
}

impl GoldenTranscript {
    /// Parse a transcript from JSON.
    ///
    /// # Errors
    /// Returns the underlying serde error if the JSON does not match the schema.
    pub fn from_json(json: &str) -> serde_json::Result<Self> {
        serde_json::from_str(json)
    }

    /// Serialize a transcript to pretty JSON.
    ///
    /// # Errors
    /// Returns the underlying serde error if serialization fails (impossible in
    /// practice for owned types but surfaced for hygiene).
    pub fn to_json(&self) -> serde_json::Result<String> {
        serde_json::to_string_pretty(self)
    }

    /// Compare an actual emitted stream against the expected one for step `idx`.
    /// Returns `Ok(())` on match, or a structured mismatch report.
    ///
    /// # Errors
    /// Returns a [`TranscriptMismatch`] describing the divergence.
    pub fn assert_step(&self, idx: usize, actual: &[Block]) -> Result<(), TranscriptMismatch> {
        let step = self
            .steps
            .get(idx)
            .ok_or(TranscriptMismatch::OutOfRange { idx })?;
        if step.expect == actual {
            Ok(())
        } else {
            Err(TranscriptMismatch::Diverged {
                idx,
                expected: step.expect.clone(),
                actual: actual.to_vec(),
            })
        }
    }
}

/// Structured comparison failure.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum TranscriptMismatch {
    /// Step index outside the transcript.
    #[error("transcript step {idx} out of range")]
    OutOfRange {
        /// Requested index.
        idx: usize,
    },
    /// Step produced output different from the expected stream.
    #[error("transcript step {idx} diverged from expected stream")]
    Diverged {
        /// Step index.
        idx: usize,
        /// Expected stream.
        expected: Vec<Block>,
        /// Actual stream.
        actual: Vec<Block>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> GoldenTranscript {
        GoldenTranscript {
            name: "hello".into(),
            steps: vec![TranscriptStep {
                input: "hi".into(),
                expect: vec![
                    Block::Text {
                        text: "hello".into(),
                    },
                    Block::Done,
                ],
            }],
        }
    }

    #[test]
    fn json_roundtrip() {
        let t = sample();
        let s = t.to_json().unwrap();
        let back = GoldenTranscript::from_json(&s).unwrap();
        assert_eq!(t, back);
    }

    #[test]
    fn assert_step_matches() {
        let t = sample();
        let actual = vec![
            Block::Text {
                text: "hello".into(),
            },
            Block::Done,
        ];
        assert!(t.assert_step(0, &actual).is_ok());
    }

    #[test]
    fn assert_step_diverges() {
        let t = sample();
        let actual = vec![
            Block::Text {
                text: "wrong".into(),
            },
            Block::Done,
        ];
        let err = t.assert_step(0, &actual).unwrap_err();
        assert!(matches!(err, TranscriptMismatch::Diverged { idx: 0, .. }));
    }

    #[test]
    fn assert_step_out_of_range() {
        let t = sample();
        let actual: Vec<Block> = vec![];
        let err = t.assert_step(7, &actual).unwrap_err();
        assert!(matches!(err, TranscriptMismatch::OutOfRange { idx: 7 }));
    }

    #[test]
    fn mismatch_display() {
        let oor = TranscriptMismatch::OutOfRange { idx: 3 };
        assert!(format!("{oor}").contains("out of range"));
        let div = TranscriptMismatch::Diverged {
            idx: 1,
            expected: vec![],
            actual: vec![],
        };
        assert!(format!("{div}").contains("diverged"));
    }
}
