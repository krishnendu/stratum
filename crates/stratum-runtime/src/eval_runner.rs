//! `EvalRunner` — runs an [`EvalSuite`] (sequence of prompts + expected
//! substrings) against an [`AgentLoop`] and produces an [`EvalReport`].
//!
//! Pure data + composition. The real `claude -p` subprocess judge lands
//! in a follow-up PR; the substring matcher here is the deterministic
//! baseline that the rest of the eval pipeline composes against.
//!
//! ## Algorithm
//!
//! For each [`EvalCase`]:
//!
//! 1. Build a [`TurnContext`] from the case's prompt + the runner's model
//!    + a monotonically-generated turn id.
//! 2. Invoke [`AgentLoop::run_turn`] with a fresh [`CancelToken`].
//! 3. Concatenate every `Block::Text` payload from the returned blocks
//!    (newline-separated for readability) — this is the matcher haystack.
//! 4. Pass requires **all** of:
//!    * `outcome == TurnOutcome::Success`
//!    * every `expected_substrings` entry appears in the haystack
//!    * no `forbidden_substrings` entry appears in the haystack
//!
//! Substring checks are case-sensitive and use `str::contains`. The first
//! failing predicate is reported via [`EvalRun::failure_reason`]; later
//! predicates are not evaluated once a failure is locked in.
//!
//! ## Persistence
//!
//! [`EvalSuite::load`] reads a JSON file produced by some external tool
//! (a docs scraper, a hand-written fixture, etc.) and validates schema
//! version + case-id uniqueness. [`EvalReport::save_atomic`] writes the
//! run summary back out using the `<file>.tmp` + rename pattern shared
//! with [`crate::transcript::TranscriptStore`].

use std::error::Error;
use std::fmt;
use std::io;
use std::path::Path;
use std::sync::Arc;
use std::time::{Instant, SystemTime};

use serde::{Deserialize, Serialize};
use stratum_types::{Block, ModelId};

use crate::agent_loop::{AgentLoop, TurnContext};
use crate::cancel::CancelToken;
use crate::conversation::TurnOutcome;
use crate::observability::TurnId;

// ---------------------------------------------------------------------------
// Schema version
// ---------------------------------------------------------------------------

/// On-disk schema version for [`EvalSuite`]. Bump in lockstep with any
/// breaking JSON-shape change.
pub const EVAL_SUITE_SCHEMA_VERSION: u32 = 1;

// ---------------------------------------------------------------------------
// EvalCase
// ---------------------------------------------------------------------------

/// One prompt + its acceptance predicates.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvalCase {
    /// Unique identifier within the parent [`EvalSuite`].
    pub id: String,
    /// Raw user prompt fed verbatim into the loop's [`TurnContext`].
    pub prompt: String,
    /// Case-sensitive substrings that must **all** appear in the
    /// concatenated `Block::Text` output for the case to pass.
    pub expected_substrings: Vec<String>,
    /// Case-sensitive substrings that must **not** appear in the
    /// concatenated `Block::Text` output.
    pub forbidden_substrings: Vec<String>,
    /// Hard upper bound on the number of blocks the case is willing to
    /// tolerate from the provider. Currently advisory: the runner does
    /// not truncate, but the field is reserved for the future judge
    /// stage that may want to cap a runaway provider.
    pub max_blocks: u8,
}

// ---------------------------------------------------------------------------
// EvalSuite
// ---------------------------------------------------------------------------

/// A named collection of [`EvalCase`]s with a schema marker.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvalSuite {
    /// File-format version; must equal [`EVAL_SUITE_SCHEMA_VERSION`] to
    /// load successfully.
    pub schema_version: u32,
    /// Human-readable suite name; surfaced in [`EvalReport::suite_name`].
    pub name: String,
    /// Cases in run order.
    pub cases: Vec<EvalCase>,
}

impl EvalSuite {
    /// Load + validate a suite from a JSON file at `path`.
    ///
    /// # Errors
    ///
    /// * [`EvalLoadError::Io`] when the file cannot be read.
    /// * [`EvalLoadError::Parse`] when the file is not valid JSON or does
    ///   not match the suite shape.
    /// * [`EvalLoadError::SchemaNewer`] when the file declares a strictly
    ///   newer `schema_version`.
    /// * [`EvalLoadError::DuplicateCaseId`] when two or more cases share an
    ///   id.
    pub fn load(path: &Path) -> Result<Self, EvalLoadError> {
        let raw = std::fs::read(path).map_err(EvalLoadError::Io)?;
        let parsed: Self =
            serde_json::from_slice(&raw).map_err(|e| EvalLoadError::Parse(e.to_string()))?;
        if parsed.schema_version > EVAL_SUITE_SCHEMA_VERSION {
            return Err(EvalLoadError::SchemaNewer {
                found: parsed.schema_version,
                supported: EVAL_SUITE_SCHEMA_VERSION,
            });
        }
        let mut seen: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
        for case in &parsed.cases {
            if !seen.insert(case.id.as_str()) {
                return Err(EvalLoadError::DuplicateCaseId(case.id.clone()));
            }
        }
        Ok(parsed)
    }
}

// ---------------------------------------------------------------------------
// EvalRun
// ---------------------------------------------------------------------------

/// Result of running one [`EvalCase`].
///
/// Intentionally **not** `Eq` — [`Block`] permits payloads (e.g. floats in
/// future variants) that may break structural equality, and the f-prefix
/// `pass_rate` lives on the parent [`EvalReport`] not here.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalRun {
    /// Echoes [`EvalCase::id`].
    pub case_id: String,
    /// `true` iff the case satisfied every acceptance predicate.
    pub passed: bool,
    /// Blocks captured from [`AgentLoop::run_turn`], verbatim.
    pub blocks: Vec<Block>,
    /// Wall-clock duration of `run_turn`, in milliseconds.
    pub duration_ms: u64,
    /// Terminal outcome reported by the loop.
    pub outcome: TurnOutcome,
    /// `Some(reason)` iff `passed == false`. Reason text begins with one
    /// of `outcome`, `missing substring`, or `forbidden substring`.
    pub failure_reason: Option<String>,
}

// ---------------------------------------------------------------------------
// EvalReport
// ---------------------------------------------------------------------------

/// Aggregate result of running an entire [`EvalSuite`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalReport {
    /// Suite name carried over from [`EvalSuite::name`].
    pub suite_name: String,
    /// Wall-clock timestamp captured before the first case ran.
    pub ran_at: SystemTime,
    /// Per-case results in the order they were executed.
    pub runs: Vec<EvalRun>,
    /// Convenience counter mirroring `runs.iter().filter(|r| r.passed).count()`.
    pub passed: u32,
    /// Convenience counter mirroring `runs.len() - passed`.
    pub failed: u32,
    /// Sum of `runs[*].duration_ms`.
    pub total_duration_ms: u64,
}

impl EvalReport {
    /// Pass rate as a fraction in `[0.0, 1.0]`. An empty report returns
    /// `0.0` (never NaN — denominator is clamped to at least 1).
    #[must_use]
    pub fn pass_rate(&self) -> f32 {
        let denom = self.runs.len().max(1);
        // Cast widths: `passed` is `u32`, `denom` is `usize` clamped to
        // at least 1. The lossy cast is intentional and well-defined for
        // realistic counts (we never run more than `u32::MAX` cases).
        #[allow(
            clippy::cast_precision_loss,
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss
        )]
        {
            self.passed as f32 / denom as f32
        }
    }

    /// Atomically write `self` as pretty-printed JSON to `path` using a
    /// `<file>.tmp` + rename swap.
    ///
    /// # Errors
    ///
    /// * [`EvalReportError::Serialize`] when JSON serialization fails.
    /// * [`EvalReportError::Io`] for any underlying filesystem error.
    pub fn save_atomic(&self, path: &Path) -> Result<(), EvalReportError> {
        let bytes = serde_json::to_vec_pretty(self)
            .map_err(|e| EvalReportError::Serialize(e.to_string()))?;
        let mut tmp_name = path
            .file_name()
            .ok_or_else(|| {
                EvalReportError::Io(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "save path is missing a file name",
                ))
            })?
            .to_os_string();
        tmp_name.push(".tmp");
        let tmp = path.with_file_name(tmp_name);

        {
            use std::io::Write;
            let mut f = std::fs::OpenOptions::new()
                .create(true)
                .truncate(true)
                .write(true)
                .open(&tmp)
                .map_err(EvalReportError::Io)?;
            f.write_all(&bytes).map_err(EvalReportError::Io)?;
            f.sync_all().map_err(EvalReportError::Io)?;
        }

        if let Err(e) = std::fs::rename(&tmp, path) {
            let _ = std::fs::remove_file(&tmp);
            return Err(EvalReportError::Io(e));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// EvalRunner
// ---------------------------------------------------------------------------

/// Drives an [`EvalSuite`] against a shared [`AgentLoop`].
pub struct EvalRunner {
    loop_: Arc<AgentLoop>,
    model: ModelId,
}

impl fmt::Debug for EvalRunner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EvalRunner")
            .field("model", &self.model)
            .finish_non_exhaustive()
    }
}

impl EvalRunner {
    /// Wire a runner around an [`AgentLoop`] and the model id every case
    /// is run against.
    #[must_use]
    pub const fn new(loop_: Arc<AgentLoop>, model: ModelId) -> Self {
        Self { loop_, model }
    }

    /// Run one case end-to-end.
    #[must_use]
    pub fn run_one(&self, case: &EvalCase) -> EvalRun {
        let ctx = TurnContext {
            user_prompt: case.prompt.clone(),
            model: self.model.clone(),
            turn_id: TurnId(0),
            started_at: SystemTime::now(),
            history: Vec::new(),
        };
        let cancel = CancelToken::new();
        let started = Instant::now();
        let result = self.loop_.run_turn(ctx, &cancel);
        let duration_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);

        let haystack = concat_text(&result.blocks);
        let (passed, failure_reason) = classify(case, &result.outcome, &haystack);

        EvalRun {
            case_id: case.id.clone(),
            passed,
            blocks: result.blocks,
            duration_ms,
            outcome: result.outcome,
            failure_reason,
        }
    }

    /// Run every case in `suite` and aggregate into an [`EvalReport`].
    #[must_use]
    pub fn run_suite(&self, suite: &EvalSuite) -> EvalReport {
        let ran_at = SystemTime::now();
        let mut runs: Vec<EvalRun> = Vec::with_capacity(suite.cases.len());
        let mut passed: u32 = 0;
        let mut failed: u32 = 0;
        let mut total_duration_ms: u64 = 0;
        for case in &suite.cases {
            let run = self.run_one(case);
            if run.passed {
                passed = passed.saturating_add(1);
            } else {
                failed = failed.saturating_add(1);
            }
            total_duration_ms = total_duration_ms.saturating_add(run.duration_ms);
            runs.push(run);
        }
        EvalReport {
            suite_name: suite.name.clone(),
            ran_at,
            runs,
            passed,
            failed,
            total_duration_ms,
        }
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors surfaced by [`EvalSuite::load`].
#[derive(Debug)]
pub enum EvalLoadError {
    /// Underlying filesystem error.
    Io(io::Error),
    /// JSON parse / shape mismatch. Inner string is the serde error's
    /// `Display`.
    Parse(String),
    /// File declares a strictly newer schema version than this binary
    /// understands.
    SchemaNewer {
        /// Version found on disk.
        found: u32,
        /// Highest version this binary supports.
        supported: u32,
    },
    /// Two or more cases share the same id.
    DuplicateCaseId(String),
}

impl fmt::Display for EvalLoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "EvalSuite load io error: {e}"),
            Self::Parse(msg) => write!(f, "EvalSuite parse error: {msg}"),
            Self::SchemaNewer { found, supported } => write!(
                f,
                "EvalSuite schema_version {found} is newer than supported {supported}"
            ),
            Self::DuplicateCaseId(id) => write!(f, "EvalSuite duplicate case id: {id}"),
        }
    }
}

impl Error for EvalLoadError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

/// Errors surfaced by [`EvalReport::save_atomic`].
#[derive(Debug)]
pub enum EvalReportError {
    /// Underlying filesystem error.
    Io(io::Error),
    /// JSON serialization failed. Inner string is the serde error's
    /// `Display`.
    Serialize(String),
}

impl fmt::Display for EvalReportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "EvalReport save io error: {e}"),
            Self::Serialize(msg) => write!(f, "EvalReport serialize error: {msg}"),
        }
    }
}

impl Error for EvalReportError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Serialize(_) => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Concatenate every `Block::Text` payload, newline-separated.
fn concat_text(blocks: &[Block]) -> String {
    let mut out = String::new();
    for block in blocks {
        if let Block::Text { text } = block {
            if !out.is_empty() {
                out.push('\n');
            }
            out.push_str(text);
        }
    }
    out
}

/// Returns `(passed, failure_reason)` for one case.
///
/// Order of checks: outcome, then missing-expected, then forbidden. The
/// first failing predicate wins; subsequent predicates are not evaluated.
fn classify(case: &EvalCase, outcome: &TurnOutcome, haystack: &str) -> (bool, Option<String>) {
    if !matches!(outcome, TurnOutcome::Success) {
        return (
            false,
            Some(format!("outcome: expected Success, got {outcome:?}")),
        );
    }
    for expected in &case.expected_substrings {
        if !haystack.contains(expected.as_str()) {
            return (false, Some(format!("missing substring: {expected}")));
        }
    }
    for forbidden in &case.forbidden_substrings {
        if haystack.contains(forbidden.as_str()) {
            return (false, Some(format!("forbidden substring: {forbidden}")));
        }
    }
    (true, None)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    use stratum_types::Capability;

    use crate::agent_factory::AgentFactory;
    use crate::agent_loop::{AgentLoop, AgentLoopConfig};
    use crate::event_log::{EventEmitter, MemoryEventSink};
    use crate::intent_router::IntentRouter;
    use crate::permission_prompt::{AllowAllResponder, PermissionStore, PromptIdGen};
    use crate::plan_mode::PlanMode;
    use crate::provider::{GenerateRequest, Provider};
    use crate::tool_invocation::RegistryDispatcher;
    use crate::tools::CapabilityMatrix;

    fn echo_runner() -> EvalRunner {
        let loop_ = Arc::new(AgentFactory::echo().unwrap());
        EvalRunner::new(loop_, ModelId::from("echo"))
    }

    fn case(id: &str, prompt: &str, expected: &[&str], forbidden: &[&str]) -> EvalCase {
        EvalCase {
            id: id.into(),
            prompt: prompt.into(),
            expected_substrings: expected.iter().map(|s| (*s).to_string()).collect(),
            forbidden_substrings: forbidden.iter().map(|s| (*s).to_string()).collect(),
            max_blocks: 32,
        }
    }

    // ---- ZeroBlockProvider for "outcome != Success" coverage ---------

    #[derive(Debug)]
    struct ZeroBlockProvider;
    impl Provider for ZeroBlockProvider {
        #[allow(clippy::unnecessary_literal_bound)]
        fn id(&self) -> &str {
            "zero"
        }
        fn capabilities(&self) -> &'static [Capability] {
            const CAPS: &[Capability] = &[Capability::Generate];
            CAPS
        }
        fn generate(&self, _req: &GenerateRequest, _cancel: &CancelToken) -> Vec<Block> {
            Vec::new()
        }
    }

    fn zero_block_runner() -> EvalRunner {
        // Hand-wire an AgentLoop directly so the factory's defaults
        // don't drag in unrelated dispatchers.
        let sink = Arc::new(MemoryEventSink::new());
        let loop_ = AgentLoop::builder()
            .with_provider(Arc::new(ZeroBlockProvider))
            .with_router(IntentRouter::empty())
            .with_permission_store(Arc::new(PermissionStore::new()))
            .with_prompt_gen(Arc::new(PromptIdGen::new()))
            .with_responder(Arc::new(AllowAllResponder))
            .with_events(Arc::new(EventEmitter::new(sink)))
            .with_capability_matrix(Arc::new(CapabilityMatrix::new()))
            .with_plan_mode(Arc::new(PlanMode::new()))
            .with_dispatcher(Arc::new(RegistryDispatcher::new()))
            .with_config(AgentLoopConfig::default())
            .build()
            .unwrap();
        EvalRunner::new(Arc::new(loop_), ModelId::from("zero"))
    }

    // ---- serde round-trips ------------------------------------------

    #[test]
    fn eval_case_serde_roundtrip() {
        let c = case("c1", "hello", &["hi"], &["nope"]);
        let s = serde_json::to_string(&c).unwrap();
        let back: EvalCase = serde_json::from_str(&s).unwrap();
        assert_eq!(c, back);
    }

    #[test]
    fn eval_suite_serde_roundtrip() {
        let s = EvalSuite {
            schema_version: EVAL_SUITE_SCHEMA_VERSION,
            name: "demo".into(),
            cases: vec![
                case("a", "alpha", &["a"], &[]),
                case("b", "beta", &["b"], &[]),
            ],
        };
        let json = serde_json::to_string(&s).unwrap();
        let back: EvalSuite = serde_json::from_str(&json).unwrap();
        assert_eq!(s, back);
    }

    #[test]
    fn eval_report_serde_roundtrip() {
        let runner = echo_runner();
        let report = runner.run_suite(&EvalSuite {
            schema_version: EVAL_SUITE_SCHEMA_VERSION,
            name: "round".into(),
            cases: vec![case("p", "hello world", &["hello"], &[])],
        });
        let json = serde_json::to_string(&report).unwrap();
        let back: EvalReport = serde_json::from_str(&json).unwrap();
        assert_eq!(report.suite_name, back.suite_name);
        assert_eq!(report.passed, back.passed);
        assert_eq!(report.failed, back.failed);
        assert_eq!(report.runs.len(), back.runs.len());
    }

    // ---- EvalSuite::load ---------------------------------------------

    #[test]
    fn eval_suite_load_happy_three_cases() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("suite.json");
        let suite = EvalSuite {
            schema_version: EVAL_SUITE_SCHEMA_VERSION,
            name: "happy".into(),
            cases: vec![
                case("c1", "p1", &["p"], &[]),
                case("c2", "p2", &["p"], &[]),
                case("c3", "p3", &["p"], &[]),
            ],
        };
        std::fs::write(&p, serde_json::to_vec_pretty(&suite).unwrap()).unwrap();
        let loaded = EvalSuite::load(&p).unwrap();
        assert_eq!(loaded.cases.len(), 3);
    }

    #[test]
    fn eval_suite_load_rejects_duplicate_case_id() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("dup.json");
        let suite = EvalSuite {
            schema_version: EVAL_SUITE_SCHEMA_VERSION,
            name: "dup".into(),
            cases: vec![case("same", "p1", &[], &[]), case("same", "p2", &[], &[])],
        };
        std::fs::write(&p, serde_json::to_vec_pretty(&suite).unwrap()).unwrap();
        let err = EvalSuite::load(&p).expect_err("duplicate must fail");
        assert!(matches!(&err, EvalLoadError::DuplicateCaseId(id) if id == "same"));
    }

    #[test]
    fn eval_suite_load_rejects_newer_schema() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("newer.json");
        let body = serde_json::json!({
            "schema_version": 999,
            "name": "future",
            "cases": [],
        });
        std::fs::write(&p, serde_json::to_vec_pretty(&body).unwrap()).unwrap();
        let err = EvalSuite::load(&p).expect_err("newer schema must fail");
        assert!(matches!(
            &err,
            EvalLoadError::SchemaNewer {
                found: 999,
                supported: EVAL_SUITE_SCHEMA_VERSION
            }
        ));
    }

    #[test]
    fn eval_suite_load_rejects_malformed_json() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("bad.json");
        std::fs::write(&p, b"{not json").unwrap();
        let err = EvalSuite::load(&p).expect_err("malformed must fail");
        assert!(matches!(err, EvalLoadError::Parse(_)));
    }

    #[test]
    fn eval_suite_load_rejects_missing_file() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("does-not-exist.json");
        let err = EvalSuite::load(&p).expect_err("missing file must fail");
        assert!(matches!(err, EvalLoadError::Io(_)));
    }

    // ---- run_one classification --------------------------------------

    #[test]
    fn run_one_expected_substring_passes() {
        let runner = echo_runner();
        let c = case("ok", "hello world", &["hello"], &[]);
        let r = runner.run_one(&c);
        assert!(
            r.passed,
            "expected pass, got failure: {:?}",
            r.failure_reason
        );
        assert!(r.failure_reason.is_none());
        assert_eq!(r.case_id, "ok");
    }

    #[test]
    fn run_one_missing_expected_substring_fails() {
        let runner = echo_runner();
        let c = case("miss", "hello world", &["zzz"], &[]);
        let r = runner.run_one(&c);
        assert!(!r.passed);
        let reason = r.failure_reason.expect("failure reason set");
        assert!(
            reason.starts_with("missing substring:"),
            "reason was {reason}"
        );
    }

    #[test]
    fn run_one_forbidden_substring_fails() {
        let runner = echo_runner();
        let c = case("forb", "hello world", &[], &["world"]);
        let r = runner.run_one(&c);
        assert!(!r.passed);
        let reason = r.failure_reason.expect("failure reason set");
        assert!(
            reason.starts_with("forbidden substring:"),
            "reason was {reason}"
        );
    }

    #[test]
    fn run_one_both_expected_and_forbidden_satisfied_passes() {
        let runner = echo_runner();
        let c = case("both", "alpha bravo", &["alpha"], &["zzz"]);
        let r = runner.run_one(&c);
        assert!(r.passed, "expected pass, got: {:?}", r.failure_reason);
    }

    #[test]
    fn run_one_outcome_not_success_fails() {
        let runner = zero_block_runner();
        let c = case("zero", "anything", &[], &[]);
        let r = runner.run_one(&c);
        assert!(!r.passed);
        let reason = r.failure_reason.expect("failure reason set");
        assert!(reason.starts_with("outcome:"), "reason was {reason}");
    }

    // ---- run_suite aggregation ---------------------------------------

    #[test]
    fn run_suite_all_pass() {
        let runner = echo_runner();
        let suite = EvalSuite {
            schema_version: EVAL_SUITE_SCHEMA_VERSION,
            name: "all-pass".into(),
            cases: vec![
                case("a", "hello", &["hello"], &[]),
                case("b", "world", &["world"], &[]),
                case("c", "foo bar", &["foo"], &[]),
            ],
        };
        let report = runner.run_suite(&suite);
        assert_eq!(report.passed, 3);
        assert_eq!(report.failed, 0);
        assert!((report.pass_rate() - 1.0).abs() < f32::EPSILON);
        assert_eq!(report.suite_name, "all-pass");
        assert_eq!(report.runs.len(), 3);
    }

    #[test]
    fn run_suite_mixed_pass_fail() {
        let runner = echo_runner();
        let suite = EvalSuite {
            schema_version: EVAL_SUITE_SCHEMA_VERSION,
            name: "mixed".into(),
            cases: vec![
                case("a", "hello world", &["hello"], &[]),
                case("b", "hello world", &["world"], &[]),
                case("c", "hello world", &["never-there"], &[]),
            ],
        };
        let report = runner.run_suite(&suite);
        assert_eq!(report.passed, 2);
        assert_eq!(report.failed, 1);
        let rate = report.pass_rate();
        assert!((rate - (2.0 / 3.0)).abs() < 0.01, "rate {rate} not ~0.66");
    }

    // ---- pass_rate edge cases ----------------------------------------

    #[test]
    fn pass_rate_empty_report_is_zero() {
        let report = EvalReport {
            suite_name: "empty".into(),
            ran_at: SystemTime::now(),
            runs: Vec::new(),
            passed: 0,
            failed: 0,
            total_duration_ms: 0,
        };
        assert!((report.pass_rate() - 0.0).abs() < f32::EPSILON);
    }

    // ---- save_atomic round-trip --------------------------------------

    #[test]
    fn save_atomic_roundtrip() {
        let runner = echo_runner();
        let suite = EvalSuite {
            schema_version: EVAL_SUITE_SCHEMA_VERSION,
            name: "save".into(),
            cases: vec![case("a", "hello world", &["hello"], &[])],
        };
        let report = runner.run_suite(&suite);

        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("report.json");
        report.save_atomic(&p).unwrap();

        let raw = std::fs::read(&p).unwrap();
        let back: EvalReport = serde_json::from_slice(&raw).unwrap();
        assert_eq!(back.suite_name, "save");
        assert_eq!(back.passed, 1);
        assert_eq!(back.failed, 0);
    }

    #[test]
    fn save_atomic_rejects_path_without_file_name() {
        // A path that ends in `..` has no `file_name()`; this exercises
        // the `EvalReportError::Io { ErrorKind::InvalidInput }` branch.
        let report = EvalReport {
            suite_name: "noname".into(),
            ran_at: SystemTime::now(),
            runs: Vec::new(),
            passed: 0,
            failed: 0,
            total_duration_ms: 0,
        };
        let dir = tempfile::tempdir().unwrap();
        let weird = dir.path().join("..");
        let err = report
            .save_atomic(&weird)
            .expect_err("path without file_name must fail");
        assert!(
            matches!(&err, EvalReportError::Io(e) if e.kind() == io::ErrorKind::InvalidInput),
            "unexpected error: {err}",
        );
    }

    #[test]
    fn zero_block_provider_descriptors() {
        // Exercise the test-only provider's metadata so it counts toward
        // coverage on the module's own helpers.
        let p = ZeroBlockProvider;
        assert_eq!(p.id(), "zero");
        assert!(p.capabilities().contains(&Capability::Generate));
    }

    #[test]
    fn save_atomic_rejects_directory_path() {
        // Passing a path whose parent doesn't exist forces the
        // OpenOptions::open call to return an `io::Error`, exercising
        // the `EvalReportError::Io` arm.
        let report = EvalReport {
            suite_name: "x".into(),
            ran_at: SystemTime::now(),
            runs: Vec::new(),
            passed: 0,
            failed: 0,
            total_duration_ms: 0,
        };
        let bogus = Path::new("/this/path/definitely/does/not/exist/report.json");
        let err = report.save_atomic(bogus).expect_err("bogus path must fail");
        assert!(matches!(err, EvalReportError::Io(_)));
    }

    // ---- Display smoke ------------------------------------------------

    #[test]
    fn eval_load_error_display_smoke() {
        let cases: Vec<EvalLoadError> = vec![
            EvalLoadError::Io(io::Error::new(io::ErrorKind::NotFound, "nf")),
            EvalLoadError::Parse("bad".into()),
            EvalLoadError::SchemaNewer {
                found: 9,
                supported: 1,
            },
            EvalLoadError::DuplicateCaseId("dup".into()),
        ];
        for e in &cases {
            let s = format!("{e}");
            assert!(!s.is_empty(), "Display empty for {e:?}");
        }
        // Source: only `Io` has a source; the rest return None.
        assert!(cases[0].source().is_some());
        assert!(cases[1].source().is_none());
        assert!(cases[2].source().is_none());
        assert!(cases[3].source().is_none());
    }

    #[test]
    fn eval_report_error_display_smoke() {
        let cases: Vec<EvalReportError> = vec![
            EvalReportError::Io(io::Error::new(io::ErrorKind::PermissionDenied, "pd")),
            EvalReportError::Serialize("oops".into()),
        ];
        for e in &cases {
            let s = format!("{e}");
            assert!(!s.is_empty(), "Display empty for {e:?}");
        }
        assert!(cases[0].source().is_some());
        assert!(cases[1].source().is_none());
    }

    // ---- Send + Sync smoke -------------------------------------------

    #[test]
    fn eval_runner_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<EvalRunner>();
    }

    // ---- Debug smoke -------------------------------------------------

    #[test]
    fn eval_runner_debug_does_not_panic() {
        let runner = echo_runner();
        let s = format!("{runner:?}");
        assert!(s.contains("EvalRunner"));
    }

    // ---- concat_text helper -----------------------------------------

    #[test]
    fn concat_text_joins_text_blocks_only() {
        let blocks = vec![
            Block::Text { text: "a".into() },
            Block::Done,
            Block::Text { text: "b".into() },
            Block::Usage {
                prompt: 1,
                completion: 1,
            },
        ];
        let out = concat_text(&blocks);
        assert_eq!(out, "a\nb");
    }
}
