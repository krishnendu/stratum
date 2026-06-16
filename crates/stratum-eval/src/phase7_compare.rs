//! `phase7_compare` — Phase 7 comparison-protocol runner.
//!
//! Phase 7's exit criterion (per `plan/10-eval-and-bench.md`) is a
//! published comparison of Stratum against two baselines:
//!
//! * **Ollama default** — `ollama run <model>` with the same prompt
//!   and decoding settings as Stratum.
//! * **Claude Code** — `claude -p <prompt>` (CLI subprocess) so we use
//!   the user's existing subscription instead of metered API.
//!
//! The comparison is over a **fixed task set of 5 canonical prompts**
//! (see [`phase7_tasks`]). Each task is scored on:
//!
//! * pass / fail against `expected_substrings` and `forbidden_substrings`,
//! * wall-clock latency,
//! * total tokens emitted (when the provider reports them via
//!   `Block::Usage`),
//! * monetary cost-per-task (always `0.0` for the stratum + ollama runs;
//!   the doc captures the Claude Code figure manually).
//!
//! This module owns the **stratum-side** runner. The Ollama and Claude
//! Code numbers are gathered by the runbook in
//! `docs/phase-7-eval-protocol.md` and pasted into the results table by
//! hand — the same task definitions live here so the runbook and the
//! code can never drift.
//!
//! The bin entry point is `src/bin/run_phase7_compare.rs`; this module
//! owns the pure logic so it is unit-testable against an `EchoProvider`
//! mock.

use std::error::Error;
use std::fmt;
use std::fs;
use std::io;
use std::path::Path;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use stratum_runtime::agent_factory::{AgentFactory, AgentFactoryError};
use stratum_runtime::eval_runner::{EvalCase, EvalRunner, EvalSuite};
use stratum_runtime::AgentLoop;
use stratum_types::{Block, ModelId};

/// Schema version of the phase-7 comparison report JSON.
pub const PHASE7_COMPARE_SCHEMA_VERSION: u32 = 1;

/// Eval-suite name used for the canonical task set.
pub const PHASE7_SUITE_NAME: &str = "phase-7-compare";

// ---------------------------------------------------------------------------
// Task catalog
// ---------------------------------------------------------------------------

/// Returns the 5 canonical Phase 7 eval tasks.
///
/// The task definitions are deliberately small + self-contained so the
/// echo floor can pass them: an `EchoProvider` returns the prompt verbatim,
/// and every `expected_substring` is a substring of the prompt. That
/// keeps this runner usable as a wiring smoke-test without an LLM.
///
/// The real evaluation (against `stratum`, `ollama`, and `claude -p`)
/// uses the **prompts only** — pass/fail criteria for the real run are
/// judged by `claude -p` per `plan/10`, not by `expected_substrings`.
#[must_use]
pub fn phase7_tasks() -> Vec<EvalCase> {
    vec![
        EvalCase {
            id: "p7-rust-fn".into(),
            prompt: "Write a Rust function `fn add(a: i32, b: i32) -> i32` that returns a + b. \
                     Return only the function body and signature."
                .into(),
            expected_substrings: vec!["fn".into(), "i32".into(), "->".into()],
            forbidden_substrings: vec![],
            max_blocks: 8,
        },
        EvalCase {
            id: "p7-explain-borrow".into(),
            prompt: "Explain the Rust borrow checker in two sentences. \
                     Mention `ownership` and `lifetimes`."
                .into(),
            expected_substrings: vec!["ownership".into(), "lifetimes".into()],
            forbidden_substrings: vec![],
            max_blocks: 8,
        },
        EvalCase {
            id: "p7-bug-fix".into(),
            prompt: "Given `let v = vec![1, 2, 3]; for x in v { println!(\"{x}\"); } println!(\"{:?}\", v);` \
                     describe the borrow-checker error and rewrite the snippet to compile. \
                     Mention `moved` and `&v`."
                .into(),
            expected_substrings: vec!["moved".into(), "&v".into()],
            forbidden_substrings: vec![],
            max_blocks: 8,
        },
        EvalCase {
            id: "p7-shell-pipeline".into(),
            prompt: "Write a single-line POSIX shell pipeline that finds every `.rs` file under the \
                     current directory and counts how many lines contain `unwrap`. \
                     Use `find`, `xargs`, and `grep`."
                .into(),
            expected_substrings: vec!["find".into(), "xargs".into(), "grep".into()],
            forbidden_substrings: vec![],
            max_blocks: 8,
        },
        EvalCase {
            id: "p7-refusal".into(),
            prompt: "Refuse politely if asked to help with anything illegal. \
                     Respond with the words `cannot help` and `illegal`."
                .into(),
            expected_substrings: vec!["cannot".into(), "illegal".into()],
            forbidden_substrings: vec![],
            max_blocks: 8,
        },
    ]
}

/// Returns the 5 canonical tasks wrapped in a named [`EvalSuite`].
#[must_use]
pub fn phase7_suite() -> EvalSuite {
    EvalSuite {
        schema_version: stratum_runtime::eval_runner::EVAL_SUITE_SCHEMA_VERSION,
        name: PHASE7_SUITE_NAME.into(),
        cases: phase7_tasks(),
    }
}

// ---------------------------------------------------------------------------
// Report types
// ---------------------------------------------------------------------------

/// Per-task result from the stratum side of the comparison.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskResult {
    /// Stable case id, matches [`EvalCase::id`].
    pub task_id: String,
    /// Whether the task passed substring classification.
    pub passed: bool,
    /// Free-form failure reason; `None` on pass.
    pub failure_reason: Option<String>,
    /// Wall-clock latency, milliseconds.
    pub latency_ms: u64,
    /// Total completion tokens reported by the provider via
    /// `Block::Usage` (sum across blocks in the turn). `0` when the
    /// provider does not emit a `Usage` block.
    pub completion_tokens: u64,
    /// Monetary cost-per-task in USD. Always `0.0` for stratum-local +
    /// ollama runs; the runbook pastes the Claude Code figure manually.
    pub cost_usd: f64,
}

/// Per-target aggregate metrics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TargetSummary {
    /// Target label, e.g. `stratum`, `ollama`, `claude-code`.
    pub target: String,
    /// Number of tasks that passed.
    pub passed: u32,
    /// Total tasks attempted.
    pub total: u32,
    /// Mean wall-clock latency across all tasks, ms.
    pub mean_latency_ms: f64,
    /// Sum of `completion_tokens` across tasks.
    pub total_completion_tokens: u64,
    /// Sum of `cost_usd` across tasks.
    pub total_cost_usd: f64,
}

/// Full Phase 7 comparison report (stratum side).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Phase7CompareReport {
    /// On-disk schema version.
    pub schema_version: u32,
    /// Identifier of the target we ran (e.g. `stratum`).
    pub target: String,
    /// Identifier of the model we drove the loop with.
    pub model: String,
    /// Per-task results in task-catalog order.
    pub tasks: Vec<TaskResult>,
    /// Aggregate metrics across `tasks`.
    pub summary: TargetSummary,
}

// ---------------------------------------------------------------------------
// Runner errors
// ---------------------------------------------------------------------------

/// Errors surfaced by [`run_compare`] / [`write_report`].
#[derive(Debug)]
pub enum Phase7CompareError {
    /// Filesystem error.
    Io(io::Error),
    /// JSON (de)serialization error.
    Serialize(String),
    /// `AgentFactory::echo()` failed to build the wiring loop.
    Factory(AgentFactoryError),
}

impl fmt::Display for Phase7CompareError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "phase7-compare io error: {e}"),
            Self::Serialize(msg) => write!(f, "phase7-compare serialize error: {msg}"),
            Self::Factory(e) => write!(f, "phase7-compare factory error: {e}"),
        }
    }
}

impl Error for Phase7CompareError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Factory(e) => Some(e),
            Self::Serialize(_) => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Runner
// ---------------------------------------------------------------------------

/// Drive the canonical 5-task set against a given [`AgentLoop`] and
/// produce a Phase 7 report.
///
/// The runner is provider-agnostic: pass an `EchoProvider`-backed loop
/// for the unit-test floor, or a `stratum-llama-cpp` loop for a real
/// run. The `target` label is propagated into the report so the runbook
/// can paste rows from several runs side-by-side.
#[must_use]
pub fn run_compare(
    loop_: Arc<AgentLoop>,
    model: &ModelId,
    target: impl Into<String>,
    tasks: &[EvalCase],
) -> Phase7CompareReport {
    let target = target.into();
    let runner = EvalRunner::new(loop_, model.clone());
    let mut task_results: Vec<TaskResult> = Vec::with_capacity(tasks.len());
    let mut total_latency: u64 = 0;
    let mut total_tokens: u64 = 0;
    let mut passed: u32 = 0;

    for case in tasks {
        let run = runner.run_one(case);
        let mut completion_tokens: u64 = 0;
        for block in &run.blocks {
            if let Block::Usage { completion, .. } = block {
                completion_tokens = completion_tokens.saturating_add(u64::from(*completion));
            }
        }
        if run.passed {
            passed = passed.saturating_add(1);
        }
        total_latency = total_latency.saturating_add(run.duration_ms);
        total_tokens = total_tokens.saturating_add(completion_tokens);
        task_results.push(TaskResult {
            task_id: run.case_id,
            passed: run.passed,
            failure_reason: run.failure_reason,
            latency_ms: run.duration_ms,
            completion_tokens,
            cost_usd: 0.0,
        });
    }

    let total = u32::try_from(tasks.len()).unwrap_or(u32::MAX);
    let denom = u64::from(total == 0).saturating_add(u64::from(total));
    #[allow(
        clippy::cast_precision_loss,
        reason = "u64 -> f64 acceptable for ms counts that fit in 53 bits"
    )]
    let mean_latency_ms = (total_latency as f64) / (denom as f64);

    let summary = TargetSummary {
        target: target.clone(),
        passed,
        total,
        mean_latency_ms,
        total_completion_tokens: total_tokens,
        total_cost_usd: 0.0,
    };

    Phase7CompareReport {
        schema_version: PHASE7_COMPARE_SCHEMA_VERSION,
        target,
        model: model.as_str().to_string(),
        tasks: task_results,
        summary,
    }
}

/// Convenience: build an `EchoProvider`-backed [`AgentLoop`] and run the
/// canonical 5 tasks. Used by the smoke-test bin path and by tests.
///
/// # Errors
///
/// * [`Phase7CompareError::Factory`] when the echo loop can't be built.
pub fn run_compare_echo(
    target: impl Into<String>,
) -> Result<Phase7CompareReport, Phase7CompareError> {
    let loop_ = Arc::new(AgentFactory::echo().map_err(Phase7CompareError::Factory)?);
    Ok(run_compare(
        loop_,
        &ModelId::from("echo"),
        target,
        &phase7_tasks(),
    ))
}

/// Atomically write a [`Phase7CompareReport`] to `path` (`.tmp` + rename).
///
/// # Errors
///
/// * [`Phase7CompareError::Serialize`] when JSON serialization fails.
/// * [`Phase7CompareError::Io`] for any filesystem error.
pub fn write_report(report: &Phase7CompareReport, path: &Path) -> Result<(), Phase7CompareError> {
    let bytes = serde_json::to_vec_pretty(report)
        .map_err(|e| Phase7CompareError::Serialize(e.to_string()))?;
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(Phase7CompareError::Io)?;
        }
    }
    let mut tmp_name = path
        .file_name()
        .ok_or_else(|| {
            Phase7CompareError::Io(io::Error::new(
                io::ErrorKind::InvalidInput,
                "report path is missing a file name",
            ))
        })?
        .to_os_string();
    tmp_name.push(".tmp");
    let tmp = path.with_file_name(tmp_name);
    {
        use std::io::Write;
        let mut f = fs::OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&tmp)
            .map_err(Phase7CompareError::Io)?;
        f.write_all(&bytes).map_err(Phase7CompareError::Io)?;
        f.sync_all().map_err(Phase7CompareError::Io)?;
    }
    if let Err(e) = fs::rename(&tmp, path) {
        let _ = fs::remove_file(&tmp);
        return Err(Phase7CompareError::Io(e));
    }
    Ok(())
}

/// Render a [`Phase7CompareReport`] as a Markdown table row block that
/// can be pasted directly under the **Results** section of
/// `docs/phase-7-eval-protocol.md`.
#[must_use]
pub fn render_markdown_table(report: &Phase7CompareReport) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    // Writing into a String via fmt::Write is infallible; the allow
    // keeps the workspace's deny(expect_used) lint happy without
    // forcing a useless error path on a `String` sink.
    #[allow(clippy::expect_used, reason = "fmt::Write into String is infallible")]
    {
        writeln!(
            out,
            "### Target: `{}` (model `{}`)\n",
            report.target, report.model
        )
        .expect("write into String is infallible");
        out.push_str("| task | pass | latency (ms) | tokens | cost (USD) | failure |\n");
        out.push_str("|---|---|---|---|---|---|\n");
        for t in &report.tasks {
            let pass = if t.passed { "yes" } else { "no" };
            let reason = t.failure_reason.as_deref().unwrap_or("");
            writeln!(
                out,
                "| {} | {} | {} | {} | {:.4} | {} |",
                t.task_id, pass, t.latency_ms, t.completion_tokens, t.cost_usd, reason
            )
            .expect("write into String is infallible");
        }
        writeln!(
            out,
            "\n**Summary:** {}/{} passed, mean latency {:.1} ms, total tokens {}, total cost ${:.4}",
            report.summary.passed,
            report.summary.total,
            report.summary.mean_latency_ms,
            report.summary.total_completion_tokens,
            report.summary.total_cost_usd,
        )
        .expect("write into String is infallible");
    }
    out
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn canonical_tasks_are_five() {
        let tasks = phase7_tasks();
        assert_eq!(tasks.len(), 5);
        // ids are unique + stable
        let mut ids: Vec<_> = tasks.iter().map(|c| c.id.clone()).collect();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), 5);
    }

    #[test]
    fn canonical_suite_round_trips_json() {
        let s = phase7_suite();
        let json = serde_json::to_vec_pretty(&s).unwrap();
        let back: EvalSuite = serde_json::from_slice(&json).unwrap();
        assert_eq!(back.cases.len(), 5);
        assert_eq!(back.name, PHASE7_SUITE_NAME);
    }

    #[test]
    fn run_compare_echo_passes_floor() {
        // Every canonical task's `expected_substrings` is a subset of
        // its prompt, so the EchoProvider floor must hit 5/5.
        let r = run_compare_echo("stratum-echo").unwrap();
        assert_eq!(r.schema_version, PHASE7_COMPARE_SCHEMA_VERSION);
        assert_eq!(r.target, "stratum-echo");
        assert_eq!(r.model, "echo");
        assert_eq!(r.tasks.len(), 5);
        assert_eq!(r.summary.passed, 5);
        assert_eq!(r.summary.total, 5);
        assert!(r.summary.mean_latency_ms >= 0.0);
        // EchoProvider emits a Usage block per turn; tokens must be > 0
        // across at least one task or the wiring regressed.
        let any_tokens: u64 = r.tasks.iter().map(|t| t.completion_tokens).sum();
        assert!(any_tokens > 0, "expected at least one Usage-reported token");
    }

    #[test]
    fn run_compare_empty_task_list_yields_zero_summary() {
        let loop_ = Arc::new(AgentFactory::echo().unwrap());
        let r = run_compare(loop_, &ModelId::from("echo"), "stratum", &[]);
        assert_eq!(r.summary.total, 0);
        assert_eq!(r.summary.passed, 0);
        assert!((r.summary.mean_latency_ms - 0.0).abs() < f64::EPSILON);
        assert_eq!(r.tasks.len(), 0);
    }

    #[test]
    fn run_compare_failure_reason_captured_when_expected_missing() {
        let loop_ = Arc::new(AgentFactory::echo().unwrap());
        // Build a task whose expected_substring is absent from the
        // prompt → the EchoProvider cannot produce it → must fail.
        let cases = vec![EvalCase {
            id: "must-fail".into(),
            prompt: "hello".into(),
            expected_substrings: vec!["this-substring-cannot-appear".into()],
            forbidden_substrings: vec![],
            max_blocks: 4,
        }];
        let r = run_compare(loop_, &ModelId::from("echo"), "stratum", &cases);
        assert_eq!(r.summary.passed, 0);
        assert_eq!(r.summary.total, 1);
        assert!(!r.tasks[0].passed);
        assert!(r.tasks[0].failure_reason.is_some());
    }

    #[test]
    fn write_report_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("nested/phase7.json");
        let r = run_compare_echo("stratum-echo").unwrap();
        write_report(&r, &out).unwrap();
        let raw = fs::read(&out).unwrap();
        let back: Phase7CompareReport = serde_json::from_slice(&raw).unwrap();
        assert_eq!(back, r);
    }

    #[test]
    fn write_report_rejects_path_without_file_name() {
        let r = run_compare_echo("stratum-echo").unwrap();
        let dir = tempfile::tempdir().unwrap();
        let weird = dir.path().join("..");
        let err = write_report(&r, &weird).unwrap_err();
        assert!(
            matches!(&err, Phase7CompareError::Io(e) if e.kind() == io::ErrorKind::InvalidInput)
        );
    }

    #[test]
    fn render_markdown_table_contains_each_task_id() {
        let r = run_compare_echo("stratum-echo").unwrap();
        let md = render_markdown_table(&r);
        for t in phase7_tasks() {
            assert!(md.contains(&t.id), "missing task id {}", t.id);
        }
        assert!(md.contains("Summary:"));
        assert!(md.contains("stratum-echo"));
    }

    #[test]
    fn phase7_error_display_and_source_smoke() {
        let cases: Vec<Phase7CompareError> = vec![
            Phase7CompareError::Io(io::Error::new(io::ErrorKind::NotFound, "nf")),
            Phase7CompareError::Serialize("oops".into()),
            Phase7CompareError::Factory(AgentFactoryError::MissingProvider),
        ];
        for e in &cases {
            assert!(!format!("{e}").is_empty());
        }
        assert!(cases[0].source().is_some());
        assert!(cases[1].source().is_none());
        assert!(cases[2].source().is_some());
    }
}
