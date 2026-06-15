//! `bench-floor` — deterministic nightly regression bench.
//!
//! Runs every `evals/<suite>.json` file in a given directory through
//! [`stratum_runtime::eval_runner::EvalRunner`] backed by the synthetic
//! `EchoProvider`. The output is a single JSON file with per-suite
//! pass-rate, mean / p95 wall-clock latency, mean completion-token
//! count, and total run duration — enough to drive a regression-tracking
//! dashboard without depending on a real model.
//!
//! The bin entry point is in `src/bin/bench_floor.rs`; this module owns
//! the pure logic so it's unit-testable.

use std::error::Error;
use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use stratum_runtime::agent_factory::{AgentFactory, AgentFactoryError};
use stratum_runtime::eval_runner::{EvalLoadError, EvalRun, EvalRunner, EvalSuite};
use stratum_types::{Block, ModelId};

/// Schema version of the bench-floor result JSON.
pub const BENCH_FLOOR_SCHEMA_VERSION: u32 = 1;

/// Per-suite metrics captured by one bench run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SuiteMetric {
    /// Suite name (echoed from [`EvalSuite::name`]).
    pub suite_name: String,
    /// Source path the suite was loaded from, relative if possible.
    pub source: String,
    /// Number of cases that passed.
    pub passed: u32,
    /// Total number of cases run.
    pub total: u32,
    /// `passed / total` as a fraction in `[0.0, 1.0]`. Zero for empty
    /// suites (denominator clamped to 1).
    pub pass_rate: f32,
    /// Mean per-case wall-clock latency, in milliseconds.
    pub mean_wall_clock_ms: f64,
    /// 95th-percentile per-case wall-clock latency, in milliseconds.
    pub p95_wall_clock_ms: u64,
    /// Mean completion tokens reported via `Block::Usage` across cases
    /// that emitted at least one usage block; zero otherwise.
    pub mean_completion_tokens: f64,
    /// Sum of every case's wall-clock duration, in milliseconds.
    pub total_duration_ms: u64,
}

/// One full bench-floor run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchFloorResult {
    /// On-disk schema version. Must equal [`BENCH_FLOOR_SCHEMA_VERSION`].
    pub schema_version: u32,
    /// Per-suite results in alphabetical suite-path order.
    pub suites: Vec<SuiteMetric>,
    /// Sum of every suite's `total_duration_ms`.
    pub total_duration_ms: u64,
}

/// Errors surfaced by [`run_bench`] / [`write_result`].
#[derive(Debug)]
pub enum BenchFloorError {
    /// Filesystem error.
    Io(io::Error),
    /// JSON (de)serialization error.
    Serialize(String),
    /// A suite file failed to load.
    SuiteLoad {
        /// Path of the offending suite file.
        path: PathBuf,
        /// Underlying load error.
        source: EvalLoadError,
    },
    /// `AgentFactory::echo()` failed to build.
    Factory(AgentFactoryError),
}

impl fmt::Display for BenchFloorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "bench-floor io error: {e}"),
            Self::Serialize(msg) => write!(f, "bench-floor serialize error: {msg}"),
            Self::SuiteLoad { path, source } => write!(
                f,
                "bench-floor suite load failed for {}: {source}",
                path.display()
            ),
            Self::Factory(e) => write!(f, "bench-floor echo factory error: {e}"),
        }
    }
}

impl Error for BenchFloorError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::SuiteLoad { source, .. } => Some(source),
            Self::Factory(e) => Some(e),
            Self::Serialize(_) => None,
        }
    }
}

/// Discover suite files (`*.json` and `*.toml`) under `evals_dir`,
/// non-recursive. Returns paths sorted lexicographically so the output
/// JSON is byte-stable across machines.
///
/// # Errors
///
/// Returns [`BenchFloorError::Io`] when the directory cannot be read.
pub fn discover_suites(evals_dir: &Path) -> Result<Vec<PathBuf>, BenchFloorError> {
    let mut out: Vec<PathBuf> = Vec::new();
    let entries = fs::read_dir(evals_dir).map_err(BenchFloorError::Io)?;
    for entry in entries {
        let entry = entry.map_err(BenchFloorError::Io)?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(ext) = path.extension().and_then(|e| e.to_str()) else {
            continue;
        };
        if ext == "json" || ext == "toml" {
            out.push(path);
        }
    }
    out.sort();
    Ok(out)
}

/// Run the bench against the given list of suite paths.
///
/// The runner is `EchoProvider`-backed; every suite is loaded via
/// [`EvalSuite::load`] and dispatched through [`EvalRunner::run_suite`].
///
/// # Errors
///
/// * [`BenchFloorError::Factory`] when the echo loop can't be built.
/// * [`BenchFloorError::SuiteLoad`] when any suite file fails to load.
pub fn run_bench(suite_paths: &[PathBuf]) -> Result<BenchFloorResult, BenchFloorError> {
    let loop_ = Arc::new(AgentFactory::echo().map_err(BenchFloorError::Factory)?);
    let runner = EvalRunner::new(loop_, ModelId::from("echo"));

    let mut suites: Vec<SuiteMetric> = Vec::with_capacity(suite_paths.len());
    let mut total_duration_ms: u64 = 0;
    let bench_started = Instant::now();

    for path in suite_paths {
        let suite = EvalSuite::load(path).map_err(|source| BenchFloorError::SuiteLoad {
            path: path.clone(),
            source,
        })?;
        let report = runner.run_suite(&suite);
        let metric = summarize(path, &suite, &report.runs, report.passed, report.failed);
        total_duration_ms = total_duration_ms.saturating_add(metric.total_duration_ms);
        suites.push(metric);
    }

    // The wall-clock of the bench loop itself is informational only —
    // we use the sum-of-suites figure (`total_duration_ms`) as the
    // primary metric because it's reproducible across machine speeds.
    let _bench_elapsed = bench_started.elapsed();

    Ok(BenchFloorResult {
        schema_version: BENCH_FLOOR_SCHEMA_VERSION,
        suites,
        total_duration_ms,
    })
}

/// Atomically write a [`BenchFloorResult`] to `path`. Uses the same
/// `<file>.tmp` + rename pattern as `EvalReport::save_atomic`.
///
/// # Errors
///
/// * [`BenchFloorError::Serialize`] when JSON serialization fails.
/// * [`BenchFloorError::Io`] for any filesystem error.
pub fn write_result(result: &BenchFloorResult, path: &Path) -> Result<(), BenchFloorError> {
    let bytes =
        serde_json::to_vec_pretty(result).map_err(|e| BenchFloorError::Serialize(e.to_string()))?;
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(BenchFloorError::Io)?;
        }
    }
    let mut tmp_name = path
        .file_name()
        .ok_or_else(|| {
            BenchFloorError::Io(io::Error::new(
                io::ErrorKind::InvalidInput,
                "result path is missing a file name",
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
            .map_err(BenchFloorError::Io)?;
        f.write_all(&bytes).map_err(BenchFloorError::Io)?;
        f.sync_all().map_err(BenchFloorError::Io)?;
    }
    if let Err(e) = fs::rename(&tmp, path) {
        let _ = fs::remove_file(&tmp);
        return Err(BenchFloorError::Io(e));
    }
    Ok(())
}

fn summarize(
    source: &Path,
    suite: &EvalSuite,
    runs: &[EvalRun],
    passed: u32,
    failed: u32,
) -> SuiteMetric {
    let total = passed.saturating_add(failed);
    let denom_total = u32::from(total == 0).saturating_add(total);
    #[allow(
        clippy::cast_precision_loss,
        reason = "u32 -> f32 lossy cast is intentional for ratios in [0,1]"
    )]
    let pass_rate = passed as f32 / denom_total as f32;

    // Collect per-case durations + completion-token totals.
    let mut durations: Vec<u64> = runs.iter().map(|r| r.duration_ms).collect();
    let total_duration_ms = durations.iter().copied().fold(0u64, u64::saturating_add);
    let mean_wall_clock_ms = if durations.is_empty() {
        0.0
    } else {
        #[allow(
            clippy::cast_precision_loss,
            reason = "u64 -> f64 acceptable for ms counts that fit comfortably in 53 bits"
        )]
        {
            (total_duration_ms as f64) / (durations.len() as f64)
        }
    };
    durations.sort_unstable();
    let p95_wall_clock_ms = percentile_u64(&durations, 95);

    let mut completion_total: u64 = 0;
    let mut completion_samples: u64 = 0;
    for run in runs {
        for block in &run.blocks {
            if let Block::Usage { completion, .. } = block {
                completion_total = completion_total.saturating_add(u64::from(*completion));
                completion_samples = completion_samples.saturating_add(1);
            }
        }
    }
    let mean_completion_tokens = if completion_samples == 0 {
        0.0
    } else {
        #[allow(
            clippy::cast_precision_loss,
            reason = "u64 -> f64 acceptable for token counts that fit in 53 bits"
        )]
        {
            (completion_total as f64) / (completion_samples as f64)
        }
    };

    SuiteMetric {
        suite_name: suite.name.clone(),
        source: source.display().to_string(),
        passed,
        total,
        pass_rate,
        mean_wall_clock_ms,
        p95_wall_clock_ms,
        mean_completion_tokens,
        total_duration_ms,
    }
}

/// Nearest-rank percentile on a pre-sorted slice. Returns `0` for an
/// empty slice and the last element for `pct == 100`.
fn percentile_u64(sorted: &[u64], pct: u8) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let len = sorted.len();
    // nearest-rank: ceil(pct/100 * len) -> 1-based index.
    let pct = u32::from(pct.min(100));
    // index = ceil(pct * len / 100), clamped to [1, len].
    #[allow(
        clippy::cast_possible_truncation,
        reason = "len bounded by realistic suite sizes; u32 is sufficient"
    )]
    let len_u32 = len as u32;
    let num = pct.saturating_mul(len_u32);
    let mut idx = num.div_ceil(100);
    if idx == 0 {
        idx = 1;
    }
    if (idx as usize) > len {
        idx = len_u32;
    }
    sorted[(idx as usize) - 1]
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use stratum_runtime::eval_runner::{EvalCase, EVAL_SUITE_SCHEMA_VERSION};

    fn write_suite(dir: &Path, name: &str, suite: &EvalSuite) -> PathBuf {
        let p = dir.join(format!("{name}.json"));
        fs::write(&p, serde_json::to_vec_pretty(suite).unwrap()).unwrap();
        p
    }

    fn case(id: &str, prompt: &str, expected: &[&str]) -> EvalCase {
        EvalCase {
            id: id.into(),
            prompt: prompt.into(),
            expected_substrings: expected.iter().map(|s| (*s).into()).collect(),
            forbidden_substrings: Vec::new(),
            max_blocks: 4,
        }
    }

    #[test]
    fn percentile_basics() {
        assert_eq!(percentile_u64(&[], 95), 0);
        assert_eq!(percentile_u64(&[5], 95), 5);
        assert_eq!(percentile_u64(&[1, 2, 3, 4, 5], 95), 5);
        assert_eq!(percentile_u64(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 95), 10);
        // pct=0 still returns the smallest element (idx clamped to 1).
        assert_eq!(percentile_u64(&[1, 2, 3], 0), 1);
    }

    #[test]
    fn discover_suites_sorted_and_filtered() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("z.json"), b"{}").unwrap();
        fs::write(dir.path().join("a.json"), b"{}").unwrap();
        fs::write(dir.path().join("m.toml"), b"").unwrap();
        fs::write(dir.path().join("ignored.txt"), b"").unwrap();
        let suites = discover_suites(dir.path()).unwrap();
        let names: Vec<_> = suites
            .iter()
            .map(|p| p.file_name().unwrap().to_str().unwrap().to_string())
            .collect();
        assert_eq!(names, vec!["a.json", "m.toml", "z.json"]);
    }

    #[test]
    fn discover_suites_missing_dir_errors() {
        let dir = tempfile::tempdir().unwrap();
        let bogus = dir.path().join("nope");
        let err = discover_suites(&bogus).unwrap_err();
        assert!(matches!(err, BenchFloorError::Io(_)));
    }

    #[test]
    fn run_bench_synthetic_suite_passes() {
        let dir = tempfile::tempdir().unwrap();
        let suite = EvalSuite {
            schema_version: EVAL_SUITE_SCHEMA_VERSION,
            name: "synth".into(),
            cases: vec![
                case("a", "hello world", &["hello"]),
                case("b", "foo bar", &["foo", "bar"]),
            ],
        };
        let path = write_suite(dir.path(), "synth", &suite);
        let res = run_bench(&[path]).unwrap();
        assert_eq!(res.schema_version, BENCH_FLOOR_SCHEMA_VERSION);
        assert_eq!(res.suites.len(), 1);
        let s = &res.suites[0];
        assert_eq!(s.suite_name, "synth");
        assert_eq!(s.passed, 2);
        assert_eq!(s.total, 2);
        assert!((s.pass_rate - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn run_bench_load_error_surfaces() {
        let dir = tempfile::tempdir().unwrap();
        let bad = dir.path().join("bad.json");
        fs::write(&bad, b"{not json").unwrap();
        let err = run_bench(std::slice::from_ref(&bad)).unwrap_err();
        match err {
            BenchFloorError::SuiteLoad { path, .. } => assert_eq!(path, bad),
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn write_result_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("nested/result.json");
        let r = BenchFloorResult {
            schema_version: BENCH_FLOOR_SCHEMA_VERSION,
            suites: vec![SuiteMetric {
                suite_name: "x".into(),
                source: "evals/x.json".into(),
                passed: 1,
                total: 1,
                pass_rate: 1.0,
                mean_wall_clock_ms: 0.0,
                p95_wall_clock_ms: 0,
                mean_completion_tokens: 0.0,
                total_duration_ms: 0,
            }],
            total_duration_ms: 0,
        };
        write_result(&r, &out).unwrap();
        let raw = fs::read(&out).unwrap();
        let back: BenchFloorResult = serde_json::from_slice(&raw).unwrap();
        assert_eq!(back, r);
    }

    #[test]
    fn write_result_rejects_path_without_file_name() {
        let r = BenchFloorResult {
            schema_version: BENCH_FLOOR_SCHEMA_VERSION,
            suites: Vec::new(),
            total_duration_ms: 0,
        };
        let dir = tempfile::tempdir().unwrap();
        let weird = dir.path().join("..");
        let err = write_result(&r, &weird).unwrap_err();
        assert!(matches!(&err, BenchFloorError::Io(e) if e.kind() == io::ErrorKind::InvalidInput));
    }

    #[test]
    fn bench_floor_error_display_smoke() {
        let cases: Vec<BenchFloorError> = vec![
            BenchFloorError::Io(io::Error::new(io::ErrorKind::NotFound, "nf")),
            BenchFloorError::Serialize("oops".into()),
            BenchFloorError::SuiteLoad {
                path: PathBuf::from("evals/x.json"),
                source: EvalLoadError::Parse("bad".into()),
            },
        ];
        for e in &cases {
            assert!(!format!("{e}").is_empty());
        }
        assert!(cases[0].source().is_some());
        assert!(cases[1].source().is_none());
        assert!(cases[2].source().is_some());
    }

    /// e2e against a real on-disk suite — runs `evals/baseline.json` if
    /// it exists at the repo root (it does in CI). Skipped silently if
    /// not so the unit-test crate stays standalone.
    #[test]
    fn e2e_real_evals_baseline_if_present() {
        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap();
        let evals = repo_root.join("evals");
        if !evals.is_dir() {
            return;
        }
        let suites = discover_suites(&evals).unwrap();
        // baseline.json is checked in; the bench should not crash.
        let baseline = suites
            .iter()
            .find(|p| p.file_name().and_then(|n| n.to_str()) == Some("baseline.json"));
        let Some(baseline) = baseline else {
            return;
        };
        let res = run_bench(std::slice::from_ref(baseline)).unwrap();
        assert_eq!(res.suites.len(), 1);
        let s = &res.suites[0];
        assert_eq!(s.suite_name, "baseline");
        assert!(s.total > 0);
    }
}
