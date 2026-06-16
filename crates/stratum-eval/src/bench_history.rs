//! `bench-history` — append a bench-floor result to a versioned
//! JSON time-series and refresh `latest.json` under a history dir.
//!
//! The bin entry is in `src/bin/bench_history.rs`; this module owns the
//! pure logic so it's unit-testable.
//!
//! The timestamp used for the date-stamped JSONL file is the result
//! file's `mtime` — never `SystemTime::now()`. Reproducibility matters:
//! re-running the workflow against the same artifact must produce a
//! byte-identical history entry.

use std::error::Error;
use std::fmt;
use std::fs;
use std::io;
use std::path::Path;
use std::time::SystemTime;

use serde::{Deserialize, Serialize};

use crate::bench_floor::BenchFloorResult;

/// Schema version of the per-row JSONL history entries.
pub const BENCH_HISTORY_SCHEMA_VERSION: u32 = 1;

/// One row of the history JSONL — the bench result plus the timestamp
/// that placed it in the file name's month bucket.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchHistoryEntry {
    /// On-disk schema version.
    pub schema_version: u32,
    /// Seconds since the unix epoch when the source result file was
    /// last modified. Stable across re-runs of the same artifact.
    pub recorded_at_unix: u64,
    /// Calendar date string `YYYY-MM-DD` derived from `recorded_at_unix`.
    pub recorded_at_date: String,
    /// The full bench-floor result inlined verbatim so a single JSONL
    /// row is self-contained.
    pub result: BenchFloorResult,
}

/// Errors surfaced by [`append_history`].
#[derive(Debug)]
pub enum BenchHistoryError {
    /// Filesystem error.
    Io(io::Error),
    /// JSON (de)serialization error.
    Serialize(String),
    /// Source result file was not valid JSON / not a `BenchFloorResult`.
    Parse(String),
}

impl fmt::Display for BenchHistoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "bench-history io error: {e}"),
            Self::Serialize(msg) => write!(f, "bench-history serialize error: {msg}"),
            Self::Parse(msg) => write!(f, "bench-history parse error: {msg}"),
        }
    }
}

impl Error for BenchHistoryError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

/// Read a bench-floor result JSON from `result_path`, append it to
/// `history_dir/<YYYY-MM>.jsonl`, and refresh `history_dir/latest.json`.
///
/// `recorded_at` is the timestamp baked into the history row. Pass the
/// result file's mtime; never `SystemTime::now()` (reproducibility).
///
/// # Errors
///
/// * [`BenchHistoryError::Io`] for filesystem errors.
/// * [`BenchHistoryError::Parse`] when the result file isn't a valid
///   `BenchFloorResult` JSON.
/// * [`BenchHistoryError::Serialize`] when re-serializing fails.
pub fn append_history(
    result_path: &Path,
    history_dir: &Path,
    recorded_at: SystemTime,
) -> Result<BenchHistoryEntry, BenchHistoryError> {
    let raw = fs::read(result_path).map_err(BenchHistoryError::Io)?;
    let result: BenchFloorResult =
        serde_json::from_slice(&raw).map_err(|e| BenchHistoryError::Parse(e.to_string()))?;

    let recorded_at_unix = system_time_to_unix(recorded_at);
    let recorded_at_date = unix_to_iso_date(recorded_at_unix);
    let month_bucket = month_bucket_from_date(&recorded_at_date);

    let entry = BenchHistoryEntry {
        schema_version: BENCH_HISTORY_SCHEMA_VERSION,
        recorded_at_unix,
        recorded_at_date,
        result,
    };

    fs::create_dir_all(history_dir).map_err(BenchHistoryError::Io)?;

    let jsonl_path = history_dir.join(format!("{month_bucket}.jsonl"));
    append_jsonl_line(&jsonl_path, &entry)?;

    let latest_path = history_dir.join("latest.json");
    write_latest(&latest_path, &entry)?;

    Ok(entry)
}

// Single-writer assumption: `O_APPEND + write_all` is only atomic for
// payloads ≤ `PIPE_BUF` (4096 bytes on Linux). A serialized
// `BenchHistoryEntry` can exceed that with enough suites, so concurrent
// writers would interleave bytes mid-line. The CI workflow guarantees
// only one writer at a time via a global `concurrency: bench-floor`
// group (see `.github/workflows/bench-floor.yml`). If this function
// is ever called outside that workflow, wrap it in an OS file lock
// (e.g. `fs2::flock`).
fn append_jsonl_line(path: &Path, entry: &BenchHistoryEntry) -> Result<(), BenchHistoryError> {
    use std::io::Write;
    let mut line =
        serde_json::to_string(entry).map_err(|e| BenchHistoryError::Serialize(e.to_string()))?;
    line.push('\n');
    let mut f = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(BenchHistoryError::Io)?;
    f.write_all(line.as_bytes())
        .map_err(BenchHistoryError::Io)?;
    f.sync_all().map_err(BenchHistoryError::Io)?;
    Ok(())
}

fn write_latest(path: &Path, entry: &BenchHistoryEntry) -> Result<(), BenchHistoryError> {
    use std::io::Write;
    let bytes = serde_json::to_vec_pretty(entry)
        .map_err(|e| BenchHistoryError::Serialize(e.to_string()))?;
    let mut tmp_name = path
        .file_name()
        .ok_or_else(|| {
            BenchHistoryError::Io(io::Error::new(
                io::ErrorKind::InvalidInput,
                "latest path is missing a file name",
            ))
        })?
        .to_os_string();
    tmp_name.push(".tmp");
    let tmp = path.with_file_name(tmp_name);
    {
        let mut f = fs::OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&tmp)
            .map_err(BenchHistoryError::Io)?;
        f.write_all(&bytes).map_err(BenchHistoryError::Io)?;
        f.sync_all().map_err(BenchHistoryError::Io)?;
    }
    if let Err(e) = fs::rename(&tmp, path) {
        let _ = fs::remove_file(&tmp);
        return Err(BenchHistoryError::Io(e));
    }
    Ok(())
}

/// Read the mtime of `path` and clamp absent values to the unix epoch.
///
/// # Errors
///
/// Returns [`BenchHistoryError::Io`] when the file metadata can't be
/// read.
pub fn mtime_of(path: &Path) -> Result<SystemTime, BenchHistoryError> {
    let meta = fs::metadata(path).map_err(BenchHistoryError::Io)?;
    meta.modified().map_err(BenchHistoryError::Io)
}

fn system_time_to_unix(t: SystemTime) -> u64 {
    t.duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Convert a unix timestamp to a `YYYY-MM-DD` date string using the
/// civil-from-days algorithm (Howard Hinnant). UTC only; no leap
/// seconds. Deterministic and dependency-free.
///
/// Works in `u64` throughout: unix timestamps are non-negative by
/// definition (the bin clamps pre-epoch times to 0 upstream), and the
/// 0001-01-01 shift baked into `Z_OFFSET` keeps every intermediate
/// non-negative for any unix second up to year ~5.84 × 10^11.
fn unix_to_iso_date(unix: u64) -> String {
    // Days since the unix epoch.
    let days = unix / 86_400;
    // Shift the day zero to a March-of-year-zero anchor so leap years
    // line up. 719_468 is days from 0000-03-01 to 1970-01-01.
    let z = days + 719_468;
    let era = z / 146_097;
    let day_of_era = z - era * 146_097; // 0..=146096
    let year_of_era =
        (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365; // 0..=399
    let mut y = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100); // 0..=365
    let mp = (5 * day_of_year + 2) / 153; // 0..=11
    let d = day_of_year - (153 * mp + 2) / 5 + 1; // 1..=31
    let m = if mp < 10 { mp + 3 } else { mp - 9 }; // 1..=12
    if m <= 2 {
        y += 1;
    }
    format!("{y:04}-{m:02}-{d:02}")
}

fn month_bucket_from_date(date: &str) -> String {
    // date is `YYYY-MM-DD`; take the first 7 chars.
    date.get(..7).unwrap_or("0000-00").to_string()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::bench_floor::{BenchFloorResult, SuiteMetric, BENCH_FLOOR_SCHEMA_VERSION};
    use std::path::PathBuf;
    use std::time::Duration;

    fn sample_result() -> BenchFloorResult {
        BenchFloorResult {
            schema_version: BENCH_FLOOR_SCHEMA_VERSION,
            suites: vec![SuiteMetric {
                suite_name: "demo".into(),
                source: "evals/demo.json".into(),
                passed: 5,
                total: 5,
                pass_rate: 1.0,
                mean_wall_clock_ms: 1.5,
                p95_wall_clock_ms: 2,
                mean_completion_tokens: 4.0,
                total_duration_ms: 7,
            }],
            total_duration_ms: 7,
        }
    }

    fn write_sample(dir: &Path) -> PathBuf {
        let p = dir.join("result.json");
        fs::write(&p, serde_json::to_vec_pretty(&sample_result()).unwrap()).unwrap();
        p
    }

    #[test]
    fn iso_date_well_known_timestamps() {
        assert_eq!(unix_to_iso_date(0), "1970-01-01");
        // 2024-01-01 00:00:00 UTC = 1704067200
        assert_eq!(unix_to_iso_date(1_704_067_200), "2024-01-01");
        // 2026-06-15 00:00:00 UTC = 1781481600
        assert_eq!(unix_to_iso_date(1_781_481_600), "2026-06-15");
        // A second before midnight rolls back to the previous day.
        assert_eq!(unix_to_iso_date(1_781_481_599), "2026-06-14");
    }

    #[test]
    fn month_bucket_well_formed() {
        assert_eq!(month_bucket_from_date("2026-06-15"), "2026-06");
    }

    #[test]
    fn append_history_writes_jsonl_and_latest() {
        let dir = tempfile::tempdir().unwrap();
        let result = write_sample(dir.path());
        let history = dir.path().join("history");
        let recorded_at = SystemTime::UNIX_EPOCH + Duration::from_secs(1_781_481_600); // 2026-06-15

        let entry = append_history(&result, &history, recorded_at).unwrap();
        assert_eq!(entry.recorded_at_date, "2026-06-15");
        assert_eq!(entry.recorded_at_unix, 1_781_481_600);

        let jsonl = history.join("2026-06.jsonl");
        assert!(jsonl.is_file());
        let body = fs::read_to_string(&jsonl).unwrap();
        assert_eq!(body.lines().count(), 1);

        let latest = history.join("latest.json");
        assert!(latest.is_file());
        let back: BenchHistoryEntry = serde_json::from_slice(&fs::read(&latest).unwrap()).unwrap();
        assert_eq!(back, entry);
    }

    #[test]
    fn append_history_appends_multiple_rows_in_same_month() {
        let dir = tempfile::tempdir().unwrap();
        let result = write_sample(dir.path());
        let history = dir.path().join("history");
        let t1 = SystemTime::UNIX_EPOCH + Duration::from_secs(1_781_481_600);
        let t2 = SystemTime::UNIX_EPOCH + Duration::from_secs(1_781_481_600 + 86_400);
        append_history(&result, &history, t1).unwrap();
        append_history(&result, &history, t2).unwrap();
        let body = fs::read_to_string(history.join("2026-06.jsonl")).unwrap();
        assert_eq!(body.lines().count(), 2);
    }

    #[test]
    fn append_history_rejects_malformed_result() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("bad.json");
        fs::write(&p, b"{not-json").unwrap();
        let history = dir.path().join("history");
        let err = append_history(&p, &history, SystemTime::UNIX_EPOCH).unwrap_err();
        assert!(matches!(err, BenchHistoryError::Parse(_)));
    }

    #[test]
    fn append_history_rejects_missing_result_file() {
        let dir = tempfile::tempdir().unwrap();
        let nope = dir.path().join("nope.json");
        let history = dir.path().join("history");
        let err = append_history(&nope, &history, SystemTime::UNIX_EPOCH).unwrap_err();
        assert!(matches!(err, BenchHistoryError::Io(_)));
    }

    #[test]
    fn mtime_of_existing_file_succeeds() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("x");
        fs::write(&p, b"hi").unwrap();
        let m = mtime_of(&p).unwrap();
        // mtime should be in the present, not the unix epoch.
        assert!(m >= SystemTime::UNIX_EPOCH);
    }

    #[test]
    fn mtime_of_missing_file_errors() {
        let dir = tempfile::tempdir().unwrap();
        let err = mtime_of(&dir.path().join("missing")).unwrap_err();
        assert!(matches!(err, BenchHistoryError::Io(_)));
    }

    #[test]
    fn system_time_to_unix_pre_epoch_clamps_to_zero() {
        // A SystemTime before UNIX_EPOCH would error in duration_since;
        // we clamp to 0.
        let before = SystemTime::UNIX_EPOCH
            .checked_sub(Duration::from_secs(1))
            .unwrap_or(SystemTime::UNIX_EPOCH);
        assert_eq!(system_time_to_unix(before), 0);
    }

    #[test]
    fn error_display_smoke() {
        let cases: Vec<BenchHistoryError> = vec![
            BenchHistoryError::Io(io::Error::other("x")),
            BenchHistoryError::Serialize("oops".into()),
            BenchHistoryError::Parse("bad".into()),
        ];
        for e in &cases {
            assert!(!format!("{e}").is_empty());
        }
        assert!(cases[0].source().is_some());
        assert!(cases[1].source().is_none());
        assert!(cases[2].source().is_none());
    }
}
