//! Panic hook + crash file writer.
//!
//! Captures unwind information at panic time and writes a JSON record into
//! `<state>/crashes/`. Detailed delivery (preview, send, opt-in) lands in
//! Phase 4 per `plan/25-crash-reports.md`; here we just persist locally so
//! a crashed run leaves diagnosable artifacts.
//!
//! When the user has opted in (via a JSON config at
//! `<state_root>/crash-reports.json` with `{"enabled": true}`) the hook also
//! writes a richer [`crate::crash_report::CrashBundle`] (record + redacted log
//! tail + environment snapshot) to `<state_root>/crashes/<id>.json` next to
//! the legacy record file. Opt-in is OFF by default; see the user-memory note
//! "crash reports opt-in".

use std::any::Any;
use std::error::Error as StdError;
use std::fmt;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use serde::{Deserialize, Serialize};
use stratum_types::error::codes::E9001_INTERNAL_PANIC;
use time::format_description::well_known::Rfc3339;
use time::OffsetDateTime;

use crate::crash_report::{build_bundle, write_bundle, CrashBundleConfig, CrashEnv};

/// One captured crash, persisted as JSON.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CrashRecord {
    /// Schema version for forward compatibility.
    pub schema_version: u32,
    /// Stratum stable error code attributed to this panic (`STRAT-E9001`).
    pub code: String,
    /// Rfc3339 timestamp at the time of capture.
    pub at: String,
    /// Stratum version (cargo crate version).
    pub stratum_version: String,
    /// Panic message as captured by `PanicHookInfo::payload`.
    pub message: String,
    /// Source `file:line:column` if available.
    pub location: Option<String>,
    /// Active thread name.
    pub thread: String,
}

impl CrashRecord {
    /// Synthesize a record (used by both the panic hook and tests).
    ///
    /// # Panics
    /// Does not panic for any caller-reachable input; the internal `expect`
    /// is justified because [`OffsetDateTime::format`] with [`Rfc3339`] is
    /// infallible. The carve-out is tracked in `docs/coverage-exclusions.md`.
    #[must_use]
    pub fn synthesize(
        version: &str,
        message: impl Into<String>,
        location: Option<String>,
        thread: impl Into<String>,
        now: OffsetDateTime,
    ) -> Self {
        #[allow(
            clippy::expect_used,
            reason = "OffsetDateTime::format with Rfc3339 is infallible"
        )]
        let at = now
            .format(&Rfc3339)
            .expect("Rfc3339 formatting of OffsetDateTime is infallible");
        Self {
            schema_version: 1,
            code: E9001_INTERNAL_PANIC.as_str().to_string(),
            at,
            stratum_version: version.into(),
            message: message.into(),
            location,
            thread: thread.into(),
        }
    }
}

/// On-disk opt-in flag for the richer crash bundle.
///
/// Stored at `<state_root>/crash-reports.json`. Missing or malformed files
/// are treated as opted-out so the default behaviour is fully local.
#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CrashReportConfig {
    /// `true` only if the user explicitly opted in. Default is `false`.
    pub enabled: bool,
}

/// Filesystem layout the panic hook needs to honour.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CrashReportPaths {
    /// Path to `<state_root>/crash-reports.json` (the opt-in toggle).
    pub config_path: PathBuf,
    /// Directory where crash artifacts are written.
    pub crashes_dir: PathBuf,
    /// Optional log buffer file whose tail is included in the opt-in bundle.
    pub log_buffer_path: Option<PathBuf>,
}

/// Errors returned by the panic-hook setup helpers.
///
/// The panic hook closure itself never propagates these; they exist purely
/// for the synchronous setup path (`install_hook_with_crash_reports`).
#[derive(Debug)]
pub enum CrashHookError {
    /// Filesystem I/O failed while preparing crash-report paths.
    Io(std::io::Error),
    /// JSON (de)serialization failed.
    Serialize(String),
}

impl fmt::Display for CrashHookError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "crash hook io error: {e}"),
            Self::Serialize(msg) => write!(f, "crash hook serialize error: {msg}"),
        }
    }
}

impl StdError for CrashHookError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Serialize(_) => None,
        }
    }
}

impl From<std::io::Error> for CrashHookError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

/// Load the opt-in [`CrashReportConfig`] from disk.
///
/// Missing or malformed files yield [`CrashReportConfig::default`] (i.e.
/// disabled). This function never panics — it is safe to call from inside
/// the panic hook.
#[must_use]
pub fn load_crash_report_config(path: &Path) -> CrashReportConfig {
    std::fs::read_to_string(path).map_or_else(
        |_| CrashReportConfig::default(),
        |s| serde_json::from_str(&s).unwrap_or_default(),
    )
}

/// Ensure the crashes directory exists, creating it (and parents) if needed.
///
/// # Errors
/// Returns [`CrashHookError::Io`] if directory creation fails.
pub fn ensure_crashes_dir(dir: &Path) -> Result<(), CrashHookError> {
    std::fs::create_dir_all(dir)?;
    Ok(())
}

/// Install the panic hook. Subsequent calls overwrite the previous hook.
///
/// The captured `crash_dir` and `version` are moved into the hook closure;
/// each install fully replaces the prior hook with the new directory.
///
/// This is the legacy entry point; it preserves backward compatibility by
/// delegating to [`install_hook_with_crash_reports`] with a paths layout that
/// disables opt-in bundle writes (no `config_path`, no log buffer).
pub fn install_hook(crash_dir: PathBuf, version: &'static str) {
    // We deliberately pass a non-existent config path so the opt-in branch
    // always reads default-disabled config and behaviour matches pre-bundle.
    let paths = CrashReportPaths {
        config_path: crash_dir.join("crash-reports.json"),
        crashes_dir: crash_dir,
        log_buffer_path: None,
    };
    let env = CrashEnv {
        app_version: version.to_string(),
        channel: "stable".to_string(),
        os: std::env::consts::OS.to_string(),
        cpu_arch: std::env::consts::ARCH.to_string(),
    };
    // Errors here are best-effort: the legacy entry point intentionally
    // hides them to match its prior signature.
    let _ = install_hook_with_crash_reports_inner(paths, env, version);
}

/// Install the panic hook with opt-in [`crate::crash_report::CrashBundle`]
/// support.
///
/// When the user has opted in via `paths.config_path`, the hook will also
/// write a redacted crash bundle into `paths.crashes_dir`. Otherwise only
/// the legacy [`CrashRecord`] is written (backward compatible).
///
/// # Errors
/// Returns [`CrashHookError::Io`] if `paths.crashes_dir` cannot be created
/// during setup. The hook closure itself never returns an error — any
/// failure inside the hook is swallowed and logged to stderr.
pub fn install_hook_with_crash_reports(
    paths: CrashReportPaths,
    env: CrashEnv,
) -> Result<(), CrashHookError> {
    let version: String = env.app_version.clone();
    install_hook_with_crash_reports_inner(paths, env, &version)
}

fn install_hook_with_crash_reports_inner(
    paths: CrashReportPaths,
    env: CrashEnv,
    version: &str,
) -> Result<(), CrashHookError> {
    ensure_crashes_dir(&paths.crashes_dir)?;
    let version_owned: String = version.to_string();
    std::panic::set_hook(Box::new(move |info| {
        let message = panic_message_from_payload(info.payload());
        let location = info.location().map(|l| format!("{l}"));
        let thread = std::thread::current()
            .name()
            .unwrap_or("unnamed")
            .to_string();
        handle_panic_with_bundle(&paths, &env, &version_owned, message, location, thread);
    }));
    Ok(())
}

/// Internal body of the panic hook, extracted for direct testing.
#[cfg(test)]
pub(crate) fn handle_panic(
    dir: &Path,
    version: &str,
    message: String,
    location: Option<String>,
    thread: String,
) {
    let record = CrashRecord::synthesize(
        version,
        message,
        location,
        thread,
        OffsetDateTime::now_utc(),
    );
    let _ = write_record(dir, &record);
    // The panic-handling boundary is one of the few legitimate sites for
    // direct stderr output; routing through `tracing` is unsafe at panic
    // time because the subscriber may be poisoned.
    #[allow(
        clippy::print_stderr,
        reason = "panic hook must not depend on tracing subscriber state"
    )]
    {
        eprintln!("[{}] {}", record.code, record.message);
    }
}

/// Bundle-aware variant of [`handle_panic`]. Always writes the legacy
/// [`CrashRecord`]; additionally writes a [`crate::crash_report::CrashBundle`]
/// when opted in via `paths.config_path`.
pub(crate) fn handle_panic_with_bundle(
    paths: &CrashReportPaths,
    env: &CrashEnv,
    version: &str,
    message: String,
    location: Option<String>,
    thread: String,
) {
    let record = CrashRecord::synthesize(
        version,
        message,
        location,
        thread,
        OffsetDateTime::now_utc(),
    );

    // Always write the legacy record so existing tooling keeps working
    // regardless of opt-in.
    let _ = write_record(&paths.crashes_dir, &record);

    // Opt-in branch: load config, build bundle, write it. Every step is
    // wrapped in a `match` / `let _` to guarantee no panic-from-panic.
    let cfg = load_crash_report_config(&paths.config_path);
    if cfg.enabled {
        let log_tail = paths
            .log_buffer_path
            .as_ref()
            .map_or_else(Vec::new, |p| read_log_tail(p, 200, 64 * 1024));
        match write_bundle_for_record(&record, &log_tail, env, &paths.crashes_dir) {
            Ok(_) => {}
            Err(e) => report_hook_error(&e),
        }
    }

    // The panic-handling boundary is one of the few legitimate sites for
    // direct stderr output; routing through `tracing` is unsafe at panic
    // time because the subscriber may be poisoned.
    #[allow(
        clippy::print_stderr,
        reason = "panic hook must not depend on tracing subscriber state"
    )]
    {
        eprintln!("[{}] {}", record.code, record.message);
    }
}

/// Pure-data helper exercised by tests: build a bundle for `record` and write
/// it to `<crashes_dir>/<crash_id>.json`.
pub(crate) fn write_bundle_for_record(
    record: &CrashRecord,
    log_tail: &[String],
    env: &CrashEnv,
    crashes_dir: &Path,
) -> Result<PathBuf, CrashHookError> {
    let now = SystemTime::now();
    let bundle = build_bundle(
        record.clone(),
        log_tail,
        env.clone(),
        &CrashBundleConfig::default(),
        now,
    );
    let crash_id = bundle_crash_id(record, now);
    let target = crashes_dir.join(format!("{crash_id}.json"));
    write_bundle(&bundle, &target).map_err(|e| CrashHookError::Serialize(format!("{e}")))?;
    Ok(target)
}

fn bundle_crash_id(record: &CrashRecord, now: SystemTime) -> String {
    let stamp = sanitize_stamp(&record.at);
    if !stamp.is_empty() {
        return format!("bundle-{stamp}");
    }
    let nanos = now
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let pid = std::process::id();
    format!("bundle-{nanos}-{pid}")
}

fn read_log_tail(path: &Path, max_lines: usize, max_bytes: usize) -> Vec<String> {
    let Ok(file) = File::open(path) else {
        return Vec::new();
    };
    let reader = BufReader::new(file);
    let mut lines: Vec<String> = Vec::new();
    for line in reader.lines() {
        match line {
            Ok(l) => lines.push(l),
            Err(_) => break,
        }
    }
    let mut start = lines.len().saturating_sub(max_lines);
    let mut total: usize = lines[start..].iter().map(String::len).sum();
    while start < lines.len() && total > max_bytes {
        total = total.saturating_sub(lines[start].len());
        start += 1;
    }
    lines.split_off(start)
}

fn report_hook_error(err: &CrashHookError) {
    // Direct stderr is acceptable inside the panic hook — see note in
    // `handle_panic`.
    #[allow(
        clippy::print_stderr,
        reason = "panic hook must not depend on tracing subscriber state"
    )]
    {
        eprintln!("stratum: crash report write failed: {err}");
    }
}

/// Decode a panic payload into a printable message.
pub(crate) fn panic_message_from_payload(payload: &dyn Any) -> String {
    if let Some(s) = payload.downcast_ref::<&'static str>() {
        return (*s).to_string();
    }
    if let Some(s) = payload.downcast_ref::<String>() {
        return s.clone();
    }
    "non-string panic payload".to_string()
}

fn write_record(dir: &Path, record: &CrashRecord) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let stamp = sanitize_stamp(&record.at);
    let path = dir.join(format!("crash-{stamp}.json"));
    let rendered = serde_json::to_string_pretty(record).unwrap_or_else(|_| "{}".to_string());
    std::fs::write(path, rendered)
}

fn sanitize_stamp(s: &str) -> String {
    s.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use tempfile::TempDir;

    use super::*;
    use crate::crash_report::{load_bundle, CrashBundle};

    /// Serializes tests that mutate the global panic hook so they do not
    /// race against one another.
    static HOOK_LOCK: Mutex<()> = Mutex::new(());

    fn env_for_test() -> CrashEnv {
        CrashEnv {
            app_version: "0.1.0".into(),
            channel: "stable".into(),
            os: "macos".into(),
            cpu_arch: "aarch64".into(),
        }
    }

    #[test]
    fn synthesize_carries_code_and_message() {
        let now = OffsetDateTime::from_unix_timestamp(1_700_000_000).unwrap();
        let rec = CrashRecord::synthesize("0.1.0", "boom", Some("a.rs:1:2".into()), "main", now);
        assert_eq!(rec.code, "STRAT-E9001");
        assert_eq!(rec.message, "boom");
        assert_eq!(rec.stratum_version, "0.1.0");
        assert_eq!(rec.thread, "main");
        assert!(rec.at.contains('T'));
    }

    #[test]
    fn synthesize_without_location() {
        let now = OffsetDateTime::from_unix_timestamp(1_700_000_000).unwrap();
        let rec = CrashRecord::synthesize("0.1.0", "x", None, "main", now);
        assert!(rec.location.is_none());
    }

    #[test]
    fn write_record_creates_file() {
        let tmp = TempDir::new().unwrap();
        let now = OffsetDateTime::from_unix_timestamp(1_700_000_000).unwrap();
        let rec = CrashRecord::synthesize("0.1.0", "boom", None, "main", now);
        write_record(tmp.path(), &rec).unwrap();
        let entries: Vec<_> = std::fs::read_dir(tmp.path())
            .unwrap()
            .filter_map(Result::ok)
            .collect();
        assert_eq!(entries.len(), 1);
        let name = entries[0].file_name().to_string_lossy().to_string();
        assert!(name.starts_with("crash-"));
        assert!(std::path::Path::new(&name)
            .extension()
            .is_some_and(|e| e == "json"));
    }

    #[test]
    fn write_record_idempotent_in_dir() {
        let tmp = TempDir::new().unwrap();
        let now = OffsetDateTime::from_unix_timestamp(1_700_000_000).unwrap();
        let rec = CrashRecord::synthesize("0.1.0", "boom", None, "main", now);
        write_record(tmp.path(), &rec).unwrap();
        // Re-running with the same timestamp overwrites the same file.
        write_record(tmp.path(), &rec).unwrap();
        let count = std::fs::read_dir(tmp.path()).unwrap().count();
        assert_eq!(count, 1);
    }

    #[test]
    fn write_record_errors_on_unwritable_dir() {
        let tmp = TempDir::new().unwrap();
        let blocker = tmp.path().join("blocker");
        std::fs::write(&blocker, b"x").unwrap();
        let inside_file = blocker.join("nested");
        let now = OffsetDateTime::from_unix_timestamp(1_700_000_000).unwrap();
        let rec = CrashRecord::synthesize("0.1.0", "boom", None, "main", now);
        assert!(write_record(&inside_file, &rec).is_err());
    }

    #[test]
    fn sanitize_stamp_replaces_punctuation() {
        let s = sanitize_stamp("2026-06-04T18:11:02Z");
        assert!(!s.contains(':'));
        assert!(!s.contains('-'));
        assert!(s.contains("2026"));
    }

    #[test]
    fn crash_record_serde_roundtrip() {
        let now = OffsetDateTime::from_unix_timestamp(1_700_000_000).unwrap();
        let rec = CrashRecord::synthesize("0.1.0", "boom", Some("a.rs:1:2".into()), "main", now);
        let s = serde_json::to_string(&rec).unwrap();
        let back: CrashRecord = serde_json::from_str(&s).unwrap();
        assert_eq!(rec, back);
    }

    #[test]
    fn install_hook_does_not_double_register() {
        let _g = HOOK_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let tmp = TempDir::new().unwrap();
        let prior = std::panic::take_hook();
        install_hook(tmp.path().to_path_buf(), env!("CARGO_PKG_VERSION"));
        install_hook(tmp.path().to_path_buf(), env!("CARGO_PKG_VERSION"));
        std::panic::set_hook(prior);
    }

    #[test]
    fn installed_hook_writes_crash_file_on_panic() {
        let _g = HOOK_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let tmp = TempDir::new().unwrap();
        let prior = std::panic::take_hook();
        install_hook(tmp.path().to_path_buf(), env!("CARGO_PKG_VERSION"));
        let result = std::panic::catch_unwind(|| panic!("synthetic-panic-for-test"));
        std::panic::set_hook(prior);
        assert!(result.is_err());
        let entries = std::fs::read_dir(tmp.path())
            .unwrap()
            .filter_map(Result::ok)
            .count();
        // legacy hook writes only the CrashRecord (no opt-in config present).
        assert!(entries >= 1);
    }

    #[test]
    fn panic_message_decodes_static_str() {
        let payload: Box<dyn std::any::Any + Send> = Box::new("static-panic");
        assert_eq!(panic_message_from_payload(payload.as_ref()), "static-panic");
    }

    #[test]
    fn panic_message_decodes_owned_string() {
        let payload: Box<dyn std::any::Any + Send> = Box::new(String::from("owned-panic"));
        assert_eq!(panic_message_from_payload(payload.as_ref()), "owned-panic");
    }

    #[test]
    fn panic_message_decodes_other_to_sentinel() {
        let payload: Box<dyn std::any::Any + Send> = Box::new(42_u64);
        assert_eq!(
            panic_message_from_payload(payload.as_ref()),
            "non-string panic payload"
        );
    }

    #[test]
    fn handle_panic_writes_record_and_does_not_panic() {
        let tmp = TempDir::new().unwrap();
        handle_panic(
            tmp.path(),
            "0.0.0",
            "synthetic".to_string(),
            Some("a.rs:1:2".into()),
            "main".to_string(),
        );
        let entries: Vec<_> = std::fs::read_dir(tmp.path())
            .unwrap()
            .filter_map(Result::ok)
            .collect();
        assert_eq!(entries.len(), 1);
    }

    #[test]
    fn handle_panic_swallows_write_errors() {
        let tmp = TempDir::new().unwrap();
        let blocker = tmp.path().join("blocker");
        std::fs::write(&blocker, b"x").unwrap();
        // Pass a path that cannot become a directory; handle_panic must not
        // propagate the error (it eats the result).
        handle_panic(
            &blocker.join("nested"),
            "0.0.0",
            "synthetic".to_string(),
            None,
            "main".to_string(),
        );
    }

    // ----- new CrashReportConfig / opt-in bundle tests -----

    #[test]
    fn crash_report_config_default_is_disabled() {
        assert!(!CrashReportConfig::default().enabled);
    }

    #[test]
    fn load_crash_report_config_missing_file_returns_default() {
        let tmp = TempDir::new().unwrap();
        let cfg = load_crash_report_config(&tmp.path().join("nope.json"));
        assert_eq!(cfg, CrashReportConfig::default());
    }

    #[test]
    fn load_crash_report_config_malformed_returns_default() {
        let tmp = TempDir::new().unwrap();
        let p = tmp.path().join("crash-reports.json");
        std::fs::write(&p, b"not-json{").unwrap();
        let cfg = load_crash_report_config(&p);
        assert_eq!(cfg, CrashReportConfig::default());
    }

    #[test]
    fn load_crash_report_config_parses_enabled_true() {
        let tmp = TempDir::new().unwrap();
        let p = tmp.path().join("crash-reports.json");
        std::fs::write(&p, br#"{"enabled": true}"#).unwrap();
        let cfg = load_crash_report_config(&p);
        assert!(cfg.enabled);
    }

    #[test]
    fn ensure_crashes_dir_creates_tree() {
        let tmp = TempDir::new().unwrap();
        let nested = tmp.path().join("a").join("b").join("c");
        ensure_crashes_dir(&nested).unwrap();
        assert!(nested.is_dir());
    }

    #[test]
    fn ensure_crashes_dir_is_idempotent() {
        let tmp = TempDir::new().unwrap();
        let nested = tmp.path().join("x");
        ensure_crashes_dir(&nested).unwrap();
        ensure_crashes_dir(&nested).unwrap();
        assert!(nested.is_dir());
    }

    #[test]
    fn ensure_crashes_dir_errors_on_blocker_file() {
        let tmp = TempDir::new().unwrap();
        let blocker = tmp.path().join("blocker");
        std::fs::write(&blocker, b"x").unwrap();
        let nested = blocker.join("inside");
        let err = ensure_crashes_dir(&nested).unwrap_err();
        assert!(matches!(err, CrashHookError::Io(_)));
    }

    #[test]
    fn crash_report_config_serde_roundtrip() {
        let cfg = CrashReportConfig { enabled: true };
        let s = serde_json::to_string(&cfg).unwrap();
        let back: CrashReportConfig = serde_json::from_str(&s).unwrap();
        assert_eq!(cfg, back);
    }

    #[test]
    fn crash_hook_error_display_io_variant() {
        let e = CrashHookError::Io(std::io::Error::other("boom"));
        let msg = format!("{e}");
        assert!(msg.contains("io error"));
        assert!(e.source().is_some());
    }

    #[test]
    fn crash_hook_error_display_serialize_variant() {
        let e = CrashHookError::Serialize("nope".to_string());
        let msg = format!("{e}");
        assert!(msg.contains("serialize"));
        assert!(e.source().is_none());
    }

    #[test]
    fn crash_hook_error_from_io() {
        let e: CrashHookError = std::io::Error::other("x").into();
        assert!(matches!(e, CrashHookError::Io(_)));
    }

    #[test]
    fn write_bundle_for_record_produces_loadable_file() {
        let tmp = TempDir::new().unwrap();
        let now = OffsetDateTime::from_unix_timestamp(1_700_000_000).unwrap();
        let rec = CrashRecord::synthesize("0.1.0", "boom", None, "main", now);
        let target = write_bundle_for_record(&rec, &[], &env_for_test(), tmp.path()).unwrap();
        assert!(target.exists());
        let loaded: CrashBundle = load_bundle(&target).unwrap();
        assert_eq!(loaded.record, rec);
        assert_eq!(loaded.env, env_for_test());
    }

    #[test]
    fn write_bundle_for_record_redacts_log_tail() {
        let tmp = TempDir::new().unwrap();
        let now = OffsetDateTime::from_unix_timestamp(1_700_000_000).unwrap();
        let rec = CrashRecord::synthesize("0.1.0", "boom", None, "main", now);
        let tail = vec!["api_key=sk-leak-me".to_string()];
        let target = write_bundle_for_record(&rec, &tail, &env_for_test(), tmp.path()).unwrap();
        let loaded: CrashBundle = load_bundle(&target).unwrap();
        assert_eq!(loaded.redacted_log_tail.len(), 1);
        assert!(loaded.redacted_log_tail[0].contains("<REDACTED>"));
        assert!(!loaded.redacted_log_tail[0].contains("sk-leak-me"));
    }

    #[test]
    fn write_bundle_for_record_errors_on_blocker() {
        let tmp = TempDir::new().unwrap();
        let blocker = tmp.path().join("blocker");
        std::fs::write(&blocker, b"x").unwrap();
        let now = OffsetDateTime::from_unix_timestamp(1_700_000_000).unwrap();
        let rec = CrashRecord::synthesize("0.1.0", "boom", None, "main", now);
        // crashes_dir is a file, not a directory — write_bundle will fail.
        let target_dir = blocker.join("nested");
        let err = write_bundle_for_record(&rec, &[], &env_for_test(), &target_dir).unwrap_err();
        assert!(matches!(err, CrashHookError::Serialize(_)));
    }

    #[test]
    fn bundle_crash_id_uses_stamp() {
        let now = OffsetDateTime::from_unix_timestamp(1_700_000_000).unwrap();
        let rec = CrashRecord::synthesize("0.1.0", "x", None, "main", now);
        let id = bundle_crash_id(&rec, SystemTime::UNIX_EPOCH);
        assert!(id.starts_with("bundle-"));
        // Sanitised stamp should still embed the year.
        assert!(id.contains("2023") || id.contains("2024") || id.contains("2025"));
    }

    #[test]
    fn bundle_crash_id_falls_back_on_empty_stamp() {
        // Manually craft a record with an empty `at`.
        let rec = CrashRecord {
            schema_version: 1,
            code: "STRAT-E9001".to_string(),
            at: String::new(),
            stratum_version: "0.1.0".to_string(),
            message: "x".to_string(),
            location: None,
            thread: "main".to_string(),
        };
        let now = SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(42);
        let id = bundle_crash_id(&rec, now);
        assert!(id.starts_with("bundle-"));
        assert!(id.contains('-'));
    }

    #[test]
    fn read_log_tail_missing_file_is_empty() {
        let tmp = TempDir::new().unwrap();
        let out = read_log_tail(&tmp.path().join("nope.log"), 10, 1024);
        assert!(out.is_empty());
    }

    #[test]
    fn read_log_tail_respects_line_budget() {
        use std::fmt::Write as _;
        let tmp = TempDir::new().unwrap();
        let p = tmp.path().join("log");
        let mut body = String::new();
        for i in 0..50 {
            writeln!(&mut body, "line-{i}").unwrap();
        }
        std::fs::write(&p, body).unwrap();
        let out = read_log_tail(&p, 5, 10_000);
        assert_eq!(out.len(), 5);
        assert_eq!(out[4], "line-49");
    }

    #[test]
    fn read_log_tail_respects_byte_budget() {
        use std::fmt::Write as _;
        let tmp = TempDir::new().unwrap();
        let p = tmp.path().join("log");
        let mut body = String::new();
        for i in 0..20 {
            writeln!(&mut body, "xx-{i:02}").unwrap();
        }
        std::fs::write(&p, body).unwrap();
        let out = read_log_tail(&p, 1000, 20);
        let total: usize = out.iter().map(String::len).sum();
        assert!(total <= 20);
        assert!(!out.is_empty());
    }

    #[test]
    fn handle_panic_with_bundle_opt_out_writes_only_record() {
        let tmp = TempDir::new().unwrap();
        let crashes = tmp.path().join("crashes");
        ensure_crashes_dir(&crashes).unwrap();
        let paths = CrashReportPaths {
            config_path: tmp.path().join("crash-reports.json"),
            crashes_dir: crashes.clone(),
            log_buffer_path: None,
        };
        handle_panic_with_bundle(
            &paths,
            &env_for_test(),
            "0.0.0",
            "synthetic".to_string(),
            None,
            "main".to_string(),
        );
        let entries: Vec<_> = std::fs::read_dir(&crashes)
            .unwrap()
            .filter_map(Result::ok)
            .collect();
        assert_eq!(entries.len(), 1);
        let name = entries[0].file_name().to_string_lossy().into_owned();
        assert!(name.starts_with("crash-"));
        assert!(!name.starts_with("bundle-"));
    }

    #[test]
    fn handle_panic_with_bundle_opt_in_writes_both() {
        let tmp = TempDir::new().unwrap();
        let crashes = tmp.path().join("crashes");
        ensure_crashes_dir(&crashes).unwrap();
        let cfg_path = tmp.path().join("crash-reports.json");
        std::fs::write(&cfg_path, br#"{"enabled": true}"#).unwrap();
        let log_path = tmp.path().join("session.log");
        std::fs::write(&log_path, b"line-1\napi_key=sk-leak\nline-3\n").unwrap();
        let paths = CrashReportPaths {
            config_path: cfg_path,
            crashes_dir: crashes.clone(),
            log_buffer_path: Some(log_path),
        };
        handle_panic_with_bundle(
            &paths,
            &env_for_test(),
            "0.0.0",
            "synthetic".to_string(),
            None,
            "main".to_string(),
        );
        let entries: Vec<_> = std::fs::read_dir(&crashes)
            .unwrap()
            .filter_map(Result::ok)
            .collect();
        // record + bundle
        assert_eq!(entries.len(), 2);
        let bundle_entry = entries
            .iter()
            .find(|e| e.file_name().to_string_lossy().starts_with("bundle-"))
            .expect("expected a bundle-*.json file");
        let bundle: CrashBundle = load_bundle(&bundle_entry.path()).unwrap();
        assert_eq!(bundle.record.message, "synthetic");
        // Secret got redacted.
        assert!(bundle
            .redacted_log_tail
            .iter()
            .any(|l| l.contains("<REDACTED>")));
        assert!(bundle
            .redacted_log_tail
            .iter()
            .all(|l| !l.contains("sk-leak")));
    }

    #[test]
    fn handle_panic_with_bundle_opt_in_no_log_buffer() {
        let tmp = TempDir::new().unwrap();
        let crashes = tmp.path().join("crashes");
        ensure_crashes_dir(&crashes).unwrap();
        let cfg_path = tmp.path().join("crash-reports.json");
        std::fs::write(&cfg_path, br#"{"enabled": true}"#).unwrap();
        let paths = CrashReportPaths {
            config_path: cfg_path,
            crashes_dir: crashes.clone(),
            log_buffer_path: None,
        };
        handle_panic_with_bundle(
            &paths,
            &env_for_test(),
            "0.0.0",
            "no-log".to_string(),
            None,
            "main".to_string(),
        );
        let bundle_count = std::fs::read_dir(&crashes)
            .unwrap()
            .filter_map(Result::ok)
            .filter(|e| e.file_name().to_string_lossy().starts_with("bundle-"))
            .count();
        assert_eq!(bundle_count, 1);
    }

    #[test]
    fn handle_panic_with_bundle_swallows_bundle_write_failure() {
        let tmp = TempDir::new().unwrap();
        // crashes_dir is fine for the record, but we will mount the bundle
        // write failure by making crashes_dir itself unable to host the
        // bundle file. We achieve that with a path whose name collides with
        // a directory we manually drop into the crashes dir.
        let crashes = tmp.path().join("crashes");
        ensure_crashes_dir(&crashes).unwrap();
        // Pre-create a directory whose name will exactly match the bundle
        // target so the write fails.
        let now = OffsetDateTime::now_utc();
        let stamp = sanitize_stamp(&now.format(&Rfc3339).unwrap_or_else(|_| "stamp".to_string()));
        let blocker_dir = crashes.join(format!("bundle-{stamp}.json"));
        std::fs::create_dir_all(&blocker_dir).unwrap();
        let cfg_path = tmp.path().join("crash-reports.json");
        std::fs::write(&cfg_path, br#"{"enabled": true}"#).unwrap();
        let paths = CrashReportPaths {
            config_path: cfg_path,
            crashes_dir: crashes,
            log_buffer_path: None,
        };
        // Should not panic even though the bundle write fails.
        handle_panic_with_bundle(
            &paths,
            &env_for_test(),
            "0.0.0",
            "boom".to_string(),
            None,
            "main".to_string(),
        );
    }

    #[test]
    fn install_hook_with_crash_reports_returns_ok() {
        let _g = HOOK_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let tmp = TempDir::new().unwrap();
        let prior = std::panic::take_hook();
        let paths = CrashReportPaths {
            config_path: tmp.path().join("crash-reports.json"),
            crashes_dir: tmp.path().join("crashes"),
            log_buffer_path: None,
        };
        let result = install_hook_with_crash_reports(paths, env_for_test());
        std::panic::set_hook(prior);
        assert!(result.is_ok());
    }

    #[test]
    fn install_hook_with_crash_reports_io_error_on_blocker() {
        let _g = HOOK_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let tmp = TempDir::new().unwrap();
        let blocker = tmp.path().join("blocker");
        std::fs::write(&blocker, b"x").unwrap();
        let prior = std::panic::take_hook();
        let paths = CrashReportPaths {
            config_path: tmp.path().join("crash-reports.json"),
            crashes_dir: blocker.join("nope"),
            log_buffer_path: None,
        };
        let result = install_hook_with_crash_reports(paths, env_for_test());
        std::panic::set_hook(prior);
        assert!(matches!(result, Err(CrashHookError::Io(_))));
    }
}
