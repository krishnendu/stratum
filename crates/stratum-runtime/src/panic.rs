//! Panic hook + crash file writer.
//!
//! Captures unwind information at panic time and writes a JSON record into
//! `<state>/crashes/`. Detailed delivery (preview, send, opt-in) lands in
//! Phase 4 per `plan/25-crash-reports.md`; here we just persist locally so
//! a crashed run leaves diagnosable artifacts.

use std::any::Any;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use stratum_types::error::codes::E9001_INTERNAL_PANIC;
use time::format_description::well_known::Rfc3339;
use time::OffsetDateTime;

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

/// Install the panic hook. Subsequent calls overwrite the previous hook.
///
/// The captured `crash_dir` and `version` are moved into the hook closure;
/// each install fully replaces the prior hook with the new directory.
pub fn install_hook(crash_dir: PathBuf, version: &'static str) {
    std::panic::set_hook(Box::new(move |info| {
        let message = panic_message_from_payload(info.payload());
        let location = info.location().map(|l| format!("{l}"));
        let thread = std::thread::current()
            .name()
            .unwrap_or("unnamed")
            .to_string();
        handle_panic(&crash_dir, version, message, location, thread);
    }));
}

/// Internal body of the panic hook, extracted for direct testing.
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

    /// Serializes tests that mutate the global panic hook so they do not
    /// race against one another.
    static HOOK_LOCK: Mutex<()> = Mutex::new(());

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
        assert_eq!(entries, 1);
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
}
