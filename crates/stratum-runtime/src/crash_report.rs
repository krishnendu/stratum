//! Opt-in crash-report bundle + log redaction.
//!
//! Sits on top of the existing [`crate::panic::CrashRecord`] without modifying
//! it. The bundle bolts on environment metadata and a redacted tail of the
//! tracing/log buffer so a user can preview-and-send a crash report without
//! leaking secrets, JWTs, or local usernames.
//!
//! See `plan/22-panic-and-crash-reports.md` and the user-memory note
//! "crash reports opt-in" — sending is always explicit; this module only
//! prepares the artifact.

use std::error::Error as StdError;
use std::fmt;
use std::fs::File;
use std::io::{BufReader, Write};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use serde::{Deserialize, Serialize};

use crate::panic::CrashRecord;

/// Current bundle schema version. Increment with any breaking on-disk change.
pub const CRASH_BUNDLE_SCHEMA_VERSION: u32 = 1;

/// Environment snapshot captured alongside the crash record.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CrashEnv {
    /// Stratum application version (cargo crate version at build time).
    pub app_version: String,
    /// Release channel: `stable` / `beta` / `nightly`.
    pub channel: String,
    /// Operating system family (e.g. `macos`, `linux`).
    pub os: String,
    /// CPU architecture (e.g. `aarch64`, `x86_64`).
    pub cpu_arch: String,
}

/// Opt-in crash bundle: a `CrashRecord` plus environment metadata and a
/// redacted slice of the log tail.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CrashBundle {
    /// Schema version for forward compatibility.
    pub schema_version: u32,
    /// The captured panic record.
    pub record: CrashRecord,
    /// Tail of the log buffer with secrets / paths / tokens redacted.
    pub redacted_log_tail: Vec<String>,
    /// Snapshot of the runtime environment.
    pub env: CrashEnv,
    /// Bundle creation time.
    pub created_at: SystemTime,
}

/// Knobs that control how much log tail is included in the bundle.
#[derive(Debug, Clone, Copy)]
pub struct CrashBundleConfig {
    /// Maximum number of log lines retained from the tail.
    pub max_log_tail_lines: usize,
    /// Maximum total bytes (sum of line lengths) retained from the tail.
    pub max_log_tail_bytes: usize,
}

impl Default for CrashBundleConfig {
    fn default() -> Self {
        Self {
            max_log_tail_lines: 200,
            max_log_tail_bytes: 64 * 1024,
        }
    }
}

/// Errors that can happen when persisting / loading a crash bundle.
#[derive(Debug)]
pub enum CrashBundleError {
    /// Filesystem I/O failed.
    Io(std::io::Error),
    /// JSON (de)serialization failed.
    Serialize(serde_json::Error),
    /// On-disk bundle was written by a newer schema than this build supports.
    SchemaNewer {
        /// Schema version found on disk.
        found: u32,
        /// Highest schema version this build understands.
        supported: u32,
    },
}

impl fmt::Display for CrashBundleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "crash bundle io error: {e}"),
            Self::Serialize(e) => write!(f, "crash bundle serialize error: {e}"),
            Self::SchemaNewer { found, supported } => write!(
                f,
                "crash bundle schema {found} is newer than supported {supported}"
            ),
        }
    }
}

impl StdError for CrashBundleError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Serialize(e) => Some(e),
            Self::SchemaNewer { .. } => None,
        }
    }
}

impl From<std::io::Error> for CrashBundleError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<serde_json::Error> for CrashBundleError {
    fn from(e: serde_json::Error) -> Self {
        Self::Serialize(e)
    }
}

/// Build a [`CrashBundle`] by redacting the raw log tail and stamping `now`.
#[must_use]
pub fn build_bundle(
    record: CrashRecord,
    log_tail_raw: &[String],
    env: CrashEnv,
    cfg: &CrashBundleConfig,
    now: SystemTime,
) -> CrashBundle {
    let redacted_log_tail = redact_log_lines(log_tail_raw, cfg);
    CrashBundle {
        schema_version: CRASH_BUNDLE_SCHEMA_VERSION,
        record,
        redacted_log_tail,
        env,
        created_at: now,
    }
}

// Sensitive key=value patterns. Each entry is the case-insensitive needle
// that introduces the secret; we replace everything that follows up to a
// whitespace, comma, or end-of-line with `<REDACTED>`.
// Order matters: more specific / nested-value needles run first so outer
// needles see already-redacted values and don't leave secrets dangling
// after their own replacement.
const NEEDLES_KV: &[&str] = &[
    "bearer ",
    "stratum_token=",
    "openai_api_key=",
    "anthropic_api_key=",
    "authorization:",
    "api_key",
    "apikey",
    "password",
    "token",
];

/// Redact one log line, returning the scrubbed form.
fn redact_one_line(line: &str) -> String {
    let mut out = line.to_string();

    for needle in NEEDLES_KV {
        // Re-scan after each replacement; cap at a small bound so a
        // pathological input cannot wedge the loop.
        for _ in 0..8 {
            let lower = out.to_lowercase();
            let Some(pos) = lower.find(needle) else {
                break;
            };
            let abs = pos + needle.len();
            let bytes = out.as_bytes();
            let mut value_start = abs;
            while value_start < bytes.len() {
                let c = bytes[value_start];
                if c == b'=' || c == b':' || c == b' ' || c == b'\t' || c == b'"' {
                    value_start += 1;
                } else {
                    break;
                }
            }
            let value_end = out[value_start..]
                .find(|c: char| c.is_whitespace() || c == ',' || c == '"' || c == ';')
                .map_or(out.len(), |i| value_start + i);
            if value_end > value_start {
                let slice = &out[value_start..value_end];
                if slice == "<REDACTED>" {
                    // Already redacted at this position — stop to avoid loops.
                    break;
                }
                out.replace_range(value_start..value_end, "<REDACTED>");
            } else {
                break;
            }
        }
    }

    // JWT-shaped tokens: `eyJ<chunk>.<chunk>.<chunk>`. Replace the whole token
    // with `<REDACTED_JWT>`.
    out = redact_jwts(&out);

    // Home-directory paths.
    out = redact_home_paths(&out);

    out
}

fn redact_jwts(s: &str) -> String {
    // Walk the string looking for `eyJ`. From each match, scan forward across
    // base64url chars and require exactly two `.` separators.
    let bytes = s.as_bytes();
    let mut result = String::with_capacity(s.len());
    let mut i = 0;
    while i < bytes.len() {
        if i + 3 <= bytes.len() && &bytes[i..i + 3] == b"eyJ" {
            // Try to match a JWT starting here.
            let mut j = i;
            let mut dots = 0u32;
            while j < bytes.len() {
                let c = bytes[j];
                let is_b64 = c.is_ascii_alphanumeric() || c == b'-' || c == b'_' || c == b'=';
                if is_b64 {
                    j += 1;
                } else if c == b'.' {
                    dots += 1;
                    j += 1;
                    if dots == 2 {
                        // Consume the third chunk.
                        while j < bytes.len() {
                            let cc = bytes[j];
                            if cc.is_ascii_alphanumeric() || cc == b'-' || cc == b'_' || cc == b'='
                            {
                                j += 1;
                            } else {
                                break;
                            }
                        }
                        break;
                    }
                } else {
                    break;
                }
            }
            if dots == 2 && j > i + 3 {
                result.push_str("<REDACTED_JWT>");
                i = j;
                continue;
            }
        }
        // Push one char of the input as-is.
        // Use char boundaries via the string slice to avoid splitting UTF-8.
        if let Some(ch) = s[i..].chars().next() {
            result.push(ch);
            i += ch.len_utf8();
        } else {
            break;
        }
    }
    result
}

fn redact_home_paths(s: &str) -> String {
    // Replace `/Users/<seg>` and `/home/<seg>` where `<seg>` is a non-`/`,
    // non-whitespace run.
    let mut out = String::with_capacity(s.len());
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let rest = &s[i..];
        let prefix_users = "/Users/";
        let prefix_home = "/home/";
        let matched_prefix = if rest.starts_with(prefix_users) {
            Some(prefix_users)
        } else if rest.starts_with(prefix_home) {
            Some(prefix_home)
        } else {
            None
        };
        if let Some(prefix) = matched_prefix {
            let after = i + prefix.len();
            // Find end of the user segment.
            let seg_end = s[after..]
                .find(|c: char| c == '/' || c.is_whitespace())
                .map_or(s.len(), |k| after + k);
            // Only redact if there's an actual segment, and it's not already
            // the literal `<USER>` sentinel.
            let seg = &s[after..seg_end];
            if !seg.is_empty() && seg != "<USER>" {
                out.push_str(prefix);
                out.push_str("<USER>");
                i = seg_end;
                continue;
            } else if seg == "<USER>" {
                out.push_str(prefix);
                out.push_str(seg);
                i = seg_end;
                continue;
            }
        }
        if let Some(ch) = rest.chars().next() {
            out.push(ch);
            i += ch.len_utf8();
        } else {
            break;
        }
    }
    out
}

/// Redact a slice of log lines: each line is scrubbed and then the tail is
/// truncated to fit both `max_log_tail_lines` and `max_log_tail_bytes`.
#[must_use]
pub fn redact_log_lines(lines: &[String], cfg: &CrashBundleConfig) -> Vec<String> {
    let redacted: Vec<String> = lines.iter().map(|l| redact_one_line(l)).collect();

    // Trim from the front until both the line and byte budgets are satisfied.
    let mut start = redacted.len().saturating_sub(cfg.max_log_tail_lines);
    let mut total_bytes: usize = redacted[start..].iter().map(String::len).sum();
    while start < redacted.len() && total_bytes > cfg.max_log_tail_bytes {
        total_bytes = total_bytes.saturating_sub(redacted[start].len());
        start += 1;
    }
    redacted[start..].to_vec()
}

/// Persist a [`CrashBundle`] to disk as pretty JSON and `fsync` the file.
///
/// # Errors
/// Returns [`CrashBundleError::Io`] if creating parent directories, opening
/// the file, writing, or `fsync` fails, and [`CrashBundleError::Serialize`]
/// if JSON encoding fails.
pub fn write_bundle(bundle: &CrashBundle, path: &Path) -> Result<(), CrashBundleError> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let mut file = File::create(path)?;
    serde_json::to_writer_pretty(&mut file, bundle)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    Ok(())
}

/// Load a [`CrashBundle`] from disk, rejecting newer schemas.
///
/// # Errors
/// Returns [`CrashBundleError::Io`] if the file cannot be opened or read,
/// [`CrashBundleError::Serialize`] if the contents are not valid JSON for
/// this shape, and [`CrashBundleError::SchemaNewer`] if the stored
/// `schema_version` exceeds [`CRASH_BUNDLE_SCHEMA_VERSION`].
pub fn load_bundle(path: &Path) -> Result<CrashBundle, CrashBundleError> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let bundle: CrashBundle = serde_json::from_reader(reader)?;
    if bundle.schema_version > CRASH_BUNDLE_SCHEMA_VERSION {
        return Err(CrashBundleError::SchemaNewer {
            found: bundle.schema_version,
            supported: CRASH_BUNDLE_SCHEMA_VERSION,
        });
    }
    Ok(bundle)
}

/// Collapse the user segment in `/Users/<name>/...` or `/home/<name>/...` to
/// the `<USER>` sentinel. Paths that don't match are returned unchanged.
#[must_use]
pub fn redact_path_user(path: &Path) -> PathBuf {
    let s = path.to_string_lossy().into_owned();
    let redacted = redact_home_paths(&s);
    PathBuf::from(redacted)
}

#[cfg(test)]
mod tests {
    use time::OffsetDateTime;

    use super::*;

    fn record_for_test() -> CrashRecord {
        let now = OffsetDateTime::from_unix_timestamp(1_700_000_000).unwrap();
        CrashRecord::synthesize("0.1.0", "boom", Some("a.rs:1:2".into()), "main", now)
    }

    fn env_for_test() -> CrashEnv {
        CrashEnv {
            app_version: "0.1.0".into(),
            channel: "stable".into(),
            os: "macos".into(),
            cpu_arch: "aarch64".into(),
        }
    }

    #[test]
    fn config_default_values_match() {
        let cfg = CrashBundleConfig::default();
        assert_eq!(cfg.max_log_tail_lines, 200);
        assert_eq!(cfg.max_log_tail_bytes, 64 * 1024);
    }

    #[test]
    fn redact_authorization_bearer_jwt_line() {
        let cfg = CrashBundleConfig::default();
        let lines = vec!["Authorization: Bearer abc.def.ghi".to_string()];
        let out = redact_log_lines(&lines, &cfg);
        assert_eq!(out.len(), 1);
        assert!(out[0].contains("<REDACTED>"), "got: {}", out[0]);
        assert!(!out[0].contains("abc.def.ghi"), "got: {}", out[0]);
    }

    #[test]
    fn redact_api_key_kv() {
        let cfg = CrashBundleConfig::default();
        let lines = vec!["loaded api_key=sk-1234567890".to_string()];
        let out = redact_log_lines(&lines, &cfg);
        assert!(out[0].contains("<REDACTED>"));
        assert!(!out[0].contains("sk-1234567890"));
    }

    #[test]
    fn redact_apikey_case_insensitive() {
        let cfg = CrashBundleConfig::default();
        let lines = vec!["APIKEY=ZZZZ9999".to_string()];
        let out = redact_log_lines(&lines, &cfg);
        assert!(out[0].contains("<REDACTED>"));
        assert!(!out[0].contains("ZZZZ9999"));
    }

    #[test]
    fn redact_jwt_shaped_token() {
        let cfg = CrashBundleConfig::default();
        let lines =
            vec!["got eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ4In0.SflKxwRJSMeKKF for user".to_string()];
        let out = redact_log_lines(&lines, &cfg);
        assert!(out[0].contains("<REDACTED_JWT>"), "got: {}", out[0]);
        assert!(!out[0].contains("eyJhbGciOiJIUzI1NiJ9"));
    }

    #[test]
    fn redact_users_path() {
        let cfg = CrashBundleConfig::default();
        let lines = vec!["opened /Users/krish/projects/stratum/foo".to_string()];
        let out = redact_log_lines(&lines, &cfg);
        assert!(
            out[0].contains("/Users/<USER>/projects/stratum/foo"),
            "got: {}",
            out[0]
        );
        assert!(!out[0].contains("krish"));
    }

    #[test]
    fn redact_home_path() {
        let cfg = CrashBundleConfig::default();
        let lines = vec!["opened /home/krish/projects/stratum/foo".to_string()];
        let out = redact_log_lines(&lines, &cfg);
        assert!(
            out[0].contains("/home/<USER>/projects/stratum/foo"),
            "got: {}",
            out[0]
        );
        assert!(!out[0].contains("krish"));
    }

    #[test]
    fn redact_stratum_token_env() {
        let cfg = CrashBundleConfig::default();
        let lines = vec!["env STRATUM_TOKEN=xyz123 set".to_string()];
        let out = redact_log_lines(&lines, &cfg);
        assert!(out[0].contains("<REDACTED>"));
        assert!(!out[0].contains("xyz123"));
    }

    #[test]
    fn redact_truncates_to_line_budget() {
        let cfg = CrashBundleConfig {
            max_log_tail_lines: 3,
            max_log_tail_bytes: 1_000_000,
        };
        let lines: Vec<String> = (0..10).map(|i| format!("line-{i}")).collect();
        let out = redact_log_lines(&lines, &cfg);
        assert_eq!(out.len(), 3);
        assert_eq!(out[0], "line-7");
        assert_eq!(out[2], "line-9");
    }

    #[test]
    fn redact_truncates_to_byte_budget() {
        let cfg = CrashBundleConfig {
            max_log_tail_lines: 1000,
            max_log_tail_bytes: 20,
        };
        let lines: Vec<String> = (0..20).map(|i| format!("xx-{i:02}")).collect();
        let out = redact_log_lines(&lines, &cfg);
        let total: usize = out.iter().map(String::len).sum();
        assert!(total <= 20, "total={total}");
        assert!(!out.is_empty());
        // Most recent lines retained.
        assert_eq!(out[out.len() - 1], "xx-19");
    }

    #[test]
    fn redact_preserves_neutral_lines() {
        let cfg = CrashBundleConfig::default();
        let lines = vec!["nothing to scrub here".to_string()];
        let out = redact_log_lines(&lines, &cfg);
        assert_eq!(out[0], "nothing to scrub here");
    }

    #[test]
    fn redact_is_idempotent() {
        let cfg = CrashBundleConfig::default();
        let lines = vec![
            "Authorization: Bearer abc.def.ghi".to_string(),
            "/Users/krish/x".to_string(),
            "api_key=sk-xyz".to_string(),
            "boring line".to_string(),
        ];
        let once = redact_log_lines(&lines, &cfg);
        let twice = redact_log_lines(&once, &cfg);
        assert_eq!(once, twice);
    }

    #[test]
    fn build_bundle_populates_every_field() {
        let cfg = CrashBundleConfig::default();
        let now = SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(1_700_000_000);
        let bundle = build_bundle(
            record_for_test(),
            &["api_key=foo".to_string()],
            env_for_test(),
            &cfg,
            now,
        );
        assert_eq!(bundle.schema_version, CRASH_BUNDLE_SCHEMA_VERSION);
        assert_eq!(bundle.record.message, "boom");
        assert_eq!(bundle.env.app_version, "0.1.0");
        assert_eq!(bundle.created_at, now);
        assert_eq!(bundle.redacted_log_tail.len(), 1);
        assert!(bundle.redacted_log_tail[0].contains("<REDACTED>"));
    }

    #[test]
    fn redact_path_user_collapses_users() {
        let out = redact_path_user(Path::new("/Users/krish/projects/foo"));
        assert_eq!(out, PathBuf::from("/Users/<USER>/projects/foo"));
    }

    #[test]
    fn redact_path_user_leaves_unrelated_paths() {
        let out = redact_path_user(Path::new("/var/log/system.log"));
        assert_eq!(out, PathBuf::from("/var/log/system.log"));
    }

    #[test]
    fn redact_path_user_collapses_home() {
        let out = redact_path_user(Path::new("/home/alice/work"));
        assert_eq!(out, PathBuf::from("/home/<USER>/work"));
    }

    #[test]
    fn write_and_load_roundtrip() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("crash.json");
        let cfg = CrashBundleConfig::default();
        let bundle = build_bundle(
            record_for_test(),
            &["plain".to_string()],
            env_for_test(),
            &cfg,
            SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(1_700_000_000),
        );
        write_bundle(&bundle, &path).unwrap();
        let loaded = load_bundle(&path).unwrap();
        assert_eq!(bundle, loaded);
    }

    #[test]
    fn load_bundle_rejects_newer_schema() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("crash.json");
        let json = serde_json::json!({
            "schema_version": 999u32,
            "record": {
                "schema_version": 1u32,
                "code": "STRAT-E9001",
                "at": "2026-06-04T18:11:02Z",
                "stratum_version": "0.1.0",
                "message": "boom",
                "location": null,
                "thread": "main",
            },
            "redacted_log_tail": [],
            "env": {
                "app_version": "0.1.0",
                "channel": "stable",
                "os": "macos",
                "cpu_arch": "aarch64",
            },
            "created_at": { "secs_since_epoch": 0u64, "nanos_since_epoch": 0u32 },
        });
        std::fs::write(&path, serde_json::to_string(&json).unwrap()).unwrap();
        let err = load_bundle(&path).unwrap_err();
        match err {
            CrashBundleError::SchemaNewer { found, supported } => {
                assert_eq!(found, 999);
                assert_eq!(supported, CRASH_BUNDLE_SCHEMA_VERSION);
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn load_bundle_errors_on_malformed_json() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("crash.json");
        std::fs::write(&path, b"not-json{").unwrap();
        let err = load_bundle(&path).unwrap_err();
        assert!(matches!(err, CrashBundleError::Serialize(_)));
    }

    #[test]
    fn load_bundle_errors_on_missing_file() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("does-not-exist.json");
        let err = load_bundle(&path).unwrap_err();
        assert!(matches!(err, CrashBundleError::Io(_)));
    }

    #[test]
    fn error_display_smoke() {
        let io = CrashBundleError::Io(std::io::Error::other("x"));
        assert!(format!("{io}").contains("io error"));
        let serde_err: serde_json::Error = serde_json::from_str::<CrashBundle>("nope").unwrap_err();
        let ser = CrashBundleError::Serialize(serde_err);
        assert!(format!("{ser}").contains("serialize"));
        let sn = CrashBundleError::SchemaNewer {
            found: 2,
            supported: 1,
        };
        let msg = format!("{sn}");
        assert!(msg.contains('2') && msg.contains('1'));
    }

    #[test]
    fn error_source_chain() {
        let io = CrashBundleError::Io(std::io::Error::other("x"));
        assert!(io.source().is_some());
        let sn = CrashBundleError::SchemaNewer {
            found: 2,
            supported: 1,
        };
        assert!(sn.source().is_none());
    }

    #[test]
    fn crash_env_serde_roundtrip() {
        let env = env_for_test();
        let s = serde_json::to_string(&env).unwrap();
        let back: CrashEnv = serde_json::from_str(&s).unwrap();
        assert_eq!(env, back);
    }

    #[test]
    fn crash_bundle_serde_roundtrip() {
        let cfg = CrashBundleConfig::default();
        let bundle = build_bundle(
            record_for_test(),
            &["x".to_string()],
            env_for_test(),
            &cfg,
            SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(42),
        );
        let s = serde_json::to_string(&bundle).unwrap();
        let back: CrashBundle = serde_json::from_str(&s).unwrap();
        assert_eq!(bundle, back);
    }

    #[test]
    fn redact_one_line_handles_unicode() {
        let cfg = CrashBundleConfig::default();
        let lines = vec!["héllo world – api_key=secret-€".to_string()];
        let out = redact_log_lines(&lines, &cfg);
        assert!(out[0].contains("<REDACTED>"));
        assert!(!out[0].contains("secret"));
    }

    #[test]
    fn redact_handles_empty_input() {
        let cfg = CrashBundleConfig::default();
        let out = redact_log_lines(&[], &cfg);
        assert!(out.is_empty());
    }

    #[test]
    fn redact_path_user_does_not_double_collapse() {
        // Calling twice yields the same value.
        let once = redact_path_user(Path::new("/Users/krish/x"));
        let twice = redact_path_user(&once);
        assert_eq!(once, twice);
    }

    #[test]
    fn config_is_copy() {
        let cfg = CrashBundleConfig::default();
        let cfg2 = cfg;
        assert_eq!(cfg.max_log_tail_lines, cfg2.max_log_tail_lines);
    }

    #[test]
    fn write_bundle_creates_parent_dirs() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("nested").join("deeper").join("crash.json");
        let cfg = CrashBundleConfig::default();
        let bundle = build_bundle(
            record_for_test(),
            &[],
            env_for_test(),
            &cfg,
            SystemTime::UNIX_EPOCH,
        );
        write_bundle(&bundle, &path).unwrap();
        assert!(path.exists());
    }

    #[test]
    fn write_bundle_io_error_on_invalid_path() {
        let tmp = tempfile::TempDir::new().unwrap();
        let blocker = tmp.path().join("blocker");
        std::fs::write(&blocker, b"x").unwrap();
        let path = blocker.join("crash.json");
        let cfg = CrashBundleConfig::default();
        let bundle = build_bundle(
            record_for_test(),
            &[],
            env_for_test(),
            &cfg,
            SystemTime::UNIX_EPOCH,
        );
        let err = write_bundle(&bundle, &path).unwrap_err();
        assert!(matches!(err, CrashBundleError::Io(_)));
    }

    #[test]
    fn redact_jwt_only_when_three_parts() {
        // `eyJfoo.bar` has only one dot — should NOT be treated as JWT.
        let cfg = CrashBundleConfig::default();
        let lines = vec!["look eyJabc.def end".to_string()];
        let out = redact_log_lines(&lines, &cfg);
        assert!(out[0].contains("eyJabc.def"), "got: {}", out[0]);
    }

    #[test]
    fn from_io_and_serde_errors() {
        let io: CrashBundleError = std::io::Error::other("x").into();
        assert!(matches!(io, CrashBundleError::Io(_)));
        let serde_err: serde_json::Error = serde_json::from_str::<CrashBundle>("oops").unwrap_err();
        let se: CrashBundleError = serde_err.into();
        assert!(matches!(se, CrashBundleError::Serialize(_)));
    }
}
