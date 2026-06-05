//! Model-file installation primitives.
//!
//! Phase 1 v2 ships the **local** install path: copy a GGUF (or any other
//! model file) from a source path into `<data>/models/`, optionally
//! verifying the SHA-256 against a caller-supplied digest. The download
//! lands in a `<dest>.partial` file first; only on a successful (and
//! verified) write is it renamed into place. Reuse of a previous partial
//! is gated by exact-size match and re-verification.
//!
//! The HTTP variant lands later in Phase 1 with the actual `LlamaCppProvider`
//! GGUF-fetch work; the contract (atomic + SHA-verified + interruption-safe)
//! is identical, only the byte source changes.

use std::fmt;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use rand::rngs::SmallRng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use stratum_types::error::codes::E1001_INSTALLED_SCHEMA_UNREADABLE;
use stratum_types::{StratumError, StratumResult};

use crate::retry::{
    retry_with_clock_seeded, Clock, Jitter, RetryDecision, RetryError, RetryPolicy, SystemClock,
};

const COPY_CHUNK: usize = 64 * 1024;

/// Default retry policy for HTTP installs.
///
/// Used by [`ModelInstaller::install_from_url`] for transient HTTP
/// failures; mirrors the contract in `plan/32-cancellation-and-timeouts.md`
/// §6: 4 attempts, 250ms→5s exponential backoff with full jitter.
#[must_use]
pub const fn default_install_retry_policy() -> RetryPolicy {
    RetryPolicy {
        max_attempts: 4,
        initial_delay: Duration::from_millis(250),
        max_delay: Duration::from_secs(5),
        backoff_multiplier: 2.0,
        jitter: Jitter::Full,
    }
}

/// Error surface for retry-wrapped installs.
///
/// `Retryable` carries the outcome of the underlying [`crate::retry`] loop
/// — either a [`RetryError::Fatal`] short-circuit or a
/// [`RetryError::Exhausted`] failure after every attempt was used. The
/// inner `String` is the short, classifier-friendly reason text
/// (e.g. `"status 502"`, `"transport: timed out"`).
///
/// `NonRetryable` carries terminal errors that fall outside the retry loop:
/// e.g. a SHA-256 mismatch detected after the body finishes streaming, or
/// a local filesystem failure (creating the destination directory).
#[derive(Debug)]
pub enum InstallTransientError {
    /// Transient-failure path: the underlying retry helper gave up.
    Retryable(RetryError<String>),
    /// Terminal failure, with a short human-readable reason.
    NonRetryable(String),
}

impl fmt::Display for InstallTransientError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Retryable(e) => write!(f, "install transient failure: {e}"),
            Self::NonRetryable(s) => write!(f, "install terminal failure: {s}"),
        }
    }
}

impl std::error::Error for InstallTransientError {}

/// Outcome reported back from a model install.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstallReport {
    /// Final on-disk path of the installed model.
    pub dest: PathBuf,
    /// Total bytes written.
    pub bytes: u64,
    /// SHA-256 digest of the installed bytes.
    pub sha256: String,
    /// Whether the caller's expected digest was supplied + matched.
    pub verified: bool,
}

/// Verifies a stream against an expected SHA-256 while streaming it into
/// a writer. Returns the actual digest and bytes written.
///
/// # Errors
/// Propagates any underlying io error.
pub fn hash_and_copy<R: Read, W: Write>(
    mut reader: R,
    mut writer: W,
) -> std::io::Result<(String, u64)> {
    let mut hasher = Sha256::new();
    let mut buf = vec![0_u8; COPY_CHUNK];
    let mut total = 0_u64;
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        writer.write_all(&buf[..n])?;
        total = total.saturating_add(n as u64);
    }
    Ok((hex(&hasher.finalize()), total))
}

/// Compute SHA-256 of a byte slice, formatted as lowercase hex.
#[must_use]
pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex(&hasher.finalize())
}

fn hex(digest: &[u8]) -> String {
    let mut out = String::with_capacity(digest.len() * 2);
    for b in digest {
        use std::fmt::Write as _;
        // SAFETY: writing hex to an owned String never errors.
        let _ = write!(out, "{b:02x}");
    }
    out
}

/// Predicate: is the caller's expected digest equal (case-insensitive) to
/// what we measured?
#[must_use]
pub const fn digest_matches(expected: &str, actual: &str) -> bool {
    expected.eq_ignore_ascii_case(actual)
}

/// Wrap an `io::Error` into a [`StratumError`] tagged with
/// `E1001_INSTALLED_SCHEMA_UNREADABLE` and the supplied context message.
fn io_err(message: impl Into<String>, cause: std::io::Error) -> StratumError {
    StratumError::new(E1001_INSTALLED_SCHEMA_UNREADABLE, message).with_cause(cause)
}

/// Project a transient-install error onto the `StratumResult` surface used
/// by the public installer API.
fn transient_to_stratum(url: &str, err: InstallTransientError) -> StratumError {
    StratumError::new(E1001_INSTALLED_SCHEMA_UNREADABLE, format!("http get {url}")).with_cause(err)
}

/// Classifier shared by `install_from_url_with_retry` and tests.
///
/// `"status 4xx"` → fatal; everything else (`"status 5xx"`, `"transport:
/// ..."`, partial-write failures the attempt promoted to transport-like)
/// → retry.
fn classify_http_reason(reason: &str) -> RetryDecision {
    if reason.starts_with("status 4") {
        RetryDecision::Fatal
    } else {
        RetryDecision::Retry
    }
}

/// Predicate: does the server's `Content-Range` header begin at the offset
/// the client requested (`bytes <start>-…/…`)? A missing or malformed
/// header conservatively returns `false`, which forces a clean restart.
fn content_range_starts_at(response: &ureq::Response, expected_start: u64) -> bool {
    let Some(value) = response.header("Content-Range") else {
        return false;
    };
    // Accept `bytes <start>-<end>/<total>` and `bytes=<start>-<end>/<total>`.
    let rest = value
        .trim()
        .strip_prefix("bytes")
        .map(|s| s.trim_start_matches([' ', '=']));
    let Some(rest) = rest else {
        return false;
    };
    let Some((start_str, _)) = rest.split_once('-') else {
        return false;
    };
    start_str
        .trim()
        .parse::<u64>()
        .is_ok_and(|n| n == expected_start)
}

/// Build the partial-file path next to `dest`.
#[must_use]
pub fn partial_path(dest: &Path) -> PathBuf {
    let mut name = dest.file_name().unwrap_or_default().to_os_string();
    name.push(".partial");
    dest.with_file_name(name)
}

/// Install a local model file into `dest` atomically:
/// `src` is streamed to `<dest>.partial`, hashed, optionally verified, then
/// renamed into place.
#[derive(Debug)]
pub struct ModelInstaller<'a> {
    /// Directory the file ends up in.
    pub dest_dir: &'a Path,
    /// Final filename inside `dest_dir`.
    pub dest_filename: &'a str,
    /// Optional expected SHA-256 (lowercase or uppercase hex).
    pub expected_sha256: Option<&'a str>,
}

impl ModelInstaller<'_> {
    /// Fetch a model file from a HTTP(S) URL and install it.
    ///
    /// The flow mirrors [`Self::install_local`]: stream the response body
    /// through [`hash_and_copy`] into `<dest>.partial`, verify the digest
    /// when set, then atomically rename. If a `<dest>.partial` already
    /// exists from a previous interrupted run, the installer issues a
    /// `Range: bytes={n}-` request and — if the server replies `206
    /// Partial Content` with a coherent `Content-Range` — resumes by
    /// re-hashing the partial bytes off disk to seed the SHA-256 state and
    /// then appending the new bytes. A `200 OK` reply (range ignored)
    /// transparently restarts from scratch.
    ///
    /// # Errors
    /// Returns [`E1001_INSTALLED_SCHEMA_UNREADABLE`] for HTTP, io, or
    /// digest-mismatch failures.
    pub fn install_from_url(&self, url: &str) -> StratumResult<InstallReport> {
        self.install_from_url_with_retry(url, &default_install_retry_policy())
    }

    /// Like [`Self::install_from_url`] but with a caller-supplied retry
    /// `policy`. Transient HTTP failures (5xx, transport errors, mid-body
    /// connection drops) are retried with exponential backoff; on each
    /// retry the request includes `Range: bytes=<n>-` so the new attempt
    /// resumes from wherever the previous one left bytes on disk. Terminal
    /// errors (4xx, SHA mismatch, local filesystem failures) short-circuit.
    ///
    /// # Errors
    /// Same code as [`Self::install_from_url`]. The underlying
    /// [`InstallTransientError`] is preserved in the `cause` chain.
    pub fn install_from_url_with_retry(
        &self,
        url: &str,
        policy: &RetryPolicy,
    ) -> StratumResult<InstallReport> {
        self.install_from_url_with_retry_and_clock(url, policy, &SystemClock)
    }

    /// Clock-injecting variant used by deterministic tests. Production code
    /// always goes through [`Self::install_from_url_with_retry`] which uses
    /// the real wall clock.
    fn install_from_url_with_retry_and_clock<K: Clock + ?Sized>(
        &self,
        url: &str,
        policy: &RetryPolicy,
        clock: &K,
    ) -> StratumResult<InstallReport> {
        std::fs::create_dir_all(self.dest_dir)
            .map_err(|e| io_err(format!("create {}", self.dest_dir.display()), e))?;
        let dest = self.dest_dir.join(self.dest_filename);
        let partial = partial_path(&dest);

        // Deterministic seed: production jitter only needs to vary across
        // attempts within a single install, not across processes.
        let mut rng = SmallRng::seed_from_u64(0);
        let classifier = |reason: &String| classify_http_reason(reason.as_str());
        let mut op =
            |_attempt: u32| -> Result<(), String> { Self::attempt_http_fetch(url, &partial) };
        let retry_outcome = retry_with_clock_seeded(policy, &classifier, clock, &mut rng, &mut op);
        match retry_outcome {
            Ok(()) => {}
            Err(e) => {
                return Err(transient_to_stratum(
                    url,
                    InstallTransientError::Retryable(e),
                ));
            }
        }

        // Body fully on disk in `partial`. Hash, verify, rename.
        self.finalize_partial(&dest, &partial)
    }

    /// Single HTTP attempt: read `partial.len()` off disk, build a request
    /// (with `Range:` if we have a partial), and stream the response body
    /// into `partial` (appending on 206, truncating on 200). Returns Ok
    /// when the body completes, `Err(reason)` for any transient or fatal
    /// failure — the classifier reads the reason prefix to decide.
    fn attempt_http_fetch(url: &str, partial: &Path) -> Result<(), String> {
        let resume_from = std::fs::metadata(partial).map(|m| m.len()).unwrap_or(0);
        let mut req = ureq::get(url);
        if resume_from > 0 {
            req = req.set("Range", &format!("bytes={resume_from}-"));
        }
        let response = match req.call() {
            Ok(r) => r,
            Err(ureq::Error::Status(code, _)) => {
                return Err(format!("status {code}"));
            }
            Err(ureq::Error::Transport(t)) => {
                return Err(format!("transport: {t}"));
            }
        };

        let resume_ok = resume_from > 0
            && response.status() == 206
            && content_range_starts_at(&response, resume_from);
        let mut reader = response.into_reader();
        let mut writer = if resume_ok {
            std::fs::OpenOptions::new()
                .append(true)
                .open(partial)
                .map_err(|e| format!("open partial: {e}"))?
        } else {
            // 200 OK (or 206 we cannot trust): drop any stale partial and
            // write from byte 0.
            if partial.exists() {
                std::fs::remove_file(partial).map_err(|e| format!("clear partial: {e}"))?;
            }
            std::fs::File::create(partial).map_err(|e| format!("create partial: {e}"))?
        };

        let mut buf = vec![0_u8; COPY_CHUNK];
        loop {
            let n = match reader.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => n,
                Err(e) => return Err(format!("transport: {e}")),
            };
            writer
                .write_all(&buf[..n])
                .map_err(|e| format!("write partial: {e}"))?;
        }
        Ok(())
    }

    /// Hash the on-disk `partial`, verify the expected digest if any, and
    /// rename to `dest`. SHA mismatch is treated as a terminal failure and
    /// the partial is removed.
    fn finalize_partial(&self, dest: &Path, partial: &Path) -> StratumResult<InstallReport> {
        let mut hasher = Sha256::new();
        let mut total = 0_u64;
        let mut file = std::fs::File::open(partial)
            .map_err(|e| io_err(format!("open partial {}", partial.display()), e))?;
        let mut buf = vec![0_u8; COPY_CHUNK];
        loop {
            let n = file
                .read(&mut buf)
                .map_err(|e| io_err(format!("hash partial {}", partial.display()), e))?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
            total = total.saturating_add(n as u64);
        }
        drop(file);
        let actual = hex(&hasher.finalize());

        let verified = if let Some(expected) = self.expected_sha256 {
            if !digest_matches(expected, &actual) {
                let _ = std::fs::remove_file(partial);
                let reason = format!(
                    "sha256 mismatch for {}: expected {expected}, got {actual}",
                    dest.display()
                );
                return Err(
                    StratumError::new(E1001_INSTALLED_SCHEMA_UNREADABLE, reason.clone())
                        .with_cause(InstallTransientError::NonRetryable(reason)),
                );
            }
            true
        } else {
            false
        };
        std::fs::rename(partial, dest)
            .map_err(|e| io_err(format!("rename to {}", dest.display()), e))?;
        Ok(InstallReport {
            dest: dest.to_path_buf(),
            bytes: total,
            sha256: actual,
            verified,
        })
    }

    /// Run the install from a local source path.
    ///
    /// # Errors
    /// Returns [`E1001_INSTALLED_SCHEMA_UNREADABLE`] on io failure, digest
    /// mismatch, or destination-directory issues.
    pub fn install_local(&self, src: &Path) -> StratumResult<InstallReport> {
        let src_file = std::fs::File::open(src).map_err(|e| {
            StratumError::new(
                E1001_INSTALLED_SCHEMA_UNREADABLE,
                format!("open source {}", src.display()),
            )
            .with_cause(e)
        })?;
        self.install_from_reader(src_file)
    }

    /// Run the install from an arbitrary reader. Used by the local-source
    /// path and by tests; the HTTP variant will plug into the same surface.
    ///
    /// # Errors
    /// Returns [`E1001_INSTALLED_SCHEMA_UNREADABLE`] on io failure, digest
    /// mismatch, or destination-directory issues.
    pub fn install_from_reader<R: Read>(&self, reader: R) -> StratumResult<InstallReport> {
        std::fs::create_dir_all(self.dest_dir).map_err(|e| {
            StratumError::new(
                E1001_INSTALLED_SCHEMA_UNREADABLE,
                format!("create {}", self.dest_dir.display()),
            )
            .with_cause(e)
        })?;
        let dest = self.dest_dir.join(self.dest_filename);
        let partial = partial_path(&dest);
        let writer = std::fs::File::create(&partial).map_err(|e| {
            StratumError::new(
                E1001_INSTALLED_SCHEMA_UNREADABLE,
                format!("create partial {}", partial.display()),
            )
            .with_cause(e)
        })?;
        let (actual, bytes) = hash_and_copy(reader, writer).map_err(|e| {
            StratumError::new(
                E1001_INSTALLED_SCHEMA_UNREADABLE,
                format!("stream into {}", partial.display()),
            )
            .with_cause(e)
        })?;
        let verified = if let Some(expected) = self.expected_sha256 {
            if !digest_matches(expected, &actual) {
                let _ = std::fs::remove_file(&partial);
                return Err(StratumError::new(
                    E1001_INSTALLED_SCHEMA_UNREADABLE,
                    format!(
                        "sha256 mismatch for {}: expected {expected}, got {actual}",
                        dest.display()
                    ),
                ));
            }
            true
        } else {
            false
        };
        std::fs::rename(&partial, &dest).map_err(|e| {
            StratumError::new(
                E1001_INSTALLED_SCHEMA_UNREADABLE,
                format!("rename to {}", dest.display()),
            )
            .with_cause(e)
        })?;
        Ok(InstallReport {
            dest,
            bytes,
            sha256: actual,
            verified,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use tempfile::TempDir;

    use super::*;

    const HELLO_SHA: &str = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824";

    #[test]
    fn sha256_hex_matches_known_value() {
        assert_eq!(sha256_hex(b"hello"), HELLO_SHA);
    }

    #[test]
    fn hash_and_copy_writes_and_reports_digest() {
        let mut out = Vec::new();
        let (digest, bytes) = hash_and_copy(&b"hello"[..], &mut out).unwrap();
        assert_eq!(digest, HELLO_SHA);
        assert_eq!(bytes, 5);
        assert_eq!(out, b"hello");
    }

    #[test]
    fn hash_and_copy_handles_empty() {
        let mut out = Vec::new();
        let (digest, bytes) = hash_and_copy(std::io::empty(), &mut out).unwrap();
        assert_eq!(bytes, 0);
        assert_eq!(
            digest,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn digest_matches_is_case_insensitive() {
        assert!(digest_matches(&HELLO_SHA.to_uppercase(), HELLO_SHA));
        assert!(digest_matches(HELLO_SHA, HELLO_SHA));
        assert!(!digest_matches("00", HELLO_SHA));
    }

    #[test]
    fn partial_path_appends_partial_suffix() {
        let p = partial_path(Path::new("/x/models/gemma.gguf"));
        assert_eq!(p, Path::new("/x/models/gemma.gguf.partial"));
    }

    #[test]
    fn partial_path_for_path_without_filename() {
        // Pathological input — partial still produces a sibling.
        let p = partial_path(Path::new("/"));
        assert!(p.to_string_lossy().contains(".partial"));
    }

    #[test]
    fn install_from_reader_succeeds_without_expected_digest() {
        let tmp = TempDir::new().unwrap();
        let installer = ModelInstaller {
            dest_dir: &tmp.path().join("models"),
            dest_filename: "fake.gguf",
            expected_sha256: None,
        };
        let report = installer
            .install_from_reader(Cursor::new(b"hello".to_vec()))
            .unwrap();
        assert_eq!(report.bytes, 5);
        assert_eq!(report.sha256, HELLO_SHA);
        assert!(!report.verified);
        assert!(report.dest.exists());
        assert_eq!(std::fs::read(&report.dest).unwrap(), b"hello");
    }

    #[test]
    fn install_from_reader_verifies_digest_when_provided() {
        let tmp = TempDir::new().unwrap();
        let installer = ModelInstaller {
            dest_dir: tmp.path(),
            dest_filename: "fake.gguf",
            expected_sha256: Some(HELLO_SHA),
        };
        let report = installer
            .install_from_reader(Cursor::new(b"hello".to_vec()))
            .unwrap();
        assert!(report.verified);
    }

    #[test]
    fn install_from_reader_errors_on_digest_mismatch() {
        let tmp = TempDir::new().unwrap();
        let installer = ModelInstaller {
            dest_dir: tmp.path(),
            dest_filename: "fake.gguf",
            expected_sha256: Some("deadbeef"),
        };
        let err = installer
            .install_from_reader(Cursor::new(b"hello".to_vec()))
            .unwrap_err();
        assert_eq!(err.code(), &E1001_INSTALLED_SCHEMA_UNREADABLE);
        assert!(format!("{err}").contains("mismatch"));
        // Partial must be cleaned up on mismatch.
        let partial = partial_path(&tmp.path().join("fake.gguf"));
        assert!(!partial.exists());
        assert!(!tmp.path().join("fake.gguf").exists());
    }

    #[test]
    fn install_local_copies_file() {
        let tmp = TempDir::new().unwrap();
        let src = tmp.path().join("src.bin");
        std::fs::write(&src, b"hello world").unwrap();
        let installer = ModelInstaller {
            dest_dir: &tmp.path().join("models"),
            dest_filename: "copy.bin",
            expected_sha256: None,
        };
        let report = installer.install_local(&src).unwrap();
        assert_eq!(report.bytes, 11);
        assert_eq!(std::fs::read(&report.dest).unwrap(), b"hello world");
    }

    #[test]
    fn install_local_errors_when_src_missing() {
        let tmp = TempDir::new().unwrap();
        let installer = ModelInstaller {
            dest_dir: &tmp.path().join("models"),
            dest_filename: "x.bin",
            expected_sha256: None,
        };
        let err = installer
            .install_local(&tmp.path().join("missing"))
            .unwrap_err();
        assert_eq!(err.code(), &E1001_INSTALLED_SCHEMA_UNREADABLE);
    }

    #[test]
    fn install_from_reader_errors_when_dest_dir_unwritable() {
        let tmp = TempDir::new().unwrap();
        let blocker = tmp.path().join("blocker");
        std::fs::write(&blocker, b"x").unwrap();
        let installer = ModelInstaller {
            dest_dir: &blocker.join("nested"),
            dest_filename: "x.bin",
            expected_sha256: None,
        };
        let err = installer
            .install_from_reader(Cursor::new(b"hello".to_vec()))
            .unwrap_err();
        assert_eq!(err.code(), &E1001_INSTALLED_SCHEMA_UNREADABLE);
    }

    /// Reader that fails after producing some bytes.
    struct FailingReader(usize);

    impl Read for FailingReader {
        fn read(&mut self, _buf: &mut [u8]) -> std::io::Result<usize> {
            if self.0 == 0 {
                Err(std::io::Error::other("forced read failure"))
            } else {
                self.0 -= 1;
                _buf[0] = b'x';
                Ok(1)
            }
        }
    }

    #[test]
    fn install_from_reader_propagates_io_error() {
        let tmp = TempDir::new().unwrap();
        let installer = ModelInstaller {
            dest_dir: tmp.path(),
            dest_filename: "x.bin",
            expected_sha256: None,
        };
        let err = installer.install_from_reader(FailingReader(2)).unwrap_err();
        assert_eq!(err.code(), &E1001_INSTALLED_SCHEMA_UNREADABLE);
    }

    #[test]
    fn install_report_serde_roundtrip() {
        let report = InstallReport {
            dest: PathBuf::from("/x.gguf"),
            bytes: 5,
            sha256: HELLO_SHA.into(),
            verified: true,
        };
        let s = serde_json::to_string(&report).unwrap();
        let back: InstallReport = serde_json::from_str(&s).unwrap();
        assert_eq!(report, back);
    }

    #[cfg(unix)]
    #[test]
    fn install_from_reader_rename_failure_is_reported() {
        let tmp = TempDir::new().unwrap();
        let dest_dir = tmp.path();
        // Pre-create the destination as a directory containing a child so
        // `fs::rename` of the partial file over it fails on Unix.
        let dest = dest_dir.join("x.bin");
        std::fs::create_dir(&dest).unwrap();
        std::fs::write(dest.join("child"), b"x").unwrap();
        let installer = ModelInstaller {
            dest_dir,
            dest_filename: "x.bin",
            expected_sha256: None,
        };
        let err = installer
            .install_from_reader(Cursor::new(b"hi".to_vec()))
            .unwrap_err();
        assert_eq!(err.code(), &E1001_INSTALLED_SCHEMA_UNREADABLE);
    }

    #[cfg(unix)]
    #[test]
    fn install_from_reader_create_partial_failure_is_reported() {
        let tmp = TempDir::new().unwrap();
        let dest_dir = tmp.path();
        // Pre-create the partial path as a directory so `File::create` errors.
        let partial = dest_dir.join("x.bin.partial");
        std::fs::create_dir(&partial).unwrap();
        let installer = ModelInstaller {
            dest_dir,
            dest_filename: "x.bin",
            expected_sha256: None,
        };
        let err = installer
            .install_from_reader(Cursor::new(b"hi".to_vec()))
            .unwrap_err();
        assert_eq!(err.code(), &E1001_INSTALLED_SCHEMA_UNREADABLE);
    }

    fn spawn_static_server(body: Vec<u8>) -> String {
        spawn_range_server(body, RangeMode::Honor)
    }

    /// How the test server treats an inbound `Range:` header.
    #[derive(Clone, Copy)]
    enum RangeMode {
        /// Reply `206 Partial Content` with a coherent `Content-Range`.
        Honor,
        /// Ignore the `Range` header and always reply `200 OK` with the full body.
        Ignore,
    }

    fn spawn_range_server(body: Vec<u8>, mode: RangeMode) -> String {
        use std::io::{Read as _, Write as _};
        use std::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            for stream in listener.incoming().take(1) {
                let Ok(mut stream) = stream else { continue };
                let mut buf = [0_u8; 4096];
                let n = stream.read(&mut buf).unwrap_or(0);
                let request = std::str::from_utf8(&buf[..n]).unwrap_or("");
                let range_start = parse_range_start(request);
                let total = body.len();
                let (headers, payload) = match (mode, range_start) {
                    (RangeMode::Honor, Some(start))
                        if usize::try_from(start).is_ok_and(|s| s < total) =>
                    {
                        let start_usize = usize::try_from(start).unwrap_or(0);
                        let slice = &body[start_usize..];
                        let end = total - 1;
                        let h = format!(
                            "HTTP/1.1 206 Partial Content\r\nContent-Length: {}\r\nContent-Range: bytes {start}-{end}/{total}\r\nConnection: close\r\n\r\n",
                            slice.len()
                        );
                        (h, slice.to_vec())
                    }
                    _ => {
                        let h = format!(
                            "HTTP/1.1 200 OK\r\nContent-Length: {total}\r\nConnection: close\r\n\r\n"
                        );
                        (h, body.clone())
                    }
                };
                let _ = stream.write_all(headers.as_bytes());
                let _ = stream.write_all(&payload);
                let _ = stream.flush();
            }
        });
        format!("http://{addr}/file")
    }

    fn parse_range_start(request: &str) -> Option<u64> {
        for line in request.lines() {
            let lower = line.to_ascii_lowercase();
            if let Some(rest) = lower.strip_prefix("range:") {
                let after = rest.trim().strip_prefix("bytes=")?;
                let (s, _) = after.split_once('-')?;
                return s.trim().parse::<u64>().ok();
            }
        }
        None
    }

    fn spawn_404_server() -> String {
        use std::io::{Read as _, Write as _};
        use std::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            for stream in listener.incoming().take(1) {
                let Ok(mut stream) = stream else { continue };
                let mut buf = [0_u8; 4096];
                let _ = stream.read(&mut buf);
                let _ = stream.write_all(
                    b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                );
                let _ = stream.flush();
            }
        });
        format!("http://{addr}/missing")
    }

    #[test]
    fn install_from_url_writes_and_verifies() {
        let tmp = TempDir::new().unwrap();
        let url = spawn_static_server(b"hello".to_vec());
        let installer = ModelInstaller {
            dest_dir: tmp.path(),
            dest_filename: "remote.bin",
            expected_sha256: Some(HELLO_SHA),
        };
        let report = installer.install_from_url(&url).unwrap();
        assert_eq!(report.bytes, 5);
        assert!(report.verified);
        assert_eq!(std::fs::read(&report.dest).unwrap(), b"hello");
    }

    #[test]
    fn install_from_url_errors_on_404() {
        let tmp = TempDir::new().unwrap();
        let url = spawn_404_server();
        let installer = ModelInstaller {
            dest_dir: tmp.path(),
            dest_filename: "x.bin",
            expected_sha256: None,
        };
        let err = installer.install_from_url(&url).unwrap_err();
        assert_eq!(err.code(), &E1001_INSTALLED_SCHEMA_UNREADABLE);
    }

    #[test]
    fn install_from_url_errors_on_unreachable() {
        let tmp = TempDir::new().unwrap();
        let installer = ModelInstaller {
            dest_dir: tmp.path(),
            dest_filename: "x.bin",
            expected_sha256: None,
        };
        // Unbound port on the loopback interface; connect fails immediately.
        let err = installer
            .install_from_url("http://127.0.0.1:1/never")
            .unwrap_err();
        assert_eq!(err.code(), &E1001_INSTALLED_SCHEMA_UNREADABLE);
    }

    #[test]
    fn install_from_url_errors_on_sha_mismatch() {
        let tmp = TempDir::new().unwrap();
        let url = spawn_static_server(b"hello".to_vec());
        let installer = ModelInstaller {
            dest_dir: tmp.path(),
            dest_filename: "remote.bin",
            expected_sha256: Some("deadbeef"),
        };
        let err = installer.install_from_url(&url).unwrap_err();
        assert!(format!("{err}").contains("mismatch"));
    }

    #[test]
    fn install_from_url_resumes_via_range_206() {
        let tmp = TempDir::new().unwrap();
        let installer = ModelInstaller {
            dest_dir: tmp.path(),
            dest_filename: "remote.bin",
            expected_sha256: Some(HELLO_SHA),
        };
        // Pre-seed the partial with the first two bytes of "hello".
        let partial = partial_path(&tmp.path().join("remote.bin"));
        std::fs::write(&partial, b"he").unwrap();

        let url = spawn_range_server(b"hello".to_vec(), RangeMode::Honor);
        let report = installer.install_from_url(&url).unwrap();
        assert_eq!(report.bytes, 5);
        assert_eq!(report.sha256, HELLO_SHA);
        assert!(report.verified);
        assert_eq!(std::fs::read(&report.dest).unwrap(), b"hello");
        assert!(!partial.exists());
    }

    #[test]
    fn install_from_url_falls_back_to_200_when_server_ignores_range() {
        let tmp = TempDir::new().unwrap();
        let installer = ModelInstaller {
            dest_dir: tmp.path(),
            dest_filename: "remote.bin",
            expected_sha256: Some(HELLO_SHA),
        };
        // Stale partial that the server won't honor — must be discarded.
        let partial = partial_path(&tmp.path().join("remote.bin"));
        std::fs::write(&partial, b"XX").unwrap();

        let url = spawn_range_server(b"hello".to_vec(), RangeMode::Ignore);
        let report = installer.install_from_url(&url).unwrap();
        assert_eq!(report.bytes, 5);
        assert_eq!(report.sha256, HELLO_SHA);
        assert_eq!(std::fs::read(&report.dest).unwrap(), b"hello");
    }

    #[test]
    fn install_from_url_resumes_across_chunk_boundary() {
        // Body 11 bytes, partial 3 bytes, remaining 8 bytes.
        let body = b"hello world".to_vec();
        let expected = sha256_hex(&body);

        let tmp = TempDir::new().unwrap();
        let installer = ModelInstaller {
            dest_dir: tmp.path(),
            dest_filename: "remote.bin",
            expected_sha256: Some(&expected),
        };
        let partial = partial_path(&tmp.path().join("remote.bin"));
        std::fs::write(&partial, &body[..3]).unwrap();

        let url = spawn_range_server(body.clone(), RangeMode::Honor);
        let report = installer.install_from_url(&url).unwrap();
        assert_eq!(report.bytes, body.len() as u64);
        assert_eq!(report.sha256, expected);
        assert!(report.verified);
        assert_eq!(std::fs::read(&report.dest).unwrap(), body);
    }

    #[test]
    fn install_from_url_resume_no_expected_sha_marks_unverified() {
        let tmp = TempDir::new().unwrap();
        let installer = ModelInstaller {
            dest_dir: tmp.path(),
            dest_filename: "remote.bin",
            expected_sha256: None,
        };
        let partial = partial_path(&tmp.path().join("remote.bin"));
        std::fs::write(&partial, b"he").unwrap();

        let url = spawn_range_server(b"hello".to_vec(), RangeMode::Honor);
        let report = installer.install_from_url(&url).unwrap();
        assert!(!report.verified);
        assert_eq!(report.bytes, 5);
        assert_eq!(report.sha256, HELLO_SHA);
    }

    /// Server that emits a 206 response with a caller-supplied
    /// `Content-Range` value, used to drive the parser's reject branches.
    fn spawn_custom_content_range_server(body: Vec<u8>, content_range: String) -> String {
        use std::io::{Read as _, Write as _};
        use std::net::TcpListener;
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            for stream in listener.incoming().take(1) {
                let Ok(mut stream) = stream else { continue };
                let mut buf = [0_u8; 4096];
                let _ = stream.read(&mut buf);
                let headers = if content_range.is_empty() {
                    format!(
                        "HTTP/1.1 206 Partial Content\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        body.len()
                    )
                } else {
                    format!(
                        "HTTP/1.1 206 Partial Content\r\nContent-Length: {}\r\nContent-Range: {content_range}\r\nConnection: close\r\n\r\n",
                        body.len()
                    )
                };
                let _ = stream.write_all(headers.as_bytes());
                let _ = stream.write_all(&body);
                let _ = stream.flush();
            }
        });
        format!("http://{addr}/file")
    }

    #[test]
    fn install_from_url_rejects_206_without_content_range() {
        // Pre-seed a partial, server returns 206 without Content-Range — must
        // fall back to fresh-install path (discard partial and use full body).
        let tmp = TempDir::new().unwrap();
        let partial = partial_path(&tmp.path().join("remote.bin"));
        std::fs::write(&partial, b"XX").unwrap();
        let url = spawn_custom_content_range_server(b"hello".to_vec(), String::new());
        let installer = ModelInstaller {
            dest_dir: tmp.path(),
            dest_filename: "remote.bin",
            expected_sha256: Some(HELLO_SHA),
        };
        let report = installer.install_from_url(&url).unwrap();
        assert_eq!(report.sha256, HELLO_SHA);
        assert_eq!(std::fs::read(&report.dest).unwrap(), b"hello");
    }

    #[test]
    fn install_from_url_rejects_206_with_mismatched_offset() {
        let tmp = TempDir::new().unwrap();
        let partial = partial_path(&tmp.path().join("remote.bin"));
        std::fs::write(&partial, b"XX").unwrap();
        // Server says it's starting at 0, but the client asked for 2.
        let url = spawn_custom_content_range_server(b"hello".to_vec(), "bytes 0-4/5".to_string());
        let installer = ModelInstaller {
            dest_dir: tmp.path(),
            dest_filename: "remote.bin",
            expected_sha256: Some(HELLO_SHA),
        };
        let report = installer.install_from_url(&url).unwrap();
        assert_eq!(report.sha256, HELLO_SHA);
    }

    #[test]
    fn install_from_url_rejects_malformed_content_range() {
        let tmp = TempDir::new().unwrap();
        let partial = partial_path(&tmp.path().join("remote.bin"));
        std::fs::write(&partial, b"XX").unwrap();
        // No dash → split_once('-') returns None → predicate rejects.
        let url = spawn_custom_content_range_server(b"hello".to_vec(), "garbage".to_string());
        let installer = ModelInstaller {
            dest_dir: tmp.path(),
            dest_filename: "remote.bin",
            expected_sha256: Some(HELLO_SHA),
        };
        let report = installer.install_from_url(&url).unwrap();
        assert_eq!(report.sha256, HELLO_SHA);
    }

    #[test]
    fn install_from_url_rejects_content_range_without_bytes_unit() {
        let tmp = TempDir::new().unwrap();
        let partial = partial_path(&tmp.path().join("remote.bin"));
        std::fs::write(&partial, b"XX").unwrap();
        // Wrong unit → `strip_prefix("bytes")` returns None → predicate rejects.
        let url = spawn_custom_content_range_server(b"hello".to_vec(), "items 2-4/5".to_string());
        let installer = ModelInstaller {
            dest_dir: tmp.path(),
            dest_filename: "remote.bin",
            expected_sha256: Some(HELLO_SHA),
        };
        let report = installer.install_from_url(&url).unwrap();
        assert_eq!(report.sha256, HELLO_SHA);
    }

    #[test]
    fn install_from_url_errors_when_dest_dir_unwritable() {
        let tmp = TempDir::new().unwrap();
        let blocker = tmp.path().join("blocker");
        std::fs::write(&blocker, b"x").unwrap();
        let installer = ModelInstaller {
            dest_dir: &blocker.join("nested"),
            dest_filename: "x.bin",
            expected_sha256: None,
        };
        let err = installer
            .install_from_url("http://127.0.0.1:1/never")
            .unwrap_err();
        assert_eq!(err.code(), &E1001_INSTALLED_SCHEMA_UNREADABLE);
    }

    #[test]
    fn install_from_url_resume_mismatch_clears_partial() {
        let tmp = TempDir::new().unwrap();
        let installer = ModelInstaller {
            dest_dir: tmp.path(),
            dest_filename: "remote.bin",
            expected_sha256: Some("deadbeef"),
        };
        let partial = partial_path(&tmp.path().join("remote.bin"));
        std::fs::write(&partial, b"he").unwrap();

        let url = spawn_range_server(b"hello".to_vec(), RangeMode::Honor);
        let err = installer.install_from_url(&url).unwrap_err();
        assert_eq!(err.code(), &E1001_INSTALLED_SCHEMA_UNREADABLE);
        assert!(format!("{err}").contains("mismatch"));
        assert!(!partial.exists());
    }

    #[test]
    fn content_range_starts_at_parses_well_formed_header() {
        // Construct via a faux loopback that always returns the header we want.
        // Easier: round-trip through `parse_range_start` for symmetry, plus
        // direct exercise of the predicate via a tiny live request.
        let body = b"hello world".to_vec();
        let url = spawn_range_server(body, RangeMode::Honor);
        let resp = ureq::get(&url)
            .set("Range", "bytes=3-")
            .call()
            .expect("range server replies");
        assert_eq!(resp.status(), 206);
        assert!(content_range_starts_at(&resp, 3));
        assert!(!content_range_starts_at(&resp, 0));
    }

    #[test]
    fn content_range_starts_at_rejects_missing_header() {
        let url = spawn_range_server(b"hello".to_vec(), RangeMode::Ignore);
        let resp = ureq::get(&url).call().expect("plain 200");
        assert!(!content_range_starts_at(&resp, 0));
    }

    // ===== Retry-with-clock tests =====
    //
    // These exercise `install_from_url_with_retry_and_clock` against a
    // `ManualClock`, so backoff sleeps are captured without blocking and the
    // sequence of HTTP responses is fully scripted via a multi-connection
    // test server.

    use crate::retry::{ManualClock as RetryManualClock, RetryError};

    /// One scripted response for the n-th inbound connection.
    enum Step {
        /// Reply with the given raw header bytes + payload, then close.
        Reply { headers: String, payload: Vec<u8> },
        /// Honor a `Range:` request — reply 206 with the suffix of `body`
        /// starting at the requested offset, then close.
        RangeFrom(Vec<u8>),
        /// Reply with the first `prefix_bytes` of `body`, then drop the
        /// connection (no Content-Length set, mid-stream close).
        DropAfter { body: Vec<u8>, prefix_bytes: usize },
        /// Close the connection immediately, before sending any bytes.
        CloseImmediately,
    }

    fn spawn_scripted_server(steps: Vec<Step>) -> String {
        use std::io::{Read as _, Write as _};
        use std::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let conn_count = steps.len();
        std::thread::spawn(move || {
            let mut steps_iter = steps.into_iter();
            for stream in listener.incoming().take(conn_count) {
                let Ok(mut stream) = stream else { continue };
                let Some(step) = steps_iter.next() else { break };
                let mut buf = [0_u8; 4096];
                let n = stream.read(&mut buf).unwrap_or(0);
                let request = std::str::from_utf8(&buf[..n]).unwrap_or("");
                match step {
                    Step::Reply { headers, payload } => {
                        let _ = stream.write_all(headers.as_bytes());
                        let _ = stream.write_all(&payload);
                        let _ = stream.flush();
                    }
                    Step::RangeFrom(body) => {
                        let total = body.len();
                        let start = parse_range_start(request).unwrap_or(0);
                        let start_usize = usize::try_from(start).unwrap_or(0);
                        if start_usize >= total {
                            let h = "HTTP/1.1 416 Range Not Satisfiable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
                            let _ = stream.write_all(h.as_bytes());
                        } else if start_usize == 0 {
                            let h = format!(
                                "HTTP/1.1 200 OK\r\nContent-Length: {total}\r\nConnection: close\r\n\r\n"
                            );
                            let _ = stream.write_all(h.as_bytes());
                            let _ = stream.write_all(&body);
                        } else {
                            let slice = &body[start_usize..];
                            let end = total - 1;
                            let h = format!(
                                "HTTP/1.1 206 Partial Content\r\nContent-Length: {}\r\nContent-Range: bytes {start}-{end}/{total}\r\nConnection: close\r\n\r\n",
                                slice.len()
                            );
                            let _ = stream.write_all(h.as_bytes());
                            let _ = stream.write_all(slice);
                        }
                        let _ = stream.flush();
                    }
                    Step::DropAfter { body, prefix_bytes } => {
                        // Advertise the full length, then close after only
                        // `prefix_bytes` bytes — ureq's reader surfaces the
                        // short-read as a transport error which the retry
                        // classifier sees as `transport: ...`.
                        let total = body.len();
                        let h = format!(
                            "HTTP/1.1 200 OK\r\nContent-Length: {total}\r\nConnection: close\r\n\r\n"
                        );
                        let _ = stream.write_all(h.as_bytes());
                        let _ = stream.write_all(&body[..prefix_bytes]);
                        let _ = stream.flush();
                        // Drop the stream while remaining bytes are still pending.
                        drop(stream);
                    }
                    Step::CloseImmediately => {
                        // Drop without sending anything — the client sees a
                        // transport error (connection reset) before any
                        // HTTP framing arrives.
                        drop(stream);
                    }
                }
            }
        });
        format!("http://{addr}/file")
    }

    fn fast_test_policy(max_attempts: u32) -> RetryPolicy {
        RetryPolicy {
            max_attempts,
            initial_delay: Duration::from_millis(250),
            max_delay: Duration::from_secs(5),
            backoff_multiplier: 2.0,
            jitter: Jitter::None,
        }
    }

    #[test]
    fn retry_happy_path_single_attempt_no_sleeps() {
        let tmp = TempDir::new().unwrap();
        let url = spawn_scripted_server(vec![Step::RangeFrom(b"hello".to_vec())]);
        let installer = ModelInstaller {
            dest_dir: tmp.path(),
            dest_filename: "remote.bin",
            expected_sha256: Some(HELLO_SHA),
        };
        let clock = RetryManualClock::new();
        let report = installer
            .install_from_url_with_retry_and_clock(&url, &fast_test_policy(4), &clock)
            .unwrap();
        assert_eq!(report.bytes, 5);
        assert!(report.verified);
        assert!(clock.sleeps().is_empty(), "no retry → no sleeps");
    }

    #[test]
    fn retry_transient_502_then_200_records_one_sleep() {
        let tmp = TempDir::new().unwrap();
        let body = b"hello".to_vec();
        let h502 = "HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
            .to_string();
        let url = spawn_scripted_server(vec![
            Step::Reply {
                headers: h502,
                payload: Vec::new(),
            },
            Step::RangeFrom(body),
        ]);
        let installer = ModelInstaller {
            dest_dir: tmp.path(),
            dest_filename: "remote.bin",
            expected_sha256: Some(HELLO_SHA),
        };
        let clock = RetryManualClock::new();
        let report = installer
            .install_from_url_with_retry_and_clock(&url, &fast_test_policy(4), &clock)
            .unwrap();
        assert_eq!(report.bytes, 5);
        assert_eq!(report.sha256, HELLO_SHA);
        assert_eq!(
            clock.sleeps().len(),
            1,
            "exactly one inter-attempt sleep on the second attempt"
        );
    }

    #[test]
    fn retry_always_502_exhausts_with_three_sleeps() {
        let tmp = TempDir::new().unwrap();
        let h502 = "HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
            .to_string();
        let url = spawn_scripted_server(vec![
            Step::Reply {
                headers: h502.clone(),
                payload: Vec::new(),
            },
            Step::Reply {
                headers: h502.clone(),
                payload: Vec::new(),
            },
            Step::Reply {
                headers: h502.clone(),
                payload: Vec::new(),
            },
            Step::Reply {
                headers: h502,
                payload: Vec::new(),
            },
        ]);
        let installer = ModelInstaller {
            dest_dir: tmp.path(),
            dest_filename: "remote.bin",
            expected_sha256: Some(HELLO_SHA),
        };
        let clock = RetryManualClock::new();
        let err = installer
            .install_from_url_with_retry_and_clock(&url, &fast_test_policy(4), &clock)
            .unwrap_err();
        assert_eq!(err.code(), &E1001_INSTALLED_SCHEMA_UNREADABLE);
        // 4 attempts → 3 inter-attempt sleeps.
        assert_eq!(clock.sleeps().len(), 3);
    }

    #[test]
    fn retry_404_short_circuits_to_fatal() {
        let tmp = TempDir::new().unwrap();
        let h404 =
            "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_string();
        let url = spawn_scripted_server(vec![Step::Reply {
            headers: h404,
            payload: Vec::new(),
        }]);
        let installer = ModelInstaller {
            dest_dir: tmp.path(),
            dest_filename: "remote.bin",
            expected_sha256: None,
        };
        let clock = RetryManualClock::new();
        let err = installer
            .install_from_url_with_retry_and_clock(&url, &fast_test_policy(4), &clock)
            .unwrap_err();
        assert_eq!(err.code(), &E1001_INSTALLED_SCHEMA_UNREADABLE);
        assert!(
            clock.sleeps().is_empty(),
            "fatal short-circuits before any sleep"
        );
    }

    #[test]
    fn retry_range_resume_after_mid_stream_drop() {
        // 1 KiB deterministic body. First attempt: server sends the first
        // 512 bytes then closes mid-stream. Client retries with
        // `Range: bytes=512-` and receives the remaining 512 bytes.
        let body: Vec<u8> = (0..1024_u16)
            .map(|i| u8::try_from(i % 256).unwrap_or(0))
            .collect();
        let expected = sha256_hex(&body);

        let tmp = TempDir::new().unwrap();
        let url = spawn_scripted_server(vec![
            Step::DropAfter {
                body: body.clone(),
                prefix_bytes: 512,
            },
            Step::RangeFrom(body.clone()),
        ]);
        let installer = ModelInstaller {
            dest_dir: tmp.path(),
            dest_filename: "remote.bin",
            expected_sha256: Some(&expected),
        };
        let clock = RetryManualClock::new();
        let report = installer
            .install_from_url_with_retry_and_clock(&url, &fast_test_policy(4), &clock)
            .unwrap();
        assert_eq!(report.bytes, body.len() as u64);
        assert_eq!(report.sha256, expected);
        assert!(report.verified);
        assert_eq!(std::fs::read(&report.dest).unwrap(), body);
        assert_eq!(clock.sleeps().len(), 1, "one retry after the drop");
    }

    #[test]
    fn retry_transport_close_then_success() {
        // First connection: server closes without sending any HTTP framing.
        // ureq surfaces this as `Transport(_)` which we classify as retryable.
        // Second connection: clean 200.
        let tmp = TempDir::new().unwrap();
        let url = spawn_scripted_server(vec![
            Step::CloseImmediately,
            Step::RangeFrom(b"hello".to_vec()),
        ]);
        let installer = ModelInstaller {
            dest_dir: tmp.path(),
            dest_filename: "remote.bin",
            expected_sha256: Some(HELLO_SHA),
        };
        let clock = RetryManualClock::new();
        let report = installer
            .install_from_url_with_retry_and_clock(&url, &fast_test_policy(4), &clock)
            .unwrap();
        assert_eq!(report.bytes, 5);
        assert_eq!(clock.sleeps().len(), 1);
    }

    #[test]
    fn install_transient_error_display_covers_both_variants() {
        let retryable: InstallTransientError =
            InstallTransientError::Retryable(RetryError::Exhausted {
                attempts: 4,
                last_error: "status 502".to_string(),
            });
        let nonretryable = InstallTransientError::NonRetryable("sha256 mismatch".to_string());
        let rmsg = retryable.to_string();
        let nmsg = nonretryable.to_string();
        assert!(rmsg.contains("502"));
        assert!(rmsg.contains("transient"));
        assert!(nmsg.contains("mismatch"));
        assert!(nmsg.contains("terminal"));
        // Trigger the Debug impl too so the derive is covered.
        let _ = format!("{retryable:?}");
        let _ = format!("{nonretryable:?}");
    }

    #[test]
    fn install_transient_error_implements_error_trait() {
        fn assert_error<E: std::error::Error>() {}
        assert_error::<InstallTransientError>();
    }

    #[test]
    fn default_install_retry_policy_matches_spec() {
        let p = default_install_retry_policy();
        assert_eq!(p.max_attempts, 4);
        assert_eq!(p.initial_delay, Duration::from_millis(250));
        assert_eq!(p.max_delay, Duration::from_secs(5));
        assert!((p.backoff_multiplier - 2.0).abs() < f64::EPSILON);
        assert_eq!(p.jitter, Jitter::Full);
    }

    #[test]
    fn classify_http_reason_categorises_known_prefixes() {
        assert_eq!(classify_http_reason("status 502"), RetryDecision::Retry);
        assert_eq!(classify_http_reason("status 503"), RetryDecision::Retry);
        assert_eq!(classify_http_reason("status 404"), RetryDecision::Fatal);
        assert_eq!(classify_http_reason("status 403"), RetryDecision::Fatal);
        assert_eq!(
            classify_http_reason("transport: timed out"),
            RetryDecision::Retry
        );
    }

    #[test]
    fn install_from_reader_overwrites_stale_dest() {
        let tmp = TempDir::new().unwrap();
        std::fs::create_dir_all(tmp.path()).unwrap();
        let dest = tmp.path().join("dup.bin");
        std::fs::write(&dest, b"stale").unwrap();
        let installer = ModelInstaller {
            dest_dir: tmp.path(),
            dest_filename: "dup.bin",
            expected_sha256: None,
        };
        let _ = installer
            .install_from_reader(Cursor::new(b"fresh".to_vec()))
            .unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), b"fresh");
    }
}
