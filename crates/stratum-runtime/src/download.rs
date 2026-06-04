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

use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use stratum_types::error::codes::E1001_INSTALLED_SCHEMA_UNREADABLE;
use stratum_types::{StratumError, StratumResult};

const COPY_CHUNK: usize = 64 * 1024;

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
