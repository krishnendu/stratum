//! Slug → local GGUF path resolver.
//!
//! `ModelResolver` is a thin, pure-runtime helper that composes
//! [`crate::model_catalog::ModelCatalog`] and a [`BlobFetcher`] into a
//! single "give me the file for this slug" entrypoint.
//!
//! The on-disk layout is `<state_root>/models/<sha256>.gguf` — content-
//! addressed by the catalog's expected SHA-256 so two entries sharing the
//! same artifact byte-for-byte naturally share a single cached file.
//!
//! # Composition (no surgery on existing types)
//!
//! `ModelResolver` does not depend on the concrete
//! [`crate::download::ModelInstaller`] type. Production callers wrap their
//! installer in a [`BlobFetcher`] implementation, and tests can stage files
//! on disk via a no-op fetcher. The actual HTTPS retry path lives inside
//! whatever fetcher the caller supplies — typically one that delegates to
//! [`crate::download::ModelInstaller::install_from_url_with_retry`].
//!
//! # Tradeoff
//!
//! `resolve` performs real HTTPS in production via the supplied fetcher.
//! Tests pre-stage the file at the expected target path (or use a
//! [`StageFetcher`] shim that emulates a successful install) so no network
//! traffic is generated.

use std::fmt;
use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::model_catalog::{ModelCatalog, ModelEntry, ModelSlug};
use crate::retry::RetryPolicy;

const HASH_CHUNK: usize = 64 * 1024;

/// Pluggable downloader contract — accepts a URL + expected SHA-256 +
/// expected byte count + destination path, and is responsible for writing
/// the verified bytes to that path atomically.
///
/// The default Stratum implementation delegates to
/// [`crate::download::ModelInstaller`]; tests use [`StageFetcher`] which
/// just copies a caller-staged byte slice into place.
pub trait BlobFetcher {
    /// Fetch `url`, verify it matches `expected_sha256` + `expected_bytes`,
    /// and write the verified bytes to `dest`.
    ///
    /// On success the file at `dest` must exist and be content-equal to
    /// what the catalog declared.
    ///
    /// # Errors
    /// Returns a short, classifier-friendly reason string the resolver
    /// surfaces as [`ResolveModelError::InstallFailed`].
    fn fetch(
        &self,
        url: &str,
        expected_sha256: &str,
        expected_bytes: u64,
        dest: &Path,
        policy: &RetryPolicy,
    ) -> Result<(), String>;
}

/// Test-only fetcher that copies caller-staged bytes into the destination
/// path. Mirrors the post-install state of a real fetcher without touching
/// the network or the retry stack.
///
/// Hidden behind `cfg(test)` so it never ships in release binaries.
#[cfg(test)]
pub(crate) struct StageFetcher {
    pub(crate) bytes: Vec<u8>,
}

#[cfg(test)]
impl BlobFetcher for StageFetcher {
    fn fetch(
        &self,
        _url: &str,
        _expected_sha256: &str,
        _expected_bytes: u64,
        dest: &Path,
        _policy: &RetryPolicy,
    ) -> Result<(), String> {
        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent).map_err(|e| format!("mkdir: {e}"))?;
        }
        fs::write(dest, &self.bytes).map_err(|e| format!("write: {e}"))
    }
}

/// Test-only fetcher that always returns a fixed error reason.
#[cfg(test)]
pub(crate) struct FailingFetcher;

#[cfg(test)]
impl BlobFetcher for FailingFetcher {
    fn fetch(
        &self,
        _url: &str,
        _expected_sha256: &str,
        _expected_bytes: u64,
        _dest: &Path,
        _policy: &RetryPolicy,
    ) -> Result<(), String> {
        Err("forced failure".to_owned())
    }
}

/// Slug → local file path resolver, backed by a curated [`ModelCatalog`]
/// and a content-addressed cache rooted at `<state_root>/models/`.
#[derive(Debug, Clone)]
pub struct ModelResolver {
    catalog: ModelCatalog,
    state_root: PathBuf,
    retry_policy: RetryPolicy,
}

impl ModelResolver {
    /// Build a resolver from a catalog + the path that contains the
    /// `models/` cache directory.
    #[must_use]
    pub const fn new(catalog: ModelCatalog, state_root: PathBuf) -> Self {
        Self {
            catalog,
            state_root,
            retry_policy: crate::download::default_install_retry_policy(),
        }
    }

    /// Override the retry policy used by `resolve`'s underlying fetch.
    #[must_use]
    pub const fn with_retry_policy(mut self, policy: RetryPolicy) -> Self {
        self.retry_policy = policy;
        self
    }

    /// Borrow the catalog the resolver was built from.
    #[must_use]
    pub const fn catalog(&self) -> &ModelCatalog {
        &self.catalog
    }

    /// Borrow the state root.
    #[must_use]
    pub fn state_root(&self) -> &Path {
        &self.state_root
    }

    /// Compute the content-addressed target path for an entry — same layout
    /// the cache check and the install path agree on.
    fn target_path(&self, entry: &ModelEntry) -> PathBuf {
        self.state_root
            .join("models")
            .join(format!("{}.gguf", entry.artifact.sha256))
    }

    fn lookup<'a>(&'a self, slug: &ModelSlug) -> Result<&'a ModelEntry, ResolveModelError> {
        self.catalog
            .get(slug)
            .ok_or_else(|| ResolveModelError::UnknownSlug(slug.as_str().to_owned()))
    }

    /// Resolve `slug` to a local file path, fetching via `fetcher` only
    /// when the cache is empty or the cached file fails verification.
    ///
    /// # Errors
    /// - [`ResolveModelError::UnknownSlug`] when the catalog has no such entry.
    /// - [`ResolveModelError::InstallFailed`] when the fetcher returns Err.
    /// - [`ResolveModelError::HashMismatch`] when the post-fetch file still
    ///   fails the catalog's SHA-256 expectation.
    /// - [`ResolveModelError::Io`] for any I/O failure while inspecting the
    ///   cache.
    pub fn resolve<F: BlobFetcher + ?Sized>(
        &self,
        slug: &ModelSlug,
        fetcher: &F,
    ) -> Result<PathBuf, ResolveModelError> {
        let entry = self.lookup(slug)?;
        let target = self.target_path(entry);

        if file_matches(&target, &entry.artifact.sha256, entry.artifact.bytes)? {
            return Ok(target);
        }

        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent).map_err(ResolveModelError::Io)?;
        }

        fetcher
            .fetch(
                &entry.artifact.url,
                &entry.artifact.sha256,
                entry.artifact.bytes,
                &target,
                &self.retry_policy,
            )
            .map_err(ResolveModelError::InstallFailed)?;

        let (actual_sha, observed_bytes) = hash_file(&target)?;
        if actual_sha != entry.artifact.sha256.to_ascii_lowercase()
            || observed_bytes != entry.artifact.bytes
        {
            return Err(ResolveModelError::HashMismatch {
                expected: entry.artifact.sha256.clone(),
                actual: actual_sha,
                bytes_observed: observed_bytes,
            });
        }
        Ok(target)
    }

    /// Quick predicate: is this slug's artifact already on disk + verified?
    ///
    /// # Errors
    /// - [`ResolveModelError::UnknownSlug`] when the catalog has no such entry.
    /// - [`ResolveModelError::Io`] for any I/O failure while hashing the cache file.
    pub fn is_cached(&self, slug: &ModelSlug) -> Result<bool, ResolveModelError> {
        let entry = self.lookup(slug)?;
        let target = self.target_path(entry);
        file_matches(&target, &entry.artifact.sha256, entry.artifact.bytes)
    }

    /// Remove the cached file for `slug` if present.
    ///
    /// Returns `Ok(true)` when a file was removed, `Ok(false)` when nothing
    /// was cached.
    ///
    /// # Errors
    /// - [`ResolveModelError::UnknownSlug`] when the catalog has no such entry.
    /// - [`ResolveModelError::Io`] for any I/O failure while removing the file.
    pub fn evict(&self, slug: &ModelSlug) -> Result<bool, ResolveModelError> {
        let entry = self.lookup(slug)?;
        let target = self.target_path(entry);
        match fs::remove_file(&target) {
            Ok(()) => Ok(true),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(false),
            Err(e) => Err(ResolveModelError::Io(e)),
        }
    }

    /// All catalog slugs whose artifact is currently cached + verified.
    ///
    /// The returned vec preserves catalog (slug) order — `ModelCatalog`
    /// iterates a `BTreeMap`, so the output is already sorted.
    ///
    /// # Errors
    /// - [`ResolveModelError::Io`] for any I/O failure while hashing a cache file.
    pub fn list_cached(&self) -> Result<Vec<ModelSlug>, ResolveModelError> {
        let mut out = Vec::new();
        for entry in self.catalog.entries.values() {
            let target = self.target_path(entry);
            if file_matches(&target, &entry.artifact.sha256, entry.artifact.bytes)? {
                out.push(entry.slug.clone());
            }
        }
        Ok(out)
    }
}

/// SHA-256 the file at `path` (streaming) and return `(hex_digest, byte_count)`.
///
/// Caller is responsible for short-circuiting on `NotFound`.
fn hash_file(path: &Path) -> Result<(String, u64), ResolveModelError> {
    let mut file = fs::File::open(path).map_err(ResolveModelError::Io)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0_u8; HASH_CHUNK];
    let mut total = 0_u64;
    loop {
        let n = file.read(&mut buf).map_err(ResolveModelError::Io)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        total = total.saturating_add(n as u64);
    }
    let digest = hasher.finalize();
    let mut hex = String::with_capacity(digest.len() * 2);
    for b in digest {
        use std::fmt::Write as _;
        let _ = write!(hex, "{b:02x}");
    }
    Ok((hex, total))
}

/// Does `path` exist AND match both the expected SHA-256 and byte count?
///
/// Missing file → `Ok(false)`. Hash mismatch / size mismatch → `Ok(false)`.
/// Real I/O failure → `Err(Io)`.
fn file_matches(
    path: &Path,
    expected_sha256: &str,
    expected_bytes: u64,
) -> Result<bool, ResolveModelError> {
    match fs::metadata(path) {
        Ok(meta) if meta.is_file() => {
            if meta.len() != expected_bytes {
                return Ok(false);
            }
            let (actual, observed) = hash_file(path)?;
            Ok(observed == expected_bytes && actual == expected_sha256.to_ascii_lowercase())
        }
        Ok(_) => Ok(false),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(e) => Err(ResolveModelError::Io(e)),
    }
}

/// Error surface for resolver operations.
#[derive(Debug)]
pub enum ResolveModelError {
    /// Catalog did not contain an entry for the requested slug.
    UnknownSlug(String),
    /// The fetcher returned an error (network failure, retry exhausted,
    /// non-2xx, fs failure, etc.). Inner string is the fetcher's classifier-
    /// friendly reason.
    InstallFailed(String),
    /// Local filesystem I/O failed while inspecting or hashing the cache.
    Io(io::Error),
    /// Post-fetch verification disagreed with the catalog's expectations.
    HashMismatch {
        /// Catalog-declared SHA-256 (lowercase hex).
        expected: String,
        /// Digest we measured off disk.
        actual: String,
        /// Byte count we read off disk.
        bytes_observed: u64,
    },
}

impl fmt::Display for ResolveModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownSlug(slug) => write!(f, "unknown model slug: {slug}"),
            Self::InstallFailed(reason) => write!(f, "model install failed: {reason}"),
            Self::Io(e) => write!(f, "model resolver io error: {e}"),
            Self::HashMismatch {
                expected,
                actual,
                bytes_observed,
            } => write!(
                f,
                "model resolver hash mismatch: expected {expected}, got {actual} ({bytes_observed} bytes)"
            ),
        }
    }
}

impl std::error::Error for ResolveModelError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::UnknownSlug(_) | Self::InstallFailed(_) | Self::HashMismatch { .. } => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use std::error::Error as _;

    use tempfile::TempDir;

    use super::*;
    use crate::download::sha256_hex;
    use crate::model_catalog::{ArtifactRef, ModelEntry, ModelTask, ModelTier};

    fn entry_with(slug: &str, bytes: &[u8]) -> ModelEntry {
        let sha = sha256_hex(bytes);
        let artifact = ArtifactRef::new(
            "https://example.test/blob".to_owned(),
            sha,
            u64::try_from(bytes.len()).unwrap_or(0),
        )
        .expect("artifact valid");
        let mut tasks = BTreeSet::new();
        tasks.insert(ModelTask::Chat);
        ModelEntry {
            slug: slug.parse().expect("slug"),
            family: "llama".to_owned(),
            display_name: format!("Display {slug}"),
            tier: ModelTier::Low,
            task: tasks,
            size_mib: 1,
            quantization: "Q4_K_M".to_owned(),
            artifact,
            license: "Apache-2.0".to_owned(),
            homepage: None,
        }
    }

    fn one_entry_catalog(slug: &str, bytes: &[u8]) -> ModelCatalog {
        let mut cat = ModelCatalog::new();
        cat.insert(entry_with(slug, bytes));
        cat
    }

    fn stage_file(state_root: &Path, sha: &str, body: &[u8]) -> PathBuf {
        let dir = state_root.join("models");
        fs::create_dir_all(&dir).expect("mkdir models");
        let path = dir.join(format!("{sha}.gguf"));
        fs::write(&path, body).expect("stage");
        path
    }

    // ---- new() smoke ----

    #[test]
    fn new_smoke_holds_catalog_and_root() {
        let tmp = TempDir::new().expect("tempdir");
        let cat = ModelCatalog::new();
        let r = ModelResolver::new(cat.clone(), tmp.path().to_path_buf());
        assert_eq!(r.catalog(), &cat);
        assert_eq!(r.state_root(), tmp.path());
    }

    // ---- resolve() ----

    #[test]
    fn resolve_unknown_slug_errors() {
        let tmp = TempDir::new().expect("tempdir");
        let r = ModelResolver::new(ModelCatalog::new(), tmp.path().to_path_buf());
        let slug: ModelSlug = "missing".parse().expect("slug");
        let err = r
            .resolve(&slug, &FailingFetcher)
            .expect_err("unknown slug must err");
        match err {
            ResolveModelError::UnknownSlug(s) => assert_eq!(s, "missing"),
            other => panic!("wrong variant: {other:?}"),
        }
    }

    #[test]
    fn resolve_cached_returns_path_without_fetching() {
        let tmp = TempDir::new().expect("tempdir");
        let body = b"abcdef".to_vec();
        let cat = one_entry_catalog("alpha", &body);
        let sha = sha256_hex(&body);
        let _staged = stage_file(tmp.path(), &sha, &body);

        let r = ModelResolver::new(cat, tmp.path().to_path_buf());
        let slug: ModelSlug = "alpha".parse().expect("slug");
        // FailingFetcher would error if we hit fetch — proves cache short-circuit.
        let got = r.resolve(&slug, &FailingFetcher).expect("cached resolves");
        assert_eq!(got, tmp.path().join("models").join(format!("{sha}.gguf")));
    }

    #[test]
    fn resolve_honors_path_layout() {
        let tmp = TempDir::new().expect("tempdir");
        let body = b"layout-bytes".to_vec();
        let cat = one_entry_catalog("layout", &body);
        let sha = sha256_hex(&body);
        let r = ModelResolver::new(cat, tmp.path().to_path_buf());
        let slug: ModelSlug = "layout".parse().expect("slug");
        let fetcher = StageFetcher { bytes: body };
        let got = r.resolve(&slug, &fetcher).expect("resolves");
        let expected = tmp.path().join("models").join(format!("{sha}.gguf"));
        assert_eq!(got, expected);
        assert!(expected.exists());
    }

    #[test]
    fn resolve_tampered_file_triggers_redownload() {
        let tmp = TempDir::new().expect("tempdir");
        let body = b"trusted-body".to_vec();
        let cat = one_entry_catalog("tamper", &body);
        let sha = sha256_hex(&body);
        // Pre-stage a tampered file that won't match.
        let _staged = stage_file(tmp.path(), &sha, b"WRONG-BYTES");
        let r = ModelResolver::new(cat, tmp.path().to_path_buf());
        let slug: ModelSlug = "tamper".parse().expect("slug");
        let fetcher = StageFetcher {
            bytes: body.clone(),
        };
        let got = r.resolve(&slug, &fetcher).expect("re-download succeeds");
        assert_eq!(fs::read(&got).expect("read"), body);
    }

    #[test]
    fn resolve_fetcher_failure_surfaces_install_failed() {
        let tmp = TempDir::new().expect("tempdir");
        let body = b"never-fetched".to_vec();
        let cat = one_entry_catalog("nope", &body);
        let r = ModelResolver::new(cat, tmp.path().to_path_buf());
        let slug: ModelSlug = "nope".parse().expect("slug");
        let err = r
            .resolve(&slug, &FailingFetcher)
            .expect_err("fetch must fail");
        assert!(matches!(err, ResolveModelError::InstallFailed(_)));
        assert!(format!("{err}").contains("forced failure"));
    }

    #[test]
    fn resolve_post_fetch_hash_mismatch_reports_observed() {
        // Fetcher "succeeds" but writes the wrong bytes — resolver must
        // catch the discrepancy via re-hash.
        struct WrongBytesFetcher;
        impl BlobFetcher for WrongBytesFetcher {
            fn fetch(
                &self,
                _url: &str,
                _expected_sha256: &str,
                _expected_bytes: u64,
                dest: &Path,
                _policy: &RetryPolicy,
            ) -> Result<(), String> {
                if let Some(parent) = dest.parent() {
                    fs::create_dir_all(parent).map_err(|e| format!("mkdir: {e}"))?;
                }
                fs::write(dest, b"corrupt").map_err(|e| format!("write: {e}"))
            }
        }
        let tmp = TempDir::new().expect("tempdir");
        let body = b"expected".to_vec();
        let cat = one_entry_catalog("mm", &body);
        let r = ModelResolver::new(cat, tmp.path().to_path_buf());
        let slug: ModelSlug = "mm".parse().expect("slug");
        let err = r
            .resolve(&slug, &WrongBytesFetcher)
            .expect_err("hash mismatch");
        match err {
            ResolveModelError::HashMismatch {
                expected,
                actual,
                bytes_observed,
            } => {
                assert_eq!(expected, sha256_hex(&body));
                assert_ne!(actual, expected);
                assert_eq!(bytes_observed, 7);
            }
            other => panic!("wrong variant: {other:?}"),
        }
    }

    // ---- is_cached() ----

    #[test]
    fn is_cached_unknown_slug_errors() {
        let tmp = TempDir::new().expect("tempdir");
        let r = ModelResolver::new(ModelCatalog::new(), tmp.path().to_path_buf());
        let slug: ModelSlug = "ghost".parse().expect("slug");
        let err = r.is_cached(&slug).expect_err("unknown slug must err");
        assert!(matches!(err, ResolveModelError::UnknownSlug(_)));
    }

    #[test]
    fn is_cached_missing_file_is_false() {
        let tmp = TempDir::new().expect("tempdir");
        let cat = one_entry_catalog("solo", b"data");
        let r = ModelResolver::new(cat, tmp.path().to_path_buf());
        let slug: ModelSlug = "solo".parse().expect("slug");
        assert!(!r.is_cached(&slug).expect("ok"));
    }

    #[test]
    fn is_cached_present_correct_file_is_true() {
        let tmp = TempDir::new().expect("tempdir");
        let body = b"good".to_vec();
        let cat = one_entry_catalog("good", &body);
        let sha = sha256_hex(&body);
        let _staged = stage_file(tmp.path(), &sha, &body);
        let r = ModelResolver::new(cat, tmp.path().to_path_buf());
        let slug: ModelSlug = "good".parse().expect("slug");
        assert!(r.is_cached(&slug).expect("ok"));
    }

    #[test]
    fn is_cached_present_wrong_sha_is_false() {
        let tmp = TempDir::new().expect("tempdir");
        let body = b"expected".to_vec();
        let cat = one_entry_catalog("wrong", &body);
        let sha = sha256_hex(&body);
        // Stage a file at the expected path but with wrong bytes (same length so
        // we exercise the hash compare path, not just the length short-circuit).
        let mut wrong = body;
        wrong[0] ^= 0xff;
        let _staged = stage_file(tmp.path(), &sha, &wrong);
        let r = ModelResolver::new(cat, tmp.path().to_path_buf());
        let slug: ModelSlug = "wrong".parse().expect("slug");
        assert!(!r.is_cached(&slug).expect("ok"));
    }

    #[test]
    fn is_cached_present_wrong_size_is_false() {
        let tmp = TempDir::new().expect("tempdir");
        let body = b"expected".to_vec();
        let cat = one_entry_catalog("size", &body);
        let sha = sha256_hex(&body);
        let _staged = stage_file(tmp.path(), &sha, b"x"); // wrong length
        let r = ModelResolver::new(cat, tmp.path().to_path_buf());
        let slug: ModelSlug = "size".parse().expect("slug");
        assert!(!r.is_cached(&slug).expect("ok"));
    }

    #[test]
    fn is_cached_path_is_directory_is_false() {
        let tmp = TempDir::new().expect("tempdir");
        let body = b"x".to_vec();
        let cat = one_entry_catalog("dir", &body);
        let sha = sha256_hex(&body);
        let dir = tmp.path().join("models").join(format!("{sha}.gguf"));
        fs::create_dir_all(&dir).expect("create dir at target path");
        let r = ModelResolver::new(cat, tmp.path().to_path_buf());
        let slug: ModelSlug = "dir".parse().expect("slug");
        assert!(!r.is_cached(&slug).expect("ok"));
    }

    // ---- evict() ----

    #[test]
    fn evict_unknown_slug_errors() {
        let tmp = TempDir::new().expect("tempdir");
        let r = ModelResolver::new(ModelCatalog::new(), tmp.path().to_path_buf());
        let slug: ModelSlug = "phantom".parse().expect("slug");
        let err = r.evict(&slug).expect_err("unknown slug must err");
        assert!(matches!(err, ResolveModelError::UnknownSlug(_)));
    }

    #[test]
    fn evict_present_removes_file_and_returns_true() {
        let tmp = TempDir::new().expect("tempdir");
        let body = b"to-remove".to_vec();
        let cat = one_entry_catalog("rm", &body);
        let sha = sha256_hex(&body);
        let staged = stage_file(tmp.path(), &sha, &body);
        let r = ModelResolver::new(cat, tmp.path().to_path_buf());
        let slug: ModelSlug = "rm".parse().expect("slug");
        assert!(r.evict(&slug).expect("evict"));
        assert!(!staged.exists());
    }

    #[test]
    fn evict_absent_returns_false() {
        let tmp = TempDir::new().expect("tempdir");
        let cat = one_entry_catalog("absent", b"x");
        let r = ModelResolver::new(cat, tmp.path().to_path_buf());
        let slug: ModelSlug = "absent".parse().expect("slug");
        assert!(!r.evict(&slug).expect("ok"));
    }

    // ---- list_cached() ----

    #[test]
    fn list_cached_empty_catalog() {
        let tmp = TempDir::new().expect("tempdir");
        let r = ModelResolver::new(ModelCatalog::new(), tmp.path().to_path_buf());
        assert!(r.list_cached().expect("ok").is_empty());
    }

    #[test]
    fn list_cached_returns_only_present_entries_sorted() {
        let tmp = TempDir::new().expect("tempdir");
        let body_b = b"second".to_vec();
        let mut cat = ModelCatalog::new();
        cat.insert(entry_with("alpha", b"first"));
        cat.insert(entry_with("bravo", &body_b));
        cat.insert(entry_with("charlie", b"third"));
        let sha_b = sha256_hex(&body_b);
        let _staged = stage_file(tmp.path(), &sha_b, &body_b);
        let r = ModelResolver::new(cat, tmp.path().to_path_buf());
        let got = r.list_cached().expect("ok");
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].as_str(), "bravo");
    }

    #[test]
    fn list_cached_returns_all_when_all_present() {
        let tmp = TempDir::new().expect("tempdir");
        let body_a = b"a-body".to_vec();
        let body_b = b"b-body".to_vec();
        let mut cat = ModelCatalog::new();
        cat.insert(entry_with("a", &body_a));
        cat.insert(entry_with("b", &body_b));
        let _sa = stage_file(tmp.path(), &sha256_hex(&body_a), &body_a);
        let _sb = stage_file(tmp.path(), &sha256_hex(&body_b), &body_b);
        let r = ModelResolver::new(cat, tmp.path().to_path_buf());
        let got: Vec<String> = r
            .list_cached()
            .expect("ok")
            .into_iter()
            .map(|s| s.as_str().to_owned())
            .collect();
        assert_eq!(got, vec!["a".to_owned(), "b".to_owned()]);
    }

    // ---- error display + variants ----

    #[test]
    fn resolve_model_error_display_unknown_slug() {
        let e = ResolveModelError::UnknownSlug("foo".to_owned());
        assert!(format!("{e}").contains("unknown"));
        assert!(format!("{e}").contains("foo"));
    }

    #[test]
    fn resolve_model_error_display_install_failed() {
        let e = ResolveModelError::InstallFailed("boom".to_owned());
        assert!(format!("{e}").contains("install"));
        assert!(format!("{e}").contains("boom"));
    }

    #[test]
    fn resolve_model_error_display_io() {
        let e = ResolveModelError::Io(io::Error::other("disk"));
        assert!(format!("{e}").contains("io"));
        // source() must surface the inner io error.
        assert!(e.source().is_some());
    }

    #[test]
    fn resolve_model_error_display_hash_mismatch_constructable() {
        let e = ResolveModelError::HashMismatch {
            expected: "aa".to_owned(),
            actual: "bb".to_owned(),
            bytes_observed: 42,
        };
        let msg = format!("{e}");
        assert!(msg.contains("aa"));
        assert!(msg.contains("bb"));
        assert!(msg.contains("42"));
        // non-Io variants have no inner source.
        assert!(e.source().is_none());
    }

    #[test]
    fn resolve_model_error_source_none_for_non_io_variants() {
        assert!(ResolveModelError::UnknownSlug("x".to_owned())
            .source()
            .is_none());
        assert!(ResolveModelError::InstallFailed("x".to_owned())
            .source()
            .is_none());
    }

    // ---- with_retry_policy override ----

    #[test]
    fn with_retry_policy_overrides_default() {
        let tmp = TempDir::new().expect("tempdir");
        let r = ModelResolver::new(ModelCatalog::new(), tmp.path().to_path_buf());
        let custom = RetryPolicy {
            max_attempts: 1,
            initial_delay: std::time::Duration::from_millis(1),
            max_delay: std::time::Duration::from_millis(2),
            backoff_multiplier: 1.5,
            jitter: crate::retry::Jitter::None,
        };
        let r2 = r.with_retry_policy(custom);
        assert_eq!(r2.retry_policy.max_attempts, custom.max_attempts);
    }

    // ---- Send + Sync smoke ----

    #[test]
    fn resolver_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ModelResolver>();
        assert_send_sync::<ResolveModelError>();
    }
}
