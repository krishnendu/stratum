//! Remote [`ModelCatalog`] fetcher.
//!
//! [`CatalogSync`] composes the curated [`ModelCatalog`] data shape with the
//! `ureq` HTTP client and the [`crate::retry`] helper to fetch a channel
//! catalog over HTTPS, validate it, and atomically replace the on-disk file.
//!
//! Per `plan/05-models-and-installer.md` §4 the catalog ships as JSON from
//! `https://catalog.stratum.dev/<channel>.json`. This module is Phase 1
//! scaffold: it stabilises the wire surface for the installer + future
//! `stratum catalog sync` CLI command. The body is bounded
//! (`max_bytes`, default 4 MiB), `https://` is mandatory in release builds,
//! and the bundled `validate()` + `schema_version` guard runs before the
//! catalog is handed to the caller.

use std::error::Error;
use std::fmt;
use std::io::Read;
use std::path::Path;
use std::time::{Duration, SystemTime};

use rand::rngs::SmallRng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

use crate::model_catalog::{CatalogError, ModelCatalog, MODEL_CATALOG_SCHEMA_VERSION};
use crate::retry::{
    retry_with_clock_seeded, Clock, RetryDecision, RetryError, RetryPolicy, SystemClock,
};

/// Default channel sync URL — matches the documented production endpoint.
pub const DEFAULT_CATALOG_URL: &str = "https://catalog.stratum.dev/stable.json";
/// Default release channel name.
pub const DEFAULT_CATALOG_CHANNEL: &str = "stable";
/// Default fetch timeout (10s) — generous enough for slow networks, short
/// enough that a wedged transport is surfaced quickly to the retry layer.
pub const DEFAULT_CATALOG_TIMEOUT: Duration = Duration::from_secs(10);
/// Default body cap (4 MiB) — comfortably above the curated catalog size,
/// well below the smallest mobile data cap a user might be on.
pub const DEFAULT_CATALOG_MAX_BYTES: u64 = 4 * 1024 * 1024;

/// Configuration for a [`CatalogSync`] fetch.
///
/// All fields are public so call sites can tweak the URL / channel / limits
/// per environment (e.g. an enterprise mirror). `Default` matches the
/// production endpoint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatalogSyncConfig {
    /// HTTPS URL of the channel JSON.
    pub url: String,
    /// Release-channel name (`stable` / `beta` / `nightly`).
    pub channel: String,
    /// Per-attempt fetch timeout.
    pub timeout: Duration,
    /// Hard cap on the response body, in bytes.
    pub max_bytes: u64,
}

impl Default for CatalogSyncConfig {
    fn default() -> Self {
        Self {
            url: DEFAULT_CATALOG_URL.to_owned(),
            channel: DEFAULT_CATALOG_CHANNEL.to_owned(),
            timeout: DEFAULT_CATALOG_TIMEOUT,
            max_bytes: DEFAULT_CATALOG_MAX_BYTES,
        }
    }
}

/// Report returned by [`CatalogSync::sync_to_file`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SyncReport {
    /// URL that was fetched.
    pub url: String,
    /// Release channel that was synced.
    pub channel: String,
    /// Wall-clock time the catalog was written to disk.
    pub fetched_at: SystemTime,
    /// Number of entries in the synced catalog.
    pub entries: usize,
}

/// Error surface for [`CatalogSync`].
#[derive(Debug)]
pub enum CatalogSyncError {
    /// Transport / IO failure (DNS, TCP, TLS, mid-body close, retry exhaustion).
    Network(String),
    /// HTTP status outside the 2xx range.
    NonOk {
        /// HTTP status code returned by the server.
        status: u16,
    },
    /// Response body exceeded `max_bytes`.
    TooLarge {
        /// Configured cap (in bytes).
        limit: u64,
    },
    /// JSON parse failure.
    Parse(String),
    /// Catalog parsed but failed [`ModelCatalog::validate`].
    InvalidContents(String),
    /// On-disk schema is newer than this binary supports.
    SchemaNewer {
        /// Schema version present in the response.
        found: u32,
        /// Highest version this build understands.
        supported: u32,
    },
    /// URL was empty, non-https (without the insecure override), or otherwise
    /// rejected before the request was made.
    BadUrl(String),
    /// Filesystem step (mostly `save_atomic`) failed.
    Io(String),
}

impl fmt::Display for CatalogSyncError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Network(s) => write!(f, "catalog sync network error: {s}"),
            Self::NonOk { status } => write!(f, "catalog sync http status {status}"),
            Self::TooLarge { limit } => {
                write!(f, "catalog sync body exceeded {limit} bytes")
            }
            Self::Parse(s) => write!(f, "catalog sync parse error: {s}"),
            Self::InvalidContents(s) => write!(f, "catalog sync invalid contents: {s}"),
            Self::SchemaNewer { found, supported } => write!(
                f,
                "catalog sync schema_version {found} is newer than supported {supported}",
            ),
            Self::BadUrl(s) => write!(f, "catalog sync bad url: {s}"),
            Self::Io(s) => write!(f, "catalog sync io error: {s}"),
        }
    }
}

impl Error for CatalogSyncError {}

impl CatalogSyncError {
    fn from_catalog(err: CatalogError) -> Self {
        match err {
            CatalogError::Io(e) => Self::Io(e.to_string()),
            CatalogError::Serialize(e) => Self::Parse(e.to_string()),
            CatalogError::Validation(s) => Self::InvalidContents(s),
            CatalogError::SchemaNewer { found, supported } => {
                Self::SchemaNewer { found, supported }
            }
        }
    }
}

/// Fetches a [`ModelCatalog`] over HTTPS and atomically writes it locally.
#[derive(Debug, Clone)]
pub struct CatalogSync {
    cfg: CatalogSyncConfig,
}

impl CatalogSync {
    /// Wrap `cfg`.
    #[must_use]
    pub const fn new(cfg: CatalogSyncConfig) -> Self {
        Self { cfg }
    }

    /// Borrow the active configuration.
    #[must_use]
    pub const fn config(&self) -> &CatalogSyncConfig {
        &self.cfg
    }

    /// Fetch and validate the catalog over the configured URL.
    ///
    /// # Errors
    /// See [`CatalogSyncError`] — covers bad URLs, transport failure, non-200,
    /// oversize bodies, JSON parse failure, validation failure, and a
    /// schema-newer guard.
    pub fn fetch(&self) -> Result<ModelCatalog, CatalogSyncError> {
        validate_url(&self.cfg.url)?;
        let response = match ureq::get(&self.cfg.url).timeout(self.cfg.timeout).call() {
            Ok(r) => r,
            Err(ureq::Error::Status(code, _)) => {
                return Err(CatalogSyncError::NonOk { status: code })
            }
            Err(ureq::Error::Transport(t)) => {
                return Err(CatalogSyncError::Network(format!("transport: {t}")))
            }
        };

        let status = response.status();
        if !(200..300).contains(&status) {
            return Err(CatalogSyncError::NonOk { status });
        }

        // Bound the body up front. Read one extra byte so we can tell the
        // difference between "body == limit" (ok) and "body > limit" (reject).
        let limit = self.cfg.max_bytes;
        let mut reader = response.into_reader().take(limit.saturating_add(1));
        let mut body = Vec::new();
        reader
            .read_to_end(&mut body)
            .map_err(|e| CatalogSyncError::Network(format!("read body: {e}")))?;
        if body.len() as u64 > limit {
            return Err(CatalogSyncError::TooLarge { limit });
        }

        let catalog: ModelCatalog =
            serde_json::from_slice(&body).map_err(|e| CatalogSyncError::Parse(e.to_string()))?;

        if catalog.schema_version > MODEL_CATALOG_SCHEMA_VERSION {
            return Err(CatalogSyncError::SchemaNewer {
                found: catalog.schema_version,
                supported: MODEL_CATALOG_SCHEMA_VERSION,
            });
        }

        catalog
            .validate()
            .map_err(|e| CatalogSyncError::InvalidContents(e.to_string()))?;

        Ok(catalog)
    }

    /// Fetch with the configured retry `policy`. 5xx + transport failures
    /// retry; 4xx, parse, validation, and schema-newer failures short-circuit.
    ///
    /// # Errors
    /// See [`CatalogSyncError`]. Retry exhaustion is reported as
    /// [`CatalogSyncError::Network`] carrying the last error.
    pub fn fetch_with_retry(&self, policy: &RetryPolicy) -> Result<ModelCatalog, CatalogSyncError> {
        self.fetch_with_retry_and_clock(policy, &SystemClock)
    }

    /// Clock-injecting variant used by deterministic tests.
    pub(crate) fn fetch_with_retry_and_clock<K: Clock + ?Sized>(
        &self,
        policy: &RetryPolicy,
        clock: &K,
    ) -> Result<ModelCatalog, CatalogSyncError> {
        let mut rng = SmallRng::seed_from_u64(0);
        let classifier = |err: &CatalogSyncError| match err {
            CatalogSyncError::Network(_) => RetryDecision::Retry,
            CatalogSyncError::NonOk { status } => {
                if (500..600).contains(status) {
                    RetryDecision::Retry
                } else {
                    RetryDecision::Fatal
                }
            }
            CatalogSyncError::TooLarge { .. }
            | CatalogSyncError::Parse(_)
            | CatalogSyncError::InvalidContents(_)
            | CatalogSyncError::SchemaNewer { .. }
            | CatalogSyncError::BadUrl(_)
            | CatalogSyncError::Io(_) => RetryDecision::Fatal,
        };
        let mut op = |_attempt: u32| -> Result<ModelCatalog, CatalogSyncError> { self.fetch() };
        match retry_with_clock_seeded(policy, &classifier, clock, &mut rng, &mut op) {
            Ok(c) => Ok(c),
            Err(RetryError::Fatal(e)) => Err(e),
            Err(RetryError::Exhausted {
                attempts,
                last_error,
            }) => Err(CatalogSyncError::Network(format!(
                "exhausted after {attempts} attempts: {last_error}"
            ))),
        }
    }

    /// Fetch and atomically write the catalog to `path`.
    ///
    /// The caller is responsible for creating the parent directory — this
    /// method does NOT auto-create it, to stay symmetric with
    /// [`ModelCatalog::save_atomic`].
    ///
    /// # Errors
    /// See [`CatalogSyncError`].
    pub fn sync_to_file(&self, path: &Path) -> Result<SyncReport, CatalogSyncError> {
        let catalog = self.fetch()?;
        catalog
            .save_atomic(path)
            .map_err(CatalogSyncError::from_catalog)?;
        Ok(SyncReport {
            url: self.cfg.url.clone(),
            channel: self.cfg.channel.clone(),
            fetched_at: SystemTime::now(),
            entries: catalog.entries.len(),
        })
    }
}

/// Returns `true` iff loopback HTTP is allowed for this build.
///
/// Debug builds always allow it (so unit tests don't need any env setup);
/// release builds require `STRATUM_ALLOW_INSECURE_URL` to be set.
fn loopback_http_allowed() -> bool {
    cfg!(debug_assertions) || std::env::var("STRATUM_ALLOW_INSECURE_URL").is_ok()
}

fn validate_url(url: &str) -> Result<(), CatalogSyncError> {
    if url.is_empty() {
        return Err(CatalogSyncError::BadUrl("url is empty".to_owned()));
    }
    if url.starts_with("https://") {
        return Ok(());
    }
    // Allow http://127.0.0.1 (any port/path) when loopback is allowed.
    if url.starts_with("http://127.0.0.1") && loopback_http_allowed() {
        return Ok(());
    }
    Err(CatalogSyncError::BadUrl(format!(
        "url must be https:// (got {url})"
    )))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::io::{Read as _, Write as _};
    use std::net::TcpListener;

    use tempfile::TempDir;

    use super::*;
    use crate::model_catalog::{ArtifactRef, ModelEntry, ModelSlug, ModelTask, ModelTier};
    use crate::retry::{Jitter, ManualClock};

    // ---------- helpers ----------

    fn good_sha() -> String {
        "a".repeat(64)
    }

    fn sample_artifact() -> ArtifactRef {
        ArtifactRef::new("https://example.com/m.gguf".to_owned(), good_sha(), 1_024)
            .expect("artifact")
    }

    fn sample_entry(slug: &str) -> ModelEntry {
        let slug: ModelSlug = slug.parse().expect("slug");
        let mut task = BTreeSet::new();
        task.insert(ModelTask::Chat);
        ModelEntry {
            slug,
            family: "llama".to_owned(),
            display_name: "Llama Test".to_owned(),
            tier: ModelTier::Low,
            task,
            size_mib: 128,
            quantization: "Q4_K_M".to_owned(),
            artifact: sample_artifact(),
            license: "Apache-2.0".to_owned(),
            homepage: None,
        }
    }

    fn sample_catalog() -> ModelCatalog {
        let mut cat = ModelCatalog::new();
        cat.insert(sample_entry("foo"));
        cat.insert(sample_entry("bar"));
        cat
    }

    fn sample_catalog_json() -> Vec<u8> {
        serde_json::to_vec_pretty(&sample_catalog()).expect("serialize")
    }

    /// Spin up a single-response server that writes `headers` + `payload`.
    fn spawn_one_shot(headers: String, payload: Vec<u8>) -> (String, u16) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");
        std::thread::spawn(move || {
            for stream in listener.incoming().take(1) {
                let Ok(mut stream) = stream else { continue };
                let mut buf = [0_u8; 4096];
                let _ = stream.read(&mut buf);
                let _ = stream.write_all(headers.as_bytes());
                let _ = stream.write_all(&payload);
                let _ = stream.flush();
            }
        });
        (format!("http://{addr}/catalog.json"), addr.port())
    }

    fn spawn_static_json(body: Vec<u8>) -> String {
        let headers = format!(
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n",
            body.len()
        );
        let (url, _) = spawn_one_shot(headers, body);
        url
    }

    fn spawn_status(code: u16, reason: &str) -> String {
        let headers =
            format!("HTTP/1.1 {code} {reason}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
        let (url, _) = spawn_one_shot(headers, Vec::new());
        url
    }

    /// Server that serves the given body for the first `n` connections.
    /// Each subsequent connection uses the next `body_for_attempt(n)`.
    fn spawn_sequence(steps: Vec<(String, Vec<u8>)>) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");
        let count = steps.len();
        std::thread::spawn(move || {
            let mut iter = steps.into_iter();
            for stream in listener.incoming().take(count) {
                let Ok(mut stream) = stream else { continue };
                let Some((headers, payload)) = iter.next() else {
                    break;
                };
                let mut buf = [0_u8; 4096];
                let _ = stream.read(&mut buf);
                let _ = stream.write_all(headers.as_bytes());
                let _ = stream.write_all(&payload);
                let _ = stream.flush();
            }
        });
        format!("http://{addr}/catalog.json")
    }

    fn cfg_with(url: String) -> CatalogSyncConfig {
        CatalogSyncConfig {
            url,
            channel: "stable".to_owned(),
            timeout: Duration::from_secs(2),
            max_bytes: DEFAULT_CATALOG_MAX_BYTES,
        }
    }

    fn fast_policy(max_attempts: u32) -> RetryPolicy {
        RetryPolicy {
            max_attempts,
            initial_delay: Duration::from_millis(1),
            max_delay: Duration::from_millis(1),
            backoff_multiplier: 1.0,
            jitter: Jitter::None,
        }
    }

    // ---------- tests ----------

    #[test]
    fn config_default_matches_documented_values() {
        let c = CatalogSyncConfig::default();
        assert_eq!(c.url, "https://catalog.stratum.dev/stable.json");
        assert_eq!(c.channel, "stable");
        assert_eq!(c.timeout, Duration::from_secs(10));
        assert_eq!(c.max_bytes, 4 * 1024 * 1024);
    }

    #[test]
    fn config_serde_round_trip() {
        let c = CatalogSyncConfig::default();
        let s = serde_json::to_string(&c).expect("serialize");
        let back: CatalogSyncConfig = serde_json::from_str(&s).expect("deserialize");
        assert_eq!(back, c);
    }

    #[test]
    fn fetch_happy_path_returns_non_empty_catalog() {
        let url = spawn_static_json(sample_catalog_json());
        let sync = CatalogSync::new(cfg_with(url));
        let catalog = sync.fetch().expect("fetch ok");
        assert_eq!(catalog.entries.len(), 2);
        assert_eq!(catalog.schema_version, MODEL_CATALOG_SCHEMA_VERSION);
    }

    #[test]
    fn fetch_rejects_non_https_url() {
        let cfg = CatalogSyncConfig {
            url: "ftp://example.com/x.json".to_owned(),
            ..CatalogSyncConfig::default()
        };
        let sync = CatalogSync::new(cfg);
        let err = sync.fetch().expect_err("must reject");
        assert!(matches!(err, CatalogSyncError::BadUrl(_)));
        assert!(format!("{err}").contains("bad url"));
    }

    #[test]
    fn fetch_rejects_empty_url() {
        let cfg = CatalogSyncConfig {
            url: String::new(),
            ..CatalogSyncConfig::default()
        };
        let sync = CatalogSync::new(cfg);
        let err = sync.fetch().expect_err("must reject");
        assert!(matches!(err, CatalogSyncError::BadUrl(_)));
    }

    #[test]
    fn fetch_allows_loopback_under_debug_assertions() {
        // In `cargo test` (debug build) the loopback exception is always on,
        // so a plain `http://127.0.0.1:<port>` URL must succeed.
        let url = spawn_static_json(sample_catalog_json());
        assert!(url.starts_with("http://127.0.0.1"));
        let sync = CatalogSync::new(cfg_with(url));
        let cat = sync.fetch().expect("loopback allowed");
        assert!(!cat.entries.is_empty());
    }

    #[test]
    fn fetch_non_200_returns_non_ok() {
        let url = spawn_status(503, "Service Unavailable");
        let sync = CatalogSync::new(cfg_with(url));
        let err = sync.fetch().expect_err("503");
        match err {
            CatalogSyncError::NonOk { status } => assert_eq!(status, 503),
            other => panic!("unexpected: {other}"),
        }
    }

    #[test]
    fn fetch_404_returns_non_ok() {
        let url = spawn_status(404, "Not Found");
        let sync = CatalogSync::new(cfg_with(url));
        let err = sync.fetch().expect_err("404");
        assert!(matches!(err, CatalogSyncError::NonOk { status: 404 }));
    }

    #[test]
    fn fetch_body_over_max_bytes_returns_too_large() {
        let big = vec![b'x'; 4096];
        let url = spawn_static_json(big);
        let cfg = CatalogSyncConfig {
            max_bytes: 16,
            ..cfg_with(url)
        };
        let sync = CatalogSync::new(cfg);
        let err = sync.fetch().expect_err("too large");
        match err {
            CatalogSyncError::TooLarge { limit } => assert_eq!(limit, 16),
            other => panic!("unexpected: {other}"),
        }
    }

    #[test]
    fn fetch_malformed_json_returns_parse() {
        let url = spawn_static_json(b"{not json".to_vec());
        let sync = CatalogSync::new(cfg_with(url));
        let err = sync.fetch().expect_err("parse");
        assert!(matches!(err, CatalogSyncError::Parse(_)));
        assert!(format!("{err}").contains("parse"));
    }

    #[test]
    fn fetch_schema_newer_returns_schema_newer() {
        let body = serde_json::json!({
            "schema_version": 999,
            "entries": {}
        });
        let bytes = serde_json::to_vec_pretty(&body).expect("serialize");
        let url = spawn_static_json(bytes);
        let sync = CatalogSync::new(cfg_with(url));
        let err = sync.fetch().expect_err("schema newer");
        match err {
            CatalogSyncError::SchemaNewer { found, supported } => {
                assert_eq!(found, 999);
                assert_eq!(supported, MODEL_CATALOG_SCHEMA_VERSION);
            }
            other => panic!("unexpected: {other}"),
        }
    }

    #[test]
    fn fetch_invalid_contents_when_validate_fails() {
        // Build a catalog where the slug key doesn't match the entry slug.
        let mut cat = sample_catalog();
        let mut entry = sample_entry("foo");
        entry.slug = "different".parse().expect("slug");
        cat.entries.insert("foo".parse().expect("key"), entry);
        let bytes = serde_json::to_vec_pretty(&cat).expect("serialize");
        let url = spawn_static_json(bytes);
        let sync = CatalogSync::new(cfg_with(url));
        let err = sync.fetch().expect_err("invalid");
        assert!(matches!(err, CatalogSyncError::InvalidContents(_)));
        assert!(format!("{err}").contains("invalid contents"));
    }

    #[test]
    fn sync_to_file_writes_catalog_atomically() {
        let bytes = sample_catalog_json();
        let url = spawn_static_json(bytes);
        let cfg = cfg_with(url);
        let sync = CatalogSync::new(cfg);

        let dir = TempDir::new().expect("tmp");
        let path = dir.path().join("catalog.json");
        let report = sync.sync_to_file(&path).expect("sync ok");
        assert_eq!(report.channel, "stable");
        assert_eq!(report.entries, 2);
        assert!(report.url.starts_with("http://127.0.0.1"));
        // File on disk round-trips.
        let on_disk = ModelCatalog::load(&path).expect("load");
        assert_eq!(on_disk, sample_catalog());
        // No leftover .tmp.
        assert!(!dir.path().join("catalog.json.tmp").exists());
    }

    #[test]
    fn sync_to_file_errors_when_parent_dir_missing() {
        let url = spawn_static_json(sample_catalog_json());
        let sync = CatalogSync::new(cfg_with(url));
        let dir = TempDir::new().expect("tmp");
        let path = dir.path().join("missing-dir").join("catalog.json");
        let err = sync.sync_to_file(&path).expect_err("io");
        assert!(matches!(err, CatalogSyncError::Io(_)));
        assert!(format!("{err}").contains("io"));
    }

    #[test]
    fn sync_report_serde_round_trip() {
        let r = SyncReport {
            url: "https://example.com/x.json".to_owned(),
            channel: "stable".to_owned(),
            fetched_at: SystemTime::UNIX_EPOCH + Duration::from_secs(1_700_000_000),
            entries: 7,
        };
        let s = serde_json::to_string(&r).expect("serialize");
        let back: SyncReport = serde_json::from_str(&s).expect("deserialize");
        assert_eq!(back, r);
    }

    #[test]
    fn catalog_sync_error_display_covers_every_variant() {
        let cases = [
            CatalogSyncError::Network("dns".to_owned()),
            CatalogSyncError::NonOk { status: 503 },
            CatalogSyncError::TooLarge { limit: 16 },
            CatalogSyncError::Parse("bad".to_owned()),
            CatalogSyncError::InvalidContents("nope".to_owned()),
            CatalogSyncError::SchemaNewer {
                found: 9,
                supported: 1,
            },
            CatalogSyncError::BadUrl("oops".to_owned()),
            CatalogSyncError::Io("denied".to_owned()),
        ];
        for c in &cases {
            let s = format!("{c}");
            assert!(s.contains("catalog sync"), "missing prefix: {s}");
        }
    }

    #[test]
    fn fetch_with_retry_retries_on_502_then_succeeds() {
        let body = sample_catalog_json();
        let h502 =
            "HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_owned();
        let h200 = format!(
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n",
            body.len()
        );
        let url = spawn_sequence(vec![(h502, Vec::new()), (h200, body)]);
        let sync = CatalogSync::new(cfg_with(url));
        let clock = ManualClock::new();
        let cat = sync
            .fetch_with_retry_and_clock(&fast_policy(4), &clock)
            .expect("retry ok");
        assert_eq!(cat.entries.len(), 2);
        assert_eq!(clock.sleeps().len(), 1, "exactly one inter-attempt sleep");
    }

    #[test]
    fn fetch_with_retry_exhausts_on_persistent_502() {
        let h502 =
            "HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_owned();
        let url = spawn_sequence(vec![
            (h502.clone(), Vec::new()),
            (h502.clone(), Vec::new()),
            (h502.clone(), Vec::new()),
            (h502, Vec::new()),
        ]);
        let sync = CatalogSync::new(cfg_with(url));
        let clock = ManualClock::new();
        let err = sync
            .fetch_with_retry_and_clock(&fast_policy(4), &clock)
            .expect_err("exhausted");
        match err {
            CatalogSyncError::Network(msg) => {
                assert!(msg.contains("exhausted"));
                assert!(msg.contains("attempts"));
            }
            other => panic!("unexpected: {other}"),
        }
        // 4 attempts → 3 inter-attempt sleeps.
        assert_eq!(clock.sleeps().len(), 3);
    }

    #[test]
    fn fetch_with_retry_404_is_fatal_single_attempt() {
        let url = spawn_status(404, "Not Found");
        let sync = CatalogSync::new(cfg_with(url));
        let clock = ManualClock::new();
        let err = sync
            .fetch_with_retry_and_clock(&fast_policy(4), &clock)
            .expect_err("fatal");
        assert!(matches!(err, CatalogSyncError::NonOk { status: 404 }));
        assert!(
            clock.sleeps().is_empty(),
            "fatal must not sleep between attempts"
        );
    }

    #[test]
    fn fetch_with_retry_schema_newer_is_fatal_single_attempt() {
        let body = serde_json::json!({
            "schema_version": 999,
            "entries": {}
        });
        let bytes = serde_json::to_vec_pretty(&body).expect("serialize");
        let url = spawn_static_json(bytes);
        let sync = CatalogSync::new(cfg_with(url));
        let clock = ManualClock::new();
        let err = sync
            .fetch_with_retry_and_clock(&fast_policy(4), &clock)
            .expect_err("fatal");
        assert!(matches!(err, CatalogSyncError::SchemaNewer { .. }));
        assert!(clock.sleeps().is_empty());
    }

    #[test]
    fn fetch_with_retry_parse_is_fatal_single_attempt() {
        let url = spawn_static_json(b"{not json".to_vec());
        let sync = CatalogSync::new(cfg_with(url));
        let clock = ManualClock::new();
        let err = sync
            .fetch_with_retry_and_clock(&fast_policy(4), &clock)
            .expect_err("fatal");
        assert!(matches!(err, CatalogSyncError::Parse(_)));
        assert!(clock.sleeps().is_empty());
    }

    #[test]
    fn fetch_with_retry_transport_failure_retries() {
        // First connection: spawn nothing — connect refused for an unbound
        // port. Second: a real server returning the catalog. Use two distinct
        // URLs by pre-binding the second server and constructing the policy
        // so the first attempt hits a dead port.
        //
        // We can't do this cleanly with the one-URL CatalogSync, so instead
        // we drive `fetch_with_retry` against an entirely-dead port and just
        // assert it's classified as Network and exhausts.
        let cfg = CatalogSyncConfig {
            url: "http://127.0.0.1:1/never".to_owned(),
            channel: "stable".to_owned(),
            timeout: Duration::from_millis(100),
            max_bytes: 4096,
        };
        let sync = CatalogSync::new(cfg);
        let clock = ManualClock::new();
        let err = sync
            .fetch_with_retry_and_clock(&fast_policy(2), &clock)
            .expect_err("exhausted");
        assert!(matches!(err, CatalogSyncError::Network(_)));
        assert_eq!(clock.sleeps().len(), 1);
    }

    #[test]
    fn catalog_sync_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<CatalogSync>();
        assert_send_sync::<CatalogSyncConfig>();
        assert_send_sync::<SyncReport>();
        assert_send_sync::<CatalogSyncError>();
    }

    #[test]
    fn config_accessor_returns_borrow() {
        let cfg = CatalogSyncConfig::default();
        let sync = CatalogSync::new(cfg.clone());
        assert_eq!(sync.config(), &cfg);
    }

    #[test]
    fn fetch_with_retry_real_clock_smoke() {
        // Exercise the public `fetch_with_retry` so SystemClock is covered.
        let url = spawn_static_json(sample_catalog_json());
        let sync = CatalogSync::new(cfg_with(url));
        let cat = sync
            .fetch_with_retry(&fast_policy(1))
            .expect("real-clock retry ok");
        assert_eq!(cat.entries.len(), 2);
    }

    #[test]
    fn loopback_http_allowed_under_debug() {
        assert!(loopback_http_allowed());
    }

    #[test]
    fn from_catalog_serialize_maps_to_parse() {
        let e = ModelCatalog::load(std::path::Path::new("/nonexistent/path/x.json"))
            .expect_err("missing");
        let mapped = CatalogSyncError::from_catalog(e);
        assert!(matches!(mapped, CatalogSyncError::Io(_)));
    }

    #[test]
    fn from_catalog_validation_maps_to_invalid_contents() {
        let mapped = CatalogSyncError::from_catalog(CatalogError::Validation("oops".to_owned()));
        assert!(matches!(mapped, CatalogSyncError::InvalidContents(_)));
    }

    #[test]
    fn from_catalog_schema_newer_maps_through() {
        let mapped = CatalogSyncError::from_catalog(CatalogError::SchemaNewer {
            found: 9,
            supported: 1,
        });
        assert!(matches!(
            mapped,
            CatalogSyncError::SchemaNewer {
                found: 9,
                supported: 1
            }
        ));
    }

    #[test]
    fn from_catalog_serialize_error_maps_to_parse() {
        let serde_err = serde_json::from_str::<ModelCatalog>("{").expect_err("bad json");
        let mapped = CatalogSyncError::from_catalog(CatalogError::Serialize(serde_err));
        assert!(matches!(mapped, CatalogSyncError::Parse(_)));
    }
}
