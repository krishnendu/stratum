//! Curated model catalog — the structured index of models the installer can
//! browse and resolve.
//!
//! Per `plan/05-models-and-installer.md` §3, the catalog is the source of
//! truth for "which models can a user pick at install time". Each entry binds
//! a stable slug to a downloadable artifact (URL + sha256 + bytes), a tier /
//! task taxonomy, and provenance fields (family, license, homepage).
//!
//! The on-disk shape is JSON for simple cross-tool consumption (`stratum`
//! installer, future website, packaging scripts). Save uses an atomic
//! `<path>.tmp` + rename pattern so a half-written file is never observed.
//!
//! Phase 1 scaffold — the installer and recommender that consume this land
//! in later phases. Today we pin the shape, validation, and serde so call
//! sites and tests can stabilize.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;
use std::fs;
use std::path::Path;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

/// Current on-disk schema version for the model catalog JSON. Bump whenever
/// the wire shape changes; older readers refuse newer files via
/// [`CatalogError::SchemaNewer`].
pub const MODEL_CATALOG_SCHEMA_VERSION: u32 = 1;

const MODEL_SLUG_MAX_LEN: usize = 64;
const SHA256_HEX_LEN: usize = 64;

/// Top-level curated catalog of installable models.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelCatalog {
    /// On-disk schema version. Older readers refuse newer files.
    pub schema_version: u32,
    /// Ordered map slug → entry, for deterministic JSON output.
    pub entries: BTreeMap<ModelSlug, ModelEntry>,
}

impl Default for ModelCatalog {
    fn default() -> Self {
        Self {
            schema_version: MODEL_CATALOG_SCHEMA_VERSION,
            entries: BTreeMap::new(),
        }
    }
}

impl ModelCatalog {
    /// Empty catalog pinned at the current schema version.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert (or replace) `entry`. Returns the previous entry under the
    /// same slug if one existed.
    pub fn insert(&mut self, entry: ModelEntry) -> Option<ModelEntry> {
        self.entries.insert(entry.slug.clone(), entry)
    }

    /// Look up an entry by slug.
    #[must_use]
    pub fn get(&self, slug: &ModelSlug) -> Option<&ModelEntry> {
        self.entries.get(slug)
    }

    /// All entries whose tier matches `tier`, sorted by slug.
    #[must_use]
    pub fn filter_by_tier(&self, tier: ModelTier) -> Vec<&ModelEntry> {
        // Entries are already iterated in slug order thanks to `BTreeMap`.
        self.entries.values().filter(|e| e.tier == tier).collect()
    }

    /// All entries whose task set contains `task`, sorted by slug.
    #[must_use]
    pub fn filter_by_task(&self, task: ModelTask) -> Vec<&ModelEntry> {
        self.entries
            .values()
            .filter(|e| e.task.contains(&task))
            .collect()
    }

    /// Recommend the smallest entry that fits the user's tier budget and
    /// covers `task`.
    ///
    /// Selection rule: tier `<=` requested AND tasks contain `task`. Among
    /// candidates, the entry with the smallest `size_mib` wins; ties are
    /// broken by slug lexicographic order.
    #[must_use]
    pub fn recommend_for(&self, tier: ModelTier, task: ModelTask) -> Option<&ModelEntry> {
        self.entries
            .values()
            .filter(|e| e.tier <= tier && e.task.contains(&task))
            .min_by(|a, b| {
                a.size_mib
                    .cmp(&b.size_mib)
                    .then_with(|| a.slug.cmp(&b.slug))
            })
    }

    /// Validate cross-entry invariants:
    /// - every entry's `slug` matches its map key,
    /// - every artifact re-validates via [`ArtifactRef::new`],
    /// - every `size_mib > 0`,
    /// - every `family` is non-empty.
    ///
    /// # Errors
    /// Returns [`CatalogError::Validation`] on the first failed invariant.
    pub fn validate(&self) -> Result<(), CatalogError> {
        for (key, entry) in &self.entries {
            if &entry.slug != key {
                return Err(CatalogError::Validation(format!(
                    "slug/key mismatch: key={key} entry.slug={entry_slug}",
                    entry_slug = entry.slug
                )));
            }
            if entry.family.trim().is_empty() {
                return Err(CatalogError::Validation(format!(
                    "entry {key} has empty family",
                )));
            }
            if entry.size_mib == 0 {
                return Err(CatalogError::Validation(format!(
                    "entry {key} has zero size_mib",
                )));
            }
            // Round-trip the artifact through the constructor so any
            // validation drift here surfaces immediately.
            ArtifactRef::new(
                entry.artifact.url.clone(),
                entry.artifact.sha256.clone(),
                entry.artifact.bytes,
            )
            .map_err(|e| CatalogError::Validation(format!("entry {key} artifact invalid: {e}")))?;
        }
        Ok(())
    }

    /// Load a catalog from JSON at `path`.
    ///
    /// # Errors
    /// - [`CatalogError::Io`] for filesystem failures.
    /// - [`CatalogError::Serialize`] for malformed JSON.
    /// - [`CatalogError::SchemaNewer`] if the file declares a schema newer
    ///   than this binary supports.
    pub fn load(path: &Path) -> Result<Self, CatalogError> {
        let bytes = fs::read(path).map_err(CatalogError::Io)?;
        let catalog: Self = serde_json::from_slice(&bytes).map_err(CatalogError::Serialize)?;
        if catalog.schema_version > MODEL_CATALOG_SCHEMA_VERSION {
            return Err(CatalogError::SchemaNewer {
                found: catalog.schema_version,
                supported: MODEL_CATALOG_SCHEMA_VERSION,
            });
        }
        Ok(catalog)
    }

    /// Atomically write `self` to `path` as pretty-printed JSON.
    ///
    /// Uses a sibling `<path>.tmp` and `rename(2)`: on POSIX the swap is
    /// atomic, so concurrent readers never see a half-written catalog.
    ///
    /// # Errors
    /// - [`CatalogError::Serialize`] if JSON serialization fails.
    /// - [`CatalogError::Io`] if any filesystem step fails.
    pub fn save_atomic(&self, path: &Path) -> Result<(), CatalogError> {
        let body = serde_json::to_vec_pretty(self).map_err(CatalogError::Serialize)?;

        let tmp = {
            let mut name = path.as_os_str().to_owned();
            name.push(".tmp");
            std::path::PathBuf::from(name)
        };

        fs::write(&tmp, &body).map_err(CatalogError::Io)?;
        if let Err(e) = fs::rename(&tmp, path) {
            // Best-effort cleanup; swallow secondary errors.
            let _ = fs::remove_file(&tmp);
            return Err(CatalogError::Io(e));
        }
        Ok(())
    }
}

/// Newtype wrapping a curated model slug.
///
/// Allowed characters: ASCII `[a-z0-9._-]`, length `1..=64`, must not start
/// with `-` or `.` (to avoid hidden-file / flag-lookalike traps).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ModelSlug(String);

impl ModelSlug {
    /// Borrow the raw slug string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    fn parse(input: &str) -> Result<Self, ModelSlugError> {
        if input.is_empty() {
            return Err(ModelSlugError::Empty);
        }
        if input.len() > MODEL_SLUG_MAX_LEN {
            return Err(ModelSlugError::TooLong { len: input.len() });
        }
        if let Some(first) = input.chars().next() {
            if first == '-' || first == '.' {
                return Err(ModelSlugError::BadPrefix { ch: first });
            }
        }
        for ch in input.chars() {
            let ok = ch.is_ascii_lowercase()
                || ch.is_ascii_digit()
                || ch == '.'
                || ch == '_'
                || ch == '-';
            if !ok {
                return Err(ModelSlugError::InvalidChar { ch });
            }
        }
        Ok(Self(input.to_owned()))
    }
}

impl FromStr for ModelSlug {
    type Err = ModelSlugError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s)
    }
}

impl fmt::Display for ModelSlug {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl AsRef<str> for ModelSlug {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

/// Validation failures for [`ModelSlug::from_str`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelSlugError {
    /// Slug was the empty string.
    Empty,
    /// Slug exceeded the maximum length.
    TooLong {
        /// Actual length of the offending input.
        len: usize,
    },
    /// Slug contained a character outside `[a-z0-9._-]`.
    InvalidChar {
        /// Offending character.
        ch: char,
    },
    /// Slug started with `-` or `.`.
    BadPrefix {
        /// Offending leading character.
        ch: char,
    },
}

impl fmt::Display for ModelSlugError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => f.write_str("model slug is empty"),
            Self::TooLong { len } => {
                write!(f, "model slug too long ({len} > {MODEL_SLUG_MAX_LEN})")
            }
            Self::InvalidChar { ch } => {
                write!(f, "model slug contains invalid character {ch:?}")
            }
            Self::BadPrefix { ch } => {
                write!(f, "model slug must not start with {ch:?}")
            }
        }
    }
}

impl Error for ModelSlugError {}

/// Coarse model size / capability bucket the installer uses to recommend
/// candidates for a system tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelTier {
    /// Smallest models, low-RAM targets.
    Low,
    /// Mid-range.
    Medium,
    /// Large but mainstream.
    High,
    /// Extra-large flagship class.
    Xl,
}

/// Task labels an entry advertises support for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelTask {
    /// General chat / instruction following.
    Chat,
    /// Code generation / completion.
    Code,
    /// Sentence-embedding model.
    Embedding,
    /// Tool / function calling.
    ToolUse,
    /// Multimodal vision.
    Vision,
    /// "Caveman-ish" rewriter — internal Stratum role.
    Cavemanish,
    /// Polisher role — turns caveman output into the final answer.
    Polisher,
}

/// A single entry in the curated model catalog.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelEntry {
    /// Stable identifier (same value as the map key in [`ModelCatalog`]).
    pub slug: ModelSlug,
    /// Upstream model family / lineage (e.g. `"llama"`, `"qwen"`).
    pub family: String,
    /// Human-friendly name shown in the installer UI.
    pub display_name: String,
    /// Coarse capability bucket.
    pub tier: ModelTier,
    /// Tasks this model is curated for.
    pub task: BTreeSet<ModelTask>,
    /// Total on-disk artifact size, in MiB. Stored as `u64` (not `f64` GiB)
    /// so the struct can derive `Eq`.
    pub size_mib: u64,
    /// Quantization tag (e.g. `"Q4_K_M"`).
    pub quantization: String,
    /// Downloadable artifact reference.
    pub artifact: ArtifactRef,
    /// SPDX-ish license identifier.
    pub license: String,
    /// Optional homepage / model card URL.
    pub homepage: Option<String>,
    /// Optional companion mmproj (multimodal projection) sidecar
    /// artifact. Set on entries whose `task` set includes
    /// [`ModelTask::Vision`]; pairs the model's GGUF weights with a
    /// separate `mmproj-*.gguf` projector file that llama.cpp's `mtmd`
    /// interface needs to encode images. `None` for text-only models.
    ///
    /// Per `plan/05-multimodal.md`: Gemma 4 E4B + companion
    /// `mmproj-*.gguf` is the v1 vision pair; the URL + sha256 are
    /// pinned the same way as `artifact`. When this field is missing
    /// from a vision-tagged entry the installer should surface a clear
    /// "no projector available" error rather than silently letting the
    /// model load without vision.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vision_mmproj: Option<ArtifactRef>,
}

/// Reference to a downloadable model artifact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactRef {
    /// HTTPS download URL.
    pub url: String,
    /// Lowercase hex SHA-256 of the artifact (exactly 64 chars).
    pub sha256: String,
    /// Expected byte size of the artifact (`> 0`).
    pub bytes: u64,
}

impl ArtifactRef {
    /// Validate and construct an [`ArtifactRef`].
    ///
    /// # Errors
    /// - [`ArtifactRefError::EmptyUrl`] if `url` is empty.
    /// - [`ArtifactRefError::NonHttpsUrl`] if `url` does not start with
    ///   `https://` (the catalog ships over HTTPS only).
    /// - [`ArtifactRefError::BadSha256Length`] if `sha256` is not exactly
    ///   64 characters.
    /// - [`ArtifactRefError::BadSha256Hex`] if `sha256` contains a
    ///   non-lowercase-hex character.
    /// - [`ArtifactRefError::ZeroBytes`] if `bytes == 0`.
    pub fn new(url: String, sha256: String, bytes: u64) -> Result<Self, ArtifactRefError> {
        if url.is_empty() {
            return Err(ArtifactRefError::EmptyUrl);
        }
        if !url.starts_with("https://") {
            return Err(ArtifactRefError::NonHttpsUrl);
        }
        if sha256.len() != SHA256_HEX_LEN {
            return Err(ArtifactRefError::BadSha256Length {
                actual: sha256.len(),
            });
        }
        if !sha256.chars().all(|c| matches!(c, '0'..='9' | 'a'..='f')) {
            return Err(ArtifactRefError::BadSha256Hex);
        }
        if bytes == 0 {
            return Err(ArtifactRefError::ZeroBytes);
        }
        Ok(Self { url, sha256, bytes })
    }
}

/// Validation failures for [`ArtifactRef::new`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArtifactRefError {
    /// `sha256` did not have exactly 64 characters.
    BadSha256Length {
        /// Observed length.
        actual: usize,
    },
    /// `sha256` contained a character outside `[0-9a-f]`.
    BadSha256Hex,
    /// `url` was the empty string.
    EmptyUrl,
    /// `url` did not start with `https://`.
    NonHttpsUrl,
    /// `bytes` was zero.
    ZeroBytes,
}

impl fmt::Display for ArtifactRefError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadSha256Length { actual } => write!(
                f,
                "artifact sha256 must be {SHA256_HEX_LEN} hex chars, got {actual}",
            ),
            Self::BadSha256Hex => f.write_str("artifact sha256 must be lowercase hex [0-9a-f]"),
            Self::EmptyUrl => f.write_str("artifact url is empty"),
            Self::NonHttpsUrl => f.write_str("artifact url must start with https://"),
            Self::ZeroBytes => f.write_str("artifact bytes must be > 0"),
        }
    }
}

impl Error for ArtifactRefError {}

/// Top-level errors emitted by catalog load / save / validation.
#[derive(Debug)]
pub enum CatalogError {
    /// Filesystem failure.
    Io(std::io::Error),
    /// JSON serialization / deserialization failure.
    Serialize(serde_json::Error),
    /// Cross-entry validation failure (see [`ModelCatalog::validate`]).
    Validation(String),
    /// On-disk schema is newer than this binary supports.
    SchemaNewer {
        /// Schema version present in the file.
        found: u32,
        /// Highest version this build understands.
        supported: u32,
    },
}

impl fmt::Display for CatalogError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "catalog io error: {e}"),
            Self::Serialize(e) => write!(f, "catalog serialize error: {e}"),
            Self::Validation(msg) => write!(f, "catalog validation error: {msg}"),
            Self::SchemaNewer { found, supported } => write!(
                f,
                "catalog schema_version {found} is newer than supported {supported}",
            ),
        }
    }
}

impl Error for CatalogError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Serialize(e) => Some(e),
            Self::Validation(_) | Self::SchemaNewer { .. } => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn good_sha() -> String {
        "a".repeat(64)
    }

    fn sample_artifact() -> ArtifactRef {
        ArtifactRef::new("https://example.com/m.gguf".to_owned(), good_sha(), 1_024).unwrap()
    }

    fn entry(slug: &str, tier: ModelTier, size_mib: u64, tasks: &[ModelTask]) -> ModelEntry {
        ModelEntry {
            slug: slug.parse().unwrap(),
            family: "llama".to_owned(),
            display_name: format!("Display {slug}"),
            tier,
            task: tasks.iter().copied().collect(),
            size_mib,
            quantization: "Q4_K_M".to_owned(),
            artifact: sample_artifact(),
            license: "Apache-2.0".to_owned(),
            homepage: None,
            vision_mmproj: None,
        }
    }

    // ---- ModelSlug ----

    #[test]
    fn slug_happy_parses() {
        let s: ModelSlug = "llama-3.1_8b".parse().unwrap();
        assert_eq!(s.as_str(), "llama-3.1_8b");
        assert_eq!(format!("{s}"), "llama-3.1_8b");
        assert_eq!(<ModelSlug as AsRef<str>>::as_ref(&s), "llama-3.1_8b");
    }

    #[test]
    fn slug_rejects_empty() {
        let err = "".parse::<ModelSlug>().unwrap_err();
        assert_eq!(err, ModelSlugError::Empty);
        assert!(format!("{err}").contains("empty"));
    }

    #[test]
    fn slug_rejects_too_long() {
        let long = "a".repeat(65);
        let err = long.parse::<ModelSlug>().unwrap_err();
        assert_eq!(err, ModelSlugError::TooLong { len: 65 });
        assert!(format!("{err}").contains("too long"));
    }

    #[test]
    fn slug_rejects_invalid_char() {
        let err = "Llama".parse::<ModelSlug>().unwrap_err();
        assert!(matches!(err, ModelSlugError::InvalidChar { ch: 'L' }));
        assert!(format!("{err}").contains("invalid"));
    }

    #[test]
    fn slug_rejects_bad_prefix_dash() {
        let err = "-foo".parse::<ModelSlug>().unwrap_err();
        assert_eq!(err, ModelSlugError::BadPrefix { ch: '-' });
        assert!(format!("{err}").contains("start"));
    }

    #[test]
    fn slug_rejects_bad_prefix_dot() {
        let err = ".foo".parse::<ModelSlug>().unwrap_err();
        assert_eq!(err, ModelSlugError::BadPrefix { ch: '.' });
    }

    // ---- ArtifactRef ----

    #[test]
    fn artifact_happy() {
        let a = ArtifactRef::new("https://example.com/x".to_owned(), good_sha(), 42).unwrap();
        assert_eq!(a.bytes, 42);
        assert_eq!(a.sha256.len(), 64);
    }

    #[test]
    fn artifact_rejects_wrong_sha_len() {
        let err =
            ArtifactRef::new("https://x.test/y".to_owned(), "deadbeef".to_owned(), 1).unwrap_err();
        assert_eq!(err, ArtifactRefError::BadSha256Length { actual: 8 });
        assert!(format!("{err}").contains("64"));
    }

    #[test]
    fn artifact_rejects_non_hex_sha() {
        // 64 chars but contains 'g'.
        let mut sha = "a".repeat(63);
        sha.push('g');
        let err = ArtifactRef::new("https://x.test/y".to_owned(), sha, 1).unwrap_err();
        assert_eq!(err, ArtifactRefError::BadSha256Hex);
        assert!(format!("{err}").contains("hex"));
    }

    #[test]
    fn artifact_rejects_empty_url() {
        let err = ArtifactRef::new(String::new(), good_sha(), 1).unwrap_err();
        assert_eq!(err, ArtifactRefError::EmptyUrl);
    }

    #[test]
    fn artifact_rejects_non_https() {
        let err = ArtifactRef::new("http://example.com/x".to_owned(), good_sha(), 1).unwrap_err();
        assert_eq!(err, ArtifactRefError::NonHttpsUrl);
        assert!(format!("{err}").contains("https"));
    }

    #[test]
    fn artifact_rejects_zero_bytes() {
        let err = ArtifactRef::new("https://example.com/x".to_owned(), good_sha(), 0).unwrap_err();
        assert_eq!(err, ArtifactRefError::ZeroBytes);
    }

    // ---- Tier / Task serde ----

    #[test]
    fn tier_serde_snake_case() {
        let s = serde_json::to_string(&ModelTier::Xl).unwrap();
        assert_eq!(s, "\"xl\"");
        let back: ModelTier = serde_json::from_str("\"medium\"").unwrap();
        assert_eq!(back, ModelTier::Medium);
    }

    #[test]
    fn task_serde_snake_case() {
        let s = serde_json::to_string(&ModelTask::ToolUse).unwrap();
        assert_eq!(s, "\"tool_use\"");
        let back: ModelTask = serde_json::from_str("\"cavemanish\"").unwrap();
        assert_eq!(back, ModelTask::Cavemanish);
    }

    // ---- ModelCatalog ----

    #[test]
    fn insert_overwrites_and_returns_prior() {
        let mut cat = ModelCatalog::new();
        let a = entry("foo", ModelTier::Low, 100, &[ModelTask::Chat]);
        let b = entry("foo", ModelTier::High, 200, &[ModelTask::Code]);
        assert!(cat.insert(a.clone()).is_none());
        let prior = cat.insert(b.clone()).expect("prior");
        assert_eq!(prior, a);
        assert_eq!(cat.get(&"foo".parse().unwrap()), Some(&b));
    }

    #[test]
    fn get_unknown_returns_none() {
        let cat = ModelCatalog::new();
        assert!(cat.get(&"nope".parse().unwrap()).is_none());
    }

    #[test]
    fn filter_by_tier_sorted_by_slug() {
        let mut cat = ModelCatalog::new();
        cat.insert(entry("b", ModelTier::Low, 100, &[ModelTask::Chat]));
        cat.insert(entry("a", ModelTier::Low, 200, &[ModelTask::Chat]));
        cat.insert(entry("c", ModelTier::High, 300, &[ModelTask::Chat]));
        let got: Vec<_> = cat
            .filter_by_tier(ModelTier::Low)
            .into_iter()
            .map(|e| e.slug.as_str().to_owned())
            .collect();
        assert_eq!(got, vec!["a".to_owned(), "b".to_owned()]);
    }

    #[test]
    fn filter_by_task_sorted_by_slug() {
        let mut cat = ModelCatalog::new();
        cat.insert(entry("z", ModelTier::Low, 100, &[ModelTask::Code]));
        cat.insert(entry(
            "m",
            ModelTier::Low,
            200,
            &[ModelTask::Code, ModelTask::Chat],
        ));
        cat.insert(entry("a", ModelTier::High, 300, &[ModelTask::Vision]));
        let got: Vec<_> = cat
            .filter_by_task(ModelTask::Code)
            .into_iter()
            .map(|e| e.slug.as_str().to_owned())
            .collect();
        assert_eq!(got, vec!["m".to_owned(), "z".to_owned()]);
    }

    #[test]
    fn recommend_picks_smallest_size() {
        let mut cat = ModelCatalog::new();
        cat.insert(entry("big", ModelTier::Low, 9_000, &[ModelTask::Chat]));
        cat.insert(entry("small", ModelTier::Low, 100, &[ModelTask::Chat]));
        cat.insert(entry("mid", ModelTier::Medium, 500, &[ModelTask::Chat]));
        let pick = cat.recommend_for(ModelTier::High, ModelTask::Chat).unwrap();
        assert_eq!(pick.slug.as_str(), "small");
    }

    #[test]
    fn recommend_returns_none_when_nothing_fits() {
        let mut cat = ModelCatalog::new();
        cat.insert(entry("only", ModelTier::Xl, 1, &[ModelTask::Chat]));
        // Tier budget too small.
        assert!(cat.recommend_for(ModelTier::Low, ModelTask::Chat).is_none());
        // Task missing.
        assert!(cat
            .recommend_for(ModelTier::Xl, ModelTask::Vision)
            .is_none());
    }

    #[test]
    fn recommend_tie_breaks_on_slug() {
        let mut cat = ModelCatalog::new();
        cat.insert(entry("zzz", ModelTier::Low, 100, &[ModelTask::Chat]));
        cat.insert(entry("aaa", ModelTier::Low, 100, &[ModelTask::Chat]));
        let pick = cat.recommend_for(ModelTier::Low, ModelTask::Chat).unwrap();
        assert_eq!(pick.slug.as_str(), "aaa");
    }

    #[test]
    fn save_atomic_then_load_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("catalog.json");

        let mut cat = ModelCatalog::new();
        cat.insert(entry("a-model", ModelTier::Low, 100, &[ModelTask::Chat]));
        cat.insert(entry("b-model", ModelTier::Medium, 300, &[ModelTask::Code]));
        cat.insert(entry(
            "c-model",
            ModelTier::High,
            500,
            &[ModelTask::Chat, ModelTask::ToolUse],
        ));

        cat.save_atomic(&path).unwrap();
        // No .tmp residue.
        let tmp = dir.path().join("catalog.json.tmp");
        assert!(!tmp.exists());

        let loaded = ModelCatalog::load(&path).unwrap();
        assert_eq!(loaded, cat);
    }

    #[test]
    fn load_rejects_newer_schema() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("catalog.json");
        let body = serde_json::json!({
            "schema_version": 999,
            "entries": {}
        });
        fs::write(&path, serde_json::to_vec_pretty(&body).unwrap()).unwrap();
        let err = ModelCatalog::load(&path).unwrap_err();
        match err {
            CatalogError::SchemaNewer { found, supported } => {
                assert_eq!(found, 999);
                assert_eq!(supported, MODEL_CATALOG_SCHEMA_VERSION);
            }
            other => panic!("unexpected: {other}"),
        }
    }

    #[test]
    fn load_rejects_malformed_json() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("catalog.json");
        fs::write(&path, b"{not json").unwrap();
        let err = ModelCatalog::load(&path).unwrap_err();
        assert!(matches!(err, CatalogError::Serialize(_)));
        assert!(format!("{err}").contains("serialize"));
    }

    #[test]
    fn load_io_error_on_missing_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("missing.json");
        let err = ModelCatalog::load(&path).unwrap_err();
        assert!(matches!(err, CatalogError::Io(_)));
        // exercise the source() impl
        assert!(err.source().is_some());
    }

    #[test]
    fn validate_accepts_valid_catalog() {
        let mut cat = ModelCatalog::new();
        cat.insert(entry("foo", ModelTier::Low, 100, &[ModelTask::Chat]));
        cat.insert(entry("bar", ModelTier::High, 200, &[ModelTask::Code]));
        cat.validate().unwrap();
    }

    #[test]
    fn validate_rejects_mismatched_slug_key() {
        let mut cat = ModelCatalog::new();
        let mut e = entry("foo", ModelTier::Low, 100, &[ModelTask::Chat]);
        // Insert under the "real" slug, then mutate the stored entry's slug
        // to forge a mismatch.
        cat.insert(e.clone());
        e.slug = "different".parse().unwrap();
        cat.entries.insert("foo".parse().unwrap(), e);
        let err = cat.validate().unwrap_err();
        assert!(matches!(err, CatalogError::Validation(_)));
        assert!(format!("{err}").contains("mismatch"));
    }

    #[test]
    fn validate_rejects_zero_size_mib() {
        let mut cat = ModelCatalog::new();
        let mut e = entry("foo", ModelTier::Low, 100, &[ModelTask::Chat]);
        e.size_mib = 0;
        // Skip the constructor's invariant check by inserting directly.
        cat.entries.insert(e.slug.clone(), e);
        let err = cat.validate().unwrap_err();
        assert!(format!("{err}").contains("zero size_mib"));
    }

    #[test]
    fn validate_rejects_empty_family() {
        let mut cat = ModelCatalog::new();
        let mut e = entry("foo", ModelTier::Low, 100, &[ModelTask::Chat]);
        e.family = "   ".to_owned();
        cat.entries.insert(e.slug.clone(), e);
        let err = cat.validate().unwrap_err();
        assert!(format!("{err}").contains("empty family"));
    }

    #[test]
    fn validate_rejects_bad_artifact() {
        let mut cat = ModelCatalog::new();
        let mut e = entry("foo", ModelTier::Low, 100, &[ModelTask::Chat]);
        e.artifact.bytes = 0;
        cat.entries.insert(e.slug.clone(), e);
        let err = cat.validate().unwrap_err();
        assert!(format!("{err}").contains("artifact invalid"));
    }

    // ---- CatalogError::Display smoke ----

    #[test]
    fn catalog_error_display_smoke() {
        let io = CatalogError::Io(std::io::Error::other("boom"));
        assert!(format!("{io}").contains("io"));
        // Serialize variant — fabricate via a parse failure.
        let serialize_err = serde_json::from_str::<ModelCatalog>("{")
            .map_err(CatalogError::Serialize)
            .unwrap_err();
        assert!(format!("{serialize_err}").contains("serialize"));
        let v = CatalogError::Validation("nope".to_owned());
        assert!(format!("{v}").contains("validation"));
        let s = CatalogError::SchemaNewer {
            found: 9,
            supported: 1,
        };
        assert!(format!("{s}").contains("newer"));
    }

    // ---- catalog serde round-trip with multiple entries ----

    #[test]
    fn catalog_serde_round_trip_three_entries() {
        let mut cat = ModelCatalog::new();
        cat.insert(entry("alpha", ModelTier::Low, 100, &[ModelTask::Chat]));
        cat.insert(entry(
            "beta",
            ModelTier::Medium,
            200,
            &[ModelTask::Code, ModelTask::ToolUse],
        ));
        cat.insert(entry(
            "gamma",
            ModelTier::Xl,
            900,
            &[ModelTask::Vision, ModelTask::Polisher],
        ));
        let json = serde_json::to_string_pretty(&cat).unwrap();
        let back: ModelCatalog = serde_json::from_str(&json).unwrap();
        assert_eq!(back, cat);
        assert_eq!(back.entries.len(), 3);
    }

    // ---- ModelSlug serde transparent ----

    #[test]
    fn slug_serde_transparent() {
        let s: ModelSlug = "ok-slug".parse().unwrap();
        let j = serde_json::to_string(&s).unwrap();
        assert_eq!(j, "\"ok-slug\"");
    }
}
