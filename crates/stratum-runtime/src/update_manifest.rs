//! Self-update channel manifest data shape.
//!
//! Phase 1 scaffold for `stratum self-update` (per
//! `plan/27-self-update-and-migrations.md` §2). This module pins the on-disk
//! shape of the channel manifest the updater consumes — the actual download,
//! signature-verify, and atomic-swap pipeline lands in a later phase.
//!
//! ## Scope
//!
//! - Pure data shape: channel, release entry, semver-ish release version,
//!   binary artifact reference, platform tag.
//! - In-process decision (`evaluate`) for "given a current version and a
//!   manifest, what should the updater do?".
//! - Atomic save / load with `.tmp` rename. No schema migration ladder yet —
//!   only the schema-newer rejection check.
//! - Validation: every release-version unique, history sorted ascending,
//!   `latest` mirrors a history entry, every `ArtifactRef` revalidates.
//!
//! ## Channels
//!
//! Three channels per the project memory note: `stable`, `beta`, `nightly`.
//! Encoded as `snake_case` on the wire.
//!
//! ## What is intentionally NOT here
//!
//! - HTTP fetch of the manifest URL.
//! - Code-signing / minisign / Ed25519 signature verification on the
//!   downloaded binary.
//! - Rollback bookkeeping (lives in the install module).
//! - Migration ladder for `schema_version` < `UPDATE_MANIFEST_SCHEMA_VERSION`
//!   — there is no older version yet to migrate from.

use std::cmp::Ordering;
use std::fmt;
use std::hash::Hash;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use serde::{Deserialize, Serialize};

/// Current update-manifest schema version.
///
/// Bumps within a major release are additive only; a manifest declaring a
/// higher value than this is refused (downgrade is unsupported).
pub const UPDATE_MANIFEST_SCHEMA_VERSION: u32 = 1;

// ---------------------------------------------------------------------------
// UpdateChannel
// ---------------------------------------------------------------------------

/// Release channel a manifest publishes against.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UpdateChannel {
    /// Stable release line — slowest moving, most-tested.
    Stable,
    /// Beta line — release candidates and previews.
    Beta,
    /// Nightly line — automated builds off `main`.
    Nightly,
}

// ---------------------------------------------------------------------------
// PlatformTag
// ---------------------------------------------------------------------------

/// Wire-encoded target platform of a binary artifact.
///
/// The exact `snake_case` forms are part of the contract — they appear in
/// signed manifests served over HTTPS — so the upper-case-acronym lint is
/// suppressed with a `reason` on the enum itself.
#[allow(clippy::upper_case_acronyms, reason = "exact wire encoding")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlatformTag {
    /// macOS on Apple Silicon (`aarch64-apple-darwin`).
    MacOsAarch64,
    /// macOS on `x86_64` (`x86_64-apple-darwin`).
    MacOsX86_64,
    /// Linux on aarch64 (`aarch64-unknown-linux-gnu`).
    LinuxAarch64,
    /// Linux on `x86_64` (`x86_64-unknown-linux-gnu`).
    LinuxX86_64,
    /// Windows on `x86_64` (`x86_64-pc-windows-msvc`).
    WindowsX86_64,
}

// ---------------------------------------------------------------------------
// ReleaseVersion
// ---------------------------------------------------------------------------

/// Parsed semver-ish release version.
///
/// Custom ordering: numeric compare on `(major, minor, patch)`, then a
/// pre-release sorts BEFORE the corresponding release (`SemVer` 2 §11).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ReleaseVersion {
    /// Major version component.
    pub major: u16,
    /// Minor version component.
    pub minor: u16,
    /// Patch version component.
    pub patch: u16,
    /// Optional pre-release tag (everything after `-`). `None` for a release.
    pub pre: Option<String>,
}

impl ReleaseVersion {
    /// Construct a release version from its parts. Convenience for tests and
    /// for callers that already have the numbers.
    #[must_use]
    pub const fn new(major: u16, minor: u16, patch: u16, pre: Option<String>) -> Self {
        Self {
            major,
            minor,
            patch,
            pre,
        }
    }

    /// Parse a `MAJOR.MINOR.PATCH[-PRE]` string.
    ///
    /// Rejections:
    /// - Wrong segment count (`1.0` or `1.0.0.0`).
    /// - Non-numeric components (`1.4.x`).
    /// - Leading zeros (`1.04.0`).
    /// - Empty pre after `-` (`1.0.0-`).
    /// - Pre containing whitespace.
    ///
    /// # Errors
    /// Returns [`ReleaseVersionError`] with a short reason string.
    pub fn parse(s: &str) -> Result<Self, ReleaseVersionError> {
        // Split off optional pre after the first `-`.
        let (core, pre) = if let Some((c, p)) = s.split_once('-') {
            (c, Some(p))
        } else {
            (s, None)
        };
        let parts: Vec<&str> = core.split('.').collect();
        if parts.len() != 3 {
            return Err(ReleaseVersionError::Format(format!(
                "expected MAJOR.MINOR.PATCH, got {core:?}"
            )));
        }
        let parse_part = |label: &str, raw: &str| -> Result<u16, ReleaseVersionError> {
            if raw.is_empty() {
                return Err(ReleaseVersionError::BadInteger(format!("{label} is empty")));
            }
            if raw.len() > 1 && raw.starts_with('0') {
                return Err(ReleaseVersionError::BadInteger(format!(
                    "{label} has leading zero: {raw}"
                )));
            }
            raw.parse::<u16>()
                .map_err(|e| ReleaseVersionError::BadInteger(format!("{label}: {e}")))
        };
        let major = parse_part("major", parts[0])?;
        let minor = parse_part("minor", parts[1])?;
        let patch = parse_part("patch", parts[2])?;

        let pre = match pre {
            None => None,
            Some(p) => {
                if p.is_empty() {
                    return Err(ReleaseVersionError::BadPre("empty pre-release".into()));
                }
                if p.chars().any(char::is_whitespace) {
                    return Err(ReleaseVersionError::BadPre(format!(
                        "whitespace in pre: {p:?}"
                    )));
                }
                Some(p.to_owned())
            }
        };
        Ok(Self {
            major,
            minor,
            patch,
            pre,
        })
    }
}

impl fmt::Display for ReleaseVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)?;
        if let Some(pre) = &self.pre {
            write!(f, "-{pre}")?;
        }
        Ok(())
    }
}

impl Ord for ReleaseVersion {
    fn cmp(&self, other: &Self) -> Ordering {
        let numeric =
            (self.major, self.minor, self.patch).cmp(&(other.major, other.minor, other.patch));
        if numeric != Ordering::Equal {
            return numeric;
        }
        // SemVer §11: a pre-release sorts BEFORE the corresponding release.
        match (&self.pre, &other.pre) {
            (None, None) => Ordering::Equal,
            (None, Some(_)) => Ordering::Greater,
            (Some(_), None) => Ordering::Less,
            (Some(a), Some(b)) => a.cmp(b),
        }
    }
}

impl PartialOrd for ReleaseVersion {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Errors produced by [`ReleaseVersion::parse`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReleaseVersionError {
    /// Wrong overall shape (segment count, separator, etc.).
    Format(String),
    /// A numeric component failed integer parsing or had a leading zero.
    BadInteger(String),
    /// The pre-release tag was empty or contained whitespace.
    BadPre(String),
}

impl fmt::Display for ReleaseVersionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Format(s) => write!(f, "release version format: {s}"),
            Self::BadInteger(s) => write!(f, "release version integer: {s}"),
            Self::BadPre(s) => write!(f, "release version pre-release: {s}"),
        }
    }
}

impl std::error::Error for ReleaseVersionError {}

// ---------------------------------------------------------------------------
// ArtifactRef
// ---------------------------------------------------------------------------

/// Reference to a downloadable build artifact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactRef {
    /// HTTPS URL the binary is fetched from.
    pub url: String,
    /// Lower-case hex SHA-256 of the binary (64 chars).
    pub sha256: String,
    /// Byte size of the binary. Must be non-zero.
    pub bytes: u64,
    /// Platform this artifact targets.
    pub platform: PlatformTag,
}

impl ArtifactRef {
    /// Construct + validate an [`ArtifactRef`].
    ///
    /// # Errors
    /// Returns [`ArtifactRefError`] for any of the validation rules:
    /// empty url, non-https url, wrong-length sha256, non-hex sha256, or
    /// zero byte count.
    pub fn new(
        url: impl Into<String>,
        sha256: impl Into<String>,
        bytes: u64,
        platform: PlatformTag,
    ) -> Result<Self, ArtifactRefError> {
        let me = Self {
            url: url.into(),
            sha256: sha256.into(),
            bytes,
            platform,
        };
        me.validate()?;
        Ok(me)
    }

    /// Re-validate an [`ArtifactRef`] (used by [`UpdateManifest::validate`]
    /// after deserialization).
    ///
    /// # Errors
    /// Same rules as [`Self::new`].
    pub fn validate(&self) -> Result<(), ArtifactRefError> {
        if self.url.is_empty() {
            return Err(ArtifactRefError::EmptyUrl);
        }
        if !self.url.starts_with("https://") {
            return Err(ArtifactRefError::NonHttpsUrl);
        }
        if self.sha256.len() != 64 {
            return Err(ArtifactRefError::BadSha256Length {
                actual: self.sha256.len(),
            });
        }
        if !self
            .sha256
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        {
            return Err(ArtifactRefError::BadSha256Hex);
        }
        if self.bytes == 0 {
            return Err(ArtifactRefError::ZeroBytes);
        }
        Ok(())
    }
}

/// Errors produced when constructing or revalidating an [`ArtifactRef`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArtifactRefError {
    /// SHA-256 hex string was not exactly 64 characters.
    BadSha256Length {
        /// Actual length the caller supplied.
        actual: usize,
    },
    /// SHA-256 string contained a non-lower-hex character.
    BadSha256Hex,
    /// URL was empty.
    EmptyUrl,
    /// URL did not begin with `https://`.
    NonHttpsUrl,
    /// Byte count was zero.
    ZeroBytes,
}

impl fmt::Display for ArtifactRefError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadSha256Length { actual } => {
                write!(f, "sha256 must be 64 hex chars, got {actual}")
            }
            Self::BadSha256Hex => f.write_str("sha256 must be lower-case hex"),
            Self::EmptyUrl => f.write_str("url is empty"),
            Self::NonHttpsUrl => f.write_str("url must be https://"),
            Self::ZeroBytes => f.write_str("artifact byte count is zero"),
        }
    }
}

impl std::error::Error for ArtifactRefError {}

// ---------------------------------------------------------------------------
// ReleaseEntry
// ---------------------------------------------------------------------------

/// One published release on a channel.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReleaseEntry {
    /// Release version.
    pub version: ReleaseVersion,
    /// Wall-clock release timestamp.
    pub released_at: SystemTime,
    /// Binary artifact for this release (one per platform — callers pick the
    /// matching artifact from the `latest` entry's siblings if multi-platform
    /// fan-out is needed).
    pub binary: ArtifactRef,
    /// Optional floor: the oldest version that can upgrade INTO this release.
    /// `None` means any older version may upgrade.
    pub min_supported_from: Option<ReleaseVersion>,
    /// HTTPS URL to the human-readable release notes.
    pub release_notes_url: String,
}

// ---------------------------------------------------------------------------
// UpdateDecision
// ---------------------------------------------------------------------------

/// The updater's decision for a given (manifest, current version) pair.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum UpdateDecision {
    /// Nothing to do — already at or above the latest version on the channel.
    UpToDate,
    /// A newer release is available and the current build is allowed to take
    /// the jump.
    Upgrade {
        /// Currently-running release version.
        from: ReleaseVersion,
        /// Target release version published as `latest`.
        to: ReleaseVersion,
    },
    /// A newer release exists, but `current < latest.min_supported_from`.
    /// The user must take an intermediate hop first.
    BlockedSchemaTooOld {
        /// Currently-running release version.
        current: ReleaseVersion,
        /// `latest.min_supported_from` — the oldest version permitted to
        /// upgrade directly.
        min_supported: ReleaseVersion,
    },
}

/// Decide what the updater should do given the manifest and the running
/// version.
///
/// Rules:
/// 1. `latest == current` → `UpToDate`.
/// 2. `latest > current` and `current >= latest.min_supported_from` (or
///    `min_supported_from` is `None`) → `Upgrade`.
/// 3. `latest > current` and `current < latest.min_supported_from` →
///    `BlockedSchemaTooOld`.
/// 4. `latest < current` → `UpToDate` (the server hasn't caught up; never
///    downgrade automatically).
#[must_use]
pub fn evaluate(manifest: &UpdateManifest, current: &ReleaseVersion) -> UpdateDecision {
    let latest = &manifest.latest.version;
    match latest.cmp(current) {
        Ordering::Equal | Ordering::Less => UpdateDecision::UpToDate,
        Ordering::Greater => {
            if let Some(floor) = &manifest.latest.min_supported_from {
                if current < floor {
                    return UpdateDecision::BlockedSchemaTooOld {
                        current: current.clone(),
                        min_supported: floor.clone(),
                    };
                }
            }
            UpdateDecision::Upgrade {
                from: current.clone(),
                to: latest.clone(),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// UpdateManifest
// ---------------------------------------------------------------------------

/// Channel manifest the updater consumes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UpdateManifest {
    /// Schema version of this manifest.
    pub schema_version: u32,
    /// Channel this manifest publishes against.
    pub channel: UpdateChannel,
    /// Currently-advertised latest release on the channel. Must match a
    /// member of `history`.
    pub latest: ReleaseEntry,
    /// Full release history, sorted ascending by version.
    pub history: Vec<ReleaseEntry>,
}

impl UpdateManifest {
    /// Return the [`ArtifactRef`] for `platform` from `self.latest`, if its
    /// binary targets that platform.
    ///
    /// The scaffold deliberately models one binary per `ReleaseEntry`; richer
    /// per-release fan-out lands later. Callers that need cross-platform
    /// matrices today should publish one manifest per platform.
    #[must_use]
    pub fn pick_artifact(&self, platform: PlatformTag) -> Option<&ArtifactRef> {
        if self.latest.binary.platform == platform {
            Some(&self.latest.binary)
        } else {
            None
        }
    }

    /// Validate cross-field invariants. See [`ManifestError::Validation`].
    ///
    /// # Errors
    /// Returns [`ManifestError::Validation`] on any rule violation, or
    /// `ManifestError` derived from a per-`ArtifactRef` failure.
    pub fn validate(&self) -> Result<(), ManifestError> {
        // 1. History sorted ascending and unique.
        for window in self.history.windows(2) {
            let (a, b) = (&window[0].version, &window[1].version);
            match a.cmp(b) {
                Ordering::Less => {}
                Ordering::Equal => {
                    return Err(ManifestError::Validation(format!(
                        "duplicate version in history: {a}"
                    )));
                }
                Ordering::Greater => {
                    return Err(ManifestError::Validation(format!(
                        "history not sorted ascending: {a} > {b}"
                    )));
                }
            }
        }
        // 2. `latest` matches a history entry.
        if !self.history.iter().any(|e| e == &self.latest) {
            return Err(ManifestError::Validation(format!(
                "latest version {} not present in history",
                self.latest.version
            )));
        }
        // 3. Every artifact revalidates.
        self.latest
            .binary
            .validate()
            .map_err(|e| ManifestError::Validation(format!("latest artifact: {e}")))?;
        for entry in &self.history {
            entry
                .binary
                .validate()
                .map_err(|e| ManifestError::Validation(format!("history artifact: {e}")))?;
        }
        Ok(())
    }

    /// Load + parse a manifest from `path`.
    ///
    /// # Errors
    /// Returns [`ManifestError::Io`] on IO failure, [`ManifestError::Serialize`]
    /// on JSON parse failure, and [`ManifestError::SchemaNewer`] if the
    /// on-disk `schema_version` exceeds [`UPDATE_MANIFEST_SCHEMA_VERSION`].
    pub fn load(path: &Path) -> Result<Self, ManifestError> {
        let raw = std::fs::read_to_string(path).map_err(ManifestError::Io)?;
        let parsed: Self = serde_json::from_str(&raw).map_err(ManifestError::Serialize)?;
        if parsed.schema_version > UPDATE_MANIFEST_SCHEMA_VERSION {
            return Err(ManifestError::SchemaNewer {
                found: parsed.schema_version,
                supported: UPDATE_MANIFEST_SCHEMA_VERSION,
            });
        }
        Ok(parsed)
    }

    /// Serialize + write to `path` via a sibling `.tmp` and atomic rename.
    ///
    /// Minimal reimplementation — does NOT chain into `install::save_atomic`,
    /// per the task brief.
    ///
    /// # Errors
    /// Returns [`ManifestError::Io`] for filesystem failures and
    /// [`ManifestError::Serialize`] for serialization failures.
    pub fn save_atomic(&self, path: &Path) -> Result<(), ManifestError> {
        let parent = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .ok_or_else(|| ManifestError::Validation("manifest path has no parent".into()))?;
        std::fs::create_dir_all(parent).map_err(ManifestError::Io)?;
        let rendered = serde_json::to_string_pretty(self).map_err(ManifestError::Serialize)?;
        let tmp = tmp_sibling(path)?;
        {
            use std::io::Write;
            let mut f = std::fs::OpenOptions::new()
                .create(true)
                .truncate(true)
                .write(true)
                .open(&tmp)
                .map_err(ManifestError::Io)?;
            f.write_all(rendered.as_bytes())
                .map_err(ManifestError::Io)?;
            f.sync_all().map_err(ManifestError::Io)?;
        }
        if let Err(e) = std::fs::rename(&tmp, path) {
            let _ = std::fs::remove_file(&tmp);
            return Err(ManifestError::Io(e));
        }
        Ok(())
    }
}

fn tmp_sibling(path: &Path) -> Result<PathBuf, ManifestError> {
    let name = path
        .file_name()
        .ok_or_else(|| ManifestError::Validation("manifest path has no file name".into()))?;
    let mut new_name = name.to_os_string();
    new_name.push(".tmp");
    Ok(path.with_file_name(new_name))
}

/// Errors emitted by [`UpdateManifest::load`] / [`UpdateManifest::save_atomic`]
/// / [`UpdateManifest::validate`].
#[derive(Debug)]
pub enum ManifestError {
    /// Underlying [`std::io::Error`].
    Io(std::io::Error),
    /// JSON serialization or deserialization failure.
    Serialize(serde_json::Error),
    /// A cross-field invariant was violated.
    Validation(String),
    /// On-disk `schema_version` is newer than this build supports.
    SchemaNewer {
        /// Version found on disk.
        found: u32,
        /// Highest version this build understands.
        supported: u32,
    },
}

impl fmt::Display for ManifestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "manifest io error: {e}"),
            Self::Serialize(e) => write!(f, "manifest serialize error: {e}"),
            Self::Validation(s) => write!(f, "manifest validation: {s}"),
            Self::SchemaNewer { found, supported } => write!(
                f,
                "manifest schema_version {found} is newer than supported {supported}",
            ),
        }
    }
}

impl std::error::Error for ManifestError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Serialize(e) => Some(e),
            Self::Validation(_) | Self::SchemaNewer { .. } => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::time::{Duration, UNIX_EPOCH};

    use tempfile::TempDir;

    use super::*;

    fn v(major: u16, minor: u16, patch: u16) -> ReleaseVersion {
        ReleaseVersion::new(major, minor, patch, None)
    }

    fn v_pre(major: u16, minor: u16, patch: u16, pre: &str) -> ReleaseVersion {
        ReleaseVersion::new(major, minor, patch, Some(pre.to_owned()))
    }

    fn good_artifact(platform: PlatformTag) -> ArtifactRef {
        ArtifactRef::new(
            "https://dl.stratum.dev/v1.0.0/stratum-macos-aarch64.tar.gz",
            "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            1024,
            platform,
        )
        .unwrap()
    }

    fn release(version: ReleaseVersion, platform: PlatformTag) -> ReleaseEntry {
        ReleaseEntry {
            version,
            released_at: UNIX_EPOCH + Duration::from_secs(1_700_000_000),
            binary: good_artifact(platform),
            min_supported_from: None,
            release_notes_url: "https://stratum.dev/releases/1.0.0".into(),
        }
    }

    fn manifest_with(latest: ReleaseEntry, history: Vec<ReleaseEntry>) -> UpdateManifest {
        UpdateManifest {
            schema_version: UPDATE_MANIFEST_SCHEMA_VERSION,
            channel: UpdateChannel::Stable,
            latest,
            history,
        }
    }

    // ---- ReleaseVersion::parse ----

    #[test]
    fn parse_happy_release() {
        let parsed = ReleaseVersion::parse("1.0.0").unwrap();
        assert_eq!(parsed, v(1, 0, 0));
    }

    #[test]
    fn parse_happy_with_pre() {
        let parsed = ReleaseVersion::parse("1.4.7-beta.3").unwrap();
        assert_eq!(parsed, v_pre(1, 4, 7, "beta.3"));
    }

    #[test]
    fn parse_rejects_too_few_segments() {
        let err = ReleaseVersion::parse("1.0").unwrap_err();
        assert!(matches!(err, ReleaseVersionError::Format(_)));
    }

    #[test]
    fn parse_rejects_non_numeric() {
        let err = ReleaseVersion::parse("1.4.x").unwrap_err();
        assert!(matches!(err, ReleaseVersionError::BadInteger(_)));
    }

    #[test]
    fn parse_rejects_leading_zero() {
        let err = ReleaseVersion::parse("1.04.0").unwrap_err();
        assert!(
            matches!(&err, ReleaseVersionError::BadInteger(msg) if msg.contains("leading zero"))
        );
    }

    #[test]
    fn parse_rejects_empty_pre() {
        let err = ReleaseVersion::parse("1.0.0-").unwrap_err();
        assert!(matches!(err, ReleaseVersionError::BadPre(_)));
    }

    #[test]
    fn parse_rejects_pre_with_space() {
        let err = ReleaseVersion::parse("1.0.0-bad pre").unwrap_err();
        assert!(matches!(err, ReleaseVersionError::BadPre(_)));
    }

    #[test]
    fn parse_rejects_too_many_segments() {
        let err = ReleaseVersion::parse("1.0.0.0").unwrap_err();
        assert!(matches!(err, ReleaseVersionError::Format(_)));
    }

    #[test]
    fn parse_rejects_empty_segment() {
        let err = ReleaseVersion::parse("1..0").unwrap_err();
        assert!(matches!(err, ReleaseVersionError::BadInteger(_)));
    }

    // ---- Display + Ord ----

    #[test]
    fn display_roundtrip_without_pre() {
        let s = v(1, 4, 7).to_string();
        assert_eq!(s, "1.4.7");
        assert_eq!(ReleaseVersion::parse(&s).unwrap(), v(1, 4, 7));
    }

    #[test]
    fn display_roundtrip_with_pre() {
        let s = v_pre(1, 4, 7, "beta.3").to_string();
        assert_eq!(s, "1.4.7-beta.3");
        assert_eq!(ReleaseVersion::parse(&s).unwrap(), v_pre(1, 4, 7, "beta.3"));
    }

    #[test]
    fn ord_numeric_simple() {
        assert!(v(1, 0, 0) > v(0, 9, 9));
        assert!(v(2, 0, 0) > v(1, 99, 99));
    }

    #[test]
    fn ord_release_beats_pre_release() {
        // 1.0.0 > 1.0.0-beta.1 (the pre sorts earlier).
        assert!(v(1, 0, 0) > v_pre(1, 0, 0, "beta.1"));
    }

    #[test]
    fn ord_pre_lexical_within_same_release() {
        assert!(v_pre(1, 0, 0, "beta.1") < v_pre(1, 0, 0, "beta.2"));
    }

    #[test]
    fn ord_partial_cmp_matches_cmp() {
        let a = v(1, 0, 0);
        let b = v(1, 0, 1);
        assert_eq!(a.partial_cmp(&b), Some(Ordering::Less));
    }

    // ---- Channel + Platform serde ----

    #[test]
    fn update_channel_serde_snake_case() {
        assert_eq!(
            serde_json::to_string(&UpdateChannel::Stable).unwrap(),
            "\"stable\""
        );
        assert_eq!(
            serde_json::to_string(&UpdateChannel::Beta).unwrap(),
            "\"beta\""
        );
        assert_eq!(
            serde_json::to_string(&UpdateChannel::Nightly).unwrap(),
            "\"nightly\""
        );
    }

    #[test]
    fn platform_tag_serde_snake_case_literal() {
        // Exact wire forms — these are part of the manifest contract.
        assert_eq!(
            serde_json::to_string(&PlatformTag::MacOsAarch64).unwrap(),
            "\"mac_os_aarch64\""
        );
        assert_eq!(
            serde_json::to_string(&PlatformTag::MacOsX86_64).unwrap(),
            "\"mac_os_x86_64\""
        );
        assert_eq!(
            serde_json::to_string(&PlatformTag::LinuxAarch64).unwrap(),
            "\"linux_aarch64\""
        );
        assert_eq!(
            serde_json::to_string(&PlatformTag::LinuxX86_64).unwrap(),
            "\"linux_x86_64\""
        );
        assert_eq!(
            serde_json::to_string(&PlatformTag::WindowsX86_64).unwrap(),
            "\"windows_x86_64\""
        );
    }

    #[test]
    fn platform_tag_roundtrip() {
        for p in [
            PlatformTag::MacOsAarch64,
            PlatformTag::MacOsX86_64,
            PlatformTag::LinuxAarch64,
            PlatformTag::LinuxX86_64,
            PlatformTag::WindowsX86_64,
        ] {
            let s = serde_json::to_string(&p).unwrap();
            let back: PlatformTag = serde_json::from_str(&s).unwrap();
            assert_eq!(p, back);
        }
    }

    // ---- ArtifactRef ----

    #[test]
    fn artifact_ref_happy() {
        let a = good_artifact(PlatformTag::MacOsAarch64);
        assert_eq!(a.bytes, 1024);
    }

    #[test]
    fn artifact_ref_rejects_empty_url() {
        let err = ArtifactRef::new("", "0".repeat(64), 1, PlatformTag::LinuxX86_64).unwrap_err();
        assert_eq!(err, ArtifactRefError::EmptyUrl);
    }

    #[test]
    fn artifact_ref_rejects_non_https() {
        let err = ArtifactRef::new(
            "http://example/x",
            "0".repeat(64),
            1,
            PlatformTag::LinuxX86_64,
        )
        .unwrap_err();
        assert_eq!(err, ArtifactRefError::NonHttpsUrl);
    }

    #[test]
    fn artifact_ref_rejects_short_sha() {
        let err =
            ArtifactRef::new("https://x/y", "deadbeef", 1, PlatformTag::LinuxX86_64).unwrap_err();
        assert_eq!(err, ArtifactRefError::BadSha256Length { actual: 8 });
    }

    #[test]
    fn artifact_ref_rejects_non_hex_sha() {
        // 64 chars but with an uppercase / non-hex char.
        let bad = format!("{}Z", "0".repeat(63));
        let err = ArtifactRef::new("https://x/y", bad, 1, PlatformTag::LinuxX86_64).unwrap_err();
        assert_eq!(err, ArtifactRefError::BadSha256Hex);
    }

    #[test]
    fn artifact_ref_rejects_zero_bytes() {
        let err = ArtifactRef::new("https://x/y", "0".repeat(64), 0, PlatformTag::LinuxX86_64)
            .unwrap_err();
        assert_eq!(err, ArtifactRefError::ZeroBytes);
    }

    // ---- evaluate ----

    fn three_release_manifest() -> UpdateManifest {
        let h = vec![
            release(v(0, 9, 0), PlatformTag::MacOsAarch64),
            release(v(0, 9, 9), PlatformTag::MacOsAarch64),
            release(v(1, 0, 0), PlatformTag::MacOsAarch64),
        ];
        manifest_with(h[2].clone(), h)
    }

    #[test]
    fn evaluate_up_to_date_on_match() {
        let m = three_release_manifest();
        let d = evaluate(&m, &v(1, 0, 0));
        assert_eq!(d, UpdateDecision::UpToDate);
    }

    #[test]
    fn evaluate_upgrade_on_newer_latest() {
        let m = three_release_manifest();
        let d = evaluate(&m, &v(0, 9, 0));
        assert_eq!(
            d,
            UpdateDecision::Upgrade {
                from: v(0, 9, 0),
                to: v(1, 0, 0),
            }
        );
    }

    #[test]
    fn evaluate_blocked_when_below_min_supported() {
        let mut m = three_release_manifest();
        m.latest.min_supported_from = Some(v(0, 9, 9));
        // Also keep history's matching entry consistent.
        m.history[2].min_supported_from = Some(v(0, 9, 9));
        let d = evaluate(&m, &v(0, 9, 0));
        assert_eq!(
            d,
            UpdateDecision::BlockedSchemaTooOld {
                current: v(0, 9, 0),
                min_supported: v(0, 9, 9),
            }
        );
    }

    #[test]
    fn evaluate_upgrade_with_satisfied_min_supported() {
        // Exercise the `if let Some(floor)` path where `current >= floor`,
        // falling through to `Upgrade` (covers the close-brace path of the
        // inner `if let` block, not just the `BlockedSchemaTooOld` return).
        let mut m = three_release_manifest();
        m.latest.min_supported_from = Some(v(0, 9, 0));
        m.history[2].min_supported_from = Some(v(0, 9, 0));
        let d = evaluate(&m, &v(0, 9, 9));
        assert_eq!(
            d,
            UpdateDecision::Upgrade {
                from: v(0, 9, 9),
                to: v(1, 0, 0),
            }
        );
    }

    #[test]
    fn evaluate_no_downgrade() {
        // Current is newer than latest (e.g. user installed a nightly past
        // the stable channel head). Updater stays put.
        let m = three_release_manifest();
        let d = evaluate(&m, &v(1, 5, 0));
        assert_eq!(d, UpdateDecision::UpToDate);
    }

    // ---- pick_artifact ----

    #[test]
    fn pick_artifact_match() {
        let m = three_release_manifest();
        let a = m.pick_artifact(PlatformTag::MacOsAarch64).unwrap();
        assert_eq!(a.platform, PlatformTag::MacOsAarch64);
    }

    #[test]
    fn pick_artifact_no_match() {
        let m = three_release_manifest();
        assert!(m.pick_artifact(PlatformTag::WindowsX86_64).is_none());
    }

    // ---- save_atomic + load ----

    #[test]
    fn save_then_load_roundtrip() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("sub").join("manifest.json");
        let m = three_release_manifest();
        m.save_atomic(&path).unwrap();
        let back = UpdateManifest::load(&path).unwrap();
        assert_eq!(m, back);
        // No .tmp residue.
        let tmp_sibling_path = path.with_file_name("manifest.json.tmp");
        assert!(!tmp_sibling_path.exists());
    }

    #[test]
    fn load_rejects_newer_schema() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("manifest.json");
        let m = three_release_manifest();
        m.save_atomic(&path).unwrap();
        let raw = std::fs::read_to_string(&path).unwrap();
        let bumped = raw.replace("\"schema_version\": 1", "\"schema_version\": 999");
        std::fs::write(&path, bumped).unwrap();
        let err = UpdateManifest::load(&path).unwrap_err();
        assert!(matches!(
            &err,
            ManifestError::SchemaNewer { found: 999, supported }
                if *supported == UPDATE_MANIFEST_SCHEMA_VERSION
        ));
    }

    #[test]
    fn load_io_error_for_missing_file() {
        let tmp = TempDir::new().unwrap();
        let missing = tmp.path().join("no-such.json");
        let err = UpdateManifest::load(&missing).unwrap_err();
        assert!(matches!(err, ManifestError::Io(_)));
    }

    #[test]
    fn load_serialize_error_on_garbage() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("manifest.json");
        std::fs::write(&path, b"::: not json").unwrap();
        let err = UpdateManifest::load(&path).unwrap_err();
        assert!(matches!(err, ManifestError::Serialize(_)));
    }

    #[test]
    fn save_atomic_errors_on_pathless_target() {
        let m = three_release_manifest();
        // An empty path has no parent.
        let err = m.save_atomic(Path::new("")).unwrap_err();
        assert!(matches!(err, ManifestError::Validation(_)));
    }

    #[cfg(unix)]
    #[test]
    fn save_atomic_io_failure_cleans_tmp() {
        // Pre-create the target as a non-empty directory; rename of a file
        // over a non-empty directory fails on Linux/macOS, exercising the
        // tmp-cleanup branch.
        let tmp = TempDir::new().unwrap();
        let target = tmp.path().join("manifest.json");
        std::fs::create_dir(&target).unwrap();
        std::fs::write(target.join("blocker"), b"x").unwrap();
        let m = three_release_manifest();
        let err = m.save_atomic(&target).unwrap_err();
        assert!(matches!(err, ManifestError::Io(_)));
        let tmp_sibling_path = tmp.path().join("manifest.json.tmp");
        assert!(!tmp_sibling_path.exists(), "tmp cleaned on failure");
    }

    // ---- validate ----

    #[test]
    fn validate_happy() {
        let m = three_release_manifest();
        m.validate().unwrap();
    }

    #[test]
    fn validate_rejects_duplicate_version_in_history() {
        let mut m = three_release_manifest();
        // Duplicate the middle entry so `0.9.9` appears twice.
        m.history
            .insert(2, release(v(0, 9, 9), PlatformTag::MacOsAarch64));
        let err = m.validate().unwrap_err();
        assert!(matches!(&err, ManifestError::Validation(s) if s.contains("duplicate")));
    }

    #[test]
    fn validate_rejects_unsorted_history() {
        let mut m = three_release_manifest();
        m.history.swap(0, 2);
        let err = m.validate().unwrap_err();
        assert!(matches!(&err, ManifestError::Validation(s) if s.contains("not sorted")));
    }

    #[test]
    fn validate_rejects_latest_not_in_history() {
        let m = manifest_with(
            release(v(2, 0, 0), PlatformTag::MacOsAarch64),
            vec![
                release(v(0, 9, 0), PlatformTag::MacOsAarch64),
                release(v(1, 0, 0), PlatformTag::MacOsAarch64),
            ],
        );
        let err = m.validate().unwrap_err();
        assert!(
            matches!(&err, ManifestError::Validation(s) if s.contains("not present in history"))
        );
    }

    #[test]
    fn validate_rejects_bad_artifact() {
        // Construct a manifest, then mutate the artifact post-hoc to break it.
        let mut m = three_release_manifest();
        m.latest.binary.sha256 = "tooshort".into();
        m.history[2].binary.sha256 = "tooshort".into();
        let err = m.validate().unwrap_err();
        assert!(matches!(&err, ManifestError::Validation(s) if s.contains("artifact")));
    }

    #[test]
    fn ord_pre_release_less_than_release_directly() {
        // Exercise the `(Some(_), None) => Less` arm of `Ord`.
        assert!(v_pre(1, 0, 0, "alpha") < v(1, 0, 0));
        assert_eq!(v_pre(1, 0, 0, "alpha").cmp(&v(1, 0, 0)), Ordering::Less);
    }

    // ---- ManifestError + ReleaseVersionError + ArtifactRefError Display ----

    #[test]
    fn manifest_error_display_each_variant() {
        let io = ManifestError::Io(std::io::Error::other("boom"));
        assert!(io.to_string().contains("io error"));
        assert!(std::error::Error::source(&io).is_some());

        let ser = ManifestError::Serialize(serde_json::from_str::<u32>("not-json").unwrap_err());
        assert!(ser.to_string().contains("serialize"));
        assert!(std::error::Error::source(&ser).is_some());

        let val = ManifestError::Validation("x".into());
        assert!(val.to_string().contains("validation"));
        assert!(std::error::Error::source(&val).is_none());

        let sn = ManifestError::SchemaNewer {
            found: 9,
            supported: 1,
        };
        assert!(sn.to_string().contains("newer than supported"));
        assert!(std::error::Error::source(&sn).is_none());
    }

    #[test]
    fn release_version_error_display_each_variant() {
        let f = ReleaseVersionError::Format("x".into());
        assert!(f.to_string().contains("format"));
        let i = ReleaseVersionError::BadInteger("x".into());
        assert!(i.to_string().contains("integer"));
        let p = ReleaseVersionError::BadPre("x".into());
        assert!(p.to_string().contains("pre-release"));
    }

    #[test]
    fn artifact_ref_error_display_each_variant() {
        assert!(ArtifactRefError::EmptyUrl.to_string().contains("empty"));
        assert!(ArtifactRefError::NonHttpsUrl.to_string().contains("https"));
        assert!(ArtifactRefError::BadSha256Length { actual: 3 }
            .to_string()
            .contains("64"));
        assert!(ArtifactRefError::BadSha256Hex.to_string().contains("hex"));
        assert!(ArtifactRefError::ZeroBytes.to_string().contains("zero"));
    }
}
