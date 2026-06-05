//! Default-on, opt-out telemetry payload shape.
//!
//! Phase 1 (data only) — the HTTP transport lands later. This module pins the
//! exact wire shape of every metric Stratum is permitted to emit, together
//! with a strict allowlist guard so a future field addition can never silently
//! widen the payload.
//!
//! See `plan/26-telemetry-and-analytics.md` and the user-memory note
//! "Project: Stratum telemetry" — default-on opt-out, strict allowlist
//! payload, distro strip flag.
//!
//! ## Allowlist
//!
//! The serialized top-level keys are exactly:
//!
//! - `schema_version`
//! - `anon_install_id`
//! - `app_version`
//! - `channel`
//! - `os`
//! - `cpu_arch`
//! - `tier`
//! - `gpu_accel`
//! - `event_kind`
//! - `at`
//!
//! [`payload_is_allowlisted`] verifies the actual JSON keys equal this set.

use std::error::Error;
use std::fmt::{self, Display, Formatter};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Schema version
// ---------------------------------------------------------------------------

/// Wire-format version of the telemetry payload.
///
/// Bumped lockstep with any breaking change to the field set or types. Tests
/// pin this value so a silent edit is caught at review time.
pub const TELEMETRY_SCHEMA_VERSION: u32 = 1;

/// Exact set of top-level JSON keys the [`payload_is_allowlisted`] guard
/// considers permissible. Sorted alphabetically; tests verify this list
/// matches the documented allowlist in the module docs.
const ALLOWED_TOP_LEVEL_KEYS: &[&str] = &[
    "anon_install_id",
    "app_version",
    "at",
    "channel",
    "cpu_arch",
    "event_kind",
    "gpu_accel",
    "os",
    "schema_version",
    "tier",
];

// ---------------------------------------------------------------------------
// AnonInstallId
// ---------------------------------------------------------------------------

/// Length of an [`AnonInstallId`] in lowercase hex chars (8 random bytes
/// rendered as 16 hex digits).
const ANON_INSTALL_ID_LEN: usize = 16;

/// Opaque anonymous install identifier.
///
/// The wire form is exactly 16 lowercase hex characters (no dashes, no
/// prefix). Constructed via [`AnonInstallId::new_random`] on first run and
/// persisted alongside `installed.toml`; the same value is replayed on every
/// later beacon so distinct installs can be counted without identifying the
/// user.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AnonInstallId(String);

impl AnonInstallId {
    /// Generate a fresh 8-byte random id rendered as 16 lowercase hex chars.
    ///
    /// The seed is derived from `SystemTime` nanoseconds mixed through
    /// `SplitMix64`, combined with a process-local monotonic counter so two
    /// calls within the same nanosecond still produce distinct ids. This is
    /// sufficient anonymity for a per-install identifier; cryptographic
    /// entropy is not required.
    #[must_use]
    pub fn new_random() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let counter = COUNTER.fetch_add(1, Ordering::Relaxed);
        let nanos: u64 = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| {
                // u128 → u64 truncation is fine for entropy mixing; the
                // SplitMix64 step that follows propagates the low bits.
                #[allow(
                    clippy::cast_possible_truncation,
                    reason = "low-bits-of-nanos is the entropy source we want"
                )]
                let n = d.as_nanos() as u64;
                n
            })
            .unwrap_or(0);
        let seed = splitmix64(nanos ^ splitmix64(counter));
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut bytes = [0u8; 8];
        rng.fill(&mut bytes);
        let mut out = String::with_capacity(ANON_INSTALL_ID_LEN);
        for byte in bytes {
            let hi = (byte >> 4) & 0xF;
            let lo = byte & 0xF;
            out.push(hex_digit(hi));
            out.push(hex_digit(lo));
        }
        Self(out)
    }

    /// Parse `s` as a 16-lowercase-hex-char install id.
    ///
    /// Intentionally a free-standing method (not the `FromStr` trait) so the
    /// caller does not need a turbofish-style `parse::<AnonInstallId>` call;
    /// the resulting signature is the same shape used by [`SecretId::new`].
    ///
    /// # Errors
    ///
    /// Returns [`AnonInstallIdError::WrongLength`] when `s` is not exactly 16
    /// characters, and [`AnonInstallIdError::InvalidHex`] when any character
    /// is not a lowercase hex digit.
    #[allow(
        clippy::should_implement_trait,
        reason = "deliberate inherent constructor; mirrors SecretId::new shape"
    )]
    pub fn from_str(s: &str) -> Result<Self, AnonInstallIdError> {
        if s.len() != ANON_INSTALL_ID_LEN {
            return Err(AnonInstallIdError::WrongLength { actual: s.len() });
        }
        for c in s.chars() {
            if !is_lower_hex(c) {
                return Err(AnonInstallIdError::InvalidHex);
            }
        }
        Ok(Self(s.to_owned()))
    }

    /// Borrow the underlying string slice.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Display for AnonInstallId {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Serialize for AnonInstallId {
    fn serialize<S: serde::Serializer>(&self, ser: S) -> Result<S::Ok, S::Error> {
        ser.serialize_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for AnonInstallId {
    fn deserialize<D: serde::Deserializer<'de>>(de: D) -> Result<Self, D::Error> {
        let s = String::deserialize(de)?;
        Self::from_str(&s).map_err(serde::de::Error::custom)
    }
}

#[inline]
const fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[inline]
const fn hex_digit(nibble: u8) -> char {
    match nibble {
        0..=9 => (b'0' + nibble) as char,
        _ => (b'a' + nibble - 10) as char,
    }
}

#[inline]
const fn is_lower_hex(c: char) -> bool {
    matches!(c, '0'..='9' | 'a'..='f')
}

/// First-failure rejection from [`AnonInstallId::from_str`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnonInstallIdError {
    /// Length was not exactly 16 chars.
    WrongLength {
        /// Observed length in chars.
        actual: usize,
    },
    /// Encountered a character outside `[0-9a-f]`.
    InvalidHex,
}

impl Display for AnonInstallIdError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongLength { actual } => write!(
                f,
                "anon install id must be {ANON_INSTALL_ID_LEN} chars; got {actual}"
            ),
            Self::InvalidHex => f.write_str("anon install id must be lowercase hex `[0-9a-f]`"),
        }
    }
}

impl Error for AnonInstallIdError {}

// ---------------------------------------------------------------------------
// Wire tag enums
// ---------------------------------------------------------------------------

/// Kind of telemetry event. The strict enum guarantees a new event variant
/// requires a code change and a serde test update.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TelemetryEventKind {
    /// First-run install beacon.
    Install,
    /// Self-update completed.
    Update,
    /// Once-per-UTC-day liveness beacon.
    DailyActive,
    /// First user-initiated chat turn after install.
    FirstChatTurn,
    /// User opted in to crash reports.
    CrashOptIn,
    /// Uninstall beacon (best-effort).
    Uninstall,
}

/// Release channel the running binary self-identifies as.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReleaseChannel {
    /// Stable / GA channel.
    Stable,
    /// Beta channel.
    Beta,
    /// Nightly channel.
    Nightly,
}

/// Coarse OS bucket. Anything outside `macos`/`linux`/`windows` collapses to
/// `other` so the cardinality stays bounded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OsTag {
    /// macOS.
    #[serde(rename = "macos")]
    MacOS,
    /// Linux.
    Linux,
    /// Windows.
    Windows,
    /// Anything else (BSD, illumos, etc.).
    Other,
}

/// Coarse CPU architecture bucket.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[allow(
    clippy::upper_case_acronyms,
    reason = "exact wire encoding — `X86_64` serializes as `x86_64`"
)]
pub enum CpuArchTag {
    /// `x86_64` / amd64.
    X86_64,
    /// 64-bit ARM.
    Aarch64,
    /// Anything else (riscv64, etc.).
    Other,
}

// ---------------------------------------------------------------------------
// TelemetryConfig
// ---------------------------------------------------------------------------

/// Runtime-side telemetry configuration.
///
/// Defaults reflect the documented opt-out posture: enabled by default, ships
/// to the canonical `telemetry.stratum.dev` ingest, on the `Stable` channel.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TelemetryConfig {
    /// Whether telemetry is currently enabled for this install.
    pub enabled: bool,
    /// Absolute URL of the ingest endpoint.
    pub endpoint: String,
    /// Release channel the running binary self-identifies as.
    pub channel: ReleaseChannel,
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            endpoint: "https://telemetry.stratum.dev/v1/ingest".to_owned(),
            channel: ReleaseChannel::Stable,
        }
    }
}

// ---------------------------------------------------------------------------
// TelemetryPayload
// ---------------------------------------------------------------------------

/// The complete, strictly-allowlisted set of fields a telemetry beacon may
/// carry. Adding a field here is a breaking schema change and **must** bump
/// [`TELEMETRY_SCHEMA_VERSION`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TelemetryPayload {
    /// Wire-format version — mirrors [`TELEMETRY_SCHEMA_VERSION`].
    pub schema_version: u32,
    /// Anonymous install identifier (16 lowercase hex chars).
    pub anon_install_id: AnonInstallId,
    /// Stratum version string (e.g. `1.4.7-rc.3`).
    pub app_version: String,
    /// Release channel.
    pub channel: ReleaseChannel,
    /// Coarse OS bucket.
    pub os: OsTag,
    /// Coarse CPU arch bucket.
    pub cpu_arch: CpuArchTag,
    /// Hardware tier label (`low` / `medium` / `high`).
    pub tier: String,
    /// GPU acceleration label (`cpu` / `metal` / `cuda` / `vulkan` / …).
    pub gpu_accel: String,
    /// Kind of event being reported.
    pub event_kind: TelemetryEventKind,
    /// Wall-clock time the event was generated, as `SystemTime`.
    pub at: SystemTime,
}

// ---------------------------------------------------------------------------
// build_payload / redact / allowlist guard
// ---------------------------------------------------------------------------

/// Assemble a [`TelemetryPayload`] from the runtime-side inputs.
///
/// The schema version is fixed to [`TELEMETRY_SCHEMA_VERSION`]; the channel
/// is taken from `cfg` so a binary on the nightly channel does not need to
/// thread a separate channel argument through every call site.
#[must_use]
#[allow(
    clippy::too_many_arguments,
    reason = "each argument corresponds to one allowlisted wire field"
)]
pub fn build_payload(
    cfg: &TelemetryConfig,
    install_id: &AnonInstallId,
    app_version: &str,
    kind: TelemetryEventKind,
    tier: &str,
    gpu_accel: &str,
    os: OsTag,
    cpu_arch: CpuArchTag,
    now: SystemTime,
) -> TelemetryPayload {
    TelemetryPayload {
        schema_version: TELEMETRY_SCHEMA_VERSION,
        anon_install_id: install_id.clone(),
        app_version: app_version.to_owned(),
        channel: cfg.channel,
        os,
        cpu_arch,
        tier: tier.to_owned(),
        gpu_accel: gpu_accel.to_owned(),
        event_kind: kind,
        at: now,
    }
}

/// Strip-flag redaction: replace `app_version` with its major-only form.
///
/// `1.4.7-rc.3` becomes `1`. A value that does not contain a `.` is taken
/// verbatim (its major component is the whole string). A value whose major
/// component is empty (e.g. leading `.`) is replaced with the empty string.
///
/// Every other field is left untouched. Idempotent.
pub fn redact(payload: &mut TelemetryPayload) {
    let major = match payload.app_version.split_once('.') {
        Some((head, _)) => head,
        None => payload.app_version.as_str(),
    };
    // Strip any pre-release suffix attached to a single-component version
    // (e.g. `1-rc.3` → `1`).
    let major = match major.split_once('-') {
        Some((head, _)) => head,
        None => major,
    };
    payload.app_version = major.to_owned();
}

/// Verify that `payload`, when serialized to JSON, exposes exactly the
/// allowlisted top-level keys — nothing more, nothing less.
///
/// This is a defense-in-depth guard against a future struct field addition
/// silently widening the payload. Tests assert the documented allowlist
/// matches this check.
///
/// # Errors
///
/// Returns [`TelemetryError::Serialize`] if `serde_json::to_value` fails (it
/// should not for the well-typed [`TelemetryPayload`]) and
/// [`TelemetryError::UnknownField`] on the first key drift detected.
pub fn payload_is_allowlisted(payload: &TelemetryPayload) -> Result<(), TelemetryError> {
    let value =
        serde_json::to_value(payload).map_err(|e| TelemetryError::Serialize(e.to_string()))?;
    check_value_allowlisted(&value)
}

fn check_value_allowlisted(value: &serde_json::Value) -> Result<(), TelemetryError> {
    let serde_json::Value::Object(map) = value else {
        return Err(TelemetryError::Serialize(
            "telemetry payload did not serialize to a JSON object".to_owned(),
        ));
    };
    // Reject any key outside the allowlist.
    for key in map.keys() {
        if !ALLOWED_TOP_LEVEL_KEYS.contains(&key.as_str()) {
            return Err(TelemetryError::UnknownField(key.clone()));
        }
    }
    // Reject any allowlisted key that is missing from the object.
    for allowed in ALLOWED_TOP_LEVEL_KEYS {
        if !map.contains_key(*allowed) {
            return Err(TelemetryError::UnknownField((*allowed).to_owned()));
        }
    }
    Ok(())
}

/// Failure modes for the allowlist guard.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TelemetryError {
    /// A top-level JSON key was outside the allowlist, or an allowlisted
    /// key was missing. The wrapped string is the offending key name.
    UnknownField(String),
    /// `serde_json::to_value` failed. Should not happen in practice; surfaced
    /// for diagnostic completeness.
    Serialize(String),
}

impl Display for TelemetryError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownField(name) => {
                write!(f, "telemetry payload key not on allowlist: {name}")
            }
            Self::Serialize(msg) => write!(f, "telemetry payload failed to serialize: {msg}"),
        }
    }
}

impl Error for TelemetryError {}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::collections::BTreeSet;
    use std::time::{Duration, UNIX_EPOCH};

    use serde_json::json;

    use super::*;

    fn sample_install_id() -> AnonInstallId {
        AnonInstallId::from_str("0123456789abcdef").unwrap()
    }

    fn sample_payload() -> TelemetryPayload {
        let cfg = TelemetryConfig::default();
        let id = sample_install_id();
        build_payload(
            &cfg,
            &id,
            "1.4.7-rc.3",
            TelemetryEventKind::Install,
            "high",
            "metal",
            OsTag::MacOS,
            CpuArchTag::Aarch64,
            UNIX_EPOCH + Duration::from_secs(1_700_000_000),
        )
    }

    // ---- AnonInstallId ----------------------------------------------------

    #[test]
    fn anon_install_id_new_random_is_16_lower_hex() {
        let id = AnonInstallId::new_random();
        let s = id.as_str();
        assert_eq!(s.len(), 16);
        for c in s.chars() {
            assert!(
                matches!(c, '0'..='9' | 'a'..='f'),
                "non-lowercase-hex char {c:?} in {s}"
            );
        }
    }

    #[test]
    fn anon_install_id_new_random_is_unique_enough() {
        // 8 bytes of OS entropy: collisions on a handful of calls are
        // astronomically unlikely. This catches a literal "always returns
        // the same value" bug.
        let a = AnonInstallId::new_random();
        let b = AnonInstallId::new_random();
        assert_ne!(a, b);
    }

    #[test]
    fn anon_install_id_from_str_accepts_valid() {
        let id = AnonInstallId::from_str("0123456789abcdef").unwrap();
        assert_eq!(id.as_str(), "0123456789abcdef");
    }

    #[test]
    fn anon_install_id_from_str_rejects_wrong_length() {
        assert_eq!(
            AnonInstallId::from_str("0123"),
            Err(AnonInstallIdError::WrongLength { actual: 4 })
        );
        assert_eq!(
            AnonInstallId::from_str("0123456789abcdef0"),
            Err(AnonInstallIdError::WrongLength { actual: 17 })
        );
    }

    #[test]
    fn anon_install_id_from_str_rejects_non_hex() {
        assert_eq!(
            AnonInstallId::from_str("0123456789abcdez"),
            Err(AnonInstallIdError::InvalidHex)
        );
    }

    #[test]
    fn anon_install_id_from_str_rejects_uppercase() {
        // Strict lowercase: uppercase hex is rejected.
        assert_eq!(
            AnonInstallId::from_str("0123456789ABCDEF"),
            Err(AnonInstallIdError::InvalidHex)
        );
    }

    #[test]
    fn anon_install_id_display_equals_inner() {
        let id = AnonInstallId::from_str("deadbeefcafef00d").unwrap();
        assert_eq!(id.to_string(), "deadbeefcafef00d");
        assert_eq!(format!("{id}"), id.as_str());
    }

    #[test]
    fn anon_install_id_serde_transparent_roundtrip() {
        let id = sample_install_id();
        let json = serde_json::to_string(&id).unwrap();
        // Transparent string encoding.
        assert_eq!(json, "\"0123456789abcdef\"");
        let back: AnonInstallId = serde_json::from_str(&json).unwrap();
        assert_eq!(back, id);
    }

    #[test]
    fn anon_install_id_serde_rejects_invalid() {
        let bad = "\"NOTHEX\"";
        let r: Result<AnonInstallId, _> = serde_json::from_str(bad);
        assert!(r.is_err());
    }

    #[test]
    fn anon_install_id_error_display_covers_all_variants() {
        let wl = AnonInstallIdError::WrongLength { actual: 4 }.to_string();
        assert!(wl.contains('4'), "got {wl}");
        assert!(wl.contains("16"), "got {wl}");
        let inv = AnonInstallIdError::InvalidHex.to_string();
        assert!(inv.contains("hex"), "got {inv}");
        // Error trait.
        let _: &dyn Error = &AnonInstallIdError::InvalidHex;
    }

    // ---- TelemetryConfig --------------------------------------------------

    #[test]
    fn telemetry_config_default_matches_docs() {
        let cfg = TelemetryConfig::default();
        assert!(cfg.enabled);
        assert_eq!(cfg.endpoint, "https://telemetry.stratum.dev/v1/ingest");
        assert_eq!(cfg.channel, ReleaseChannel::Stable);
    }

    // ---- build_payload ----------------------------------------------------

    #[test]
    fn build_payload_populates_every_field() {
        let cfg = TelemetryConfig {
            channel: ReleaseChannel::Nightly,
            ..TelemetryConfig::default()
        };
        let id = sample_install_id();
        let at = UNIX_EPOCH + Duration::from_secs(42);
        let p = build_payload(
            &cfg,
            &id,
            "0.1.0",
            TelemetryEventKind::DailyActive,
            "medium",
            "cpu",
            OsTag::Linux,
            CpuArchTag::X86_64,
            at,
        );
        assert_eq!(p.schema_version, TELEMETRY_SCHEMA_VERSION);
        assert_eq!(p.anon_install_id, id);
        assert_eq!(p.app_version, "0.1.0");
        assert_eq!(p.channel, ReleaseChannel::Nightly);
        assert_eq!(p.os, OsTag::Linux);
        assert_eq!(p.cpu_arch, CpuArchTag::X86_64);
        assert_eq!(p.tier, "medium");
        assert_eq!(p.gpu_accel, "cpu");
        assert_eq!(p.event_kind, TelemetryEventKind::DailyActive);
        assert_eq!(p.at, at);
    }

    // ---- redact -----------------------------------------------------------

    #[test]
    fn redact_truncates_app_version_to_major() {
        let mut p = sample_payload();
        assert_eq!(p.app_version, "1.4.7-rc.3");
        redact(&mut p);
        assert_eq!(p.app_version, "1");
    }

    #[test]
    fn redact_leaves_other_fields_untouched() {
        let mut p = sample_payload();
        let before = p.clone();
        redact(&mut p);
        assert_eq!(p.schema_version, before.schema_version);
        assert_eq!(p.anon_install_id, before.anon_install_id);
        assert_eq!(p.channel, before.channel);
        assert_eq!(p.os, before.os);
        assert_eq!(p.cpu_arch, before.cpu_arch);
        assert_eq!(p.tier, before.tier);
        assert_eq!(p.gpu_accel, before.gpu_accel);
        assert_eq!(p.event_kind, before.event_kind);
        assert_eq!(p.at, before.at);
    }

    #[test]
    fn redact_is_idempotent() {
        let mut p = sample_payload();
        redact(&mut p);
        let once = p.clone();
        redact(&mut p);
        assert_eq!(p, once);
        assert_eq!(p.app_version, "1");
    }

    #[test]
    fn redact_handles_versions_without_dot() {
        let mut p = sample_payload();
        p.app_version = "nightly".to_owned();
        redact(&mut p);
        assert_eq!(p.app_version, "nightly");

        let mut p2 = sample_payload();
        p2.app_version = "2-rc.1".to_owned();
        redact(&mut p2);
        assert_eq!(p2.app_version, "2");
    }

    // ---- payload_is_allowlisted -------------------------------------------

    #[test]
    fn payload_is_allowlisted_accepts_default_built() {
        let p = sample_payload();
        payload_is_allowlisted(&p).unwrap();
    }

    #[test]
    fn payload_is_allowlisted_rejects_extra_top_level_key() {
        // Hand-craft a JSON object that mirrors the payload shape plus one
        // forbidden key, then run the inner check directly.
        let v = json!({
            "schema_version": 1,
            "anon_install_id": "0123456789abcdef",
            "app_version": "0.1.0",
            "channel": "stable",
            "os": "linux",
            "cpu_arch": "x86_64",
            "tier": "low",
            "gpu_accel": "cpu",
            "event_kind": "install",
            "at": { "secs_since_epoch": 0, "nanos_since_epoch": 0 },
            "user_email": "leaked@example.com"
        });
        let err = check_value_allowlisted(&v).unwrap_err();
        assert!(matches!(err, TelemetryError::UnknownField(ref k) if k == "user_email"));
    }

    #[test]
    fn payload_is_allowlisted_rejects_missing_allowlisted_key() {
        // Build a JSON object missing `gpu_accel`.
        let v = json!({
            "schema_version": 1,
            "anon_install_id": "0123456789abcdef",
            "app_version": "0.1.0",
            "channel": "stable",
            "os": "linux",
            "cpu_arch": "x86_64",
            "tier": "low",
            "event_kind": "install",
            "at": { "secs_since_epoch": 0, "nanos_since_epoch": 0 }
        });
        let err = check_value_allowlisted(&v).unwrap_err();
        assert!(matches!(err, TelemetryError::UnknownField(ref k) if k == "gpu_accel"));
    }

    #[test]
    fn payload_is_allowlisted_rejects_non_object() {
        let v = json!(["nope"]);
        let err = check_value_allowlisted(&v).unwrap_err();
        assert!(matches!(err, TelemetryError::Serialize(_)));
    }

    #[test]
    fn allowlist_matches_documented_keys() {
        // Lock the documented allowlist against silent drift. Any future
        // field addition must touch this set explicitly.
        let actual: BTreeSet<&str> = ALLOWED_TOP_LEVEL_KEYS.iter().copied().collect();
        let documented: BTreeSet<&str> = [
            "schema_version",
            "anon_install_id",
            "app_version",
            "channel",
            "os",
            "cpu_arch",
            "tier",
            "gpu_accel",
            "event_kind",
            "at",
        ]
        .into_iter()
        .collect();
        assert_eq!(actual, documented);
    }

    #[test]
    fn allowlist_top_level_keys_match_serialized_payload() {
        // The runtime guarantee: a freshly serialized payload's top-level
        // keys equal the allowlist exactly.
        let p = sample_payload();
        let v = serde_json::to_value(&p).unwrap();
        let obj = v.as_object().unwrap();
        let actual: BTreeSet<&str> = obj.keys().map(String::as_str).collect();
        let allowed: BTreeSet<&str> = ALLOWED_TOP_LEVEL_KEYS.iter().copied().collect();
        assert_eq!(actual, allowed);
    }

    // ---- serde shapes for the wire tag enums ------------------------------

    #[test]
    fn release_channel_serde_values_are_exact() {
        assert_eq!(
            serde_json::to_string(&ReleaseChannel::Stable).unwrap(),
            "\"stable\""
        );
        assert_eq!(
            serde_json::to_string(&ReleaseChannel::Beta).unwrap(),
            "\"beta\""
        );
        assert_eq!(
            serde_json::to_string(&ReleaseChannel::Nightly).unwrap(),
            "\"nightly\""
        );
    }

    #[test]
    fn os_tag_serde_values_are_exact() {
        assert_eq!(serde_json::to_string(&OsTag::MacOS).unwrap(), "\"macos\"");
        assert_eq!(serde_json::to_string(&OsTag::Linux).unwrap(), "\"linux\"");
        assert_eq!(
            serde_json::to_string(&OsTag::Windows).unwrap(),
            "\"windows\""
        );
        assert_eq!(serde_json::to_string(&OsTag::Other).unwrap(), "\"other\"");
    }

    #[test]
    fn cpu_arch_tag_serde_values_are_exact() {
        assert_eq!(
            serde_json::to_string(&CpuArchTag::X86_64).unwrap(),
            "\"x86_64\""
        );
        assert_eq!(
            serde_json::to_string(&CpuArchTag::Aarch64).unwrap(),
            "\"aarch64\""
        );
        assert_eq!(
            serde_json::to_string(&CpuArchTag::Other).unwrap(),
            "\"other\""
        );
    }

    #[test]
    fn telemetry_event_kind_serde_values_are_exact() {
        let cases = [
            (TelemetryEventKind::Install, "\"install\""),
            (TelemetryEventKind::Update, "\"update\""),
            (TelemetryEventKind::DailyActive, "\"daily_active\""),
            (TelemetryEventKind::FirstChatTurn, "\"first_chat_turn\""),
            (TelemetryEventKind::CrashOptIn, "\"crash_opt_in\""),
            (TelemetryEventKind::Uninstall, "\"uninstall\""),
        ];
        for (kind, expected) in cases {
            let got = serde_json::to_string(&kind).unwrap();
            assert_eq!(got, expected);
        }
    }

    // ---- payload round-trip per event kind --------------------------------

    #[test]
    fn telemetry_payload_roundtrip_for_each_event_kind() {
        let kinds = [
            TelemetryEventKind::Install,
            TelemetryEventKind::Update,
            TelemetryEventKind::DailyActive,
            TelemetryEventKind::FirstChatTurn,
            TelemetryEventKind::CrashOptIn,
            TelemetryEventKind::Uninstall,
        ];
        for kind in kinds {
            let mut p = sample_payload();
            p.event_kind = kind;
            let json = serde_json::to_string(&p).unwrap();
            let back: TelemetryPayload = serde_json::from_str(&json).unwrap();
            assert_eq!(back, p, "kind {kind:?} did not round-trip");
        }
    }

    // ---- TelemetryError ---------------------------------------------------

    #[test]
    fn telemetry_error_display_smoke() {
        let unk = TelemetryError::UnknownField("ghost".into()).to_string();
        assert!(unk.contains("ghost"), "got {unk}");
        let ser = TelemetryError::Serialize("oops".into()).to_string();
        assert!(ser.contains("oops"), "got {ser}");
        // Error trait.
        let _: &dyn Error = &TelemetryError::UnknownField("x".into());
    }

    // ---- schema version pin -----------------------------------------------

    #[test]
    fn schema_version_is_pinned_to_one() {
        assert_eq!(TELEMETRY_SCHEMA_VERSION, 1);
    }
}
