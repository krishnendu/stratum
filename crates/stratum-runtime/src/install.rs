//! First-run install record and atomic TOML writer.
//!
//! `installed.toml` is the marker file Stratum looks for at startup to decide
//! whether to run the wizard. Schema is versioned; backups are written on
//! migration per `plan/23-updates-and-upgrades.md` §5.

use std::path::Path;

use serde::{Deserialize, Serialize};
use stratum_types::error::codes::E1001_INSTALLED_SCHEMA_UNREADABLE;
use stratum_types::{StratumError, StratumResult};
use time::format_description::well_known::Rfc3339;
use time::OffsetDateTime;

use crate::probe::{GpuBackend, HardwareProbe};
use crate::tier::Tier;

const SCHEMA_VERSION: u32 = 1;

/// The subset of the probe captured at install time for later comparison by
/// `stratum doctor` and the tier-upgrade suggestion logic.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TierInputs {
    /// RAM total in mebibytes at install time.
    pub ram_total_mib: u32,
    /// CPU feature set at install time.
    pub cpu_features: Vec<String>,
    /// GPU backend at install time.
    pub gpu: GpuBackend,
    /// Operating system at install time.
    pub os: String,
}

impl TierInputs {
    /// Capture the relevant subset from a probe.
    #[must_use]
    pub fn from_probe(probe: &HardwareProbe) -> Self {
        Self {
            ram_total_mib: probe.ram_total_mib,
            cpu_features: probe.cpu_features.clone(),
            gpu: probe.gpu,
            os: probe.os.clone(),
        }
    }
}

/// On-disk install record.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstalledToml {
    /// Schema version for forward compatibility.
    pub schema_version: u32,
    /// RFC3339 timestamp of the install.
    pub installed_at: String,
    /// Classified tier at install time.
    pub tier: Tier,
    /// Probe inputs at install time.
    pub tier_inputs: TierInputs,
    /// GPU acceleration backend selected for the runtime.
    pub gpu_accel: GpuBackend,
}

impl InstalledToml {
    /// Build a fresh record from a probe and a classified tier.
    ///
    /// # Panics
    /// Does not panic for any caller-reachable input; the internal `expect`
    /// is justified because [`OffsetDateTime::format`] with [`Rfc3339`] is
    /// infallible for any well-formed `OffsetDateTime`. The carve-out is
    /// tracked in `docs/coverage-exclusions.md`.
    #[must_use]
    pub fn new(probe: &HardwareProbe, tier: Tier, now: OffsetDateTime) -> Self {
        #[allow(
            clippy::expect_used,
            reason = "OffsetDateTime::format with Rfc3339 is infallible"
        )]
        let installed_at = now
            .format(&Rfc3339)
            .expect("Rfc3339 formatting of OffsetDateTime is infallible");
        Self {
            schema_version: SCHEMA_VERSION,
            installed_at,
            tier,
            tier_inputs: TierInputs::from_probe(probe),
            gpu_accel: probe.gpu,
        }
    }

    /// Write atomically to `path`: serialize, write to a sibling temp file in
    /// the same directory, then rename.
    ///
    /// # Errors
    /// Returns [`E1001_INSTALLED_SCHEMA_UNREADABLE`] wrapping the underlying
    /// io error.
    ///
    /// # Panics
    /// Does not panic on any input the public API produces; the internal
    /// `expect` is justified by the fact that [`InstalledToml`] contains only
    /// primitive types and `Vec<String>`, for which `toml_edit::ser::to_string`
    /// is infallible. The carve-out is tracked in `docs/coverage-exclusions.md`.
    pub fn write_atomic(&self, path: &Path) -> StratumResult<()> {
        let parent = path.parent().ok_or_else(|| {
            StratumError::new(
                E1001_INSTALLED_SCHEMA_UNREADABLE,
                "installed.toml path has no parent",
            )
        })?;
        std::fs::create_dir_all(parent).map_err(|e| {
            StratumError::new(
                E1001_INSTALLED_SCHEMA_UNREADABLE,
                format!("cannot create {}", parent.display()),
            )
            .with_cause(e)
        })?;
        // `InstalledToml` contains only primitive types and `Vec<String>`;
        // toml_edit serialization is infallible for this shape. The carve-out
        // is tracked in `docs/coverage-exclusions.md`.
        #[allow(
            clippy::expect_used,
            reason = "InstalledToml serialization is infallible (primitives only)"
        )]
        let rendered =
            toml_edit::ser::to_string(self).expect("InstalledToml serialization is infallible");
        let tmp = path.with_extension("toml.tmp");
        std::fs::write(&tmp, rendered.as_bytes()).map_err(|e| {
            StratumError::new(
                E1001_INSTALLED_SCHEMA_UNREADABLE,
                format!("write {}", tmp.display()),
            )
            .with_cause(e)
        })?;
        std::fs::rename(&tmp, path).map_err(|e| {
            StratumError::new(
                E1001_INSTALLED_SCHEMA_UNREADABLE,
                format!("rename to {}", path.display()),
            )
            .with_cause(e)
        })?;
        Ok(())
    }

    /// Load + parse from `path`.
    ///
    /// # Errors
    /// Returns [`E1001_INSTALLED_SCHEMA_UNREADABLE`] for io or parse failures.
    pub fn load(path: &Path) -> StratumResult<Self> {
        let raw = std::fs::read_to_string(path).map_err(|e| {
            StratumError::new(
                E1001_INSTALLED_SCHEMA_UNREADABLE,
                format!("read {}", path.display()),
            )
            .with_cause(e)
        })?;
        toml_edit::de::from_str::<Self>(&raw).map_err(|e| {
            StratumError::new(
                E1001_INSTALLED_SCHEMA_UNREADABLE,
                format!("parse {}", path.display()),
            )
            .with_cause(e)
        })
    }
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::*;
    use crate::probe::HardwareProbe;
    use crate::tier::Tier;

    fn fixture_probe() -> HardwareProbe {
        HardwareProbe::synthetic(
            12 * 1024,
            7 * 1024,
            "aarch64",
            vec!["neon"],
            8,
            GpuBackend::Metal,
            "macos",
        )
    }

    #[test]
    fn new_captures_probe_inputs() {
        let p = fixture_probe();
        let now = OffsetDateTime::from_unix_timestamp(1_700_000_000).unwrap();
        let rec = InstalledToml::new(&p, Tier::High, now);
        assert_eq!(rec.schema_version, SCHEMA_VERSION);
        assert_eq!(rec.tier, Tier::High);
        assert_eq!(rec.tier_inputs.ram_total_mib, 12 * 1024);
        assert_eq!(rec.tier_inputs.gpu, GpuBackend::Metal);
        assert_eq!(rec.gpu_accel, GpuBackend::Metal);
    }

    #[test]
    fn installed_at_is_rfc3339_with_tz() {
        let p = fixture_probe();
        let now = OffsetDateTime::from_unix_timestamp(1_700_000_000).unwrap();
        let rec = InstalledToml::new(&p, Tier::High, now);
        assert!(rec.installed_at.contains('T'));
        // OffsetDateTime::from_unix_timestamp produces UTC; Rfc3339 ends with `Z`.
        assert!(rec.installed_at.ends_with('Z'));
    }

    #[test]
    fn write_then_load_roundtrip() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("config").join("installed.toml");
        let p = fixture_probe();
        let now = OffsetDateTime::from_unix_timestamp(1_700_000_000).unwrap();
        let rec = InstalledToml::new(&p, Tier::High, now);
        rec.write_atomic(&path).unwrap();
        assert!(path.exists());
        let back = InstalledToml::load(&path).unwrap();
        assert_eq!(rec, back);
    }

    #[test]
    fn write_atomic_overwrites_existing() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("installed.toml");
        std::fs::write(&path, b"stale").unwrap();
        let p = fixture_probe();
        let now = OffsetDateTime::from_unix_timestamp(1_700_000_000).unwrap();
        let rec = InstalledToml::new(&p, Tier::Medium, now);
        rec.write_atomic(&path).unwrap();
        let back = InstalledToml::load(&path).unwrap();
        assert_eq!(back.tier, Tier::Medium);
    }

    #[test]
    fn load_missing_file_errors() {
        let tmp = TempDir::new().unwrap();
        let missing = tmp.path().join("missing.toml");
        let err = InstalledToml::load(&missing).unwrap_err();
        assert_eq!(err.code(), &E1001_INSTALLED_SCHEMA_UNREADABLE);
    }

    #[test]
    fn load_malformed_file_errors() {
        let tmp = TempDir::new().unwrap();
        let bad = tmp.path().join("bad.toml");
        std::fs::write(&bad, b"this is = not [ valid toml").unwrap();
        let err = InstalledToml::load(&bad).unwrap_err();
        assert_eq!(err.code(), &E1001_INSTALLED_SCHEMA_UNREADABLE);
    }

    #[test]
    fn write_to_root_no_parent_errors() {
        // Path with no parent component cannot be the target of an atomic write.
        let bare = Path::new("installed-bare.toml");
        let _ = std::fs::remove_file(bare); // best-effort cleanup
        let p = fixture_probe();
        let now = OffsetDateTime::from_unix_timestamp(1_700_000_000).unwrap();
        let rec = InstalledToml::new(&p, Tier::Medium, now);
        // Use a path whose parent is "" — only the empty string has no parent.
        let no_parent = Path::new("");
        let err = rec.write_atomic(no_parent).unwrap_err();
        assert_eq!(err.code(), &E1001_INSTALLED_SCHEMA_UNREADABLE);
    }

    #[test]
    fn write_into_file_as_dir_errors() {
        let tmp = TempDir::new().unwrap();
        let blocker = tmp.path().join("blocker");
        std::fs::write(&blocker, b"x").unwrap();
        // Treat the regular file as if it were a parent directory.
        let target = blocker.join("installed.toml");
        let p = fixture_probe();
        let now = OffsetDateTime::from_unix_timestamp(1_700_000_000).unwrap();
        let rec = InstalledToml::new(&p, Tier::Medium, now);
        let err = rec.write_atomic(&target).unwrap_err();
        assert_eq!(err.code(), &E1001_INSTALLED_SCHEMA_UNREADABLE);
    }

    #[cfg(unix)]
    #[test]
    fn write_atomic_write_failure_is_reported() {
        // Pre-create the tmp file path as a directory; `fs::write` then fails
        // because the destination is a directory.
        let tmp = TempDir::new().unwrap();
        let target = tmp.path().join("installed.toml");
        let blocker = tmp.path().join("installed.toml.tmp");
        std::fs::create_dir(&blocker).unwrap();
        let p = fixture_probe();
        let now = OffsetDateTime::from_unix_timestamp(1_700_000_000).unwrap();
        let rec = InstalledToml::new(&p, Tier::Medium, now);
        let err = rec.write_atomic(&target).unwrap_err();
        assert_eq!(err.code(), &E1001_INSTALLED_SCHEMA_UNREADABLE);
    }

    #[cfg(unix)]
    #[test]
    fn write_atomic_rename_failure_is_reported() {
        // Pre-create the target as a non-empty directory; `fs::rename` of a
        // file over a non-empty directory fails on Unix.
        let tmp = TempDir::new().unwrap();
        let target = tmp.path().join("installed.toml");
        std::fs::create_dir(&target).unwrap();
        std::fs::write(target.join("dummy"), b"x").unwrap();
        let p = fixture_probe();
        let now = OffsetDateTime::from_unix_timestamp(1_700_000_000).unwrap();
        let rec = InstalledToml::new(&p, Tier::Medium, now);
        let err = rec.write_atomic(&target).unwrap_err();
        assert_eq!(err.code(), &E1001_INSTALLED_SCHEMA_UNREADABLE);
    }

    #[test]
    fn tier_inputs_serde_roundtrip() {
        let p = fixture_probe();
        let inputs = TierInputs::from_probe(&p);
        let s = serde_json::to_string(&inputs).unwrap();
        let back: TierInputs = serde_json::from_str(&s).unwrap();
        assert_eq!(inputs, back);
    }
}
