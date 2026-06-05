//! First-run install record and atomic TOML writer.
//!
//! `installed.toml` is the marker file Stratum looks for at startup to decide
//! whether to run the wizard. Schema is versioned; backups are written on
//! migration per `plan/23-updates-and-upgrades.md` §5.

use std::fmt;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use stratum_types::error::codes::E1001_INSTALLED_SCHEMA_UNREADABLE;
use stratum_types::{StratumError, StratumResult};
use time::format_description::well_known::Rfc3339;
use time::OffsetDateTime;

use crate::probe::{GpuBackend, HardwareProbe};
use crate::tier::Tier;

/// Current `installed.toml` schema version.
///
/// Bumps within a major release are additive only. A file declaring a higher
/// version than this constant is refused (downgrade is unsupported); a file
/// declaring a lower version is migrated forward in [`load_with_migration`].
pub const CURRENT_SCHEMA_VERSION: u32 = 1;

const SCHEMA_VERSION: u32 = CURRENT_SCHEMA_VERSION;

const fn default_schema_version() -> u32 {
    // Older files predate the explicit field; default to 1.
    1
}

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
    #[serde(default = "default_schema_version")]
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

/// IO-side errors emitted by [`save_atomic`] / [`restore_backup`].
///
/// Plain enum; no STRAT-E code mapping yet (this scaffold predates the error
/// code allocations for migrations).
#[derive(Debug)]
pub enum InstallIoError {
    /// Underlying [`std::io::Error`].
    Io(std::io::Error),
    /// Target path has no parent directory component.
    MissingParent,
    /// Backup hardlink-or-copy failed (and we refuse to clobber the live file).
    BackupFailed {
        /// Human-readable reason.
        reason: String,
    },
}

impl fmt::Display for InstallIoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "io error: {e}"),
            Self::MissingParent => f.write_str("target path has no parent directory"),
            Self::BackupFailed { reason } => write!(f, "backup failed: {reason}"),
        }
    }
}

impl std::error::Error for InstallIoError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for InstallIoError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

/// Load-side errors emitted by [`load_with_migration`].
#[derive(Debug)]
pub enum InstallLoadError {
    /// Underlying [`std::io::Error`].
    Io(std::io::Error),
    /// TOML parse failure.
    Parse(String),
    /// The on-disk schema version is newer than [`CURRENT_SCHEMA_VERSION`];
    /// downgrade is unsupported.
    SchemaNewer {
        /// Version found on disk.
        found: u32,
        /// Highest version this build understands.
        supported: u32,
    },
}

impl fmt::Display for InstallLoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "io error: {e}"),
            Self::Parse(s) => write!(f, "parse error: {s}"),
            Self::SchemaNewer { found, supported } => write!(
                f,
                "installed.toml schema_version {found} is newer than supported {supported}; \
                 downgrade is not supported",
            ),
        }
    }
}

impl std::error::Error for InstallLoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for InstallLoadError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

/// Compute the `.bak` sibling path of `installed_path`.
///
/// Appends `.bak` to the existing file name (preserving any prior extension);
/// for example `/x/installed.toml` -> `/x/installed.toml.bak`. If `path` has
/// no file-name component (e.g. `/`) returns the path with `.bak` appended
/// as a final segment.
#[must_use]
pub fn backup_path(installed_path: &Path) -> PathBuf {
    installed_path.file_name().map_or_else(
        || {
            let mut p = installed_path.as_os_str().to_os_string();
            p.push(".bak");
            PathBuf::from(p)
        },
        |name| {
            let mut new_name = name.to_os_string();
            new_name.push(".bak");
            installed_path.with_file_name(new_name)
        },
    )
}

fn link_or_copy(src: &Path, dst: &Path) -> Result<(), InstallIoError> {
    // Try a hardlink first (cheap, atomic on POSIX). If the filesystem refuses
    // (cross-device, file already exists, unsupported FS), fall back to a
    // byte-copy — the backup semantics are identical, only the disk-cost
    // differs.
    if std::fs::hard_link(src, dst).is_ok() {
        return Ok(());
    }
    std::fs::copy(src, dst)
        .map(|_| ())
        .map_err(|e| InstallIoError::BackupFailed {
            reason: format!("hardlink and copy both failed: {e}"),
        })
}

/// Atomically write `value` to `path` with a one-shot `.bak` of the prior
/// contents.
///
/// Sequence:
/// 1. Serialize to a sibling `<path>.tmp` and fsync the temp file.
/// 2. If `<path>` exists and `<path>.bak` does NOT, hardlink-or-copy the
///    current `<path>` to `<path>.bak`. A pre-existing backup is preserved
///    (we never overwrite the most recent known-good copy).
/// 3. `rename(<path>.tmp, <path>)` — atomic on POSIX, best-effort on Windows.
/// 4. On Unix, fsync the parent directory so the rename is durable.
///
/// # Errors
/// Returns [`InstallIoError`] if any step fails. On rename failure the temp
/// file is best-effort cleaned up; on success no `.tmp` remains.
///
/// # Panics
/// Does not panic for any input the public API produces; the internal
/// `expect` is justified because [`InstalledToml`] contains only primitive
/// types and `Vec<String>`, for which `toml_edit::ser::to_string` is
/// infallible. The carve-out matches [`InstalledToml::write_atomic`] and is
/// tracked in `docs/coverage-exclusions.md`.
pub fn save_atomic(path: &Path, value: &InstalledToml) -> Result<(), InstallIoError> {
    let parent = path.parent().ok_or(InstallIoError::MissingParent)?;
    // For paths like "installed.toml" the parent is "", which we treat as the
    // current working directory (no mkdir needed).
    if !parent.as_os_str().is_empty() {
        std::fs::create_dir_all(parent)?;
    }

    // `InstalledToml` contains only primitives and `Vec<String>`; toml_edit
    // serialization is infallible for this shape (matches the carve-out used
    // by `InstalledToml::write_atomic` and tracked in
    // `docs/coverage-exclusions.md`).
    #[allow(
        clippy::expect_used,
        reason = "InstalledToml serialization is infallible (primitives only)"
    )]
    let rendered =
        toml_edit::ser::to_string(value).expect("InstalledToml serialization is infallible");

    let tmp = {
        let mut name = path
            .file_name()
            .ok_or(InstallIoError::MissingParent)?
            .to_os_string();
        name.push(".tmp");
        path.with_file_name(name)
    };

    // Write + fsync the temp.
    {
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&tmp)?;
        f.write_all(rendered.as_bytes())?;
        f.sync_all()?;
    }

    // Stage the backup of the current live file (one-shot — never clobber an
    // existing .bak).
    let bak = backup_path(path);
    if path.exists() && !bak.exists() {
        if let Err(e) = link_or_copy(path, &bak) {
            // Clean up the temp before bubbling.
            let _ = std::fs::remove_file(&tmp);
            return Err(e);
        }
    }

    // Atomic rename.
    if let Err(e) = std::fs::rename(&tmp, path) {
        let _ = std::fs::remove_file(&tmp);
        return Err(InstallIoError::Io(e));
    }

    // Fsync parent dir (Unix only).
    #[cfg(unix)]
    {
        if let Ok(dir) = std::fs::File::open(parent) {
            let _ = dir.sync_all();
        }
    }

    Ok(())
}

/// Load `path` and apply forward-only schema migration.
///
/// # Back-compat policy
/// - Schema bumps within a major release are **additive only**: new optional
///   fields with `#[serde(default)]`. A reader from a newer minor build
///   always understands an older file.
/// - If the file declares `schema_version > CURRENT_SCHEMA_VERSION`, this
///   returns [`InstallLoadError::SchemaNewer`]. Downgrade is unsupported —
///   the user must reinstall or upgrade.
/// - If the file declares `schema_version < CURRENT_SCHEMA_VERSION`, the
///   record is read with current defaults filling absent fields and then
///   rewritten via [`save_atomic`]. This rewrite hook is where future
///   per-version migration steps slot in (each step adapts shape, then
///   bumps the version field, then `save_atomic` persists).
/// - If the file already matches `CURRENT_SCHEMA_VERSION`, no rewrite.
///
/// # Errors
/// Returns [`InstallLoadError`] on io / parse / version-mismatch failure.
pub fn load_with_migration(path: &Path) -> Result<InstalledToml, InstallLoadError> {
    let raw = std::fs::read_to_string(path)?;
    let parsed: InstalledToml =
        toml_edit::de::from_str(&raw).map_err(|e| InstallLoadError::Parse(e.to_string()))?;

    if parsed.schema_version > CURRENT_SCHEMA_VERSION {
        return Err(InstallLoadError::SchemaNewer {
            found: parsed.schema_version,
            supported: CURRENT_SCHEMA_VERSION,
        });
    }
    if parsed.schema_version < CURRENT_SCHEMA_VERSION {
        // No older versions exist yet. When v2 ships, the per-version
        // migration ladder will run here (v1 -> v2 -> ... -> CURRENT),
        // each step adapting the in-memory shape and bumping the field,
        // and the final save_atomic call below persists the migrated form.
        let mut migrated = parsed;
        migrated.schema_version = CURRENT_SCHEMA_VERSION;
        save_atomic(path, &migrated).map_err(|e| match e {
            InstallIoError::Io(io) => InstallLoadError::Io(io),
            other => InstallLoadError::Parse(other.to_string()),
        })?;
        return Ok(migrated);
    }

    Ok(parsed)
}

/// Restore `<path>.bak` over `<path>`, returning `true` if a backup existed.
///
/// Uses `rename` (atomic on POSIX). Returns `Ok(false)` if no backup file is
/// present; this is the steady-state success path (nothing to roll back).
///
/// # Errors
/// Returns [`InstallIoError::Io`] if the rename fails.
pub fn restore_backup(path: &Path) -> Result<bool, InstallIoError> {
    let bak = backup_path(path);
    if !bak.exists() {
        return Ok(false);
    }
    std::fs::rename(&bak, path)?;
    Ok(true)
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

    // ---- backup / save_atomic / load_with_migration tests ----

    fn fixture_record() -> InstalledToml {
        let p = fixture_probe();
        let now = OffsetDateTime::from_unix_timestamp(1_700_000_000).unwrap();
        InstalledToml::new(&p, Tier::High, now)
    }

    #[test]
    fn current_schema_version_matches_internal_constant() {
        assert_eq!(CURRENT_SCHEMA_VERSION, SCHEMA_VERSION);
    }

    #[test]
    fn backup_path_posix_style() {
        let p = Path::new("/x/installed.toml");
        assert_eq!(backup_path(p), PathBuf::from("/x/installed.toml.bak"));
    }

    #[test]
    fn backup_path_windows_style() {
        // Use a relative path with backslashes — exercised on all platforms
        // by file_name semantics on `\`-separated input. We use a
        // forward-slash subdir to remain platform-portable; the contract is
        // simply "append .bak to the file name".
        let p = Path::new("dir/installed.toml");
        assert_eq!(backup_path(p), PathBuf::from("dir/installed.toml.bak"));
        // Path with no file_name (root): falls back to appending to the
        // whole path.
        let r = Path::new("/");
        let got = backup_path(r);
        assert!(got.as_os_str().to_string_lossy().ends_with(".bak"));
    }

    #[test]
    fn save_atomic_writes_file_and_leaves_no_tmp() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("installed.toml");
        let rec = fixture_record();
        save_atomic(&path, &rec).unwrap();
        assert!(path.exists());
        let tmp_sibling = tmp.path().join("installed.toml.tmp");
        assert!(!tmp_sibling.exists(), "no .tmp residue");
    }

    #[test]
    fn save_atomic_creates_bak_if_file_exists_and_no_bak() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("installed.toml");
        let rec = fixture_record();
        // First save: no prior file, no .bak.
        save_atomic(&path, &rec).unwrap();
        assert!(!backup_path(&path).exists(), "no bak on first save");
        // Second save: prior file exists, no .bak yet -> .bak is created.
        save_atomic(&path, &rec).unwrap();
        assert!(backup_path(&path).exists(), "bak created on second save");
    }

    #[test]
    fn save_atomic_does_not_overwrite_existing_bak() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("installed.toml");
        let bak = backup_path(&path);
        // Seed a sentinel bak; save_atomic must preserve it byte-for-byte.
        std::fs::write(&path, b"original").unwrap();
        std::fs::write(&bak, b"PRE-EXISTING-BAK").unwrap();
        let rec = fixture_record();
        save_atomic(&path, &rec).unwrap();
        let bak_contents = std::fs::read(&bak).unwrap();
        assert_eq!(bak_contents, b"PRE-EXISTING-BAK");
    }

    #[test]
    fn save_atomic_succeeds_without_prior_file() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("sub").join("installed.toml");
        let rec = fixture_record();
        save_atomic(&path, &rec).unwrap();
        assert!(path.exists());
        assert!(!backup_path(&path).exists(), "no .bak when no prior file");
    }

    #[test]
    fn save_atomic_errors_on_missing_parent() {
        let rec = fixture_record();
        let no_parent = Path::new("");
        let err = save_atomic(no_parent, &rec).unwrap_err();
        match err {
            InstallIoError::MissingParent => {}
            other => panic!("expected MissingParent, got {other:?}"),
        }
    }

    #[test]
    fn restore_backup_returns_false_for_missing_backup() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("installed.toml");
        assert!(!restore_backup(&path).unwrap());
    }

    #[test]
    fn restore_backup_swaps_and_returns_true() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("installed.toml");
        let bak = backup_path(&path);
        std::fs::write(&path, b"NEW").unwrap();
        std::fs::write(&bak, b"OLD").unwrap();
        assert!(restore_backup(&path).unwrap());
        let restored = std::fs::read(&path).unwrap();
        assert_eq!(restored, b"OLD");
        assert!(!bak.exists(), "backup consumed by restore");
    }

    #[test]
    fn load_with_migration_happy_path_v1() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("installed.toml");
        let rec = fixture_record();
        save_atomic(&path, &rec).unwrap();
        let got = load_with_migration(&path).unwrap();
        assert_eq!(got, rec);
    }

    #[test]
    fn load_with_migration_errors_on_newer_schema() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("installed.toml");
        let rec = fixture_record();
        save_atomic(&path, &rec).unwrap();
        // Hand-edit the schema_version up to 999.
        let raw = std::fs::read_to_string(&path).unwrap();
        let bumped = raw.replace("schema_version = 1", "schema_version = 999");
        std::fs::write(&path, bumped).unwrap();
        let err = load_with_migration(&path).unwrap_err();
        match err {
            InstallLoadError::SchemaNewer { found, supported } => {
                assert_eq!(found, 999);
                assert_eq!(supported, CURRENT_SCHEMA_VERSION);
            }
            other => panic!("expected SchemaNewer, got {other:?}"),
        }
    }

    #[test]
    fn load_with_migration_defaults_missing_schema_version() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("installed.toml");
        let rec = fixture_record();
        save_atomic(&path, &rec).unwrap();
        // Strip the schema_version line so serde must fall back to the
        // default (1).
        let raw = std::fs::read_to_string(&path).unwrap();
        let stripped: String = raw
            .lines()
            .filter(|l| !l.trim_start().starts_with("schema_version"))
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(&path, stripped).unwrap();
        let got = load_with_migration(&path).unwrap();
        assert_eq!(got.schema_version, CURRENT_SCHEMA_VERSION);
    }

    #[test]
    fn save_then_load_with_migration_roundtrip() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("installed.toml");
        let rec = fixture_record();
        save_atomic(&path, &rec).unwrap();
        let back = load_with_migration(&path).unwrap();
        assert_eq!(back, rec);
    }

    #[test]
    fn backup_then_restore_lifecycle_preserves_original() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("installed.toml");
        let original = fixture_record();
        save_atomic(&path, &original).unwrap();

        // Second save with a mutated record creates the .bak from the
        // original.
        let p = fixture_probe();
        let now = OffsetDateTime::from_unix_timestamp(1_700_000_000).unwrap();
        let mutated = InstalledToml::new(&p, Tier::Low, now);
        save_atomic(&path, &mutated).unwrap();
        assert!(backup_path(&path).exists());

        // The live file is now the mutated one.
        let live = load_with_migration(&path).unwrap();
        assert_eq!(live, mutated);

        // Restore -> live file is back to the original record.
        assert!(restore_backup(&path).unwrap());
        let restored = load_with_migration(&path).unwrap();
        assert_eq!(restored, original);
    }

    #[test]
    fn load_with_migration_propagates_io_error_for_missing_file() {
        let tmp = TempDir::new().unwrap();
        let missing = tmp.path().join("no-such.toml");
        let err = load_with_migration(&missing).unwrap_err();
        match err {
            InstallLoadError::Io(_) => {}
            other => panic!("expected Io, got {other:?}"),
        }
    }

    #[test]
    fn load_with_migration_propagates_parse_error() {
        let tmp = TempDir::new().unwrap();
        let bad = tmp.path().join("bad.toml");
        std::fs::write(&bad, b"::: not toml").unwrap();
        let err = load_with_migration(&bad).unwrap_err();
        match err {
            InstallLoadError::Parse(_) => {}
            other => panic!("expected Parse, got {other:?}"),
        }
    }

    #[test]
    fn install_io_error_display_and_source() {
        let mp = InstallIoError::MissingParent;
        assert!(mp.to_string().contains("no parent"));
        assert!(std::error::Error::source(&mp).is_none());

        let bf = InstallIoError::BackupFailed { reason: "x".into() };
        assert!(bf.to_string().contains("backup failed"));
        assert!(std::error::Error::source(&bf).is_none());

        let io = InstallIoError::from(std::io::Error::other("boom"));
        assert!(io.to_string().contains("io error"));
        assert!(std::error::Error::source(&io).is_some());
    }

    #[cfg(unix)]
    #[test]
    fn save_atomic_failure_cleans_up_tmp() {
        // Pre-create target as a non-empty directory; the backup-or-rename
        // path fails (depending on platform: backup of a directory fails on
        // macOS; rename of file over non-empty dir fails on Linux). In
        // either case, the .tmp is cleaned up.
        let tmp = TempDir::new().unwrap();
        let target = tmp.path().join("installed.toml");
        std::fs::create_dir(&target).unwrap();
        std::fs::write(target.join("blocker"), b"x").unwrap();
        let rec = fixture_record();
        let err = save_atomic(&target, &rec).unwrap_err();
        match err {
            InstallIoError::Io(_) | InstallIoError::BackupFailed { .. } => {}
            other @ InstallIoError::MissingParent => {
                panic!("expected Io or BackupFailed, got {other:?}")
            }
        }
        let tmp_sibling = tmp.path().join("installed.toml.tmp");
        assert!(!tmp_sibling.exists(), "tmp cleaned up on failure");
    }

    #[test]
    fn load_with_migration_rewrites_lower_schema_version() {
        // No older version exists yet; simulate one by hand-editing
        // schema_version to 0 (a value below CURRENT). The migration path
        // bumps it to CURRENT and rewrites via save_atomic.
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("installed.toml");
        let rec = fixture_record();
        save_atomic(&path, &rec).unwrap();
        let raw = std::fs::read_to_string(&path).unwrap();
        let downgraded = raw.replace("schema_version = 1", "schema_version = 0");
        std::fs::write(&path, downgraded).unwrap();
        let got = load_with_migration(&path).unwrap();
        assert_eq!(got.schema_version, CURRENT_SCHEMA_VERSION);
        // And it was persisted.
        let reread = std::fs::read_to_string(&path).unwrap();
        assert!(reread.contains(&format!("schema_version = {CURRENT_SCHEMA_VERSION}")));
    }

    #[test]
    fn install_load_error_display_and_source() {
        let s = InstallLoadError::SchemaNewer {
            found: 9,
            supported: 1,
        };
        assert!(s.to_string().contains("newer than supported"));
        assert!(std::error::Error::source(&s).is_none());

        let p = InstallLoadError::Parse("oops".into());
        assert!(p.to_string().contains("parse error"));
        assert!(std::error::Error::source(&p).is_none());

        let io = InstallLoadError::from(std::io::Error::other("boom"));
        assert!(io.to_string().contains("io error"));
        assert!(std::error::Error::source(&io).is_some());
    }
}
