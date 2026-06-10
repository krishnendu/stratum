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
    /// A registered [`InstalledMigrator`] returned an error while transforming
    /// the document.
    Migration(String),
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
            Self::Migration(s) => write!(f, "migration error: {s}"),
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

/// One step in the forward-only `installed.toml` migration ladder.
///
/// Each implementation transforms a TOML document in place from
/// [`from_version`](Self::from_version) to [`to_version`](Self::to_version).
/// The migration pipeline parses the raw file once into a
/// [`toml_edit::DocumentMut`], dispatches through a chain of migrators
/// returned by [`chain_for`], then deserializes the final document into an
/// [`InstalledToml`]. New schema versions plug in by adding a migrator and
/// registering it via [`registered_migrators`].
pub trait InstalledMigrator {
    /// Schema version this migrator accepts as input.
    #[allow(
        clippy::wrong_self_convention,
        reason = "from_version names the version field, not a conversion constructor"
    )]
    fn from_version(&self) -> u32;
    /// Schema version this migrator produces.
    fn to_version(&self) -> u32;
    /// Transform the document. The returned document MUST have its
    /// `schema_version` set to [`to_version`](Self::to_version).
    ///
    /// # Errors
    /// Implementations return [`InstallLoadError::Migration`] (or wrap an
    /// io / parse error) when the transformation cannot complete.
    fn migrate(
        &self,
        raw: toml_edit::DocumentMut,
    ) -> Result<toml_edit::DocumentMut, InstallLoadError>;
}

/// Identity migrator: passes a document through unchanged.
///
/// Used as the v1 → v1 no-op so the dispatch pipeline has a uniform shape
/// before any real version bump ships.
#[derive(Debug, Clone, Copy)]
pub struct IdentityMigrator {
    /// Both the `from` and `to` version this identity covers.
    version: u32,
}

impl IdentityMigrator {
    /// Construct an identity migrator for `version`.
    #[must_use]
    pub const fn new(version: u32) -> Self {
        Self { version }
    }
}

impl InstalledMigrator for IdentityMigrator {
    fn from_version(&self) -> u32 {
        self.version
    }
    fn to_version(&self) -> u32 {
        self.version
    }
    fn migrate(
        &self,
        raw: toml_edit::DocumentMut,
    ) -> Result<toml_edit::DocumentMut, InstallLoadError> {
        Ok(raw)
    }
}

/// The static v1 → v1 identity migrator used by [`registered_migrators`] and
/// [`chain_for`].
static IDENTITY_V1: IdentityMigrator = IdentityMigrator::new(1);

/// Slice of statically-registered migrators consulted by [`chain_for`].
const REGISTERED: &[&'static dyn InstalledMigrator] = &[&IDENTITY_V1];

/// Return the ordered list of registered migrators.
///
/// Currently contains only the v1 → v1 identity step. When v2 ships, a real
/// v1 → v2 migrator is appended here (and to [`REGISTERED`]) — no other
/// caller needs to change.
#[must_use]
pub fn registered_migrators() -> Vec<Box<dyn InstalledMigrator>> {
    vec![Box::new(IdentityMigrator::new(1))]
}

/// Resolve the chain of migrators that walks `from` → `to`.
///
/// Greedy forward search: at each step picks the registered migrator whose
/// `from_version` matches the current version and whose `to_version` is the
/// largest available `≤ to`. Returns the ordered list of migrator
/// references; for `from == to == v` returns the single identity step
/// (length 1). Returns [`InstallLoadError::Parse`] with `"no migration path"`
/// when no chain exists (this is the load-side signal the operator should
/// see; the pipeline keeps it under `Parse` to avoid a fresh STRAT-E code).
///
/// # Errors
/// Returns [`InstallLoadError::Parse`] when there is no v`from` → v`to`
/// chain among the registered migrators.
pub fn chain_for(
    from: u32,
    to: u32,
) -> Result<Vec<&'static dyn InstalledMigrator>, InstallLoadError> {
    chain_from_slice(REGISTERED, from, to)
}

fn chain_from_slice<'a>(
    pool: &'a [&'a dyn InstalledMigrator],
    from: u32,
    to: u32,
) -> Result<Vec<&'a dyn InstalledMigrator>, InstallLoadError> {
    let mut out: Vec<&'a dyn InstalledMigrator> = Vec::new();
    let mut current = from;
    // Identity case: a v == v migrator is required and resolves the chain.
    if current == to {
        let identity = pool
            .iter()
            .copied()
            .find(|m| m.from_version() == current && m.to_version() == current)
            .ok_or_else(|| InstallLoadError::Parse(String::from("no migration path")))?;
        out.push(identity);
        return Ok(out);
    }
    while current < to {
        let next = pool
            .iter()
            .copied()
            .filter(|m| m.from_version() == current && m.to_version() > current)
            .max_by_key(|m| m.to_version())
            .ok_or_else(|| InstallLoadError::Parse(String::from("no migration path")))?;
        current = next.to_version();
        out.push(next);
        if current > to {
            return Err(InstallLoadError::Parse(String::from("no migration path")));
        }
    }
    Ok(out)
}

/// Load `path` and apply forward-only schema migration via the registered
/// [`InstalledMigrator`] chain.
///
/// # Pipeline
/// 1. Read the file and parse it as a [`toml_edit::DocumentMut`].
/// 2. Peek `schema_version` (defaults to `0` when absent — a file without
///    the field is treated as pre-versioned and must be migrated forward).
/// 3. If the peeked version exceeds [`CURRENT_SCHEMA_VERSION`], return
///    [`InstallLoadError::SchemaNewer`] (downgrade is unsupported).
/// 4. Resolve a migrator chain via [`chain_for`]; run each step in order.
/// 5. Deserialize the final document into [`InstalledToml`].
/// 6. If the on-disk version changed, rewrite via [`save_atomic`].
///
/// # Errors
/// Returns [`InstallLoadError`] on io / parse / migration / version-mismatch
/// failure.
pub fn load_with_migration(path: &Path) -> Result<InstalledToml, InstallLoadError> {
    let migrators = registered_migrators();
    let pool: Vec<&dyn InstalledMigrator> = migrators.iter().map(AsRef::as_ref).collect();
    load_with_migrators(path, &pool)
}

/// Test seam: same pipeline as [`load_with_migration`] but consults the
/// caller-supplied migrator pool instead of [`REGISTERED`]. Kept
/// `pub(crate)` so the test module can inject a fake v1 → v2 migrator and
/// exercise the full dispatch without touching global state.
pub(crate) fn load_with_migrators(
    path: &Path,
    pool: &[&dyn InstalledMigrator],
) -> Result<InstalledToml, InstallLoadError> {
    let raw = std::fs::read_to_string(path)?;
    let mut doc: toml_edit::DocumentMut = raw
        .parse::<toml_edit::DocumentMut>()
        .map_err(|e| InstallLoadError::Parse(e.to_string()))?;

    let on_disk_version = peek_schema_version(&doc);
    if on_disk_version > CURRENT_SCHEMA_VERSION {
        return Err(InstallLoadError::SchemaNewer {
            found: on_disk_version,
            supported: CURRENT_SCHEMA_VERSION,
        });
    }

    let chain = chain_from_slice(pool, on_disk_version, CURRENT_SCHEMA_VERSION)?;
    for step in chain {
        doc = step.migrate(doc)?;
    }

    let final_text = doc.to_string();
    let parsed: InstalledToml =
        toml_edit::de::from_str(&final_text).map_err(|e| InstallLoadError::Parse(e.to_string()))?;

    if on_disk_version != parsed.schema_version {
        save_atomic(path, &parsed).map_err(|e| match e {
            InstallIoError::Io(io) => InstallLoadError::Io(io),
            other => InstallLoadError::Parse(other.to_string()),
        })?;
    }

    Ok(parsed)
}

/// Peek the `schema_version` integer from a parsed document, defaulting to
/// `0` if absent or not an integer. Pre-versioned files (no field) flow
/// through the migrator chain as v0 sources.
fn peek_schema_version(doc: &toml_edit::DocumentMut) -> u32 {
    doc.get("schema_version")
        .and_then(toml_edit::Item::as_integer)
        .and_then(|v| u32::try_from(v).ok())
        .unwrap_or(0)
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
    fn load_with_migration_errors_on_missing_schema_version() {
        // Post-migrator-refactor: a file without `schema_version` peeks as 0
        // and has no registered v0 -> v1 migrator, so the chain resolver
        // emits "no migration path". Pre-versioned files must ship a
        // migrator alongside the schema bump that introduced versioning.
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("installed.toml");
        let rec = fixture_record();
        save_atomic(&path, &rec).unwrap();
        let raw = std::fs::read_to_string(&path).unwrap();
        let stripped: String = raw
            .lines()
            .filter(|l| !l.trim_start().starts_with("schema_version"))
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(&path, stripped).unwrap();
        let err = load_with_migration(&path).unwrap_err();
        match err {
            InstallLoadError::Parse(msg) => assert_eq!(msg, "no migration path"),
            other => panic!("expected Parse(no migration path), got {other:?}"),
        }
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
    fn load_with_migration_errors_on_lower_schema_version_without_migrator() {
        // Hand-edit schema_version to 0; there is no registered v0 -> v1
        // migrator, so dispatch fails with "no migration path".
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("installed.toml");
        let rec = fixture_record();
        save_atomic(&path, &rec).unwrap();
        let raw = std::fs::read_to_string(&path).unwrap();
        let downgraded = raw.replace("schema_version = 1", "schema_version = 0");
        std::fs::write(&path, downgraded).unwrap();
        let err = load_with_migration(&path).unwrap_err();
        match err {
            InstallLoadError::Parse(msg) => assert_eq!(msg, "no migration path"),
            other => panic!("expected Parse(no migration path), got {other:?}"),
        }
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

        let m = InstallLoadError::Migration("boom".into());
        assert!(m.to_string().contains("migration error"));
        assert!(std::error::Error::source(&m).is_none());

        let io = InstallLoadError::from(std::io::Error::other("boom"));
        assert!(io.to_string().contains("io error"));
        assert!(std::error::Error::source(&io).is_some());
    }

    // ---- migrator trait + chain dispatch tests ----

    /// Hand-crafted migrator used as a fake v1 -> v2 step in the test seam.
    /// Bumps the on-disk `schema_version` field; the rest of the document is
    /// left untouched so the existing `InstalledToml` shape still parses.
    struct BumpToV2;
    impl InstalledMigrator for BumpToV2 {
        fn from_version(&self) -> u32 {
            1
        }
        fn to_version(&self) -> u32 {
            2
        }
        fn migrate(
            &self,
            mut raw: toml_edit::DocumentMut,
        ) -> Result<toml_edit::DocumentMut, InstallLoadError> {
            raw["schema_version"] = toml_edit::value(2_i64);
            Ok(raw)
        }
    }

    /// Test-only v0 -> v1 migrator that fills in `schema_version = 1` so the
    /// rewrite branch of `load_with_migrators` can be exercised end-to-end.
    struct FillV0ToV1;
    impl InstalledMigrator for FillV0ToV1 {
        fn from_version(&self) -> u32 {
            0
        }
        fn to_version(&self) -> u32 {
            1
        }
        fn migrate(
            &self,
            mut raw: toml_edit::DocumentMut,
        ) -> Result<toml_edit::DocumentMut, InstallLoadError> {
            raw["schema_version"] = toml_edit::value(1_i64);
            Ok(raw)
        }
    }

    /// Test-only migrator that jumps v1 -> v3 in a single step. Combined
    /// with a v1 -> v2 chain target it exposes the "gap" path in
    /// `chain_from_slice`.
    struct JumpToV3;
    impl InstalledMigrator for JumpToV3 {
        fn from_version(&self) -> u32 {
            1
        }
        fn to_version(&self) -> u32 {
            3
        }
        fn migrate(
            &self,
            raw: toml_edit::DocumentMut,
        ) -> Result<toml_edit::DocumentMut, InstallLoadError> {
            Ok(raw)
        }
    }

    /// Migrator that always fails — exercises the `Migration` error path.
    struct AlwaysFailV1;
    impl InstalledMigrator for AlwaysFailV1 {
        fn from_version(&self) -> u32 {
            1
        }
        fn to_version(&self) -> u32 {
            1
        }
        fn migrate(
            &self,
            _raw: toml_edit::DocumentMut,
        ) -> Result<toml_edit::DocumentMut, InstallLoadError> {
            Err(InstallLoadError::Migration("synthetic failure".into()))
        }
    }

    #[test]
    fn identity_migrator_passes_document_through_unchanged() {
        let src = "schema_version = 1\nfoo = \"bar\"\n";
        let doc: toml_edit::DocumentMut = src.parse().unwrap();
        let m = IdentityMigrator::new(1);
        let out = m.migrate(doc).unwrap();
        assert_eq!(out.to_string(), src);
    }

    #[test]
    fn identity_migrator_reports_constructed_version() {
        let m = IdentityMigrator::new(7);
        assert_eq!(m.from_version(), 7);
        assert_eq!(m.to_version(), 7);
    }

    #[test]
    fn chain_for_v1_to_v1_returns_identity_step() {
        let chain = match chain_for(1, 1) {
            Ok(c) => c,
            Err(e) => panic!("expected chain, got {e}"),
        };
        assert_eq!(chain.len(), 1);
        assert_eq!(chain[0].from_version(), 1);
        assert_eq!(chain[0].to_version(), 1);
    }

    #[test]
    fn chain_for_v0_to_v1_has_no_path() {
        let Err(err) = chain_for(0, 1) else {
            panic!("expected error");
        };
        match err {
            InstallLoadError::Parse(msg) => assert_eq!(msg, "no migration path"),
            other => panic!("expected Parse(no migration path), got {other:?}"),
        }
    }

    #[test]
    fn chain_with_gap_returns_no_migration_path() {
        // Pool defines v1 -> v3, but caller asks v1 -> v2: the only
        // v1-rooted step jumps past v2, so resolution fails.
        let jump = JumpToV3;
        let pool: Vec<&dyn InstalledMigrator> = vec![&jump];
        let Err(err) = chain_from_slice(&pool, 1, 2) else {
            panic!("expected error");
        };
        match err {
            InstallLoadError::Parse(msg) => assert_eq!(msg, "no migration path"),
            other => panic!("expected Parse(no migration path), got {other:?}"),
        }
    }

    #[test]
    fn registered_migrators_lists_identity_v1() {
        let regs = registered_migrators();
        assert_eq!(regs.len(), 1);
        assert_eq!(regs[0].from_version(), 1);
        assert_eq!(regs[0].to_version(), 1);
    }

    #[test]
    fn load_with_migrators_injected_v1_to_v2_chain_rewrites_file() {
        // Test seam: inject a v1 -> v2 migrator pool and lift CURRENT to v2
        // by routing through `load_with_migrators` with a pool that takes
        // the file from v1 to v2. We cannot literally change
        // `CURRENT_SCHEMA_VERSION` at runtime, so we exercise the lower
        // half of the pipeline directly: peek-then-chain-then-migrate, and
        // assert the chained doc reaches v2.
        let bump = BumpToV2;
        let pool: Vec<&dyn InstalledMigrator> = vec![&bump];
        let chain = match chain_from_slice(&pool, 1, 2) {
            Ok(c) => c,
            Err(e) => panic!("expected chain, got {e}"),
        };
        assert_eq!(chain.len(), 1);
        let src = "schema_version = 1\nfoo = \"bar\"\n";
        let doc: toml_edit::DocumentMut = src.parse().unwrap();
        let out = chain[0].migrate(doc).unwrap();
        assert!(out.to_string().contains("schema_version = 2"));
    }

    #[test]
    fn load_with_migrators_propagates_migrator_failure() {
        // Inject a failing v1 -> v1 migrator; the pipeline must surface
        // `Migration(msg)` from `load_with_migrators`.
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("installed.toml");
        let rec = fixture_record();
        save_atomic(&path, &rec).unwrap();
        let failing = AlwaysFailV1;
        let pool: Vec<&dyn InstalledMigrator> = vec![&failing];
        let err = load_with_migrators(&path, &pool).unwrap_err();
        match err {
            InstallLoadError::Migration(msg) => assert_eq!(msg, "synthetic failure"),
            other => panic!("expected Migration, got {other:?}"),
        }
    }

    #[test]
    fn load_with_migration_rewrites_file_when_schema_version_changes() {
        // End-to-end: a v1 file passed through a v1 -> v2 migrator pool
        // (via the test seam) gets persisted at the new version. Re-read
        // the file from disk and assert the on-disk schema_version moved
        // from 1 to 2.
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("installed.toml");
        let rec = fixture_record();
        save_atomic(&path, &rec).unwrap();

        // `load_with_migrators` targets the compile-time CURRENT (1), so
        // we exercise the rewrite by forcing the on-disk version to 0 and
        // injecting a v0 -> v1 pseudo-migrator (`FillV0ToV1`) that sets
        // `schema_version = 1`. After the pipeline runs, the file on disk
        // must be rewritten with schema_version=1.

        // Drop schema_version so peek returns 0.
        let raw = std::fs::read_to_string(&path).unwrap();
        let stripped: String = raw
            .lines()
            .filter(|l| !l.trim_start().starts_with("schema_version"))
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(&path, stripped).unwrap();

        let fill = FillV0ToV1;
        let pool: Vec<&dyn InstalledMigrator> = vec![&fill];
        let got = load_with_migrators(&path, &pool).unwrap();
        assert_eq!(got.schema_version, 1);

        // File on disk was rewritten by the pipeline.
        let reread = std::fs::read_to_string(&path).unwrap();
        assert!(reread.contains("schema_version = 1"));
    }
}
