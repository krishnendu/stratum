//! Sandbox-profile resolver.
//!
//! Phase 3 v2 scaffold per `plan/15-sandbox-and-process.md` §4. Takes a
//! `SandboxProfile`, a `CapabilityMatrix`, the workspace root, and the
//! chosen backend, and produces a concrete `SandboxLaunchSpec` the
//! future launcher layer can drive.
//!
//! This module is composition-only: it does not duplicate the
//! `SandboxBackend` enum from `sandbox.rs`, it mirrors the variants
//! into a `BackendChoice` keyed for the launcher surface and
//! co-exists alongside the detection report.

use std::collections::{BTreeMap, BTreeSet};
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::sandbox::SandboxBackend;
use crate::sandbox_profile::{NetPolicy, SandboxProfile};
use crate::tools::CapabilityMatrix;

/// Mount mode inside the launched sandbox.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MountMode {
    /// Read-only bind.
    ReadOnly,
    /// Read-write bind.
    ReadWrite,
    /// Tempfs of the given size (MiB).
    TempFs {
        /// Size in MiB.
        size_mib: u64,
    },
}

/// One resolved mount: a host path projected to a guest path with a mode.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ResolvedMount {
    /// Host path. Must be absolute.
    pub host: PathBuf,
    /// Guest (in-sandbox) path.
    pub guest: PathBuf,
    /// Mount mode.
    pub mode: MountMode,
}

/// Resolved network policy.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResolvedNet {
    /// No network at all.
    Off,
    /// Loopback only.
    Loopback,
    /// Hosts allowlist.
    Hosts {
        /// Allowed DNS names / hostnames.
        allow: BTreeSet<String>,
    },
}

/// Backend choice fed to the launcher layer. Mirrors `SandboxBackend`
/// without taking a direct dependency on its rename rules; the two are
/// kept in lock-step by `From` impls.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendChoice {
    /// `bubblewrap` user-namespace sandbox.
    Bwrap,
    /// macOS `sandbox-exec`.
    SandboxExec,
    /// Windows Job Object + `AppContainer`.
    WindowsJob,
    /// No isolation (passthrough).
    Passthrough,
}

impl From<SandboxBackend> for BackendChoice {
    fn from(b: SandboxBackend) -> Self {
        match b {
            SandboxBackend::Bwrap => Self::Bwrap,
            SandboxBackend::SandboxExec => Self::SandboxExec,
            SandboxBackend::WindowsJob => Self::WindowsJob,
            SandboxBackend::Passthrough => Self::Passthrough,
        }
    }
}

impl From<BackendChoice> for SandboxBackend {
    fn from(b: BackendChoice) -> Self {
        match b {
            BackendChoice::Bwrap => Self::Bwrap,
            BackendChoice::SandboxExec => Self::SandboxExec,
            BackendChoice::WindowsJob => Self::WindowsJob,
            BackendChoice::Passthrough => Self::Passthrough,
        }
    }
}

/// Concrete launch spec: everything the launcher needs to drive a
/// child under one backend.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SandboxLaunchSpec {
    /// Resolved mount list, sorted by guest path.
    pub mounts: Vec<ResolvedMount>,
    /// Resolved network policy.
    pub net: ResolvedNet,
    /// Environment variables for the child (sorted by key).
    pub env: BTreeMap<String, String>,
    /// Flat set of capability keys the matrix allows.
    pub allowed_caps: BTreeSet<String>,
    /// Flat set of capability keys explicitly denied. Wins over allow.
    pub denied_caps: BTreeSet<String>,
    /// Working directory the child starts in (guest path).
    pub working_dir: PathBuf,
    /// CPU quota percent (1..=100), if any.
    pub cpu_quota_pct: Option<u8>,
    /// Memory limit in MiB, if any.
    pub memory_limit_mib: Option<u64>,
    /// Backend selected for the launch.
    pub backend: BackendChoice,
}

impl SandboxLaunchSpec {
    /// Deterministic 64-bit hash of the canonical Serde form of the
    /// spec. The future caching layer uses this to detect spec drift.
    #[must_use]
    pub fn stable_hash(&self) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        match serde_json::to_vec(self) {
            Ok(canonical) => canonical.hash(&mut hasher),
            Err(_) => 0u8.hash(&mut hasher),
        }
        hasher.finish()
    }
}

/// Errors and warnings emitted by `resolve`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResolveError {
    /// A mount's host path was not absolute.
    NonAbsoluteHostPath(PathBuf),
    /// A capability key appears as both allowed and denied.
    ConflictingCapability {
        /// The conflicting key.
        key: String,
    },
    /// The backend was not recognized (reserved for forward-compat).
    UnknownBackend,
    /// Two mounts collide on the same guest path.
    MountCollision {
        /// The duplicated guest path.
        guest: PathBuf,
    },
}

impl std::fmt::Display for ResolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonAbsoluteHostPath(p) => {
                write!(
                    f,
                    "sandbox mount host path is not absolute: {}",
                    p.display()
                )
            }
            Self::ConflictingCapability { key } => {
                write!(f, "capability `{key}` is both allowed and denied")
            }
            Self::UnknownBackend => f.write_str("unknown sandbox backend"),
            Self::MountCollision { guest } => {
                write!(
                    f,
                    "two sandbox mounts collide on guest path: {}",
                    guest.display()
                )
            }
        }
    }
}

impl std::error::Error for ResolveError {}

/// Default guest mount point for the workspace.
const WORKSPACE_GUEST: &str = "/workspace";

/// Sandbox env marker injected into every spec.
const SANDBOX_MARKER_KEY: &str = "STRATUM_SANDBOX";
const SANDBOX_MARKER_VALUE: &str = "1";

/// Resolve a profile into a concrete launch spec. Strict variant:
/// returns the first warning as an error.
///
/// # Errors
///
/// Returns the first `ResolveError` produced by validation: a
/// non-absolute host path, a guest-path mount collision, or a
/// capability key that appears in both allow and deny sets.
pub fn resolve(
    profile: &SandboxProfile,
    caps: &CapabilityMatrix,
    workspace_root: &Path,
    backend: BackendChoice,
) -> Result<SandboxLaunchSpec, ResolveError> {
    let (spec, warnings) = resolve_with_warnings(profile, caps, workspace_root, backend);
    if let Some(first) = warnings.into_iter().next() {
        return Err(first);
    }
    Ok(spec)
}

/// Lenient variant of `resolve`.
///
/// Returns the best-effort spec together with any warnings encountered.
/// Catastrophic errors (non-absolute host path, mount collision) still
/// surface as warnings here so the strict `resolve` can fail-fast on them.
#[must_use]
pub fn resolve_with_warnings(
    profile: &SandboxProfile,
    caps: &CapabilityMatrix,
    workspace_root: &Path,
    backend: BackendChoice,
) -> (SandboxLaunchSpec, Vec<ResolveError>) {
    let mut warnings: Vec<ResolveError> = Vec::new();

    // --- Mounts ---------------------------------------------------------
    let mut mounts: Vec<ResolvedMount> = Vec::with_capacity(profile.mounts.len() + 1);
    let workspace_guest = PathBuf::from(WORKSPACE_GUEST);

    // Track the guest paths the profile already covers, so a caller-supplied
    // /workspace override wins over the auto-prepended one.
    let profile_has_workspace = profile
        .mounts
        .iter()
        .any(|m| m.destination == workspace_guest);

    if !profile_has_workspace {
        mounts.push(ResolvedMount {
            host: workspace_root.to_path_buf(),
            guest: workspace_guest.clone(),
            mode: MountMode::ReadOnly,
        });
    }

    for m in &profile.mounts {
        mounts.push(ResolvedMount {
            host: m.source.clone(),
            guest: m.destination.clone(),
            mode: if m.read_only {
                MountMode::ReadOnly
            } else {
                MountMode::ReadWrite
            },
        });
    }

    // Validate absolute host paths.
    for m in &mounts {
        if !m.host.is_absolute() {
            warnings.push(ResolveError::NonAbsoluteHostPath(m.host.clone()));
        }
    }

    // Detect guest-path collisions before sorting (so reporting is stable).
    let mut seen: BTreeSet<PathBuf> = BTreeSet::new();
    for m in &mounts {
        if !seen.insert(m.guest.clone()) {
            warnings.push(ResolveError::MountCollision {
                guest: m.guest.clone(),
            });
        }
    }

    mounts.sort_by(|a, b| a.guest.cmp(&b.guest));

    // --- Net ------------------------------------------------------------
    let net = match &profile.net {
        NetPolicy::None => ResolvedNet::Off,
        NetPolicy::LocalhostOnly => ResolvedNet::Loopback,
        NetPolicy::AllowList(set) => ResolvedNet::Hosts { allow: set.clone() },
        NetPolicy::Full => ResolvedNet::Hosts {
            allow: BTreeSet::new(),
        },
    };

    // --- Capabilities ---------------------------------------------------
    // The matrix is an allow-list; the deny-list comes from entries the
    // matrix exposes via the conventional `!verb` prefix. This scaffold
    // treats every matrix entry as an allow, and lets the caller seed
    // denials via a future deny-prefix; conflict detection still flows.
    let mut allowed_caps: BTreeSet<String> = BTreeSet::new();
    let mut denied_caps: BTreeSet<String> = BTreeSet::new();
    for entry in caps.entries() {
        let raw = entry.as_str();
        if let Some(rest) = raw.strip_prefix('!') {
            denied_caps.insert(rest.to_string());
        } else {
            allowed_caps.insert(raw.to_string());
        }
    }

    // Conflict resolution: anything in both → keep only in deny, emit warning.
    let conflicts: Vec<String> = allowed_caps.intersection(&denied_caps).cloned().collect();
    for key in &conflicts {
        allowed_caps.remove(key);
        warnings.push(ResolveError::ConflictingCapability { key: key.clone() });
    }

    // --- Env ------------------------------------------------------------
    let mut env: BTreeMap<String, String> = BTreeMap::new();
    for key in &profile.env_passthrough {
        // Phase 3 scaffold: we record the key with an empty value; the
        // launcher fills the value from the parent env at spawn time.
        env.insert(key.clone(), String::new());
    }
    env.insert(
        SANDBOX_MARKER_KEY.to_string(),
        SANDBOX_MARKER_VALUE.to_string(),
    );

    let spec = SandboxLaunchSpec {
        mounts,
        net,
        env,
        allowed_caps,
        denied_caps,
        working_dir: workspace_guest,
        cpu_quota_pct: None,
        memory_limit_mib: None,
        backend,
    };

    (spec, warnings)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sandbox_profile::Mount;

    fn empty_caps() -> CapabilityMatrix {
        CapabilityMatrix::new()
    }

    fn trivial_profile() -> SandboxProfile {
        SandboxProfile {
            name: "test".into(),
            backend: SandboxBackend::Passthrough,
            mounts: Vec::new(),
            net: NetPolicy::None,
            env_passthrough: BTreeSet::new(),
        }
    }

    fn ws() -> PathBuf {
        PathBuf::from("/ws")
    }

    #[test]
    fn trivial_profile_injects_workspace_mount() {
        let p = trivial_profile();
        let spec = resolve(&p, &empty_caps(), &ws(), BackendChoice::Passthrough).expect("resolve");
        assert_eq!(spec.mounts.len(), 1);
        assert_eq!(spec.mounts[0].guest, PathBuf::from("/workspace"));
        assert_eq!(spec.mounts[0].host, ws());
    }

    #[test]
    fn workspace_mount_is_read_only_by_default() {
        let spec = resolve(
            &trivial_profile(),
            &empty_caps(),
            &ws(),
            BackendChoice::Passthrough,
        )
        .expect("resolve");
        assert_eq!(spec.mounts[0].mode, MountMode::ReadOnly);
    }

    #[test]
    fn profile_workspace_override_is_not_duplicated() {
        let mut p = trivial_profile();
        p.mounts.push(Mount {
            source: PathBuf::from("/elsewhere"),
            destination: PathBuf::from("/workspace"),
            read_only: false,
        });
        let spec = resolve(&p, &empty_caps(), &ws(), BackendChoice::Passthrough).expect("resolve");
        let workspace_mounts: Vec<&ResolvedMount> = spec
            .mounts
            .iter()
            .filter(|m| m.guest == PathBuf::from("/workspace"))
            .collect();
        assert_eq!(workspace_mounts.len(), 1);
        assert_eq!(workspace_mounts[0].host, PathBuf::from("/elsewhere"));
        assert_eq!(workspace_mounts[0].mode, MountMode::ReadWrite);
    }

    #[test]
    fn non_absolute_host_path_returns_error_strict() {
        let mut p = trivial_profile();
        p.mounts.push(Mount {
            source: PathBuf::from("relative/path"),
            destination: PathBuf::from("/g"),
            read_only: true,
        });
        let err = resolve(&p, &empty_caps(), &ws(), BackendChoice::Passthrough)
            .expect_err("should error");
        assert!(matches!(err, ResolveError::NonAbsoluteHostPath(_)));
    }

    #[test]
    fn mount_collision_returns_error_strict() {
        let mut p = trivial_profile();
        p.mounts.push(Mount {
            source: PathBuf::from("/a"),
            destination: PathBuf::from("/dup"),
            read_only: true,
        });
        p.mounts.push(Mount {
            source: PathBuf::from("/b"),
            destination: PathBuf::from("/dup"),
            read_only: true,
        });
        let err = resolve(&p, &empty_caps(), &ws(), BackendChoice::Passthrough)
            .expect_err("should error");
        assert!(matches!(err, ResolveError::MountCollision { .. }));
    }

    #[test]
    fn net_off_maps_through() {
        let p = trivial_profile();
        let spec = resolve(&p, &empty_caps(), &ws(), BackendChoice::Passthrough).expect("resolve");
        assert_eq!(spec.net, ResolvedNet::Off);
    }

    #[test]
    fn net_loopback_maps_through() {
        let mut p = trivial_profile();
        p.net = NetPolicy::LocalhostOnly;
        let spec = resolve(&p, &empty_caps(), &ws(), BackendChoice::Passthrough).expect("resolve");
        assert_eq!(spec.net, ResolvedNet::Loopback);
    }

    #[test]
    fn net_hosts_maps_through() {
        let mut p = trivial_profile();
        let allow = BTreeSet::from(["api.example.com".to_string(), "z.example".to_string()]);
        p.net = NetPolicy::AllowList(allow.clone());
        let spec = resolve(&p, &empty_caps(), &ws(), BackendChoice::Passthrough).expect("resolve");
        match spec.net {
            ResolvedNet::Hosts { allow: got } => assert_eq!(got, allow),
            other => panic!("expected Hosts, got {other:?}"),
        }
    }

    #[test]
    fn capability_conflict_lands_in_deny_only_lenient() {
        let p = trivial_profile();
        let caps = CapabilityMatrix::from_entries(["fs.read", "!fs.read"]);
        let (spec, warnings) = resolve_with_warnings(&p, &caps, &ws(), BackendChoice::Passthrough);
        assert!(spec.denied_caps.contains("fs.read"));
        assert!(!spec.allowed_caps.contains("fs.read"));
        assert!(warnings
            .iter()
            .any(|w| matches!(w, ResolveError::ConflictingCapability { key } if key == "fs.read")));
    }

    #[test]
    fn env_includes_sandbox_marker() {
        let spec = resolve(
            &trivial_profile(),
            &empty_caps(),
            &ws(),
            BackendChoice::Passthrough,
        )
        .expect("resolve");
        assert_eq!(spec.env.get("STRATUM_SANDBOX"), Some(&"1".to_string()));
    }

    #[test]
    fn mounts_sorted_by_guest_path() {
        let mut p = trivial_profile();
        p.mounts.push(Mount {
            source: PathBuf::from("/zz"),
            destination: PathBuf::from("/zz-guest"),
            read_only: true,
        });
        p.mounts.push(Mount {
            source: PathBuf::from("/aa"),
            destination: PathBuf::from("/aa-guest"),
            read_only: true,
        });
        let spec = resolve(&p, &empty_caps(), &ws(), BackendChoice::Passthrough).expect("resolve");
        let guests: Vec<&PathBuf> = spec.mounts.iter().map(|m| &m.guest).collect();
        let mut sorted = guests.clone();
        sorted.sort();
        assert_eq!(guests, sorted);
    }

    #[test]
    fn backend_choice_bwrap_flows_through() {
        let spec = resolve(
            &trivial_profile(),
            &empty_caps(),
            &ws(),
            BackendChoice::Bwrap,
        )
        .expect("resolve");
        assert_eq!(spec.backend, BackendChoice::Bwrap);
    }

    #[test]
    fn stable_hash_is_deterministic() {
        let a = resolve(
            &trivial_profile(),
            &empty_caps(),
            &ws(),
            BackendChoice::Passthrough,
        )
        .expect("resolve");
        let b = resolve(
            &trivial_profile(),
            &empty_caps(),
            &ws(),
            BackendChoice::Passthrough,
        )
        .expect("resolve");
        assert_eq!(a.stable_hash(), b.stable_hash());
    }

    #[test]
    fn stable_hash_differs_on_mount_mode_change() {
        let mut p1 = trivial_profile();
        p1.mounts.push(Mount {
            source: PathBuf::from("/a"),
            destination: PathBuf::from("/g"),
            read_only: true,
        });
        let mut p2 = trivial_profile();
        p2.mounts.push(Mount {
            source: PathBuf::from("/a"),
            destination: PathBuf::from("/g"),
            read_only: false,
        });
        let s1 = resolve(&p1, &empty_caps(), &ws(), BackendChoice::Passthrough).expect("resolve");
        let s2 = resolve(&p2, &empty_caps(), &ws(), BackendChoice::Passthrough).expect("resolve");
        assert_ne!(s1.stable_hash(), s2.stable_hash());
    }

    #[test]
    fn launch_spec_serde_roundtrip() {
        let spec = resolve(
            &trivial_profile(),
            &empty_caps(),
            &ws(),
            BackendChoice::Passthrough,
        )
        .expect("resolve");
        let s = serde_json::to_string(&spec).expect("ser");
        let back: SandboxLaunchSpec = serde_json::from_str(&s).expect("de");
        assert_eq!(spec, back);
    }

    #[test]
    fn resolve_error_display_smoke() {
        let variants = [
            ResolveError::NonAbsoluteHostPath(PathBuf::from("rel")),
            ResolveError::ConflictingCapability {
                key: "fs.read".into(),
            },
            ResolveError::UnknownBackend,
            ResolveError::MountCollision {
                guest: PathBuf::from("/dup"),
            },
        ];
        for v in &variants {
            let s = format!("{v}");
            assert!(!s.is_empty());
        }
    }

    #[test]
    fn lenient_returns_spec_and_warnings_on_conflict() {
        let p = trivial_profile();
        let caps = CapabilityMatrix::from_entries(["fs.read", "!fs.read"]);
        let (spec, warnings) = resolve_with_warnings(&p, &caps, &ws(), BackendChoice::Passthrough);
        assert!(!warnings.is_empty());
        assert!(spec.denied_caps.contains("fs.read"));
    }

    #[test]
    fn backend_choice_round_trips_with_sandbox_backend() {
        for b in [
            SandboxBackend::Bwrap,
            SandboxBackend::SandboxExec,
            SandboxBackend::WindowsJob,
            SandboxBackend::Passthrough,
        ] {
            let bc: BackendChoice = b.into();
            let back: SandboxBackend = bc.into();
            assert_eq!(b, back);
        }
    }

    #[test]
    fn net_policy_full_maps_to_empty_hosts_allowlist() {
        let mut p = trivial_profile();
        p.net = NetPolicy::Full;
        let spec = resolve(&p, &empty_caps(), &ws(), BackendChoice::Passthrough).expect("resolve");
        match spec.net {
            ResolvedNet::Hosts { allow } => assert!(allow.is_empty()),
            other => panic!("expected Hosts, got {other:?}"),
        }
    }

    #[test]
    fn env_passthrough_keys_make_it_into_env() {
        let mut p = trivial_profile();
        p.env_passthrough = BTreeSet::from(["PATH".to_string(), "HOME".to_string()]);
        let spec = resolve(&p, &empty_caps(), &ws(), BackendChoice::Passthrough).expect("resolve");
        assert!(spec.env.contains_key("PATH"));
        assert!(spec.env.contains_key("HOME"));
        assert!(spec.env.contains_key("STRATUM_SANDBOX"));
    }

    #[test]
    fn allowed_caps_collected_when_no_deny_prefix() {
        let p = trivial_profile();
        let caps = CapabilityMatrix::from_entries(["fs.read", "fs.write:src/**"]);
        let spec = resolve(&p, &caps, &ws(), BackendChoice::Passthrough).expect("resolve");
        assert!(spec.allowed_caps.contains("fs.read"));
        assert!(spec.allowed_caps.contains("fs.write:src/**"));
        assert!(spec.denied_caps.is_empty());
    }

    #[test]
    fn working_dir_is_workspace_guest() {
        let spec = resolve(
            &trivial_profile(),
            &empty_caps(),
            &ws(),
            BackendChoice::Passthrough,
        )
        .expect("resolve");
        assert_eq!(spec.working_dir, PathBuf::from("/workspace"));
    }
}
