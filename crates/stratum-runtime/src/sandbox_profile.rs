//! Sandbox profile bodies.
//!
//! Phase 3 v2 — concrete `bwrap-*`, `macos-*`, and `passthrough`
//! profile definitions per `plan/31-tool-sandbox-and-secrets.md`. The
//! profile is a *data* shape today (mount list, network policy, env
//! filter); the actual invocation that turns a profile into a child
//! process (bwrap argv assembly, sandbox-exec SBPL emission) lands in
//! the same phase alongside the `stratum-tools` crate.

use std::collections::BTreeSet;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::sandbox::SandboxBackend;

/// Network egress policy for a profile.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NetPolicy {
    /// No network is allowed.
    None,
    /// Allow LAN-loopback only (`127.0.0.1`, `::1`).
    LocalhostOnly,
    /// Allow the explicit DNS allowlist.
    AllowList(BTreeSet<String>),
    /// Wide-open network. Used only by `passthrough`.
    Full,
}

/// One mount entry inside the sandbox.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Mount {
    /// Source path on the host.
    pub source: PathBuf,
    /// Destination path inside the sandbox.
    pub destination: PathBuf,
    /// Read-only?
    pub read_only: bool,
}

/// A sandbox profile. Pairs a backend with the policy + mount + env
/// fragments that backend needs to assemble the child.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SandboxProfile {
    /// Stable name (`"bwrap-strict"`, `"macos-net"`, …).
    pub name: String,
    /// Backend that runs this profile.
    pub backend: SandboxBackend,
    /// Read-only and read-write mount list.
    pub mounts: Vec<Mount>,
    /// Network policy.
    pub net: NetPolicy,
    /// Whitelisted env vars passed through to the child. The empty list
    /// means the child gets an empty env.
    pub env_passthrough: BTreeSet<String>,
}

impl SandboxProfile {
    /// `passthrough`: no isolation. Always available.
    #[must_use]
    pub fn passthrough() -> Self {
        Self {
            name: "passthrough".to_string(),
            backend: SandboxBackend::Passthrough,
            mounts: Vec::new(),
            net: NetPolicy::Full,
            env_passthrough: BTreeSet::new(),
        }
    }

    /// `bwrap-strict`: Linux user-namespace isolation with no network
    /// and a workspace-only mount bind.
    #[must_use]
    pub fn bwrap_strict(workspace: PathBuf) -> Self {
        Self {
            name: "bwrap-strict".to_string(),
            backend: SandboxBackend::Bwrap,
            mounts: vec![
                Mount {
                    source: PathBuf::from("/usr"),
                    destination: PathBuf::from("/usr"),
                    read_only: true,
                },
                Mount {
                    source: PathBuf::from("/etc/alternatives"),
                    destination: PathBuf::from("/etc/alternatives"),
                    read_only: true,
                },
                Mount {
                    source: PathBuf::from("/etc/ssl"),
                    destination: PathBuf::from("/etc/ssl"),
                    read_only: true,
                },
                Mount {
                    source: workspace.clone(),
                    destination: workspace,
                    read_only: false,
                },
            ],
            net: NetPolicy::None,
            env_passthrough: BTreeSet::from(["PATH".into()]),
        }
    }

    /// `bwrap-net`: like `bwrap-strict` but with the explicit network
    /// allowlist.
    #[must_use]
    pub fn bwrap_net(workspace: PathBuf, allow: BTreeSet<String>) -> Self {
        let mut p = Self::bwrap_strict(workspace);
        p.name = "bwrap-net".to_string();
        p.net = NetPolicy::AllowList(allow);
        p
    }

    /// `macos-strict`: SBPL-style policy expressing the same constraints
    /// as `bwrap-strict` (file-write only under the workspace; deny net).
    #[must_use]
    pub fn macos_strict(workspace: PathBuf) -> Self {
        Self {
            name: "macos-strict".to_string(),
            backend: SandboxBackend::SandboxExec,
            mounts: vec![Mount {
                source: workspace.clone(),
                destination: workspace,
                read_only: false,
            }],
            net: NetPolicy::None,
            env_passthrough: BTreeSet::from(["PATH".into()]),
        }
    }

    /// `macos-net`: same as `macos-strict` plus the DNS allowlist.
    #[must_use]
    pub fn macos_net(workspace: PathBuf, allow: BTreeSet<String>) -> Self {
        let mut p = Self::macos_strict(workspace);
        p.name = "macos-net".to_string();
        p.net = NetPolicy::AllowList(allow);
        p
    }

    /// Is this profile network-enabled at all?
    #[must_use]
    pub const fn has_network(&self) -> bool {
        !matches!(self.net, NetPolicy::None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn allowlist() -> BTreeSet<String> {
        BTreeSet::from([
            "api.anthropic.com".to_string(),
            "huggingface.co".to_string(),
        ])
    }

    #[test]
    fn passthrough_has_full_network_and_no_mounts() {
        let p = SandboxProfile::passthrough();
        assert_eq!(p.name, "passthrough");
        assert_eq!(p.backend, SandboxBackend::Passthrough);
        assert!(p.mounts.is_empty());
        assert_eq!(p.net, NetPolicy::Full);
        assert!(p.env_passthrough.is_empty());
        assert!(p.has_network());
    }

    #[test]
    fn bwrap_strict_binds_workspace_and_denies_net() {
        let p = SandboxProfile::bwrap_strict(PathBuf::from("/home/krish/projects/stratum"));
        assert_eq!(p.name, "bwrap-strict");
        assert_eq!(p.backend, SandboxBackend::Bwrap);
        assert_eq!(p.net, NetPolicy::None);
        assert!(!p.has_network());
        // The workspace path appears as a read-write bind.
        let ws = p
            .mounts
            .iter()
            .find(|m| !m.read_only)
            .expect("workspace mount");
        assert_eq!(ws.source, PathBuf::from("/home/krish/projects/stratum"));
        // /usr is read-only.
        let usr = p
            .mounts
            .iter()
            .find(|m| m.source == PathBuf::from("/usr"))
            .expect("/usr mount");
        assert!(usr.read_only);
    }

    #[test]
    fn bwrap_net_carries_allowlist() {
        let p = SandboxProfile::bwrap_net(PathBuf::from("/x"), allowlist());
        assert_eq!(p.name, "bwrap-net");
        assert_eq!(p.backend, SandboxBackend::Bwrap);
        match &p.net {
            NetPolicy::AllowList(set) => {
                assert!(set.contains("api.anthropic.com"));
                assert!(set.contains("huggingface.co"));
            }
            other => panic!("expected AllowList, got {other:?}"),
        }
        assert!(p.has_network());
    }

    #[test]
    fn macos_strict_binds_workspace_and_denies_net() {
        let p = SandboxProfile::macos_strict(PathBuf::from("/Users/krish/dev"));
        assert_eq!(p.name, "macos-strict");
        assert_eq!(p.backend, SandboxBackend::SandboxExec);
        assert_eq!(p.net, NetPolicy::None);
        assert_eq!(p.mounts.len(), 1);
    }

    #[test]
    fn macos_net_carries_allowlist() {
        let p = SandboxProfile::macos_net(PathBuf::from("/x"), allowlist());
        assert_eq!(p.name, "macos-net");
        assert_eq!(p.backend, SandboxBackend::SandboxExec);
        matches!(p.net, NetPolicy::AllowList(_));
    }

    #[test]
    fn has_network_for_localhost_only_is_true() {
        let p = SandboxProfile {
            name: "test".into(),
            backend: SandboxBackend::Passthrough,
            mounts: Vec::new(),
            net: NetPolicy::LocalhostOnly,
            env_passthrough: BTreeSet::new(),
        };
        assert!(p.has_network());
    }

    #[test]
    fn mount_serde_roundtrip() {
        let m = Mount {
            source: PathBuf::from("/a"),
            destination: PathBuf::from("/b"),
            read_only: true,
        };
        let s = serde_json::to_string(&m).unwrap();
        let back: Mount = serde_json::from_str(&s).unwrap();
        assert_eq!(m, back);
    }

    #[test]
    fn net_policy_serde_roundtrip() {
        for p in [
            NetPolicy::None,
            NetPolicy::LocalhostOnly,
            NetPolicy::AllowList(BTreeSet::from(["x".to_string()])),
            NetPolicy::Full,
        ] {
            let s = serde_json::to_string(&p).unwrap();
            let back: NetPolicy = serde_json::from_str(&s).unwrap();
            assert_eq!(p, back);
        }
    }

    #[test]
    fn sandbox_profile_serde_roundtrip() {
        let p = SandboxProfile::bwrap_net(PathBuf::from("/x"), allowlist());
        let s = serde_json::to_string(&p).unwrap();
        let back: SandboxProfile = serde_json::from_str(&s).unwrap();
        assert_eq!(p, back);
    }

    #[test]
    fn bwrap_strict_includes_path_env() {
        let p = SandboxProfile::bwrap_strict(PathBuf::from("/x"));
        assert!(p.env_passthrough.contains("PATH"));
    }
}
