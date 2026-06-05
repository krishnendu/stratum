//! MCP (Model Context Protocol) client + server data shapes.
//!
//! Phase 3 (data only) — the real protocol (JSON-RPC over stdio / HTTP,
//! spec-version handshake, streaming tool outputs) lands in Phase 6. This
//! module pins the workspace `stratum.toml` shape and the namespace-prefixed
//! tool entries so the global capability matrix can intersect them today.
//!
//! Per `plan/33-mcp-and-external-tools.md` §2-3.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::tools::{CapabilityEntry, CapabilityMatrix};

/// Transport an upstream MCP server speaks. Mirrors the `[[mcp.servers]]`
/// `transport = "stdio" | "http"` discriminator from §2.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "transport", rename_all = "snake_case")]
pub enum McpTransport {
    /// Spawn a long-lived subprocess and speak JSON-RPC over its stdio.
    Stdio {
        /// Executable to spawn.
        command: String,
        /// Argument vector. Defaults to empty when absent.
        #[serde(default)]
        args: Vec<String>,
        /// Extra environment variables merged onto the workspace `secrets`
        /// inherited env. Sorted, so the serialized form is deterministic.
        #[serde(default)]
        env: BTreeMap<String, String>,
    },
    /// Connect to a remote MCP endpoint over HTTP.
    Http {
        /// Endpoint URL (validated by the live client, not this data shape).
        url: String,
        /// Optional `keyring://...` URI carrying a bearer token. `None`
        /// means the endpoint is unauthenticated (rare; usually a local
        /// sidecar).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        bearer_token_uri: Option<String>,
    },
}

/// One configured upstream MCP server.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpServerConfig {
    /// Logical name used to key the server in [`McpServerSet`] and to
    /// build the `mcp.<name>.<verb>` capability prefix.
    pub name: String,
    /// Transport-specific connection details.
    #[serde(flatten)]
    pub transport: McpTransport,
    /// Tool keywords (without the `mcp.<server>.` prefix) the user
    /// explicitly allows. Intersected with the global capability matrix.
    #[serde(default)]
    pub allow: Vec<String>,
    /// Tool keywords the user explicitly denies. The denial wins over
    /// `allow`; the live client enforces the rule.
    #[serde(default)]
    pub deny: Vec<String>,
}

/// Live state of one MCP server, used by the `/mcp list` palette.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum McpServerStatus {
    /// Connected and responding to JSON-RPC.
    Live,
    /// Configured but not currently spawned (idle eviction or never used).
    Dormant,
    /// Last spawn or call failed; `reason` carries the human-readable
    /// detail surfaced in the palette. Encoded as a struct variant
    /// because the enum is internally tagged (`#[serde(tag = "state")]`),
    /// which forbids newtype-of-primitive variants.
    Failed {
        /// Human-readable failure reason.
        reason: String,
    },
}

/// Keyed registry of [`McpServerConfig`] entries.
///
/// Keyed by `McpServerConfig::name`; iteration is sorted by that key so
/// CLI / TUI rendering is deterministic.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct McpServerSet(BTreeMap<String, McpServerConfig>);

impl McpServerSet {
    /// Empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register `config`. If a server with the same name was already
    /// present the previous entry is returned (caller decides whether to
    /// surface a warning).
    pub fn insert(&mut self, config: McpServerConfig) -> Option<McpServerConfig> {
        self.0.insert(config.name.clone(), config)
    }

    /// Borrow a configured server by its logical name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&McpServerConfig> {
        self.0.get(name)
    }

    /// Drop a configured server. Returns the removed entry.
    pub fn remove(&mut self, name: &str) -> Option<McpServerConfig> {
        self.0.remove(name)
    }

    /// Count of registered servers.
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Is the registry empty?
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Walk every `(name, config)` pair in alphabetical order by name.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &McpServerConfig)> {
        self.0.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Translate the named server's `allow` list into a `CapabilityMatrix`
    /// of `mcp.<server>.<verb>` entries. Returns an empty matrix when the
    /// server is unknown or has no allow entries.
    #[must_use]
    pub fn effective_capabilities(&self, server_name: &str) -> CapabilityMatrix {
        let Some(server) = self.0.get(server_name) else {
            return CapabilityMatrix::new();
        };
        CapabilityMatrix::from_entries(
            server
                .allow
                .iter()
                .map(|verb| CapabilityEntry::new(format!("mcp.{server_name}.{verb}"))),
        )
    }
}

/// Transport Stratum's own MCP server listens on. Mirrors the
/// `[mcp_server]` `transport = "stdio" | "http"` discriminator from §3.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "transport", rename_all = "snake_case")]
pub enum McpServeTransport {
    /// Stdio sidecar invoked by a single local client (Claude Desktop,
    /// Zed, Cursor).
    Stdio,
    /// HTTP listener; the optional `token_uri` points at the keyring
    /// entry that carries the bearer the listener must accept.
    Http {
        /// `keyring://...` URI for the listener's bearer token. `None`
        /// only makes sense when `allow_any_client = true`; this module
        /// only encodes the shape and leaves the enforcement to the live
        /// server.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        token_uri: Option<String>,
    },
}

/// `[mcp_server]` table from `stratum.toml` (§3).
///
/// Stratum exposes a curated subset of its tool registry to outside MCP
/// clients. The whole feature is **off by default**; this shape only
/// carries the configuration — it does not start a listener.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpServerExpose {
    /// Master switch. Wizard never flips this implicitly.
    pub enabled: bool,
    /// How external clients reach this Stratum instance.
    #[serde(flatten)]
    pub transport: McpServeTransport,
    /// Global capability names exposed to clients (e.g. `fs.read`,
    /// `git.diff`). Serialized in sorted order — the underlying
    /// `BTreeSet` guarantees it.
    #[serde(default)]
    pub expose: BTreeSet<String>,
    /// If `true`, the listener skips bearer-token auth. Only sensible
    /// for stdio; the live server enforces the policy.
    #[serde(default)]
    pub allow_any_client: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stdio_cfg() -> McpServerConfig {
        McpServerConfig {
            name: "filesystem".into(),
            transport: McpTransport::Stdio {
                command: "uvx".into(),
                args: vec!["mcp-server-filesystem".into(), "--root".into(), ".".into()],
                env: BTreeMap::new(),
            },
            allow: vec!["read".into(), "list".into()],
            deny: vec!["write".into()],
        }
    }

    fn http_cfg(token: Option<&str>) -> McpServerConfig {
        McpServerConfig {
            name: "linear".into(),
            transport: McpTransport::Http {
                url: "https://mcp.linear.app".into(),
                bearer_token_uri: token.map(str::to_owned),
            },
            allow: vec!["issue.read".into()],
            deny: vec![],
        }
    }

    #[test]
    fn stdio_transport_roundtrips_via_json() {
        let cfg = stdio_cfg();
        let json = serde_json::to_string(&cfg).expect("serialize");
        let back: McpServerConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(cfg, back);
    }

    #[test]
    fn http_transport_with_bearer_roundtrips() {
        let cfg = http_cfg(Some("keyring://linear/personal"));
        let json = serde_json::to_string(&cfg).expect("serialize");
        let back: McpServerConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(cfg, back);
        assert!(json.contains("keyring://linear/personal"));
    }

    #[test]
    fn http_transport_without_bearer_skips_field() {
        let cfg = http_cfg(None);
        let json = serde_json::to_string(&cfg).expect("serialize");
        assert!(!json.contains("bearer_token_uri"));
        let back: McpServerConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(cfg, back);
    }

    #[test]
    fn server_config_parses_stdio_toml() {
        let toml = r#"
name = "filesystem"
transport = "stdio"
command = "uvx"
args = ["mcp-server-filesystem", "--root", "."]
allow = ["read", "list", "search"]
deny = ["write"]
"#;
        let cfg: McpServerConfig = toml_edit::de::from_str(toml).expect("parse");
        assert_eq!(cfg.name, "filesystem");
        match cfg.transport {
            McpTransport::Stdio {
                ref command,
                ref args,
                ref env,
            } => {
                assert_eq!(command, "uvx");
                assert_eq!(args.len(), 3);
                assert!(env.is_empty());
            }
            McpTransport::Http { .. } => panic!("expected stdio"),
        }
        assert_eq!(cfg.allow, vec!["read", "list", "search"]);
        assert_eq!(cfg.deny, vec!["write"]);
    }

    #[test]
    fn server_config_parses_http_toml() {
        let toml = r#"
name = "linear"
transport = "http"
url = "https://mcp.linear.app"
bearer_token_uri = "keyring://linear/personal"
allow = ["issue.read"]
"#;
        let cfg: McpServerConfig = toml_edit::de::from_str(toml).expect("parse");
        match cfg.transport {
            McpTransport::Http {
                ref url,
                ref bearer_token_uri,
            } => {
                assert_eq!(url, "https://mcp.linear.app");
                assert_eq!(
                    bearer_token_uri.as_deref(),
                    Some("keyring://linear/personal")
                );
            }
            McpTransport::Stdio { .. } => panic!("expected http"),
        }
        assert!(cfg.deny.is_empty());
    }

    #[test]
    fn server_set_insert_returns_prior() {
        let mut set = McpServerSet::new();
        assert!(set.is_empty());
        assert!(set.insert(stdio_cfg()).is_none());
        let mut renamed = stdio_cfg();
        renamed.allow = vec!["read".into()];
        let prior = set.insert(renamed).expect("prior config");
        assert_eq!(prior.allow, vec!["read".to_owned(), "list".to_owned()]);
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn server_set_get_and_remove() {
        let mut set = McpServerSet::new();
        set.insert(stdio_cfg());
        assert!(set.get("filesystem").is_some());
        assert!(set.get("missing").is_none());
        let removed = set.remove("filesystem").expect("removed");
        assert_eq!(removed.name, "filesystem");
        assert!(set.is_empty());
        assert!(set.remove("filesystem").is_none());
    }

    #[test]
    fn effective_capabilities_prefixes_allow_entries() {
        let mut set = McpServerSet::new();
        set.insert(McpServerConfig {
            name: "fs".into(),
            transport: McpTransport::Stdio {
                command: "mcp-fs".into(),
                args: vec![],
                env: BTreeMap::new(),
            },
            allow: vec!["read".into()],
            deny: vec![],
        });
        let matrix = set.effective_capabilities("fs");
        assert_eq!(matrix.len(), 1);
        assert!(matrix.allows("mcp.fs.read", None));
        let names: Vec<&str> = matrix.entries().map(CapabilityEntry::as_str).collect();
        assert_eq!(names, vec!["mcp.fs.read"]);
    }

    #[test]
    fn effective_capabilities_empty_allow_is_empty_matrix() {
        let mut set = McpServerSet::new();
        set.insert(McpServerConfig {
            name: "fs".into(),
            transport: McpTransport::Stdio {
                command: "mcp-fs".into(),
                args: vec![],
                env: BTreeMap::new(),
            },
            allow: vec![],
            deny: vec![],
        });
        assert!(set.effective_capabilities("fs").is_empty());
    }

    #[test]
    fn effective_capabilities_unknown_server_is_empty_matrix() {
        let set = McpServerSet::new();
        assert!(set.effective_capabilities("nope").is_empty());
    }

    #[test]
    fn iter_walks_servers_alphabetically() {
        let mut set = McpServerSet::new();
        set.insert(McpServerConfig {
            name: "zeta".into(),
            transport: McpTransport::Stdio {
                command: "z".into(),
                args: vec![],
                env: BTreeMap::new(),
            },
            allow: vec![],
            deny: vec![],
        });
        set.insert(McpServerConfig {
            name: "alpha".into(),
            transport: McpTransport::Stdio {
                command: "a".into(),
                args: vec![],
                env: BTreeMap::new(),
            },
            allow: vec![],
            deny: vec![],
        });
        set.insert(McpServerConfig {
            name: "mid".into(),
            transport: McpTransport::Stdio {
                command: "m".into(),
                args: vec![],
                env: BTreeMap::new(),
            },
            allow: vec![],
            deny: vec![],
        });
        let names: Vec<&str> = set.iter().map(|(name, _)| name).collect();
        assert_eq!(names, vec!["alpha", "mid", "zeta"]);
    }

    #[test]
    fn status_failed_roundtrips_through_serde() {
        let status = McpServerStatus::Failed {
            reason: "connection refused".into(),
        };
        let json = serde_json::to_string(&status).expect("serialize");
        // Internally-tagged: the discriminator is `state`.
        assert!(json.contains("\"state\""));
        let back: McpServerStatus = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(status, back);
    }

    #[test]
    fn status_live_and_dormant_roundtrip() {
        for s in [McpServerStatus::Live, McpServerStatus::Dormant] {
            let json = serde_json::to_string(&s).expect("serialize");
            let back: McpServerStatus = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(s, back);
        }
    }

    #[test]
    fn server_expose_http_with_token_roundtrips() {
        let cfg = McpServerExpose {
            enabled: true,
            transport: McpServeTransport::Http {
                token_uri: Some("keyring://stratum/mcp-serve-token".into()),
            },
            expose: BTreeSet::from(["fs.read".to_owned(), "git.diff".to_owned()]),
            allow_any_client: false,
        };
        let toml = toml_edit::ser::to_string(&cfg).expect("serialize");
        let back: McpServerExpose = toml_edit::de::from_str(&toml).expect("deserialize");
        assert_eq!(cfg, back);
    }

    #[test]
    fn server_expose_stdio_allow_any_client_optional_token() {
        let cfg = McpServerExpose {
            enabled: true,
            transport: McpServeTransport::Stdio,
            expose: BTreeSet::from(["rag.search".to_owned()]),
            allow_any_client: true,
        };
        let toml = toml_edit::ser::to_string(&cfg).expect("serialize");
        // Stdio variant has no `token_uri` field at all.
        assert!(!toml.contains("token_uri"));
        let back: McpServerExpose = toml_edit::de::from_str(&toml).expect("deserialize");
        assert_eq!(cfg, back);
    }

    #[test]
    fn server_expose_serializes_expose_sorted() {
        let cfg = McpServerExpose {
            enabled: true,
            transport: McpServeTransport::Stdio,
            expose: BTreeSet::from([
                "git.diff".to_owned(),
                "fs.read".to_owned(),
                "rag.search".to_owned(),
            ]),
            allow_any_client: false,
        };
        let json = serde_json::to_string(&cfg).expect("serialize");
        // Sorted: fs.read < git.diff < rag.search.
        let fs_idx = json.find("fs.read").expect("fs.read present");
        let git_idx = json.find("git.diff").expect("git.diff present");
        let rag_idx = json.find("rag.search").expect("rag.search present");
        assert!(fs_idx < git_idx);
        assert!(git_idx < rag_idx);
    }

    #[test]
    fn stdio_transport_env_roundtrips() {
        let mut env = BTreeMap::new();
        env.insert("RAG_INDEX".to_owned(), "/var/rag".to_owned());
        env.insert("LOG".to_owned(), "info".to_owned());
        let cfg = McpServerConfig {
            name: "rag".into(),
            transport: McpTransport::Stdio {
                command: "stratum-mcp-rag".into(),
                args: vec![],
                env,
            },
            allow: vec!["search".into()],
            deny: vec![],
        };
        let json = serde_json::to_string(&cfg).expect("serialize");
        // BTreeMap sorts: LOG < RAG_INDEX.
        let log_idx = json.find("LOG").expect("LOG present");
        let rag_idx = json.find("RAG_INDEX").expect("RAG_INDEX present");
        assert!(log_idx < rag_idx);
        let back: McpServerConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(cfg, back);
    }
}
