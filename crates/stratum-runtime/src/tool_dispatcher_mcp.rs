//! `ToolDispatcher` bridge from a [`crate::mcp_jsonrpc::McpJsonRpcClient`] to
//! the runtime's [`crate::tool_invocation::ToolDispatcher`] surface.
//!
//! Phase 6 scaffold — wires MCP server tools into the same
//! [`crate::tool_invocation::RegistryDispatcher`] the agent loop already
//! routes `fs.read` / `shell.exec` through. The dispatcher receives a
//! [`crate::tool_invocation::ToolInvocation`] whose `tool_id` is
//! `mcp.<server>.<verb>`, splits the prefix, calls `tools/call` on the
//! underlying [`crate::mcp_jsonrpc::McpJsonRpcClient`], and maps the
//! [`crate::mcp_jsonrpc::ToolCallResult`] back into a
//! [`crate::tool_invocation::ToolResult`].
//!
//! ## Error code policy
//!
//! Mirrors `tool_dispatchers.rs`: this module ships local
//! `E_DISPATCH_MCP_*` sentinels rather than introducing new
//! `STRAT-E####` codes until the agent-loop dispatch step lands a stable
//! surface area. See `plan/29-error-taxonomy-and-logging.md` §8.

// xtask-check-error-codes: ignore-file
//
// Reason: this module uses local `E_DISPATCH_MCP_*` sentinels (mirroring
// the `E_DISPATCH_*` precedent in `tool_dispatchers.rs`) rather than
// catalog `STRAT-E####` entries.

use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::{Map, Value};

use crate::mcp_jsonrpc::{McpJsonRpcClient, McpRpcError, ToolCallResult};
use crate::tool_invocation::{ToolDispatcher, ToolInvocation, ToolResult};

/// Local sentinel: invocation targeted a different server name.
const E_DISPATCH_MCP_WRONG_SERVER: &str = "E_DISPATCH_MCP_WRONG_SERVER";
/// Local sentinel: verb was not on the per-dispatcher allowlist.
const E_DISPATCH_MCP_TOOL_DENIED: &str = "E_DISPATCH_MCP_TOOL_DENIED";
/// Local sentinel: server returned `isError: true`.
const E_DISPATCH_MCP_TOOL_ERROR: &str = "E_DISPATCH_MCP_TOOL_ERROR";
/// Local sentinel: RPC did not produce a response before the timeout.
const E_DISPATCH_TIMEOUT: &str = "E_DISPATCH_TIMEOUT";
/// Local sentinel: transport-layer failure talking to the MCP child.
const E_DISPATCH_MCP_TRANSPORT: &str = "E_DISPATCH_MCP_TRANSPORT";

/// Default per-call deadline for MCP `tools/call` requests.
const DEFAULT_MCP_TIMEOUT: Duration = Duration::from_secs(30);

/// `ToolDispatcher` that proxies `mcp.<server>.<verb>` calls into an
/// underlying [`McpJsonRpcClient`].
///
/// One dispatcher instance is bound to exactly one logical MCP server:
/// the `<server>` segment of an incoming `tool_id` must match the
/// dispatcher's `server_name`, otherwise the call is refused with
/// [`E_DISPATCH_MCP_WRONG_SERVER`]. Multiple servers are supported by
/// registering one dispatcher per server in the same
/// [`crate::tool_invocation::RegistryDispatcher`].
pub struct McpToolDispatcher {
    id: String,
    client: Arc<Mutex<McpJsonRpcClient>>,
    server_name: String,
    timeout: Duration,
    allowed_tools: BTreeSet<String>,
}

impl std::fmt::Debug for McpToolDispatcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("McpToolDispatcher")
            .field("id", &self.id)
            .field("server_name", &self.server_name)
            .field("timeout", &self.timeout)
            .field("allowed_tools", &self.allowed_tools)
            .finish_non_exhaustive()
    }
}

impl McpToolDispatcher {
    /// Build a new MCP dispatcher bound to `server_name` and the shared
    /// `client` handle. The default per-call timeout is 30 seconds and
    /// the allowlist is empty (which means "allow every verb").
    #[must_use]
    pub fn new(server_name: String, client: Arc<Mutex<McpJsonRpcClient>>) -> Self {
        let id = format!("mcp.{server_name}");
        Self {
            id,
            client,
            server_name,
            timeout: DEFAULT_MCP_TIMEOUT,
            allowed_tools: BTreeSet::new(),
        }
    }

    /// Override the per-call wall-clock timeout.
    #[must_use]
    pub const fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Restrict the dispatcher to a fixed verb allowlist. An empty set
    /// (the default) allows every verb.
    #[must_use]
    pub fn with_allowed_tools(mut self, allowed: BTreeSet<String>) -> Self {
        self.allowed_tools = allowed;
        self
    }

    /// Configured per-call timeout.
    #[must_use]
    pub const fn timeout(&self) -> Duration {
        self.timeout
    }

    /// Configured server name (the `<server>` segment of `mcp.<server>.<verb>`).
    #[must_use]
    pub fn server_name(&self) -> &str {
        &self.server_name
    }

    /// Read-only view of the configured verb allowlist.
    #[must_use]
    pub const fn allowed_tools(&self) -> &BTreeSet<String> {
        &self.allowed_tools
    }

    fn err(code: &str, message: impl Into<String>, tool_id: &str) -> ToolResult {
        ToolResult::Err {
            tool_id: tool_id.to_string(),
            code: code.to_string(),
            message: message.into(),
        }
    }

    fn args_to_value(inv: &ToolInvocation) -> Value {
        let mut map = Map::with_capacity(inv.args.len());
        for (k, v) in &inv.args {
            map.insert(k.clone(), v.clone());
        }
        Value::Object(map)
    }

    fn map_call_result(tool_id: &str, result: &ToolCallResult) -> ToolResult {
        // Aggregate text content blocks: the budget tracker charges
        // against raw text bytes; the MCP spec scopes "content" to text
        // for this scaffold (image/audio blocks land in later phases).
        let texts: Vec<String> = result
            .content
            .iter()
            .filter_map(|b| b.text.clone())
            .collect();
        if result.is_error {
            let message = if texts.is_empty() {
                "MCP tool reported error".to_string()
            } else {
                texts.join("\n")
            };
            return ToolResult::Err {
                tool_id: tool_id.to_string(),
                code: E_DISPATCH_MCP_TOOL_ERROR.to_string(),
                message,
            };
        }
        let bytes: u64 = texts.iter().map(|t| t.len() as u64).sum();
        let body = serde_json::json!({ "content": texts });
        ToolResult::Ok {
            tool_id: tool_id.to_string(),
            body,
            bytes,
        }
    }
}

impl ToolDispatcher for McpToolDispatcher {
    fn invoke(&self, inv: &ToolInvocation) -> ToolResult {
        let Some((server, verb)) = parse_mcp_tool_id(&inv.tool_id) else {
            return Self::err(
                E_DISPATCH_MCP_WRONG_SERVER,
                format!(
                    "tool_id `{}` is not a well-formed mcp.<server>.<verb>",
                    inv.tool_id
                ),
                &inv.tool_id,
            );
        };
        if server != self.server_name {
            return Self::err(
                E_DISPATCH_MCP_WRONG_SERVER,
                format!(
                    "tool_id targets server `{server}` but dispatcher is bound to `{}`",
                    self.server_name
                ),
                &inv.tool_id,
            );
        }
        if !self.allowed_tools.is_empty() && !self.allowed_tools.contains(&verb) {
            return Self::err(
                E_DISPATCH_MCP_TOOL_DENIED,
                format!("verb `{verb}` is not on the allowlist"),
                &inv.tool_id,
            );
        }

        let args = Self::args_to_value(inv);
        let outcome = {
            // Scope the lock guard tightly: hold the mutex only for the
            // RPC round-trip. Poisoned mutex maps to a transport error
            // sentinel rather than a panic.
            let Ok(guard) = self.client.lock() else {
                return Self::err(
                    E_DISPATCH_MCP_TRANSPORT,
                    "MCP client mutex poisoned",
                    &inv.tool_id,
                );
            };
            guard.call_tool(&verb, &args, self.timeout)
        };

        match outcome {
            Ok(result) => Self::map_call_result(&inv.tool_id, &result),
            Err(McpRpcError::ResponseTimeout { after }) => Self::err(
                E_DISPATCH_TIMEOUT,
                format!("MCP tools/call timed out after {} ms", after.as_millis()),
                &inv.tool_id,
            ),
            Err(McpRpcError::JsonRpcError { code, message }) => ToolResult::Err {
                tool_id: inv.tool_id.clone(),
                code: format!("E_DISPATCH_MCP_JSONRPC_{code}"),
                message,
            },
            Err(other) => Self::err(E_DISPATCH_MCP_TRANSPORT, other.to_string(), &inv.tool_id),
        }
    }

    fn supports(&self, tool_id: &str) -> bool {
        let prefix = format!("mcp.{}.", self.server_name);
        tool_id.starts_with(&prefix)
    }

    fn id(&self) -> &str {
        &self.id
    }
}

/// Split a `mcp.<server>.<verb>` tool id into `(server, verb)`.
///
/// Returns [`None`] when the id is not well-formed: at minimum it must
/// have the literal `mcp.` prefix, a non-empty `<server>`, a separator,
/// and a non-empty `<verb>`. Verbs may themselves contain dots (e.g.
/// `read.deep`), in which case the entire remaining suffix is returned
/// as the verb.
#[must_use]
pub fn parse_mcp_tool_id(tool_id: &str) -> Option<(String, String)> {
    let rest = tool_id.strip_prefix("mcp.")?;
    let (server, verb) = rest.split_once('.')?;
    if server.is_empty() || verb.is_empty() {
        return None;
    }
    Some((server.to_string(), verb.to_string()))
}

// =====================================================================
// Tests
// =====================================================================

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};
    use std::path::PathBuf;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use super::*;
    use crate::mcp::McpStdioConfig;
    use crate::mcp_jsonrpc::{
        ClientCapabilities, ClientInfo, McpInitializeParams, McpJsonRpcClient,
        RootsClientCapability, ToolsClientCapability,
    };
    use crate::tool_invocation::{RegistryDispatcher, ToolInvocation, ToolResult};

    fn sample_params() -> McpInitializeParams {
        McpInitializeParams {
            protocol_version: "2024-11-05".into(),
            client_info: ClientInfo {
                name: "stratum".into(),
                version: "0.1.0".into(),
            },
            capabilities: ClientCapabilities {
                tools: Some(ToolsClientCapability {}),
                roots: Some(RootsClientCapability {}),
            },
        }
    }

    fn invocation(tool_id: &str) -> ToolInvocation {
        let mut args = BTreeMap::new();
        args.insert("path".to_string(), serde_json::json!("/tmp/x"));
        ToolInvocation {
            tool_id: tool_id.to_string(),
            args,
            capability: "mcp".to_string(),
            turn_id: 1,
        }
    }

    #[cfg(unix)]
    fn fake_server_cfg(script: &str) -> McpStdioConfig {
        McpStdioConfig {
            command: PathBuf::from("/bin/sh"),
            args: vec!["-c".into(), script.into()],
            env: BTreeMap::new(),
            workdir: None,
            init_timeout: Duration::from_millis(200),
        }
    }

    #[cfg(unix)]
    fn spawn_and_init(server: &str, script: &str) -> Arc<Mutex<McpJsonRpcClient>> {
        let cfg = fake_server_cfg(script);
        let client = McpJsonRpcClient::spawn(server.into(), &cfg).expect("spawn ok");
        client
            .initialize(&sample_params(), Duration::from_secs(2))
            .expect("init ok");
        Arc::new(Mutex::new(client))
    }

    // Use a wrapped (handle-less) client whenever we only need a value to
    // hold on to — RPC against it surfaces a transport error, perfect for
    // the non-IO-exercising assertions.
    fn wrapped_client() -> Arc<Mutex<McpJsonRpcClient>> {
        #[cfg(unix)]
        {
            use crate::mcp::McpStdioSession;
            let cfg = McpStdioConfig {
                command: PathBuf::from("/bin/sh"),
                args: vec!["-c".into(), "sleep 1".into()],
                env: BTreeMap::new(),
                workdir: None,
                init_timeout: Duration::from_millis(100),
            };
            let session = McpStdioSession::spawn("wrapped".into(), &cfg).expect("session ok");
            Arc::new(Mutex::new(McpJsonRpcClient::wrap(session)))
        }
        #[cfg(not(unix))]
        {
            // Non-unix CI: skip — the `wrap` constructor still requires
            // a session, which itself requires a child. The unix matrix
            // covers every non-IO assertion these tests exercise.
            unimplemented!("wrapped_client requires unix in tests");
        }
    }

    fn assert_ok_with(result: ToolResult) -> (String, serde_json::Value, u64) {
        if let ToolResult::Ok {
            tool_id,
            body,
            bytes,
        } = result
        {
            (tool_id, body, bytes)
        } else {
            panic!("expected Ok, got {result:?}");
        }
    }

    fn assert_err_with(result: ToolResult, expected_code: &str) -> String {
        if let ToolResult::Err { code, message, .. } = result {
            assert_eq!(code, expected_code, "wrong code (message was: {message})");
            message
        } else {
            panic!("expected Err({expected_code}), got {result:?}");
        }
    }

    // ---- parse_mcp_tool_id --------------------------------------------

    #[test]
    fn parse_mcp_tool_id_simple() {
        assert_eq!(
            parse_mcp_tool_id("mcp.fs.read"),
            Some(("fs".to_string(), "read".to_string()))
        );
    }

    #[test]
    fn parse_mcp_tool_id_no_verb_is_none() {
        assert_eq!(parse_mcp_tool_id("mcp.fs"), None);
    }

    #[test]
    fn parse_mcp_tool_id_no_mcp_prefix_is_none() {
        assert_eq!(parse_mcp_tool_id("fs.read"), None);
    }

    #[test]
    fn parse_mcp_tool_id_multi_level_verb() {
        assert_eq!(
            parse_mcp_tool_id("mcp.fs.read.deep"),
            Some(("fs".to_string(), "read.deep".to_string()))
        );
    }

    #[test]
    fn parse_mcp_tool_id_empty_segments_is_none() {
        assert_eq!(parse_mcp_tool_id("mcp..read"), None);
        assert_eq!(parse_mcp_tool_id("mcp.fs."), None);
    }

    // ---- field round-trip + constructor defaults ----------------------

    #[cfg(unix)]
    #[test]
    fn new_sets_defaults() {
        let client = wrapped_client();
        let d = McpToolDispatcher::new("fs".into(), client);
        assert_eq!(d.id(), "mcp.fs");
        assert_eq!(d.server_name(), "fs");
        assert_eq!(d.timeout(), DEFAULT_MCP_TIMEOUT);
        assert!(d.allowed_tools().is_empty());
    }

    #[cfg(unix)]
    #[test]
    fn with_timeout_and_with_allowed_tools_round_trip() {
        let client = wrapped_client();
        let mut allowed = BTreeSet::new();
        allowed.insert("read".to_string());
        allowed.insert("list".to_string());
        let d = McpToolDispatcher::new("fs".into(), client)
            .with_timeout(Duration::from_millis(750))
            .with_allowed_tools(allowed.clone());
        assert_eq!(d.timeout(), Duration::from_millis(750));
        assert_eq!(d.allowed_tools(), &allowed);
    }

    #[cfg(unix)]
    #[test]
    fn supports_only_matching_server_prefix() {
        let client = wrapped_client();
        let d = McpToolDispatcher::new("fs".into(), client);
        assert!(d.supports("mcp.fs.read"));
        assert!(d.supports("mcp.fs.list"));
        assert!(!d.supports("mcp.other.read"));
        assert!(!d.supports("mcp.fs")); // no verb segment
        assert!(!d.supports("fs.read"));
        assert!(!d.supports("shell.exec"));
    }

    #[cfg(unix)]
    #[test]
    fn debug_smoke_redacts_client() {
        let client = wrapped_client();
        let d = McpToolDispatcher::new("fs".into(), client);
        let rendered = format!("{d:?}");
        assert!(rendered.contains("mcp.fs"));
        assert!(rendered.contains("fs"));
    }

    // ---- invoke happy path against a fake server ---------------------

    #[cfg(unix)]
    #[test]
    fn invoke_against_fake_server_ok_returns_content_text() {
        let script = r#"while IFS= read -r line; do
  case "$line" in
    *'"initialize"'*) echo '{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"v","serverInfo":{"name":"f","version":"0"},"capabilities":{}}}' ;;
    *'"tools/call"'*) echo '{"jsonrpc":"2.0","id":2,"result":{"content":[{"type":"text","text":"hello"},{"type":"text","text":"world"}]}}' ;;
  esac
done"#;
        let client = spawn_and_init("fs", script);
        let d = McpToolDispatcher::new("fs".into(), client);
        let inv = invocation("mcp.fs.read");
        let (tool_id, body, bytes) = assert_ok_with(d.invoke(&inv));
        assert_eq!(tool_id, "mcp.fs.read");
        // "hello" + "world" → 10 bytes total.
        assert_eq!(bytes, 10);
        let arr = body
            .get("content")
            .and_then(|v| v.as_array())
            .expect("content array");
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[0], serde_json::json!("hello"));
        assert_eq!(arr[1], serde_json::json!("world"));
    }

    #[cfg(unix)]
    #[test]
    fn invoke_is_error_returns_tool_error_with_message() {
        let script = r#"while IFS= read -r line; do
  case "$line" in
    *'"initialize"'*) echo '{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"v","serverInfo":{"name":"f","version":"0"},"capabilities":{}}}' ;;
    *'"tools/call"'*) echo '{"jsonrpc":"2.0","id":2,"result":{"isError":true,"content":[{"type":"text","text":"boom"}]}}' ;;
  esac
done"#;
        let client = spawn_and_init("fs", script);
        let d = McpToolDispatcher::new("fs".into(), client);
        let inv = invocation("mcp.fs.explode");
        let msg = assert_err_with(d.invoke(&inv), E_DISPATCH_MCP_TOOL_ERROR);
        assert!(msg.contains("boom"), "unexpected message: {msg}");
    }

    #[cfg(unix)]
    #[test]
    fn invoke_is_error_without_text_yields_default_message() {
        let script = r#"while IFS= read -r line; do
  case "$line" in
    *'"initialize"'*) echo '{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"v","serverInfo":{"name":"f","version":"0"},"capabilities":{}}}' ;;
    *'"tools/call"'*) echo '{"jsonrpc":"2.0","id":2,"result":{"isError":true,"content":[]}}' ;;
  esac
done"#;
        let client = spawn_and_init("fs", script);
        let d = McpToolDispatcher::new("fs".into(), client);
        let inv = invocation("mcp.fs.explode");
        let msg = assert_err_with(d.invoke(&inv), E_DISPATCH_MCP_TOOL_ERROR);
        assert!(msg.contains("MCP tool reported error"));
    }

    #[cfg(unix)]
    #[test]
    fn invoke_response_timeout_maps_to_dispatch_timeout() {
        // Server completes initialize but ignores tools/call.
        let script = r#"while IFS= read -r line; do
  case "$line" in
    *'"initialize"'*) echo '{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"v","serverInfo":{"name":"f","version":"0"},"capabilities":{}}}' ;;
    *) sleep 5 ;;
  esac
done"#;
        let client = spawn_and_init("fs", script);
        let d =
            McpToolDispatcher::new("fs".into(), client).with_timeout(Duration::from_millis(120));
        let inv = invocation("mcp.fs.read");
        let msg = assert_err_with(d.invoke(&inv), E_DISPATCH_TIMEOUT);
        assert!(msg.contains("timed out"));
    }

    #[cfg(unix)]
    #[test]
    fn invoke_jsonrpc_error_maps_to_namespaced_code() {
        let script = r#"while IFS= read -r line; do
  case "$line" in
    *'"initialize"'*) echo '{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"v","serverInfo":{"name":"f","version":"0"},"capabilities":{}}}' ;;
    *'"tools/call"'*) echo '{"jsonrpc":"2.0","id":2,"error":{"code":-32601,"message":"Method not found"}}' ;;
  esac
done"#;
        let client = spawn_and_init("fs", script);
        let d = McpToolDispatcher::new("fs".into(), client);
        let inv = invocation("mcp.fs.read");
        let result = d.invoke(&inv);
        if let ToolResult::Err { code, message, .. } = result {
            assert_eq!(code, "E_DISPATCH_MCP_JSONRPC_-32601");
            assert!(message.contains("Method not found"));
        } else {
            panic!("expected Err");
        }
    }

    #[cfg(unix)]
    #[test]
    fn invoke_transport_error_for_wrapped_client() {
        // Wrapped clients can't drive RPC — call_tool returns Session,
        // which we fold into E_DISPATCH_MCP_TRANSPORT.
        let client = wrapped_client();
        let d = McpToolDispatcher::new("fs".into(), client);
        let inv = invocation("mcp.fs.read");
        let msg = assert_err_with(d.invoke(&inv), E_DISPATCH_MCP_TRANSPORT);
        assert!(!msg.is_empty());
    }

    // ---- invoke pre-flight refusal paths ------------------------------

    #[cfg(unix)]
    #[test]
    fn invoke_wrong_server_returns_wrong_server() {
        let client = wrapped_client();
        let d = McpToolDispatcher::new("fs".into(), client);
        let inv = invocation("mcp.other.read");
        assert_err_with(d.invoke(&inv), E_DISPATCH_MCP_WRONG_SERVER);
    }

    #[cfg(unix)]
    #[test]
    fn invoke_malformed_tool_id_returns_wrong_server() {
        let client = wrapped_client();
        let d = McpToolDispatcher::new("fs".into(), client);
        // Missing the verb segment — parse returns None.
        let inv = invocation("mcp.fs");
        assert_err_with(d.invoke(&inv), E_DISPATCH_MCP_WRONG_SERVER);
        // Wrong prefix entirely.
        let inv2 = invocation("fs.read");
        assert_err_with(d.invoke(&inv2), E_DISPATCH_MCP_WRONG_SERVER);
    }

    #[cfg(unix)]
    #[test]
    fn invoke_denied_verb_returns_tool_denied() {
        let client = wrapped_client();
        let mut allowed = BTreeSet::new();
        allowed.insert("read".to_string());
        let d = McpToolDispatcher::new("fs".into(), client).with_allowed_tools(allowed);
        let inv = invocation("mcp.fs.write");
        assert_err_with(d.invoke(&inv), E_DISPATCH_MCP_TOOL_DENIED);
    }

    // ---- trait-object + Send/Sync smoke -------------------------------

    #[cfg(unix)]
    #[test]
    fn dispatcher_via_arc_dyn() {
        let client = wrapped_client();
        let d: Arc<dyn ToolDispatcher> = Arc::new(McpToolDispatcher::new("fs".into(), client));
        assert_eq!(d.id(), "mcp.fs");
        assert!(d.supports("mcp.fs.read"));
    }

    #[test]
    fn dispatcher_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<McpToolDispatcher>();
    }

    // ---- RegistryDispatcher integration --------------------------------

    #[cfg(unix)]
    #[test]
    fn dispatches_through_registry() {
        let script = r#"while IFS= read -r line; do
  case "$line" in
    *'"initialize"'*) echo '{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"v","serverInfo":{"name":"f","version":"0"},"capabilities":{}}}' ;;
    *'"tools/call"'*) echo '{"jsonrpc":"2.0","id":2,"result":{"content":[{"type":"text","text":"ok"}]}}' ;;
  esac
done"#;
        let client = spawn_and_init("fs", script);
        let d = McpToolDispatcher::new("fs".into(), client);
        let mut reg = RegistryDispatcher::new();
        reg.register(Box::new(d)).expect("register");
        let inv = invocation("mcp.fs.read");
        let (tool_id, body, _) = assert_ok_with(reg.dispatch(&inv));
        assert_eq!(tool_id, "mcp.fs.read");
        let arr = body
            .get("content")
            .and_then(|v| v.as_array())
            .expect("content array");
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0], serde_json::json!("ok"));
    }

    // ---- args_to_value helper exercise --------------------------------

    #[cfg(unix)]
    #[test]
    fn args_to_value_preserves_keys_and_values() {
        let mut args = BTreeMap::new();
        args.insert("a".to_string(), serde_json::json!(1));
        args.insert("b".to_string(), serde_json::json!("two"));
        args.insert("c".to_string(), serde_json::json!([3, 4]));
        let inv = ToolInvocation {
            tool_id: "mcp.fs.read".to_string(),
            args,
            capability: "mcp".to_string(),
            turn_id: 0,
        };
        let v = McpToolDispatcher::args_to_value(&inv);
        let obj = v.as_object().expect("object");
        assert_eq!(obj.get("a"), Some(&serde_json::json!(1)));
        assert_eq!(obj.get("b"), Some(&serde_json::json!("two")));
        assert_eq!(obj.get("c"), Some(&serde_json::json!([3, 4])));
    }

    #[test]
    fn default_timeout_constant_is_thirty_seconds() {
        assert_eq!(DEFAULT_MCP_TIMEOUT, Duration::from_secs(30));
    }

    // ---- parse_mcp_tool_id additional coverage ------------------------

    #[test]
    fn parse_mcp_tool_id_just_mcp_dot_is_none() {
        assert_eq!(parse_mcp_tool_id("mcp."), None);
        assert_eq!(parse_mcp_tool_id("mcp"), None);
        assert_eq!(parse_mcp_tool_id(""), None);
    }
}
