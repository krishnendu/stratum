//! Tool invocation data shape + dispatcher trait.
//!
//! Phase 3 v2 scaffold — pure data + a `ToolDispatcher` trait that the future
//! `AgentLoop` will call once the capability-matrix check passes. Real tool
//! implementations (fs.read, shell.exec, mcp.<server>.<verb>, etc.) plug in
//! later; this module pins the surface so the orchestrator can wire its
//! dispatch step today.
//!
//! Per `plan/19-user-agents-and-plugins.md` §7 and
//! `plan/31-tool-sandbox-and-secrets.md` §7.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};

/// A single tool call request issued by the agent loop.
///
/// `tool_id` is the fully-qualified verb (`fs.read`, `shell.exec`,
/// `mcp.github.list_issues`, …). `args` is the structured argument
/// payload — any JSON value the dispatcher chooses to accept.
/// `capability` is the capability-matrix entry that authorized this
/// call; recorded so dispatchers can double-check intent. `turn_id`
/// ties the call to a specific orchestrator turn.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolInvocation {
    /// Fully-qualified verb identifying the target dispatcher.
    pub tool_id: String,
    /// Structured arguments — any JSON object the dispatcher accepts.
    pub args: BTreeMap<String, serde_json::Value>,
    /// Capability-matrix entry that authorized this call.
    pub capability: String,
    /// Turn identifier — ties the call back to the agent loop.
    pub turn_id: u64,
}

/// Outcome of a tool dispatch.
///
/// Serialized with a `status` tag so consumers can branch on
/// `"status": "ok"` vs `"status": "err"` without peeking inside.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "lowercase")]
pub enum ToolResult {
    /// Successful outcome with an opaque JSON body and a byte count
    /// (used by the budget tracker to charge against the turn budget).
    Ok {
        /// Tool the result came from.
        tool_id: String,
        /// Structured response body.
        body: serde_json::Value,
        /// Approximate output size in bytes.
        bytes: u64,
    },
    /// Failed outcome with a stable STRAT-E code and a message.
    Err {
        /// Tool the failure came from.
        tool_id: String,
        /// Stable error code (e.g. `STRAT-E5004`).
        code: String,
        /// Human-readable message.
        message: String,
    },
}

/// Trait implemented by concrete tool backends.
///
/// Implementations MUST be `Send + Sync` so the registry can hand them
/// out across the async turn loop.
pub trait ToolDispatcher: Send + Sync {
    /// Run the call. Never panics; all errors land in `ToolResult::Err`.
    fn invoke(&self, inv: &ToolInvocation) -> ToolResult;
    /// Does this dispatcher handle `tool_id`?
    fn supports(&self, tool_id: &str) -> bool;
    /// Stable identifier used for duplicate-registration detection.
    fn id(&self) -> &str;
}

/// A trivial dispatcher that echoes the call's args back to the caller.
/// Used in tests and as a placeholder while real tools are stubbed.
#[derive(Debug, Default, Clone, Copy)]
pub struct EchoDispatcher;

impl ToolDispatcher for EchoDispatcher {
    fn invoke(&self, inv: &ToolInvocation) -> ToolResult {
        let body = serde_json::to_value(&inv.args).unwrap_or(serde_json::Value::Null);
        let bytes = body.to_string().len() as u64;
        ToolResult::Ok {
            tool_id: inv.tool_id.clone(),
            body,
            bytes,
        }
    }

    fn supports(&self, tool_id: &str) -> bool {
        tool_id == "echo"
    }

    fn id(&self) -> &str {
        Self::ID
    }
}

impl EchoDispatcher {
    const ID: &'static str = "echo";
}

/// A dispatcher that always refuses, carrying a reason. Useful for
/// plan-mode fences and capability denials.
#[derive(Debug, Clone)]
pub struct DenyDispatcher {
    id: String,
    reason: String,
}

impl DenyDispatcher {
    /// Build a deny-all dispatcher with the given id + reason.
    #[must_use]
    pub fn new(id: impl Into<String>, reason: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            reason: reason.into(),
        }
    }
}

impl ToolDispatcher for DenyDispatcher {
    fn invoke(&self, inv: &ToolInvocation) -> ToolResult {
        ToolResult::Err {
            tool_id: inv.tool_id.clone(),
            code: "STRAT-E5004".to_string(),
            message: self.reason.clone(),
        }
    }

    fn supports(&self, _tool_id: &str) -> bool {
        true
    }

    fn id(&self) -> &str {
        &self.id
    }
}

/// Composite dispatcher that routes calls to the first registered
/// backend whose `supports()` returns `true`.
#[derive(Default)]
pub struct RegistryDispatcher {
    dispatchers: Vec<Box<dyn ToolDispatcher>>,
}

impl fmt::Debug for RegistryDispatcher {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RegistryDispatcher")
            .field("ids", &self.ids())
            .finish()
    }
}

impl RegistryDispatcher {
    /// Empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            dispatchers: Vec::new(),
        }
    }

    /// Register a dispatcher. Fails if a dispatcher with the same id
    /// is already registered.
    ///
    /// # Errors
    ///
    /// Returns [`DispatchError::DuplicateId`] if `d.id()` collides with an
    /// already-registered dispatcher.
    pub fn register(&mut self, d: Box<dyn ToolDispatcher>) -> Result<(), DispatchError> {
        let new_id = d.id().to_string();
        if self
            .dispatchers
            .iter()
            .any(|existing| existing.id() == new_id)
        {
            return Err(DispatchError::DuplicateId(new_id));
        }
        self.dispatchers.push(d);
        Ok(())
    }

    /// Route the call to the first dispatcher that supports it.
    ///
    /// If none match returns a `STRAT-E5005` error result.
    #[must_use]
    pub fn dispatch(&self, inv: &ToolInvocation) -> ToolResult {
        for d in &self.dispatchers {
            if d.supports(&inv.tool_id) {
                return d.invoke(inv);
            }
        }
        ToolResult::Err {
            tool_id: inv.tool_id.clone(),
            code: "STRAT-E5005".to_string(),
            message: format!("no dispatcher for {}", inv.tool_id),
        }
    }

    /// Registered ids, in insertion order.
    #[must_use]
    pub fn ids(&self) -> Vec<&str> {
        self.dispatchers.iter().map(|d| d.id()).collect()
    }

    /// True when at least one registered dispatcher supports `tool_id`.
    /// Used by the agent loop to reject hallucinated tool names BEFORE
    /// the permission flow runs, so the user doesn't see a fake
    /// "asking for permission to run X" prompt for a nonexistent
    /// tool.
    #[must_use]
    pub fn supports(&self, tool_id: &str) -> bool {
        self.dispatchers.iter().any(|d| d.supports(tool_id))
    }
}

/// Errors returned by [`RegistryDispatcher`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DispatchError {
    /// A dispatcher with this id is already registered.
    DuplicateId(String),
}

impl fmt::Display for DispatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateId(id) => write!(f, "duplicate dispatcher id: {id}"),
        }
    }
}

impl Error for DispatchError {}

/// Convenience helper: dispatch a single owned invocation against a
/// concrete dispatcher without going through the registry.
#[allow(clippy::needless_pass_by_value)]
pub fn quick_dispatch<D: ToolDispatcher>(d: &D, inv: ToolInvocation) -> ToolResult {
    d.invoke(&inv)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::thread;

    use super::*;

    fn sample_invocation(tool: &str) -> ToolInvocation {
        let mut args = BTreeMap::new();
        args.insert("path".to_string(), serde_json::json!("README.md"));
        args.insert("limit".to_string(), serde_json::json!(42));
        ToolInvocation {
            tool_id: tool.to_string(),
            args,
            capability: "fs.read".to_string(),
            turn_id: 7,
        }
    }

    #[test]
    fn tool_invocation_serde_roundtrip() {
        let inv = sample_invocation("fs.read");
        let s = serde_json::to_string(&inv).expect("serialize");
        let back: ToolInvocation = serde_json::from_str(&s).expect("deserialize");
        assert_eq!(inv, back);
    }

    #[test]
    fn tool_result_ok_serde_roundtrip() {
        let r = ToolResult::Ok {
            tool_id: "echo".to_string(),
            body: serde_json::json!({"hello": "world"}),
            bytes: 17,
        };
        let s = serde_json::to_string(&r).expect("serialize");
        assert!(s.contains("\"status\":\"ok\""));
        let back: ToolResult = serde_json::from_str(&s).expect("deserialize");
        assert_eq!(r, back);
    }

    #[test]
    fn tool_result_err_serde_roundtrip() {
        let r = ToolResult::Err {
            tool_id: "fs.write".to_string(),
            code: "STRAT-E5004".to_string(),
            message: "denied by plan mode".to_string(),
        };
        let s = serde_json::to_string(&r).expect("serialize");
        assert!(s.contains("\"status\":\"err\""));
        let back: ToolResult = serde_json::from_str(&s).expect("deserialize");
        assert_eq!(r, back);
    }

    #[test]
    fn echo_dispatcher_supports() {
        let d = EchoDispatcher;
        assert!(d.supports("echo"));
        assert!(!d.supports("fs.read"));
    }

    #[test]
    fn echo_dispatcher_invoke_returns_args_as_body() {
        let d = EchoDispatcher;
        let inv = sample_invocation("echo");
        let result = d.invoke(&inv);
        match result {
            ToolResult::Ok { tool_id, body, .. } => {
                assert_eq!(tool_id, "echo");
                let expected = serde_json::to_value(&inv.args).expect("serialize args");
                assert_eq!(body, expected);
            }
            ToolResult::Err { .. } => panic!("expected Ok"),
        }
    }

    #[test]
    fn deny_dispatcher_returns_e5004() {
        let d = DenyDispatcher::new("deny", "blocked");
        let inv = sample_invocation("fs.write");
        match d.invoke(&inv) {
            ToolResult::Err { code, message, .. } => {
                assert_eq!(code, "STRAT-E5004");
                assert_eq!(message, "blocked");
            }
            ToolResult::Ok { .. } => panic!("expected Err"),
        }
    }

    #[test]
    fn registry_register_rejects_duplicate() {
        let mut reg = RegistryDispatcher::new();
        reg.register(Box::new(EchoDispatcher)).expect("first ok");
        let err = reg
            .register(Box::new(EchoDispatcher))
            .expect_err("dup must fail");
        assert_eq!(err, DispatchError::DuplicateId("echo".to_string()));
    }

    #[test]
    fn registry_dispatch_routes_to_match() {
        let mut reg = RegistryDispatcher::new();
        reg.register(Box::new(EchoDispatcher)).expect("register");
        let inv = sample_invocation("echo");
        match reg.dispatch(&inv) {
            ToolResult::Ok { tool_id, .. } => assert_eq!(tool_id, "echo"),
            ToolResult::Err { .. } => panic!("expected Ok"),
        }
    }

    #[test]
    fn registry_dispatch_returns_e5005_when_no_match() {
        let reg = RegistryDispatcher::new();
        let inv = sample_invocation("fs.read");
        match reg.dispatch(&inv) {
            ToolResult::Err { code, message, .. } => {
                assert_eq!(code, "STRAT-E5005");
                assert!(message.contains("fs.read"));
            }
            ToolResult::Ok { .. } => panic!("expected Err"),
        }
    }

    #[test]
    fn registry_dispatch_empty_returns_e5005() {
        let reg = RegistryDispatcher::new();
        let inv = sample_invocation("anything");
        match reg.dispatch(&inv) {
            ToolResult::Err { code, .. } => assert_eq!(code, "STRAT-E5005"),
            ToolResult::Ok { .. } => panic!("expected Err"),
        }
    }

    #[test]
    fn registry_ids_insertion_order() {
        let mut reg = RegistryDispatcher::new();
        reg.register(Box::new(EchoDispatcher)).expect("echo");
        reg.register(Box::new(DenyDispatcher::new("deny", "no")))
            .expect("deny");
        assert_eq!(reg.ids(), vec!["echo", "deny"]);
    }

    #[test]
    fn quick_dispatch_with_echo() {
        let d = EchoDispatcher;
        let inv = sample_invocation("echo");
        let result = quick_dispatch(&d, inv);
        match result {
            ToolResult::Ok { tool_id, .. } => assert_eq!(tool_id, "echo"),
            ToolResult::Err { .. } => panic!("expected Ok"),
        }
    }

    #[test]
    fn dispatch_error_display_smoke() {
        let err = DispatchError::DuplicateId("echo".to_string());
        let s = format!("{err}");
        assert!(s.contains("echo"));
        assert!(s.contains("duplicate"));
    }

    #[test]
    fn tool_invocation_eq() {
        let a = sample_invocation("echo");
        let b = sample_invocation("echo");
        assert_eq!(a, b);
    }

    #[test]
    fn tool_result_ok_eq() {
        let a = ToolResult::Ok {
            tool_id: "echo".to_string(),
            body: serde_json::json!("x"),
            bytes: 1,
        };
        let b = ToolResult::Ok {
            tool_id: "echo".to_string(),
            body: serde_json::json!("x"),
            bytes: 1,
        };
        assert_eq!(a, b);
    }

    #[test]
    fn tool_result_ok_vs_err_neq() {
        let ok = ToolResult::Ok {
            tool_id: "echo".to_string(),
            body: serde_json::json!("x"),
            bytes: 1,
        };
        let err = ToolResult::Err {
            tool_id: "echo".to_string(),
            code: "STRAT-E5005".to_string(),
            message: "no".to_string(),
        };
        assert_ne!(ok, err);
    }

    #[test]
    fn registry_concurrent_dispatch() {
        let mut reg = RegistryDispatcher::new();
        reg.register(Box::new(EchoDispatcher)).expect("echo");
        let shared = Arc::new(reg);
        let mut handles = Vec::new();
        for thread_id in 0u64..4 {
            let reg = Arc::clone(&shared);
            handles.push(thread::spawn(move || {
                let mut count_ok = 0;
                for i in 0u64..50 {
                    let mut args = BTreeMap::new();
                    args.insert("i".to_string(), serde_json::json!(thread_id * 100 + i));
                    let inv = ToolInvocation {
                        tool_id: "echo".to_string(),
                        args,
                        capability: "echo".to_string(),
                        turn_id: thread_id,
                    };
                    if matches!(reg.dispatch(&inv), ToolResult::Ok { .. }) {
                        count_ok += 1;
                    }
                }
                count_ok
            }));
        }
        let mut total = 0;
        for h in handles {
            total += h.join().expect("join");
        }
        assert_eq!(total, 4 * 50);
    }

    #[test]
    fn registry_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<RegistryDispatcher>();
    }

    #[test]
    fn tool_invocation_accepts_varied_json_args() {
        let mut args = BTreeMap::new();
        args.insert("n".to_string(), serde_json::json!(2.5_f64));
        args.insert("s".to_string(), serde_json::json!("hello"));
        args.insert("o".to_string(), serde_json::json!({"nested": [1, 2, 3]}));
        let inv = ToolInvocation {
            tool_id: "echo".to_string(),
            args,
            capability: "echo".to_string(),
            turn_id: 0,
        };
        let s = serde_json::to_string(&inv).expect("serialize");
        let back: ToolInvocation = serde_json::from_str(&s).expect("deserialize");
        assert_eq!(inv, back);
    }

    #[test]
    fn tool_result_ok_bytes_roundtrip() {
        let r = ToolResult::Ok {
            tool_id: "echo".to_string(),
            body: serde_json::json!(null),
            bytes: 9_999_999,
        };
        let s = serde_json::to_string(&r).expect("serialize");
        let back: ToolResult = serde_json::from_str(&s).expect("deserialize");
        match back {
            ToolResult::Ok { bytes, .. } => assert_eq!(bytes, 9_999_999),
            ToolResult::Err { .. } => panic!("expected Ok"),
        }
    }

    #[test]
    fn registry_dispatch_picks_first_match() {
        let mut reg = RegistryDispatcher::new();
        reg.register(Box::new(EchoDispatcher)).expect("echo");
        reg.register(Box::new(DenyDispatcher::new("deny", "blocked")))
            .expect("deny");
        // EchoDispatcher only supports "echo"; DenyDispatcher supports anything.
        // For "echo", EchoDispatcher wins (registered first).
        let inv = sample_invocation("echo");
        assert!(matches!(reg.dispatch(&inv), ToolResult::Ok { .. }));
        // For "fs.read", only DenyDispatcher supports → Err.
        let inv2 = sample_invocation("fs.read");
        match reg.dispatch(&inv2) {
            ToolResult::Err { code, .. } => assert_eq!(code, "STRAT-E5004"),
            ToolResult::Ok { .. } => panic!("expected Err"),
        }
    }

    #[test]
    fn deny_dispatcher_id_returns_configured() {
        let d = DenyDispatcher::new("plan_mode_fence", "read-only");
        assert_eq!(d.id(), "plan_mode_fence");
    }

    #[test]
    fn registry_debug_lists_ids() {
        let mut reg = RegistryDispatcher::new();
        reg.register(Box::new(EchoDispatcher)).expect("echo");
        let s = format!("{reg:?}");
        assert!(s.contains("echo"));
    }
}
