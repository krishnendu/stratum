//! `AgentFactory` — fluent builder that constructs a fully-wired
//! [`AgentLoop`] from an [`AgentFactoryConfig`] plus the minimum required
//! dependency (a [`Provider`]). Every other component falls back to a
//! sensible default.
//!
//! ## Why this exists
//!
//! [`AgentLoopBuilder`] is the low-level fluent constructor: it requires
//! every collaborator (router, permission store, prompt id gen, responder,
//! event emitter, capability matrix, plan-mode flag, config) to be
//! explicitly provided. That ergonomics tax is appropriate for the
//! orchestrator itself, but every call site (CLI chat, tests, the future
//! `stratum serve` daemon) ends up repeating the same 9-field stanza.
//!
//! `AgentFactory` reduces that to *"hand me a provider, take a loop"* —
//! while still letting any of the defaults be overridden.
//!
//! ## Permission semantics in the runtime layer
//!
//! The runtime cannot draw a real interactive prompt — that's a CLI/TUI
//! concern. The [`PermissionMode::Prompt`] variant therefore falls back
//! to [`DenyAllResponder`] inside the runtime. The CLI layer is expected
//! to replace the responder via [`AgentLoopBuilder::with_responder`]
//! before handing the loop off to a real user.
//!
//! ## Errors
//!
//! Only `MissingProvider` is structurally surfaced. Inner
//! [`AgentLoopBuildError`] strings are wrapped in `AgentLoopBuild` for
//! forward compatibility — when [`AgentLoopBuilder::build`] gains new
//! error shapes, this factory keeps a stable surface.

use std::error::Error;
use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use serde::{Deserialize, Serialize};

use crate::agent_loop::{AgentLoop, AgentLoopConfig};
use crate::event_log::{EventEmitter, MemoryEventSink};
use crate::intent_router::IntentRouter;
use crate::permission_prompt::{
    AllowAllResponder, DenyAllResponder, PermissionStore, PromptIdGen, PromptResponder,
};
use crate::plan_mode::PlanMode;
use crate::provider::{EchoProvider, Provider};
use crate::sandbox::{SandboxBackend, SandboxSpawn};
use crate::sandbox_resolve::{BackendChoice, ResolvedNet, SandboxLaunchSpec};
use crate::tool_dispatchers::default_dispatchers;
use crate::tool_invocation::RegistryDispatcher;
use crate::tools::CapabilityMatrix;

// ---------------------------------------------------------------------------
// AgentFactoryConfig
// ---------------------------------------------------------------------------

/// Static, copyable configuration handed to [`AgentFactory::with_config`].
///
/// Serializable so it can live in a `stratum.toml` once Phase 4 lands the
/// project-config story.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentFactoryConfig {
    /// Activate plan mode at build time when `true`. Maps onto the
    /// `AgentLoopConfig::plan_mode` flag plus an `activate(now)` call on
    /// the default plan-mode handle.
    pub plan_mode: bool,
    /// Wall-clock cap on a single `run_turn`, in milliseconds. Defaults
    /// to `300_000` (5 minutes) to match [`AgentLoopConfig::default`].
    pub max_turn_duration_ms: u64,
    /// Upper bound on the number of `Block::ToolCall` permission checks
    /// performed within one turn.
    pub max_tool_calls_per_turn: u8,
    /// Default permission responder mode (see [`PermissionMode`]).
    pub permission_default: PermissionMode,
}

impl Default for AgentFactoryConfig {
    fn default() -> Self {
        Self {
            plan_mode: false,
            max_turn_duration_ms: 300_000,
            max_tool_calls_per_turn: 8,
            permission_default: PermissionMode::Prompt,
        }
    }
}

// ---------------------------------------------------------------------------
// PermissionMode
// ---------------------------------------------------------------------------

/// Permission-responder selector used by [`AgentFactory::build`].
///
/// `Prompt` is intentionally the default but, in the runtime layer,
/// resolves to a [`DenyAllResponder`]. The interactive TUI prompt lives
/// in the CLI crate; runtime code must default to deny.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PermissionMode {
    /// Allow every tool call without prompting. Useful for tests and
    /// non-interactive batch runs.
    AllowAll,
    /// Deny every tool call.
    DenyAll,
    /// Interactive prompt. The real prompter is wired in at the CLI
    /// layer; the runtime falls back to deny-by-default.
    Prompt,
}

// ---------------------------------------------------------------------------
// AgentFactoryError
// ---------------------------------------------------------------------------

/// Errors surfaced by [`AgentFactory::build`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AgentFactoryError {
    /// No provider was set on the factory.
    MissingProvider,
    /// The wrapped [`AgentLoopBuilder::build`] returned an error. The
    /// inner string is the underlying error's `Display`.
    AgentLoopBuild(String),
}

impl fmt::Display for AgentFactoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingProvider => f.write_str("AgentFactory: provider is required"),
            Self::AgentLoopBuild(msg) => write!(f, "AgentFactory: agent-loop build failed: {msg}"),
        }
    }
}

impl Error for AgentFactoryError {}

// ---------------------------------------------------------------------------
// AgentFactory
// ---------------------------------------------------------------------------

/// Fluent factory that composes an [`AgentLoop`] from defaults plus any
/// user-supplied overrides.
///
/// See the module docs for the rationale.
#[derive(Clone)]
pub struct AgentFactory {
    provider: Option<Arc<dyn Provider>>,
    router: Option<IntentRouter>,
    dispatcher: Option<Arc<RegistryDispatcher>>,
    capability_matrix: Option<Arc<CapabilityMatrix>>,
    plan_mode: Option<Arc<PlanMode>>,
    config: AgentFactoryConfig,
}

impl fmt::Debug for AgentFactory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AgentFactory")
            .field("provider_set", &self.provider.is_some())
            .field("router_set", &self.router.is_some())
            .field("dispatcher_set", &self.dispatcher.is_some())
            .field("capability_matrix_set", &self.capability_matrix.is_some())
            .field("plan_mode_set", &self.plan_mode.is_some())
            .field("config", &self.config)
            .finish()
    }
}

impl Default for AgentFactory {
    fn default() -> Self {
        Self::new()
    }
}

impl AgentFactory {
    /// Build a fresh factory with no provider and a default config.
    #[must_use]
    pub fn new() -> Self {
        Self {
            provider: None,
            router: None,
            dispatcher: None,
            capability_matrix: None,
            plan_mode: None,
            config: AgentFactoryConfig::default(),
        }
    }

    /// Set the provider. **Required** for [`Self::build`].
    #[must_use]
    pub fn with_provider(mut self, provider: Arc<dyn Provider>) -> Self {
        self.provider = Some(provider);
        self
    }

    /// Override the intent router. Defaults to [`IntentRouter::default`].
    #[must_use]
    pub fn with_router(mut self, router: IntentRouter) -> Self {
        self.router = Some(router);
        self
    }

    /// Override the tool dispatcher. Defaults to an empty
    /// [`RegistryDispatcher`].
    #[must_use]
    pub fn with_dispatcher(mut self, dispatcher: Arc<RegistryDispatcher>) -> Self {
        self.dispatcher = Some(dispatcher);
        self
    }

    /// Override the capability matrix. Defaults to
    /// [`CapabilityMatrix::default`].
    #[must_use]
    pub fn with_capability_matrix(mut self, matrix: Arc<CapabilityMatrix>) -> Self {
        self.capability_matrix = Some(matrix);
        self
    }

    /// Override the plan-mode handle. Defaults to a fresh
    /// [`PlanMode::new`].
    #[must_use]
    pub fn with_plan_mode(mut self, plan_mode: Arc<PlanMode>) -> Self {
        self.plan_mode = Some(plan_mode);
        self
    }

    /// Replace the config wholesale.
    #[must_use]
    pub const fn with_config(mut self, config: AgentFactoryConfig) -> Self {
        self.config = config;
        self
    }

    /// Snapshot of the currently-configured [`AgentFactoryConfig`].
    #[must_use]
    pub const fn config(&self) -> AgentFactoryConfig {
        self.config
    }

    /// Build the underlying [`AgentLoop`], filling every collaborator with
    /// a default when it was not explicitly supplied.
    ///
    /// # Errors
    ///
    /// * [`AgentFactoryError::MissingProvider`] — no provider was set.
    /// * [`AgentFactoryError::AgentLoopBuild`] — the inner builder
    ///   surfaced an error; the string is the underlying `Display`.
    pub fn build(self) -> Result<AgentLoop, AgentFactoryError> {
        let provider = self.provider.ok_or(AgentFactoryError::MissingProvider)?;
        let router = self.router.unwrap_or_default();
        let dispatcher = self
            .dispatcher
            .unwrap_or_else(|| Arc::new(RegistryDispatcher::new()));
        let capability_matrix = self
            .capability_matrix
            .unwrap_or_else(|| Arc::new(CapabilityMatrix::default()));
        let plan_mode = self.plan_mode.unwrap_or_else(|| Arc::new(PlanMode::new()));
        if self.config.plan_mode {
            plan_mode.activate(SystemTime::now());
        }

        let responder: Arc<dyn PromptResponder> = match self.config.permission_default {
            PermissionMode::AllowAll => Arc::new(AllowAllResponder),
            PermissionMode::DenyAll | PermissionMode::Prompt => Arc::new(DenyAllResponder),
        };

        let permission_store = Arc::new(PermissionStore::default());
        let prompt_gen = Arc::new(PromptIdGen::new());
        let events = Arc::new(EventEmitter::new(Arc::new(MemoryEventSink::new())));

        let loop_config = AgentLoopConfig {
            plan_mode: self.config.plan_mode,
            max_turn_duration: Duration::from_millis(self.config.max_turn_duration_ms),
            max_tool_calls_per_turn: self.config.max_tool_calls_per_turn,
            max_agentic_steps: AgentLoopConfig::default().max_agentic_steps,
            // Production factory: schema-validate tool calls so missing
            // required args (fs.write w/o `content`, fs.edit w/o
            // `old_string`) are caught before dispatch instead of
            // silently corrupting files.
            validate_tool_args: true,
        };

        AgentLoop::builder()
            .with_provider(provider)
            .with_router(router)
            .with_permission_store(permission_store)
            .with_prompt_gen(prompt_gen)
            .with_responder(responder)
            .with_events(events)
            .with_capability_matrix(capability_matrix)
            .with_plan_mode(plan_mode)
            .with_dispatcher(dispatcher)
            .with_config(loop_config)
            .build()
            .map_err(|e| AgentFactoryError::AgentLoopBuild(e.to_string()))
    }

    /// Convenience: one-liner producing a fully-defaulted
    /// [`EchoProvider`]-backed loop. Useful as the CLI default and for
    /// tests that just need *some* working loop.
    ///
    /// # Errors
    ///
    /// Propagates any [`AgentLoopBuildError`] from the inner builder.
    /// Cannot return [`AgentFactoryError::MissingProvider`] in practice.
    pub fn echo() -> Result<AgentLoop, AgentFactoryError> {
        Self::new()
            .with_provider(Arc::new(EchoProvider::new("")))
            .build()
    }
}

// ---------------------------------------------------------------------------
// default_factory_with_dispatchers
// ---------------------------------------------------------------------------

/// "Out-of-box no-sandbox" factory composition.
///
/// Returns an [`AgentFactory`] pre-wired with a
/// [`crate::tool_dispatchers::FsReadToolDispatcher`] +
/// [`crate::tool_dispatchers::ShellToolDispatcher`] driving the
/// [`BackendChoice::Passthrough`] backend.
///
/// Intended for development and CLI default invocations. **Production
/// deployments must replace the dispatcher with a real sandboxed
/// composition** — `Passthrough` runs commands directly against the
/// host, which is exactly what every sandbox-resolution path is
/// designed to avoid.
#[must_use]
pub fn default_factory_with_dispatchers(workspace_root: PathBuf) -> AgentFactory {
    let sandbox = SandboxSpawn::new(SandboxBackend::Passthrough);
    let base_spec = SandboxLaunchSpec {
        mounts: Vec::new(),
        net: ResolvedNet::Off,
        env: std::collections::BTreeMap::new(),
        allowed_caps: std::collections::BTreeSet::new(),
        denied_caps: std::collections::BTreeSet::new(),
        working_dir: workspace_root.clone(),
        cpu_quota_pct: None,
        memory_limit_mib: None,
        backend: BackendChoice::Passthrough,
    };
    let registry = default_dispatchers(workspace_root, sandbox, base_spec);
    AgentFactory::new().with_dispatcher(Arc::new(registry))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::cancel::CancelToken;
    use crate::conversation::TurnOutcome;
    use crate::intent_router::{Intent, IntentPattern, IntentRule, SuggestedRole};
    use crate::model_catalog::ModelTier;
    use crate::observability::TurnId;
    use crate::provider::{EchoProvider, GenerateRequest};
    use std::sync::Mutex;
    use std::thread;
    use std::time::UNIX_EPOCH;
    use stratum_types::{Block, Capability, ModelId};

    fn t0() -> SystemTime {
        UNIX_EPOCH + Duration::from_secs(1_700_000_000)
    }

    fn ctx(prompt: &str) -> crate::agent_loop::TurnContext {
        crate::agent_loop::TurnContext {
            user_prompt: prompt.into(),
            model: ModelId::from("echo"),
            turn_id: TurnId(1),
            started_at: t0(),
            history: Vec::new(),
        }
    }

    #[derive(Debug)]
    struct ScriptedProvider {
        script: Mutex<Vec<Vec<Block>>>,
    }

    impl ScriptedProvider {
        fn new(initial: Vec<Block>) -> Self {
            Self {
                script: Mutex::new(vec![initial]),
            }
        }
    }

    impl Provider for ScriptedProvider {
        #[allow(clippy::unnecessary_literal_bound)]
        fn id(&self) -> &str {
            "scripted"
        }
        fn capabilities(&self) -> &'static [Capability] {
            const CAPS: &[Capability] = &[Capability::Generate];
            CAPS
        }
        fn generate(&self, _req: &GenerateRequest, _cancel: &CancelToken) -> Vec<Block> {
            let mut g = self.script.lock().unwrap();
            if g.is_empty() {
                return Vec::new();
            }
            g.remove(0)
        }
    }

    /// Builds a ToolCall block with `args: "{}"`. **Only safe for
    /// tools that have no required args in `missing_required_args` —
    /// e.g. the synthetic `"echo"` used in dispatcher and budget
    /// tests.** Calling this with a tool that *does* have a required
    /// schema (e.g. `fs.read` needs `path`) will short-circuit through
    /// the schema gate (`validate_tool_args = true` in production
    /// factory) and surface `STRAT-E5006` instead of whatever
    /// downstream gate the test was trying to exercise. For real
    /// tools, use [`fs_read_call`] or build a `Block::ToolCall`
    /// inline with the required args populated.
    fn tool_call(id: &str, tool: &str) -> Block {
        Block::ToolCall {
            id: id.into(),
            tool: tool.into(),
            args: "{}".into(),
        }
    }

    /// fs.read call with a syntactically valid `path` arg. Use this when
    /// the test wants the call to clear the schema gate
    /// (`validate_tool_args = true`) and reach the permission gate.
    fn fs_read_call(id: &str) -> Block {
        Block::ToolCall {
            id: id.into(),
            tool: "fs.read".into(),
            args: r#"{"path":"hello.txt"}"#.into(),
        }
    }

    #[derive(Debug)]
    struct AcceptAllDispatcher;
    impl crate::tool_invocation::ToolDispatcher for AcceptAllDispatcher {
        fn invoke(
            &self,
            inv: &crate::tool_invocation::ToolInvocation,
        ) -> crate::tool_invocation::ToolResult {
            crate::tool_invocation::ToolResult::Ok {
                tool_id: inv.tool_id.clone(),
                body: serde_json::Value::Null,
                bytes: 0,
            }
        }
        fn supports(&self, _tool_id: &str) -> bool {
            true
        }
        fn id(&self) -> &'static str {
            "accept_all"
        }
    }

    fn permissive_registry() -> Arc<RegistryDispatcher> {
        let mut reg = RegistryDispatcher::new();
        reg.register(Box::new(AcceptAllDispatcher)).unwrap();
        Arc::new(reg)
    }

    // -- AgentFactoryConfig --------------------------------------------------

    #[test]
    fn config_default_matches_documented_values() {
        let c = AgentFactoryConfig::default();
        assert!(!c.plan_mode);
        assert_eq!(c.max_turn_duration_ms, 300_000);
        assert_eq!(c.max_tool_calls_per_turn, 8);
        assert_eq!(c.permission_default, PermissionMode::Prompt);
    }

    #[test]
    fn config_serde_round_trip() {
        let c = AgentFactoryConfig {
            plan_mode: true,
            max_turn_duration_ms: 1_234,
            max_tool_calls_per_turn: 3,
            permission_default: PermissionMode::AllowAll,
        };
        let s = serde_json::to_string(&c).unwrap();
        let back: AgentFactoryConfig = serde_json::from_str(&s).unwrap();
        assert_eq!(back, c);
    }

    #[test]
    fn permission_mode_serde_snake_case() {
        assert_eq!(
            serde_json::to_string(&PermissionMode::AllowAll).unwrap(),
            "\"allow_all\""
        );
        assert_eq!(
            serde_json::to_string(&PermissionMode::DenyAll).unwrap(),
            "\"deny_all\""
        );
        assert_eq!(
            serde_json::to_string(&PermissionMode::Prompt).unwrap(),
            "\"prompt\""
        );
    }

    // -- AgentFactory::new + missing provider --------------------------------

    #[test]
    fn new_has_no_provider_and_build_fails() {
        let factory = AgentFactory::new();
        assert!(factory.provider.is_none());
        let err = factory.build().unwrap_err();
        assert_eq!(err, AgentFactoryError::MissingProvider);
    }

    #[test]
    fn default_factory_is_same_as_new() {
        let a = format!("{:?}", AgentFactory::default());
        let b = format!("{:?}", AgentFactory::new());
        assert_eq!(a, b);
    }

    // -- AgentFactory::echo --------------------------------------------------

    #[test]
    fn echo_factory_runs_turn_successfully() {
        let loop_ = AgentFactory::echo().unwrap();
        let res = loop_.run_turn(ctx("hello"), &CancelToken::new());
        assert!(matches!(res.outcome, TurnOutcome::Success));
        assert!(!res.blocks.is_empty());
    }

    // -- with_provider basic build success -----------------------------------

    #[test]
    fn with_provider_alone_builds_successfully() {
        let loop_ = AgentFactory::new()
            .with_provider(Arc::new(EchoProvider::new("")))
            .build()
            .unwrap();
        let res = loop_.run_turn(ctx("hi"), &CancelToken::new());
        assert!(matches!(res.outcome, TurnOutcome::Success));
    }

    // -- with_router plumbs through ------------------------------------------

    #[test]
    fn with_router_empty_plumbs_through() {
        // An empty router classifies every prompt to the fallback intent;
        // verify the loop still completes successfully.
        let loop_ = AgentFactory::new()
            .with_provider(Arc::new(EchoProvider::new("")))
            .with_router(IntentRouter::empty())
            .build()
            .unwrap();
        let res = loop_.run_turn(ctx("anything"), &CancelToken::new());
        assert!(matches!(res.outcome, TurnOutcome::Success));
    }

    // -- with_dispatcher plumbs through --------------------------------------

    #[test]
    fn with_dispatcher_routes_tool_call() {
        let scripted = Arc::new(ScriptedProvider::new(vec![tool_call("echo#1", "echo")]));
        let loop_ = AgentFactory::new()
            .with_provider(scripted)
            .with_dispatcher(permissive_registry())
            .with_config(AgentFactoryConfig {
                permission_default: PermissionMode::AllowAll,
                ..AgentFactoryConfig::default()
            })
            .build()
            .unwrap();
        let res = loop_.run_turn(ctx("call echo"), &CancelToken::new());
        assert!(
            matches!(res.outcome, TurnOutcome::Success),
            "got {:?}",
            res.outcome
        );
    }

    // -- with_capability_matrix plumbs through -------------------------------

    #[test]
    fn with_capability_matrix_plumbs_through() {
        // Build a non-empty matrix, run a turn, and assert the loop
        // still completes — the matrix is consulted on every ToolCall
        // dispatch via `capability_matrix.entries()`.
        let matrix = Arc::new(CapabilityMatrix::from_entries(["fs.read".to_string()]));
        let loop_ = AgentFactory::new()
            .with_provider(Arc::new(EchoProvider::new("")))
            .with_capability_matrix(matrix)
            .build()
            .unwrap();
        let res = loop_.run_turn(ctx("plain text"), &CancelToken::new());
        assert!(matches!(res.outcome, TurnOutcome::Success));
    }

    // -- with_plan_mode plumbs through ---------------------------------------

    #[test]
    fn with_plan_mode_preserves_external_state() {
        let plan = Arc::new(PlanMode::new());
        plan.activate(t0());
        assert!(plan.is_active());
        let loop_ = AgentFactory::new()
            .with_provider(Arc::new(EchoProvider::new("")))
            .with_plan_mode(plan)
            .build()
            .unwrap();
        assert!(loop_.is_plan_mode_active());
    }

    #[test]
    fn config_plan_mode_true_activates_default_plan_mode() {
        let plan = Arc::new(PlanMode::new());
        assert!(!plan.is_active());
        let loop_ = AgentFactory::new()
            .with_provider(Arc::new(EchoProvider::new("")))
            .with_plan_mode(plan)
            .with_config(AgentFactoryConfig {
                plan_mode: true,
                ..AgentFactoryConfig::default()
            })
            .build()
            .unwrap();
        assert!(loop_.is_plan_mode_active());
    }

    // -- PermissionMode wiring ----------------------------------------------

    #[test]
    fn permission_mode_allow_all_succeeds_on_tool_call() {
        // ScriptedProvider emits a ToolCall; with AllowAll the loop must
        // reach the dispatcher. The permissive registry returns Ok for
        // any tool, so the turn must finish successfully.
        let scripted = Arc::new(ScriptedProvider::new(vec![tool_call("echo#1", "echo")]));
        let loop_ = AgentFactory::new()
            .with_provider(scripted)
            .with_dispatcher(permissive_registry())
            .with_config(AgentFactoryConfig {
                permission_default: PermissionMode::AllowAll,
                ..AgentFactoryConfig::default()
            })
            .build()
            .unwrap();
        let res = loop_.run_turn(ctx("call echo"), &CancelToken::new());
        assert!(matches!(res.outcome, TurnOutcome::Success));
    }

    #[test]
    fn permission_mode_deny_all_fails_tool_call_with_e5004() {
        let scripted = Arc::new(ScriptedProvider::new(vec![fs_read_call("fs.read#1")]));
        let loop_ = AgentFactory::new()
            .with_provider(scripted)
            .with_config(AgentFactoryConfig {
                permission_default: PermissionMode::DenyAll,
                ..AgentFactoryConfig::default()
            })
            .build()
            .unwrap();
        let res = loop_.run_turn(ctx("call tool"), &CancelToken::new());
        match res.outcome {
            TurnOutcome::ToolFailure { tool_id, code } => {
                // ToolFailure reports the tool name, not the per-call
                // correlation id.
                assert_eq!(tool_id, "fs.read");
                assert_eq!(code, "STRAT-E5004");
            }
            other => panic!("expected ToolFailure STRAT-E5004, got {other:?}"),
        }
    }

    #[test]
    fn permission_mode_prompt_falls_back_to_deny_in_runtime() {
        // The runtime layer maps Prompt -> DenyAllResponder. This test
        // pins that contract — the CLI is the layer that swaps in the
        // real interactive responder.
        let scripted = Arc::new(ScriptedProvider::new(vec![fs_read_call("fs.read#1")]));
        let loop_ = AgentFactory::new()
            .with_provider(scripted)
            .with_config(AgentFactoryConfig {
                permission_default: PermissionMode::Prompt,
                ..AgentFactoryConfig::default()
            })
            .build()
            .unwrap();
        let res = loop_.run_turn(ctx("call tool"), &CancelToken::new());
        match res.outcome {
            TurnOutcome::ToolFailure { code, .. } => assert_eq!(code, "STRAT-E5004"),
            other => panic!("expected ToolFailure(STRAT-E5004), got {other:?}"),
        }
    }

    // -- Error display -------------------------------------------------------

    #[test]
    fn factory_error_display_missing_provider() {
        let s = AgentFactoryError::MissingProvider.to_string();
        assert!(s.contains("provider is required"), "got: {s}");
    }

    #[test]
    fn factory_error_display_agent_loop_build() {
        let s = AgentFactoryError::AgentLoopBuild("nope".into()).to_string();
        assert!(s.contains("nope"), "got: {s}");
    }

    #[test]
    fn factory_error_is_an_error_type() {
        fn assert_error<T: Error>(_: &T) {}
        assert_error(&AgentFactoryError::MissingProvider);
    }

    // -- default_factory_with_dispatchers -----------------------------------

    #[test]
    fn default_factory_with_dispatchers_has_fs_and_shell() {
        let tmp = tempfile::TempDir::new().unwrap();
        let factory = default_factory_with_dispatchers(tmp.path().to_path_buf());
        // Inspect the dispatcher we configured.
        let dispatcher = factory.dispatcher.as_ref().expect("dispatcher set");
        let ids = dispatcher.ids();
        assert!(ids.contains(&"fs.read"), "ids: {ids:?}");
        assert!(ids.contains(&"shell.exec"), "ids: {ids:?}");
    }

    #[test]
    fn default_factory_with_dispatchers_builds_with_provider() {
        let tmp = tempfile::TempDir::new().unwrap();
        let factory = default_factory_with_dispatchers(tmp.path().to_path_buf())
            .with_provider(Arc::new(EchoProvider::new("")));
        let loop_ = factory.build().unwrap();
        let res = loop_.run_turn(ctx("hi"), &CancelToken::new());
        assert!(matches!(res.outcome, TurnOutcome::Success));
    }

    // -- Send + Sync + Clone smoke ------------------------------------------

    #[test]
    fn factory_is_send_sync_and_cloneable() {
        fn assert_send_sync_clone<T: Send + Sync + Clone>() {}
        assert_send_sync_clone::<AgentFactory>();
    }

    #[test]
    fn concurrent_build_produces_independent_loops() {
        let handles: Vec<_> = (0..4)
            .map(|_| {
                thread::spawn(|| {
                    let loop_ = AgentFactory::echo().unwrap();
                    let res = loop_.run_turn(ctx("hi"), &CancelToken::new());
                    matches!(res.outcome, TurnOutcome::Success)
                })
            })
            .collect();
        for h in handles {
            assert!(h.join().unwrap());
        }
    }

    // -- with_config override ------------------------------------------------

    #[test]
    fn with_config_overrides_individual_fields() {
        let factory = AgentFactory::new().with_config(AgentFactoryConfig {
            max_tool_calls_per_turn: 1,
            max_turn_duration_ms: 42,
            ..AgentFactoryConfig::default()
        });
        let c = factory.config();
        assert_eq!(c.max_tool_calls_per_turn, 1);
        assert_eq!(c.max_turn_duration_ms, 42);
        assert!(!c.plan_mode);
    }

    #[test]
    fn tool_call_budget_one_hits_budget_exceeded_on_second_call() {
        // Two ToolCall blocks; budget = 1; the second must trigger
        // BudgetExceeded { kind: "tool_calls" }. Allow-all so we don't
        // fail at the permission check first; permissive registry so
        // the first call returns Ok and the loop advances to the second.
        let scripted = Arc::new(ScriptedProvider::new(vec![
            tool_call("echo#1", "echo"),
            tool_call("echo#2", "echo"),
        ]));
        let loop_ = AgentFactory::new()
            .with_provider(scripted)
            .with_dispatcher(permissive_registry())
            .with_config(AgentFactoryConfig {
                max_tool_calls_per_turn: 1,
                permission_default: PermissionMode::AllowAll,
                ..AgentFactoryConfig::default()
            })
            .build()
            .unwrap();
        let res = loop_.run_turn(ctx("call tools"), &CancelToken::new());
        match res.outcome {
            TurnOutcome::BudgetExceeded { kind } => assert_eq!(kind, "tool_calls"),
            other => panic!("expected BudgetExceeded(tool_calls), got {other:?}"),
        }
    }

    // -- Router classify smoke (with_router plumbs through) ------------------

    #[test]
    fn with_custom_router_routed_intent_runs_to_completion() {
        let router = IntentRouter::with_rules(vec![IntentRule {
            pattern: IntentPattern::Contains("hello".into()),
            intent: Intent::Chat,
            weight: 1.0,
            tier: ModelTier::Low,
            role: SuggestedRole::Default,
            caps: vec!["chat".into()],
        }])
        .unwrap();
        let loop_ = AgentFactory::new()
            .with_provider(Arc::new(EchoProvider::new("")))
            .with_router(router)
            .build()
            .unwrap();
        let res = loop_.run_turn(ctx("hello world"), &CancelToken::new());
        assert!(matches!(res.outcome, TurnOutcome::Success));
    }

    // -- ScriptedProvider self-cover ----------------------------------------

    #[test]
    fn scripted_provider_id_caps_and_empty_script_are_covered() {
        // ScriptedProvider's id()/capabilities()/empty-script branches
        // are not naturally exercised by the agent-loop happy paths.
        // This test pins them so the test-helper carries its own
        // coverage instead of relying on incidental hits.
        let p = ScriptedProvider {
            script: Mutex::new(Vec::new()),
        };
        assert_eq!(p.id(), "scripted");
        assert_eq!(p.capabilities().len(), 1);
        // empty-script branch: returns `Vec::new()` without popping.
        let blocks = p.generate(
            &GenerateRequest {
                model: ModelId::from("any"),
                prompt: "x".into(),
                max_blocks: 1,
                system_override: None,
                history: Vec::new(),
                sampler: crate::provider::SamplerParams::default(),
            },
            &CancelToken::new(),
        );
        assert!(blocks.is_empty());
    }

    // -- Debug smoke --------------------------------------------------------

    #[test]
    fn factory_debug_smoke() {
        let factory = AgentFactory::new().with_provider(Arc::new(EchoProvider::new("")));
        let rendered = format!("{factory:?}");
        assert!(rendered.contains("AgentFactory"));
        assert!(rendered.contains("provider_set: true"));
    }
}
