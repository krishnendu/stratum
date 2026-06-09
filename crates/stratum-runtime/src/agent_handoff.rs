//! `AgentHandoff` — multi-role coordinator.
//!
//! Routes a turn through one or more [`AgentLoop`]s based on a hand-off
//! sentinel emitted by the assistant's last block. Each role corresponds
//! to a fully-built [`AgentLoop`] registered under a [`SuggestedRole`].
//!
//! ## Sentinel
//!
//! The orchestrator inspects the **last** [`Block::Text`] emitted by a
//! successful turn. If its text starts with `<handoff:<role>>`, the next
//! turn is routed to that role. The marker is stripped from the prompt
//! handed to the follow-up role so the chain does not loop infinitely on
//! the same sentinel (subject to `HandoffPolicy::allow_self_handoff`).
//!
//! ## Depth + self-handoff
//!
//! Chains are capped at [`HandoffPolicy::max_chain_depth`] hops (counting
//! the initial role as hop 0; the cap controls how many *follow-ups* can
//! be chained). When the limit is hit the orchestrator returns
//! [`HandoffError::ChainTooDeep`]. Self-hand-offs (a role nominating
//! itself) are governed by [`HandoffPolicy::allow_self_handoff`].
//!
//! ## Error catalog
//!
//! No new `STRAT-Exxxx` codes — failures surface as typed variants of
//! [`HandoffError`]. The downstream observability layer maps them onto
//! existing codes when needed.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use stratum_types::Block;

use crate::agent_loop::{AgentLoop, TurnContext, TurnResult};
use crate::cancel::CancelToken;
use crate::conversation::TurnOutcome;
use crate::intent_router::{RoutedIntent, SuggestedRole};

// ---------------------------------------------------------------------------
// Policy
// ---------------------------------------------------------------------------

/// Static policy controlling [`AgentHandoff::run_turn_with_handoff`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct HandoffPolicy {
    /// When `true`, a role may nominate itself as the next hop. When
    /// `false`, a self-nomination is rejected with
    /// [`HandoffError::SelfHandoffNotAllowed`].
    pub allow_self_handoff: bool,
    /// Maximum number of follow-up hops. The initial role is hop 0; each
    /// observed sentinel grows the depth by 1. Exceeding this bound
    /// surfaces [`HandoffError::ChainTooDeep`].
    pub max_chain_depth: u8,
}

impl Default for HandoffPolicy {
    fn default() -> Self {
        Self {
            allow_self_handoff: false,
            max_chain_depth: 4,
        }
    }
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

/// Adapter wrapping [`SuggestedRole`] with a stable `Ord` impl so we
/// can use it as a `BTreeMap` key without altering the upstream type.
/// Order matches the declaration order in [`SuggestedRole`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct OrdRole(SuggestedRole);

impl OrdRole {
    const fn rank(self) -> u8 {
        match self.0 {
            SuggestedRole::Default => 0,
            SuggestedRole::Cavemanish => 1,
            SuggestedRole::Coder => 2,
            SuggestedRole::Polisher => 3,
            SuggestedRole::Researcher => 4,
        }
    }
}

impl PartialOrd for OrdRole {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrdRole {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.rank().cmp(&other.rank())
    }
}

/// Role → [`AgentLoop`] lookup table.
#[derive(Default, Clone)]
pub struct AgentRegistry {
    agents: BTreeMap<OrdRole, Arc<AgentLoop>>,
}

impl fmt::Debug for AgentRegistry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AgentRegistry")
            .field("len", &self.agents.len())
            .field("roles", &self.roles())
            .finish()
    }
}

impl AgentRegistry {
    /// Construct an empty registry.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            agents: BTreeMap::new(),
        }
    }

    /// Register `loop_` under `role`, returning the previously-registered
    /// loop for that role if any.
    pub fn register(
        &mut self,
        role: SuggestedRole,
        loop_: Arc<AgentLoop>,
    ) -> Option<Arc<AgentLoop>> {
        self.agents.insert(OrdRole(role), loop_)
    }

    /// Look up the loop registered for `role`.
    #[must_use]
    #[allow(
        clippy::trivially_copy_pass_by_ref,
        reason = "spec asks for &SuggestedRole to match the rest of the runtime surface"
    )]
    pub fn get(&self, role: &SuggestedRole) -> Option<Arc<AgentLoop>> {
        self.agents.get(&OrdRole(*role)).map(Arc::clone)
    }

    /// All registered roles, in declaration-order.
    #[must_use]
    pub fn roles(&self) -> Vec<SuggestedRole> {
        self.agents.keys().map(|k| k.0).collect()
    }

    /// Number of registered roles.
    #[must_use]
    pub fn len(&self) -> usize {
        self.agents.len()
    }

    /// Whether the registry has no roles registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.agents.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Result / step
// ---------------------------------------------------------------------------

/// One link in a hand-off chain.
#[derive(Debug, Clone)]
pub struct HandoffStep {
    /// Role that emitted the sentinel (for the first hop, the role that
    /// initially ran).
    pub from_role: SuggestedRole,
    /// Role nominated for this hop. For hop 0 this equals `from_role`.
    pub to_role: SuggestedRole,
    /// Underlying [`TurnResult`] for this hop.
    pub turn_result: TurnResult,
}

/// Aggregate result of a multi-hop hand-off chain.
#[derive(Debug, Clone)]
pub struct HandoffResult {
    /// Role that produced the final hop.
    pub final_role: SuggestedRole,
    /// Each hop, in arrival order.
    pub steps: Vec<HandoffStep>,
    /// Blocks emitted by the final hop, verbatim.
    pub final_blocks: Vec<Block>,
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Typed failure surface of
/// [`AgentHandoff::run_turn_with_handoff`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HandoffError {
    /// No agent is registered for the requested role and the default
    /// also has no entry.
    NoSuchRole(SuggestedRole),
    /// The chain exceeded [`HandoffPolicy::max_chain_depth`].
    ChainTooDeep,
    /// A role nominated itself but
    /// [`HandoffPolicy::allow_self_handoff`] is `false`.
    SelfHandoffNotAllowed {
        /// The role that attempted the self-hand-off.
        role: SuggestedRole,
    },
    /// The sentinel was syntactically valid but referenced an unknown
    /// role string.
    BadHandoffMarker(String),
}

impl fmt::Display for HandoffError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoSuchRole(role) => write!(f, "no agent registered for role: {role:?}"),
            Self::ChainTooDeep => f.write_str("hand-off chain exceeded max_chain_depth"),
            Self::SelfHandoffNotAllowed { role } => {
                write!(f, "self-hand-off rejected by policy for role: {role:?}")
            }
            Self::BadHandoffMarker(s) => write!(f, "bad hand-off marker: {s}"),
        }
    }
}

impl Error for HandoffError {}

// ---------------------------------------------------------------------------
// AgentHandoff
// ---------------------------------------------------------------------------

/// Multi-role coordinator.
#[derive(Debug, Clone)]
pub struct AgentHandoff {
    registry: AgentRegistry,
    default_role: SuggestedRole,
    policy: HandoffPolicy,
}

impl AgentHandoff {
    /// Construct a coordinator from a registry, fall-back role, and
    /// chain policy.
    #[must_use]
    pub const fn new(
        registry: AgentRegistry,
        default_role: SuggestedRole,
        policy: HandoffPolicy,
    ) -> Self {
        Self {
            registry,
            default_role,
            policy,
        }
    }

    /// Snapshot of the registered roles.
    #[must_use]
    pub fn roles(&self) -> Vec<SuggestedRole> {
        self.registry.roles()
    }

    /// Fallback role.
    #[must_use]
    pub const fn default_role(&self) -> SuggestedRole {
        self.default_role
    }

    /// Active policy.
    #[must_use]
    pub const fn policy(&self) -> HandoffPolicy {
        self.policy
    }

    /// Drive a turn, following any hand-off sentinels emitted along the
    /// way.
    ///
    /// # Errors
    ///
    /// * [`HandoffError::NoSuchRole`] — neither the suggested role nor
    ///   the default role is registered.
    /// * [`HandoffError::ChainTooDeep`] — sentinel chain longer than
    ///   [`HandoffPolicy::max_chain_depth`].
    /// * [`HandoffError::SelfHandoffNotAllowed`] — sentinel pointed at
    ///   the same role under a `allow_self_handoff = false` policy.
    #[allow(
        clippy::needless_pass_by_value,
        reason = "owning the intent matches the agent-loop run_turn signature and is the spec'd surface"
    )]
    pub fn run_turn_with_handoff(
        &self,
        ctx: TurnContext,
        intent: RoutedIntent,
        cancel: &CancelToken,
    ) -> Result<HandoffResult, HandoffError> {
        // Resolve initial role: suggested first, fall back to default.
        let initial_role = self.resolve_role(intent.suggested_role)?;

        let mut current_role = initial_role;
        let mut current_ctx = ctx;
        let mut steps: Vec<HandoffStep> = Vec::new();
        let mut depth: u8 = 0;

        loop {
            // Defensive: the resolved role must still be registered. Any
            // mid-loop divergence (e.g. a sentinel pointing at a
            // never-registered role) is caught here.
            let Some(agent) = self.registry.get(&current_role) else {
                return Err(HandoffError::NoSuchRole(current_role));
            };

            let result = agent.run_turn(current_ctx.clone(), cancel);
            let from_role = current_role;
            let to_role = current_role;
            let outcome_success = matches!(result.outcome, TurnOutcome::Success);
            let parsed_marker = if outcome_success {
                last_text_block(&result.blocks).and_then(parse_handoff_marker_with_stripped)
            } else {
                None
            };

            steps.push(HandoffStep {
                from_role,
                to_role,
                turn_result: result.clone(),
            });

            let Some((next_role, stripped)) = parsed_marker else {
                // No sentinel — chain ends.
                return Ok(HandoffResult {
                    final_role: current_role,
                    final_blocks: result.blocks,
                    steps,
                });
            };

            // Self-hand-off policy gate.
            if next_role == current_role && !self.policy.allow_self_handoff {
                return Err(HandoffError::SelfHandoffNotAllowed { role: next_role });
            }

            // Depth gate — count this would-be hop.
            if depth >= self.policy.max_chain_depth {
                return Err(HandoffError::ChainTooDeep);
            }
            depth = depth.saturating_add(1);

            // Resolve the next role through the same fall-back path.
            current_role = self.resolve_role(next_role)?;

            // Build the next context: same model, fresh turn id (bumped),
            // marker stripped from the prompt so the chain does not loop
            // on the same sentinel.
            let next_turn_id = crate::observability::TurnId(current_ctx.turn_id.0 + 1);
            current_ctx = TurnContext {
                user_prompt: stripped,
                model: current_ctx.model.clone(),
                turn_id: next_turn_id,
                started_at: current_ctx.started_at,
            };
        }
    }

    fn resolve_role(&self, requested: SuggestedRole) -> Result<SuggestedRole, HandoffError> {
        if self.registry.get(&requested).is_some() {
            return Ok(requested);
        }
        if self.registry.get(&self.default_role).is_some() {
            return Ok(self.default_role);
        }
        Err(HandoffError::NoSuchRole(requested))
    }
}

// ---------------------------------------------------------------------------
// Sentinel parsing
// ---------------------------------------------------------------------------

/// Parse a hand-off marker from `text`.
///
/// Returns `Some(role)` when `text` starts with the literal sequence
/// `<handoff:<role>>` and `<role>` deserialises as a [`SuggestedRole`]
/// (snake-case or `PascalCase`). Returns `None` otherwise.
#[must_use]
pub fn parse_handoff_marker(text: &str) -> Option<SuggestedRole> {
    parse_handoff_marker_with_stripped(text).map(|(role, _)| role)
}

/// Same as [`parse_handoff_marker`] but also returns the prompt text
/// with the marker stripped. Crate-private so callers cannot grow a
/// second public surface by accident.
fn parse_handoff_marker_with_stripped(text: &str) -> Option<(SuggestedRole, String)> {
    let rest = text.strip_prefix("<handoff:")?;
    let end = rest.find('>')?;
    let role_str = &rest[..end];
    let remainder = &rest[end + 1..];
    let role = parse_role(role_str)?;
    Some((role, remainder.to_string()))
}

fn parse_role(s: &str) -> Option<SuggestedRole> {
    // Try snake_case first (the serde rename_all), then PascalCase.
    let snake = format!("\"{s}\"");
    if let Ok(role) = serde_json::from_str::<SuggestedRole>(&snake) {
        return Some(role);
    }
    // Fall back to a case-insensitive match against the known variants.
    match s.to_ascii_lowercase().as_str() {
        "default" => Some(SuggestedRole::Default),
        "cavemanish" => Some(SuggestedRole::Cavemanish),
        "polisher" => Some(SuggestedRole::Polisher),
        "coder" => Some(SuggestedRole::Coder),
        "researcher" => Some(SuggestedRole::Researcher),
        _ => None,
    }
}

fn last_text_block(blocks: &[Block]) -> Option<&str> {
    blocks.iter().rev().find_map(|b| match b {
        Block::Text { text } => Some(text.as_str()),
        _ => None,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    use std::collections::BTreeSet;
    use std::sync::Mutex;
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    use stratum_types::{Capability, ModelId};

    use crate::agent_factory::AgentFactory;
    use crate::agent_loop::AgentLoopConfig;
    use crate::event_log::{EventEmitter, FixedEventClock, MemoryEventSink};
    use crate::intent_router::{Intent, IntentRouter};
    use crate::model_catalog::ModelTier;
    use crate::observability::TurnId;
    use crate::permission_prompt::{AllowAllResponder, PermissionStore, PromptIdGen};
    use crate::plan_mode::PlanMode;
    use crate::provider::{GenerateRequest, Provider};
    use crate::tool_invocation::RegistryDispatcher;
    use crate::tools::CapabilityMatrix;

    fn t0() -> SystemTime {
        UNIX_EPOCH + Duration::from_secs(1_700_000_000)
    }

    fn ctx(prompt: &str) -> TurnContext {
        TurnContext {
            user_prompt: prompt.into(),
            model: ModelId::from("echo"),
            turn_id: TurnId(1),
            started_at: t0(),
        }
    }

    fn echo_loop_arc() -> Arc<AgentLoop> {
        Arc::new(AgentFactory::echo().expect("echo factory builds"))
    }

    fn routed_with_role(role: SuggestedRole) -> RoutedIntent {
        RoutedIntent {
            intent: Intent::Chat,
            confidence: 0.0,
            required_tier: ModelTier::Low,
            suggested_role: role,
            hinted_capabilities: BTreeSet::new(),
        }
    }

    // ---- Provider used for chain / self-handoff tests --------------------

    #[derive(Debug)]
    struct ConstTextProvider {
        text: String,
    }

    impl Provider for ConstTextProvider {
        #[allow(clippy::unnecessary_literal_bound)]
        fn id(&self) -> &str {
            "const_text"
        }
        fn capabilities(&self) -> &'static [Capability] {
            const CAPS: &[Capability] = &[Capability::Generate];
            CAPS
        }
        fn generate(&self, _req: &GenerateRequest, _cancel: &CancelToken) -> Vec<Block> {
            vec![Block::Text {
                text: self.text.clone(),
            }]
        }
    }

    /// Provider that always echoes the prompt verbatim as a single
    /// `Block::Text` — used to drive `ChainTooDeep` where every hop
    /// re-emits the same sentinel.
    #[derive(Debug)]
    struct PromptEchoProvider;

    impl Provider for PromptEchoProvider {
        #[allow(clippy::unnecessary_literal_bound)]
        fn id(&self) -> &str {
            "prompt_echo"
        }
        fn capabilities(&self) -> &'static [Capability] {
            const CAPS: &[Capability] = &[Capability::Generate];
            CAPS
        }
        fn generate(&self, req: &GenerateRequest, _cancel: &CancelToken) -> Vec<Block> {
            // Keep the prompt as-is so the sentinel survives intact.
            vec![Block::Text {
                text: req.prompt.clone(),
            }]
        }
    }

    /// Provider that records each prompt it sees so the test can pin
    /// the marker-stripping contract.
    #[derive(Debug, Default)]
    struct CapturingProvider {
        prompts: Mutex<Vec<String>>,
    }

    impl Provider for CapturingProvider {
        #[allow(clippy::unnecessary_literal_bound)]
        fn id(&self) -> &str {
            "capturing"
        }
        fn capabilities(&self) -> &'static [Capability] {
            const CAPS: &[Capability] = &[Capability::Generate];
            CAPS
        }
        fn generate(&self, req: &GenerateRequest, _cancel: &CancelToken) -> Vec<Block> {
            self.prompts.lock().unwrap().push(req.prompt.clone());
            // First hop emits a sentinel; subsequent hops emit plain text
            // so the chain terminates.
            if req.prompt.starts_with("<handoff:") {
                vec![Block::Text {
                    text: req.prompt.clone(),
                }]
            } else {
                vec![Block::Text {
                    text: "done".into(),
                }]
            }
        }
    }

    fn build_loop_with_provider(provider: Arc<dyn Provider>) -> Arc<AgentLoop> {
        let sink = Arc::new(MemoryEventSink::new());
        let events = Arc::new(EventEmitter::with_clock(
            sink,
            Box::new(FixedEventClock(t0())),
        ));
        let l = AgentLoop::builder()
            .with_provider(provider)
            .with_router(IntentRouter::empty())
            .with_permission_store(Arc::new(PermissionStore::new()))
            .with_prompt_gen(Arc::new(PromptIdGen::new()))
            .with_responder(Arc::new(AllowAllResponder))
            .with_events(events)
            .with_capability_matrix(Arc::new(CapabilityMatrix::new()))
            .with_plan_mode(Arc::new(PlanMode::new()))
            .with_dispatcher(Arc::new(RegistryDispatcher::new()))
            .with_config(AgentLoopConfig::default())
            .build()
            .expect("build loop");
        Arc::new(l)
    }

    // ---- HandoffPolicy ---------------------------------------------------

    #[test]
    fn handoff_policy_default_matches_documented() {
        let p = HandoffPolicy::default();
        assert!(!p.allow_self_handoff);
        assert_eq!(p.max_chain_depth, 4);
    }

    #[test]
    fn handoff_policy_is_copy_and_debug() {
        let p = HandoffPolicy::default();
        let q = p; // Copy.
        let _ = format!("{p:?} {q:?}");
        assert_eq!(p, q);
    }

    // ---- AgentRegistry ---------------------------------------------------

    #[test]
    fn registry_register_and_get_round_trip() {
        let mut reg = AgentRegistry::new();
        let l = echo_loop_arc();
        assert!(reg.register(SuggestedRole::Coder, Arc::clone(&l)).is_none());
        let got = reg.get(&SuggestedRole::Coder).expect("registered");
        assert!(Arc::ptr_eq(&got, &l));
        // Re-registering returns the previous entry.
        let l2 = echo_loop_arc();
        let prev = reg.register(SuggestedRole::Coder, Arc::clone(&l2));
        assert!(prev.is_some());
        assert!(Arc::ptr_eq(&prev.unwrap(), &l));
    }

    #[test]
    fn registry_roles_returns_sorted_list() {
        let mut reg = AgentRegistry::new();
        reg.register(SuggestedRole::Researcher, echo_loop_arc());
        reg.register(SuggestedRole::Default, echo_loop_arc());
        reg.register(SuggestedRole::Cavemanish, echo_loop_arc());
        let roles = reg.roles();
        // BTreeMap iteration is ordered; we don't care about the
        // particular ordering rule, only that it is stable + sorted.
        let mut sorted = roles.clone();
        sorted.sort_by_key(|r| match r {
            SuggestedRole::Default => 0,
            SuggestedRole::Cavemanish => 1,
            SuggestedRole::Coder => 2,
            SuggestedRole::Polisher => 3,
            SuggestedRole::Researcher => 4,
        });
        assert_eq!(roles, sorted);
        assert_eq!(roles.len(), 3);
    }

    #[test]
    fn registry_len_tracks_entries() {
        let mut reg = AgentRegistry::new();
        assert_eq!(reg.len(), 0);
        assert!(reg.is_empty());
        reg.register(SuggestedRole::Default, echo_loop_arc());
        reg.register(SuggestedRole::Coder, echo_loop_arc());
        assert_eq!(reg.len(), 2);
        assert!(!reg.is_empty());
    }

    #[test]
    fn empty_registry_has_no_roles() {
        let reg = AgentRegistry::new();
        assert!(reg.roles().is_empty());
        assert_eq!(reg.len(), 0);
        assert!(reg.is_empty());
    }

    #[test]
    fn registry_default_is_empty() {
        let reg = AgentRegistry::default();
        assert!(reg.is_empty());
    }

    #[test]
    fn registry_debug_smoke() {
        let mut reg = AgentRegistry::new();
        reg.register(SuggestedRole::Coder, echo_loop_arc());
        let s = format!("{reg:?}");
        assert!(s.contains("AgentRegistry"));
        assert!(s.contains("len"));
    }

    // ---- parse_handoff_marker -------------------------------------------

    #[test]
    fn parse_marker_snake_case_polisher_returns_some() {
        assert_eq!(
            parse_handoff_marker("<handoff:polisher>hello"),
            Some(SuggestedRole::Polisher)
        );
    }

    #[test]
    fn parse_marker_pascal_case_polisher_returns_some() {
        // Case-insensitive fallback covers both PascalCase and
        // ALL-CAPS variants the model might emit.
        assert_eq!(
            parse_handoff_marker("<handoff:Polisher>"),
            Some(SuggestedRole::Polisher)
        );
    }

    #[test]
    fn parse_marker_plain_text_returns_none() {
        assert_eq!(parse_handoff_marker("hello"), None);
    }

    #[test]
    fn parse_marker_bogus_role_returns_none() {
        assert_eq!(parse_handoff_marker("<handoff:bogus>"), None);
    }

    #[test]
    fn parse_marker_no_closing_bracket_returns_none() {
        assert_eq!(parse_handoff_marker("<handoff:polisher"), None);
    }

    #[test]
    fn parse_marker_all_roles_round_trip() {
        for role in [
            SuggestedRole::Default,
            SuggestedRole::Cavemanish,
            SuggestedRole::Polisher,
            SuggestedRole::Coder,
            SuggestedRole::Researcher,
        ] {
            let snake = serde_json::to_string(&role).unwrap();
            // `"polisher"` -> `polisher`.
            let bare = snake.trim_matches('"');
            let text = format!("<handoff:{bare}>tail");
            assert_eq!(parse_handoff_marker(&text), Some(role));
        }
    }

    // ---- run_turn_with_handoff happy paths ------------------------------

    #[test]
    fn run_turn_with_handoff_no_marker_is_single_step() {
        let mut reg = AgentRegistry::new();
        let loop_ = build_loop_with_provider(Arc::new(ConstTextProvider {
            text: "all done".into(),
        }));
        reg.register(SuggestedRole::Default, loop_);
        let h = AgentHandoff::new(reg, SuggestedRole::Default, HandoffPolicy::default());

        let result = h
            .run_turn_with_handoff(
                ctx("hi"),
                routed_with_role(SuggestedRole::Default),
                &CancelToken::new(),
            )
            .expect("handoff ok");
        assert_eq!(result.steps.len(), 1);
        assert_eq!(result.final_role, SuggestedRole::Default);
        assert!(!result.final_blocks.is_empty());
    }

    #[test]
    fn run_turn_with_handoff_single_hop_yields_two_steps() {
        // First hop emits a sentinel pointing at Polisher; Polisher
        // returns a plain text block so the chain terminates.
        let first = build_loop_with_provider(Arc::new(ConstTextProvider {
            text: "<handoff:polisher>follow-up prompt".into(),
        }));
        let second = build_loop_with_provider(Arc::new(ConstTextProvider {
            text: "all polished".into(),
        }));
        let mut reg = AgentRegistry::new();
        reg.register(SuggestedRole::Coder, first);
        reg.register(SuggestedRole::Polisher, second);
        let h = AgentHandoff::new(reg, SuggestedRole::Coder, HandoffPolicy::default());

        let result = h
            .run_turn_with_handoff(
                ctx("write code"),
                routed_with_role(SuggestedRole::Coder),
                &CancelToken::new(),
            )
            .expect("handoff ok");
        assert_eq!(result.steps.len(), 2);
        assert_eq!(result.steps[0].from_role, SuggestedRole::Coder);
        assert_eq!(result.steps[1].from_role, SuggestedRole::Polisher);
        assert_eq!(result.final_role, SuggestedRole::Polisher);
    }

    #[test]
    fn run_turn_with_handoff_chain_too_deep_when_sentinel_loops() {
        // PromptEchoProvider re-emits whatever prompt it receives. We
        // seed the chain with a stack of self-targeted sentinels under
        // `allow_self_handoff = true` so the policy gate does not fire
        // before the depth cap does.
        let policy = HandoffPolicy {
            allow_self_handoff: true,
            max_chain_depth: 2,
        };
        let coder = build_loop_with_provider(Arc::new(PromptEchoProvider));
        let mut reg = AgentRegistry::new();
        reg.register(SuggestedRole::Coder, coder);
        let h = AgentHandoff::new(reg, SuggestedRole::Coder, policy);
        let result = h.run_turn_with_handoff(
            ctx("<handoff:coder><handoff:coder><handoff:coder><handoff:coder><handoff:coder>tail"),
            routed_with_role(SuggestedRole::Coder),
            &CancelToken::new(),
        );
        match result {
            Err(HandoffError::ChainTooDeep) => {}
            other => panic!("expected ChainTooDeep, got {other:?}"),
        }
    }

    #[test]
    fn run_turn_with_handoff_self_handoff_rejected_when_policy_off() {
        let coder = build_loop_with_provider(Arc::new(ConstTextProvider {
            text: "<handoff:coder>more".into(),
        }));
        let mut reg = AgentRegistry::new();
        reg.register(SuggestedRole::Coder, coder);
        let h = AgentHandoff::new(reg, SuggestedRole::Coder, HandoffPolicy::default());
        let result = h.run_turn_with_handoff(
            ctx("seed"),
            routed_with_role(SuggestedRole::Coder),
            &CancelToken::new(),
        );
        match result {
            Err(HandoffError::SelfHandoffNotAllowed { role }) => {
                assert_eq!(role, SuggestedRole::Coder);
            }
            other => panic!("expected SelfHandoffNotAllowed, got {other:?}"),
        }
    }

    #[test]
    fn run_turn_with_handoff_self_handoff_allowed_when_policy_on() {
        // First call emits a self-sentinel; second call emits a plain
        // text block so the chain terminates after one hop.
        struct ToggleProvider {
            n: Mutex<u8>,
        }
        impl std::fmt::Debug for ToggleProvider {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("ToggleProvider")
            }
        }
        impl Provider for ToggleProvider {
            #[allow(clippy::unnecessary_literal_bound)]
            fn id(&self) -> &str {
                "toggle"
            }
            fn capabilities(&self) -> &'static [Capability] {
                const CAPS: &[Capability] = &[Capability::Generate];
                CAPS
            }
            fn generate(&self, _req: &GenerateRequest, _cancel: &CancelToken) -> Vec<Block> {
                let mut g = self.n.lock().unwrap();
                *g = g.saturating_add(1);
                if *g == 1 {
                    vec![Block::Text {
                        text: "<handoff:coder>next".into(),
                    }]
                } else {
                    vec![Block::Text {
                        text: "settled".into(),
                    }]
                }
            }
        }
        let coder = build_loop_with_provider(Arc::new(ToggleProvider { n: Mutex::new(0) }));
        let mut reg = AgentRegistry::new();
        reg.register(SuggestedRole::Coder, coder);
        let policy = HandoffPolicy {
            allow_self_handoff: true,
            max_chain_depth: 4,
        };
        let h = AgentHandoff::new(reg, SuggestedRole::Coder, policy);
        let result = h
            .run_turn_with_handoff(
                ctx("seed"),
                routed_with_role(SuggestedRole::Coder),
                &CancelToken::new(),
            )
            .expect("self hop allowed");
        assert_eq!(result.steps.len(), 2);
        assert_eq!(result.final_role, SuggestedRole::Coder);
    }

    // ---- fall-back semantics --------------------------------------------

    #[test]
    fn run_turn_falls_back_to_default_when_intent_role_missing() {
        // Registry has only Default; intent suggests Coder.
        let only_default = build_loop_with_provider(Arc::new(ConstTextProvider {
            text: "served by default".into(),
        }));
        let mut reg = AgentRegistry::new();
        reg.register(SuggestedRole::Default, only_default);
        let h = AgentHandoff::new(reg, SuggestedRole::Default, HandoffPolicy::default());
        let result = h
            .run_turn_with_handoff(
                ctx("hi"),
                routed_with_role(SuggestedRole::Coder),
                &CancelToken::new(),
            )
            .expect("falls back");
        assert_eq!(result.final_role, SuggestedRole::Default);
        assert_eq!(result.steps.len(), 1);
    }

    #[test]
    fn run_turn_no_such_role_when_neither_intent_nor_default_registered() {
        let reg = AgentRegistry::new();
        let h = AgentHandoff::new(reg, SuggestedRole::Default, HandoffPolicy::default());
        let err = h
            .run_turn_with_handoff(
                ctx("hi"),
                routed_with_role(SuggestedRole::Coder),
                &CancelToken::new(),
            )
            .expect_err("no role");
        match err {
            HandoffError::NoSuchRole(role) => assert_eq!(role, SuggestedRole::Coder),
            other => panic!("expected NoSuchRole, got {other:?}"),
        }
    }

    #[test]
    fn run_turn_initial_role_exists_default_ignored() {
        // Registry has Coder only; intent suggests Coder. Default
        // (Researcher) is not registered but should not matter.
        let coder = build_loop_with_provider(Arc::new(ConstTextProvider {
            text: "coded".into(),
        }));
        let mut reg = AgentRegistry::new();
        reg.register(SuggestedRole::Coder, coder);
        let h = AgentHandoff::new(reg, SuggestedRole::Researcher, HandoffPolicy::default());
        let result = h
            .run_turn_with_handoff(
                ctx("hi"),
                routed_with_role(SuggestedRole::Coder),
                &CancelToken::new(),
            )
            .expect("served");
        assert_eq!(result.final_role, SuggestedRole::Coder);
        assert_eq!(result.steps.len(), 1);
    }

    // ---- HandoffError Display -------------------------------------------

    #[test]
    fn handoff_error_display_no_such_role() {
        let s = HandoffError::NoSuchRole(SuggestedRole::Polisher).to_string();
        assert!(s.contains("no agent"), "got: {s}");
    }

    #[test]
    fn handoff_error_display_chain_too_deep() {
        let s = HandoffError::ChainTooDeep.to_string();
        assert!(s.contains("max_chain_depth"), "got: {s}");
    }

    #[test]
    fn handoff_error_display_self_handoff_not_allowed() {
        let s = HandoffError::SelfHandoffNotAllowed {
            role: SuggestedRole::Coder,
        }
        .to_string();
        assert!(s.contains("self-hand-off"), "got: {s}");
    }

    #[test]
    fn handoff_error_display_bad_handoff_marker() {
        let s = HandoffError::BadHandoffMarker("garbage".into()).to_string();
        assert!(s.contains("bad hand-off marker"), "got: {s}");
        assert!(s.contains("garbage"));
    }

    #[test]
    fn handoff_error_is_an_error_type() {
        fn assert_error<T: Error>(_: &T) {}
        assert_error(&HandoffError::ChainTooDeep);
    }

    // ---- Send + Sync smoke ----------------------------------------------

    #[test]
    fn agent_handoff_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<AgentHandoff>();
        assert_send_sync::<AgentRegistry>();
        assert_send_sync::<HandoffPolicy>();
        assert_send_sync::<HandoffResult>();
        assert_send_sync::<HandoffStep>();
        assert_send_sync::<HandoffError>();
    }

    // ---- Accessor smoke -------------------------------------------------

    #[test]
    fn agent_handoff_accessors_round_trip() {
        let mut reg = AgentRegistry::new();
        reg.register(SuggestedRole::Coder, echo_loop_arc());
        let policy = HandoffPolicy {
            allow_self_handoff: true,
            max_chain_depth: 7,
        };
        let h = AgentHandoff::new(reg, SuggestedRole::Default, policy);
        assert_eq!(h.default_role(), SuggestedRole::Default);
        assert_eq!(h.policy(), policy);
        assert_eq!(h.roles(), vec![SuggestedRole::Coder]);
        let s = format!("{h:?}");
        assert!(s.contains("AgentHandoff"));
    }

    // ---- CapturingProvider pin (marker stripped) ------------------------

    #[test]
    fn marker_is_stripped_before_next_hop() {
        let cap = Arc::new(CapturingProvider::default());
        // Same Arc handed both as the dyn Provider and as the test
        // observer.
        let provider_dyn: Arc<dyn Provider> = cap.clone();
        let l = build_loop_with_provider(provider_dyn);
        let mut reg = AgentRegistry::new();
        reg.register(SuggestedRole::Coder, l);
        let policy = HandoffPolicy {
            allow_self_handoff: true,
            max_chain_depth: 4,
        };
        let h = AgentHandoff::new(reg, SuggestedRole::Coder, policy);
        let _ = h
            .run_turn_with_handoff(
                ctx("<handoff:coder>follow-up text"),
                routed_with_role(SuggestedRole::Coder),
                &CancelToken::new(),
            )
            .expect("chain ok");
        let prompts = {
            let g = cap.prompts.lock().unwrap();
            g.clone()
        };
        assert_eq!(prompts.len(), 2);
        assert_eq!(prompts[0], "<handoff:coder>follow-up text");
        assert_eq!(prompts[1], "follow-up text");
    }
}
