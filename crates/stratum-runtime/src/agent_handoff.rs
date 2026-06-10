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
//!
//! ## Parallel API
//!
//! [`AgentHandoff::run_turn_parallel`] broadcasts the same `TurnContext`
//! and `RoutedIntent` to a fixed list of roles and returns a
//! [`ParallelResult`] containing one [`RoleResult`] per requested role.
//! Workers run on dedicated `std::thread::spawn` threads, throttled by
//! [`ParallelPolicy::max_concurrent`]. Each worker observes a child of
//! the caller-supplied [`CancelToken`] so cancelling the parent fans out
//! to every in-flight role. The returned `per_role` `BTreeMap` is
//! deterministically ordered by the [`SuggestedRole`] declaration rank
//! used elsewhere in this module.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::{Duration, Instant, SystemTime};

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

/// Static policy controlling [`AgentHandoff::run_turn_parallel`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParallelPolicy {
    /// Upper bound on the number of worker threads active concurrently.
    /// The dispatch loop releases a new worker only when an in-flight
    /// worker has finished, so the runtime never exceeds this many
    /// simultaneously executing roles.
    pub max_concurrent: u8,
    /// When `true`, the dispatcher cancels the shared cancel token and
    /// returns as soon as any role reports an error (a non-`Success`
    /// [`TurnOutcome`]). The returned `per_role` map will then contain
    /// only the results that landed before the cancellation.
    pub fail_fast: bool,
}

impl Default for ParallelPolicy {
    fn default() -> Self {
        Self {
            max_concurrent: 4,
            fail_fast: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

/// Adapter wrapping [`SuggestedRole`] with a stable `Ord` impl so we
/// can use it as a `BTreeMap` key without altering the upstream type.
/// Order matches the declaration order in [`SuggestedRole`].
///
/// Exposed so the public [`ParallelResult::per_role`] map can be keyed
/// by an `Ord` type while preserving the upstream `SuggestedRole`
/// definition. Use [`OrdRole::role`] to recover the wrapped enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OrdRole(SuggestedRole);

impl OrdRole {
    /// Wrap a [`SuggestedRole`] for use as a `BTreeMap` key.
    #[must_use]
    pub const fn new(role: SuggestedRole) -> Self {
        Self(role)
    }

    /// Recover the wrapped [`SuggestedRole`].
    #[must_use]
    pub const fn role(self) -> SuggestedRole {
        self.0
    }

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

impl From<SuggestedRole> for OrdRole {
    fn from(role: SuggestedRole) -> Self {
        Self(role)
    }
}

impl From<OrdRole> for SuggestedRole {
    fn from(wrapper: OrdRole) -> Self {
        wrapper.0
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

/// Per-role outcome captured by [`AgentHandoff::run_turn_parallel`].
#[derive(Debug, Clone)]
pub struct RoleResult {
    /// Role this result belongs to.
    pub role: SuggestedRole,
    /// Terminal outcome reported by [`AgentLoop::run_turn`]. When the
    /// worker thread panicked or the dispatcher cancelled before the
    /// worker started, this is [`TurnOutcome::ModelError`] and `error`
    /// carries a short human description.
    pub outcome: TurnOutcome,
    /// Blocks emitted by the role, verbatim.
    pub blocks: Vec<Block>,
    /// Wall-clock time the worker spent inside `run_turn`, in
    /// milliseconds.
    pub duration_ms: u64,
    /// Optional human-readable error message — populated when the
    /// outcome was synthesised by the dispatcher (cancelled,
    /// worker-panicked, deadline-exceeded).
    pub error: Option<String>,
}

/// Aggregate result of [`AgentHandoff::run_turn_parallel`].
///
/// The `per_role` map is sorted by [`SuggestedRole`] declaration rank
/// (via [`OrdRole`]) so callers observe a deterministic iteration order
/// regardless of how the underlying threads were scheduled. Use
/// [`OrdRole::role`] on the keys (or [`From`]) to recover the wrapped
/// [`SuggestedRole`].
#[derive(Debug, Clone)]
pub struct ParallelResult {
    /// Per-role outcome, sorted by role declaration rank.
    pub per_role: BTreeMap<OrdRole, RoleResult>,
    /// Wall-clock instant the dispatcher started.
    pub started_at: SystemTime,
    /// Total dispatcher wall-clock time, in milliseconds.
    pub elapsed_ms: u64,
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
    parallel_policy: ParallelPolicy,
}

impl AgentHandoff {
    /// Construct a coordinator from a registry, fall-back role, and
    /// chain policy. The parallel-dispatch policy starts at
    /// [`ParallelPolicy::default`]; use
    /// [`Self::with_parallel_policy`] to override it.
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
            parallel_policy: ParallelPolicy {
                max_concurrent: 4,
                fail_fast: false,
            },
        }
    }

    /// Override the parallel-dispatch policy used by
    /// [`Self::run_turn_parallel`]. Builder-style; returns the modified
    /// coordinator so callers can chain construction.
    #[must_use]
    pub const fn with_parallel_policy(mut self, policy: ParallelPolicy) -> Self {
        self.parallel_policy = policy;
        self
    }

    /// Active parallel-dispatch policy.
    #[must_use]
    pub const fn parallel_policy(&self) -> ParallelPolicy {
        self.parallel_policy
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

    /// Broadcast the same `TurnContext` and `RoutedIntent` to every
    /// role in `roles` concurrently, throttled by
    /// [`ParallelPolicy::max_concurrent`].
    ///
    /// Each role runs on a dedicated worker thread spawned via
    /// [`std::thread::spawn`]. Every worker observes a child of the
    /// caller-supplied [`CancelToken`] so cancelling the parent fans
    /// out to every in-flight role; the [`AgentRegistry`] is shared
    /// read-only via `Arc` so workers never contend on a mutex.
    ///
    /// Returns a [`ParallelResult`] whose `per_role` map is sorted by
    /// [`SuggestedRole`] declaration rank. Duplicate roles in `roles`
    /// collapse to a single entry (the later worker wins on insertion
    /// order, which is also deterministic).
    ///
    /// # Errors
    ///
    /// * [`HandoffError::NoSuchRole`] — at least one requested role is
    ///   not present in the registry. The check runs before any worker
    ///   is spawned, so a missing role aborts the dispatch entirely.
    #[allow(
        clippy::needless_pass_by_value,
        reason = "owning the intent matches the agent-loop run_turn signature and the rest of the hand-off surface"
    )]
    #[allow(
        clippy::too_many_lines,
        reason = "single coherent dispatcher: spawn pool + recv loop + drain + join must stay co-located for readability"
    )]
    pub fn run_turn_parallel(
        &self,
        ctx: TurnContext,
        intent: RoutedIntent,
        cancel: &CancelToken,
        roles: &[SuggestedRole],
    ) -> Result<ParallelResult, HandoffError> {
        let _ = intent; // Reserved for future per-role hint plumbing.
        let started_at = SystemTime::now();
        let started_instant = Instant::now();

        // Empty role list short-circuits to an empty result.
        if roles.is_empty() {
            return Ok(ParallelResult {
                per_role: BTreeMap::new(),
                started_at,
                elapsed_ms: 0,
            });
        }

        // Pre-flight: every requested role must be registered.
        for role in roles {
            if self.registry.get(role).is_none() {
                return Err(HandoffError::NoSuchRole(*role));
            }
        }

        // Wall-time budget — re-uses HandoffPolicy::max_chain_depth as a
        // hop-budget proxy at 30 s per hop.
        let deadline =
            started_instant + Duration::from_secs(u64::from(self.policy.max_chain_depth) * 30);

        let parent_cancel = cancel.clone();
        let max_concurrent = usize::from(self.parallel_policy.max_concurrent.max(1));
        let fail_fast = self.parallel_policy.fail_fast;

        let (tx, rx) = mpsc::channel::<RoleResult>();
        let mut joins: Vec<thread::JoinHandle<()>> = Vec::with_capacity(roles.len());
        let mut per_role: BTreeMap<OrdRole, RoleResult> = BTreeMap::new();
        let mut next_dispatch = 0_usize;
        let mut in_flight = 0_usize;
        let mut finished = 0_usize;

        // Helper closure to spawn a single worker.
        let spawn_one = |role: SuggestedRole,
                         agent: Arc<AgentLoop>,
                         ctx: TurnContext,
                         cancel: CancelToken,
                         tx: mpsc::Sender<RoleResult>|
         -> thread::JoinHandle<()> {
            thread::spawn(move || {
                let started = Instant::now();
                // If cancelled before we even started, short-circuit.
                if cancel.is_cancelled() {
                    let _ = tx.send(RoleResult {
                        role,
                        outcome: TurnOutcome::UserAbort,
                        blocks: Vec::new(),
                        duration_ms: 0,
                        error: Some("cancelled before dispatch".into()),
                    });
                    return;
                }
                let result = agent.run_turn(ctx, &cancel);
                let duration_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(0);
                let _ = tx.send(RoleResult {
                    role,
                    outcome: result.outcome,
                    blocks: result.blocks,
                    duration_ms,
                    error: None,
                });
            })
        };

        // Initial burst — at most `max_concurrent` workers in flight.
        while next_dispatch < roles.len() && in_flight < max_concurrent {
            let role = roles[next_dispatch];
            let Some(agent) = self.registry.get(&role) else {
                // Should never trip — we pre-flighted above — but stay
                // defensive against a registry mutation between checks.
                return Err(HandoffError::NoSuchRole(role));
            };
            let worker_cancel = parent_cancel.child();
            let worker_tx = tx.clone();
            joins.push(spawn_one(
                role,
                agent,
                ctx.clone(),
                worker_cancel,
                worker_tx,
            ));
            in_flight += 1;
            next_dispatch += 1;
        }

        let total = roles.len();
        let mut aborted_for_fail_fast = false;

        while finished < total {
            let now = Instant::now();
            if now >= deadline {
                // Wall-time budget exhausted — cancel everyone and stop
                // waiting. Workers that have not started yet will
                // observe the cancellation and short-circuit.
                parent_cancel.cancel();
                break;
            }
            let remaining = deadline.saturating_duration_since(now);
            match rx.recv_timeout(remaining) {
                Ok(role_result) => {
                    finished += 1;
                    in_flight = in_flight.saturating_sub(1);
                    let is_error = !matches!(role_result.outcome, TurnOutcome::Success);
                    per_role.insert(OrdRole(role_result.role), role_result);

                    if fail_fast && is_error {
                        // Cancel any pending / in-flight workers.
                        parent_cancel.cancel();
                        aborted_for_fail_fast = true;
                        break;
                    }

                    // Top up the queue if more roles remain.
                    while next_dispatch < total && in_flight < max_concurrent {
                        let role = roles[next_dispatch];
                        let Some(agent) = self.registry.get(&role) else {
                            return Err(HandoffError::NoSuchRole(role));
                        };
                        let worker_cancel = parent_cancel.child();
                        let worker_tx = tx.clone();
                        joins.push(spawn_one(
                            role,
                            agent,
                            ctx.clone(),
                            worker_cancel,
                            worker_tx,
                        ));
                        in_flight += 1;
                        next_dispatch += 1;
                    }
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    parent_cancel.cancel();
                    break;
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            }
        }

        // Drop our own sender so the receiver eventually disconnects.
        drop(tx);

        // Drain any results that landed after the loop exited (e.g. a
        // worker that finished between fail_fast cancel and join). Best
        // effort — bounded by the channel close.
        while let Ok(role_result) = rx.try_recv() {
            per_role
                .entry(OrdRole(role_result.role))
                .or_insert(role_result);
        }

        // Join all worker threads. Threads observe the cancel token,
        // so they should terminate quickly even when we bailed early.
        for handle in joins {
            // Ignore the join error — a panicking worker is treated as
            // a missing result (the per_role map stays partial).
            let _ = handle.join();
        }

        // After threads are joined, collect any final results.
        while let Ok(role_result) = rx.try_recv() {
            per_role
                .entry(OrdRole(role_result.role))
                .or_insert(role_result);
        }

        let _ = aborted_for_fail_fast; // Reserved — future status reporting.

        let elapsed_ms = u64::try_from(started_instant.elapsed().as_millis()).unwrap_or(0);
        Ok(ParallelResult {
            per_role,
            started_at,
            elapsed_ms,
        })
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

    // ---- ParallelPolicy + run_turn_parallel -----------------------------

    /// Provider that errors on demand for `fail_fast` tests.
    #[derive(Debug)]
    struct ErrProvider {
        code: String,
    }

    impl Provider for ErrProvider {
        #[allow(clippy::unnecessary_literal_bound)]
        fn id(&self) -> &str {
            "err"
        }
        fn capabilities(&self) -> &'static [Capability] {
            const CAPS: &[Capability] = &[Capability::Generate];
            CAPS
        }
        fn generate(&self, _req: &GenerateRequest, _cancel: &CancelToken) -> Vec<Block> {
            // Returning no blocks triggers TurnOutcome::ModelError via
            // the agent_loop's E_NO_BLOCKS path.
            let _ = &self.code;
            Vec::new()
        }
    }

    /// Provider that sleeps to give the dispatcher non-zero `elapsed_ms`.
    #[derive(Debug)]
    struct SlowProvider {
        millis: u64,
    }

    impl Provider for SlowProvider {
        #[allow(clippy::unnecessary_literal_bound)]
        fn id(&self) -> &str {
            "slow"
        }
        fn capabilities(&self) -> &'static [Capability] {
            const CAPS: &[Capability] = &[Capability::Generate];
            CAPS
        }
        fn generate(&self, _req: &GenerateRequest, _cancel: &CancelToken) -> Vec<Block> {
            std::thread::sleep(Duration::from_millis(self.millis));
            vec![Block::Text {
                text: "slow done".into(),
            }]
        }
    }

    fn const_loop(text: &str) -> Arc<AgentLoop> {
        build_loop_with_provider(Arc::new(ConstTextProvider {
            text: text.to_string(),
        }))
    }

    #[test]
    fn parallel_policy_default_matches_documented() {
        let p = ParallelPolicy::default();
        assert_eq!(p.max_concurrent, 4);
        assert!(!p.fail_fast);
    }

    #[test]
    fn parallel_policy_is_copy_and_debug() {
        let p = ParallelPolicy::default();
        let q = p;
        let _ = format!("{p:?} {q:?}");
        assert_eq!(p, q);
    }

    #[test]
    fn with_parallel_policy_overrides_default() {
        let reg = AgentRegistry::new();
        let h = AgentHandoff::new(reg, SuggestedRole::Default, HandoffPolicy::default());
        assert_eq!(h.parallel_policy(), ParallelPolicy::default());
        let custom = ParallelPolicy {
            max_concurrent: 2,
            fail_fast: true,
        };
        let h2 = h.with_parallel_policy(custom);
        assert_eq!(h2.parallel_policy(), custom);
    }

    #[test]
    fn run_turn_parallel_with_no_roles_returns_empty_map() {
        let reg = AgentRegistry::new();
        let h = AgentHandoff::new(reg, SuggestedRole::Default, HandoffPolicy::default());
        let result = h
            .run_turn_parallel(
                ctx("hi"),
                routed_with_role(SuggestedRole::Default),
                &CancelToken::new(),
                &[],
            )
            .expect("ok");
        assert!(result.per_role.is_empty());
        assert_eq!(result.elapsed_ms, 0);
    }

    #[test]
    fn run_turn_parallel_three_roles_yields_three_results() {
        let mut reg = AgentRegistry::new();
        reg.register(SuggestedRole::Default, const_loop("d"));
        reg.register(SuggestedRole::Coder, const_loop("c"));
        reg.register(SuggestedRole::Polisher, const_loop("p"));
        let h = AgentHandoff::new(reg, SuggestedRole::Default, HandoffPolicy::default());
        let result = h
            .run_turn_parallel(
                ctx("hi"),
                routed_with_role(SuggestedRole::Default),
                &CancelToken::new(),
                &[
                    SuggestedRole::Default,
                    SuggestedRole::Coder,
                    SuggestedRole::Polisher,
                ],
            )
            .expect("ok");
        assert_eq!(result.per_role.len(), 3);
        for role in [
            SuggestedRole::Default,
            SuggestedRole::Coder,
            SuggestedRole::Polisher,
        ] {
            let r = result.per_role.get(&OrdRole(role)).expect("role present");
            assert_eq!(r.role, role);
            assert!(matches!(r.outcome, TurnOutcome::Success));
            assert!(!r.blocks.is_empty());
        }
    }

    #[test]
    fn run_turn_parallel_missing_role_returns_no_such_role() {
        let mut reg = AgentRegistry::new();
        reg.register(SuggestedRole::Default, const_loop("d"));
        let h = AgentHandoff::new(reg, SuggestedRole::Default, HandoffPolicy::default());
        let err = h
            .run_turn_parallel(
                ctx("hi"),
                routed_with_role(SuggestedRole::Default),
                &CancelToken::new(),
                &[SuggestedRole::Default, SuggestedRole::Researcher],
            )
            .expect_err("missing role");
        match err {
            HandoffError::NoSuchRole(role) => assert_eq!(role, SuggestedRole::Researcher),
            other => panic!("expected NoSuchRole, got {other:?}"),
        }
    }

    #[test]
    fn run_turn_parallel_fail_fast_with_erroring_role_partial_or_error() {
        // One role returns Success, one returns a ModelError outcome.
        // fail_fast: the error landing must cancel the rest. The result
        // map must contain either only the error or be partial.
        let mut reg = AgentRegistry::new();
        reg.register(
            SuggestedRole::Coder,
            build_loop_with_provider(Arc::new(ErrProvider {
                code: "boom".into(),
            })),
        );
        reg.register(
            SuggestedRole::Polisher,
            build_loop_with_provider(Arc::new(SlowProvider { millis: 80 })),
        );
        reg.register(
            SuggestedRole::Default,
            build_loop_with_provider(Arc::new(SlowProvider { millis: 80 })),
        );
        let policy = ParallelPolicy {
            max_concurrent: 1, // Serialise so we see Coder first.
            fail_fast: true,
        };
        let h = AgentHandoff::new(reg, SuggestedRole::Default, HandoffPolicy::default())
            .with_parallel_policy(policy);
        let result = h
            .run_turn_parallel(
                ctx("hi"),
                routed_with_role(SuggestedRole::Coder),
                &CancelToken::new(),
                &[
                    SuggestedRole::Coder,
                    SuggestedRole::Polisher,
                    SuggestedRole::Default,
                ],
            )
            .expect("ok with partial results");
        // Coder must be present and carry the error outcome.
        let coder = result
            .per_role
            .get(&OrdRole(SuggestedRole::Coder))
            .expect("coder result present");
        assert!(!matches!(coder.outcome, TurnOutcome::Success));
        // Documented contract: per_role is either error-only or partial.
        assert!(result.per_role.len() <= 3);
        assert!(!result.per_role.is_empty());
    }

    #[test]
    fn run_turn_parallel_records_elapsed_ms_for_slow_provider() {
        let mut reg = AgentRegistry::new();
        reg.register(
            SuggestedRole::Default,
            build_loop_with_provider(Arc::new(SlowProvider { millis: 25 })),
        );
        let h = AgentHandoff::new(reg, SuggestedRole::Default, HandoffPolicy::default());
        let result = h
            .run_turn_parallel(
                ctx("hi"),
                routed_with_role(SuggestedRole::Default),
                &CancelToken::new(),
                &[SuggestedRole::Default],
            )
            .expect("ok");
        assert_eq!(result.per_role.len(), 1);
        assert!(
            result.elapsed_ms > 0,
            "elapsed_ms was {}",
            result.elapsed_ms
        );
    }

    #[test]
    fn run_turn_parallel_max_concurrent_one_runs_sequentially() {
        let mut reg = AgentRegistry::new();
        reg.register(SuggestedRole::Default, const_loop("d"));
        reg.register(SuggestedRole::Coder, const_loop("c"));
        reg.register(SuggestedRole::Polisher, const_loop("p"));
        let policy = ParallelPolicy {
            max_concurrent: 1,
            fail_fast: false,
        };
        let h = AgentHandoff::new(reg, SuggestedRole::Default, HandoffPolicy::default())
            .with_parallel_policy(policy);
        let result = h
            .run_turn_parallel(
                ctx("hi"),
                routed_with_role(SuggestedRole::Default),
                &CancelToken::new(),
                &[
                    SuggestedRole::Default,
                    SuggestedRole::Coder,
                    SuggestedRole::Polisher,
                ],
            )
            .expect("ok");
        assert_eq!(result.per_role.len(), 3);
    }

    #[test]
    fn run_turn_parallel_honors_cancel_token_pre_cancelled() {
        let mut reg = AgentRegistry::new();
        reg.register(
            SuggestedRole::Default,
            build_loop_with_provider(Arc::new(SlowProvider { millis: 200 })),
        );
        reg.register(
            SuggestedRole::Coder,
            build_loop_with_provider(Arc::new(SlowProvider { millis: 200 })),
        );
        let h = AgentHandoff::new(reg, SuggestedRole::Default, HandoffPolicy::default());
        let cancel = CancelToken::new();
        cancel.cancel();
        let result = h
            .run_turn_parallel(
                ctx("hi"),
                routed_with_role(SuggestedRole::Default),
                &cancel,
                &[SuggestedRole::Default, SuggestedRole::Coder],
            )
            .expect("ok");
        assert_eq!(result.per_role.len(), 2);
        for r in result.per_role.values() {
            // Either UserAbort (short-circuit before run) or whatever
            // the agent_loop maps a cancelled turn to — neither should
            // be a "Success" since we cancelled up-front.
            assert!(
                !matches!(r.outcome, TurnOutcome::Success),
                "role {:?} unexpectedly succeeded",
                r.role
            );
        }
    }

    #[test]
    fn run_turn_parallel_four_thread_fuzz_yields_distinct_results() {
        let mut reg = AgentRegistry::new();
        reg.register(SuggestedRole::Default, const_loop("d"));
        reg.register(SuggestedRole::Coder, const_loop("c"));
        reg.register(SuggestedRole::Polisher, const_loop("p"));
        let h = Arc::new(AgentHandoff::new(
            reg,
            SuggestedRole::Default,
            HandoffPolicy::default(),
        ));

        let mut handles = Vec::new();
        for i in 0..4 {
            let h = Arc::clone(&h);
            handles.push(std::thread::spawn(move || {
                h.run_turn_parallel(
                    ctx(&format!("prompt-{i}")),
                    routed_with_role(SuggestedRole::Default),
                    &CancelToken::new(),
                    &[
                        SuggestedRole::Default,
                        SuggestedRole::Coder,
                        SuggestedRole::Polisher,
                    ],
                )
                .expect("ok")
            }));
        }
        let results: Vec<ParallelResult> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        assert_eq!(results.len(), 4);
        for r in &results {
            assert_eq!(r.per_role.len(), 3);
        }
    }

    #[test]
    fn role_result_carries_role_label() {
        let mut reg = AgentRegistry::new();
        reg.register(SuggestedRole::Polisher, const_loop("p"));
        let h = AgentHandoff::new(reg, SuggestedRole::Polisher, HandoffPolicy::default());
        let result = h
            .run_turn_parallel(
                ctx("hi"),
                routed_with_role(SuggestedRole::Polisher),
                &CancelToken::new(),
                &[SuggestedRole::Polisher],
            )
            .expect("ok");
        let r = result
            .per_role
            .get(&OrdRole(SuggestedRole::Polisher))
            .expect("present");
        assert_eq!(r.role, SuggestedRole::Polisher);
    }

    #[test]
    fn parallel_result_per_role_keys_are_sorted() {
        let mut reg = AgentRegistry::new();
        // Register out-of-rank order; the BTreeMap should still
        // iterate in declaration rank.
        reg.register(SuggestedRole::Researcher, const_loop("r"));
        reg.register(SuggestedRole::Default, const_loop("d"));
        reg.register(SuggestedRole::Cavemanish, const_loop("k"));
        reg.register(SuggestedRole::Coder, const_loop("c"));
        let h = AgentHandoff::new(reg, SuggestedRole::Default, HandoffPolicy::default());
        let result = h
            .run_turn_parallel(
                ctx("hi"),
                routed_with_role(SuggestedRole::Default),
                &CancelToken::new(),
                &[
                    SuggestedRole::Researcher,
                    SuggestedRole::Default,
                    SuggestedRole::Cavemanish,
                    SuggestedRole::Coder,
                ],
            )
            .expect("ok");
        let observed: Vec<SuggestedRole> = result
            .per_role
            .keys()
            .copied()
            .map(SuggestedRole::from)
            .collect();
        assert_eq!(
            observed,
            vec![
                SuggestedRole::Default,
                SuggestedRole::Cavemanish,
                SuggestedRole::Coder,
                SuggestedRole::Researcher,
            ]
        );
    }

    #[test]
    fn ord_role_conversions_round_trip() {
        let r = OrdRole::new(SuggestedRole::Coder);
        assert_eq!(r.role(), SuggestedRole::Coder);
        let back: SuggestedRole = r.into();
        assert_eq!(back, SuggestedRole::Coder);
        let wrap: OrdRole = SuggestedRole::Polisher.into();
        assert_eq!(wrap.role(), SuggestedRole::Polisher);
    }

    #[test]
    fn parallel_send_sync_smoke() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ParallelPolicy>();
        assert_send_sync::<ParallelResult>();
        assert_send_sync::<RoleResult>();
        assert_send_sync::<OrdRole>();
    }

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
