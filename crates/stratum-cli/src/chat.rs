//! Minimal `stratum chat` TUI built on ratatui + crossterm.
//!
//! Phase 1 surface: chat pane + status bar + input line, backed by the
//! deterministic [`EchoProvider`]. Real LLM inference lands when the
//! `LlamaCppProvider` work concludes.
//!
//! The TUI is split into a pure [`ChatState`] (deterministic, tested with
//! `ratatui::backend::TestBackend`) and a thin event loop (`run`).

// This module is private to the binary crate; clippy's `unreachable_pub`
// lint would fire on every public item because none of them cross the
// crate boundary. The visibility is intentional for readability.
#![allow(
    unreachable_pub,
    reason = "private module by design; pub kept for readability"
)]
#![allow(
    clippy::redundant_pub_crate,
    reason = "internal API kept pub for documentation; module itself is private"
)]
#![allow(
    dead_code,
    reason = "EventEmitter wiring is exposed for the upcoming JSONL CLI path and TUI events panel; kept pub even though the bin build does not yet consume it"
)]

use std::io;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use ratatui::backend::{Backend, CrosstermBackend};
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block as TuiBlock, Borders, Paragraph, Wrap};
use ratatui::Terminal;
use stratum_runtime::{
    format_tokens_per_second, AgentHandoff, AgentLoop, AgentLoopConfig, CancelToken,
    CapabilityMatrix, EchoProvider, Event as RtEvent, EventEmitter, EventRecord, IntentRouter,
    MemoryEventSink, Paths, PendingPrompt, PermissionDecision, PermissionRequest, PermissionStore,
    PlanMode, PromptId, PromptIdGen, Provider, RoleTimer, SuggestedRole, Tier, Transcript,
    TranscriptTurn, TurnContext, TurnId, TurnMetrics, TurnOutcome, TurnRecorder, TurnResult,
};
use stratum_types::{Block, ModelId, StratumResult};

use crate::palette::{self, Palette};

// Permission-prompt responder lives as a sibling file; declare it inline so
// the binary crate root (`main.rs`) does not have to mention it.
#[path = "permission_prompter.rs"]
mod permission_prompter;

pub use permission_prompter::TuiPromptResponder;

/// Multi-line `/help` body listing every wired palette command.
const HELP_TEXT: &str = "available commands:\n\
    /plan [on|off] — toggle (or set) plan mode\n\
    /cancel — cancel the in-flight turn\n\
    /clear — clear the transcript\n\
    /agents — list registered roles (multi-agent mode only)\n\
    /parallel <role1,role2,…> — fan the next turn out across the listed roles \
(multi-agent mode only)\n\
    /budget — show the latest turn metrics (tokens · ms · tok/s · turn id)\n\
    /help — show this message\n\
    /quit, /exit — exit the TUI";

/// One entry in the chat transcript.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Turn {
    /// What the user typed.
    User(String),
    /// What the provider returned.
    Assistant(Vec<Block>),
    /// Cancellation marker (Ctrl-C mid-stream).
    Cancelled,
    /// Slash command was invoked from the palette and dispatched.
    Command {
        /// The command keyword (without the leading `/`) as the user typed it.
        text: String,
        /// `true` for an acknowledged dispatch, `false` for a rejection
        /// (e.g. unknown command).
        ok: bool,
        /// Human-friendly message rendered in the transcript.
        message: String,
    },
}

/// Outcome of dispatching a palette command via
/// [`ChatState::execute_palette_command`]. Each variant carries the
/// message that the transcript renders.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PaletteOutcome {
    /// The command was recognised and executed.
    Acknowledged {
        /// Human-friendly message describing the effect.
        message: String,
    },
    /// The command was unknown or otherwise rejected.
    Rejected {
        /// Human-friendly explanation.
        message: String,
    },
}

/// Pure TUI state. Driven by events; rendered into any [`Backend`].
#[derive(Debug)]
pub struct ChatState {
    transcript: Vec<Turn>,
    input: String,
    provider: EchoProvider,
    cancel: CancelToken,
    tier: Tier,
    quit: bool,
    status: String,
    palette: Option<Palette>,
    /// Monotonic turn counter; the next submitted turn gets this id.
    next_turn_id: u64,
    /// Metrics from the most recently completed turn (renders in status bar).
    last_metrics: Option<TurnMetrics>,
    /// Structured-event emitter wired into each completed turn.
    ///
    /// Defaults to an in-process [`MemoryEventSink`]; the CLI binary may
    /// inject a JSONL-backed sink via [`ChatState::with_events`].
    pub events: Arc<EventEmitter>,
    /// In-memory sink handle, when the emitter is backed by [`MemoryEventSink`].
    ///
    /// `None` after [`ChatState::with_events`] is called with a non-memory
    /// emitter — the emitter's sink is opaque (no `Any` bound on the trait),
    /// so we track the typed handle alongside it for snapshotting.
    memory_sink: Option<Arc<MemoryEventSink>>,
    /// Orchestrator used by [`Self::submit`] to drive a single turn through
    /// the FSM, intent router, permission store, plan-mode fence, and
    /// provider — replacing the direct `provider.generate` call.
    agent_loop: Arc<AgentLoop>,
    /// Most recent [`TurnResult`] returned by [`AgentLoop::run_turn`].
    last_turn_result: Option<TurnResult>,
    /// Plan-mode handle wired into both this state (for `/plan` palette
    /// toggling) and the [`AgentLoop`] (for capability gating). Sharing
    /// the same `Arc` is what makes `/plan` semantically meaningful: a
    /// toggle here immediately reflects in the loop's fence.
    plan_mode: Arc<PlanMode>,
    /// Number of [`Turn`]s prepended from a resumed [`Transcript`].
    ///
    /// Set by [`Self::with_resumed_transcript`]; defaults to zero. Lets
    /// callers render a "Resumed N turns" banner without re-walking the
    /// transcript to count.
    resumed_count: usize,
    /// Shared TUI permission prompter. Cloned into the [`AgentLoop`] as the
    /// responder and read by the event loop to surface pending requests.
    permission_prompter: Arc<TuiPromptResponder>,
    /// Wall-clock instant at which the current in-flight turn started.
    ///
    /// `Some` while [`Self::submit`] is running; cleared back to `None` on
    /// completion. Used by [`Self::status_bar_text`] to render the live
    /// `[generating… <N>s]` indicator.
    in_flight_since: Option<Instant>,
    /// Approximate completion-token count from the most recent turn.
    ///
    /// Computed as `sum(Block::Text(text).len()) / 4` across the assistant
    /// blocks: a coarse 4-chars-per-token heuristic that matches the ballpark
    /// for English-prose tokenizers (GPT/Llama byte-pair encodings average
    /// ~3.5–4.5 chars per token on natural-language text). Cheap, allocation-
    /// free, and good enough for a status-bar gauge — the precise token count
    /// from the provider's `Block::Usage` still flows through `TurnRecorder`.
    last_token_count: u64,
    /// Optional multi-role coordinator. When `Some`, [`Self::submit`] routes
    /// each turn through [`AgentHandoff::run_turn_with_handoff`] instead of
    /// the single-loop default path. `None` preserves the Phase 1 single-loop
    /// behaviour.
    handoff: Option<Arc<AgentHandoff>>,
    /// Role currently driving the chat, when multi-agent mode is active.
    ///
    /// `None` in single-loop mode (no handoff installed). `Some(role)` after
    /// [`Self::with_handoff`] — seeded with [`SuggestedRole::Default`] and
    /// updated after every successful [`AgentHandoff::run_turn_with_handoff`]
    /// to the chain's final role.
    current_role: Option<SuggestedRole>,
}

impl Default for ChatState {
    fn default() -> Self {
        Self::new(EchoProvider::default(), Tier::High, String::new())
    }
}

impl ChatState {
    /// Build a fresh state with the given header (status bar) and tier.
    #[must_use]
    pub fn new(provider: EchoProvider, tier: Tier, status: String) -> Self {
        let sink = Arc::new(MemoryEventSink::new());
        let events = Arc::new(EventEmitter::new(sink.clone()));
        let plan_mode = Arc::new(PlanMode::new());
        let permission_prompter = Arc::new(TuiPromptResponder::default());
        #[allow(
            clippy::expect_used,
            reason = "default_agent_loop sets all nine required builder fields; build() cannot return MissingField on this code path"
        )]
        let agent_loop = Arc::new(
            default_agent_loop(
                provider.clone(),
                events.clone(),
                plan_mode.clone(),
                permission_prompter.clone(),
            )
            .expect("default AgentLoop builder sets every required field"),
        );
        Self {
            transcript: Vec::new(),
            input: String::new(),
            provider,
            cancel: CancelToken::new(),
            tier,
            quit: false,
            status,
            palette: None,
            next_turn_id: 0,
            last_metrics: None,
            events,
            memory_sink: Some(sink),
            agent_loop,
            last_turn_result: None,
            plan_mode,
            resumed_count: 0,
            permission_prompter,
            in_flight_since: None,
            last_token_count: 0,
            handoff: None,
            current_role: None,
        }
    }

    /// Build a state wrapping the supplied [`AgentLoop`]. Test-friendly
    /// builder: lets callers inject providers, responders, or capability
    /// matrices that the default constructor does not expose.
    ///
    /// The supplied loop owns its own emitter; the state still defaults to
    /// a memory-backed [`EventEmitter`] for status-bar / palette wiring,
    /// but [`Self::submit`] will emit through the loop's emitter.
    #[must_use]
    pub fn with_agent_loop(loop_: Arc<AgentLoop>) -> Self {
        let mut state = Self::new(EchoProvider::new("echo: "), Tier::High, String::new());
        state.agent_loop = loop_;
        state
    }

    /// Route every subsequent [`Self::submit`] turn through the supplied
    /// [`AgentHandoff`] instead of the default single-loop path.
    ///
    /// Calling this is what turns `stratum chat --agents-dir <path>` into a
    /// multi-role surface: `submit` builds a [`stratum_runtime::RoutedIntent`]
    /// via [`IntentRouter::default`] and hands the turn (plus context) to
    /// [`AgentHandoff::run_turn_with_handoff`]. The default-loop path remains
    /// available — it just is not used while `handoff` is `Some`.
    #[must_use]
    pub fn with_handoff(mut self, h: Arc<AgentHandoff>) -> Self {
        self.handoff = Some(h);
        // Seed the visible role so the status bar shows `agent: default`
        // immediately on first paint — before any turn has run. Subsequent
        // submits update this to the chain's final role.
        self.current_role = Some(SuggestedRole::Default);
        self
    }

    /// Whether [`Self::submit`] currently routes through an [`AgentHandoff`].
    /// Test-friendly accessor; production callers do not branch on this.
    #[must_use]
    pub const fn has_handoff(&self) -> bool {
        self.handoff.is_some()
    }

    /// Status-bar label describing the active role.
    ///
    /// * Single-loop mode (no [`Self::with_handoff`]): the empty string.
    /// * Multi-agent mode: `"agent: <snake_case_name>"` for the role that
    ///   produced the most recent hop (or [`SuggestedRole::Default`] if no
    ///   submit has run yet).
    ///
    /// The role name is rendered through [`role_name`] so the formatted
    /// value matches the `serde(rename_all = "snake_case")` projection used
    /// throughout the runtime.
    #[must_use]
    pub fn current_role_label(&self) -> String {
        self.current_role
            .map_or_else(String::new, |role| format!("agent: {}", role_name(role)))
    }

    /// Replace the structured-event emitter (e.g. with a JSONL-backed one).
    ///
    /// The default [`MemoryEventSink`] handle is dropped; calls to
    /// [`Self::events_snapshot`] will return `None` afterwards. The
    /// underlying [`AgentLoop`] is rebuilt against the new emitter so
    /// turn-level events also land in the swapped sink.
    #[must_use]
    pub fn with_events(mut self, events: Arc<EventEmitter>) -> Self {
        #[allow(
            clippy::expect_used,
            reason = "default_agent_loop sets all nine required builder fields; build() cannot return MissingField on this code path"
        )]
        let agent_loop = Arc::new(
            default_agent_loop(
                self.provider.clone(),
                events.clone(),
                self.plan_mode.clone(),
                self.permission_prompter.clone(),
            )
            .expect("default AgentLoop builder sets every required field"),
        );
        self.agent_loop = agent_loop;
        self.events = events;
        self.memory_sink = None;
        self
    }

    /// Pop the next pending permission request from the shared TUI prompter,
    /// if any. The event loop polls this each tick; `Some` triggers the
    /// modal overlay path in [`Self::render`].
    ///
    /// This consumes the request from the queue — callers that need a
    /// non-destructive view should use [`Self::peek_pending_permission`].
    #[must_use]
    pub fn pending_permission_request(&self) -> Option<PendingPrompt> {
        self.permission_prompter.pending_request()
    }

    /// Non-destructively look at the next pending permission request.
    /// Used by the modal-render path so the queue is observable across
    /// multiple paint frames.
    #[must_use]
    pub fn peek_pending_permission(&self) -> Option<PendingPrompt> {
        self.permission_prompter.peek_request()
    }

    /// Record the user's answer for `id` on the shared TUI prompter, waking
    /// the worker thread that is blocked inside [`stratum_runtime::PromptResponder::ask`].
    pub fn answer_permission(&self, id: PromptId, decision: PermissionDecision) {
        self.permission_prompter.submit_decision(id, decision);
    }

    /// Pre-populate the scrollback with the turns from a previously persisted
    /// [`Transcript`] so a resumed session shows its prior context.
    ///
    /// Each [`TranscriptTurn`] is mapped to its in-memory [`Turn`] counterpart:
    ///
    /// * [`TranscriptTurn::User`] → [`Turn::User`].
    /// * [`TranscriptTurn::Assistant`] → [`Turn::Assistant`] with each
    ///   [`stratum_runtime::TranscriptBlock`] folded into a [`Block::Text`]
    ///   carrying its rendered text. This keeps the user-visible content while
    ///   sidestepping the impedance mismatch between the persisted block taxonomy
    ///   and the streaming-provider [`Block`] enum.
    /// * [`TranscriptTurn::System`] → [`Turn::Command`] with `text` prefixed by
    ///   `"(system) "`, mirroring how the palette renders informational lines.
    /// * [`TranscriptTurn::Command`] → [`Turn::Command`] preserving `ok`.
    ///
    /// Call this once, before [`Self::submit`] / [`Self::submit_with_prompt`];
    /// existing transcript entries are preserved and the resumed turns are
    /// appended at the current end. [`Self::resumed_count`] is updated by the
    /// number of turns folded in.
    #[must_use]
    pub fn with_resumed_transcript(mut self, t: Transcript) -> Self {
        let mut added = 0_usize;
        for turn in t.turns {
            let mapped = match turn {
                TranscriptTurn::User { text, .. } => Turn::User(text),
                TranscriptTurn::Assistant { blocks, .. } => {
                    let mapped_blocks: Vec<Block> = blocks
                        .into_iter()
                        .map(|b| Block::Text { text: b.text })
                        .collect();
                    Turn::Assistant(mapped_blocks)
                }
                TranscriptTurn::System { text, .. } => Turn::Command {
                    text: format!("(system) {text}"),
                    ok: true,
                    message: String::new(),
                },
                TranscriptTurn::Command { text, ok, .. } => Turn::Command {
                    text,
                    ok,
                    message: String::new(),
                },
            };
            self.transcript.push(mapped);
            added = added.saturating_add(1);
        }
        self.resumed_count = self.resumed_count.saturating_add(added);
        self
    }

    /// Number of turns prepended by [`Self::with_resumed_transcript`].
    ///
    /// Returns zero when no transcript was resumed.
    #[must_use]
    pub const fn resumed_count(&self) -> usize {
        self.resumed_count
    }

    /// Borrow the most recent [`TurnResult`], if any.
    #[must_use]
    pub const fn last_turn_result(&self) -> Option<&TurnResult> {
        self.last_turn_result.as_ref()
    }

    /// Snapshot the in-memory event log if the emitter is backed by a
    /// [`MemoryEventSink`].
    ///
    /// Returns `None` when [`Self::with_events`] swapped in an opaque sink
    /// (the trait does not carry an `Any` bound, so the runtime sink type is
    /// not recoverable from the emitter alone).
    #[must_use]
    pub fn events_snapshot(&self) -> Option<Vec<EventRecord>> {
        self.memory_sink.as_ref().map(|sink| sink.snapshot())
    }

    /// Borrow the last recorded turn metrics, if any. Used by tests.
    #[must_use]
    #[cfg(test)]
    const fn last_metrics(&self) -> Option<&TurnMetrics> {
        self.last_metrics.as_ref()
    }

    /// Is the slash-command palette currently visible?
    #[must_use]
    #[cfg(test)]
    const fn palette_open(&self) -> bool {
        self.palette.is_some()
    }

    /// Has the user asked to quit?
    #[must_use]
    pub const fn should_quit(&self) -> bool {
        self.quit
    }

    /// Borrow the transcript (used in tests).
    #[must_use]
    #[cfg(test)]
    fn transcript(&self) -> &[Turn] {
        &self.transcript
    }

    /// Borrow the current input buffer (used in tests).
    #[must_use]
    #[cfg(test)]
    fn input(&self) -> &str {
        &self.input
    }

    /// Apply a keyboard event.
    pub fn handle_key(&mut self, key: KeyEvent) {
        // Permission modal owns the keyboard while a request is pending.
        if let Some(pending) = self.permission_prompter.peek_request() {
            if let KeyCode::Char(c) = key.code {
                if let Some(decision) = decision_from_key(c) {
                    // Drain the request from the queue and answer it.
                    let _ = self.permission_prompter.pending_request();
                    self.permission_prompter
                        .submit_decision(pending.id, decision);
                }
            }
            // Unknown / non-char keys are swallowed while the modal is open
            // so they don't leak into the input buffer.
            return;
        }
        // Palette mode owns the keyboard while open.
        if let Some(palette) = self.palette.as_mut() {
            match palette.handle_key(key) {
                palette::Action::None => {}
                palette::Action::Close => {
                    self.palette = None;
                }
                palette::Action::Execute(name) => {
                    self.palette = None;
                    self.execute_command(&name);
                }
            }
            return;
        }
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        match key.code {
            KeyCode::Esc => self.quit = true,
            KeyCode::Char('c' | 'C') if ctrl => {
                self.cancel.cancel();
                self.transcript.push(Turn::Cancelled);
                self.quit = true;
            }
            KeyCode::Char('/') if self.input.is_empty() => {
                self.palette = Some(Palette::new());
            }
            KeyCode::Char(c) => self.input.push(c),
            KeyCode::Backspace => {
                self.input.pop();
            }
            KeyCode::Enter => self.submit(),
            _ => {}
        }
    }

    /// Internal palette-flush bridge: the palette emits a bare command
    /// name (no leading `/`); we re-attach the slash and route through
    /// [`Self::execute_palette_command`].
    fn execute_command(&mut self, name: &str) {
        let with_slash = format!("/{name}");
        let _ = self.execute_palette_command(&with_slash);
    }

    /// Dispatch a single palette command. Recognised commands mutate
    /// state (toggle `plan_mode`, fire the `cancel` token, clear the
    /// transcript, set `should_quit`) and push a
    /// [`Turn::Command`] entry to the transcript. Unknown commands are
    /// rejected. The return value also exposes the outcome so callers
    /// can render a status message without re-walking the transcript.
    ///
    /// `cmd` must start with `/` (the palette parser preserves the
    /// slash). An empty string or a command without the leading slash
    /// is rejected with `"unknown command: <cmd>"`.
    ///
    /// `/plan` is a toggle: if plan mode is currently active it is
    /// deactivated, else it is activated. Pass `/plan on` or
    /// `/plan off` to force a specific state.
    pub fn execute_palette_command(&mut self, cmd: &str) -> PaletteOutcome {
        // `/clear` is the only command whose post-state is the empty
        // transcript: it erases history *including* the marker for the
        // clear itself. Detect it up front so we can clear after the
        // push and end up with an empty transcript.
        let is_clear = cmd.trim() == "/clear";
        let outcome = self.dispatch_command(cmd);
        let (ok, message) = match &outcome {
            PaletteOutcome::Acknowledged { message } => (true, message.clone()),
            PaletteOutcome::Rejected { message } => (false, message.clone()),
        };
        self.transcript.push(Turn::Command {
            text: cmd.to_string(),
            ok,
            message,
        });
        if is_clear && ok {
            self.transcript.clear();
        }
        outcome
    }

    fn dispatch_command(&mut self, cmd: &str) -> PaletteOutcome {
        let Some(rest) = cmd.strip_prefix('/') else {
            return PaletteOutcome::Rejected {
                message: format!("unknown command: {cmd}"),
            };
        };
        let trimmed = rest.trim();
        if trimmed.is_empty() {
            return PaletteOutcome::Rejected {
                message: format!("unknown command: {cmd}"),
            };
        }
        let mut parts = trimmed.split_whitespace();
        let head = parts.next().unwrap_or("");
        let arg = parts.next();
        match head {
            "plan" => {
                let message = match arg {
                    Some("on") => {
                        self.plan_mode.activate(SystemTime::now());
                        "plan mode: on".to_string()
                    }
                    Some("off") => {
                        self.plan_mode.deactivate();
                        "plan mode: off".to_string()
                    }
                    None => {
                        if self.plan_mode.is_active() {
                            self.plan_mode.deactivate();
                            "plan mode: off".to_string()
                        } else {
                            self.plan_mode.activate(SystemTime::now());
                            "plan mode: on".to_string()
                        }
                    }
                    Some(other) => {
                        return PaletteOutcome::Rejected {
                            message: format!("unknown command: /plan {other}"),
                        };
                    }
                };
                PaletteOutcome::Acknowledged { message }
            }
            "cancel" => {
                self.cancel.cancel();
                PaletteOutcome::Acknowledged {
                    message: "cancel signal sent".to_string(),
                }
            }
            "clear" => {
                self.transcript.clear();
                PaletteOutcome::Acknowledged {
                    message: "transcript cleared".to_string(),
                }
            }
            "help" => PaletteOutcome::Acknowledged {
                message: HELP_TEXT.to_string(),
            },
            "agents" => self.dispatch_agents(),
            "parallel" => {
                let tail = trimmed.strip_prefix("parallel").unwrap_or("").trim();
                self.dispatch_parallel(tail)
            }
            "budget" => self.dispatch_budget(),
            "quit" | "exit" => {
                self.quit = true;
                PaletteOutcome::Acknowledged {
                    message: "exiting".to_string(),
                }
            }
            _ => PaletteOutcome::Rejected {
                message: format!("unknown command: {cmd}"),
            },
        }
    }

    /// Render the `/agents` palette command output.
    ///
    /// Without an installed [`AgentHandoff`] (single-loop mode), reject with
    /// a pointer to `--agents-dir`. Otherwise enumerate the registered roles
    /// via [`AgentHandoff::roles`] and tag the current driver role.
    fn dispatch_agents(&self) -> PaletteOutcome {
        let Some(handoff) = self.handoff.as_ref() else {
            return PaletteOutcome::Rejected {
                message: "no multi-agent mode; pass --agents-dir to enable".to_string(),
            };
        };
        let roles = handoff.roles();
        let joined = roles
            .iter()
            .copied()
            .map(role_name)
            .collect::<Vec<_>>()
            .join(", ");
        let current = self.current_role.map_or("default", |role| role_name(role));
        PaletteOutcome::Acknowledged {
            message: format!("roles: {joined} (current: {current})"),
        }
    }

    /// Render the `/parallel <role1,role2,…>` palette command output.
    ///
    /// Without an installed [`AgentHandoff`] (single-loop mode), reject with
    /// a pointer to `--agents-dir`. With a handoff: parse the role list
    /// (`snake_case` to match `SuggestedRole`'s serde projection), drive one
    /// turn through [`AgentHandoff::run_turn_parallel`], and append the
    /// per-role results to the transcript as a series of [`Turn::Command`]
    /// summaries plus a [`Turn::Assistant`] block carrying the concatenated
    /// text of every role's output.
    ///
    /// Unknown roles, an empty list, or `NoSuchRole` from the dispatcher
    /// all surface as `PaletteOutcome::Rejected`. Any other dispatcher
    /// error also rejects — the transcript receives no assistant turn.
    fn dispatch_parallel(&mut self, args: &str) -> PaletteOutcome {
        let Some(handoff) = self.handoff.clone() else {
            return PaletteOutcome::Rejected {
                message: "no multi-agent mode; pass --agents-dir to enable".to_string(),
            };
        };
        if args.is_empty() {
            return PaletteOutcome::Rejected {
                message: "unknown role: ".to_string(),
            };
        }
        let mut roles = Vec::new();
        for raw in args.split(',') {
            let label = raw.trim();
            let Some(role) = parse_role_label(label) else {
                return PaletteOutcome::Rejected {
                    message: format!("unknown role: {label}"),
                };
            };
            roles.push(role);
        }

        let turn_id = TurnId(self.next_turn_id);
        self.next_turn_id = self.next_turn_id.saturating_add(1);
        let prompt = self
            .transcript
            .iter()
            .rev()
            .find_map(|t| match t {
                Turn::User(text) => Some(text.clone()),
                _ => None,
            })
            .unwrap_or_default();
        let ctx = TurnContext {
            user_prompt: prompt.clone(),
            model: ModelId::from("echo"),
            turn_id,
            started_at: SystemTime::now(),
        };
        let intent = IntentRouter::default().classify(&prompt);
        let result = match handoff.run_turn_parallel(ctx, intent, &self.cancel, &roles) {
            Ok(r) => r,
            Err(stratum_runtime::HandoffError::NoSuchRole(role)) => {
                return PaletteOutcome::Rejected {
                    message: format!("unknown role: {}", role_name(role)),
                };
            }
            Err(e) => {
                return PaletteOutcome::Rejected {
                    message: format!("parallel dispatch failed: {e}"),
                };
            }
        };

        let mut combined: Vec<Block> = Vec::new();
        let mut summaries: Vec<(String, bool, String)> = Vec::new();
        for (key, role_result) in &result.per_role {
            let role = role_name(key.role());
            let ok = matches!(role_result.outcome, TurnOutcome::Success);
            let text = concat_text_blocks(&role_result.blocks);
            combined.push(Block::Text {
                text: format!("=== {role} ({}ms) ===\n{text}\n", role_result.duration_ms),
            });
            let summary = if ok {
                format!(
                    "[{role}] {}ms ({} chars)",
                    role_result.duration_ms,
                    text.len()
                )
            } else {
                let err = role_result
                    .error
                    .as_deref()
                    .unwrap_or("non-success outcome");
                format!("[{role}] error: {err}")
            };
            summaries.push((format!("/parallel {role}"), ok, summary));
        }

        if !combined.is_empty() {
            self.transcript.push(Turn::Assistant(combined));
        }
        for (text, ok, message) in summaries {
            self.transcript.push(Turn::Command { text, ok, message });
        }

        let all_ok = result
            .per_role
            .values()
            .all(|r| matches!(r.outcome, TurnOutcome::Success));
        PaletteOutcome::Acknowledged {
            message: format!(
                "parallel: {} role(s) in {}ms (all_ok={all_ok})",
                result.per_role.len(),
                result.elapsed_ms,
            ),
        }
    }

    /// Render the `/budget` palette command output.
    ///
    /// Without recorded [`TurnMetrics`] (no submit yet), acknowledges with the
    /// sentinel string `"no turn metrics yet"`. With metrics in hand, formats
    /// the latest turn's completion tokens, summed role-step wall-clock, and
    /// tok/s (via [`format_tokens_per_second`]) alongside the turn id.
    fn dispatch_budget(&self) -> PaletteOutcome {
        let Some(metrics) = self.last_metrics.as_ref() else {
            return PaletteOutcome::Acknowledged {
                message: "no turn metrics yet".to_string(),
            };
        };
        let wall_ms = metrics
            .role_steps
            .iter()
            .map(|step| step.duration_ms)
            .fold(0_u32, u32::saturating_add);
        let tps = format_tokens_per_second(metrics.completion_tokens, wall_ms);
        PaletteOutcome::Acknowledged {
            message: format!(
                "metrics: {} tokens · {wall_ms}ms · {tps:.1} tok/s · turn id {}",
                metrics.completion_tokens, metrics.turn_id.0,
            ),
        }
    }

    /// Stage `prompt` into the input buffer and dispatch [`Self::submit`].
    ///
    /// Helper used by the non-interactive `stratum chat --prompt <STR>` path
    /// (and by tests) — it replaces the current input wholesale so callers
    /// don't have to drive a stream of `KeyCode::Char` events.
    pub fn submit_with_prompt(&mut self, prompt: &str) {
        self.input.clear();
        self.input.push_str(prompt);
        self.submit();
    }

    /// Join the most recent [`Turn::Assistant`] entry's text blocks into a
    /// single string.
    ///
    /// Returns `None` when the transcript contains no assistant turn or the
    /// last assistant turn has no [`Block::Text`] blocks. Useful for the
    /// `--prompt` non-interactive path and integration tests that need to
    /// inspect what the provider produced without re-walking the transcript.
    #[must_use]
    pub fn last_assistant_text(&self) -> Option<String> {
        let blocks = self.transcript.iter().rev().find_map(|t| match t {
            Turn::Assistant(b) => Some(b),
            _ => None,
        })?;
        let texts: Vec<&str> = blocks
            .iter()
            .filter_map(|b| match b {
                Block::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        if texts.is_empty() {
            None
        } else {
            Some(texts.join(""))
        }
    }

    /// Submit the current input through the [`AgentLoop`] and append the
    /// resulting blocks to the transcript.
    ///
    /// When [`Self::with_handoff`] has installed an [`AgentHandoff`], the
    /// turn is routed through [`AgentHandoff::run_turn_with_handoff`]
    /// instead — a multi-role classifier + sentinel-driven chain — and any
    /// extra hops surface as `(handoff: from → to)` command lines in the
    /// transcript so the UI shows the chain that produced the final blocks.
    pub fn submit(&mut self) {
        if self.input.trim().is_empty() {
            return;
        }
        let prompt = std::mem::take(&mut self.input);
        let turn_id = TurnId(self.next_turn_id);
        self.next_turn_id = self.next_turn_id.saturating_add(1);

        // Mark the turn as in-flight so `status_bar_text` renders the live
        // `[generating… <N>s]` indicator. Cleared unconditionally at the end.
        self.in_flight_since = Some(Instant::now());

        let ctx = TurnContext {
            user_prompt: prompt.clone(),
            model: ModelId::from("echo"),
            turn_id,
            started_at: std::time::SystemTime::now(),
        };

        let role_timer = RoleTimer::start();
        let (blocks, last_turn_result, handoff_lines, final_role) =
            if let Some(handoff) = self.handoff.as_ref() {
                self.run_turn_via_handoff(handoff.as_ref(), ctx, &prompt)
            } else {
                let turn_result = self.agent_loop.run_turn(ctx, &self.cancel);
                let blocks = turn_result.blocks.clone();
                (blocks, Some(turn_result), Vec::new(), None)
            };
        let step_ms = role_timer.stop_ms();
        // Multi-agent mode only: track which role drove the chain to its
        // final hop. Single-loop mode leaves `current_role` untouched (None).
        if let Some(role) = final_role {
            self.current_role = Some(role);
        }

        let mut recorder = TurnRecorder::new(turn_id);
        for block in &blocks {
            recorder.record_block(block);
        }
        recorder.record_step("generate", step_ms);

        // Coarse 4-chars-per-token approximation across assistant text blocks.
        // Replaces (not accumulates) `last_token_count` so the status bar
        // reflects only the most recent turn — matching `last_metrics`.
        let tokens_generated = approximate_token_count(&blocks);
        self.last_token_count = tokens_generated;
        self.last_metrics = Some(recorder.finish());

        self.transcript.push(Turn::User(prompt));
        self.transcript.push(Turn::Assistant(blocks));
        for line in handoff_lines {
            self.transcript.push(line);
        }
        self.last_turn_result = last_turn_result;

        // Turn finished: drop the in-flight indicator.
        self.in_flight_since = None;
    }

    /// Drive one turn through the supplied [`AgentHandoff`]. Returns the
    /// final hop's blocks, the matching [`TurnResult`] (for `last_turn_result`),
    /// and zero-or-more `Turn::Command` rows describing each step in the
    /// chain (one per step, only when `result.steps.len() > 1`).
    ///
    /// On [`stratum_runtime::HandoffError`] the transcript receives an empty
    /// assistant block, a single `Turn::Command` with `ok = false`, and the
    /// `last_turn_result` is cleared so the status bar does not surface
    /// stale metrics.
    fn run_turn_via_handoff(
        &self,
        handoff: &AgentHandoff,
        ctx: TurnContext,
        prompt: &str,
    ) -> (
        Vec<Block>,
        Option<TurnResult>,
        Vec<Turn>,
        Option<SuggestedRole>,
    ) {
        let intent = IntentRouter::default().classify(prompt);
        match handoff.run_turn_with_handoff(ctx, intent, &self.cancel) {
            Ok(result) => {
                let extra: Vec<Turn> = if result.steps.len() > 1 {
                    result
                        .steps
                        .iter()
                        .map(|step| Turn::Command {
                            text: format!("(handoff: {:?} → {:?})", step.from_role, step.to_role),
                            ok: true,
                            message: String::new(),
                        })
                        .collect()
                } else {
                    Vec::new()
                };
                let last = result.steps.last().map(|s| s.turn_result.clone());
                let final_role = Some(result.final_role);
                (result.final_blocks, last, extra, final_role)
            }
            Err(e) => {
                let line = Turn::Command {
                    text: "(handoff failed)".to_string(),
                    ok: false,
                    message: e.to_string(),
                };
                (Vec::new(), None, vec![line], None)
            }
        }
    }

    /// Render the status-bar text for the current turn state.
    ///
    /// * While a turn is in flight: `[generating… <N>s]` where `<N>` is the
    ///   wall-clock seconds since [`Self::submit`] started.
    /// * After a completed turn (and no in-flight turn): a token-rate summary
    ///   `"<count> tokens in <ms>ms (<tok/s> tok/s)"` derived from
    ///   `last_token_count`, the most recent `TurnMetrics`, and
    ///   [`format_tokens_per_second`].
    /// * Otherwise (fresh state, no submit yet): the empty string.
    ///
    /// The in-flight branch wins over the completed branch — a fresh submit
    /// after a previous turn renders `[generating…]` even though
    /// `last_metrics` is still `Some`.
    #[must_use]
    pub fn status_bar_text(&self) -> String {
        if let Some(started) = self.in_flight_since {
            let elapsed = started.elapsed().as_secs();
            return format!("[generating… {elapsed}s]");
        }
        if let Some(metrics) = self.last_metrics.as_ref() {
            let role_ms = metrics
                .role_steps
                .iter()
                .map(|step| step.duration_ms)
                .fold(0_u32, u32::saturating_add);
            // `format_tokens_per_second` takes `u32`; saturate the approximate
            // count without panicking on the (impossible) overflow case.
            let count_u32 = u32::try_from(self.last_token_count).unwrap_or(u32::MAX);
            let tps = format_tokens_per_second(count_u32, role_ms);
            return format!(
                "{} tokens in {}ms ({tps:.1} tok/s)",
                self.last_token_count, role_ms,
            );
        }
        String::new()
    }

    /// Borrow the approximate token count from the most recent turn. Used by
    /// tests; production callers prefer the precise `last_metrics` count.
    #[must_use]
    #[cfg(test)]
    const fn last_token_count(&self) -> u64 {
        self.last_token_count
    }

    /// Record a completed turn: emit structured events for the generated
    /// blocks, update metrics, and append the user / assistant transcript
    /// entries.
    ///
    /// Factored out of [`Self::submit`] so tests can drive the event-emission
    /// path with synthetic block streams.
    fn finish_turn(&mut self, prompt: String, blocks: Vec<Block>, provider_id: &str, step_ms: u32) {
        let turn_id = TurnId(self.next_turn_id);
        self.next_turn_id = self.next_turn_id.saturating_add(1);
        let mut recorder = TurnRecorder::new(turn_id);
        for block in &blocks {
            recorder.record_block(block);
        }
        recorder.record_step("generate", step_ms);
        self.last_metrics = Some(recorder.finish());

        // No blocks at all on a non-empty prompt = provider failure.
        if blocks.is_empty() {
            self.events.emit(
                RtEvent::ProviderError {
                    provider: provider_id.to_string(),
                    code: "E_NO_BLOCKS".to_string(),
                    message: "provider returned no blocks".to_string(),
                },
                Some(turn_id.0),
            );
        } else {
            for block in &blocks {
                if let Block::ToolCall { id, .. } = block {
                    self.events.emit(
                        RtEvent::ToolCall {
                            tool_id: id.clone(),
                            ok: true,
                            duration_ms: u64::from(step_ms),
                        },
                        Some(turn_id.0),
                    );
                }
            }
        }

        self.transcript.push(Turn::User(prompt));
        self.transcript.push(Turn::Assistant(blocks));
    }

    /// Render the entire TUI into the given frame.
    #[allow(
        clippy::too_many_lines,
        reason = "render walks every overlay (status, chat, palette, modal); splitting fragments the buffer plumbing for no gain"
    )]
    pub fn render(&self, area: Rect, buf: &mut ratatui::buffer::Buffer) {
        let palette_height = if self.palette.is_some() { 10 } else { 3 };
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1),
                Constraint::Min(3),
                Constraint::Length(palette_height),
            ])
            .split(area);

        let mut status_spans = vec![
            Span::styled("stratum", Style::default().add_modifier(Modifier::BOLD)),
            Span::raw(" · "),
            Span::raw(format!("tier={}", self.tier)),
            Span::raw(" · "),
            Span::raw(&self.status),
        ];
        if let Some(metrics) = self.last_metrics.as_ref() {
            let role_ms = metrics
                .role_steps
                .iter()
                .map(|step| step.duration_ms)
                .fold(0_u32, u32::saturating_add);
            let tps = format_tokens_per_second(metrics.completion_tokens, role_ms);
            status_spans.push(Span::raw(" · "));
            status_spans.push(Span::styled(
                format!(
                    "turn {} · prompt:{} compl:{} · {tps:.1} tok/s",
                    metrics.turn_id.0, metrics.prompt_tokens, metrics.completion_tokens,
                ),
                Style::default().add_modifier(Modifier::DIM),
            ));
        }
        let status_bar = self.status_bar_text();
        if !status_bar.is_empty() {
            status_spans.push(Span::raw(" · "));
            status_spans.push(Span::styled(
                status_bar,
                Style::default().add_modifier(Modifier::DIM),
            ));
        }
        let role_label = self.current_role_label();
        if !role_label.is_empty() {
            status_spans.push(Span::raw(" · "));
            status_spans.push(Span::styled(
                role_label,
                Style::default().add_modifier(Modifier::DIM),
            ));
        }
        status_spans.push(Span::raw(" · "));
        status_spans.push(Span::raw("Esc/Ctrl-C exit"));
        let status = Paragraph::new(Line::from(status_spans));
        ratatui::widgets::Widget::render(status, chunks[0], buf);

        let mut lines: Vec<Line<'_>> = Vec::new();
        for turn in &self.transcript {
            match turn {
                Turn::User(text) => lines.push(Line::from(vec![
                    Span::styled("you: ", Style::default().add_modifier(Modifier::BOLD)),
                    Span::raw(text.clone()),
                ])),
                Turn::Assistant(blocks) => {
                    for block in blocks {
                        if let Some(line) = render_block(block) {
                            lines.push(line);
                        }
                    }
                }
                Turn::Cancelled => lines.push(Line::from(Span::styled(
                    "(cancelled)",
                    Style::default().add_modifier(Modifier::ITALIC),
                ))),
                Turn::Command { text, ok, message } => {
                    let prefix = if *ok { "executed" } else { "rejected" };
                    for line in message.lines() {
                        lines.push(Line::from(Span::styled(
                            format!("({prefix} {text}: {line})"),
                            Style::default().add_modifier(Modifier::DIM),
                        )));
                    }
                }
            }
        }
        let chat = Paragraph::new(lines)
            .block(TuiBlock::default().borders(Borders::ALL).title("chat"))
            .wrap(Wrap { trim: false });
        ratatui::widgets::Widget::render(chat, chunks[1], buf);

        // Permission-request modal renders as an overlay inside the chat
        // pane when a request is pending. We peek without popping so the
        // modal persists across re-renders until a decision is supplied.
        if let Some(pending) = self.peek_pending_permission() {
            let modal_lines: Vec<Line<'_>> = vec![
                Line::from(Span::styled(
                    "permission request",
                    Style::default().add_modifier(Modifier::BOLD),
                )),
                Line::from(Span::raw(describe_request(&pending.request))),
                Line::from(Span::styled(
                    "[a] allow once  [s] allow session  [f] allow forever  [d] deny  [F] deny forever",
                    Style::default().add_modifier(Modifier::DIM),
                )),
            ];
            let modal = Paragraph::new(modal_lines)
                .block(
                    TuiBlock::default()
                        .borders(Borders::ALL)
                        .title("permission"),
                )
                .wrap(Wrap { trim: false });
            // Overlay across the chat pane only (chunks[1]).
            ratatui::widgets::Widget::render(modal, chunks[1], buf);
        }

        if let Some(palette) = self.palette.as_ref() {
            let matches = palette.matches();
            let mut palette_lines: Vec<Line<'_>> = Vec::new();
            palette_lines.push(Line::from(vec![
                Span::styled("palette: ", Style::default().add_modifier(Modifier::BOLD)),
                Span::raw(palette.filter().to_string()),
            ]));
            for (idx, cmd) in matches.iter().enumerate() {
                let style = if idx == palette.cursor() {
                    Style::default().add_modifier(Modifier::REVERSED)
                } else {
                    Style::default()
                };
                palette_lines.push(Line::from(Span::styled(format!("/{}", cmd.name), style)));
            }
            let palette_widget = Paragraph::new(palette_lines)
                .block(TuiBlock::default().borders(Borders::ALL).title("palette"));
            ratatui::widgets::Widget::render(palette_widget, chunks[2], buf);
        } else {
            let input = Paragraph::new(Line::from(vec![Span::raw("> "), Span::raw(&self.input)]))
                .block(TuiBlock::default().borders(Borders::ALL).title("input"));
            ratatui::widgets::Widget::render(input, chunks[2], buf);
        }
    }
}

/// Construct an [`AgentLoop`] wired with the supplied provider and shared
/// emitter, plus the documented default permission store / responder /
/// router / plan-mode / capability matrix.
///
/// All nine required builder fields are populated below, so `build()` is
/// total. We propagate the `Result` rather than panicking; callers fall
/// back to a `None`-loop state if the builder ever grows a new required
/// field that this function forgets to set.
fn default_agent_loop(
    provider: EchoProvider,
    events: Arc<EventEmitter>,
    plan_mode: Arc<PlanMode>,
    prompter: Arc<TuiPromptResponder>,
) -> Result<AgentLoop, stratum_runtime::AgentLoopBuildError> {
    let provider_arc: Arc<dyn Provider> = Arc::new(provider);
    let responder: Arc<dyn stratum_runtime::PromptResponder> = prompter;
    AgentLoop::builder()
        .with_provider(provider_arc)
        .with_router(IntentRouter::default())
        .with_permission_store(Arc::new(PermissionStore::new()))
        .with_prompt_gen(Arc::new(PromptIdGen::new()))
        .with_responder(responder)
        .with_events(events)
        .with_capability_matrix(Arc::new(CapabilityMatrix::new()))
        .with_plan_mode(plan_mode)
        .with_config(AgentLoopConfig::default())
        .build()
}

/// Map a single keystroke to a [`PermissionDecision`] for the modal.
///
/// `a` → `AllowOnce`, `s` → `AllowSession`, `f` → `AllowForever`,
/// `d` → `Deny`, `F` → `DenyForever`. Returns `None` for unknown keys so
/// the caller can re-display the modal.
const fn decision_from_key(c: char) -> Option<PermissionDecision> {
    match c {
        'a' => Some(PermissionDecision::AllowOnce),
        's' => Some(PermissionDecision::AllowSession),
        'f' => Some(PermissionDecision::AllowForever),
        'd' => Some(PermissionDecision::Deny),
        'F' => Some(PermissionDecision::DenyForever),
        _ => None,
    }
}

/// Render `request` into a short human-readable label for the modal.
fn describe_request(req: &PermissionRequest) -> String {
    match req {
        PermissionRequest::CapabilityGrant {
            capability,
            target,
            reason,
        } => target.as_ref().map_or_else(
            || format!("grant {capability} ({reason})"),
            |t| format!("grant {capability} on {t} ({reason})"),
        ),
        PermissionRequest::SecretAccess { secret_ref, scope } => {
            format!("access secret {secret_ref} [{scope}]")
        }
        PermissionRequest::NetworkHost { host, port } => port.as_ref().map_or_else(
            || format!("connect to {host}"),
            |p| format!("connect to {host}:{p}"),
        ),
        PermissionRequest::FileWrite { path } => format!("write to {}", path.display()),
        PermissionRequest::ToolUse { tool_id } => format!("invoke tool {tool_id}"),
    }
}

/// Approximate the completion-token count from a slice of assistant blocks.
///
/// Sums character lengths of every [`Block::Text`] block and divides by 4,
/// matching the 4-chars-per-token rough heuristic documented on
/// [`ChatState::last_token_count`]. Non-text blocks (usage, tool calls, etc.)
/// contribute nothing. Pure / allocation-free.
/// Render a [`SuggestedRole`] as its `snake_case` label.
///
/// Matches the `serde(rename_all = "snake_case")` projection on the enum so
/// status-bar and `/agents` output align with how roles are spelled in
/// agents-dir YAML and `agents list` output.
const fn role_name(role: SuggestedRole) -> &'static str {
    match role {
        SuggestedRole::Default => "default",
        SuggestedRole::Cavemanish => "cavemanish",
        SuggestedRole::Polisher => "polisher",
        SuggestedRole::Coder => "coder",
        SuggestedRole::Researcher => "researcher",
    }
}

/// Resolve a `snake_case` role label to its [`SuggestedRole`] variant.
///
/// Mirrors the `serde(rename_all = "snake_case")` projection on
/// [`SuggestedRole`] so the `/parallel` palette command accepts the same
/// spelling as the on-disk agent TOML's `roles = […]` field.
fn parse_role_label(s: &str) -> Option<SuggestedRole> {
    match s {
        "default" => Some(SuggestedRole::Default),
        "cavemanish" => Some(SuggestedRole::Cavemanish),
        "polisher" => Some(SuggestedRole::Polisher),
        "coder" => Some(SuggestedRole::Coder),
        "researcher" => Some(SuggestedRole::Researcher),
        _ => None,
    }
}

/// Concatenate every [`Block::Text`] payload in `blocks` in order. Used by
/// the `/parallel` palette dispatcher to render each role's output as a
/// single transcript line. Non-text blocks (usage, tool calls, etc.) are
/// skipped.
fn concat_text_blocks(blocks: &[Block]) -> String {
    let mut out = String::new();
    for block in blocks {
        if let Block::Text { text } = block {
            out.push_str(text);
        }
    }
    out
}

fn approximate_token_count(blocks: &[Block]) -> u64 {
    const CHARS_PER_TOKEN: u64 = 4;
    let total_chars: u64 = blocks
        .iter()
        .filter_map(|b| match b {
            Block::Text { text } => Some(text.len() as u64),
            _ => None,
        })
        .sum();
    total_chars / CHARS_PER_TOKEN
}

fn render_block(block: &Block) -> Option<Line<'_>> {
    match block {
        Block::Text { text } => Some(Line::from(vec![
            Span::styled("ai:  ", Style::default().add_modifier(Modifier::BOLD)),
            Span::raw(text.clone()),
        ])),
        Block::Usage { prompt, completion } => Some(Line::from(Span::styled(
            format!("(usage: prompt={prompt} completion={completion})"),
            Style::default().add_modifier(Modifier::DIM),
        ))),
        Block::Cancelled { reason } => Some(Line::from(Span::styled(
            format!("(cancelled: {reason})"),
            Style::default().add_modifier(Modifier::ITALIC),
        ))),
        Block::Done | Block::ToolCall { .. } | Block::ToolResult { .. } => None,
    }
}

/// Build the status string shown in the TUI header.
#[must_use]
pub fn status_for(paths: &Paths) -> String {
    if paths.installed_toml().exists() {
        "installed".to_string()
    } else {
        "not installed; run `stratum init`".to_string()
    }
}

/// Drive the live TUI until the user quits. Returns when the state's
/// `should_quit` is set.
///
/// # Errors
/// Propagates terminal-init failures as [`io::Error`].
pub fn run(paths: &Paths, tier: Tier) -> StratumResult<()> {
    let provider = EchoProvider::new("echo: ");
    let state = ChatState::new(provider, tier, status_for(paths));
    run_with_state(state)
}

/// Drive the live TUI against a caller-supplied [`ChatState`]. Used by the
/// `--model` path so the resolved [`AgentLoop`] (wrapping the real
/// `LlamaCppProvider`) backs the TUI in place of the default [`EchoProvider`].
///
/// # Errors
/// Propagates terminal-init failures as [`io::Error`].
pub fn run_with_state(mut state: ChatState) -> StratumResult<()> {
    let mut stdout = io::stdout();
    enable_raw_mode().map_err(map_io_error)?;
    execute!(stdout, EnterAlternateScreen).map_err(map_io_error)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend).map_err(map_io_error)?;
    let result = event_loop(&mut terminal, &mut state);
    let _ = disable_raw_mode();
    let _ = execute!(io::stdout(), LeaveAlternateScreen);
    result
}

fn event_loop<B: Backend>(terminal: &mut Terminal<B>, state: &mut ChatState) -> StratumResult<()> {
    loop {
        let evt = if event::poll(Duration::from_millis(100)).map_err(map_io_error)? {
            Some(event::read().map_err(map_io_error)?)
        } else {
            None
        };
        step(terminal, state, evt.as_ref())?;
        if state.should_quit() {
            return Ok(());
        }
    }
}

fn step<B: Backend>(
    terminal: &mut Terminal<B>,
    state: &mut ChatState,
    event: Option<&Event>,
) -> StratumResult<()> {
    terminal
        .draw(|f| state.render(f.area(), f.buffer_mut()))
        .map_err(map_io_error)?;
    if let Some(Event::Key(key)) = event {
        state.handle_key(*key);
    }
    Ok(())
}

fn map_io_error(err: io::Error) -> stratum_types::StratumError {
    stratum_types::StratumError::new(
        stratum_types::error::codes::E1001_INSTALLED_SCHEMA_UNREADABLE,
        "TUI io error",
    )
    .with_cause(err)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::thread;

    use crossterm::event::KeyCode;
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;
    use stratum_runtime::EventSink;

    use super::*;

    fn key(code: KeyCode, mods: KeyModifiers) -> KeyEvent {
        KeyEvent::new(code, mods)
    }

    fn state() -> ChatState {
        ChatState::new(EchoProvider::new("echo: "), Tier::High, "ready".into())
    }

    #[test]
    fn typing_appends_to_input() {
        let mut s = state();
        s.handle_key(key(KeyCode::Char('h'), KeyModifiers::NONE));
        s.handle_key(key(KeyCode::Char('i'), KeyModifiers::NONE));
        assert_eq!(s.input(), "hi");
    }

    #[test]
    fn backspace_pops_last_char() {
        let mut s = state();
        s.handle_key(key(KeyCode::Char('a'), KeyModifiers::NONE));
        s.handle_key(key(KeyCode::Char('b'), KeyModifiers::NONE));
        s.handle_key(key(KeyCode::Backspace, KeyModifiers::NONE));
        assert_eq!(s.input(), "a");
    }

    #[test]
    fn backspace_on_empty_is_noop() {
        let mut s = state();
        s.handle_key(key(KeyCode::Backspace, KeyModifiers::NONE));
        assert_eq!(s.input(), "");
    }

    #[test]
    fn enter_submits_and_clears_input() {
        let mut s = state();
        for c in "hello world".chars() {
            s.handle_key(key(KeyCode::Char(c), KeyModifiers::NONE));
        }
        s.handle_key(key(KeyCode::Enter, KeyModifiers::NONE));
        assert!(s.input().is_empty());
        assert!(matches!(s.transcript()[0], Turn::User(_)));
        assert!(matches!(s.transcript()[1], Turn::Assistant(_)));
    }

    #[test]
    fn enter_with_empty_input_is_noop() {
        let mut s = state();
        s.handle_key(key(KeyCode::Enter, KeyModifiers::NONE));
        assert!(s.transcript().is_empty());
    }

    #[test]
    fn esc_quits() {
        let mut s = state();
        s.handle_key(key(KeyCode::Esc, KeyModifiers::NONE));
        assert!(s.should_quit());
    }

    #[test]
    fn ctrl_c_quits_and_pushes_cancelled() {
        let mut s = state();
        s.handle_key(key(KeyCode::Char('c'), KeyModifiers::CONTROL));
        assert!(s.should_quit());
        assert!(matches!(s.transcript().last(), Some(Turn::Cancelled)));
    }

    #[test]
    fn ctrl_uppercase_c_also_cancels() {
        let mut s = state();
        s.handle_key(key(KeyCode::Char('C'), KeyModifiers::CONTROL));
        assert!(s.should_quit());
    }

    #[test]
    fn unhandled_key_is_ignored() {
        let mut s = state();
        s.handle_key(key(KeyCode::F(5), KeyModifiers::NONE));
        assert_eq!(s.input(), "");
        assert!(!s.should_quit());
    }

    #[test]
    fn slash_with_empty_input_opens_palette() {
        let mut s = state();
        s.handle_key(key(KeyCode::Char('/'), KeyModifiers::NONE));
        assert!(s.palette_open());
    }

    #[test]
    fn slash_with_text_input_just_appends() {
        let mut s = state();
        s.handle_key(key(KeyCode::Char('a'), KeyModifiers::NONE));
        s.handle_key(key(KeyCode::Char('/'), KeyModifiers::NONE));
        assert!(!s.palette_open());
        assert_eq!(s.input(), "a/");
    }

    #[test]
    fn palette_esc_closes_without_execute() {
        let mut s = state();
        s.handle_key(key(KeyCode::Char('/'), KeyModifiers::NONE));
        s.handle_key(key(KeyCode::Esc, KeyModifiers::NONE));
        assert!(!s.palette_open());
        assert!(s.transcript().is_empty());
    }

    #[test]
    fn palette_enter_executes_command() {
        let mut s = state();
        s.handle_key(key(KeyCode::Char('/'), KeyModifiers::NONE));
        // First alphabetical match is "active" — unknown to the
        // dispatcher, so it lands as a rejected command turn.
        s.handle_key(key(KeyCode::Enter, KeyModifiers::NONE));
        assert!(!s.palette_open());
        assert!(matches!(s.transcript().last(), Some(Turn::Command { .. })));
    }

    #[test]
    fn palette_quit_command_sets_quit() {
        let mut s = state();
        s.handle_key(key(KeyCode::Char('/'), KeyModifiers::NONE));
        for c in "qui".chars() {
            s.handle_key(key(KeyCode::Char(c), KeyModifiers::NONE));
        }
        s.handle_key(key(KeyCode::Enter, KeyModifiers::NONE));
        assert!(s.should_quit());
        let Some(Turn::Command { text, ok, .. }) = s.transcript().last() else {
            panic!("expected command turn")
        };
        assert_eq!(text, "/quit");
        assert!(*ok);
    }

    #[test]
    fn palette_ctrl_c_closes_without_executing() {
        let mut s = state();
        s.handle_key(key(KeyCode::Char('/'), KeyModifiers::NONE));
        s.handle_key(key(KeyCode::Char('c'), KeyModifiers::CONTROL));
        assert!(!s.palette_open());
        assert!(s.transcript().is_empty());
    }

    #[test]
    fn palette_typing_filters() {
        let mut s = state();
        s.handle_key(key(KeyCode::Char('/'), KeyModifiers::NONE));
        s.handle_key(key(KeyCode::Char('m'), KeyModifiers::NONE));
        s.handle_key(key(KeyCode::Enter, KeyModifiers::NONE));
        let Some(Turn::Command { text, .. }) = s.transcript().last() else {
            panic!("expected command turn")
        };
        // Filter "m" leaves "model" and "models"; cursor=0 picks "model".
        // The palette flush prepends the slash before dispatch.
        assert_eq!(text, "/model");
    }

    fn rendered_text(state: &ChatState, width: u16, height: u16) -> String {
        let backend = TestBackend::new(width, height);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|f| state.render(f.area(), f.buffer_mut()))
            .unwrap();
        let buf = terminal.backend().buffer();
        (0..buf.area.height)
            .map(|y| {
                (0..buf.area.width)
                    .map(|x| buf[(x, y)].symbol().to_string())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn render_shows_status_and_input() {
        let s = state();
        let text = rendered_text(&s, 60, 10);
        assert!(text.contains("stratum"));
        assert!(text.contains("tier=high"));
        assert!(text.contains("ready"));
        assert!(text.contains("input"));
    }

    #[test]
    fn render_shows_transcript_after_submit() {
        let mut s = state();
        for c in "hi".chars() {
            s.handle_key(key(KeyCode::Char(c), KeyModifiers::NONE));
        }
        s.handle_key(key(KeyCode::Enter, KeyModifiers::NONE));
        let text = rendered_text(&s, 80, 12);
        assert!(text.contains("you:"));
        assert!(text.contains("hi"));
        assert!(text.contains("echo: hi"));
    }

    #[test]
    fn submit_records_turn_metrics() {
        let mut s = state();
        for c in "hi".chars() {
            s.handle_key(key(KeyCode::Char(c), KeyModifiers::NONE));
        }
        s.handle_key(key(KeyCode::Enter, KeyModifiers::NONE));
        let metrics = s.last_metrics().expect("metrics recorded");
        assert_eq!(metrics.turn_id.0, 0);
        assert!(metrics.total_blocks >= 1);
        assert_eq!(metrics.role_steps.len(), 1);
        assert_eq!(metrics.role_steps[0].name, "generate");
    }

    #[test]
    fn turn_ids_increment_per_submit() {
        let mut s = state();
        for c in "hi".chars() {
            s.handle_key(key(KeyCode::Char(c), KeyModifiers::NONE));
        }
        s.handle_key(key(KeyCode::Enter, KeyModifiers::NONE));
        for c in "bye".chars() {
            s.handle_key(key(KeyCode::Char(c), KeyModifiers::NONE));
        }
        s.handle_key(key(KeyCode::Enter, KeyModifiers::NONE));
        let metrics = s.last_metrics().expect("metrics recorded");
        assert_eq!(metrics.turn_id.0, 1);
    }

    #[test]
    fn render_shows_token_meter_after_submit() {
        let mut s = state();
        for c in "hi".chars() {
            s.handle_key(key(KeyCode::Char(c), KeyModifiers::NONE));
        }
        s.handle_key(key(KeyCode::Enter, KeyModifiers::NONE));
        let text = rendered_text(&s, 100, 12);
        assert!(text.contains("turn 0"));
        assert!(text.contains("prompt:"));
        assert!(text.contains("compl:"));
        assert!(text.contains("tok/s"));
    }

    #[test]
    fn render_shows_palette_when_open() {
        let mut s = state();
        s.handle_key(key(KeyCode::Char('/'), KeyModifiers::NONE));
        s.handle_key(key(KeyCode::Char('m'), KeyModifiers::NONE));
        let text = rendered_text(&s, 60, 12);
        assert!(text.contains("palette"));
        assert!(text.contains("/model"));
        // Input pane is replaced; "input" title should be gone.
        assert!(!text.contains(" input "));
    }

    #[test]
    fn render_shows_executed_command_marker() {
        let mut s = state();
        // Dispatch a known command so the transcript shows the
        // "executed /<cmd>" marker.
        let outcome = s.execute_palette_command("/help");
        assert!(matches!(outcome, PaletteOutcome::Acknowledged { .. }));
        let text = rendered_text(&s, 80, 14);
        assert!(text.contains("executed /help"));
    }

    #[test]
    fn render_shows_cancelled_marker() {
        let mut s = state();
        s.handle_key(key(KeyCode::Char('c'), KeyModifiers::CONTROL));
        let text = rendered_text(&s, 60, 10);
        assert!(text.contains("(cancelled)"));
    }

    #[test]
    fn status_for_installed_marker() {
        let tmp = tempfile::TempDir::new().unwrap();
        let paths = Paths::under(tmp.path());
        paths.ensure_dirs().unwrap();
        std::fs::write(paths.installed_toml(), b"x").unwrap();
        assert_eq!(status_for(&paths), "installed");
    }

    #[test]
    fn status_for_uninstalled_hints_init() {
        let tmp = tempfile::TempDir::new().unwrap();
        let paths = Paths::under(tmp.path());
        assert!(status_for(&paths).contains("not installed"));
    }

    #[test]
    fn render_block_text_emits_ai_prefix() {
        let block = Block::Text { text: "hi".into() };
        let line = render_block(&block).unwrap();
        let rendered: String = line.spans.iter().map(|s| s.content.to_string()).collect();
        assert!(rendered.contains("ai:"));
        assert!(rendered.contains("hi"));
    }

    #[test]
    fn render_block_usage_emits_meter() {
        let line = render_block(&Block::Usage {
            prompt: 3,
            completion: 4,
        })
        .unwrap();
        let rendered: String = line.spans.iter().map(|s| s.content.to_string()).collect();
        assert!(rendered.contains("usage"));
    }

    #[test]
    fn render_block_done_returns_none() {
        assert!(render_block(&Block::Done).is_none());
    }

    #[test]
    fn render_block_tool_call_returns_none() {
        assert!(render_block(&Block::ToolCall {
            id: "t1".into(),
            tool: "fs.read".into(),
            args: "{}".into(),
        })
        .is_none());
    }

    #[test]
    fn render_block_tool_result_returns_none() {
        assert!(render_block(&Block::ToolResult {
            id: "t1".into(),
            output: "ok".into(),
        })
        .is_none());
    }

    #[test]
    fn render_block_cancelled_returns_some_with_reason() {
        let block = Block::Cancelled {
            reason: "STRAT-E4002".into(),
        };
        let line = render_block(&block).unwrap();
        let rendered: String = line.spans.iter().map(|s| s.content.to_string()).collect();
        assert!(rendered.contains("STRAT-E4002"));
    }

    #[test]
    fn map_io_error_carries_code() {
        let err = map_io_error(io::Error::other("x"));
        assert_eq!(
            err.code(),
            &stratum_types::error::codes::E1001_INSTALLED_SCHEMA_UNREADABLE
        );
    }

    #[test]
    fn step_renders_and_dispatches_key_event() {
        let mut s = state();
        let backend = TestBackend::new(40, 8);
        let mut terminal = Terminal::new(backend).unwrap();
        let evt = Event::Key(key(KeyCode::Char('a'), KeyModifiers::NONE));
        step(&mut terminal, &mut s, Some(&evt)).unwrap();
        assert_eq!(s.input(), "a");
    }

    #[test]
    fn step_renders_without_event() {
        let mut s = state();
        let backend = TestBackend::new(40, 8);
        let mut terminal = Terminal::new(backend).unwrap();
        step(&mut terminal, &mut s, None).unwrap();
        assert_eq!(s.input(), "");
    }

    #[test]
    fn step_ignores_non_key_events() {
        let mut s = state();
        let backend = TestBackend::new(40, 8);
        let mut terminal = Terminal::new(backend).unwrap();
        let evt = Event::FocusGained;
        step(&mut terminal, &mut s, Some(&evt)).unwrap();
        assert_eq!(s.input(), "");
        assert!(!s.should_quit());
    }

    // ---------- structured-event instrumentation ----------

    /// Drop-in `EventSink` for the "opaque sink" `with_events` test.
    #[derive(Debug, Default)]
    struct NullSink;
    impl EventSink for NullSink {
        fn write(&self, _record: EventRecord) {}
    }

    #[test]
    fn default_state_has_memory_event_emitter() {
        let s = ChatState::default();
        // Default sink is memory-backed -> snapshot is observable and empty.
        let snap = s.events_snapshot().expect("memory snapshot available");
        assert!(snap.is_empty());
    }

    #[test]
    fn echo_submit_emits_agent_handoff_event() {
        let mut s = state();
        for c in "hello world".chars() {
            s.handle_key(key(KeyCode::Char(c), KeyModifiers::NONE));
        }
        s.handle_key(key(KeyCode::Enter, KeyModifiers::NONE));
        // `submit` now routes through `AgentLoop::run_turn`, which emits
        // an `AgentHandoff` at the start of every turn. EchoProvider's
        // Text+Usage+Done blocks produce no further events (no ToolCall,
        // non-empty so no ProviderError).
        let snap = s.events_snapshot().expect("memory snapshot");
        assert_eq!(snap.len(), 1, "got events: {snap:?}");
        assert!(matches!(snap[0].event, RtEvent::AgentHandoff { .. }));
    }

    #[test]
    fn submit_with_zero_blocks_emits_provider_error() {
        let mut s = state();
        s.finish_turn("hi".to_string(), Vec::new(), "echo", 0);
        let snap = s.events_snapshot().expect("memory snapshot");
        assert_eq!(snap.len(), 1);
        let RtEvent::ProviderError {
            provider,
            code,
            message,
        } = &snap[0].event
        else {
            panic!("expected ProviderError, got {:?}", snap[0].event);
        };
        assert_eq!(provider, "echo");
        assert_eq!(code, "E_NO_BLOCKS");
        assert!(!message.is_empty());
    }

    #[test]
    fn tool_call_block_emits_tool_call_event() {
        let mut s = state();
        let blocks = vec![
            Block::ToolCall {
                id: "t1".into(),
                tool: "fs.read".into(),
                args: "{}".into(),
            },
            Block::Done,
        ];
        s.finish_turn("run".to_string(), blocks, "echo", 7);
        let snap = s.events_snapshot().expect("memory snapshot");
        assert_eq!(snap.len(), 1);
        let RtEvent::ToolCall {
            tool_id,
            ok,
            duration_ms,
        } = &snap[0].event
        else {
            panic!("expected ToolCall, got {:?}", snap[0].event);
        };
        assert_eq!(tool_id, "t1");
        assert!(*ok);
        assert_eq!(*duration_ms, 7);
        assert_eq!(snap[0].turn_id, Some(0));
    }

    #[test]
    fn events_snapshot_aggregates_across_submits() {
        let mut s = state();
        // First turn: tool call.
        s.finish_turn(
            "a".to_string(),
            vec![Block::ToolCall {
                id: "t1".into(),
                tool: "fs.read".into(),
                args: "{}".into(),
            }],
            "echo",
            1,
        );
        // Second turn: provider error (zero blocks).
        s.finish_turn("b".to_string(), Vec::new(), "echo", 2);
        // Third turn: clean text turn — no events.
        s.finish_turn(
            "c".to_string(),
            vec![Block::Text { text: "ok".into() }],
            "echo",
            3,
        );
        let snap = s.events_snapshot().expect("memory snapshot");
        assert_eq!(snap.len(), 2);
        assert!(matches!(snap[0].event, RtEvent::ToolCall { .. }));
        assert!(matches!(snap[1].event, RtEvent::ProviderError { .. }));
        assert_eq!(snap[0].turn_id, Some(0));
        assert_eq!(snap[1].turn_id, Some(1));
    }

    #[test]
    fn with_events_non_memory_sink_yields_none_snapshot() {
        let sink: Arc<dyn EventSink> = Arc::new(NullSink);
        let emitter = Arc::new(EventEmitter::new(sink));
        let s = state().with_events(emitter);
        assert!(s.events_snapshot().is_none());
    }

    #[test]
    fn with_events_swaps_emitter_target() {
        // Build a memory sink we own and watch directly.
        let sink = Arc::new(MemoryEventSink::new());
        let sink_dyn: Arc<dyn EventSink> = sink.clone();
        let emitter = Arc::new(EventEmitter::new(sink_dyn));
        let mut s = state().with_events(emitter);
        s.finish_turn("x".to_string(), Vec::new(), "echo", 0);
        // ChatState no longer has its own MemoryEventSink handle.
        assert!(s.events_snapshot().is_none());
        // But the externally-owned sink received the event.
        let external = sink.snapshot();
        assert_eq!(external.len(), 1);
        assert!(matches!(external[0].event, RtEvent::ProviderError { .. }));
    }

    #[test]
    fn concurrent_submits_produce_monotonic_event_ids() {
        // Drive the emitter directly from many threads through a shared
        // ChatState-style emitter. We can't share &mut ChatState across
        // threads, so exercise the emitter that backs it.
        let sink = Arc::new(MemoryEventSink::new());
        let sink_dyn: Arc<dyn EventSink> = sink.clone();
        let emitter = Arc::new(EventEmitter::new(sink_dyn));
        let mut handles = Vec::new();
        for t in 0..4_u64 {
            let em = emitter.clone();
            handles.push(thread::spawn(move || {
                for _ in 0..50_u64 {
                    em.emit(
                        RtEvent::ProviderError {
                            provider: "echo".to_string(),
                            code: "E_NO_BLOCKS".to_string(),
                            message: "x".to_string(),
                        },
                        Some(t),
                    );
                }
            }));
        }
        for h in handles {
            h.join().expect("join");
        }
        let snap = sink.snapshot();
        assert_eq!(snap.len(), 200);
        let mut ids: Vec<u64> = snap.iter().map(|r| r.id).collect();
        ids.sort_unstable();
        // Strictly monotonic, unique, starting at 1.
        assert_eq!(*ids.first().expect("first"), 1);
        assert_eq!(*ids.last().expect("last"), 200);
        let mut dedup = ids.clone();
        dedup.dedup();
        assert_eq!(dedup.len(), ids.len());
    }

    // ---------- AgentLoop integration ----------

    /// Test provider whose blocks come from a script. Empty script ⇒ no
    /// blocks (lets us assert the zero-block path).
    #[derive(Debug)]
    struct ScriptedProvider {
        script: std::sync::Mutex<Vec<Block>>,
    }

    impl ScriptedProvider {
        fn empty() -> Self {
            Self {
                script: std::sync::Mutex::new(Vec::new()),
            }
        }
    }

    impl Provider for ScriptedProvider {
        fn id(&self) -> &'static str {
            "scripted"
        }
        fn capabilities(&self) -> &'static [stratum_types::Capability] {
            const CAPS: &[stratum_types::Capability] = &[stratum_types::Capability::Generate];
            CAPS
        }
        fn generate(
            &self,
            _req: &stratum_runtime::GenerateRequest,
            _cancel: &CancelToken,
        ) -> Vec<Block> {
            self.script
                .lock()
                .map(|mut v| std::mem::take(&mut *v))
                .unwrap_or_default()
        }
    }

    fn build_loop(provider: Arc<dyn Provider>, events: Arc<EventEmitter>) -> Arc<AgentLoop> {
        Arc::new(
            AgentLoop::builder()
                .with_provider(provider)
                .with_router(stratum_runtime::IntentRouter::default())
                .with_permission_store(Arc::new(stratum_runtime::PermissionStore::new()))
                .with_prompt_gen(Arc::new(stratum_runtime::PromptIdGen::new()))
                .with_responder(Arc::new(stratum_runtime::AllowAllResponder))
                .with_events(events)
                .with_capability_matrix(Arc::new(stratum_runtime::CapabilityMatrix::new()))
                .with_plan_mode(Arc::new(stratum_runtime::PlanMode::new()))
                .with_config(stratum_runtime::AgentLoopConfig::default())
                .build()
                .unwrap(),
        )
    }

    #[test]
    fn default_state_constructs_without_panic() {
        // Smoke: default state can be built and probed without unwrapping.
        let s = ChatState::default();
        assert!(s.last_turn_result().is_none());
        assert!(s.transcript().is_empty());
    }

    #[test]
    fn submit_hello_records_user_then_assistant() {
        let mut s = state();
        for c in "hello".chars() {
            s.handle_key(key(KeyCode::Char(c), KeyModifiers::NONE));
        }
        s.handle_key(key(KeyCode::Enter, KeyModifiers::NONE));
        assert_eq!(s.transcript().len(), 2);
        match &s.transcript()[0] {
            Turn::User(p) => assert_eq!(p, "hello"),
            other => panic!("expected Turn::User, got {other:?}"),
        }
        assert!(matches!(s.transcript()[1], Turn::Assistant(_)));
    }

    #[test]
    fn submit_with_scripted_zero_block_provider_emits_provider_error() {
        // Wire a sink we can inspect, point the AgentLoop at the same
        // emitter, and feed the state via `with_agent_loop` so the
        // scripted-zero-blocks path is exercised end-to-end.
        let sink = Arc::new(MemoryEventSink::new());
        let sink_dyn: Arc<dyn EventSink> = sink.clone();
        let events = Arc::new(EventEmitter::new(sink_dyn));
        let provider: Arc<dyn Provider> = Arc::new(ScriptedProvider::empty());
        let loop_ = build_loop(provider, events);
        let mut s = ChatState::with_agent_loop(loop_);

        for c in "anything".chars() {
            s.handle_key(key(KeyCode::Char(c), KeyModifiers::NONE));
        }
        s.handle_key(key(KeyCode::Enter, KeyModifiers::NONE));

        assert_eq!(s.transcript().len(), 2);
        match &s.transcript()[1] {
            Turn::Assistant(blocks) => assert!(blocks.is_empty(), "expected empty blocks"),
            other => panic!("expected Turn::Assistant, got {other:?}"),
        }
        let snap = sink.snapshot();
        assert!(
            snap.iter()
                .any(|r| matches!(r.event, RtEvent::ProviderError { .. })),
            "expected a ProviderError event, got: {snap:?}"
        );
    }

    #[test]
    fn last_turn_result_none_before_submit_some_after() {
        let mut s = state();
        assert!(s.last_turn_result().is_none());
        for c in "hi".chars() {
            s.handle_key(key(KeyCode::Char(c), KeyModifiers::NONE));
        }
        s.handle_key(key(KeyCode::Enter, KeyModifiers::NONE));
        let tr = s.last_turn_result().expect("turn result populated");
        assert!(matches!(tr.outcome, stratum_runtime::TurnOutcome::Success));
    }

    #[test]
    fn empty_prompt_is_noop_through_agent_loop() {
        let mut s = state();
        // Type only whitespace, then Enter.
        s.handle_key(key(KeyCode::Char(' '), KeyModifiers::NONE));
        s.handle_key(key(KeyCode::Enter, KeyModifiers::NONE));
        assert!(s.transcript().is_empty());
        assert!(s.last_turn_result().is_none());
    }

    #[test]
    fn concurrent_submits_across_threads_do_not_panic() {
        use std::sync::Mutex;
        // 4 threads × 25 submits = 200 user+assistant pairs distributed
        // across 4 shared ChatStates (one per thread). Each thread owns its
        // own state — concurrency at the per-state level is exercised by
        // serial submits, while the AgentLoop is the same `Arc` shared
        // across threads via the loops built per state.
        let states: Vec<Arc<Mutex<ChatState>>> =
            (0..4).map(|_| Arc::new(Mutex::new(state()))).collect();
        let mut handles = Vec::new();
        for s in &states {
            let s = s.clone();
            handles.push(thread::spawn(move || {
                for i in 0..25_u32 {
                    let mut g = s.lock().unwrap();
                    for c in format!("hi{i}").chars() {
                        g.handle_key(key(KeyCode::Char(c), KeyModifiers::NONE));
                    }
                    g.handle_key(key(KeyCode::Enter, KeyModifiers::NONE));
                }
            }));
        }
        for h in handles {
            h.join().expect("thread join");
        }
        let total: usize = states
            .iter()
            .map(|s| s.lock().unwrap().transcript().len())
            .sum();
        assert_eq!(total, 200);
    }

    #[test]
    fn with_agent_loop_swaps_orchestrator() {
        // The `with_agent_loop` builder returns a state that delegates to
        // the supplied loop. Smoke-check that submit still records a turn.
        let sink: Arc<dyn EventSink> = Arc::new(MemoryEventSink::new());
        let events = Arc::new(EventEmitter::new(sink));
        let provider: Arc<dyn Provider> = Arc::new(EchoProvider::new("ECHO> "));
        let loop_ = build_loop(provider, events);
        let mut s = ChatState::with_agent_loop(loop_);
        for c in "ping".chars() {
            s.handle_key(key(KeyCode::Char(c), KeyModifiers::NONE));
        }
        s.handle_key(key(KeyCode::Enter, KeyModifiers::NONE));
        assert_eq!(s.transcript().len(), 2);
    }

    // ---------- palette command dispatch ----------

    #[test]
    fn execute_plan_activates_and_acknowledges() {
        let mut s = state();
        assert!(!s.plan_mode.is_active());
        let outcome = s.execute_palette_command("/plan");
        assert!(matches!(outcome, PaletteOutcome::Acknowledged { .. }));
        assert!(s.plan_mode.is_active());
    }

    #[test]
    fn execute_plan_toggle_deactivates_when_already_active() {
        let mut s = state();
        s.execute_palette_command("/plan");
        assert!(s.plan_mode.is_active());
        let outcome = s.execute_palette_command("/plan");
        assert!(matches!(outcome, PaletteOutcome::Acknowledged { .. }));
        assert!(!s.plan_mode.is_active());
    }

    #[test]
    fn execute_plan_on_force_activates_regardless_of_state() {
        let mut s = state();
        // Inactive -> on.
        s.execute_palette_command("/plan on");
        assert!(s.plan_mode.is_active());
        // Active -> on (still active).
        s.execute_palette_command("/plan on");
        assert!(s.plan_mode.is_active());
    }

    #[test]
    fn execute_plan_off_force_deactivates_regardless_of_state() {
        let mut s = state();
        // Inactive -> off (still inactive).
        s.execute_palette_command("/plan off");
        assert!(!s.plan_mode.is_active());
        // Activate, then off.
        s.execute_palette_command("/plan on");
        assert!(s.plan_mode.is_active());
        s.execute_palette_command("/plan off");
        assert!(!s.plan_mode.is_active());
    }

    #[test]
    fn execute_plan_unknown_arg_is_rejected() {
        let mut s = state();
        let outcome = s.execute_palette_command("/plan maybe");
        assert!(matches!(outcome, PaletteOutcome::Rejected { .. }));
        assert!(!s.plan_mode.is_active());
    }

    #[test]
    fn execute_cancel_fires_cancel_token() {
        let mut s = state();
        assert!(!s.cancel.is_cancelled());
        let outcome = s.execute_palette_command("/cancel");
        assert!(matches!(outcome, PaletteOutcome::Acknowledged { .. }));
        assert!(s.cancel.is_cancelled());
    }

    #[test]
    fn execute_clear_empties_transcript() {
        // /clear is a meta-action: it removes prior turns *and* its own
        // marker, leaving the transcript empty.
        let mut s = state();
        for c in "hi".chars() {
            s.handle_key(key(KeyCode::Char(c), KeyModifiers::NONE));
        }
        s.handle_key(key(KeyCode::Enter, KeyModifiers::NONE));
        assert!(!s.transcript().is_empty());
        s.execute_palette_command("/clear");
        assert!(s.transcript().is_empty());
    }

    #[test]
    fn execute_help_returns_acknowledged_with_nonempty_message() {
        let mut s = state();
        let outcome = s.execute_palette_command("/help");
        let PaletteOutcome::Acknowledged { message } = outcome else {
            panic!("expected acknowledged");
        };
        assert!(!message.is_empty());
        assert!(message.contains("/plan"));
        assert!(message.contains("/cancel"));
        assert!(message.contains("/clear"));
        assert!(message.contains("/budget"));
        assert!(message.contains("/quit"));
    }

    #[test]
    fn help_text_const_advertises_budget() {
        assert!(HELP_TEXT.contains("/budget"));
    }

    #[test]
    fn execute_budget_with_no_metrics_returns_sentinel() {
        let mut s = state();
        let outcome = s.execute_palette_command("/budget");
        let PaletteOutcome::Acknowledged { message } = outcome else {
            panic!("expected acknowledged");
        };
        assert_eq!(message, "no turn metrics yet");
    }

    #[test]
    fn execute_budget_after_submit_reports_formatted_metrics() {
        let mut s = state();
        s.submit_with_prompt("hello world");
        let outcome = s.execute_palette_command("/budget");
        let PaletteOutcome::Acknowledged { message } = outcome else {
            panic!("expected acknowledged");
        };
        // Format: "metrics: <tokens> tokens · <wall_ms>ms · <tps> tok/s · turn id <N>"
        assert!(message.starts_with("metrics: "), "got: {message}");
        assert!(message.contains(" tokens · "), "got: {message}");
        assert!(message.contains("ms · "), "got: {message}");
        assert!(message.contains(" tok/s · turn id "), "got: {message}");
        // After one submit, turn id 0 was recorded.
        assert!(message.ends_with("turn id 0"), "got: {message}");
    }

    #[test]
    fn execute_quit_sets_should_quit() {
        let mut s = state();
        assert!(!s.should_quit());
        let outcome = s.execute_palette_command("/quit");
        assert!(matches!(outcome, PaletteOutcome::Acknowledged { .. }));
        assert!(s.should_quit());
    }

    #[test]
    fn execute_exit_is_alias_for_quit() {
        let mut s = state();
        assert!(!s.should_quit());
        let outcome = s.execute_palette_command("/exit");
        assert!(matches!(outcome, PaletteOutcome::Acknowledged { .. }));
        assert!(s.should_quit());
    }

    #[test]
    fn execute_unknown_command_returns_rejected() {
        let mut s = state();
        let outcome = s.execute_palette_command("/unknown");
        let PaletteOutcome::Rejected { message } = outcome else {
            panic!("expected rejected");
        };
        assert_eq!(message, "unknown command: /unknown");
    }

    #[test]
    fn execute_empty_command_returns_rejected() {
        let mut s = state();
        let outcome = s.execute_palette_command("");
        assert!(matches!(outcome, PaletteOutcome::Rejected { .. }));
    }

    #[test]
    fn execute_slash_only_returns_rejected() {
        // `cmd = "/"` strips the prefix but leaves an empty body — covered
        // by the dedicated empty-trim branch.
        let mut s = state();
        let outcome = s.execute_palette_command("/");
        assert!(matches!(outcome, PaletteOutcome::Rejected { .. }));
    }

    #[test]
    fn execute_command_via_palette_flush() {
        // Exercise the palette → execute_command → execute_palette_command
        // bridge (lower-level than the existing `Enter`-key tests). Pick a
        // recognised command so the dispatch path is OK.
        let mut s = state();
        s.execute_command("help");
        let Turn::Command { text, ok, .. } = s.transcript().last().expect("turn") else {
            panic!("expected command turn")
        };
        assert_eq!(text, "/help");
        assert!(*ok);
    }

    #[test]
    fn render_rejected_command_shows_marker() {
        let mut s = state();
        s.execute_palette_command("/unknown");
        let text = rendered_text(&s, 80, 14);
        assert!(text.contains("rejected /unknown"));
    }

    #[test]
    fn transcript_records_ok_true_after_plan() {
        let mut s = state();
        s.execute_palette_command("/plan");
        let Turn::Command { ok, text, .. } = s.transcript().last().expect("turn") else {
            panic!("expected command turn")
        };
        assert!(*ok);
        assert_eq!(text, "/plan");
    }

    #[test]
    fn transcript_records_ok_false_after_unknown() {
        let mut s = state();
        s.execute_palette_command("/unknown");
        let Turn::Command { ok, text, message } = s.transcript().last().expect("turn") else {
            panic!("expected command turn")
        };
        assert!(!*ok);
        assert_eq!(text, "/unknown");
        assert_eq!(message, "unknown command: /unknown");
    }

    #[test]
    fn clear_then_help_results_in_single_help_entry() {
        let mut s = state();
        s.execute_palette_command("/plan");
        s.execute_palette_command("/clear");
        // /clear wipes its own marker, so the transcript is empty here.
        assert!(s.transcript().is_empty());
        s.execute_palette_command("/help");
        assert_eq!(s.transcript().len(), 1);
        let Turn::Command { text, .. } = &s.transcript()[0] else {
            panic!("expected command turn")
        };
        assert_eq!(text, "/help");
    }

    #[test]
    fn submit_after_cancel_still_proceeds_via_agent_loop() {
        // Document the current AgentLoop behavior: a parent cancel set
        // before submit causes `run_turn` to return a UserAbort outcome,
        // and the state still pushes a User+Assistant pair. We assert
        // the user-abort outcome to pin the contract.
        let mut s = state();
        s.execute_palette_command("/cancel");
        assert!(s.cancel.is_cancelled());
        for c in "hi".chars() {
            s.handle_key(key(KeyCode::Char(c), KeyModifiers::NONE));
        }
        s.handle_key(key(KeyCode::Enter, KeyModifiers::NONE));
        let tr = s.last_turn_result().expect("turn result populated");
        // AgentLoop sees the cancel and short-circuits with UserAbort.
        assert!(matches!(
            tr.outcome,
            stratum_runtime::TurnOutcome::UserAbort
        ));
    }

    #[test]
    fn should_quit_is_false_initially() {
        let s = state();
        assert!(!s.should_quit());
    }

    // ---------- non-interactive --prompt helpers ----------

    #[test]
    fn submit_with_prompt_stages_input_and_runs_turn() {
        let mut s = state();
        s.submit_with_prompt("hello");
        // Input must be drained after submit, transcript holds a
        // user+assistant pair.
        assert!(s.input().is_empty());
        assert_eq!(s.transcript().len(), 2);
        match &s.transcript()[0] {
            Turn::User(p) => assert_eq!(p, "hello"),
            other => panic!("expected Turn::User, got {other:?}"),
        }
    }

    #[test]
    fn last_assistant_text_returns_echo_output() {
        let mut s = state();
        s.submit_with_prompt("hello");
        let text = s.last_assistant_text().expect("assistant text");
        // EchoProvider's prefix here is "echo: ".
        assert!(text.contains("hello"), "got: {text}");
        assert!(text.starts_with("echo: "), "got: {text}");
    }

    #[test]
    fn last_assistant_text_none_when_no_turn() {
        let s = state();
        assert!(s.last_assistant_text().is_none());
    }

    #[test]
    fn last_assistant_text_none_when_assistant_has_only_usage() {
        let mut s = state();
        // Push a synthetic assistant turn with no Text blocks — only
        // Usage. Helper exercises the "no text blocks at all" branch.
        s.transcript.push(Turn::User("x".to_string()));
        s.transcript.push(Turn::Assistant(vec![Block::Usage {
            prompt: 1,
            completion: 1,
        }]));
        assert!(s.last_assistant_text().is_none());
    }

    #[test]
    fn submit_with_prompt_empty_is_noop() {
        let mut s = state();
        s.submit_with_prompt("");
        assert!(s.transcript().is_empty());
    }

    // ---------- --resume / with_resumed_transcript ----------

    fn fixed_at() -> std::time::SystemTime {
        std::time::SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(1_700_000_000)
    }

    fn resume_fixture() -> stratum_runtime::Transcript {
        use stratum_runtime::{
            SessionId, TranscriptBlock, TranscriptBlockKind, TRANSCRIPT_SCHEMA_VERSION,
        };
        stratum_runtime::Transcript {
            schema_version: TRANSCRIPT_SCHEMA_VERSION,
            session_id: SessionId::from_str("deadbeefcafef00d").expect("valid id"),
            created_at: fixed_at(),
            turns: vec![
                stratum_runtime::TranscriptTurn::User {
                    at: fixed_at(),
                    text: "earlier-q".to_owned(),
                },
                stratum_runtime::TranscriptTurn::Assistant {
                    at: fixed_at(),
                    blocks: vec![TranscriptBlock {
                        kind: TranscriptBlockKind::Text,
                        text: "earlier-a".to_owned(),
                    }],
                },
                stratum_runtime::TranscriptTurn::System {
                    at: fixed_at(),
                    text: "sysline".to_owned(),
                },
                stratum_runtime::TranscriptTurn::Command {
                    at: fixed_at(),
                    text: "/help".to_owned(),
                    ok: true,
                },
            ],
        }
    }

    #[test]
    fn with_resumed_transcript_counts_turns_and_populates_scrollback() {
        let t = resume_fixture();
        let expected = t.turns.len();
        let s = ChatState::default().with_resumed_transcript(t);
        assert_eq!(s.resumed_count(), expected);
        assert_eq!(s.transcript().len(), expected);
        // User and Assistant turns must round-trip through the in-memory
        // shapes — System and Command both fold to Turn::Command.
        assert!(matches!(&s.transcript()[0], Turn::User(text) if text == "earlier-q"));
        match &s.transcript()[1] {
            Turn::Assistant(blocks) => {
                assert_eq!(blocks.len(), 1);
                assert!(matches!(&blocks[0], Block::Text { text } if text == "earlier-a"));
            }
            other => panic!("expected Turn::Assistant, got {other:?}"),
        }
        match &s.transcript()[2] {
            Turn::Command { text, ok, .. } => {
                assert!(text.starts_with("(system)"), "got: {text}");
                assert!(ok);
            }
            other => panic!("expected Turn::Command for system, got {other:?}"),
        }
        match &s.transcript()[3] {
            Turn::Command { text, ok, .. } => {
                assert_eq!(text, "/help");
                assert!(ok);
            }
            other => panic!("expected Turn::Command for command, got {other:?}"),
        }
    }

    #[test]
    fn with_resumed_transcript_default_count_is_zero() {
        let s = ChatState::default();
        assert_eq!(s.resumed_count(), 0);
    }

    #[test]
    fn with_resumed_transcript_empty_turns_yields_zero_count() {
        use stratum_runtime::{SessionId, TRANSCRIPT_SCHEMA_VERSION};
        let t = stratum_runtime::Transcript {
            schema_version: TRANSCRIPT_SCHEMA_VERSION,
            session_id: SessionId::from_str("0123456789abcdef").expect("valid id"),
            created_at: fixed_at(),
            turns: vec![],
        };
        let s = ChatState::default().with_resumed_transcript(t);
        assert_eq!(s.resumed_count(), 0);
        assert!(s.transcript().is_empty());
    }

    // ---------- TuiPromptResponder integration ----------

    /// Dispatcher that handles `fs.write` invocations by returning a
    /// trivial success. Used by the permission-flow tests.
    #[derive(Debug)]
    struct FsWriteDispatcher;
    impl stratum_runtime::ToolDispatcher for FsWriteDispatcher {
        fn invoke(&self, inv: &stratum_runtime::ToolInvocation) -> stratum_runtime::ToolResult {
            let body = serde_json::Value::Bool(true);
            let bytes = body.to_string().len() as u64;
            stratum_runtime::ToolResult::Ok {
                tool_id: inv.tool_id.clone(),
                body,
                bytes,
            }
        }
        fn supports(&self, tool_id: &str) -> bool {
            tool_id == "fs.write"
        }
        fn id(&self) -> &'static str {
            "fs.write"
        }
    }

    fn loop_with_prompter(
        provider: Arc<dyn Provider>,
        prompter: Arc<TuiPromptResponder>,
    ) -> Arc<AgentLoop> {
        let sink: Arc<dyn EventSink> = Arc::new(MemoryEventSink::new());
        let events = Arc::new(EventEmitter::new(sink));
        let mut dispatcher = stratum_runtime::RegistryDispatcher::new();
        dispatcher
            .register(Box::new(FsWriteDispatcher))
            .expect("register");
        let responder: Arc<dyn stratum_runtime::PromptResponder> = prompter;
        Arc::new(
            AgentLoop::builder()
                .with_provider(provider)
                .with_router(stratum_runtime::IntentRouter::default())
                .with_permission_store(Arc::new(stratum_runtime::PermissionStore::new()))
                .with_prompt_gen(Arc::new(stratum_runtime::PromptIdGen::new()))
                .with_responder(responder)
                .with_events(events)
                .with_capability_matrix(Arc::new(stratum_runtime::CapabilityMatrix::new()))
                .with_plan_mode(Arc::new(stratum_runtime::PlanMode::new()))
                .with_config(stratum_runtime::AgentLoopConfig::default())
                .with_dispatcher(Arc::new(dispatcher))
                .build()
                .unwrap(),
        )
    }

    #[test]
    fn pending_permission_request_initially_none() {
        let s = state();
        assert!(s.pending_permission_request().is_none());
        assert!(s.peek_pending_permission().is_none());
    }

    #[test]
    fn answer_permission_unblocks_agent_loop_with_allow_once() {
        // ScriptedProvider yields one ToolCall for "fs.write". The
        // AgentLoop will dispatch the permission flow through the shared
        // TuiPromptResponder; the main thread answers AllowOnce; the
        // background `submit` thread completes successfully.
        let prompter = Arc::new(TuiPromptResponder::new(Duration::from_secs(2)));
        let provider = ScriptedProvider {
            script: std::sync::Mutex::new(vec![Block::ToolCall {
                id: "fs.write".into(),
                tool: "fs.write".into(),
                args: "{}".into(),
            }]),
        };
        let provider_arc: Arc<dyn Provider> = Arc::new(provider);
        let loop_ = loop_with_prompter(provider_arc, prompter.clone());

        // Inject the same prompter into a ChatState alongside the loop.
        let mut s = ChatState::with_agent_loop(loop_);
        s.permission_prompter = prompter.clone();

        let s_arc = Arc::new(std::sync::Mutex::new(s));
        let s_bg = s_arc.clone();
        let bg = thread::spawn(move || {
            let mut guard = s_bg.lock().expect("lock");
            guard.submit_with_prompt("write a file");
        });

        // Wait until the worker has enqueued its request.
        let pending = loop {
            if let Some(p) = prompter.peek_request() {
                break p;
            }
            thread::sleep(Duration::from_millis(5));
        };
        prompter.submit_decision(pending.id, PermissionDecision::AllowOnce);

        bg.join().expect("background submit completes");
        let outcome = {
            let guard = s_arc.lock().expect("lock");
            guard
                .last_turn_result()
                .expect("turn result")
                .outcome
                .clone()
        };
        assert!(
            matches!(outcome, stratum_runtime::TurnOutcome::Success),
            "expected Success, got {outcome:?}",
        );
    }

    #[test]
    fn answer_permission_deny_short_circuits_turn() {
        let prompter = Arc::new(TuiPromptResponder::new(Duration::from_secs(2)));
        let provider = ScriptedProvider {
            script: std::sync::Mutex::new(vec![Block::ToolCall {
                id: "fs.write".into(),
                tool: "fs.write".into(),
                args: "{}".into(),
            }]),
        };
        let provider_arc: Arc<dyn Provider> = Arc::new(provider);
        let loop_ = loop_with_prompter(provider_arc, prompter.clone());

        let mut s = ChatState::with_agent_loop(loop_);
        s.permission_prompter = prompter.clone();
        let s_arc = Arc::new(std::sync::Mutex::new(s));
        let s_bg = s_arc.clone();
        let bg = thread::spawn(move || {
            let mut guard = s_bg.lock().expect("lock");
            guard.submit_with_prompt("write a file");
        });

        let pending = loop {
            if let Some(p) = prompter.peek_request() {
                break p;
            }
            thread::sleep(Duration::from_millis(5));
        };
        // Use the public delegate.
        {
            let guard = s_arc.lock().expect("lock");
            guard.answer_permission(pending.id, PermissionDecision::Deny);
        }

        bg.join().expect("background submit completes");
        let outcome = {
            let guard = s_arc.lock().expect("lock");
            guard
                .last_turn_result()
                .expect("turn result")
                .outcome
                .clone()
        };
        assert!(
            matches!(
                outcome,
                stratum_runtime::TurnOutcome::ToolFailure { ref code, .. } if code == "STRAT-E5004"
            ),
            "expected ToolFailure with STRAT-E5004, got {outcome:?}",
        );
    }

    #[test]
    fn prompter_times_out_to_deny_when_unanswered() {
        // Wire a short-timeout prompter and never answer; the AgentLoop
        // should see a Deny and short-circuit the turn.
        let prompter = Arc::new(TuiPromptResponder::new(Duration::from_millis(100)));
        let provider = ScriptedProvider {
            script: std::sync::Mutex::new(vec![Block::ToolCall {
                id: "fs.write".into(),
                tool: "fs.write".into(),
                args: "{}".into(),
            }]),
        };
        let provider_arc: Arc<dyn Provider> = Arc::new(provider);
        let loop_ = loop_with_prompter(provider_arc, prompter.clone());
        let mut s = ChatState::with_agent_loop(loop_);
        s.permission_prompter = prompter;
        s.submit_with_prompt("write a file");
        let tr = s.last_turn_result().expect("turn result");
        assert!(
            matches!(
                tr.outcome,
                stratum_runtime::TurnOutcome::ToolFailure { ref code, .. } if code == "STRAT-E5004"
            ),
            "expected ToolFailure with STRAT-E5004, got {:?}",
            tr.outcome,
        );
    }

    #[test]
    fn submit_decision_then_pending_returns_none_after_consumed() {
        // Submitting a decision is independent of the queue; once
        // pending_request has popped the request, a subsequent call
        // returns None even after a decision was recorded.
        let prompter = TuiPromptResponder::new(Duration::from_secs(1));
        let p = PendingPrompt {
            id: PromptId(99),
            request: PermissionRequest::ToolUse {
                tool_id: "fs.write".into(),
            },
            issued_at: SystemTime::UNIX_EPOCH,
        };
        // Manually enqueue + drain (no waiter).
        prompter.requeue_for_redisplay(p);
        let popped = prompter.pending_request().expect("popped");
        assert_eq!(popped.id, PromptId(99));
        prompter.submit_decision(PromptId(99), PermissionDecision::AllowOnce);
        // Queue is empty: the recorded decision does not put the request
        // back into the queue.
        assert!(prompter.pending_request().is_none());
    }

    #[test]
    fn decision_from_key_maps_all_documented_keys() {
        assert_eq!(decision_from_key('a'), Some(PermissionDecision::AllowOnce));
        assert_eq!(
            decision_from_key('s'),
            Some(PermissionDecision::AllowSession)
        );
        assert_eq!(
            decision_from_key('f'),
            Some(PermissionDecision::AllowForever)
        );
        assert_eq!(decision_from_key('d'), Some(PermissionDecision::Deny));
        assert_eq!(
            decision_from_key('F'),
            Some(PermissionDecision::DenyForever)
        );
        assert_eq!(decision_from_key('x'), None);
        assert_eq!(decision_from_key('A'), None);
    }

    #[test]
    fn describe_request_covers_every_variant() {
        let cap = PermissionRequest::CapabilityGrant {
            capability: "net".into(),
            target: Some("example.com".into()),
            reason: "fetch".into(),
        };
        assert!(describe_request(&cap).contains("net"));
        assert!(describe_request(&cap).contains("example.com"));
        let cap_no_target = PermissionRequest::CapabilityGrant {
            capability: "shell".into(),
            target: None,
            reason: "run".into(),
        };
        assert!(describe_request(&cap_no_target).contains("shell"));
        let secret = PermissionRequest::SecretAccess {
            secret_ref: "p/k".into(),
            scope: "read".into(),
        };
        assert!(describe_request(&secret).contains("p/k"));
        let net = PermissionRequest::NetworkHost {
            host: "api.example".into(),
            port: Some(443),
        };
        assert!(describe_request(&net).contains("443"));
        let net_anyport = PermissionRequest::NetworkHost {
            host: "api.example".into(),
            port: None,
        };
        assert!(describe_request(&net_anyport).contains("api.example"));
        let file = PermissionRequest::FileWrite {
            path: std::path::PathBuf::from("/tmp/x"),
        };
        assert!(describe_request(&file).contains("/tmp/x"));
        let tool = PermissionRequest::ToolUse {
            tool_id: "fs.write".into(),
        };
        assert!(describe_request(&tool).contains("fs.write"));
    }

    #[test]
    fn handle_key_with_pending_request_swallows_unknown_char() {
        let mut s = state();
        let pending = PendingPrompt {
            id: PromptId(5),
            request: PermissionRequest::ToolUse {
                tool_id: "fs.write".into(),
            },
            issued_at: SystemTime::UNIX_EPOCH,
        };
        s.permission_prompter.requeue_for_redisplay(pending);
        // 'z' is not one of the modal keys: input must stay clean.
        s.handle_key(key(KeyCode::Char('z'), KeyModifiers::NONE));
        assert!(s.input().is_empty());
        // Modal still pending.
        assert!(s.peek_pending_permission().is_some());
    }

    #[test]
    fn handle_key_with_pending_request_dispatches_known_decision_key() {
        let mut s = state();
        let pending = PendingPrompt {
            id: PromptId(6),
            request: PermissionRequest::ToolUse {
                tool_id: "fs.write".into(),
            },
            issued_at: SystemTime::UNIX_EPOCH,
        };
        s.permission_prompter.requeue_for_redisplay(pending);
        s.handle_key(key(KeyCode::Char('a'), KeyModifiers::NONE));
        // Queue drained; no further pending.
        assert!(s.peek_pending_permission().is_none());
    }

    #[test]
    fn render_shows_permission_modal_when_pending() {
        let s = state();
        let pending = PendingPrompt {
            id: PromptId(7),
            request: PermissionRequest::ToolUse {
                tool_id: "fs.write".into(),
            },
            issued_at: SystemTime::UNIX_EPOCH,
        };
        s.permission_prompter.requeue_for_redisplay(pending);
        let text = rendered_text(&s, 100, 14);
        assert!(text.contains("permission"));
        assert!(text.contains("fs.write"));
        assert!(text.contains("[a]"));
    }

    // ---------- streaming-progress status bar ----------

    #[test]
    fn status_bar_text_initially_empty() {
        let s = state();
        assert_eq!(s.status_bar_text(), "");
    }

    #[test]
    fn status_bar_text_after_submit_contains_tokens_and_tok_per_sec() {
        // ScriptedProvider returns one Block::Text of length 11 ("hello world")
        // so we can pin the approximate-token-count math.
        let sink: Arc<dyn EventSink> = Arc::new(MemoryEventSink::new());
        let events = Arc::new(EventEmitter::new(sink));
        let provider: Arc<dyn Provider> = Arc::new(ScriptedProvider {
            script: std::sync::Mutex::new(vec![
                Block::Text {
                    text: "hello world".into(),
                },
                Block::Done,
            ]),
        });
        let loop_ = build_loop(provider, events);
        let mut s = ChatState::with_agent_loop(loop_);
        s.submit_with_prompt("go");
        let bar = s.status_bar_text();
        assert!(bar.contains("tokens"), "got: {bar}");
        assert!(bar.contains("tok/s"), "got: {bar}");
    }

    #[test]
    fn status_bar_last_token_count_uses_four_chars_per_token() {
        // 11 chars / 4 = 2 (integer division). Pin a small bound to allow for
        // a future tweak of the heuristic without rewriting this test.
        let sink: Arc<dyn EventSink> = Arc::new(MemoryEventSink::new());
        let events = Arc::new(EventEmitter::new(sink));
        let provider: Arc<dyn Provider> = Arc::new(ScriptedProvider {
            script: std::sync::Mutex::new(vec![Block::Text {
                text: "hello world".into(),
            }]),
        });
        let loop_ = build_loop(provider, events);
        let mut s = ChatState::with_agent_loop(loop_);
        s.submit_with_prompt("go");
        let count = s.last_token_count();
        assert!(
            (2..=3).contains(&count),
            "expected approximate count in [2, 3], got {count}",
        );
    }

    #[test]
    fn in_flight_since_cleared_after_submit_completes() {
        let mut s = state();
        s.submit_with_prompt("hi");
        assert!(s.in_flight_since.is_none());
    }

    #[test]
    fn status_bar_in_flight_shows_generating() {
        let mut s = state();
        // Pin a synthetic in-flight stamp 2 seconds in the past so the
        // formatted elapsed seconds are deterministic.
        s.in_flight_since = Instant::now().checked_sub(Duration::from_secs(2));
        assert!(s.in_flight_since.is_some());
        let bar = s.status_bar_text();
        assert!(bar.contains("generating"), "got: {bar}");
    }

    #[test]
    fn status_bar_in_flight_takes_precedence_over_last_metrics() {
        // First, run a turn so `last_metrics` is populated.
        let mut s = state();
        s.submit_with_prompt("hi");
        assert!(s.last_metrics().is_some());
        // Now mark a fresh turn as in-flight — the in-flight indicator
        // must win over the completed-turn summary.
        s.in_flight_since = Some(Instant::now());
        let bar = s.status_bar_text();
        assert!(bar.contains("generating"), "got: {bar}");
        assert!(!bar.contains("tok/s"), "got: {bar}");
    }

    #[test]
    fn status_bar_multiple_submits_replace_last_metrics() {
        let mut s = state();
        s.submit_with_prompt("hi");
        let first = s.last_metrics().expect("first metrics").turn_id;
        s.submit_with_prompt("bye");
        let second = s.last_metrics().expect("second metrics").turn_id;
        // The most recent submit replaces — does not append.
        assert_ne!(first, second);
        assert_eq!(second.0, first.0 + 1);
    }

    #[test]
    fn approximate_token_count_zero_for_empty_blocks() {
        assert_eq!(approximate_token_count(&[]), 0);
    }

    #[test]
    fn approximate_token_count_ignores_non_text_blocks() {
        // Usage / Done / ToolCall must contribute nothing.
        let blocks = vec![
            Block::Usage {
                prompt: 100,
                completion: 200,
            },
            Block::Done,
            Block::ToolCall {
                id: "t1".into(),
                tool: "fs.read".into(),
                args: "{}".into(),
            },
        ];
        assert_eq!(approximate_token_count(&blocks), 0);
    }

    #[test]
    fn approximate_token_count_sums_text_lengths() {
        // "abcd" + "efgh" = 8 chars / 4 = 2 tokens.
        let blocks = vec![
            Block::Text {
                text: "abcd".into(),
            },
            Block::Text {
                text: "efgh".into(),
            },
        ];
        assert_eq!(approximate_token_count(&blocks), 2);
    }

    #[test]
    fn render_shows_generating_indicator_when_in_flight() {
        let mut s = state();
        s.in_flight_since = Some(Instant::now());
        let text = rendered_text(&s, 100, 12);
        assert!(text.contains("generating"), "got render:\n{text}");
    }

    #[test]
    fn resumed_count_independent_of_submit() {
        // Submit a new turn after resume; resumed_count must NOT increase.
        let t = resume_fixture();
        let mut s = ChatState::default().with_resumed_transcript(t);
        let baseline = s.resumed_count();
        let pre_len = s.transcript().len();
        s.submit_with_prompt("hi");
        assert_eq!(s.resumed_count(), baseline);
        assert!(s.transcript().len() > pre_len);
    }

    // -- with_handoff -------------------------------------------------------

    /// Build a tiny [`AgentHandoff`] wrapping a registry that has a single
    /// `Default` role backed by [`AgentFactory::echo`].
    fn handoff_default_only() -> Arc<AgentHandoff> {
        use stratum_runtime::{AgentFactory, AgentRegistry, HandoffPolicy, SuggestedRole};
        let l = Arc::new(AgentFactory::echo().expect("echo factory builds"));
        let mut reg = AgentRegistry::new();
        reg.register(SuggestedRole::Default, l);
        Arc::new(AgentHandoff::new(
            reg,
            SuggestedRole::Default,
            HandoffPolicy::default(),
        ))
    }

    #[test]
    fn default_state_has_no_handoff() {
        let s = state();
        assert!(!s.has_handoff());
    }

    #[test]
    fn with_handoff_installs_coordinator() {
        let s = state().with_handoff(handoff_default_only());
        assert!(s.has_handoff());
    }

    #[test]
    fn submit_with_handoff_records_user_and_assistant_turns() {
        let mut s = state().with_handoff(handoff_default_only());
        s.submit_with_prompt("hi");
        // At minimum we expect a User then an Assistant entry.
        assert!(matches!(s.transcript()[0], Turn::User(_)));
        assert!(matches!(s.transcript()[1], Turn::Assistant(_)));
    }

    #[test]
    fn submit_without_handoff_still_uses_single_loop_path() {
        // Regression: omitting `with_handoff` preserves the Phase 1 behaviour
        // — `submit` routes through `agent_loop.run_turn` and records a
        // `User` + `Assistant` pair with no command lines.
        let mut s = state();
        s.submit_with_prompt("hi");
        assert_eq!(s.transcript().len(), 2);
        assert!(matches!(s.transcript()[0], Turn::User(_)));
        assert!(matches!(s.transcript()[1], Turn::Assistant(_)));
    }

    // -- current_role + /agents palette ------------------------------------

    /// Registry with Default + Coder, both backed by the echo factory.
    fn handoff_default_and_coder() -> Arc<AgentHandoff> {
        use stratum_runtime::{AgentFactory, AgentRegistry, HandoffPolicy, SuggestedRole};
        let l1 = Arc::new(AgentFactory::echo().expect("echo factory builds"));
        let l2 = Arc::new(AgentFactory::echo().expect("echo factory builds"));
        let mut reg = AgentRegistry::new();
        reg.register(SuggestedRole::Default, l1);
        reg.register(SuggestedRole::Coder, l2);
        Arc::new(AgentHandoff::new(
            reg,
            SuggestedRole::Default,
            HandoffPolicy::default(),
        ))
    }

    #[test]
    fn current_role_label_empty_in_single_loop_mode() {
        let s = state();
        assert!(s.current_role_label().is_empty());
    }

    #[test]
    fn current_role_label_seeds_default_after_with_handoff() {
        let s = state().with_handoff(handoff_default_and_coder());
        assert_eq!(s.current_role_label(), "agent: default");
    }

    #[test]
    fn current_role_label_updates_after_coder_routed_submit() {
        // The default IntentRouter rules route a prompt containing
        // "stack trace" to `SuggestedRole::Coder`. The registry above has
        // Coder registered, so the handoff lands there on the first hop
        // and `current_role` flips to `Coder`.
        let mut s = state().with_handoff(handoff_default_and_coder());
        s.submit_with_prompt("debug this stack trace");
        assert_eq!(s.current_role_label(), "agent: coder");
    }

    #[test]
    fn agents_command_without_handoff_is_rejected_with_hint() {
        let mut s = state();
        let outcome = s.execute_palette_command("/agents");
        let PaletteOutcome::Rejected { message } = outcome else {
            panic!("expected rejected");
        };
        assert!(message.contains("no multi-agent"), "got: {message}");
        assert!(message.contains("--agents-dir"), "got: {message}");
    }

    #[test]
    fn agents_command_with_handoff_lists_roles_and_current() {
        let mut s = state().with_handoff(handoff_default_and_coder());
        let outcome = s.execute_palette_command("/agents");
        let PaletteOutcome::Acknowledged { message } = outcome else {
            panic!("expected acknowledged");
        };
        assert!(message.starts_with("roles:"), "got: {message}");
        assert!(message.contains("(current:"), "got: {message}");
        assert!(message.contains("default"), "got: {message}");
        assert!(message.contains("coder"), "got: {message}");
    }

    #[test]
    fn agents_command_in_single_loop_mode_does_not_panic() {
        // Regression: even when /agents is rejected (no handoff installed),
        // the dispatch path must complete cleanly and append a rejected
        // command turn to the transcript.
        let mut s = state();
        let pre_len = s.transcript().len();
        let _outcome = s.execute_palette_command("/agents");
        let last = s.transcript().last().expect("command turn appended");
        match last {
            Turn::Command { text, ok, .. } => {
                assert_eq!(text, "/agents");
                assert!(!*ok);
            }
            other => panic!("expected Turn::Command, got {other:?}"),
        }
        assert_eq!(s.transcript().len(), pre_len + 1);
    }

    #[test]
    fn status_bar_render_includes_role_in_multi_agent_mode() {
        let s = state().with_handoff(handoff_default_and_coder());
        let text = rendered_text(&s, 100, 12);
        assert!(text.contains("agent:"), "render:\n{text}");
        assert!(text.contains("default"), "render:\n{text}");
    }

    #[test]
    fn status_bar_render_omits_role_in_single_loop_mode() {
        let s = state();
        let text = rendered_text(&s, 100, 12);
        assert!(!text.contains("agent:"), "render:\n{text}");
    }

    #[test]
    fn help_text_lists_agents_command() {
        // HELP_TEXT is what /help echoes; it must enumerate every wired
        // palette command, including /agents.
        assert!(HELP_TEXT.contains("/agents"), "got: {HELP_TEXT}");
    }

    #[test]
    fn role_name_covers_every_suggested_role() {
        assert_eq!(role_name(SuggestedRole::Default), "default");
        assert_eq!(role_name(SuggestedRole::Cavemanish), "cavemanish");
        assert_eq!(role_name(SuggestedRole::Polisher), "polisher");
        assert_eq!(role_name(SuggestedRole::Coder), "coder");
        assert_eq!(role_name(SuggestedRole::Researcher), "researcher");
    }

    // -- /parallel palette --------------------------------------------------

    /// Registry with Cavemanish + Coder, both backed by the echo factory.
    /// Used by the `/parallel` palette tests so the dispatcher has at
    /// least two distinct roles to fan a turn out to.
    fn handoff_cavemanish_and_coder() -> Arc<AgentHandoff> {
        use stratum_runtime::{AgentFactory, AgentRegistry, HandoffPolicy, SuggestedRole};
        let l1 = Arc::new(AgentFactory::echo().expect("echo factory builds"));
        let l2 = Arc::new(AgentFactory::echo().expect("echo factory builds"));
        let mut reg = AgentRegistry::new();
        reg.register(SuggestedRole::Cavemanish, l1);
        reg.register(SuggestedRole::Coder, l2);
        Arc::new(AgentHandoff::new(
            reg,
            SuggestedRole::Default,
            HandoffPolicy::default(),
        ))
    }

    #[test]
    fn parallel_command_without_handoff_is_rejected_with_hint() {
        let mut s = state();
        let outcome = s.execute_palette_command("/parallel cavemanish,coder");
        let PaletteOutcome::Rejected { message } = outcome else {
            panic!("expected rejected, got: {outcome:?}");
        };
        assert!(message.contains("no multi-agent"), "got: {message}");
        assert!(message.contains("--agents-dir"), "got: {message}");
    }

    #[test]
    fn parallel_command_with_handoff_appends_assistant_turn() {
        let mut s = state().with_handoff(handoff_cavemanish_and_coder());
        let outcome = s.execute_palette_command("/parallel cavemanish,coder");
        let PaletteOutcome::Acknowledged { message } = outcome else {
            panic!("expected acknowledged, got: {outcome:?}");
        };
        assert!(message.starts_with("parallel:"), "got: {message}");
        // Transcript should now contain an Assistant turn whose concatenated
        // text mentions both role names (each role gets its own section
        // header).
        let combined = s
            .transcript()
            .iter()
            .filter_map(|t| match t {
                Turn::Assistant(blocks) => Some(concat_text_blocks(blocks)),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            combined.contains("cavemanish") && combined.contains("coder"),
            "expected both role headers in assistant turn; got: {combined:?}"
        );
    }

    #[test]
    fn parallel_command_unknown_role_is_rejected() {
        let mut s = state().with_handoff(handoff_cavemanish_and_coder());
        let outcome = s.execute_palette_command("/parallel unknown-role");
        let PaletteOutcome::Rejected { message } = outcome else {
            panic!("expected rejected, got: {outcome:?}");
        };
        assert!(message.contains("unknown role"), "got: {message}");
    }

    #[test]
    fn parallel_command_empty_args_is_rejected() {
        let mut s = state().with_handoff(handoff_cavemanish_and_coder());
        let outcome = s.execute_palette_command("/parallel");
        let PaletteOutcome::Rejected { message } = outcome else {
            panic!("expected rejected, got: {outcome:?}");
        };
        assert!(message.contains("unknown role"), "got: {message}");
    }

    #[test]
    fn help_text_lists_parallel_command() {
        assert!(HELP_TEXT.contains("/parallel"), "got: {HELP_TEXT}");
    }

    #[test]
    fn parse_role_label_covers_every_suggested_role() {
        assert_eq!(parse_role_label("default"), Some(SuggestedRole::Default));
        assert_eq!(
            parse_role_label("cavemanish"),
            Some(SuggestedRole::Cavemanish)
        );
        assert_eq!(parse_role_label("polisher"), Some(SuggestedRole::Polisher));
        assert_eq!(parse_role_label("coder"), Some(SuggestedRole::Coder));
        assert_eq!(
            parse_role_label("researcher"),
            Some(SuggestedRole::Researcher)
        );
        assert_eq!(parse_role_label("unknown"), None);
        assert_eq!(parse_role_label(""), None);
    }
}
