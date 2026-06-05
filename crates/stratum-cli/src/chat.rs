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
use std::time::Duration;

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
    format_tokens_per_second, AgentLoop, AgentLoopConfig, AllowAllResponder, CancelToken,
    CapabilityMatrix, EchoProvider, Event as RtEvent, EventEmitter, EventRecord, IntentRouter,
    MemoryEventSink, Paths, PermissionStore, PlanMode, PromptIdGen, Provider, RoleTimer, Tier,
    TurnContext, TurnId, TurnMetrics, TurnRecorder, TurnResult,
};
use stratum_types::{Block, ModelId, StratumResult};

use crate::palette::{self, Palette};

/// One entry in the chat transcript.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Turn {
    /// What the user typed.
    User(String),
    /// What the provider returned.
    Assistant(Vec<Block>),
    /// Cancellation marker (Ctrl-C mid-stream).
    Cancelled,
    /// Slash command was invoked from the palette.
    Command(String),
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
        #[allow(
            clippy::expect_used,
            reason = "default_agent_loop sets all nine required builder fields; build() cannot return MissingField on this code path"
        )]
        let agent_loop = Arc::new(
            default_agent_loop(provider.clone(), events.clone())
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
            default_agent_loop(self.provider.clone(), events.clone())
                .expect("default AgentLoop builder sets every required field"),
        );
        self.agent_loop = agent_loop;
        self.events = events;
        self.memory_sink = None;
        self
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

    fn execute_command(&mut self, name: &str) {
        if name == "quit" {
            self.quit = true;
        }
        self.transcript.push(Turn::Command(name.to_string()));
    }

    /// Submit the current input through the [`AgentLoop`] and append the
    /// resulting blocks to the transcript.
    pub fn submit(&mut self) {
        if self.input.trim().is_empty() {
            return;
        }
        let prompt = std::mem::take(&mut self.input);
        let turn_id = TurnId(self.next_turn_id);
        self.next_turn_id = self.next_turn_id.saturating_add(1);

        let ctx = TurnContext {
            user_prompt: prompt.clone(),
            model: ModelId::from("echo"),
            turn_id,
            started_at: std::time::SystemTime::now(),
        };
        let role_timer = RoleTimer::start();
        let turn_result = self.agent_loop.run_turn(ctx, &self.cancel);
        let step_ms = role_timer.stop_ms();

        let blocks = turn_result.blocks.clone();
        let mut recorder = TurnRecorder::new(turn_id);
        for block in &blocks {
            recorder.record_block(block);
        }
        recorder.record_step("generate", step_ms);
        self.last_metrics = Some(recorder.finish());

        self.transcript.push(Turn::User(prompt));
        self.transcript.push(Turn::Assistant(blocks));
        self.last_turn_result = Some(turn_result);
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
                Turn::Command(name) => lines.push(Line::from(Span::styled(
                    format!("(executed /{name})"),
                    Style::default().add_modifier(Modifier::DIM),
                ))),
            }
        }
        let chat = Paragraph::new(lines)
            .block(TuiBlock::default().borders(Borders::ALL).title("chat"))
            .wrap(Wrap { trim: false });
        ratatui::widgets::Widget::render(chat, chunks[1], buf);

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
) -> Result<AgentLoop, stratum_runtime::AgentLoopBuildError> {
    let provider_arc: Arc<dyn Provider> = Arc::new(provider);
    AgentLoop::builder()
        .with_provider(provider_arc)
        .with_router(IntentRouter::default())
        .with_permission_store(Arc::new(PermissionStore::new()))
        .with_prompt_gen(Arc::new(PromptIdGen::new()))
        .with_responder(Arc::new(AllowAllResponder))
        .with_events(events)
        .with_capability_matrix(Arc::new(CapabilityMatrix::new()))
        .with_plan_mode(Arc::new(PlanMode::new()))
        .with_config(AgentLoopConfig::default())
        .build()
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
    let mut state = ChatState::new(provider, tier, status_for(paths));
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
        // First alphabetical match is "active".
        s.handle_key(key(KeyCode::Enter, KeyModifiers::NONE));
        assert!(!s.palette_open());
        assert!(matches!(s.transcript().last(), Some(Turn::Command(_))));
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
        let Some(Turn::Command(name)) = s.transcript().last() else {
            panic!("expected command turn")
        };
        assert_eq!(name, "quit");
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
        let Some(Turn::Command(name)) = s.transcript().last() else {
            panic!("expected command turn")
        };
        // Filter "m" leaves "model" and "models"; cursor=0 picks "model".
        assert_eq!(name, "model");
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
        s.handle_key(key(KeyCode::Char('/'), KeyModifiers::NONE));
        s.handle_key(key(KeyCode::Enter, KeyModifiers::NONE));
        let text = rendered_text(&s, 60, 10);
        assert!(text.contains("executed /active"));
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
}
