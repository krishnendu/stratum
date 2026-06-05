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

use std::io;
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
use stratum_runtime::{CancelToken, EchoProvider, GenerateRequest, Paths, Provider, Tier};
use stratum_types::{Block, ModelId, StratumResult};

/// One entry in the chat transcript.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Turn {
    /// What the user typed.
    User(String),
    /// What the provider returned.
    Assistant(Vec<Block>),
    /// Cancellation marker (Ctrl-C mid-stream).
    Cancelled,
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
}

impl ChatState {
    /// Build a fresh state with the given header (status bar) and tier.
    #[must_use]
    pub fn new(provider: EchoProvider, tier: Tier, status: String) -> Self {
        Self {
            transcript: Vec::new(),
            input: String::new(),
            provider,
            cancel: CancelToken::new(),
            tier,
            quit: false,
            status,
        }
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
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        match key.code {
            KeyCode::Esc => self.quit = true,
            KeyCode::Char('c' | 'C') if ctrl => {
                self.cancel.cancel();
                self.transcript.push(Turn::Cancelled);
                self.quit = true;
            }
            KeyCode::Char(c) => self.input.push(c),
            KeyCode::Backspace => {
                self.input.pop();
            }
            KeyCode::Enter => self.submit(),
            _ => {}
        }
    }

    /// Submit the current input to the provider and append the result.
    pub fn submit(&mut self) {
        if self.input.trim().is_empty() {
            return;
        }
        let prompt = std::mem::take(&mut self.input);
        let request = GenerateRequest {
            model: ModelId::from("echo"),
            prompt: prompt.clone(),
            max_blocks: 64,
        };
        let blocks = self.provider.generate(&request, &self.cancel);
        self.transcript.push(Turn::User(prompt));
        self.transcript.push(Turn::Assistant(blocks));
    }

    /// Render the entire TUI into the given frame.
    pub fn render(&self, area: Rect, buf: &mut ratatui::buffer::Buffer) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1),
                Constraint::Min(3),
                Constraint::Length(3),
            ])
            .split(area);

        let status = Paragraph::new(Line::from(vec![
            Span::styled("stratum", Style::default().add_modifier(Modifier::BOLD)),
            Span::raw(" · "),
            Span::raw(format!("tier={}", self.tier)),
            Span::raw(" · "),
            Span::raw(&self.status),
            Span::raw(" · "),
            Span::raw("Esc/Ctrl-C exit"),
        ]));
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
            }
        }
        let chat = Paragraph::new(lines)
            .block(TuiBlock::default().borders(Borders::ALL).title("chat"))
            .wrap(Wrap { trim: false });
        ratatui::widgets::Widget::render(chat, chunks[1], buf);

        let input = Paragraph::new(Line::from(vec![Span::raw("> "), Span::raw(&self.input)]))
            .block(TuiBlock::default().borders(Borders::ALL).title("input"));
        ratatui::widgets::Widget::render(input, chunks[2], buf);
    }
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
    use crossterm::event::KeyCode;
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;

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
}
