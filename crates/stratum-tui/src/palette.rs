//! Slash-command palette state machine.
//!
//! Phase 2 v2 minimal: a substring-filter palette opened on `/`, listing
//! every registered command, with `Up`/`Down` to move the highlight,
//! `Enter` to invoke, `Esc` to close. Fuzzy matching via `nucleo` and
//! sticky recents land in Phase 3 (CI invariant + arg sub-palette also
//! Phase 3) per `plan/13-approaches-considered.md` A17.

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

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

/// One registered slash command. Static for now; user-defined commands
/// (from agents and tool registration) hook in in later phases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Command {
    /// The command keyword (without the leading `/`).
    pub name: &'static str,
    /// One-line description shown in the palette.
    pub help: &'static str,
}

/// Static command catalog. Sorted by name for stable rendering.
/// MUST mirror every match arm in `chat.rs::ChatState::dispatch_command`.
pub const COMMANDS: &[Command] = &[
    Command { name: "active", help: "Show currently active model." },
    Command { name: "agent", help: "Delegate next turn to a subagent. Usage: /agent <name> <task>." },
    Command { name: "agents", help: "List registered roles (multi-agent mode only)." },
    Command { name: "budget", help: "Show last turn metrics (tokens · ms · tok/s · turn id)." },
    Command { name: "cancel", help: "Cancel the in-flight turn." },
    Command { name: "clear", help: "Clear the chat transcript." },
    Command { name: "compact", help: "Compress older turns into a summary; keeps the last 4 verbatim." },
    Command { name: "cost", help: "Show the latest turn metrics (alias for /budget)." },
    Command { name: "diff", help: "Show recent fs.write / fs.edit calls." },
    Command { name: "editor", help: "Open the current input in $VISUAL / $EDITOR (Ctrl+G shortcut)." },
    Command { name: "exit", help: "Exit the TUI." },
    Command { name: "export", help: "Dump the chat transcript to a file. Usage: /export [path]." },
    Command { name: "help", help: "Show available commands." },
    Command { name: "image", help: "Attach an image to the next turn. Usage: /image <path>." },
    Command { name: "init", help: "Scaffold STRATUM.md for the current workspace." },
    Command { name: "model", help: "Show active model, or alias for /switch <slug>." },
    Command { name: "models", help: "List available models from the catalog." },
    Command { name: "parallel", help: "Fan next turn across roles. Usage: /parallel <role1,role2,…>." },
    Command { name: "plan", help: "Toggle (or set) plan mode. Usage: /plan [on|off]." },
    Command { name: "quit", help: "Exit the TUI." },
    Command { name: "recap", help: "One-line session summary." },
    Command { name: "select", help: "Toggle mouse-capture (Ctrl+T): off = drag to select text + copy; on = scroll-wheel + mouse events." },
    Command { name: "redo", help: "Redo (placeholder — Phase 3 v2)." },
    Command { name: "subagents", help: "List available subagents (built-in + user-defined)." },
    Command { name: "switch", help: "Swap to a different model mid-session. Usage: /switch <slug>." },
    Command { name: "theme", help: "Switch chat theme. Usage: /theme <name>. See /themes for choices." },
    Command { name: "themes", help: "List available themes (built-in + user JSON)." },
    Command { name: "tier", help: "Show current host tier (low/medium/high)." },
    Command { name: "undo", help: "Undo (placeholder — Phase 3 v2)." },
    Command { name: "usage", help: "Show the latest turn metrics (alias for /budget)." },
    Command { name: "version", help: "Show stratum version." },
    Command { name: "welcome", help: "Re-show the Stratum greeting + a tip." },
];

/// What a key event resolves to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Action {
    /// Stay in palette mode; nothing emitted.
    None,
    /// Close the palette without executing anything.
    Close,
    /// Execute the named command.
    Execute(String),
}

/// Substring-filtered command palette.
#[derive(Debug, Clone)]
pub struct Palette {
    filter: String,
    cursor: usize,
}

impl Default for Palette {
    fn default() -> Self {
        Self::new()
    }
}

impl Palette {
    /// Fresh palette: empty filter, cursor at the first match.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            filter: String::new(),
            cursor: 0,
        }
    }

    /// Current filter text.
    #[must_use]
    pub fn filter(&self) -> &str {
        &self.filter
    }

    /// Index of the highlighted entry in `matches()`.
    #[must_use]
    pub const fn cursor(&self) -> usize {
        self.cursor
    }

    /// Substring-filter the catalog and return the matching commands.
    #[must_use]
    pub fn matches(&self) -> Vec<&'static Command> {
        let needle = self.filter.to_ascii_lowercase();
        COMMANDS
            .iter()
            .filter(|c| c.name.to_ascii_lowercase().contains(&needle))
            .collect()
    }

    /// Apply a key event. Returns the next [`Action`] for the caller.
    pub fn handle_key(&mut self, key: KeyEvent) -> Action {
        // Ctrl-C inside the palette is treated as Close, not as a global
        // cancel; the caller handles the outer cancel token.
        if key.modifiers.contains(KeyModifiers::CONTROL)
            && matches!(key.code, KeyCode::Char('c' | 'C'))
        {
            return Action::Close;
        }
        match key.code {
            KeyCode::Esc => Action::Close,
            KeyCode::Enter => {
                let matches = self.matches();
                matches
                    .get(self.cursor)
                    .map_or(Action::Close, |c| Action::Execute(c.name.to_string()))
            }
            KeyCode::Up => {
                self.cursor = self.cursor.saturating_sub(1);
                Action::None
            }
            KeyCode::Down => {
                let len = self.matches().len();
                if self.cursor + 1 < len {
                    self.cursor += 1;
                }
                Action::None
            }
            KeyCode::Backspace => {
                self.filter.pop();
                self.cursor = 0;
                Action::None
            }
            KeyCode::Char(c) => {
                self.filter.push(c);
                self.cursor = 0;
                Action::None
            }
            _ => Action::None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    fn key_ctrl(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::CONTROL)
    }

    #[test]
    fn empty_filter_lists_every_command() {
        let p = Palette::new();
        assert_eq!(p.matches().len(), COMMANDS.len());
    }

    #[test]
    fn substring_filter_narrows() {
        let mut p = Palette::new();
        p.handle_key(key(KeyCode::Char('m')));
        let names: Vec<&str> = p.matches().iter().map(|c| c.name).collect();
        assert!(names.contains(&"model"));
        assert!(names.contains(&"models"));
        assert!(!names.contains(&"quit"));
    }

    #[test]
    fn case_insensitive_filter() {
        let mut p = Palette::new();
        p.handle_key(key(KeyCode::Char('M')));
        assert!(p.matches().iter().any(|c| c.name == "model"));
    }

    #[test]
    fn backspace_pops_filter() {
        let mut p = Palette::new();
        p.handle_key(key(KeyCode::Char('q')));
        p.handle_key(key(KeyCode::Char('u')));
        p.handle_key(key(KeyCode::Backspace));
        assert_eq!(p.filter(), "q");
    }

    #[test]
    fn typing_resets_cursor_to_zero() {
        let mut p = Palette::new();
        p.handle_key(key(KeyCode::Down));
        p.handle_key(key(KeyCode::Down));
        p.handle_key(key(KeyCode::Char('m')));
        assert_eq!(p.cursor(), 0);
    }

    #[test]
    fn down_moves_cursor_within_bounds() {
        let mut p = Palette::new();
        p.handle_key(key(KeyCode::Down));
        assert_eq!(p.cursor(), 1);
        // Press past the end — cursor saturates.
        for _ in 0..100 {
            p.handle_key(key(KeyCode::Down));
        }
        assert_eq!(p.cursor(), COMMANDS.len() - 1);
    }

    #[test]
    fn up_saturates_at_zero() {
        let mut p = Palette::new();
        p.handle_key(key(KeyCode::Up));
        assert_eq!(p.cursor(), 0);
    }

    #[test]
    fn esc_closes() {
        let mut p = Palette::new();
        assert_eq!(p.handle_key(key(KeyCode::Esc)), Action::Close);
    }

    #[test]
    fn ctrl_c_closes() {
        let mut p = Palette::new();
        assert_eq!(p.handle_key(key_ctrl(KeyCode::Char('c'))), Action::Close);
    }

    #[test]
    fn enter_on_match_executes() {
        let mut p = Palette::new();
        // First entry alphabetically is "active".
        assert_eq!(
            p.handle_key(key(KeyCode::Enter)),
            Action::Execute("active".to_string())
        );
    }

    #[test]
    fn enter_on_no_match_closes() {
        let mut p = Palette::new();
        for c in "zzzzz".chars() {
            p.handle_key(key(KeyCode::Char(c)));
        }
        assert_eq!(p.handle_key(key(KeyCode::Enter)), Action::Close);
    }

    #[test]
    fn unhandled_key_returns_none() {
        let mut p = Palette::new();
        assert_eq!(p.handle_key(key(KeyCode::F(5))), Action::None);
    }

    #[test]
    fn palette_clone_is_independent() {
        let mut a = Palette::new();
        a.handle_key(key(KeyCode::Char('m')));
        let b = a.clone();
        assert_eq!(a.filter(), b.filter());
    }

    #[test]
    fn default_matches_new() {
        let a = Palette::default();
        let b = Palette::new();
        assert_eq!(a.filter(), b.filter());
        assert_eq!(a.cursor(), b.cursor());
    }

    #[test]
    fn command_struct_renders_via_debug() {
        let c = Command {
            name: "x",
            help: "y",
        };
        assert!(format!("{c:?}").contains("Command"));
    }

    #[test]
    fn action_equality() {
        assert_eq!(Action::None, Action::None);
        assert_ne!(Action::Close, Action::None);
        assert_eq!(
            Action::Execute("a".to_string()),
            Action::Execute("a".to_string())
        );
    }
}
