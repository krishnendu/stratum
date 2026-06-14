//! Hooks runtime — minimal viable surface for the eight highest-value
//! events from `plan/42`. v1 supports the `command` handler type only;
//! `http` / `mcp_tool` / `prompt` / `agent` types are planned but not
//! built.
//!
//! Event coverage in v1:
//!
//! - [`Event::SessionStart`] — inject org/workspace context into the
//!   system prompt
//! - [`Event::UserPromptSubmit`] — transform / reject the user's prompt
//! - [`Event::PreToolUse`] — veto a tool call (`exit != 0`)
//! - [`Event::PostToolUse`] — formatter / observer on success
//! - [`Event::FileChanged`] — fires when fs.edit / fs.write writes
//! - [`Event::PreCompact`] / [`Event::PostCompact`] — compaction
//! - [`Event::Stop`] — end-of-turn observer
//!
//! Other event types from `plan/42 §2` land in Phase 4 v2 — they
//! plug into the same dispatcher.

use std::collections::BTreeMap;
use std::process::{Command, Stdio};
use std::time::Duration;

use serde::{Deserialize, Serialize};

/// The eight events the v1 dispatcher knows about.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Event {
    /// TUI launched, prior session replayed.
    SessionStart,
    /// User pressed Enter; prompt about to go to provider.
    UserPromptSubmit,
    /// Tool call about to be dispatched (post-permission, pre-exec).
    PreToolUse,
    /// Tool call returned successfully.
    PostToolUse,
    /// fs.edit / fs.write succeeded + the OS file watch confirms write.
    FileChanged,
    /// `/compact` about to run.
    PreCompact,
    /// `/compact` completed.
    PostCompact,
    /// End of turn (success / cancel / error).
    Stop,
}

impl Event {
    /// Stable string label written to the hook command's stdin payload
    /// and to the per-event matcher table.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::SessionStart => "SessionStart",
            Self::UserPromptSubmit => "UserPromptSubmit",
            Self::PreToolUse => "PreToolUse",
            Self::PostToolUse => "PostToolUse",
            Self::FileChanged => "FileChanged",
            Self::PreCompact => "PreCompact",
            Self::PostCompact => "PostCompact",
            Self::Stop => "Stop",
        }
    }

    /// Parse a hook event from a settings-file string. Case-sensitive
    /// to match Claude Code's convention.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "SessionStart" => Self::SessionStart,
            "UserPromptSubmit" => Self::UserPromptSubmit,
            "PreToolUse" => Self::PreToolUse,
            "PostToolUse" => Self::PostToolUse,
            "FileChanged" => Self::FileChanged,
            "PreCompact" => Self::PreCompact,
            "PostCompact" => Self::PostCompact,
            "Stop" => Self::Stop,
            _ => return None,
        })
    }
}

/// One hook config row from `settings.json hooks`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HookConfig {
    /// Event the hook subscribes to.
    pub event: String,
    /// Optional matcher — e.g. `fs.edit(*.rs)`, `shell.exec(rm *)`.
    /// `None` matches all invocations of the event.
    #[serde(default)]
    pub matcher: Option<String>,
    /// Handler type. v1 supports `"command"` only.
    #[serde(rename = "type")]
    pub kind: String,
    /// For `type = "command"`: the shell command to run. Receives the
    /// event payload as JSON on stdin.
    #[serde(default)]
    pub command: Option<String>,
    /// Per-hook timeout in milliseconds (default 2000).
    #[serde(default)]
    pub timeout_ms: Option<u64>,
}

/// What the dispatcher returns for a fired event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    /// All hooks (or no hooks at all) allowed the action.
    Allow {
        /// Transformed payload if any hook mutated stdout — caller may
        /// substitute the prompt / args body.
        transform: Option<String>,
    },
    /// Some hook vetoed (exit code != 0).
    Veto {
        /// Reason string (hook's stderr or a synthetic message).
        reason: String,
    },
    /// Some hook timed out.
    Timeout {
        /// The hook that timed out (label form).
        hook: String,
    },
}

/// Hook dispatcher. Holds the parsed hook table; `fire` looks up the
/// event, runs every matching hook in source order, and aggregates
/// the outcomes.
#[derive(Debug, Clone, Default)]
pub struct HookDispatcher {
    by_event: BTreeMap<String, Vec<HookConfig>>,
}

impl HookDispatcher {
    /// Build from a flat `settings.json hooks` list. Entries whose
    /// `event` doesn't parse are silently dropped (the loader logs).
    #[must_use]
    pub fn from_configs(configs: Vec<HookConfig>) -> Self {
        let mut by_event: BTreeMap<String, Vec<HookConfig>> = BTreeMap::new();
        for c in configs {
            if Event::parse(&c.event).is_some() {
                by_event.entry(c.event.clone()).or_default().push(c);
            }
        }
        Self { by_event }
    }

    /// True when at least one hook is registered for `event`.
    #[must_use]
    pub fn has_hooks(&self, event: Event) -> bool {
        self.by_event
            .get(event.label())
            .is_some_and(|v| !v.is_empty())
    }

    /// Fire all matching hooks for `event` with the given payload.
    /// `matcher_target` is the matcher input (e.g. tool id + first
    /// arg) used when comparing against a hook's `matcher` glob.
    #[must_use]
    pub fn fire(&self, event: Event, matcher_target: &str, payload_json: &str) -> Outcome {
        let Some(hooks) = self.by_event.get(event.label()) else {
            return Outcome::Allow { transform: None };
        };
        let mut last_transform: Option<String> = None;
        for hook in hooks {
            if !matcher_matches(hook.matcher.as_deref(), matcher_target) {
                continue;
            }
            if hook.kind != "command" {
                // Other handler types not built yet — silently allow
                // so callers don't break when a v2 hook is configured.
                continue;
            }
            let Some(cmd) = hook.command.as_deref() else {
                continue;
            };
            let timeout = Duration::from_millis(hook.timeout_ms.unwrap_or(2000));
            match run_command_hook(cmd, payload_json, timeout) {
                CommandResult::Allow { stdout } => {
                    if !stdout.is_empty() {
                        last_transform = Some(stdout);
                    }
                }
                CommandResult::Veto { stderr } => {
                    return Outcome::Veto {
                        reason: if stderr.is_empty() {
                            format!("hook command refused: {cmd}")
                        } else {
                            stderr
                        },
                    };
                }
                CommandResult::Timeout => {
                    return Outcome::Timeout {
                        hook: event.label().to_string(),
                    };
                }
            }
        }
        Outcome::Allow {
            transform: last_transform,
        }
    }
}

enum CommandResult {
    Allow { stdout: String },
    Veto { stderr: String },
    Timeout,
}

fn run_command_hook(cmd: &str, payload: &str, timeout: Duration) -> CommandResult {
    use std::io::Write;
    let Ok(mut child) = Command::new("sh")
        .arg("-c")
        .arg(cmd)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    else {
        // Spawn failure — treat as observe (allow) rather than veto,
        // matching Claude Code's "broken hook can't break the
        // session" stance. Logger upstream should surface this.
        return CommandResult::Allow {
            stdout: String::new(),
        };
    };
    if let Some(stdin) = child.stdin.take() {
        let mut stdin = stdin;
        let _ = stdin.write_all(payload.as_bytes());
    }
    let start = std::time::Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let mut stdout = String::new();
                let mut stderr = String::new();
                if let Some(mut out) = child.stdout.take() {
                    use std::io::Read;
                    let _ = out.read_to_string(&mut stdout);
                }
                if let Some(mut err) = child.stderr.take() {
                    use std::io::Read;
                    let _ = err.read_to_string(&mut stderr);
                }
                if status.success() {
                    return CommandResult::Allow {
                        stdout: stdout.trim_end().to_string(),
                    };
                }
                return CommandResult::Veto {
                    stderr: stderr.trim_end().to_string(),
                };
            }
            Ok(None) => {
                if start.elapsed() >= timeout {
                    let _ = child.kill();
                    return CommandResult::Timeout;
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(_) => {
                return CommandResult::Allow {
                    stdout: String::new(),
                };
            }
        }
    }
}

/// Cheap glob/exact matcher reused by the dispatcher. Returns true
/// for an absent matcher (= match all).
fn matcher_matches(matcher: Option<&str>, target: &str) -> bool {
    let Some(m) = matcher else { return true };
    if m == "*" || m == "**" {
        return true;
    }
    // Try the permission_rules glob first (reuses fs.write(*.rs) syntax).
    if let Some(rule) = crate::permission_rules::Rule::parse(m) {
        // Match on the full target string treated as a single arg.
        return rule.matches(target.split('(').next().unwrap_or(target), &[target]);
    }
    m == target
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn event_label_parse_round_trip() {
        for ev in [
            Event::SessionStart,
            Event::UserPromptSubmit,
            Event::PreToolUse,
            Event::PostToolUse,
            Event::FileChanged,
            Event::PreCompact,
            Event::PostCompact,
            Event::Stop,
        ] {
            let back = Event::parse(ev.label()).unwrap();
            assert_eq!(back, ev);
        }
    }

    #[test]
    fn dispatcher_with_no_hooks_allows_everything() {
        let d = HookDispatcher::default();
        assert!(matches!(
            d.fire(Event::PreToolUse, "fs.read", "{}"),
            Outcome::Allow { transform: None }
        ));
    }

    #[test]
    fn matching_command_hook_that_succeeds_allows() {
        let d = HookDispatcher::from_configs(vec![HookConfig {
            event: "PreToolUse".to_string(),
            matcher: None,
            kind: "command".to_string(),
            command: Some("true".to_string()),
            timeout_ms: Some(500),
        }]);
        let outcome = d.fire(Event::PreToolUse, "fs.read", "{}");
        assert!(matches!(outcome, Outcome::Allow { .. }));
    }

    #[test]
    fn matching_command_hook_that_fails_vetoes() {
        let d = HookDispatcher::from_configs(vec![HookConfig {
            event: "PreToolUse".to_string(),
            matcher: None,
            kind: "command".to_string(),
            command: Some("false".to_string()),
            timeout_ms: Some(500),
        }]);
        let outcome = d.fire(Event::PreToolUse, "fs.read", "{}");
        assert!(matches!(outcome, Outcome::Veto { .. }));
    }

    #[test]
    fn timeout_returns_timeout_outcome() {
        let d = HookDispatcher::from_configs(vec![HookConfig {
            event: "PreToolUse".to_string(),
            matcher: None,
            kind: "command".to_string(),
            command: Some("sleep 5".to_string()),
            timeout_ms: Some(100),
        }]);
        let outcome = d.fire(Event::PreToolUse, "fs.read", "{}");
        assert!(matches!(outcome, Outcome::Timeout { .. }));
    }

    #[test]
    fn hook_stdout_becomes_transform_payload() {
        let d = HookDispatcher::from_configs(vec![HookConfig {
            event: "UserPromptSubmit".to_string(),
            matcher: None,
            kind: "command".to_string(),
            command: Some("echo 'rewritten prompt'".to_string()),
            timeout_ms: Some(500),
        }]);
        match d.fire(Event::UserPromptSubmit, "*", "original") {
            Outcome::Allow { transform: Some(t) } => {
                assert!(t.contains("rewritten prompt"));
            }
            other => panic!("expected Allow w/ transform; got {other:?}"),
        }
    }

    #[test]
    fn unknown_kind_silently_allows() {
        let d = HookDispatcher::from_configs(vec![HookConfig {
            event: "PreToolUse".to_string(),
            matcher: None,
            kind: "http".to_string(), // v2 type
            command: None,
            timeout_ms: None,
        }]);
        let outcome = d.fire(Event::PreToolUse, "fs.read", "{}");
        assert!(matches!(outcome, Outcome::Allow { .. }));
    }

    #[test]
    fn has_hooks_reports_correctly() {
        let d = HookDispatcher::from_configs(vec![HookConfig {
            event: "Stop".to_string(),
            matcher: None,
            kind: "command".to_string(),
            command: Some("true".to_string()),
            timeout_ms: None,
        }]);
        assert!(d.has_hooks(Event::Stop));
        assert!(!d.has_hooks(Event::PreToolUse));
    }

    #[test]
    fn matcher_star_matches_anything() {
        assert!(matcher_matches(Some("*"), "fs.read"));
        assert!(matcher_matches(None, "fs.read"));
    }

    #[test]
    fn unknown_event_label_drops_from_dispatcher() {
        let d = HookDispatcher::from_configs(vec![HookConfig {
            event: "NotARealEvent".to_string(),
            matcher: None,
            kind: "command".to_string(),
            command: Some("true".to_string()),
            timeout_ms: None,
        }]);
        // Nothing registered → has_hooks for real events is false.
        assert!(!d.has_hooks(Event::PreToolUse));
    }
}
