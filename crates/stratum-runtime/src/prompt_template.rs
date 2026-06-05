//! Structured prompt template + composition layer.
//!
//! See `plan/09-prompts-and-roles.md`. This is the data-plus-interpolation
//! shape that the runtime uses to build the final string passed to a
//! provider, conceptually:
//!
//! ```text
//! system_prompt + agent_header + user_turn
//! ```
//!
//! Nothing here invokes a model — the module only holds the template, the
//! per-turn context (vars + user turn + tool results), and the renderer that
//! substitutes a tiny `{ $name }` placeholder set into the three template
//! slots.
//!
//! ## Placeholder syntax
//!
//! - `{ $var }` — looked up in [`PromptContext::vars`]. Missing → error.
//! - `{ $user_turn }` — substitutes [`PromptContext::user_turn`].
//! - `{ $tool_results }` — substitutes a deterministic per-line rendering of
//!   [`PromptContext::tool_results`].
//!
//! Braces must come in matched `{ ... }` pairs; an unmatched `{` or a
//! placeholder that does not start with `$` is a [`PromptRenderError::BadTemplate`].
//!
//! ## Budgets
//!
//! [`render_with_budget`] truncates `tool_results` first by line count, then
//! by cumulative byte size (dropping the oldest entries first — the most
//! recent tool results are the most relevant context), and finally enforces
//! a hard cap on the total rendered byte size.
//!
//! No new `STRAT-E####` codes are introduced; all failure modes are returned
//! as variants of the local error types.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::{self, Debug, Display, Formatter};
use std::str::FromStr;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// TemplateId
// ---------------------------------------------------------------------------

/// Short stable identifier for a [`PromptTemplate`].
///
/// Accepts ASCII `[a-z0-9._-]`, length 1..=64, and may not start with `-` or
/// `.` (those prefixes are reserved for future namespacing / hidden ids).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TemplateId(String);

impl TemplateId {
    /// Validate and wrap `s`.
    ///
    /// # Errors
    ///
    /// See [`TemplateIdError`].
    pub fn new(s: &str) -> Result<Self, TemplateIdError> {
        if s.is_empty() {
            return Err(TemplateIdError::Empty);
        }
        if s.len() > 64 {
            return Err(TemplateIdError::TooLong { len: s.len() });
        }
        for ch in s.chars() {
            let ok = ch.is_ascii_lowercase()
                || ch.is_ascii_digit()
                || ch == '.'
                || ch == '_'
                || ch == '-';
            if !ok {
                return Err(TemplateIdError::InvalidChar { ch });
            }
        }
        // Length checked above as >= 1, and the char-loop guarantees ASCII,
        // so the first byte is a one-byte char.
        let first = s.as_bytes()[0] as char;
        if first == '-' || first == '.' {
            return Err(TemplateIdError::BadPrefix { ch: first });
        }
        Ok(Self(s.to_owned()))
    }

    /// Borrow the underlying id as a `&str`.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl FromStr for TemplateId {
    type Err = TemplateIdError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::new(s)
    }
}

impl Display for TemplateId {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl AsRef<str> for TemplateId {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl Serialize for TemplateId {
    fn serialize<S: serde::Serializer>(&self, ser: S) -> Result<S::Ok, S::Error> {
        ser.serialize_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for TemplateId {
    fn deserialize<D: serde::Deserializer<'de>>(de: D) -> Result<Self, D::Error> {
        let s = String::deserialize(de)?;
        Self::new(&s).map_err(serde::de::Error::custom)
    }
}

/// Reasons [`TemplateId::new`] rejects an input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TemplateIdError {
    /// Input was the empty string.
    Empty,
    /// Input exceeded the 64-byte cap.
    TooLong {
        /// Offending length in bytes.
        len: usize,
    },
    /// Input contained a character outside `[a-z0-9._-]`.
    InvalidChar {
        /// Offending character.
        ch: char,
    },
    /// Input started with a reserved prefix character (`-` or `.`).
    BadPrefix {
        /// Offending leading character.
        ch: char,
    },
}

impl Display for TemplateIdError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => f.write_str("template id is empty"),
            Self::TooLong { len } => write!(f, "template id is {len} bytes, max 64"),
            Self::InvalidChar { ch } => {
                write!(f, "template id contains invalid character {ch:?}")
            }
            Self::BadPrefix { ch } => {
                write!(f, "template id may not start with {ch:?}")
            }
        }
    }
}

impl Error for TemplateIdError {}

// ---------------------------------------------------------------------------
// PromptTemplate
// ---------------------------------------------------------------------------

/// A reusable structured prompt template.
///
/// The three text slots are interpolated independently and then handed to
/// the provider as `system + agent_header + turn`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromptTemplate {
    /// Stable identifier — used for cache keys, eval reports, telemetry.
    pub id: TemplateId,
    /// System slot template body. May contain `{ $var }` placeholders.
    pub system: String,
    /// Agent-header slot template body. May contain `{ $var }` placeholders.
    pub agent_header: String,
    /// Per-turn slot template body. May contain `{ $var }`, `{ $user_turn }`,
    /// and `{ $tool_results }`.
    pub turn_format: String,
}

// ---------------------------------------------------------------------------
// PromptContext
// ---------------------------------------------------------------------------

/// Per-turn inputs supplied to [`render`].
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct PromptContext {
    /// Latest user-authored turn text.
    pub user_turn: String,
    /// Named variables available to `{ $name }` placeholders.
    pub vars: BTreeMap<String, String>,
    /// Recent tool-call results in oldest-to-newest order.
    pub tool_results: Vec<ToolResultSnippet>,
}

/// One rendered tool result line.
///
/// `bytes` is recorded explicitly so callers don't have to recompute it for
/// budget enforcement.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolResultSnippet {
    /// Tool identifier the result came from.
    pub tool_id: String,
    /// Whether the tool call succeeded.
    pub ok: bool,
    /// Body of the result (may be truncated upstream).
    pub body: String,
    /// Byte size attributed to this snippet for budget bookkeeping.
    pub bytes: usize,
}

// ---------------------------------------------------------------------------
// RenderedPrompt
// ---------------------------------------------------------------------------

/// Result of [`render`] / [`render_with_budget`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RenderedPrompt {
    /// Interpolated system slot.
    pub system: String,
    /// Interpolated agent-header slot.
    pub agent_header: String,
    /// Interpolated per-turn slot.
    pub turn: String,
    /// Total bytes across the three slots.
    pub total_bytes: usize,
}

// ---------------------------------------------------------------------------
// PromptBudget
// ---------------------------------------------------------------------------

/// Soft and hard caps applied by [`render_with_budget`].
#[allow(clippy::struct_field_names)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromptBudget {
    /// Hard cap on the total rendered byte size.
    pub max_total_bytes: usize,
    /// Maximum number of tool-result lines to include.
    pub max_tool_result_lines: usize,
    /// Maximum cumulative byte size across kept tool-result snippets.
    pub max_tool_result_bytes: usize,
}

impl Default for PromptBudget {
    fn default() -> Self {
        Self {
            max_total_bytes: 32_768,
            max_tool_result_lines: 50,
            max_tool_result_bytes: 8_192,
        }
    }
}

// ---------------------------------------------------------------------------
// PromptRenderError
// ---------------------------------------------------------------------------

/// Reasons rendering can fail.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PromptRenderError {
    /// A `{ $name }` placeholder was not found in [`PromptContext::vars`].
    MissingVar {
        /// Name of the missing variable.
        name: String,
    },
    /// Template text was malformed (e.g. unmatched `{` / `}` or a placeholder
    /// that did not start with `$`).
    BadTemplate(String),
    /// Total rendered size exceeded [`PromptBudget::max_total_bytes`].
    BodyTooLarge {
        /// Actual rendered byte size.
        actual: usize,
        /// Cap that was exceeded.
        max: usize,
    },
}

impl Display for PromptRenderError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingVar { name } => write!(f, "missing prompt variable {name:?}"),
            Self::BadTemplate(msg) => write!(f, "bad prompt template: {msg}"),
            Self::BodyTooLarge { actual, max } => {
                write!(f, "rendered prompt is {actual} bytes, max {max}")
            }
        }
    }
}

impl Error for PromptRenderError {}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

/// Render a template against a context, returning the three interpolated
/// slots and the total byte size.
///
/// # Errors
///
/// See [`PromptRenderError`].
pub fn render(
    template: &PromptTemplate,
    ctx: &PromptContext,
) -> Result<RenderedPrompt, PromptRenderError> {
    let tool_block = render_tool_results(&ctx.tool_results);
    let system = interpolate(&template.system, ctx, &tool_block)?;
    let agent_header = interpolate(&template.agent_header, ctx, &tool_block)?;
    let turn = interpolate(&template.turn_format, ctx, &tool_block)?;
    let total_bytes = system.len() + agent_header.len() + turn.len();
    Ok(RenderedPrompt {
        system,
        agent_header,
        turn,
        total_bytes,
    })
}

/// Render a template under a [`PromptBudget`].
///
/// Tool results are first trimmed via [`truncate_tool_results`], the prompt
/// is rendered, and finally the total byte size is verified against
/// [`PromptBudget::max_total_bytes`].
///
/// # Errors
///
/// See [`PromptRenderError`]; in particular [`PromptRenderError::BodyTooLarge`].
pub fn render_with_budget(
    template: &PromptTemplate,
    ctx: &PromptContext,
    budget: &PromptBudget,
) -> Result<RenderedPrompt, PromptRenderError> {
    let trimmed = truncate_tool_results(&ctx.tool_results, budget);
    let trimmed_ctx = PromptContext {
        user_turn: ctx.user_turn.clone(),
        vars: ctx.vars.clone(),
        tool_results: trimmed,
    };
    let rendered = render(template, &trimmed_ctx)?;
    if rendered.total_bytes > budget.max_total_bytes {
        return Err(PromptRenderError::BodyTooLarge {
            actual: rendered.total_bytes,
            max: budget.max_total_bytes,
        });
    }
    Ok(rendered)
}

/// Trim a tool-result slice to satisfy [`PromptBudget::max_tool_result_lines`]
/// and [`PromptBudget::max_tool_result_bytes`].
///
/// Oldest entries are dropped first; the most recent results are preserved.
#[must_use]
pub fn truncate_tool_results(
    results: &[ToolResultSnippet],
    budget: &PromptBudget,
) -> Vec<ToolResultSnippet> {
    if budget.max_tool_result_lines == 0 || results.is_empty() {
        return Vec::new();
    }
    // First, cap by line count from the END (most recent N).
    let start = results.len().saturating_sub(budget.max_tool_result_lines);
    let mut kept: Vec<ToolResultSnippet> = results[start..].to_vec();
    // Then, cap by cumulative bytes, dropping from the FRONT (oldest first).
    let mut total: usize = kept.iter().map(|s| s.bytes).sum();
    while total > budget.max_tool_result_bytes && !kept.is_empty() {
        let removed = kept.remove(0);
        total = total.saturating_sub(removed.bytes);
    }
    kept
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

/// Render the `{ $tool_results }` block deterministically.
fn render_tool_results(results: &[ToolResultSnippet]) -> String {
    use std::fmt::Write as _;
    let mut buf = String::new();
    for (i, r) in results.iter().enumerate() {
        if i > 0 {
            buf.push('\n');
        }
        // Writing into a `String` is infallible; ignore the result to satisfy
        // the no-`unwrap`/`expect` workspace rule.
        let _ = write!(
            buf,
            "[tool:{} ok={} {}B] {}",
            r.tool_id, r.ok, r.bytes, r.body
        );
    }
    buf
}

/// Interpolate `{ $name }` placeholders in `template`.
fn interpolate(
    template: &str,
    ctx: &PromptContext,
    tool_block: &str,
) -> Result<String, PromptRenderError> {
    let mut out = String::with_capacity(template.len());
    let bytes = template.as_bytes();
    let mut cursor = 0;
    while cursor < bytes.len() {
        // Fast scan to the next `{` or `}` — both are ASCII so byte indexing
        // lands on a char boundary even if the template contains multi-byte
        // text in between.
        let mut next = cursor;
        while next < bytes.len() && bytes[next] != b'{' && bytes[next] != b'}' {
            next += 1;
        }
        if next > cursor {
            out.push_str(&template[cursor..next]);
        }
        if next == bytes.len() {
            break;
        }
        if bytes[next] == b'}' {
            return Err(PromptRenderError::BadTemplate(
                "unmatched closing brace".to_owned(),
            ));
        }
        // bytes[next] == b'{'
        let close = find_close(bytes, next + 1)?;
        let inner_str = &template[next + 1..close];
        let name = parse_placeholder(inner_str)?;
        match name {
            "user_turn" => out.push_str(&ctx.user_turn),
            "tool_results" => out.push_str(tool_block),
            other => {
                let Some(v) = ctx.vars.get(other) else {
                    return Err(PromptRenderError::MissingVar {
                        name: other.to_owned(),
                    });
                };
                out.push_str(v);
            }
        }
        cursor = close + 1;
    }
    Ok(out)
}

/// Find the byte index of the next `}` starting at `from`.
fn find_close(bytes: &[u8], from: usize) -> Result<usize, PromptRenderError> {
    let mut j = from;
    while j < bytes.len() {
        if bytes[j] == b'}' {
            return Ok(j);
        }
        if bytes[j] == b'{' {
            return Err(PromptRenderError::BadTemplate(
                "nested opening brace inside placeholder".to_owned(),
            ));
        }
        j += 1;
    }
    Err(PromptRenderError::BadTemplate(
        "unmatched opening brace".to_owned(),
    ))
}

/// Parse `{ $name }` body, returning the bare `name`.
fn parse_placeholder(inner: &str) -> Result<&str, PromptRenderError> {
    let trimmed = inner.trim();
    let Some(name) = trimmed.strip_prefix('$') else {
        return Err(PromptRenderError::BadTemplate(format!(
            "placeholder {trimmed:?} does not start with $"
        )));
    };
    if name.is_empty() {
        return Err(PromptRenderError::BadTemplate(
            "empty placeholder name".to_owned(),
        ));
    }
    Ok(name)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn tmpl(system: &str, header: &str, turn: &str) -> PromptTemplate {
        PromptTemplate {
            id: TemplateId::new("t1").expect("valid id"),
            system: system.to_owned(),
            agent_header: header.to_owned(),
            turn_format: turn.to_owned(),
        }
    }

    fn snip(id: &str, body: &str) -> ToolResultSnippet {
        ToolResultSnippet {
            tool_id: id.to_owned(),
            ok: true,
            body: body.to_owned(),
            bytes: body.len(),
        }
    }

    #[test]
    fn template_id_happy() {
        let id = TemplateId::from_str("planner.v1_alpha-2").expect("valid");
        assert_eq!(id.as_str(), "planner.v1_alpha-2");
        assert_eq!(format!("{id}"), "planner.v1_alpha-2");
    }

    #[test]
    fn template_id_empty_rejected() {
        assert_eq!(TemplateId::from_str(""), Err(TemplateIdError::Empty));
    }

    #[test]
    fn template_id_too_long_rejected() {
        let big = "a".repeat(65);
        assert_eq!(
            TemplateId::from_str(&big),
            Err(TemplateIdError::TooLong { len: 65 })
        );
    }

    #[test]
    fn template_id_invalid_char_rejected() {
        assert_eq!(
            TemplateId::from_str("Bad"),
            Err(TemplateIdError::InvalidChar { ch: 'B' })
        );
    }

    #[test]
    fn template_id_bad_prefix_rejected() {
        assert_eq!(
            TemplateId::from_str("-leading"),
            Err(TemplateIdError::BadPrefix { ch: '-' })
        );
        assert_eq!(
            TemplateId::from_str(".hidden"),
            Err(TemplateIdError::BadPrefix { ch: '.' })
        );
    }

    #[test]
    fn template_id_max_length_ok() {
        let max = "a".repeat(64);
        let id = TemplateId::from_str(&max).expect("64 is allowed");
        assert_eq!(id.as_str().len(), 64);
    }

    #[test]
    fn prompt_template_serde_roundtrip() {
        let t = tmpl("sys", "hdr", "turn");
        let s = serde_json::to_string(&t).expect("ser");
        let back: PromptTemplate = serde_json::from_str(&s).expect("de");
        assert_eq!(t, back);
    }

    #[test]
    fn prompt_context_serde_roundtrip() {
        let mut vars = BTreeMap::new();
        vars.insert("k".to_owned(), "v".to_owned());
        let ctx = PromptContext {
            user_turn: "hi".to_owned(),
            vars,
            tool_results: vec![snip("ls", "ok")],
        };
        let s = serde_json::to_string(&ctx).expect("ser");
        let back: PromptContext = serde_json::from_str(&s).expect("de");
        assert_eq!(ctx, back);
    }

    #[test]
    fn render_interpolates_var() {
        let t = tmpl("hello { $name }", "", "");
        let mut vars = BTreeMap::new();
        vars.insert("name".to_owned(), "world".to_owned());
        let ctx = PromptContext {
            user_turn: String::new(),
            vars,
            tool_results: Vec::new(),
        };
        let r = render(&t, &ctx).expect("render");
        assert_eq!(r.system, "hello world");
    }

    #[test]
    fn render_interpolates_user_turn() {
        let t = tmpl("", "", "USER: { $user_turn }");
        let ctx = PromptContext {
            user_turn: "make me a sandwich".to_owned(),
            vars: BTreeMap::new(),
            tool_results: Vec::new(),
        };
        let r = render(&t, &ctx).expect("render");
        assert_eq!(r.turn, "USER: make me a sandwich");
    }

    #[test]
    fn render_interpolates_tool_results() {
        let t = tmpl("", "", "tools:\n{ $tool_results }");
        let ctx = PromptContext {
            user_turn: String::new(),
            vars: BTreeMap::new(),
            tool_results: vec![snip("ls", "a.txt"), snip("cat", "hi")],
        };
        let r = render(&t, &ctx).expect("render");
        assert_eq!(
            r.turn,
            "tools:\n[tool:ls ok=true 5B] a.txt\n[tool:cat ok=true 2B] hi"
        );
    }

    #[test]
    fn render_missing_var_errors() {
        let t = tmpl("hi { $name }", "", "");
        let ctx = PromptContext::default();
        let err = render(&t, &ctx).expect_err("should fail");
        assert_eq!(
            err,
            PromptRenderError::MissingVar {
                name: "name".to_owned()
            }
        );
    }

    #[test]
    fn render_bad_template_unmatched_open() {
        let t = tmpl("hi { $name", "", "");
        let ctx = PromptContext::default();
        let err = render(&t, &ctx).expect_err("should fail");
        assert!(
            matches!(err, PromptRenderError::BadTemplate(_)),
            "expected BadTemplate, got {err:?}"
        );
    }

    #[test]
    fn render_bad_template_unmatched_close() {
        let t = tmpl("hi } name", "", "");
        let ctx = PromptContext::default();
        let err = render(&t, &ctx).expect_err("should fail");
        assert!(
            matches!(err, PromptRenderError::BadTemplate(_)),
            "expected BadTemplate, got {err:?}"
        );
    }

    #[test]
    fn render_bad_template_no_dollar() {
        let t = tmpl("hi { name }", "", "");
        let ctx = PromptContext::default();
        let err = render(&t, &ctx).expect_err("should fail");
        assert!(
            matches!(err, PromptRenderError::BadTemplate(_)),
            "expected BadTemplate, got {err:?}"
        );
    }

    #[test]
    fn render_total_bytes_matches_sum() {
        let t = tmpl("aaa", "bb", "c");
        let ctx = PromptContext::default();
        let r = render(&t, &ctx).expect("render");
        assert_eq!(r.total_bytes, 3 + 2 + 1);
    }

    #[test]
    fn render_empty_template() {
        let t = tmpl("", "", "");
        let ctx = PromptContext::default();
        let r = render(&t, &ctx).expect("render");
        assert_eq!(r.system, "");
        assert_eq!(r.agent_header, "");
        assert_eq!(r.turn, "");
        assert_eq!(r.total_bytes, 0);
    }

    #[test]
    fn render_no_placeholders_no_vars_ok() {
        let t = tmpl("static system", "static header", "static turn");
        let ctx = PromptContext::default();
        let r = render(&t, &ctx).expect("render");
        assert_eq!(r.system, "static system");
    }

    #[test]
    fn render_with_budget_trims_lines() {
        let t = tmpl("", "", "{ $tool_results }");
        let ctx = PromptContext {
            user_turn: String::new(),
            vars: BTreeMap::new(),
            tool_results: (0..5)
                .map(|i| snip(&format!("t{i}"), &format!("body{i}")))
                .collect(),
        };
        let budget = PromptBudget {
            max_total_bytes: 32_768,
            max_tool_result_lines: 2,
            max_tool_result_bytes: 8_192,
        };
        let r = render_with_budget(&t, &ctx, &budget).expect("render");
        // Last two snippets preserved: t3, t4.
        assert!(r.turn.contains("[tool:t3"));
        assert!(r.turn.contains("[tool:t4"));
        assert!(!r.turn.contains("[tool:t0"));
        assert!(!r.turn.contains("[tool:t1"));
        assert!(!r.turn.contains("[tool:t2"));
    }

    #[test]
    fn render_with_budget_trims_bytes() {
        let t = tmpl("", "", "{ $tool_results }");
        // Each snippet is 5 bytes; 6 entries = 30 bytes; budget of 12 keeps last 2.
        let ctx = PromptContext {
            user_turn: String::new(),
            vars: BTreeMap::new(),
            tool_results: (0..6).map(|i| snip(&format!("t{i}"), "abcde")).collect(),
        };
        let budget = PromptBudget {
            max_total_bytes: 32_768,
            max_tool_result_lines: 100,
            max_tool_result_bytes: 12,
        };
        let r = render_with_budget(&t, &ctx, &budget).expect("render");
        assert!(r.turn.contains("[tool:t4"));
        assert!(r.turn.contains("[tool:t5"));
        assert!(!r.turn.contains("[tool:t0"));
    }

    #[test]
    fn render_with_budget_body_too_large() {
        let big = "x".repeat(1_000);
        let t = tmpl(&big, &big, &big);
        let ctx = PromptContext::default();
        let budget = PromptBudget {
            max_total_bytes: 100,
            max_tool_result_lines: 0,
            max_tool_result_bytes: 0,
        };
        let err = render_with_budget(&t, &ctx, &budget).expect_err("should fail");
        assert_eq!(
            err,
            PromptRenderError::BodyTooLarge {
                actual: 3_000,
                max: 100
            }
        );
    }

    #[test]
    fn truncate_preserves_last_n() {
        let results: Vec<ToolResultSnippet> = (0..5).map(|i| snip(&format!("t{i}"), "x")).collect();
        let budget = PromptBudget {
            max_total_bytes: 32_768,
            max_tool_result_lines: 3,
            max_tool_result_bytes: 8_192,
        };
        let trimmed = truncate_tool_results(&results, &budget);
        assert_eq!(trimmed.len(), 3);
        assert_eq!(trimmed[0].tool_id, "t2");
        assert_eq!(trimmed[2].tool_id, "t4");
    }

    #[test]
    fn truncate_empty_input() {
        let budget = PromptBudget::default();
        let trimmed = truncate_tool_results(&[], &budget);
        assert!(trimmed.is_empty());
    }

    #[test]
    fn truncate_zero_lines() {
        let results = vec![snip("t1", "body")];
        let budget = PromptBudget {
            max_total_bytes: 32_768,
            max_tool_result_lines: 0,
            max_tool_result_bytes: 8_192,
        };
        assert!(truncate_tool_results(&results, &budget).is_empty());
    }

    #[test]
    fn truncate_drops_oldest_for_byte_budget() {
        let results: Vec<ToolResultSnippet> =
            (0..4).map(|i| snip(&format!("t{i}"), "abcde")).collect();
        let budget = PromptBudget {
            max_total_bytes: 32_768,
            max_tool_result_lines: 100,
            max_tool_result_bytes: 10,
        };
        let trimmed = truncate_tool_results(&results, &budget);
        // 4 entries of 5 bytes = 20; budget 10 → drop oldest two, keep t2,t3.
        assert_eq!(trimmed.len(), 2);
        assert_eq!(trimmed[0].tool_id, "t2");
        assert_eq!(trimmed[1].tool_id, "t3");
    }

    #[test]
    fn prompt_budget_default_values() {
        let d = PromptBudget::default();
        assert_eq!(d.max_total_bytes, 32_768);
        assert_eq!(d.max_tool_result_lines, 50);
        assert_eq!(d.max_tool_result_bytes, 8_192);
    }

    #[test]
    fn rendered_prompt_serde_roundtrip() {
        let r = RenderedPrompt {
            system: "s".to_owned(),
            agent_header: "h".to_owned(),
            turn: "t".to_owned(),
            total_bytes: 3,
        };
        let s = serde_json::to_string(&r).expect("ser");
        let back: RenderedPrompt = serde_json::from_str(&s).expect("de");
        assert_eq!(r, back);
    }

    #[test]
    fn render_error_display_variants() {
        let e = PromptRenderError::MissingVar {
            name: "x".to_owned(),
        };
        assert!(format!("{e}").contains("missing prompt variable"));
        let e = PromptRenderError::BadTemplate("oops".to_owned());
        assert!(format!("{e}").contains("bad prompt template"));
        let e = PromptRenderError::BodyTooLarge { actual: 5, max: 1 };
        assert!(format!("{e}").contains("rendered prompt is 5 bytes"));
    }

    #[test]
    fn template_id_error_display_variants() {
        assert!(format!("{}", TemplateIdError::Empty).contains("empty"));
        assert!(format!("{}", TemplateIdError::TooLong { len: 99 }).contains("99"));
        assert!(format!("{}", TemplateIdError::InvalidChar { ch: '!' }).contains("'!'"));
        assert!(format!("{}", TemplateIdError::BadPrefix { ch: '-' }).contains("'-'"));
    }

    #[test]
    fn tool_result_snippet_format_matches_doc() {
        let t = tmpl("", "", "{ $tool_results }");
        let mut s = snip("grep", "hello world");
        s.ok = true;
        s.bytes = 124;
        let ctx = PromptContext {
            user_turn: String::new(),
            vars: BTreeMap::new(),
            tool_results: vec![s],
        };
        let r = render(&t, &ctx).expect("render");
        assert_eq!(r.turn, "[tool:grep ok=true 124B] hello world");
    }

    #[test]
    fn template_id_serde_transparent() {
        let id = TemplateId::new("planner.v1").expect("valid");
        let s = serde_json::to_string(&id).expect("ser");
        assert_eq!(s, "\"planner.v1\"");
        let back: TemplateId = serde_json::from_str(&s).expect("de");
        assert_eq!(id, back);
    }

    #[test]
    fn template_id_serde_rejects_invalid() {
        let bad = "\"Bad!\"";
        let r: Result<TemplateId, _> = serde_json::from_str(bad);
        assert!(r.is_err());
    }

    #[test]
    fn template_id_as_ref() {
        let id = TemplateId::new("ok").expect("valid");
        let s: &str = id.as_ref();
        assert_eq!(s, "ok");
    }

    #[test]
    fn render_var_used_in_all_three_slots() {
        let t = tmpl("S:{ $x }", "H:{ $x }", "T:{ $x }");
        let mut vars = BTreeMap::new();
        vars.insert("x".to_owned(), "Z".to_owned());
        let ctx = PromptContext {
            user_turn: String::new(),
            vars,
            tool_results: Vec::new(),
        };
        let r = render(&t, &ctx).expect("render");
        assert_eq!(r.system, "S:Z");
        assert_eq!(r.agent_header, "H:Z");
        assert_eq!(r.turn, "T:Z");
    }

    #[test]
    fn render_empty_placeholder_name_rejected() {
        let t = tmpl("hi { $ }", "", "");
        let ctx = PromptContext::default();
        let err = render(&t, &ctx).expect_err("should fail");
        assert!(
            matches!(err, PromptRenderError::BadTemplate(_)),
            "expected BadTemplate, got {err:?}"
        );
    }

    #[test]
    fn render_nested_brace_rejected() {
        let t = tmpl("hi { $a { b }", "", "");
        let ctx = PromptContext::default();
        let err = render(&t, &ctx).expect_err("should fail");
        assert!(
            matches!(err, PromptRenderError::BadTemplate(_)),
            "expected BadTemplate, got {err:?}"
        );
    }
}
