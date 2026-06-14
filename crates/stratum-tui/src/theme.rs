//! Color theming for the chat TUI.
//!
//! Themes are layered: a `default` theme baked into the binary,
//! plus optional named JSON overrides loaded from
//! `<config>/stratum/themes/<name>.json`. The active theme is held
//! in a process-global atomic so render helpers can call
//! [`current()`] without threading a parameter through every layer.
//!
//! ## File format
//!
//! ```json
//! {
//!   "user_prefix_fg":  "cyan",
//!   "ai_prefix_fg":    "magenta",
//!   "keyword_fg":      "blue",
//!   "string_fg":       "green",
//!   "comment_fg":      "gray",
//!   "header_fg":       "yellow",
//!   "quote_fg":        "gray",
//!   "bullet_fg":       "white",
//!   "dim_fg":          "gray"
//! }
//! ```
//!
//! Every field is optional; missing fields fall back to the active
//! built-in theme. Color names accept the 16 ratatui named colors
//! (case-insensitive). Unknown names silently fall back to the
//! built-in default — themes never panic the renderer.

#![allow(
    clippy::module_name_repetitions,
    reason = "Theme is the primary public type; the module is named after it"
)]
#![allow(
    unreachable_pub,
    reason = "private module by design; pub kept for readability"
)]
#![allow(
    clippy::redundant_pub_crate,
    reason = "internal API kept pub for documentation; module itself is private"
)]

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Mutex, OnceLock};

use ratatui::style::{Color, Modifier, Style};
use serde::{Deserialize, Serialize};

/// One named theme variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Theme {
    /// `you:` prefix style (kept for compatibility).
    pub user_prefix: Style,
    /// `ai:` prefix style (kept for compatibility).
    pub ai_prefix: Style,
    /// Left-gutter bar color for user turns.
    pub user_gutter: Style,
    /// Left-gutter bar color for assistant turns.
    pub ai_gutter: Style,
    /// Style for tool-call icons / lines.
    pub tool: Style,
    /// Bold inline markdown.
    pub bold: Style,
    /// Italic inline markdown.
    pub italic: Style,
    /// `` `inline code` ``.
    pub inline_code: Style,
    /// `# Header` lines.
    pub header: Style,
    /// `- bullet` markers.
    pub bullet: Style,
    /// `> blockquote` prefix.
    pub quote: Style,
    /// Generic dim text (status hints, code-fence frame).
    pub dim: Style,
    /// Language keywords inside a code fence.
    pub keyword: Style,
    /// String literals inside a code fence.
    pub string_lit: Style,
    /// Line comments inside a code fence.
    pub comment: Style,
}

impl Default for Theme {
    fn default() -> Self {
        Self::built_in_default()
    }
}

impl Theme {
    /// Stratum's branded default theme — uses brand-primary teal for
    /// the user gutter + headers, brand-accent amber for the AI
    /// gutter + tool icons + inline code. Looks branded out-of-box
    /// per plan/44. Falls back to ratatui's truecolor downsampler on
    /// 256-color terminals.
    ///
    /// Users on broken / e-ink terminals can switch to `plain` (the
    /// previous modifier-only behavior) with `/theme plain`.
    #[must_use]
    pub const fn built_in_default() -> Self {
        // Brand primary #1E5E5E — warm slate teal.
        let primary = Color::Rgb(0x1E, 0x5E, 0x5E);
        // Brand accent #D9844D — warm amber.
        let accent = Color::Rgb(0xD9, 0x84, 0x4D);
        let dim = Style::new().add_modifier(Modifier::DIM);
        Self {
            user_prefix: Style::new().fg(primary).add_modifier(Modifier::BOLD),
            ai_prefix: Style::new().fg(accent).add_modifier(Modifier::BOLD),
            user_gutter: Style::new().fg(primary),
            ai_gutter: Style::new().fg(accent),
            tool: Style::new().fg(accent).add_modifier(Modifier::DIM),
            bold: Style::new().add_modifier(Modifier::BOLD),
            italic: Style::new().add_modifier(Modifier::ITALIC),
            inline_code: Style::new().fg(accent).add_modifier(Modifier::REVERSED),
            header: Style::new().fg(primary).add_modifier(Modifier::BOLD),
            bullet: Style::new().fg(primary),
            quote: dim,
            dim,
            keyword: Style::new().fg(primary).add_modifier(Modifier::BOLD),
            string_lit: Style::new().fg(accent).add_modifier(Modifier::ITALIC),
            comment: dim,
        }
    }

    /// Plain theme — pure modifiers, no colors. Use on terminals
    /// with broken 256-color palettes, e-ink screens, or by
    /// preference. Pre-brand v0.2 default.
    #[must_use]
    pub const fn plain() -> Self {
        let dim = Style::new().add_modifier(Modifier::DIM);
        Self {
            user_prefix: Style::new().add_modifier(Modifier::BOLD),
            ai_prefix: Style::new().add_modifier(Modifier::BOLD),
            user_gutter: Style::new().add_modifier(Modifier::DIM),
            ai_gutter: Style::new().add_modifier(Modifier::DIM),
            tool: Style::new().add_modifier(Modifier::DIM),
            bold: Style::new().add_modifier(Modifier::BOLD),
            italic: Style::new().add_modifier(Modifier::ITALIC),
            inline_code: Style::new().add_modifier(Modifier::REVERSED),
            header: Style::new().add_modifier(Modifier::BOLD),
            bullet: Style::new(),
            quote: dim,
            dim,
            keyword: Style::new().add_modifier(Modifier::BOLD),
            string_lit: Style::new().add_modifier(Modifier::ITALIC),
            comment: dim,
        }
    }

    /// High-contrast monochrome — alias of `plain`. Kept for backwards
    /// compatibility with anyone who hardcoded `/theme mono`.
    #[must_use]
    pub const fn mono() -> Self {
        Self::plain()
    }

    /// Color-forward variant: cyan/magenta prefixes, blue keywords,
    /// green strings, gray comments. Targets dark backgrounds.
    #[must_use]
    pub const fn vivid() -> Self {
        Self {
            user_prefix: Style::new()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
            ai_prefix: Style::new()
                .fg(Color::Magenta)
                .add_modifier(Modifier::BOLD),
            user_gutter: Style::new().fg(Color::Cyan),
            ai_gutter: Style::new().fg(Color::Magenta),
            tool: Style::new().fg(Color::Yellow).add_modifier(Modifier::DIM),
            bold: Style::new().add_modifier(Modifier::BOLD),
            italic: Style::new().add_modifier(Modifier::ITALIC),
            inline_code: Style::new()
                .fg(Color::Yellow)
                .add_modifier(Modifier::REVERSED),
            header: Style::new()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
            bullet: Style::new().fg(Color::Cyan),
            quote: Style::new()
                .fg(Color::Gray)
                .add_modifier(Modifier::DIM),
            dim: Style::new().add_modifier(Modifier::DIM),
            keyword: Style::new()
                .fg(Color::Blue)
                .add_modifier(Modifier::BOLD),
            string_lit: Style::new()
                .fg(Color::Green)
                .add_modifier(Modifier::ITALIC),
            comment: Style::new()
                .fg(Color::Gray)
                .add_modifier(Modifier::DIM),
        }
    }

    /// Solarized-dark inspired palette.
    #[must_use]
    pub const fn ocean() -> Self {
        Self {
            user_prefix: Style::new()
                .fg(Color::LightCyan)
                .add_modifier(Modifier::BOLD),
            ai_prefix: Style::new()
                .fg(Color::LightBlue)
                .add_modifier(Modifier::BOLD),
            user_gutter: Style::new().fg(Color::LightCyan),
            ai_gutter: Style::new().fg(Color::LightBlue),
            tool: Style::new().fg(Color::LightYellow).add_modifier(Modifier::DIM),
            bold: Style::new().add_modifier(Modifier::BOLD),
            italic: Style::new().add_modifier(Modifier::ITALIC),
            inline_code: Style::new()
                .fg(Color::LightYellow)
                .add_modifier(Modifier::REVERSED),
            header: Style::new()
                .fg(Color::LightYellow)
                .add_modifier(Modifier::BOLD),
            bullet: Style::new().fg(Color::LightBlue),
            quote: Style::new()
                .fg(Color::DarkGray)
                .add_modifier(Modifier::DIM),
            dim: Style::new().add_modifier(Modifier::DIM),
            keyword: Style::new()
                .fg(Color::LightMagenta)
                .add_modifier(Modifier::BOLD),
            string_lit: Style::new()
                .fg(Color::LightGreen)
                .add_modifier(Modifier::ITALIC),
            comment: Style::new()
                .fg(Color::DarkGray)
                .add_modifier(Modifier::DIM),
        }
    }
}

/// Built-in theme variants known by name. `plain` is the explicit
/// modifier-only theme (pre-brand v0.2 default); `mono` aliases it
/// for backwards compat. `default` is now the branded theme per
/// plan/44.
const BUILT_INS: &[(&str, fn() -> Theme)] = &[
    ("default", Theme::built_in_default),
    ("plain", Theme::plain),
    ("mono", Theme::mono),
    ("vivid", Theme::vivid),
    ("ocean", Theme::ocean),
];

/// Atomic theme slot. Encodes the active built-in index when the
/// `OVERRIDE` slot is empty; reads of `current()` consult the override
/// first so user-loaded themes win.
static ACTIVE: AtomicU8 = AtomicU8::new(0);

/// Overrides loaded from disk replace the built-in slot wholesale.
fn override_slot() -> &'static Mutex<Option<Theme>> {
    static SLOT: OnceLock<Mutex<Option<Theme>>> = OnceLock::new();
    SLOT.get_or_init(|| Mutex::new(None))
}

/// Active theme. Cheap — no allocations on the hot render path.
#[must_use]
pub fn current() -> Theme {
    if let Ok(g) = override_slot().lock() {
        if let Some(t) = *g {
            return t;
        }
    }
    let i = ACTIVE.load(Ordering::Relaxed) as usize;
    BUILT_INS
        .get(i)
        .map_or_else(Theme::built_in_default, |(_, f)| f())
}

/// Apply a theme by name. Built-in names win; otherwise the JSON
/// loader is tried against `<themes_dir>/<name>.json`. On success the
/// new theme becomes the process-wide active theme.
///
/// # Errors
/// Returns a stable error string suitable for the palette outcome
/// when neither built-in nor JSON file resolves the name.
pub fn set_by_name(name: &str, themes_dir: Option<&Path>) -> Result<(), String> {
    let name = name.trim();
    if name.is_empty() {
        return Err("theme name required".to_string());
    }
    if let Some(idx) = BUILT_INS
        .iter()
        .position(|(n, _)| n.eq_ignore_ascii_case(name))
    {
        if let Ok(mut g) = override_slot().lock() {
            *g = None;
        }
        #[allow(
            clippy::cast_possible_truncation,
            reason = "BUILT_INS length is a small constant"
        )]
        ACTIVE.store(idx as u8, Ordering::Relaxed);
        return Ok(());
    }
    let Some(dir) = themes_dir else {
        return Err(format!("unknown theme: {name}"));
    };
    let path = dir.join(format!("{name}.json"));
    let bytes = std::fs::read(&path)
        .map_err(|e| format!("failed to read {}: {e}", path.display()))?;
    let parsed: ThemeFile = serde_json::from_slice(&bytes)
        .map_err(|e| format!("failed to parse {}: {e}", path.display()))?;
    let mut t = Theme::built_in_default();
    parsed.apply(&mut t);
    if let Ok(mut g) = override_slot().lock() {
        *g = Some(t);
    }
    Ok(())
}

/// Names a user can pass to `/theme <name>`: built-ins plus any
/// `*.json` file in `themes_dir`.
#[must_use]
pub fn list(themes_dir: Option<&Path>) -> Vec<String> {
    let mut out: Vec<String> = BUILT_INS.iter().map(|(n, _)| (*n).to_string()).collect();
    if let Some(dir) = themes_dir {
        if let Ok(rd) = std::fs::read_dir(dir) {
            for entry in rd.flatten() {
                if let Some(name) = entry.path().file_stem().and_then(|s| s.to_str()) {
                    if entry
                        .path()
                        .extension()
                        .and_then(|e| e.to_str())
                        .is_some_and(|e| e.eq_ignore_ascii_case("json"))
                        && !out.iter().any(|n| n.eq_ignore_ascii_case(name))
                    {
                        out.push(name.to_string());
                    }
                }
            }
        }
    }
    out.sort();
    out
}

/// Compute the default themes directory relative to a config root.
#[must_use]
#[allow(
    dead_code,
    reason = "only called from the provider-llama-cpp-feature chat_with_model"
)]
pub fn themes_dir_for(config_root: &Path) -> PathBuf {
    config_root.join("themes")
}

/// Read the persisted theme name (one line) from a state file. Used
/// at startup so the previous `/theme` choice survives restart.
#[must_use]
pub fn read_persisted(state_file: &Path) -> Option<String> {
    let raw = std::fs::read_to_string(state_file).ok()?;
    let line = raw.lines().next()?.trim();
    if line.is_empty() {
        None
    } else {
        Some(line.to_string())
    }
}

/// Write the active theme name to a state file. Best-effort —
/// failures are silently ignored; the user will simply re-pick the
/// theme on next launch.
pub fn write_persisted(state_file: &Path, name: &str) {
    if let Some(parent) = state_file.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _ = std::fs::write(state_file, format!("{name}\n"));
}

#[derive(Debug, Deserialize, Serialize, Default)]
struct ThemeFile {
    user_prefix_fg: Option<String>,
    ai_prefix_fg: Option<String>,
    keyword_fg: Option<String>,
    string_fg: Option<String>,
    comment_fg: Option<String>,
    header_fg: Option<String>,
    quote_fg: Option<String>,
    bullet_fg: Option<String>,
    dim_fg: Option<String>,
    inline_code_fg: Option<String>,
}

impl ThemeFile {
    fn apply(&self, t: &mut Theme) {
        fn fg(s: &Option<String>) -> Option<Color> {
            s.as_deref().and_then(parse_color)
        }
        if let Some(c) = fg(&self.user_prefix_fg) {
            t.user_prefix = t.user_prefix.fg(c);
        }
        if let Some(c) = fg(&self.ai_prefix_fg) {
            t.ai_prefix = t.ai_prefix.fg(c);
        }
        if let Some(c) = fg(&self.keyword_fg) {
            t.keyword = t.keyword.fg(c);
        }
        if let Some(c) = fg(&self.string_fg) {
            t.string_lit = t.string_lit.fg(c);
        }
        if let Some(c) = fg(&self.comment_fg) {
            t.comment = t.comment.fg(c);
        }
        if let Some(c) = fg(&self.header_fg) {
            t.header = t.header.fg(c);
        }
        if let Some(c) = fg(&self.quote_fg) {
            t.quote = t.quote.fg(c);
        }
        if let Some(c) = fg(&self.bullet_fg) {
            t.bullet = t.bullet.fg(c);
        }
        if let Some(c) = fg(&self.dim_fg) {
            t.dim = t.dim.fg(c);
        }
        if let Some(c) = fg(&self.inline_code_fg) {
            t.inline_code = t.inline_code.fg(c);
        }
    }
}

fn parse_color(s: &str) -> Option<Color> {
    Some(match s.trim().to_ascii_lowercase().as_str() {
        "black" => Color::Black,
        "red" => Color::Red,
        "green" => Color::Green,
        "yellow" => Color::Yellow,
        "blue" => Color::Blue,
        "magenta" | "purple" => Color::Magenta,
        "cyan" => Color::Cyan,
        "gray" | "grey" => Color::Gray,
        "darkgray" | "darkgrey" => Color::DarkGray,
        "lightred" => Color::LightRed,
        "lightgreen" => Color::LightGreen,
        "lightyellow" => Color::LightYellow,
        "lightblue" => Color::LightBlue,
        "lightmagenta" | "lightpurple" => Color::LightMagenta,
        "lightcyan" => Color::LightCyan,
        "white" => Color::White,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn default_is_built_in() {
        let t = Theme::default();
        assert_eq!(t, Theme::built_in_default());
    }

    #[test]
    fn set_by_name_switches_built_in() {
        set_by_name("vivid", None).unwrap();
        assert_eq!(current(), Theme::vivid());
        set_by_name("default", None).unwrap();
        assert_eq!(current(), Theme::built_in_default());
    }

    #[test]
    fn set_by_name_is_case_insensitive() {
        set_by_name("default", None).unwrap();
        set_by_name("VIVID", None).unwrap();
        assert_eq!(current(), Theme::vivid());
        set_by_name("default", None).unwrap();
    }

    #[test]
    fn set_by_name_unknown_returns_err() {
        let err = set_by_name("doesnotexist", None).unwrap_err();
        assert!(err.contains("unknown"));
    }

    #[test]
    fn list_returns_builtins_when_no_dir() {
        let names = list(None);
        assert!(names.contains(&"default".to_string()));
        assert!(names.contains(&"vivid".to_string()));
    }

    #[test]
    fn list_includes_json_files_from_dir() {
        let tmp = TempDir::new().unwrap();
        std::fs::write(tmp.path().join("forest.json"), b"{}").unwrap();
        std::fs::write(tmp.path().join("notes.txt"), b"x").unwrap();
        let names = list(Some(tmp.path()));
        assert!(names.contains(&"forest".to_string()));
        assert!(!names.contains(&"notes".to_string()));
    }

    #[test]
    fn json_file_overrides_colors() {
        let tmp = TempDir::new().unwrap();
        std::fs::write(
            tmp.path().join("custom.json"),
            br#"{ "ai_prefix_fg": "red", "keyword_fg": "green" }"#,
        )
        .unwrap();
        set_by_name("custom", Some(tmp.path())).unwrap();
        let t = current();
        assert_eq!(t.ai_prefix.fg, Some(Color::Red));
        assert_eq!(t.keyword.fg, Some(Color::Green));
        // Reset for other tests.
        set_by_name("default", None).unwrap();
    }

    #[test]
    fn json_file_missing_keys_keep_defaults() {
        let tmp = TempDir::new().unwrap();
        std::fs::write(tmp.path().join("partial.json"), br"{}").unwrap();
        set_by_name("partial", Some(tmp.path())).unwrap();
        let t = current();
        // No overrides → built-in default carried through.
        assert_eq!(t.bold, Theme::built_in_default().bold);
        set_by_name("default", None).unwrap();
    }

    #[test]
    fn json_file_invalid_returns_err() {
        let tmp = TempDir::new().unwrap();
        std::fs::write(tmp.path().join("broken.json"), b"{ this is not json").unwrap();
        let err = set_by_name("broken", Some(tmp.path())).unwrap_err();
        assert!(err.contains("parse"));
    }

    #[test]
    fn parse_color_accepts_aliases() {
        assert_eq!(parse_color("PURPLE"), Some(Color::Magenta));
        assert_eq!(parse_color("grey"), Some(Color::Gray));
        assert_eq!(parse_color("LightCyan"), Some(Color::LightCyan));
        assert_eq!(parse_color("not-a-color"), None);
    }

    #[test]
    fn read_and_write_persisted_round_trip() {
        let tmp = TempDir::new().unwrap();
        let state = tmp.path().join("nested").join("theme.txt");
        write_persisted(&state, "vivid");
        assert_eq!(read_persisted(&state).as_deref(), Some("vivid"));
    }

    #[test]
    fn read_persisted_missing_file_returns_none() {
        let tmp = TempDir::new().unwrap();
        assert_eq!(read_persisted(&tmp.path().join("missing")), None);
    }

    #[test]
    fn themes_dir_for_appends_themes_segment() {
        let p = PathBuf::from("/cfg");
        assert_eq!(themes_dir_for(&p), PathBuf::from("/cfg/themes"));
    }
}
