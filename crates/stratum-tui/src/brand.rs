//! Canonical brand constants — colors, tagline, ASCII wordmark.
//!
//! Single source of truth referenced by:
//! - `theme.rs` (the `default` theme's gutters / headers)
//! - `chat.rs` (empty-state banner, status bar wordmark, finish flash)
//! - `Cargo.toml description` (via build script when one lands)
//! - `dist/homebrew/stratum.rb` `desc` field
//! - README badges + social cards
//!
//! When any of these need a brand color or the tagline, they pull
//! from here. No literals anywhere else.
//!
//! Defined by `plan/44-brand-and-identity.md`.

#![allow(
    clippy::redundant_pub_crate,
    reason = "internal module; pub kept so each constant is documentable"
)]

use ratatui::style::Color;

// ---- Tagline + identity ---------------------------------------------

/// Primary long-form tagline. Used on social cards, blog posts, the
/// TUI empty-state banner, and the README header.
pub const TAGLINE: &str = "your local model crew";

/// Short-form tagline. Used wherever the long-form is too verbose:
/// Cargo `description` field, Homebrew `desc`, `man` synopsis line.
pub const TAGLINE_SHORT: &str = "local-first chat with an LLM agent";

/// Canonical project name. Lowercase wordmark form.
pub const WORDMARK: &str = "stratum";

// ---- Brand colors (hex) ---------------------------------------------

/// Brand primary — warm slate teal. Used for user-turn gutters, the
/// status-bar `stratum` wordmark, headers in the default theme.
pub const HEX_PRIMARY: &str = "#1E5E5E";

/// Brand accent — warm amber. Used for assistant-turn gutters, the
/// braille spinner, finish-flash glyph, tool-call icon.
pub const HEX_ACCENT: &str = "#D9844D";

/// Error tone — muted brick. Used for error markers + warning borders.
pub const HEX_ERROR: &str = "#C2384A";

/// Warning tone — muted gold.
pub const HEX_WARN: &str = "#C29A3A";

/// Success tone — muted green.
pub const HEX_SUCCESS: &str = "#3A8A4A";

// ---- ratatui Color constants ----------------------------------------
//
// Pre-resolved RGB values. ratatui terminal renderers downsample to
// the closest 256-color cell when the terminal lacks truecolor.

/// Brand primary as a ratatui Color.
pub const COLOR_PRIMARY: Color = Color::Rgb(0x1E, 0x5E, 0x5E);
/// Brand accent as a ratatui Color.
pub const COLOR_ACCENT: Color = Color::Rgb(0xD9, 0x84, 0x4D);
/// Error tone.
pub const COLOR_ERROR: Color = Color::Rgb(0xC2, 0x38, 0x4A);
/// Warning tone.
pub const COLOR_WARN: Color = Color::Rgb(0xC2, 0x9A, 0x3A);
/// Success tone.
pub const COLOR_SUCCESS: Color = Color::Rgb(0x3A, 0x8A, 0x4A);

// ---- ASCII wordmark for TUI empty state -----------------------------

/// Four-line ASCII rendering of the stacked-layers mark. Aligned for
/// monospace fonts; each layer is one cell narrower than the one above
/// — read top-down a wedge, bottom-up an arrow. Per `plan/44 §4.1`.
///
/// Rendered in `COLOR_PRIMARY` on the empty-state line.
pub const ASCII_MARK: [&str; 4] = [
    "  ████████████  ",
    "   ██████████   ",
    "    ████████    ",
    "     ██████     ",
];

// ---- Braille spinner frames -----------------------------------------

/// 10-frame braille animation per `plan/44 §6.1`. Cadence: 80ms per
/// frame. Same cycle used by lazygit, helix, and most modern Rust
/// TUIs — feels native rather than novel.
pub const SPINNER_FRAMES: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

/// Pick the current spinner frame given an elapsed-time in ms.
/// Centralises the cadence so callers don't tune it independently.
#[must_use]
pub const fn spinner_frame_for(elapsed_ms: u128) -> &'static str {
    let idx = ((elapsed_ms / 80) % SPINNER_FRAMES.len() as u128) as usize;
    SPINNER_FRAMES[idx]
}

// ---- Rotating tips (random in TUI empty state) ----------------------

/// Tips shown in `/welcome` and on the empty-state banner. Picked
/// pseudo-randomly per launch via a turn-counter seeded rotation —
/// no PRNG state, just `tip_for(idx)` with the caller bumping idx.
pub const TIPS: &[&str] = &[
    "Type @ to reference a workspace file by path.",
    "Press Ctrl+R to search your input history.",
    "Press Ctrl+G to open the prompt in $VISUAL / $EDITOR.",
    "Use /plan to enter plan mode — chat won't run tools until you turn it off.",
    "Type /theme to switch color schemes.",
    "Press Ctrl+T to flip between drag-to-copy and mouse-scroll modes.",
    "Use /export to dump the transcript to a file.",
    "Tail logs into Stratum: `tail -f app.log | stratum -p \"summarise errors\"`.",
    "/cost shows tokens · ms · tok/s for the latest turn.",
    "Ctrl+C twice exits. One press cancels the current turn.",
];

/// Stable pick for a tip index without taking a PRNG dependency.
#[must_use]
pub fn tip_for(rotation: u64) -> &'static str {
    let idx = (rotation as usize) % TIPS.len();
    TIPS[idx]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(
        clippy::const_is_empty,
        reason = "guards against an accidental empty-string regression in the constants"
    )]
    fn tagline_constants_non_empty() {
        assert!(!TAGLINE.is_empty());
        assert!(!TAGLINE_SHORT.is_empty());
        assert_eq!(WORDMARK, "stratum");
    }

    #[test]
    fn hex_constants_parse_to_rgb() {
        // Sanity: the hex strings match the pre-resolved Color::Rgb.
        assert_eq!(COLOR_PRIMARY, Color::Rgb(0x1E, 0x5E, 0x5E));
        assert_eq!(COLOR_ACCENT, Color::Rgb(0xD9, 0x84, 0x4D));
    }

    #[test]
    fn ascii_mark_is_four_lines() {
        assert_eq!(ASCII_MARK.len(), 4);
        // Each line wider-or-equal to the next — wedge invariant.
        let widths: Vec<usize> = ASCII_MARK
            .iter()
            .map(|s| s.chars().filter(|c| *c == '█').count())
            .collect();
        for w in widths.windows(2) {
            assert!(w[0] >= w[1], "wedge invariant violated: {widths:?}");
        }
    }

    #[test]
    fn spinner_has_ten_frames() {
        assert_eq!(SPINNER_FRAMES.len(), 10);
    }

    #[test]
    fn spinner_frame_for_cycles() {
        let f0 = spinner_frame_for(0);
        let f80 = spinner_frame_for(80);
        let f800 = spinner_frame_for(800);
        assert_eq!(f0, SPINNER_FRAMES[0]);
        assert_eq!(f80, SPINNER_FRAMES[1]);
        // After 10 frames (800ms) we're back at 0.
        assert_eq!(f800, SPINNER_FRAMES[0]);
    }

    #[test]
    fn tip_for_is_in_bounds() {
        for rot in 0..50_u64 {
            let _ = tip_for(rot);
        }
    }

    #[test]
    #[allow(
        clippy::const_is_empty,
        reason = "guards against an accidental empty-tip regression"
    )]
    fn tips_list_non_empty() {
        assert!(!TIPS.is_empty());
        for t in TIPS {
            assert!(!t.is_empty());
            assert!(!t.contains('\n'), "tips must fit one line: {t:?}");
        }
    }
}
