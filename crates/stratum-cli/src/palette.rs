//! Re-export of the slash-command palette from `stratum-tui`.
//!
//! The palette catalog moved to its own backend-agnostic crate per
//! `plan/38 §Phase B`. This file is the thin shim so existing
//! `crate::palette::*` references in `chat.rs` keep compiling.

pub use stratum_tui::palette::*;
