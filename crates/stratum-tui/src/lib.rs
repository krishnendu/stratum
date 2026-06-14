//! Stratum TUI — backend-agnostic terminal UI primitives.
//!
//! Per `plan/38-tui-architecture-and-gap-fix.md` Phase B. Houses
//! palette catalog, themes, brand constants, and (eventually) the
//! full chat renderer. The CLI binary depends on this crate; it
//! does NOT depend on `stratum-runtime` directly.
//!
//! ## Current scope (this landing)
//!
//! - `palette` — slash-command catalog + filter state
//! - `theme`   — color themes (default + plain + mono + vivid + ocean +
//!               JSON-loaded user themes)
//! - `brand`   — canonical brand constants (colors, spinner frames,
//!               tagline, ASCII wordmark)
//!
//! ## Deferred (next session)
//!
//! - `permission_prompter` — depends on chat.rs and moves with it.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod brand;
/// Chat state, renderer, and event loop. Backend-agnostic via the
/// [`chat::ChatBackend`] trait; the CLI wires `AgentLoop`.
pub mod chat;
pub mod palette;
pub mod theme;
