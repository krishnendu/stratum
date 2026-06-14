//! Stratum TUI — backend-agnostic terminal UI primitives.
//!
//! Per `plan/38-tui-architecture-and-gap-fix.md` Phase B. Houses
//! palette catalog, themes, brand constants, and (eventually) the
//! full chat renderer. The CLI binary depends on this crate; it
//! does NOT depend on `stratum-runtime` directly.
//!
//! ## Workspace-internal — no semver
//!
//! `Cargo.toml` sets `publish = false`. The `chat` module re-exports a
//! broad `pub` surface for the CLI's convenience (via `pub use
//! stratum_tui::chat::*`); that surface is **not** a public API
//! commitment. External consumers must not depend on it.
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
// stratum-tui is a workspace-internal crate (`publish = false`) housing
// the 6800-LOC chat module moved from the CLI. We accept the broader
// pedantic / nursery lint surface here — it would otherwise demand
// hundreds of per-call allows on what is effectively an internal seam.
#![allow(
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::format_push_string,
    clippy::needless_pass_by_value,
    clippy::option_if_let_else,
    clippy::redundant_else,
    clippy::doc_lazy_continuation,
    clippy::doc_overindented_list_items,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::too_long_first_doc_paragraph,
    clippy::needless_continue,
    clippy::type_complexity,
    clippy::or_fun_call,
    clippy::single_match_else,
    clippy::manual_let_else,
    clippy::redundant_clone,
    clippy::no_effect_underscore_binding,
    clippy::unused_self,
    clippy::wildcard_imports,
    clippy::nonminimal_bool,
    clippy::useless_let_if_seq,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::struct_excessive_bools,
    clippy::cognitive_complexity,
    clippy::map_unwrap_or,
    clippy::if_same_then_else,
    clippy::branches_sharing_code,
    clippy::items_after_statements,
    clippy::assigning_clones,
    clippy::struct_field_names,
    clippy::ref_option,
    reason = "workspace-internal crate; chat surface accumulated 6800 LOC of pre-existing patterns"
)]

pub mod brand;
/// Chat state, renderer, and event loop. Backend-agnostic via the
/// [`chat::ChatBackend`] trait; the CLI wires `AgentLoop`.
pub mod chat;
pub mod palette;
pub mod theme;
