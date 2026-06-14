//! Re-export of the chat module from `stratum-tui`.
//!
//! The full `ChatState` + render loop lives in `stratum-tui::chat`. The
//! CLI keeps a glob re-export so existing `crate::chat::*` call sites
//! (e.g. `app::run_chat_with_loop`) keep compiling without churn.

#[allow(
    unreachable_pub,
    reason = "binary crate has no external API; pub kept for crate::chat::* call sites"
)]
pub use stratum_tui::chat::*;
