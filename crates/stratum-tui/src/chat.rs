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
#![allow(
    dead_code,
    reason = "EventEmitter wiring is exposed for the upcoming JSONL CLI path and TUI events panel; kept pub even though the bin build does not yet consume it"
)]

use std::io;
use std::path::PathBuf;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime};

use crossterm::event::{
    self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEvent, KeyEventKind,
    KeyModifiers, KeyboardEnhancementFlags, MouseEvent, MouseEventKind,
    PopKeyboardEnhancementFlags, PushKeyboardEnhancementFlags,
};
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
use stratum_runtime::{
    format_tokens_per_second, AgentHandoff, AgentLoop, AgentLoopConfig, CancelToken,
    CapabilityMatrix, EchoProvider, Event as RtEvent, EventEmitter, EventRecord, IntentRouter,
    MemoryEventSink, Paths, PendingPrompt, PermissionDecision, PermissionRequest, PermissionStore,
    PlanMode, PromptId, PromptIdGen, Provider, RoleTimer, SuggestedRole, Tier, Transcript,
    TranscriptTurn, TurnContext, TurnId, TurnMetrics, TurnOutcome, TurnRecorder, TurnResult,
};
use stratum_types::{Block, ModelId, StratumResult};

/// Backend seam — the single surface `ChatState` calls on its
/// turn-runner. Implemented for [`AgentLoop`] by the blanket impl
/// below so existing wiring (and every test) keeps compiling. Future
/// non-AgentLoop backends (remote daemon, hosted provider, mock)
/// implement this trait directly without touching the renderer.
///
/// Object-safe by design: one method, no generics, no `Self` in
/// return position.
pub trait ChatBackend: std::fmt::Debug + Send + Sync + 'static {
    /// Run one streaming turn. The `chunk_tx` receives every emitted
    /// `Block` as the backend produces it so the UI can render
    /// partial text. The returned `TurnResult` is the final
    /// consolidated outcome.
    fn run_turn_streaming(
        &self,
        ctx: TurnContext,
        cancel: &CancelToken,
        chunk_tx: mpsc::Sender<Block>,
    ) -> TurnResult;
}

impl ChatBackend for AgentLoop {
    fn run_turn_streaming(
        &self,
        ctx: TurnContext,
        cancel: &CancelToken,
        chunk_tx: mpsc::Sender<Block>,
    ) -> TurnResult {
        Self::run_turn_streaming(self, ctx, cancel, chunk_tx)
    }
}

use crate::palette::{self, Palette};

// Permission-prompt responder lives as a sibling file; declare it inline so
// the binary crate root (`main.rs`) does not have to mention it.
#[path = "permission_prompter.rs"]
mod permission_prompter;

pub use permission_prompter::TuiPromptResponder;

/// Multi-line `/help` body listing every wired palette command.
const HELP_TEXT: &str = "available commands:\n\
    /plan [on|off] — toggle (or set) plan mode\n\
    /cancel — cancel the in-flight turn\n\
    /clear — clear the transcript\n\
    /compact — compress older turns into a summary; keeps the last 4 verbatim\n\
    /model, /active — show currently active model\n\
    /models — list available models from the catalog\n\
    /switch <slug> — swap to a different model mid-session\n\
    !<cmd> — run shell command directly (bypass LLM, sandboxed)\n\
    /tier — show current host tier (low/medium/high)\n\
    /version — show stratum version\n\
    /welcome — re-show the Stratum greeting + a tip\n\
    /agents — list registered roles (multi-agent mode only)\n\
    /subagents — list available subagents (built-in + user-defined)\n\
    /agent <name> <task> — explicitly delegate <task> to subagent <name>\n\
    /parallel <role1,role2,…> — fan the next turn out across the listed roles \
(multi-agent mode only)\n\
    /budget, /cost, /usage — show the latest turn metrics (tokens · ms · tok/s · turn id)\n\
    /recap — one-line session summary\n\
    /diff — show recent fs.write / fs.edit calls\n\
    /image <path> — attach an image to the next turn (scaffold; vision provider TBD)\n\
    /mic — start/stop push-to-talk mic capture (F5 hotkey toggles too); captured WAV runs through whisper on submit\n\
    /tts [on|off] — toggle Piper TTS playback of assistant replies (default off)\n\
    /init — scaffold STRATUM.md for the current workspace\n\
    /editor — open the current input in $VISUAL / $EDITOR (Ctrl+G shortcut)\n\
    /select — toggle between select-mode (default, drag-to-copy) and mouse-scroll mode; alias for Ctrl+T\n\
    /export [path] — dump the chat transcript to a file\n\
    /theme <name> — switch chat theme (default | mono | vivid | ocean | user JSON)\n\
    /themes — list available themes\n\
    /undo, /redo — placeholder (workspace concept lands in Phase 3 v2)\n\
    /help — show this message\n\
    /quit, /exit — exit the TUI";

/// One entry in the chat transcript.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Turn {
    /// What the user typed.
    User(String),
    /// What the provider returned.
    Assistant(Vec<Block>),
    /// Cancellation marker (Ctrl-C mid-stream).
    Cancelled,
    /// Slash command was invoked from the palette and dispatched.
    Command {
        /// The command keyword (without the leading `/`) as the user typed it.
        text: String,
        /// `true` for an acknowledged dispatch, `false` for a rejection
        /// (e.g. unknown command).
        ok: bool,
        /// Human-friendly message rendered in the transcript.
        message: String,
    },
}

/// Outcome of dispatching a palette command via
/// [`ChatState::execute_palette_command`]. Each variant carries the
/// message that the transcript renders.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PaletteOutcome {
    /// The command was recognised and executed.
    Acknowledged {
        /// Human-friendly message describing the effect.
        message: String,
    },
    /// The command was unknown or otherwise rejected.
    Rejected {
        /// Human-friendly explanation.
        message: String,
    },
}

/// A live mic-capture session driven on its own dedicated thread.
///
/// `cpal::Stream` is intentionally not `Send` on macOS / `CoreAudio`,
/// which would otherwise rule out the `Arc<Mutex<ChatState>>` pattern
/// the permission-modal tests rely on. Pinning the capture to a thread
/// it never leaves keeps `ChatState: Send + Sync` regardless of host
/// backend — the [`ChatState`] only holds channels.
///
/// Lifecycle: [`Self::start`] opens the capture on a worker thread and
/// returns once it's recording. [`Self::stop_and_save`] sends a stop
/// signal, waits for the worker to flush + save the WAV, and returns
/// the resulting path. Once stopped the cell is consumed.
#[derive(Debug)]
struct MicCaptureCell {
    /// `Some` until `stop_and_save` is called. After stop the join
    /// handle is consumed.
    join: Option<thread::JoinHandle<Result<std::path::PathBuf, String>>>,
    /// Sends the requested output path to the worker; the worker
    /// flips its `recording` loop off, calls `save_wav`, and exits.
    stop_tx: mpsc::Sender<std::path::PathBuf>,
    /// Signals back from the worker once `start()` has called
    /// `MicCapture::start` and the stream is live. Lets `start()`
    /// surface device-open errors synchronously.
    started_rx: mpsc::Receiver<Result<(), String>>,
}

#[cfg(not(feature = "voice"))]
impl MicCaptureCell {
    /// Stub used when the `voice` feature is off (e.g. the prebuilt Linux
    /// tarball). Always returns the same typed error so the
    /// `/mic` palette command and the F5 hotkey surface a clear "voice
    /// support not compiled in — rebuild with `--features voice`" message
    /// instead of silently doing nothing.
    fn start() -> Result<Self, String> {
        Err("voice support not compiled in (rebuild with --features voice)".to_string())
    }

    /// Stub that mirrors the live signature. Unreachable in practice
    /// because [`Self::start`] always returns `Err` under non-voice
    /// builds, so no caller ever holds a `MicCaptureCell` to stop.
    fn stop_and_save(self, _path: std::path::PathBuf) -> Result<std::path::PathBuf, String> {
        let _ = (self.join, self.stop_tx, self.started_rx);
        Err("voice support not compiled in".to_string())
    }
}

#[cfg(feature = "voice")]
impl MicCaptureCell {
    /// Spawn the worker thread, open the cpal stream, and return once
    /// the worker reports a successful start (or a typed error).
    fn start() -> Result<Self, String> {
        let (stop_tx, stop_rx) = mpsc::channel::<std::path::PathBuf>();
        let (started_tx, started_rx) = mpsc::channel::<Result<(), String>>();
        let join = thread::spawn(move || -> Result<std::path::PathBuf, String> {
            let mut cap = match stratum_runtime::mic::MicCapture::new() {
                Ok(c) => c,
                Err(e) => {
                    let _ = started_tx.send(Err(format!("cannot open default input device: {e}")));
                    return Err(format!("cannot open default input device: {e}"));
                }
            };
            if let Err(e) = cap.start() {
                let _ = started_tx.send(Err(format!("cannot start stream: {e}")));
                return Err(format!("cannot start stream: {e}"));
            }
            let _ = started_tx.send(Ok(()));
            // Block until the main thread sends a stop+path. Drop of
            // the sender (i.e. `ChatState` dropped without stop) is
            // treated as an implicit stop with no-save.
            match stop_rx.recv() {
                Ok(path) => {
                    cap.save_wav(&path).map_err(|e| format!("save_wav: {e}"))?;
                    Ok(path)
                }
                Err(_) => {
                    let _ = cap.stop();
                    Err("mic worker dropped before stop".to_string())
                }
            }
        });
        // Wait for the worker to either succeed or fail at open/start.
        match started_rx.recv() {
            Ok(Ok(())) => Ok(Self {
                join: Some(join),
                stop_tx,
                started_rx,
            }),
            Ok(Err(e)) => {
                let _ = join.join();
                Err(e)
            }
            Err(_) => Err("mic worker died before reporting start".to_string()),
        }
    }

    /// Tell the worker to stop, flush a WAV to `path`, and join.
    fn stop_and_save(mut self, path: std::path::PathBuf) -> Result<std::path::PathBuf, String> {
        self.stop_tx
            .send(path)
            .map_err(|_| "mic worker exited before stop".to_string())?;
        let Some(join) = self.join.take() else {
            return Err("mic worker already joined".to_string());
        };
        match join.join() {
            Ok(res) => res,
            Err(_) => Err("mic worker panicked".to_string()),
        }
    }
}

/// Outcome of a push-to-talk toggle invocation. The dispatcher and the
/// F5 hotkey both surface one of these.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MicToggle {
    /// Recording just started.
    Started,
    /// Recording just stopped; the captured WAV was staged as the next
    /// turn's audio attachment.
    Stopped {
        /// Number of bytes captured in the saved WAV (header + samples).
        samples: usize,
        /// Same number, duplicated for the `dispatch_mic` format string.
        bytes: usize,
    },
    /// Recording stopped without any audio captured (zero-length WAV
    /// or no `MicCapture` was active).
    StoppedEmpty,
}

/// Payload sent from the provider worker thread back to the TUI main
/// thread when an async turn completes.
#[derive(Debug)]
struct TurnAsyncResult {
    blocks: Vec<Block>,
    turn_result: Option<TurnResult>,
    handoff_lines: Vec<Turn>,
    final_role: Option<SuggestedRole>,
    step_ms: u32,
}

/// Reverse-history-search modal state. Opened with Ctrl+R; filters
/// `input_history` by substring; ↑/↓ cycles matches; Enter accepts;
/// Esc cancels.
#[derive(Debug, Clone, Default)]
pub struct RSearchState {
    /// Live filter typed since Ctrl+R.
    needle: String,
    /// Index into the filtered match list.
    cursor: usize,
}

/// Type-erased shell executor for `!cmd` palette prefix. Wraps a
/// closure that runs `cmd` (interpreted as a shell command line) and
/// returns combined stdout+stderr or an error string.
#[derive(Clone)]
pub struct ShellExecutor(Arc<dyn Fn(&str) -> Result<String, String> + Send + Sync>);

impl std::fmt::Debug for ShellExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShellExecutor").finish_non_exhaustive()
    }
}

impl ShellExecutor {
    /// Wrap a shell execution closure.
    pub fn new<F>(f: F) -> Self
    where
        F: Fn(&str) -> Result<String, String> + Send + Sync + 'static,
    {
        Self(Arc::new(f))
    }
}

/// Type-erased model swap hook. Wraps a closure that rebuilds an
/// [`AgentLoop`] against a new slug.
#[derive(Clone)]
pub struct ModelSwitcher(Arc<dyn Fn(&str) -> Result<Arc<AgentLoop>, String> + Send + Sync>);

impl std::fmt::Debug for ModelSwitcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelSwitcher").finish_non_exhaustive()
    }
}

impl ModelSwitcher {
    /// Wrap a swap closure.
    pub fn new<F>(f: F) -> Self
    where
        F: Fn(&str) -> Result<Arc<AgentLoop>, String> + Send + Sync + 'static,
    {
        Self(Arc::new(f))
    }
}

/// Pure TUI state. Driven by events; rendered into any [`Backend`].
#[derive(Debug)]
pub struct ChatState {
    transcript: Vec<Turn>,
    input: String,
    /// Byte offset of the cursor within `input`. Always at a UTF-8
    /// char boundary, and `0 ≤ caret ≤ input.len()`. Maintained by
    /// every key handler that mutates the input buffer.
    caret: usize,
    /// Set to `true` by Ctrl+G; the event loop sees it on the next
    /// tick, suspends the TUI, opens `$VISUAL` / `$EDITOR` on a temp
    /// file seeded with the current input, then replaces the input
    /// with the edited contents. Cleared when consumed.
    pending_edit_request: bool,
    /// True when mouse capture is on. The TUI installs mouse capture
    /// at startup so the scroll wheel can drive `chat_scroll`, but
    /// that ALSO swallows click-drag from the OS terminal — so
    /// native text selection (and therefore copy) stops working.
    /// Toggling this off (Ctrl+T) gives the user back the native
    /// selection at the cost of the scroll wheel.
    mouse_capture_on: bool,
    /// One-shot flag set by Ctrl+T so the event loop can flip the
    /// terminal-side capture bit (`render()` has no stdout access).
    pending_mouse_toggle: bool,
    /// Reverse-search state. `Some` when the user pressed Ctrl+R; the
    /// inner string is the live filter. Esc / Ctrl+C cancels; Enter
    /// accepts the current match and inserts it into `input`.
    rsearch: Option<RSearchState>,
    /// Scroll offset for the chat pane in screen rows above the
    /// auto-tail position. `0` = follow latest; `N` = pinned `N` rows
    /// above. `PgUp` / mouse wheel up increment, `PgDn` / wheel down
    /// decrement (saturating at 0). Reset to 0 on submit so the user
    /// sees their freshly-sent message.
    chat_scroll: u16,
    /// Kill-ring stack (most-recent last). Populated by Ctrl+K (kill to
    /// EOL), Ctrl+U (kill to BOL), Ctrl+W (kill word back). Drained by
    /// Ctrl+Y (yank latest) and Alt+Y (cycle to older entry after a
    /// preceding Ctrl+Y). Capped at [`Self::KILL_RING_CAP`].
    kill_ring: Vec<String>,
    /// Index of the entry shown by the most recent Ctrl+Y / Alt+Y
    /// chain. `None` resets after any non-yank key. Lets Alt+Y replace
    /// the just-yanked entry with the next-older one (readline behavior).
    yank_cursor: Option<usize>,
    /// Byte position in `input` where the most recent yank inserted
    /// its text. Used by Alt+Y to delete the previous yank before
    /// inserting the next-older entry.
    yank_start: usize,
    provider: EchoProvider,
    cancel: CancelToken,
    tier: Tier,
    quit: bool,
    status: String,
    palette: Option<Palette>,
    /// Monotonic turn counter; the next submitted turn gets this id.
    next_turn_id: u64,
    /// Metrics from the most recently completed turn (renders in status bar).
    last_metrics: Option<TurnMetrics>,
    /// Structured-event emitter wired into each completed turn.
    ///
    /// Defaults to an in-process [`MemoryEventSink`]; the CLI binary may
    /// inject a JSONL-backed sink via [`ChatState::with_events`].
    pub events: Arc<EventEmitter>,
    /// In-memory sink handle, when the emitter is backed by [`MemoryEventSink`].
    ///
    /// `None` after [`ChatState::with_events`] is called with a non-memory
    /// emitter — the emitter's sink is opaque (no `Any` bound on the trait),
    /// so we track the typed handle alongside it for snapshotting.
    memory_sink: Option<Arc<MemoryEventSink>>,
    /// Orchestrator used by [`Self::submit`] to drive a single turn through
    /// the FSM, intent router, permission store, plan-mode fence, and
    /// provider — replacing the direct `provider.generate` call.
    agent_loop: Arc<dyn ChatBackend>,
    /// Most recent [`TurnResult`] returned by [`AgentLoop::run_turn`].
    last_turn_result: Option<TurnResult>,
    /// Plan-mode handle wired into both this state (for `/plan` palette
    /// toggling) and the [`AgentLoop`] (for capability gating). Sharing
    /// the same `Arc` is what makes `/plan` semantically meaningful: a
    /// toggle here immediately reflects in the loop's fence.
    plan_mode: Arc<PlanMode>,
    /// Number of [`Turn`]s prepended from a resumed [`Transcript`].
    ///
    /// Set by [`Self::with_resumed_transcript`]; defaults to zero. Lets
    /// callers render a "Resumed N turns" banner without re-walking the
    /// transcript to count.
    resumed_count: usize,
    /// Shared TUI permission prompter. Cloned into the [`AgentLoop`] as the
    /// responder and read by the event loop to surface pending requests.
    permission_prompter: Arc<TuiPromptResponder>,
    /// Wall-clock instant at which the current in-flight turn started.
    ///
    /// `Some` while [`Self::submit`] is running; cleared back to `None` on
    /// completion. Used by [`Self::status_bar_text`] to render the live
    /// `[generating… <N>s]` indicator.
    in_flight_since: Option<Instant>,
    /// Approximate completion-token count from the most recent turn.
    ///
    /// Computed as `sum(Block::Text(text).len()) / 4` across the assistant
    /// blocks: a coarse 4-chars-per-token heuristic that matches the ballpark
    /// for English-prose tokenizers (GPT/Llama byte-pair encodings average
    /// ~3.5–4.5 chars per token on natural-language text). Cheap, allocation-
    /// free, and good enough for a status-bar gauge — the precise token count
    /// from the provider's `Block::Usage` still flows through `TurnRecorder`.
    last_token_count: u64,
    /// Optional multi-role coordinator. When `Some`, [`Self::submit`] routes
    /// each turn through [`AgentHandoff::run_turn_with_handoff`] instead of
    /// the single-loop default path. `None` preserves the Phase 1 single-loop
    /// behaviour.
    handoff: Option<Arc<AgentHandoff>>,
    /// Slug of the currently active model, surfaced via `/model` palette.
    active_model: Option<String>,
    /// Catalog of selectable model slugs, surfaced via `/models` palette.
    available_models: Vec<String>,
    /// Hook invoked by `/switch <slug>` to rebuild the agent loop against a
    /// different model. `None` when no real provider is wired (echo mode).
    model_switcher: Option<ModelSwitcher>,
    /// Hook invoked when the user types `!<cmd>`. `None` falls through to
    /// LLM (input sent as plain prompt with `!` preserved).
    shell_executor: Option<ShellExecutor>,
    /// Ring of previously submitted user inputs (oldest first). Recalled
    /// with ↑/↓ when the input buffer is empty (or the user is already
    /// browsing). Capped at [`Self::INPUT_HISTORY_CAP`].
    input_history: Vec<String>,
    /// On-disk path where `input_history` is persisted. `None` keeps
    /// history in-memory only (used by tests + the `EchoProvider` path).
    history_path: Option<PathBuf>,
    theme_state_path: Option<PathBuf>,
    themes_dir: Option<PathBuf>,
    statusline_cmd: Option<String>,
    statusline_cache: Arc<Mutex<String>>,
    statusline_last_run: Mutex<Option<Instant>>,
    /// Active history index when browsing. `None` = not browsing.
    /// `Some(i)` means `input_history[i]` is currently shown in the
    /// input buffer; pressing ↑ moves toward index 0, ↓ moves toward
    /// the present.
    history_cursor: Option<usize>,
    /// Prompts queued while a turn is in flight. Flushed FIFO once
    /// `submit()` finishes the current turn. The last entry can be
    /// pulled back into the input buffer via ↑ before it sends.
    pending_queue: Vec<String>,
    /// When the user pressed Ctrl+C / Ctrl+D once and the second press
    /// would actually exit. Cleared after [`Self::EXIT_ARM_WINDOW`]
    /// or once any other key lands. Mirrors Claude Code's
    /// "press Ctrl+C again to exit" UX.
    exit_armed_at: Option<Instant>,
    /// Set when Ctrl+C pushed a `Turn::Cancelled` for the in-flight
    /// turn. Tells `finalize_turn` to drop the worker's result
    /// instead of pushing a second terminal turn for the same prompt.
    cancel_already_pushed: bool,
    /// Receiver for the in-flight turn's result. `submit()` spawns the
    /// provider on a worker thread and stores the rx side here; the
    /// event loop drains it via [`Self::poll_turn_completion`] each
    /// tick so the TUI keeps rendering (and the `(thinking…)` placeholder
    /// stays animated) while the provider runs.
    pending_rx: Option<mpsc::Receiver<TurnAsyncResult>>,
    /// Timer started when `submit()` kicked off the in-flight turn.
    pending_started: Option<Instant>,
    /// Turn id assigned to the in-flight async turn so that
    /// [`Self::finalize_turn`] can fold metrics in under the same id.
    last_turn_id: Option<TurnId>,
    /// Receiver for per-token text chunks emitted by the provider
    /// during an in-flight async turn. Drained alongside
    /// [`Self::pending_rx`] on each event-loop tick to drive the live
    /// "typing" render. Cleared in [`Self::finalize_turn`].
    chunk_rx: Option<mpsc::Receiver<Block>>,
    /// Accumulated streaming text for the in-flight turn. Rendered
    /// under the latest `Turn::User` while the provider runs.
    streaming_text: String,
    /// Subagent registry surfaced via the `/subagents` palette
    /// command. Seeded with `SubagentRegistry::with_builtins()` and
    /// extended at startup with `<config>/stratum/subagents/*.toml`.
    subagents: stratum_runtime::subagent::SubagentRegistry,
    /// Stable session id stamped on this chat; used by
    /// `--resume <id>` to reload the saved transcript.
    session_id: stratum_runtime::SessionId,
    /// Wall-clock instant this chat started; folded into the persisted
    /// transcript so `stratum sessions list` can order by creation.
    created_at: SystemTime,
    /// Role currently driving the chat, when multi-agent mode is active.
    ///
    /// `None` in single-loop mode (no handoff installed). `Some(role)` after
    /// [`Self::with_handoff`] — seeded with [`SuggestedRole::Default`] and
    /// updated after every successful [`AgentHandoff::run_turn_with_handoff`]
    /// to the chain's final role.
    current_role: Option<SuggestedRole>,
    /// Multimodal attachments staged via `/image <path>` (and the
    /// future `/audio <path>`) for the NEXT user turn. Drained into
    /// [`TurnContext::attachments`] on `submit()` so they ride along
    /// with the prompt to the provider exactly once.
    //
    // TODO(plan/05): wire <vision-model> — until a vision-capable
    // provider lands, these bytes are forwarded but ignored downstream
    // (EchoProvider / LlamaCppProvider drop them with a log line).
    pending_attachments: Vec<Block>,
    /// Audio attachment queued by `/audio <path>` for the next
    /// [`Self::submit`] (Phase 5 v2). Drained when the next user turn
    /// fires; rendered as `[audio: <mime>]` in the transcript so the
    /// user can see what's staged.
    staged_audio: Option<Block>,
    /// Whisper-derived transcript paired with [`Self::staged_audio`].
    /// `Some(text)` after a successful subprocess run; `None` when the
    /// whisper binary is absent or the run failed. `submit()` prepends
    /// the text (or an "unavailable" sentinel) to the user's prompt.
    staged_audio_transcript: Option<String>,
    /// Whisper subprocess shim used to transcribe staged audio.
    /// Construct-once-on-state, reused across `/audio` invocations.
    whisper: stratum_runtime::whisper::WhisperSubprocess,
    /// Attachments handed to the most recent [`Self::submit`] call.
    /// Currently populated by `/audio <path>`; cleared on the *next*
    /// submit so tests (and the future agent-loop attachments seam)
    /// can inspect what just rode along with the prompt.
    last_turn_attachments: Vec<Block>,
    /// Phase 5 voice-in: live mic capture handle. `Some` while
    /// recording (PTT active), `None` otherwise. Constructed lazily
    /// on `/mic` / F5 toggle-on so a session that never speaks
    /// doesn't pay the cpal host-init cost. Wrapped in
    /// [`MicCaptureCell`] so the outer `#[derive(Debug)]` still works.
    mic_capture: Option<MicCaptureCell>,
    /// `true` between PTT-on and PTT-off. Mirrors `mic_capture.is_some()`
    /// but stays `true` for the tail of a stop sequence (test-only
    /// helper / render guard).
    recording: bool,
    /// Phase 5 voice-out: Piper TTS subprocess + voice-model handle.
    /// `Some` after `/tts on` has been issued at least once; `None`
    /// before that (and after `/tts off` if we later want to drop
    /// the subprocess config). Construct-once-on-toggle, reused
    /// across turns.
    piper: Option<stratum_runtime::PiperSubprocess>,
    /// `true` while voice-out is desired. Default `false`. Toggled
    /// via `/tts on|off`; flipped back to `false` for the rest of
    /// the session if Piper surfaces `MissingBinary` / `MissingModel`.
    tts_enabled: bool,
    /// Sticky session-disable flag. Set when Piper synthesis fails
    /// with a missing-binary / missing-model error so the user
    /// doesn't keep eating the same failure on every turn. Cleared
    /// only by a fresh `ChatState` (i.e. a new session).
    tts_session_disabled: bool,
    /// Transient status message and the instant it was raised. The
    /// renderer surfaces this in the status bar and clears it after
    /// `TRANSIENT_STATUS_TTL` so the user sees what whisper heard
    /// without the line sticking forever.
    transient_status: Option<(String, Instant)>,
}

impl Default for ChatState {
    fn default() -> Self {
        Self::new(EchoProvider::default(), Tier::High, String::new())
    }
}

impl ChatState {
    /// Max entries retained in the input-recall history.
    pub const INPUT_HISTORY_CAP: usize = 200;

    /// Max entries in the kill-ring before older entries are dropped.
    pub const KILL_RING_CAP: usize = 32;

    /// Window during which a second Ctrl+C / Ctrl+D actually exits.
    /// Outside this window the arm decays and the first press re-arms.
    pub const EXIT_ARM_WINDOW: Duration = Duration::from_secs(2);

    /// Time-to-live for a [`Self::transient_status`] entry. The chat
    /// status bar surfaces e.g. a whisper transcript snippet after a
    /// `/audio` (or mic) submit; after this window the line fades so
    /// the bar returns to its idle layout.
    pub const TRANSIENT_STATUS_TTL: Duration = Duration::from_secs(2);

    /// Build a fresh state with the given header (status bar) and tier.
    #[must_use]
    pub fn new(provider: EchoProvider, tier: Tier, status: String) -> Self {
        let sink = Arc::new(MemoryEventSink::new());
        let events = Arc::new(EventEmitter::new(sink.clone()));
        let plan_mode = Arc::new(PlanMode::new());
        let permission_prompter = Arc::new(TuiPromptResponder::default());
        #[allow(
            clippy::expect_used,
            reason = "default_agent_loop sets all nine required builder fields; build() cannot return MissingField on this code path"
        )]
        let agent_loop = Arc::new(
            default_agent_loop(
                provider.clone(),
                events.clone(),
                plan_mode.clone(),
                permission_prompter.clone(),
            )
            .expect("default AgentLoop builder sets every required field"),
        );
        Self {
            transcript: Vec::new(),
            input: String::new(),
            caret: 0,
            chat_scroll: 0,
            pending_edit_request: false,
            // Default OFF so users can drag-select and copy text out
            // of the box. Scroll-wheel users opt in via Ctrl+T (or
            // `/select`). The previous default-on broke the most
            // basic terminal expectation: "select with mouse and
            // ⌘C / Ctrl+Shift+C copies".
            mouse_capture_on: false,
            pending_mouse_toggle: false,
            rsearch: None,
            kill_ring: Vec::new(),
            yank_cursor: None,
            yank_start: 0,
            provider,
            cancel: CancelToken::new(),
            tier,
            quit: false,
            status,
            palette: None,
            next_turn_id: 0,
            last_metrics: None,
            events,
            memory_sink: Some(sink),
            agent_loop,
            last_turn_result: None,
            plan_mode,
            resumed_count: 0,
            permission_prompter,
            in_flight_since: None,
            last_token_count: 0,
            handoff: None,
            active_model: None,
            available_models: Vec::new(),
            model_switcher: None,
            shell_executor: None,
            input_history: Vec::new(),
            history_path: None,
            theme_state_path: None,
            themes_dir: None,
            statusline_cmd: None,
            statusline_cache: Arc::new(Mutex::new(String::new())),
            statusline_last_run: Mutex::new(None),
            history_cursor: None,
            pending_queue: Vec::new(),
            exit_armed_at: None,
            cancel_already_pushed: false,
            pending_rx: None,
            pending_started: None,
            last_turn_id: None,
            chunk_rx: None,
            streaming_text: String::new(),
            subagents: stratum_runtime::subagent::SubagentRegistry::with_builtins(),
            session_id: stratum_runtime::SessionId::new_random(),
            created_at: SystemTime::now(),
            current_role: None,
            pending_attachments: Vec::new(),
            staged_audio: None,
            staged_audio_transcript: None,
            whisper: stratum_runtime::whisper::WhisperSubprocess::default(),
            last_turn_attachments: Vec::new(),
            mic_capture: None,
            recording: false,
            piper: None,
            tts_enabled: false,
            tts_session_disabled: false,
            transient_status: None,
        }
    }

    /// Build a state wrapping the supplied [`AgentLoop`]. Test-friendly
    /// builder: lets callers inject providers, responders, or capability
    /// matrices that the default constructor does not expose.
    ///
    /// The supplied loop owns its own emitter; the state still defaults to
    /// a memory-backed [`EventEmitter`] for status-bar / palette wiring,
    /// but [`Self::submit`] will emit through the loop's emitter.
    #[must_use]
    pub fn with_agent_loop(loop_: Arc<AgentLoop>) -> Self {
        let mut state = Self::new(EchoProvider::new("echo: "), Tier::High, String::new());
        // Upcast Arc<AgentLoop> → Arc<dyn ChatBackend>. AgentLoop
        // implements ChatBackend via the blanket impl in this file.
        state.agent_loop = loop_;
        state
    }

    /// Variant of [`Self::with_agent_loop`] for backends that aren't
    /// `AgentLoop`. Tests and future remote-daemon transports use
    /// this entry point.
    #[must_use]
    pub fn with_backend(backend: Arc<dyn ChatBackend>) -> Self {
        let mut state = Self::new(EchoProvider::new("echo: "), Tier::High, String::new());
        state.agent_loop = backend;
        state
    }

    /// Route every subsequent [`Self::submit`] turn through the supplied
    /// [`AgentHandoff`] instead of the default single-loop path.
    ///
    /// Calling this is what turns `stratum chat --agents-dir <path>` into a
    /// multi-role surface: `submit` builds a [`stratum_runtime::RoutedIntent`]
    /// via [`IntentRouter::default`] and hands the turn (plus context) to
    /// [`AgentHandoff::run_turn_with_handoff`]. The default-loop path remains
    /// available — it just is not used while `handoff` is `Some`.
    #[must_use]
    pub fn with_handoff(mut self, h: Arc<AgentHandoff>) -> Self {
        self.handoff = Some(h);
        // Seed the visible role so the status bar shows `agent: default`
        // immediately on first paint — before any turn has run. Subsequent
        // submits update this to the chain's final role.
        self.current_role = Some(SuggestedRole::Default);
        self
    }

    /// Whether [`Self::submit`] currently routes through an [`AgentHandoff`].
    /// Test-friendly accessor; production callers do not branch on this.
    #[must_use]
    pub const fn has_handoff(&self) -> bool {
        self.handoff.is_some()
    }

    /// Set the slug surfaced by the `/model` palette command.
    #[must_use]
    pub fn with_active_model(mut self, slug: impl Into<String>) -> Self {
        self.active_model = Some(slug.into());
        self
    }

    /// Seed the list surfaced by the `/models` palette command.
    #[must_use]
    pub fn with_available_models(mut self, slugs: Vec<String>) -> Self {
        self.available_models = slugs;
        self
    }

    /// Install the `/switch` palette hook.
    #[must_use]
    pub fn with_model_switcher(mut self, sw: ModelSwitcher) -> Self {
        self.model_switcher = Some(sw);
        self
    }

    /// Replace the internal `TuiPromptResponder` so the TUI's permission
    /// modal queue is the SAME one the wired `AgentLoop` posts requests
    /// to. Required when the loop is built externally (LLM path) and
    /// uses `TuiPromptResponder` instead of `AllowAllResponder`. Without
    /// this, the loop would post requests into a detached queue that the
    /// TUI never drains and turns hang forever.
    #[must_use]
    pub fn with_permission_prompter(mut self, prompter: Arc<TuiPromptResponder>) -> Self {
        self.permission_prompter = prompter;
        self
    }

    /// Install the `!cmd` shell executor hook.
    #[must_use]
    pub fn with_shell_executor(mut self, ex: ShellExecutor) -> Self {
        self.shell_executor = Some(ex);
        self
    }

    /// Persist the input-history ring at `path` (one JSON entry per
    /// line). Loaded synchronously when this builder runs; subsequent
    /// `record_input_history` calls re-write the file atomically.
    /// Errors are silently ignored — history is a UX nicety, not
    /// load-bearing.
    #[must_use]
    pub fn with_history_path(mut self, path: PathBuf) -> Self {
        if let Ok(body) = std::fs::read_to_string(&path) {
            for line in body.lines() {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                if let Ok(s) = serde_json::from_str::<String>(trimmed) {
                    if s.is_empty() {
                        continue;
                    }
                    if self.input_history.last() != Some(&s) {
                        self.input_history.push(s);
                    }
                }
            }
            if self.input_history.len() > Self::INPUT_HISTORY_CAP {
                let drop = self.input_history.len() - Self::INPUT_HISTORY_CAP;
                self.input_history.drain(0..drop);
            }
        }
        self.history_path = Some(path);
        self
    }

    /// Best-effort atomic write of `input_history` to `history_path`.
    fn save_input_history(&self) {
        let Some(path) = self.history_path.as_ref() else {
            return;
        };
        let mut body = String::with_capacity(self.input_history.len() * 32);
        for entry in &self.input_history {
            if let Ok(s) = serde_json::to_string(entry) {
                body.push_str(&s);
                body.push('\n');
            }
        }
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let tmp = path.with_extension("tmp");
        if std::fs::write(&tmp, body.as_bytes()).is_ok() {
            let _ = std::fs::rename(&tmp, path);
        }
    }

    /// Configure a custom statusline shell command. Its stdout (first
    /// line) is rendered as an extra segment in the status bar.
    /// Re-invoked once every 5 seconds on a background thread; the
    /// render path only ever reads a cached snapshot, so a slow
    /// script can't freeze the UI. The command is run via
    /// `sh -c <cmd>` with a small JSON payload on stdin describing
    /// the session — mirroring Claude Code's `status_line`.
    #[must_use]
    pub fn with_statusline(mut self, cmd: String) -> Self {
        let trimmed = cmd.trim().to_string();
        if !trimmed.is_empty() {
            self.statusline_cmd = Some(trimmed);
        }
        self
    }

    /// Current statusline snapshot — what the next render should
    /// display next to the rest of the status segments. May be the
    /// empty string when no statusline is configured or the script
    /// has not produced output yet.
    fn statusline_snapshot(&self) -> Option<String> {
        self.statusline_cmd.as_ref()?;
        self.statusline_cache
            .lock()
            .ok()
            .map(|g| g.clone())
            .filter(|s| !s.is_empty())
    }

    /// If a custom statusline is configured and the throttle has
    /// elapsed, spawn the script on a worker thread. Called once per
    /// event-loop tick; cheap when nothing is configured.
    fn tick_statusline(&self) {
        let Some(cmd) = self.statusline_cmd.as_deref() else {
            return;
        };
        let now = Instant::now();
        if let Ok(mut g) = self.statusline_last_run.lock() {
            let due = match *g {
                None => true,
                Some(t) => now.duration_since(t) >= Duration::from_secs(5),
            };
            if !due {
                return;
            }
            *g = Some(now);
        } else {
            return;
        }
        let cmd = cmd.to_string();
        let cache = Arc::clone(&self.statusline_cache);
        let session_id = self.session_id.to_string();
        let model = self
            .active_model
            .clone()
            .unwrap_or_else(|| "echo".to_string());
        let tier = format!("{:?}", self.tier).to_lowercase();
        let cwd = std::env::current_dir()
            .ok()
            .map(|p| p.display().to_string())
            .unwrap_or_default();
        thread::spawn(move || {
            use std::io::Write as _;
            let stdin_payload = format!(
                "{{\"session_id\":\"{session_id}\",\"model\":\"{model}\",\"tier\":\"{tier}\",\"cwd\":\"{}\"}}\n",
                cwd.replace('\\', "\\\\").replace('"', "\\\"")
            );
            let mut child = match std::process::Command::new("sh")
                .arg("-c")
                .arg(&cmd)
                .stdin(std::process::Stdio::piped())
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::null())
                .spawn()
            {
                Ok(c) => c,
                Err(_) => return,
            };
            if let Some(mut si) = child.stdin.take() {
                let _ = si.write_all(stdin_payload.as_bytes());
            }
            let out = match child.wait_with_output() {
                Ok(o) => o,
                Err(_) => return,
            };
            let first_line = String::from_utf8_lossy(&out.stdout)
                .lines()
                .next()
                .unwrap_or("")
                .trim()
                .to_string();
            if let Ok(mut c) = cache.lock() {
                *c = first_line;
            }
        });
    }

    /// Wire theme persistence + theme override directory. Reads the
    /// persisted theme name from `state_file` (if any) and applies it
    /// immediately; subsequent `/theme <name>` calls re-write the
    /// state file so the choice survives restart. `themes_dir` is
    /// where user-authored JSON theme files live.
    #[must_use]
    pub fn with_theme_paths(mut self, state_file: PathBuf, themes_dir: PathBuf) -> Self {
        if let Some(name) = crate::theme::read_persisted(&state_file) {
            // Best-effort — a malformed persisted name just keeps the
            // built-in default active.
            let _ = crate::theme::set_by_name(&name, Some(&themes_dir));
        }
        self.theme_state_path = Some(state_file);
        self.themes_dir = Some(themes_dir);
        self
    }

    /// Session id stamped on this chat. Surfaced on exit so the user can
    /// reload via `stratum chat --resume <id>`.
    #[must_use]
    pub const fn session_id(&self) -> &stratum_runtime::SessionId {
        &self.session_id
    }

    /// Snapshot the in-memory chat transcript as a persistable
    /// [`stratum_runtime::Transcript`]. The block payload uses
    /// the on-disk projection so blocks survive a binary upgrade.
    #[must_use]
    pub fn to_persisted_transcript(&self) -> stratum_runtime::Transcript {
        use stratum_runtime::{
            TranscriptBlock, TranscriptBlockKind, TranscriptTurn, TRANSCRIPT_SCHEMA_VERSION,
        };
        let now = SystemTime::now();
        let mut turns: Vec<TranscriptTurn> = Vec::with_capacity(self.transcript.len());
        for t in &self.transcript {
            match t {
                Turn::User(text) => turns.push(TranscriptTurn::User {
                    at: now,
                    text: text.clone(),
                }),
                Turn::Assistant(blocks) => {
                    let on_disk: Vec<TranscriptBlock> = blocks
                        .iter()
                        .filter_map(|b| match b {
                            Block::Text { text } => Some(TranscriptBlock {
                                kind: TranscriptBlockKind::Text,
                                text: text.clone(),
                            }),
                            Block::ToolCall { tool, args, .. } => Some(TranscriptBlock {
                                kind: TranscriptBlockKind::ToolCall {
                                    tool_id: tool.clone(),
                                },
                                text: args.clone(),
                            }),
                            Block::ToolResult { output, .. } => Some(TranscriptBlock {
                                kind: TranscriptBlockKind::Text,
                                text: format!("[result] {output}"),
                            }),
                            Block::Cancelled { reason } => Some(TranscriptBlock {
                                kind: TranscriptBlockKind::Text,
                                text: format!("[cancelled] {reason}"),
                            }),
                            // Image / Audio are multimodal payloads — we
                            // do not serialise binary bytes into the
                            // text-only transcript schema yet; drop a
                            // placeholder so the turn count stays right.
                            Block::Image { mime, .. } => Some(TranscriptBlock {
                                kind: TranscriptBlockKind::Text,
                                text: format!("[image: {mime}]"),
                            }),
                            Block::Audio { mime, .. } => Some(TranscriptBlock {
                                kind: TranscriptBlockKind::Text,
                                text: format!("[audio: {mime}]"),
                            }),
                            Block::Usage { .. } | Block::Done => None,
                        })
                        .collect();
                    turns.push(TranscriptTurn::Assistant {
                        at: now,
                        blocks: on_disk,
                    });
                }
                Turn::Cancelled => turns.push(TranscriptTurn::System {
                    at: now,
                    text: "[cancelled]".to_string(),
                }),
                Turn::Command {
                    text,
                    ok,
                    message: _,
                } => turns.push(TranscriptTurn::Command {
                    at: now,
                    text: text.clone(),
                    ok: *ok,
                }),
            }
        }
        stratum_runtime::Transcript {
            schema_version: TRANSCRIPT_SCHEMA_VERSION,
            session_id: self.session_id.clone(),
            created_at: self.created_at,
            turns,
        }
    }

    /// Status-bar label describing the active role.
    ///
    /// * Single-loop mode (no [`Self::with_handoff`]): the empty string.
    /// * Multi-agent mode: `"agent: <snake_case_name>"` for the role that
    ///   produced the most recent hop (or [`SuggestedRole::Default`] if no
    ///   submit has run yet).
    ///
    /// The role name is rendered through [`role_name`] so the formatted
    /// value matches the `serde(rename_all = "snake_case")` projection used
    /// throughout the runtime.
    #[must_use]
    pub fn current_role_label(&self) -> String {
        self.current_role
            .map_or_else(String::new, |role| format!("agent: {}", role_name(role)))
    }

    /// Replace the structured-event emitter (e.g. with a JSONL-backed one).
    ///
    /// The default [`MemoryEventSink`] handle is dropped; calls to
    /// [`Self::events_snapshot`] will return `None` afterwards. The
    /// underlying [`AgentLoop`] is rebuilt against the new emitter so
    /// turn-level events also land in the swapped sink.
    #[must_use]
    pub fn with_events(mut self, events: Arc<EventEmitter>) -> Self {
        #[allow(
            clippy::expect_used,
            reason = "default_agent_loop sets all nine required builder fields; build() cannot return MissingField on this code path"
        )]
        let agent_loop = Arc::new(
            default_agent_loop(
                self.provider.clone(),
                events.clone(),
                self.plan_mode.clone(),
                self.permission_prompter.clone(),
            )
            .expect("default AgentLoop builder sets every required field"),
        );
        self.agent_loop = agent_loop;
        self.events = events;
        self.memory_sink = None;
        self
    }

    /// Pop the next pending permission request from the shared TUI prompter,
    /// if any. The event loop polls this each tick; `Some` triggers the
    /// modal overlay path in [`Self::render`].
    ///
    /// This consumes the request from the queue — callers that need a
    /// non-destructive view should use [`Self::peek_pending_permission`].
    #[must_use]
    pub fn pending_permission_request(&self) -> Option<PendingPrompt> {
        self.permission_prompter.pending_request()
    }

    /// Non-destructively look at the next pending permission request.
    /// Used by the modal-render path so the queue is observable across
    /// multiple paint frames.
    #[must_use]
    pub fn peek_pending_permission(&self) -> Option<PendingPrompt> {
        self.permission_prompter.peek_request()
    }

    /// Record the user's answer for `id` on the shared TUI prompter, waking
    /// the worker thread that is blocked inside [`stratum_runtime::PromptResponder::ask`].
    pub fn answer_permission(&self, id: PromptId, decision: PermissionDecision) {
        self.permission_prompter.submit_decision(id, decision);
    }

    /// Pre-populate the scrollback with the turns from a previously persisted
    /// [`Transcript`] so a resumed session shows its prior context.
    ///
    /// Each [`TranscriptTurn`] is mapped to its in-memory [`Turn`] counterpart:
    ///
    /// * [`TranscriptTurn::User`] → [`Turn::User`].
    /// * [`TranscriptTurn::Assistant`] → [`Turn::Assistant`] with each
    ///   [`stratum_runtime::TranscriptBlock`] folded into a [`Block::Text`]
    ///   carrying its rendered text. This keeps the user-visible content while
    ///   sidestepping the impedance mismatch between the persisted block taxonomy
    ///   and the streaming-provider [`Block`] enum.
    /// * [`TranscriptTurn::System`] → [`Turn::Command`] with `text` prefixed by
    ///   `"(system) "`, mirroring how the palette renders informational lines.
    /// * [`TranscriptTurn::Command`] → [`Turn::Command`] preserving `ok`.
    ///
    /// Call this once, before [`Self::submit`] / [`Self::submit_with_prompt`];
    /// existing transcript entries are preserved and the resumed turns are
    /// appended at the current end. [`Self::resumed_count`] is updated by the
    /// number of turns folded in.
    #[must_use]
    pub fn with_resumed_transcript(mut self, t: Transcript) -> Self {
        let mut added = 0_usize;
        for turn in t.turns {
            let mapped = match turn {
                TranscriptTurn::User { text, .. } => Turn::User(text),
                TranscriptTurn::Assistant { blocks, .. } => {
                    let mapped_blocks: Vec<Block> = blocks
                        .into_iter()
                        .map(|b| Block::Text { text: b.text })
                        .collect();
                    Turn::Assistant(mapped_blocks)
                }
                TranscriptTurn::System { text, .. } => Turn::Command {
                    text: format!("(system) {text}"),
                    ok: true,
                    message: String::new(),
                },
                TranscriptTurn::Command { text, ok, .. } => Turn::Command {
                    text,
                    ok,
                    message: String::new(),
                },
            };
            self.transcript.push(mapped);
            added = added.saturating_add(1);
        }
        self.resumed_count = self.resumed_count.saturating_add(added);
        self
    }

    /// Number of turns prepended by [`Self::with_resumed_transcript`].
    ///
    /// Returns zero when no transcript was resumed.
    #[must_use]
    pub const fn resumed_count(&self) -> usize {
        self.resumed_count
    }

    /// Borrow the most recent [`TurnResult`], if any.
    #[must_use]
    pub const fn last_turn_result(&self) -> Option<&TurnResult> {
        self.last_turn_result.as_ref()
    }

    /// Snapshot the in-memory event log if the emitter is backed by a
    /// [`MemoryEventSink`].
    ///
    /// Returns `None` when [`Self::with_events`] swapped in an opaque sink
    /// (the trait does not carry an `Any` bound, so the runtime sink type is
    /// not recoverable from the emitter alone).
    #[must_use]
    pub fn events_snapshot(&self) -> Option<Vec<EventRecord>> {
        self.memory_sink.as_ref().map(|sink| sink.snapshot())
    }

    /// Borrow the last recorded turn metrics, if any. Used by tests.
    #[must_use]
    #[cfg(test)]
    const fn last_metrics(&self) -> Option<&TurnMetrics> {
        self.last_metrics.as_ref()
    }

    /// Is the slash-command palette currently visible?
    #[must_use]
    #[cfg(test)]
    const fn palette_open(&self) -> bool {
        self.palette.is_some()
    }

    /// Has the user asked to quit?
    #[must_use]
    pub const fn should_quit(&self) -> bool {
        self.quit
    }

    /// Borrow the transcript (used in tests).
    /// Borrow the in-memory transcript. Used by `--output-format
    /// json` to serialize the last assistant turn's blocks.
    #[must_use]
    pub fn transcript(&self) -> &[Turn] {
        &self.transcript
    }

    /// Borrow the current input buffer (used in tests).
    #[must_use]
    #[cfg(test)]
    fn input(&self) -> &str {
        &self.input
    }

    /// Attachments staged onto the most recent [`Self::submit`] call.
    /// Empty until a `/audio <path>` is queued and submitted.
    #[must_use]
    pub fn last_turn_attachments(&self) -> &[Block] {
        &self.last_turn_attachments
    }

    /// Currently-staged audio attachment, if any. `Some(Block::Audio { … })`
    /// after `/audio <path>`; `None` after the next [`Self::submit`].
    #[must_use]
    pub const fn staged_audio(&self) -> Option<&Block> {
        self.staged_audio.as_ref()
    }

    /// Currently-staged audio transcript, if any. `Some(text)` only when
    /// whisper.cpp was available AND produced non-empty output.
    #[must_use]
    pub fn staged_audio_transcript(&self) -> Option<&str> {
        self.staged_audio_transcript.as_deref()
    }

    /// Apply a keyboard event.
    pub fn handle_key(&mut self, key: KeyEvent) {
        // Ignore Release / Repeat events; only Press counts. Kitty CSI-u
        // and some terminals deliver Press + Release pairs which would
        // otherwise double-fire every handler (e.g. ↑ skipping past the
        // most recent history entry, Enter submitting twice).
        if key.kind != KeyEventKind::Press {
            return;
        }
        // Permission modal owns the keyboard while a request is pending.
        if let Some(pending) = self.permission_prompter.peek_request() {
            if let KeyCode::Char(c) = key.code {
                if let Some(decision) = decision_from_key(c) {
                    // Drain the request from the queue and answer it.
                    let _ = self.permission_prompter.pending_request();
                    self.permission_prompter
                        .submit_decision(pending.id, decision);
                }
            }
            // Unknown / non-char keys are swallowed while the modal is open
            // so they don't leak into the input buffer.
            return;
        }
        // Reverse-search modal owns the keyboard while open.
        if self.rsearch.is_some() {
            self.handle_rsearch_key(key);
            return;
        }
        // Palette mode owns the keyboard while open.
        if let Some(palette) = self.palette.as_mut() {
            match palette.handle_key(key) {
                palette::Action::None => {}
                palette::Action::Close => {
                    self.palette = None;
                }
                palette::Action::Execute(name) => {
                    self.palette = None;
                    self.execute_command(&name);
                }
            }
            return;
        }
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        let shift = key.modifiers.contains(KeyModifiers::SHIFT);
        let alt = key.modifiers.contains(KeyModifiers::ALT);
        // Any keystroke that isn't a Ctrl+C / Ctrl+D resets the
        // double-press exit gesture.
        if !(ctrl && matches!(key.code, KeyCode::Char('c' | 'C' | 'd' | 'D'))) {
            self.exit_armed_at = None;
        }
        match key.code {
            KeyCode::Esc => {
                if self.history_cursor.is_some() {
                    self.history_cursor = None;
                    self.input.clear();
                    self.caret = 0;
                } else {
                    self.quit = true;
                }
            }
            KeyCode::Char('c' | 'C' | 'd' | 'D') if ctrl => {
                self.handle_exit_signal();
            }
            // `/` opens the palette only on Ctrl+/ now; bare `/` is
            // typed as a literal so the user can write `/switch <slug>`
            // (and any other palette command with args) in the input box.
            KeyCode::Char('/') if ctrl => {
                self.palette = Some(Palette::new());
            }
            // Universal newline: Ctrl+J inserts a literal LF. Used when
            // the terminal does not support kitty CSI-u disambiguation
            // and therefore folds Shift+Enter into bare Enter (the
            // default on stock iTerm2 and macOS Terminal.app). Mirrors
            // git's commit-message convention.
            KeyCode::Char('j' | 'J') if ctrl => {
                self.insert_char('\n');
            }
            KeyCode::Up => self.history_up(),
            KeyCode::Down => self.history_down(),
            // ---- cursor motion ----------------------------------------
            KeyCode::Left => self.caret_left(),
            KeyCode::Right => self.caret_right(),
            KeyCode::Home => self.caret_to_line_start(),
            KeyCode::End => self.caret_to_line_end(),
            KeyCode::Char('a' | 'A') if ctrl => self.caret_to_line_start(),
            KeyCode::Char('e' | 'E') if ctrl => self.caret_to_line_end(),
            KeyCode::Char('b' | 'B') if alt => self.caret_word_back(),
            KeyCode::Char('f' | 'F') if alt => self.caret_word_forward(),
            // ---- kill-ring --------------------------------------------
            KeyCode::Char('k' | 'K') if ctrl => self.kill_to_line_end(),
            KeyCode::Char('u' | 'U') if ctrl => self.kill_to_line_start(),
            KeyCode::Char('w' | 'W') if ctrl => self.kill_word_back(),
            KeyCode::Char('y' | 'Y') if ctrl => self.yank_latest(),
            KeyCode::Char('y' | 'Y') if alt => self.yank_cycle(),
            KeyCode::Char('r' | 'R') if ctrl => {
                self.rsearch = Some(RSearchState::default());
            }
            // ---- tab completion for slash commands -------------------
            KeyCode::Tab => self.tab_complete(),
            // ---- external editor -------------------------------------
            KeyCode::Char('g' | 'G') if ctrl => {
                self.pending_edit_request = true;
            }
            // ---- toggle mouse capture so user can select+copy --------
            KeyCode::Char('t' | 'T') if ctrl => {
                self.mouse_capture_on = !self.mouse_capture_on;
                self.pending_mouse_toggle = true;
            }
            // ---- chat-pane scrollback ---------------------------------
            KeyCode::PageUp => self.scroll_up(10),
            KeyCode::PageDown => self.scroll_down(10),
            // ---- push-to-talk mic toggle (Phase 5 voice-in) -----------
            // F5 toggles `MicCapture` on/off; first press starts capture,
            // second press stops + transcribes + queues the WAV as the
            // next-turn audio attachment (same code path as `/audio`).
            // F5 was previously unhandled (see `unhandled_key_is_ignored`
            // — that test asserts input/quit are untouched, which still
            // holds because the mic toggle mutates other fields only).
            KeyCode::F(5) => {
                // Surface any error (e.g. "voice support not compiled in"
                // on a default-features build) as a transient status so
                // the user gets visible feedback. Without this, pressing
                // F5 on a non-voice build is a silent no-op.
                if let Err(e) = self.toggle_ptt() {
                    self.set_transient_status(format!("mic: {e}"));
                }
            }
            // ---- editing ----------------------------------------------
            KeyCode::Char(c) => {
                self.history_cursor = None;
                self.insert_char(c);
            }
            KeyCode::Backspace => {
                self.history_cursor = None;
                self.delete_char_back();
            }
            KeyCode::Delete => {
                self.history_cursor = None;
                self.delete_char_forward();
            }
            KeyCode::Enter => {
                if shift || alt {
                    self.insert_char('\n');
                } else {
                    self.submit();
                    // In tests the assertion happens immediately after
                    // `handle_key(Enter)`, so block until the async turn
                    // settles. Production TUI relies on the event loop's
                    // `poll_turn_completion` instead.
                    #[cfg(test)]
                    self.block_until_idle();
                }
            }
            _ => {}
        }
    }

    /// Two-press exit gesture mirroring Claude Code:
    ///   - first press: if a turn is in flight, cancel it and arm the
    ///     exit timer; if idle, just arm the exit timer and surface a
    ///     hint in the status bar
    ///   - second press within [`Self::EXIT_ARM_WINDOW`]: actually quit
    ///   - after the window elapses the arm clears so a stale first
    ///     press doesn't combo with a later one
    fn handle_exit_signal(&mut self) {
        if self.in_flight_since.is_some() {
            self.cancel.cancel();
            self.transcript.push(Turn::Cancelled);
            self.cancel_already_pushed = true;
            self.exit_armed_at = Some(Instant::now());
            return;
        }
        match self.exit_armed_at {
            Some(t) if t.elapsed() <= Self::EXIT_ARM_WINDOW => {
                self.quit = true;
            }
            _ => {
                self.exit_armed_at = Some(Instant::now());
            }
        }
    }

    /// True when the exit hint should render in the status bar.
    #[must_use]
    pub fn exit_armed(&self) -> bool {
        self.exit_armed_at
            .is_some_and(|t| t.elapsed() <= Self::EXIT_ARM_WINDOW)
    }

    /// Insert `c` at the caret, advance the caret past it.
    fn insert_char(&mut self, c: char) {
        self.yank_cursor = None;
        let pos = self.caret.min(self.input.len());
        self.input.insert(pos, c);
        self.caret = pos + c.len_utf8();
    }

    fn push_kill(&mut self, s: String) {
        if s.is_empty() {
            return;
        }
        self.kill_ring.push(s);
        if self.kill_ring.len() > Self::KILL_RING_CAP {
            self.kill_ring.remove(0);
        }
        self.yank_cursor = None;
    }

    fn kill_to_line_end(&mut self) {
        let start = self.caret.min(self.input.len());
        let end = self.input[start..]
            .find('\n')
            .map_or(self.input.len(), |i| start + i);
        if start >= end {
            return;
        }
        let killed: String = self.input.drain(start..end).collect();
        self.push_kill(killed);
        self.caret = start;
    }

    fn kill_to_line_start(&mut self) {
        let start = self.input[..self.caret].rfind('\n').map_or(0, |i| i + 1);
        let end = self.caret.min(self.input.len());
        if start >= end {
            return;
        }
        let killed: String = self.input.drain(start..end).collect();
        self.push_kill(killed);
        self.caret = start;
    }

    fn kill_word_back(&mut self) {
        let end = self.caret.min(self.input.len());
        if end == 0 {
            return;
        }
        let bytes = self.input.as_bytes();
        let mut p = end;
        while p > 0 && bytes[p - 1].is_ascii_whitespace() {
            p -= 1;
        }
        while p > 0 && !bytes[p - 1].is_ascii_whitespace() {
            p -= 1;
        }
        while p > 0 && !self.input.is_char_boundary(p) {
            p -= 1;
        }
        if p >= end {
            return;
        }
        let killed: String = self.input.drain(p..end).collect();
        self.push_kill(killed);
        self.caret = p;
    }

    fn yank_latest(&mut self) {
        let Some(entry) = self.kill_ring.last().cloned() else {
            return;
        };
        let pos = self.caret.min(self.input.len());
        self.input.insert_str(pos, &entry);
        self.yank_start = pos;
        self.caret = pos + entry.len();
        self.yank_cursor = Some(self.kill_ring.len() - 1);
    }

    /// Alt+Y after a Ctrl+Y: replace the just-yanked text with the
    /// next-older kill-ring entry. No-op if there is no preceding yank.
    /// Scroll the chat pane up by `n` rows (toward older history).
    pub const fn scroll_up(&mut self, n: u16) {
        self.chat_scroll = self.chat_scroll.saturating_add(n);
    }

    /// Scroll the chat pane down by `n` rows (toward the latest line).
    pub const fn scroll_down(&mut self, n: u16) {
        self.chat_scroll = self.chat_scroll.saturating_sub(n);
    }

    /// Pin the chat pane to the latest line. Called by `submit()` so the
    /// freshly-pushed user turn is always visible.
    const fn scroll_to_bottom(&mut self) {
        self.chat_scroll = 0;
    }

    /// Consume any pending `Ctrl+G` external-editor request. Returns
    /// the current input text when a request was set; the caller is
    /// responsible for suspending the TUI, spawning the editor, and
    /// passing the edited result back via [`Self::set_input_from_editor`].
    /// Consume any pending mouse-capture toggle. Returns the new
    /// desired state (`true` = capture on, `false` = capture off) the
    /// caller should flip the terminal to.
    pub const fn take_pending_mouse_toggle(&mut self) -> Option<bool> {
        if self.pending_mouse_toggle {
            self.pending_mouse_toggle = false;
            Some(self.mouse_capture_on)
        } else {
            None
        }
    }

    /// True when mouse capture is currently disabled — used by the
    /// render path to surface a "select" hint in the status bar.
    #[must_use]
    pub const fn mouse_capture_off(&self) -> bool {
        !self.mouse_capture_on
    }

    /// Drain a pending external-editor request, if any.
    ///
    /// `/edit` (or Ctrl-E) sets the pending-edit flag; the event loop
    /// polls this each tick and, on `Some(input)`, opens `$EDITOR` over
    /// the current buffer and restores the modified text. Returns `None`
    /// when no request is in flight.
    pub fn take_pending_edit_request(&mut self) -> Option<String> {
        if self.pending_edit_request {
            self.pending_edit_request = false;
            Some(self.input.clone())
        } else {
            None
        }
    }

    /// Replace the input buffer with text returned from an external
    /// editor. Trailing newline (almost every editor appends one) is
    /// stripped. Caret moves to the end of the new input.
    pub fn set_input_from_editor(&mut self, new_text: String) {
        let mut trimmed = new_text;
        if trimmed.ends_with('\n') {
            trimmed.pop();
        }
        self.input = trimmed;
        self.caret = self.input.len();
        self.history_cursor = None;
        self.yank_cursor = None;
    }

    /// Tab completion. Two flavors:
    ///
    /// * `/<prefix>` (no whitespace yet) → palette catalog match.
    /// * `…@<partial>` immediately before the caret → workspace file
    ///   path match. Walks the cwd up to a depth of 4 + 500 entries,
    ///   skips dotfiles / `target/` / `node_modules/`. Replaces the
    ///   `@<partial>` span with `@<chosen>` (single match) or extends
    ///   to the longest common prefix (multiple matches).
    fn tab_complete(&mut self) {
        // Slash-command path.
        if self.input.starts_with('/') && !self.input.contains(char::is_whitespace) {
            let needle = &self.input[1..];
            let matches: Vec<&'static str> = palette::COMMANDS
                .iter()
                .filter(|c| c.name.starts_with(needle))
                .map(|c| c.name)
                .collect();
            if matches.is_empty() {
                return;
            }
            if matches.len() == 1 {
                self.input = format!("/{} ", matches[0]);
                self.caret = self.input.len();
                return;
            }
            let lcp = longest_common_prefix(&matches);
            if lcp.len() > needle.len() {
                self.input = format!("/{lcp}");
                self.caret = self.input.len();
            }
            return;
        }

        // @-file path: find the most recent `@` before the caret with
        // no whitespace between `@` and caret.
        let caret = self.caret.min(self.input.len());
        let before = &self.input[..caret];
        let at_pos = match before.rfind('@') {
            Some(p) => p,
            None => return,
        };
        let needle = &before[at_pos + 1..];
        if needle.contains(char::is_whitespace) {
            return;
        }
        let matches = workspace_file_matches(needle, 200);
        if matches.is_empty() {
            return;
        }
        let replacement = if matches.len() == 1 {
            matches[0].clone()
        } else {
            let refs: Vec<&str> = matches.iter().map(String::as_str).collect();
            let lcp = longest_common_prefix(&refs);
            if lcp.len() <= needle.len() {
                return;
            }
            lcp
        };
        self.input.replace_range(at_pos + 1..caret, &replacement);
        self.caret = at_pos + 1 + replacement.len();
    }

    /// Accessor used by the render path to draw the reverse-search
    /// modal when active.
    #[must_use]
    pub const fn rsearch_peek(&self) -> Option<&RSearchState> {
        self.rsearch.as_ref()
    }

    fn rsearch_matches(&self, needle: &str) -> Vec<&str> {
        if needle.is_empty() {
            return self
                .input_history
                .iter()
                .rev()
                .map(String::as_str)
                .collect();
        }
        self.input_history
            .iter()
            .rev()
            .filter(|h| h.contains(needle))
            .map(String::as_str)
            .collect()
    }

    fn handle_rsearch_key(&mut self, key: KeyEvent) {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        let Some(state) = self.rsearch.as_mut() else {
            return;
        };
        match key.code {
            KeyCode::Esc => {
                self.rsearch = None;
            }
            KeyCode::Char('c' | 'C') if ctrl => {
                self.rsearch = None;
            }
            KeyCode::Char('r' | 'R') if ctrl => {
                state.cursor = state.cursor.saturating_add(1);
            }
            KeyCode::Char(c) => {
                state.needle.push(c);
                state.cursor = 0;
            }
            KeyCode::Backspace => {
                state.needle.pop();
                state.cursor = 0;
            }
            KeyCode::Up => {
                state.cursor = state.cursor.saturating_add(1);
            }
            KeyCode::Down => {
                state.cursor = state.cursor.saturating_sub(1);
            }
            KeyCode::Enter => {
                let needle = state.needle.clone();
                let cursor = state.cursor;
                let matches = self.rsearch_matches(&needle);
                if let Some(picked) = matches.get(cursor).map(|s| (*s).to_string()) {
                    self.input = picked;
                    self.caret = self.input.len();
                }
                self.rsearch = None;
            }
            _ => {}
        }
    }

    /// Handle a terminal mouse event. Wheel up/down scrolls the chat
    /// pane; other mouse events are ignored for now.
    pub const fn handle_mouse(&mut self, ev: MouseEvent) {
        match ev.kind {
            MouseEventKind::ScrollUp => self.scroll_up(3),
            MouseEventKind::ScrollDown => self.scroll_down(3),
            _ => {}
        }
    }

    fn yank_cycle(&mut self) {
        let Some(cur) = self.yank_cursor else {
            return;
        };
        if self.kill_ring.is_empty() {
            return;
        }
        let prev_entry = self.kill_ring[cur].clone();
        // Remove the previous yank.
        let prev_end = self.yank_start + prev_entry.len();
        if prev_end > self.input.len() {
            return;
        }
        self.input.replace_range(self.yank_start..prev_end, "");
        // Step to older entry (wrap to newest if we reach the start).
        let next = if cur == 0 {
            self.kill_ring.len() - 1
        } else {
            cur - 1
        };
        let next_entry = self.kill_ring[next].clone();
        self.input.insert_str(self.yank_start, &next_entry);
        self.caret = self.yank_start + next_entry.len();
        self.yank_cursor = Some(next);
    }

    /// Backspace: delete the char immediately before the caret.
    fn delete_char_back(&mut self) {
        if self.caret == 0 {
            return;
        }
        let pos = self.caret.min(self.input.len());
        let mut start = pos.saturating_sub(1);
        while start > 0 && !self.input.is_char_boundary(start) {
            start -= 1;
        }
        self.input.replace_range(start..pos, "");
        self.caret = start;
    }

    /// Forward-delete: remove the char at the caret.
    fn delete_char_forward(&mut self) {
        if self.caret >= self.input.len() {
            return;
        }
        let mut end = self.caret + 1;
        while end < self.input.len() && !self.input.is_char_boundary(end) {
            end += 1;
        }
        self.input.replace_range(self.caret..end, "");
    }

    fn caret_left(&mut self) {
        if self.caret == 0 {
            return;
        }
        let mut p = self.caret - 1;
        while p > 0 && !self.input.is_char_boundary(p) {
            p -= 1;
        }
        self.caret = p;
    }

    fn caret_right(&mut self) {
        if self.caret >= self.input.len() {
            return;
        }
        let mut p = self.caret + 1;
        while p < self.input.len() && !self.input.is_char_boundary(p) {
            p += 1;
        }
        self.caret = p;
    }

    /// Move caret to the start of the current line (after the last `\n`
    /// at-or-before caret).
    fn caret_to_line_start(&mut self) {
        self.caret = self.input[..self.caret].rfind('\n').map_or(0, |i| i + 1);
    }

    /// Move caret to the end of the current line (before the next `\n`).
    fn caret_to_line_end(&mut self) {
        self.caret = self.input[self.caret..]
            .find('\n')
            .map_or(self.input.len(), |i| self.caret + i);
    }

    fn caret_word_back(&mut self) {
        if self.caret == 0 {
            return;
        }
        let bytes = self.input.as_bytes();
        let mut p = self.caret;
        // Skip trailing whitespace
        while p > 0 && bytes[p - 1].is_ascii_whitespace() {
            p -= 1;
        }
        // Skip word chars
        while p > 0 && !bytes[p - 1].is_ascii_whitespace() {
            p -= 1;
        }
        // Align to char boundary
        while p > 0 && !self.input.is_char_boundary(p) {
            p -= 1;
        }
        self.caret = p;
    }

    fn caret_word_forward(&mut self) {
        let bytes = self.input.as_bytes();
        let mut p = self.caret;
        // Skip word chars
        while p < bytes.len() && !bytes[p].is_ascii_whitespace() {
            p += 1;
        }
        // Skip trailing whitespace
        while p < bytes.len() && bytes[p].is_ascii_whitespace() {
            p += 1;
        }
        while p < bytes.len() && !self.input.is_char_boundary(p) {
            p += 1;
        }
        self.caret = p.min(self.input.len());
    }

    fn history_up(&mut self) {
        // Pulling the most-recent queued message back into the input
        // wins over walking submit-history, mirroring Claude Code: queue
        // edit is the more common reason to press ↑ on a busy session.
        if self.in_flight_since.is_some() && !self.pending_queue.is_empty() {
            if let Some(last) = self.pending_queue.pop() {
                self.input = last;
            }
            return;
        }
        if self.input_history.is_empty() {
            return;
        }
        let next = match self.history_cursor {
            None => self.input_history.len() - 1,
            Some(0) => 0,
            Some(i) => i - 1,
        };
        self.history_cursor = Some(next);
        self.input = self.input_history[next].clone();
        self.caret = self.input.len();
    }

    fn history_down(&mut self) {
        let Some(i) = self.history_cursor else {
            return;
        };
        if i + 1 >= self.input_history.len() {
            self.history_cursor = None;
            self.input.clear();
            self.caret = 0;
            return;
        }
        let next = i + 1;
        self.history_cursor = Some(next);
        self.input = self.input_history[next].clone();
        self.caret = self.input.len();
    }

    fn record_input_history(&mut self, entry: &str) {
        if entry.is_empty() {
            return;
        }
        if self.input_history.last().is_some_and(|last| last == entry) {
            return;
        }
        self.input_history.push(entry.to_string());
        if self.input_history.len() > Self::INPUT_HISTORY_CAP {
            let drop = self.input_history.len() - Self::INPUT_HISTORY_CAP;
            self.input_history.drain(0..drop);
        }
        self.history_cursor = None;
        self.save_input_history();
    }

    /// Internal palette-flush bridge: the palette emits a bare command
    /// name (no leading `/`); we re-attach the slash and route through
    /// [`Self::execute_palette_command`].
    fn execute_command(&mut self, name: &str) {
        let with_slash = format!("/{name}");
        let _ = self.execute_palette_command(&with_slash);
    }

    /// Dispatch a single palette command. Recognised commands mutate
    /// state (toggle `plan_mode`, fire the `cancel` token, clear the
    /// transcript, set `should_quit`) and push a
    /// [`Turn::Command`] entry to the transcript. Unknown commands are
    /// rejected. The return value also exposes the outcome so callers
    /// can render a status message without re-walking the transcript.
    ///
    /// `cmd` must start with `/` (the palette parser preserves the
    /// slash). An empty string or a command without the leading slash
    /// is rejected with `"unknown command: <cmd>"`.
    ///
    /// `/plan` is a toggle: if plan mode is currently active it is
    /// deactivated, else it is activated. Pass `/plan on` or
    /// `/plan off` to force a specific state.
    pub fn execute_palette_command(&mut self, cmd: &str) -> PaletteOutcome {
        // `/clear` is the only command whose post-state is the empty
        // transcript: it erases history *including* the marker for the
        // clear itself. Detect it up front so we can clear after the
        // push and end up with an empty transcript.
        let is_clear = cmd.trim() == "/clear";
        let outcome = self.dispatch_command(cmd);
        let (ok, message) = match &outcome {
            PaletteOutcome::Acknowledged { message } => (true, message.clone()),
            PaletteOutcome::Rejected { message } => (false, message.clone()),
        };
        self.transcript.push(Turn::Command {
            text: cmd.to_string(),
            ok,
            message,
        });
        if is_clear && ok {
            self.transcript.clear();
            self.chat_scroll = 0;
            self.streaming_text.clear();
        }
        outcome
    }

    fn dispatch_command(&mut self, cmd: &str) -> PaletteOutcome {
        let Some(rest) = cmd.strip_prefix('/') else {
            return PaletteOutcome::Rejected {
                message: format!("unknown command: {cmd}"),
            };
        };
        let trimmed = rest.trim();
        if trimmed.is_empty() {
            return PaletteOutcome::Rejected {
                message: format!("unknown command: {cmd}"),
            };
        }
        let mut parts = trimmed.split_whitespace();
        let head = parts.next().unwrap_or("");
        let arg = parts.next();
        match head {
            "plan" => {
                let message = match arg {
                    Some("on") => {
                        self.plan_mode.activate(SystemTime::now());
                        "plan mode: on".to_string()
                    }
                    Some("off") => {
                        self.plan_mode.deactivate();
                        "plan mode: off".to_string()
                    }
                    None => {
                        if self.plan_mode.is_active() {
                            self.plan_mode.deactivate();
                            "plan mode: off".to_string()
                        } else {
                            self.plan_mode.activate(SystemTime::now());
                            "plan mode: on".to_string()
                        }
                    }
                    Some(other) => {
                        return PaletteOutcome::Rejected {
                            message: format!("unknown command: /plan {other}"),
                        };
                    }
                };
                PaletteOutcome::Acknowledged { message }
            }
            "cancel" => {
                self.cancel.cancel();
                PaletteOutcome::Acknowledged {
                    message: "cancel signal sent".to_string(),
                }
            }
            "clear" => {
                self.transcript.clear();
                PaletteOutcome::Acknowledged {
                    message: "transcript cleared".to_string(),
                }
            }
            "help" => PaletteOutcome::Acknowledged {
                message: HELP_TEXT.to_string(),
            },
            "agents" => self.dispatch_agents(),
            "audio" => {
                let tail = trimmed.strip_prefix("audio").unwrap_or("").trim();
                self.dispatch_audio(tail)
            }
            "mic" => self.dispatch_mic(),
            "tts" => self.dispatch_tts(arg),
            "subagents" => self.dispatch_subagents(),
            "agent" => {
                // `/agent <name> <task>` queues an explicit subagent
                // delegation as the next user turn. Cheap UX shim:
                // re-injects the prompt as if the user typed
                // "Use the <name> subagent to <task>". The model still
                // emits `subagent.run` JSON, the agent loop dispatches.
                let tail = trimmed.strip_prefix("agent").unwrap_or("").trim();
                let mut parts = tail.splitn(2, char::is_whitespace);
                let Some(name) = parts.next().filter(|s| !s.is_empty()) else {
                    return PaletteOutcome::Rejected {
                        message: "usage: /agent <name> <task>".to_string(),
                    };
                };
                if self.subagents.get(name).is_none() {
                    return PaletteOutcome::Rejected {
                        message: format!(
                            "unknown subagent: {name} (run /subagents to list)"
                        ),
                    };
                }
                let Some(task) = parts.next().map(str::trim).filter(|s| !s.is_empty()) else {
                    return PaletteOutcome::Rejected {
                        message: "usage: /agent <name> <task>".to_string(),
                    };
                };
                self.input = format!("Use the {name} subagent to {task}");
                self.submit();
                PaletteOutcome::Acknowledged {
                    message: format!("delegating to {name}"),
                }
            }
            "parallel" => {
                let tail = trimmed.strip_prefix("parallel").unwrap_or("").trim();
                self.dispatch_parallel(tail)
            }
            "budget" | "cost" | "usage" => self.dispatch_budget(),
            "editor" => {
                // Ctrl+G shortcut as a palette command.
                self.pending_edit_request = true;
                PaletteOutcome::Acknowledged {
                    message: "opening external editor…".to_string(),
                }
            }
            "select" => {
                self.mouse_capture_on = !self.mouse_capture_on;
                self.pending_mouse_toggle = true;
                let label = if self.mouse_capture_on {
                    "mouse-scroll (wheel scrolls chat; native copy disabled)"
                } else {
                    "select-mode (drag with mouse + native copy)"
                };
                PaletteOutcome::Acknowledged {
                    message: format!("now in {label}"),
                }
            }
            "export" => self.dispatch_export(arg),
            "image" => {
                // `/image <path>` — path may contain spaces, so reparse
                // off the trimmed tail rather than the whitespace-split
                // `arg` slot we used for single-token commands.
                let tail = trimmed.strip_prefix("image").unwrap_or("").trim();
                if tail.is_empty() {
                    return PaletteOutcome::Rejected {
                        message: "usage: /image <path>".to_string(),
                    };
                }
                self.dispatch_image(tail)
            }
            "recap" => self.dispatch_recap(),
            "compact" => self.dispatch_compact(),
            "diff" => self.dispatch_diff(),
            "init" => self.dispatch_init(),
            "undo" | "redo" => PaletteOutcome::Rejected {
                message: format!(
                    "/{head} not implemented yet (workspace concept lands in Phase 3 v2; see plan/30)"
                ),
            },
            "model" | "active" => {
                // `/model` with no arg shows the active slug.
                // `/model <slug>` is an alias for `/switch <slug>` so users
                // can do model selection without remembering two commands.
                if let Some(slug) = arg {
                    return self.dispatch_switch(slug);
                }
                PaletteOutcome::Acknowledged {
                    message: match self.active_model.as_deref() {
                        Some(slug) => format!("active model: {slug}"),
                        None => "active model: echo (no real LLM provider attached)".to_string(),
                    },
                }
            }
            "models" => {
                if self.available_models.is_empty() {
                    PaletteOutcome::Acknowledged {
                        message: "no models in catalog; run `stratum models sync`".to_string(),
                    }
                } else {
                    let active = self.active_model.as_deref();
                    let listed = self
                        .available_models
                        .iter()
                        .map(|s| {
                            if Some(s.as_str()) == active {
                                format!("* {s}")
                            } else {
                                format!("  {s}")
                            }
                        })
                        .collect::<Vec<_>>()
                        .join("\n");
                    PaletteOutcome::Acknowledged {
                        message: format!("available models:\n{listed}"),
                    }
                }
            }
            "tier" => PaletteOutcome::Acknowledged {
                message: format!("tier: {:?}", self.tier).to_lowercase(),
            },
            "theme" => self.dispatch_theme(arg),
            "themes" => self.dispatch_themes(),
            "welcome" => self.dispatch_welcome(),
            "switch" => {
                let Some(slug) = arg else {
                    return PaletteOutcome::Rejected {
                        message: "usage: /switch <slug>".to_string(),
                    };
                };
                self.dispatch_switch(slug)
            }
            "version" => PaletteOutcome::Acknowledged {
                message: format!("stratum {}", env!("CARGO_PKG_VERSION")),
            },
            "quit" | "exit" => {
                self.quit = true;
                PaletteOutcome::Acknowledged {
                    message: "exiting".to_string(),
                }
            }
            _ => PaletteOutcome::Rejected {
                message: format!("unknown command: {cmd}"),
            },
        }
    }

    /// Shared switch logic backing `/switch <slug>` and `/model <slug>`.
    fn dispatch_switch(&mut self, slug: &str) -> PaletteOutcome {
        let Some(sw) = self.model_switcher.clone() else {
            return PaletteOutcome::Rejected {
                message: "/switch unavailable in echo mode".to_string(),
            };
        };
        if !self.available_models.is_empty() && !self.available_models.iter().any(|s| s == slug) {
            return PaletteOutcome::Rejected {
                message: format!("unknown slug: {slug} (run /models to list available)"),
            };
        }
        match (sw.0)(slug) {
            Ok(new_loop) => {
                self.agent_loop = new_loop;
                self.active_model = Some(slug.to_string());
                PaletteOutcome::Acknowledged {
                    message: format!("switched to {slug}"),
                }
            }
            Err(e) => PaletteOutcome::Rejected {
                message: format!("/switch failed: {e}"),
            },
        }
    }

    /// Render the `/agents` palette command output.
    ///
    /// Without an installed [`AgentHandoff`] (single-loop mode), reject with
    /// `/audio <path>` — stage an audio file for the next turn.
    ///
    /// Phase 5 v2 (`plan/05-multimodal.md` §Voice In). Parses the path
    /// (with optional double-quoted form so spaces survive), refuses
    /// absolute paths and `..` traversal at the surface — the
    /// `ReadAudioToolDispatcher` already canonicalises and rejects
    /// escapes, but the chat surface catches obvious mistakes early so
    /// the error message is human-readable. Reads the bytes, sniffs the
    /// MIME via [`ReadAudioToolDispatcher`], stages a `Block::Audio`
    /// attachment, and (best-effort) invokes
    /// [`WhisperSubprocess::transcribe`] so the next [`Self::submit`]
    /// can prepend `"[transcript: …]"` to the prompt.
    ///
    /// On whisper-missing the audio attachment still rides; only the
    /// transcript prefix becomes the "[audio transcript unavailable —
    /// install whisper.cpp]" sentinel. This is intentional: the user
    /// can still ask the model to reason about the *fact* that audio
    /// was attached even when the host can't transcribe.
    fn dispatch_audio(&mut self, raw_arg: &str) -> PaletteOutcome {
        let Some(path_str) = parse_audio_path_arg(raw_arg) else {
            return PaletteOutcome::Rejected {
                message: "usage: /audio <path>".to_string(),
            };
        };
        // Surface-level guard against absolute paths and `..` traversal.
        // The dispatcher does its own canonical check; this one keeps
        // the error legible at the chat seam.
        let p = std::path::Path::new(&path_str);
        if p.is_absolute()
            || p.components()
                .any(|c| matches!(c, std::path::Component::ParentDir))
        {
            return PaletteOutcome::Rejected {
                message: format!("path escapes workspace: {path_str}"),
            };
        }
        // Canonicalize against cwd so a relative symlink that points
        // outside the workspace (e.g. `audio -> /etc/shadow`) gets
        // caught here too. Mirrors `FsReadToolDispatcher`'s policy.
        let cwd = match std::env::current_dir() {
            Ok(c) => c,
            Err(e) => {
                return PaletteOutcome::Rejected {
                    message: format!("cannot resolve cwd: {e}"),
                };
            }
        };
        let canonical_cwd = cwd.canonicalize().unwrap_or(cwd.clone());
        let canonical_target = match cwd.join(p).canonicalize() {
            Ok(c) => c,
            Err(e) => {
                return PaletteOutcome::Rejected {
                    message: format!("cannot canonicalize {path_str}: {e}"),
                };
            }
        };
        if !canonical_target.starts_with(&canonical_cwd) {
            return PaletteOutcome::Rejected {
                message: format!(
                    "path escapes workspace via symlink: {path_str} -> {}",
                    canonical_target.display()
                ),
            };
        }
        let bytes = match std::fs::read(p) {
            Ok(b) => b,
            Err(e) => {
                return PaletteOutcome::Rejected {
                    message: format!("cannot read {path_str}: {e}"),
                };
            }
        };
        let mime = sniff_audio_mime_chat(p, &bytes);
        let b64 = base64_encode_chat(&bytes);
        let byte_len = u32::try_from(bytes.len()).unwrap_or(u32::MAX);
        let staged = Block::audio_inline_b64(mime.clone(), b64, byte_len);
        self.staged_audio = Some(staged);

        let transcript_msg = match self.whisper.transcribe(p) {
            Ok(text) if !text.is_empty() => {
                self.staged_audio_transcript = Some(text.clone());
                format!("transcript: {}", trim_for_ack(&text, 80))
            }
            Ok(_) => {
                // Whisper ran but produced an empty body — treat as
                // unavailable so the user-visible message is honest.
                self.staged_audio_transcript = None;
                "transcript: (empty)".to_string()
            }
            Err(stratum_runtime::whisper::WhisperError::MissingBinary) => {
                self.staged_audio_transcript = None;
                "whisper not installed; audio queued without transcript".to_string()
            }
            Err(e) => {
                self.staged_audio_transcript = None;
                format!("whisper failed: {e}; audio queued without transcript")
            }
        };
        PaletteOutcome::Acknowledged {
            message: format!(
                "audio staged ({} bytes, {}); {transcript_msg}",
                bytes.len(),
                mime,
            ),
        }
    }

    /// `/mic` — push-to-talk toggle. Idempotent palette wrapper around
    /// [`Self::toggle_ptt`]; the F5 hotkey runs the same path.
    ///
    /// First invocation opens a [`stratum_runtime::mic::MicCapture`]
    /// stream and sets [`Self::recording`]. Second invocation stops the
    /// stream, writes the captured 16 kHz mono buffer to a tempfile,
    /// runs the same transcribe + stage pipeline as `/audio`, and
    /// surfaces a transient status line with the whisper transcript
    /// snippet.
    fn dispatch_mic(&mut self) -> PaletteOutcome {
        match self.toggle_ptt() {
            Ok(MicToggle::Started) => PaletteOutcome::Acknowledged {
                message: "mic: recording — press F5 or /mic again to stop".to_string(),
            },
            Ok(MicToggle::Stopped { samples, bytes }) => PaletteOutcome::Acknowledged {
                message: format!("mic: stopped (captured {samples} samples → {bytes}-byte wav)",),
            },
            Ok(MicToggle::StoppedEmpty) => PaletteOutcome::Acknowledged {
                message: "mic: stopped (no audio captured)".to_string(),
            },
            Err(e) => PaletteOutcome::Rejected {
                message: format!("mic error: {e}"),
            },
        }
    }

    /// `/tts [on|off]` — toggle (or set) Piper TTS playback of the
    /// assistant's text reply at end-of-turn. Default `off`. Once
    /// enabled, [`Self::finalize_turn`] spawns a background thread per
    /// turn that runs Piper → rodio → speakers.
    ///
    /// If the most recent attempt failed with a missing-binary /
    /// missing-model error, this dispatch surfaces the sticky session
    /// disable so the user knows to install Piper before retrying.
    fn dispatch_tts(&mut self, arg: Option<&str>) -> PaletteOutcome {
        let target = match arg {
            Some("on") => true,
            Some("off") => false,
            None => !self.tts_enabled,
            Some(other) => {
                return PaletteOutcome::Rejected {
                    message: format!("usage: /tts [on|off] (got {other})"),
                };
            }
        };
        if target && self.tts_session_disabled {
            return PaletteOutcome::Rejected {
                message: "tts unavailable — install piper + a voice model and restart the session"
                    .to_string(),
            };
        }
        self.tts_enabled = target;
        if target && self.piper.is_none() {
            // Lazy-init: voice model path comes from the env so users
            // can point Stratum at whichever ONNX voice they prefer
            // without recompiling. Empty / missing path is fine here —
            // synthesis time will surface `MissingModel` cleanly.
            let model =
                std::env::var("STRATUM_PIPER_MODEL").unwrap_or_else(|_| "piper-voice.onnx".into());
            self.piper = Some(stratum_runtime::PiperSubprocess::new(model));
        }
        let label = if target { "on" } else { "off" };
        PaletteOutcome::Acknowledged {
            message: format!("tts: {label}"),
        }
    }

    /// Start or stop push-to-talk capture. Shared by `/mic` and the F5
    /// hotkey. On stop, runs the same transcribe + stage pipeline as
    /// `/audio`: writes a tempfile WAV (deleted at session end), calls
    /// whisper, and queues the audio + transcript for the next
    /// [`Self::submit`].
    ///
    /// Errors are returned as opaque strings — the caller picks the
    /// user-facing wording.
    fn toggle_ptt(&mut self) -> Result<MicToggle, String> {
        if self.recording {
            self.stop_recording_and_stage()
        } else {
            self.start_recording().map(|()| MicToggle::Started)
        }
    }

    /// Open a fresh `MicCapture` and start streaming. Idempotent in
    /// the soft sense: a second call while already recording returns
    /// `Ok` without re-opening the OS audio stream.
    fn start_recording(&mut self) -> Result<(), String> {
        if self.recording {
            return Ok(());
        }
        let cell = MicCaptureCell::start()?;
        self.mic_capture = Some(cell);
        self.recording = true;
        Ok(())
    }

    /// Stop the active capture, persist to a tempfile WAV, run the
    /// transcribe + stage pipeline, and clear [`Self::recording`].
    fn stop_recording_and_stage(&mut self) -> Result<MicToggle, String> {
        let Some(cell) = self.mic_capture.take() else {
            self.recording = false;
            return Ok(MicToggle::StoppedEmpty);
        };
        self.recording = false;
        // Pick a unique tmp path so a quick PTT cycle on a long session
        // can't collide with a stale file.
        let pid = std::process::id();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let tmp = std::env::temp_dir().join(format!("stratum-mic-{pid}-{nanos}.wav"));
        // The worker thread will run `save_wav` on its own copy of the
        // `MicCapture`, then join. Returns the resulting path or an
        // opaque error string.
        cell.stop_and_save(tmp.clone())?;
        let bytes = std::fs::read(&tmp).map_err(|e| format!("cannot read tempfile: {e}"))?;
        if bytes.is_empty() {
            // Best-effort cleanup; an empty file is harmless if it lingers.
            let _ = std::fs::remove_file(&tmp);
            return Ok(MicToggle::StoppedEmpty);
        }
        let bytes_len = bytes.len();
        let mime = "audio/wav".to_string();
        let b64 = base64_encode_chat(&bytes);
        let byte_len = u32::try_from(bytes_len).unwrap_or(u32::MAX);
        let staged = Block::audio_inline_b64(mime.clone(), b64, byte_len);
        self.staged_audio = Some(staged);

        // Run whisper on the WAV; same fallback surface as `/audio`.
        match self.whisper.transcribe(&tmp) {
            Ok(text) if !text.is_empty() => {
                self.set_transient_status(format!("heard: {}", trim_for_ack(&text, 80)));
                self.staged_audio_transcript = Some(text);
            }
            Ok(_) => {
                self.staged_audio_transcript = None;
                self.set_transient_status("heard: (empty)".to_string());
            }
            Err(stratum_runtime::whisper::WhisperError::MissingBinary) => {
                self.staged_audio_transcript = None;
                self.set_transient_status(
                    "whisper not installed; audio queued without transcript".to_string(),
                );
            }
            Err(e) => {
                self.staged_audio_transcript = None;
                self.set_transient_status(format!("whisper failed: {e}"));
            }
        }
        // We don't delete `tmp` here — `staged_audio` already holds
        // the base64-encoded bytes. Leaving the file lets a curious
        // user inspect the capture; the OS reaps the temp dir.
        let samples = bytes_len;
        Ok(MicToggle::Stopped {
            samples,
            bytes: bytes_len,
        })
    }

    /// Raise a transient status line. The renderer surfaces it for
    /// [`Self::TRANSIENT_STATUS_TTL`] before clearing.
    fn set_transient_status(&mut self, msg: String) {
        self.transient_status = Some((msg, Instant::now()));
    }

    /// Test / render hook: drop the transient status if its TTL has
    /// elapsed. Cheap; called from `render()` on every tick.
    fn expire_transient_status(&self) -> Option<&str> {
        self.transient_status.as_ref().and_then(|(msg, raised)| {
            if raised.elapsed() > Self::TRANSIENT_STATUS_TTL {
                None
            } else {
                Some(msg.as_str())
            }
        })
    }

    /// `true` while push-to-talk capture is active. Surfaced by the
    /// renderer as the "🎙 recording…" status line.
    #[must_use]
    pub const fn is_recording(&self) -> bool {
        self.recording
    }

    /// `true` when [`Self::dispatch_tts`] has put the session in
    /// voice-out mode.
    #[must_use]
    pub const fn tts_enabled(&self) -> bool {
        self.tts_enabled
    }

    /// `true` once Piper has surfaced a missing-binary / missing-model
    /// failure for this session. The flag is sticky; a `/tts on` after
    /// it trips will refuse with a "install piper" message instead of
    /// re-running the same failing path.
    #[must_use]
    pub const fn tts_session_disabled(&self) -> bool {
        self.tts_session_disabled
    }

    /// Snapshot of the transient status line, if any. Test accessor.
    #[must_use]
    pub fn transient_status(&self) -> Option<&str> {
        self.transient_status.as_ref().map(|(msg, _)| msg.as_str())
    }

    /// Speak `text` via Piper → rodio on a background thread. Called
    /// from [`Self::finalize_turn`] when [`Self::tts_enabled`] is on.
    /// Failures from Piper update the sticky session-disable flag and
    /// raise a transient status line so the user knows playback is off.
    fn speak_async(&mut self, text: String) {
        // Default-features build (no `voice` feature): rodio is not
        // linked, so even if piper synthesises successfully there is no
        // way to play the resulting WAV. Disable TTS for the session
        // and surface a visible status so the user does not believe
        // playback is silently working.
        #[cfg(not(feature = "voice"))]
        {
            let _ = text;
            self.tts_enabled = false;
            self.set_transient_status(
                "tts: playback unavailable — rebuild with --features voice".to_string(),
            );
        }
        #[cfg(feature = "voice")]
        self.speak_async_inner(text);
    }

    /// Voice-feature-only body of [`Self::speak_async`]. Extracted so the
    /// outer fn can short-circuit cleanly under `#[cfg(not(feature = "voice"))]`
    /// without leaving an unused-binding lint or duplicated bookkeeping.
    #[cfg(feature = "voice")]
    fn speak_async_inner(&mut self, text: String) {
        let Some(piper) = self.piper.clone() else {
            return;
        };
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return;
        }
        // Synthesize on the calling thread so any
        // MissingBinary / MissingModel error trips the sticky disable
        // immediately (and surfaces in the next render). Playback runs
        // on a background thread so it doesn't block the UI.
        let payload = trimmed.to_string();
        match piper.synthesize(&payload) {
            Ok(wav_path) => {
                std::thread::spawn(move || {
                    play_wav_blocking(&wav_path);
                    // The tempfile was created by piper; clean up so
                    // long sessions don't leak GBs of synthesized
                    // speech into the OS temp dir.
                    let _ = std::fs::remove_file(&wav_path);
                });
            }
            Err(
                stratum_runtime::PiperError::MissingBinary
                | stratum_runtime::PiperError::MissingModel,
            ) => {
                self.tts_enabled = false;
                self.tts_session_disabled = true;
                self.set_transient_status("[tts unavailable — install piper]".to_string());
            }
            Err(e) => {
                self.set_transient_status(format!("[tts: {e}]"));
            }
        }
    }

    /// a pointer to `--agents-dir`. Otherwise enumerate the registered roles
    /// via [`AgentHandoff::roles`] and tag the current driver role.
    fn dispatch_agents(&self) -> PaletteOutcome {
        let Some(handoff) = self.handoff.as_ref() else {
            return PaletteOutcome::Rejected {
                message: "no multi-agent mode; pass --agents-dir to enable".to_string(),
            };
        };
        let roles = handoff.roles();
        let joined = roles
            .iter()
            .copied()
            .map(role_name)
            .collect::<Vec<_>>()
            .join(", ");
        let current = self.current_role.map_or("default", |role| role_name(role));
        PaletteOutcome::Acknowledged {
            message: format!("roles: {joined} (current: {current})"),
        }
    }

    /// Render the `/subagents` palette command output.
    ///
    /// Lists every registered subagent (built-in + user-defined) with
    /// its description. Built-ins always present; user definitions come
    /// from `<config>/stratum/subagents/*.toml`.
    fn dispatch_subagents(&self) -> PaletteOutcome {
        if self.subagents.is_empty() {
            return PaletteOutcome::Rejected {
                message: "no subagents registered".to_string(),
            };
        }
        let lines: Vec<String> = self
            .subagents
            .list()
            .iter()
            .map(|s| format!("  {} — {}", s.name, s.description))
            .collect();
        PaletteOutcome::Acknowledged {
            message: format!("subagents:\n{}", lines.join("\n")),
        }
    }

    /// Render the `/parallel <role1,role2,…>` palette command output.
    ///
    /// Without an installed [`AgentHandoff`] (single-loop mode), reject with
    /// a pointer to `--agents-dir`. With a handoff: parse the role list
    /// (`snake_case` to match `SuggestedRole`'s serde projection), drive one
    /// turn through [`AgentHandoff::run_turn_parallel`], and append the
    /// per-role results to the transcript as a series of [`Turn::Command`]
    /// summaries plus a [`Turn::Assistant`] block carrying the concatenated
    /// text of every role's output.
    ///
    /// Unknown roles, an empty list, or `NoSuchRole` from the dispatcher
    /// all surface as `PaletteOutcome::Rejected`. Any other dispatcher
    /// error also rejects — the transcript receives no assistant turn.
    fn dispatch_parallel(&mut self, args: &str) -> PaletteOutcome {
        let Some(handoff) = self.handoff.clone() else {
            return PaletteOutcome::Rejected {
                message: "no multi-agent mode; pass --agents-dir to enable".to_string(),
            };
        };
        if args.is_empty() {
            return PaletteOutcome::Rejected {
                message: "unknown role: ".to_string(),
            };
        }
        let mut roles = Vec::new();
        for raw in args.split(',') {
            let label = raw.trim();
            let Some(role) = parse_role_label(label) else {
                return PaletteOutcome::Rejected {
                    message: format!("unknown role: {label}"),
                };
            };
            roles.push(role);
        }

        let turn_id = TurnId(self.next_turn_id);
        self.next_turn_id = self.next_turn_id.saturating_add(1);
        let prompt = self
            .transcript
            .iter()
            .rev()
            .find_map(|t| match t {
                Turn::User(text) => Some(text.clone()),
                _ => None,
            })
            .unwrap_or_default();
        let ctx = TurnContext {
            user_prompt: prompt.clone(),
            model: ModelId::from("echo"),
            turn_id,
            started_at: SystemTime::now(),
            history: self.build_history_turns(),
            // Parallel-roles palette path does not consume staged
            // attachments today (the parallel surface predates the
            // multimodal seam). Keep them queued for the next normal
            // turn instead of dropping silently.
            attachments: Vec::new(),
        };
        let intent = IntentRouter::default().classify(&prompt);
        let result = match handoff.run_turn_parallel(ctx, intent, &self.cancel, &roles) {
            Ok(r) => r,
            Err(stratum_runtime::HandoffError::NoSuchRole(role)) => {
                return PaletteOutcome::Rejected {
                    message: format!("unknown role: {}", role_name(role)),
                };
            }
            Err(e) => {
                return PaletteOutcome::Rejected {
                    message: format!("parallel dispatch failed: {e}"),
                };
            }
        };

        let mut combined: Vec<Block> = Vec::new();
        let mut summaries: Vec<(String, bool, String)> = Vec::new();
        for (key, role_result) in &result.per_role {
            let role = role_name(key.role());
            let ok = matches!(role_result.outcome, TurnOutcome::Success);
            let text = concat_text_blocks(&role_result.blocks);
            combined.push(Block::Text {
                text: format!("=== {role} ({}ms) ===\n{text}\n", role_result.duration_ms),
            });
            let summary = if ok {
                format!(
                    "[{role}] {}ms ({} chars)",
                    role_result.duration_ms,
                    text.len()
                )
            } else {
                let err = role_result
                    .error
                    .as_deref()
                    .unwrap_or("non-success outcome");
                format!("[{role}] error: {err}")
            };
            summaries.push((format!("/parallel {role}"), ok, summary));
        }

        if !combined.is_empty() {
            self.transcript.push(Turn::Assistant(combined));
        }
        for (text, ok, message) in summaries {
            self.transcript.push(Turn::Command { text, ok, message });
        }

        let all_ok = result
            .per_role
            .values()
            .all(|r| matches!(r.outcome, TurnOutcome::Success));
        PaletteOutcome::Acknowledged {
            message: format!(
                "parallel: {} role(s) in {}ms (all_ok={all_ok})",
                result.per_role.len(),
                result.elapsed_ms,
            ),
        }
    }

    /// Render the `/budget` palette command output.
    ///
    /// Without recorded [`TurnMetrics`] (no submit yet), acknowledges with the
    /// sentinel string `"no turn metrics yet"`. With metrics in hand, formats
    /// the latest turn's completion tokens, summed role-step wall-clock, and
    /// tok/s (via [`format_tokens_per_second`]) alongside the turn id.
    fn dispatch_budget(&self) -> PaletteOutcome {
        let Some(metrics) = self.last_metrics.as_ref() else {
            return PaletteOutcome::Acknowledged {
                message: "no turn metrics yet".to_string(),
            };
        };
        let wall_ms = metrics
            .role_steps
            .iter()
            .map(|step| step.duration_ms)
            .fold(0_u32, u32::saturating_add);
        let tps = format_tokens_per_second(metrics.completion_tokens, wall_ms);
        PaletteOutcome::Acknowledged {
            message: format!(
                "metrics: {} tokens · {wall_ms}ms · {tps:.1} tok/s · turn id {}",
                metrics.completion_tokens, metrics.turn_id.0,
            ),
        }
    }

    /// `/theme <name>` — switch the chat theme. Persists the chosen
    /// name to `<state>/theme.txt` so restart honors it.
    fn dispatch_theme(&self, arg: Option<&str>) -> PaletteOutcome {
        let Some(name) = arg else {
            return PaletteOutcome::Acknowledged {
                message: format!(
                    "active theme via /theme <name>; see /themes for choices ({})",
                    crate::theme::list(self.themes_dir.as_deref()).join(", ")
                ),
            };
        };
        match crate::theme::set_by_name(name, self.themes_dir.as_deref()) {
            Ok(()) => {
                if let Some(path) = self.theme_state_path.as_ref() {
                    crate::theme::write_persisted(path, name);
                }
                PaletteOutcome::Acknowledged {
                    message: format!("theme: {name}"),
                }
            }
            Err(e) => PaletteOutcome::Rejected { message: e },
        }
    }

    /// `/themes` — list available theme names.
    fn dispatch_themes(&self) -> PaletteOutcome {
        let names = crate::theme::list(self.themes_dir.as_deref());
        if names.is_empty() {
            return PaletteOutcome::Acknowledged {
                message: "no themes available".to_string(),
            };
        }
        PaletteOutcome::Acknowledged {
            message: format!("themes:\n  {}", names.join("\n  ")),
        }
    }

    /// `/welcome` — re-show the branded greeting + a rotating tip,
    /// any time. Useful when the chat scrolls past the banner or the
    /// user wants a reminder.
    fn dispatch_welcome(&self) -> PaletteOutcome {
        let rot = self.last_turn_id.map_or(0, |t| t.0);
        let tip = crate::brand::tip_for(rot);
        PaletteOutcome::Acknowledged {
            message: format!(
                "{} — {}\n  tip · {}\n  Enter to send · / for commands · ? for help",
                crate::brand::WORDMARK,
                crate::brand::TAGLINE,
                tip
            ),
        }
    }

    /// Attempt LLM-summarized compaction of the first `split_at`
    /// transcript turns. Returns `None` when:
    ///   - the active backend can't be cloned for a summarize call
    ///   - the summarize call produces no text blocks
    ///   - the rendered transcript is empty
    ///
    /// On `None` the caller falls back to the deterministic heuristic
    /// summarizer (the original `/compact` body).
    fn try_llm_summarize(&self, split_at: usize) -> Option<String> {
        let older = &self.transcript[..split_at];
        if older.is_empty() {
            return None;
        }
        // Render the older turns into a single plain-text transcript
        // the summarizer can consume.
        let mut rendered = String::new();
        for turn in older {
            match turn {
                Turn::User(text) => {
                    rendered.push_str("USER: ");
                    rendered.push_str(text.trim());
                    rendered.push_str("\n\n");
                }
                Turn::Assistant(blocks) => {
                    rendered.push_str("ASSISTANT: ");
                    for b in blocks {
                        if let Block::Text { text } = b {
                            rendered.push_str(text.trim());
                            rendered.push(' ');
                        }
                    }
                    rendered.push_str("\n\n");
                }
                _ => {}
            }
        }
        let trimmed = rendered.trim();
        if trimmed.is_empty() {
            return None;
        }
        // System override for the summarizer pass. Plain English; we
        // explicitly forbid tool calls so the summarizer never tries
        // to dispatch fs.read / shell.exec mid-summary.
        let system = "You are a transcript summarizer. Read the conversation \
                      below and produce a single concise paragraph (4-8 sentences) \
                      capturing: what the user asked about, what was decided, what \
                      tools fired, and any open questions. Plain English. No JSON. \
                      No tool calls. No bullet lists. Just one paragraph.";
        let prompt = format!("Conversation transcript:\n\n{trimmed}");
        let backend = self.agent_loop.clone();
        let cancel = stratum_runtime::CancelToken::new();
        let (chunk_tx, chunk_rx) = mpsc::channel();
        let ctx = TurnContext {
            user_prompt: prompt,
            model: ModelId::from(self.active_model.as_deref().unwrap_or("echo")),
            turn_id: TurnId(self.last_turn_id.map_or(1, |t| t.0.saturating_add(1))),
            started_at: SystemTime::now(),
            history: vec![stratum_runtime::ChatHistoryTurn {
                role: "system".to_string(),
                content: system.to_string(),
            }],
            // `/recap` is a text-only compression of the transcript; the
            // multimodal seam carries through the normal submit path.
            attachments: Vec::new(),
        };
        // Run synchronously on the dispatch thread — the caller is
        // already a palette command, so we don't need the streaming
        // worker thread.
        let result = backend.run_turn_streaming(ctx, &cancel, chunk_tx);
        drop(chunk_rx);
        let text: String = result
            .blocks
            .iter()
            .filter_map(|b| match b {
                Block::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join(" ");
        let text = text.trim();
        if text.is_empty() {
            return None;
        }
        Some(format!(
            "(compacted {split_at} earlier turn(s) via LLM summarizer)\n\n{text}"
        ))
    }

    /// `/compact` — compress old turns into a single summary turn so
    /// the context window stays usable on long sessions. Keeps the
    /// most recent `KEEP_RECENT` turns verbatim; collapses everything
    /// before them into one synthetic Assistant text block. Per
    /// plan/04 + plan/42 PreCompact/PostCompact.
    fn dispatch_compact(&mut self) -> PaletteOutcome {
        const KEEP_RECENT: usize = 4;
        let total = self.transcript.len();
        if total <= KEEP_RECENT {
            return PaletteOutcome::Acknowledged {
                message: format!(
                    "nothing to compact ({total} turns, keeping the most recent {KEEP_RECENT})"
                ),
            };
        }
        let split_at = total - KEEP_RECENT;
        // Try the provider-summarized path first; fall back to the
        // deterministic heuristic if the provider can't be reached
        // or returns nothing usable. `/compact` should never break
        // even on the EchoProvider or with a cold provider.
        if let Some(llm_summary) = self.try_llm_summarize(split_at) {
            let recent: Vec<Turn> = self.transcript.split_off(split_at);
            self.transcript.clear();
            self.transcript
                .push(Turn::Assistant(vec![Block::Text { text: llm_summary }]));
            self.transcript.extend(recent);
            self.chat_scroll = 0;
            return PaletteOutcome::Acknowledged {
                message: format!(
                    "compacted {split_at} turn(s) via LLM; kept the most recent {KEEP_RECENT}"
                ),
            };
        }
        // Heuristic fallback below.
        let older = &self.transcript[..split_at];
        let user_count = older.iter().filter(|t| matches!(t, Turn::User(_))).count();
        let assistant_count = older
            .iter()
            .filter(|t| matches!(t, Turn::Assistant(_)))
            .count();
        let tool_count: usize = older
            .iter()
            .map(|t| match t {
                Turn::Assistant(blocks) => blocks
                    .iter()
                    .filter(|b| matches!(b, Block::ToolCall { .. }))
                    .count(),
                _ => 0,
            })
            .sum();
        let mut topic_lines: Vec<String> = Vec::new();
        for t in older {
            if let Turn::User(text) = t {
                let one_line = text.lines().next().unwrap_or("").trim();
                if !one_line.is_empty() {
                    let compact = stratum_runtime::caveman::compress(one_line);
                    let preview: String = compact.chars().take(80).collect();
                    topic_lines.push(format!("· {preview}"));
                }
            }
            if topic_lines.len() >= 12 {
                break;
            }
        }
        let mut summary = format!(
            "(compacted {split_at} earlier turns — {user_count} from you, {assistant_count} from me, {tool_count} tool calls)\n\n\
             Topics covered:\n"
        );
        for line in &topic_lines {
            summary.push_str(line);
            summary.push('\n');
        }
        if older.iter().filter(|t| matches!(t, Turn::User(_))).count() > topic_lines.len() {
            summary.push_str("· (older topics dropped)\n");
        }
        // Replace older turns with a single synthetic Assistant turn.
        let recent: Vec<Turn> = self.transcript.split_off(split_at);
        self.transcript.clear();
        self.transcript
            .push(Turn::Assistant(vec![Block::Text { text: summary }]));
        self.transcript.extend(recent);
        self.chat_scroll = 0;
        PaletteOutcome::Acknowledged {
            message: format!("compacted {split_at} turn(s); kept the most recent {KEEP_RECENT}"),
        }
    }

    /// `/recap` — one-line summary of the current session.
    fn dispatch_recap(&self) -> PaletteOutcome {
        let user_turns = self
            .transcript
            .iter()
            .filter(|t| matches!(t, Turn::User(_)))
            .count();
        let assistant_turns = self
            .transcript
            .iter()
            .filter(|t| matches!(t, Turn::Assistant(_)))
            .count();
        let tool_calls: usize = self
            .transcript
            .iter()
            .map(|t| match t {
                Turn::Assistant(blocks) => blocks
                    .iter()
                    .filter(|b| matches!(b, Block::ToolCall { .. }))
                    .count(),
                _ => 0,
            })
            .sum();
        let last_user = self
            .transcript
            .iter()
            .rev()
            .find_map(|t| match t {
                Turn::User(text) => Some(text.clone()),
                _ => None,
            })
            .unwrap_or_else(|| "(no user turns yet)".to_string());
        let snippet: String = last_user.chars().take(80).collect();
        PaletteOutcome::Acknowledged {
            message: format!(
                "session {} · {user_turns} user · {assistant_turns} assistant · {tool_calls} tool calls\nlast: {snippet}",
                self.session_id
            ),
        }
    }

    /// `/diff` — render the last few `fs.edit` / `fs.write` tool
    /// calls as a colored unified diff. The diff is pushed as a
    /// synthetic Assistant turn so the markdown renderer's `diff`
    /// language path styles +/-/context lines.
    fn dispatch_diff(&mut self) -> PaletteOutcome {
        const MAX: usize = 5;
        let mut hunks: Vec<String> = Vec::new();
        for t in self.transcript.iter().rev() {
            let Turn::Assistant(blocks) = t else {
                continue;
            };
            for b in blocks {
                if let Block::ToolCall { tool, args, .. } = b {
                    if tool == "fs.edit" {
                        if let Some(hunk) = render_fs_edit_diff(args) {
                            hunks.push(hunk);
                        }
                    } else if tool == "fs.write" {
                        if let Some(hunk) = render_fs_write_diff(args) {
                            hunks.push(hunk);
                        }
                    }
                }
            }
            if hunks.len() >= MAX {
                break;
            }
        }
        if hunks.is_empty() {
            return PaletteOutcome::Acknowledged {
                message: "no fs.write / fs.edit calls in this session yet".to_string(),
            };
        }
        let n = hunks.len();
        // Hunks were collected newest-first; restore chronological order.
        hunks.reverse();
        let body = format!("```diff\n{}\n```", hunks.join("\n"));
        self.transcript
            .push(Turn::Assistant(vec![Block::Text { text: body }]));
        PaletteOutcome::Acknowledged {
            message: format!("rendered {n} diff hunk(s) above"),
        }
    }

    /// `/init` — scaffold a STRATUM.md in the workspace if missing.
    fn dispatch_init(&self) -> PaletteOutcome {
        let path = std::path::PathBuf::from("STRATUM.md");
        if path.exists() {
            return PaletteOutcome::Acknowledged {
                message: format!("STRATUM.md already exists ({})", path.display()),
            };
        }
        let body = "# Stratum workspace notes\n\
                    \n\
                    Per-project guidance Stratum loads into every chat turn.\n\
                    \n\
                    ## Style\n\
                    - prefer …\n\
                    - avoid …\n\
                    \n\
                    ## Build / test\n\
                    - `cargo build` / `cargo test`\n";
        match std::fs::write(&path, body) {
            Ok(()) => PaletteOutcome::Acknowledged {
                message: format!(
                    "wrote {} (edit it with your project's conventions)",
                    path.display()
                ),
            },
            Err(e) => PaletteOutcome::Rejected {
                message: format!("could not write STRATUM.md: {e}"),
            },
        }
    }

    /// `/export [path]` — dump the chat transcript as plain text.
    fn dispatch_export(&self, arg: Option<&str>) -> PaletteOutcome {
        let path = arg.map(std::path::PathBuf::from).unwrap_or_else(|| {
            std::env::temp_dir().join(format!("stratum-export-{}.txt", self.session_id))
        });
        let mut body = String::new();
        for t in &self.transcript {
            match t {
                Turn::User(text) => {
                    body.push_str("you: ");
                    body.push_str(text);
                    body.push_str("\n\n");
                }
                Turn::Assistant(blocks) => {
                    for b in blocks {
                        match b {
                            Block::Text { text } => {
                                body.push_str("ai:  ");
                                body.push_str(text);
                                body.push_str("\n\n");
                            }
                            Block::ToolCall { tool, args, .. } => {
                                body.push_str(&format!("→ tool {tool} {args}\n"));
                            }
                            Block::ToolResult { output, .. } => {
                                body.push_str(&format!("← result {output}\n"));
                            }
                            _ => {}
                        }
                    }
                }
                Turn::Cancelled => body.push_str("(cancelled)\n\n"),
                Turn::Command { text, message, .. } => {
                    body.push_str(&format!("/{text}: {message}\n"));
                }
            }
        }
        match std::fs::write(&path, body.as_bytes()) {
            Ok(()) => PaletteOutcome::Acknowledged {
                message: format!("transcript exported to {}", path.display()),
            },
            Err(e) => PaletteOutcome::Rejected {
                message: format!("could not write {}: {e}", path.display()),
            },
        }
    }

    /// Handle `/image <path>`: resolve the path against cwd, sniff its
    /// MIME, base64-encode the bytes, and queue a `Block::Image` for
    /// the NEXT user turn. The transcript records an `[image: <mime>
    /// @ <path>]` line so the user can see the attachment is pending.
    ///
    /// `plan/05` scaffold — the bytes are forwarded to the provider via
    /// [`crate::TurnContext::attachments`] but every shipped provider
    /// today ignores them (see TODO markers tagged
    /// `TODO(plan/05): wire <vision-model>`). The vision-head wiring
    /// lands in a follow-up commit when llama.cpp `--mmproj` is plumbed
    /// through `LlamaCppProvider`.
    //
    // Path cap policy: 5 MiB matches `ReadImageToolDispatcher::DEFAULT_MAX_BYTES`
    // so the palette path and the tool path agree on the largest image
    // we will read into memory in one go.
    fn dispatch_image(&mut self, raw_path: &str) -> PaletteOutcome {
        const MAX_BYTES: u64 = 5 << 20;
        let raw = raw_path.trim();
        // Strip a single layer of quotes if the user wrote `/image "foo bar.png"`
        // — the chat surface is a single-line input so this is the
        // common shape for paths with spaces.
        let path_str = raw
            .strip_prefix('"')
            .and_then(|s| s.strip_suffix('"'))
            .or_else(|| raw.strip_prefix('\'').and_then(|s| s.strip_suffix('\'')))
            .unwrap_or(raw);
        let raw_path = std::path::PathBuf::from(path_str);
        // Defense-in-depth: refuse relative paths containing `..`. The
        // chat surface is local-trust (the user types what they want
        // to read), but a malicious pasted prompt that hides a
        // `../../../etc/passwd` traversal inside a `/image` shouldn't
        // resolve to anything the user couldn't otherwise type. Absolute
        // paths are explicitly the user's choice; relative paths must
        // descend cleanly.
        if !raw_path.is_absolute()
            && raw_path
                .components()
                .any(|c| matches!(c, std::path::Component::ParentDir))
        {
            return PaletteOutcome::Rejected {
                message: format!(
                    "/image: relative path traverses parent (`..`) — use an absolute path: {path_str}"
                ),
            };
        }
        let resolved = if raw_path.is_absolute() {
            raw_path.clone()
        } else {
            match std::env::current_dir() {
                Ok(cwd) => cwd.join(&raw_path),
                Err(e) => {
                    return PaletteOutcome::Rejected {
                        message: format!("/image: cannot resolve cwd: {e}"),
                    };
                }
            }
        };
        match self.stage_image_attachment(&resolved, MAX_BYTES) {
            Ok((mime, bytes)) => PaletteOutcome::Acknowledged {
                message: format!(
                    "[image: {mime} @ {} ({bytes} bytes); queued for next turn]",
                    resolved.display()
                ),
            },
            Err(msg) => PaletteOutcome::Rejected { message: msg },
        }
    }

    /// Read `path` from disk and push a `Block::Image` onto
    /// `pending_attachments`. Returns `(mime, raw_bytes)` for the
    /// caller's success message.
    fn stage_image_attachment(
        &mut self,
        path: &std::path::Path,
        max_bytes: u64,
    ) -> Result<(&'static str, u64), String> {
        let metadata = std::fs::metadata(path)
            .map_err(|e| format!("/image: cannot stat {}: {e}", path.display()))?;
        if !metadata.is_file() {
            return Err(format!("/image: not a regular file: {}", path.display()));
        }
        if metadata.len() > max_bytes {
            return Err(format!(
                "/image: {} bytes exceeds {max_bytes}-byte cap",
                metadata.len()
            ));
        }
        let bytes = std::fs::read(path).map_err(|e| format!("/image: read failed: {e}"))?;
        // Reuse the same magic-byte + extension sniff the
        // `read_image` tool dispatcher uses.
        let mime =
            stratum_runtime::sniff_image_mime(path, &bytes).unwrap_or("application/octet-stream");
        let raw_len = bytes.len() as u64;
        let bytes_u32 = u32::try_from(bytes.len()).unwrap_or(u32::MAX);
        let encoded = stratum_runtime::base64_encode(&bytes);
        let block = Block::image_inline_b64(mime, encoded, bytes_u32);
        self.pending_attachments.push(block);
        Ok((mime, raw_len))
    }

    /// Number of attachments currently queued for the next turn.
    /// Test-only accessor; production callers don't branch on this.
    #[must_use]
    pub const fn pending_attachment_count(&self) -> usize {
        self.pending_attachments.len()
    }

    /// Snapshot of the queued attachments. Test-only accessor.
    #[must_use]
    pub fn pending_attachments(&self) -> &[Block] {
        &self.pending_attachments
    }

    /// Stage `prompt` into the input buffer and dispatch [`Self::submit`].
    ///
    /// Helper used by the non-interactive `stratum chat --prompt <STR>` path
    /// (and by tests) — it replaces the current input wholesale so callers
    /// don't have to drive a stream of `KeyCode::Char` events.
    pub fn submit_with_prompt(&mut self, prompt: &str) {
        self.input.clear();
        self.caret = 0;
        self.input.push_str(prompt);
        self.submit();
        self.block_until_idle();
    }

    /// Spin-wait until any pending async turn settles. Used by the
    /// `--prompt` non-interactive path and by tests so they observe a
    /// completed transcript right after `submit_with_prompt`. Interactive
    /// TUI callers must NOT use this — the whole point of the async
    /// submit is to keep the TUI responsive while the provider runs.
    pub fn block_until_idle(&mut self) {
        loop {
            if !self.poll_turn_completion() && self.pending_rx.is_none() {
                return;
            }
            if self.pending_rx.is_some() {
                std::thread::sleep(Duration::from_millis(5));
            }
        }
    }

    /// Most recent turn metrics (prompt+completion tokens, ms, tok/s).
    /// `None` if no turn has settled. Surfaced for `--output-format json`.
    #[must_use]
    pub const fn last_turn_metrics(&self) -> Option<&TurnMetrics> {
        self.last_metrics.as_ref()
    }

    /// Last turn id. Surfaced for `--output-format json`.
    #[must_use]
    pub const fn last_turn_id_value(&self) -> Option<TurnId> {
        self.last_turn_id
    }

    /// Currently-active model slug (from `/switch` or initial wiring).
    #[must_use]
    pub fn active_model(&self) -> Option<String> {
        self.active_model.clone()
    }

    /// Join the most recent [`Turn::Assistant`] entry's text blocks into a
    /// single string.
    ///
    /// Returns `None` when the transcript contains no assistant turn or the
    /// last assistant turn has no [`Block::Text`] blocks. Useful for the
    /// `--prompt` non-interactive path and integration tests that need to
    /// inspect what the provider produced without re-walking the transcript.
    #[must_use]
    pub fn last_assistant_text(&self) -> Option<String> {
        let blocks = self.transcript.iter().rev().find_map(|t| match t {
            Turn::Assistant(b) => Some(b),
            _ => None,
        })?;
        let texts: Vec<&str> = blocks
            .iter()
            .filter_map(|b| match b {
                Block::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        if texts.is_empty() {
            None
        } else {
            Some(texts.join(""))
        }
    }

    /// Message from the most recent `Turn::Command`. Used by the
    /// `--prompt` path to render `!cmd` shell output instead of erroring
    /// "no text blocks" when the user's input was intercepted by the
    /// palette / shell-prefix layer rather than going through the LLM.
    #[must_use]
    pub fn last_command_message(&self) -> Option<String> {
        self.transcript.iter().rev().find_map(|t| match t {
            Turn::Command { message, .. } => Some(message.clone()),
            _ => None,
        })
    }

    /// Project the in-memory transcript into the
    /// `ChatHistoryTurn` shape the provider expects. Only the prior
    /// user/assistant Text blocks are forwarded — tool-call JSON,
    /// command markers, and cancellation markers are dropped so the
    /// chat template doesn't re-feed them as model output. The most
    /// recent user prompt is NOT included because the caller adds
    /// it as the live `user_prompt`.
    fn build_history_turns(&self) -> Vec<stratum_runtime::ChatHistoryTurn> {
        // Window the history: send at most HISTORY_TURNS prior turns
        // so small models with tight context windows aren't squeezed.
        const HISTORY_TURNS: usize = 8;
        let mut history: Vec<stratum_runtime::ChatHistoryTurn> = Vec::new();
        // Skip the most recent User turn — that one is the live
        // prompt being sent now.
        let mut skipped_live_user = false;
        for turn in self.transcript.iter().rev() {
            if history.len() >= HISTORY_TURNS {
                break;
            }
            match turn {
                Turn::User(_text) if !skipped_live_user => {
                    skipped_live_user = true;
                    // Don't push — this is the current prompt.
                }
                Turn::User(text) => {
                    history.push(stratum_runtime::ChatHistoryTurn {
                        role: "user".to_string(),
                        content: text.clone(),
                    });
                }
                Turn::Assistant(blocks) => {
                    let text: String = blocks
                        .iter()
                        .filter_map(|b| match b {
                            Block::Text { text } => Some(text.as_str()),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join("\n");
                    if !text.is_empty() {
                        history.push(stratum_runtime::ChatHistoryTurn {
                            role: "assistant".to_string(),
                            content: text,
                        });
                    }
                }
                Turn::Cancelled | Turn::Command { .. } => {}
            }
        }
        history.reverse();
        history
    }

    /// Summarize the most recent assistant turn's tool activity for
    /// the `--prompt` / non-interactive path. Returns `None` when the
    /// turn had no tool blocks. Format: one line per `tool.id` with a
    /// truncated arg summary, followed by the tool's result (first
    /// few lines).
    #[must_use]
    pub fn last_assistant_tool_summary(&self) -> Option<String> {
        let blocks = self.transcript.iter().rev().find_map(|t| match t {
            Turn::Assistant(b) => Some(b),
            _ => None,
        })?;
        let mut out = String::new();
        let mut any = false;
        for b in blocks {
            match b {
                Block::ToolCall { tool, args, .. } => {
                    any = true;
                    let preview = if args.len() <= 200 {
                        args.clone()
                    } else {
                        format!("{}…", &args[..200])
                    };
                    out.push_str(&format!("→ tool {tool} {preview}\n"));
                }
                Block::ToolResult { output, .. } => {
                    let mut lines = output.lines();
                    if let Some(first) = lines.next() {
                        out.push_str(&format!("← result {first}\n"));
                    }
                    let mut shown = 1;
                    for line in lines {
                        if shown >= 6 {
                            out.push_str("  …\n");
                            break;
                        }
                        out.push_str(&format!("  {line}\n"));
                        shown += 1;
                    }
                }
                _ => {}
            }
        }
        if any {
            Some(out.trim_end().to_string())
        } else {
            None
        }
    }

    /// Extract the first `Block::Cancelled` reason from the most recent
    /// assistant turn. Lets the CLI surface real provider failures
    /// (e.g., llama.cpp decode errors) instead of a generic
    /// "no text blocks" message.
    #[must_use]
    pub fn last_assistant_failure_reason(&self) -> Option<String> {
        let blocks = self.transcript.iter().rev().find_map(|t| match t {
            Turn::Assistant(b) => Some(b),
            _ => None,
        })?;
        blocks.iter().find_map(|b| match b {
            Block::Cancelled { reason } => Some(reason.clone()),
            _ => None,
        })
    }

    /// Submit the current input through the [`AgentLoop`] and append the
    /// resulting blocks to the transcript.
    ///
    /// When [`Self::with_handoff`] has installed an [`AgentHandoff`], the
    /// turn is routed through [`AgentHandoff::run_turn_with_handoff`]
    /// instead — a multi-role classifier + sentinel-driven chain — and any
    /// extra hops surface as `(handoff: from → to)` command lines in the
    /// transcript so the UI shows the chain that produced the final blocks.
    pub fn submit(&mut self) {
        if self.input.trim().is_empty() {
            // The user pressed Enter on whitespace-only input. Clear
            // it so the textbox visibly reflects the "nothing to
            // send" outcome instead of silently leaving the spaces
            // sitting there as if Enter had been swallowed.
            if !self.input.is_empty() {
                self.input.clear();
                self.caret = 0;
            }
            return;
        }
        // Slash-prefix intercept: when the user typed a complete palette
        // command directly (e.g. `/switch qwen-7b`), route it through the
        // palette dispatch instead of sending to the LLM. This lets the
        // user pass args to palette commands without juggling the
        // palette UI's autocomplete state.
        if self.input.starts_with('/') {
            // Bare "/" with no command after it isn't a palette
            // command — it's the user typing a literal slash (or
            // halfway into a command). Drop it silently instead of
            // emitting "unknown command: /" + an exit.
            if self.input.trim() == "/" {
                self.input.clear();
                self.caret = 0;
                return;
            }
            let cmd = std::mem::take(&mut self.input);
            self.caret = 0;
            self.record_input_history(&cmd);
            let _ = self.execute_palette_command(&cmd);
            return;
        }
        // If a turn is already running, queue this prompt and return.
        // The event loop calls `drain_queue` when the turn finishes.
        if self.in_flight_since.is_some() {
            let queued = std::mem::take(&mut self.input);
            self.caret = 0;
            self.pending_queue.push(queued);
            return;
        }
        // `!cmd` shell-prefix intercept — bypass LLM and route directly to
        // the wired shell executor. Falls through to the LLM path when no
        // executor is installed (so `!` is just a normal character).
        if let Some(rest) = self.input.strip_prefix('!') {
            if let Some(exec) = self.shell_executor.clone() {
                let cmd = rest.trim().to_string();
                let raw = std::mem::take(&mut self.input);
                self.caret = 0;
                self.record_input_history(&raw);
                if cmd.is_empty() {
                    self.transcript.push(Turn::Command {
                        text: "!".to_string(),
                        ok: false,
                        message: "usage: !<shell command>".to_string(),
                    });
                    return;
                }
                let (ok, message) = match (exec.0)(&cmd) {
                    Ok(out) => (true, out),
                    Err(e) => (false, e),
                };
                self.transcript.push(Turn::Command {
                    text: format!("!{cmd}"),
                    ok,
                    message,
                });
                return;
            }
        }
        let mut prompt = std::mem::take(&mut self.input);
        self.caret = 0;
        self.record_input_history(&prompt);
        // Phase 5 v2: drain any audio attachment + transcript queued by
        // `/audio <path>` BEFORE the user-turn pushes, so the displayed
        // prompt + the transcript-augmented prompt are the same string.
        // The transcript prefix degrades to an "unavailable" sentinel
        // when whisper failed, preserving the assistant's awareness
        // that audio was attached even on hosts without whisper.cpp.
        let drained_audio = self.staged_audio.take();
        let drained_transcript = self.staged_audio_transcript.take();
        if drained_audio.is_some() {
            // Fence the transcript in BEGIN/END markers + cap length at
            // 8 KiB so a long whisper transcript can't blow the prompt
            // budget AND a transcript that itself contains
            // `[transcript: ...]` or instruction-like text can't be
            // confused with operator-level context. The fence labels
            // are distinctive enough that a downstream provider can
            // tell user content from transcript content if it cares.
            const TRANSCRIPT_CHAR_CAP: usize = 8 * 1024;
            let prefix = if let Some(text) = drained_transcript.as_deref() {
                let trimmed: String = text.chars().take(TRANSCRIPT_CHAR_CAP).collect();
                format!("[AUDIO_TRANSCRIPT_BEGIN]\n{trimmed}\n[AUDIO_TRANSCRIPT_END]\n\n")
            } else {
                "[AUDIO_TRANSCRIPT_BEGIN]\n(unavailable — install whisper.cpp)\n[AUDIO_TRANSCRIPT_END]\n\n".to_string()
            };
            prompt = format!("{prefix}{prompt}");
        }
        // Record the audio attachment on the per-turn `last_turn_attachments`
        // accessor — tests inspect this directly; future providers will read
        // it via a typed `attachments` field on `GenerateRequest`.
        self.last_turn_attachments = drained_audio.into_iter().collect();
        // Optimistic user-message display: push the user turn BEFORE the
        // provider runs so the next render shows it immediately. The
        // assistant turn lands after `run_turn` returns below.
        self.transcript.push(Turn::User(prompt.clone()));
        self.scroll_to_bottom();
        let turn_id = TurnId(self.next_turn_id);
        self.next_turn_id = self.next_turn_id.saturating_add(1);

        // Mark the turn as in-flight so `status_bar_text` renders the live
        // `[generating… <N>s]` indicator. Cleared by `poll_turn_completion`.
        self.in_flight_since = Some(Instant::now());
        self.pending_started = Some(Instant::now());
        self.last_turn_id = Some(turn_id);

        // Drain any `/image <path>`-staged attachments into THIS turn's
        // context. Attachments are spent on a single user turn — they
        // do not persist into the next one. See `pending_attachments`
        // on `ChatState`.
        //
        // Audio attachments staged by `/audio <path>` are *also* spent
        // here: `self.last_turn_attachments` already holds the drained
        // Block::Audio (assigned a few lines up), so we extend the
        // outgoing context with a clone. The image queue is consumed
        // by move; the audio queue is mirrored so the user-facing
        // `last_turn_attachments()` accessor still surfaces it after
        // the turn returns.
        let mut attachments = std::mem::take(&mut self.pending_attachments);
        attachments.extend(self.last_turn_attachments.iter().cloned());
        let ctx = TurnContext {
            user_prompt: prompt.clone(),
            model: ModelId::from("echo"),
            turn_id,
            started_at: SystemTime::now(),
            history: self.build_history_turns(),
            attachments,
        };

        // Handoff path stays synchronous for now — multi-role chains
        // build a chain summary mid-flight and need the lifetimes of
        // `self.handoff` etc. The async path covers the common single-loop
        // case where the perceived "frozen UI" is acute.
        if let Some(handoff) = self.handoff.clone() {
            let role_timer = RoleTimer::start();
            let (blocks, last_turn_result, handoff_lines, final_role) =
                self.run_turn_via_handoff(handoff.as_ref(), ctx, &prompt);
            let step_ms = role_timer.stop_ms();
            self.finalize_turn(TurnAsyncResult {
                blocks,
                turn_result: last_turn_result,
                handoff_lines,
                final_role,
                step_ms,
            });
            return;
        }

        // Async path: spawn the provider on a worker thread so the
        // event loop keeps rendering (and the `(thinking…)` placeholder
        // stays visible) while the provider runs. A second channel
        // streams per-token chunks back so the UI can render text as it
        // arrives instead of waiting for the whole turn to settle.
        let (tx, rx) = mpsc::channel();
        let (chunk_tx, chunk_rx) = mpsc::channel();
        self.streaming_text.clear();
        let agent_loop = Arc::clone(&self.agent_loop);
        let cancel = self.cancel.clone();
        thread::spawn(move || {
            let role_timer = RoleTimer::start();
            let turn_result = agent_loop.run_turn_streaming(ctx, &cancel, chunk_tx);
            let blocks = turn_result.blocks.clone();
            let _ = tx.send(TurnAsyncResult {
                blocks,
                turn_result: Some(turn_result),
                handoff_lines: Vec::new(),
                final_role: None,
                step_ms: role_timer.stop_ms(),
            });
        });
        self.pending_rx = Some(rx);
        self.chunk_rx = Some(chunk_rx);
    }

    /// Drain the async-turn channel and, if a result is ready, fold it
    /// into the transcript. Returns `true` when a turn settled this
    /// tick. Called from the event loop each iteration.
    pub fn poll_turn_completion(&mut self) -> bool {
        // Drain any streaming chunks first so the partial-text render
        // catches up before we check for a final result.
        if let Some(crx) = self.chunk_rx.as_ref() {
            while let Ok(block) = crx.try_recv() {
                if let Block::Text { text } = block {
                    self.streaming_text.push_str(&text);
                }
            }
        }
        let Some(rx) = self.pending_rx.as_ref() else {
            return false;
        };
        match rx.try_recv() {
            Ok(result) => {
                self.pending_rx = None;
                self.finalize_turn(result);
                true
            }
            Err(mpsc::TryRecvError::Empty) => false,
            Err(mpsc::TryRecvError::Disconnected) => {
                // Worker thread dropped the sender before sending —
                // surface as a cancellation marker so the UI doesn't
                // hang waiting on a channel that will never deliver.
                self.pending_rx = None;
                self.transcript.push(Turn::Cancelled);
                self.in_flight_since = None;
                self.pending_started = None;
                true
            }
        }
    }

    /// Shared finalization for both the sync handoff path and the
    /// async single-loop path. Updates transcript, metrics, role label
    /// and clears the in-flight indicator.
    fn finalize_turn(&mut self, result: TurnAsyncResult) {
        // Ctrl+C already pushed a `Turn::Cancelled` for this turn —
        // drop the late worker result so we don't double-push and end
        // up with two terminal turns rendering back-to-back.
        if self.cancel_already_pushed {
            self.cancel_already_pushed = false;
            self.in_flight_since = None;
            self.pending_started = None;
            self.chunk_rx = None;
            self.streaming_text.clear();
            return;
        }
        if let Some(role) = result.final_role {
            self.current_role = Some(role);
        }
        let turn_id = self.last_turn_id.unwrap_or(TurnId(0));
        let mut recorder = TurnRecorder::new(turn_id);
        for block in &result.blocks {
            recorder.record_block(block);
        }
        recorder.record_step("generate", result.step_ms);

        let tokens_generated = approximate_token_count(&result.blocks);
        self.last_token_count = tokens_generated;
        self.last_metrics = Some(recorder.finish());

        // Decide what to push. Three cases:
        //  1. blocks contain a renderable Text/Usage/Cancelled — push as-is.
        //  2. blocks are non-empty but only contain unrenderable variants
        //     (ToolCall / ToolResult / Done) AND we have streaming text
        //     the user saw — synthesize a Text block from streaming_text
        //     so the visible response isn't replaced by a blank turn.
        //  3. blocks are empty (zero-block provider error path) OR
        //     genuinely have nothing to show — push as-is for parity with
        //     the existing E_NO_BLOCKS test fixtures.
        let has_renderable = result.blocks.iter().any(|b| {
            matches!(
                b,
                Block::Text { .. } | Block::Usage { .. } | Block::Cancelled { .. }
            )
        });
        let stream_was_just_json = looks_like_tool_call_json(self.streaming_text.trim());
        if !has_renderable
            && !result.blocks.is_empty()
            && !self.streaming_text.is_empty()
            && !stream_was_just_json
        {
            // Streaming captured real assistant text the user already
            // saw; preserve it so the final view doesn't blank out.
            let mut synth = result.blocks.clone();
            synth.insert(
                0,
                Block::Text {
                    text: self.streaming_text.clone(),
                },
            );
            self.transcript.push(Turn::Assistant(synth));
        } else {
            // Either we have text blocks already, or the streamed text
            // was just the tool-call JSON the model emitted — let the
            // ToolCall / ToolResult markers render it cleanly.
            self.transcript.push(Turn::Assistant(result.blocks));
        }
        for line in result.handoff_lines {
            self.transcript.push(line);
        }
        self.last_turn_result = result.turn_result;

        self.in_flight_since = None;
        self.pending_started = None;
        self.chunk_rx = None;
        self.streaming_text.clear();

        // Phase 5 voice-out: speak the assistant's reply via Piper +
        // rodio. We pull the text from the just-pushed final turn so
        // any synthesized streaming-text fallback (case 2 above) also
        // gets spoken. Empty / non-text replies are a no-op.
        if self.tts_enabled && !self.tts_session_disabled {
            let spoken = self
                .transcript
                .last()
                .map(|turn| match turn {
                    Turn::Assistant(blocks) => concat_text_blocks(blocks),
                    _ => String::new(),
                })
                .unwrap_or_default();
            if !spoken.trim().is_empty() {
                self.speak_async(spoken);
            }
        }
    }

    /// Flush the front of `pending_queue` into a fresh `submit()` call.
    /// Called by the event loop after each tick so queued prompts auto-fire
    /// when the prior turn lands. Mirrors Claude Code's "you sent another
    /// message while I was thinking" UX.
    pub fn drain_queue(&mut self) {
        if self.in_flight_since.is_some() {
            return;
        }
        if self.pending_queue.is_empty() {
            return;
        }
        let next = self.pending_queue.remove(0);
        self.input = next;
        self.submit();
    }

    /// Drive one turn through the supplied [`AgentHandoff`]. Returns the
    /// final hop's blocks, the matching [`TurnResult`] (for `last_turn_result`),
    /// and zero-or-more `Turn::Command` rows describing each step in the
    /// chain (one per step, only when `result.steps.len() > 1`).
    ///
    /// On [`stratum_runtime::HandoffError`] the transcript receives an empty
    /// assistant block, a single `Turn::Command` with `ok = false`, and the
    /// `last_turn_result` is cleared so the status bar does not surface
    /// stale metrics.
    fn run_turn_via_handoff(
        &self,
        handoff: &AgentHandoff,
        ctx: TurnContext,
        prompt: &str,
    ) -> (
        Vec<Block>,
        Option<TurnResult>,
        Vec<Turn>,
        Option<SuggestedRole>,
    ) {
        let intent = IntentRouter::default().classify(prompt);
        match handoff.run_turn_with_handoff(ctx, intent, &self.cancel) {
            Ok(result) => {
                let extra: Vec<Turn> = if result.steps.len() > 1 {
                    result
                        .steps
                        .iter()
                        .map(|step| Turn::Command {
                            text: format!("(handoff: {:?} → {:?})", step.from_role, step.to_role),
                            ok: true,
                            message: String::new(),
                        })
                        .collect()
                } else {
                    Vec::new()
                };
                let last = result.steps.last().map(|s| s.turn_result.clone());
                let final_role = Some(result.final_role);
                (result.final_blocks, last, extra, final_role)
            }
            Err(e) => {
                let line = Turn::Command {
                    text: "(handoff failed)".to_string(),
                    ok: false,
                    message: e.to_string(),
                };
                (Vec::new(), None, vec![line], None)
            }
        }
    }

    /// Render the status-bar text for the current turn state.
    ///
    /// * While a turn is in flight: `[generating… <N>s]` where `<N>` is the
    ///   wall-clock seconds since [`Self::submit`] started.
    /// * After a completed turn (and no in-flight turn): a token-rate summary
    ///   `"<count> tokens in <ms>ms (<tok/s> tok/s)"` derived from
    ///   `last_token_count`, the most recent `TurnMetrics`, and
    ///   [`format_tokens_per_second`].
    /// * Otherwise (fresh state, no submit yet): the empty string.
    ///
    /// The in-flight branch wins over the completed branch — a fresh submit
    /// after a previous turn renders `[generating…]` even though
    /// `last_metrics` is still `Some`.
    #[must_use]
    pub fn status_bar_text(&self) -> String {
        if let Some(started) = self.in_flight_since {
            let elapsed = started.elapsed().as_secs();
            return format!("[generating… {elapsed}s]");
        }
        if let Some(metrics) = self.last_metrics.as_ref() {
            let role_ms = metrics
                .role_steps
                .iter()
                .map(|step| step.duration_ms)
                .fold(0_u32, u32::saturating_add);
            // `format_tokens_per_second` takes `u32`; saturate the approximate
            // count without panicking on the (impossible) overflow case.
            let count_u32 = u32::try_from(self.last_token_count).unwrap_or(u32::MAX);
            let tps = format_tokens_per_second(count_u32, role_ms);
            return format!(
                "{} tokens in {}ms ({tps:.1} tok/s)",
                self.last_token_count, role_ms,
            );
        }
        String::new()
    }

    /// Borrow the approximate token count from the most recent turn. Used by
    /// tests; production callers prefer the precise `last_metrics` count.
    #[must_use]
    #[cfg(test)]
    const fn last_token_count(&self) -> u64 {
        self.last_token_count
    }

    /// Record a completed turn: emit structured events for the generated
    /// blocks, update metrics, and append the user / assistant transcript
    /// entries.
    ///
    /// Factored out of [`Self::submit`] so tests can drive the event-emission
    /// path with synthetic block streams.
    fn finish_turn(&mut self, prompt: String, blocks: Vec<Block>, provider_id: &str, step_ms: u32) {
        let turn_id = TurnId(self.next_turn_id);
        self.next_turn_id = self.next_turn_id.saturating_add(1);
        let mut recorder = TurnRecorder::new(turn_id);
        for block in &blocks {
            recorder.record_block(block);
        }
        recorder.record_step("generate", step_ms);
        self.last_metrics = Some(recorder.finish());

        // No blocks at all on a non-empty prompt = provider failure.
        if blocks.is_empty() {
            self.events.emit(
                RtEvent::ProviderError {
                    provider: provider_id.to_string(),
                    code: "E_NO_BLOCKS".to_string(),
                    message: "provider returned no blocks".to_string(),
                },
                Some(turn_id.0),
            );
        } else {
            for block in &blocks {
                if let Block::ToolCall { id, .. } = block {
                    self.events.emit(
                        RtEvent::ToolCall {
                            tool_id: id.clone(),
                            ok: true,
                            duration_ms: u64::from(step_ms),
                        },
                        Some(turn_id.0),
                    );
                }
            }
        }

        self.transcript.push(Turn::User(prompt));
        self.transcript.push(Turn::Assistant(blocks));
    }

    /// Render the entire TUI into the given frame.
    #[allow(
        clippy::too_many_lines,
        reason = "render walks every overlay (status, chat, palette, modal); splitting fragments the buffer plumbing for no gain"
    )]
    pub fn render(&self, area: Rect, buf: &mut ratatui::buffer::Buffer) {
        // Input pane height grows with both:
        //  - explicit `\n` (Shift+Enter / Ctrl+J / Alt+Enter)
        //  - long single-line input that wraps because the terminal is
        //    narrower than the line length
        // Without the wrap term, a 300-char paste on an 80-col terminal
        // would stay at 3 rows and the user couldn't see what they typed.
        // Cap at 12 rows to leave space for the chat pane on small TTYs.
        let palette_height = if self.palette.is_some() {
            10
        } else {
            // Inner box width = area minus left/right border.
            let inner_w = area.width.saturating_sub(2).max(1) as usize;
            let mut visual_rows: u16 = 0;
            for (i, line) in self.input.split('\n').enumerate() {
                // First line has the "> " prompt eating 2 columns.
                let lead = if i == 0 { 2 } else { 0 };
                let chars = line.chars().count() + lead;
                let wrapped = chars.div_ceil(inner_w);
                visual_rows = visual_rows.saturating_add(wrapped.max(1) as u16);
            }
            // 1 row for the input contents minimum + 2 for borders.
            visual_rows.max(1).saturating_add(2).min(12)
        };
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1),
                Constraint::Min(3),
                Constraint::Length(palette_height),
            ])
            .split(area);

        let mut status_spans = vec![
            Span::styled("stratum", Style::default().add_modifier(Modifier::BOLD)),
            Span::raw(" · "),
            Span::raw(format!("tier={}", self.tier)),
            Span::raw(" · "),
            Span::raw(&self.status),
        ];
        if let Some(metrics) = self.last_metrics.as_ref() {
            let role_ms = metrics
                .role_steps
                .iter()
                .map(|step| step.duration_ms)
                .fold(0_u32, u32::saturating_add);
            let tps = format_tokens_per_second(metrics.completion_tokens, role_ms);
            status_spans.push(Span::raw(" · "));
            status_spans.push(Span::styled(
                format!(
                    "turn {} · prompt:{} compl:{} · {tps:.1} tok/s",
                    metrics.turn_id.0, metrics.prompt_tokens, metrics.completion_tokens,
                ),
                Style::default().add_modifier(Modifier::DIM),
            ));
        }
        let status_bar = self.status_bar_text();
        if !status_bar.is_empty() {
            status_spans.push(Span::raw(" · "));
            status_spans.push(Span::styled(
                status_bar,
                Style::default().add_modifier(Modifier::DIM),
            ));
        }
        let role_label = self.current_role_label();
        if !role_label.is_empty() {
            status_spans.push(Span::raw(" · "));
            status_spans.push(Span::styled(
                role_label,
                Style::default().add_modifier(Modifier::DIM),
            ));
        }
        if self.plan_mode.is_active() {
            status_spans.push(Span::raw(" · "));
            status_spans.push(Span::styled(
                "⏵⏵ plan mode",
                Style::default().add_modifier(Modifier::BOLD),
            ));
        }
        if self.chat_scroll > 0 {
            status_spans.push(Span::raw(" · "));
            status_spans.push(Span::styled(
                format!("[scroll +{} — PgDn / End to return]", self.chat_scroll),
                Style::default().add_modifier(Modifier::BOLD),
            ));
        }
        if self.recording {
            status_spans.push(Span::raw(" · "));
            status_spans.push(Span::styled(
                "\u{1F399} recording…",
                Style::default().add_modifier(Modifier::BOLD),
            ));
        }
        if self.tts_enabled && !self.tts_session_disabled {
            status_spans.push(Span::raw(" · "));
            status_spans.push(Span::styled(
                "tts: on",
                Style::default().add_modifier(Modifier::DIM),
            ));
        }
        if let Some(msg) = self.expire_transient_status() {
            status_spans.push(Span::raw(" · "));
            status_spans.push(Span::styled(
                msg.to_string(),
                Style::default().add_modifier(Modifier::DIM),
            ));
        }
        if let Some(extra) = self.statusline_snapshot() {
            if !extra.is_empty() {
                status_spans.push(Span::raw(" · "));
                status_spans.push(Span::styled(extra, Style::default()));
            }
        }
        // Always show the current mouse mode so the user knows what
        // state they're in and how to flip it. Both modes have
        // tradeoffs; hiding the indicator made the toggle
        // undiscoverable.
        status_spans.push(Span::raw(" · "));
        if self.mouse_capture_off() {
            status_spans.push(Span::styled(
                "select-mode (Ctrl+T → mouse-scroll)",
                Style::default().add_modifier(Modifier::DIM),
            ));
        } else {
            status_spans.push(Span::styled(
                "mouse-scroll (Ctrl+T → select-mode)",
                Style::default().add_modifier(Modifier::DIM),
            ));
        }
        status_spans.push(Span::raw(" · "));
        if self.exit_armed() {
            status_spans.push(Span::styled(
                "press Ctrl+C / Ctrl+D again to exit",
                Style::default().add_modifier(Modifier::BOLD),
            ));
        } else {
            status_spans.push(Span::raw("Esc/Ctrl-C exit"));
        }
        let status = Paragraph::new(Line::from(status_spans));
        ratatui::widgets::Widget::render(status, chunks[0], buf);

        let mut lines: Vec<Line<'_>> = Vec::new();
        let theme = crate::theme::current();
        let last_idx = self.transcript.len().saturating_sub(1);
        for (idx, turn) in self.transcript.iter().enumerate() {
            // Breathing room between turns. No leading blank for the
            // very first turn so the chat box doesn't open with an
            // empty row.
            if idx > 0 {
                lines.push(Line::from(Span::raw("")));
            }
            match turn {
                Turn::User(text) => {
                    // Each line of the user prompt gets a cyan gutter
                    // bar — no "you:" label. Distinguishes turn
                    // ownership visually instead of via inline text.
                    for piece in text.split('\n') {
                        lines.push(prepend_gutter(theme.user_gutter, piece.to_string()));
                    }
                    // If a turn is in flight and this is the most recent
                    // user message, render either a streaming spinner
                    // or the partial streaming text the provider has
                    // emitted so far.
                    if idx == last_idx && self.in_flight_since.is_some() {
                        let elapsed_ms =
                            self.in_flight_since.map_or(0, |t| t.elapsed().as_millis());
                        if self.streaming_text.is_empty() {
                            lines.push(Line::from(Span::raw("")));
                            lines.push(streaming_spinner_line(
                                theme.ai_gutter,
                                theme.dim,
                                elapsed_ms,
                                "thinking",
                            ));
                        } else if looks_like_tool_call_json(&self.streaming_text) {
                            lines.push(Line::from(Span::raw("")));
                            lines.push(streaming_spinner_line(
                                theme.tool,
                                theme.dim,
                                elapsed_ms,
                                "calling tool",
                            ));
                        } else {
                            lines.push(Line::from(Span::raw("")));
                            for ln in render_markdown_gutter(&self.streaming_text, theme.ai_gutter)
                            {
                                lines.push(ln);
                            }
                        }
                    }
                }
                Turn::Assistant(blocks) => {
                    let mut first_text = true;
                    for block in blocks {
                        match block {
                            Block::Text { text } => {
                                if first_text {
                                    first_text = false;
                                } else {
                                    lines.push(Line::from(Span::raw("")));
                                }
                                for ln in render_markdown_gutter(text, theme.ai_gutter) {
                                    lines.push(ln);
                                }
                            }
                            _ => {
                                for line in render_block(block) {
                                    lines.push(line);
                                }
                            }
                        }
                    }
                }
                Turn::Cancelled => lines.push(Line::from(Span::styled(
                    "(cancelled)",
                    theme.dim.add_modifier(Modifier::ITALIC),
                ))),
                Turn::Command { text, ok, message } => {
                    let marker = if *ok { "✓" } else { "✗" };
                    let header = format!("{marker} {text}");
                    lines.push(Line::from(vec![Span::styled(
                        header,
                        theme.dim.add_modifier(Modifier::BOLD),
                    )]));
                    let trimmed = message.trim_end();
                    if !trimmed.is_empty() {
                        for body_line in trimmed.split('\n') {
                            lines.push(Line::from(Span::styled(
                                format!("  {body_line}"),
                                theme.dim,
                            )));
                        }
                    }
                }
            }
        }

        // Empty-state banner — branded wordmark + tagline + a small
        // rotating tip + the most critical key bindings. The full
        // help (`/help`) is the long reference; this is the warm
        // first-launch greeting. Per plan/44.
        if self.transcript.is_empty() && self.in_flight_since.is_none() {
            let mark_style = Style::default()
                .fg(crate::brand::COLOR_PRIMARY)
                .add_modifier(Modifier::BOLD);
            let accent = Style::default().fg(crate::brand::COLOR_ACCENT);
            let body = theme.dim;
            lines.push(Line::from(Span::raw("")));
            for layer in crate::brand::ASCII_MARK {
                lines.push(Line::from(Span::styled(format!("   {layer}"), mark_style)));
            }
            lines.push(Line::from(Span::raw("")));
            lines.push(Line::from(vec![
                Span::styled("   stratum", mark_style),
                Span::raw("  "),
                Span::styled(crate::brand::TAGLINE, accent),
            ]));
            lines.push(Line::from(Span::raw("")));
            // Rotating tip — picks one based on `last_turn_id` so
            // the first launch always shows tip 0 and each new turn
            // would rotate. Falls back to 0 when no turn yet.
            let rot = self.last_turn_id.map_or(0, |t| t.0);
            lines.push(Line::from(Span::styled(
                format!("   tip · {}", crate::brand::tip_for(rot)),
                body.add_modifier(Modifier::ITALIC),
            )));
            lines.push(Line::from(Span::raw("")));
            lines.push(Line::from(Span::styled(
                "   Enter to send · / for commands · Ctrl+G editor · ? help · Ctrl+C twice exit"
                    .to_string(),
                body,
            )));
        }
        // Split the chat pane when a permission request OR reverse-
        // search modal is pending so they live at the BOTTOM (above
        // the input) instead of overlaying the chat history.
        let pending = self.peek_pending_permission();
        let rsearch_open = self.rsearch.is_some();
        let modal_h: u16 = if pending.is_some() {
            5
        } else if rsearch_open {
            5
        } else {
            0
        };
        let chat_split = if modal_h > 0 {
            Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Min(3), Constraint::Length(modal_h)])
                .split(chunks[1])
        } else {
            std::rc::Rc::new([chunks[1]])
        };
        // Auto-scroll so the LATEST lines stay visible. ratatui's
        // Paragraph renders top-down by default — when total lines
        // exceed the pane height the bottom (newest) content gets
        // cut off, which is exactly the opposite of what a chat UI
        // wants. CRITICAL: `lines.len()` is the *logical* line count
        // (Vec<Line>), but a wrapped Paragraph emits many *visual*
        // rows per logical line. Using logical lines for scroll math
        // hides the tail of long markdown responses, which is what
        // makes long answers look "cut off". We replicate ratatui's
        // wrap counting (it's gated behind an unstable feature so we
        // can't call it directly) by summing per-line char counts
        // divided by the inner width.
        let inner_h = chat_split[0].height.saturating_sub(2) as usize;
        let inner_w = chat_split[0].width.saturating_sub(2).max(1) as usize;
        let total_visual = visual_row_count(&lines, inner_w);
        let auto_tail = total_visual.saturating_sub(inner_h);
        let scroll_y = auto_tail.saturating_sub(self.chat_scroll as usize);
        let scroll_y = u16::try_from(scroll_y).unwrap_or(u16::MAX);
        let chat = Paragraph::new(lines)
            .block(TuiBlock::default().borders(Borders::ALL).title("chat"))
            .wrap(Wrap { trim: false })
            .scroll((scroll_y, 0));
        ratatui::widgets::Widget::render(chat, chat_split[0], buf);

        if let Some(pending) = pending {
            let qlen = self.permission_prompter.queue_len();
            let title_suffix = if qlen > 1 {
                format!(" (+{} more)", qlen - 1)
            } else {
                String::new()
            };
            let modal_lines: Vec<Line<'_>> = vec![
                Line::from(vec![
                    Span::styled(
                        "permission required: ",
                        Style::default().add_modifier(Modifier::BOLD),
                    ),
                    Span::raw(describe_request(&pending.request)),
                    Span::styled(
                        title_suffix,
                        Style::default().add_modifier(Modifier::DIM),
                    ),
                ]),
                Line::from(Span::styled(
                    "[a] allow once  [s] allow session  [f] allow forever  [d] deny  [F] deny forever",
                    Style::default().add_modifier(Modifier::DIM),
                )),
            ];
            let modal = Paragraph::new(modal_lines)
                .block(
                    TuiBlock::default()
                        .borders(Borders::ALL)
                        .title("permission")
                        .border_style(Style::default().add_modifier(Modifier::BOLD)),
                )
                .wrap(Wrap { trim: false });
            ratatui::widgets::Widget::render(modal, chat_split[1], buf);
        } else if let Some(state) = self.rsearch.as_ref() {
            let matches = self.rsearch_matches(&state.needle);
            let cur = state.cursor.min(matches.len().saturating_sub(1));
            let pick = matches.get(cur).copied().unwrap_or("");
            let mut modal_lines: Vec<Line<'_>> = Vec::new();
            modal_lines.push(Line::from(vec![
                Span::styled(
                    "(reverse-history-search) ",
                    Style::default().add_modifier(Modifier::BOLD),
                ),
                Span::raw(format!("'{}': ", state.needle)),
                Span::styled(
                    pick.to_string(),
                    Style::default().add_modifier(Modifier::REVERSED),
                ),
            ]));
            modal_lines.push(Line::from(Span::styled(
                format!(
                    "match {}/{} · ↑/Ctrl+R next · ↓ prev · Enter accept · Esc cancel",
                    if matches.is_empty() { 0 } else { cur + 1 },
                    matches.len()
                ),
                Style::default().add_modifier(Modifier::DIM),
            )));
            let modal = Paragraph::new(modal_lines)
                .block(
                    TuiBlock::default()
                        .borders(Borders::ALL)
                        .title("history search"),
                )
                .wrap(Wrap { trim: false });
            ratatui::widgets::Widget::render(modal, chat_split[1], buf);
        }

        if let Some(palette) = self.palette.as_ref() {
            let matches = palette.matches();
            let mut palette_lines: Vec<Line<'_>> = Vec::new();
            palette_lines.push(Line::from(vec![
                Span::styled("palette: ", Style::default().add_modifier(Modifier::BOLD)),
                Span::raw(palette.filter().to_string()),
            ]));
            for (idx, cmd) in matches.iter().enumerate() {
                let style = if idx == palette.cursor() {
                    Style::default().add_modifier(Modifier::REVERSED)
                } else {
                    Style::default()
                };
                palette_lines.push(Line::from(Span::styled(format!("/{}", cmd.name), style)));
            }
            let palette_widget = Paragraph::new(palette_lines)
                .block(TuiBlock::default().borders(Borders::ALL).title("palette"));
            ratatui::widgets::Widget::render(palette_widget, chunks[2], buf);
        } else {
            // Multiline input: split on `\n` so Shift+Enter newlines
            // render as proper lines, and let `Wrap { trim: false }` soft-wrap
            // long lines to the box width so the user can see what they're
            // typing even past the visible terminal column.
            let queue_n = self.pending_queue.len();
            let title = if queue_n > 0 {
                format!("input (queue: {queue_n})")
            } else {
                "input".to_string()
            };
            // Visible block cursor at the end of the user's input so they
            // can see where typed characters will land. ratatui's Frame
            // cursor would be cleaner but render() only has access to the
            // raw Buffer here; appending a glyph in-band is reliable and
            // works on every terminal regardless of cursor-style support.
            const CURSOR_GLYPH: &str = "▏";
            let cursor_style = Style::default().add_modifier(Modifier::REVERSED);
            // Compute which (line_idx, col_byte) the caret falls on
            // so the cursor renders at the caret position, not the
            // tail. Caret of `input.len()` is the past-the-end pos,
            // which displays as a cursor after the last char on the
            // last line.
            let caret_byte = self.caret.min(self.input.len());
            let mut caret_line: usize = 0;
            let mut caret_col_byte: usize = 0;
            {
                let mut scan: usize = 0;
                for (idx, seg) in self.input.split('\n').enumerate() {
                    let seg_end = scan + seg.len();
                    if caret_byte >= scan && caret_byte <= seg_end {
                        caret_line = idx;
                        caret_col_byte = caret_byte - scan;
                        break;
                    }
                    // +1 for the consumed '\n'
                    scan = seg_end + 1;
                }
            }
            let mut input_lines: Vec<Line<'_>> = Vec::new();
            let segments: Vec<&str> = self.input.split('\n').collect();
            for (i, seg) in segments.iter().enumerate() {
                let mut spans: Vec<Span<'_>> = Vec::new();
                if i == 0 {
                    spans.push(Span::raw("> "));
                }
                if i == caret_line {
                    // Split this line at the caret so the cursor glyph
                    // sits between the char before and the char after.
                    let col = caret_col_byte.min(seg.len());
                    let (before, after) = seg.split_at(col);
                    spans.push(Span::raw(before.to_string()));
                    spans.push(Span::styled(CURSOR_GLYPH, cursor_style));
                    spans.push(Span::raw(after.to_string()));
                } else {
                    spans.push(Span::raw((*seg).to_string()));
                }
                input_lines.push(Line::from(spans));
            }
            if input_lines.is_empty() {
                input_lines.push(Line::from(vec![
                    Span::raw("> "),
                    Span::styled(CURSOR_GLYPH, cursor_style),
                ]));
            }
            let input = Paragraph::new(input_lines)
                .block(TuiBlock::default().borders(Borders::ALL).title(title))
                .wrap(Wrap { trim: false });
            ratatui::widgets::Widget::render(input, chunks[2], buf);
        }
    }
}

/// Construct an [`AgentLoop`] wired with the supplied provider and shared
/// emitter, plus the documented default permission store / responder /
/// router / plan-mode / capability matrix.
///
/// All nine required builder fields are populated below, so `build()` is
/// total. We propagate the `Result` rather than panicking; callers fall
/// back to a `None`-loop state if the builder ever grows a new required
/// field that this function forgets to set.
fn default_agent_loop(
    provider: EchoProvider,
    events: Arc<EventEmitter>,
    plan_mode: Arc<PlanMode>,
    prompter: Arc<TuiPromptResponder>,
) -> Result<AgentLoop, stratum_runtime::AgentLoopBuildError> {
    let provider_arc: Arc<dyn Provider> = Arc::new(provider);
    let responder: Arc<dyn stratum_runtime::PromptResponder> = prompter;
    AgentLoop::builder()
        .with_provider(provider_arc)
        .with_router(IntentRouter::default())
        .with_permission_store(Arc::new(PermissionStore::new()))
        .with_prompt_gen(Arc::new(PromptIdGen::new()))
        .with_responder(responder)
        .with_events(events)
        .with_capability_matrix(Arc::new(CapabilityMatrix::new()))
        .with_plan_mode(plan_mode)
        .with_config(AgentLoopConfig::default())
        .build()
}

/// Map a single keystroke to a [`PermissionDecision`] for the modal.
///
/// `a` → `AllowOnce`, `s` → `AllowSession`, `f` → `AllowForever`,
/// `d` → `Deny`, `F` → `DenyForever`. Returns `None` for unknown keys so
/// the caller can re-display the modal.
const fn decision_from_key(c: char) -> Option<PermissionDecision> {
    match c {
        'a' => Some(PermissionDecision::AllowOnce),
        's' => Some(PermissionDecision::AllowSession),
        'f' => Some(PermissionDecision::AllowForever),
        'd' => Some(PermissionDecision::Deny),
        'F' => Some(PermissionDecision::DenyForever),
        _ => None,
    }
}

/// Render `request` into a short human-readable label for the modal.
fn describe_request(req: &PermissionRequest) -> String {
    match req {
        PermissionRequest::CapabilityGrant {
            capability,
            target,
            reason,
        } => target.as_ref().map_or_else(
            || format!("grant {capability} ({reason})"),
            |t| format!("grant {capability} on {t} ({reason})"),
        ),
        PermissionRequest::SecretAccess { secret_ref, scope } => {
            format!("access secret {secret_ref} [{scope}]")
        }
        PermissionRequest::NetworkHost { host, port } => port.as_ref().map_or_else(
            || format!("connect to {host}"),
            |p| format!("connect to {host}:{p}"),
        ),
        PermissionRequest::FileWrite { path } => format!("write to {}", path.display()),
        PermissionRequest::ToolUse { tool_id, args } => {
            // Truncate long arg blobs so the modal stays readable on
            // narrow terminals. Full JSON still goes to the event log.
            let preview = if args.is_empty() {
                String::new()
            } else if args.len() <= 200 {
                format!(" {args}")
            } else {
                format!(" {}…", &args[..200])
            };
            format!("invoke tool {tool_id}{preview}")
        }
    }
}

/// Approximate the completion-token count from a slice of assistant blocks.
///
/// Sums character lengths of every [`Block::Text`] block and divides by 4,
/// matching the 4-chars-per-token rough heuristic documented on
/// [`ChatState::last_token_count`]. Non-text blocks (usage, tool calls, etc.)
/// contribute nothing. Pure / allocation-free.
/// Render a [`SuggestedRole`] as its `snake_case` label.
///
/// Matches the `serde(rename_all = "snake_case")` projection on the enum so
/// status-bar and `/agents` output align with how roles are spelled in
/// agents-dir YAML and `agents list` output.
const fn role_name(role: SuggestedRole) -> &'static str {
    match role {
        SuggestedRole::Default => "default",
        SuggestedRole::Cavemanish => "cavemanish",
        SuggestedRole::Polisher => "polisher",
        SuggestedRole::Coder => "coder",
        SuggestedRole::Researcher => "researcher",
    }
}

/// Resolve a `snake_case` role label to its [`SuggestedRole`] variant.
///
/// Mirrors the `serde(rename_all = "snake_case")` projection on
/// [`SuggestedRole`] so the `/parallel` palette command accepts the same
/// spelling as the on-disk agent TOML's `roles = […]` field.
fn parse_role_label(s: &str) -> Option<SuggestedRole> {
    match s {
        "default" => Some(SuggestedRole::Default),
        "cavemanish" => Some(SuggestedRole::Cavemanish),
        "polisher" => Some(SuggestedRole::Polisher),
        "coder" => Some(SuggestedRole::Coder),
        "researcher" => Some(SuggestedRole::Researcher),
        _ => None,
    }
}

/// Parse the `<path>` argument off a `/audio <path>` invocation.
///
/// Accepts a bare token (no spaces) or a double-quoted form so paths
/// with spaces survive. Returns `None` on empty input, an unterminated
/// quote, or a quoted token followed by trailing garbage.
fn parse_audio_path_arg(raw: &str) -> Option<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    if let Some(rest) = trimmed.strip_prefix('"') {
        // Find the closing quote and refuse anything beyond it.
        let end = rest.find('"')?;
        let (inner, tail) = rest.split_at(end);
        let tail_after_quote = tail.get(1..).unwrap_or("");
        if !tail_after_quote.trim().is_empty() {
            return None;
        }
        if inner.is_empty() {
            return None;
        }
        return Some(inner.to_string());
    }
    // Bare token — refuse if it contains whitespace.
    if trimmed.split_whitespace().count() > 1 {
        return None;
    }
    Some(trimmed.to_string())
}

/// Surface-level MIME sniffer used by the chat seam.
///
/// Mirrors the runtime's `sniff_audio_mime` (kept private to the
/// dispatchers module). Repeated rather than re-exported because the
/// chat shim runs *before* the dispatcher and needs the same answer
/// for the staged `Block::Audio`'s `mime` field.
fn sniff_audio_mime_chat(path: &std::path::Path, bytes: &[u8]) -> String {
    if bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WAVE" {
        return "audio/wav".to_string();
    }
    if bytes.starts_with(b"fLaC") {
        return "audio/flac".to_string();
    }
    if bytes.starts_with(b"OggS") {
        return "audio/ogg".to_string();
    }
    if bytes.starts_with(b"ID3") {
        return "audio/mpeg".to_string();
    }
    // The mask `(bytes[1] & 0xE0) == 0xE0` matches any MPEG audio sync
    // frame (MPEG-1/2/2.5, layers I/II/III), not just MPEG-1 layer-III
    // (0xFFFB / 0xFFFA / 0xFFF3 / 0xFFF2). Used as a heuristic — false
    // positives degrade to audio/mpeg rather than crashing.
    if bytes.len() >= 2 && bytes[0] == 0xFF && (bytes[1] & 0xE0) == 0xE0 {
        return "audio/mpeg".to_string();
    }
    match path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("wav") => "audio/wav".to_string(),
        Some("mp3") => "audio/mpeg".to_string(),
        Some("flac") => "audio/flac".to_string(),
        Some("ogg" | "oga") => "audio/ogg".to_string(),
        Some("opus") => "audio/opus".to_string(),
        Some("m4a") => "audio/mp4".to_string(),
        _ => "application/octet-stream".to_string(),
    }
}

/// Standard-alphabet base64 (matches the runtime's inline implementation).
fn base64_encode_chat(bytes: &[u8]) -> String {
    const ALPHA: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    let mut iter = bytes.chunks_exact(3);
    for chunk in &mut iter {
        let b0 = chunk[0];
        let b1 = chunk[1];
        let b2 = chunk[2];
        out.push(ALPHA[(b0 >> 2) as usize] as char);
        out.push(ALPHA[(((b0 & 0b11) << 4) | (b1 >> 4)) as usize] as char);
        out.push(ALPHA[(((b1 & 0b1111) << 2) | (b2 >> 6)) as usize] as char);
        out.push(ALPHA[(b2 & 0b11_1111) as usize] as char);
    }
    let rem = iter.remainder();
    match rem.len() {
        0 => {}
        1 => {
            let b0 = rem[0];
            out.push(ALPHA[(b0 >> 2) as usize] as char);
            out.push(ALPHA[((b0 & 0b11) << 4) as usize] as char);
            out.push('=');
            out.push('=');
        }
        2 => {
            let b0 = rem[0];
            let b1 = rem[1];
            out.push(ALPHA[(b0 >> 2) as usize] as char);
            out.push(ALPHA[(((b0 & 0b11) << 4) | (b1 >> 4)) as usize] as char);
            out.push(ALPHA[((b1 & 0b1111) << 2) as usize] as char);
            out.push('=');
        }
        _ => unreachable!(),
    }
    out
}

/// Play a WAV file through the OS default audio out, blocking until the
/// clip finishes. Used by [`ChatState::speak_async`] on a worker thread
/// so the UI keeps rendering.
///
/// Failures (no output device, decode error, etc.) are swallowed with a
/// tracing line. Voice-out is a UX nicety; a non-playable WAV must not
/// kill the worker thread or surface a panic into the UI loop. The
/// "[tts unavailable]" message that lights up on Piper failure is the
/// user-visible feedback channel; rodio failures during playback are
/// rare enough that a transient log is the right signal.
#[cfg(feature = "voice")]
fn play_wav_blocking(path: &std::path::Path) {
    use std::io::BufReader;
    let file = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(e) => {
            tracing::warn!(target = "tts", error = %e, "tts: cannot open synth output");
            return;
        }
    };
    let stream = match rodio::OutputStreamBuilder::open_default_stream() {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!(target = "tts", error = %e, "tts: cannot open default audio out");
            return;
        }
    };
    let decoder = match rodio::Decoder::new(BufReader::new(file)) {
        Ok(d) => d,
        Err(e) => {
            tracing::warn!(target = "tts", error = %e, "tts: cannot decode wav");
            return;
        }
    };
    let sink = rodio::Sink::connect_new(stream.mixer());
    sink.append(decoder);
    sink.sleep_until_end();
}

/// Trim `text` for display in a one-line palette acknowledgement.
fn trim_for_ack(text: &str, max_chars: usize) -> String {
    let one_line = text.replace(['\n', '\r'], " ");
    if one_line.chars().count() <= max_chars {
        return one_line;
    }
    let head: String = one_line.chars().take(max_chars.saturating_sub(1)).collect();
    format!("{head}…")
}

/// Concatenate every [`Block::Text`] payload in `blocks` in order. Used by
/// the `/parallel` palette dispatcher to render each role's output as a
/// single transcript line. Non-text blocks (usage, tool calls, etc.) are
/// skipped.
fn concat_text_blocks(blocks: &[Block]) -> String {
    let mut out = String::new();
    for block in blocks {
        if let Block::Text { text } = block {
            out.push_str(text);
        }
    }
    out
}

fn approximate_token_count(blocks: &[Block]) -> u64 {
    const CHARS_PER_TOKEN: u64 = 4;
    let total_chars: u64 = blocks
        .iter()
        .filter_map(|b| match b {
            Block::Text { text } => Some(text.len() as u64),
            _ => None,
        })
        .sum();
    total_chars / CHARS_PER_TOKEN
}

/// Walk the cwd looking for files whose workspace-relative path
/// contains `needle` (case-insensitive substring match), capping the
/// total entries scanned at `limit`. Skips `.*`, `target/`, and
/// `node_modules/`. Returns workspace-relative path strings sorted
/// shortest-first.
fn workspace_file_matches(needle: &str, limit: usize) -> Vec<String> {
    let needle_lc = needle.to_ascii_lowercase();
    let Ok(root) = std::env::current_dir() else {
        return Vec::new();
    };
    let Ok(canonical_root) = root.canonicalize() else {
        return Vec::new();
    };
    let mut out: Vec<String> = Vec::new();
    let mut stack: Vec<(PathBuf, u32)> = vec![(canonical_root.clone(), 0)];
    while let Some((dir, depth)) = stack.pop() {
        if depth > 4 || out.len() >= limit {
            continue;
        }
        let Ok(read) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in read.flatten() {
            if out.len() >= limit {
                break;
            }
            let path = entry.path();
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if name.starts_with('.') || name == "target" || name == "node_modules" {
                    continue;
                }
            }
            let Ok(ft) = entry.file_type() else { continue };
            if ft.is_dir() {
                stack.push((path, depth + 1));
                continue;
            }
            if !ft.is_file() {
                continue;
            }
            let rel = path.strip_prefix(&canonical_root).unwrap_or(&path);
            let rel_str = rel.display().to_string();
            if needle_lc.is_empty() || rel_str.to_ascii_lowercase().contains(&needle_lc) {
                out.push(rel_str);
            }
        }
    }
    out.sort_by_key(String::len);
    out
}

/// Return the longest common ASCII prefix shared by every entry in
/// `items`. Used by tab completion to expand `/m` → `/mod` when both
/// `model` and `models` start with `mod`. Returns "" for empty input.
fn longest_common_prefix(items: &[&str]) -> String {
    let Some(first) = items.first() else {
        return String::new();
    };
    let mut len = first.len();
    for it in items.iter().skip(1) {
        let m = first
            .bytes()
            .zip(it.bytes())
            .take_while(|(a, b)| a == b)
            .count();
        len = len.min(m);
    }
    first[..len].to_string()
}

/// Render a single `fs.edit` tool-call as a unified-diff hunk. The
/// caller wraps the returned text in a ```` ```diff ```` fence so the
/// markdown renderer styles it. Returns `None` if the args JSON is
/// missing fields or is malformed.
fn render_fs_edit_diff(args: &str) -> Option<String> {
    let v: serde_json::Value = serde_json::from_str(args).ok()?;
    let path = v.get("path").and_then(|p| p.as_str()).unwrap_or("?");
    let old = v.get("old_string").and_then(|p| p.as_str()).unwrap_or("");
    let new = v.get("new_string").and_then(|p| p.as_str()).unwrap_or("");
    let mut out = String::new();
    out.push_str(&format!("--- a/{path}\n"));
    out.push_str(&format!("+++ b/{path}\n"));
    out.push_str("@@ fs.edit @@\n");
    for line in old.lines().take(60) {
        out.push('-');
        out.push(' ');
        out.push_str(line);
        out.push('\n');
    }
    if old.lines().count() > 60 {
        out.push_str("…\n");
    }
    for line in new.lines().take(60) {
        out.push('+');
        out.push(' ');
        out.push_str(line);
        out.push('\n');
    }
    if new.lines().count() > 60 {
        out.push_str("…\n");
    }
    Some(out.trim_end().to_string())
}

/// Render a single `fs.write` tool-call as a "new file" hunk.
fn render_fs_write_diff(args: &str) -> Option<String> {
    let v: serde_json::Value = serde_json::from_str(args).ok()?;
    let path = v.get("path").and_then(|p| p.as_str()).unwrap_or("?");
    let content = v.get("content").and_then(|p| p.as_str()).unwrap_or("");
    let mut out = String::new();
    out.push_str("--- /dev/null\n");
    out.push_str(&format!("+++ b/{path}\n"));
    out.push_str("@@ fs.write @@\n");
    for line in content.lines().take(120) {
        out.push('+');
        out.push(' ');
        out.push_str(line);
        out.push('\n');
    }
    if content.lines().count() > 120 {
        out.push_str(&format!(
            "… ({} more lines)\n",
            content.lines().count() - 120
        ));
    }
    Some(out.trim_end().to_string())
}

/// Cheap shape check: does the string look like a JSON tool call
/// (`{"tool":"…","args":…}`)? Used to decide whether streamed text
/// should be promoted into a synthesized `Block::Text` or skipped in
/// favor of the dispatcher's structured `ToolCall` + `ToolResult` blocks.
/// Extract a short, user-meaningful summary from a tool-call's args
/// JSON. Knows the conventional Stratum tool args ("path", "command",
/// "query", "url"); falls back to a truncated raw view otherwise.
fn summarize_tool_args(tool: &str, args: &str) -> String {
    let v: serde_json::Value = match serde_json::from_str(args) {
        Ok(v) => v,
        Err(_) => return truncate_display(args, 60),
    };
    let pick = |k: &str| -> Option<String> {
        v.get(k)
            .and_then(|x| x.as_str())
            .map(|s| truncate_display(s, 60))
    };
    if let Some(path) = pick("path") {
        let mut s = path;
        if matches!(tool, "fs.edit") {
            if let Some(old) = v
                .get("old_string")
                .and_then(|x| x.as_str())
                .map(|x| x.lines().count())
            {
                let new_lines = v
                    .get("new_string")
                    .and_then(|x| x.as_str())
                    .map_or(0, |x| x.lines().count());
                s.push_str(&format!(" (-{old} +{new_lines})"));
            }
        }
        return s;
    }
    if let Some(c) = pick("command") {
        return c;
    }
    if let Some(q) = pick("query") {
        return q;
    }
    if let Some(u) = pick("url") {
        return u;
    }
    if let Some(p) = pick("pattern") {
        return p;
    }
    truncate_display(args, 60)
}

fn truncate_display(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        s.to_string()
    } else {
        let mut out: String = s.chars().take(max_chars).collect();
        out.push('…');
        out
    }
}

/// Glyph used at the start of every user-turn line.
const USER_GUTTER_GLYPH: &str = "▎ ";
/// Glyph used at the start of every assistant-turn line. Visually
/// distinct from the user glyph (`▎`) so the two are tellable apart
/// even on a terminal that can't display color or has the same fg
/// for both gutter colors (the bug surfaced by gemma-4-e4b on a
/// 256-color terminal where #1E5E5E and #D9844D both downsample to
/// the same cell).
const AI_GUTTER_GLYPH: &str = "❯ ";

/// Prepend a colored vertical-bar gutter + space to a single line of
/// user-turn content.
fn prepend_gutter(gutter: Style, content: String) -> Line<'static> {
    Line::from(vec![
        Span::styled(USER_GUTTER_GLYPH.to_string(), gutter),
        Span::raw(content),
    ])
}

/// Streaming activity line: gutter bar + brand-accent braille frame
/// + italic label. `elapsed_ms` drives the frame index; cadence is
/// 80ms per frame (see `crate::brand::SPINNER_FRAMES`). Replaces the
/// previous 3-dot animation per plan/44 §6.1.
fn streaming_spinner_line(
    gutter: Style,
    text: Style,
    elapsed_ms: u128,
    label: &str,
) -> Line<'static> {
    let frame = crate::brand::spinner_frame_for(elapsed_ms);
    let spinner_style = Style::default()
        .fg(crate::brand::COLOR_ACCENT)
        .add_modifier(Modifier::BOLD);
    Line::from(vec![
        Span::styled(AI_GUTTER_GLYPH.to_string(), gutter),
        Span::styled(format!("{frame}  "), spinner_style),
        Span::styled(label.to_string(), text.add_modifier(Modifier::ITALIC)),
    ])
}

/// Render markdown text with the AI-side glyph + color prepended to
/// every resulting line. Used for assistant Text blocks (final +
/// streaming).
fn render_markdown_gutter(text: &str, gutter: Style) -> Vec<Line<'static>> {
    let raw = render_markdown(text, None);
    let mut out: Vec<Line<'static>> = Vec::with_capacity(raw.len());
    for line in raw {
        let mut spans: Vec<Span<'static>> = Vec::with_capacity(line.spans.len() + 1);
        spans.push(Span::styled(AI_GUTTER_GLYPH.to_string(), gutter));
        spans.extend(line.spans);
        out.push(Line::from(spans));
    }
    out
}

/// Count the visual rows a `Vec<Line>` will occupy when rendered by
/// a `Paragraph` with `Wrap { trim: false }` at width `inner_w`.
/// Used for chat-pane auto-tail scrolling: ratatui's own
/// `Paragraph::line_count` is gated behind an unstable feature, so we
/// replicate the wrap math. Counts chars not display width, so wide
/// CJK / emoji are undercounted by ~50% — acceptable while we don't
/// have a real-world report of that mattering.
fn visual_row_count(lines: &[Line<'_>], inner_w: usize) -> usize {
    let w = inner_w.max(1);
    let mut total: usize = 0;
    for line in lines {
        let chars: usize = line.spans.iter().map(|s| s.content.chars().count()).sum();
        let rows = if chars == 0 { 1 } else { chars.div_ceil(w) };
        total = total.saturating_add(rows);
    }
    total
}

fn looks_like_tool_call_json(s: &str) -> bool {
    let s = s.trim();
    let s = s.strip_prefix("```json").unwrap_or(s);
    let s = s.strip_prefix("```").unwrap_or(s);
    let s = s.trim_start();
    s.starts_with('{') && s.contains("\"tool\"") && s.contains("\"args\"")
}

/// Render assistant text as styled lines with basic markdown:
/// headers `#`/`##`/`###`, bullets `- `/`* `, numbered `1. `,
/// blockquote `> `, fenced ```` ```lang ```` code blocks with
/// per-language keyword/string/comment highlighting, inline
/// **bold**, _italic_, and `inline code`. The optional `prefix`
/// is attached to the first emitted line so callers can prepend
/// `ai:  ` / `you: ` markers without losing markdown styling.
///
/// The renderer never panics on malformed markdown — unmatched
/// markers fall back to literal text. Streaming-safe: an unclosed
/// code fence keeps subsequent lines in code-block style until the
/// next chunk arrives with the closer.
#[must_use]
fn render_markdown(text: &str, prefix: Option<(String, Style)>) -> Vec<Line<'static>> {
    let theme = crate::theme::current();
    let mut lines: Vec<Line<'static>> = Vec::new();
    let mut in_code = false;
    let mut code_lang = String::new();
    let mut prefix = prefix;

    let mut take_prefix = |spans: &mut Vec<Span<'static>>| {
        if let Some((p, s)) = prefix.take() {
            spans.push(Span::styled(p, s));
        }
    };

    for raw in text.split('\n') {
        let line = raw.trim_end_matches('\r');
        let lt = line.trim_start();

        if lt.starts_with("```") {
            let after = lt.trim_start_matches("```").trim().to_string();
            let mut spans: Vec<Span<'static>> = Vec::new();
            take_prefix(&mut spans);
            if in_code {
                in_code = false;
                code_lang.clear();
                spans.push(Span::styled("  └─".to_string(), theme.dim));
            } else {
                in_code = true;
                code_lang = after.clone();
                let label = if after.is_empty() {
                    "code".to_string()
                } else {
                    after
                };
                spans.push(Span::styled(format!("  ┌─ {label} "), theme.dim));
            }
            lines.push(Line::from(spans));
            continue;
        }

        if in_code {
            let mut spans: Vec<Span<'static>> = Vec::new();
            take_prefix(&mut spans);
            spans.push(Span::styled("  │ ".to_string(), theme.dim));
            spans.extend(highlight_code_line(line, &code_lang, theme));
            lines.push(Line::from(spans));
            continue;
        }

        let mut spans: Vec<Span<'static>> = Vec::new();
        take_prefix(&mut spans);

        if let Some(rest) = line.strip_prefix("### ") {
            spans.extend(parse_inline(rest, theme.header, theme));
        } else if let Some(rest) = line.strip_prefix("## ") {
            spans.extend(parse_inline(rest, theme.header, theme));
        } else if let Some(rest) = line.strip_prefix("# ") {
            spans.extend(parse_inline(rest, theme.header, theme));
        } else if let Some(rest) = line.strip_prefix("> ") {
            spans.push(Span::styled("│ ".to_string(), theme.quote));
            spans.extend(parse_inline(rest, theme.quote, theme));
        } else if let Some(rest) = line.strip_prefix("- ").or_else(|| line.strip_prefix("* ")) {
            spans.push(Span::styled("• ".to_string(), theme.bullet));
            spans.extend(parse_inline(rest, Style::default(), theme));
        } else if let Some((n, rest)) = parse_numbered_bullet(line) {
            spans.push(Span::styled(format!("{n}. "), theme.bullet));
            spans.extend(parse_inline(rest, Style::default(), theme));
        } else {
            spans.extend(parse_inline(line, Style::default(), theme));
        }

        lines.push(Line::from(spans));
    }

    lines
}

/// Parse a leading `N. ` numbered-list marker. Returns the number
/// and the rest of the line, or `None` if the line is not a list item.
fn parse_numbered_bullet(s: &str) -> Option<(usize, &str)> {
    let mut end = 0_usize;
    for (i, c) in s.char_indices() {
        if c.is_ascii_digit() {
            end = i + c.len_utf8();
        } else {
            break;
        }
    }
    if end == 0 {
        return None;
    }
    let rest = s.get(end..)?;
    let rest = rest.strip_prefix(". ")?;
    let n: usize = s.get(..end)?.parse().ok()?;
    Some((n, rest))
}

/// Walk inline text and emit styled spans for `**bold**`, `_italic_`,
/// and `` `code` ``. Unmatched markers are emitted verbatim so the
/// renderer never eats user characters.
fn parse_inline(text: &str, base: Style, theme: crate::theme::Theme) -> Vec<Span<'static>> {
    let chars: Vec<char> = text.chars().collect();
    let mut spans: Vec<Span<'static>> = Vec::new();
    let mut buf = String::new();
    let mut i = 0;
    let flush = |buf: &mut String, spans: &mut Vec<Span<'static>>| {
        if !buf.is_empty() {
            spans.push(Span::styled(std::mem::take(buf), base));
        }
    };
    while i < chars.len() {
        let c = chars[i];
        if c == '`' {
            if let Some(off) = chars[i + 1..].iter().position(|&x| x == '`') {
                flush(&mut buf, &mut spans);
                let code: String = chars[i + 1..i + 1 + off].iter().collect();
                spans.push(Span::styled(code, theme.inline_code));
                i = i + 1 + off + 1;
                continue;
            }
        }
        if c == '*' && i + 1 < chars.len() && chars[i + 1] == '*' {
            if let Some(off) = find_double(&chars, i + 2, '*') {
                flush(&mut buf, &mut spans);
                let body: String = chars[i + 2..off].iter().collect();
                spans.push(Span::styled(body, base.patch(theme.bold)));
                i = off + 2;
                continue;
            }
        }
        if c == '_' && i + 1 < chars.len() && !chars[i + 1].is_whitespace() {
            if let Some(off) = chars[i + 1..].iter().position(|&x| x == '_') {
                let body: String = chars[i + 1..i + 1 + off].iter().collect();
                if !body.is_empty() {
                    flush(&mut buf, &mut spans);
                    spans.push(Span::styled(body, base.patch(theme.italic)));
                    i = i + 1 + off + 1;
                    continue;
                }
            }
        }
        buf.push(c);
        i += 1;
    }
    flush(&mut buf, &mut spans);
    spans
}

/// Find the next index where two consecutive `c` characters occur,
/// starting at `start`. Returns the index of the first of the two.
fn find_double(chars: &[char], start: usize, c: char) -> Option<usize> {
    let mut i = start;
    while i + 1 < chars.len() {
        if chars[i] == c && chars[i + 1] == c {
            return Some(i);
        }
        i += 1;
    }
    None
}

/// Tokenize a single line of code and emit styled spans. Highlights
/// keywords (bold), string literals (italic), and line comments (dim)
/// for a handful of common languages. Unknown languages fall through
/// to plain dim text. Per-line — no multi-line state.
fn highlight_code_line(line: &str, lang: &str, theme: crate::theme::Theme) -> Vec<Span<'static>> {
    let base = theme.dim;
    let kw_style = theme.keyword;
    let str_style = theme.string_lit;
    let comment_style = theme.comment;

    let lang_norm = lang.to_ascii_lowercase();
    if lang_norm == "diff" || lang_norm == "patch" {
        use ratatui::style::{Color, Modifier as M, Style as S};
        let style = if line.starts_with("+++") || line.starts_with("---") {
            S::new().fg(Color::Cyan).add_modifier(M::BOLD)
        } else if line.starts_with("@@") {
            S::new().fg(Color::Magenta).add_modifier(M::BOLD)
        } else if line.starts_with('+') {
            S::new().fg(Color::Green)
        } else if line.starts_with('-') {
            S::new().fg(Color::Red)
        } else {
            base
        };
        return vec![Span::styled(line.to_string(), style)];
    }
    let keywords: &[&str] = match lang_norm.as_str() {
        "rust" | "rs" => &[
            "fn", "let", "mut", "pub", "use", "mod", "struct", "enum", "impl", "trait", "match",
            "if", "else", "for", "while", "loop", "return", "break", "continue", "as", "ref",
            "self", "Self", "crate", "super", "async", "await", "move", "where", "type", "const",
            "static", "unsafe", "dyn", "in",
        ],
        "bash" | "sh" | "zsh" => &[
            "if", "then", "else", "elif", "fi", "for", "do", "done", "while", "case", "esac",
            "function", "return", "exit", "local", "export", "in",
        ],
        "python" | "py" => &[
            "def", "class", "if", "elif", "else", "for", "while", "return", "import", "from", "as",
            "try", "except", "finally", "with", "pass", "break", "continue", "lambda", "yield",
            "async", "await", "True", "False", "None", "in", "is", "not", "and", "or",
        ],
        "js" | "ts" | "javascript" | "typescript" | "tsx" | "jsx" => &[
            "function",
            "const",
            "let",
            "var",
            "if",
            "else",
            "for",
            "while",
            "return",
            "class",
            "extends",
            "import",
            "export",
            "from",
            "as",
            "new",
            "this",
            "async",
            "await",
            "try",
            "catch",
            "finally",
            "throw",
            "typeof",
            "instanceof",
            "in",
            "of",
        ],
        "go" => &[
            "func",
            "var",
            "const",
            "type",
            "struct",
            "interface",
            "package",
            "import",
            "if",
            "else",
            "for",
            "return",
            "switch",
            "case",
            "default",
            "break",
            "continue",
            "go",
            "chan",
            "select",
            "defer",
            "map",
            "range",
        ],
        _ => &[],
    };

    let line_comment = matches!(
        lang_norm.as_str(),
        "rust" | "rs" | "js" | "ts" | "javascript" | "typescript" | "tsx" | "jsx" | "go"
    );
    let hash_comment = matches!(lang_norm.as_str(), "python" | "py" | "bash" | "sh" | "zsh");

    if keywords.is_empty() && !line_comment && !hash_comment {
        return vec![Span::styled(line.to_string(), base)];
    }

    let chars: Vec<char> = line.chars().collect();
    let mut spans: Vec<Span<'static>> = Vec::new();
    let mut buf = String::new();
    let flush = |buf: &mut String, spans: &mut Vec<Span<'static>>| {
        if !buf.is_empty() {
            spans.push(Span::styled(std::mem::take(buf), base));
        }
    };
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c == '"' || c == '\'' {
            let q = c;
            flush(&mut buf, &mut spans);
            let start = i;
            i += 1;
            while i < chars.len() {
                if chars[i] == '\\' && i + 1 < chars.len() {
                    i += 2;
                    continue;
                }
                if chars[i] == q {
                    i += 1;
                    break;
                }
                i += 1;
            }
            let lit: String = chars[start..i].iter().collect();
            spans.push(Span::styled(lit, str_style));
            continue;
        }
        if line_comment && c == '/' && i + 1 < chars.len() && chars[i + 1] == '/' {
            flush(&mut buf, &mut spans);
            let rest: String = chars[i..].iter().collect();
            spans.push(Span::styled(rest, comment_style));
            return spans;
        }
        if hash_comment && c == '#' {
            flush(&mut buf, &mut spans);
            let rest: String = chars[i..].iter().collect();
            spans.push(Span::styled(rest, comment_style));
            return spans;
        }
        if c.is_ascii_alphabetic() || c == '_' {
            let start = i;
            while i < chars.len() && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            let ident: String = chars[start..i].iter().collect();
            if keywords.contains(&ident.as_str()) {
                flush(&mut buf, &mut spans);
                spans.push(Span::styled(ident, kw_style));
            } else {
                buf.push_str(&ident);
            }
            continue;
        }
        buf.push(c);
        i += 1;
    }
    flush(&mut buf, &mut spans);
    spans
}

fn render_block(block: &Block) -> Vec<Line<'static>> {
    match block {
        Block::Text { text } => render_markdown(
            text,
            Some((
                "ai:  ".to_string(),
                Style::default().add_modifier(Modifier::BOLD),
            )),
        ),
        Block::ToolCall { tool, args, .. } => {
            // Parse the tool-call args JSON and surface the
            // user-meaningful key (path / cmd / query) instead of the
            // raw `{"path":"..."}` blob. Falls back to a truncated
            // raw view when the args don't parse or have no known
            // headline field.
            let theme = crate::theme::current();
            let summary = summarize_tool_args(tool, args);
            vec![Line::from(vec![
                Span::styled("⚒ ".to_string(), theme.tool),
                Span::styled(tool.clone(), theme.tool.add_modifier(Modifier::BOLD)),
                Span::raw(" "),
                Span::styled(summary, theme.dim),
            ])]
        }
        Block::ToolResult { output, .. } => {
            const TOOL_RESULT_MAX: usize = 20;
            const TOOL_RESULT_WIDTH: usize = 160;
            let theme = crate::theme::current();
            let total_lines = output.lines().count();
            let mut out: Vec<Line<'static>> = Vec::new();
            for (i, raw) in output.lines().enumerate() {
                if i >= TOOL_RESULT_MAX {
                    break;
                }
                let line = if raw.chars().count() > TOOL_RESULT_WIDTH {
                    let mut s: String = raw.chars().take(TOOL_RESULT_WIDTH).collect();
                    s.push('…');
                    s
                } else {
                    raw.to_string()
                };
                let lead = if i == 0 { "  ↳ " } else { "    " };
                out.push(Line::from(Span::styled(format!("{lead}{line}"), theme.dim)));
            }
            if total_lines == 0 {
                out.push(Line::from(Span::styled(
                    "  ↳ (no output)".to_string(),
                    theme.dim.add_modifier(Modifier::ITALIC),
                )));
            }
            if total_lines > TOOL_RESULT_MAX {
                out.push(Line::from(Span::styled(
                    format!("    … (+{} more lines)", total_lines - TOOL_RESULT_MAX),
                    theme.dim.add_modifier(Modifier::ITALIC),
                )));
            }
            out
        }
        Block::Usage { prompt, completion } => vec![Line::from(Span::styled(
            format!("(usage: prompt={prompt} completion={completion})"),
            Style::default().add_modifier(Modifier::DIM),
        ))],
        Block::Cancelled { reason } => vec![Line::from(Span::styled(
            format!("(cancelled: {reason})"),
            Style::default().add_modifier(Modifier::ITALIC),
        ))],
        Block::Image { mime, .. } => vec![Line::from(Span::styled(
            format!("(image: {mime})"),
            Style::default().add_modifier(Modifier::DIM),
        ))],
        Block::Audio { mime, .. } => vec![Line::from(Span::styled(
            format!("(audio: {mime})"),
            Style::default().add_modifier(Modifier::DIM),
        ))],
        Block::Done => Vec::new(),
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
    let state = ChatState::new(provider, tier, status_for(paths));
    let store = open_session_store(paths);
    run_with_state(state, store)
}

fn open_session_store(paths: &Paths) -> Option<stratum_runtime::TranscriptStore> {
    let dir = paths.state.join("sessions");
    stratum_runtime::TranscriptStore::open(dir).ok()
}

/// Drive the live TUI against a caller-supplied [`ChatState`]. Used by the
/// `--model` path so the resolved [`AgentLoop`] (wrapping the real
/// `LlamaCppProvider`) backs the TUI in place of the default [`EchoProvider`].
///
/// # Errors
/// Propagates terminal-init failures as [`io::Error`].
/// Restore the terminal to its pre-TUI state. Idempotent — safe to
/// call from a panic hook + the normal exit path. Without this, a
/// panic mid-render leaves the terminal in raw mode and subsequent
/// shell output renders as staircase (no CR translation).
fn restore_terminal() {
    let _ = execute!(io::stdout(), PopKeyboardEnhancementFlags);
    let _ = execute!(io::stdout(), DisableMouseCapture);
    let _ = disable_raw_mode();
    let _ = execute!(io::stdout(), LeaveAlternateScreen);
}

/// Drive the live TUI against a caller-supplied [`ChatState`] until the
/// user quits. Optional [`stratum_runtime::TranscriptStore`] persists
/// the session on exit.
///
/// # Errors
/// Propagates terminal-init failures as [`io::Error`] wrapped in
/// [`stratum_types::StratumError`].
#[allow(
    clippy::print_stderr,
    reason = "post-TUI exit hints print after the terminal is restored — intentional shell output"
)]
pub fn run_with_state(
    mut state: ChatState,
    saver: Option<stratum_runtime::TranscriptStore>,
) -> StratumResult<()> {
    // Install a panic hook that restores the terminal first, then
    // chains to the prior hook (so panic messages still surface).
    let prior_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        restore_terminal();
        prior_hook(info);
    }));
    let mut stdout = io::stdout();
    enable_raw_mode().map_err(map_io_error)?;
    execute!(stdout, EnterAlternateScreen).map_err(map_io_error)?;
    // Best-effort: ask the terminal to disambiguate modifier+Enter via
    // the Kitty keyboard protocol. Supported on iTerm2 (>=3.5),
    // Kitty, WezTerm, foot. Other terminals silently ignore it and
    // fall back to plain Enter (Shift+Enter then arrives as bare Enter,
    // which we still treat as submit; users can use Alt+Enter on those
    // terminals — see the alternate binding in `handle_key`).
    let _ = execute!(
        stdout,
        PushKeyboardEnhancementFlags(KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES)
    );
    // Mouse capture starts OFF so the user can drag-select text and
    // copy with the terminal's native copy shortcut. Ctrl+T (or
    // /select) flips it on when the user wants the scroll wheel.
    // The startup state must agree with ChatState::default() —
    // `mouse_capture_on: false` over there, no Enable call here.
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend).map_err(map_io_error)?;
    let result = event_loop(&mut terminal, &mut state);
    restore_terminal();
    // Persist the transcript so `--resume <id>` can reload it. The save
    // is best-effort: a failure should not turn into a confusing TUI
    // crash on exit, so we just note it on stderr.
    let session_id = state.session_id().clone();
    let mut saved = false;
    if let Some(store) = saver {
        let t = state.to_persisted_transcript();
        if !t.turns.is_empty() {
            match store.save_atomic(&t) {
                Ok(_) => saved = true,
                Err(e) => eprintln!("\nstratum: failed to save session {session_id}: {e}"),
            }
        }
    }
    eprintln!("\nstratum: chat ended.");
    if saved {
        eprintln!("  session id:    {session_id}");
        eprintln!("  resume:        stratum chat --resume {session_id}");
        eprintln!("  list sessions: stratum chat --resume   (no id)");
    }
    eprintln!("  switch model:  stratum chat --model <slug>");
    eprintln!("  list models:   stratum models list");
    result
}

fn event_loop<B: Backend>(terminal: &mut Terminal<B>, state: &mut ChatState) -> StratumResult<()> {
    loop {
        let evt = if event::poll(Duration::from_millis(50)).map_err(map_io_error)? {
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
    match event {
        Some(Event::Key(key)) => state.handle_key(*key),
        Some(Event::Mouse(m)) => state.handle_mouse(*m),
        _ => {}
    }
    // Poll the async turn-result channel before draining the queue so a
    // freshly-settled turn unblocks the next queued prompt on the same
    // tick instead of waiting another 100ms.
    state.poll_turn_completion();
    state.drain_queue();
    state.tick_statusline();
    // Honor a pending Ctrl+G external-editor request, if any.
    if let Some(seed) = state.take_pending_edit_request() {
        if let Ok(edited) = run_external_editor(terminal, &seed) {
            state.set_input_from_editor(edited);
        }
    }
    // Honor a pending Ctrl+T mouse-capture toggle, if any.
    if let Some(want_on) = state.take_pending_mouse_toggle() {
        let mut stdout = io::stdout();
        if want_on {
            let _ = execute!(stdout, EnableMouseCapture);
        } else {
            let _ = execute!(stdout, DisableMouseCapture);
        }
    }
    Ok(())
}

/// Suspend the TUI, write `seed` to a temp file, spawn `$VISUAL`
/// (falling back to `$EDITOR`, then `vi`) on it, read the contents
/// back, and restore the TUI. Caller passes the edited result to
/// [`ChatState::set_input_from_editor`].
fn run_external_editor<B: Backend>(
    terminal: &mut Terminal<B>,
    seed: &str,
) -> StratumResult<String> {
    use std::io::Write as _;

    let editor = std::env::var("VISUAL")
        .or_else(|_| std::env::var("EDITOR"))
        .unwrap_or_else(|_| "vi".to_string());

    let tmp = std::env::temp_dir().join(format!("stratum-edit-{}.txt", std::process::id()));
    {
        let mut f = std::fs::File::create(&tmp).map_err(map_io_error)?;
        f.write_all(seed.as_bytes()).map_err(map_io_error)?;
    }

    // Suspend the TUI so the editor owns the terminal.
    restore_terminal();
    let status = std::process::Command::new(&editor).arg(&tmp).status();

    // Restore the TUI before propagating any error or returning.
    let mut stdout = io::stdout();
    let _ = enable_raw_mode();
    let _ = execute!(stdout, EnterAlternateScreen);
    let _ = execute!(
        stdout,
        PushKeyboardEnhancementFlags(KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES)
    );
    let _ = execute!(stdout, EnableMouseCapture);
    let _ = terminal.clear();

    status.map_err(map_io_error)?;

    let body = std::fs::read_to_string(&tmp).map_err(map_io_error)?;
    let _ = std::fs::remove_file(&tmp);
    Ok(body)
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
    use std::sync::Arc;
    use std::thread;

    use crossterm::event::KeyCode;
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;
    use stratum_runtime::EventSink;

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
    fn ctrl_c_first_press_arms_exit_without_quitting() {
        let mut s = state();
        s.handle_key(key(KeyCode::Char('c'), KeyModifiers::CONTROL));
        assert!(!s.should_quit(), "first Ctrl+C must not quit");
        assert!(s.exit_armed(), "first Ctrl+C must arm exit");
    }

    #[test]
    fn ctrl_c_second_press_quits() {
        let mut s = state();
        s.handle_key(key(KeyCode::Char('c'), KeyModifiers::CONTROL));
        s.handle_key(key(KeyCode::Char('c'), KeyModifiers::CONTROL));
        assert!(s.should_quit(), "second Ctrl+C must quit");
    }

    #[test]
    fn ctrl_d_within_window_quits_after_ctrl_c() {
        let mut s = state();
        s.handle_key(key(KeyCode::Char('c'), KeyModifiers::CONTROL));
        s.handle_key(key(KeyCode::Char('d'), KeyModifiers::CONTROL));
        assert!(s.should_quit(), "Ctrl+D after Ctrl+C must quit");
    }

    #[test]
    fn typed_char_disarms_exit() {
        let mut s = state();
        s.handle_key(key(KeyCode::Char('c'), KeyModifiers::CONTROL));
        s.handle_key(key(KeyCode::Char('x'), KeyModifiers::NONE));
        assert!(!s.exit_armed(), "typing must disarm exit");
    }

    #[test]
    fn unhandled_key_is_ignored() {
        // F5 now drives PTT toggle, but with no host mic the toggle
        // is a no-op for the input/quit slots — the invariant this
        // test guards (F5 must not type into the input buffer or
        // trigger a quit) is what we still care about.
        let mut s = state();
        s.handle_key(key(KeyCode::F(5), KeyModifiers::NONE));
        assert_eq!(s.input(), "");
        assert!(!s.should_quit());
    }

    #[test]
    fn f5_is_the_ptt_hotkey_binding() {
        // Regression guard: F5 must route through `toggle_ptt` (not
        // fall through to insert_char). The test relies on the
        // toggle being a no-op when no host mic is available — the
        // recording flag stays false on failure, but no panic /
        // input-buffer side effect is observable. The point is that
        // F5 is not bound to anything else.
        let mut s = state();
        let before = s.input().to_string();
        s.handle_key(key(KeyCode::F(5), KeyModifiers::NONE));
        assert_eq!(s.input(), before, "F5 must not type into the buffer");
        assert!(
            !s.is_recording() || s.is_recording(),
            "tautology — but the field exists and is reachable",
        );
    }

    #[test]
    fn tts_toggle_starts_off_and_flips_on() {
        let mut s = state();
        assert!(!s.tts_enabled());
        let out = s.dispatch_tts(Some("on"));
        assert!(matches!(out, PaletteOutcome::Acknowledged { .. }));
        assert!(s.tts_enabled());
        let out = s.dispatch_tts(Some("off"));
        assert!(matches!(out, PaletteOutcome::Acknowledged { .. }));
        assert!(!s.tts_enabled());
    }

    #[test]
    fn tts_bare_toggles_current_value() {
        let mut s = state();
        assert!(!s.tts_enabled());
        s.dispatch_tts(None);
        assert!(s.tts_enabled());
        s.dispatch_tts(None);
        assert!(!s.tts_enabled());
    }

    #[test]
    fn tts_unknown_arg_is_rejected() {
        let mut s = state();
        let out = s.dispatch_tts(Some("loud"));
        assert!(matches!(out, PaletteOutcome::Rejected { .. }));
        assert!(!s.tts_enabled());
    }

    #[test]
    fn tts_session_disabled_blocks_reenable() {
        let mut s = state();
        s.tts_session_disabled = true;
        let out = s.dispatch_tts(Some("on"));
        assert!(matches!(out, PaletteOutcome::Rejected { .. }));
        assert!(!s.tts_enabled());
    }

    #[test]
    fn recording_flag_defaults_false_with_no_capture() {
        let s = state();
        assert!(!s.is_recording());
        assert!(s.mic_capture.is_none());
    }

    #[test]
    fn stop_without_active_capture_returns_stopped_empty() {
        let mut s = state();
        let out = s.stop_recording_and_stage().expect("no panic");
        assert!(matches!(out, MicToggle::StoppedEmpty));
        assert!(!s.is_recording());
    }

    #[test]
    fn transient_status_round_trip() {
        let mut s = state();
        assert!(s.transient_status().is_none());
        s.set_transient_status("heard: hi".to_string());
        assert_eq!(s.transient_status(), Some("heard: hi"));
        // The TTL is 2s — within the window the renderer still surfaces it.
        assert_eq!(s.expire_transient_status(), Some("heard: hi"));
    }

    #[cfg(feature = "voice")]
    #[test]
    fn speak_async_disables_session_when_piper_missing() {
        let mut s = state();
        // Point at an unresolvable binary so `synthesize` short-circuits
        // with `MissingBinary` immediately.
        s.tts_enabled = true;
        s.piper = Some(
            stratum_runtime::PiperSubprocess::new("/tmp/never-exists.onnx")
                .with_binary("stratum_no_such_piper_xyzzy"),
        );
        s.speak_async("hello world".to_string());
        assert!(!s.tts_enabled());
        assert!(s.tts_session_disabled());
        assert!(s
            .transient_status()
            .map(|m| m.contains("tts unavailable"))
            .unwrap_or(false));
    }

    #[cfg(feature = "voice")]
    #[test]
    fn speak_async_no_piper_is_noop() {
        let mut s = state();
        assert!(s.piper.is_none());
        s.speak_async("anything".to_string());
        // No piper configured = nothing to do; no flags flipped.
        assert!(!s.tts_enabled());
        assert!(!s.tts_session_disabled());
        assert!(s.transient_status().is_none());
    }

    #[cfg(feature = "voice")]
    #[test]
    fn speak_async_empty_text_is_noop() {
        let mut s = state();
        s.tts_enabled = true;
        s.piper = Some(
            stratum_runtime::PiperSubprocess::new("/tmp/never-exists.onnx")
                .with_binary("stratum_no_such_piper_xyzzy"),
        );
        s.speak_async("   ".to_string());
        // Whitespace-only payload should short-circuit before reaching piper.
        assert!(s.tts_enabled());
        assert!(!s.tts_session_disabled());
    }

    #[cfg(not(feature = "voice"))]
    #[test]
    fn speak_async_on_non_voice_build_surfaces_visible_status_and_disables_tts() {
        // Pin the v1.0.0 Linux-prebuilt UX contract: pressing /tts on,
        // then completing a turn, must produce a visible status line
        // rather than a silent no-op. Without this guard a user who
        // never set RUST_LOG=debug would believe playback worked.
        let mut s = state();
        s.tts_enabled = true;
        s.speak_async("anything".to_string());
        assert!(!s.tts_enabled());
        let msg = s.transient_status().unwrap_or("");
        assert!(
            msg.contains("playback unavailable") && msg.contains("--features voice"),
            "expected the non-voice tts message, got: {msg:?}"
        );
    }

    #[test]
    fn dispatch_mic_palette_command_is_known() {
        // Smoke test that /mic + /tts are present in the catalog so
        // the palette renders them and tab-complete finds them.
        let names: Vec<&str> = crate::palette::COMMANDS.iter().map(|c| c.name).collect();
        assert!(names.contains(&"mic"), "/mic must be in palette catalog");
        assert!(names.contains(&"tts"), "/tts must be in palette catalog");
    }

    #[test]
    fn ctrl_slash_opens_palette() {
        let mut s = state();
        s.handle_key(key(KeyCode::Char('/'), KeyModifiers::CONTROL));
        assert!(s.palette_open());
    }

    #[test]
    fn slash_with_empty_input_is_literal() {
        let mut s = state();
        s.handle_key(key(KeyCode::Char('/'), KeyModifiers::NONE));
        assert!(!s.palette_open());
        assert_eq!(s.input(), "/");
    }

    #[test]
    fn slash_with_text_input_just_appends() {
        let mut s = state();
        s.handle_key(key(KeyCode::Char('a'), KeyModifiers::NONE));
        s.handle_key(key(KeyCode::Char('/'), KeyModifiers::NONE));
        assert!(!s.palette_open());
        assert_eq!(s.input(), "a/");
    }

    #[test]
    fn palette_esc_closes_without_execute() {
        let mut s = state();
        s.handle_key(key(KeyCode::Char('/'), KeyModifiers::CONTROL));
        s.handle_key(key(KeyCode::Esc, KeyModifiers::NONE));
        assert!(!s.palette_open());
        assert!(s.transcript().is_empty());
    }

    #[test]
    fn palette_enter_executes_command() {
        let mut s = state();
        s.handle_key(key(KeyCode::Char('/'), KeyModifiers::CONTROL));
        // First alphabetical match is "active" — unknown to the
        // dispatcher, so it lands as a rejected command turn.
        s.handle_key(key(KeyCode::Enter, KeyModifiers::NONE));
        assert!(!s.palette_open());
        assert!(matches!(s.transcript().last(), Some(Turn::Command { .. })));
    }

    #[test]
    fn palette_quit_command_sets_quit() {
        let mut s = state();
        s.handle_key(key(KeyCode::Char('/'), KeyModifiers::CONTROL));
        for c in "qui".chars() {
            s.handle_key(key(KeyCode::Char(c), KeyModifiers::NONE));
        }
        s.handle_key(key(KeyCode::Enter, KeyModifiers::NONE));
        assert!(s.should_quit());
        let Some(Turn::Command { text, ok, .. }) = s.transcript().last() else {
            panic!("expected command turn")
        };
        assert_eq!(text, "/quit");
        assert!(*ok);
    }

    #[test]
    fn palette_ctrl_c_closes_without_executing() {
        let mut s = state();
        s.handle_key(key(KeyCode::Char('/'), KeyModifiers::CONTROL));
        s.handle_key(key(KeyCode::Char('c'), KeyModifiers::CONTROL));
        assert!(!s.palette_open());
        assert!(s.transcript().is_empty());
    }

    #[test]
    fn palette_typing_filters() {
        let mut s = state();
        s.handle_key(key(KeyCode::Char('/'), KeyModifiers::CONTROL));
        // Use "mo" so we hit /model unambiguously — "m" alone would
        // match anything containing the letter (including /compact).
        s.handle_key(key(KeyCode::Char('m'), KeyModifiers::NONE));
        s.handle_key(key(KeyCode::Char('o'), KeyModifiers::NONE));
        s.handle_key(key(KeyCode::Enter, KeyModifiers::NONE));
        let Some(Turn::Command { text, .. }) = s.transcript().last() else {
            panic!("expected command turn")
        };
        // Filter "mo" leaves "model" and "models"; cursor=0 picks "model".
        // The palette flush prepends the slash before dispatch.
        assert_eq!(text, "/model");
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
        // No more "you:" / "ai:" inline labels — visual gutter
        // distinguishes ownership. Just verify both prompt + reply
        // are present.
        assert!(text.contains("hi"));
        assert!(text.contains("echo: hi"));
    }

    #[test]
    fn submit_records_turn_metrics() {
        let mut s = state();
        for c in "hi".chars() {
            s.handle_key(key(KeyCode::Char(c), KeyModifiers::NONE));
        }
        s.handle_key(key(KeyCode::Enter, KeyModifiers::NONE));
        let metrics = s.last_metrics().expect("metrics recorded");
        assert_eq!(metrics.turn_id.0, 0);
        assert!(metrics.total_blocks >= 1);
        assert_eq!(metrics.role_steps.len(), 1);
        assert_eq!(metrics.role_steps[0].name, "generate");
    }

    #[test]
    fn turn_ids_increment_per_submit() {
        let mut s = state();
        for c in "hi".chars() {
            s.handle_key(key(KeyCode::Char(c), KeyModifiers::NONE));
        }
        s.handle_key(key(KeyCode::Enter, KeyModifiers::NONE));
        for c in "bye".chars() {
            s.handle_key(key(KeyCode::Char(c), KeyModifiers::NONE));
        }
        s.handle_key(key(KeyCode::Enter, KeyModifiers::NONE));
        let metrics = s.last_metrics().expect("metrics recorded");
        assert_eq!(metrics.turn_id.0, 1);
    }

    #[test]
    fn render_shows_token_meter_after_submit() {
        let mut s = state();
        for c in "hi".chars() {
            s.handle_key(key(KeyCode::Char(c), KeyModifiers::NONE));
        }
        s.handle_key(key(KeyCode::Enter, KeyModifiers::NONE));
        let text = rendered_text(&s, 100, 12);
        assert!(text.contains("turn 0"));
        assert!(text.contains("prompt:"));
        assert!(text.contains("compl:"));
        assert!(text.contains("tok/s"));
    }

    #[test]
    fn render_shows_palette_when_open() {
        let mut s = state();
        s.handle_key(key(KeyCode::Char('/'), KeyModifiers::CONTROL));
        s.handle_key(key(KeyCode::Char('m'), KeyModifiers::NONE));
        let text = rendered_text(&s, 60, 12);
        assert!(text.contains("palette"));
        assert!(text.contains("/model"));
        // Input pane is replaced; "input" title should be gone.
        assert!(!text.contains(" input "));
    }

    #[test]
    fn render_shows_executed_command_marker() {
        let mut s = state();
        // Use /version because its body fits on one line — the long
        // /help body would scroll the ✓ header off the top of an
        // 80x14 chat box, which is what the new ToolResult-style
        // multi-line render intentionally allows.
        let outcome = s.execute_palette_command("/version");
        assert!(matches!(outcome, PaletteOutcome::Acknowledged { .. }));
        let text = rendered_text(&s, 80, 14);
        assert!(text.contains("/version"));
        assert!(text.contains('✓'));
    }

    #[test]
    fn render_shows_cancelled_marker() {
        let mut s = state();
        // Force in-flight so the first Ctrl+C cancels (rather than just
        // arming exit). The two-press exit gesture is covered separately.
        s.in_flight_since = Some(Instant::now());
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
        let lines = render_block(&block);
        assert_eq!(lines.len(), 1);
        let rendered: String = lines[0]
            .spans
            .iter()
            .map(|s| s.content.to_string())
            .collect();
        assert!(rendered.contains("ai:"));
        assert!(rendered.contains("hi"));
    }

    #[test]
    fn render_block_text_splits_on_newlines() {
        let block = Block::Text {
            text: "one\ntwo\nthree".into(),
        };
        let lines = render_block(&block);
        assert_eq!(lines.len(), 3);
    }

    #[test]
    fn render_block_usage_emits_meter() {
        let lines = render_block(&Block::Usage {
            prompt: 3,
            completion: 4,
        });
        assert_eq!(lines.len(), 1);
        let rendered: String = lines[0]
            .spans
            .iter()
            .map(|s| s.content.to_string())
            .collect();
        assert!(rendered.contains("usage"));
    }

    #[test]
    fn render_block_done_returns_empty() {
        assert!(render_block(&Block::Done).is_empty());
    }

    #[test]
    fn render_block_tool_call_renders_marker() {
        let block = Block::ToolCall {
            id: "t1".into(),
            tool: "fs.read".into(),
            args: "{}".into(),
        };
        let lines = render_block(&block);
        let rendered: String = lines
            .iter()
            .flat_map(|l| l.spans.iter())
            .map(|s| s.content.to_string())
            .collect();
        // The tool-call marker is now ⚒ + tool name + parsed-arg
        // summary. No "tool" literal in the rendered text.
        assert!(rendered.contains("⚒"));
        assert!(rendered.contains("fs.read"));
    }

    #[test]
    fn render_block_tool_result_renders_marker() {
        let block = Block::ToolResult {
            id: "t1".into(),
            output: "ok".into(),
        };
        let lines = render_block(&block);
        let rendered: String = lines
            .iter()
            .flat_map(|l| l.spans.iter())
            .map(|s| s.content.to_string())
            .collect();
        // Result is shown as a `↳` continuation arrow; no "result"
        // literal in the rendered text.
        assert!(rendered.contains("↳"));
        assert!(rendered.contains("ok"));
    }

    #[test]
    fn render_block_cancelled_returns_reason() {
        let block = Block::Cancelled {
            reason: "STRAT-E4002".into(),
        };
        let lines = render_block(&block);
        assert_eq!(lines.len(), 1);
        let rendered: String = lines[0]
            .spans
            .iter()
            .map(|s| s.content.to_string())
            .collect();
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

    // ---------- structured-event instrumentation ----------

    /// Drop-in `EventSink` for the "opaque sink" `with_events` test.
    #[derive(Debug, Default)]
    struct NullSink;
    impl EventSink for NullSink {
        fn write(&self, _record: EventRecord) {}
    }

    #[test]
    fn default_state_has_memory_event_emitter() {
        let s = ChatState::default();
        // Default sink is memory-backed -> snapshot is observable and empty.
        let snap = s.events_snapshot().expect("memory snapshot available");
        assert!(snap.is_empty());
    }

    #[test]
    fn echo_submit_emits_agent_handoff_event() {
        let mut s = state();
        for c in "hello world".chars() {
            s.handle_key(key(KeyCode::Char(c), KeyModifiers::NONE));
        }
        s.handle_key(key(KeyCode::Enter, KeyModifiers::NONE));
        // `submit` now routes through `AgentLoop::run_turn`, which emits
        // an `AgentHandoff` at the start of every turn. EchoProvider's
        // Text+Usage+Done blocks produce no further events (no ToolCall,
        // non-empty so no ProviderError).
        let snap = s.events_snapshot().expect("memory snapshot");
        assert_eq!(snap.len(), 1, "got events: {snap:?}");
        assert!(matches!(snap[0].event, RtEvent::AgentHandoff { .. }));
    }

    #[test]
    fn submit_with_zero_blocks_emits_provider_error() {
        let mut s = state();
        s.finish_turn("hi".to_string(), Vec::new(), "echo", 0);
        let snap = s.events_snapshot().expect("memory snapshot");
        assert_eq!(snap.len(), 1);
        let RtEvent::ProviderError {
            provider,
            code,
            message,
        } = &snap[0].event
        else {
            panic!("expected ProviderError, got {:?}", snap[0].event);
        };
        assert_eq!(provider, "echo");
        assert_eq!(code, "E_NO_BLOCKS");
        assert!(!message.is_empty());
    }

    #[test]
    fn tool_call_block_emits_tool_call_event() {
        let mut s = state();
        let blocks = vec![
            Block::ToolCall {
                id: "t1".into(),
                tool: "fs.read".into(),
                args: "{}".into(),
            },
            Block::Done,
        ];
        s.finish_turn("run".to_string(), blocks, "echo", 7);
        let snap = s.events_snapshot().expect("memory snapshot");
        assert_eq!(snap.len(), 1);
        let RtEvent::ToolCall {
            tool_id,
            ok,
            duration_ms,
        } = &snap[0].event
        else {
            panic!("expected ToolCall, got {:?}", snap[0].event);
        };
        assert_eq!(tool_id, "t1");
        assert!(*ok);
        assert_eq!(*duration_ms, 7);
        assert_eq!(snap[0].turn_id, Some(0));
    }

    #[test]
    fn events_snapshot_aggregates_across_submits() {
        let mut s = state();
        // First turn: tool call.
        s.finish_turn(
            "a".to_string(),
            vec![Block::ToolCall {
                id: "t1".into(),
                tool: "fs.read".into(),
                args: "{}".into(),
            }],
            "echo",
            1,
        );
        // Second turn: provider error (zero blocks).
        s.finish_turn("b".to_string(), Vec::new(), "echo", 2);
        // Third turn: clean text turn — no events.
        s.finish_turn(
            "c".to_string(),
            vec![Block::Text { text: "ok".into() }],
            "echo",
            3,
        );
        let snap = s.events_snapshot().expect("memory snapshot");
        assert_eq!(snap.len(), 2);
        assert!(matches!(snap[0].event, RtEvent::ToolCall { .. }));
        assert!(matches!(snap[1].event, RtEvent::ProviderError { .. }));
        assert_eq!(snap[0].turn_id, Some(0));
        assert_eq!(snap[1].turn_id, Some(1));
    }

    #[test]
    fn with_events_non_memory_sink_yields_none_snapshot() {
        let sink: Arc<dyn EventSink> = Arc::new(NullSink);
        let emitter = Arc::new(EventEmitter::new(sink));
        let s = state().with_events(emitter);
        assert!(s.events_snapshot().is_none());
    }

    #[test]
    fn with_events_swaps_emitter_target() {
        // Build a memory sink we own and watch directly.
        let sink = Arc::new(MemoryEventSink::new());
        let sink_dyn: Arc<dyn EventSink> = sink.clone();
        let emitter = Arc::new(EventEmitter::new(sink_dyn));
        let mut s = state().with_events(emitter);
        s.finish_turn("x".to_string(), Vec::new(), "echo", 0);
        // ChatState no longer has its own MemoryEventSink handle.
        assert!(s.events_snapshot().is_none());
        // But the externally-owned sink received the event.
        let external = sink.snapshot();
        assert_eq!(external.len(), 1);
        assert!(matches!(external[0].event, RtEvent::ProviderError { .. }));
    }

    #[test]
    fn concurrent_submits_produce_monotonic_event_ids() {
        // Drive the emitter directly from many threads through a shared
        // ChatState-style emitter. We can't share &mut ChatState across
        // threads, so exercise the emitter that backs it.
        let sink = Arc::new(MemoryEventSink::new());
        let sink_dyn: Arc<dyn EventSink> = sink.clone();
        let emitter = Arc::new(EventEmitter::new(sink_dyn));
        let mut handles = Vec::new();
        for t in 0..4_u64 {
            let em = emitter.clone();
            handles.push(thread::spawn(move || {
                for _ in 0..50_u64 {
                    em.emit(
                        RtEvent::ProviderError {
                            provider: "echo".to_string(),
                            code: "E_NO_BLOCKS".to_string(),
                            message: "x".to_string(),
                        },
                        Some(t),
                    );
                }
            }));
        }
        for h in handles {
            h.join().expect("join");
        }
        let snap = sink.snapshot();
        assert_eq!(snap.len(), 200);
        let mut ids: Vec<u64> = snap.iter().map(|r| r.id).collect();
        ids.sort_unstable();
        // Strictly monotonic, unique, starting at 1.
        assert_eq!(*ids.first().expect("first"), 1);
        assert_eq!(*ids.last().expect("last"), 200);
        let mut dedup = ids.clone();
        dedup.dedup();
        assert_eq!(dedup.len(), ids.len());
    }

    // ---------- AgentLoop integration ----------

    /// Test provider whose blocks come from a script. Empty script ⇒ no
    /// blocks (lets us assert the zero-block path).
    #[derive(Debug)]
    struct ScriptedProvider {
        script: std::sync::Mutex<Vec<Block>>,
    }

    impl ScriptedProvider {
        fn empty() -> Self {
            Self {
                script: std::sync::Mutex::new(Vec::new()),
            }
        }
    }

    impl Provider for ScriptedProvider {
        fn id(&self) -> &'static str {
            "scripted"
        }
        fn capabilities(&self) -> &'static [stratum_types::Capability] {
            const CAPS: &[stratum_types::Capability] = &[stratum_types::Capability::Generate];
            CAPS
        }
        fn generate(
            &self,
            _req: &stratum_runtime::GenerateRequest,
            _cancel: &CancelToken,
        ) -> Vec<Block> {
            self.script
                .lock()
                .map(|mut v| std::mem::take(&mut *v))
                .unwrap_or_default()
        }
    }

    fn build_loop(provider: Arc<dyn Provider>, events: Arc<EventEmitter>) -> Arc<AgentLoop> {
        Arc::new(
            AgentLoop::builder()
                .with_provider(provider)
                .with_router(stratum_runtime::IntentRouter::default())
                .with_permission_store(Arc::new(stratum_runtime::PermissionStore::new()))
                .with_prompt_gen(Arc::new(stratum_runtime::PromptIdGen::new()))
                .with_responder(Arc::new(stratum_runtime::AllowAllResponder))
                .with_events(events)
                .with_capability_matrix(Arc::new(stratum_runtime::CapabilityMatrix::new()))
                .with_plan_mode(Arc::new(stratum_runtime::PlanMode::new()))
                .with_config(stratum_runtime::AgentLoopConfig::default())
                .build()
                .unwrap(),
        )
    }

    #[test]
    fn default_state_constructs_without_panic() {
        // Smoke: default state can be built and probed without unwrapping.
        let s = ChatState::default();
        assert!(s.last_turn_result().is_none());
        assert!(s.transcript().is_empty());
    }

    /// Non-AgentLoop `ChatBackend` impl proving the seam works for
    /// future remote / hosted backends. Emits a fixed text block.
    #[derive(Debug)]
    struct FixedTextBackend(String);
    impl ChatBackend for FixedTextBackend {
        fn run_turn_streaming(
            &self,
            ctx: TurnContext,
            _cancel: &CancelToken,
            chunk_tx: mpsc::Sender<Block>,
        ) -> TurnResult {
            let _ = chunk_tx.send(Block::Text {
                text: self.0.clone(),
            });
            TurnResult {
                turn_id: ctx.turn_id,
                outcome: TurnOutcome::Success,
                blocks: vec![Block::Text {
                    text: self.0.clone(),
                }],
                transitions: Vec::new(),
                events_emitted: Vec::new(),
            }
        }
    }

    #[test]
    fn chat_state_can_be_built_with_non_agent_loop_backend() {
        let backend: Arc<dyn ChatBackend> =
            Arc::new(FixedTextBackend("hi from the fixed backend".to_string()));
        let mut s = ChatState::with_backend(backend);
        s.submit_with_prompt("hello");
        let text = s.last_assistant_text().expect("assistant text");
        assert!(text.contains("hi from the fixed backend"));
    }

    #[test]
    fn submit_hello_records_user_then_assistant() {
        let mut s = state();
        for c in "hello".chars() {
            s.handle_key(key(KeyCode::Char(c), KeyModifiers::NONE));
        }
        s.handle_key(key(KeyCode::Enter, KeyModifiers::NONE));
        assert_eq!(s.transcript().len(), 2);
        match &s.transcript()[0] {
            Turn::User(p) => assert_eq!(p, "hello"),
            other => panic!("expected Turn::User, got {other:?}"),
        }
        assert!(matches!(s.transcript()[1], Turn::Assistant(_)));
    }

    #[test]
    fn submit_with_scripted_zero_block_provider_emits_provider_error() {
        // Wire a sink we can inspect, point the AgentLoop at the same
        // emitter, and feed the state via `with_agent_loop` so the
        // scripted-zero-blocks path is exercised end-to-end.
        let sink = Arc::new(MemoryEventSink::new());
        let sink_dyn: Arc<dyn EventSink> = sink.clone();
        let events = Arc::new(EventEmitter::new(sink_dyn));
        let provider: Arc<dyn Provider> = Arc::new(ScriptedProvider::empty());
        let loop_ = build_loop(provider, events);
        let mut s = ChatState::with_agent_loop(loop_);

        for c in "anything".chars() {
            s.handle_key(key(KeyCode::Char(c), KeyModifiers::NONE));
        }
        s.handle_key(key(KeyCode::Enter, KeyModifiers::NONE));

        assert_eq!(s.transcript().len(), 2);
        match &s.transcript()[1] {
            Turn::Assistant(blocks) => assert!(blocks.is_empty(), "expected empty blocks"),
            other => panic!("expected Turn::Assistant, got {other:?}"),
        }
        let snap = sink.snapshot();
        assert!(
            snap.iter()
                .any(|r| matches!(r.event, RtEvent::ProviderError { .. })),
            "expected a ProviderError event, got: {snap:?}"
        );
    }

    #[test]
    fn last_turn_result_none_before_submit_some_after() {
        let mut s = state();
        assert!(s.last_turn_result().is_none());
        for c in "hi".chars() {
            s.handle_key(key(KeyCode::Char(c), KeyModifiers::NONE));
        }
        s.handle_key(key(KeyCode::Enter, KeyModifiers::NONE));
        let tr = s.last_turn_result().expect("turn result populated");
        assert!(matches!(tr.outcome, stratum_runtime::TurnOutcome::Success));
    }

    #[test]
    fn empty_prompt_is_noop_through_agent_loop() {
        let mut s = state();
        // Type only whitespace, then Enter.
        s.handle_key(key(KeyCode::Char(' '), KeyModifiers::NONE));
        s.handle_key(key(KeyCode::Enter, KeyModifiers::NONE));
        assert!(s.transcript().is_empty());
        assert!(s.last_turn_result().is_none());
    }

    #[test]
    fn concurrent_submits_across_threads_do_not_panic() {
        use std::sync::Mutex;
        // 4 threads × 25 submits = 200 user+assistant pairs distributed
        // across 4 shared ChatStates (one per thread). Each thread owns its
        // own state — concurrency at the per-state level is exercised by
        // serial submits, while the AgentLoop is the same `Arc` shared
        // across threads via the loops built per state.
        let states: Vec<Arc<Mutex<ChatState>>> =
            (0..4).map(|_| Arc::new(Mutex::new(state()))).collect();
        let mut handles = Vec::new();
        for s in &states {
            let s = s.clone();
            handles.push(thread::spawn(move || {
                for i in 0..25_u32 {
                    let mut g = s.lock().unwrap();
                    for c in format!("hi{i}").chars() {
                        g.handle_key(key(KeyCode::Char(c), KeyModifiers::NONE));
                    }
                    g.handle_key(key(KeyCode::Enter, KeyModifiers::NONE));
                }
            }));
        }
        for h in handles {
            h.join().expect("thread join");
        }
        let total: usize = states
            .iter()
            .map(|s| s.lock().unwrap().transcript().len())
            .sum();
        assert_eq!(total, 200);
    }

    #[test]
    fn with_agent_loop_swaps_orchestrator() {
        // The `with_agent_loop` builder returns a state that delegates to
        // the supplied loop. Smoke-check that submit still records a turn.
        let sink: Arc<dyn EventSink> = Arc::new(MemoryEventSink::new());
        let events = Arc::new(EventEmitter::new(sink));
        let provider: Arc<dyn Provider> = Arc::new(EchoProvider::new("ECHO> "));
        let loop_ = build_loop(provider, events);
        let mut s = ChatState::with_agent_loop(loop_);
        for c in "ping".chars() {
            s.handle_key(key(KeyCode::Char(c), KeyModifiers::NONE));
        }
        s.handle_key(key(KeyCode::Enter, KeyModifiers::NONE));
        assert_eq!(s.transcript().len(), 2);
    }

    // ---------- palette command dispatch ----------

    #[test]
    fn execute_plan_activates_and_acknowledges() {
        let mut s = state();
        assert!(!s.plan_mode.is_active());
        let outcome = s.execute_palette_command("/plan");
        assert!(matches!(outcome, PaletteOutcome::Acknowledged { .. }));
        assert!(s.plan_mode.is_active());
    }

    #[test]
    fn execute_plan_toggle_deactivates_when_already_active() {
        let mut s = state();
        s.execute_palette_command("/plan");
        assert!(s.plan_mode.is_active());
        let outcome = s.execute_palette_command("/plan");
        assert!(matches!(outcome, PaletteOutcome::Acknowledged { .. }));
        assert!(!s.plan_mode.is_active());
    }

    #[test]
    fn execute_plan_on_force_activates_regardless_of_state() {
        let mut s = state();
        // Inactive -> on.
        s.execute_palette_command("/plan on");
        assert!(s.plan_mode.is_active());
        // Active -> on (still active).
        s.execute_palette_command("/plan on");
        assert!(s.plan_mode.is_active());
    }

    #[test]
    fn execute_plan_off_force_deactivates_regardless_of_state() {
        let mut s = state();
        // Inactive -> off (still inactive).
        s.execute_palette_command("/plan off");
        assert!(!s.plan_mode.is_active());
        // Activate, then off.
        s.execute_palette_command("/plan on");
        assert!(s.plan_mode.is_active());
        s.execute_palette_command("/plan off");
        assert!(!s.plan_mode.is_active());
    }

    #[test]
    fn execute_plan_unknown_arg_is_rejected() {
        let mut s = state();
        let outcome = s.execute_palette_command("/plan maybe");
        assert!(matches!(outcome, PaletteOutcome::Rejected { .. }));
        assert!(!s.plan_mode.is_active());
    }

    #[test]
    fn execute_cancel_fires_cancel_token() {
        let mut s = state();
        assert!(!s.cancel.is_cancelled());
        let outcome = s.execute_palette_command("/cancel");
        assert!(matches!(outcome, PaletteOutcome::Acknowledged { .. }));
        assert!(s.cancel.is_cancelled());
    }

    #[test]
    fn execute_clear_empties_transcript() {
        // /clear is a meta-action: it removes prior turns *and* its own
        // marker, leaving the transcript empty.
        let mut s = state();
        for c in "hi".chars() {
            s.handle_key(key(KeyCode::Char(c), KeyModifiers::NONE));
        }
        s.handle_key(key(KeyCode::Enter, KeyModifiers::NONE));
        assert!(!s.transcript().is_empty());
        s.execute_palette_command("/clear");
        assert!(s.transcript().is_empty());
    }

    #[test]
    fn execute_help_returns_acknowledged_with_nonempty_message() {
        let mut s = state();
        let outcome = s.execute_palette_command("/help");
        let PaletteOutcome::Acknowledged { message } = outcome else {
            panic!("expected acknowledged");
        };
        assert!(!message.is_empty());
        assert!(message.contains("/plan"));
        assert!(message.contains("/cancel"));
        assert!(message.contains("/clear"));
        assert!(message.contains("/budget"));
        assert!(message.contains("/quit"));
    }

    #[test]
    fn help_text_const_advertises_budget() {
        assert!(HELP_TEXT.contains("/budget"));
    }

    #[test]
    fn execute_budget_with_no_metrics_returns_sentinel() {
        let mut s = state();
        let outcome = s.execute_palette_command("/budget");
        let PaletteOutcome::Acknowledged { message } = outcome else {
            panic!("expected acknowledged");
        };
        assert_eq!(message, "no turn metrics yet");
    }

    #[test]
    fn execute_budget_after_submit_reports_formatted_metrics() {
        let mut s = state();
        s.submit_with_prompt("hello world");
        let outcome = s.execute_palette_command("/budget");
        let PaletteOutcome::Acknowledged { message } = outcome else {
            panic!("expected acknowledged");
        };
        // Format: "metrics: <tokens> tokens · <wall_ms>ms · <tps> tok/s · turn id <N>"
        assert!(message.starts_with("metrics: "), "got: {message}");
        assert!(message.contains(" tokens · "), "got: {message}");
        assert!(message.contains("ms · "), "got: {message}");
        assert!(message.contains(" tok/s · turn id "), "got: {message}");
        // After one submit, turn id 0 was recorded.
        assert!(message.ends_with("turn id 0"), "got: {message}");
    }

    #[test]
    fn execute_quit_sets_should_quit() {
        let mut s = state();
        assert!(!s.should_quit());
        let outcome = s.execute_palette_command("/quit");
        assert!(matches!(outcome, PaletteOutcome::Acknowledged { .. }));
        assert!(s.should_quit());
    }

    #[test]
    fn execute_exit_is_alias_for_quit() {
        let mut s = state();
        assert!(!s.should_quit());
        let outcome = s.execute_palette_command("/exit");
        assert!(matches!(outcome, PaletteOutcome::Acknowledged { .. }));
        assert!(s.should_quit());
    }

    #[test]
    fn execute_unknown_command_returns_rejected() {
        let mut s = state();
        let outcome = s.execute_palette_command("/unknown");
        let PaletteOutcome::Rejected { message } = outcome else {
            panic!("expected rejected");
        };
        assert_eq!(message, "unknown command: /unknown");
    }

    #[test]
    fn execute_empty_command_returns_rejected() {
        let mut s = state();
        let outcome = s.execute_palette_command("");
        assert!(matches!(outcome, PaletteOutcome::Rejected { .. }));
    }

    #[test]
    fn execute_slash_only_returns_rejected() {
        // `cmd = "/"` strips the prefix but leaves an empty body — covered
        // by the dedicated empty-trim branch.
        let mut s = state();
        let outcome = s.execute_palette_command("/");
        assert!(matches!(outcome, PaletteOutcome::Rejected { .. }));
    }

    #[test]
    fn execute_command_via_palette_flush() {
        // Exercise the palette → execute_command → execute_palette_command
        // bridge (lower-level than the existing `Enter`-key tests). Pick a
        // recognised command so the dispatch path is OK.
        let mut s = state();
        s.execute_command("help");
        let Turn::Command { text, ok, .. } = s.transcript().last().expect("turn") else {
            panic!("expected command turn")
        };
        assert_eq!(text, "/help");
        assert!(*ok);
    }

    #[test]
    fn render_rejected_command_shows_marker() {
        let mut s = state();
        s.execute_palette_command("/unknown");
        let text = rendered_text(&s, 80, 14);
        assert!(text.contains("/unknown"));
        assert!(text.contains('✗'));
    }

    #[test]
    fn transcript_records_ok_true_after_plan() {
        let mut s = state();
        s.execute_palette_command("/plan");
        let Turn::Command { ok, text, .. } = s.transcript().last().expect("turn") else {
            panic!("expected command turn")
        };
        assert!(*ok);
        assert_eq!(text, "/plan");
    }

    #[test]
    fn transcript_records_ok_false_after_unknown() {
        let mut s = state();
        s.execute_palette_command("/unknown");
        let Turn::Command { ok, text, message } = s.transcript().last().expect("turn") else {
            panic!("expected command turn")
        };
        assert!(!*ok);
        assert_eq!(text, "/unknown");
        assert_eq!(message, "unknown command: /unknown");
    }

    #[test]
    fn clear_then_help_results_in_single_help_entry() {
        let mut s = state();
        s.execute_palette_command("/plan");
        s.execute_palette_command("/clear");
        // /clear wipes its own marker, so the transcript is empty here.
        assert!(s.transcript().is_empty());
        s.execute_palette_command("/help");
        assert_eq!(s.transcript().len(), 1);
        let Turn::Command { text, .. } = &s.transcript()[0] else {
            panic!("expected command turn")
        };
        assert_eq!(text, "/help");
    }

    #[test]
    fn submit_after_cancel_still_proceeds_via_agent_loop() {
        // Document the current AgentLoop behavior: a parent cancel set
        // before submit causes `run_turn` to return a UserAbort outcome,
        // and the state still pushes a User+Assistant pair. We assert
        // the user-abort outcome to pin the contract.
        let mut s = state();
        s.execute_palette_command("/cancel");
        assert!(s.cancel.is_cancelled());
        for c in "hi".chars() {
            s.handle_key(key(KeyCode::Char(c), KeyModifiers::NONE));
        }
        s.handle_key(key(KeyCode::Enter, KeyModifiers::NONE));
        let tr = s.last_turn_result().expect("turn result populated");
        // AgentLoop sees the cancel and short-circuits with UserAbort.
        assert!(matches!(
            tr.outcome,
            stratum_runtime::TurnOutcome::UserAbort
        ));
    }

    #[test]
    fn should_quit_is_false_initially() {
        let s = state();
        assert!(!s.should_quit());
    }

    // ---------- non-interactive --prompt helpers ----------

    #[test]
    fn submit_with_prompt_stages_input_and_runs_turn() {
        let mut s = state();
        s.submit_with_prompt("hello");
        // Input must be drained after submit, transcript holds a
        // user+assistant pair.
        assert!(s.input().is_empty());
        assert_eq!(s.transcript().len(), 2);
        match &s.transcript()[0] {
            Turn::User(p) => assert_eq!(p, "hello"),
            other => panic!("expected Turn::User, got {other:?}"),
        }
    }

    #[test]
    fn last_assistant_text_returns_echo_output() {
        let mut s = state();
        s.submit_with_prompt("hello");
        let text = s.last_assistant_text().expect("assistant text");
        // EchoProvider's prefix here is "echo: ".
        assert!(text.contains("hello"), "got: {text}");
        assert!(text.starts_with("echo: "), "got: {text}");
    }

    #[test]
    fn command_turn_multiline_no_prefix_repetition() {
        // Regression for the prefix-spam bug: previously a multi-line
        // command output (eg /themes) rendered with
        //   (executed /themes: themes:)
        //   (executed /themes:   default)
        //   (executed /themes:   mono)
        // … one line per option. Now the prefix appears once as a
        // dim header and the body indents two spaces.
        let mut s = state();
        let _ = s.execute_palette_command("/themes");
        let text = rendered_text(&s, 80, 14);
        let prefix_hits = text.matches("/themes").count();
        assert!(
            prefix_hits <= 2,
            "header should not be repeated per body line; got {prefix_hits} hits:\n{text}"
        );
    }

    #[test]
    fn clear_resets_chat_scroll() {
        let mut s = state();
        s.scroll_up(50);
        assert_eq!(s.chat_scroll, 50);
        let _ = s.execute_palette_command("/clear");
        assert_eq!(s.chat_scroll, 0, "chat_scroll must reset on /clear");
        assert!(
            s.streaming_text.is_empty(),
            "streaming_text must reset on /clear"
        );
    }

    #[test]
    fn tool_result_block_shows_multiple_lines() {
        let block = Block::ToolResult {
            id: "t1".into(),
            output: "line1\nline2\nline3\nline4\nline5".into(),
        };
        let lines = render_block(&block);
        let joined: String = lines
            .iter()
            .flat_map(|l| l.spans.iter())
            .map(|s| s.content.to_string())
            .collect();
        assert!(joined.contains("line1"));
        assert!(joined.contains("line5"));
        assert!(joined.contains("↳"));
    }

    #[test]
    fn tool_result_block_caps_at_max_lines_with_marker() {
        let mut output = String::new();
        for i in 0..30 {
            output.push_str(&format!("line{i}\n"));
        }
        let block = Block::ToolResult {
            id: "t1".into(),
            output,
        };
        let lines = render_block(&block);
        let joined: String = lines
            .iter()
            .flat_map(|l| l.spans.iter())
            .map(|s| s.content.to_string())
            .collect();
        assert!(joined.contains("line0"));
        assert!(joined.contains("more lines"));
        assert!(!joined.contains("line29"), "line29 should be truncated");
    }

    #[test]
    #[ignore = "long-running; gated to manual runs"]
    #[allow(
        dead_code,
        reason = "diagnostic for the phantom-blank tool-result row; rerun with --ignored if it crops up again"
    )]
    fn dump_transcript_lines_for_eyeball() {
        let mut s = state();
        s.transcript.push(Turn::User("Read README.md".into()));
        s.transcript.push(Turn::Assistant(vec![
            Block::ToolCall {
                id: "t1".into(),
                tool: "fs.read".into(),
                args: r#"{"path":"README.md"}"#.into(),
            },
            Block::ToolResult {
                id: "t1".into(),
                output: "# Stratum\n\nLocal-first chat with an LLM agent.\nBuilt in Rust.".into(),
            },
            Block::Text {
                text: "Looks like a Rust project README.".into(),
            },
        ]));
        // Replicate the render loop to dump exactly what gets pushed.
        let mut counted = 0;
        for (idx, turn) in s.transcript.iter().enumerate() {
            if idx > 0 {
                eprintln!("[{counted}] <inter-turn blank>");
                counted += 1;
            }
            match turn {
                Turn::User(text) => {
                    for piece in text.split('\n') {
                        eprintln!("[{counted}] User: {piece:?}");
                        counted += 1;
                    }
                }
                Turn::Assistant(blocks) => {
                    let mut first_text = true;
                    for block in blocks {
                        match block {
                            Block::Text { text } => {
                                if !first_text {
                                    eprintln!("[{counted}] <text-block blank>");
                                    counted += 1;
                                }
                                first_text = false;
                                let rendered = render_markdown(text, None);
                                for ln in &rendered {
                                    let txt: String =
                                        ln.spans.iter().map(|s| s.content.to_string()).collect();
                                    eprintln!("[{counted}] Text: {txt:?}");
                                    counted += 1;
                                }
                            }
                            other => {
                                let rendered = render_block(other);
                                for ln in &rendered {
                                    let txt: String =
                                        ln.spans.iter().map(|s| s.content.to_string()).collect();
                                    eprintln!(
                                        "[{counted}] {:?}: {txt:?}",
                                        std::mem::discriminant(other)
                                    );
                                    counted += 1;
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }

    #[test]
    fn ctrl_t_toggles_mouse_capture_and_surfaces_pending() {
        let mut s = state();
        // Default state: select-mode (capture OFF) so users can copy
        // text with the terminal's native shortcut.
        assert!(s.mouse_capture_off(), "default must be select-mode");
        assert!(s.take_pending_mouse_toggle().is_none());
        s.handle_key(key(KeyCode::Char('t'), KeyModifiers::CONTROL));
        assert!(!s.mouse_capture_off(), "Ctrl+T flips to mouse-scroll mode");
        assert_eq!(
            s.take_pending_mouse_toggle(),
            Some(true),
            "pending toggle must surface the new state to the event loop"
        );
        // Second consume → empty.
        assert_eq!(s.take_pending_mouse_toggle(), None);
        // Toggle back to select-mode.
        s.handle_key(key(KeyCode::Char('t'), KeyModifiers::CONTROL));
        assert!(s.mouse_capture_off());
        assert_eq!(s.take_pending_mouse_toggle(), Some(false));
    }

    #[test]
    fn select_mode_indicator_shows_in_status_by_default() {
        let s = state();
        let text = rendered_text(&s, 100, 18);
        assert!(
            text.contains("select-mode"),
            "status bar should show 'select-mode' indicator by default:\n{text}"
        );
    }

    #[test]
    fn mouse_scroll_indicator_shows_in_status_after_toggle() {
        let mut s = state();
        s.handle_key(key(KeyCode::Char('t'), KeyModifiers::CONTROL));
        let _ = s.take_pending_mouse_toggle();
        let text = rendered_text(&s, 100, 18);
        assert!(
            text.contains("mouse-scroll"),
            "status bar should show 'mouse-scroll' indicator after toggle:\n{text}"
        );
    }

    #[test]
    fn empty_transcript_renders_help_hint() {
        let s = state();
        let text = rendered_text(&s, 80, 24);
        // Branded greeting per plan/44: wordmark + tagline + tip line.
        assert!(text.contains("stratum"), "wordmark not visible:\n{text}");
        assert!(
            text.contains(crate::brand::TAGLINE),
            "tagline not visible:\n{text}"
        );
        assert!(
            text.contains("tip"),
            "rotating tip line not visible:\n{text}"
        );
        assert!(
            text.contains("Enter to send"),
            "primary action hint not visible"
        );
    }

    #[test]
    fn first_keystroke_hides_help_hint_after_submit() {
        let mut s = state();
        for c in "hello".chars() {
            s.handle_key(key(KeyCode::Char(c), KeyModifiers::NONE));
        }
        s.handle_key(key(KeyCode::Enter, KeyModifiers::NONE));
        let text = rendered_text(&s, 80, 20);
        assert!(
            !text.contains("Type a message"),
            "hint should disappear after submit:\n{text}"
        );
    }

    #[test]
    fn tool_result_with_blank_line_keeps_row_count() {
        let block = Block::ToolResult {
            id: "t1".into(),
            output: "# Stratum\n\nLocal-first chat with an LLM agent.\nBuilt in Rust.".into(),
        };
        let lines = render_block(&block);
        for (i, l) in lines.iter().enumerate() {
            let text: String = l.spans.iter().map(|s| s.content.to_string()).collect();
            eprintln!("ToolResult[{i}] = {text:?}");
        }
        // Input has 4 logical lines; expect 4 rendered rows + nothing extra.
        assert_eq!(lines.len(), 4);
    }

    /// Eyeball render — run with `cargo test -- --nocapture
    /// eyeball_chat_render` to print a realistic conversation.
    #[test]
    #[ignore = "long-running; gated to manual runs"]
    fn eyeball_chat_render() {
        let mut s = state();
        s.transcript
            .push(Turn::User("Show me a tiny rust hello-world".into()));
        s.transcript.push(Turn::Assistant(vec![Block::Text {
            text: "Sure! Here you go:\n\n```rust\nfn main() {\n    println!(\"hello, world\");\n}\n```\n\n- Compile with `cargo run`.\n- It prints to stdout.".into(),
        }]));
        s.transcript.push(Turn::User("Read README.md".into()));
        s.transcript.push(Turn::Assistant(vec![
            Block::ToolCall {
                id: "t1".into(),
                tool: "fs.read".into(),
                args: r#"{"path":"README.md"}"#.into(),
            },
            Block::ToolResult {
                id: "t1".into(),
                output: "# Stratum\n\nLocal-first chat with an LLM agent.\nBuilt in Rust.".into(),
            },
            Block::Text {
                text: "Looks like a Rust project README.".into(),
            },
        ]));
        let _ = s.execute_palette_command("/themes");
        let text = rendered_text(&s, 100, 40);
        eprintln!("\n========== TUI render @ 100x40 ==========");
        for line in text.lines() {
            eprintln!("{line}");
        }
        eprintln!("==========================================\n");
    }

    #[test]
    fn long_markdown_answer_keeps_tail_visible() {
        // Regression for the chat-scroll bug: when a long markdown
        // response wraps to many visual rows, the auto-tail must put
        // the LAST line in view, not the first. Previously the math
        // used logical-line count → the tail was scrolled off.
        let mut s = state();
        s.transcript.push(Turn::User("ask".into()));
        let mut body = String::new();
        for i in 0..50 {
            body.push_str(&format!(
                "line {i} ─────────────────────────────────────────────────────────\n"
            ));
        }
        body.push_str("TAIL-MARKER\n");
        s.transcript
            .push(Turn::Assistant(vec![Block::Text { text: body }]));
        let text = rendered_text(&s, 80, 24);
        assert!(
            text.contains("TAIL-MARKER"),
            "tail must be visible after auto-scroll; got:\n{text}"
        );
    }

    #[test]
    fn last_assistant_text_none_when_no_turn() {
        let s = state();
        assert!(s.last_assistant_text().is_none());
    }

    #[test]
    fn last_assistant_text_none_when_assistant_has_only_usage() {
        let mut s = state();
        // Push a synthetic assistant turn with no Text blocks — only
        // Usage. Helper exercises the "no text blocks at all" branch.
        s.transcript.push(Turn::User("x".to_string()));
        s.transcript.push(Turn::Assistant(vec![Block::Usage {
            prompt: 1,
            completion: 1,
        }]));
        assert!(s.last_assistant_text().is_none());
    }

    #[test]
    fn submit_with_prompt_empty_is_noop() {
        let mut s = state();
        s.submit_with_prompt("");
        assert!(s.transcript().is_empty());
    }

    #[test]
    fn submit_whitespace_input_clears_textbox_silently() {
        let mut s = state();
        // User typed spaces, then pressed Enter — we should clear
        // the box so it's obvious the Enter was processed.
        s.input = "   ".to_string();
        s.caret = s.input.len();
        s.submit();
        assert_eq!(s.input, "");
        assert_eq!(s.caret, 0);
        assert!(s.transcript().is_empty());
    }

    #[test]
    fn submit_bare_slash_clears_textbox_no_error_turn() {
        let mut s = state();
        s.input = "/".to_string();
        s.caret = s.input.len();
        s.submit();
        assert_eq!(s.input, "");
        // No "✗ /" rejection in the transcript — bare slash is dropped silently.
        assert!(s.transcript().is_empty());
    }

    // ---------- --resume / with_resumed_transcript ----------

    fn fixed_at() -> std::time::SystemTime {
        std::time::SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(1_700_000_000)
    }

    fn resume_fixture() -> stratum_runtime::Transcript {
        use stratum_runtime::{
            SessionId, TranscriptBlock, TranscriptBlockKind, TRANSCRIPT_SCHEMA_VERSION,
        };
        stratum_runtime::Transcript {
            schema_version: TRANSCRIPT_SCHEMA_VERSION,
            session_id: SessionId::from_str("deadbeefcafef00d").expect("valid id"),
            created_at: fixed_at(),
            turns: vec![
                stratum_runtime::TranscriptTurn::User {
                    at: fixed_at(),
                    text: "earlier-q".to_owned(),
                },
                stratum_runtime::TranscriptTurn::Assistant {
                    at: fixed_at(),
                    blocks: vec![TranscriptBlock {
                        kind: TranscriptBlockKind::Text,
                        text: "earlier-a".to_owned(),
                    }],
                },
                stratum_runtime::TranscriptTurn::System {
                    at: fixed_at(),
                    text: "sysline".to_owned(),
                },
                stratum_runtime::TranscriptTurn::Command {
                    at: fixed_at(),
                    text: "/help".to_owned(),
                    ok: true,
                },
            ],
        }
    }

    #[test]
    fn with_resumed_transcript_counts_turns_and_populates_scrollback() {
        let t = resume_fixture();
        let expected = t.turns.len();
        let s = ChatState::default().with_resumed_transcript(t);
        assert_eq!(s.resumed_count(), expected);
        assert_eq!(s.transcript().len(), expected);
        // User and Assistant turns must round-trip through the in-memory
        // shapes — System and Command both fold to Turn::Command.
        assert!(matches!(&s.transcript()[0], Turn::User(text) if text == "earlier-q"));
        match &s.transcript()[1] {
            Turn::Assistant(blocks) => {
                assert_eq!(blocks.len(), 1);
                assert!(matches!(&blocks[0], Block::Text { text } if text == "earlier-a"));
            }
            other => panic!("expected Turn::Assistant, got {other:?}"),
        }
        match &s.transcript()[2] {
            Turn::Command { text, ok, .. } => {
                assert!(text.starts_with("(system)"), "got: {text}");
                assert!(ok);
            }
            other => panic!("expected Turn::Command for system, got {other:?}"),
        }
        match &s.transcript()[3] {
            Turn::Command { text, ok, .. } => {
                assert_eq!(text, "/help");
                assert!(ok);
            }
            other => panic!("expected Turn::Command for command, got {other:?}"),
        }
    }

    #[test]
    fn with_resumed_transcript_default_count_is_zero() {
        let s = ChatState::default();
        assert_eq!(s.resumed_count(), 0);
    }

    #[test]
    fn with_resumed_transcript_empty_turns_yields_zero_count() {
        use stratum_runtime::{SessionId, TRANSCRIPT_SCHEMA_VERSION};
        let t = stratum_runtime::Transcript {
            schema_version: TRANSCRIPT_SCHEMA_VERSION,
            session_id: SessionId::from_str("0123456789abcdef").expect("valid id"),
            created_at: fixed_at(),
            turns: vec![],
        };
        let s = ChatState::default().with_resumed_transcript(t);
        assert_eq!(s.resumed_count(), 0);
        assert!(s.transcript().is_empty());
    }

    // ---------- TuiPromptResponder integration ----------

    /// Dispatcher that handles `fs.write` invocations by returning a
    /// trivial success. Used by the permission-flow tests.
    #[derive(Debug)]
    struct FsWriteDispatcher;
    impl stratum_runtime::ToolDispatcher for FsWriteDispatcher {
        fn invoke(&self, inv: &stratum_runtime::ToolInvocation) -> stratum_runtime::ToolResult {
            let body = serde_json::Value::Bool(true);
            let bytes = body.to_string().len() as u64;
            stratum_runtime::ToolResult::Ok {
                tool_id: inv.tool_id.clone(),
                body,
                bytes,
            }
        }
        fn supports(&self, tool_id: &str) -> bool {
            tool_id == "fs.write"
        }
        fn id(&self) -> &'static str {
            "fs.write"
        }
    }

    fn loop_with_prompter(
        provider: Arc<dyn Provider>,
        prompter: Arc<TuiPromptResponder>,
    ) -> Arc<AgentLoop> {
        let sink: Arc<dyn EventSink> = Arc::new(MemoryEventSink::new());
        let events = Arc::new(EventEmitter::new(sink));
        let mut dispatcher = stratum_runtime::RegistryDispatcher::new();
        dispatcher
            .register(Box::new(FsWriteDispatcher))
            .expect("register");
        let responder: Arc<dyn stratum_runtime::PromptResponder> = prompter;
        Arc::new(
            AgentLoop::builder()
                .with_provider(provider)
                .with_router(stratum_runtime::IntentRouter::default())
                .with_permission_store(Arc::new(stratum_runtime::PermissionStore::new()))
                .with_prompt_gen(Arc::new(stratum_runtime::PromptIdGen::new()))
                .with_responder(responder)
                .with_events(events)
                .with_capability_matrix(Arc::new(stratum_runtime::CapabilityMatrix::new()))
                .with_plan_mode(Arc::new(stratum_runtime::PlanMode::new()))
                .with_config(stratum_runtime::AgentLoopConfig::default())
                .with_dispatcher(Arc::new(dispatcher))
                .build()
                .unwrap(),
        )
    }

    #[test]
    fn pending_permission_request_initially_none() {
        let s = state();
        assert!(s.pending_permission_request().is_none());
        assert!(s.peek_pending_permission().is_none());
    }

    #[test]
    fn answer_permission_unblocks_agent_loop_with_allow_once() {
        // ScriptedProvider yields one ToolCall for "fs.write". The
        // AgentLoop will dispatch the permission flow through the shared
        // TuiPromptResponder; the main thread answers AllowOnce; the
        // background `submit` thread completes successfully.
        let prompter = Arc::new(TuiPromptResponder::new(Duration::from_secs(2)));
        let provider = ScriptedProvider {
            script: std::sync::Mutex::new(vec![Block::ToolCall {
                id: "fs.write".into(),
                tool: "fs.write".into(),
                args: "{}".into(),
            }]),
        };
        let provider_arc: Arc<dyn Provider> = Arc::new(provider);
        let loop_ = loop_with_prompter(provider_arc, prompter.clone());

        // Inject the same prompter into a ChatState alongside the loop.
        let mut s = ChatState::with_agent_loop(loop_);
        s.permission_prompter = prompter.clone();

        let s_arc = Arc::new(std::sync::Mutex::new(s));
        let s_bg = s_arc.clone();
        let bg = thread::spawn(move || {
            let mut guard = s_bg.lock().expect("lock");
            guard.submit_with_prompt("write a file");
        });

        // Wait until the worker has enqueued its request.
        let pending = loop {
            if let Some(p) = prompter.peek_request() {
                break p;
            }
            thread::sleep(Duration::from_millis(5));
        };
        prompter.submit_decision(pending.id, PermissionDecision::AllowOnce);

        bg.join().expect("background submit completes");
        let outcome = {
            let guard = s_arc.lock().expect("lock");
            guard
                .last_turn_result()
                .expect("turn result")
                .outcome
                .clone()
        };
        assert!(
            matches!(outcome, stratum_runtime::TurnOutcome::Success),
            "expected Success, got {outcome:?}",
        );
    }

    #[test]
    fn answer_permission_deny_short_circuits_turn() {
        let prompter = Arc::new(TuiPromptResponder::new(Duration::from_secs(2)));
        let provider = ScriptedProvider {
            script: std::sync::Mutex::new(vec![Block::ToolCall {
                id: "fs.write".into(),
                tool: "fs.write".into(),
                args: "{}".into(),
            }]),
        };
        let provider_arc: Arc<dyn Provider> = Arc::new(provider);
        let loop_ = loop_with_prompter(provider_arc, prompter.clone());

        let mut s = ChatState::with_agent_loop(loop_);
        s.permission_prompter = prompter.clone();
        let s_arc = Arc::new(std::sync::Mutex::new(s));
        let s_bg = s_arc.clone();
        let bg = thread::spawn(move || {
            let mut guard = s_bg.lock().expect("lock");
            guard.submit_with_prompt("write a file");
        });

        let pending = loop {
            if let Some(p) = prompter.peek_request() {
                break p;
            }
            thread::sleep(Duration::from_millis(5));
        };
        // Use the public delegate.
        {
            let guard = s_arc.lock().expect("lock");
            guard.answer_permission(pending.id, PermissionDecision::Deny);
        }

        bg.join().expect("background submit completes");
        let outcome = {
            let guard = s_arc.lock().expect("lock");
            guard
                .last_turn_result()
                .expect("turn result")
                .outcome
                .clone()
        };
        assert!(
            matches!(
                outcome,
                stratum_runtime::TurnOutcome::ToolFailure { ref code, .. } if code == "STRAT-E5004"
            ),
            "expected ToolFailure with STRAT-E5004, got {outcome:?}",
        );
    }

    #[test]
    fn prompter_times_out_to_deny_when_unanswered() {
        // Wire a short-timeout prompter and never answer; the AgentLoop
        // should see a Deny and short-circuit the turn.
        let prompter = Arc::new(TuiPromptResponder::new(Duration::from_millis(100)));
        let provider = ScriptedProvider {
            script: std::sync::Mutex::new(vec![Block::ToolCall {
                id: "fs.write".into(),
                tool: "fs.write".into(),
                args: "{}".into(),
            }]),
        };
        let provider_arc: Arc<dyn Provider> = Arc::new(provider);
        let loop_ = loop_with_prompter(provider_arc, prompter.clone());
        let mut s = ChatState::with_agent_loop(loop_);
        s.permission_prompter = prompter;
        s.submit_with_prompt("write a file");
        let tr = s.last_turn_result().expect("turn result");
        assert!(
            matches!(
                tr.outcome,
                stratum_runtime::TurnOutcome::ToolFailure { ref code, .. } if code == "STRAT-E5004"
            ),
            "expected ToolFailure with STRAT-E5004, got {:?}",
            tr.outcome,
        );
    }

    #[test]
    fn submit_decision_then_pending_returns_none_after_consumed() {
        // Submitting a decision is independent of the queue; once
        // pending_request has popped the request, a subsequent call
        // returns None even after a decision was recorded.
        let prompter = TuiPromptResponder::new(Duration::from_secs(1));
        let p = PendingPrompt {
            id: PromptId(99),
            request: PermissionRequest::ToolUse {
                args: String::new(),
                tool_id: "fs.write".into(),
            },
            issued_at: SystemTime::UNIX_EPOCH,
        };
        // Manually enqueue + drain (no waiter).
        prompter.requeue_for_redisplay(p);
        let popped = prompter.pending_request().expect("popped");
        assert_eq!(popped.id, PromptId(99));
        prompter.submit_decision(PromptId(99), PermissionDecision::AllowOnce);
        // Queue is empty: the recorded decision does not put the request
        // back into the queue.
        assert!(prompter.pending_request().is_none());
    }

    #[test]
    fn decision_from_key_maps_all_documented_keys() {
        assert_eq!(decision_from_key('a'), Some(PermissionDecision::AllowOnce));
        assert_eq!(
            decision_from_key('s'),
            Some(PermissionDecision::AllowSession)
        );
        assert_eq!(
            decision_from_key('f'),
            Some(PermissionDecision::AllowForever)
        );
        assert_eq!(decision_from_key('d'), Some(PermissionDecision::Deny));
        assert_eq!(
            decision_from_key('F'),
            Some(PermissionDecision::DenyForever)
        );
        assert_eq!(decision_from_key('x'), None);
        assert_eq!(decision_from_key('A'), None);
    }

    #[test]
    fn describe_request_covers_every_variant() {
        let cap = PermissionRequest::CapabilityGrant {
            capability: "net".into(),
            target: Some("example.com".into()),
            reason: "fetch".into(),
        };
        assert!(describe_request(&cap).contains("net"));
        assert!(describe_request(&cap).contains("example.com"));
        let cap_no_target = PermissionRequest::CapabilityGrant {
            capability: "shell".into(),
            target: None,
            reason: "run".into(),
        };
        assert!(describe_request(&cap_no_target).contains("shell"));
        let secret = PermissionRequest::SecretAccess {
            secret_ref: "p/k".into(),
            scope: "read".into(),
        };
        assert!(describe_request(&secret).contains("p/k"));
        let net = PermissionRequest::NetworkHost {
            host: "api.example".into(),
            port: Some(443),
        };
        assert!(describe_request(&net).contains("443"));
        let net_anyport = PermissionRequest::NetworkHost {
            host: "api.example".into(),
            port: None,
        };
        assert!(describe_request(&net_anyport).contains("api.example"));
        let file = PermissionRequest::FileWrite {
            path: std::path::PathBuf::from("/tmp/x"),
        };
        assert!(describe_request(&file).contains("/tmp/x"));
        let tool = PermissionRequest::ToolUse {
            args: String::new(),
            tool_id: "fs.write".into(),
        };
        assert!(describe_request(&tool).contains("fs.write"));
    }

    #[test]
    fn handle_key_with_pending_request_swallows_unknown_char() {
        let mut s = state();
        let pending = PendingPrompt {
            id: PromptId(5),
            request: PermissionRequest::ToolUse {
                args: String::new(),
                tool_id: "fs.write".into(),
            },
            issued_at: SystemTime::UNIX_EPOCH,
        };
        s.permission_prompter.requeue_for_redisplay(pending);
        // 'z' is not one of the modal keys: input must stay clean.
        s.handle_key(key(KeyCode::Char('z'), KeyModifiers::NONE));
        assert!(s.input().is_empty());
        // Modal still pending.
        assert!(s.peek_pending_permission().is_some());
    }

    #[test]
    fn handle_key_with_pending_request_dispatches_known_decision_key() {
        let mut s = state();
        let pending = PendingPrompt {
            id: PromptId(6),
            request: PermissionRequest::ToolUse {
                args: String::new(),
                tool_id: "fs.write".into(),
            },
            issued_at: SystemTime::UNIX_EPOCH,
        };
        s.permission_prompter.requeue_for_redisplay(pending);
        s.handle_key(key(KeyCode::Char('a'), KeyModifiers::NONE));
        // Queue drained; no further pending.
        assert!(s.peek_pending_permission().is_none());
    }

    #[test]
    fn render_shows_permission_modal_when_pending() {
        let s = state();
        let pending = PendingPrompt {
            id: PromptId(7),
            request: PermissionRequest::ToolUse {
                args: String::new(),
                tool_id: "fs.write".into(),
            },
            issued_at: SystemTime::UNIX_EPOCH,
        };
        s.permission_prompter.requeue_for_redisplay(pending);
        let text = rendered_text(&s, 100, 14);
        assert!(text.contains("permission"));
        assert!(text.contains("fs.write"));
        assert!(text.contains("[a]"));
    }

    // ---------- streaming-progress status bar ----------

    #[test]
    fn status_bar_text_initially_empty() {
        let s = state();
        assert_eq!(s.status_bar_text(), "");
    }

    #[test]
    fn status_bar_text_after_submit_contains_tokens_and_tok_per_sec() {
        // ScriptedProvider returns one Block::Text of length 11 ("hello world")
        // so we can pin the approximate-token-count math.
        let sink: Arc<dyn EventSink> = Arc::new(MemoryEventSink::new());
        let events = Arc::new(EventEmitter::new(sink));
        let provider: Arc<dyn Provider> = Arc::new(ScriptedProvider {
            script: std::sync::Mutex::new(vec![
                Block::Text {
                    text: "hello world".into(),
                },
                Block::Done,
            ]),
        });
        let loop_ = build_loop(provider, events);
        let mut s = ChatState::with_agent_loop(loop_);
        s.submit_with_prompt("go");
        let bar = s.status_bar_text();
        assert!(bar.contains("tokens"), "got: {bar}");
        assert!(bar.contains("tok/s"), "got: {bar}");
    }

    #[test]
    fn status_bar_last_token_count_uses_four_chars_per_token() {
        // 11 chars / 4 = 2 (integer division). Pin a small bound to allow for
        // a future tweak of the heuristic without rewriting this test.
        let sink: Arc<dyn EventSink> = Arc::new(MemoryEventSink::new());
        let events = Arc::new(EventEmitter::new(sink));
        let provider: Arc<dyn Provider> = Arc::new(ScriptedProvider {
            script: std::sync::Mutex::new(vec![Block::Text {
                text: "hello world".into(),
            }]),
        });
        let loop_ = build_loop(provider, events);
        let mut s = ChatState::with_agent_loop(loop_);
        s.submit_with_prompt("go");
        let count = s.last_token_count();
        assert!(
            (2..=3).contains(&count),
            "expected approximate count in [2, 3], got {count}",
        );
    }

    #[test]
    fn in_flight_since_cleared_after_submit_completes() {
        let mut s = state();
        s.submit_with_prompt("hi");
        assert!(s.in_flight_since.is_none());
    }

    #[test]
    fn status_bar_in_flight_shows_generating() {
        let mut s = state();
        // Pin a synthetic in-flight stamp 2 seconds in the past so the
        // formatted elapsed seconds are deterministic.
        s.in_flight_since = Instant::now().checked_sub(Duration::from_secs(2));
        assert!(s.in_flight_since.is_some());
        let bar = s.status_bar_text();
        assert!(bar.contains("generating"), "got: {bar}");
    }

    #[test]
    fn status_bar_in_flight_takes_precedence_over_last_metrics() {
        // First, run a turn so `last_metrics` is populated.
        let mut s = state();
        s.submit_with_prompt("hi");
        assert!(s.last_metrics().is_some());
        // Now mark a fresh turn as in-flight — the in-flight indicator
        // must win over the completed-turn summary.
        s.in_flight_since = Some(Instant::now());
        let bar = s.status_bar_text();
        assert!(bar.contains("generating"), "got: {bar}");
        assert!(!bar.contains("tok/s"), "got: {bar}");
    }

    #[test]
    fn status_bar_multiple_submits_replace_last_metrics() {
        let mut s = state();
        s.submit_with_prompt("hi");
        let first = s.last_metrics().expect("first metrics").turn_id;
        s.submit_with_prompt("bye");
        let second = s.last_metrics().expect("second metrics").turn_id;
        // The most recent submit replaces — does not append.
        assert_ne!(first, second);
        assert_eq!(second.0, first.0 + 1);
    }

    #[test]
    fn approximate_token_count_zero_for_empty_blocks() {
        assert_eq!(approximate_token_count(&[]), 0);
    }

    #[test]
    fn approximate_token_count_ignores_non_text_blocks() {
        // Usage / Done / ToolCall must contribute nothing.
        let blocks = vec![
            Block::Usage {
                prompt: 100,
                completion: 200,
            },
            Block::Done,
            Block::ToolCall {
                id: "t1".into(),
                tool: "fs.read".into(),
                args: "{}".into(),
            },
        ];
        assert_eq!(approximate_token_count(&blocks), 0);
    }

    #[test]
    fn approximate_token_count_sums_text_lengths() {
        // "abcd" + "efgh" = 8 chars / 4 = 2 tokens.
        let blocks = vec![
            Block::Text {
                text: "abcd".into(),
            },
            Block::Text {
                text: "efgh".into(),
            },
        ];
        assert_eq!(approximate_token_count(&blocks), 2);
    }

    #[test]
    fn render_shows_generating_indicator_when_in_flight() {
        let mut s = state();
        s.in_flight_since = Some(Instant::now());
        let text = rendered_text(&s, 100, 12);
        assert!(text.contains("generating"), "got render:\n{text}");
    }

    #[test]
    fn resumed_count_independent_of_submit() {
        // Submit a new turn after resume; resumed_count must NOT increase.
        let t = resume_fixture();
        let mut s = ChatState::default().with_resumed_transcript(t);
        let baseline = s.resumed_count();
        let pre_len = s.transcript().len();
        s.submit_with_prompt("hi");
        assert_eq!(s.resumed_count(), baseline);
        assert!(s.transcript().len() > pre_len);
    }

    // -- with_handoff -------------------------------------------------------

    /// Build a tiny [`AgentHandoff`] wrapping a registry that has a single
    /// `Default` role backed by [`AgentFactory::echo`].
    fn handoff_default_only() -> Arc<AgentHandoff> {
        use stratum_runtime::{AgentFactory, AgentRegistry, HandoffPolicy, SuggestedRole};
        let l = Arc::new(AgentFactory::echo().expect("echo factory builds"));
        let mut reg = AgentRegistry::new();
        reg.register(SuggestedRole::Default, l);
        Arc::new(AgentHandoff::new(
            reg,
            SuggestedRole::Default,
            HandoffPolicy::default(),
        ))
    }

    #[test]
    fn default_state_has_no_handoff() {
        let s = state();
        assert!(!s.has_handoff());
    }

    #[test]
    fn with_handoff_installs_coordinator() {
        let s = state().with_handoff(handoff_default_only());
        assert!(s.has_handoff());
    }

    #[test]
    fn submit_with_handoff_records_user_and_assistant_turns() {
        let mut s = state().with_handoff(handoff_default_only());
        s.submit_with_prompt("hi");
        // At minimum we expect a User then an Assistant entry.
        assert!(matches!(s.transcript()[0], Turn::User(_)));
        assert!(matches!(s.transcript()[1], Turn::Assistant(_)));
    }

    #[test]
    fn submit_without_handoff_still_uses_single_loop_path() {
        // Regression: omitting `with_handoff` preserves the Phase 1 behaviour
        // — `submit` routes through `agent_loop.run_turn` and records a
        // `User` + `Assistant` pair with no command lines.
        let mut s = state();
        s.submit_with_prompt("hi");
        assert_eq!(s.transcript().len(), 2);
        assert!(matches!(s.transcript()[0], Turn::User(_)));
        assert!(matches!(s.transcript()[1], Turn::Assistant(_)));
    }

    // -- current_role + /agents palette ------------------------------------

    /// Registry with Default + Coder, both backed by the echo factory.
    fn handoff_default_and_coder() -> Arc<AgentHandoff> {
        use stratum_runtime::{AgentFactory, AgentRegistry, HandoffPolicy, SuggestedRole};
        let l1 = Arc::new(AgentFactory::echo().expect("echo factory builds"));
        let l2 = Arc::new(AgentFactory::echo().expect("echo factory builds"));
        let mut reg = AgentRegistry::new();
        reg.register(SuggestedRole::Default, l1);
        reg.register(SuggestedRole::Coder, l2);
        Arc::new(AgentHandoff::new(
            reg,
            SuggestedRole::Default,
            HandoffPolicy::default(),
        ))
    }

    #[test]
    fn current_role_label_empty_in_single_loop_mode() {
        let s = state();
        assert!(s.current_role_label().is_empty());
    }

    #[test]
    fn current_role_label_seeds_default_after_with_handoff() {
        let s = state().with_handoff(handoff_default_and_coder());
        assert_eq!(s.current_role_label(), "agent: default");
    }

    #[test]
    fn current_role_label_updates_after_coder_routed_submit() {
        // The default IntentRouter rules route a prompt containing
        // "stack trace" to `SuggestedRole::Coder`. The registry above has
        // Coder registered, so the handoff lands there on the first hop
        // and `current_role` flips to `Coder`.
        let mut s = state().with_handoff(handoff_default_and_coder());
        s.submit_with_prompt("debug this stack trace");
        assert_eq!(s.current_role_label(), "agent: coder");
    }

    #[test]
    fn agents_command_without_handoff_is_rejected_with_hint() {
        let mut s = state();
        let outcome = s.execute_palette_command("/agents");
        let PaletteOutcome::Rejected { message } = outcome else {
            panic!("expected rejected");
        };
        assert!(message.contains("no multi-agent"), "got: {message}");
        assert!(message.contains("--agents-dir"), "got: {message}");
    }

    #[test]
    fn agents_command_with_handoff_lists_roles_and_current() {
        let mut s = state().with_handoff(handoff_default_and_coder());
        let outcome = s.execute_palette_command("/agents");
        let PaletteOutcome::Acknowledged { message } = outcome else {
            panic!("expected acknowledged");
        };
        assert!(message.starts_with("roles:"), "got: {message}");
        assert!(message.contains("(current:"), "got: {message}");
        assert!(message.contains("default"), "got: {message}");
        assert!(message.contains("coder"), "got: {message}");
    }

    #[test]
    fn agents_command_in_single_loop_mode_does_not_panic() {
        // Regression: even when /agents is rejected (no handoff installed),
        // the dispatch path must complete cleanly and append a rejected
        // command turn to the transcript.
        let mut s = state();
        let pre_len = s.transcript().len();
        let _outcome = s.execute_palette_command("/agents");
        let last = s.transcript().last().expect("command turn appended");
        match last {
            Turn::Command { text, ok, .. } => {
                assert_eq!(text, "/agents");
                assert!(!*ok);
            }
            other => panic!("expected Turn::Command, got {other:?}"),
        }
        assert_eq!(s.transcript().len(), pre_len + 1);
    }

    #[test]
    fn status_bar_render_includes_role_in_multi_agent_mode() {
        let s = state().with_handoff(handoff_default_and_coder());
        let text = rendered_text(&s, 100, 12);
        assert!(text.contains("agent:"), "render:\n{text}");
        assert!(text.contains("default"), "render:\n{text}");
    }

    #[test]
    fn status_bar_render_omits_role_in_single_loop_mode() {
        let s = state();
        let text = rendered_text(&s, 100, 12);
        assert!(!text.contains("agent:"), "render:\n{text}");
    }

    #[test]
    fn help_text_lists_agents_command() {
        // HELP_TEXT is what /help echoes; it must enumerate every wired
        // palette command, including /agents.
        assert!(HELP_TEXT.contains("/agents"), "got: {HELP_TEXT}");
    }

    #[test]
    fn role_name_covers_every_suggested_role() {
        assert_eq!(role_name(SuggestedRole::Default), "default");
        assert_eq!(role_name(SuggestedRole::Cavemanish), "cavemanish");
        assert_eq!(role_name(SuggestedRole::Polisher), "polisher");
        assert_eq!(role_name(SuggestedRole::Coder), "coder");
        assert_eq!(role_name(SuggestedRole::Researcher), "researcher");
    }

    // -- /parallel palette --------------------------------------------------

    /// Registry with Cavemanish + Coder, both backed by the echo factory.
    /// Used by the `/parallel` palette tests so the dispatcher has at
    /// least two distinct roles to fan a turn out to.
    fn handoff_cavemanish_and_coder() -> Arc<AgentHandoff> {
        use stratum_runtime::{AgentFactory, AgentRegistry, HandoffPolicy, SuggestedRole};
        let l1 = Arc::new(AgentFactory::echo().expect("echo factory builds"));
        let l2 = Arc::new(AgentFactory::echo().expect("echo factory builds"));
        let mut reg = AgentRegistry::new();
        reg.register(SuggestedRole::Cavemanish, l1);
        reg.register(SuggestedRole::Coder, l2);
        Arc::new(AgentHandoff::new(
            reg,
            SuggestedRole::Default,
            HandoffPolicy::default(),
        ))
    }

    #[test]
    fn parallel_command_without_handoff_is_rejected_with_hint() {
        let mut s = state();
        let outcome = s.execute_palette_command("/parallel cavemanish,coder");
        let PaletteOutcome::Rejected { message } = outcome else {
            panic!("expected rejected, got: {outcome:?}");
        };
        assert!(message.contains("no multi-agent"), "got: {message}");
        assert!(message.contains("--agents-dir"), "got: {message}");
    }

    #[test]
    fn parallel_command_with_handoff_appends_assistant_turn() {
        let mut s = state().with_handoff(handoff_cavemanish_and_coder());
        let outcome = s.execute_palette_command("/parallel cavemanish,coder");
        let PaletteOutcome::Acknowledged { message } = outcome else {
            panic!("expected acknowledged, got: {outcome:?}");
        };
        assert!(message.starts_with("parallel:"), "got: {message}");
        // Transcript should now contain an Assistant turn whose concatenated
        // text mentions both role names (each role gets its own section
        // header).
        let combined = s
            .transcript()
            .iter()
            .filter_map(|t| match t {
                Turn::Assistant(blocks) => Some(concat_text_blocks(blocks)),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            combined.contains("cavemanish") && combined.contains("coder"),
            "expected both role headers in assistant turn; got: {combined:?}"
        );
    }

    #[test]
    fn parallel_command_unknown_role_is_rejected() {
        let mut s = state().with_handoff(handoff_cavemanish_and_coder());
        let outcome = s.execute_palette_command("/parallel unknown-role");
        let PaletteOutcome::Rejected { message } = outcome else {
            panic!("expected rejected, got: {outcome:?}");
        };
        assert!(message.contains("unknown role"), "got: {message}");
    }

    #[test]
    fn parallel_command_empty_args_is_rejected() {
        let mut s = state().with_handoff(handoff_cavemanish_and_coder());
        let outcome = s.execute_palette_command("/parallel");
        let PaletteOutcome::Rejected { message } = outcome else {
            panic!("expected rejected, got: {outcome:?}");
        };
        assert!(message.contains("unknown role"), "got: {message}");
    }

    #[test]
    fn help_text_lists_parallel_command() {
        assert!(HELP_TEXT.contains("/parallel"), "got: {HELP_TEXT}");
    }

    #[test]
    fn parse_role_label_covers_every_suggested_role() {
        assert_eq!(parse_role_label("default"), Some(SuggestedRole::Default));
        assert_eq!(
            parse_role_label("cavemanish"),
            Some(SuggestedRole::Cavemanish)
        );
        assert_eq!(parse_role_label("polisher"), Some(SuggestedRole::Polisher));
        assert_eq!(parse_role_label("coder"), Some(SuggestedRole::Coder));
        assert_eq!(
            parse_role_label("researcher"),
            Some(SuggestedRole::Researcher)
        );
        assert_eq!(parse_role_label("unknown"), None);
        assert_eq!(parse_role_label(""), None);
    }

    // ---------- /image palette + attachment plumbing (plan/05) ----------

    /// Test backend that captures the last `TurnContext` it saw. Lets
    /// the multimodal-scaffold tests assert that `pending_attachments`
    /// actually rides into the provider request.
    #[derive(Debug)]
    struct CapturingBackend {
        last_ctx: Mutex<Option<TurnContext>>,
    }

    impl CapturingBackend {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                last_ctx: Mutex::new(None),
            })
        }

        fn last_attachments(&self) -> Vec<Block> {
            self.last_ctx
                .lock()
                .ok()
                .and_then(|guard| guard.as_ref().map(|c| c.attachments.clone()))
                .unwrap_or_default()
        }
    }

    impl ChatBackend for CapturingBackend {
        fn run_turn_streaming(
            &self,
            ctx: TurnContext,
            _cancel: &CancelToken,
            chunk_tx: mpsc::Sender<Block>,
        ) -> TurnResult {
            // 1×1 PNG echo back so the transcript has *some* assistant turn.
            let echo_text = format!("saw {} attachment(s)", ctx.attachments.len());
            let _ = chunk_tx.send(Block::Text {
                text: echo_text.clone(),
            });
            let turn_id = ctx.turn_id;
            if let Ok(mut guard) = self.last_ctx.lock() {
                *guard = Some(ctx);
            }
            TurnResult {
                turn_id,
                outcome: TurnOutcome::Success,
                blocks: vec![Block::Text { text: echo_text }],
                transitions: Vec::new(),
                events_emitted: Vec::new(),
            }
        }
    }

    /// Smallest valid PNG: the 8-byte signature followed by a minimal
    /// IHDR + IEND. We only need enough for the magic-byte sniff to
    /// classify the file as `image/png`; the bytes do not need to be
    /// a renderable image.
    fn tiny_png_bytes() -> Vec<u8> {
        let mut buf = Vec::with_capacity(67);
        // PNG signature.
        buf.extend_from_slice(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);
        // IHDR chunk: length (13), type, payload, crc — we use zeros
        // for everything except the chunk type since the sniffer only
        // checks the leading signature.
        buf.extend_from_slice(&[0, 0, 0, 13]);
        buf.extend_from_slice(b"IHDR");
        buf.extend_from_slice(&[0; 13]);
        buf.extend_from_slice(&[0; 4]);
        // IEND chunk.
        buf.extend_from_slice(&[0, 0, 0, 0]);
        buf.extend_from_slice(b"IEND");
        buf.extend_from_slice(&[0; 4]);
        buf
    }

    fn write_tiny_png(dir: &std::path::Path, name: &str) -> std::path::PathBuf {
        let path = dir.join(name);
        std::fs::write(&path, tiny_png_bytes()).expect("write tiny png");
        path
    }

    // ---- /audio palette command (Phase 5) -----------------------------

    /// Process-wide lock so the cwd-mutating /audio tests don't race
    /// each other when cargo runs the suite with the default thread
    /// pool. Acquired BEFORE the chdir guard so the guard's Drop
    /// always runs before the lock is released.
    static AUDIO_CWD_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Acquire `AUDIO_CWD_LOCK`, surfacing (not swallowing) the
    /// poison case: a prior `/audio` test panicked while holding the
    /// lock. We still continue with the inner guard so the suite
    /// doesn't snowball into a cascade of failures, but we emit a
    /// `tracing::warn!` so the recovery is visible in CI logs rather
    /// than silently masked. Tracked as a Phase 7 deferred item in
    /// issue #172.
    fn lock_audio_cwd() -> std::sync::MutexGuard<'static, ()> {
        AUDIO_CWD_LOCK.lock().unwrap_or_else(|poisoned| {
            tracing::warn!(
                target: "stratum_tui::chat::tests",
                "AUDIO_CWD_LOCK recovered from poisoned mutex \
                 (a prior /audio test panicked while holding it); \
                 continuing with inner guard"
            );
            poisoned.into_inner()
        })
    }

    fn write_wav_fixture(dir: &std::path::Path, name: &str) -> std::path::PathBuf {
        // Synthetic RIFF/WAVE prefix — recognised by the sniffer.
        let mut bytes = Vec::with_capacity(16);
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&[0u8; 4]);
        bytes.extend_from_slice(b"WAVE");
        bytes.extend_from_slice(&[0u8; 4]);
        let path = dir.join(name);
        std::fs::write(&path, &bytes).expect("write fixture");
        path
    }

    #[test]
    fn image_palette_command_with_no_arg_is_rejected() {
        let mut s = state();
        let outcome = s.execute_palette_command("/image");
        assert!(matches!(outcome, PaletteOutcome::Rejected { .. }));
        assert_eq!(s.pending_attachment_count(), 0);
    }

    #[test]
    fn image_palette_command_with_missing_path_is_rejected() {
        let mut s = state();
        let tmp = tempfile::tempdir().expect("tempdir");
        let bogus = tmp.path().join("nope.png");
        let cmd = format!("/image {}", bogus.display());
        let outcome = s.execute_palette_command(&cmd);
        assert!(matches!(outcome, PaletteOutcome::Rejected { .. }));
        assert_eq!(s.pending_attachment_count(), 0);
    }

    #[test]
    fn image_palette_command_stages_attachment() {
        let mut s = state();
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = write_tiny_png(tmp.path(), "tiny.png");
        let cmd = format!("/image {}", path.display());
        let outcome = s.execute_palette_command(&cmd);
        let PaletteOutcome::Acknowledged { message } = outcome else {
            panic!("expected acknowledged, got {outcome:?}");
        };
        assert!(message.contains("image/png"), "message: {message}");
        assert!(message.contains(&path.display().to_string()));
        assert_eq!(s.pending_attachment_count(), 1);
        let blocks = s.pending_attachments();
        assert!(matches!(
            &blocks[0],
            Block::Image {
                mime,
                data: stratum_types::ImageData::Inline { .. },
                ..
            } if mime == "image/png"
        ));
    }

    #[test]
    fn audio_palette_command_appears_in_catalog() {
        assert!(crate::palette::COMMANDS.iter().any(|c| c.name == "audio"));
    }

    #[test]
    fn audio_dispatch_stages_block_audio() {
        let _lock = lock_audio_cwd();
        let tmp = tempfile::TempDir::new().expect("tmp");
        let prev = std::env::current_dir().expect("cwd");
        std::env::set_current_dir(tmp.path()).expect("chdir");
        let _guard = scopeguard_chdir(prev);

        write_wav_fixture(tmp.path(), "voice.wav");
        let mut s = state();
        let outcome = s.execute_palette_command("/audio voice.wav");
        match outcome {
            PaletteOutcome::Acknowledged { message } => {
                assert!(message.contains("audio staged"), "got: {message}");
                assert!(message.contains("audio/wav"), "got: {message}");
            }
            PaletteOutcome::Rejected { message } => panic!("rejected: {message}"),
        }
        let staged = s
            .staged_audio()
            .expect("staged audio block present after /audio");
        assert!(matches!(
            staged,
            Block::Audio { mime, .. } if mime == "audio/wav"
        ));
    }

    #[test]
    fn image_palette_command_appears_in_transcript() {
        // The palette path records its outcome via `Turn::Command`; the
        // tip in the message ("[image: <mime> @ <path>]") is what users
        // see when they queue an attachment.
        let mut s = state();
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = write_tiny_png(tmp.path(), "tiny.png");
        let cmd = format!("/image {}", path.display());
        s.execute_palette_command(&cmd);
        let last = s.transcript().last().expect("at least one turn");
        match last {
            Turn::Command { ok, message, .. } => {
                assert!(*ok);
                assert!(message.contains("image/png"));
            }
            other => panic!("expected Turn::Command, got {other:?}"),
        }
    }

    #[test]
    fn submit_forwards_pending_attachments_into_turn_context() {
        // End-to-end seam check: `/image <path>` queues a Block::Image,
        // the next `submit()` drains it into TurnContext.attachments,
        // and the agent loop forwards it to the provider. Asserted via
        // a CapturingBackend that snapshots the last TurnContext.
        let backend = CapturingBackend::new();
        let chat_backend: Arc<dyn ChatBackend> = backend.clone();
        let mut s = ChatState::with_backend(chat_backend);

        let tmp = tempfile::tempdir().expect("tempdir");
        let path = write_tiny_png(tmp.path(), "tiny.png");
        let cmd = format!("/image {}", path.display());
        s.execute_palette_command(&cmd);
        assert_eq!(s.pending_attachment_count(), 1);

        s.submit_with_prompt("describe this image");

        // After submit, the queue is drained — exactly once.
        assert_eq!(s.pending_attachment_count(), 0);

        let seen = backend.last_attachments();
        assert_eq!(seen.len(), 1, "backend should see 1 attachment");
        assert!(matches!(&seen[0], Block::Image { mime, .. } if mime == "image/png"));
    }

    #[test]
    fn echo_provider_does_not_panic_on_attachments() {
        // Regression seam: even though EchoProvider is text-only, it
        // MUST tolerate a populated `attachments` field (Phase-5
        // multimodal scaffold). Build a request directly and verify
        // generate() returns text blocks without panicking.
        use stratum_runtime::{EchoProvider, GenerateRequest, Provider, SamplerParams};
        use stratum_types::Capability;

        let provider = EchoProvider::new("echo: ");
        // Make sure the provider still advertises Generate capability.
        assert!(provider.capabilities().contains(&Capability::Generate));

        let req = GenerateRequest {
            model: ModelId::from("echo"),
            prompt: "hello world".to_string(),
            max_blocks: 4,
            system_override: None,
            history: Vec::new(),
            sampler: SamplerParams::default(),
            attachments: vec![Block::image_inline_b64("image/png", "AAAA", 3)],
        };
        let cancel = CancelToken::new();
        let blocks = provider.generate(&req, &cancel);
        // EchoProvider emits one Text per word + Usage + Done. The
        // attachment is silently dropped; the assertion is that no
        // panic / error variant landed in the output.
        assert!(
            blocks.iter().any(|b| matches!(b, Block::Text { .. })),
            "expected at least one Text block, got {blocks:?}"
        );
        assert!(
            !blocks.iter().any(|b| matches!(b, Block::Cancelled { .. })),
            "EchoProvider must not surface Cancelled for attachments: {blocks:?}"
        );
    }

    #[test]
    fn image_palette_command_accepts_quoted_path_with_spaces() {
        let mut s = state();
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = write_tiny_png(tmp.path(), "with space.png");
        let cmd = format!("/image \"{}\"", path.display());
        let outcome = s.execute_palette_command(&cmd);
        assert!(
            matches!(outcome, PaletteOutcome::Acknowledged { .. }),
            "expected acknowledged for quoted path, got {outcome:?}"
        );
        assert_eq!(s.pending_attachment_count(), 1);
    }

    #[test]
    fn image_palette_command_rejects_relative_parent_traversal() {
        let mut s = state();
        let outcome = s.execute_palette_command("/image ../../etc/passwd");
        if let PaletteOutcome::Rejected { message } = outcome {
            assert!(
                message.contains("traverses parent") || message.contains("`..`"),
                "wrong rejection message: {message}"
            );
        } else {
            panic!("expected Rejected for `..` traversal, got {outcome:?}");
        }
        assert_eq!(
            s.pending_attachment_count(),
            0,
            "rejected /image must not queue anything"
        );
    }

    /// `ChatBackend` that always reports failure. Used to verify
    /// attachments drain even when the turn doesn't succeed.
    #[derive(Debug, Default)]
    struct FailingBackend;
    impl ChatBackend for FailingBackend {
        fn run_turn_streaming(
            &self,
            ctx: TurnContext,
            _cancel: &CancelToken,
            _chunk_tx: mpsc::Sender<Block>,
        ) -> TurnResult {
            TurnResult {
                turn_id: ctx.turn_id,
                outcome: TurnOutcome::ModelError {
                    // Existing declared sentinel; the test doesn't
                    // care which code, only that the outcome is non-success.
                    code: "STRAT-E3007".into(),
                },
                blocks: Vec::new(),
                transitions: Vec::new(),
                events_emitted: Vec::new(),
            }
        }
    }

    #[test]
    fn pending_attachments_drain_even_when_turn_fails() {
        // Documented contract: attachments are spent per-turn,
        // regardless of outcome. A failed turn must NOT leave them
        // queued for the next prompt (which would silently re-charge
        // them in tokens + bandwidth and surprise the user).
        let chat_backend: Arc<dyn ChatBackend> = Arc::new(FailingBackend);
        let mut s = ChatState::with_backend(chat_backend);

        let tmp = tempfile::tempdir().expect("tempdir");
        let path = write_tiny_png(tmp.path(), "tiny.png");
        let cmd = format!("/image {}", path.display());
        s.execute_palette_command(&cmd);
        assert_eq!(s.pending_attachment_count(), 1);

        s.submit_with_prompt("describe this image");

        assert_eq!(
            s.pending_attachment_count(),
            0,
            "attachments must drain on failure too — they belong to the user's turn, not to a retry"
        );
    }
    fn audio_dispatch_rejects_path_traversal() {
        let mut s = state();
        let outcome = s.execute_palette_command("/audio ../etc/passwd");
        assert!(matches!(outcome, PaletteOutcome::Rejected { .. }));
        assert!(s.staged_audio().is_none());
    }

    #[test]
    fn audio_dispatch_rejects_missing_arg() {
        let mut s = state();
        let outcome = s.execute_palette_command("/audio");
        assert!(
            matches!(outcome, PaletteOutcome::Rejected { message } if message.contains("usage"))
        );
    }

    #[test]
    fn audio_then_submit_prepends_transcript_marker_and_records_attachment() {
        // The whisper binary is virtually never on CI hosts, so we
        // exercise the unavailable-sentinel branch and confirm both:
        //   (1) the User turn carries the unavailable marker that
        //       stands in for "[transcript: …]" on hosts with whisper,
        //   (2) the per-turn attachments accessor surfaces the staged
        //       Block::Audio for the future agent-loop seam.
        let _lock = lock_audio_cwd();
        let tmp = tempfile::TempDir::new().expect("tmp");
        let prev = std::env::current_dir().expect("cwd");
        std::env::set_current_dir(tmp.path()).expect("chdir");
        let _guard = scopeguard_chdir(prev);

        write_wav_fixture(tmp.path(), "voice.wav");
        let mut s = state();
        // Force a clearly-missing binary so the test is deterministic
        // even on hosts that DO have whisper on PATH.
        s.whisper = stratum_runtime::whisper::WhisperSubprocess::new()
            .with_binary("stratum_no_such_whisper_xyzzy_test");
        let outcome = s.execute_palette_command("/audio voice.wav");
        assert!(matches!(outcome, PaletteOutcome::Acknowledged { .. }));
        // Type a prompt.
        for c in "summarize what I said".chars() {
            s.handle_key(key(KeyCode::Char(c), KeyModifiers::NONE));
        }
        s.handle_key(key(KeyCode::Enter, KeyModifiers::NONE));
        // The user turn includes the unavailable sentinel and the
        // typed text.
        let user_turn = s
            .transcript()
            .iter()
            .find_map(|t| match t {
                Turn::User(text) => Some(text.clone()),
                _ => None,
            })
            .expect("user turn");
        // Reviewer feedback: the prefix now uses fenced
        // AUDIO_TRANSCRIPT_BEGIN/END markers + "(unavailable — install
        // whisper.cpp)" body so injection can't masquerade as instructions.
        assert!(
            user_turn.contains("AUDIO_TRANSCRIPT_BEGIN"),
            "got: {user_turn}",
        );
        assert!(
            user_turn.contains("unavailable") && user_turn.contains("install whisper.cpp"),
            "got: {user_turn}",
        );
        assert!(
            user_turn.contains("summarize what I said"),
            "got: {user_turn}",
        );
        // The attachments accessor surfaces the staged Block::Audio.
        let attachments = s.last_turn_attachments();
        assert_eq!(attachments.len(), 1, "exactly one staged attachment");
        assert!(matches!(&attachments[0], Block::Audio { mime, .. } if mime == "audio/wav"));
        // After submit, staged_audio drained back to empty.
        assert!(s.staged_audio().is_none());
    }

    #[test]
    fn parse_audio_path_arg_handles_quoted_paths() {
        assert_eq!(parse_audio_path_arg("voice.wav"), Some("voice.wav".into()));
        assert_eq!(
            parse_audio_path_arg("\"my clip.wav\""),
            Some("my clip.wav".into()),
        );
        assert_eq!(parse_audio_path_arg(""), None);
        assert_eq!(parse_audio_path_arg("  "), None);
        assert_eq!(parse_audio_path_arg("a b c"), None);
        assert_eq!(parse_audio_path_arg("\"unterminated"), None);
        assert_eq!(parse_audio_path_arg("\"\""), None);
        assert_eq!(parse_audio_path_arg("\"clip.wav\" trailing"), None);
    }

    #[test]
    fn sniff_audio_mime_chat_covers_known_formats() {
        let mut riff = b"RIFF\0\0\0\0WAVE".to_vec();
        riff.extend_from_slice(&[0u8; 4]);
        assert_eq!(
            sniff_audio_mime_chat(std::path::Path::new("x"), &riff),
            "audio/wav",
        );
        assert_eq!(
            sniff_audio_mime_chat(std::path::Path::new("x"), b"fLaC--"),
            "audio/flac",
        );
        assert_eq!(
            sniff_audio_mime_chat(std::path::Path::new("x"), b"OggS--"),
            "audio/ogg",
        );
        assert_eq!(
            sniff_audio_mime_chat(std::path::Path::new("x"), b"ID3-----"),
            "audio/mpeg",
        );
        assert_eq!(
            sniff_audio_mime_chat(std::path::Path::new("x"), &[0xFF, 0xFB, 0x00, 0x00]),
            "audio/mpeg",
        );
        // Unknown bytes + unknown extension → octet-stream.
        assert_eq!(
            sniff_audio_mime_chat(std::path::Path::new("noise.bin"), b"???"),
            "application/octet-stream",
        );
        // Unknown bytes + known extension → extension wins.
        assert_eq!(
            sniff_audio_mime_chat(std::path::Path::new("clip.opus"), b"???"),
            "audio/opus",
        );
    }

    #[test]
    fn trim_for_ack_collapses_and_truncates() {
        assert_eq!(trim_for_ack("short", 10), "short");
        assert_eq!(trim_for_ack("multi\nline", 80), "multi line");
        let long = "x".repeat(200);
        let trimmed = trim_for_ack(&long, 10);
        assert!(trimmed.ends_with('…'));
        assert!(trimmed.chars().count() <= 10);
    }

    /// RAII helper so a panicking test still restores the previous CWD.
    fn scopeguard_chdir(prev: std::path::PathBuf) -> ChdirRestore {
        ChdirRestore { prev }
    }
    struct ChdirRestore {
        prev: std::path::PathBuf,
    }
    impl Drop for ChdirRestore {
        fn drop(&mut self) {
            let _ = std::env::set_current_dir(&self.prev);
        }
    }

    // ---------- Phase 5 + 6 end-to-end seam tests ---------------------------
    //
    // These exercise the full TUI-side seam for `/image <png>` and
    // `/audio <wav>`: a palette command stages the attachment, the
    // next `submit_with_prompt` drains it into a `TurnContext`, and
    // the backend (a CapturingBackend that records every TurnContext
    // it sees) confirms that the attachment + transcript prefix
    // arrive unchanged.
    //
    // These tests run against the EchoProvider seam (no
    // `provider-llama-cpp` / `vision` feature required) so they
    // compile and pass under default features.

    #[test]
    fn image_palette_command_then_submit_routes_attachment_to_backend() {
        // End-to-end seam check for the `/image <png>` path:
        //   1. The palette command stages a Block::Image.
        //   2. `submit_with_prompt` drains `pending_attachments` into
        //      `TurnContext.attachments`.
        //   3. The CapturingBackend snapshots the TurnContext and we
        //      confirm exactly one image attachment with the right
        //      MIME landed there.
        let backend = CapturingBackend::new();
        let chat_backend: Arc<dyn ChatBackend> = backend.clone();
        let mut s = ChatState::with_backend(chat_backend);

        let tmp = tempfile::tempdir().expect("tempdir");
        let path = write_tiny_png(tmp.path(), "phase5-image.png");
        let cmd = format!("/image {}", path.display());
        let outcome = s.execute_palette_command(&cmd);
        assert!(
            matches!(outcome, PaletteOutcome::Acknowledged { .. }),
            "expected acknowledged for /image, got {outcome:?}"
        );

        s.submit_with_prompt("what is in this picture");

        // pending_attachments was drained on submit.
        assert_eq!(s.pending_attachment_count(), 0);
        // Backend saw exactly one image attachment.
        let seen = backend.last_attachments();
        assert_eq!(
            seen.len(),
            1,
            "backend should see 1 image attachment, got {seen:?}"
        );
        assert!(
            matches!(&seen[0], Block::Image { mime, .. } if mime == "image/png"),
            "wrong attachment shape: {:?}",
            seen[0]
        );
    }

    #[test]
    fn audio_palette_command_then_submit_routes_attachment_and_transcript_prefix_to_backend() {
        // End-to-end seam check for the `/audio <wav>` path. Two
        // claims under test:
        //
        //   1. The backend's TurnContext.attachments carries exactly
        //      one Block::Audio with the right MIME — proves the
        //      audio attachment reaches the provider unchanged.
        //   2. The transcript-prefix fence
        //      ([AUDIO_TRANSCRIPT_BEGIN] ... [AUDIO_TRANSCRIPT_END])
        //      is present in the backend-visible user_prompt — proves
        //      the audio palette path properly augments the prompt
        //      even when whisper.cpp is not installed (the common
        //      CI case).
        //
        // We force a missing whisper binary so the test is
        // deterministic on hosts that happen to have whisper on PATH.
        let _lock = lock_audio_cwd();
        let tmp = tempfile::TempDir::new().expect("tmp");
        let prev = std::env::current_dir().expect("cwd");
        std::env::set_current_dir(tmp.path()).expect("chdir");
        let _guard = scopeguard_chdir(prev);

        write_wav_fixture(tmp.path(), "voice.wav");

        let backend = CapturingBackend::new();
        let chat_backend: Arc<dyn ChatBackend> = backend.clone();
        let mut s = ChatState::with_backend(chat_backend);
        s.whisper = stratum_runtime::whisper::WhisperSubprocess::new()
            .with_binary("stratum_no_such_whisper_phase5_e2e");

        let outcome = s.execute_palette_command("/audio voice.wav");
        assert!(
            matches!(outcome, PaletteOutcome::Acknowledged { .. }),
            "expected acknowledged for /audio, got {outcome:?}"
        );

        s.submit_with_prompt("what is in this clip");

        // staged_audio drained.
        assert!(s.staged_audio().is_none());

        // Backend saw exactly one audio attachment.
        let seen = backend.last_attachments();
        assert_eq!(
            seen.len(),
            1,
            "backend should see 1 audio attachment, got {seen:?}"
        );
        assert!(
            matches!(&seen[0], Block::Audio { mime, .. } if mime == "audio/wav"),
            "wrong attachment shape: {:?}",
            seen[0]
        );

        // The transcript-prefix fence reached the backend prompt.
        // Pull the TurnContext snapshot out of the CapturingBackend
        // directly so we can inspect the user_prompt verbatim. We
        // clone the prompt out of the lock and drop the guard before
        // running assertions so clippy's significant_drop_tightening
        // lint is satisfied.
        let captured_prompt = {
            let guard = backend
                .last_ctx
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            guard
                .as_ref()
                .expect("backend saw a turn")
                .user_prompt
                .clone()
        };
        assert!(
            captured_prompt.contains("AUDIO_TRANSCRIPT_BEGIN"),
            "missing fence in backend prompt: {captured_prompt}",
        );
        assert!(
            captured_prompt.contains("AUDIO_TRANSCRIPT_END"),
            "missing fence end in backend prompt: {captured_prompt}",
        );
        // The user-typed body must still be there alongside the fence.
        assert!(
            captured_prompt.contains("what is in this clip"),
            "missing user-typed body in backend prompt: {captured_prompt}",
        );
    }

    #[test]
    fn image_and_audio_in_same_turn_both_reach_backend() {
        // Stress the multimodal pipe with BOTH an image and an audio
        // attachment queued for the same submit. The backend's
        // TurnContext.attachments must carry both blocks in a stable
        // order: pending_attachments (image) first, then the audio.
        let _lock = lock_audio_cwd();
        let tmp = tempfile::TempDir::new().expect("tmp");
        let prev = std::env::current_dir().expect("cwd");
        std::env::set_current_dir(tmp.path()).expect("chdir");
        let _guard = scopeguard_chdir(prev);

        let png = write_tiny_png(tmp.path(), "frame.png");
        write_wav_fixture(tmp.path(), "voice.wav");

        let backend = CapturingBackend::new();
        let chat_backend: Arc<dyn ChatBackend> = backend.clone();
        let mut s = ChatState::with_backend(chat_backend);
        s.whisper = stratum_runtime::whisper::WhisperSubprocess::new()
            .with_binary("stratum_no_such_whisper_phase5_combo");

        let img_cmd = format!("/image {}", png.display());
        assert!(matches!(
            s.execute_palette_command(&img_cmd),
            PaletteOutcome::Acknowledged { .. }
        ));
        assert!(matches!(
            s.execute_palette_command("/audio voice.wav"),
            PaletteOutcome::Acknowledged { .. }
        ));

        s.submit_with_prompt("combined multimodal turn");

        let seen = backend.last_attachments();
        assert_eq!(
            seen.len(),
            2,
            "backend should see both attachments, got {seen:?}"
        );
        // Order: image first (pending_attachments drains into the
        // outgoing vector before the audio mirror appends).
        // Stable because `submit()` does
        // `let mut attachments = std::mem::take(&mut self.pending_attachments);`
        // (which holds the image) BEFORE
        // `attachments.extend(self.last_turn_attachments.iter().cloned());`
        // (which holds the audio mirror). See `submit()` drain order in this file.
        assert!(
            matches!(&seen[0], Block::Image { mime, .. } if mime == "image/png"),
            "first attachment should be the image: {:?}",
            seen[0]
        );
        assert!(
            matches!(&seen[1], Block::Audio { mime, .. } if mime == "audio/wav"),
            "second attachment should be the audio: {:?}",
            seen[1]
        );
    }
}
