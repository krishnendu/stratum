//! `AgentSession` — high-level wrapper composing [`AgentLoop`],
//! [`TranscriptStore`], and [`EventEmitter`] into a single
//! `next_turn(prompt) -> TurnResult` surface.
//!
//! The session owns the conversation lifecycle: per-turn id allocation,
//! transcript appends and atomic saves, and a shared [`CancelToken`] so
//! a caller (the TUI palette, a test harness, the future `serve` daemon)
//! can stop the in-flight turn without poking the loop directly.
//!
//! # Lifecycle
//!
//! 1. [`AgentSession::open`] loads the on-disk transcript when one exists,
//!    otherwise starts a fresh [`Transcript`] and writes it atomically so
//!    the session file exists from the first call.
//! 2. [`AgentSession::next_turn`] increments the turn counter, drives one
//!    [`AgentLoop::run_turn`], appends `User` then `Assistant` turns to
//!    the in-memory transcript, and persists.
//! 3. [`AgentSession::push_system_message`] adds a synthetic `System`
//!    turn — useful for plan-mode toggles, command outcomes, etc.
//! 4. [`AgentSession::close`] writes the final transcript and consumes
//!    the session.
//!
//! # Cancellation
//!
//! The session owns a single [`CancelToken`]. Once
//! [`AgentSession::cancel`] is fired, every subsequent
//! [`AgentSession::next_turn`] short-circuits to
//! [`SessionError::Cancelled`] — the cancel is sticky by design (matches
//! `CancelToken` semantics).
//!
//! # Errors
//!
//! All transcript / store failures bubble up as [`SessionError::Store`].
//! Provider-level failures inside `run_turn` are *not* re-thrown — they
//! remain inside [`TurnResult::outcome`] so the caller can show them in
//! the UI without an error path.

use std::error::Error;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

use stratum_types::{Block, ModelId};

use crate::agent_loop::{AgentLoop, TurnContext, TurnResult};
use crate::cancel::CancelToken;
use crate::conversation::TurnOutcome;
use crate::event_log::{EventEmitter, EventRecord, MemoryEventSink};
use crate::observability::TurnId;
use crate::transcript::{
    SessionId, Transcript, TranscriptBlock, TranscriptBlockKind, TranscriptError, TranscriptStore,
    TranscriptTurn, TRANSCRIPT_SCHEMA_VERSION,
};

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors surfaced by [`AgentSession`].
#[derive(Debug)]
pub enum SessionError {
    /// The transcript store failed an I/O or serialization step.
    Store(TranscriptError),
    /// The session was cancelled before (or during) the call.
    Cancelled,
    /// A provider returned an error code that the session promotes to a
    /// hard error. Currently unused by the scaffold — reserved for the
    /// future "fail-fast on provider error" wiring.
    Provider(String),
}

impl fmt::Display for SessionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Store(e) => write!(f, "agent session store error: {e}"),
            Self::Cancelled => f.write_str("agent session cancelled"),
            Self::Provider(msg) => write!(f, "agent session provider error: {msg}"),
        }
    }
}

impl Error for SessionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Store(e) => Some(e),
            Self::Cancelled | Self::Provider(_) => None,
        }
    }
}

impl From<TranscriptError> for SessionError {
    fn from(e: TranscriptError) -> Self {
        Self::Store(e)
    }
}

// ---------------------------------------------------------------------------
// AgentSession
// ---------------------------------------------------------------------------

/// Composed agent + transcript + event session.
///
/// See the module docs for the lifecycle.
pub struct AgentSession {
    id: SessionId,
    loop_: Arc<AgentLoop>,
    store: Arc<TranscriptStore>,
    #[allow(
        dead_code,
        reason = "events emitter is wired for future per-turn event flush; held to keep the Arc alive"
    )]
    events: Arc<EventEmitter>,
    /// Optional handle to a [`MemoryEventSink`] backing `events`. When
    /// present, [`AgentSession::events_snapshot`] yields a clone of the
    /// recorded events; otherwise it returns `None`.
    memory_sink: Option<Arc<MemoryEventSink>>,
    transcript: Mutex<Transcript>,
    model: ModelId,
    next_turn_id: AtomicU64,
    cancel: CancelToken,
}

impl fmt::Debug for AgentSession {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AgentSession")
            .field("id", &self.id)
            .field("model", &self.model)
            .field("next_turn_id", &self.next_turn_id)
            .field("has_memory_sink", &self.memory_sink.is_some())
            .finish_non_exhaustive()
    }
}

impl AgentSession {
    /// Open (or resume) a session.
    ///
    /// If a transcript file already exists for `id`, it is loaded via
    /// [`TranscriptStore::load`]. Otherwise a fresh [`Transcript`] is
    /// created and written atomically so the on-disk file exists from
    /// the start.
    ///
    /// # Errors
    ///
    /// Returns [`SessionError::Store`] on any I/O or serialization
    /// failure from the underlying [`TranscriptStore`].
    pub fn open(
        id: SessionId,
        loop_: Arc<AgentLoop>,
        store: Arc<TranscriptStore>,
        events: Arc<EventEmitter>,
        model: ModelId,
    ) -> Result<Self, SessionError> {
        let transcript = match store.load(&id) {
            Ok(t) => t,
            Err(TranscriptError::Io(e)) if e.kind() == std::io::ErrorKind::NotFound => Transcript {
                schema_version: TRANSCRIPT_SCHEMA_VERSION,
                session_id: id.clone(),
                created_at: SystemTime::now(),
                turns: Vec::new(),
            },
            Err(other) => return Err(SessionError::Store(other)),
        };
        // Save atomically so the session file exists from the start.
        store.save_atomic(&transcript)?;

        Ok(Self {
            id,
            loop_,
            store,
            events,
            memory_sink: None,
            transcript: Mutex::new(transcript),
            model,
            next_turn_id: AtomicU64::new(0),
            cancel: CancelToken::new(),
        })
    }

    /// Attach an in-memory event-sink handle so
    /// [`AgentSession::events_snapshot`] returns `Some`. The caller is
    /// expected to have built the [`EventEmitter`] handed to
    /// [`AgentSession::open`] from the same sink.
    #[must_use]
    pub fn with_memory_sink(mut self, sink: Arc<MemoryEventSink>) -> Self {
        self.memory_sink = Some(sink);
        self
    }

    /// Borrow the session id.
    #[must_use]
    pub const fn id(&self) -> &SessionId {
        &self.id
    }

    /// Run one user → assistant turn end-to-end.
    ///
    /// Increments the per-turn counter, builds a [`TurnContext`], drives
    /// [`AgentLoop::run_turn`], appends a `User` + `Assistant` pair to
    /// the transcript, and persists atomically.
    ///
    /// # Errors
    ///
    /// * [`SessionError::Cancelled`] if the session's [`CancelToken`]
    ///   has been fired before the call. (Mid-turn cancels surface
    ///   inside [`TurnResult::outcome`] as [`TurnOutcome::UserAbort`]
    ///   — the cancel is sticky, so subsequent calls then *do* hit
    ///   this error path.)
    /// * [`SessionError::Store`] on any transcript persistence failure.
    #[allow(
        clippy::significant_drop_tightening,
        reason = "guard must span save_atomic to preserve write ordering"
    )]
    pub fn next_turn(&self, prompt: &str) -> Result<TurnResult, SessionError> {
        if self.cancel.is_cancelled() {
            return Err(SessionError::Cancelled);
        }

        let next = self.next_turn_id.fetch_add(1, Ordering::SeqCst) + 1;
        let started_at = SystemTime::now();

        let ctx = TurnContext {
            user_prompt: prompt.to_string(),
            model: self.model.clone(),
            turn_id: TurnId(next),
            started_at,
        };

        let result = self.loop_.run_turn(ctx, &self.cancel);

        // Append User and Assistant turns. We always append both, even
        // when the assistant's blocks are empty — the transcript should
        // record that the turn happened.
        let user_turn = TranscriptTurn::User {
            at: started_at,
            text: prompt.to_string(),
        };
        let assistant_turn = TranscriptTurn::Assistant {
            at: SystemTime::now(),
            blocks: blocks_to_transcript_blocks(&result.blocks),
        };

        // Hold the lock across the atomic save so concurrent callers
        // serialize their writes — the store path is intentionally not
        // racy from the session's perspective.
        {
            let mut guard = self
                .transcript
                .lock()
                .map_err(|_| transcript_lock_poisoned())?;
            guard.turns.push(user_turn);
            guard.turns.push(assistant_turn);
            self.store.save_atomic(&guard)?;
        }

        Ok(result)
    }

    /// Append a synthetic `System` turn and persist.
    ///
    /// # Errors
    ///
    /// Returns [`SessionError::Store`] on transcript persistence failure.
    #[allow(
        clippy::significant_drop_tightening,
        reason = "guard must span save_atomic to preserve write ordering"
    )]
    pub fn push_system_message(&self, text: &str) -> Result<(), SessionError> {
        let turn = TranscriptTurn::System {
            at: SystemTime::now(),
            text: text.to_string(),
        };
        let mut guard = self
            .transcript
            .lock()
            .map_err(|_| transcript_lock_poisoned())?;
        guard.turns.push(turn);
        self.store.save_atomic(&guard)?;
        Ok(())
    }

    /// Fire the session's cancel token. Idempotent.
    ///
    /// The cancel is sticky — every subsequent [`AgentSession::next_turn`]
    /// returns [`SessionError::Cancelled`] without invoking the loop.
    pub fn cancel(&self) {
        self.cancel.cancel();
    }

    /// Clone of the current transcript.
    #[must_use]
    pub fn transcript_snapshot(&self) -> Transcript {
        self.transcript
            .lock()
            .map_or_else(|poisoned| poisoned.into_inner().clone(), |g| g.clone())
    }

    /// Snapshot of the underlying event sink — only available when the
    /// session was built [`with_memory_sink`].
    ///
    /// [`with_memory_sink`]: AgentSession::with_memory_sink
    #[must_use]
    pub fn events_snapshot(&self) -> Option<Vec<EventRecord>> {
        self.memory_sink.as_ref().map(|s| s.snapshot())
    }

    /// Persist one final transcript and consume the session.
    ///
    /// # Errors
    ///
    /// Returns [`SessionError::Store`] on transcript persistence failure.
    pub fn close(self) -> Result<(), SessionError> {
        let final_transcript = self
            .transcript
            .lock()
            .map_or_else(|p| p.into_inner().clone(), |g| g.clone());
        self.store.save_atomic(&final_transcript)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn transcript_lock_poisoned() -> TranscriptError {
    TranscriptError::Validation("agent session transcript mutex poisoned".to_string())
}

/// Convert provider [`Block`]s to renderable [`TranscriptBlock`]s.
///
/// `Block::Text` → `Text`; `Block::ToolCall` → `ToolCall { tool_id }`
/// carrying the JSON args blob as text. Non-rendering variants
/// (`Usage`, `Done`, `Cancelled`, `ToolResult`) are dropped — they are
/// observability concerns, not transcript content.
fn blocks_to_transcript_blocks(blocks: &[Block]) -> Vec<TranscriptBlock> {
    let mut out = Vec::with_capacity(blocks.len());
    for b in blocks {
        match b {
            Block::Text { text } => out.push(TranscriptBlock {
                kind: TranscriptBlockKind::Text,
                text: text.clone(),
            }),
            Block::ToolCall { tool, args, .. } => out.push(TranscriptBlock {
                kind: TranscriptBlockKind::ToolCall {
                    tool_id: tool.clone(),
                },
                text: args.clone(),
            }),
            Block::ToolResult { .. }
            | Block::Usage { .. }
            | Block::Done
            | Block::Cancelled { .. } => {
                // intentionally not surfaced in the transcript
            }
        }
    }
    out
}

// Avoid unused-imports warning on TurnOutcome — it's part of the public
// API surface re-exported by the loop, and is referenced in the rustdoc
// above. Keep a no-op reference so future refactors don't drop the use.
#[doc(hidden)]
#[allow(dead_code)]
const _ASSERT_TURN_OUTCOME_USED: fn() -> Option<TurnOutcome> = || None;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::agent_factory::AgentFactory;
    use crate::event_log::{EventEmitter, EventSink, JsonlEventSink, MemoryEventSink};
    use crate::transcript::{SessionId, TranscriptStore, TranscriptTurn};
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;
    use tempfile::TempDir;

    fn fresh_store(tmp: &TempDir) -> Arc<TranscriptStore> {
        Arc::new(TranscriptStore::open(tmp.path().to_path_buf()).unwrap())
    }

    fn fresh_loop() -> Arc<AgentLoop> {
        Arc::new(AgentFactory::echo().unwrap())
    }

    fn fresh_emitter() -> (Arc<EventEmitter>, Arc<MemoryEventSink>) {
        let sink = Arc::new(MemoryEventSink::new());
        let emitter = Arc::new(EventEmitter::new(sink.clone()));
        (emitter, sink)
    }

    fn fresh_session() -> (AgentSession, TempDir, Arc<MemoryEventSink>) {
        let tmp = TempDir::new().unwrap();
        let store = fresh_store(&tmp);
        let loop_ = fresh_loop();
        let (emitter, sink) = fresh_emitter();
        let id = SessionId::new_random();
        let session = AgentSession::open(id, loop_, store, emitter, ModelId::from("echo"))
            .unwrap()
            .with_memory_sink(sink.clone());
        (session, tmp, sink)
    }

    // ---------- open() ----------------------------------------------------

    #[test]
    fn open_writes_fresh_transcript_when_no_file_exists() {
        let tmp = TempDir::new().unwrap();
        let store = fresh_store(&tmp);
        let loop_ = fresh_loop();
        let (emitter, _sink) = fresh_emitter();
        let id = SessionId::from_str("0123456789abcdef").unwrap();
        let session = AgentSession::open(
            id.clone(),
            loop_,
            store.clone(),
            emitter,
            ModelId::from("echo"),
        )
        .unwrap();
        // File must exist on disk after open.
        let expected = tmp.path().join("0123456789abcdef.json");
        assert!(expected.exists(), "session file should exist after open");
        // In-memory transcript is empty.
        assert!(session.transcript_snapshot().turns.is_empty());
        // Disk also round-trips.
        let loaded = store.load(&id).unwrap();
        assert!(loaded.turns.is_empty());
    }

    #[test]
    fn open_loads_existing_transcript_when_file_exists() {
        let tmp = TempDir::new().unwrap();
        let store = fresh_store(&tmp);
        let id = SessionId::from_str("aaaaaaaabbbbbbbb").unwrap();
        // Pre-seed a transcript with one User turn.
        let seeded = Transcript {
            schema_version: TRANSCRIPT_SCHEMA_VERSION,
            session_id: id.clone(),
            created_at: SystemTime::now(),
            turns: vec![TranscriptTurn::User {
                at: SystemTime::now(),
                text: "before-restart".into(),
            }],
        };
        store.save_atomic(&seeded).unwrap();

        let loop_ = fresh_loop();
        let (emitter, _sink) = fresh_emitter();
        let session = AgentSession::open(id, loop_, store, emitter, ModelId::from("echo")).unwrap();
        let snap = session.transcript_snapshot();
        assert_eq!(snap.turns.len(), 1);
        match &snap.turns[0] {
            TranscriptTurn::User { text, .. } => assert_eq!(text, "before-restart"),
            other => panic!("expected User turn, got {other:?}"),
        }
    }

    // ---------- next_turn -------------------------------------------------

    #[test]
    fn next_turn_appends_user_then_assistant() {
        let (session, _tmp, _sink) = fresh_session();
        let _ = session.next_turn("hello").unwrap();
        let snap = session.transcript_snapshot();
        assert_eq!(snap.turns.len(), 2);
        assert!(matches!(snap.turns[0], TranscriptTurn::User { .. }));
        assert!(matches!(snap.turns[1], TranscriptTurn::Assistant { .. }));
    }

    #[test]
    fn next_turn_user_text_matches_prompt() {
        let (session, _tmp, _sink) = fresh_session();
        let _ = session.next_turn("the prompt text").unwrap();
        let snap = session.transcript_snapshot();
        match &snap.turns[0] {
            TranscriptTurn::User { text, .. } => assert_eq!(text, "the prompt text"),
            other => panic!("unexpected first turn: {other:?}"),
        }
    }

    #[test]
    fn next_turn_increments_turn_id() {
        let (session, _tmp, _sink) = fresh_session();
        let r1 = session.next_turn("a").unwrap();
        let r2 = session.next_turn("b").unwrap();
        let r3 = session.next_turn("c").unwrap();
        assert_eq!(r1.turn_id, TurnId(1));
        assert_eq!(r2.turn_id, TurnId(2));
        assert_eq!(r3.turn_id, TurnId(3));
    }

    #[test]
    fn next_turn_with_echo_provider_succeeds() {
        let (session, _tmp, _sink) = fresh_session();
        let r = session.next_turn("hello").unwrap();
        assert!(matches!(r.outcome, TurnOutcome::Success));
    }

    #[test]
    fn next_turn_saves_atomically_file_updates() {
        let tmp = TempDir::new().unwrap();
        let store = fresh_store(&tmp);
        let loop_ = fresh_loop();
        let (emitter, _sink) = fresh_emitter();
        let id = SessionId::from_str("0123456789abcdef").unwrap();
        let session = AgentSession::open(
            id.clone(),
            loop_,
            store.clone(),
            emitter,
            ModelId::from("echo"),
        )
        .unwrap();
        let path = tmp.path().join("0123456789abcdef.json");
        let before_len = std::fs::metadata(&path).unwrap().len();
        let _ = session.next_turn("hello").unwrap();
        let after_len = std::fs::metadata(&path).unwrap().len();
        assert!(
            after_len > before_len,
            "file len should grow after a turn (before={before_len}, after={after_len})"
        );
        // And the on-disk file still deserializes.
        let loaded = store.load(&id).unwrap();
        assert_eq!(loaded.turns.len(), 2);
    }

    #[test]
    fn next_turn_after_cancel_returns_cancelled_error() {
        let (session, _tmp, _sink) = fresh_session();
        session.cancel();
        let err = session.next_turn("anything").unwrap_err();
        assert!(matches!(err, SessionError::Cancelled));
    }

    #[test]
    fn cancel_is_sticky_subsequent_calls_also_fail() {
        let (session, _tmp, _sink) = fresh_session();
        session.cancel();
        for _ in 0..3 {
            let err = session.next_turn("x").unwrap_err();
            assert!(matches!(err, SessionError::Cancelled));
        }
    }

    // ---------- transcript_snapshot --------------------------------------

    #[test]
    fn transcript_snapshot_reflects_appended_turns() {
        let (session, _tmp, _sink) = fresh_session();
        assert!(session.transcript_snapshot().turns.is_empty());
        let _ = session.next_turn("one").unwrap();
        assert_eq!(session.transcript_snapshot().turns.len(), 2);
        let _ = session.next_turn("two").unwrap();
        assert_eq!(session.transcript_snapshot().turns.len(), 4);
    }

    #[test]
    fn transcript_snapshot_is_a_clone_modifying_does_not_affect_session() {
        let (session, _tmp, _sink) = fresh_session();
        let _ = session.next_turn("hello").unwrap();
        let mut snap = session.transcript_snapshot();
        snap.turns.clear();
        // Session's own state unchanged.
        assert_eq!(session.transcript_snapshot().turns.len(), 2);
    }

    // ---------- events_snapshot ------------------------------------------

    #[test]
    fn events_snapshot_returns_some_for_memory_backed_emitter() {
        let (session, _tmp, sink) = fresh_session();
        // The session's emitter is independent of the loop's internal
        // event emitter (the loop owns its own). Emit a marker event
        // directly into the sink so the snapshot has observable content.
        let (emitter, _other) = fresh_emitter();
        let _ = emitter; // (loop's own emitter is separate; we drive the sink directly)
        sink.write(crate::event_log::EventRecord {
            id: 1,
            at: SystemTime::UNIX_EPOCH,
            turn_id: None,
            event: crate::event_log::Event::AgentHandoff {
                from: "session".into(),
                to: "loop".into(),
                reason: "scaffold".into(),
            },
        });
        let _ = session.next_turn("hello").unwrap();
        let snap = session.events_snapshot();
        assert!(snap.is_some());
        let snap = snap.unwrap();
        assert!(
            !snap.is_empty(),
            "memory sink should expose the marker event"
        );
    }

    #[test]
    fn events_snapshot_returns_none_for_jsonl_backed_emitter() {
        let tmp = TempDir::new().unwrap();
        let store = fresh_store(&tmp);
        let loop_ = fresh_loop();
        let jsonl = JsonlEventSink::open(tmp.path().join("events.jsonl")).expect("jsonl open");
        let emitter = Arc::new(EventEmitter::new(Arc::new(jsonl)));
        let id = SessionId::new_random();
        let session = AgentSession::open(id, loop_, store, emitter, ModelId::from("echo")).unwrap();
        assert!(session.events_snapshot().is_none());
    }

    // ---------- push_system_message --------------------------------------

    #[test]
    fn push_system_message_appends_system_turn_and_persists() {
        let (session, tmp, _sink) = fresh_session();
        session.push_system_message("plan mode activated").unwrap();
        let snap = session.transcript_snapshot();
        assert_eq!(snap.turns.len(), 1);
        match &snap.turns[0] {
            TranscriptTurn::System { text, .. } => {
                assert_eq!(text, "plan mode activated");
            }
            other => panic!("expected System turn, got {other:?}"),
        }
        // Persistence: reload from disk.
        let store = TranscriptStore::open(tmp.path().to_path_buf()).unwrap();
        let loaded = store.load(session.id()).unwrap();
        assert_eq!(loaded.turns.len(), 1);
    }

    // ---------- id() -----------------------------------------------------

    #[test]
    fn id_returns_session_id() {
        let tmp = TempDir::new().unwrap();
        let store = fresh_store(&tmp);
        let loop_ = fresh_loop();
        let (emitter, _sink) = fresh_emitter();
        let id = SessionId::from_str("ffffffffeeeeeeee").unwrap();
        let session =
            AgentSession::open(id.clone(), loop_, store, emitter, ModelId::from("echo")).unwrap();
        assert_eq!(session.id(), &id);
    }

    // ---------- close ----------------------------------------------------

    #[test]
    fn close_persists_then_consumes() {
        let (session, tmp, _sink) = fresh_session();
        let _ = session.next_turn("hello").unwrap();
        let id = session.id().clone();
        session.close().unwrap();
        let store = TranscriptStore::open(tmp.path().to_path_buf()).unwrap();
        let loaded = store.load(&id).unwrap();
        assert_eq!(loaded.turns.len(), 2);
    }

    // ---------- SessionError::Display + From -----------------------------

    #[test]
    fn session_error_display_smoke_each_variant() {
        let store_err = SessionError::Store(TranscriptError::Validation("bad".into()));
        assert!(store_err.to_string().contains("store error"));
        assert!(store_err.source().is_some());

        let cancelled = SessionError::Cancelled;
        assert!(cancelled.to_string().contains("cancelled"));
        assert!(cancelled.source().is_none());

        let prov = SessionError::Provider("nope".into());
        assert!(prov.to_string().contains("provider"));
        assert!(prov.source().is_none());
    }

    #[test]
    fn session_error_from_transcript_error_works() {
        let err: SessionError = TranscriptError::Validation("oops".into()).into();
        assert!(matches!(err, SessionError::Store(_)));
    }

    // ---------- Send + Sync ----------------------------------------------

    #[test]
    fn agent_session_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<AgentSession>();
        assert_send_sync::<SessionError>();
    }

    // ---------- Concurrent next_turn -------------------------------------

    #[test]
    fn concurrent_next_turn_appends_all_turns_without_panic() {
        let (session, _tmp, _sink) = fresh_session();
        let session = Arc::new(session);
        let mut handles = Vec::new();
        for _ in 0..4 {
            let s = Arc::clone(&session);
            handles.push(thread::spawn(move || {
                for _ in 0..5 {
                    let _ = s.next_turn("hi").unwrap();
                }
            }));
        }
        for h in handles {
            h.join().expect("thread join");
        }
        let snap = session.transcript_snapshot();
        // 4 threads * 5 turns * 2 entries (User+Assistant) = 40
        assert_eq!(snap.turns.len(), 40);
    }

    // ---------- Round-trip: open + next_turn + close + reopen -----------

    #[test]
    fn round_trip_drop_and_reopen_preserves_transcript() {
        let tmp = TempDir::new().unwrap();
        let id = SessionId::new_random();
        {
            let store = fresh_store(&tmp);
            let loop_ = fresh_loop();
            let (emitter, _sink) = fresh_emitter();
            let session =
                AgentSession::open(id.clone(), loop_, store, emitter, ModelId::from("echo"))
                    .unwrap();
            let _ = session.next_turn("one").unwrap();
            let _ = session.next_turn("two").unwrap();
            let _ = session.next_turn("three").unwrap();
            // Drop without explicit close.
        }
        // Reopen with the same id.
        let store = fresh_store(&tmp);
        let loop_ = fresh_loop();
        let (emitter, _sink) = fresh_emitter();
        let reopened =
            AgentSession::open(id, loop_, store, emitter, ModelId::from("echo")).unwrap();
        let snap = reopened.transcript_snapshot();
        assert_eq!(snap.turns.len(), 6);
    }

    #[test]
    fn on_disk_file_after_close_round_trips_via_store_load() {
        let (session, tmp, _sink) = fresh_session();
        let id = session.id().clone();
        let _ = session.next_turn("hello").unwrap();
        session.close().unwrap();
        let store = TranscriptStore::open(tmp.path().to_path_buf()).unwrap();
        let loaded = store.load(&id).unwrap();
        assert_eq!(loaded.schema_version, TRANSCRIPT_SCHEMA_VERSION);
        assert_eq!(loaded.session_id, id);
        assert_eq!(loaded.turns.len(), 2);
    }

    // ---------- Debug ----------------------------------------------------

    #[test]
    fn agent_session_debug_smoke() {
        let (session, _tmp, _sink) = fresh_session();
        let rendered = format!("{session:?}");
        assert!(rendered.contains("AgentSession"));
        assert!(rendered.contains("model"));
    }

    // ---------- Tool-call rendering --------------------------------------

    #[test]
    fn assistant_block_text_is_recorded_in_transcript() {
        let (session, _tmp, _sink) = fresh_session();
        let _ = session.next_turn("hi").unwrap();
        let snap = session.transcript_snapshot();
        match &snap.turns[1] {
            TranscriptTurn::Assistant { blocks, .. } => {
                // Echo provider emits at least one text block.
                assert!(blocks
                    .iter()
                    .any(|b| matches!(b.kind, TranscriptBlockKind::Text)));
            }
            other => panic!("expected Assistant, got {other:?}"),
        }
    }

    #[test]
    fn blocks_to_transcript_blocks_filters_non_rendering_variants() {
        let blocks = vec![
            Block::Text { text: "a".into() },
            Block::Usage {
                prompt: 1,
                completion: 2,
            },
            Block::Done,
            Block::Cancelled {
                reason: "STRAT-Exxxx".into(),
            },
            Block::ToolResult {
                id: "t1".into(),
                output: "out".into(),
            },
            Block::ToolCall {
                id: "t2".into(),
                tool: "fs.read".into(),
                args: "{}".into(),
            },
        ];
        let out = blocks_to_transcript_blocks(&blocks);
        assert_eq!(out.len(), 2);
        assert!(matches!(out[0].kind, TranscriptBlockKind::Text));
        assert!(matches!(out[1].kind, TranscriptBlockKind::ToolCall { .. }));
    }

    // ---------- with_memory_sink default --------------------------------

    #[test]
    fn without_memory_sink_events_snapshot_is_none() {
        let tmp = TempDir::new().unwrap();
        let store = fresh_store(&tmp);
        let loop_ = fresh_loop();
        let (emitter, _sink) = fresh_emitter();
        let id = SessionId::new_random();
        let session = AgentSession::open(id, loop_, store, emitter, ModelId::from("echo")).unwrap();
        // We never called `with_memory_sink`, so even though the emitter
        // happens to be memory-backed the session can't expose it.
        assert!(session.events_snapshot().is_none());
    }

    // ---------- open propagates store errors -----------------------------

    #[test]
    fn open_propagates_unexpected_store_errors() {
        // Pre-populate a session file with a strictly-newer schema; load
        // must return SchemaNewer, which `open` should pass through as
        // SessionError::Store.
        let tmp = TempDir::new().unwrap();
        let id = SessionId::from_str("0123456789abcdef").unwrap();
        let raw = r#"{
            "schema_version": 9999,
            "session_id": "0123456789abcdef",
            "created_at": { "secs_since_epoch": 0, "nanos_since_epoch": 0 },
            "turns": []
        }"#;
        std::fs::write(tmp.path().join("0123456789abcdef.json"), raw).unwrap();
        let store = fresh_store(&tmp);
        let loop_ = fresh_loop();
        let (emitter, _sink) = fresh_emitter();
        let err = AgentSession::open(id, loop_, store, emitter, ModelId::from("echo"))
            .expect_err("schema-newer must surface");
        assert!(matches!(
            err,
            SessionError::Store(TranscriptError::SchemaNewer { .. })
        ));
    }

    // ---------- close vs drop persistence --------------------------------

    #[test]
    fn close_writes_even_with_no_new_turns() {
        // Open writes the initial empty transcript. We then close
        // without any turns — the file should still exist and load.
        let (session, tmp, _sink) = fresh_session();
        let id = session.id().clone();
        session.close().unwrap();
        let store = TranscriptStore::open(tmp.path().to_path_buf()).unwrap();
        let loaded = store.load(&id).unwrap();
        assert!(loaded.turns.is_empty());
    }

    // ---------- next_turn timing growth ----------------------------------

    #[test]
    fn many_turns_in_sequence_stay_consistent() {
        let (session, _tmp, _sink) = fresh_session();
        for _ in 0..6 {
            let _ = session.next_turn("ping").unwrap();
            // Brief pause so the wall-clock advances; not strictly
            // required but mirrors a real interactive loop.
            thread::sleep(Duration::from_millis(1));
        }
        let snap = session.transcript_snapshot();
        assert_eq!(snap.turns.len(), 12);
    }
}
