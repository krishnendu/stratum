// xtask-check-error-codes: ignore-file
//! Append-only structured event log.
//!
//! Data side of `plan/29-error-taxonomy-and-logging.md` §5 — complementary to
//! [`crate::observability`] which handles per-turn metrics. Where the recorder
//! aggregates token counts and role-step latencies, this log captures discrete
//! events the orchestrator needs to *audit*: tool invocations, permission
//! prompts, agent hand-offs, provider failures, sandbox launches.
//!
//! Pieces:
//!
//! * [`Event`] — tagged enum of the event kinds we currently care about.
//! * [`EventRecord`] — an [`Event`] wrapped with a monotonic id, timestamp,
//!   and optional [`TurnId`]-style turn correlation.
//! * [`EventSink`] — the write side. Two implementations ship: an in-memory
//!   [`MemoryEventSink`] for tests and a line-delimited
//!   [`JsonlEventSink`] that lands records on disk in append-only mode.
//! * [`EventEmitter`] — assigns ids, stamps `at` from an injected
//!   [`EventClock`], and hands records to a sink.
//!
//! The log is intentionally schema-light: every persisted event survives a
//! serde round-trip, so downstream consumers (the CLI's `events` subcommand,
//! the crash bundler, future tooling) can read the file without taking a
//! source dependency on this crate.

use std::error::Error;
use std::fmt;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

use serde::{Deserialize, Serialize};

/// One discrete event recorded by the orchestrator.
///
/// The `kind` tag in the serialized form lets consumers route on event type
/// without ever instantiating a typed variant — see the `kind: "tool_call"`
/// invariant exercised in the tests.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Event {
    /// A tool invocation completed (successfully or not).
    ToolCall {
        /// Stable identifier of the tool that was invoked.
        tool_id: String,
        /// Whether the tool reported success.
        ok: bool,
        /// Wall-clock duration of the call, in milliseconds.
        duration_ms: u64,
    },
    /// A permission prompt was shown to the user.
    PermissionAsked {
        /// Human-readable description of what was requested.
        request: String,
        /// The user's resolved decision (e.g. `"allow_once"`, `"deny"`).
        decision: String,
    },
    /// Control was handed from one agent role to another.
    AgentHandoff {
        /// Role name that yielded control.
        from: String,
        /// Role name that received control.
        to: String,
        /// Free-form reason supplied by the orchestrator.
        reason: String,
    },
    /// A provider returned an error.
    ProviderError {
        /// Provider identifier (e.g. `"llama-cpp"`, `"echo"`).
        provider: String,
        /// Short error code surfaced by the provider.
        code: String,
        /// Human-readable error message.
        message: String,
    },
    /// A sandboxed process was launched.
    SandboxLaunched {
        /// Backend that was selected (e.g. `"bwrap"`, `"macos"`).
        backend: String,
        /// Resolved sandbox profile name.
        profile: String,
    },
}

/// An [`Event`] paired with the metadata that makes it auditable.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EventRecord {
    /// Monotonic id assigned by the [`EventEmitter`].
    pub id: u64,
    /// Wall-clock instant the event was emitted.
    pub at: SystemTime,
    /// Optional correlation to a [`crate::observability::TurnId`].
    pub turn_id: Option<u64>,
    /// The event payload.
    pub event: Event,
}

/// Write side of the event log.
///
/// Implementors must be safe to share across orchestrator tasks. `flush` has a
/// sensible default for in-memory sinks where it is a no-op.
pub trait EventSink: Send + Sync {
    /// Append a record to the sink.
    fn write(&self, record: EventRecord);
    /// Flush any buffered records to durable storage.
    fn flush(&self) {}
}

/// In-memory [`EventSink`] used by tests and short-lived sessions.
#[derive(Debug, Default)]
pub struct MemoryEventSink {
    events: Mutex<Vec<EventRecord>>,
}

impl MemoryEventSink {
    /// Build an empty sink.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Return a clone of every record written so far, in write order.
    ///
    /// A poisoned mutex yields an empty snapshot rather than panicking — the
    /// sink prefers data loss to taking the whole orchestrator down.
    #[must_use]
    pub fn snapshot(&self) -> Vec<EventRecord> {
        self.events
            .lock()
            .map_or_else(|_| Vec::new(), |guard| guard.clone())
    }

    /// Drop every recorded event.
    pub fn clear(&self) {
        if let Ok(mut guard) = self.events.lock() {
            guard.clear();
        }
    }
}

impl EventSink for MemoryEventSink {
    fn write(&self, record: EventRecord) {
        if let Ok(mut guard) = self.events.lock() {
            guard.push(record);
        }
    }
}

/// Append-only JSON-lines [`EventSink`] backed by a single file.
#[derive(Debug)]
pub struct JsonlEventSink {
    path: PathBuf,
    file: Mutex<BufWriter<File>>,
    auto_flush: Mutex<Option<Duration>>,
    last_flush: Mutex<Instant>,
}

impl JsonlEventSink {
    /// Open `path` for append, creating it if missing.
    ///
    /// # Errors
    /// Returns [`EventLogError::Io`] if the file cannot be opened.
    pub fn open(path: PathBuf) -> Result<Self, EventLogError> {
        let file = OpenOptions::new()
            .append(true)
            .create(true)
            .open(&path)
            .map_err(EventLogError::Io)?;
        Ok(Self {
            path,
            file: Mutex::new(BufWriter::new(file)),
            auto_flush: Mutex::new(None),
            last_flush: Mutex::new(Instant::now()),
        })
    }

    /// Configure an auto-flush threshold. On `write`, if more than `dur` has
    /// passed since the last flush the writer is flushed automatically.
    pub fn auto_flush_after(&self, dur: Duration) {
        if let Ok(mut guard) = self.auto_flush.lock() {
            *guard = Some(dur);
        }
    }

    /// Return the path this sink writes to.
    #[must_use]
    pub const fn path(&self) -> &PathBuf {
        &self.path
    }

    fn write_inner(&self, record: &EventRecord) -> Result<(), EventLogError> {
        let line = serde_json::to_string(record).map_err(EventLogError::Serialize)?;
        {
            let mut guard = self.file.lock().map_err(|_| {
                EventLogError::Io(std::io::Error::other("event log mutex poisoned"))
            })?;
            guard
                .write_all(line.as_bytes())
                .map_err(EventLogError::Io)?;
            guard.write_all(b"\n").map_err(EventLogError::Io)?;
        }
        Ok(())
    }

    fn maybe_auto_flush(&self) {
        let threshold = match self.auto_flush.lock() {
            Ok(guard) => *guard,
            Err(_) => return,
        };
        let Some(threshold) = threshold else {
            return;
        };
        let elapsed = match self.last_flush.lock() {
            Ok(guard) => guard.elapsed(),
            Err(_) => return,
        };
        if elapsed > threshold {
            self.flush();
        }
    }
}

impl EventSink for JsonlEventSink {
    fn write(&self, record: EventRecord) {
        // Persisted sinks may fail mid-write (disk full, EIO). The log is best
        // effort: an orchestrator that cannot record an event must not crash
        // the turn, so we swallow the error here. Production builds should
        // pair this sink with a tracing subscriber that surfaces the IO error.
        let _ = self.write_inner(&record);
        self.maybe_auto_flush();
    }

    fn flush(&self) {
        let Ok(mut guard) = self.file.lock() else {
            return;
        };
        if guard.flush().is_err() {
            return;
        }
        if let Ok(inner) = guard.get_ref().sync_data() {
            // sync_data returns (), `inner` is `()` here; explicit binding
            // keeps the optimizer from eliding the call.
            let () = inner;
        }
        if let Ok(mut last) = self.last_flush.lock() {
            *last = Instant::now();
        }
    }
}

/// Clock abstraction used by [`EventEmitter`] to stamp `at`.
pub trait EventClock: Send + Sync {
    /// Return the current wall-clock instant.
    fn now(&self) -> SystemTime;
}

/// Production [`EventClock`] backed by [`SystemTime::now`].
#[derive(Debug, Default, Clone, Copy)]
pub struct SystemEventClock;

impl EventClock for SystemEventClock {
    fn now(&self) -> SystemTime {
        SystemTime::now()
    }
}

/// Test-only [`EventClock`] that always reports the same instant.
#[derive(Debug, Clone, Copy)]
pub struct FixedEventClock(pub SystemTime);

impl EventClock for FixedEventClock {
    fn now(&self) -> SystemTime {
        self.0
    }
}

/// Assigns monotonic ids and timestamps to events, then hands them to a sink.
pub struct EventEmitter {
    next_id: AtomicU64,
    clock: Box<dyn EventClock>,
    sink: Arc<dyn EventSink>,
}

impl fmt::Debug for EventEmitter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EventEmitter")
            .field("next_id", &self.next_id)
            .finish_non_exhaustive()
    }
}

impl EventEmitter {
    /// Build an emitter that uses [`SystemEventClock`].
    #[must_use]
    pub fn new(sink: Arc<dyn EventSink>) -> Self {
        Self::with_clock(sink, Box::new(SystemEventClock))
    }

    /// Build an emitter with a custom clock (used by tests).
    #[must_use]
    pub fn with_clock(sink: Arc<dyn EventSink>, clock: Box<dyn EventClock>) -> Self {
        Self {
            next_id: AtomicU64::new(1),
            clock,
            sink,
        }
    }

    /// Emit a single event. Returns the id that was assigned.
    pub fn emit(&self, event: Event, turn_id: Option<u64>) -> u64 {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let record = EventRecord {
            id,
            at: self.clock.now(),
            turn_id,
            event,
        };
        self.sink.write(record);
        id
    }
}

/// Errors produced by [`JsonlEventSink`] open / serialize.
#[derive(Debug)]
pub enum EventLogError {
    /// Filesystem error.
    Io(std::io::Error),
    /// Serde failed to encode a record.
    Serialize(serde_json::Error),
}

impl fmt::Display for EventLogError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "event log io error: {err}"),
            Self::Serialize(err) => write!(f, "event log serialize error: {err}"),
        }
    }
}

impl Error for EventLogError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            Self::Serialize(err) => Some(err),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::{BufRead, BufReader};
    use std::sync::Arc;
    use std::thread;
    use tempfile::TempDir;

    fn sample_tool_call() -> Event {
        Event::ToolCall {
            tool_id: "fs.read".into(),
            ok: true,
            duration_ms: 12,
        }
    }

    fn fixed_clock() -> FixedEventClock {
        FixedEventClock(SystemTime::UNIX_EPOCH + Duration::from_secs(1_700_000_000))
    }

    #[test]
    fn memory_sink_round_trip_preserves_order() {
        let sink = MemoryEventSink::new();
        for i in 0..3 {
            sink.write(EventRecord {
                id: i,
                at: SystemTime::UNIX_EPOCH,
                turn_id: None,
                event: Event::ToolCall {
                    tool_id: format!("t{i}"),
                    ok: true,
                    duration_ms: i,
                },
            });
        }
        let snap = sink.snapshot();
        assert_eq!(snap.len(), 3);
        assert_eq!(snap[0].id, 0);
        assert_eq!(snap[2].id, 2);
    }

    #[test]
    fn memory_sink_clear_empties() {
        let sink = MemoryEventSink::new();
        sink.write(EventRecord {
            id: 1,
            at: SystemTime::UNIX_EPOCH,
            turn_id: None,
            event: sample_tool_call(),
        });
        sink.clear();
        assert!(sink.snapshot().is_empty());
    }

    #[test]
    fn memory_sink_empty_snapshot_is_empty() {
        let sink = MemoryEventSink::new();
        assert!(sink.snapshot().is_empty());
    }

    #[test]
    fn emitter_assigns_monotonic_ids_starting_at_one() {
        let sink = Arc::new(MemoryEventSink::new());
        let emitter = EventEmitter::new(sink.clone());
        let a = emitter.emit(sample_tool_call(), None);
        let b = emitter.emit(sample_tool_call(), None);
        let c = emitter.emit(sample_tool_call(), None);
        assert_eq!(a, 1);
        assert_eq!(b, 2);
        assert_eq!(c, 3);
        let snap = sink.snapshot();
        assert_eq!(snap.iter().map(|r| r.id).collect::<Vec<_>>(), vec![1, 2, 3]);
    }

    #[test]
    fn emitter_preserves_turn_id() {
        let sink = Arc::new(MemoryEventSink::new());
        let emitter = EventEmitter::new(sink.clone());
        let _ = emitter.emit(sample_tool_call(), Some(42));
        let snap = sink.snapshot();
        assert_eq!(snap[0].turn_id, Some(42));
    }

    #[test]
    fn emitter_uses_injected_clock() {
        let sink = Arc::new(MemoryEventSink::new());
        let clock = fixed_clock();
        let emitter = EventEmitter::with_clock(sink.clone(), Box::new(clock));
        let _ = emitter.emit(sample_tool_call(), None);
        let snap = sink.snapshot();
        assert_eq!(snap[0].at, clock.0);
    }

    #[test]
    fn emit_returns_same_id_as_recorded() {
        let sink = Arc::new(MemoryEventSink::new());
        let emitter = EventEmitter::new(sink.clone());
        let id = emitter.emit(sample_tool_call(), None);
        let snap = sink.snapshot();
        assert_eq!(snap[0].id, id);
    }

    #[test]
    fn memory_sink_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<MemoryEventSink>();
    }

    #[test]
    fn tool_call_serde_round_trip() {
        let rec = EventRecord {
            id: 7,
            at: SystemTime::UNIX_EPOCH + Duration::from_secs(5),
            turn_id: Some(3),
            event: sample_tool_call(),
        };
        let json = serde_json::to_string(&rec).expect("serialize");
        let back: EventRecord = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(rec, back);
    }

    #[test]
    fn permission_asked_serde_round_trip() {
        let rec = EventRecord {
            id: 1,
            at: SystemTime::UNIX_EPOCH,
            turn_id: None,
            event: Event::PermissionAsked {
                request: "net.connect example.com:443".into(),
                decision: "allow_once".into(),
            },
        };
        let json = serde_json::to_string(&rec).expect("serialize");
        let back: EventRecord = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(rec, back);
    }

    #[test]
    fn agent_handoff_serde_round_trip() {
        let rec = EventRecord {
            id: 1,
            at: SystemTime::UNIX_EPOCH,
            turn_id: None,
            event: Event::AgentHandoff {
                from: "planner".into(),
                to: "coder".into(),
                reason: "ready".into(),
            },
        };
        let json = serde_json::to_string(&rec).expect("serialize");
        let back: EventRecord = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(rec, back);
    }

    #[test]
    fn provider_error_serde_round_trip() {
        let rec = EventRecord {
            id: 1,
            at: SystemTime::UNIX_EPOCH,
            turn_id: None,
            event: Event::ProviderError {
                provider: "llama-cpp".into(),
                code: "STRAT-E1234".into(),
                message: "context overflow".into(),
            },
        };
        let json = serde_json::to_string(&rec).expect("serialize");
        let back: EventRecord = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(rec, back);
    }

    #[test]
    fn sandbox_launched_serde_round_trip() {
        let rec = EventRecord {
            id: 1,
            at: SystemTime::UNIX_EPOCH,
            turn_id: None,
            event: Event::SandboxLaunched {
                backend: "bwrap".into(),
                profile: "default".into(),
            },
        };
        let json = serde_json::to_string(&rec).expect("serialize");
        let back: EventRecord = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(rec, back);
    }

    #[test]
    fn tool_call_kind_tag_is_snake_case() {
        let json = serde_json::to_string(&sample_tool_call()).expect("serialize");
        assert!(
            json.contains("\"kind\":\"tool_call\""),
            "kind tag missing in {json}"
        );
    }

    #[test]
    fn event_record_id_survives_serde() {
        let rec = EventRecord {
            id: 9_999,
            at: SystemTime::UNIX_EPOCH,
            turn_id: Some(11),
            event: sample_tool_call(),
        };
        let json = serde_json::to_string(&rec).expect("serialize");
        let back: EventRecord = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.id, 9_999);
        assert_eq!(back.turn_id, Some(11));
    }

    #[test]
    fn jsonl_sink_creates_file_when_missing() {
        let dir = TempDir::new().expect("tempdir");
        let path = dir.path().join("events.jsonl");
        assert!(!path.exists());
        let sink = JsonlEventSink::open(path.clone()).expect("open");
        sink.flush();
        assert!(path.exists());
        assert_eq!(sink.path(), &path);
    }

    #[test]
    fn jsonl_sink_writes_one_line_per_event() {
        let dir = TempDir::new().expect("tempdir");
        let path = dir.path().join("events.jsonl");
        let sink = JsonlEventSink::open(path.clone()).expect("open");
        for i in 0..4 {
            sink.write(EventRecord {
                id: i,
                at: SystemTime::UNIX_EPOCH,
                turn_id: None,
                event: Event::ToolCall {
                    tool_id: format!("t{i}"),
                    ok: true,
                    duration_ms: i,
                },
            });
        }
        sink.flush();
        let contents = fs::read_to_string(&path).expect("read");
        let line_count = contents.lines().count();
        assert_eq!(line_count, 4);
    }

    #[test]
    fn jsonl_sink_is_append_only_across_reopen() {
        let dir = TempDir::new().expect("tempdir");
        let path = dir.path().join("events.jsonl");
        {
            let sink = JsonlEventSink::open(path.clone()).expect("open");
            sink.write(EventRecord {
                id: 1,
                at: SystemTime::UNIX_EPOCH,
                turn_id: None,
                event: sample_tool_call(),
            });
            sink.write(EventRecord {
                id: 2,
                at: SystemTime::UNIX_EPOCH,
                turn_id: None,
                event: sample_tool_call(),
            });
            sink.flush();
        }
        {
            let sink = JsonlEventSink::open(path.clone()).expect("reopen");
            sink.write(EventRecord {
                id: 3,
                at: SystemTime::UNIX_EPOCH,
                turn_id: None,
                event: sample_tool_call(),
            });
            sink.flush();
        }
        let contents = fs::read_to_string(&path).expect("read");
        assert_eq!(contents.lines().count(), 3);
    }

    #[test]
    fn jsonl_sink_flush_is_idempotent() {
        let dir = TempDir::new().expect("tempdir");
        let path = dir.path().join("events.jsonl");
        let sink = JsonlEventSink::open(path).expect("open");
        sink.write(EventRecord {
            id: 1,
            at: SystemTime::UNIX_EPOCH,
            turn_id: None,
            event: sample_tool_call(),
        });
        sink.flush();
        sink.flush();
    }

    #[test]
    fn jsonl_sink_auto_flush_triggers_on_next_write() {
        let dir = TempDir::new().expect("tempdir");
        let path = dir.path().join("events.jsonl");
        let sink = JsonlEventSink::open(path.clone()).expect("open");
        sink.auto_flush_after(Duration::from_nanos(1));
        // First write loads the buffer, then the auto-flush check fires
        // because the elapsed time since `open` already exceeds 1ns.
        sink.write(EventRecord {
            id: 1,
            at: SystemTime::UNIX_EPOCH,
            turn_id: None,
            event: sample_tool_call(),
        });
        // The file should be observable without an explicit flush call.
        let contents = fs::read_to_string(&path).expect("read");
        assert_eq!(contents.lines().count(), 1);
    }

    #[test]
    fn event_log_error_display_smoke() {
        let io = EventLogError::Io(std::io::Error::other("boom"));
        assert!(format!("{io}").contains("io error"));
        let serde_err = serde_json::from_str::<EventRecord>("not json").unwrap_err();
        let ser = EventLogError::Serialize(serde_err);
        assert!(format!("{ser}").contains("serialize error"));
        // Source chain populated.
        assert!(Error::source(&io).is_some());
        assert!(Error::source(&ser).is_some());
    }

    #[test]
    fn concurrent_emit_assigns_unique_monotonic_ids() {
        let sink = Arc::new(MemoryEventSink::new());
        let emitter = Arc::new(EventEmitter::new(sink.clone()));
        let per_thread = 25_u64;
        let threads = 4_u64;
        let mut handles = Vec::new();
        for _ in 0..threads {
            let em = emitter.clone();
            handles.push(thread::spawn(move || {
                for _ in 0..per_thread {
                    em.emit(sample_tool_call(), None);
                }
            }));
        }
        for h in handles {
            h.join().expect("join");
        }
        let snap = sink.snapshot();
        assert_eq!(snap.len() as u64, threads * per_thread);
        let mut ids: Vec<u64> = snap.iter().map(|r| r.id).collect();
        ids.sort_unstable();
        let unique = {
            let mut copy = ids.clone();
            copy.dedup();
            copy.len()
        };
        assert_eq!(unique, ids.len());
        assert_eq!(*ids.first().expect("first"), 1);
        assert_eq!(*ids.last().expect("last"), threads * per_thread);
    }

    #[test]
    fn emitter_with_jsonl_sink_round_trips_via_file() {
        let dir = TempDir::new().expect("tempdir");
        let path = dir.path().join("events.jsonl");
        let sink: Arc<dyn EventSink> = Arc::new(JsonlEventSink::open(path.clone()).expect("open"));
        let emitter = EventEmitter::with_clock(sink.clone(), Box::new(fixed_clock()));
        emitter.emit(
            Event::ToolCall {
                tool_id: "fs.write".into(),
                ok: false,
                duration_ms: 99,
            },
            Some(7),
        );
        emitter.emit(
            Event::AgentHandoff {
                from: "planner".into(),
                to: "coder".into(),
                reason: "compile".into(),
            },
            Some(7),
        );
        sink.flush();
        let file = fs::File::open(&path).expect("open file");
        let reader = BufReader::new(file);
        let mut records: Vec<EventRecord> = Vec::new();
        for line in reader.lines() {
            let line = line.expect("line");
            let rec: EventRecord = serde_json::from_str(&line).expect("parse");
            records.push(rec);
        }
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].id, 1);
        assert_eq!(records[1].id, 2);
        assert_eq!(records[0].turn_id, Some(7));
        assert_eq!(
            records[0].event,
            Event::ToolCall {
                tool_id: "fs.write".into(),
                ok: false,
                duration_ms: 99,
            }
        );
        assert_eq!(
            records[1].event,
            Event::AgentHandoff {
                from: "planner".into(),
                to: "coder".into(),
                reason: "compile".into(),
            }
        );
    }

    #[test]
    fn system_event_clock_advances() {
        let clock = SystemEventClock;
        let a = clock.now();
        let b = clock.now();
        assert!(b >= a);
    }

    #[test]
    fn memory_sink_default_flush_is_noop() {
        // Exercises the default `flush` body on the trait.
        let sink = MemoryEventSink::new();
        sink.write(EventRecord {
            id: 1,
            at: SystemTime::UNIX_EPOCH,
            turn_id: None,
            event: sample_tool_call(),
        });
        EventSink::flush(&sink);
        assert_eq!(sink.snapshot().len(), 1);
    }

    #[test]
    fn event_emitter_debug_includes_next_id() {
        let sink = Arc::new(MemoryEventSink::new());
        let emitter = EventEmitter::new(sink);
        let dbg = format!("{emitter:?}");
        assert!(dbg.contains("EventEmitter"));
        assert!(dbg.contains("next_id"));
    }
}
