//! On-disk conversation-transcript shape and store.
//!
//! Phase 1 scaffold for the persistence layer the future TUI will write to so
//! chat survives restarts. See `plan/10-chat-and-scrollback.md`.
//!
//! # Shape
//!
//! A [`Transcript`] is a sequence of [`TranscriptTurn`]s carrying the schema
//! version, a session id, and a creation timestamp. Each turn is one of
//! `User`, `Assistant`, `System`, or `Command`. The `Assistant` variant carries
//! a list of [`TranscriptBlock`]s so future renderers can split prose, code
//! fences, and tool calls without re-parsing.
//!
//! # On-disk
//!
//! [`TranscriptStore`] writes one `<session_id>.json` per session under a
//! base directory, using a `<file>.tmp` + rename atomic save. A configurable
//! per-session byte ceiling protects the chat dir from runaway dumps; the
//! default is 64 MiB.

use std::error::Error;
use std::fmt::{self, Display, Formatter};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Current on-disk schema version for [`Transcript`].
///
/// Bumped only via the same forward-only additive policy used by
/// `installed.toml` (new optional fields with `#[serde(default)]`).
pub const TRANSCRIPT_SCHEMA_VERSION: u32 = 1;

/// Length in lowercase hex chars of a [`SessionId`] (8 random bytes → 16 hex).
const SESSION_ID_LEN: usize = 16;

/// Default per-session size ceiling (64 MiB).
const DEFAULT_MAX_PER_SESSION_BYTES: u64 = 64 * 1024 * 1024;

// ---------------------------------------------------------------------------
// SessionId
// ---------------------------------------------------------------------------

/// Opaque per-session identifier.
///
/// The wire form is exactly 16 lowercase hex characters. Mirrors
/// `AnonInstallId`'s shape so callers learn one validation rule for all
/// runtime hex ids.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SessionId(String);

impl SessionId {
    /// Generate a fresh 8-byte random id rendered as 16 lowercase hex chars.
    ///
    /// Entropy mixes `SystemTime` nanos with a process-local monotonic
    /// counter through `SplitMix64`. Cryptographic strength is not required;
    /// the goal is per-process uniqueness for a chat session.
    #[must_use]
    pub fn new_random() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let counter = COUNTER.fetch_add(1, Ordering::Relaxed);
        let nanos: u64 = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| {
                #[allow(
                    clippy::cast_possible_truncation,
                    reason = "low-bits-of-nanos is the entropy source we want"
                )]
                let n = d.as_nanos() as u64;
                n
            })
            .unwrap_or(0);
        let seed = splitmix64(nanos ^ splitmix64(counter));
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut bytes = [0u8; 8];
        rng.fill(&mut bytes);
        let mut out = String::with_capacity(SESSION_ID_LEN);
        for byte in bytes {
            let hi = (byte >> 4) & 0xF;
            let lo = byte & 0xF;
            out.push(hex_digit(hi));
            out.push(hex_digit(lo));
        }
        Self(out)
    }

    /// Parse `s` as a 16-lowercase-hex-char session id.
    ///
    /// Free-standing constructor (not the `FromStr` trait) so the caller does
    /// not need a turbofish — mirrors `AnonInstallId::from_str`.
    ///
    /// # Errors
    ///
    /// Returns [`SessionIdError::WrongLength`] when `s` is not exactly 16
    /// characters, and [`SessionIdError::InvalidHex`] when any character is
    /// not a lowercase hex digit.
    #[allow(
        clippy::should_implement_trait,
        reason = "deliberate inherent constructor; mirrors AnonInstallId::from_str shape"
    )]
    pub fn from_str(s: &str) -> Result<Self, SessionIdError> {
        if s.len() != SESSION_ID_LEN {
            return Err(SessionIdError::WrongLength { actual: s.len() });
        }
        for c in s.chars() {
            if !is_lower_hex(c) {
                return Err(SessionIdError::InvalidHex);
            }
        }
        Ok(Self(s.to_owned()))
    }

    /// Borrow the underlying string slice.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Display for SessionId {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Serialize for SessionId {
    fn serialize<S: serde::Serializer>(&self, ser: S) -> Result<S::Ok, S::Error> {
        ser.serialize_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for SessionId {
    fn deserialize<D: serde::Deserializer<'de>>(de: D) -> Result<Self, D::Error> {
        let s = String::deserialize(de)?;
        Self::from_str(&s).map_err(serde::de::Error::custom)
    }
}

/// First-failure rejection from [`SessionId::from_str`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionIdError {
    /// Length was not exactly 16 chars.
    WrongLength {
        /// Observed length in chars.
        actual: usize,
    },
    /// Encountered a character outside `[0-9a-f]`.
    InvalidHex,
}

impl Display for SessionIdError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongLength { actual } => {
                write!(f, "session id must be {SESSION_ID_LEN} chars; got {actual}")
            }
            Self::InvalidHex => f.write_str("session id must be lowercase hex `[0-9a-f]`"),
        }
    }
}

impl Error for SessionIdError {}

#[inline]
const fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[inline]
const fn hex_digit(nibble: u8) -> char {
    match nibble {
        0..=9 => (b'0' + nibble) as char,
        _ => (b'a' + nibble - 10) as char,
    }
}

#[inline]
const fn is_lower_hex(c: char) -> bool {
    matches!(c, '0'..='9' | 'a'..='f')
}

// ---------------------------------------------------------------------------
// Block / Turn shape
// ---------------------------------------------------------------------------

/// Kind of an assistant block. The strict enum keeps the wire form
/// closed: a new variant requires a code change and a serde test.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TranscriptBlockKind {
    /// Plain prose.
    Text,
    /// Fenced code block.
    Code {
        /// Lower-case language tag, e.g. `rust`. Empty string when unknown.
        language: String,
    },
    /// Tool invocation rendered inline in the transcript.
    ToolCall {
        /// Opaque id of the tool being invoked.
        tool_id: String,
    },
}

/// A single span of assistant output (prose, code, or a tool call) plus its
/// rendered text.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TranscriptBlock {
    /// Block discriminator + per-kind data.
    pub kind: TranscriptBlockKind,
    /// Already-rendered text for this block.
    pub text: String,
}

/// A single turn in a conversation transcript.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TranscriptTurn {
    /// Free-form user message.
    User {
        /// Wall-clock timestamp when the turn was recorded.
        at: SystemTime,
        /// User-entered text.
        text: String,
    },
    /// Assistant reply, split into renderable blocks.
    Assistant {
        /// Wall-clock timestamp when the turn was recorded.
        at: SystemTime,
        /// Blocks in display order.
        blocks: Vec<TranscriptBlock>,
    },
    /// System / banner line.
    System {
        /// Wall-clock timestamp when the turn was recorded.
        at: SystemTime,
        /// System text.
        text: String,
    },
    /// User-issued slash command + its outcome.
    Command {
        /// Wall-clock timestamp when the turn was recorded.
        at: SystemTime,
        /// Raw command line, e.g. `/help`.
        text: String,
        /// Whether the command completed without error.
        ok: bool,
    },
}

// ---------------------------------------------------------------------------
// Transcript
// ---------------------------------------------------------------------------

/// Full on-disk conversation transcript.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Transcript {
    /// On-disk schema version. Always [`TRANSCRIPT_SCHEMA_VERSION`] for fresh
    /// writes; loader rejects strictly-newer values.
    pub schema_version: u32,
    /// Opaque session id; matches the on-disk filename stem.
    pub session_id: SessionId,
    /// Wall-clock timestamp at session creation.
    pub created_at: SystemTime,
    /// Turns in temporal order.
    pub turns: Vec<TranscriptTurn>,
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// All failure modes for [`TranscriptStore`] and friends.
#[derive(Debug)]
pub enum TranscriptError {
    /// Filesystem I/O error.
    Io(std::io::Error),
    /// `serde_json` encode/decode failure.
    Serialize(serde_json::Error),
    /// Caller-supplied invariant violation (e.g. bad session id length).
    Validation(String),
    /// Serialized payload exceeded the configured per-session ceiling.
    TooLarge {
        /// Observed byte length.
        bytes: u64,
    },
    /// On-disk `schema_version` is strictly newer than this binary supports.
    SchemaNewer {
        /// Version found on disk.
        found: u32,
        /// Highest version this binary understands.
        supported: u32,
    },
    /// Filename did not parse as a [`SessionId`].
    BadSessionId(SessionIdError),
}

impl Display for TranscriptError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "transcript io error: {e}"),
            Self::Serialize(e) => write!(f, "transcript serialize error: {e}"),
            Self::Validation(msg) => write!(f, "transcript validation error: {msg}"),
            Self::TooLarge { bytes } => {
                write!(f, "transcript exceeds per-session ceiling: {bytes} bytes")
            }
            Self::SchemaNewer { found, supported } => write!(
                f,
                "transcript schema_version {found} is newer than supported {supported}"
            ),
            Self::BadSessionId(e) => write!(f, "transcript filename is not a session id: {e}"),
        }
    }
}

impl Error for TranscriptError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Serialize(e) => Some(e),
            Self::BadSessionId(e) => Some(e),
            Self::Validation(_) | Self::TooLarge { .. } | Self::SchemaNewer { .. } => None,
        }
    }
}

impl From<std::io::Error> for TranscriptError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<serde_json::Error> for TranscriptError {
    fn from(e: serde_json::Error) -> Self {
        Self::Serialize(e)
    }
}

impl From<SessionIdError> for TranscriptError {
    fn from(e: SessionIdError) -> Self {
        Self::BadSessionId(e)
    }
}

// ---------------------------------------------------------------------------
// Redaction hook
// ---------------------------------------------------------------------------

/// PII-redaction hook applied per turn before persistence.
///
/// Currently a no-op stub: the production redaction pass (regexes for paths,
/// tokens, emails) lands with the chat UI in a later commit. Keeping the
/// function here means the call-site in the future `TurnRecorder → Transcript`
/// flush path is already wired, and future commits only need to replace the
/// body.
#[allow(
    clippy::ptr_arg,
    clippy::needless_pass_by_ref_mut,
    reason = "signature must remain `&mut` so future redaction can mutate in place"
)]
pub const fn redact_pii(_turn: &mut TranscriptTurn) {
    // Intentionally empty. See doc comment.
}

// ---------------------------------------------------------------------------
// Store
// ---------------------------------------------------------------------------

/// On-disk transcript directory.
///
/// One JSON file per session under `dir`. Writes are atomic (`<file>.tmp` +
/// rename) and capped at `max_per_session_bytes`.
#[derive(Debug)]
pub struct TranscriptStore {
    dir: PathBuf,
    max_per_session_bytes: u64,
}

impl TranscriptStore {
    /// Open (or create) a store rooted at `dir` with the default per-session
    /// ceiling.
    ///
    /// # Errors
    /// Returns [`TranscriptError::Io`] if `dir` cannot be created.
    pub fn open(dir: PathBuf) -> Result<Self, TranscriptError> {
        std::fs::create_dir_all(&dir)?;
        Ok(Self {
            dir,
            max_per_session_bytes: DEFAULT_MAX_PER_SESSION_BYTES,
        })
    }

    /// Override the per-session size ceiling (bytes).
    #[must_use]
    pub const fn with_max_per_session_bytes(mut self, bytes: u64) -> Self {
        self.max_per_session_bytes = bytes;
        self
    }

    /// Filesystem path for `session_id`'s file.
    fn path_for(&self, session_id: &SessionId) -> PathBuf {
        self.dir.join(format!("{session_id}.json"))
    }

    /// Atomically write `t` to `<dir>/<session_id>.json` via `<file>.tmp` +
    /// rename.
    ///
    /// # Errors
    ///
    /// - [`TranscriptError::TooLarge`] if the serialized JSON exceeds the
    ///   configured ceiling.
    /// - [`TranscriptError::Serialize`] / [`TranscriptError::Io`] on the
    ///   respective subsystem failure.
    pub fn save_atomic(&self, t: &Transcript) -> Result<PathBuf, TranscriptError> {
        let bytes = serde_json::to_vec_pretty(t)?;
        let len = bytes.len() as u64;
        if len > self.max_per_session_bytes {
            return Err(TranscriptError::TooLarge { bytes: len });
        }

        let final_path = self.path_for(&t.session_id);
        let tmp = {
            let mut name = final_path
                .file_name()
                .ok_or_else(|| TranscriptError::Validation("missing file name".to_owned()))?
                .to_os_string();
            name.push(".tmp");
            final_path.with_file_name(name)
        };

        {
            use std::io::Write;
            let mut f = std::fs::OpenOptions::new()
                .create(true)
                .truncate(true)
                .write(true)
                .open(&tmp)?;
            f.write_all(&bytes)?;
            f.sync_all()?;
        }

        if let Err(e) = std::fs::rename(&tmp, &final_path) {
            let _ = std::fs::remove_file(&tmp);
            return Err(TranscriptError::Io(e));
        }

        #[cfg(unix)]
        {
            if let Ok(d) = std::fs::File::open(&self.dir) {
                let _ = d.sync_all();
            }
        }

        Ok(final_path)
    }

    /// Load the transcript for `session_id`.
    ///
    /// # Errors
    ///
    /// - [`TranscriptError::Io`] if the file does not exist or cannot be read.
    /// - [`TranscriptError::Serialize`] on malformed JSON.
    /// - [`TranscriptError::SchemaNewer`] if the file declares a strictly
    ///   newer `schema_version`.
    pub fn load(&self, session_id: &SessionId) -> Result<Transcript, TranscriptError> {
        let path = self.path_for(session_id);
        let raw = std::fs::read(&path)?;
        let parsed: Transcript = serde_json::from_slice(&raw)?;
        if parsed.schema_version > TRANSCRIPT_SCHEMA_VERSION {
            return Err(TranscriptError::SchemaNewer {
                found: parsed.schema_version,
                supported: TRANSCRIPT_SCHEMA_VERSION,
            });
        }
        Ok(parsed)
    }

    /// List all session ids currently on disk, sorted ascending.
    ///
    /// Files whose stem does not parse as a [`SessionId`] are silently
    /// skipped — this keeps stray editor swap files from poisoning the list.
    ///
    /// # Errors
    /// Returns [`TranscriptError::Io`] if the directory cannot be read.
    pub fn list(&self) -> Result<Vec<SessionId>, TranscriptError> {
        let mut out: Vec<SessionId> = Vec::new();
        for entry in std::fs::read_dir(&self.dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("json") {
                continue;
            }
            let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            if let Ok(id) = SessionId::from_str(stem) {
                out.push(id);
            }
        }
        out.sort();
        Ok(out)
    }

    /// Delete the transcript for `session_id`. Returns `true` if a file was
    /// removed, `false` if it did not exist.
    ///
    /// # Errors
    /// Returns [`TranscriptError::Io`] on any error other than `NotFound`.
    pub fn delete(&self, session_id: &SessionId) -> Result<bool, TranscriptError> {
        let path = self.path_for(session_id);
        match std::fs::remove_file(&path) {
            Ok(()) => Ok(true),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
            Err(e) => Err(TranscriptError::Io(e)),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tempfile::TempDir;

    fn fixed_time() -> SystemTime {
        UNIX_EPOCH + Duration::from_secs(1_700_000_000)
    }

    fn sample_session_id() -> SessionId {
        SessionId::from_str("deadbeefcafef00d").unwrap()
    }

    fn sample_transcript(id: SessionId, turns: Vec<TranscriptTurn>) -> Transcript {
        Transcript {
            schema_version: TRANSCRIPT_SCHEMA_VERSION,
            session_id: id,
            created_at: fixed_time(),
            turns,
        }
    }

    // ---- SessionId --------------------------------------------------------

    #[test]
    fn session_id_new_random_is_16_lower_hex() {
        let id = SessionId::new_random();
        assert_eq!(id.as_str().len(), SESSION_ID_LEN);
        for c in id.as_str().chars() {
            assert!(is_lower_hex(c), "non-hex char: {c}");
        }
    }

    #[test]
    fn session_id_from_str_happy() {
        let id = SessionId::from_str("0123456789abcdef").unwrap();
        assert_eq!(id.as_str(), "0123456789abcdef");
        assert_eq!(id.to_string(), "0123456789abcdef");
    }

    #[test]
    fn session_id_rejects_wrong_length() {
        assert_eq!(
            SessionId::from_str("0123"),
            Err(SessionIdError::WrongLength { actual: 4 })
        );
        assert_eq!(
            SessionId::from_str("0123456789abcdef0"),
            Err(SessionIdError::WrongLength { actual: 17 })
        );
    }

    #[test]
    fn session_id_rejects_non_hex() {
        assert_eq!(
            SessionId::from_str("0123456789abcdez"),
            Err(SessionIdError::InvalidHex)
        );
    }

    #[test]
    fn session_id_rejects_uppercase() {
        assert_eq!(
            SessionId::from_str("0123456789ABCDEF"),
            Err(SessionIdError::InvalidHex)
        );
    }

    #[test]
    fn session_id_serde_transparent() {
        let id = sample_session_id();
        let json = serde_json::to_string(&id).unwrap();
        assert_eq!(json, "\"deadbeefcafef00d\"");
        let back: SessionId = serde_json::from_str(&json).unwrap();
        assert_eq!(back, id);
    }

    #[test]
    fn session_id_serde_rejects_garbage() {
        let r: Result<SessionId, _> = serde_json::from_str("\"not-a-session-id\"");
        assert!(r.is_err());
    }

    // ---- Transcript turn round-trip --------------------------------------

    #[test]
    fn transcript_round_trip_user_turn() {
        let t = sample_transcript(
            sample_session_id(),
            vec![TranscriptTurn::User {
                at: fixed_time(),
                text: "hello".to_owned(),
            }],
        );
        let json = serde_json::to_string(&t).unwrap();
        let back: Transcript = serde_json::from_str(&json).unwrap();
        assert_eq!(back, t);
    }

    #[test]
    fn transcript_round_trip_assistant_turn() {
        let t = sample_transcript(
            sample_session_id(),
            vec![TranscriptTurn::Assistant {
                at: fixed_time(),
                blocks: vec![TranscriptBlock {
                    kind: TranscriptBlockKind::Text,
                    text: "hi".to_owned(),
                }],
            }],
        );
        let json = serde_json::to_string(&t).unwrap();
        let back: Transcript = serde_json::from_str(&json).unwrap();
        assert_eq!(back, t);
    }

    #[test]
    fn transcript_round_trip_system_turn() {
        let t = sample_transcript(
            sample_session_id(),
            vec![TranscriptTurn::System {
                at: fixed_time(),
                text: "model swapped".to_owned(),
            }],
        );
        let json = serde_json::to_string(&t).unwrap();
        let back: Transcript = serde_json::from_str(&json).unwrap();
        assert_eq!(back, t);
    }

    #[test]
    fn transcript_round_trip_command_turn() {
        let t = sample_transcript(
            sample_session_id(),
            vec![TranscriptTurn::Command {
                at: fixed_time(),
                text: "/help".to_owned(),
                ok: true,
            }],
        );
        let json = serde_json::to_string(&t).unwrap();
        let back: Transcript = serde_json::from_str(&json).unwrap();
        assert_eq!(back, t);
    }

    // ---- TranscriptBlockKind ---------------------------------------------

    #[test]
    fn block_round_trip_text() {
        let b = TranscriptBlock {
            kind: TranscriptBlockKind::Text,
            text: "plain".to_owned(),
        };
        let json = serde_json::to_string(&b).unwrap();
        let back: TranscriptBlock = serde_json::from_str(&json).unwrap();
        assert_eq!(back, b);
    }

    #[test]
    fn block_round_trip_code() {
        let b = TranscriptBlock {
            kind: TranscriptBlockKind::Code {
                language: "rust".to_owned(),
            },
            text: "fn main() {}".to_owned(),
        };
        let json = serde_json::to_string(&b).unwrap();
        let back: TranscriptBlock = serde_json::from_str(&json).unwrap();
        assert_eq!(back, b);
    }

    #[test]
    fn block_round_trip_tool_call() {
        let b = TranscriptBlock {
            kind: TranscriptBlockKind::ToolCall {
                tool_id: "fs.read".to_owned(),
            },
            text: "{...}".to_owned(),
        };
        let json = serde_json::to_string(&b).unwrap();
        let back: TranscriptBlock = serde_json::from_str(&json).unwrap();
        assert_eq!(back, b);
    }

    // ---- TranscriptStore --------------------------------------------------

    #[test]
    fn store_open_creates_missing_dir() {
        let tmp = TempDir::new().unwrap();
        let target = tmp.path().join("nested").join("chat");
        assert!(!target.exists());
        let _store = TranscriptStore::open(target.clone()).unwrap();
        assert!(target.exists());
    }

    #[test]
    fn store_save_writes_expected_path() {
        let tmp = TempDir::new().unwrap();
        let store = TranscriptStore::open(tmp.path().to_path_buf()).unwrap();
        let id = sample_session_id();
        let t = sample_transcript(id, vec![]);
        let written = store.save_atomic(&t).unwrap();
        assert_eq!(written, tmp.path().join("deadbeefcafef00d.json"));
        assert!(written.exists());
    }

    #[test]
    fn store_save_and_load_round_trip() {
        let tmp = TempDir::new().unwrap();
        let store = TranscriptStore::open(tmp.path().to_path_buf()).unwrap();
        let id = sample_session_id();
        let t = sample_transcript(
            id.clone(),
            vec![
                TranscriptTurn::User {
                    at: fixed_time(),
                    text: "hi".to_owned(),
                },
                TranscriptTurn::Assistant {
                    at: fixed_time() + Duration::from_secs(1),
                    blocks: vec![TranscriptBlock {
                        kind: TranscriptBlockKind::Text,
                        text: "hello".to_owned(),
                    }],
                },
            ],
        );
        store.save_atomic(&t).unwrap();
        let back = store.load(&id).unwrap();
        assert_eq!(back, t);
    }

    #[test]
    fn store_save_rejects_oversized_transcript() {
        let tmp = TempDir::new().unwrap();
        let store = TranscriptStore::open(tmp.path().to_path_buf())
            .unwrap()
            .with_max_per_session_bytes(64);
        let id = sample_session_id();
        let t = sample_transcript(
            id,
            vec![TranscriptTurn::User {
                at: fixed_time(),
                text: "x".repeat(1024),
            }],
        );
        match store.save_atomic(&t) {
            Err(TranscriptError::TooLarge { bytes }) => assert!(bytes > 64),
            other => panic!("expected TooLarge, got {other:?}"),
        }
    }

    #[test]
    fn store_load_rejects_newer_schema() {
        let tmp = TempDir::new().unwrap();
        let store = TranscriptStore::open(tmp.path().to_path_buf()).unwrap();
        let id = sample_session_id();
        let path = tmp.path().join("deadbeefcafef00d.json");
        let raw = r#"{
            "schema_version": 999,
            "session_id": "deadbeefcafef00d",
            "created_at": { "secs_since_epoch": 0, "nanos_since_epoch": 0 },
            "turns": []
        }"#;
        std::fs::write(&path, raw).unwrap();
        match store.load(&id) {
            Err(TranscriptError::SchemaNewer { found, supported }) => {
                assert_eq!(found, 999);
                assert_eq!(supported, TRANSCRIPT_SCHEMA_VERSION);
            }
            other => panic!("expected SchemaNewer, got {other:?}"),
        }
    }

    #[test]
    fn store_load_errors_on_malformed_json() {
        let tmp = TempDir::new().unwrap();
        let store = TranscriptStore::open(tmp.path().to_path_buf()).unwrap();
        let id = sample_session_id();
        let path = tmp.path().join("deadbeefcafef00d.json");
        std::fs::write(&path, b"{ not json").unwrap();
        match store.load(&id) {
            Err(TranscriptError::Serialize(_)) => (),
            other => panic!("expected Serialize, got {other:?}"),
        }
    }

    #[test]
    fn store_load_errors_when_missing() {
        let tmp = TempDir::new().unwrap();
        let store = TranscriptStore::open(tmp.path().to_path_buf()).unwrap();
        let id = sample_session_id();
        match store.load(&id) {
            Err(TranscriptError::Io(e)) => assert_eq!(e.kind(), std::io::ErrorKind::NotFound),
            other => panic!("expected Io NotFound, got {other:?}"),
        }
    }

    #[test]
    fn store_list_empty_dir() {
        let tmp = TempDir::new().unwrap();
        let store = TranscriptStore::open(tmp.path().to_path_buf()).unwrap();
        assert!(store.list().unwrap().is_empty());
    }

    #[test]
    fn store_list_returns_sorted_ids() {
        let tmp = TempDir::new().unwrap();
        let store = TranscriptStore::open(tmp.path().to_path_buf()).unwrap();
        let a = SessionId::from_str("aaaaaaaaaaaaaaaa").unwrap();
        let b = SessionId::from_str("bbbbbbbbbbbbbbbb").unwrap();
        let c = SessionId::from_str("cccccccccccccccc").unwrap();
        // Save out of order.
        store
            .save_atomic(&sample_transcript(c.clone(), vec![]))
            .unwrap();
        store
            .save_atomic(&sample_transcript(a.clone(), vec![]))
            .unwrap();
        store
            .save_atomic(&sample_transcript(b.clone(), vec![]))
            .unwrap();
        // Junk that must be ignored.
        std::fs::write(tmp.path().join("not-a-session.json"), b"{}").unwrap();
        std::fs::write(tmp.path().join("notes.txt"), b"hi").unwrap();
        let listed = store.list().unwrap();
        assert_eq!(listed, vec![a, b, c]);
    }

    #[test]
    fn store_delete_returns_true_for_existing() {
        let tmp = TempDir::new().unwrap();
        let store = TranscriptStore::open(tmp.path().to_path_buf()).unwrap();
        let id = sample_session_id();
        store
            .save_atomic(&sample_transcript(id.clone(), vec![]))
            .unwrap();
        assert!(store.delete(&id).unwrap());
    }

    #[test]
    fn store_delete_returns_false_for_missing() {
        let tmp = TempDir::new().unwrap();
        let store = TranscriptStore::open(tmp.path().to_path_buf()).unwrap();
        let id = sample_session_id();
        assert!(!store.delete(&id).unwrap());
    }

    #[test]
    fn store_delete_removes_file_from_disk() {
        let tmp = TempDir::new().unwrap();
        let store = TranscriptStore::open(tmp.path().to_path_buf()).unwrap();
        let id = sample_session_id();
        let path = store
            .save_atomic(&sample_transcript(id.clone(), vec![]))
            .unwrap();
        assert!(path.exists());
        assert!(store.delete(&id).unwrap());
        assert!(!path.exists());
    }

    // ---- redact_pii (stub) ------------------------------------------------

    #[test]
    fn redact_pii_is_noop_today() {
        let mut turn = TranscriptTurn::User {
            at: fixed_time(),
            text: "secret".to_owned(),
        };
        let before = turn.clone();
        redact_pii(&mut turn);
        assert_eq!(turn, before);
    }

    // ---- Error Display + source -------------------------------------------

    #[test]
    fn transcript_error_display_smoke() {
        let io = TranscriptError::Io(std::io::Error::other("boom"));
        assert!(io.to_string().contains("transcript io error"));
        assert!(io.source().is_some());

        let ser: serde_json::Error = serde_json::from_str::<u8>("nope").unwrap_err();
        let s = TranscriptError::Serialize(ser);
        assert!(s.to_string().contains("transcript serialize error"));
        assert!(s.source().is_some());

        let v = TranscriptError::Validation("bad".to_owned());
        assert!(v.to_string().contains("validation"));
        assert!(v.source().is_none());

        let tl = TranscriptError::TooLarge { bytes: 99 };
        assert!(tl.to_string().contains("99"));
        assert!(tl.source().is_none());

        let sn = TranscriptError::SchemaNewer {
            found: 5,
            supported: 1,
        };
        assert!(sn.to_string().contains("newer than supported"));
        assert!(sn.source().is_none());

        let bs = TranscriptError::BadSessionId(SessionIdError::InvalidHex);
        assert!(bs.to_string().contains("filename"));
        assert!(bs.source().is_some());
    }

    #[test]
    fn session_id_error_display_smoke() {
        let wl = SessionIdError::WrongLength { actual: 4 }.to_string();
        assert!(wl.contains("16"));
        assert!(wl.contains('4'));
        let inv = SessionIdError::InvalidHex.to_string();
        assert!(inv.contains("hex"));
        let _: &dyn Error = &SessionIdError::InvalidHex;
    }

    #[test]
    fn transcript_error_from_conversions() {
        let _: TranscriptError = std::io::Error::from(std::io::ErrorKind::Other).into();
        let serde_err: serde_json::Error = serde_json::from_str::<u8>("nope").unwrap_err();
        let _: TranscriptError = serde_err.into();
        let _: TranscriptError = SessionIdError::InvalidHex.into();
    }

    // ---- Order preservation ----------------------------------------------

    #[test]
    fn round_trip_preserves_block_order() {
        let blocks = vec![
            TranscriptBlock {
                kind: TranscriptBlockKind::Text,
                text: "first".to_owned(),
            },
            TranscriptBlock {
                kind: TranscriptBlockKind::Code {
                    language: "rust".to_owned(),
                },
                text: "second".to_owned(),
            },
            TranscriptBlock {
                kind: TranscriptBlockKind::ToolCall {
                    tool_id: "fs.read".to_owned(),
                },
                text: "third".to_owned(),
            },
        ];
        let t = sample_transcript(
            sample_session_id(),
            vec![TranscriptTurn::Assistant {
                at: fixed_time(),
                blocks: blocks.clone(),
            }],
        );
        let json = serde_json::to_string(&t).unwrap();
        let back: Transcript = serde_json::from_str(&json).unwrap();
        match &back.turns[0] {
            TranscriptTurn::Assistant { blocks: b, .. } => assert_eq!(b, &blocks),
            other => panic!("unexpected turn shape: {other:?}"),
        }
    }

    #[test]
    fn round_trip_preserves_turn_order() {
        let turns = vec![
            TranscriptTurn::System {
                at: fixed_time(),
                text: "boot".to_owned(),
            },
            TranscriptTurn::User {
                at: fixed_time() + Duration::from_secs(1),
                text: "hi".to_owned(),
            },
            TranscriptTurn::Command {
                at: fixed_time() + Duration::from_secs(2),
                text: "/help".to_owned(),
                ok: false,
            },
            TranscriptTurn::Assistant {
                at: fixed_time() + Duration::from_secs(3),
                blocks: vec![],
            },
        ];
        let t = sample_transcript(sample_session_id(), turns.clone());
        let json = serde_json::to_string(&t).unwrap();
        let back: Transcript = serde_json::from_str(&json).unwrap();
        assert_eq!(back.turns, turns);
    }

    #[test]
    fn store_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<TranscriptStore>();
        assert_send_sync::<Transcript>();
        assert_send_sync::<SessionId>();
        assert_send_sync::<TranscriptError>();
    }
}
