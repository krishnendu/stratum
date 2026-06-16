//! Phase 5 exit test — voice-driven 5-turn agentic dialogue.
//!
//! See `plan/05-multimodal.md` §Exit. This locks down the assertion the
//! phase's exit criterion bakes in: a user can carry a 5-turn dialogue
//! to resolution by attaching audio per turn, with no keyboard-typed
//! semantic content. The model surface is the `EchoProvider` floor so
//! the test runs everywhere; the whisper subprocess is replaced with a
//! tiny `#!/bin/sh` script on a tempdir-prepended `PATH` that maps a
//! marker byte at offset 12 of each WAV to the matching transcript.
//!
//! What this exercises end-to-end:
//!
//! * `/audio <path>` stages a `Block::Audio` attachment AND populates
//!   `staged_audio_transcript` from the (faked) whisper subprocess.
//! * `submit()` drains both into the next user turn, wrapping the
//!   transcript in the `[AUDIO_TRANSCRIPT_BEGIN] … [AUDIO_TRANSCRIPT_END]`
//!   fence the chat surface inserts.
//! * The 5-turn conversation reaches turn 5 with the transcript history
//!   ordered as `phrase_1 … phrase_5`.
//! * Each turn's `last_turn_attachments` carries exactly one
//!   `Block::Audio` with mime `audio/wav`.
//!
//! What is intentionally NOT exercised:
//!
//! * Real whisper.cpp transcription. CI hosts don't ship the model and
//!   the binary is platform-specific — the fake script + marker byte is
//!   the documented swap-in seam.
//! * The mic/PTT capture path. That has its own unit coverage in
//!   `stratum-runtime::mic`; here we feed pre-recorded WAVs because the
//!   integration target is the chat-surface plumbing, not cpal.

// Integration test binary: every fn here exists only for `cargo test`. The
// helpers below panic on setup failures by design; clippy's `expect_used` /
// `unwrap_used` denials only apply to non-test code.
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "integration test helpers may panic on setup failures"
)]
// This file currently runs only on Unix because the fake whisper is a
// `#!/bin/sh` script. Windows would need a `.bat` shim with the same
// marker-byte → transcript mapping; out of scope for the exit test.
#![cfg(unix)]

use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

use stratum_runtime::{EchoProvider, Tier};
use stratum_tui::chat::{ChatState, Turn};
use stratum_types::Block;
use tempfile::TempDir;

/// The five canned voice phrases the user "speaks" across the dialogue.
/// Order is the assertion contract: turn N produces `PHRASES[N - 1]`.
const PHRASES: [&str; 5] = [
    "list directory",
    "open the readme",
    "summarise it",
    "is there a test directory",
    "thanks done",
];

/// Build a minimal RIFF/WAVE byte sequence with a marker byte embedded
/// at offset 12. The 12-byte RIFF/WAVE prefix is what
/// `sniff_audio_mime_chat` looks at to classify the file as
/// `audio/wav`; the marker rides immediately after so the fake whisper
/// script can `dd bs=1 skip=12 count=1` to pick a transcript.
fn synthetic_wav_with_marker(marker: u8) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(16);
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&[0u8; 4]);
    bytes.extend_from_slice(b"WAVE");
    // Marker byte. The fake whisper reads exactly this byte to choose
    // which transcript to emit; any value 1..=5 routes deterministically.
    bytes.push(marker);
    // Pad so the total is non-trivial — keeps the on-disk size readable
    // in the chat command's acknowledgement message ("audio staged (N
    // bytes, audio/wav)").
    bytes.extend_from_slice(&[0u8; 3]);
    bytes
}

/// Write the fake whisper shell script into `dir` and chmod it 0o755.
///
/// The script signature mirrors real whisper.cpp: `whisper -f <input>
/// -otxt -of <stem>`. The arguments land at positions $1..$5; we read
/// the marker byte from `$2` and write the matching transcript to
/// `$5.txt` (whisper.cpp appends `.txt` to the `-of` stem).
fn write_fake_whisper(dir: &Path) -> PathBuf {
    let p = dir.join("whisper");
    // The marker byte is read with `dd bs=1 skip=12 count=1` then piped
    // to `od -An -tu1` to print the unsigned decimal value (e.g. ` 3`).
    // `tr -d ' '` strips the leading padding so `case` matches cleanly.
    let body = r#"#!/bin/sh
set -e
INPUT="$2"
STEM="$5"
MARK="$(dd if="$INPUT" bs=1 skip=12 count=1 2>/dev/null | od -An -tu1 | tr -d ' \n')"
case "$MARK" in
  1) TEXT="list directory" ;;
  2) TEXT="open the readme" ;;
  3) TEXT="summarise it" ;;
  4) TEXT="is there a test directory" ;;
  5) TEXT="thanks done" ;;
  *) TEXT="" ;;
esac
printf '%s\n' "$TEXT" > "${STEM}.txt"
exit 0
"#;
    std::fs::write(&p, body).expect("write fake whisper");
    std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755))
        .expect("chmod fake whisper");
    p
}

/// Prepend `dir` to the process `PATH` for the lifetime of the returned
/// guard. `WhisperSubprocess::new()` walks `PATH` for `"whisper"`; this
/// is how we make the fake script "the" whisper without touching the
/// (crate-private) `whisper` field on `ChatState`.
struct PathPrepend {
    previous: Option<std::ffi::OsString>,
}

impl PathPrepend {
    fn new(dir: &Path) -> Self {
        let previous = std::env::var_os("PATH");
        let new_path = previous.as_ref().map_or_else(
            || std::ffi::OsString::from(dir),
            |prev| {
                let mut joined = std::ffi::OsString::from(dir);
                joined.push(":");
                joined.push(prev);
                joined
            },
        );
        // `set_var` is process-global. The integration test runs
        // single-threaded inside its own binary and restores the
        // previous PATH on drop.
        std::env::set_var("PATH", &new_path);
        Self { previous }
    }
}

impl Drop for PathPrepend {
    fn drop(&mut self) {
        match self.previous.take() {
            Some(prev) => std::env::set_var("PATH", prev),
            None => std::env::remove_var("PATH"),
        }
    }
}

/// Restore the process working directory on drop. The chat surface's
/// `/audio` dispatcher canonicalises the WAV path against `current_dir`
/// and rejects anything outside it, so the test has to `chdir` into the
/// tempdir for the duration of the dialogue.
struct CwdGuard {
    previous: PathBuf,
}

impl CwdGuard {
    fn enter(target: &Path) -> Self {
        let previous = std::env::current_dir().expect("cwd");
        std::env::set_current_dir(target).expect("chdir into tempdir");
        Self { previous }
    }
}

impl Drop for CwdGuard {
    fn drop(&mut self) {
        let _ = std::env::set_current_dir(&self.previous);
    }
}

#[test]
fn voice_driven_five_turn_dialogue_resolves_with_no_keyboard() {
    let tmp = TempDir::new().expect("tempdir");

    // Stage the 5 WAVs in the workspace cwd so the chat surface accepts
    // them as relative paths under cwd.
    let cwd_guard = CwdGuard::enter(tmp.path());
    for (i, _) in PHRASES.iter().enumerate() {
        let marker = u8::try_from(i + 1).expect("marker fits in u8");
        let bytes = synthetic_wav_with_marker(marker);
        std::fs::write(tmp.path().join(format!("voice{}.wav", i + 1)), &bytes)
            .expect("write wav fixture");
    }

    // Stage the fake whisper in its own tempdir, then prepend that dir
    // to PATH so `WhisperSubprocess::new()`'s `which_in_path("whisper")`
    // resolves to our script.
    let whisper_dir = TempDir::new().expect("whisper tempdir");
    let _whisper_path = write_fake_whisper(whisper_dir.path());
    let _path_guard = PathPrepend::new(whisper_dir.path());

    let mut state = ChatState::new(EchoProvider::new("echo: "), Tier::High, "ready".to_string());

    // Drive 5 turns. Each turn:
    //   1. /audio voice{N}.wav — stages Block::Audio + transcript.
    //   2. submit_with_prompt(".") — drains both into the user turn.
    //      The "." is a non-whitespace placeholder so submit() doesn't
    //      bail on the empty-input fast path; the *semantic* content of
    //      the turn comes from the fenced transcript.
    for (i, expected_phrase) in PHRASES.iter().enumerate() {
        let cmd = format!("/audio voice{}.wav", i + 1);
        let _ = state.execute_palette_command(&cmd);
        assert_eq!(
            state.staged_audio_transcript(),
            Some(*expected_phrase),
            "turn {}: whisper should produce {expected_phrase:?}",
            i + 1,
        );
        state.submit_with_prompt(".");

        // The audio attachment rode along on this turn and only this
        // turn — the chat surface clears `staged_audio` on submit.
        assert!(
            state.staged_audio().is_none(),
            "turn {}: staged audio should drain on submit",
            i + 1,
        );
        let attachments = state.last_turn_attachments();
        assert_eq!(
            attachments.len(),
            1,
            "turn {}: exactly one audio attachment per turn",
            i + 1,
        );
        assert!(
            matches!(&attachments[0], Block::Audio { mime, .. } if mime == "audio/wav"),
            "turn {}: attachment should be audio/wav",
            i + 1,
        );
    }

    // Walk the transcript and pull out the 5 user turns in order. Each
    // user turn must carry the fence + the matching phrase. The
    // EchoProvider's assistant turns are interleaved but are not the
    // contract this test locks down.
    let user_turns: Vec<String> = state
        .transcript()
        .iter()
        .filter_map(|t| match t {
            Turn::User(text) => Some(text.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(
        user_turns.len(),
        PHRASES.len(),
        "expected exactly 5 user turns, got {}: {:?}",
        user_turns.len(),
        user_turns,
    );
    for (i, (turn, expected_phrase)) in user_turns.iter().zip(PHRASES.iter()).enumerate() {
        assert!(
            turn.contains("[AUDIO_TRANSCRIPT_BEGIN]"),
            "turn {}: user message missing BEGIN fence: {turn:?}",
            i + 1,
        );
        assert!(
            turn.contains("[AUDIO_TRANSCRIPT_END]"),
            "turn {}: user message missing END fence: {turn:?}",
            i + 1,
        );
        assert!(
            turn.contains(expected_phrase),
            "turn {}: user message missing phrase {expected_phrase:?}: {turn:?}",
            i + 1,
        );
    }

    // Drop guards explicitly so PATH / cwd restoration is part of the
    // test body (not the post-test teardown unwind), making any
    // restoration panic surface as a test failure rather than a noisy
    // process exit.
    drop(cwd_guard);
}
