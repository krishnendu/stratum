//! whisper.cpp subprocess scaffold.
//!
//! Phase 5 v2 — see `plan/05-multimodal.md` §Voice In. Stratum does NOT
//! link `whisper-rs` or any whisper crate; instead we shell out to an
//! optional `whisper` binary on `PATH`. When the binary is missing the
//! caller (today: the `/audio` palette command) degrades gracefully:
//! the audio attachment still rides on the next turn, only the
//! "[transcript: …]" prefix is replaced with an "unavailable" sentinel.
//!
//! ## Subprocess shape
//!
//! `whisper -f <input> -otxt -of <tmp-stem>`
//!
//! whisper.cpp writes the transcript to `<tmp-stem>.txt`. We read that
//! file back, return the trimmed contents, and best-effort delete the
//! tmp file. The `-of` argument lets us own the output path so we don't
//! collide with `<input>.txt` next to a user-controlled location.
//!
//! Timeout is a coarse wall-clock cap; the typical CPU-only `small`
//! model runs at ~5x real-time on modern laptops, so a 5-minute cap
//! covers an hour-long clip with headroom.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;

use crate::tool_dispatchers::which_in_path;

/// Errors a [`WhisperSubprocess::transcribe`] call can surface.
#[derive(Debug)]
pub enum WhisperError {
    /// No `whisper` binary on `PATH`. The caller should fall back to
    /// the "[audio transcript unavailable — install whisper.cpp]"
    /// surface text and keep the audio attachment in flight.
    MissingBinary,
    /// Subprocess failed to spawn at the OS layer.
    Spawn(std::io::Error),
    /// Subprocess exited non-zero. Captures the exit code and a tail of
    /// stderr for diagnosis (rendered in the chat command outcome).
    NonZero {
        /// OS exit code (or `-1` if the process was killed by signal).
        status: i32,
        /// Tail of stderr, capped at 256 chars.
        stderr_tail: String,
    },
    /// The wall-clock timeout fired before the subprocess exited.
    Timeout,
    /// Subprocess returned 0 but the `.txt` output file is missing or
    /// could not be read.
    OutputMissing(std::io::Error),
}

impl std::fmt::Display for WhisperError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingBinary => f.write_str("whisper binary not found on PATH"),
            Self::Spawn(e) => write!(f, "whisper spawn failed: {e}"),
            Self::NonZero {
                status,
                stderr_tail,
            } => write!(f, "whisper exit {status}: {stderr_tail}"),
            Self::Timeout => f.write_str("whisper timeout"),
            Self::OutputMissing(e) => write!(f, "whisper output unreadable: {e}"),
        }
    }
}

impl std::error::Error for WhisperError {}

/// Subprocess-backed whisper.cpp transcription.
///
/// Configured with the binary name to look up on `PATH` (default
/// `"whisper"`) and a per-call wall-clock timeout. Construct once,
/// reuse for many calls.
#[derive(Debug, Clone)]
pub struct WhisperSubprocess {
    binary: String,
    timeout: Duration,
}

impl Default for WhisperSubprocess {
    fn default() -> Self {
        Self::new()
    }
}

impl WhisperSubprocess {
    /// Default wall-clock cap for a single transcription. Covers the
    /// `small` model at ~5x real-time on a CPU host for clips up to
    /// ~25 minutes; bigger inputs can extend this via
    /// [`Self::with_timeout`].
    pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(300);

    /// New subprocess configured to look up the `whisper` binary on PATH.
    #[must_use]
    pub fn new() -> Self {
        Self {
            binary: "whisper".to_string(),
            timeout: Self::DEFAULT_TIMEOUT,
        }
    }

    /// Override the binary name. Useful for vendored builds shipped as
    /// `whisper-cli` or similar.
    #[must_use]
    pub fn with_binary(mut self, binary: impl Into<String>) -> Self {
        self.binary = binary.into();
        self
    }

    /// Override the per-call wall-clock timeout.
    #[must_use]
    pub const fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Inspect the configured binary name.
    #[must_use]
    pub fn binary(&self) -> &str {
        &self.binary
    }

    /// Inspect the per-call wall-clock timeout.
    #[must_use]
    pub const fn timeout(&self) -> Duration {
        self.timeout
    }

    /// Returns `true` iff the configured binary is resolvable on `PATH`.
    /// Cheaper than [`Self::transcribe`] when callers only need to know
    /// whether to render the "unavailable" sentinel.
    #[must_use]
    pub fn is_available(&self) -> bool {
        which_in_path(&self.binary).is_some()
    }

    /// Transcribe the audio file at `input` to text.
    ///
    /// Returns the trimmed transcript on success. On any error, callers
    /// in the TUI typically degrade gracefully — they keep the audio
    /// attachment but skip the transcript prefix.
    ///
    /// # Errors
    ///
    /// Returns [`WhisperError::MissingBinary`] when `whisper` is absent
    /// on `PATH`; [`WhisperError::Spawn`] when the subprocess could not
    /// be started; [`WhisperError::Timeout`] when wall-clock cap fired
    /// before exit; [`WhisperError::NonZero`] when whisper exited with
    /// a failure status; [`WhisperError::OutputMissing`] when the
    /// `.txt` output file is unreadable after a successful run.
    pub fn transcribe(&self, input: &Path) -> Result<String, WhisperError> {
        let bin = which_in_path(&self.binary).ok_or(WhisperError::MissingBinary)?;

        // Pick a unique tmp stem so `-of <stem>` doesn't collide with
        // sibling runs. whisper.cpp appends `.txt`; we manage the full
        // file lifecycle.
        let tmp_dir = std::env::temp_dir();
        let pid = std::process::id();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let stem = format!("stratum-whisper-{pid}-{nanos}");
        let out_stem: PathBuf = tmp_dir.join(&stem);
        let out_txt: PathBuf = tmp_dir.join(format!("{stem}.txt"));

        let mut cmd = Command::new(&bin);
        cmd.arg("-f")
            .arg(input)
            .arg("-otxt")
            .arg("-of")
            .arg(&out_stem)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let mut child = cmd.spawn().map_err(WhisperError::Spawn)?;

        // Poll wait — keep dep surface flat, mirror the pattern used by
        // `wait_with_timeout` in `tool_dispatchers`.
        let start = std::time::Instant::now();
        let poll = Duration::from_millis(50);
        let status = loop {
            match child.try_wait() {
                Ok(Some(status)) => break status,
                Ok(None) => {
                    if start.elapsed() >= self.timeout {
                        let _ = child.kill();
                        let _ = child.wait();
                        let _ = std::fs::remove_file(&out_txt);
                        return Err(WhisperError::Timeout);
                    }
                    std::thread::sleep(poll);
                }
                Err(e) => return Err(WhisperError::Spawn(e)),
            }
        };

        if !status.success() {
            let mut stderr_buf = Vec::new();
            if let Some(mut s) = child.stderr.take() {
                use std::io::Read;
                let _ = s.read_to_end(&mut stderr_buf);
            }
            let tail = String::from_utf8_lossy(&stderr_buf);
            // Capture up to the last 256 chars of stderr without
            // mirroring the order — collect to Vec<char>, slice the tail.
            let chars: Vec<char> = tail.chars().collect();
            let start = chars.len().saturating_sub(256);
            let stderr_tail: String = chars[start..].iter().collect();
            let _ = std::fs::remove_file(&out_txt);
            return Err(WhisperError::NonZero {
                status: status.code().unwrap_or(-1),
                stderr_tail,
            });
        }

        let text = std::fs::read_to_string(&out_txt).map_err(WhisperError::OutputMissing)?;
        let _ = std::fs::remove_file(&out_txt);
        Ok(text.trim().to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_binary_yields_typed_error() {
        // Configure the subprocess with a binary name that cannot exist
        // on PATH so we deterministically hit the MissingBinary arm
        // without depending on the test host having (or not having) the
        // real `whisper` installed.
        let w = WhisperSubprocess::new().with_binary("stratum_no_such_whisper_binary_xyzzy_12345");
        assert!(!w.is_available());
        let err = w.transcribe(Path::new("/tmp/anything.wav")).unwrap_err();
        assert!(matches!(err, WhisperError::MissingBinary), "got {err:?}");
    }

    #[test]
    fn defaults_round_trip() {
        let w = WhisperSubprocess::default();
        assert_eq!(w.binary(), "whisper");
        assert_eq!(w.timeout(), WhisperSubprocess::DEFAULT_TIMEOUT);
    }

    #[test]
    fn with_timeout_round_trips() {
        let w = WhisperSubprocess::new().with_timeout(Duration::from_millis(123));
        assert_eq!(w.timeout(), Duration::from_millis(123));
    }

    #[test]
    fn with_binary_round_trips() {
        let w = WhisperSubprocess::new().with_binary("whisper-cli");
        assert_eq!(w.binary(), "whisper-cli");
    }

    #[test]
    fn whisper_error_renders_for_each_variant() {
        // Cheap smoke: confirm Display doesn't panic for any variant.
        let _ = format!("{}", WhisperError::MissingBinary);
        let _ = format!("{}", WhisperError::Spawn(std::io::Error::other("x")),);
        let _ = format!(
            "{}",
            WhisperError::NonZero {
                status: 1,
                stderr_tail: "boom".to_string(),
            },
        );
        let _ = format!("{}", WhisperError::Timeout);
        let _ = format!(
            "{}",
            WhisperError::OutputMissing(std::io::Error::new(std::io::ErrorKind::NotFound, "x")),
        );
    }

    #[test]
    fn whisper_subprocess_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<WhisperSubprocess>();
    }
}
