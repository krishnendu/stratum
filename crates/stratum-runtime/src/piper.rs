//! piper TTS subprocess scaffold.
//!
//! Phase 5 v2 — see `plan/05-multimodal.md` §Voice Out. Stratum does NOT
//! link a TTS crate; instead we shell out to an optional `piper` binary
//! on `PATH` (see <https://github.com/rhasspy/piper>). When the binary or
//! model is missing the caller (today: the `/say` palette command)
//! degrades gracefully — the rendered text is still surfaced, only the
//! WAV synthesis is skipped.
//!
//! ## Subprocess shape
//!
//! `piper --model <onnx> --output_file <wav>`
//!
//! piper reads the text to synthesize on stdin (line-terminated) and
//! writes a RIFF/WAV file at `<wav>`. We pick a unique tempfile for the
//! output and return its path; the caller owns deletion (typically after
//! handing the bytes to the audio sink).
//!
//! Timeout is a coarse wall-clock cap; piper's small voices synthesize
//! at ~10x real-time on CPU, so the default 2-minute cap covers a
//! ~20-minute monologue with headroom.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;

use crate::tool_dispatchers::which_in_path;

/// Errors a [`PiperSubprocess::synthesize`] call can surface.
#[derive(Debug)]
pub enum PiperError {
    /// No `piper` binary on `PATH` (or the configured absolute path does
    /// not exist). Callers should degrade to a text-only surface.
    MissingBinary,
    /// The configured ONNX voice model does not exist on disk. piper
    /// would otherwise fail with an opaque non-zero exit; surface this
    /// distinctly so callers can prompt the user to install a voice.
    MissingModel,
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
    /// Subprocess returned 0 but the `.wav` output file is missing or
    /// could not be opened.
    OutputMissing(std::io::Error),
}

impl std::fmt::Display for PiperError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingBinary => f.write_str("piper binary not found on PATH"),
            Self::MissingModel => f.write_str("piper model file not found"),
            Self::Spawn(e) => write!(f, "piper spawn failed: {e}"),
            Self::NonZero {
                status,
                stderr_tail,
            } => write!(f, "piper exit {status}: {stderr_tail}"),
            Self::Timeout => f.write_str("piper timeout"),
            Self::OutputMissing(e) => write!(f, "piper output unreadable: {e}"),
        }
    }
}

impl std::error::Error for PiperError {}

/// Subprocess-backed piper text-to-speech.
///
/// Configured with the binary name to look up on `PATH` (default
/// `"piper"`), an ONNX voice model path, and a per-call wall-clock
/// timeout. Construct once, reuse for many calls.
#[derive(Debug, Clone)]
pub struct PiperSubprocess {
    binary: String,
    model: PathBuf,
    timeout: Duration,
}

impl PiperSubprocess {
    /// Default wall-clock cap for a single synthesis call. Covers a
    /// small voice at ~10x real-time on a CPU host for clips up to
    /// ~20 minutes; bigger inputs can extend this via
    /// [`Self::with_timeout`].
    pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(120);

    /// New subprocess configured to look up the `piper` binary on PATH,
    /// pointing at the given ONNX voice model.
    #[must_use]
    pub fn new(model: impl Into<PathBuf>) -> Self {
        Self {
            binary: "piper".to_string(),
            model: model.into(),
            timeout: Self::DEFAULT_TIMEOUT,
        }
    }

    /// Override the binary name. Useful for vendored builds shipped as
    /// `piper-tts` or similar.
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

    /// Override the ONNX voice model path.
    #[must_use]
    pub fn with_model(mut self, model: impl Into<PathBuf>) -> Self {
        self.model = model.into();
        self
    }

    /// Inspect the configured binary name.
    #[must_use]
    pub fn binary(&self) -> &str {
        &self.binary
    }

    /// Inspect the configured voice-model path.
    #[must_use]
    pub fn model(&self) -> &Path {
        &self.model
    }

    /// Inspect the per-call wall-clock timeout.
    #[must_use]
    pub const fn timeout(&self) -> Duration {
        self.timeout
    }

    /// Returns `true` iff the configured binary is resolvable.
    ///
    /// Single-component names (no directory separator) are looked up on
    /// `PATH`; multi-component paths (containing any `Path` component
    /// beyond the leaf) are checked for file existence directly so tests
    /// — and users with a non-`PATH` install — can point at an absolute
    /// binary. Cheaper than [`Self::synthesize`] when callers only need
    /// to know whether to render the "unavailable" sentinel.
    #[must_use]
    pub fn is_available(&self) -> bool {
        if Path::new(&self.binary).components().count() > 1 {
            Path::new(&self.binary).is_file()
        } else {
            which_in_path(&self.binary).is_some()
        }
    }

    /// Synthesize `text` to a WAV file and return its path.
    ///
    /// The caller owns the returned tempfile and is responsible for
    /// deletion (typically after streaming bytes to the audio sink).
    ///
    /// # Errors
    ///
    /// Returns [`PiperError::MissingBinary`] when `piper` is absent on
    /// `PATH` (or the configured absolute path does not exist);
    /// [`PiperError::MissingModel`] when the ONNX voice model is
    /// missing; [`PiperError::Spawn`] when the subprocess could not be
    /// started; [`PiperError::Timeout`] when the wall-clock cap fired
    /// before exit; [`PiperError::NonZero`] when piper exited with a
    /// failure status; [`PiperError::OutputMissing`] when the `.wav`
    /// output file is unreadable after a successful run.
    pub fn synthesize(&self, text: &str) -> Result<PathBuf, PiperError> {
        // Resolve the binary: single-component names go through PATH
        // lookup; multi-component paths are accepted as-is so tests /
        // sideloads can target a binary outside `PATH`. Component count
        // is platform-agnostic — works for both `/usr/bin/piper` and
        // `C:\tools\piper.exe` without a separator-character branch.
        let bin: PathBuf = if Path::new(&self.binary).components().count() > 1 {
            let p = Path::new(&self.binary);
            if !p.is_file() {
                return Err(PiperError::MissingBinary);
            }
            p.to_path_buf()
        } else {
            which_in_path(&self.binary).ok_or(PiperError::MissingBinary)?
        };

        if !self.model.is_file() {
            return Err(PiperError::MissingModel);
        }

        // Pick a unique tmp path so concurrent calls don't collide.
        let tmp_dir = std::env::temp_dir();
        let pid = std::process::id();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let out_wav: PathBuf = tmp_dir.join(format!("stratum-piper-{pid}-{nanos}.wav"));

        let mut cmd = Command::new(&bin);
        cmd.arg("--model")
            .arg(&self.model)
            .arg("--output_file")
            .arg(&out_wav)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let mut child = cmd.spawn().map_err(PiperError::Spawn)?;

        // Feed the text + trailing newline on stdin so piper's line
        // reader flushes a single synthesis batch. We close stdin by
        // dropping the handle so piper sees EOF and exits.
        if let Some(mut stdin) = child.stdin.take() {
            let _ = stdin.write_all(text.as_bytes());
            let _ = stdin.write_all(b"\n");
            drop(stdin);
        }

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
                        let _ = std::fs::remove_file(&out_wav);
                        return Err(PiperError::Timeout);
                    }
                    std::thread::sleep(poll);
                }
                Err(e) => return Err(PiperError::Spawn(e)),
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
            let _ = std::fs::remove_file(&out_wav);
            return Err(PiperError::NonZero {
                status: status.code().unwrap_or(-1),
                stderr_tail,
            });
        }

        // Confirm the WAV exists and is openable; surface as
        // OutputMissing if not. We don't read the contents — caller
        // owns the bytes (and deletion).
        std::fs::File::open(&out_wav).map_err(PiperError::OutputMissing)?;
        Ok(out_wav)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_binary_yields_typed_error() {
        // Configure with a binary name that cannot exist on PATH so we
        // deterministically hit the MissingBinary arm without depending
        // on the test host having (or not having) the real `piper`
        // installed.
        let p = PiperSubprocess::new("/tmp/whatever.onnx")
            .with_binary("stratum_no_such_piper_binary_xyzzy_12345");
        assert!(!p.is_available());
        let err = p.synthesize("hello").unwrap_err();
        assert!(matches!(err, PiperError::MissingBinary), "got {err:?}");
    }

    #[test]
    fn defaults_round_trip() {
        let p = PiperSubprocess::new("/some/voice.onnx");
        assert_eq!(p.binary(), "piper");
        assert_eq!(p.model(), Path::new("/some/voice.onnx"));
        assert_eq!(p.timeout(), PiperSubprocess::DEFAULT_TIMEOUT);
    }

    #[test]
    fn with_timeout_round_trips() {
        let p = PiperSubprocess::new("/v.onnx").with_timeout(Duration::from_millis(123));
        assert_eq!(p.timeout(), Duration::from_millis(123));
    }

    #[test]
    fn with_binary_round_trips() {
        let p = PiperSubprocess::new("/v.onnx").with_binary("piper-tts");
        assert_eq!(p.binary(), "piper-tts");
    }

    #[test]
    fn with_model_round_trips() {
        let p = PiperSubprocess::new("/v.onnx").with_model("/other/voice.onnx");
        assert_eq!(p.model(), Path::new("/other/voice.onnx"));
    }

    #[test]
    fn piper_error_renders_for_each_variant() {
        // Cheap smoke: confirm Display doesn't panic for any variant.
        let _ = format!("{}", PiperError::MissingBinary);
        let _ = format!("{}", PiperError::MissingModel);
        let _ = format!("{}", PiperError::Spawn(std::io::Error::other("x")),);
        let _ = format!(
            "{}",
            PiperError::NonZero {
                status: 1,
                stderr_tail: "boom".to_string(),
            },
        );
        let _ = format!("{}", PiperError::Timeout);
        let _ = format!(
            "{}",
            PiperError::OutputMissing(std::io::Error::new(std::io::ErrorKind::NotFound, "x")),
        );
    }

    #[test]
    fn piper_subprocess_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<PiperSubprocess>();
    }

    // The script-driven tests below stand up a fake piper via a shell
    // script in a tempdir and point `PiperSubprocess` at the absolute
    // path. They cover the `synthesize()` body — spawn, poll, success,
    // NonZero, Timeout, OutputMissing — without depending on a real
    // piper install on the test host.
    #[cfg(unix)]
    mod script_driven {
        use std::os::unix::fs::PermissionsExt;
        use std::path::PathBuf;

        use tempfile::TempDir;

        use super::*;

        /// Write a `#!/bin/sh` script under `tmp` with mode 0o755 and
        /// return its absolute path. The script body is appended after a
        /// `#!/bin/sh\n` shebang line.
        fn write_script(tmp: &TempDir, name: &str, body: &str) -> PathBuf {
            let p = tmp.path().join(name);
            std::fs::write(&p, format!("#!/bin/sh\n{body}")).expect("write script");
            std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755))
                .expect("chmod script");
            p
        }

        /// Convert a tempdir-backed script path to `&str` without lossy
        /// substitution. Fails loudly if the temp directory is non-UTF-8
        /// rather than silently substituting `U+FFFD` and producing a
        /// path that no longer exists.
        fn script_str(p: &Path) -> &str {
            p.to_str()
                .expect("tempdir-backed script path must be valid UTF-8")
        }

        /// Build a fake voice-model file under `tmp` so the
        /// `MissingModel` guard passes.
        fn make_model(tmp: &TempDir) -> PathBuf {
            let m = tmp.path().join("voice.onnx");
            std::fs::write(&m, b"fake-onnx").expect("model");
            m
        }

        #[test]
        fn happy_path_writes_wav_and_returns_path() {
            let tmp = TempDir::new().expect("tmp");
            // The fake piper writes WAV bytes to `$5`, the value of the
            // `--output_file` arg. Args (1-indexed): 1=`--model`,
            // 2=<model>, 3=`--output_file`, 4=<wav>.
            //
            // NOTE: $5 is unused in piper's real argv shape, but we
            // accept any argv slot pointing at the output file. To keep
            // the test aligned with whisper's "$5 -> .txt" convention we
            // shift the args by inserting a leading stdin drain.
            let script = write_script(
                &tmp,
                "fake-piper.sh",
                // Drain stdin to /dev/null so the test isn't sensitive
                // to whether piper closes stdout before stdin reads.
                // Then write fake WAV bytes to the `--output_file` arg
                // (which is $4 with our `--model <m> --output_file <w>`
                // shape).
                "cat > /dev/null\nprintf 'RIFFfake' > \"$4\"\nexit 0\n",
            );
            let model = make_model(&tmp);
            let p = PiperSubprocess::new(&model).with_binary(script_str(&script));
            assert!(p.is_available(), "absolute path should be available");
            let out = p.synthesize("hello world").expect("synthesize");
            let bytes = std::fs::read(&out).expect("read out");
            assert_eq!(&bytes, b"RIFFfake", "wav contents round-trip");
            // Caller owns deletion.
            let _ = std::fs::remove_file(&out);
        }

        #[test]
        fn nonzero_exit_captures_stderr_tail() {
            let tmp = TempDir::new().expect("tmp");
            let script = write_script(
                &tmp,
                "fake-piper.sh",
                "cat > /dev/null\necho 'piper boom on stderr' 1>&2\nexit 7\n",
            );
            let model = make_model(&tmp);
            let p = PiperSubprocess::new(&model).with_binary(script_str(&script));
            let err = p.synthesize("anything").expect_err("expected NonZero");
            match err {
                PiperError::NonZero {
                    status,
                    stderr_tail,
                } => {
                    assert_eq!(status, 7);
                    assert!(
                        stderr_tail.contains("piper boom on stderr"),
                        "stderr_tail = {stderr_tail:?}"
                    );
                }
                other => panic!("expected NonZero, got {other:?}"),
            }
        }

        #[test]
        fn timeout_kills_child_and_returns_typed_error() {
            let tmp = TempDir::new().expect("tmp");
            // Sleep longer than the configured timeout so the poll loop
            // hits its wall-clock cap.
            let script = write_script(&tmp, "fake-piper.sh", "cat > /dev/null\nsleep 5\n");
            let model = make_model(&tmp);
            let p = PiperSubprocess::new(&model)
                .with_binary(script_str(&script))
                // Bumped above the 50 ms poll interval by ~10× so a busy
                // CI host that takes >50 ms to execve /bin/sh + reach
                // the first poll still has comfortable margin before
                // the wall-clock cap fires. The child sleeps 5 s so kill
                // is fast regardless of the budget.
                .with_timeout(Duration::from_millis(500));
            let err = p.synthesize("anything").expect_err("expected Timeout");
            assert!(matches!(err, PiperError::Timeout), "got {err:?}");
        }

        #[test]
        fn success_but_missing_output_returns_output_missing() {
            let tmp = TempDir::new().expect("tmp");
            // Exit 0 without writing the `.wav`, exercising the
            // OutputMissing arm.
            let script = write_script(&tmp, "fake-piper.sh", "cat > /dev/null\nexit 0\n");
            let model = make_model(&tmp);
            let p = PiperSubprocess::new(&model).with_binary(script_str(&script));
            let err = p
                .synthesize("anything")
                .expect_err("expected OutputMissing");
            assert!(matches!(err, PiperError::OutputMissing(_)), "got {err:?}");
        }

        #[test]
        fn absolute_path_to_missing_file_yields_missing_binary() {
            let tmp = TempDir::new().expect("tmp");
            let model = make_model(&tmp);
            let p =
                PiperSubprocess::new(&model).with_binary("/tmp/definitely-not-a-real-piper-binary");
            assert!(!p.is_available());
            let err = p
                .synthesize("anything")
                .expect_err("expected MissingBinary");
            assert!(matches!(err, PiperError::MissingBinary), "got {err:?}");
        }

        #[test]
        fn missing_model_yields_typed_error() {
            let tmp = TempDir::new().expect("tmp");
            // Real binary (the script) but no model on disk.
            let script = write_script(&tmp, "fake-piper.sh", "cat > /dev/null\nexit 0\n");
            let p = PiperSubprocess::new(tmp.path().join("does-not-exist.onnx"))
                .with_binary(script_str(&script));
            assert!(p.is_available(), "binary should resolve");
            let err = p.synthesize("anything").expect_err("expected MissingModel");
            assert!(matches!(err, PiperError::MissingModel), "got {err:?}");
        }
    }
}
