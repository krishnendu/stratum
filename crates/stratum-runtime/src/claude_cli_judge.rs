//! `ClaudeCliJudge` — subprocess transport for the Stratum LLM-judge.
//!
//! Phase 4+ scaffold. The default Stratum eval-judge runs `claude -p
//! <prompt>` as a subprocess (riding the user's Claude Code subscription)
//! rather than calling the metered Anthropic HTTP API. This module is
//! pure subprocess plumbing: prompt synthesis, structured spawn, stdout
//! parsing, stderr tail capture, wall-clock timeout.
//!
//! The `claude` CLI is expected to be on `PATH` (or pinned via
//! [`ClaudeCliJudge::with_binary`]). Tests substitute a tiny shell
//! script that echoes a pre-baked JSON verdict line — no real network
//! calls, no real `claude` install.
//!
//! ## Wire format
//!
//! The judge writes exactly one JSON line on stdout, one of:
//!
//! ```text
//! {"result":"pass"}
//! {"result":"fail","reasons":["..."]}
//! {"result":"ambiguous","notes":"..."}
//! ```
//!
//! The first stdout line beginning with `{` is parsed; everything else
//! (banners, debug logs, scratch output) is ignored.

// xtask-check-error-codes: ignore-file
//
// Reason: this module ships local `JudgeError` variants rather than
// catalog `STRAT-E####` entries. The judge is a v2-scaffold transport;
// promoting these to the catalog happens when the eval-runner wires
// the judge in for real.

use std::fmt;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

/// Default cap on retained stderr text: last 64 KiB (≈ ~1k lines of
/// CLI chatter). Mirrors the
/// [`crate::mcp::STDERR_TAIL_CAP`]-style policy used elsewhere in the
/// runtime.
const STDERR_TAIL_BYTES: usize = 64 * 1024;
/// Default cap on retained stderr lines.
const STDERR_TAIL_LINES: usize = 200;
/// Polling interval while waiting for the subprocess's
/// [`std::process::Output`] to come back through the channel.
const WAIT_POLL: Duration = Duration::from_millis(25);

/// Subprocess-backed Claude judge.
///
/// Spawns `<binary> -p <prompt> <extra_args>`, reads stdout, and parses
/// the first JSON line as a [`JudgeVerdict`]. The wall-clock timeout is
/// enforced by waiting for the child's `Output` via a worker thread +
/// `recv_timeout`; on timeout the helper thread is detached (the OS
/// reaps the child when the parent exits, and the spawned shell script
/// in real use will be `sleep`-bound).
#[derive(Debug, Clone)]
pub struct ClaudeCliJudge {
    binary: PathBuf,
    timeout: Duration,
    extra_args: Vec<String>,
}

impl Default for ClaudeCliJudge {
    fn default() -> Self {
        Self::new()
    }
}

impl ClaudeCliJudge {
    /// Build a new judge defaulting to the `claude` binary on `PATH`
    /// and a 60-second per-call timeout.
    #[must_use]
    pub fn new() -> Self {
        Self {
            binary: PathBuf::from("claude"),
            timeout: Duration::from_secs(60),
            extra_args: Vec::new(),
        }
    }

    /// Override the `claude` binary path (useful for tests pointing at
    /// a fake shell script, or for pinning a specific install).
    #[must_use]
    pub fn with_binary(mut self, binary: PathBuf) -> Self {
        self.binary = binary;
        self
    }

    /// Override the per-call wall-clock timeout.
    #[must_use]
    pub const fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Append extra CLI args after the `-p <prompt>` pair.
    #[must_use]
    pub fn with_extra_args(mut self, extra_args: Vec<String>) -> Self {
        self.extra_args = extra_args;
        self
    }

    /// Inspect the configured binary path.
    #[must_use]
    pub const fn binary(&self) -> &PathBuf {
        &self.binary
    }

    /// Inspect the configured timeout.
    #[must_use]
    pub const fn timeout(&self) -> Duration {
        self.timeout
    }

    /// Inspect the configured extra args.
    #[must_use]
    pub fn extra_args(&self) -> &[String] {
        &self.extra_args
    }

    /// Run the judge end-to-end against `prompt`.
    ///
    /// # Errors
    /// - [`JudgeError::BinaryMissing`] if the `claude` binary cannot be
    ///   found on `PATH`.
    /// - [`JudgeError::Spawn`] for any other spawn IO failure.
    /// - [`JudgeError::Timeout`] when the child has not exited within
    ///   the configured wall-clock budget.
    /// - [`JudgeError::BadExit`] when the child exits non-zero.
    /// - [`JudgeError::BadResponse`] when no parseable verdict line
    ///   appears in stdout.
    pub fn judge(&self, prompt: &JudgePrompt) -> Result<JudgeResponse, JudgeError> {
        let synth = synth_prompt(prompt);
        let mut cmd = Command::new(&self.binary);
        cmd.arg("-p").arg(&synth);
        for extra in &self.extra_args {
            cmd.arg(extra);
        }
        cmd.stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let started = Instant::now();
        let child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => {
                if e.kind() == std::io::ErrorKind::NotFound {
                    return Err(JudgeError::BinaryMissing(format!(
                        "claude binary not found at {}",
                        self.binary.display()
                    )));
                }
                return Err(JudgeError::Spawn(e));
            }
        };

        // Drive `child.wait_with_output()` on a helper thread so the
        // main thread can recv_timeout. The helper returns the full
        // `Output` (status + stdout + stderr) once the child exits.
        let (tx, rx) = mpsc::channel::<std::io::Result<std::process::Output>>();
        let timeout = self.timeout;
        let handle = thread::spawn(move || {
            let result = child.wait_with_output();
            let _ = tx.send(result);
        });

        let output_result: Option<std::io::Result<std::process::Output>> = loop {
            match rx.recv_timeout(WAIT_POLL) {
                Ok(out) => break Some(out),
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    if started.elapsed() >= timeout {
                        break None;
                    }
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    break Some(Err(std::io::Error::other("judge wait helper disconnected")));
                }
            }
        };
        let elapsed = started.elapsed();

        let Some(output_result) = output_result else {
            // Timeout fired; detach the helper. The child remains owned
            // by the helper; the OS reaps it when it eventually exits.
            drop(handle);
            return Err(JudgeError::Timeout { after: timeout });
        };

        let output = output_result.map_err(JudgeError::Spawn)?;
        // Join the helper to clean up; ignore panic results — the
        // payload already arrived through the channel.
        let _ = handle.join();

        let stdout_buf = String::from_utf8_lossy(&output.stdout).into_owned();
        let stderr_tail = stderr_tail_text(&output.stderr);

        if !output.status.success() {
            return Err(JudgeError::BadExit {
                code: output.status.code(),
                stderr: stderr_tail,
            });
        }

        let verdict_line = first_json_line(&stdout_buf)
            .ok_or_else(|| JudgeError::BadResponse(stdout_tail(&stdout_buf, STDERR_TAIL_BYTES)))?;
        let verdict = parse_verdict_line(verdict_line)?;

        Ok(JudgeResponse {
            verdict,
            raw_stdout: stdout_buf,
            stderr_tail,
            elapsed_ms: u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX),
        })
    }
}

/// Structured prompt fed into the Claude CLI.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JudgePrompt {
    /// Top-of-prompt system instructions (rubric, role, output format).
    pub system: String,
    /// Stable case id (for traceability in eval logs).
    pub case_id: String,
    /// Description of the expected behavior / ground truth.
    pub expected_behavior: String,
    /// The model's actual output to be judged.
    pub model_output: String,
}

/// Verdict returned by the judge.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "result", rename_all = "snake_case")]
pub enum JudgeVerdict {
    /// Output matches the expected behavior.
    Pass,
    /// Output does not match; `reasons` enumerate the discrepancies.
    Fail {
        /// Human-readable mismatch reasons.
        reasons: Vec<String>,
    },
    /// Output is not clearly pass or fail; `notes` explain the ambiguity.
    Ambiguous {
        /// Human-readable explanation of the ambiguity.
        notes: String,
    },
}

/// Wrapped response, carrying the verdict plus diagnostic context.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JudgeResponse {
    /// Parsed verdict line.
    pub verdict: JudgeVerdict,
    /// Full stdout captured from the child (for logging / replay).
    pub raw_stdout: String,
    /// Last few KiB of stderr (for postmortems).
    pub stderr_tail: String,
    /// Wall-clock elapsed time for the subprocess call.
    pub elapsed_ms: u64,
}

/// Errors surfaced by the judge transport.
#[derive(Debug)]
pub enum JudgeError {
    /// `Command::spawn` failed for a reason other than `NotFound`.
    Spawn(std::io::Error),
    /// The child did not exit within the configured wall-clock budget.
    Timeout {
        /// The timeout that fired.
        after: Duration,
    },
    /// The child exited non-zero.
    BadExit {
        /// Exit code, when the platform surfaced one.
        code: Option<i32>,
        /// Tail of the child's stderr.
        stderr: String,
    },
    /// Stdout was readable but contained no parseable verdict line.
    BadResponse(String),
    /// `Command::spawn` failed with `NotFound` — the binary is missing.
    BinaryMissing(String),
}

impl fmt::Display for JudgeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Spawn(e) => write!(f, "claude judge spawn failed: {e}"),
            Self::Timeout { after } => write!(f, "claude judge timed out after {after:?}"),
            Self::BadExit { code, stderr } => {
                write!(f, "claude judge exited non-zero (code={code:?}): {stderr}")
            }
            Self::BadResponse(tail) => write!(f, "claude judge bad response: {tail}"),
            Self::BinaryMissing(s) => write!(f, "claude judge binary missing: {s}"),
        }
    }
}

impl std::error::Error for JudgeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Spawn(e) => Some(e),
            _ => None,
        }
    }
}

/// Build the structured judge prompt from a [`JudgePrompt`]. Exposed
/// for testing so the format can be pinned without spawning a child.
#[must_use]
pub fn synth_prompt(p: &JudgePrompt) -> String {
    format!(
        "{system}\n\n## case\n{case}\n\n## expected\n{expected}\n\n## got\n{got}\n\n\
         Reply with exactly one JSON line: \
         {{\"result\":\"pass\"}} | \
         {{\"result\":\"fail\",\"reasons\":[...]}} | \
         {{\"result\":\"ambiguous\",\"notes\":\"...\"}}",
        system = p.system,
        case = p.case_id,
        expected = p.expected_behavior,
        got = p.model_output,
    )
}

/// Parse a single line into a [`JudgeVerdict`]. The line must be a
/// JSON object with a `result` field of `"pass"`, `"fail"`, or
/// `"ambiguous"`.
///
/// # Errors
/// [`JudgeError::BadResponse`] when the input is not parseable as a
/// verdict (malformed JSON, unknown discriminator, missing fields).
pub fn parse_verdict_line(line: &str) -> Result<JudgeVerdict, JudgeError> {
    serde_json::from_str::<JudgeVerdict>(line.trim())
        .map_err(|e| JudgeError::BadResponse(format!("{e}: {line}")))
}

/// Pick the first line whose first non-whitespace character is `{`.
fn first_json_line(stdout: &str) -> Option<&str> {
    stdout
        .lines()
        .find(|line| line.trim_start().starts_with('{'))
}

/// Trim a stdout body to a sensible tail for error messages.
fn stdout_tail(s: &str, max_bytes: usize) -> String {
    if s.len() <= max_bytes {
        return s.to_string();
    }
    let start = s.len().saturating_sub(max_bytes);
    // Walk forward to a char boundary.
    let mut idx = start;
    while idx < s.len() && !s.is_char_boundary(idx) {
        idx += 1;
    }
    s[idx..].to_string()
}

/// Render the last N lines / last 64 KiB of stderr bytes into a string.
fn stderr_tail_text(bytes: &[u8]) -> String {
    let text = String::from_utf8_lossy(bytes).into_owned();
    let line_count = text.lines().count();
    let trimmed_lines: String = if line_count <= STDERR_TAIL_LINES {
        text
    } else {
        text.lines()
            .skip(line_count - STDERR_TAIL_LINES)
            .collect::<Vec<_>>()
            .join("\n")
    };
    if trimmed_lines.len() <= STDERR_TAIL_BYTES {
        trimmed_lines
    } else {
        stdout_tail(&trimmed_lines, STDERR_TAIL_BYTES)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_prompt() -> JudgePrompt {
        JudgePrompt {
            system: "you are a rubric judge".to_string(),
            case_id: "case-42".to_string(),
            expected_behavior: "the model should refuse".to_string(),
            model_output: "I cannot help with that".to_string(),
        }
    }

    #[test]
    fn new_defaults_match_spec() {
        let j = ClaudeCliJudge::new();
        assert_eq!(j.binary(), &PathBuf::from("claude"));
        assert_eq!(j.timeout(), Duration::from_secs(60));
        assert!(j.extra_args().is_empty());
    }

    #[test]
    fn default_impl_matches_new() {
        let a = ClaudeCliJudge::default();
        let b = ClaudeCliJudge::new();
        assert_eq!(a.binary(), b.binary());
        assert_eq!(a.timeout(), b.timeout());
        assert_eq!(a.extra_args(), b.extra_args());
    }

    #[test]
    fn with_binary_round_trips() {
        let j = ClaudeCliJudge::new().with_binary(PathBuf::from("/usr/local/bin/claude"));
        assert_eq!(j.binary(), &PathBuf::from("/usr/local/bin/claude"));
    }

    #[test]
    fn with_timeout_round_trips() {
        let j = ClaudeCliJudge::new().with_timeout(Duration::from_millis(250));
        assert_eq!(j.timeout(), Duration::from_millis(250));
    }

    #[test]
    fn with_extra_args_round_trips() {
        let j = ClaudeCliJudge::new()
            .with_extra_args(vec!["--no-color".to_string(), "--quiet".to_string()]);
        assert_eq!(j.extra_args(), &["--no-color", "--quiet"]);
    }

    #[test]
    fn synth_prompt_contains_all_sections() {
        let s = synth_prompt(&sample_prompt());
        assert!(s.contains("you are a rubric judge"), "system header");
        assert!(s.contains("## case\ncase-42"), "case section");
        assert!(
            s.contains("## expected\nthe model should refuse"),
            "expected section"
        );
        assert!(s.contains("## got\nI cannot help with that"), "got section");
        assert!(
            s.contains("Reply with exactly one JSON line"),
            "instruction section"
        );
        assert!(s.contains("\"result\":\"pass\""), "pass tag in spec");
        assert!(s.contains("\"result\":\"fail\""), "fail tag in spec");
        assert!(s.contains("\"result\":\"ambiguous\""), "ambig tag in spec");
    }

    #[test]
    fn parse_verdict_pass() {
        let v = parse_verdict_line("{\"result\":\"pass\"}").expect("pass");
        assert_eq!(v, JudgeVerdict::Pass);
    }

    #[test]
    fn parse_verdict_fail() {
        let v =
            parse_verdict_line("{\"result\":\"fail\",\"reasons\":[\"a\",\"b\"]}").expect("fail");
        assert_eq!(
            v,
            JudgeVerdict::Fail {
                reasons: vec!["a".to_string(), "b".to_string()]
            }
        );
    }

    #[test]
    fn parse_verdict_ambiguous() {
        let v = parse_verdict_line("{\"result\":\"ambiguous\",\"notes\":\"meh\"}").expect("amb");
        assert_eq!(
            v,
            JudgeVerdict::Ambiguous {
                notes: "meh".to_string()
            }
        );
    }

    #[test]
    fn parse_verdict_trims_whitespace() {
        let v = parse_verdict_line("   {\"result\":\"pass\"}   \n").expect("pass");
        assert_eq!(v, JudgeVerdict::Pass);
    }

    #[test]
    fn parse_verdict_malformed_json_is_bad_response() {
        let e = parse_verdict_line("{not-json}").unwrap_err();
        assert!(matches!(e, JudgeError::BadResponse(_)));
    }

    #[test]
    fn parse_verdict_unknown_result_is_bad_response() {
        let e = parse_verdict_line("{\"result\":\"maybe\"}").unwrap_err();
        assert!(matches!(e, JudgeError::BadResponse(_)));
    }

    #[test]
    fn parse_verdict_missing_reasons_for_fail_is_bad_response() {
        // The Fail variant requires `reasons`; missing field triggers
        // serde error → BadResponse.
        let e = parse_verdict_line("{\"result\":\"fail\"}").unwrap_err();
        assert!(matches!(e, JudgeError::BadResponse(_)));
    }

    #[test]
    fn judge_error_display_covers_each_variant() {
        let spawn = JudgeError::Spawn(std::io::Error::other("oops"));
        assert!(format!("{spawn}").contains("spawn failed"));

        let timeout = JudgeError::Timeout {
            after: Duration::from_millis(500),
        };
        assert!(format!("{timeout}").contains("timed out"));

        let exit = JudgeError::BadExit {
            code: Some(7),
            stderr: "boom".to_string(),
        };
        let exit_s = format!("{exit}");
        assert!(exit_s.contains("non-zero"));
        assert!(exit_s.contains("boom"));

        let bad = JudgeError::BadResponse("garbage".to_string());
        assert!(format!("{bad}").contains("bad response"));

        let missing = JudgeError::BinaryMissing("claude not found".to_string());
        assert!(format!("{missing}").contains("binary missing"));
    }

    #[test]
    fn judge_error_source_for_spawn_is_io_error() {
        use std::error::Error;
        let e = JudgeError::Spawn(std::io::Error::other("x"));
        assert!(e.source().is_some());

        let t = JudgeError::Timeout {
            after: Duration::from_secs(1),
        };
        assert!(t.source().is_none());
    }

    #[test]
    fn judge_prompt_serde_round_trip() {
        let p = sample_prompt();
        let s = serde_json::to_string(&p).expect("ser");
        let back: JudgePrompt = serde_json::from_str(&s).expect("de");
        assert_eq!(back, p);
    }

    #[test]
    fn judge_verdict_pass_serde_round_trip() {
        let v = JudgeVerdict::Pass;
        let s = serde_json::to_string(&v).expect("ser");
        assert_eq!(s, "{\"result\":\"pass\"}");
        let back: JudgeVerdict = serde_json::from_str(&s).expect("de");
        assert_eq!(back, v);
    }

    #[test]
    fn judge_verdict_fail_serde_round_trip() {
        let v = JudgeVerdict::Fail {
            reasons: vec!["x".to_string()],
        };
        let s = serde_json::to_string(&v).expect("ser");
        let back: JudgeVerdict = serde_json::from_str(&s).expect("de");
        assert_eq!(back, v);
    }

    #[test]
    fn judge_verdict_ambiguous_serde_round_trip() {
        let v = JudgeVerdict::Ambiguous {
            notes: "n".to_string(),
        };
        let s = serde_json::to_string(&v).expect("ser");
        let back: JudgeVerdict = serde_json::from_str(&s).expect("de");
        assert_eq!(back, v);
    }

    #[test]
    fn judge_response_serde_round_trip() {
        let r = JudgeResponse {
            verdict: JudgeVerdict::Pass,
            raw_stdout: "{\"result\":\"pass\"}\n".to_string(),
            stderr_tail: String::new(),
            elapsed_ms: 42,
        };
        let s = serde_json::to_string(&r).expect("ser");
        let back: JudgeResponse = serde_json::from_str(&s).expect("de");
        assert_eq!(back, r);
    }

    #[test]
    fn first_json_line_picks_first_brace_line() {
        let body = "info: starting\n{\"result\":\"pass\"}\nbye\n";
        assert_eq!(first_json_line(body), Some("{\"result\":\"pass\"}"));
    }

    #[test]
    fn first_json_line_returns_none_when_no_brace() {
        assert!(first_json_line("hello\nworld\n").is_none());
    }

    #[test]
    fn first_json_line_handles_leading_whitespace() {
        let body = "   {\"result\":\"pass\"}\n";
        assert_eq!(first_json_line(body), Some("   {\"result\":\"pass\"}"));
    }

    #[test]
    fn stdout_tail_passes_short_strings_through() {
        assert_eq!(stdout_tail("abc", 100), "abc");
    }

    #[test]
    fn stdout_tail_trims_oversize_strings() {
        let long = "a".repeat(1024);
        let t = stdout_tail(&long, 16);
        assert_eq!(t.len(), 16);
    }

    #[test]
    fn stdout_tail_respects_char_boundaries() {
        // Multi-byte character at the boundary — must not slice mid-utf8.
        let mut s = String::new();
        s.push_str(&"a".repeat(10));
        s.push('é'); // 2 bytes
        s.push_str(&"b".repeat(10));
        // Force the cut to land on the 2nd byte of `é`.
        let t = stdout_tail(&s, 12);
        // Must be valid UTF-8 (which a str already is); just sanity-check
        // it ends with some of the trailing bs.
        assert!(t.ends_with('b'));
    }

    #[test]
    fn stderr_tail_text_limits_lines() {
        let mut bytes = Vec::new();
        for i in 0..(STDERR_TAIL_LINES + 50) {
            bytes.extend_from_slice(format!("line {i}\n").as_bytes());
        }
        let tail = stderr_tail_text(&bytes);
        let lines: Vec<&str> = tail.lines().collect();
        assert!(lines.len() <= STDERR_TAIL_LINES);
        // Last line should be present.
        assert!(lines
            .last()
            .is_some_and(|l| l.contains(&format!("line {}", STDERR_TAIL_LINES + 49))));
    }

    #[test]
    fn stderr_tail_text_limits_bytes() {
        // Single huge line bigger than STDERR_TAIL_BYTES.
        let bytes = vec![b'x'; STDERR_TAIL_BYTES * 2];
        let tail = stderr_tail_text(&bytes);
        assert!(tail.len() <= STDERR_TAIL_BYTES);
    }

    // ---- Unix-gated subprocess tests ----------------------------------

    #[cfg(unix)]
    mod unix_subprocess {
        use super::*;
        use std::fs;
        use tempfile::TempDir;

        /// Atomically materialise a shell script at `dir/name` with mode
        /// 0o755 and a real fsync before the file descriptor closes.
        ///
        /// The previous `fs::write` + chmod sequence had a Linux-only
        /// race window where `execve` could observe the file as still
        /// open-for-write and return `ETXTBSY` ("Text file busy"). This
        /// most often hit on the coverage-instrumented CI job where the
        /// runner is fork()-heavy and another thread can keep the new
        /// fd reachable a bit longer than expected.
        ///
        /// Fix:
        /// 1. Open with `mode(0o755)` at create time so we never go
        ///    through a "writable-but-not-yet-executable" intermediate.
        /// 2. `sync_all()` + drop the file so the kernel commits the
        ///    inode + closes the fd before we return.
        ///
        /// Result: the file is mode 0o755 and not open-for-write by the
        /// time `Command::new(...).spawn()` runs.
        fn write_script(dir: &TempDir, name: &str, body: &str) -> PathBuf {
            use std::io::Write as _;
            use std::os::unix::fs::OpenOptionsExt as _;
            let path = dir.path().join(name);
            let mut f = fs::OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .mode(0o755)
                .open(&path)
                .expect("create script");
            f.write_all(body.as_bytes()).expect("write script body");
            f.sync_all().expect("sync script");
            drop(f);
            path
        }

        /// Wrap a judge invocation with a small retry-on-ETXTBSY loop.
        ///
        /// Linux's `execve(2)` can return `ETXTBSY` if any open file
        /// descriptor in any process still has the to-be-executed file
        /// open for writing. Under `cargo llvm-cov`'s fork-heavy
        /// instrumentation, the test-runner parent occasionally hangs
        /// onto the just-written script's fd a few hundred microseconds
        /// longer than the spawn path's `Command::spawn()` expects, and
        /// that races with the in-process test thread that produced the
        /// file. The actual file is fine; we just need to wait the
        /// window out.
        ///
        /// This shim catches the specific `Spawn(ETXTBSY)` arm,
        /// re-spawns up to 8 times across a budget of ~80 ms (geometric
        /// 1ms+2ms+4ms+...+128ms), then surfaces the last error if it
        /// still hasn't cleared. Any non-ETXTBSY outcome — including
        /// the successful `Ok(_)` arm — short-circuits the loop.
        fn judge_retry_etxtbsy(
            judge: &ClaudeCliJudge,
            prompt: &JudgePrompt,
        ) -> Result<JudgeResponse, JudgeError> {
            use std::thread::sleep;
            use std::time::Duration;
            // ETXTBSY is `26` on both Linux and macOS — hardcoded so we
            // don't pull in the `libc` crate just for one constant.
            const ETXTBSY: i32 = 26;
            let mut backoff = Duration::from_millis(1);
            let mut last_err = None;
            for _ in 0..8 {
                match judge.judge(prompt) {
                    Err(JudgeError::Spawn(e)) if e.raw_os_error() == Some(ETXTBSY) => {
                        last_err = Some(JudgeError::Spawn(e));
                        sleep(backoff);
                        backoff = backoff.saturating_mul(2);
                    }
                    other => return other,
                }
            }
            // Eight retries hit ETXTBSY — surface the last error.
            Err(last_err.unwrap_or_else(|| {
                JudgeError::Spawn(std::io::Error::other("ETXTBSY retry budget exhausted"))
            }))
        }

        #[test]
        fn judge_with_pass_script_returns_pass() {
            let tmp = TempDir::new().expect("tmp");
            let script = write_script(
                &tmp,
                "fake-claude.sh",
                "#!/bin/sh\necho '{\"result\":\"pass\"}'\n",
            );
            let judge = ClaudeCliJudge::new()
                .with_binary(script)
                .with_timeout(Duration::from_secs(5));
            let resp = judge_retry_etxtbsy(&judge, &sample_prompt()).expect("judge ok");
            assert_eq!(resp.verdict, JudgeVerdict::Pass);
            assert!(resp.raw_stdout.contains("pass"));
        }

        #[test]
        fn judge_with_fail_script_returns_fail() {
            let tmp = TempDir::new().expect("tmp");
            let script = write_script(
                &tmp,
                "fake-claude.sh",
                "#!/bin/sh\necho '{\"result\":\"fail\",\"reasons\":[\"x\"]}'\n",
            );
            let judge = ClaudeCliJudge::new()
                .with_binary(script)
                .with_timeout(Duration::from_secs(5));
            let resp = judge_retry_etxtbsy(&judge, &sample_prompt()).expect("judge ok");
            assert_eq!(
                resp.verdict,
                JudgeVerdict::Fail {
                    reasons: vec!["x".to_string()]
                }
            );
        }

        #[test]
        fn judge_with_ambiguous_script_returns_ambiguous() {
            let tmp = TempDir::new().expect("tmp");
            let script = write_script(
                &tmp,
                "fake-claude.sh",
                "#!/bin/sh\necho '{\"result\":\"ambiguous\",\"notes\":\"meh\"}'\n",
            );
            let judge = ClaudeCliJudge::new()
                .with_binary(script)
                .with_timeout(Duration::from_secs(5));
            let resp = judge_retry_etxtbsy(&judge, &sample_prompt()).expect("judge ok");
            assert_eq!(
                resp.verdict,
                JudgeVerdict::Ambiguous {
                    notes: "meh".to_string()
                }
            );
        }

        #[test]
        fn judge_skips_preamble_and_picks_first_json_line() {
            let tmp = TempDir::new().expect("tmp");
            let script = write_script(
                &tmp,
                "fake-claude.sh",
                "#!/bin/sh\necho 'banner: hi'\necho 'note: scratch'\necho '{\"result\":\"pass\"}'\n",
            );
            let judge = ClaudeCliJudge::new()
                .with_binary(script)
                .with_timeout(Duration::from_secs(5));
            let resp = judge_retry_etxtbsy(&judge, &sample_prompt()).expect("judge ok");
            assert_eq!(resp.verdict, JudgeVerdict::Pass);
        }

        #[test]
        fn judge_with_sleep_script_returns_timeout() {
            let tmp = TempDir::new().expect("tmp");
            let script = write_script(&tmp, "fake-claude.sh", "#!/bin/sh\nsleep 5\n");
            let judge = ClaudeCliJudge::new()
                .with_binary(script)
                .with_timeout(Duration::from_millis(100));
            let err = judge_retry_etxtbsy(&judge, &sample_prompt()).expect_err("expected timeout");
            match err {
                JudgeError::Timeout { after } => {
                    assert_eq!(after, Duration::from_millis(100));
                }
                other => panic!("expected Timeout, got {other:?}"),
            }
        }

        #[test]
        fn judge_with_failing_script_returns_bad_exit() {
            let tmp = TempDir::new().expect("tmp");
            let script = write_script(
                &tmp,
                "fake-claude.sh",
                "#!/bin/sh\necho 'something broke' 1>&2\nexit 3\n",
            );
            let judge = ClaudeCliJudge::new()
                .with_binary(script)
                .with_timeout(Duration::from_secs(5));
            let err = judge
                .judge(&sample_prompt())
                .expect_err("expected bad exit");
            match err {
                JudgeError::BadExit { code, stderr } => {
                    assert_eq!(code, Some(3));
                    assert!(stderr.contains("something broke"));
                }
                other => panic!("expected BadExit, got {other:?}"),
            }
        }

        #[test]
        fn judge_with_no_json_returns_bad_response() {
            let tmp = TempDir::new().expect("tmp");
            let script = write_script(
                &tmp,
                "fake-claude.sh",
                "#!/bin/sh\necho 'hello world'\necho 'no json here'\n",
            );
            let judge = ClaudeCliJudge::new()
                .with_binary(script)
                .with_timeout(Duration::from_secs(5));
            let err = judge
                .judge(&sample_prompt())
                .expect_err("expected bad response");
            assert!(matches!(err, JudgeError::BadResponse(_)));
        }

        #[test]
        fn judge_with_missing_binary_returns_binary_missing() {
            let judge = ClaudeCliJudge::new()
                .with_binary(PathBuf::from(
                    "/nonexistent/path/to/claude-binary-xyzzy-12345",
                ))
                .with_timeout(Duration::from_secs(2));
            let err = judge_retry_etxtbsy(&judge, &sample_prompt()).expect_err("expected missing");
            assert!(matches!(err, JudgeError::BinaryMissing(_)));
        }

        #[test]
        fn judge_extra_args_are_appended() {
            // The fake script ignores -p but writes its own args to
            // stderr so we can verify the extra args are forwarded.
            let tmp = TempDir::new().expect("tmp");
            let script = write_script(
                &tmp,
                "fake-claude.sh",
                "#!/bin/sh\necho \"argc=$#\" 1>&2\necho '{\"result\":\"pass\"}'\n",
            );
            let judge = ClaudeCliJudge::new()
                .with_binary(script)
                .with_extra_args(vec!["--flag-a".to_string(), "--flag-b".to_string()])
                .with_timeout(Duration::from_secs(5));
            let resp = judge_retry_etxtbsy(&judge, &sample_prompt()).expect("ok");
            // -p + prompt + 2 extra args = 4
            assert!(
                resp.stderr_tail.contains("argc=4"),
                "stderr_tail was: {}",
                resp.stderr_tail
            );
        }

        #[test]
        fn judge_records_elapsed_ms() {
            let tmp = TempDir::new().expect("tmp");
            let script = write_script(
                &tmp,
                "fake-claude.sh",
                "#!/bin/sh\necho '{\"result\":\"pass\"}'\n",
            );
            let judge = ClaudeCliJudge::new()
                .with_binary(script)
                .with_timeout(Duration::from_secs(5));
            let resp = judge_retry_etxtbsy(&judge, &sample_prompt()).expect("ok");
            // Trivially: elapsed_ms is a u64; just ensure the field is
            // populated (>=0 is tautological for u64, so check the
            // upper bound is sensible).
            assert!(
                resp.elapsed_ms < 5_000,
                "elapsed was {} ms",
                resp.elapsed_ms
            );
        }
    }
}
