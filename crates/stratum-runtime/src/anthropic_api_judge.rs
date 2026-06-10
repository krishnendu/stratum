//! `AnthropicApiJudge` — metered HTTP transport for the Stratum LLM-judge.
//!
//! Phase 4+ scaffold. The **default** Stratum eval-judge runs `claude -p`
//! as a subprocess riding the user's Claude Code subscription (see
//! [`crate::claude_cli_judge`]). This module is the **opt-in alternative**:
//! a maintainer who would rather pay metered Anthropic API rates than be
//! throttled by subscription-tier limits can wire `AnthropicApiJudge` into
//! the eval-runner instead.
//!
//! The transport speaks the Anthropic Messages API
//! (`POST https://api.anthropic.com/v1/messages`) over `ureq`, hashes the
//! response body's first `content[].text` block, and parses it as a single
//! JSON line matching [`JudgeVerdict`] — the exact same wire format as the
//! CLI subprocess judge. The two transports are otherwise interchangeable:
//! they share [`JudgePrompt`], [`JudgeResponse`], and [`JudgeError`].
//!
//! ## Wire format
//!
//! Request body (`POST /v1/messages`):
//!
//! ```json
//! {
//!   "model": "claude-sonnet-4-5",
//!   "max_tokens": 1024,
//!   "system": "<system + structured rubric>",
//!   "messages": [
//!     { "role": "user", "content": "<case>\n\n## expected\n...\n\n## got\n..." }
//!   ]
//! }
//! ```
//!
//! Required headers:
//!
//! - `x-api-key: <api_key>`
//! - `anthropic-version: 2023-06-01`
//! - `content-type: application/json`
//!
//! Response — the judge reads `response.content[0].text` and parses it as
//! one of:
//!
//! ```text
//! {"result":"pass"}
//! {"result":"fail","reasons":["..."]}
//! {"result":"ambiguous","notes":"..."}
//! ```

// xtask-check-error-codes: ignore-file
//
// Reason: this module reuses the local `JudgeError` variants defined in
// `claude_cli_judge.rs` rather than catalog `STRAT-E####` entries. The
// judge is a v2-scaffold transport; promoting these to the catalog
// happens when the eval-runner wires the judge in for real.

use std::io::Read;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

#[cfg(test)]
use crate::claude_cli_judge::JudgeVerdict;
use crate::claude_cli_judge::{JudgeError, JudgePrompt, JudgeResponse};

/// Default Anthropic Messages API endpoint.
const DEFAULT_ENDPOINT: &str = "https://api.anthropic.com/v1/messages";
/// Default model slug.
const DEFAULT_MODEL: &str = "claude-sonnet-4-5";
/// Default per-call `max_tokens`.
const DEFAULT_MAX_TOKENS: u32 = 1024;
/// Default wall-clock timeout. Matches the CLI judge.
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(60);
/// Cap on retained response-body text used to populate `BadExit.stderr`
/// and `BadResponse(_)` payloads, mirroring the
/// [`crate::claude_cli_judge`] policy (last 64 KiB).
const BODY_TAIL_BYTES: usize = 64 * 1024;
/// Cap on the response body bytes the judge will read into memory
/// before bailing with a transport-level failure. Anthropic responses
/// are tiny JSON blobs; 1 MiB is generous and bounds adversarial
/// payloads.
const BODY_MAX_BYTES: u64 = 1024 * 1024;
/// `anthropic-version` header value.
const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Static configuration for the metered HTTP judge.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnthropicApiConfig {
    /// Anthropic API key. **Empty by default — caller must fill.** The
    /// judge refuses to send a request when this is empty, returning a
    /// [`JudgeError::BadExit`] with a sentinel stderr.
    pub api_key: String,
    /// Messages API endpoint. Overridable for the in-process mock used
    /// in tests.
    pub endpoint: String,
    /// Anthropic model slug, used as `body.model`.
    pub model: String,
    /// `body.max_tokens`. Bounds the verdict-line response.
    pub max_tokens: u32,
    /// Per-call wall-clock timeout — applied as `ureq.timeout(..)`.
    pub timeout: Duration,
}

impl Default for AnthropicApiConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            endpoint: DEFAULT_ENDPOINT.to_string(),
            model: DEFAULT_MODEL.to_string(),
            max_tokens: DEFAULT_MAX_TOKENS,
            timeout: DEFAULT_TIMEOUT,
        }
    }
}

/// HTTP-backed Anthropic Messages judge.
///
/// Builds a Messages request from a [`JudgePrompt`], POSTs it via
/// `ureq`, and parses the first `content[].text` block as a single
/// JSON-line [`JudgeVerdict`]. Errors are surfaced through the shared
/// [`JudgeError`] enum so callers can swap this in for
/// [`crate::claude_cli_judge::ClaudeCliJudge`] without changing call
/// sites.
#[derive(Debug, Clone)]
pub struct AnthropicApiJudge {
    cfg: AnthropicApiConfig,
}

impl AnthropicApiJudge {
    /// Build a new judge from `cfg`.
    #[must_use]
    pub const fn new(cfg: AnthropicApiConfig) -> Self {
        Self { cfg }
    }

    /// Inspect the configured endpoint.
    #[must_use]
    pub fn endpoint(&self) -> &str {
        &self.cfg.endpoint
    }

    /// Inspect the configured model slug.
    #[must_use]
    pub fn model(&self) -> &str {
        &self.cfg.model
    }

    /// Inspect the configured timeout.
    #[must_use]
    pub const fn timeout(&self) -> Duration {
        self.cfg.timeout
    }

    /// Inspect the configured `max_tokens` cap.
    #[must_use]
    pub const fn max_tokens(&self) -> u32 {
        self.cfg.max_tokens
    }

    /// Run the judge end-to-end against `prompt`.
    ///
    /// # Errors
    /// - [`JudgeError::BadExit`] when `cfg.api_key` is empty, when the
    ///   server returns a non-2xx HTTP status, or when the transport
    ///   layer surfaces an IO failure that is not a timeout.
    /// - [`JudgeError::Timeout`] when the server does not respond
    ///   within `cfg.timeout`.
    /// - [`JudgeError::BadResponse`] when the JSON envelope cannot be
    ///   parsed, contains no text block, or the text is not a parseable
    ///   verdict line.
    pub fn judge(&self, prompt: &JudgePrompt) -> Result<JudgeResponse, JudgeError> {
        if self.cfg.api_key.is_empty() {
            return Err(JudgeError::BadExit {
                code: None,
                stderr: "ANTHROPIC_API_KEY not set".to_string(),
            });
        }

        let body = synth_anthropic_body(prompt, &self.cfg);
        let started = Instant::now();
        let response = ureq::post(&self.cfg.endpoint)
            .timeout(self.cfg.timeout)
            .set("x-api-key", &self.cfg.api_key)
            .set("anthropic-version", ANTHROPIC_VERSION)
            .set("content-type", "application/json")
            .send_string(&body);

        let response = match response {
            Ok(r) => r,
            Err(ureq::Error::Status(code, resp)) => {
                let tail = read_response_tail(resp);
                return Err(JudgeError::BadExit {
                    code: Some(i32::from(code)),
                    stderr: tail,
                });
            }
            Err(ureq::Error::Transport(t)) => {
                // ureq surfaces both wall-clock timeouts and connect
                // timeouts through `Transport`; the elapsed clock tells
                // us which side of the line we're on.
                if started.elapsed() >= self.cfg.timeout {
                    return Err(JudgeError::Timeout {
                        after: self.cfg.timeout,
                    });
                }
                return Err(JudgeError::BadExit {
                    code: None,
                    stderr: format!("transport: {t}"),
                });
            }
        };

        let body_text = read_response_tail(response);
        let elapsed = started.elapsed();

        let envelope: AnthropicEnvelope = serde_json::from_str(&body_text)
            .map_err(|e| JudgeError::BadResponse(format!("envelope parse: {e}: {body_text}")))?;
        let verdict_text = envelope
            .content
            .iter()
            .find(|block| block.kind == "text")
            .map(|block| block.text.clone())
            .ok_or_else(|| {
                JudgeError::BadResponse(format!("no text block in response: {body_text}"))
            })?;
        let verdict = crate::claude_cli_judge::parse_verdict_line(&verdict_text)?;

        Ok(JudgeResponse {
            verdict,
            raw_stdout: body_text,
            stderr_tail: String::new(),
            elapsed_ms: u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX),
        })
    }
}

/// Build the Anthropic Messages API JSON body from a [`JudgePrompt`].
/// Exposed for testing so the wire shape can be pinned without an HTTP
/// round trip.
#[must_use]
pub fn synth_anthropic_body(p: &JudgePrompt, cfg: &AnthropicApiConfig) -> String {
    let system = format!(
        "{system}\n\nYou are a rubric judge. Reply with exactly one JSON line: \
         {{\"result\":\"pass\"}} | \
         {{\"result\":\"fail\",\"reasons\":[...]}} | \
         {{\"result\":\"ambiguous\",\"notes\":\"...\"}}",
        system = p.system,
    );
    let user = format!(
        "{case}\n\n## expected\n{expected}\n\n## got\n{got}",
        case = p.case_id,
        expected = p.expected_behavior,
        got = p.model_output,
    );
    let body = AnthropicRequestBody {
        model: &cfg.model,
        max_tokens: cfg.max_tokens,
        system: &system,
        messages: vec![AnthropicRequestMessage {
            role: "user",
            content: &user,
        }],
    };
    // Manual fallback if serialization somehow fails (it cannot for
    // owned-string fields; the fallback exists only so this function
    // does not unwrap).
    serde_json::to_string(&body).unwrap_or_else(|_| String::new())
}

/// Read up to [`BODY_MAX_BYTES`] from `response` and return the last
/// [`BODY_TAIL_BYTES`] as a `String`. Used for both happy-path JSON
/// envelopes and non-2xx error tails.
fn read_response_tail(response: ureq::Response) -> String {
    let mut reader = response.into_reader().take(BODY_MAX_BYTES);
    let mut buf = Vec::new();
    let _ = reader.read_to_end(&mut buf);
    let text = String::from_utf8_lossy(&buf).into_owned();
    if text.len() <= BODY_TAIL_BYTES {
        return text;
    }
    let start = text.len().saturating_sub(BODY_TAIL_BYTES);
    let mut idx = start;
    while idx < text.len() && !text.is_char_boundary(idx) {
        idx += 1;
    }
    text[idx..].to_string()
}

#[derive(Debug, Serialize)]
struct AnthropicRequestBody<'a> {
    model: &'a str,
    max_tokens: u32,
    system: &'a str,
    messages: Vec<AnthropicRequestMessage<'a>>,
}

#[derive(Debug, Serialize)]
struct AnthropicRequestMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Debug, Deserialize)]
struct AnthropicEnvelope {
    #[serde(default)]
    content: Vec<AnthropicContentBlock>,
}

#[derive(Debug, Deserialize)]
struct AnthropicContentBlock {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    text: String,
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
    fn config_default_matches_spec() {
        let cfg = AnthropicApiConfig::default();
        assert_eq!(cfg.api_key, "");
        assert_eq!(cfg.endpoint, "https://api.anthropic.com/v1/messages");
        assert_eq!(cfg.model, "claude-sonnet-4-5");
        assert_eq!(cfg.max_tokens, 1024);
        assert_eq!(cfg.timeout, Duration::from_secs(60));
    }

    #[test]
    fn config_serde_round_trip() {
        let cfg = AnthropicApiConfig {
            api_key: "sk-test".to_string(),
            endpoint: "http://127.0.0.1:9999/v1/messages".to_string(),
            model: "claude-opus-9-9".to_string(),
            max_tokens: 256,
            timeout: Duration::from_millis(1500),
        };
        let s = serde_json::to_string(&cfg).expect("ser");
        let back: AnthropicApiConfig = serde_json::from_str(&s).expect("de");
        assert_eq!(back, cfg);
    }

    #[test]
    fn judge_constructor_smoke() {
        let cfg = AnthropicApiConfig::default();
        let judge = AnthropicApiJudge::new(cfg.clone());
        assert_eq!(judge.endpoint(), cfg.endpoint);
        assert_eq!(judge.model(), cfg.model);
        assert_eq!(judge.timeout(), cfg.timeout);
        assert_eq!(judge.max_tokens(), cfg.max_tokens);
    }

    #[test]
    fn synth_body_contains_case_expected_got() {
        let cfg = AnthropicApiConfig::default();
        let body = synth_anthropic_body(&sample_prompt(), &cfg);
        assert!(body.contains("case-42"), "case id: {body}");
        assert!(body.contains("the model should refuse"), "expected: {body}");
        assert!(body.contains("I cannot help with that"), "got: {body}");
        assert!(
            body.contains("Reply with exactly one JSON line"),
            "verdict instruction: {body}"
        );
        // The system text contains JSON snippets like `{"result":"pass"}`
        // which serde escapes when embedded into the body's `system`
        // string field — each `"` becomes `\"` on the wire.
        assert!(
            body.contains("\\\"result\\\":\\\"pass\\\""),
            "pass tag: {body}"
        );
        assert!(
            body.contains("\\\"result\\\":\\\"fail\\\""),
            "fail tag: {body}"
        );
        assert!(
            body.contains("\\\"result\\\":\\\"ambiguous\\\""),
            "ambig tag: {body}"
        );
    }

    #[test]
    fn synth_body_references_model_from_config() {
        let cfg = AnthropicApiConfig {
            model: "claude-opus-9-9".to_string(),
            ..AnthropicApiConfig::default()
        };
        let body = synth_anthropic_body(&sample_prompt(), &cfg);
        assert!(body.contains("\"model\":\"claude-opus-9-9\""), "{body}");
        assert!(body.contains("\"max_tokens\":1024"), "{body}");
    }

    #[test]
    fn synth_body_uses_max_tokens_from_config() {
        let cfg = AnthropicApiConfig {
            max_tokens: 256,
            ..AnthropicApiConfig::default()
        };
        let body = synth_anthropic_body(&sample_prompt(), &cfg);
        assert!(body.contains("\"max_tokens\":256"), "{body}");
    }

    #[test]
    fn synth_body_is_valid_json() {
        let cfg = AnthropicApiConfig::default();
        let body = synth_anthropic_body(&sample_prompt(), &cfg);
        let parsed: serde_json::Value = serde_json::from_str(&body).expect("parse");
        assert_eq!(parsed["model"], "claude-sonnet-4-5");
        assert_eq!(parsed["max_tokens"], 1024);
        assert!(parsed["system"]
            .as_str()
            .is_some_and(|s| s.contains("rubric")));
        let messages = parsed["messages"].as_array().expect("messages");
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["role"], "user");
        let content = messages[0]["content"].as_str().expect("content");
        assert!(content.contains("case-42"));
        assert!(content.contains("## expected"));
        assert!(content.contains("## got"));
    }

    #[test]
    fn empty_api_key_returns_bad_exit_without_http_call() {
        // Endpoint is intentionally unbound — if the judge tried to
        // make a real call this test would surface as a transport
        // error instead of `BadExit`.
        let cfg = AnthropicApiConfig {
            endpoint: "http://127.0.0.1:1/v1/messages".to_string(),
            ..AnthropicApiConfig::default()
        };
        let judge = AnthropicApiJudge::new(cfg);
        let err = judge.judge(&sample_prompt()).expect_err("must error");
        match err {
            JudgeError::BadExit { code, stderr } => {
                assert_eq!(code, None);
                assert!(stderr.contains("ANTHROPIC_API_KEY"), "stderr: {stderr}");
            }
            other => panic!("expected BadExit, got {other:?}"),
        }
    }

    // ---- In-process HTTP mock tests (mirror download.rs pattern) ----------

    #[cfg(unix)]
    mod http_mock {
        use super::*;
        use std::io::Write as _;
        use std::net::TcpListener;

        /// Spawn a one-shot HTTP server that returns a canned status +
        /// body. Returns the bound URL. Drains the inbound request body
        /// fully before responding so the client's POST stream isn't
        /// reset mid-write.
        fn spawn_canned_server(status_line: &'static str, body: String) -> String {
            let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
            let addr = listener.local_addr().expect("addr");
            std::thread::spawn(move || {
                for stream in listener.incoming().take(1) {
                    let Ok(mut stream) = stream else { continue };
                    drain_request(&mut stream);
                    let headers = format!(
                        "HTTP/1.1 {status_line}\r\nContent-Length: {len}\r\nConnection: close\r\n\r\n",
                        len = body.len(),
                    );
                    let _ = stream.write_all(headers.as_bytes());
                    let _ = stream.write_all(body.as_bytes());
                    let _ = stream.flush();
                }
            });
            format!("http://{addr}/v1/messages")
        }

        /// Read the request headers, then the body if `Content-Length`
        /// announced one. Best-effort: any IO failure short-circuits.
        fn drain_request(stream: &mut std::net::TcpStream) {
            let mut accumulated: Vec<u8> = Vec::new();
            let mut chunk = [0_u8; 4096];
            let header_end = loop {
                let n = match stream.read(&mut chunk) {
                    Ok(0) | Err(_) => return,
                    Ok(n) => n,
                };
                accumulated.extend_from_slice(&chunk[..n]);
                if let Some(idx) = find_header_end(&accumulated) {
                    break idx;
                }
                if accumulated.len() > 64 * 1024 {
                    return;
                }
            };
            let header_text = String::from_utf8_lossy(&accumulated[..header_end]);
            let content_length = parse_content_length(&header_text).unwrap_or(0);
            let body_already = accumulated.len().saturating_sub(header_end + 4);
            let mut remaining = content_length.saturating_sub(body_already);
            while remaining > 0 {
                let want = remaining.min(chunk.len());
                let n = match stream.read(&mut chunk[..want]) {
                    Ok(0) | Err(_) => return,
                    Ok(n) => n,
                };
                remaining = remaining.saturating_sub(n);
            }
        }

        fn find_header_end(buf: &[u8]) -> Option<usize> {
            buf.windows(4).position(|w| w == b"\r\n\r\n")
        }

        fn parse_content_length(headers: &str) -> Option<usize> {
            for line in headers.lines() {
                let lower = line.to_ascii_lowercase();
                if let Some(rest) = lower.strip_prefix("content-length:") {
                    return rest.trim().parse::<usize>().ok();
                }
            }
            None
        }

        /// Spawn a server that accepts the connection but never replies
        /// (until the test thread joins back). Forces a wall-clock
        /// timeout in the client.
        fn spawn_silent_server() -> String {
            let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
            let addr = listener.local_addr().expect("addr");
            std::thread::spawn(move || {
                for stream in listener.incoming().take(1) {
                    let Ok(stream) = stream else { continue };
                    // Hold the connection open by parking the worker
                    // until the OS reaps the test process. Dropping the
                    // stream here would make ureq see EOF.
                    std::thread::sleep(Duration::from_secs(5));
                    drop(stream);
                }
            });
            format!("http://{addr}/v1/messages")
        }

        fn make_envelope(text: &str) -> String {
            serde_json::json!({
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "content": [
                    { "type": "text", "text": text }
                ],
                "model": "claude-sonnet-4-5",
                "stop_reason": "end_turn",
            })
            .to_string()
        }

        fn judge_with_endpoint(endpoint: String, timeout: Duration) -> AnthropicApiJudge {
            AnthropicApiJudge::new(AnthropicApiConfig {
                api_key: "sk-test".to_string(),
                endpoint,
                model: "claude-sonnet-4-5".to_string(),
                max_tokens: 1024,
                timeout,
            })
        }

        #[test]
        fn happy_path_pass_verdict() {
            let url = spawn_canned_server("200 OK", make_envelope("{\"result\":\"pass\"}"));
            let judge = judge_with_endpoint(url, Duration::from_secs(5));
            let resp = judge.judge(&sample_prompt()).expect("judge ok");
            assert_eq!(resp.verdict, JudgeVerdict::Pass);
            assert!(resp.raw_stdout.contains("pass"));
        }

        #[test]
        fn fail_verdict_round_trips() {
            let url = spawn_canned_server(
                "200 OK",
                make_envelope("{\"result\":\"fail\",\"reasons\":[\"x\",\"y\"]}"),
            );
            let judge = judge_with_endpoint(url, Duration::from_secs(5));
            let resp = judge.judge(&sample_prompt()).expect("judge ok");
            assert_eq!(
                resp.verdict,
                JudgeVerdict::Fail {
                    reasons: vec!["x".to_string(), "y".to_string()],
                }
            );
        }

        #[test]
        fn ambiguous_verdict_round_trips() {
            let url = spawn_canned_server(
                "200 OK",
                make_envelope("{\"result\":\"ambiguous\",\"notes\":\"meh\"}"),
            );
            let judge = judge_with_endpoint(url, Duration::from_secs(5));
            let resp = judge.judge(&sample_prompt()).expect("judge ok");
            assert_eq!(
                resp.verdict,
                JudgeVerdict::Ambiguous {
                    notes: "meh".to_string(),
                }
            );
        }

        #[test]
        fn malformed_text_returns_bad_response() {
            let url = spawn_canned_server("200 OK", make_envelope("not a json verdict"));
            let judge = judge_with_endpoint(url, Duration::from_secs(5));
            let err = judge.judge(&sample_prompt()).expect_err("must err");
            assert!(matches!(err, JudgeError::BadResponse(_)), "{err:?}");
        }

        #[test]
        fn envelope_without_text_block_returns_bad_response() {
            let envelope = serde_json::json!({
                "id": "msg_test",
                "type": "message",
                "content": [],
            })
            .to_string();
            let url = spawn_canned_server("200 OK", envelope);
            let judge = judge_with_endpoint(url, Duration::from_secs(5));
            let err = judge.judge(&sample_prompt()).expect_err("must err");
            match err {
                JudgeError::BadResponse(tail) => {
                    assert!(tail.contains("no text block"), "tail: {tail}");
                }
                other => panic!("expected BadResponse, got {other:?}"),
            }
        }

        #[test]
        fn envelope_with_unparseable_json_returns_bad_response() {
            let url = spawn_canned_server("200 OK", "{this is not json".to_string());
            let judge = judge_with_endpoint(url, Duration::from_secs(5));
            let err = judge.judge(&sample_prompt()).expect_err("must err");
            match err {
                JudgeError::BadResponse(tail) => {
                    assert!(tail.contains("envelope parse"), "tail: {tail}");
                }
                other => panic!("expected BadResponse, got {other:?}"),
            }
        }

        #[test]
        fn http_400_returns_bad_exit_with_code_400() {
            let url = spawn_canned_server(
                "400 Bad Request",
                "{\"error\":{\"type\":\"invalid_request\"}}".to_string(),
            );
            let judge = judge_with_endpoint(url, Duration::from_secs(5));
            let err = judge.judge(&sample_prompt()).expect_err("must err");
            match err {
                JudgeError::BadExit { code, stderr } => {
                    assert_eq!(code, Some(400));
                    assert!(stderr.contains("invalid_request"), "stderr: {stderr}");
                }
                other => panic!("expected BadExit, got {other:?}"),
            }
        }

        #[test]
        fn http_500_returns_bad_exit_with_code_500() {
            let url = spawn_canned_server(
                "500 Internal Server Error",
                "{\"error\":{\"type\":\"overloaded\"}}".to_string(),
            );
            let judge = judge_with_endpoint(url, Duration::from_secs(5));
            let err = judge.judge(&sample_prompt()).expect_err("must err");
            match err {
                JudgeError::BadExit { code, stderr } => {
                    assert_eq!(code, Some(500));
                    assert!(stderr.contains("overloaded"), "stderr: {stderr}");
                }
                other => panic!("expected BadExit, got {other:?}"),
            }
        }

        #[test]
        fn silent_server_triggers_timeout() {
            let url = spawn_silent_server();
            let judge = judge_with_endpoint(url, Duration::from_millis(150));
            let err = judge.judge(&sample_prompt()).expect_err("must err");
            match err {
                JudgeError::Timeout { after } => {
                    assert_eq!(after, Duration::from_millis(150));
                }
                other => panic!("expected Timeout, got {other:?}"),
            }
        }

        #[test]
        fn transport_failure_unbound_port_returns_bad_exit() {
            // Loopback port 1 is unbound; ureq will fail before any
            // timeout elapses, so the judge maps the transport error
            // to `BadExit { code: None }`.
            let judge = judge_with_endpoint(
                "http://127.0.0.1:1/v1/messages".to_string(),
                Duration::from_secs(5),
            );
            let err = judge.judge(&sample_prompt()).expect_err("must err");
            match err {
                JudgeError::BadExit { code, stderr } => {
                    assert_eq!(code, None);
                    assert!(stderr.contains("transport"), "stderr: {stderr}");
                }
                other => panic!("expected BadExit, got {other:?}"),
            }
        }

        #[test]
        fn happy_path_records_elapsed_ms() {
            let url = spawn_canned_server("200 OK", make_envelope("{\"result\":\"pass\"}"));
            let judge = judge_with_endpoint(url, Duration::from_secs(5));
            let resp = judge.judge(&sample_prompt()).expect("ok");
            assert!(resp.elapsed_ms < 5_000, "elapsed: {}", resp.elapsed_ms);
            assert_eq!(resp.stderr_tail, "");
        }
    }
}
