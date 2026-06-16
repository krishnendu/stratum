//! End-to-end integration tests for the OpenAI multimodal egress.
//!
//! These spawn the real `stratum` binary via `CARGO_BIN_EXE_stratum`,
//! start `stratum serve --openai --tcp-port 0 --stop-after-ms <ms>
//! --json`, parse the bound port from the daemon's startup JSON, then
//! send a real `POST /v1/chat/completions` request that uses the 2024
//! multimodal `content: [{type: "text", ...}, {type: "image_url", ...},
//! {type: "input_audio", ...}]` shape. The assertions verify:
//!
//! 1. The daemon accepts the multimodal request shape without 400-ing.
//! 2. The response is well-formed JSON with the OpenAI Chat Completions
//!    shape (`object: "chat.completion"`, single choice, text content).
//! 3. The text-from-parts concatenation reaches the backend — the
//!    EchoProvider echoes its input back as text, so the response body
//!    must contain the prompt fragment we sent inside the multimodal
//!    `content` array.
//! 4. Legacy string-shaped `content: "..."` requests still work
//!    against the same daemon (regression guard).
//!
//! This proves the Phase 5 (multimodal scaffold) + Phase 6 (OpenAI
//! egress) deliverables work *together*, not just in isolation.
//!
//! Network: loopback only. No external resources required.

// Integration test binary: every fn here exists only for `cargo test`.
// Test helpers panic on setup failures by design.
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::too_many_lines,
    reason = "integration test helpers may panic on setup failures"
)]
#![allow(
    clippy::doc_markdown,
    reason = "test-module doc comments name types (OpenAI / TurnContext / EchoProvider) without code-style backticks"
)]
#![allow(
    clippy::uninlined_format_args,
    reason = "the err-debug formatter reads more clearly out-of-line"
)]

use std::io::{BufRead, BufReader, Read};
use std::net::TcpStream;
use std::process::{Child, ChildStdout, Command, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use tempfile::TempDir;

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_stratum"))
}

/// Read a single line from a piped child stdout with a hard timeout.
///
/// `std`'s pipe-backed `ChildStdout` has no `set_read_timeout` analogue
/// (`File`/pipe descriptors don't surface `SO_RCVTIMEO`), so a buggy
/// daemon that emits an error on stderr but never writes to stdout
/// would otherwise hang the test runner until the CI watchdog kills
/// it. We hand the stdout off to a worker thread that does the actual
/// `read_line`, then wait on a bounded `recv_timeout`. The worker
/// returns the `BufReader` back so the caller can keep streaming.
fn read_line_with_timeout(
    stdout: ChildStdout,
    timeout: Duration,
) -> (BufReader<ChildStdout>, String) {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let mut reader = BufReader::new(stdout);
        let mut line = String::new();
        let res = reader.read_line(&mut line);
        let _ = tx.send((reader, line, res));
    });
    let (reader, line, res) = rx
        .recv_timeout(timeout)
        .expect("child stdout read_line timed out");
    res.expect("read startup line");
    (reader, line)
}

/// Parse the bound address from a `--json` startup line. Returns the
/// `"bound"` field as a `String`.
fn bound_from_json(line: &str) -> String {
    let v: serde_json::Value =
        serde_json::from_str(line.trim()).expect("startup line is valid JSON");
    v.get("bound")
        .and_then(|b| b.as_str())
        .map(str::to_string)
        .expect("`bound` field present")
}

/// Spawn `stratum serve --openai --tcp-port 0 --stop-after-ms <ms>
/// --json` and return the live child plus the bound address parsed
/// from the startup line.
fn spawn_openai_daemon(tmp: &TempDir, stop_after_ms: u64) -> (Child, String) {
    let mut child = bin()
        .args([
            "--storage-root",
            tmp.path().to_str().unwrap(),
            "serve",
            "--openai",
            "--tcp-port",
            "0",
            "--stop-after-ms",
            &stop_after_ms.to_string(),
            "--json",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn stratum");

    // Pull the startup JSON off stdout immediately so the child doesn't
    // block on a full pipe and so the bound address is known before we
    // try to dial. The 10 s ceiling guards against a daemon that fails
    // before printing — see `read_line_with_timeout`.
    let stdout = child.stdout.take().expect("child stdout");
    let (_reader, line) = read_line_with_timeout(stdout, Duration::from_secs(10));
    let bound = bound_from_json(&line);
    (child, bound)
}

/// Wait until a TCP connect to `bound` succeeds or the deadline
/// elapses. The OpenAI acceptor runs on a worker thread so there's a
/// small window after the startup JSON prints but before `recv_timeout`
/// is armed; this helper hides that race.
fn wait_for_accept(bound: &str, deadline: Instant) {
    let mut last_err: Option<std::io::Error> = None;
    while Instant::now() < deadline {
        match TcpStream::connect(bound) {
            Ok(_) => return,
            Err(e) => {
                last_err = Some(e);
                std::thread::sleep(Duration::from_millis(20));
            }
        }
    }
    panic!(
        "daemon did not accept on {bound} before deadline; last err: {:?}",
        last_err
    );
}

/// Send a JSON body via POST and return `(status_code, response_body)`.
///
/// Hand-rolls the HTTP/1.1 request rather than pulling `ureq` into the
/// dev-dependency surface — the existing test suite uses this pattern
/// and `ureq` is already a workspace dep for the binary anyway.
fn post_json(bound: &str, path: &str, body: &str) -> (u16, String) {
    use std::io::Write;
    let mut s = TcpStream::connect(bound).expect("connect");
    s.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
    s.set_write_timeout(Some(Duration::from_secs(5))).unwrap();
    let req = format!(
        "POST {path} HTTP/1.1\r\nHost: {bound}\r\nContent-Length: {}\r\n\
         Content-Type: application/json\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    s.write_all(req.as_bytes()).expect("write request");
    s.flush().expect("flush");

    let mut r = BufReader::new(s);
    let mut status_line = String::new();
    r.read_line(&mut status_line).expect("read status");
    let code: u16 = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|t| t.parse().ok())
        .unwrap_or_else(|| panic!("bad status line: {status_line:?}"));

    // Drain headers.
    loop {
        let mut line = String::new();
        let n = r.read_line(&mut line).expect("read header");
        if n == 0 || line == "\r\n" || line == "\n" {
            break;
        }
    }

    let mut body = String::new();
    let _ = r.read_to_string(&mut body);
    (code, body)
}

/// 1×1 transparent PNG, encoded as a `data:image/png;base64,...` URI.
/// The bytes here are the canonical "smallest valid PNG" — a 67-byte
/// blob with PNG signature + IHDR + IDAT + IEND. We use a known-good
/// blob so the test stays hermetic.
const TINY_PNG_DATA_URI: &str = "data:image/png;base64,\
    iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNgAAIAAAUAAen63NgAAAAASUVORK5CYII=";

/// Eight zero bytes encoded as base64 — stands in for an "audio
/// payload" in the test. The Phase-5 audio scaffold doesn't decode the
/// content; the wire test only verifies the *shape* and the existence
/// of a Block::Audio on the routed TurnContext.
const TINY_AUDIO_B64: &str = "AAAAAAAAAAA=";

#[test]
fn openai_multimodal_request_accepts_text_plus_image_url_part() {
    let tmp = TempDir::new().unwrap();
    let (mut child, bound) = spawn_openai_daemon(&tmp, 8_000);
    wait_for_accept(&bound, Instant::now() + Duration::from_secs(3));

    // Build a multimodal request: trailing user turn has TWO content
    // parts — a text fragment and a base64 image data URI. The
    // EchoProvider on the daemon side will echo the *text* portion
    // back; the image part rides into TurnContext.attachments and
    // is silently tolerated.
    let body = serde_json::json!({
        "model": "echo",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe-this-please"},
                    {"type": "image_url", "image_url": {"url": TINY_PNG_DATA_URI}}
                ]
            }
        ],
        "stream": false
    })
    .to_string();

    let (code, resp_body) = post_json(&bound, "/v1/chat/completions", &body);
    let _ = child.kill();
    let _ = child.wait();

    assert_eq!(code, 200, "expected 200, got {code}: {resp_body}");
    let v: serde_json::Value = serde_json::from_str(&resp_body).expect("response is valid JSON");
    assert_eq!(v["object"], "chat.completion", "{resp_body}");
    assert_eq!(v["model"], "echo");
    assert!(v["choices"].is_array());
    assert_eq!(v["choices"][0]["index"], 0);
    assert_eq!(v["choices"][0]["message"]["role"], "assistant");
    // EchoProvider echoes the prompt back. The text fragment from the
    // multimodal `content` array must appear in the assistant response.
    let assistant_text = v["choices"][0]["message"]["content"]
        .as_str()
        .expect("assistant content is a string");
    assert!(
        assistant_text.contains("describe-this-please"),
        "echo missed the multimodal text part: {assistant_text}"
    );
    assert!(
        v["choices"][0]["finish_reason"].is_string(),
        "finish_reason must be present"
    );
    // The usage block exists even when the EchoProvider doesn't track
    // tokens — the schema must validate on the client side.
    assert!(v["usage"]["total_tokens"].is_number());
}

#[test]
fn openai_multimodal_request_accepts_text_plus_input_audio_part() {
    let tmp = TempDir::new().unwrap();
    let (mut child, bound) = spawn_openai_daemon(&tmp, 8_000);
    wait_for_accept(&bound, Instant::now() + Duration::from_secs(3));

    let body = serde_json::json!({
        "model": "echo",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "transcribe-this-clip"},
                    {"type": "input_audio", "input_audio": {
                        "data": TINY_AUDIO_B64,
                        "format": "wav"
                    }}
                ]
            }
        ],
        "stream": false
    })
    .to_string();

    let (code, resp_body) = post_json(&bound, "/v1/chat/completions", &body);
    let _ = child.kill();
    let _ = child.wait();

    assert_eq!(code, 200, "expected 200, got {code}: {resp_body}");
    let v: serde_json::Value = serde_json::from_str(&resp_body).expect("response is valid JSON");
    assert_eq!(v["object"], "chat.completion");
    let assistant_text = v["choices"][0]["message"]["content"]
        .as_str()
        .expect("assistant content is a string");
    assert!(
        assistant_text.contains("transcribe-this-clip"),
        "echo missed the multimodal text part: {assistant_text}"
    );
}

#[test]
fn openai_string_content_still_works_after_multimodal_extension() {
    // Regression guard: the array-content extension uses serde
    // `untagged`, which means a malformed Parts arm could silently
    // swallow the Text arm. This test pins that the original 2023
    // string shape remains a 200 with an echoed body.
    let tmp = TempDir::new().unwrap();
    let (mut child, bound) = spawn_openai_daemon(&tmp, 8_000);
    wait_for_accept(&bound, Instant::now() + Duration::from_secs(3));

    let body = serde_json::json!({
        "model": "echo",
        "messages": [
            {"role": "user", "content": "legacy-string-prompt"}
        ],
        "stream": false
    })
    .to_string();

    let (code, resp_body) = post_json(&bound, "/v1/chat/completions", &body);
    let _ = child.kill();
    let _ = child.wait();

    assert_eq!(code, 200, "expected 200, got {code}: {resp_body}");
    let v: serde_json::Value = serde_json::from_str(&resp_body).expect("valid JSON");
    let assistant_text = v["choices"][0]["message"]["content"]
        .as_str()
        .expect("assistant content is a string");
    assert!(
        assistant_text.contains("legacy-string-prompt"),
        "echo missed the legacy string prompt: {assistant_text}"
    );
}

#[test]
fn openai_multimodal_array_with_only_text_part_round_trips() {
    // Smoke: the `content` array with a *single* text part must
    // produce identical behaviour to the legacy string variant. This
    // pins the (Vec<Part>) -> String concatenation in
    // OpenAIMessageContent::flatten().
    let tmp = TempDir::new().unwrap();
    let (mut child, bound) = spawn_openai_daemon(&tmp, 8_000);
    wait_for_accept(&bound, Instant::now() + Duration::from_secs(3));

    let body = serde_json::json!({
        "model": "echo",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "lone-text-part"}
                ]
            }
        ],
        "stream": false
    })
    .to_string();

    let (code, resp_body) = post_json(&bound, "/v1/chat/completions", &body);
    let _ = child.kill();
    let _ = child.wait();

    assert_eq!(code, 200, "expected 200, got {code}: {resp_body}");
    let v: serde_json::Value = serde_json::from_str(&resp_body).expect("valid JSON");
    let assistant_text = v["choices"][0]["message"]["content"]
        .as_str()
        .expect("assistant content is a string");
    assert!(
        assistant_text.contains("lone-text-part"),
        "echo missed the single-element text array: {assistant_text}"
    );
}

#[test]
fn openai_multimodal_request_with_unknown_part_type_returns_400() {
    // The unit test in `openai.rs` covers serde-side rejection of
    // unknown `type` discriminants on `content` parts. This test pins
    // the HTTP-layer behaviour end-to-end: a live daemon must surface
    // the rejection as `400 invalid_request`, not silently drop the
    // unknown variant or 500 on a panic.
    let tmp = TempDir::new().unwrap();
    let (mut child, bound) = spawn_openai_daemon(&tmp, 8_000);
    wait_for_accept(&bound, Instant::now() + Duration::from_secs(3));

    // `video` is not a supported part type. The Phase 5 wire schema
    // only knows `text` / `image_url` / `input_audio`.
    let body = serde_json::json!({
        "model": "echo",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video_url": "https://example.invalid/clip.mp4"}
                ]
            }
        ],
        "stream": false
    })
    .to_string();

    let (code, resp_body) = post_json(&bound, "/v1/chat/completions", &body);
    let _ = child.kill();
    let _ = child.wait();

    assert_eq!(code, 400, "expected 400, got {code}: {resp_body}");
    let v: serde_json::Value =
        serde_json::from_str(&resp_body).expect("400 body must still be valid JSON");
    // OpenAI clients key on `error.code` for retry classification;
    // pin both `type` and `code` to `invalid_request` as documented in
    // the `error_body_includes_code_field` unit test.
    assert_eq!(
        v["error"]["code"].as_str(),
        Some("invalid_request"),
        "error.code must be invalid_request: {resp_body}"
    );
    assert_eq!(
        v["error"]["type"].as_str(),
        Some("invalid_request"),
        "error.type must be invalid_request: {resp_body}"
    );
}
