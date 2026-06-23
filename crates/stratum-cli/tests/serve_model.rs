//! Integration tests for `stratum serve --model <slug>`.
//!
//! These spawn the real CLI binary via `CARGO_BIN_EXE_stratum` and cover:
//!
//! * Default (no `--model`): provider line names `echo`, exit 0.
//! * `--model X` without the `provider-llama-cpp` feature: exit 1 +
//!   STRAT-E1001 pointing at the missing feature.
//! * `--model unknown-slug` with the feature on: exit 1 + STRAT-E1001
//!   "unknown slug" (only compiled when the feature is enabled).
//! * End-to-end TCP `ping` round-trip with `--model X --json` only in the
//!   feature-off configuration so default CI stays fast; the feature-on
//!   path needs a real on-disk GGUF and is exercised by the on-demand
//!   llama workflow instead.

// Integration test binary: every fn here exists only for `cargo test`.
// Test helpers panic on setup failures by design; clippy's
// `expect_used` / `unwrap_used` denials are scoped to non-test code.
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    reason = "integration test helpers may panic on setup failures"
)]

use std::process::Command;

use tempfile::TempDir;

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_stratum"))
}

#[test]
fn serve_without_model_reports_echo_provider() {
    let tmp = TempDir::new().unwrap();
    let output = bin()
        .args([
            "--storage-root",
            tmp.path().to_str().unwrap(),
            "serve",
            "--tcp-port",
            "0",
            "--stop-after-ms",
            "200",
        ])
        .output()
        .expect("spawn stratum");
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("provider=echo"),
        "expected provider=echo in startup line, got: {stdout}"
    );
}

#[cfg(not(feature = "provider-llama-cpp"))]
#[test]
fn serve_with_model_without_feature_errors() {
    let tmp = TempDir::new().unwrap();
    let output = bin()
        .args([
            "--storage-root",
            tmp.path().to_str().unwrap(),
            "serve",
            "--tcp-port",
            "0",
            "--stop-after-ms",
            "200",
            "--model",
            "X",
        ])
        .output()
        .expect("spawn stratum");
    assert_eq!(
        output.status.code(),
        Some(1),
        "expected exit 1, stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("STRAT-E1001"),
        "expected STRAT-E1001 in stderr, got: {stderr}"
    );
    assert!(
        stderr.contains("provider-llama-cpp"),
        "expected provider-llama-cpp hint in stderr, got: {stderr}"
    );
}

#[cfg(feature = "provider-llama-cpp")]
#[test]
fn serve_with_unknown_model_slug_errors() {
    let tmp = TempDir::new().unwrap();
    let output = bin()
        .args([
            "--storage-root",
            tmp.path().to_str().unwrap(),
            "serve",
            "--tcp-port",
            "0",
            "--stop-after-ms",
            "200",
            "--model",
            "unknown-slug",
        ])
        .output()
        .expect("spawn stratum");
    assert_eq!(
        output.status.code(),
        Some(1),
        "expected exit 1, stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("STRAT-E1001"),
        "expected STRAT-E1001 in stderr, got: {stderr}"
    );
    assert!(
        stderr.contains("unknown slug"),
        "expected 'unknown slug' in stderr, got: {stderr}"
    );
}

/// End-to-end wiring proof for the feature-off configuration: spawn
/// the daemon WITHOUT `--model` (so the `EchoProvider` path is exercised),
/// parse the bound TCP port from the `--json` startup payload, open a
/// real TCP connection, send a JSON-RPC `ping`, and assert the response
/// contains `"id":1` with a `result` field — i.e. the daemon's
/// `AgentLoop` wiring still works under the new `resolve_serve_provider`
/// indirection.
///
/// The feature-on equivalent (a real `ping` round-trip with a working
/// `LlamaCppProvider`) is owned by the on-demand llama workflow because
/// it requires a real GGUF on disk and would otherwise blow the default
/// CI budget.
#[cfg(not(feature = "provider-llama-cpp"))]
#[test]
#[allow(
    clippy::too_many_lines,
    reason = "integration test inlines a daemon-spawn + timeout-bounded stdout reader + TCP round-trip; splitting fragments the test's narrative without removing complexity"
)]
fn serve_default_provider_accepts_jsonrpc_ping_over_tcp() {
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::TcpStream;
    use std::process::{Child, ChildStdout, Stdio};
    use std::sync::mpsc;
    use std::thread;
    use std::time::{Duration, Instant};

    // Mirror `serve_daemon.rs::read_startup_line`: hand stdout off to a
    // worker thread that does the actual `read_line`, then bound the
    // wait via `recv_timeout`. Without this, a buggy daemon that emits
    // an error on stderr but never writes to stdout would hang this
    // test until the CI watchdog killed the runner. Tracked as a
    // follow-up in the v1.0 cleanup batch (originally surfaced in the
    // PR #171 review comments).
    fn read_startup_line(stdout: ChildStdout, timeout: Duration, child: &mut Child) -> String {
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || {
            let mut reader = BufReader::new(stdout);
            let mut line = String::new();
            let res = reader.read_line(&mut line);
            let _ = tx.send((line, res));
        });
        if let Ok((line, res)) = rx.recv_timeout(timeout) {
            res.expect("read startup line");
            line
        } else {
            // Kill the child so it does not linger as a zombie holding
            // the stdout pipe open, then panic the test.
            let _ = child.kill();
            let _ = child.wait();
            panic!("child stdout read_line timed out");
        }
    }

    let tmp = TempDir::new().unwrap();
    let mut child = bin()
        .args([
            "--storage-root",
            tmp.path().to_str().unwrap(),
            "serve",
            "--tcp-port",
            "0",
            "--stop-after-ms",
            "5000",
            "--json",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn stratum");

    let stdout = child.stdout.take().expect("child stdout");
    let stderr = child.stderr.take().expect("child stderr");
    let line = read_startup_line(stdout, Duration::from_secs(10), &mut child);

    let parsed: serde_json::Value =
        serde_json::from_str(line.trim()).expect("startup line is valid JSON");
    let bound = parsed
        .get("bound")
        .and_then(|b| b.as_str())
        .map(str::to_string)
        .expect("`bound` field present");
    assert_eq!(
        parsed
            .get("provider")
            .and_then(|p| p.as_str())
            .expect("`provider` field present"),
        "echo"
    );

    let stderr_join = thread::spawn(move || {
        let mut buf = Vec::new();
        let _ = BufReader::new(stderr).read_to_end(&mut buf);
        buf
    });

    let response = {
        let deadline = Instant::now() + Duration::from_secs(2);
        let mut last_err: Option<std::io::Error> = None;
        loop {
            match TcpStream::connect(&bound) {
                Ok(mut stream) => {
                    stream
                        .set_read_timeout(Some(Duration::from_secs(2)))
                        .expect("rto");
                    stream
                        .write_all(br#"{"jsonrpc":"2.0","id":1,"method":"ping","params":{}}"#)
                        .expect("write");
                    stream.write_all(b"\n").expect("nl");
                    let mut buf = BufReader::new(stream);
                    let mut resp = String::new();
                    buf.read_line(&mut resp).expect("read response");
                    break resp;
                }
                Err(e) if Instant::now() < deadline => {
                    last_err = Some(e);
                    thread::sleep(Duration::from_millis(50));
                }
                Err(e) => panic!("connect failed: {e}; prior: {last_err:?}"),
            }
        }
    };

    let parsed: serde_json::Value =
        serde_json::from_str(response.trim()).expect("response is JSON");
    assert_eq!(parsed["jsonrpc"], "2.0");
    assert_eq!(parsed["id"], 1);
    assert!(
        parsed.get("result").is_some(),
        "expected `result`, got {response}"
    );

    let status = child.wait().expect("wait on child");
    assert!(
        status.success(),
        "child exit: {status:?}; stderr: {:?}",
        String::from_utf8_lossy(&stderr_join.join().expect("join stderr"))
    );
}
