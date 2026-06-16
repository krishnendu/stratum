//! Integration tests for `stratum serve`.
//!
//! These spawn the real CLI binary via `CARGO_BIN_EXE_stratum` and drive
//! the daemon over a real loopback socket (or, on Unix, a real Unix
//! socket). They cover:
//!
//! * default TCP loopback with a stopwatch-driven shutdown,
//! * explicit `--tcp-port 0` plus `--json` startup payload shape,
//! * Unix-socket binding (gated `cfg(unix)`),
//! * mutually-exclusive `--unix-socket` / `--tcp-port` rejection,
//! * a real JSON-RPC `ping` round-trip through the wire — the integration
//!   test that proves the runtime wiring is hooked up correctly.

// Integration test binary: every fn here exists only for `cargo test`.
// Test helpers panic on setup failures by design; clippy's
// `expect_used` / `unwrap_used` denials are scoped to non-test code.
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    reason = "integration test helpers may panic on setup failures"
)]

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::process::{ChildStdout, Command, Stdio};
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
/// `read_line`, then wait on a bounded `recv_timeout`.
fn read_startup_line(stdout: ChildStdout, timeout: Duration) -> String {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let mut reader = BufReader::new(stdout);
        let mut line = String::new();
        let res = reader.read_line(&mut line);
        let _ = tx.send((line, res));
    });
    let (line, res) = rx
        .recv_timeout(timeout)
        .expect("child stdout read_line timed out");
    res.expect("read startup line");
    line
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

#[test]
fn serve_tcp_zero_json_prints_bound_address_and_exits_zero() {
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
            "--json",
        ])
        .output()
        .expect("spawn stratum");
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("utf8");
    let bound = bound_from_json(&stdout);
    assert!(
        bound.starts_with("127.0.0.1:"),
        "expected loopback addr, got {bound}"
    );
}

#[test]
fn serve_no_flags_defaults_to_tcp_loopback() {
    let tmp = TempDir::new().unwrap();
    let output = bin()
        .args([
            "--storage-root",
            tmp.path().to_str().unwrap(),
            "serve",
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
        "expected provider=echo, got: {stdout}"
    );
    assert!(
        stdout.contains("listening on 127.0.0.1:"),
        "expected loopback addr, got: {stdout}"
    );
}

#[cfg(unix)]
#[test]
fn serve_unix_socket_json_reports_socket_path() {
    let tmp = TempDir::new().unwrap();
    // Place the socket inside the temp dir so the test is hermetic and
    // never fights with a stale `/tmp` entry left by a prior run.
    let sock = tmp.path().join("stratum-test.sock");
    let output = bin()
        .args([
            "--storage-root",
            tmp.path().to_str().unwrap(),
            "serve",
            "--unix-socket",
            sock.to_str().unwrap(),
            "--stop-after-ms",
            "200",
            "--json",
        ])
        .output()
        .expect("spawn stratum");
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("utf8");
    let bound = bound_from_json(&stdout);
    assert!(
        bound.contains("stratum-test.sock"),
        "expected socket path in bound, got {bound}"
    );
}

#[test]
fn serve_unix_socket_and_tcp_port_are_mutually_exclusive() {
    let tmp = TempDir::new().unwrap();
    let output = bin()
        .args([
            "--storage-root",
            tmp.path().to_str().unwrap(),
            "serve",
            "--unix-socket",
            "/tmp/x.sock",
            "--tcp-port",
            "5000",
        ])
        .output()
        .expect("spawn stratum");
    assert_eq!(output.status.code(), Some(64));
}

/// End-to-end wiring proof: spawn the daemon, parse the bound TCP port
/// from its `--json` startup payload, open a TCP connection, send a
/// JSON-RPC `ping`, and assert the response contains `"id":1`.
///
/// This is the only test that exercises the live socket — every other
/// case relies on the stopwatch shutdown happening before the test
/// finishes.
#[test]
fn serve_accepts_jsonrpc_ping_over_tcp() {
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

    // The CLI flushes stdout after writing the startup JSON, so a single
    // `read_line` on the child's stdout is enough to grab the address.
    // The 10 s ceiling guards against a daemon that fails on stderr but
    // never writes to stdout — see `read_startup_line`.
    let stdout = child.stdout.take().expect("child stdout");
    let stderr = child.stderr.take().expect("child stderr");
    let line = read_startup_line(stdout, Duration::from_secs(10));
    let bound = bound_from_json(&line);

    // Drain stderr in the background so the child doesn't block on a
    // full pipe.
    let stderr_join = thread::spawn(move || {
        let mut buf = Vec::new();
        let _ = BufReader::new(stderr).read_to_end(&mut buf);
        buf
    });

    // Connect and round-trip a ping. Retry briefly because the bound
    // address may be reported a hair before the acceptor's first poll
    // cycle on slow CI.
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

    // Wait for the stopwatch (5s) to bring the daemon down. To keep the
    // test fast we don't wait the full 5s — we just confirm the child
    // exits cleanly when it does.
    let status = child.wait().expect("wait on child");
    assert!(
        status.success(),
        "child exit: {status:?}; stderr: {:?}",
        String::from_utf8_lossy(&stderr_join.join().expect("join stderr"))
    );
}
