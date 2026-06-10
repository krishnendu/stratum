//! Integration tests for `stratum client`.
//!
//! These spawn the actual CLI binary via `CARGO_BIN_EXE_stratum` and
//! drive a single JSON-RPC round-trip against a fake server that runs
//! inside the test process. The fake covers:
//!
//! * happy-path `result` over TCP (with `--json`),
//! * `--params` flows the JSON payload through to the server,
//! * mutually exclusive `--tcp` / `--unix-socket` rejection (exit 64),
//! * connection refused on a closed port (exit 1),
//! * server-side `error` envelope (exit 1, stderr mentions code/message),
//! * `--timeout-ms` against a server that never responds (exit 1 +
//!   `STRAT-E1001`),
//! * Unix-socket happy path (gated `cfg(unix)`),
//! * no transport flags + no daemon on the default loopback port (exit 1,
//!   connection refused).

// Integration test binary: every fn here exists only for `cargo test`.
// Test helpers panic on setup failures by design.
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    reason = "integration test helpers may panic on setup failures"
)]

use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use tempfile::TempDir;

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_stratum"))
}

/// Drain one newline-delimited JSON-RPC request line from `stream`,
/// decode it, and return the parsed `Value` along with the raw line.
fn read_request_line(stream: &TcpStream) -> serde_json::Value {
    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    reader.read_line(&mut line).expect("read line");
    serde_json::from_str(line.trim()).expect("request is JSON")
}

/// Bind a TCP loopback listener on a kernel-assigned port. Returns the
/// listener and the bound `host:port` string.
fn bind_loopback() -> (TcpListener, String) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let addr = listener.local_addr().expect("local_addr").to_string();
    (listener, addr)
}

/// Spawn a fake JSON-RPC server on `listener` that accepts a single
/// connection, reads one request line, and replies with a canned
/// `result: {"echo": <method>, "params": <params>}` envelope.
///
/// The shared `Mutex<Option<Value>>` is populated with the parsed
/// request so the test can assert on what the client actually sent.
fn spawn_echo_server(
    listener: TcpListener,
    captured: Arc<Mutex<Option<serde_json::Value>>>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        let (stream, _) = listener.accept().expect("accept");
        let request = read_request_line(&stream);
        let id = request
            .get("id")
            .cloned()
            .unwrap_or(serde_json::Value::Null);
        let method = request
            .get("method")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("")
            .to_string();
        let params = request
            .get("params")
            .cloned()
            .unwrap_or(serde_json::Value::Null);
        *captured.lock().unwrap() = Some(request);
        let response = serde_json::json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": { "echo": method, "params": params },
        });
        let mut writer = stream;
        writer
            .write_all(response.to_string().as_bytes())
            .expect("write");
        writer.write_all(b"\n").expect("nl");
    })
}

/// Spawn a fake server that replies with a JSON-RPC `error` envelope.
fn spawn_error_server(listener: TcpListener) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        let (stream, _) = listener.accept().expect("accept");
        let request = read_request_line(&stream);
        let id = request
            .get("id")
            .cloned()
            .unwrap_or(serde_json::Value::Null);
        let response = serde_json::json!({
            "jsonrpc": "2.0",
            "id": id,
            "error": { "code": -32601, "message": "method not found" },
        });
        let mut writer = stream;
        writer
            .write_all(response.to_string().as_bytes())
            .expect("write");
        writer.write_all(b"\n").expect("nl");
    })
}

/// Spawn a fake server that accepts the connection but never sends a
/// response — used to exercise the `--timeout-ms` deadline path.
fn spawn_silent_server(listener: TcpListener) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        let (stream, _) = listener.accept().expect("accept");
        // Hold the stream open until the test's --timeout-ms elapses.
        thread::sleep(Duration::from_secs(3));
        drop(stream);
    })
}

#[test]
fn client_ping_over_tcp_returns_ok_json() {
    let (listener, addr) = bind_loopback();
    let captured = Arc::new(Mutex::new(None));
    let join = spawn_echo_server(listener, captured);

    let output = bin()
        .args(["client", "--method", "ping", "--tcp", &addr, "--json"])
        .output()
        .expect("spawn stratum");

    join.join().expect("server join");

    assert!(
        output.status.success(),
        "exit: {:?}; stderr: {}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("\"echo\""), "stdout: {stdout}");
    assert!(stdout.contains("\"ping\""), "stdout: {stdout}");
}

#[test]
fn client_params_flows_through_to_server() {
    let (listener, addr) = bind_loopback();
    let captured = Arc::new(Mutex::new(None));
    let join = spawn_echo_server(listener, captured.clone());

    let output = bin()
        .args([
            "client",
            "--method",
            "ping",
            "--params",
            "{\"foo\":1}",
            "--tcp",
            &addr,
            "--json",
        ])
        .output()
        .expect("spawn stratum");

    join.join().expect("server join");

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    // The fake echoes params back inside result, so we can see the flag
    // landed on the wire.
    assert!(stdout.contains("\"foo\""), "stdout: {stdout}");
    let received = captured
        .lock()
        .unwrap()
        .clone()
        .expect("server saw a request");
    assert_eq!(received["params"]["foo"], 1);
    assert_eq!(received["jsonrpc"], "2.0");
    assert_eq!(received["method"], "ping");
}

#[test]
fn client_tcp_and_unix_socket_are_mutually_exclusive() {
    let output = bin()
        .args([
            "client",
            "--method",
            "ping",
            "--tcp",
            "127.0.0.1:1",
            "--unix-socket",
            "/tmp/stratum-client-test.sock",
        ])
        .output()
        .expect("spawn stratum");
    assert_eq!(output.status.code(), Some(64));
}

#[test]
fn client_connection_refused_exits_one() {
    // Bind a listener to claim a port, capture the address, then drop the
    // listener so the port is closed. The subsequent `client` call gets
    // a clean "connection refused" without any race against another test.
    let addr = {
        let (listener, addr) = bind_loopback();
        drop(listener);
        addr
    };
    let output = bin()
        .args([
            "client",
            "--method",
            "ping",
            "--tcp",
            &addr,
            "--timeout-ms",
            "500",
        ])
        .output()
        .expect("spawn stratum");
    assert_eq!(output.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("STRAT-E1001"), "stderr: {stderr}");
}

#[test]
fn client_error_response_exits_one_with_diagnostic() {
    let (listener, addr) = bind_loopback();
    let join = spawn_error_server(listener);

    let output = bin()
        .args(["client", "--method", "bogus", "--tcp", &addr])
        .output()
        .expect("spawn stratum");

    join.join().expect("server join");

    assert_eq!(output.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("-32601"), "stderr: {stderr}");
    assert!(stderr.contains("method not found"), "stderr: {stderr}");
}

#[test]
fn client_timeout_against_silent_server_exits_one() {
    let (listener, addr) = bind_loopback();
    let join = spawn_silent_server(listener);

    let output = bin()
        .args([
            "client",
            "--method",
            "ping",
            "--tcp",
            &addr,
            "--timeout-ms",
            "100",
        ])
        .output()
        .expect("spawn stratum");

    join.join().expect("server join");

    assert_eq!(output.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("STRAT-E1001"), "stderr: {stderr}");
    assert!(
        stderr.contains("timeout"),
        "expected timeout diagnostic, got: {stderr}"
    );
}

#[cfg(unix)]
#[test]
fn client_unix_socket_round_trip() {
    use std::os::unix::net::UnixListener;

    let tmp = TempDir::new().unwrap();
    let sock = tmp.path().join("stratum-client.sock");
    let listener = UnixListener::bind(&sock).expect("bind unix");

    let join = thread::spawn(move || {
        let (mut stream, _) = listener.accept().expect("accept");
        let request = {
            let mut reader = BufReader::new(&stream);
            let mut line = String::new();
            reader.read_line(&mut line).expect("read");
            serde_json::from_str::<serde_json::Value>(line.trim()).expect("json")
        };
        let id = request
            .get("id")
            .cloned()
            .unwrap_or(serde_json::Value::Null);
        let response = serde_json::json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": { "echo": "ping" },
        });
        stream
            .write_all(response.to_string().as_bytes())
            .expect("write");
        stream.write_all(b"\n").expect("nl");
    });

    let output = bin()
        .args([
            "client",
            "--method",
            "ping",
            "--unix-socket",
            sock.to_str().unwrap(),
            "--json",
        ])
        .output()
        .expect("spawn stratum");

    join.join().expect("server join");

    assert!(
        output.status.success(),
        "exit: {:?}; stderr: {}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("\"echo\""), "stdout: {stdout}");
    assert!(stdout.contains("\"ping\""), "stdout: {stdout}");
}

#[test]
fn client_default_tcp_when_no_daemon_listens() {
    // No --tcp / --unix-socket -> client defaults to 127.0.0.1:54321.
    // No daemon is listening there in this test process; expect exit 1
    // with a connection-refused diagnostic. (If something else on the
    // host happens to be listening on 54321 the test would falsely pass;
    // accepted because the default needs to be reasonable and this is
    // the documented contract.)
    let output = bin()
        .args(["client", "--method", "ping", "--timeout-ms", "500"])
        .output()
        .expect("spawn stratum");
    // Either succeeds (unlikely — some daemon already there) or fails
    // with our error code. We tolerate both but require the exit-1 path
    // to surface STRAT-E1001 when it does fail.
    if output.status.code() == Some(0) {
        return;
    }
    assert_eq!(output.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("STRAT-E1001"), "stderr: {stderr}");
}
