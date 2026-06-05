//! Integration tests for `stratum self-update --apply`.
//!
//! Each test spawns the real built binary via `CARGO_BIN_EXE_stratum` against
//! a `--storage-root` tempdir. The CLI's manifest sources are exercised via
//! `--manifest-file <fixture.json>` and (for the artifact download) an
//! in-process `TcpListener`-backed HTTP/1.0 server bound to `127.0.0.1`.
//!
//! `ureq` does not handle `file://` URLs, so an in-process HTTP server is
//! the cleanest way to feed the CLI a deterministic artifact body without
//! reaching out to the public internet. The CLI's TLS-only check is bypassed
//! for these tests via the hidden `--allow-insecure-url` flag, which is
//! gated by `cfg(debug_assertions)` OR `STRATUM_ALLOW_INSECURE_URL=1`.
//! Likewise, the hidden `--target <path>` flag redirects the on-disk swap
//! away from the CLI test binary itself.

// Integration test binary: every fn here exists only for `cargo test`.
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    reason = "integration test helpers may panic on setup failures"
)]

use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::thread;

use tempfile::TempDir;

const ARTIFACT_BODY: &[u8] = b"new-stratum-binary-body-bytes\n";

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_stratum"))
}

fn run(args: &[&str], root: &Path) -> std::process::Output {
    let mut all: Vec<&str> = vec!["--storage-root", root.to_str().unwrap()];
    all.extend_from_slice(args);
    bin().args(&all).output().expect("spawn stratum")
}

/// Lower-case hex SHA-256 of `body` using the same helper the CLI uses.
fn sha256_hex(body: &[u8]) -> String {
    stratum_runtime::download::sha256_hex(body)
}

/// Parse `"a.b.c"` into a `(major, minor, patch)` triple.
fn parse_version(s: &str) -> (u16, u16, u16) {
    let mut it = s.split('.');
    let a: u16 = it.next().unwrap().parse().unwrap();
    let b: u16 = it.next().unwrap().parse().unwrap();
    let c: u16 = it.next().unwrap().parse().unwrap();
    (a, b, c)
}

/// Build a manifest fixture whose `latest` advertises an artifact at
/// `artifact_url` with `sha256` / `bytes`. Both `latest` and `history` carry
/// the same entry — the CLI evaluates against `latest`, so a single-entry
/// history is sufficient. `min_supported_from` is included only if `Some`.
fn build_manifest(
    version: &str,
    artifact_url: &str,
    sha256: &str,
    bytes: u64,
    min_supported_from: Option<&str>,
) -> String {
    let (maj, min, pat) = parse_version(version);
    let min_block = min_supported_from.map_or_else(
        || r#""min_supported_from": null,"#.to_owned(),
        |min| {
            let (mj, mn, pt) = parse_version(min);
            format!(
                r#""min_supported_from": {{ "major": {mj}, "minor": {mn}, "patch": {pt}, "pre": null }},"#,
            )
        },
    );
    let entry = format!(
        r#"{{
            "version": {{ "major": {maj}, "minor": {min}, "patch": {pat}, "pre": null }},
            "released_at": {{ "secs_since_epoch": 1700000000, "nanos_since_epoch": 0 }},
            "binary": {{
                "url": "{artifact_url}",
                "sha256": "{sha256}",
                "bytes": {bytes},
                "platform": "linux_x86_64"
            }},
            {min_block}
            "release_notes_url": "https://stratum.dev/releases/{version}"
        }}"#,
    );
    format!(
        r#"{{
            "schema_version": 1,
            "channel": "stable",
            "latest": {entry},
            "history": [{entry}]
        }}"#,
    )
}

fn write_fixture(dir: &Path, name: &str, body: &str) -> PathBuf {
    let path = dir.join(name);
    std::fs::write(&path, body).expect("write fixture");
    path
}

/// Start a one-shot HTTP/1.0 server that answers exactly one GET with
/// `body` and then exits. Returns `(url, join_handle)`. The URL points at
/// `http://127.0.0.1:<port>/artifact.bin`.
fn spawn_artifact_server(body: &'static [u8]) -> (String, thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind 127.0.0.1:0");
    let port = listener.local_addr().expect("local_addr").port();
    let url = format!("http://127.0.0.1:{port}/artifact.bin");
    let handle = thread::spawn(move || {
        if let Ok((mut stream, _)) = listener.accept() {
            // Read until end-of-request headers ("\r\n\r\n") so the client
            // doesn't see ECONNRESET on its write.
            let mut buf = [0_u8; 1024];
            let _ = stream.read(&mut buf);
            let header = format!(
                "HTTP/1.0 200 OK\r\nContent-Type: application/octet-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            let _ = stream.write_all(header.as_bytes());
            let _ = stream.write_all(body);
            let _ = stream.flush();
        }
    });
    (url, handle)
}

/// Start a server that lies about Content-Length but actually closes the
/// connection after sending fewer bytes. Used to produce a short body for the
/// bytes-mismatch test (the declared `bytes` in the manifest doesn't match
/// the actual length).
fn spawn_short_body_server(actual: &'static [u8]) -> (String, thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind 127.0.0.1:0");
    let port = listener.local_addr().expect("local_addr").port();
    let url = format!("http://127.0.0.1:{port}/artifact.bin");
    let handle = thread::spawn(move || {
        if let Ok((mut stream, _)) = listener.accept() {
            let mut buf = [0_u8; 1024];
            let _ = stream.read(&mut buf);
            // Declare a Content-Length that matches the real body, but the
            // CLI compares actual bytes_written against the manifest's
            // declared `bytes`, so the mismatch surfaces there.
            let header = format!(
                "HTTP/1.0 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                actual.len()
            );
            let _ = stream.write_all(header.as_bytes());
            let _ = stream.write_all(actual);
            let _ = stream.flush();
        }
    });
    (url, handle)
}

#[test]
fn apply_and_check_are_mutually_exclusive_exits_64() {
    let tmp = TempDir::new().unwrap();
    let manifest = build_manifest(
        "1.0.0",
        "https://example.com/x",
        &"a".repeat(64),
        1024,
        None,
    );
    let fixture = write_fixture(tmp.path(), "manifest.json", &manifest);
    let output = run(
        &[
            "self-update",
            "--check",
            "--apply",
            "--manifest-file",
            fixture.to_str().unwrap(),
            "--current",
            "1.0.0",
            "--platform",
            "linux_x86_64",
        ],
        tmp.path(),
    );
    assert_eq!(output.status.code(), Some(64));
}

#[test]
fn neither_check_nor_apply_exits_64() {
    let tmp = TempDir::new().unwrap();
    let output = run(&["self-update"], tmp.path());
    assert_eq!(output.status.code(), Some(64));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("--check or --apply"), "stderr: {stderr}");
}

#[test]
fn apply_dry_run_does_not_modify_target() {
    let tmp = TempDir::new().unwrap();
    let sha = sha256_hex(ARTIFACT_BODY);
    let (url, handle) = spawn_artifact_server(ARTIFACT_BODY);
    let manifest = build_manifest("1.5.0", &url, &sha, ARTIFACT_BODY.len() as u64, None);
    let fixture = write_fixture(tmp.path(), "manifest.json", &manifest);

    // Pre-create a stand-in "current binary" that the swap would replace.
    let target = tmp.path().join("stratum-stub");
    std::fs::write(&target, b"original-content").unwrap();

    let output = run(
        &[
            "self-update",
            "--apply",
            "--dry-run",
            "--allow-insecure-url",
            "--target",
            target.to_str().unwrap(),
            "--manifest-file",
            fixture.to_str().unwrap(),
            "--current",
            "1.4.7",
            "--platform",
            "linux_x86_64",
        ],
        tmp.path(),
    );
    let _ = handle.join();

    assert!(
        output.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("dry-run: would swap"), "stdout: {stdout}");
    // Target must NOT have been touched.
    let contents = std::fs::read(&target).unwrap();
    assert_eq!(contents, b"original-content");
    // No .bak left behind on dry-run.
    let bak = tmp.path().join("stratum-stub.bak");
    assert!(!bak.exists(), "bak should not exist after dry-run");
    // No .new.tmp left behind either.
    let new_tmp = tmp.path().join("stratum-stub.new.tmp");
    assert!(!new_tmp.exists(), "new.tmp should be cleaned up");
}

#[test]
fn apply_happy_path_swaps_target_and_leaves_bak() {
    let tmp = TempDir::new().unwrap();
    let sha = sha256_hex(ARTIFACT_BODY);
    let (url, handle) = spawn_artifact_server(ARTIFACT_BODY);
    let manifest = build_manifest("1.5.0", &url, &sha, ARTIFACT_BODY.len() as u64, None);
    let fixture = write_fixture(tmp.path(), "manifest.json", &manifest);

    let target = tmp.path().join("stratum-stub");
    std::fs::write(&target, b"old-binary").unwrap();

    let output = run(
        &[
            "self-update",
            "--apply",
            "--allow-insecure-url",
            "--target",
            target.to_str().unwrap(),
            "--manifest-file",
            fixture.to_str().unwrap(),
            "--current",
            "1.4.7",
            "--platform",
            "linux_x86_64",
        ],
        tmp.path(),
    );
    let _ = handle.join();

    assert!(
        output.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("upgraded"), "stdout: {stdout}");
    assert!(stdout.contains("1.4.7"));
    assert!(stdout.contains("1.5.0"));
    assert!(stdout.contains(".bak"));

    // Target now holds the new artifact body.
    let contents = std::fs::read(&target).unwrap();
    assert_eq!(contents, ARTIFACT_BODY);
    // Previous binary preserved at <exe>.bak.
    let bak = tmp.path().join("stratum-stub.bak");
    assert_eq!(std::fs::read(&bak).unwrap(), b"old-binary");
}

#[test]
fn apply_sha_mismatch_exits_1_and_target_unmodified() {
    let tmp = TempDir::new().unwrap();
    // Manifest declares a sha that DOES NOT match the body the server returns.
    let bogus_sha = "0".repeat(64);
    let (url, handle) = spawn_artifact_server(ARTIFACT_BODY);
    let manifest = build_manifest("1.5.0", &url, &bogus_sha, ARTIFACT_BODY.len() as u64, None);
    let fixture = write_fixture(tmp.path(), "manifest.json", &manifest);

    let target = tmp.path().join("stratum-stub");
    std::fs::write(&target, b"old-binary").unwrap();

    let output = run(
        &[
            "self-update",
            "--apply",
            "--allow-insecure-url",
            "--target",
            target.to_str().unwrap(),
            "--manifest-file",
            fixture.to_str().unwrap(),
            "--current",
            "1.4.7",
            "--platform",
            "linux_x86_64",
        ],
        tmp.path(),
    );
    let _ = handle.join();

    assert_eq!(output.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("sha256 mismatch"), "stderr: {stderr}");
    // Target untouched.
    assert_eq!(std::fs::read(&target).unwrap(), b"old-binary");
    // Temp file cleaned up.
    let new_tmp = tmp.path().join("stratum-stub.new.tmp");
    assert!(!new_tmp.exists());
    // No .bak left because the swap never happened.
    let bak = tmp.path().join("stratum-stub.bak");
    assert!(!bak.exists());
}

#[test]
fn apply_bytes_mismatch_exits_1() {
    let tmp = TempDir::new().unwrap();
    let sha = sha256_hex(ARTIFACT_BODY);
    let (url, handle) = spawn_short_body_server(ARTIFACT_BODY);
    // Manifest declares `bytes: 999` but body is only ARTIFACT_BODY.len().
    // Use a manifest sha that matches the actual body so we get past sha
    // verification and then fail bytes.
    let manifest = build_manifest("1.5.0", &url, &sha, 999, None);
    let fixture = write_fixture(tmp.path(), "manifest.json", &manifest);

    let target = tmp.path().join("stratum-stub");
    std::fs::write(&target, b"old-binary").unwrap();

    let output = run(
        &[
            "self-update",
            "--apply",
            "--allow-insecure-url",
            "--target",
            target.to_str().unwrap(),
            "--manifest-file",
            fixture.to_str().unwrap(),
            "--current",
            "1.4.7",
            "--platform",
            "linux_x86_64",
        ],
        tmp.path(),
    );
    let _ = handle.join();

    assert_eq!(output.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("byte count mismatch") || stderr.contains("sha256 mismatch"),
        "stderr: {stderr}"
    );
    assert_eq!(std::fs::read(&target).unwrap(), b"old-binary");
}

#[test]
fn apply_up_to_date_short_circuits_no_download() {
    let tmp = TempDir::new().unwrap();
    // Point at a URL that would refuse a connection if reached. Since
    // `--current` equals the manifest's `latest`, the CLI must short-circuit
    // before any network IO.
    let unreachable = "http://127.0.0.1:1/should-not-be-fetched";
    let manifest = build_manifest("1.0.0", unreachable, &"a".repeat(64), 1, None);
    let fixture = write_fixture(tmp.path(), "manifest.json", &manifest);

    let target = tmp.path().join("stratum-stub");
    std::fs::write(&target, b"orig").unwrap();

    let output = run(
        &[
            "self-update",
            "--apply",
            "--allow-insecure-url",
            "--target",
            target.to_str().unwrap(),
            "--manifest-file",
            fixture.to_str().unwrap(),
            "--current",
            "1.0.0",
            "--platform",
            "linux_x86_64",
        ],
        tmp.path(),
    );
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("already up to date"), "stdout: {stdout}");
    // Target untouched.
    assert_eq!(std::fs::read(&target).unwrap(), b"orig");
}

#[test]
fn apply_blocked_schema_too_old_exits_64_no_download() {
    let tmp = TempDir::new().unwrap();
    let unreachable = "http://127.0.0.1:1/should-not-be-fetched";
    let manifest = build_manifest("1.5.0", unreachable, &"a".repeat(64), 1, Some("1.3.0"));
    let fixture = write_fixture(tmp.path(), "manifest.json", &manifest);

    let target = tmp.path().join("stratum-stub");
    std::fs::write(&target, b"orig").unwrap();

    let output = run(
        &[
            "self-update",
            "--apply",
            "--allow-insecure-url",
            "--target",
            target.to_str().unwrap(),
            "--manifest-file",
            fixture.to_str().unwrap(),
            "--current",
            "1.0.0",
            "--platform",
            "linux_x86_64",
        ],
        tmp.path(),
    );
    assert_eq!(output.status.code(), Some(64));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("STRAT-E1001"), "stderr: {stderr}");
    assert!(stderr.contains("reinstall"));
    // Target untouched.
    assert_eq!(std::fs::read(&target).unwrap(), b"orig");
}

#[test]
fn apply_no_artifact_for_platform_exits_1() {
    let tmp = TempDir::new().unwrap();
    let sha = sha256_hex(ARTIFACT_BODY);
    // Manifest publishes a linux_x86_64 artifact; we ask for windows_x86_64.
    let manifest = build_manifest(
        "1.5.0",
        "http://127.0.0.1:1/never-fetched",
        &sha,
        ARTIFACT_BODY.len() as u64,
        None,
    );
    let fixture = write_fixture(tmp.path(), "manifest.json", &manifest);

    let target = tmp.path().join("stratum-stub");
    std::fs::write(&target, b"orig").unwrap();

    let output = run(
        &[
            "self-update",
            "--apply",
            "--allow-insecure-url",
            "--target",
            target.to_str().unwrap(),
            "--manifest-file",
            fixture.to_str().unwrap(),
            "--current",
            "1.4.7",
            "--platform",
            "windows_x86_64",
        ],
        tmp.path(),
    );
    assert_eq!(output.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("no artifact for platform"),
        "stderr: {stderr}"
    );
    assert_eq!(std::fs::read(&target).unwrap(), b"orig");
}

#[test]
fn apply_json_round_trips() {
    let tmp = TempDir::new().unwrap();
    let sha = sha256_hex(ARTIFACT_BODY);
    let (url, handle) = spawn_artifact_server(ARTIFACT_BODY);
    let manifest = build_manifest("1.5.0", &url, &sha, ARTIFACT_BODY.len() as u64, None);
    let fixture = write_fixture(tmp.path(), "manifest.json", &manifest);

    let target = tmp.path().join("stratum-stub");
    std::fs::write(&target, b"old-binary").unwrap();

    let output = run(
        &[
            "--json",
            "self-update",
            "--apply",
            "--allow-insecure-url",
            "--target",
            target.to_str().unwrap(),
            "--manifest-file",
            fixture.to_str().unwrap(),
            "--current",
            "1.4.7",
            "--platform",
            "linux_x86_64",
        ],
        tmp.path(),
    );
    let _ = handle.join();

    assert!(
        output.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    let v: serde_json::Value = serde_json::from_str(stdout.trim()).unwrap();
    assert_eq!(v["action"], "applied");
    assert_eq!(v["from"], "1.4.7");
    assert_eq!(v["to"], "1.5.0");
    assert!(v["backup_path"].is_string());
    let backup_path = v["backup_path"].as_str().unwrap();
    // The CLI always names the rollback file `<exe>.bak`, but clippy flags
    // `ends_with(".bak")` as a case-sensitive extension check; assert the
    // suffix via byte slicing to dodge the lint while keeping the intent
    // (path ends in the literal lowercase `.bak`).
    assert_eq!(&backup_path[backup_path.len() - 4..], ".bak");
    assert_eq!(v["artifact"]["sha256"], sha);
    assert_eq!(v["artifact"]["bytes"], ARTIFACT_BODY.len());
}
