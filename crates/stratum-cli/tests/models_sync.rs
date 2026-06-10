//! Integration tests for `stratum models sync`.
//!
//! Each test spawns the built binary via `CARGO_BIN_EXE_stratum` against a
//! `--storage-root` tempdir. The two catalog sources (`--manifest-file`
//! and `--manifest-url`) are exercised independently:
//!
//! * `--manifest-file <PATH>` is fed a JSON fixture produced from the runtime
//!   `ModelCatalog` types so the on-disk round-trip is byte-stable.
//! * `--manifest-url <URL>` is pointed at an in-process `TcpListener`-backed
//!   HTTP/1.0 server bound to `127.0.0.1:<ephemeral>`. The CLI's TLS-only
//!   guard is bypassed via the hidden `--allow-insecure-url` flag (gated by
//!   `cfg(debug_assertions)` OR `STRATUM_ALLOW_INSECURE_URL=1`).

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

const GOOD_SHA: &str = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824";

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_stratum"))
}

fn run(root: &Path, extra: &[&str]) -> std::process::Output {
    let mut cmd = bin();
    cmd.args(["--storage-root", root.to_str().expect("tempdir utf-8")]);
    cmd.args(extra);
    cmd.output().expect("spawn stratum")
}

/// Build a valid two-entry `ModelCatalog` JSON. Matches the runtime
/// `model_catalog` schema (`schema_version` 1).
fn sample_catalog_json() -> String {
    format!(
        r#"{{
            "schema_version": 1,
            "entries": {{
                "alpha": {{
                    "slug": "alpha",
                    "family": "llama",
                    "display_name": "Alpha",
                    "tier": "low",
                    "task": ["chat"],
                    "size_mib": 128,
                    "quantization": "Q4_K_M",
                    "artifact": {{
                        "url": "https://example.com/alpha.gguf",
                        "sha256": "{GOOD_SHA}",
                        "bytes": 1024
                    }},
                    "license": "Apache-2.0",
                    "homepage": null
                }},
                "beta": {{
                    "slug": "beta",
                    "family": "llama",
                    "display_name": "Beta",
                    "tier": "low",
                    "task": ["chat"],
                    "size_mib": 256,
                    "quantization": "Q4_K_M",
                    "artifact": {{
                        "url": "https://example.com/beta.gguf",
                        "sha256": "{GOOD_SHA}",
                        "bytes": 2048
                    }},
                    "license": "Apache-2.0",
                    "homepage": null
                }}
            }}
        }}"#
    )
}

fn write_fixture(dir: &Path, name: &str, body: &str) -> PathBuf {
    let path = dir.join(name);
    std::fs::write(&path, body).expect("write fixture");
    path
}

/// Start a one-shot HTTP/1.0 server that answers exactly one GET with
/// `body` (Content-Type: application/json) and then exits. Returns
/// `(url, join_handle)`.
fn spawn_catalog_server(body: String) -> (String, thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind 127.0.0.1:0");
    let port = listener.local_addr().expect("local_addr").port();
    let url = format!("http://127.0.0.1:{port}/catalog.json");
    let handle = thread::spawn(move || {
        if let Ok((mut stream, _)) = listener.accept() {
            let mut buf = [0_u8; 4096];
            let _ = stream.read(&mut buf);
            let header = format!(
                "HTTP/1.0 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            let _ = stream.write_all(header.as_bytes());
            let _ = stream.write_all(body.as_bytes());
            let _ = stream.flush();
        }
    });
    (url, handle)
}

fn state_dir(root: &Path) -> PathBuf {
    root.join("state")
}

#[test]
fn sync_from_manifest_file_writes_default_catalog_path() {
    let tmp = TempDir::new().unwrap();
    let fixture = write_fixture(tmp.path(), "src.json", &sample_catalog_json());

    let out = run(
        tmp.path(),
        &[
            "models",
            "sync",
            "--manifest-file",
            fixture.to_str().unwrap(),
        ],
    );
    assert!(
        out.status.success(),
        "exit={:?} stderr={}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("synced"), "stdout: {stdout}");
    assert!(stdout.contains("entries=2"), "stdout: {stdout}");

    let written = state_dir(tmp.path()).join("models.json");
    assert!(written.exists(), "catalog must be written to default path");
    let body = std::fs::read_to_string(&written).unwrap();
    assert!(body.contains("\"alpha\""));
    assert!(body.contains("\"beta\""));
}

#[test]
fn sync_from_missing_manifest_file_exits_1_with_strat_e1001() {
    let tmp = TempDir::new().unwrap();
    let bogus = tmp.path().join("does-not-exist.json");

    let out = run(
        tmp.path(),
        &["models", "sync", "--manifest-file", bogus.to_str().unwrap()],
    );
    assert_eq!(out.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("STRAT-E1001"), "stderr: {stderr}");
}

#[test]
fn sync_from_malformed_manifest_file_exits_1_with_strat_e1001() {
    let tmp = TempDir::new().unwrap();
    let fixture = write_fixture(tmp.path(), "bad.json", "{this is not json");

    let out = run(
        tmp.path(),
        &[
            "models",
            "sync",
            "--manifest-file",
            fixture.to_str().unwrap(),
        ],
    );
    assert_eq!(out.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("STRAT-E1001"), "stderr: {stderr}");
}

#[test]
fn sync_with_explicit_out_writes_to_custom_path() {
    let tmp = TempDir::new().unwrap();
    let fixture = write_fixture(tmp.path(), "src.json", &sample_catalog_json());
    let custom = tmp.path().join("nested").join("custom-models.json");

    let out = run(
        tmp.path(),
        &[
            "models",
            "sync",
            "--manifest-file",
            fixture.to_str().unwrap(),
            "--out",
            custom.to_str().unwrap(),
        ],
    );
    assert!(
        out.status.success(),
        "exit={:?} stderr={}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(custom.exists(), "custom out path must exist");
    // Default path must NOT exist when --out redirected.
    let default = state_dir(tmp.path()).join("models.json");
    assert!(
        !default.exists(),
        "default catalog must not be written when --out is set"
    );
}

#[test]
fn sync_channel_beta_round_trips_through_json_summary() {
    let tmp = TempDir::new().unwrap();
    let fixture = write_fixture(tmp.path(), "src.json", &sample_catalog_json());

    let out = run(
        tmp.path(),
        &[
            "models",
            "sync",
            "--manifest-file",
            fixture.to_str().unwrap(),
            "--channel",
            "beta",
            "--json",
        ],
    );
    assert!(
        out.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    let value: serde_json::Value = serde_json::from_str(stdout.trim()).expect("valid json");
    assert_eq!(value["channel"], "beta");
    assert_eq!(value["entries"], 2);
    let url = value["url"].as_str().expect("url string");
    assert!(url.starts_with("file://"), "got: {url}");
    let out_path = value["out"].as_str().expect("out string");
    assert!(out_path.ends_with("models.json"), "got: {out_path}");
}

#[test]
fn sync_from_loopback_url_with_allow_insecure_succeeds() {
    let tmp = TempDir::new().unwrap();
    let (url, handle) = spawn_catalog_server(sample_catalog_json());

    let out = run(
        tmp.path(),
        &[
            "models",
            "sync",
            "--manifest-url",
            &url,
            "--allow-insecure-url",
            "--json",
        ],
    );
    let _ = handle.join();
    assert!(
        out.status.success(),
        "exit={:?} stderr={}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    let value: serde_json::Value = serde_json::from_str(stdout.trim()).expect("valid json");
    assert_eq!(value["entries"], 2);
    assert_eq!(value["url"], url);
    let written = state_dir(tmp.path()).join("models.json");
    assert!(written.exists());
}

#[test]
fn sync_from_loopback_url_without_allow_insecure_exits_1() {
    let tmp = TempDir::new().unwrap();
    // Bind a listener so the URL has a real port, but don't bother handling
    // the request — the CLI must reject the non-https URL before connecting.
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let port = listener.local_addr().expect("addr").port();
    let url = format!("http://127.0.0.1:{port}/catalog.json");

    let out = run(tmp.path(), &["models", "sync", "--manifest-url", &url]);
    drop(listener);
    assert_eq!(out.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("STRAT-E1001"), "stderr: {stderr}");
    assert!(stderr.contains("https"), "stderr: {stderr}");
}
