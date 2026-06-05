//! End-to-end tests for the `stratum models` catalog subcommands.
//!
//! These spawn the built binary via `CARGO_BIN_EXE_stratum` against a
//! `TempDir`-rooted `--storage-root`, exercising the `ModelCatalog`-backed
//! list/add/remove/recommend/validate surface introduced in this PR.

// Integration test binary: every fn here exists only for `cargo test`. Test
// helpers panic on setup failures by design; clippy's `expect_used` /
// `unwrap_used` / `panic` denials are scoped to non-test code.
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    reason = "integration test helpers may panic on setup failures"
)]

use std::path::Path;
use std::process::Command;

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

fn add_entry(root: &Path, slug: &str, tier: &str, task: &str, size_mib: &str) {
    let out = run(
        root,
        &[
            "models",
            "add",
            "--slug",
            slug,
            "--family",
            "llama",
            "--display-name",
            "Display",
            "--tier",
            tier,
            "--task",
            task,
            "--size-mib",
            size_mib,
            "--quantization",
            "Q4_K_M",
            "--url",
            "https://example.com/m.gguf",
            "--sha256",
            GOOD_SHA,
            "--bytes",
            "1024",
            "--license",
            "Apache-2.0",
        ],
    );
    assert!(
        out.status.success(),
        "add failed: stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn list_missing_catalog_is_empty() {
    let tmp = TempDir::new().unwrap();
    let out = run(tmp.path(), &["models", "list"]);
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("no catalog entries"), "got: {stdout}");
}

#[test]
fn add_then_list_sees_entry() {
    let tmp = TempDir::new().unwrap();
    add_entry(tmp.path(), "tiny-chat", "low", "chat", "100");

    // Catalog file must exist under <state>/models.json — Paths::under puts
    // state at <root>/state.
    let catalog = tmp.path().join("state").join("models.json");
    assert!(catalog.exists(), "catalog not written");

    let out = run(tmp.path(), &["models", "list"]);
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("tiny-chat"), "got: {stdout}");
}

#[test]
fn add_rejects_invalid_sha256() {
    let tmp = TempDir::new().unwrap();
    let out = run(
        tmp.path(),
        &[
            "models",
            "add",
            "--slug",
            "x",
            "--family",
            "llama",
            "--display-name",
            "X",
            "--tier",
            "low",
            "--task",
            "chat",
            "--size-mib",
            "1",
            "--quantization",
            "Q",
            "--url",
            "https://example.com/x",
            "--sha256",
            "deadbeef",
            "--bytes",
            "1",
            "--license",
            "MIT",
        ],
    );
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stderr).contains("invalid artifact"));
}

#[test]
fn add_rejects_non_https_url() {
    let tmp = TempDir::new().unwrap();
    let out = run(
        tmp.path(),
        &[
            "models",
            "add",
            "--slug",
            "x",
            "--family",
            "llama",
            "--display-name",
            "X",
            "--tier",
            "low",
            "--task",
            "chat",
            "--size-mib",
            "1",
            "--quantization",
            "Q",
            "--url",
            "http://insecure.example.com/x",
            "--sha256",
            GOOD_SHA,
            "--bytes",
            "1",
            "--license",
            "MIT",
        ],
    );
    assert_eq!(out.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("https"), "got stderr: {stderr}");
}

#[test]
fn add_rejects_malformed_slug() {
    let tmp = TempDir::new().unwrap();
    let out = run(
        tmp.path(),
        &[
            "models",
            "add",
            "--slug",
            "BAD-Slug-With-Upper",
            "--family",
            "llama",
            "--display-name",
            "X",
            "--tier",
            "low",
            "--task",
            "chat",
            "--size-mib",
            "1",
            "--quantization",
            "Q",
            "--url",
            "https://example.com/x",
            "--sha256",
            GOOD_SHA,
            "--bytes",
            "1",
            "--license",
            "MIT",
        ],
    );
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stderr).contains("invalid --slug"));
}

#[test]
fn list_filter_by_tier() {
    let tmp = TempDir::new().unwrap();
    add_entry(tmp.path(), "low-1", "low", "chat", "100");
    add_entry(tmp.path(), "med-1", "medium", "chat", "200");
    add_entry(tmp.path(), "high-1", "high", "chat", "400");

    let out = run(tmp.path(), &["--json", "models", "list", "--tier", "high"]);
    assert!(out.status.success());
    let stdout = String::from_utf8(out.stdout).unwrap();
    let v: serde_json::Value = serde_json::from_str(stdout.trim()).unwrap();
    let arr = v.as_array().unwrap();
    assert_eq!(arr.len(), 1);
    assert_eq!(arr[0]["slug"], "high-1");
}

#[test]
fn list_filter_by_task() {
    let tmp = TempDir::new().unwrap();
    add_entry(tmp.path(), "chat-only", "low", "chat", "100");
    add_entry(tmp.path(), "code-only", "low", "code", "200");

    let out = run(tmp.path(), &["--json", "models", "list", "--task", "code"]);
    assert!(out.status.success());
    let stdout = String::from_utf8(out.stdout).unwrap();
    let v: serde_json::Value = serde_json::from_str(stdout.trim()).unwrap();
    let arr = v.as_array().unwrap();
    assert_eq!(arr.len(), 1);
    assert_eq!(arr[0]["slug"], "code-only");
}

#[test]
fn list_json_round_trip() {
    let tmp = TempDir::new().unwrap();
    add_entry(tmp.path(), "alpha", "low", "chat", "100");
    let out = run(tmp.path(), &["--json", "models", "list"]);
    assert!(out.status.success());
    let stdout = String::from_utf8(out.stdout).unwrap();
    let v: serde_json::Value = serde_json::from_str(stdout.trim()).unwrap();
    assert!(v.is_array());
    assert_eq!(v[0]["slug"], "alpha");
    assert_eq!(v[0]["family"], "llama");
}

#[test]
fn remove_then_list_empty() {
    let tmp = TempDir::new().unwrap();
    add_entry(tmp.path(), "removeme", "low", "chat", "100");
    let out = run(tmp.path(), &["models", "remove", "--slug", "removeme"]);
    assert!(out.status.success());

    let list = run(tmp.path(), &["--json", "models", "list"]);
    let stdout = String::from_utf8(list.stdout).unwrap();
    let v: serde_json::Value = serde_json::from_str(stdout.trim()).unwrap();
    assert!(v.as_array().unwrap().is_empty());
}

#[test]
fn remove_missing_slug_exits_1() {
    let tmp = TempDir::new().unwrap();
    let out = run(tmp.path(), &["models", "remove", "--slug", "ghost"]);
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stderr).contains("no such slug"));
}

#[test]
fn recommend_after_seed_prints_choice() {
    let tmp = TempDir::new().unwrap();
    add_entry(tmp.path(), "tiny", "low", "chat", "100");
    add_entry(tmp.path(), "big", "low", "chat", "9000");

    let out = run(
        tmp.path(),
        &["models", "recommend", "--tier", "low", "--task", "chat"],
    );
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("tiny"), "got: {stdout}");
}

#[test]
fn recommend_empty_catalog_exits_1() {
    let tmp = TempDir::new().unwrap();
    let out = run(
        tmp.path(),
        &["models", "recommend", "--tier", "low", "--task", "chat"],
    );
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stderr).contains("no model fits"));
}

#[test]
fn validate_bad_catalog_exits_1() {
    let tmp = TempDir::new().unwrap();
    // Hand-craft a catalog with mismatched slug-vs-key.
    let state_dir = tmp.path().join("state");
    std::fs::create_dir_all(&state_dir).unwrap();
    let catalog_path = state_dir.join("models.json");
    let body = serde_json::json!({
        "schema_version": 1,
        "entries": {
            "foo": {
                "slug": "bar",
                "family": "llama",
                "display_name": "Disp",
                "tier": "low",
                "task": ["chat"],
                "size_mib": 100,
                "quantization": "Q",
                "artifact": {
                    "url": "https://example.com/x",
                    "sha256": GOOD_SHA,
                    "bytes": 1
                },
                "license": "MIT",
                "homepage": null
            }
        }
    });
    std::fs::write(&catalog_path, serde_json::to_vec_pretty(&body).unwrap()).unwrap();

    let out = run(tmp.path(), &["models", "validate"]);
    assert_eq!(out.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("mismatch"), "got stderr: {stderr}");
}
