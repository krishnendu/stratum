//! Integration tests for `stratum agents list|show`.
//!
//! These spawn the built binary against an isolated `--storage-root` and
//! stage agent TOML fixtures under `<state>/agents/` that match the on-disk
//! shape parsed by `AgentLoader::load_file`.

// Integration test binary: every fn here exists only for `cargo test`. Test
// helpers panic on setup failures by design; clippy's `expect_used` /
// `unwrap_used` denials are scoped to non-test code.
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "integration test helpers may panic on setup failures"
)]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use tempfile::TempDir;

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_stratum"))
}

fn agents_dir(root: &Path) -> PathBuf {
    root.join("state").join("agents")
}

fn write_agent(root: &Path, name: &str, body: &str) -> PathBuf {
    let dir = agents_dir(root);
    fs::create_dir_all(&dir).expect("create agents dir");
    let path = dir.join(format!("{name}.toml"));
    fs::write(&path, body).expect("write agent toml");
    path
}

fn run_agents(root: &Path, sub: &[&str]) -> std::process::Output {
    let mut cmd = bin();
    cmd.args(["--storage-root", root.to_str().expect("utf-8 root")]);
    cmd.arg("agents");
    cmd.args(sub);
    cmd.output().expect("spawn stratum agents")
}

const fn cavemanish_body() -> &'static str {
    r#"
schema_version = 1
name = "cavemanish-rewriter"
description = "terse caveman rewriter"
roles = ["cavemanish"]
model = "echo"
tools = ["fs.read", "fs.write"]
sandbox = "passthrough"
"#
}

#[test]
fn list_empty_dir_prints_no_registered_roles() {
    let tmp = TempDir::new().unwrap();
    let out = run_agents(tmp.path(), &["list"]);
    let stdout = String::from_utf8(out.stdout).unwrap();
    let stderr = String::from_utf8(out.stderr).unwrap();
    assert!(out.status.success(), "stderr={stderr}");
    assert!(
        stdout.contains("registered roles (sorted):"),
        "stdout={stdout}"
    );
    assert!(stdout.contains("skipped: 0"), "stdout={stdout}");
    assert!(stdout.contains("errors: 0"), "stdout={stdout}");
    // No registered role line.
    assert!(!stdout.contains("  - cavemanish"), "stdout={stdout}");
}

#[test]
fn list_single_valid_agent_shows_one_role() {
    let tmp = TempDir::new().unwrap();
    write_agent(tmp.path(), "cavemanish", cavemanish_body());
    let out = run_agents(tmp.path(), &["list"]);
    assert!(out.status.success());
    let stdout = String::from_utf8(out.stdout).unwrap();
    assert!(stdout.contains("registered roles (sorted):"));
    assert!(stdout.contains("  - cavemanish"));
    assert!(stdout.contains("skipped: 0"));
    assert!(stdout.contains("errors: 0"));
}

#[test]
fn list_json_round_trips_through_serde_value() {
    let tmp = TempDir::new().unwrap();
    write_agent(tmp.path(), "cavemanish", cavemanish_body());
    let out = run_agents(tmp.path(), &["list", "--json"]);
    assert!(out.status.success());
    let stdout = String::from_utf8(out.stdout).unwrap();
    let v: serde_json::Value = serde_json::from_str(stdout.trim()).expect("valid json");
    let registered = v["registered"].as_array().expect("registered array");
    assert_eq!(registered.len(), 1);
    assert_eq!(registered[0].as_str(), Some("cavemanish"));
    assert!(v["skipped"].as_array().is_some());
    assert!(v["errors"].as_array().is_some());
}

#[test]
fn show_prose_lists_agent_fields() {
    let tmp = TempDir::new().unwrap();
    write_agent(tmp.path(), "cavemanish", cavemanish_body());
    let out = run_agents(tmp.path(), &["show", "--role", "cavemanish"]);
    assert!(out.status.success());
    let stdout = String::from_utf8(out.stdout).unwrap();
    assert!(stdout.contains("role: cavemanish"));
    assert!(stdout.contains("name: cavemanish-rewriter"));
    assert!(stdout.contains("description: terse caveman rewriter"));
    assert!(stdout.contains("sandbox: passthrough"));
    assert!(stdout.contains("capabilities: fs.read, fs.write"));
}

#[test]
fn show_missing_role_returns_one_and_strat_e1001() {
    let tmp = TempDir::new().unwrap();
    write_agent(tmp.path(), "cavemanish", cavemanish_body());
    let out = run_agents(tmp.path(), &["show", "--role", "researcher"]);
    assert!(!out.status.success());
    assert_eq!(out.status.code(), Some(1));
    let stderr = String::from_utf8(out.stderr).unwrap();
    assert!(stderr.contains("STRAT-E1001"), "stderr={stderr}");
}

#[test]
fn show_json_emits_valid_json() {
    let tmp = TempDir::new().unwrap();
    write_agent(tmp.path(), "cavemanish", cavemanish_body());
    let out = run_agents(tmp.path(), &["show", "--role", "cavemanish", "--json"]);
    assert!(out.status.success());
    let stdout = String::from_utf8(out.stdout).unwrap();
    let v: serde_json::Value = serde_json::from_str(stdout.trim()).expect("valid json");
    assert_eq!(v["name"].as_str(), Some("cavemanish-rewriter"));
    assert_eq!(v["sandbox"].as_str(), Some("passthrough"));
    let roles = v["roles"].as_array().expect("roles array");
    assert_eq!(roles[0].as_str(), Some("cavemanish"));
}

#[test]
fn list_with_malformed_toml_reports_error_but_exits_zero() {
    let tmp = TempDir::new().unwrap();
    let dir = agents_dir(tmp.path());
    fs::create_dir_all(&dir).unwrap();
    fs::write(dir.join("broken.toml"), b"not = [ valid").unwrap();
    let out = run_agents(tmp.path(), &["list"]);
    assert!(out.status.success());
    let stdout = String::from_utf8(out.stdout).unwrap();
    assert!(stdout.contains("errors: 1"), "stdout={stdout}");
    assert!(stdout.contains("broken.toml"), "stdout={stdout}");
}

#[test]
fn list_with_unknown_role_reports_skip_but_exits_zero() {
    let tmp = TempDir::new().unwrap();
    let body = r#"
schema_version = 1
name = "weird"
description = "unknown role"
roles = ["not-a-real-role"]
model = "echo"
tools = []
sandbox = "passthrough"
"#;
    write_agent(tmp.path(), "weird", body);
    let out = run_agents(tmp.path(), &["list"]);
    assert!(out.status.success());
    let stdout = String::from_utf8(out.stdout).unwrap();
    assert!(stdout.contains("skipped: 1"), "stdout={stdout}");
    assert!(stdout.contains("not-a-real-role"), "stdout={stdout}");
}
