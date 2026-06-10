//! End-to-end tests for `stratum chat --parallel <roles>`.
//!
//! Each test spawns the built CLI binary against an isolated
//! `--storage-root` `TempDir`, stages two `EchoProvider`-backed agent
//! TOML files (one per parallel role) under `<state>/agents/`, and
//! exercises the documented `--parallel` surface:
//!
//! * Happy path: `--parallel cavemanish,coder --prompt "hi"` fans the
//!   prompt out to both roles, exits 0, and stdout carries each role's
//!   section header plus the echoed prompt.
//! * Unknown role: `--parallel <bad-role>` exits 1 with `STRAT-E1001
//!   unknown role: <name>`.
//! * Missing `--agents-dir`: clap's `requires` declaration on the flag
//!   surfaces as exit code 64 — `app::run_with` maps every clap parse
//!   error to sysexits `EX_USAGE`.
//! * `--json` mode: the dispatcher emits a pretty-printed JSON object
//!   with a `per_role` map keyed by `snake_case` role name.

// Integration test binary: every fn here exists only for `cargo test`. The
// helpers below panic on setup failures by design; clippy's `expect_used` /
// `unwrap_used` / `panic` denials only apply to non-test code.
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    reason = "integration test helpers may panic on setup failures"
)]

use std::path::{Path, PathBuf};
use std::process::Command;

use tempfile::TempDir;

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_stratum"))
}

fn agents_dir(root: &Path) -> PathBuf {
    let dir = root.join("state").join("agents");
    std::fs::create_dir_all(&dir).expect("mkdir state/agents");
    dir
}

/// Minimal `AgentDef` body matching the loader's serde expectation.
fn body_for(role: &str) -> String {
    format!(
        r#"
schema_version = 1
name = "{role}-agent"
description = "smoke agent for {role}"
roles = ["{role}"]
model = "echo"
tools = []
sandbox = "passthrough"
"#
    )
}

fn write_agent(dir: &Path, name: &str, role: &str) {
    let path = dir.join(format!("{name}.toml"));
    std::fs::write(&path, body_for(role)).expect("write agent toml");
}

fn run_chat(root: &Path, args: &[&str]) -> std::process::Output {
    let mut cmd = bin();
    cmd.args(["--storage-root", root.to_str().expect("utf-8 root")]);
    cmd.arg("chat");
    cmd.args(args);
    cmd.output().expect("spawn stratum chat")
}

fn run_chat_with_global(root: &Path, globals: &[&str], args: &[&str]) -> std::process::Output {
    let mut cmd = bin();
    cmd.args(["--storage-root", root.to_str().expect("utf-8 root")]);
    cmd.args(globals);
    cmd.arg("chat");
    cmd.args(args);
    cmd.output().expect("spawn stratum chat")
}

// -- happy path --------------------------------------------------------------

#[test]
fn chat_parallel_runs_each_role_and_prints_section_headers() {
    let tmp = TempDir::new().unwrap();
    let dir = agents_dir(tmp.path());
    write_agent(&dir, "cavemanish", "cavemanish");
    write_agent(&dir, "coder", "coder");

    let out = run_chat(
        tmp.path(),
        &[
            "--agents-dir",
            dir.to_str().expect("utf-8 agents dir"),
            "--parallel",
            "cavemanish,coder",
            "--prompt",
            "hi",
        ],
    );
    assert!(
        out.status.success(),
        "expected success; stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("cavemanish") && stdout.contains("coder"),
        "expected both role names in stdout; got {stdout:?}"
    );
    assert!(
        stdout.contains("hi"),
        "expected echoed prompt 'hi' in stdout; got {stdout:?}"
    );
}

// -- unknown role -----------------------------------------------------------

#[test]
fn chat_parallel_unknown_role_exits_one_with_strat_e1001() {
    let tmp = TempDir::new().unwrap();
    let dir = agents_dir(tmp.path());
    write_agent(&dir, "cavemanish", "cavemanish");
    write_agent(&dir, "coder", "coder");

    let out = run_chat(
        tmp.path(),
        &[
            "--agents-dir",
            dir.to_str().expect("utf-8 agents dir"),
            "--parallel",
            "not-a-real-role",
            "--prompt",
            "hi",
        ],
    );
    assert!(!out.status.success(), "expected non-zero exit");
    assert_eq!(out.status.code(), Some(1), "expected exit code 1");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("STRAT-E1001"),
        "expected STRAT-E1001 in stderr; got {stderr:?}"
    );
    assert!(
        stderr.contains("unknown role"),
        "expected 'unknown role' diagnostic; got {stderr:?}"
    );
}

// -- requires --agents-dir --------------------------------------------------

#[test]
fn chat_parallel_without_agents_dir_is_a_clap_error() {
    let tmp = TempDir::new().unwrap();
    let out = run_chat(
        tmp.path(),
        &["--parallel", "cavemanish,coder", "--prompt", "hi"],
    );
    assert!(!out.status.success(), "expected non-zero exit");
    // `app::run_with` maps every clap parse error to sysexits `EX_USAGE`
    // (64), preserving the convention used by the rest of the binary.
    let code = out.status.code().expect("process exited normally");
    assert_eq!(code, 64, "expected EX_USAGE (64) for missing --agents-dir");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("--agents-dir") || stderr.contains("agents-dir"),
        "expected --agents-dir mention in stderr; got {stderr:?}"
    );
}

// -- JSON output -----------------------------------------------------------

#[test]
fn chat_parallel_json_emits_per_role_object() {
    let tmp = TempDir::new().unwrap();
    let dir = agents_dir(tmp.path());
    write_agent(&dir, "cavemanish", "cavemanish");
    write_agent(&dir, "coder", "coder");

    let out = run_chat_with_global(
        tmp.path(),
        &["--json"],
        &[
            "--agents-dir",
            dir.to_str().expect("utf-8 agents dir"),
            "--parallel",
            "cavemanish,coder",
            "--prompt",
            "hi",
        ],
    );
    assert!(
        out.status.success(),
        "expected success; stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    let parsed: serde_json::Value = serde_json::from_str(&stdout).expect("stdout is valid JSON");
    let per_role = parsed
        .get("per_role")
        .and_then(|v| v.as_object())
        .expect("per_role object present");
    assert!(
        per_role.contains_key("cavemanish"),
        "expected cavemanish key; got {:?}",
        per_role.keys().collect::<Vec<_>>()
    );
    assert!(
        per_role.contains_key("coder"),
        "expected coder key; got {:?}",
        per_role.keys().collect::<Vec<_>>()
    );
}
