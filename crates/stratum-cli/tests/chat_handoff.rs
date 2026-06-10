//! Integration tests for `stratum chat --agents-dir <PATH>`.
//!
//! Each test spawns the built CLI binary against an isolated
//! `--storage-root` tempdir and stages valid / empty agent TOML files in
//! `<state>/agents/` to exercise the multi-role hand-off path the brief
//! documents.

// Integration test binary: every fn here exists only for `cargo test`. The
// helpers below panic on setup failures by design; clippy's `expect_used` /
// `unwrap_used` denials only apply to non-test code.
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

fn write_agent(dir: &Path, name: &str, role: &str) -> PathBuf {
    fs::create_dir_all(dir).expect("create agents dir");
    let path = dir.join(format!("{name}.toml"));
    let body = format!(
        r#"
schema_version = 1
description = "test agent"
roles = ["{role}"]
model = "echo"
tools = []
sandbox = "passthrough"
"#
    );
    fs::write(&path, body).expect("write agent toml");
    path
}

fn run_chat(root: &Path, args: &[&str]) -> std::process::Output {
    let mut cmd = bin();
    cmd.args(["--storage-root", root.to_str().expect("utf-8 root")]);
    cmd.arg("chat");
    cmd.args(args);
    cmd.output().expect("spawn stratum chat")
}

#[test]
fn chat_with_agents_dir_runs_prompt_and_prints_echo() {
    let tmp = TempDir::new().expect("tempdir");
    let agents = agents_dir(tmp.path());
    // Stage two valid agents. We use `default` for the first so the
    // AgentHandoff fallback role resolves; `polisher` covers the
    // remaining brief requirement (two valid TOMLs).
    write_agent(&agents, "cavemanish", "default");
    write_agent(&agents, "polisher", "polisher");

    let agents_str = agents.to_str().expect("utf-8 path");
    let out = run_chat(tmp.path(), &["--agents-dir", agents_str, "--prompt", "hi"]);
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8(out.stdout).expect("utf-8 stdout");
    assert!(stdout.contains("hi"), "stdout missing 'hi': {stdout:?}");
}

#[test]
fn chat_with_empty_agents_dir_exits_strat_e1001() {
    let tmp = TempDir::new().expect("tempdir");
    let agents = agents_dir(tmp.path());
    fs::create_dir_all(&agents).expect("create empty agents dir");

    let agents_str = agents.to_str().expect("utf-8 path");
    let out = run_chat(tmp.path(), &["--agents-dir", agents_str, "--prompt", "hi"]);
    assert!(
        !out.status.success(),
        "expected failure, got success; stdout: {}",
        String::from_utf8_lossy(&out.stdout)
    );
    let code = out.status.code().expect("exit code present");
    assert_eq!(code, 1, "expected exit 1, got {code}");
    let stderr = String::from_utf8(out.stderr).expect("utf-8 stderr");
    assert!(
        stderr.contains("STRAT-E1001"),
        "stderr missing STRAT-E1001: {stderr:?}"
    );
    assert!(
        stderr.contains("no agents"),
        "stderr missing 'no agents' hint: {stderr:?}"
    );
}

#[test]
fn chat_with_agents_dir_handoff_marker_triggers_followup() {
    let tmp = TempDir::new().expect("tempdir");
    let agents = agents_dir(tmp.path());
    // Default + Coder so the marker can route to Coder. We deliberately
    // target Coder (not Polisher) because the default IntentRouter
    // matches the substring "polish" — present inside "polisher" — and
    // would route the initial turn to Polisher, causing a self-hand-off
    // rejection on the marker. Coder has no such trigger pattern, so
    // the initial turn lands on Default and the marker drives a real
    // Default → Coder transition.
    write_agent(&agents, "cavemanish", "default");
    write_agent(&agents, "coder", "coder");

    let agents_str = agents.to_str().expect("utf-8 path");
    // EchoProvider returns the prompt verbatim (one Text block per
    // whitespace-separated token). Using a single token ensures the
    // marker is the entire first/last text block, which is what
    // `AgentHandoff` looks at.
    let out = run_chat(
        tmp.path(),
        &[
            "--agents-dir",
            agents_str,
            "--prompt",
            "<handoff:coder>followup",
        ],
    );
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8(out.stdout).expect("utf-8 stdout");
    // The follow-up turn runs against the Coder with the marker stripped
    // from the prompt — the EchoProvider then emits "followup" as its
    // final text block, which is what `last_assistant_text` prints.
    assert!(
        stdout.contains("followup"),
        "stdout missing follow-up text 'followup': {stdout:?}"
    );
}

#[test]
fn chat_without_agents_dir_default_behavior_unchanged() {
    // Regression check: the existing single-loop EchoProvider path
    // must continue to work when `--agents-dir` is omitted.
    let tmp = TempDir::new().expect("tempdir");
    let out = run_chat(tmp.path(), &["--prompt", "regression"]);
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8(out.stdout).expect("utf-8 stdout");
    // Default Echo prefix is "echo: " on the no-`--model` path.
    assert!(
        stdout.contains("regression"),
        "stdout missing prompt echo: {stdout:?}"
    );
}
