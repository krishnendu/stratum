//! Integration tests for `stratum chat --events-jsonl <path>`.
//!
//! These spawn the built CLI binary against a fresh `--storage-root` tempdir
//! and assert that:
//!
//! * The flag causes a JSONL file to be created at `<path>`.
//! * Each line is a valid `EventRecord` and at least one `AgentHandoff` is
//!   emitted by the [`AgentLoop`] at the start of every turn.
//! * A missing parent directory surfaces a STRAT-E1001 + exit 1.
//! * Without the flag, no JSONL file is created.
//! * Re-running with the same path appends rather than truncates.

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

fn run_chat(root: &Path, args: &[&str]) -> std::process::Output {
    let mut cmd = bin();
    cmd.args(["--storage-root", root.to_str().expect("utf-8 root")]);
    cmd.arg("chat");
    cmd.args(args);
    cmd.output().expect("spawn stratum chat")
}

fn read_lines(path: &Path) -> Vec<String> {
    let body = fs::read_to_string(path).expect("read events JSONL");
    body.lines().map(ToString::to_string).collect()
}

#[test]
fn events_jsonl_flag_writes_records_for_single_prompt() {
    let tmp = TempDir::new().unwrap();
    let events_path: PathBuf = tmp.path().join("my-events.jsonl");
    let output = run_chat(
        tmp.path(),
        &[
            "--prompt",
            "hello",
            "--events-jsonl",
            events_path.to_str().unwrap(),
        ],
    );
    assert!(
        output.status.success(),
        "exit={:?} stderr={}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(events_path.exists(), "expected JSONL file to be created");

    let lines = read_lines(&events_path);
    assert!(!lines.is_empty(), "expected at least one event record");

    // Every line must parse as JSON.
    let mut saw_agent_handoff = false;
    for line in &lines {
        let value: serde_json::Value =
            serde_json::from_str(line).expect("each line must be valid JSON");
        // Each EventRecord carries an `id`, `at`, and `event` field.
        assert!(value.get("id").is_some(), "record missing id: {line}");
        assert!(value.get("at").is_some(), "record missing at: {line}");
        let event = value.get("event").expect("record missing event");
        // The event is serde-tagged; `AgentHandoff` lands as
        // `{ "kind": "agent_handoff", ... }`.
        if event.get("kind").and_then(|v| v.as_str()) == Some("agent_handoff") {
            saw_agent_handoff = true;
        }
    }
    assert!(
        saw_agent_handoff,
        "expected at least one AgentHandoff event; got lines: {lines:?}"
    );
}

#[test]
fn events_jsonl_flag_missing_parent_dir_errors_with_e1001() {
    let tmp = TempDir::new().unwrap();
    let events_path: PathBuf = tmp.path().join("nested/missing/file.jsonl");
    assert!(
        !events_path.parent().unwrap().exists(),
        "test fixture: parent must not exist"
    );
    let output = run_chat(
        tmp.path(),
        &[
            "--prompt",
            "hi",
            "--events-jsonl",
            events_path.to_str().unwrap(),
        ],
    );
    assert_eq!(
        output.status.code(),
        Some(1),
        "expected exit 1; stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("STRAT-E1001"),
        "expected STRAT-E1001 in stderr; got: {stderr}"
    );
    assert!(!events_path.exists(), "no file should be created on error");
}

#[test]
fn no_flag_means_no_jsonl_file() {
    let tmp = TempDir::new().unwrap();
    let candidate: PathBuf = tmp.path().join("should-not-exist.jsonl");
    let output = run_chat(tmp.path(), &["--prompt", "hi"]);
    assert!(
        output.status.success(),
        "exit={:?} stderr={}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        !candidate.exists(),
        "no JSONL file should be created without the flag"
    );
}

#[test]
fn events_jsonl_flag_appends_across_runs() {
    let tmp = TempDir::new().unwrap();
    let events_path: PathBuf = tmp.path().join("events.jsonl");

    // First run.
    let out1 = run_chat(
        tmp.path(),
        &[
            "--prompt",
            "one",
            "--events-jsonl",
            events_path.to_str().unwrap(),
        ],
    );
    assert!(out1.status.success(), "first run failed");
    let lines1 = read_lines(&events_path);
    assert!(!lines1.is_empty(), "first run produced no records");
    let first_line = lines1[0].clone();

    // Second run against the same file.
    let out2 = run_chat(
        tmp.path(),
        &[
            "--prompt",
            "two",
            "--events-jsonl",
            events_path.to_str().unwrap(),
        ],
    );
    assert!(out2.status.success(), "second run failed");
    let lines2 = read_lines(&events_path);
    assert!(
        lines2.len() > lines1.len(),
        "expected line count to grow: before={} after={}",
        lines1.len(),
        lines2.len()
    );
    assert_eq!(
        lines2[0], first_line,
        "first line must still match (append-only)"
    );
}

#[cfg(not(feature = "provider-llama-cpp"))]
#[test]
fn events_jsonl_flag_combined_with_model_parses() {
    // Without the `provider-llama-cpp` feature, `--model` errors with
    // STRAT-E1001 + a feature-flag hint and exits 1 — but the important
    // assertion here is that clap accepts the combination of `--model` and
    // `--events-jsonl` (no usage / exit 64). We assert exit 1 (the
    // feature-disabled diag), not exit 64 (clap rejection).
    let tmp = TempDir::new().unwrap();
    let events_path: PathBuf = tmp.path().join("events.jsonl");
    let output = run_chat(
        tmp.path(),
        &[
            "--model",
            "qwen",
            "--prompt",
            "hi",
            "--events-jsonl",
            events_path.to_str().unwrap(),
        ],
    );
    // Exit code must NOT be 64 (clap usage error). It will be 1 because the
    // feature is off.
    assert_ne!(
        output.status.code(),
        Some(64),
        "clap rejected the flag combo; stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("STRAT-E1001"),
        "expected STRAT-E1001 in stderr; got: {stderr}"
    );
}

#[cfg(feature = "provider-llama-cpp")]
#[test]
fn events_jsonl_flag_combined_with_model_parses() {
    // With the `provider-llama-cpp` feature, an unknown slug errors with
    // STRAT-E1001. The important assertion is that clap accepts the
    // combination of `--model` and `--events-jsonl` (no exit 64 usage error).
    let tmp = TempDir::new().unwrap();
    let events_path: PathBuf = tmp.path().join("events.jsonl");
    let output = run_chat(
        tmp.path(),
        &[
            "--model",
            "unknown-slug",
            "--prompt",
            "hi",
            "--events-jsonl",
            events_path.to_str().unwrap(),
        ],
    );
    assert_ne!(
        output.status.code(),
        Some(64),
        "clap rejected the flag combo; stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn help_documents_events_jsonl_flag() {
    // The CLI's top-level `run_with` routes every clap error (including
    // `--help`, which clap reports as `ErrorKind::DisplayHelp`) through
    // `writeln!(err, "{e}")` and exits 64. So the help body lands on stderr
    // rather than stdout. We assert against the combined stream — the
    // important property is that the flag appears in the help output and
    // is described as a JSONL sink.
    let output = bin()
        .args(["chat", "--help"])
        .output()
        .expect("spawn stratum chat --help");
    let mut combined = String::new();
    combined.push_str(&String::from_utf8_lossy(&output.stdout));
    combined.push_str(&String::from_utf8_lossy(&output.stderr));
    assert!(
        combined.contains("--events-jsonl"),
        "help text missing --events-jsonl: {combined}"
    );
    assert!(
        combined.contains("JSONL"),
        "help text missing JSONL description: {combined}"
    );
}
