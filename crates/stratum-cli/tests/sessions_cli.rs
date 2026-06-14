//! Integration tests for `stratum sessions list|show|delete`.
//!
//! These spawn the built binary against an isolated `--storage-root` and
//! stage transcript fixtures on disk by writing JSON files that match the
//! exact `Transcript` serialization the runtime emits.

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
use std::time::{Duration, SystemTime};

use stratum_runtime::{
    SessionId, Transcript, TranscriptBlock, TranscriptBlockKind, TranscriptTurn,
    TRANSCRIPT_SCHEMA_VERSION,
};
use tempfile::TempDir;

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_stratum"))
}

fn transcripts_dir(root: &Path) -> PathBuf {
    root.join("state").join("sessions")
}

fn fixed_at() -> SystemTime {
    SystemTime::UNIX_EPOCH + Duration::from_secs(1_700_000_000)
}

fn sample_transcript(id_hex: &str) -> Transcript {
    let id = SessionId::from_str(id_hex).expect("valid session id");
    Transcript {
        schema_version: TRANSCRIPT_SCHEMA_VERSION,
        session_id: id,
        created_at: fixed_at(),
        turns: vec![
            TranscriptTurn::User {
                at: fixed_at(),
                text: "hello".to_owned(),
            },
            TranscriptTurn::Assistant {
                at: fixed_at() + Duration::from_secs(1),
                blocks: vec![
                    TranscriptBlock {
                        kind: TranscriptBlockKind::Text,
                        text: "hi".to_owned(),
                    },
                    TranscriptBlock {
                        kind: TranscriptBlockKind::Code {
                            language: "rust".to_owned(),
                        },
                        text: "fn main() {}".to_owned(),
                    },
                ],
            },
        ],
    }
}

fn stage_transcript(root: &Path, t: &Transcript) -> PathBuf {
    let dir = transcripts_dir(root);
    fs::create_dir_all(&dir).expect("create transcripts dir");
    let path = dir.join(format!("{}.json", t.session_id.as_str()));
    let f = fs::File::create(&path).expect("create transcript file");
    serde_json::to_writer(f, t).expect("write transcript json");
    path
}

fn run_sessions(root: &Path, global: &[&str], sub: &[&str]) -> std::process::Output {
    let mut cmd = bin();
    cmd.args(["--storage-root", root.to_str().expect("utf-8 root")]);
    cmd.args(global);
    cmd.arg("sessions");
    cmd.args(sub);
    cmd.output().expect("spawn stratum sessions")
}

#[test]
fn list_empty_dir_exits_zero_no_output() {
    let tmp = TempDir::new().expect("tempdir");
    let out = run_sessions(tmp.path(), &[], &["list"]);
    assert!(out.status.success(), "stderr: {:?}", out.stderr);
    assert!(out.stdout.is_empty(), "stdout: {:?}", out.stdout);
}

#[test]
fn list_json_empty_dir_emits_empty_array() {
    let tmp = TempDir::new().expect("tempdir");
    let out = run_sessions(tmp.path(), &["--json"], &["list"]);
    assert!(out.status.success(), "stderr: {:?}", out.stderr);
    let stdout = String::from_utf8(out.stdout).expect("utf-8");
    let v: serde_json::Value = serde_json::from_str(stdout.trim()).expect("valid json");
    let arr = v.as_array().expect("array");
    assert!(arr.is_empty(), "expected empty array, got {arr:?}");
}

#[test]
fn list_lists_staged_id() {
    let tmp = TempDir::new().expect("tempdir");
    let t = sample_transcript("deadbeefcafef00d");
    stage_transcript(tmp.path(), &t);
    let out = run_sessions(tmp.path(), &[], &["list"]);
    assert!(out.status.success(), "stderr: {:?}", out.stderr);
    let stdout = String::from_utf8(out.stdout).expect("utf-8");
    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(lines.len(), 1, "got: {stdout}");
    assert_eq!(lines[0], "deadbeefcafef00d");
}

#[test]
fn list_json_returns_valid_json_array() {
    let tmp = TempDir::new().expect("tempdir");
    let t = sample_transcript("deadbeefcafef00d");
    stage_transcript(tmp.path(), &t);
    let out = run_sessions(tmp.path(), &["--json"], &["list"]);
    assert!(out.status.success(), "stderr: {:?}", out.stderr);
    let stdout = String::from_utf8(out.stdout).expect("utf-8");
    let v: serde_json::Value = serde_json::from_str(stdout.trim()).expect("valid json");
    let arr = v.as_array().expect("array");
    assert_eq!(arr.len(), 1);
    assert_eq!(arr[0], "deadbeefcafef00d");
}

#[test]
fn list_returns_sorted_ids() {
    let tmp = TempDir::new().expect("tempdir");
    // Stage out of order: c, a, b.
    let c = sample_transcript("cccccccccccccccc");
    let a = sample_transcript("aaaaaaaaaaaaaaaa");
    let b = sample_transcript("bbbbbbbbbbbbbbbb");
    stage_transcript(tmp.path(), &c);
    stage_transcript(tmp.path(), &a);
    stage_transcript(tmp.path(), &b);
    let out = run_sessions(tmp.path(), &[], &["list"]);
    assert!(out.status.success(), "stderr: {:?}", out.stderr);
    let stdout = String::from_utf8(out.stdout).expect("utf-8");
    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(lines.len(), 3);
    assert_eq!(lines[0], "aaaaaaaaaaaaaaaa");
    assert_eq!(lines[1], "bbbbbbbbbbbbbbbb");
    assert_eq!(lines[2], "cccccccccccccccc");
}

#[test]
fn show_prose_emits_session_header() {
    let tmp = TempDir::new().expect("tempdir");
    let t = sample_transcript("deadbeefcafef00d");
    stage_transcript(tmp.path(), &t);
    let out = run_sessions(tmp.path(), &[], &["show", "--id", "deadbeefcafef00d"]);
    assert!(out.status.success(), "stderr: {:?}", out.stderr);
    let stdout = String::from_utf8(out.stdout).expect("utf-8");
    assert!(
        stdout.contains("session: deadbeefcafef00d"),
        "got: {stdout}"
    );
    assert!(stdout.contains("turns: 2"), "got: {stdout}");
    assert!(stdout.contains("user: hello"), "got: {stdout}");
    assert!(stdout.contains("assistant:"), "got: {stdout}");
    assert!(stdout.contains("----"), "got: {stdout}");
}

#[test]
fn show_json_round_trips_through_serde_value() {
    let tmp = TempDir::new().expect("tempdir");
    let t = sample_transcript("deadbeefcafef00d");
    stage_transcript(tmp.path(), &t);
    let out = run_sessions(
        tmp.path(),
        &["--json"],
        &["show", "--id", "deadbeefcafef00d"],
    );
    assert!(out.status.success(), "stderr: {:?}", out.stderr);
    let stdout = String::from_utf8(out.stdout).expect("utf-8");
    let v: serde_json::Value = serde_json::from_str(stdout.trim()).expect("valid json");
    assert_eq!(v["session_id"], "deadbeefcafef00d");
    assert_eq!(v["schema_version"], TRANSCRIPT_SCHEMA_VERSION);
    let turns = v["turns"].as_array().expect("turns is array");
    assert_eq!(turns.len(), 2);
}

#[test]
fn show_missing_session_exits_one() {
    let tmp = TempDir::new().expect("tempdir");
    let out = run_sessions(tmp.path(), &[], &["show", "--id", "0123456789abcdef"]);
    assert!(!out.status.success());
    assert_eq!(out.status.code(), Some(1));
    let stderr = String::from_utf8(out.stderr).expect("utf-8");
    assert!(stderr.contains("STRAT-E1001"), "stderr: {stderr}");
}

#[test]
fn show_bad_id_exits_two() {
    let tmp = TempDir::new().expect("tempdir");
    let out = run_sessions(tmp.path(), &[], &["show", "--id", "ZZZZ"]);
    assert!(!out.status.success());
    assert_eq!(out.status.code(), Some(2));
    let stderr = String::from_utf8(out.stderr).expect("utf-8");
    assert!(stderr.contains("STRAT-E1001"), "stderr: {stderr}");
}

#[test]
fn delete_existing_returns_zero_and_file_gone() {
    let tmp = TempDir::new().expect("tempdir");
    let t = sample_transcript("deadbeefcafef00d");
    let path = stage_transcript(tmp.path(), &t);
    assert!(path.exists());
    let out = run_sessions(tmp.path(), &[], &["delete", "--id", "deadbeefcafef00d"]);
    assert!(out.status.success(), "stderr: {:?}", out.stderr);
    assert!(!path.exists(), "transcript file should be deleted");
}

#[test]
fn delete_missing_returns_one() {
    let tmp = TempDir::new().expect("tempdir");
    let out = run_sessions(tmp.path(), &[], &["delete", "--id", "0123456789abcdef"]);
    assert!(!out.status.success());
    assert_eq!(out.status.code(), Some(1));
    let stderr = String::from_utf8(out.stderr).expect("utf-8");
    assert!(stderr.contains("STRAT-E1001"), "stderr: {stderr}");
}

#[test]
fn delete_bad_id_returns_two() {
    let tmp = TempDir::new().expect("tempdir");
    // Uppercase hex is rejected; SessionId requires lowercase.
    let out = run_sessions(tmp.path(), &[], &["delete", "--id", "DEADBEEFCAFEF00D"]);
    assert!(!out.status.success());
    assert_eq!(out.status.code(), Some(2));
    let stderr = String::from_utf8(out.stderr).expect("utf-8");
    assert!(stderr.contains("STRAT-E1001"), "stderr: {stderr}");
}

#[test]
fn delete_wrong_length_id_returns_two() {
    let tmp = TempDir::new().expect("tempdir");
    let out = run_sessions(tmp.path(), &[], &["delete", "--id", "deadbeef"]);
    assert!(!out.status.success());
    assert_eq!(out.status.code(), Some(2));
}
