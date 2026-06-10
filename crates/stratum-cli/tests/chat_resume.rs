//! Integration tests for `stratum chat --resume <session-id>`.
//!
//! Each test spawns the built CLI binary against an isolated
//! `--storage-root` tempdir and stages a `<state>/transcripts/<id>.json`
//! fixture (or deliberately omits / corrupts it) to exercise the load,
//! bad-id, missing-file, and malformed-JSON paths the brief documents.

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
    root.join("state").join("transcripts")
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
                text: "earlier-question".to_owned(),
            },
            TranscriptTurn::Assistant {
                at: fixed_at() + Duration::from_secs(1),
                blocks: vec![TranscriptBlock {
                    kind: TranscriptBlockKind::Text,
                    text: "earlier-answer".to_owned(),
                }],
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

fn run_chat(root: &Path, args: &[&str]) -> std::process::Output {
    let mut cmd = bin();
    cmd.args(["--storage-root", root.to_str().expect("utf-8 root")]);
    cmd.arg("chat");
    cmd.args(args);
    cmd.output().expect("spawn stratum chat")
}

#[test]
fn resume_with_prompt_echoes_resumed_turns_then_new_response() {
    let tmp = TempDir::new().expect("tempdir");
    let id_hex = "deadbeefcafef00d";
    let t = sample_transcript(id_hex);
    stage_transcript(tmp.path(), &t);

    let out = run_chat(tmp.path(), &["--resume", id_hex, "--prompt", "hi"]);
    assert!(out.status.success(), "stderr: {:?}", out.stderr);
    let stdout = String::from_utf8(out.stdout).expect("utf-8 stdout");
    // Resumed content present, in order, before the new "hi" reply.
    let pos_q = stdout.find("earlier-question").expect("user turn echoed");
    let pos_a = stdout
        .find("earlier-answer")
        .expect("assistant turn echoed");
    let pos_new = stdout.find("hi").expect("new response present");
    assert!(pos_q < pos_a, "user turn must precede assistant turn");
    assert!(pos_a < pos_new, "resumed turns must precede new reply");
}

#[test]
fn resume_with_bad_session_id_exits_2_and_emits_strat_e1001() {
    let tmp = TempDir::new().expect("tempdir");
    let out = run_chat(tmp.path(), &["--resume", "bad-id", "--prompt", "hi"]);
    let code = out.status.code().unwrap_or_default();
    assert_eq!(code, 2, "expected exit 2, stderr: {:?}", out.stderr);
    let stderr = String::from_utf8(out.stderr).expect("utf-8 stderr");
    assert!(
        stderr.contains("STRAT-E1001"),
        "stderr missing STRAT-E1001: {stderr}"
    );
}

#[test]
fn resume_with_missing_transcript_exits_1_with_sessions_list_hint() {
    let tmp = TempDir::new().expect("tempdir");
    // Stage nothing — the session id is valid format but no file exists.
    let id_hex = "0123456789abcdef";
    let out = run_chat(tmp.path(), &["--resume", id_hex, "--prompt", "hi"]);
    let code = out.status.code().unwrap_or_default();
    assert_eq!(code, 1, "expected exit 1, stderr: {:?}", out.stderr);
    let stderr = String::from_utf8(out.stderr).expect("utf-8 stderr");
    assert!(
        stderr.contains("STRAT-E1001"),
        "stderr missing STRAT-E1001: {stderr}"
    );
    assert!(
        stderr.contains("stratum sessions list"),
        "stderr missing hint: {stderr}"
    );
}

#[test]
fn resume_with_malformed_json_exits_1_and_emits_strat_e1001() {
    let tmp = TempDir::new().expect("tempdir");
    let id_hex = "abcdefabcdefabcd";
    let dir = transcripts_dir(tmp.path());
    fs::create_dir_all(&dir).expect("create transcripts dir");
    let path = dir.join(format!("{id_hex}.json"));
    fs::write(&path, b"{ not valid json").expect("write malformed");

    let out = run_chat(tmp.path(), &["--resume", id_hex, "--prompt", "hi"]);
    let code = out.status.code().unwrap_or_default();
    assert_eq!(code, 1, "expected exit 1, stderr: {:?}", out.stderr);
    let stderr = String::from_utf8(out.stderr).expect("utf-8 stderr");
    assert!(
        stderr.contains("STRAT-E1001"),
        "stderr missing STRAT-E1001: {stderr}"
    );
}

#[test]
fn prompt_without_resume_succeeds_with_no_resumed_content() {
    // Regression check: omitting --resume must keep the existing
    // EchoProvider --prompt flow intact.
    let tmp = TempDir::new().expect("tempdir");
    let out = run_chat(tmp.path(), &["--prompt", "hello-no-resume"]);
    assert!(out.status.success(), "stderr: {:?}", out.stderr);
    let stdout = String::from_utf8(out.stdout).expect("utf-8 stdout");
    assert!(
        stdout.contains("hello-no-resume"),
        "expected echo of prompt in stdout: {stdout}"
    );
    // No transcript was staged, so resumed-turn markers must be absent.
    assert!(
        !stdout.contains("user:"),
        "stray resumed-user line: {stdout}"
    );
    assert!(
        !stdout.contains("assistant:"),
        "stray resumed-assistant line: {stdout}"
    );
}
