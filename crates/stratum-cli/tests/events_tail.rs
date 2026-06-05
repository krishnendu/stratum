//! Integration tests for `stratum events tail`.
//!
//! These spawn the built binary against an isolated `--storage-root` and
//! exercise the various filters by hand-rolling a fixture `events.jsonl` that
//! matches the on-disk format written by `stratum_runtime::JsonlEventSink`.

// Integration test binary: every fn here exists only for `cargo test`. Test
// helpers panic on setup failures by design; clippy's `expect_used` /
// `unwrap_used` / `needless_collect` / `single_char_add_str` denials are
// scoped to non-test code.
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::needless_collect,
    clippy::single_char_add_str,
    reason = "integration test helpers may panic on setup failures"
)]

use std::fs;
use std::path::Path;
use std::process::Command;
use std::time::{Duration, SystemTime};

use stratum_runtime::{Event, EventRecord};
use tempfile::TempDir;

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_stratum"))
}

fn state_dir(root: &Path) -> std::path::PathBuf {
    let dir = root.join("state");
    fs::create_dir_all(&dir).expect("create state dir");
    dir
}

fn events_path(root: &Path) -> std::path::PathBuf {
    state_dir(root).join("events.jsonl")
}

fn fixed_at() -> SystemTime {
    SystemTime::UNIX_EPOCH + Duration::from_secs(1_700_000_000)
}

fn record(id: u64, event: Event) -> EventRecord {
    EventRecord {
        id,
        at: fixed_at(),
        turn_id: None,
        event,
    }
}

fn write_records(root: &Path, records: &[EventRecord]) {
    let path = events_path(root);
    let mut body = String::new();
    for r in records {
        body.push_str(&serde_json::to_string(r).expect("serialize"));
        body.push('\n');
    }
    fs::write(path, body).expect("write events.jsonl");
}

fn three_basic_records() -> Vec<EventRecord> {
    vec![
        record(
            1,
            Event::ToolCall {
                tool_id: "fs.read".into(),
                ok: true,
                duration_ms: 12,
            },
        ),
        record(
            2,
            Event::PermissionAsked {
                request: "net.connect example.com:443".into(),
                decision: "allow_once".into(),
            },
        ),
        record(
            3,
            Event::ToolCall {
                tool_id: "fs.write".into(),
                ok: false,
                duration_ms: 50,
            },
        ),
    ]
}

fn run_tail(root: &Path, args: &[&str]) -> std::process::Output {
    let mut cmd = bin();
    cmd.args(["--storage-root", root.to_str().expect("utf-8 root")]);
    cmd.args(["events", "tail"]);
    cmd.args(args);
    cmd.output().expect("spawn stratum events tail")
}

#[test]
fn tail_missing_file_exits_zero_empty() {
    let tmp = TempDir::new().expect("tempdir");
    // do NOT create events.jsonl
    let _ = state_dir(tmp.path());
    let out = run_tail(tmp.path(), &[]);
    assert!(out.status.success(), "stderr: {:?}", out.stderr);
    assert!(out.stdout.is_empty(), "stdout: {:?}", out.stdout);
}

#[test]
fn tail_prints_all_records_prose() {
    let tmp = TempDir::new().expect("tempdir");
    write_records(tmp.path(), &three_basic_records());
    let out = run_tail(tmp.path(), &[]);
    assert!(out.status.success());
    let stdout = String::from_utf8(out.stdout).expect("utf-8");
    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(lines.len(), 3, "got: {stdout}");
    assert!(lines[0].contains("tool_call"));
    assert!(lines[0].contains("fs.read"));
    assert!(lines[0].contains("ok=true"));
    assert!(lines[1].contains("permission_asked"));
    assert!(lines[2].contains("tool_call"));
    assert!(lines[2].contains("fs.write"));
}

#[test]
fn tail_since_id_skips_lower_or_equal() {
    let tmp = TempDir::new().expect("tempdir");
    write_records(tmp.path(), &three_basic_records());
    let out = run_tail(tmp.path(), &["--since-id", "2"]);
    assert!(out.status.success());
    let stdout = String::from_utf8(out.stdout).expect("utf-8");
    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(lines.len(), 1, "got: {stdout}");
    assert!(lines[0].starts_with("[3]"));
}

#[test]
fn tail_limit_caps_output() {
    let tmp = TempDir::new().expect("tempdir");
    write_records(tmp.path(), &three_basic_records());
    let out = run_tail(tmp.path(), &["--limit", "2"]);
    assert!(out.status.success());
    let stdout = String::from_utf8(out.stdout).expect("utf-8");
    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(lines.len(), 2, "got: {stdout}");
}

#[test]
fn tail_kind_filter_only_tool_calls() {
    let tmp = TempDir::new().expect("tempdir");
    write_records(tmp.path(), &three_basic_records());
    let out = run_tail(tmp.path(), &["--kind", "tool_call"]);
    assert!(out.status.success());
    let stdout = String::from_utf8(out.stdout).expect("utf-8");
    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(lines.len(), 2, "got: {stdout}");
    for line in &lines {
        assert!(line.contains("tool_call"));
        assert!(!line.contains("permission_asked"));
    }
}

#[test]
fn tail_json_emits_one_record_per_line() {
    let tmp = TempDir::new().expect("tempdir");
    write_records(tmp.path(), &three_basic_records());
    let out = run_tail(tmp.path(), &["--json"]);
    assert!(out.status.success());
    let stdout = String::from_utf8(out.stdout).expect("utf-8");
    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(lines.len(), 3);
    for line in &lines {
        let v: serde_json::Value = serde_json::from_str(line).expect("valid json line");
        assert!(v["id"].is_number());
        assert!(v["event"]["kind"].is_string());
    }
}

#[test]
fn tail_skips_malformed_lines_and_continues() {
    let tmp = TempDir::new().expect("tempdir");
    let path = events_path(tmp.path());
    let good_1 = record(
        1,
        Event::ToolCall {
            tool_id: "fs.read".into(),
            ok: true,
            duration_ms: 1,
        },
    );
    let good_2 = record(
        2,
        Event::ToolCall {
            tool_id: "fs.write".into(),
            ok: true,
            duration_ms: 2,
        },
    );
    let mut body = String::new();
    body.push_str(&serde_json::to_string(&good_1).expect("serialize"));
    body.push('\n');
    body.push_str("{this is not valid json}\n");
    body.push_str("\n"); // empty line: should be skipped silently
    body.push_str(&serde_json::to_string(&good_2).expect("serialize"));
    body.push('\n');
    fs::write(&path, body).expect("write events.jsonl");

    let out = run_tail(tmp.path(), &[]);
    assert!(out.status.success(), "stderr: {:?}", out.stderr);
    let stdout = String::from_utf8(out.stdout).expect("utf-8");
    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(lines.len(), 2, "got: {stdout}");
    assert!(lines[0].contains("fs.read"));
    assert!(lines[1].contains("fs.write"));
}

#[test]
fn tail_follow_exits_after_max_seconds_env() {
    let tmp = TempDir::new().expect("tempdir");
    write_records(tmp.path(), &three_basic_records());
    let start = std::time::Instant::now();
    let mut cmd = bin();
    cmd.env("STRATUM_EVENTS_TAIL_MAX_S", "1");
    cmd.args(["--storage-root", tmp.path().to_str().expect("utf-8 root")]);
    cmd.args(["events", "tail", "--follow"]);
    let out = cmd.output().expect("spawn");
    let elapsed = start.elapsed();
    assert!(out.status.success(), "stderr: {:?}", out.stderr);
    // The follow loop should exit within a small bound after the configured
    // 1-second window. Be generous to absorb CI jitter.
    assert!(
        elapsed < Duration::from_secs(10),
        "follow loop ran too long: {elapsed:?}"
    );
    // It should have printed the seeded records before entering the poll loop.
    let stdout = String::from_utf8(out.stdout).expect("utf-8");
    assert_eq!(stdout.lines().count(), 3);
}

#[test]
fn tail_follow_waits_for_missing_file_then_exits() {
    let tmp = TempDir::new().expect("tempdir");
    // do NOT seed any events.jsonl
    let _ = state_dir(tmp.path());
    let mut cmd = bin();
    cmd.env("STRATUM_EVENTS_TAIL_MAX_S", "1");
    cmd.args(["--storage-root", tmp.path().to_str().expect("utf-8 root")]);
    cmd.args(["events", "tail", "--follow"]);
    let out = cmd.output().expect("spawn");
    assert!(out.status.success(), "stderr: {:?}", out.stderr);
    assert!(out.stdout.is_empty());
}

#[test]
fn tail_unknown_kind_returns_clap_error() {
    let tmp = TempDir::new().expect("tempdir");
    let _ = state_dir(tmp.path());
    let out = run_tail(tmp.path(), &["--kind", "unknown_kind"]);
    assert!(!out.status.success());
    assert_eq!(out.status.code(), Some(64));
}
