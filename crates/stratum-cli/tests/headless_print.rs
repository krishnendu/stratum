//! Integration tests for `stratum -p "<prompt>"` headless mode (plan/43).
//!
//! These tests exercise the top-level `-p` / `--print` shortcut and its
//! `--output-format` siblings. Each test spawns the built `stratum`
//! binary against an isolated `--storage-root` tempdir so the runs
//! don't interfere with each other or the host's real `~/.config`.
//!
//! Coverage:
//!
//! * Success path on the `EchoProvider` floor (text output exits 0).
//! * Empty prompt aborts with exit 64 + a diagnostic on stderr.
//! * Passing `-p` together with a subcommand exits 64 (the two paths
//!   are mutually exclusive).
//! * `--output-format json` emits a well-formed envelope with the
//!   block array + metrics + outcome fields plan/43 §3.1 documents.
//! * `--output-format stream-json` emits one JSON object per block and
//!   a final `done: true` summary line.

// Integration test binary: every fn here exists only for `cargo test`.
// Test helpers panic on setup failures by design; clippy's
// `expect_used` / `unwrap_used` denials only apply to non-test code.
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "integration test helpers may panic on setup failures"
)]

use std::path::Path;
use std::process::Command;

use tempfile::TempDir;

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_stratum"))
}

/// Spawn `stratum --storage-root <root> <args...>`.
fn run(root: &Path, args: &[&str]) -> std::process::Output {
    let mut cmd = bin();
    cmd.args(["--storage-root", root.to_str().expect("utf-8 root")]);
    cmd.args(args);
    cmd.output().expect("spawn stratum")
}

#[test]
fn print_short_flag_runs_one_turn_and_exits_zero() {
    let tmp = TempDir::new().expect("tempdir");
    let out = run(tmp.path(), &["-p", "hello"]);
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8(out.stdout).expect("utf-8 stdout");
    // The `EchoProvider` (no prefix in the headless path) collapses
    // the prompt back into a single token — but the prompt body must
    // be visible on stdout because that's the assistant text.
    assert!(
        stdout.contains("hello"),
        "expected echo of prompt in stdout, got: {stdout}"
    );
}

#[test]
fn print_long_flag_is_equivalent_to_short_flag() {
    let tmp = TempDir::new().expect("tempdir");
    let out = run(tmp.path(), &["--print", "long-form-prompt"]);
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8(out.stdout).expect("utf-8 stdout");
    assert!(stdout.contains("long-form-prompt"), "stdout: {stdout}");
}

#[test]
fn empty_prompt_exits_64_with_diagnostic() {
    let tmp = TempDir::new().expect("tempdir");
    let out = run(tmp.path(), &["-p", ""]);
    let code = out.status.code().unwrap_or_default();
    assert_eq!(code, 64, "expected exit 64, stderr: {:?}", out.stderr);
    let stderr = String::from_utf8(out.stderr).expect("utf-8 stderr");
    assert!(
        stderr.contains("--print"),
        "stderr missing flag name: {stderr}"
    );
}

#[test]
fn whitespace_only_prompt_exits_64() {
    let tmp = TempDir::new().expect("tempdir");
    let out = run(tmp.path(), &["-p", "   \t\n  "]);
    let code = out.status.code().unwrap_or_default();
    assert_eq!(code, 64, "expected exit 64, stderr: {:?}", out.stderr);
}

#[test]
fn print_combined_with_subcommand_exits_64() {
    let tmp = TempDir::new().expect("tempdir");
    let out = run(tmp.path(), &["-p", "hi", "doctor"]);
    let code = out.status.code().unwrap_or_default();
    assert_eq!(code, 64, "expected exit 64, stderr: {:?}", out.stderr);
    let stderr = String::from_utf8(out.stderr).expect("utf-8 stderr");
    assert!(
        stderr.contains("mutually exclusive"),
        "stderr missing mutual-exclusion diagnostic: {stderr}"
    );
}

#[test]
fn output_format_without_print_is_rejected_by_clap() {
    // `--output-format` is gated on `--print` via clap's `requires`
    // attribute, so passing it alone exits 64 (clap's bad-args code).
    let tmp = TempDir::new().expect("tempdir");
    let out = run(tmp.path(), &["--output-format", "json"]);
    let code = out.status.code().unwrap_or_default();
    assert_eq!(code, 64, "expected exit 64, stderr: {:?}", out.stderr);
}

#[test]
fn output_format_json_emits_envelope_with_blocks_and_metrics() {
    let tmp = TempDir::new().expect("tempdir");
    let out = run(
        tmp.path(),
        &["-p", "alpha bravo", "--output-format", "json"],
    );
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8(out.stdout).expect("utf-8 stdout");
    let env: serde_json::Value = serde_json::from_str(stdout.trim()).expect("valid json envelope");

    // Plan/43 §3.1 — the envelope must carry these top-level keys.
    let obj = env.as_object().expect("envelope is a JSON object");
    for key in [
        "session_id",
        "turn_id",
        "model",
        "blocks",
        "metrics",
        "outcome",
    ] {
        assert!(
            obj.contains_key(key),
            "missing top-level key `{key}` in envelope: {env}"
        );
    }

    // Outcome must be "success" on the floor EchoProvider path.
    assert_eq!(
        env["outcome"].as_str(),
        Some("success"),
        "outcome wrong: {env}"
    );

    // The block array must include both prompt words as separate
    // text blocks (the EchoProvider's documented per-word fan-out).
    let blocks = env["blocks"].as_array().expect("blocks is a JSON array");
    let text_blocks: Vec<&str> = blocks
        .iter()
        .filter(|b| b.get("kind").and_then(|k| k.as_str()) == Some("text"))
        .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
        .collect();
    assert!(
        text_blocks.iter().any(|t| t.contains("alpha")),
        "expected `alpha` in a text block: {text_blocks:?}"
    );
    assert!(
        text_blocks.iter().any(|t| t.contains("bravo")),
        "expected `bravo` in a text block: {text_blocks:?}"
    );

    // Metrics must surface as an object with the documented counters.
    let metrics = env["metrics"]
        .as_object()
        .expect("metrics is a JSON object");
    assert!(
        metrics.contains_key("prompt_tokens"),
        "metrics missing `prompt_tokens`: {env}"
    );
    assert!(
        metrics.contains_key("completion_tokens"),
        "metrics missing `completion_tokens`: {env}"
    );
}

#[test]
fn output_format_stream_json_emits_ndjson_with_done_footer() {
    let tmp = TempDir::new().expect("tempdir");
    let out = run(
        tmp.path(),
        &["-p", "stream-test", "--output-format", "stream-json"],
    );
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8(out.stdout).expect("utf-8 stdout");

    // Each non-empty line must parse as JSON; the last one must have
    // `done: true` per plan/43 §3.2.
    let lines: Vec<&str> = stdout.lines().filter(|l| !l.is_empty()).collect();
    assert!(
        lines.len() >= 2,
        "stream-json should emit at least one block + a done footer, got {lines:?}"
    );
    for line in &lines {
        let _v: serde_json::Value = serde_json::from_str(line).expect("each line is valid JSON");
    }
    let footer: serde_json::Value = serde_json::from_str(lines.last().expect("at least one line"))
        .expect("footer is valid JSON");
    assert_eq!(
        footer["done"].as_bool(),
        Some(true),
        "footer missing `done: true`: {footer}"
    );
    assert_eq!(
        footer["outcome"].as_str(),
        Some("success"),
        "footer outcome wrong: {footer}"
    );
}

#[test]
fn text_format_default_prints_assistant_text_only() {
    // The default format must produce a clean human-readable line
    // with no JSON envelope wrappers — pipe friendliness is the whole
    // point of `stratum -p` (plan/43 §3 row 1).
    let tmp = TempDir::new().expect("tempdir");
    let out = run(tmp.path(), &["-p", "plain"]);
    assert!(out.status.success());
    let stdout = String::from_utf8(out.stdout).expect("utf-8 stdout");
    assert!(
        !stdout.trim_start().starts_with('{'),
        "text format must not emit a JSON envelope: {stdout}"
    );
    assert!(stdout.contains("plain"), "missing assistant text: {stdout}");
}
