//! Integration tests for `stratum eval run --suite <PATH> [--out <PATH>] [--json] [--model <SLUG>]`.
//!
//! Each test spawns the built `stratum` binary against an isolated
//! `--storage-root` (a `TempDir`) and stages an [`EvalSuite`] JSON fixture
//! on disk. The runner wraps `AgentFactory::echo`, so the expected
//! substring `"hello"` will be present in the haystack whenever the prompt
//! contains it (the Echo provider repeats its prompt verbatim).

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

use stratum_runtime::{EvalCase, EvalSuite, EVAL_SUITE_SCHEMA_VERSION};
use tempfile::TempDir;

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_stratum"))
}

fn write_suite(path: &Path, suite: &EvalSuite) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("create suite parent dir");
    }
    let body = serde_json::to_vec_pretty(suite).expect("serialize suite");
    fs::write(path, body).expect("write suite file");
}

fn case(id: &str, prompt: &str, expected: &[&str], forbidden: &[&str]) -> EvalCase {
    EvalCase {
        id: id.into(),
        prompt: prompt.into(),
        expected_substrings: expected.iter().map(|s| (*s).to_string()).collect(),
        forbidden_substrings: forbidden.iter().map(|s| (*s).to_string()).collect(),
        max_blocks: 32, system_override: None,
    }
}

/// Spawn `stratum --storage-root <root> eval run <sub-args...>`.
fn run_eval(root: &Path, sub: &[&str]) -> std::process::Output {
    let mut cmd = bin();
    cmd.args(["--storage-root", root.to_str().expect("utf-8 root")]);
    cmd.args(["eval", "run"]);
    cmd.args(sub);
    cmd.output().expect("spawn stratum eval run")
}

#[test]
fn happy_all_pass_suite_exits_zero_with_prose_summary() {
    let tmp = TempDir::new().expect("tempdir");
    let suite_path = tmp.path().join("suite.json");
    let suite = EvalSuite {
        schema_version: EVAL_SUITE_SCHEMA_VERSION,
        name: "happy".into(),
        cases: vec![
            case("c1", "hello world", &["hello"], &[]),
            case("c2", "alpha bravo", &["alpha"], &[]),
        ],
    };
    write_suite(&suite_path, &suite);

    let out = run_eval(
        tmp.path(),
        &["--suite", suite_path.to_str().expect("utf-8 suite")],
    );
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8(out.stdout).expect("utf-8 stdout");
    assert!(stdout.contains("suite: happy"), "got: {stdout}");
    assert!(stdout.contains("passed: 2/2 (100.0%)"), "got: {stdout}");
    assert!(stdout.contains("failed: 0"), "got: {stdout}");
    assert!(stdout.contains("[pass] c1"), "got: {stdout}");
    assert!(stdout.contains("[pass] c2"), "got: {stdout}");
    assert!(stdout.contains("report saved to:"), "got: {stdout}");
}

#[test]
fn mixed_pass_fail_suite_exits_one_with_summary() {
    let tmp = TempDir::new().expect("tempdir");
    let suite_path = tmp.path().join("mixed.json");
    let suite = EvalSuite {
        schema_version: EVAL_SUITE_SCHEMA_VERSION,
        name: "mixed".into(),
        cases: vec![
            case("ok", "hello world", &["hello"], &[]),
            case("bad", "hello world", &["never-there"], &[]),
        ],
    };
    write_suite(&suite_path, &suite);

    let out = run_eval(
        tmp.path(),
        &["--suite", suite_path.to_str().expect("utf-8 suite")],
    );
    assert!(!out.status.success());
    assert_eq!(out.status.code(), Some(1));
    let stdout = String::from_utf8(out.stdout).expect("utf-8 stdout");
    assert!(stdout.contains("passed: 1/2 (50.0%)"), "got: {stdout}");
    assert!(stdout.contains("failed: 1"), "got: {stdout}");
    assert!(stdout.contains("[pass] ok"), "got: {stdout}");
    assert!(stdout.contains("[fail] bad"), "got: {stdout}");
    assert!(
        stdout.contains("missing substring"),
        "expected failure reason in stdout, got: {stdout}"
    );
}

#[test]
fn json_flag_emits_valid_eval_report_json() {
    let tmp = TempDir::new().expect("tempdir");
    let suite_path = tmp.path().join("json-suite.json");
    let suite = EvalSuite {
        schema_version: EVAL_SUITE_SCHEMA_VERSION,
        name: "json-suite".into(),
        cases: vec![case("only", "hello world", &["hello"], &[])],
    };
    write_suite(&suite_path, &suite);

    let out = run_eval(
        tmp.path(),
        &[
            "--suite",
            suite_path.to_str().expect("utf-8 suite"),
            "--json",
        ],
    );
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8(out.stdout).expect("utf-8 stdout");
    let v: serde_json::Value = serde_json::from_str(stdout.trim()).expect("stdout is valid JSON");
    assert_eq!(v["suite_name"], "json-suite");
    assert_eq!(v["passed"], 1);
    assert_eq!(v["failed"], 0);
    let runs = v["runs"].as_array().expect("runs is array");
    assert_eq!(runs.len(), 1);
    assert_eq!(runs[0]["case_id"], "only");
    assert_eq!(runs[0]["passed"], true);
}

#[test]
fn custom_out_flag_writes_report_to_requested_path() {
    let tmp = TempDir::new().expect("tempdir");
    let suite_path = tmp.path().join("suite.json");
    let custom_out: PathBuf = tmp.path().join("nested").join("custom-report.json");
    let suite = EvalSuite {
        schema_version: EVAL_SUITE_SCHEMA_VERSION,
        name: "custom-out".into(),
        cases: vec![case("c", "hello world", &["hello"], &[])],
    };
    write_suite(&suite_path, &suite);

    let out = run_eval(
        tmp.path(),
        &[
            "--suite",
            suite_path.to_str().expect("utf-8 suite"),
            "--out",
            custom_out.to_str().expect("utf-8 out"),
        ],
    );
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(custom_out.exists(), "expected custom out path to exist");
    let body = fs::read(&custom_out).expect("read custom report");
    let v: serde_json::Value = serde_json::from_slice(&body).expect("custom report is valid JSON");
    assert_eq!(v["suite_name"], "custom-out");
    assert_eq!(v["passed"], 1);

    // The default eval-reports dir under <state> should NOT have been touched.
    let default_dir = tmp.path().join("state").join("eval-reports");
    assert!(
        !default_dir.exists()
            || fs::read_dir(&default_dir)
                .expect("read dir")
                .next()
                .is_none(),
        "default eval-reports dir should be empty when --out is set",
    );
}

#[test]
fn default_out_resolves_under_state_eval_reports() {
    let tmp = TempDir::new().expect("tempdir");
    let suite_path = tmp.path().join("suite.json");
    let suite = EvalSuite {
        schema_version: EVAL_SUITE_SCHEMA_VERSION,
        name: "default out".into(), // space → underscore in slug
        cases: vec![case("c", "hello world", &["hello"], &[])],
    };
    write_suite(&suite_path, &suite);

    let out = run_eval(
        tmp.path(),
        &["--suite", suite_path.to_str().expect("utf-8 suite")],
    );
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8(out.stdout).expect("utf-8 stdout");
    let default_dir = tmp.path().join("state").join("eval-reports");
    assert!(default_dir.exists(), "default eval-reports dir not created");
    let entries: Vec<_> = fs::read_dir(&default_dir)
        .expect("read default dir")
        .filter_map(Result::ok)
        .collect();
    assert_eq!(entries.len(), 1, "expected one report file");
    let name = entries[0].file_name();
    let name_str = name.to_string_lossy();
    assert!(
        name_str.starts_with("default_out-"),
        "report filename should be slugified: got {name_str}",
    );
    assert!(
        name_str.ends_with(".json"),
        "report filename should end with .json: got {name_str}",
    );
    // Prose summary should reference the same path.
    assert!(
        stdout.contains(default_dir.to_str().expect("utf-8 path")),
        "stdout should mention default eval-reports dir, got: {stdout}",
    );
}

#[test]
fn missing_suite_flag_exits_64() {
    let tmp = TempDir::new().expect("tempdir");
    let out = run_eval(tmp.path(), &[]);
    assert!(!out.status.success());
    // clap parse failure → exit code 64.
    assert_eq!(out.status.code(), Some(64));
}

#[test]
fn missing_suite_file_exits_one_with_strat_e1001() {
    let tmp = TempDir::new().expect("tempdir");
    let missing = tmp.path().join("nope.json");
    let out = run_eval(
        tmp.path(),
        &["--suite", missing.to_str().expect("utf-8 path")],
    );
    assert!(!out.status.success());
    assert_eq!(out.status.code(), Some(1));
    let stderr = String::from_utf8(out.stderr).expect("utf-8 stderr");
    assert!(stderr.contains("STRAT-E1001"), "stderr: {stderr}");
}

#[test]
fn malformed_suite_json_exits_one_with_strat_e1001() {
    let tmp = TempDir::new().expect("tempdir");
    let suite_path = tmp.path().join("bad.json");
    fs::write(&suite_path, b"{not json").expect("write bad json");
    let out = run_eval(
        tmp.path(),
        &["--suite", suite_path.to_str().expect("utf-8 path")],
    );
    assert!(!out.status.success());
    assert_eq!(out.status.code(), Some(1));
    let stderr = String::from_utf8(out.stderr).expect("utf-8 stderr");
    assert!(stderr.contains("STRAT-E1001"), "stderr: {stderr}");
}

/// Resolve a path relative to the workspace root (`CARGO_MANIFEST_DIR`
/// points at `crates/stratum-cli`; the workspace is two levels up).
fn workspace_path(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates dir parent")
        .parent()
        .expect("workspace dir parent")
        .join(rel)
}

#[test]
fn shipped_baseline_suite_passes_against_echo() {
    let tmp = TempDir::new().expect("tempdir");
    let suite = workspace_path("evals/baseline.json");
    let out = run_eval(
        tmp.path(),
        &["--suite", suite.to_str().expect("utf-8 suite")],
    );
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8(out.stdout).expect("utf-8 stdout");
    assert!(stdout.contains("passed: 20/20"), "got: {stdout}");
}

#[test]
fn shipped_coder_suite_passes_against_echo() {
    let tmp = TempDir::new().expect("tempdir");
    let suite = workspace_path("evals/coder.json");
    let out = run_eval(
        tmp.path(),
        &["--suite", suite.to_str().expect("utf-8 suite")],
    );
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8(out.stdout).expect("utf-8 stdout");
    assert!(stdout.contains("passed: 10/10"), "got: {stdout}");
}

#[test]
fn shipped_polisher_suite_passes_against_echo() {
    let tmp = TempDir::new().expect("tempdir");
    let suite = workspace_path("evals/polisher.json");
    let out = run_eval(
        tmp.path(),
        &["--suite", suite.to_str().expect("utf-8 suite")],
    );
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8(out.stdout).expect("utf-8 stdout");
    assert!(stdout.contains("passed: 10/10"), "got: {stdout}");
}

#[test]
fn model_flag_is_accepted_and_ignored() {
    // The flag is parsed for forward compatibility but ignored — the Echo
    // backbone still runs and produces a normal report.
    let tmp = TempDir::new().expect("tempdir");
    let suite_path = tmp.path().join("suite.json");
    let suite = EvalSuite {
        schema_version: EVAL_SUITE_SCHEMA_VERSION,
        name: "with-model".into(),
        cases: vec![case("c", "hello world", &["hello"], &[])],
    };
    write_suite(&suite_path, &suite);
    let out = run_eval(
        tmp.path(),
        &[
            "--suite",
            suite_path.to_str().expect("utf-8 suite"),
            "--model",
            "anything-goes",
        ],
    );
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}
