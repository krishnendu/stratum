//! End-to-end tests for `stratum agents list [--json]` and
//! `stratum agents show --role <role> [--json]`.
//!
//! Each test spawns the built `stratum` binary against a `TempDir`-rooted
//! `--storage-root` and asserts on the documented output / exit code
//! contract:
//!
//! * `list` against an empty directory exits 0 with the prose preamble and
//!   no role rows.
//! * `list` after a valid `<state>/agents/cavemanish.toml` lists that role.
//! * `list --json` round-trips through `serde_json::Value` with the
//!   `registered` / `skipped` / `errors` keys.
//! * `show --role cavemanish` prints the prose record.
//! * `show --role missing` exits `1` with a `STRAT-E1001` marker.
//! * `show --role cavemanish --json` round-trips through `serde_json::Value`.
//! * A malformed TOML stages a `LoadFailure` and surfaces under `errors:`.
//! * An unknown role in TOML stages a `SkipReason::UnknownRole` and
//!   surfaces under `skipped:`.

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

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_stratum"))
}

fn run(root: &Path, extra: &[&str]) -> std::process::Output {
    let mut cmd = bin();
    cmd.args(["--storage-root", root.to_str().expect("tempdir utf-8")]);
    cmd.args(extra);
    cmd.output().expect("spawn stratum")
}

fn agents_dir(root: &Path) -> std::path::PathBuf {
    let dir = root.join("state").join("agents");
    std::fs::create_dir_all(&dir).expect("mkdir state/agents");
    dir
}

/// Body shape matches [`stratum_runtime::AgentLoader::load_file`]'s serde
/// expectation — `schema_version` + `roles` + `model` + `tools` + `sandbox`,
/// plus the optional `name` / `description` fields used by the prose renderer.
fn minimal_body(name: &str, role: &str) -> String {
    format!(
        r#"
schema_version = 1
name = "{name}"
description = "small reasoner for {role}"
roles = ["{role}"]
model = "echo"
tools = ["fs.read"]
sandbox = "bwrap-strict"
"#
    )
}

#[test]
fn list_empty_dir_prose_exit_zero() {
    let tmp = TempDir::new().unwrap();
    let out = run(tmp.path(), &["agents", "list"]);
    assert!(
        out.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("registered roles (sorted):"),
        "expected preamble in {stdout:?}",
    );
    assert!(
        stdout.contains("skipped: 0"),
        "expected 'skipped: 0' in {stdout:?}",
    );
    assert!(
        stdout.contains("errors: 0"),
        "expected 'errors: 0' in {stdout:?}",
    );
    // No role rows under the preamble.
    let role_rows: Vec<&str> = stdout.lines().filter(|l| l.starts_with("  - ")).collect();
    assert!(
        role_rows.is_empty(),
        "expected no role rows, got {role_rows:?}",
    );
}

#[test]
fn list_with_one_valid_agent_shows_role() {
    let tmp = TempDir::new().unwrap();
    let dir = agents_dir(tmp.path());
    std::fs::write(
        dir.join("cavemanish.toml"),
        minimal_body("cavemanish", "cavemanish"),
    )
    .unwrap();
    let out = run(tmp.path(), &["agents", "list"]);
    assert!(
        out.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("- cavemanish"),
        "expected '- cavemanish' in {stdout:?}",
    );
    assert!(
        stdout.contains("skipped: 0"),
        "expected 'skipped: 0' in {stdout:?}",
    );
    assert!(
        stdout.contains("errors: 0"),
        "expected 'errors: 0' in {stdout:?}",
    );
}

#[test]
fn list_json_roundtrips_through_serde_json_value() {
    let tmp = TempDir::new().unwrap();
    let dir = agents_dir(tmp.path());
    std::fs::write(
        dir.join("polisher.toml"),
        minimal_body("polisher", "polisher"),
    )
    .unwrap();
    let out = run(tmp.path(), &["agents", "list", "--json"]);
    assert!(
        out.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    let parsed: serde_json::Value =
        serde_json::from_str(&stdout).expect("stdout must be valid JSON");
    let obj = parsed
        .as_object()
        .expect("top-level JSON shape is an object");
    assert!(obj.contains_key("registered"));
    assert!(obj.contains_key("skipped"));
    assert!(obj.contains_key("errors"));
    let registered = obj["registered"].as_array().expect("registered is array");
    assert_eq!(registered.len(), 1);
    assert_eq!(registered[0], serde_json::Value::String("polisher".into()));
}

#[test]
fn show_role_prose_prints_fields() {
    let tmp = TempDir::new().unwrap();
    let dir = agents_dir(tmp.path());
    std::fs::write(
        dir.join("cavemanish.toml"),
        minimal_body("cavemanish", "cavemanish"),
    )
    .unwrap();
    let out = run(tmp.path(), &["agents", "show", "--role", "cavemanish"]);
    assert!(
        out.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("role: cavemanish"),
        "missing role line in {stdout:?}",
    );
    assert!(
        stdout.contains("name: cavemanish"),
        "missing name line in {stdout:?}",
    );
    assert!(
        stdout.contains("description: small reasoner for cavemanish"),
        "missing description in {stdout:?}",
    );
    assert!(
        stdout.contains("model: echo"),
        "missing model line in {stdout:?}",
    );
    assert!(
        stdout.contains("sandbox: bwrap-strict"),
        "missing sandbox line in {stdout:?}",
    );
    assert!(
        stdout.contains("capabilities: fs.read"),
        "missing capabilities line in {stdout:?}",
    );
}

#[test]
fn show_missing_role_exits_one_with_strat_e1001() {
    let tmp = TempDir::new().unwrap();
    agents_dir(tmp.path());
    let out = run(tmp.path(), &["agents", "show", "--role", "missing"]);
    assert!(!out.status.success());
    assert_eq!(out.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("STRAT-E1001"),
        "expected STRAT-E1001 marker, got {stderr:?}",
    );
}

#[test]
fn show_role_json_roundtrips_through_serde_json_value() {
    let tmp = TempDir::new().unwrap();
    let dir = agents_dir(tmp.path());
    std::fs::write(
        dir.join("cavemanish.toml"),
        minimal_body("cavemanish", "cavemanish"),
    )
    .unwrap();
    let out = run(
        tmp.path(),
        &["agents", "show", "--role", "cavemanish", "--json"],
    );
    assert!(
        out.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    let parsed: serde_json::Value =
        serde_json::from_str(&stdout).expect("stdout must be valid JSON");
    let obj = parsed
        .as_object()
        .expect("top-level JSON shape is an object");
    assert_eq!(obj["name"], serde_json::Value::String("cavemanish".into()));
    assert_eq!(obj["model"], serde_json::Value::String("echo".into()));
    assert_eq!(
        obj["sandbox"],
        serde_json::Value::String("bwrap-strict".into())
    );
}

#[test]
fn list_malformed_toml_lands_in_errors_exit_zero() {
    let tmp = TempDir::new().unwrap();
    let dir = agents_dir(tmp.path());
    std::fs::write(dir.join("bad.toml"), b"not = [ valid").unwrap();
    let out = run(tmp.path(), &["agents", "list"]);
    assert!(
        out.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("errors: 1"),
        "expected 'errors: 1' in {stdout:?}",
    );
    assert!(
        stdout.contains("bad.toml"),
        "expected bad.toml in errors list, got {stdout:?}",
    );
}

#[test]
fn list_unknown_role_lands_in_skipped_exit_zero() {
    let tmp = TempDir::new().unwrap();
    let dir = agents_dir(tmp.path());
    std::fs::write(
        dir.join("weird.toml"),
        minimal_body("weird", "not-a-real-role"),
    )
    .unwrap();
    let out = run(tmp.path(), &["agents", "list"]);
    assert!(
        out.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("skipped: 1"),
        "expected 'skipped: 1' in {stdout:?}",
    );
    assert!(
        stdout.contains("unknown role \"not-a-real-role\""),
        "expected unknown role message in {stdout:?}",
    );
}
