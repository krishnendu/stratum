//! `stratum doctor --telemetry` integration tests.
//!
//! These exercise the full payload-assembly path end-to-end against a real
//! CLI binary spawned by `cargo test`:
//!
//! - default (no config file) → telemetry is enabled, payload is printed
//! - opt-out via `<state>/telemetry.json` → `disabled` shown, no payload
//! - `--telemetry-event` flag flows through to the emitted `event_kind`
//! - the persistent `anon_install_id` survives across invocations
//! - malformed config / id files do not break the command (opt-out posture)
//!
//! The brief calls out an implicit allowlist check: if the assembled
//! payload ever drifts beyond the runtime allowlist, the CLI exits non-zero.
//! Every test below asserts exit-0, which implicitly proves the
//! allowlist guard passed.

// Integration test binary — helpers panic on setup failures by design.
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

/// Run `stratum --storage-root <root> doctor <extra...>` and return
/// (exit-success, stdout, stderr).
fn run_doctor(root: &Path, extra: &[&str]) -> (bool, String, String) {
    let mut cmd = bin();
    cmd.args(["--storage-root", root.to_str().expect("temp path is utf-8")]);
    // Mirror the CLI's flag ordering: global `--json` lives before the
    // subcommand, telemetry flags after.
    let mut json_first = false;
    for arg in extra {
        if *arg == "--json" {
            json_first = true;
            break;
        }
    }
    if json_first {
        cmd.arg("--json");
    }
    cmd.arg("doctor");
    for arg in extra {
        if *arg == "--json" {
            continue;
        }
        cmd.arg(arg);
    }
    let output = cmd.output().expect("spawn stratum doctor");
    (
        output.status.success(),
        String::from_utf8(output.stdout).expect("stdout utf-8"),
        String::from_utf8(output.stderr).expect("stderr utf-8"),
    )
}

fn parse_json(out: &str) -> serde_json::Value {
    serde_json::from_str(out.trim()).expect("doctor --json emits valid JSON")
}

fn state_dir(root: &Path) -> std::path::PathBuf {
    root.join("state")
}

fn write_telemetry_config(root: &Path, body: &str) {
    let dir = state_dir(root);
    std::fs::create_dir_all(&dir).expect("create state dir");
    std::fs::write(dir.join("telemetry.json"), body).expect("write telemetry.json");
}

#[test]
fn doctor_telemetry_prose_prints_payload_when_enabled() {
    let tmp = TempDir::new().expect("tempdir");
    let (ok, out, err) = run_doctor(tmp.path(), &["--telemetry"]);
    assert!(ok, "stratum doctor failed: stderr={err}");
    assert!(
        out.contains("--- telemetry ---"),
        "missing telemetry block in stdout:\n{out}",
    );
    // Sanity check on a couple of payload fields.
    assert!(out.contains("\"schema_version\""), "got:\n{out}");
    assert!(out.contains("\"event_kind\""), "got:\n{out}");
    assert!(out.contains("\"daily_active\""), "got:\n{out}");
}

#[test]
fn doctor_telemetry_json_includes_payload() {
    let tmp = TempDir::new().expect("tempdir");
    let (ok, out, err) = run_doctor(tmp.path(), &["--json", "--telemetry"]);
    assert!(ok, "stratum doctor failed: stderr={err}");
    let v = parse_json(&out);
    let payload = v
        .get("telemetry")
        .expect("doctor JSON missing `telemetry` key");
    assert!(
        payload.is_object(),
        "telemetry must be an object when enabled, got: {payload}"
    );
    assert_eq!(payload["schema_version"], serde_json::json!(1));
    assert_eq!(payload["event_kind"], serde_json::json!("daily_active"));
    assert!(
        payload["anon_install_id"]
            .as_str()
            .is_some_and(|s| s.len() == 16),
        "anon_install_id must be 16 chars, got: {payload}"
    );
}

#[test]
fn doctor_telemetry_prose_opt_out_prints_disabled() {
    let tmp = TempDir::new().expect("tempdir");
    write_telemetry_config(tmp.path(), r#"{"enabled": false}"#);
    let (ok, out, err) = run_doctor(tmp.path(), &["--telemetry"]);
    assert!(ok, "stratum doctor failed: stderr={err}");
    assert!(
        out.contains("--- telemetry: disabled ---"),
        "expected disabled marker, got:\n{out}"
    );
    assert!(
        !out.contains("--- telemetry ---\n"),
        "must not print payload header when disabled, got:\n{out}"
    );
}

#[test]
fn doctor_telemetry_json_opt_out_emits_null() {
    let tmp = TempDir::new().expect("tempdir");
    write_telemetry_config(tmp.path(), r#"{"enabled": false}"#);
    let (ok, out, err) = run_doctor(tmp.path(), &["--json", "--telemetry"]);
    assert!(ok, "stratum doctor failed: stderr={err}");
    let v = parse_json(&out);
    assert!(
        v.get("telemetry").is_some(),
        "doctor JSON missing `telemetry` key entirely: {v}"
    );
    assert_eq!(
        v["telemetry"],
        serde_json::Value::Null,
        "telemetry must serialize as null when disabled, got: {v}",
    );
}

#[test]
fn doctor_telemetry_event_override_flows_through() {
    let tmp = TempDir::new().expect("tempdir");
    let (ok, out, err) = run_doctor(
        tmp.path(),
        &["--json", "--telemetry", "--telemetry-event", "update"],
    );
    assert!(ok, "stratum doctor failed: stderr={err}");
    let v = parse_json(&out);
    assert_eq!(v["telemetry"]["event_kind"], serde_json::json!("update"));
}

#[test]
fn doctor_telemetry_creates_anon_install_id_file() {
    let tmp = TempDir::new().expect("tempdir");
    let id_path = state_dir(tmp.path()).join("anon_install_id");
    assert!(
        !id_path.exists(),
        "precondition: anon_install_id must not exist yet"
    );
    let (ok, _out, err) = run_doctor(tmp.path(), &["--telemetry"]);
    assert!(ok, "stratum doctor failed: stderr={err}");
    assert!(
        id_path.exists(),
        "doctor --telemetry must create {id_path:?}"
    );
    let body = std::fs::read_to_string(&id_path).expect("read anon_install_id");
    assert_eq!(
        body.trim().len(),
        16,
        "anon_install_id must be 16 hex chars"
    );
}

#[test]
fn doctor_telemetry_persists_anon_install_id_across_runs() {
    let tmp = TempDir::new().expect("tempdir");
    let (ok1, out1, err1) = run_doctor(tmp.path(), &["--json", "--telemetry"]);
    assert!(ok1, "first run failed: stderr={err1}");
    let v1 = parse_json(&out1);
    let id1 = v1["telemetry"]["anon_install_id"]
        .as_str()
        .expect("anon_install_id present on first run")
        .to_owned();

    let (ok2, out2, err2) = run_doctor(tmp.path(), &["--json", "--telemetry"]);
    assert!(ok2, "second run failed: stderr={err2}");
    let v2 = parse_json(&out2);
    let id2 = v2["telemetry"]["anon_install_id"]
        .as_str()
        .expect("anon_install_id present on second run");
    assert_eq!(id1, id2, "anon_install_id must persist across runs");
}

#[test]
fn doctor_telemetry_malformed_config_falls_back_to_enabled() {
    let tmp = TempDir::new().expect("tempdir");
    write_telemetry_config(tmp.path(), "this is not json");
    let (ok, out, err) = run_doctor(tmp.path(), &["--json", "--telemetry"]);
    assert!(
        ok,
        "doctor must succeed even with malformed telemetry.json: stderr={err}"
    );
    let v = parse_json(&out);
    assert!(
        v["telemetry"].is_object(),
        "malformed config must fall back to enabled=true (telemetry object expected), got: {v}",
    );
}

#[test]
fn doctor_telemetry_malformed_anon_install_id_is_regenerated() {
    let tmp = TempDir::new().expect("tempdir");
    let dir = state_dir(tmp.path());
    std::fs::create_dir_all(&dir).expect("create state dir");
    let id_path = dir.join("anon_install_id");
    std::fs::write(&id_path, "garbage").expect("seed malformed id");

    let (ok, out, err) = run_doctor(tmp.path(), &["--json", "--telemetry"]);
    assert!(ok, "doctor must succeed: stderr={err}");
    let v = parse_json(&out);
    let id = v["telemetry"]["anon_install_id"]
        .as_str()
        .expect("anon_install_id present");
    assert_eq!(id.len(), 16);
    // The file should now contain the regenerated id, not the garbage.
    let body = std::fs::read_to_string(&id_path).expect("read regenerated id");
    assert_eq!(body.trim(), id);
    assert_ne!(body.trim(), "garbage");
}

#[test]
fn doctor_without_telemetry_flag_omits_payload_in_prose() {
    // Sanity: the existing doctor surface is unchanged when --telemetry is
    // not requested.
    let tmp = TempDir::new().expect("tempdir");
    let (ok, out, err) = run_doctor(tmp.path(), &[]);
    assert!(ok, "stratum doctor failed: stderr={err}");
    assert!(!out.contains("--- telemetry"), "got:\n{out}");
}

#[test]
fn doctor_without_telemetry_flag_emits_null_in_json() {
    let tmp = TempDir::new().expect("tempdir");
    let (ok, out, err) = run_doctor(tmp.path(), &["--json"]);
    assert!(ok, "stratum doctor failed: stderr={err}");
    let v = parse_json(&out);
    // Field is present (stable shape) but null because --telemetry was
    // not passed.
    assert_eq!(v["telemetry"], serde_json::Value::Null);
}
