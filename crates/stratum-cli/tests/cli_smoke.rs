//! End-to-end integration tests for the Stratum CLI.
//!
//! These spawn the actual built binary via `CARGO_BIN_EXE_stratum`, exercising
//! the same code path users will hit. Phase 0 surface only.

use std::process::Command;

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_stratum"))
}

#[test]
fn default_prints_hello() {
    let output = bin().output().expect("spawn stratum");
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("hello, tier=unknown"), "got: {stdout}");
}

#[test]
fn doctor_prose_succeeds() {
    let output = bin().arg("doctor").output().expect("spawn stratum");
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("phase 0 stub"));
}

#[test]
fn doctor_json_is_parseable() {
    let output = bin()
        .args(["--json", "doctor"])
        .output()
        .expect("spawn stratum");
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(stdout.trim()).unwrap();
    assert_eq!(parsed["schema_version"], 1);
    assert_eq!(parsed["tier"], "unknown");
}

#[test]
fn unknown_subcommand_exits_64() {
    let output = bin().arg("wat").output().expect("spawn stratum");
    assert_eq!(output.status.code(), Some(64));
}

#[test]
fn version_flag_prints_version() {
    let output = bin().arg("--version").output().expect("spawn stratum");
    assert!(output.status.success() || output.status.code() == Some(64));
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stdout.contains("stratum") || stderr.contains("stratum"));
}
