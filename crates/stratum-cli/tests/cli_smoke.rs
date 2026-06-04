//! End-to-end integration tests for the Stratum CLI.
//!
//! These spawn the actual built binary via `CARGO_BIN_EXE_stratum`, exercising
//! the same code path users will hit. Phase 1 surface: greeting + doctor + init,
//! parameterized by `--storage-root` so each test gets a fresh temp directory.

use std::process::Command;

use tempfile::TempDir;

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_stratum"))
}

#[test]
fn default_prints_hello_and_status() {
    let tmp = TempDir::new().unwrap();
    let output = bin()
        .args(["--storage-root", tmp.path().to_str().unwrap()])
        .output()
        .expect("spawn stratum");
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("hello, tier=unknown"), "got: {stdout}");
    assert!(stdout.contains("not installed"));
}

#[test]
fn doctor_json_is_parseable() {
    let tmp = TempDir::new().unwrap();
    let output = bin()
        .args([
            "--storage-root",
            tmp.path().to_str().unwrap(),
            "--json",
            "doctor",
        ])
        .output()
        .expect("spawn stratum");
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(stdout.trim()).unwrap();
    assert_eq!(parsed["schema_version"], 1);
    assert!(parsed["probe"]["ram_total_mib"].as_u64().unwrap() > 0);
    assert_eq!(parsed["installed"], false);
}

#[test]
fn init_then_doctor_marks_installed() {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path().to_str().unwrap();
    let init = bin()
        .args(["--storage-root", root, "init"])
        .output()
        .unwrap();
    assert!(
        init.status.success(),
        "init stderr: {}",
        String::from_utf8_lossy(&init.stderr)
    );
    let doctor = bin()
        .args(["--storage-root", root, "--json", "doctor"])
        .output()
        .unwrap();
    let stdout = String::from_utf8(doctor.stdout).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(stdout.trim()).unwrap();
    assert_eq!(parsed["installed"], true);
}

#[test]
fn unknown_subcommand_exits_64() {
    let tmp = TempDir::new().unwrap();
    let output = bin()
        .args(["--storage-root", tmp.path().to_str().unwrap(), "wat"])
        .output()
        .expect("spawn stratum");
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
