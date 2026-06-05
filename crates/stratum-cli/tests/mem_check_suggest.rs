//! Integration tests for the `--requested`/`--requested-mib`/`--loaded-file`
//! surface of `stratum mem-check`.
//!
//! These spawn the built binary against an isolated `--storage-root` and
//! exercise the three modes that the subcommand now dispatches:
//!
//! * No flags → print available RAM.
//! * `--requested` + `--requested-mib` → consult
//!   `MemoryGate::suggest_unloads` against `<state>/loaded.json` (or
//!   `--loaded-file`).
//! * Legacy `--weight-rss`/`--kv-per-token`/`--context` (covered by the
//!   in-file unit tests).

// Integration test binary: every fn here exists only for `cargo test`. Test
// helpers panic on setup failures by design.
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

fn state_dir(root: &Path) -> PathBuf {
    let dir = root.join("state");
    fs::create_dir_all(&dir).expect("create state dir");
    dir
}

fn loaded_path(root: &Path) -> PathBuf {
    state_dir(root).join("loaded.json")
}

fn write_loaded_json(root: &Path, body: &str) {
    fs::write(loaded_path(root), body).expect("write loaded.json");
}

/// Pretty assertion helper: report stdout/stderr on failure so test logs
/// stay readable when the binary panics or returns an unexpected code.
fn run(args: &[&str], storage_root: &Path) -> (i32, String, String) {
    let output = bin()
        .arg("--storage-root")
        .arg(storage_root)
        .args(args)
        .output()
        .expect("spawn stratum binary");
    let stdout = String::from_utf8(output.stdout).expect("stdout utf8");
    let stderr = String::from_utf8(output.stderr).expect("stderr utf8");
    let code = output.status.code().unwrap_or(-1);
    (code, stdout, stderr)
}

#[test]
fn mem_check_no_flags_prints_available_ram() {
    let tmp = TempDir::new().expect("tempdir");
    let (code, stdout, stderr) = run(&["mem-check"], tmp.path());
    assert_eq!(code, 0, "stderr was: {stderr}");
    assert!(stdout.contains("available:"), "stdout was: {stdout}");
    assert!(stdout.contains(" MiB"), "stdout was: {stdout}");
    assert!(stderr.is_empty(), "stderr was: {stderr}");
}

#[test]
fn mem_check_requested_without_loaded_file_fits_without_eviction() {
    // No loaded.json on disk and a tiny 100 MiB request → always fits on any
    // host the CI is going to run on.
    let tmp = TempDir::new().expect("tempdir");
    let (code, stdout, stderr) = run(
        &["mem-check", "--requested", "foo", "--requested-mib", "100"],
        tmp.path(),
    );
    assert_eq!(code, 0, "stderr was: {stderr}");
    assert!(
        stdout.contains("fits without eviction"),
        "stdout was: {stdout}",
    );
}

#[test]
fn mem_check_requested_with_heavy_loaded_set_suggests_evictions() {
    let tmp = TempDir::new().expect("tempdir");
    // Three resident models. The request is large enough that on any host the
    // gate must propose at least one eviction. We size the request at u32::MAX
    // MiB / 2 so the gate's check_with always refuses regardless of the
    // host's actual RAM; the resident footprints are large enough that the
    // greedy suggestion picks them.
    write_loaded_json(
        tmp.path(),
        r#"[
            {"slug": "alpha", "footprint_mib": 2000000000, "last_used_unix_secs": 100},
            {"slug": "beta",  "footprint_mib": 1500000000, "last_used_unix_secs": 200},
            {"slug": "gamma", "footprint_mib": 1000000000, "last_used_unix_secs": 300}
        ]"#,
    );
    let (code, stdout, stderr) = run(
        &[
            "mem-check",
            "--requested",
            "delta",
            "--requested-mib",
            "2000000000",
        ],
        tmp.path(),
    );
    assert_eq!(code, 0, "stderr was: {stderr}");
    assert!(
        stdout.contains("to make room for delta"),
        "stdout was: {stdout}",
    );
    // The greedy strategy in suggest_unloads picks the largest first, so
    // `alpha` (2_000_000_000 MiB) must be in the list.
    assert!(stdout.contains("alpha"), "stdout was: {stdout}");
}

#[test]
fn mem_check_requested_mib_zero_errors_via_clap() {
    let tmp = TempDir::new().expect("tempdir");
    let (code, _stdout, stderr) = run(
        &["mem-check", "--requested", "foo", "--requested-mib", "0"],
        tmp.path(),
    );
    assert_eq!(code, 64, "stderr was: {stderr}");
    // Clap's range-validator emits "not in" / "invalid value" — either form
    // is acceptable; we just confirm the binary refused before doing real
    // work.
    let lower = stderr.to_lowercase();
    assert!(
        lower.contains("invalid value") || lower.contains("not in"),
        "stderr was: {stderr}",
    );
}

#[test]
fn mem_check_json_round_trips_required_keys() {
    let tmp = TempDir::new().expect("tempdir");
    let (code, stdout, stderr) = run(
        &[
            "--json",
            "mem-check",
            "--requested",
            "foo",
            "--requested-mib",
            "1",
        ],
        tmp.path(),
    );
    assert_eq!(code, 0, "stderr was: {stderr}");
    let value: serde_json::Value = serde_json::from_str(stdout.trim()).expect("parse json");
    assert!(value.get("available_mib").is_some(), "json was: {value}");
    assert_eq!(value["requested"], "foo");
    assert_eq!(value["requested_mib"], 1);
    assert!(value["suggested_evictions"].is_array(), "json was: {value}",);
}

#[test]
fn mem_check_json_no_flags_omits_requested_keys() {
    let tmp = TempDir::new().expect("tempdir");
    let (code, stdout, stderr) = run(&["--json", "mem-check"], tmp.path());
    assert_eq!(code, 0, "stderr was: {stderr}");
    let value: serde_json::Value = serde_json::from_str(stdout.trim()).expect("parse json");
    assert!(value.get("available_mib").is_some());
    // `serde(skip_serializing_if = "Option::is_none")` drops these.
    assert!(value.get("requested").is_none(), "json was: {value}");
    assert!(value.get("requested_mib").is_none(), "json was: {value}");
    assert_eq!(
        value["suggested_evictions"].as_array().map(Vec::len),
        Some(0),
    );
}

#[test]
fn mem_check_malformed_loaded_json_exits_1_with_clear_error() {
    let tmp = TempDir::new().expect("tempdir");
    write_loaded_json(tmp.path(), "{not json");
    let (code, _stdout, stderr) = run(
        &["mem-check", "--requested", "foo", "--requested-mib", "10"],
        tmp.path(),
    );
    assert_eq!(code, 1, "stderr was: {stderr}");
    assert!(stderr.contains("STRAT-E1001"), "stderr was: {stderr}");
    assert!(stderr.contains("loaded.json"), "stderr was: {stderr}");
}

#[test]
fn mem_check_requested_mib_without_requested_errors_via_clap() {
    let tmp = TempDir::new().expect("tempdir");
    let (code, _stdout, stderr) = run(&["mem-check", "--requested-mib", "100"], tmp.path());
    assert_eq!(code, 64, "stderr was: {stderr}");
    let lower = stderr.to_lowercase();
    assert!(
        lower.contains("--requested") || lower.contains("required"),
        "stderr was: {stderr}",
    );
}

#[test]
fn mem_check_requested_without_requested_mib_errors_via_clap() {
    let tmp = TempDir::new().expect("tempdir");
    let (code, _stdout, stderr) = run(&["mem-check", "--requested", "foo"], tmp.path());
    assert_eq!(code, 64, "stderr was: {stderr}");
    let lower = stderr.to_lowercase();
    assert!(
        lower.contains("--requested-mib") || lower.contains("required"),
        "stderr was: {stderr}",
    );
}

#[test]
fn mem_check_missing_loaded_file_treated_as_empty_list() {
    let tmp = TempDir::new().expect("tempdir");
    // Point --loaded-file at a path that doesn't exist. With a tiny request
    // and no resident set, the gate must say "fits".
    let missing = tmp.path().join("nowhere/loaded.json");
    let (code, stdout, stderr) = run(
        &[
            "mem-check",
            "--requested",
            "foo",
            "--requested-mib",
            "1",
            "--loaded-file",
            missing.to_str().expect("path utf8"),
        ],
        tmp.path(),
    );
    assert_eq!(code, 0, "stderr was: {stderr}");
    assert!(
        stdout.contains("fits without eviction"),
        "stdout was: {stdout}",
    );
}

#[test]
fn mem_check_explicit_loaded_file_is_honored() {
    // Write the file under a non-default name so we exercise --loaded-file
    // routing rather than the default <state>/loaded.json path.
    let tmp = TempDir::new().expect("tempdir");
    let custom = tmp.path().join("custom_loaded.json");
    fs::write(
        &custom,
        r#"[
            {"slug": "router", "footprint_mib": 100, "last_used_unix_secs": 0}
        ]"#,
    )
    .expect("write custom loaded file");
    let (code, stdout, stderr) = run(
        &[
            "mem-check",
            "--requested",
            "newcomer",
            "--requested-mib",
            "1",
            "--loaded-file",
            custom.to_str().expect("path utf8"),
        ],
        tmp.path(),
    );
    assert_eq!(code, 0, "stderr was: {stderr}");
    // Tiny request still fits; the resident set is just parsed.
    assert!(
        stdout.contains("fits without eviction"),
        "stdout was: {stdout}",
    );
}
