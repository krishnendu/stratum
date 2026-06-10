//! Integration tests for `stratum self-update --check`.
//!
//! Each test spawns the real built binary via `CARGO_BIN_EXE_stratum` against a
//! `--storage-root` tempdir and a hand-built local manifest fixture, so we
//! don't depend on network reachability. The HTTP fetch path is exercised
//! manually outside of CI per the task brief.

// Integration test binary: every fn here exists only for `cargo test`.
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    reason = "integration test helpers may panic on setup failures"
)]

use std::path::Path;
use std::process::Command;

use tempfile::TempDir;

const GOOD_SHA: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_stratum"))
}

/// Build a manifest entry as a JSON value. `min_supported_from` is optional.
fn release_entry(version: &str, platform_wire: &str, min_supported_from: Option<&str>) -> String {
    let min_block = min_supported_from.map_or_else(
        || r#""min_supported_from": null,"#.to_owned(),
        |min| {
            let parts = parse_version(min);
            format!(
                r#""min_supported_from": {{ "major": {}, "minor": {}, "patch": {}, "pre": null }},"#,
                parts.0, parts.1, parts.2,
            )
        },
    );
    let v = parse_version(version);
    format!(
        r#"{{
            "version": {{ "major": {}, "minor": {}, "patch": {}, "pre": null }},
            "released_at": {{ "secs_since_epoch": 1700000000, "nanos_since_epoch": 0 }},
            "binary": {{
                "url": "https://dl.stratum.dev/v{version}/stratum-{platform_wire}.tar.gz",
                "sha256": "{GOOD_SHA}",
                "bytes": 1024,
                "platform": "{platform_wire}"
            }},
            {min_block}
            "release_notes_url": "https://stratum.dev/releases/{version}"
        }}"#,
        v.0, v.1, v.2,
    )
}

fn parse_version(s: &str) -> (u16, u16, u16) {
    let mut it = s.split('.');
    let a: u16 = it.next().unwrap().parse().unwrap();
    let b: u16 = it.next().unwrap().parse().unwrap();
    let c: u16 = it.next().unwrap().parse().unwrap();
    (a, b, c)
}

/// Build a complete manifest JSON given an ascending list of versions, a
/// platform (in `update_manifest` wire form, e.g. `linux_x86_64`), and an
/// optional `min_supported_from` applied only to the *latest* entry.
fn build_manifest_json(
    versions: &[&str],
    platform_wire: &str,
    latest_min_from: Option<&str>,
) -> String {
    let history: Vec<String> = versions
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let min = if i + 1 == versions.len() {
                latest_min_from
            } else {
                None
            };
            release_entry(v, platform_wire, min)
        })
        .collect();
    let latest = release_entry(
        versions.last().expect("at least one version"),
        platform_wire,
        latest_min_from,
    );
    format!(
        r#"{{
            "schema_version": 1,
            "channel": "stable",
            "latest": {latest},
            "history": [{}]
        }}"#,
        history.join(",")
    )
}

/// CLI `--platform` and wire form are now the same after the explicit
/// per-variant rename in `update_manifest::PlatformTag` (fixes the
/// `mac_os_*` mismatch reported by users running `self-update --check`
/// against the release workflow's `stable.json`).
fn wire_platform_for(cli_platform: &str) -> &'static str {
    match cli_platform {
        "macos_aarch64" => "macos_aarch64",
        "macos_x86_64" => "macos_x86_64",
        "linux_aarch64" => "linux_aarch64",
        "linux_x86_64" => "linux_x86_64",
        "windows_x86_64" => "windows_x86_64",
        _ => panic!("unknown platform: {cli_platform}"),
    }
}

fn write_fixture(dir: &Path, name: &str, body: &str) -> std::path::PathBuf {
    let path = dir.join(name);
    std::fs::write(&path, body).expect("write fixture");
    path
}

fn run(args: &[&str], root: &Path) -> std::process::Output {
    let mut all: Vec<&str> = vec!["--storage-root", root.to_str().unwrap()];
    all.extend_from_slice(args);
    bin().args(&all).output().expect("spawn stratum")
}

#[test]
fn check_up_to_date_when_latest_equals_current() {
    let tmp = TempDir::new().unwrap();
    let manifest = build_manifest_json(&["1.0.0"], wire_platform_for("linux_x86_64"), None);
    let fixture = write_fixture(tmp.path(), "manifest.json", &manifest);
    let output = run(
        &[
            "self-update",
            "--check",
            "--manifest-file",
            fixture.to_str().unwrap(),
            "--current",
            "1.0.0",
            "--channel",
            "stable",
            "--platform",
            "linux_x86_64",
        ],
        tmp.path(),
    );
    assert!(
        output.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("up to date"), "stdout: {stdout}");
    assert!(stdout.contains("1.0.0"));
    assert!(stdout.contains("stable"));
}

#[test]
fn check_upgrade_when_newer_available() {
    let tmp = TempDir::new().unwrap();
    let manifest =
        build_manifest_json(&["1.4.7", "1.5.0"], wire_platform_for("linux_x86_64"), None);
    let fixture = write_fixture(tmp.path(), "manifest.json", &manifest);
    let output = run(
        &[
            "self-update",
            "--check",
            "--manifest-file",
            fixture.to_str().unwrap(),
            "--current",
            "1.4.7",
            "--channel",
            "stable",
            "--platform",
            "linux_x86_64",
        ],
        tmp.path(),
    );
    assert!(
        output.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("upgrade available"), "stdout: {stdout}");
    assert!(stdout.contains("1.4.7"));
    assert!(stdout.contains("1.5.0"));
    assert!(stdout.contains("artifact:"));
}

#[test]
fn check_blocked_when_below_min_supported() {
    let tmp = TempDir::new().unwrap();
    let manifest = build_manifest_json(
        &["1.0.0", "1.3.0", "1.5.0"],
        wire_platform_for("linux_x86_64"),
        Some("1.3.0"),
    );
    let fixture = write_fixture(tmp.path(), "manifest.json", &manifest);
    let output = run(
        &[
            "self-update",
            "--check",
            "--manifest-file",
            fixture.to_str().unwrap(),
            "--current",
            "1.0.0",
            "--channel",
            "stable",
            "--platform",
            "linux_x86_64",
        ],
        tmp.path(),
    );
    assert_eq!(output.status.code(), Some(64));
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("1.0.0"));
    assert!(stdout.contains("1.3.0"));
    assert!(stdout.contains("full reinstall required"));
}

#[test]
fn check_json_up_to_date_round_trips() {
    let tmp = TempDir::new().unwrap();
    let manifest = build_manifest_json(&["1.0.0"], wire_platform_for("linux_x86_64"), None);
    let fixture = write_fixture(tmp.path(), "manifest.json", &manifest);
    let output = run(
        &[
            "--json",
            "self-update",
            "--check",
            "--manifest-file",
            fixture.to_str().unwrap(),
            "--current",
            "1.0.0",
            "--channel",
            "stable",
            "--platform",
            "linux_x86_64",
        ],
        tmp.path(),
    );
    assert!(
        output.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    let v: serde_json::Value = serde_json::from_str(stdout.trim()).unwrap();
    assert_eq!(v["decision"], "UpToDate");
    assert_eq!(v["from"], "1.0.0");
    assert_eq!(v["channel"], "stable");
    assert_eq!(v["platform"], "linux_x86_64");
    assert!(v["artifact"].is_object());
}

#[test]
fn check_json_upgrade_round_trips() {
    let tmp = TempDir::new().unwrap();
    let manifest =
        build_manifest_json(&["1.4.7", "1.5.0"], wire_platform_for("linux_x86_64"), None);
    let fixture = write_fixture(tmp.path(), "manifest.json", &manifest);
    let output = run(
        &[
            "--json",
            "self-update",
            "--check",
            "--manifest-file",
            fixture.to_str().unwrap(),
            "--current",
            "1.4.7",
            "--channel",
            "stable",
            "--platform",
            "linux_x86_64",
        ],
        tmp.path(),
    );
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    let v: serde_json::Value = serde_json::from_str(stdout.trim()).unwrap();
    assert_eq!(v["decision"], "Upgrade");
    assert_eq!(v["from"], "1.4.7");
    assert_eq!(v["to"], "1.5.0");
    assert_eq!(v["artifact"]["bytes"], 1024);
    assert_eq!(v["artifact"]["sha256"], GOOD_SHA);
}

#[test]
fn check_json_blocked_round_trips() {
    let tmp = TempDir::new().unwrap();
    let manifest = build_manifest_json(
        &["1.0.0", "1.3.0", "1.5.0"],
        wire_platform_for("linux_x86_64"),
        Some("1.3.0"),
    );
    let fixture = write_fixture(tmp.path(), "manifest.json", &manifest);
    let output = run(
        &[
            "--json",
            "self-update",
            "--check",
            "--manifest-file",
            fixture.to_str().unwrap(),
            "--current",
            "1.0.0",
            "--channel",
            "stable",
            "--platform",
            "linux_x86_64",
        ],
        tmp.path(),
    );
    assert_eq!(output.status.code(), Some(64));
    let stdout = String::from_utf8(output.stdout).unwrap();
    let v: serde_json::Value = serde_json::from_str(stdout.trim()).unwrap();
    assert_eq!(v["decision"], "BlockedSchemaTooOld");
    assert_eq!(v["from"], "1.0.0");
    assert_eq!(v["to"], "1.3.0");
}

#[test]
fn check_rejects_nonexistent_manifest_file() {
    let tmp = TempDir::new().unwrap();
    let missing = tmp.path().join("no-such.json");
    let output = run(
        &[
            "self-update",
            "--check",
            "--manifest-file",
            missing.to_str().unwrap(),
            "--current",
            "1.0.0",
            "--platform",
            "linux_x86_64",
        ],
        tmp.path(),
    );
    assert_eq!(output.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("STRAT-E1001"));
}

#[test]
fn check_rejects_malformed_manifest_file() {
    let tmp = TempDir::new().unwrap();
    let fixture = write_fixture(tmp.path(), "manifest.json", "{not json");
    let output = run(
        &[
            "self-update",
            "--check",
            "--manifest-file",
            fixture.to_str().unwrap(),
            "--current",
            "1.0.0",
            "--platform",
            "linux_x86_64",
        ],
        tmp.path(),
    );
    assert_eq!(output.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("STRAT-E1001"));
}

#[test]
fn check_rejects_newer_schema_version() {
    let tmp = TempDir::new().unwrap();
    let manifest = build_manifest_json(&["1.0.0"], wire_platform_for("linux_x86_64"), None);
    // Bump schema_version to 999 to simulate an unknown-newer manifest.
    let bumped = manifest.replace("\"schema_version\": 1", "\"schema_version\": 999");
    let fixture = write_fixture(tmp.path(), "manifest.json", &bumped);
    let output = run(
        &[
            "self-update",
            "--check",
            "--manifest-file",
            fixture.to_str().unwrap(),
            "--current",
            "1.0.0",
            "--platform",
            "linux_x86_64",
        ],
        tmp.path(),
    );
    assert_eq!(output.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("STRAT-E1001"));
    assert!(stderr.contains("newer than supported"));
}

#[test]
fn check_current_override_takes_precedence() {
    let tmp = TempDir::new().unwrap();
    let manifest =
        build_manifest_json(&["1.4.7", "1.5.0"], wire_platform_for("linux_x86_64"), None);
    let fixture = write_fixture(tmp.path(), "manifest.json", &manifest);
    // Without --current we'd pick up CARGO_PKG_VERSION (0.0.0) and report
    // Upgrade; passing --current 1.5.0 must report UpToDate.
    let output = run(
        &[
            "self-update",
            "--check",
            "--manifest-file",
            fixture.to_str().unwrap(),
            "--current",
            "1.5.0",
            "--platform",
            "linux_x86_64",
        ],
        tmp.path(),
    );
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("up to date"), "stdout: {stdout}");
}

#[test]
fn check_platform_override_changes_artifact_match() {
    let tmp = TempDir::new().unwrap();
    // Manifest publishes a linux_x86_64 binary. Asking for windows_x86_64
    // should produce JSON with `artifact: null` even on an Upgrade.
    let manifest =
        build_manifest_json(&["1.4.7", "1.5.0"], wire_platform_for("linux_x86_64"), None);
    let fixture = write_fixture(tmp.path(), "manifest.json", &manifest);
    let output = run(
        &[
            "--json",
            "self-update",
            "--check",
            "--manifest-file",
            fixture.to_str().unwrap(),
            "--current",
            "1.4.7",
            "--platform",
            "windows_x86_64",
        ],
        tmp.path(),
    );
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    let v: serde_json::Value = serde_json::from_str(stdout.trim()).unwrap();
    assert_eq!(v["decision"], "Upgrade");
    assert_eq!(v["platform"], "windows_x86_64");
    // `artifact` is `skip_serializing_if = Option::is_none`, so when the
    // platform doesn't match it should be absent from the JSON entirely.
    assert!(v.get("artifact").is_none() || v["artifact"].is_null());
}

#[test]
fn check_rejects_mutually_exclusive_manifest_sources() {
    let tmp = TempDir::new().unwrap();
    let fixture = write_fixture(
        tmp.path(),
        "manifest.json",
        &build_manifest_json(&["1.0.0"], wire_platform_for("linux_x86_64"), None),
    );
    let output = run(
        &[
            "self-update",
            "--check",
            "--manifest-file",
            fixture.to_str().unwrap(),
            "--manifest-url",
            "https://example.com/stable.json",
            "--current",
            "1.0.0",
            "--platform",
            "linux_x86_64",
        ],
        tmp.path(),
    );
    // Clap's `conflicts_with` returns exit 64 (per the CLI's parse-error wrapper).
    assert_eq!(output.status.code(), Some(64));
}

#[test]
fn check_default_current_falls_back_to_cargo_pkg_version() {
    // A manifest whose latest matches the workspace's `CARGO_PKG_VERSION`
    // yields UpToDate without `--current`.
    let pkg_version = env!("CARGO_PKG_VERSION");
    let tmp = TempDir::new().unwrap();
    let manifest = build_manifest_json(&[pkg_version], wire_platform_for("linux_x86_64"), None);
    let fixture = write_fixture(tmp.path(), "manifest.json", &manifest);
    let output = run(
        &[
            "self-update",
            "--check",
            "--manifest-file",
            fixture.to_str().unwrap(),
            "--platform",
            "linux_x86_64",
        ],
        tmp.path(),
    );
    assert!(
        output.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("up to date"), "stdout: {stdout}");
    assert!(stdout.contains(pkg_version));
}
