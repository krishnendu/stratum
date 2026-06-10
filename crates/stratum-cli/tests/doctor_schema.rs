//! `stratum doctor --json` schema invariant test.
//!
//! xtask-check-error-codes: ignore-file (test fixture references synthetic STRAT-E9999).
//!
//! Pins the on-the-wire shape of the doctor report against
//! `docs/schemas/doctor.v1.json`. The fixture is the schema file itself,
//! loaded at compile time via `include_str!`, so a drift in either the schema
//! or the runtime payload surfaces here as a named CI failure (see the
//! `doctor-schema-check` workflow job and `plan/29-error-taxonomy-and-logging.md`
//! §7-8).
//!
//! The test deliberately walks the JSON by hand instead of pulling in a full
//! schema validator: we want zero new dependencies on this critical path, and
//! the schema is small enough that "required field present + enum value
//! matches + regex pattern holds" is checkable in ~150 lines.

// Integration test binary: every fn here exists only for `cargo test`. Test
// helpers panic on setup failures by design; clippy's `expect_used` /
// `unwrap_used` denials are scoped to non-test code.
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "integration test helpers may panic on setup failures"
)]

use std::process::Command;

use tempfile::TempDir;

/// Embedded copy of `docs/schemas/doctor.v1.json`. Used by the
/// "schema is valid JSON" test below; the structural assertions are
/// hard-coded against the schema's contract (changing the schema in a
/// breaking way means updating both files).
const DOCTOR_V1_SCHEMA: &str = include_str!("../../../docs/schemas/doctor.v1.json");

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_stratum"))
}

/// Spawn `stratum --json doctor` against a fresh `--storage-root` and return
/// the parsed JSON value of stdout.
fn doctor_json(root: &std::path::Path) -> serde_json::Value {
    let output = bin()
        .args([
            "--storage-root",
            root.to_str().expect("temp path is utf-8"),
            "--json",
            "doctor",
        ])
        .output()
        .expect("spawn stratum");
    assert!(
        output.status.success(),
        "stratum exited non-zero: stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("stdout is utf-8");
    serde_json::from_str(stdout.trim()).expect("stratum doctor --json emits valid JSON")
}

#[test]
fn schema_file_is_valid_json() {
    let v: serde_json::Value =
        serde_json::from_str(DOCTOR_V1_SCHEMA).expect("docs/schemas/doctor.v1.json parses");
    // Accept either of the two common draft-07 $schema URLs.
    let schema_url = v["$schema"].as_str().unwrap_or("");
    assert!(
        schema_url.contains("draft-07") || schema_url.contains("draft/draft-07"),
        "$schema must reference draft-07, got: {schema_url}"
    );
    assert_eq!(v["title"].as_str(), Some("Stratum doctor report (v1)"));
    // Sanity-check the top-level required list matches what we assert below.
    let required = v["required"].as_array().expect("schema has `required`");
    for field in [
        "schema_version",
        "stratum_version",
        "tier",
        "probe",
        "gpu_accel",
        "sandbox",
        "installed",
        "issues",
    ] {
        assert!(
            required.iter().any(|x| x.as_str() == Some(field)),
            "schema `required` missing `{field}`"
        );
    }
}

#[test]
fn doctor_json_has_every_required_top_level_field() {
    let tmp = TempDir::new().expect("tempdir");
    let v = doctor_json(tmp.path());
    let obj = v.as_object().expect("top level is an object");

    for field in [
        "schema_version",
        "stratum_version",
        "tier",
        "probe",
        "gpu_accel",
        "sandbox",
        "installed",
        "issues",
    ] {
        assert!(
            obj.contains_key(field),
            "missing required top-level field `{field}`; got keys: {:?}",
            obj.keys().collect::<Vec<_>>()
        );
    }
}

#[test]
fn doctor_json_top_level_types_and_enums() {
    let tmp = TempDir::new().expect("tempdir");
    let v = doctor_json(tmp.path());

    assert_eq!(
        v["schema_version"].as_u64(),
        Some(1),
        "schema_version must be the integer 1"
    );
    assert!(
        v["stratum_version"].as_str().is_some_and(|s| !s.is_empty()),
        "stratum_version must be a non-empty string"
    );
    assert!(
        matches!(v["tier"].as_str(), Some("low" | "medium" | "high")),
        "tier must be one of low/medium/high, got: {:?}",
        v["tier"]
    );
    assert!(
        matches!(
            v["gpu_accel"].as_str(),
            Some("metal" | "cuda" | "vulkan" | "cpu")
        ),
        "gpu_accel must be one of metal/cuda/vulkan/cpu, got: {:?}",
        v["gpu_accel"]
    );
    assert!(v["installed"].is_boolean(), "installed must be a boolean");
    assert!(v["issues"].is_array(), "issues must be an array");
}

#[test]
fn doctor_json_probe_subschema() {
    let tmp = TempDir::new().expect("tempdir");
    let v = doctor_json(tmp.path());
    let probe = v["probe"].as_object().expect("probe must be an object");

    for field in [
        "ram_total_mib",
        "ram_available_mib",
        "cpu_arch",
        "cpu_features",
        "cpu_cores",
        "gpu",
        "os",
    ] {
        assert!(
            probe.contains_key(field),
            "probe missing required field `{field}`; got keys: {:?}",
            probe.keys().collect::<Vec<_>>()
        );
    }

    assert!(probe["ram_total_mib"].as_u64().is_some());
    assert!(probe["ram_available_mib"].as_u64().is_some());
    assert!(probe["cpu_cores"].as_u64().is_some());
    assert!(probe["cpu_arch"].as_str().is_some_and(|s| !s.is_empty()));
    assert!(probe["os"].as_str().is_some_and(|s| !s.is_empty()));
    assert!(probe["cpu_features"].is_array());
    assert!(
        matches!(
            probe["gpu"].as_str(),
            Some("metal" | "cuda" | "vulkan" | "cpu")
        ),
        "probe.gpu must be one of metal/cuda/vulkan/cpu, got: {:?}",
        probe["gpu"]
    );
}

#[test]
fn doctor_json_sandbox_subschema() {
    let tmp = TempDir::new().expect("tempdir");
    let v = doctor_json(tmp.path());
    let sandbox = v["sandbox"].as_object().expect("sandbox must be an object");
    let available = sandbox["available"]
        .as_array()
        .expect("sandbox.available must be an array");
    assert!(
        !available.is_empty(),
        "sandbox.available must include at least `passthrough`"
    );
    for entry in available {
        let s = entry
            .as_str()
            .expect("sandbox.available entries are strings");
        assert!(
            matches!(s, "bwrap" | "sandbox_exec" | "windows_job" | "passthrough"),
            "sandbox.available entry `{s}` is not in the v1 enum"
        );
    }
}

#[test]
fn doctor_json_issue_codes_match_regex() {
    // Before init, the doctor must surface an `installed=false` issue,
    // exercising the full issues-array path.
    let tmp = TempDir::new().expect("tempdir");
    let v = doctor_json(tmp.path());
    let issues = v["issues"].as_array().expect("issues is an array");
    assert!(
        !issues.is_empty(),
        "issues must not be empty on a fresh storage root"
    );

    for issue in issues {
        let obj = issue.as_object().expect("each issue is an object");
        for field in ["code", "level", "message"] {
            assert!(
                obj.contains_key(field),
                "issue missing field `{field}`: {obj:?}"
            );
        }
        let code = obj["code"].as_str().expect("issue.code is a string");
        assert!(
            is_strat_error_code(code),
            "issue.code `{code}` does not match ^STRAT-E\\d{{4}}$"
        );
        let level = obj["level"].as_str().expect("issue.level is a string");
        assert!(
            matches!(level, "info" | "warn" | "error"),
            "issue.level `{level}` is not info/warn/error"
        );
        assert!(
            obj["message"].as_str().is_some_and(|s| !s.is_empty()),
            "issue.message must be a non-empty string"
        );
    }
}

/// Pure-Rust regex check for `^STRAT-E\d{4}$` without pulling in the `regex`
/// crate. Returns `true` iff `s` is literally `STRAT-E` followed by exactly
/// four ASCII digits.
fn is_strat_error_code(s: &str) -> bool {
    let Some(rest) = s.strip_prefix("STRAT-E") else {
        return false;
    };
    rest.len() == 4 && rest.bytes().all(|b| b.is_ascii_digit())
}

#[test]
fn strat_error_code_matcher_accepts_known_codes() {
    assert!(is_strat_error_code("STRAT-E1001"));
    assert!(is_strat_error_code("STRAT-E2003"));
    assert!(is_strat_error_code("STRAT-E9999"));
}

#[test]
fn strat_error_code_matcher_rejects_garbage() {
    assert!(!is_strat_error_code(""));
    assert!(!is_strat_error_code("STRAT-E"));
    assert!(!is_strat_error_code("STRAT-E12"));
    assert!(!is_strat_error_code("STRAT-E12345"));
    assert!(!is_strat_error_code("STRAT-W2001"));
    assert!(!is_strat_error_code("strat-e2003"));
    assert!(!is_strat_error_code("STRAT-Eabcd"));
}

#[test]
fn doctor_strict_json_exits_zero_on_real_report() {
    // The runtime `DoctorReport` is always shape-valid; `--strict --json`
    // must therefore exit 0 against a freshly-rolled storage root.
    let tmp = TempDir::new().expect("tempdir");
    let output = bin()
        .args([
            "--storage-root",
            tmp.path().to_str().expect("temp path utf-8"),
            "--json",
            "doctor",
            "--strict",
        ])
        .output()
        .expect("spawn stratum");
    assert!(
        output.status.success(),
        "expected exit 0, got: stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("stdout utf-8");
    let v: serde_json::Value = serde_json::from_str(stdout.trim()).expect("json");
    assert_eq!(v["schema_version"], 1);
}

#[test]
fn doctor_strict_without_json_is_silent_noop() {
    // `--strict` only activates under `--json`. Without `--json` the flag
    // is accepted but does not affect output or exit code.
    let tmp = TempDir::new().expect("tempdir");
    let output = bin()
        .args([
            "--storage-root",
            tmp.path().to_str().expect("temp path utf-8"),
            "doctor",
            "--strict",
        ])
        .output()
        .expect("spawn stratum");
    assert!(
        output.status.success(),
        "expected exit 0, got: stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("stdout utf-8");
    assert!(stdout.contains("stratum "), "got: {stdout}");
}

#[test]
fn doctor_json_after_init_drops_install_issue() {
    // After `stratum init` writes `installed.toml`, the
    // "no installed.toml" info-level issue disappears. Confirms the
    // issue-array path is exercised on both sides of the boolean.
    let tmp = TempDir::new().expect("tempdir");
    let init = bin()
        .args([
            "--storage-root",
            tmp.path().to_str().expect("temp path utf-8"),
            "init",
        ])
        .output()
        .expect("spawn stratum init");
    assert!(
        init.status.success(),
        "init failed: {}",
        String::from_utf8_lossy(&init.stderr)
    );
    let v = doctor_json(tmp.path());
    assert_eq!(v["installed"], serde_json::Value::Bool(true));
    let issues = v["issues"].as_array().expect("issues array");
    for issue in issues {
        let code = issue["code"].as_str().unwrap_or("");
        assert_ne!(
            code, "STRAT-E2003",
            "install issue must not appear after init"
        );
    }
}
