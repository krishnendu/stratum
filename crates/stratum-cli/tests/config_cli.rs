//! End-to-end tests for `stratum config get|set|list|unset`.
//!
//! These spawn the built binary via `CARGO_BIN_EXE_stratum` against a
//! `TempDir`-rooted `--storage-root` and exercise the documented
//! shapes of the four subcommands plus the typed-value coercions and
//! the missing-key / bad-type error paths.

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

fn assert_ok(out: &std::process::Output) {
    assert!(
        out.status.success(),
        "expected success; stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
}

fn assert_e1001(out: &std::process::Output) {
    assert!(!out.status.success(), "expected non-zero exit");
    assert_eq!(out.status.code(), Some(1), "expected exit 1");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("STRAT-E1001"),
        "expected STRAT-E1001, got {stderr:?}"
    );
}

fn stdout_of(out: &std::process::Output) -> String {
    String::from_utf8_lossy(&out.stdout).into_owned()
}

#[test]
fn list_missing_file_prose_exit_zero_empty() {
    let tmp = TempDir::new().unwrap();
    let out = run(tmp.path(), &["config", "list"]);
    assert_ok(&out);
    let stdout = stdout_of(&out);
    assert!(
        stdout.trim().is_empty(),
        "expected empty stdout, got {stdout:?}"
    );
}

#[test]
fn list_missing_file_json_exit_zero_empty_object() {
    let tmp = TempDir::new().unwrap();
    let out = run(tmp.path(), &["config", "list", "--json"]);
    assert_ok(&out);
    let stdout = stdout_of(&out);
    let parsed: serde_json::Value =
        serde_json::from_str(&stdout).expect("stdout must be valid JSON");
    let map = parsed
        .as_object()
        .expect("top-level shape is a JSON object");
    assert!(map.is_empty(), "expected empty object, got {map:?}");
}

#[test]
fn set_then_get_string_roundtrip() {
    let tmp = TempDir::new().unwrap();
    let set = run(
        tmp.path(),
        &["config", "set", "chat.default_model", "qwen-0.5b"],
    );
    assert_ok(&set);
    let get = run(tmp.path(), &["config", "get", "chat.default_model"]);
    assert_ok(&get);
    let stdout = stdout_of(&get);
    assert_eq!(stdout.trim(), "qwen-0.5b");
}

#[test]
fn set_int_then_get_returns_number_not_string() {
    let tmp = TempDir::new().unwrap();
    let set = run(
        tmp.path(),
        &["config", "set", "foo.bar", "42", "--type", "int"],
    );
    assert_ok(&set);

    // Prose: bare integer literal.
    let get = run(tmp.path(), &["config", "get", "foo.bar"]);
    assert_ok(&get);
    assert_eq!(stdout_of(&get).trim(), "42");

    // JSON: real JSON number.
    let get_json = run(tmp.path(), &["config", "get", "foo.bar", "--json"]);
    assert_ok(&get_json);
    let parsed: serde_json::Value =
        serde_json::from_str(&stdout_of(&get_json)).expect("valid JSON");
    let n = parsed.as_i64().expect("JSON number i64");
    assert_eq!(n, 42);
}

#[test]
fn set_bool_then_get_json_is_real_bool() {
    let tmp = TempDir::new().unwrap();
    let set = run(
        tmp.path(),
        &["config", "set", "flag", "true", "--type", "bool"],
    );
    assert_ok(&set);
    let get = run(tmp.path(), &["config", "get", "flag", "--json"]);
    assert_ok(&get);
    let parsed: serde_json::Value = serde_json::from_str(&stdout_of(&get)).expect("valid JSON");
    assert_eq!(parsed, serde_json::Value::Bool(true));
}

#[test]
fn set_float_then_get_json_is_number() {
    let tmp = TempDir::new().unwrap();
    let set = run(
        tmp.path(),
        &["config", "set", "ratio", "2.5", "--type", "float"],
    );
    assert_ok(&set);
    let get = run(tmp.path(), &["config", "get", "ratio", "--json"]);
    assert_ok(&get);
    let parsed: serde_json::Value = serde_json::from_str(&stdout_of(&get)).expect("valid JSON");
    let n = parsed.as_f64().expect("JSON number f64");
    assert!((n - 2.5_f64).abs() < 1e-9, "expected ~2.5, got {n}");
}

#[test]
fn get_missing_key_exits_one_with_strat_e1001() {
    let tmp = TempDir::new().unwrap();
    let out = run(tmp.path(), &["config", "get", "missing"]);
    assert_e1001(&out);
}

#[test]
fn unset_after_set_removes_key_then_get_errors_missing() {
    let tmp = TempDir::new().unwrap();
    let set = run(
        tmp.path(),
        &["config", "set", "chat.default_model", "qwen-0.5b"],
    );
    assert_ok(&set);

    let unset = run(tmp.path(), &["config", "unset", "chat.default_model"]);
    assert_ok(&unset);

    let get = run(tmp.path(), &["config", "get", "chat.default_model"]);
    assert_e1001(&get);
    let stderr = String::from_utf8_lossy(&get.stderr);
    assert!(
        stderr.to_lowercase().contains("missing"),
        "expected 'missing' in stderr, got {stderr:?}"
    );
}

#[test]
fn unset_missing_key_exits_one() {
    let tmp = TempDir::new().unwrap();
    let out = run(tmp.path(), &["config", "unset", "never.set"]);
    assert_e1001(&out);
}

#[test]
fn list_json_roundtrips_through_serde_json_value() {
    let tmp = TempDir::new().unwrap();
    assert_ok(&run(
        tmp.path(),
        &["config", "set", "chat.default_model", "qwen-0.5b"],
    ));
    assert_ok(&run(
        tmp.path(),
        &[
            "config",
            "set",
            "serve.default_port",
            "47474",
            "--type",
            "int",
        ],
    ));
    assert_ok(&run(
        tmp.path(),
        &[
            "config",
            "set",
            "telemetry.enabled",
            "false",
            "--type",
            "bool",
        ],
    ));

    let out = run(tmp.path(), &["config", "list", "--json"]);
    assert_ok(&out);
    let stdout = stdout_of(&out);
    let parsed: serde_json::Value =
        serde_json::from_str(&stdout).expect("stdout must be valid JSON");
    let map = parsed
        .as_object()
        .expect("top-level shape is a JSON object");
    assert_eq!(
        map.get("chat.default_model"),
        Some(&serde_json::Value::String("qwen-0.5b".into())),
        "got {map:?}"
    );
    assert_eq!(
        map.get("serve.default_port")
            .and_then(serde_json::Value::as_i64),
        Some(47474),
        "got {map:?}"
    );
    assert_eq!(
        map.get("telemetry.enabled"),
        Some(&serde_json::Value::Bool(false)),
        "got {map:?}"
    );
}

#[test]
fn set_bool_with_non_bool_value_exits_one() {
    let tmp = TempDir::new().unwrap();
    let out = run(
        tmp.path(),
        &["config", "set", "flag", "banana", "--type", "bool"],
    );
    assert_e1001(&out);
}

#[test]
fn set_int_with_non_int_value_exits_one() {
    let tmp = TempDir::new().unwrap();
    let out = run(
        tmp.path(),
        &["config", "set", "n", "not-a-number", "--type", "int"],
    );
    assert_e1001(&out);
}

#[test]
fn deep_nested_key_shows_up_in_list() {
    let tmp = TempDir::new().unwrap();
    let set = run(
        tmp.path(),
        &["config", "set", "a.b.c.d", "1", "--type", "int"],
    );
    assert_ok(&set);

    // Prose list contains the dotted key.
    let list = run(tmp.path(), &["config", "list"]);
    assert_ok(&list);
    let stdout = stdout_of(&list);
    assert!(
        stdout.contains("a.b.c.d"),
        "expected 'a.b.c.d' in prose list, got {stdout:?}"
    );

    // get returns the integer.
    let get = run(tmp.path(), &["config", "get", "a.b.c.d", "--json"]);
    assert_ok(&get);
    let parsed: serde_json::Value = serde_json::from_str(&stdout_of(&get)).expect("valid JSON");
    assert_eq!(parsed.as_i64(), Some(1));

    // JSON list carries the flattened key.
    let list_json = run(tmp.path(), &["config", "list", "--json"]);
    assert_ok(&list_json);
    let parsed: serde_json::Value =
        serde_json::from_str(&stdout_of(&list_json)).expect("valid JSON");
    let map = parsed.as_object().expect("object");
    assert_eq!(
        map.get("a.b.c.d").and_then(serde_json::Value::as_i64),
        Some(1),
        "got {map:?}"
    );
}

#[test]
fn set_without_type_defaults_to_string() {
    let tmp = TempDir::new().unwrap();
    let set = run(tmp.path(), &["config", "set", "key", "value"]);
    assert_ok(&set);

    let get = run(tmp.path(), &["config", "get", "key", "--json"]);
    assert_ok(&get);
    let parsed: serde_json::Value = serde_json::from_str(&stdout_of(&get)).expect("valid JSON");
    assert_eq!(parsed, serde_json::Value::String("value".into()));
}

#[test]
fn set_then_get_preserves_value_through_separate_processes() {
    // Sanity check that values genuinely round-trip via the on-disk
    // file (not just via in-process state): we exec twice.
    let tmp = TempDir::new().unwrap();
    assert_ok(&run(tmp.path(), &["config", "set", "a", "first"]));
    assert_ok(&run(tmp.path(), &["config", "set", "b", "second"]));
    let list = run(tmp.path(), &["config", "list"]);
    assert_ok(&list);
    let stdout = stdout_of(&list);
    assert!(stdout.contains("a = first"), "got {stdout:?}");
    assert!(stdout.contains("b = second"), "got {stdout:?}");
}
