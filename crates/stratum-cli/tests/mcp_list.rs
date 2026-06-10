//! End-to-end tests for `stratum mcp list [--json]`.
//!
//! These spawn the built binary via `CARGO_BIN_EXE_stratum` against a
//! `TempDir`-rooted `--storage-root` and exercise the four documented
//! shapes of the subcommand: missing-file prose, missing-file JSON, a
//! valid two-stdio + one-http fixture, and a malformed file.

// Integration test binary: every fn here exists only for `cargo test`. Test
// helpers panic on setup failures by design; clippy's `expect_used` /
// `unwrap_used` / `panic` denials are scoped to non-test code.
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    reason = "integration test helpers may panic on setup failures"
)]

use std::collections::BTreeMap;
use std::path::Path;
use std::process::Command;

use stratum_runtime::{McpServerConfig, McpServerSet, McpTransport};
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

fn write_fixture(root: &Path, body: &str) {
    let state = root.join("state");
    std::fs::create_dir_all(&state).expect("mkdir state");
    std::fs::write(state.join("mcp.toml"), body).expect("write fixture");
}

/// Build a [`McpServerSet`] with two stdio + one http entry through the
/// runtime types, then serialize it to TOML so the fixture matches the
/// runtime's serde shape exactly.
fn build_three_server_toml() -> String {
    let mut set = McpServerSet::new();
    set.insert(McpServerConfig {
        name: "filesystem".to_owned(),
        transport: McpTransport::Stdio {
            command: "fs".to_owned(),
            args: vec!["--root".to_owned(), "/tmp".to_owned()],
            env: BTreeMap::new(),
        },
        allow: vec!["fs.read".to_owned(), "fs.write".to_owned()],
        deny: vec!["net.fetch".to_owned()],
    });
    set.insert(McpServerConfig {
        name: "rag".to_owned(),
        transport: McpTransport::Stdio {
            command: "rag".to_owned(),
            args: vec![],
            env: BTreeMap::new(),
        },
        allow: vec!["rag.search".to_owned()],
        deny: vec![],
    });
    set.insert(McpServerConfig {
        name: "web".to_owned(),
        transport: McpTransport::Http {
            url: "https://example.com/mcp".to_owned(),
            bearer_token_uri: None,
        },
        allow: vec!["tools.fetch".to_owned()],
        deny: vec![],
    });
    toml_edit::ser::to_string(&set).expect("serialize fixture to TOML")
}

#[test]
fn list_missing_file_prose_is_friendly_exit_zero() {
    let tmp = TempDir::new().unwrap();
    let out = run(tmp.path(), &["mcp", "list"]);
    assert!(
        out.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.to_lowercase().contains("no mcp"),
        "expected 'no MCP' marker, got {stdout:?}"
    );
}

#[test]
fn list_missing_file_json_is_empty_object_exit_zero() {
    let tmp = TempDir::new().unwrap();
    let out = run(tmp.path(), &["mcp", "list", "--json"]);
    assert!(
        out.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    // `McpServerSet` is `#[serde(transparent)]` over `BTreeMap`, so an
    // empty set serializes to `{}`.
    let trimmed = stdout.trim();
    assert_eq!(trimmed, "{}", "expected '{{}}', got {trimmed:?}");
}

#[test]
fn list_prose_renders_one_row_per_server() {
    let tmp = TempDir::new().unwrap();
    write_fixture(tmp.path(), &build_three_server_toml());
    let out = run(tmp.path(), &["mcp", "list"]);
    assert!(
        out.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);

    // Header + separator + 3 data rows = at least 5 non-empty lines.
    let data_rows: Vec<&str> = stdout
        .lines()
        .filter(|l| !l.is_empty())
        .filter(|l| {
            // Drop the header and dashed separator.
            !l.starts_with("name") && !l.trim_start().starts_with('-')
        })
        .collect();
    assert_eq!(
        data_rows.len(),
        3,
        "expected 3 data rows, got {data_rows:?}"
    );

    // Each configured server shows up exactly once.
    assert!(stdout.contains("filesystem"));
    assert!(stdout.contains("rag"));
    assert!(stdout.contains("web"));
    // Transport shapes show through the renderer.
    assert!(stdout.contains("stdio (cmd=fs)"));
    assert!(stdout.contains("stdio (cmd=rag)"));
    assert!(stdout.contains("http (https://example.com/mcp)"));
    // The allow / deny columns are populated.
    assert!(stdout.contains("fs.read, fs.write"));
    assert!(stdout.contains("net.fetch"));
    assert!(stdout.contains("tools.fetch"));
}

#[test]
fn list_json_roundtrips_through_serde_json_value() {
    let tmp = TempDir::new().unwrap();
    write_fixture(tmp.path(), &build_three_server_toml());
    let out = run(tmp.path(), &["mcp", "list", "--json"]);
    assert!(
        out.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    let parsed: serde_json::Value =
        serde_json::from_str(&stdout).expect("stdout must be valid JSON");
    let map = parsed
        .as_object()
        .expect("top-level shape is a JSON object (transparent BTreeMap)");
    assert_eq!(map.len(), 3, "expected 3 entries, got {map:?}");
    for key in ["filesystem", "rag", "web"] {
        assert!(map.contains_key(key), "expected key {key:?} in {map:?}");
    }
    // The filesystem entry preserves its transport discriminator + command.
    let fs = &map["filesystem"];
    assert_eq!(fs["transport"], serde_json::Value::String("stdio".into()));
    assert_eq!(fs["command"], serde_json::Value::String("fs".into()));
}

#[test]
fn list_malformed_toml_reports_strat_e1001_exit_one() {
    let tmp = TempDir::new().unwrap();
    // Deliberately invalid TOML — unclosed inline table, no `=`.
    write_fixture(tmp.path(), "this is not toml = {{{\n");
    let out = run(tmp.path(), &["mcp", "list"]);
    assert!(!out.status.success(), "expected non-zero exit");
    assert_eq!(out.status.code(), Some(1), "expected exit 1");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("STRAT-E1001"),
        "expected STRAT-E1001 marker, got {stderr:?}"
    );
}
