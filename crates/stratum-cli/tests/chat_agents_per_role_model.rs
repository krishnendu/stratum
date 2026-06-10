//! End-to-end tests for `stratum chat --agents-dir` per-role model
//! resolution.
//!
//! Runtime PR #144 wired a [`stratum_runtime::ProviderResolver`] hook into
//! [`stratum_runtime::AgentRegistryLoader`]; this PR threads a real
//! catalog-backed resolver into the CLI. These tests exercise that flow:
//!
//! * An agent TOML with `model = "echo"` still resolves to the default
//!   [`EchoProvider`] regardless of catalog state — the floor behaviour.
//! * An agent TOML with `model = "qwen-0.5b"` against an EMPTY catalog
//!   surfaces the slug as a per-file load error; the echo-backed role
//!   still loads and the chat still runs (since at least one role is
//!   registered).
//! * Same agent TOML against a catalog that DOES list `qwen-0.5b` errors
//!   on the feature-gate ("requires --features provider-llama-cpp")
//!   instead of "unknown slug" — proving the catalog lookup succeeded
//!   and only the backend wiring is missing in the default build.
//!
//! Default-feature CI runs without `provider-llama-cpp`, so these tests
//! only assert the no-feature path. The feature-on path is covered by
//! the `chat_gguf.rs` smoke matrix.

// Integration test binary: every fn here exists only for `cargo test`. The
// helpers below panic on setup failures by design; clippy's `expect_used` /
// `unwrap_used` / `panic` denials only apply to non-test code.
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    reason = "integration test helpers may panic on setup failures"
)]

use std::path::{Path, PathBuf};
use std::process::Command;

use tempfile::TempDir;

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_stratum"))
}

fn agents_dir(root: &Path) -> PathBuf {
    let dir = root.join("state").join("agents");
    std::fs::create_dir_all(&dir).expect("mkdir state/agents");
    dir
}

fn state_dir(root: &Path) -> PathBuf {
    let dir = root.join("state");
    std::fs::create_dir_all(&dir).expect("mkdir state");
    dir
}

/// Minimal `AgentDef` body with a configurable `model` slug — the loader
/// hands `slug` straight to the `ProviderResolver`, so this is the field
/// that drives per-role provider selection.
fn agent_body(role: &str, model: &str) -> String {
    format!(
        r#"
schema_version = 1
name = "{role}-agent"
description = "smoke agent for {role}"
roles = ["{role}"]
model = "{model}"
tools = []
sandbox = "passthrough"
"#
    )
}

fn write_agent(dir: &Path, name: &str, role: &str, model: &str) {
    let path = dir.join(format!("{name}.toml"));
    std::fs::write(&path, agent_body(role, model)).expect("write agent toml");
}

/// Write a `models.json` carrying a single `qwen-0.5b` entry pointing at
/// a fake HTTPS URL. The catalog itself only needs the URL to parse; no
/// network I/O happens in the no-feature path because the resolver
/// short-circuits before touching `ModelInstaller`.
#[cfg_attr(
    feature = "provider-llama-cpp",
    allow(
        dead_code,
        reason = "helper is only consumed by no-feature tests"
    )
)]
fn write_catalog_with_qwen(state: &Path) {
    let catalog = r#"{
      "schema_version": 1,
      "entries": {
        "qwen-0.5b": {
          "slug": "qwen-0.5b",
          "family": "qwen",
          "display_name": "Qwen 0.5B",
          "tier": "low",
          "task": ["chat"],
          "size_mib": 400,
          "quantization": "Q4_K_M",
          "artifact": {
            "url": "https://example.invalid/qwen-0.5b.gguf",
            "sha256": "0000000000000000000000000000000000000000000000000000000000000000",
            "bytes": 1
          },
          "license": "Apache-2.0",
          "homepage": null
        }
      }
    }"#;
    std::fs::write(state.join("models.json"), catalog).expect("write models.json");
}

/// Write an empty (no-entry) `models.json` — every non-echo slug will
/// land in [`stratum_runtime::ProviderResolveError::UnknownSlug`].
fn write_empty_catalog(state: &Path) {
    let catalog = r#"{ "schema_version": 1, "entries": {} }"#;
    std::fs::write(state.join("models.json"), catalog).expect("write models.json");
}

fn run_chat(root: &Path, args: &[&str]) -> std::process::Output {
    let mut cmd = bin();
    cmd.args(["--storage-root", root.to_str().expect("utf-8 root")]);
    cmd.arg("chat");
    cmd.args(args);
    cmd.output().expect("spawn stratum chat")
}

// -- echo slug still works regardless of catalog ----------------------------

#[test]
fn chat_with_agents_dir_resolves_echo_slug_without_catalog() {
    let tmp = TempDir::new().unwrap();
    let dir = agents_dir(tmp.path());
    // Both roles request the echo provider — the floor case that must
    // work even when `<state>/models.json` is missing entirely.
    write_agent(&dir, "default", "default", "echo");
    write_agent(&dir, "coder", "coder", "echo");

    let out = run_chat(
        tmp.path(),
        &[
            "--agents-dir",
            dir.to_str().expect("utf-8 agents dir"),
            "--prompt",
            "hi",
        ],
    );
    assert!(
        out.status.success(),
        "expected success; stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("hi"),
        "expected echo of 'hi' in stdout; got {stdout:?}"
    );
}

// -- unknown slug against empty catalog -------------------------------------

#[test]
fn chat_with_agents_dir_unknown_slug_loads_one_role_and_runs() {
    let tmp = TempDir::new().unwrap();
    let state = state_dir(tmp.path());
    let dir = agents_dir(tmp.path());
    write_agent(&dir, "default", "default", "echo");
    write_agent(&dir, "coder", "coder", "qwen-0.5b");
    write_empty_catalog(&state);

    let out = run_chat(
        tmp.path(),
        &[
            "--agents-dir",
            dir.to_str().expect("utf-8 agents dir"),
            "--prompt",
            "hi",
        ],
    );
    // Default role loaded fine; coder role failed silently into the
    // load report but the registry still has one entry, so the chat
    // runs.
    assert!(
        out.status.success(),
        "expected success because one role loaded; stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("hi"),
        "expected echo of 'hi' from the surviving default role; got {stdout:?}"
    );
}

// -- known slug, feature off -----------------------------------------------

#[cfg(not(feature = "provider-llama-cpp"))]
#[test]
fn chat_with_agents_dir_known_slug_without_feature_loads_only_echo_role() {
    let tmp = TempDir::new().unwrap();
    let state = state_dir(tmp.path());
    let dir = agents_dir(tmp.path());
    write_agent(&dir, "default", "default", "echo");
    write_agent(&dir, "coder", "coder", "qwen-0.5b");
    write_catalog_with_qwen(&state);

    let out = run_chat(
        tmp.path(),
        &[
            "--agents-dir",
            dir.to_str().expect("utf-8 agents dir"),
            "--prompt",
            "hi",
        ],
    );
    // The catalog DOES list qwen-0.5b, so the resolver gets past the
    // unknown-slug check. Without `--features provider-llama-cpp`, the
    // resolver returns `Backend("requires --features provider-llama-cpp")`,
    // which the loader records as a per-file error. The default role
    // still loaded so the chat runs.
    assert!(
        out.status.success(),
        "expected success because the echo role loaded; stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("hi"),
        "expected echo of 'hi' from the surviving default role; got {stdout:?}"
    );
}

// -- all-roles-fail surfaces STRAT-E1001 -----------------------------------

#[cfg(not(feature = "provider-llama-cpp"))]
#[test]
fn chat_with_agents_dir_all_roles_fail_surfaces_no_agents_error() {
    let tmp = TempDir::new().unwrap();
    let state = state_dir(tmp.path());
    let dir = agents_dir(tmp.path());
    // Every TOML asks for `qwen-0.5b`; without the feature every load
    // ends up in `report.errors` and the registry is empty.
    write_agent(&dir, "default", "default", "qwen-0.5b");
    write_agent(&dir, "coder", "coder", "qwen-0.5b");
    write_catalog_with_qwen(&state);

    let out = run_chat(
        tmp.path(),
        &[
            "--agents-dir",
            dir.to_str().expect("utf-8 agents dir"),
            "--prompt",
            "hi",
        ],
    );
    assert!(!out.status.success(), "expected non-zero exit");
    assert_eq!(out.status.code(), Some(1), "expected exit code 1");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("STRAT-E1001"),
        "expected STRAT-E1001 in stderr; got {stderr:?}"
    );
}
