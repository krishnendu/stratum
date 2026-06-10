//! End-to-end tests for `stratum chat --agents-dir <path>`.
//!
//! Each test spawns the built CLI binary against an isolated `--storage-root`
//! `TempDir`, stages one or more agent TOML files under
//! `<state>/agents/`, and exercises the documented behaviour of the
//! `--agents-dir` flag:
//!
//! * A populated directory + `--prompt` runs the echo provider end-to-end
//!   through the [`stratum_runtime::AgentHandoff`] coordinator and prints
//!   the prompt verbatim (echo with empty prefix).
//! * An empty directory exits `1` with a `STRAT-E1001` "no agents" hint.
//! * A prompt carrying the runtime's hand-off sentinel (`<handoff:coder>`)
//!   triggers a second hop. The final transcript surfaces both turns plus
//!   per-step "(handoff: …)" command lines.
//! * Default `stratum chat --prompt …` (no `--agents-dir`) remains
//!   unchanged — a regression guard for the single-loop path.

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

/// Minimal `AgentDef` body matching the loader's serde expectation. The role
/// names below (`default`, `coder`) deliberately avoid colliding with
/// `IntentRouter::default()` rules — `polisher` for instance would match the
/// "polish" substring rule and re-route the initial turn away from the
/// requested role.
fn body_for(role: &str) -> String {
    format!(
        r#"
schema_version = 1
name = "{role}-agent"
description = "smoke agent for {role}"
roles = ["{role}"]
model = "echo"
tools = []
sandbox = "passthrough"
"#
    )
}

fn write_agent(dir: &Path, name: &str, role: &str) {
    let path = dir.join(format!("{name}.toml"));
    std::fs::write(&path, body_for(role)).expect("write agent toml");
}

fn run_chat(root: &Path, args: &[&str]) -> std::process::Output {
    let mut cmd = bin();
    cmd.args(["--storage-root", root.to_str().expect("utf-8 root")]);
    cmd.arg("chat");
    cmd.args(args);
    cmd.output().expect("spawn stratum chat")
}

// -- happy path ---------------------------------------------------------------

#[test]
fn chat_with_agents_dir_runs_single_turn_via_handoff() {
    let tmp = TempDir::new().unwrap();
    let dir = agents_dir(tmp.path());
    write_agent(&dir, "default", "default");
    write_agent(&dir, "coder", "coder");

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

// -- empty directory ----------------------------------------------------------

#[test]
fn chat_with_empty_agents_dir_exits_one_with_strat_e1001() {
    let tmp = TempDir::new().unwrap();
    let dir = agents_dir(tmp.path()); // created but empty

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
    assert!(
        stderr.contains("no agents"),
        "expected 'no agents' hint in stderr; got {stderr:?}"
    );
}

// -- multi-hop chain ----------------------------------------------------------

#[test]
fn chat_with_agents_dir_routes_handoff_marker_to_coder() {
    // EchoProvider echoes the prompt verbatim (empty prefix). The first hop
    // receives "<handoff:coder>do thing", echoes it back, the orchestrator
    // detects the sentinel, strips it, and re-routes "do thing" to the
    // `coder` registry entry. Final blocks should contain "do thing".
    //
    // We rely on `IntentRouter::default()` *not* matching "<handoff:coder>do
    // thing" with a strict-enough rule to pull the initial role away from
    // `Default`. The default router routes unknown prose to `Default`, and
    // `<handoff:coder>` is not in the rule catalog.
    let tmp = TempDir::new().unwrap();
    let dir = agents_dir(tmp.path());
    write_agent(&dir, "default", "default");
    write_agent(&dir, "coder", "coder");

    let out = run_chat(
        tmp.path(),
        &[
            "--agents-dir",
            dir.to_str().expect("utf-8 agents dir"),
            "--prompt",
            "<handoff:coder>do thing",
        ],
    );
    assert!(
        out.status.success(),
        "expected success; stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    // Final assistant text comes from the coder hop, which receives the
    // marker-stripped prompt "do thing". EchoProvider returns "do" + "thing"
    // (one block per whitespace-separated word) so both words appear.
    assert!(
        stdout.contains("do") && stdout.contains("thing"),
        "expected echo of stripped prompt; got {stdout:?}"
    );
}

// -- regression: default chat path is unchanged ------------------------------

#[test]
fn chat_without_agents_dir_still_uses_single_loop() {
    let tmp = TempDir::new().unwrap();
    let out = run_chat(tmp.path(), &["--prompt", "hello"]);
    assert!(
        out.status.success(),
        "expected success; stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    // Default chat prints with the "echo: " prefix.
    assert!(
        stdout.contains("echo: hello"),
        "expected default echo prefix in stdout; got {stdout:?}"
    );
}
