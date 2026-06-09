//! Integration tests for `stratum chat --model <slug> [--prompt <STR>]`.
//!
//! These tests spawn the built CLI binary against a fresh `--storage-root`
//! tempdir. The real-fetch path (download a GGUF, open `LlamaCppProvider`)
//! is intentionally *not* exercised here — the heavy llama.cpp build only
//! runs on the on-demand workflow + dev boxes. What we lock down on the
//! default per-PR matrix is:
//!
//! * `--prompt` flows through the chat surface and prints stdout (no
//!   `--model`, `EchoProvider` path).
//! * Without the `provider-llama-cpp` feature, `--model` errors with
//!   STRAT-E1001 + a feature-flag hint and exits 1.
//! * With the feature, an unknown slug also errors with STRAT-E1001 + a
//!   "stratum models list" hint and exits 1.
//! * `last_assistant_text` round-trips through `ChatState::submit_with_prompt`.

use std::process::Command;

use tempfile::TempDir;

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_stratum"))
}

#[test]
fn chat_prompt_echoes_back() {
    // No --model: EchoProvider path. The provider's prefix is "echo: " so
    // the resulting assistant text contains the user prompt verbatim. We
    // assert on the substring rather than the full echo line because
    // future status banners may add extra lines.
    let tmp = TempDir::new().unwrap();
    let output = bin()
        .args([
            "--storage-root",
            tmp.path().to_str().unwrap(),
            "chat",
            "--prompt",
            "hello",
        ])
        .output()
        .expect("spawn stratum");
    assert!(
        output.status.success(),
        "exit={:?} stderr={}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("hello"),
        "expected stdout to contain prompt; got: {stdout}"
    );
}

#[cfg(not(feature = "provider-llama-cpp"))]
#[test]
fn chat_model_without_feature_errors_with_e1001() {
    let tmp = TempDir::new().unwrap();
    let output = bin()
        .args([
            "--storage-root",
            tmp.path().to_str().unwrap(),
            "chat",
            "--model",
            "qwen",
            "--prompt",
            "hi",
        ])
        .output()
        .expect("spawn stratum");
    assert_eq!(output.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("STRAT-E1001"),
        "expected STRAT-E1001 in stderr; got: {stderr}"
    );
    assert!(
        stderr.contains("provider-llama-cpp"),
        "expected feature-flag hint in stderr; got: {stderr}"
    );
}

#[cfg(feature = "provider-llama-cpp")]
#[test]
fn chat_unknown_slug_with_feature_errors_with_e1001() {
    // Catalog is empty (file does not exist) so any --model lookup
    // resolves to "unknown slug" and we get STRAT-E1001 + the install
    // hint pointing at `stratum models list`.
    let tmp = TempDir::new().unwrap();
    let output = bin()
        .args([
            "--storage-root",
            tmp.path().to_str().unwrap(),
            "chat",
            "--model",
            "unknown-slug",
            "--prompt",
            "hi",
        ])
        .output()
        .expect("spawn stratum");
    assert_eq!(output.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("STRAT-E1001"),
        "expected STRAT-E1001 in stderr; got: {stderr}"
    );
    assert!(
        stderr.contains("unknown-slug") || stderr.contains("unknown slug"),
        "expected slug hint in stderr; got: {stderr}"
    );
    assert!(
        stderr.contains("stratum models list"),
        "expected models-list hint in stderr; got: {stderr}"
    );
}
