//! Integration tests for `stratum completions <shell>`.
//!
//! Each test spawns the actual built binary via `CARGO_BIN_EXE_stratum` and
//! asserts that the emitted completion script matches the canonical
//! `clap_complete` format for that shell. The script bodies are
//! intentionally checked by sentinel substrings (e.g. `#compdef stratum`)
//! rather than full snapshots so a `clap_complete` patch bump doesn't break
//! the suite over cosmetic whitespace.

use std::process::Command;

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_stratum"))
}

#[test]
fn completions_bash_emits_bash_script() {
    let output = bin()
        .args(["completions", "bash"])
        .output()
        .expect("spawn stratum");
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("stdout is utf-8");
    // `clap_complete` emits a function named after the binary; see
    // `cargo run -- completions bash | head -1` → `_stratum()`.
    assert!(
        stdout.contains("_stratum()"),
        "expected _stratum() in bash completion, got first 200 bytes: {}",
        &stdout[..stdout.len().min(200)]
    );
}

#[test]
fn completions_zsh_emits_compdef_directive() {
    let output = bin()
        .args(["completions", "zsh"])
        .output()
        .expect("spawn stratum");
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).expect("stdout is utf-8");
    assert!(
        stdout.contains("#compdef stratum"),
        "expected #compdef stratum in zsh completion"
    );
}

#[test]
fn completions_fish_emits_complete_directive() {
    let output = bin()
        .args(["completions", "fish"])
        .output()
        .expect("spawn stratum");
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).expect("stdout is utf-8");
    assert!(
        stdout.contains("complete -c stratum"),
        "expected `complete -c stratum` in fish completion"
    );
}

#[test]
fn completions_powershell_emits_non_empty_script() {
    let output = bin()
        .args(["completions", "powershell"])
        .output()
        .expect("spawn stratum");
    assert!(output.status.success());
    assert!(
        !output.stdout.is_empty(),
        "powershell completion must be non-empty"
    );
    let stdout = String::from_utf8(output.stdout).expect("stdout is utf-8");
    // PowerShell script references the binary name somewhere.
    assert!(
        stdout.contains("stratum"),
        "powershell completion must mention 'stratum'"
    );
}

#[test]
fn completions_elvish_emits_non_empty_script() {
    let output = bin()
        .args(["completions", "elvish"])
        .output()
        .expect("spawn stratum");
    assert!(output.status.success());
    assert!(
        !output.stdout.is_empty(),
        "elvish completion must be non-empty"
    );
    let stdout = String::from_utf8(output.stdout).expect("stdout is utf-8");
    assert!(
        stdout.contains("stratum"),
        "elvish completion must mention 'stratum'"
    );
}

#[test]
fn completions_unknown_shell_exits_64() {
    let output = bin()
        .args(["completions", "unknown-shell"])
        .output()
        .expect("spawn stratum");
    assert_eq!(output.status.code(), Some(64));
}

#[test]
fn completions_missing_shell_arg_exits_64() {
    let output = bin().args(["completions"]).output().expect("spawn stratum");
    assert_eq!(output.status.code(), Some(64));
}
