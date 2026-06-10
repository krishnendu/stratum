//! Integration tests for the long-form `--version` output.
//!
//! Verifies that `build.rs` wired up `STRATUM_BUILD_SHA` and
//! `STRATUM_BUILD_DATE` correctly, and that clap's short `-V` stays
//! terse while `--version` carries the full identification string
//! used in bug reports.
//!
//! Implementation detail: `run_with` funnels every clap parse outcome
//! (including the "errors" clap raises for `--version` and `--help`)
//! through a single error-writer, so the version string lands on
//! stderr and the process exits 64. We accept either stdout or stderr
//! and either exit 0 or 64 to stay forward-compatible if that wiring
//! changes upstream.

use std::process::Command;

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_stratum"))
}

const PKG_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Combined stdout+stderr from a CLI invocation — version output may
/// land on either stream depending on how the binary funnels clap's
/// `DisplayVersion` "error".
///
/// Only ever called from `#[test]` functions in this file; the
/// `expect_used` allow is scoped here so the rest of the crate keeps
/// the workspace-level deny.
#[allow(
    clippy::expect_used,
    reason = "test helper only invoked from #[test] fns; allow-expect-in-tests doesn't reach helpers"
)]
fn run_and_combine(args: &[&str]) -> String {
    let output = bin().args(args).output().expect("spawn stratum");
    // Accept clap's exit 64 (treated as a parse-time event) or 0.
    let code = output.status.code();
    assert!(
        code == Some(0) || code == Some(64),
        "unexpected exit {code:?} for {args:?}"
    );
    let mut combined = String::new();
    combined.push_str(&String::from_utf8_lossy(&output.stdout));
    combined.push_str(&String::from_utf8_lossy(&output.stderr));
    combined
}

#[test]
fn long_version_contains_pkg_sha_and_date() {
    let out = run_and_combine(&["--version"]);

    // Package version.
    assert!(
        out.contains(PKG_VERSION),
        "expected version {PKG_VERSION:?} in: {out:?}"
    );
    // "built " marker introduced by the LONG_VERSION concat in app.rs.
    assert!(
        out.contains("built "),
        "expected build-date marker in: {out:?}"
    );
    // SHA section appears between "(" and " built".
    let open = out.find('(').expect("expected '(' in long version");
    let built = out.find(" built ").expect("expected ' built ' marker");
    assert!(
        built > open,
        "marker ordering wrong: open={open}, built={built}, out={out:?}"
    );
    let sha = &out[open + 1..built];
    assert!(!sha.is_empty(), "SHA section must be non-empty: {out:?}");
}

#[test]
fn short_version_is_terse() {
    let out = run_and_combine(&["-V"]);
    assert!(
        out.contains(PKG_VERSION),
        "expected {PKG_VERSION:?} in short version: {out:?}"
    );
    // The short form must NOT carry the SHA or the build-date marker;
    // that's the whole reason we distinguish long vs short.
    assert!(
        !out.contains("built "),
        "short version leaked long form: {out:?}"
    );
    assert!(
        !out.contains('('),
        "short version leaked parenthesized SHA: {out:?}"
    );
}

#[test]
fn build_date_matches_rfc3339_shape() {
    let out = run_and_combine(&["--version"]);

    // Find the substring after "built " and before the trailing ')'.
    let after = out
        .split_once("built ")
        .map(|(_, rest)| rest)
        .expect("'built ' marker missing");
    let date = after.split_once(')').map(|(d, _)| d).expect("')' missing");

    // "unknown" is an acceptable fallback only when both `date` and
    // SystemTime failed; otherwise the date must match
    // YYYY-MM-DDTHH:MM:SSZ shape exactly (length 20, fixed punctuation).
    if date == "unknown" {
        return;
    }
    assert_eq!(date.len(), 20, "wrong length for {date:?}");
    let bytes = date.as_bytes();
    // Positions 0..4 year digits, 4 '-', 5..7 month, 7 '-', 8..10 day,
    // 10 'T', 11..13 hh, 13 ':', 14..16 mm, 16 ':', 17..19 ss, 19 'Z'.
    assert_eq!(bytes[4], b'-', "{date:?}");
    assert_eq!(bytes[7], b'-', "{date:?}");
    assert_eq!(bytes[10], b'T', "{date:?}");
    assert_eq!(bytes[13], b':', "{date:?}");
    assert_eq!(bytes[16], b':', "{date:?}");
    assert_eq!(bytes[19], b'Z', "{date:?}");
    for &pos in &[0, 1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18] {
        assert!(
            bytes[pos].is_ascii_digit(),
            "non-digit at {pos} in {date:?}"
        );
    }
}

#[test]
fn build_sha_present_or_unknown() {
    let out = run_and_combine(&["--version"]);

    let after_paren = out
        .split_once('(')
        .map(|(_, rest)| rest)
        .expect("'(' missing");
    let sha = after_paren
        .split_once(' ')
        .map(|(s, _)| s)
        .expect("space after SHA missing");
    assert!(!sha.is_empty(), "SHA must be non-empty");
    // Either the literal fallback or printable ASCII (real short SHA).
    if sha != "unknown" {
        for c in sha.chars() {
            assert!(
                c.is_ascii_alphanumeric(),
                "SHA char {c:?} not alphanumeric in {sha:?}"
            );
        }
    }
}
