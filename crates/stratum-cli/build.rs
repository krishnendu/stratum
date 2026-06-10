//! Build script for `stratum-cli`.
//!
//! Emits two `rustc-env` variables consumed by `app.rs` to build the
//! long-form `--version` string:
//!
//! - `STRATUM_BUILD_SHA` — short git SHA (`git rev-parse --short HEAD`),
//!   or `"unknown"` when git isn't available or the source isn't a repo
//!   (tarball builds, Cargo registry packages, vendored sources).
//! - `STRATUM_BUILD_DATE` — UTC RFC3339-ish timestamp
//!   (`YYYY-MM-DDTHH:MM:SSZ`). Prefers the system `date -u` because it
//!   matches the format users see in commit hashes / changelogs, and
//!   falls back to `std::time::SystemTime` math when `date` isn't on
//!   PATH (Windows, minimal containers).
//!
//! Both vars are *always* set so `env!()` in `app.rs` never fails to
//! compile. The build script is deliberately infallible: any error path
//! collapses to `"unknown"` rather than aborting the build, because
//! version-string ornamentation must never break `cargo build` in a
//! tarball / offline / no-git environment.

#![allow(
    // Pedantic-tier lints we knowingly relax in this build-only
    // helper. The civil-from-days math operates exclusively on
    // non-negative unix timestamps and small bounded intermediates;
    // every `as`-cast below has been audited to be in-range for any
    // timestamp Stratum will ever ship at.
    clippy::cast_possible_truncation,
    clippy::similar_names,
)]

use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn main() {
    // Rebuild when HEAD moves so the SHA stays fresh across commits.
    // `.git/HEAD` changes on every checkout / commit; for detached-HEAD
    // and branch tips this is sufficient. Missing file is a no-op for
    // cargo, which is the right behavior in tarball builds.
    println!("cargo:rerun-if-changed=.git/HEAD");

    let sha = git_short_sha().unwrap_or_else(|| "unknown".to_string());
    println!("cargo:rustc-env=STRATUM_BUILD_SHA={sha}");

    let date = build_date_utc();
    println!("cargo:rustc-env=STRATUM_BUILD_DATE={date}");
}

/// Try to read the short git SHA of HEAD. Returns `None` on any failure
/// (no git binary, not in a repo, non-zero exit, non-UTF-8 output).
fn git_short_sha() -> Option<String> {
    let output = Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let sha = String::from_utf8(output.stdout).ok()?;
    let trimmed = sha.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

/// Build-time UTC timestamp formatted as `YYYY-MM-DDTHH:MM:SSZ`.
///
/// Prefers `date -u +%Y-%m-%dT%H:%M:%SZ` because it matches what users
/// expect to see in CI logs. Falls back to a hand-rolled formatter over
/// `SystemTime` (no chrono / time dependency in build.rs — workspace
/// rules forbid extra build-side deps) when `date` is unavailable
/// (Windows, scratch containers).
fn build_date_utc() -> String {
    if let Some(d) = date_via_command() {
        return d;
    }
    date_via_systemtime().unwrap_or_else(|| "unknown".to_string())
}

fn date_via_command() -> Option<String> {
    let output = Command::new("date")
        .args(["-u", "+%Y-%m-%dT%H:%M:%SZ"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let s = String::from_utf8(output.stdout).ok()?;
    let trimmed = s.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

/// Hand-rolled UTC formatter. Converts a unix timestamp (seconds since
/// epoch) into `YYYY-MM-DDTHH:MM:SSZ`. Implements the civil-from-days
/// algorithm by Howard Hinnant (`date.h`, public domain), inlined here
/// to avoid pulling a build-time dependency.
fn date_via_systemtime() -> Option<String> {
    let secs = SystemTime::now().duration_since(UNIX_EPOCH).ok()?.as_secs();
    let days = secs / 86_400;
    let secs_of_day = secs % 86_400;
    let hh = secs_of_day / 3600;
    let mm = (secs_of_day % 3600) / 60;
    let ss = secs_of_day % 60;

    let (year, month, day) = civil_from_days(days);
    Some(format!(
        "{year:04}-{month:02}-{day:02}T{hh:02}:{mm:02}:{ss:02}Z"
    ))
}

/// Hinnant's `civil_from_days` adapted for unsigned, post-1970 inputs:
/// days since 1970-01-01 → (year, month, day). Returns month in
/// `1..=12` and day in `1..=31`. Since Stratum never ships pre-1970
/// builds, taking `u64` keeps the math sign-loss-free.
const fn civil_from_days(days: u64) -> (u64, u64, u64) {
    let z = days + 719_468;
    let era = z / 146_097;
    let doe = z - era * 146_097; // [0, 146_096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let year_pre = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let day = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let month = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    let year = if month <= 2 { year_pre + 1 } else { year_pre };
    (year, month, day)
}
