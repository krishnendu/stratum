//! Integration test: every catalog constant in
//! [`stratum_types::error::codes`] renders to its documented
//! `STRAT-E####` string.
//!
//! This test also doubles as the workspace reference site for catalog
//! entries that are declared today but not yet emitted from a real error
//! path (the upcoming phases will wire them up). Without these references
//! the `xtask check-error-codes` orphan check would flag those constants.
//! Treat this file as the canonical "catalog smoke test"; when a new code
//! is added to the catalog, add a row here.

use stratum_types::error::codes::{
    E1001_INSTALLED_SCHEMA_UNREADABLE, E1003_SECRET_UNSET, E2001_INSUFFICIENT_RAM,
    E2003_TIER_DOWNGRADE_REFUSED, E3007_MODEL_LOAD_REFUSED, E4002_AGENT_SHADOW, E4003_TOKEN_BUDGET,
    E4004_WALL_BUDGET, E4005_CLIENT_DISCONNECT, E4006_INJECTION_REFUSAL, E5004_TOOL_DENIED,
    E5005_NET_DENIED, E6001_UPDATE_SIG, E7002_RAG_VERSION, E8001_RATE_LIMIT, E9001_INTERNAL_PANIC,
};
use stratum_types::error::ErrorCode;

fn assert_renders(code: &ErrorCode, expected: &str) {
    assert_eq!(
        code.as_str(),
        expected,
        "catalog constant rendered {} but expected {expected}",
        code.as_str(),
    );
}

#[test]
fn catalog_codes_render_expected_strings() {
    assert_renders(&E1001_INSTALLED_SCHEMA_UNREADABLE, "STRAT-E1001");
    assert_renders(&E1003_SECRET_UNSET, "STRAT-E1003");
    assert_renders(&E2001_INSUFFICIENT_RAM, "STRAT-E2001");
    assert_renders(&E2003_TIER_DOWNGRADE_REFUSED, "STRAT-E2003");
    assert_renders(&E3007_MODEL_LOAD_REFUSED, "STRAT-E3007");
    assert_renders(&E4002_AGENT_SHADOW, "STRAT-E4002");
    assert_renders(&E4003_TOKEN_BUDGET, "STRAT-E4003");
    assert_renders(&E4004_WALL_BUDGET, "STRAT-E4004");
    assert_renders(&E4005_CLIENT_DISCONNECT, "STRAT-E4005");
    assert_renders(&E4006_INJECTION_REFUSAL, "STRAT-E4006");
    assert_renders(&E5004_TOOL_DENIED, "STRAT-E5004");
    assert_renders(&E5005_NET_DENIED, "STRAT-E5005");
    assert_renders(&E6001_UPDATE_SIG, "STRAT-E6001");
    assert_renders(&E7002_RAG_VERSION, "STRAT-E7002");
    assert_renders(&E8001_RATE_LIMIT, "STRAT-E8001");
    assert_renders(&E9001_INTERNAL_PANIC, "STRAT-E9001");
}
