//! Stratum evaluation harness — library surface.
//!
//! Hosts the Phase 7 bench-floor + bench-history modules so the two
//! sibling binaries (`bench-floor`, `bench-history`) and the unit tests
//! can share the same code paths.
//!
//! The top-level `stratum-eval` CLI is still in `src/main.rs`; this lib
//! crate only re-exports the bench modules.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod bench_floor;
pub mod bench_history;
