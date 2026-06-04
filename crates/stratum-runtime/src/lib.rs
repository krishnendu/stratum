//! Runtime foundations.
//!
//! Phase 1 surface: the primitives every later subsystem (providers, agents,
//! tools, TUI) leans on — filesystem path resolution, hardware probe, tier
//! classifier, and the `installed.toml` first-run marker.
//!
//! See `plan/18-first-run-and-system-tiers.md` and `plan/28-finalization-v2.md`.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

/// First-run install record and atomic TOML writer.
pub mod install;
/// XDG-aware filesystem path resolution.
pub mod paths;
/// Hardware probe: RAM, CPU features, GPU backend, OS.
pub mod probe;
/// Composite tier classifier (low / medium / high).
pub mod tier;

pub use install::{InstalledToml, TierInputs};
pub use paths::Paths;
pub use probe::{GpuBackend, HardwareProbe};
pub use tier::Tier;
