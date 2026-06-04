//! Runtime foundations.
//!
//! Phase 1 surface: the primitives every later subsystem (providers, agents,
//! tools, TUI) leans on — filesystem path resolution, hardware probe, tier
//! classifier, and the `installed.toml` first-run marker.
//!
//! See `plan/18-first-run-and-system-tiers.md` and `plan/28-finalization-v2.md`.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

/// Cooperative cancellation token.
pub mod cancel;
/// Model-file install: SHA-256 verification, atomic copy with `.partial` swap.
pub mod download;
/// First-run install record and atomic TOML writer.
pub mod install;
/// `tracing` subscriber initialization with env-filter + file output.
pub mod logging;
/// Panic hook + crash report file writer.
pub mod panic;
/// XDG-aware filesystem path resolution.
pub mod paths;
/// Hardware probe: RAM, CPU features, GPU backend, OS.
pub mod probe;
/// Provider abstractions and concrete `EchoProvider` for end-to-end loop tests.
pub mod provider;
/// Composite tier classifier (low / medium / high).
pub mod tier;

pub use cancel::CancelToken;
pub use download::{InstallReport, ModelInstaller};
pub use install::{InstalledToml, TierInputs};
pub use paths::Paths;
pub use probe::{GpuBackend, HardwareProbe};
pub use provider::{EchoProvider, GenerateRequest};
pub use tier::Tier;
