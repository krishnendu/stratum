//! Runtime foundations.
//!
//! Phase 1 surface: the primitives every later subsystem (providers, agents,
//! tools, TUI) leans on — filesystem path resolution, hardware probe, tier
//! classifier, and the `installed.toml` first-run marker.
//!
//! See `plan/18-first-run-and-system-tiers.md` and `plan/28-finalization-v2.md`.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

/// User-agent loader.
pub mod agents;
/// Cooperative cancellation token.
pub mod cancel;
/// Model-file install: SHA-256 verification, atomic copy with `.partial` swap.
pub mod download;
/// Memory-safety gate.
///
/// Refuses model loads when free RAM minus the would-be hot footprint falls
/// below the configured margin.
pub mod gate;
/// Prompt-injection defense primitives.
pub mod injection;
/// First-run install record and atomic TOML writer.
pub mod install;
/// `tracing` subscriber initialization with env-filter + file output.
pub mod logging;
/// Turn-level observability primitives: token meter, latency steps, tok/s.
pub mod observability;
/// Panic hook + crash report file writer.
pub mod panic;
/// XDG-aware filesystem path resolution.
pub mod paths;
/// Hardware probe: RAM, CPU features, GPU backend, OS.
pub mod probe;
/// Embedded caveman-rewriter and polisher system prompts.
pub mod prompts;
/// Provider abstractions and concrete `EchoProvider` for end-to-end loop tests.
pub mod provider;
/// Provider registry + role-to-provider routing table.
pub mod registry;
/// Sandbox backend detection.
pub mod sandbox;
/// Sandbox profile bodies (bwrap-*, macos-*, passthrough).
pub mod sandbox_profile;
/// Composite tier classifier (low / medium / high).
pub mod tier;
/// Tool registry and capability matrix.
pub mod tools;
/// Workspace / project discovery (`stratum.toml`, `.stratumignore`).
pub mod workspace;

pub use agents::{AgentBudget, AgentDef, AgentLoader};
pub use cancel::CancelToken;
pub use download::{InstallReport, ModelInstaller};
pub use gate::{LoadedModel, MemoryGate, DEFAULT_MARGIN_MIB};
pub use injection::{fence, is_suspicious, suspicion_score, FenceSource, SUSPICION_THRESHOLD};
pub use install::{InstalledToml, TierInputs};
pub use observability::{
    format_tokens_per_second, RoleStep, RoleTimer, TurnId, TurnIdGen, TurnMetrics, TurnRecorder,
};
pub use paths::Paths;
pub use probe::{GpuBackend, HardwareProbe};
pub use prompts::{system_prompt, PromptRole};
pub use provider::{EchoProvider, GenerateRequest, Provider};
pub use registry::Registry;
pub use sandbox::{SandboxBackend, SandboxReport};
pub use sandbox_profile::{Mount, NetPolicy, SandboxProfile};
pub use tier::Tier;
pub use tools::{CapabilityEntry, CapabilityMatrix};
pub use workspace::{Workspace, WorkspaceConfig};
