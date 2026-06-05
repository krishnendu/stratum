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
/// Per-turn budget tracker layered over an `AgentBudget` + `CancelToken`.
pub mod budget;
/// Cooperative cancellation token.
pub mod cancel;
/// Hierarchical cancellation cascade with reasons + RAII deadlines.
pub mod cancel_cascade;
/// Opt-in crash-report bundle + log redaction (Phase 4 scaffold).
pub mod crash_report;
/// Model-file install: SHA-256 verification, atomic copy with `.partial` swap.
pub mod download;
/// Embedding backend trait + deterministic `HashEmbedder` stub.
pub mod embedder;
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
/// MCP client + server data shapes (Phase 3 data-only scaffold for Phase 6).
pub mod mcp;
/// Curated model catalog: structured index of installer-resolvable models.
pub mod model_catalog;
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
/// RAG index data shape and in-memory index (Phase 1 scaffold for Phase 4+).
pub mod rag;
/// Token-bucket rate limiter primitives (scaffold for `stratum serve`).
pub mod rate_limit;
/// Provider registry + role-to-provider routing table.
pub mod registry;
/// Deterministic retry-with-backoff helper for transient errors.
pub mod retry;
/// Sandbox backend detection.
pub mod sandbox;
/// Sandbox profile bodies (bwrap-*, macos-*, passthrough).
pub mod sandbox_profile;
/// Sandbox-profile resolver — combine profile + caps + workspace → launch spec.
pub mod sandbox_resolve;
/// Secrets / keyring data shape (Phase 1 scaffold; real OS backend lands later).
pub mod secrets;
/// Default-on opt-out telemetry payload shape + allowlist guard.
pub mod telemetry;
/// Composite tier classifier (low / medium / high).
pub mod tier;
/// Tool registry and capability matrix.
pub mod tools;
/// On-disk conversation-transcript shape + atomic JSON store.
pub mod transcript;
/// `stratum self-update` channel-manifest data shape (Phase 1 scaffold).
pub mod update_manifest;
/// Workspace / project discovery (`stratum.toml`, `.stratumignore`).
pub mod workspace;

pub use agents::{AgentBudget, AgentDef, AgentLoader};
pub use budget::{BudgetCheck, BudgetTracker};
pub use cancel::CancelToken;
pub use cancel_cascade::{CancelError, CancelReason, CascadeToken, DeadlineGuard};
pub use crash_report::{
    build_bundle, load_bundle, redact_log_lines, redact_path_user, write_bundle, CrashBundle,
    CrashBundleConfig, CrashBundleError, CrashEnv, CRASH_BUNDLE_SCHEMA_VERSION,
};
pub use download::{InstallReport, ModelInstaller};
pub use embedder::{
    cosine_similarity, top_k, EmbedError, Embedder, EmbeddingDim, EmbeddingVector, HashEmbedder,
    InMemoryVectorStore,
};
pub use gate::{LoadedModel, MemoryGate, DEFAULT_MARGIN_MIB};
pub use injection::{fence, is_suspicious, suspicion_score, FenceSource, SUSPICION_THRESHOLD};
pub use install::{
    backup_path, load_with_migration, restore_backup, save_atomic, InstallIoError,
    InstallLoadError, InstalledToml, TierInputs, CURRENT_SCHEMA_VERSION,
};
pub use mcp::{
    McpServeTransport, McpServerConfig, McpServerExpose, McpServerSet, McpServerStatus,
    McpTransport,
};
pub use model_catalog::{
    ArtifactRef, ArtifactRefError, CatalogError, ModelCatalog, ModelEntry, ModelSlug,
    ModelSlugError, ModelTask, ModelTier, MODEL_CATALOG_SCHEMA_VERSION,
};
pub use observability::{
    format_tokens_per_second, RoleStep, RoleTimer, TurnId, TurnIdGen, TurnMetrics, TurnRecorder,
};
pub use paths::Paths;
pub use probe::{GpuBackend, HardwareProbe};
pub use prompts::{system_prompt, PromptRole};
pub use provider::{EchoProvider, GenerateRequest, Provider};
pub use rag::{chunk_document, Chunk, ChunkPlan, ChunkSpan, DocumentId, RagDocument, RagIndex};
pub use rate_limit::{
    Clock, KeyedRateLimiter, ManualClock, RateLimitError, SystemClock, TokenBucket,
    TokenBucketConfig,
};
pub use registry::Registry;
pub use retry::{
    retry, retry_with_clock, Clock as RetryClock, Jitter, ManualClock as RetryManualClock,
    RetryClassifier, RetryDecision, RetryError, RetryPolicy, SystemClock as RetrySystemClock,
};
pub use sandbox::{SandboxBackend, SandboxReport};
pub use sandbox_profile::{Mount, NetPolicy, SandboxProfile};
pub use sandbox_resolve::{
    resolve, resolve_with_warnings, BackendChoice, MountMode, ResolveError, ResolvedMount,
    ResolvedNet, SandboxLaunchSpec,
};
pub use secrets::{
    redact_for_log, InMemorySecretStore, ProjectId, SecretId, SecretIdError, SecretRef,
    SecretScope, SecretStore, SecretStoreError, SecretValue,
};
pub use telemetry::{
    build_payload, payload_is_allowlisted, redact, AnonInstallId, AnonInstallIdError, CpuArchTag,
    OsTag, ReleaseChannel, TelemetryConfig, TelemetryError, TelemetryEventKind, TelemetryPayload,
    TELEMETRY_SCHEMA_VERSION,
};
pub use tier::Tier;
pub use tools::{CapabilityEntry, CapabilityMatrix};
pub use transcript::{
    redact_pii, SessionId, SessionIdError, Transcript, TranscriptBlock, TranscriptBlockKind,
    TranscriptError, TranscriptStore, TranscriptTurn, TRANSCRIPT_SCHEMA_VERSION,
};
pub use update_manifest::{
    evaluate, ArtifactRef as UpdateArtifactRef, ArtifactRefError as UpdateArtifactRefError,
    ManifestError, PlatformTag, ReleaseEntry, ReleaseVersion, ReleaseVersionError, UpdateChannel,
    UpdateDecision, UpdateManifest, UPDATE_MANIFEST_SCHEMA_VERSION,
};
pub use workspace::{IgnoreRule, StratumIgnore, Workspace, WorkspaceConfig};
