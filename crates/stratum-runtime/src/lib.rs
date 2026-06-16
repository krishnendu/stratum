//! Runtime foundations.
//!
//! Phase 1 surface. See `plan/18-first-run-and-system-tiers.md`.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
// 50+ `pub mod` declarations each carry a one-line doc; the lint
// fires on spurious cross-decl spans we cannot rephrase further.
#![allow(
    clippy::too_long_first_doc_paragraph,
    reason = "false positive on adjacent pub-mod doc lines"
)]
// `clippy::large_stack_arrays` emits a span-less warning during the
// `lib-test` compilation of `tool_dispatchers::tests` once that
// module's combined macro expansions cross an upstream clippy
// heuristic. The production `lib` build alone is clean (no array
// literal in the workspace approaches the 16 KiB cap); the warning
// has no real source location. Gate behind `cfg(test)` so the lint
// surface for downstream callers of the library is unchanged.
#![cfg_attr(test, allow(clippy::large_stack_arrays))]

/// `AgentFactory` builder.
///
/// Fluent builder that composes a fully-wired [`agent_loop::AgentLoop`]
/// from a config struct + the minimum required dependency (a
/// [`provider::Provider`]).
pub mod agent_factory;
/// `AgentHandoff` multi-role coordinator.
///
/// Routes a turn through one or more [`agent_loop::AgentLoop`]s based on
/// a hand-off sentinel emitted by the assistant's last block.
pub mod agent_handoff;
/// `AgentLoop` orchestrator.
///
/// Composes the FSM, intent router, provider, permission store, event
/// emitter, plan-mode fence, and capability matrix into a single
/// `run_turn` entry point.
pub mod agent_loop;
/// `AgentRegistryLoader` — populates an [`agent_handoff::AgentRegistry`].
///
/// Walks `<state>/agents/*.toml` and builds one [`agent_loop::AgentLoop`]
/// per file via [`agent_factory::AgentFactory`].
pub mod agent_registry_loader;
/// `AgentSession` — high-level conversation wrapper.
///
/// Composes [`agent_loop::AgentLoop`], [`transcript::TranscriptStore`],
/// and [`event_log::EventEmitter`] into a single
/// `next_turn(prompt) -> TurnResult` surface.
pub mod agent_session;
/// User-agent loader.
pub mod agents;
/// `AnthropicApiJudge` — metered HTTP transport for the Stratum LLM-judge.
///
/// Opt-in alternative to the default subscription-backed
/// [`claude_cli_judge`] subprocess; speaks the Anthropic Messages API
/// over `ureq`.
pub mod anthropic_api_judge;
/// Auto-memory storage layer — per-repo MEMORY.md index + body files
/// the agent maintains across sessions (plan/40).
pub mod auto_memory;
/// Per-turn budget tracker layered over an `AgentBudget` + `CancelToken`.
pub mod budget;
/// Session-level cumulative budget meter (totals + per-role breakdown + hard cap).
pub mod budget_meter;
/// Cooperative cancellation token.
pub mod cancel;
/// Hierarchical cancellation cascade with reasons + RAII deadlines.
pub mod cancel_cascade;
/// Candle-backed provider scaffold (Phase 2 v2 embeddings landing).
pub mod candle_provider;
/// Remote `ModelCatalog` sync over HTTPS — fetch + validate + atomic write.
pub mod catalog_sync;
/// Deterministic Caveman compressor — heuristic prose → fragment
/// for inter-agent messages and tool-result re-injection.
pub mod caveman;
/// `claude -p` subprocess transport for the Stratum LLM-judge.
pub mod claude_cli_judge;
/// Context compressor trait — Caveman v1, LLMLingua-2 plug point.
pub mod compressor;
/// Per-turn conversation state machine driving the agentic loop.
pub mod conversation;
/// Opt-in crash-report bundle + log redaction (Phase 4 scaffold).
pub mod crash_report;
/// Model-file install: SHA-256 verification, atomic copy with `.partial` swap.
pub mod download;
/// Embedding backend trait + deterministic `HashEmbedder` stub.
pub mod embedder;
/// Prompt-based eval suites over an [`agent_loop::AgentLoop`].
///
/// `EvalRunner` runs an [`eval_runner::EvalSuite`] (sequence of prompts +
/// expected substrings) against an [`agent_loop::AgentLoop`] and produces
/// an [`eval_runner::EvalReport`].
pub mod eval_runner;
/// Append-only structured event log for tool calls, permissions, hand-offs.
pub mod event_log;
/// Memory-safety gate.
///
/// Refuses model loads when free RAM minus the would-be hot footprint falls
/// below the configured margin.
pub mod gate;
/// Hooks runtime — settings.json hooks dispatcher (plan/42).
pub mod hooks;
/// Filesystem hot-reload for settings / agents / hooks (plan/30 §10.3).
pub mod hot_reload;
/// Fluent-style i18n catalog + lookup (Phase 1 scaffold).
pub mod i18n;
/// Prompt-injection defense primitives.
pub mod injection;
/// First-run install record and atomic TOML writer.
pub mod install;
/// Pure-rules intent classifier mapping a prompt to a [`RoutedIntent`].
pub mod intent_router;
/// Llama.cpp-backed provider scaffold (feature-gated, off by default).
#[cfg(feature = "provider-llama-cpp")]
pub mod llama_provider;
/// `tracing` subscriber initialization with env-filter + file output.
pub mod logging;
/// MCP client + server data shapes (Phase 3 data-only scaffold for Phase 6).
pub mod mcp;
/// Real JSON-RPC 2.0 client over an MCP stdio child (Phase 6 scaffold).
pub mod mcp_jsonrpc;
/// `STRATUM.md` walk-up loader with `@file` imports (plan/39).
pub mod memory_loader;
/// cpal-backed microphone capture scaffold for Phase 5 voice-in.
pub mod mic;
/// Curated model catalog: structured index of installer-resolvable models.
pub mod model_catalog;
/// Slug → local GGUF path resolver composing a [`model_catalog::ModelCatalog`]
/// + a pluggable [`model_resolver::BlobFetcher`] over a content-addressed
/// `<state_root>/models/<sha256>.gguf` cache.
pub mod model_resolver;
/// Turn-level observability primitives: token meter, latency steps, tok/s.
pub mod observability;
/// OpenAI-compatible Chat Completions HTTP egress (`stratum serve --openai`).
pub mod openai;
/// Canonical multi-role orchestrator wrapping AgentLoop with router +
/// reviewer + polish per plan/03 + plan/17.
pub mod orchestrator;
/// Panic hook + crash report file writer.
pub mod panic;
/// XDG-aware filesystem path resolution.
pub mod paths;
/// Interactive permission-prompt data shape + remembered-decision store.
pub mod permission_prompt;
/// Permission rule DSL — `fs.write(*.rs)`, `Bash(npm *)` (plan/30 §10.1).
pub mod permission_rules;
/// piper TTS subprocess scaffold for Phase 5 voice-out synthesis.
pub mod piper;
/// Plan-mode capability fence (read-only sandbox).
pub mod plan_mode;
/// Hardware probe: RAM, CPU features, GPU backend, OS.
pub mod probe;
/// Prefix prompt cache + reuse-key fingerprinting.
///
/// Sha-256 over system + agent-header text — lets providers skip
/// re-tokenizing the static prefix on every turn. See
/// `plan/13-prompt-cache.md`.
pub mod prompt_cache;
/// Structured prompt template + composition layer.
pub mod prompt_template;
/// Embedded caveman-rewriter and polisher system prompts.
pub mod prompts;
/// Provider abstractions and concrete `EchoProvider` for end-to-end loop tests.
pub mod provider;
/// Provider warm-up / LRU cache + RAM-budget eviction.
pub mod provider_cache;
/// RAG index data shape and in-memory index (Phase 1 scaffold for Phase 4+).
pub mod rag;
/// `RagIndexBuilder` — workspace tree → `RagIndex` + `InMemoryVectorStore` pipeline.
pub mod rag_index_builder;
/// `RagQuery` — prompt → ranked passages over a `BuiltIndex`.
pub mod rag_query;
/// Token-bucket rate limiter primitives (scaffold for `stratum serve`).
pub mod rate_limit;
/// Provider registry + role-to-provider routing table.
pub mod registry;
/// Deterministic retry-with-backoff helper for transient errors.
pub mod retry;
/// Reviewer pass — score an assistant draft with a SECOND provider
/// (anti-self-bias per plan/17).
pub mod reviewer;
/// Role-based model swap controller (plan/02 §Roster).
pub mod role_router;
/// Sandbox backend detection.
pub mod sandbox;
/// Sandbox profile bodies (bwrap-*, macos-*, passthrough).
pub mod sandbox_profile;
/// Sandbox-profile resolver — combine profile + caps + workspace → launch spec.
pub mod sandbox_resolve;
/// Secrets / keyring data shape (Phase 1 scaffold; real OS backend lands later).
pub mod secrets;
/// `AgentServeHandler` — first production [`serve_server::ServeHandler`].
///
/// Wires real [`agent_session::AgentSession`]s behind the `stratum serve`
/// JSON-RPC socket.
pub mod serve_handler_agent;
/// Composable middleware layers wrapping a [`serve_server::ServeHandler`].
///
/// Provides [`serve_middleware::RateLimitedHandler`],
/// [`serve_middleware::AuthTokenHandler`],
/// [`serve_middleware::LoggingHandler`], and the
/// [`serve_middleware::chain`] helper for outermost-first composition.
pub mod serve_middleware;
/// `stratum serve` JSON-RPC 2.0 wire-protocol data shapes.
pub mod serve_protocol;
/// Synchronous `stratum serve` JSON-RPC dispatch server over Unix or TCP loopback.
pub mod serve_server;
/// Four-tier settings loader (managed/user/project/local, plan/30 §10).
pub mod settings_loader;
/// `.stratumignore` matcher for fs.* + glob + grep (plan/30 §3.1).
pub mod stratumignore;
/// Subagent primitive: TOML schema + loader + builtin seed.
pub mod subagent;
/// Default-on opt-out telemetry payload shape + allowlist guard.
pub mod telemetry;
/// Composite tier classifier (low / medium / high).
pub mod tier;
/// `McpToolDispatcher` — bridges an MCP `tools/call` client into the
/// `ToolDispatcher` trait (Phase 6 scaffold).
///
/// Routes `mcp.<server>.<verb>` calls through the same
/// `RegistryDispatcher` as the local `fs.read` / `shell.exec` tools.
pub mod tool_dispatcher_mcp;
/// Concrete `shell.exec` + `fs.read` tool dispatchers (Phase 3 v2).
pub mod tool_dispatchers;
/// Tool invocation data shape + dispatcher trait (Phase 3 scaffold).
pub mod tool_invocation;
/// Third-party tool plugin SDK — manifest + filesystem registry + subprocess dispatcher.
pub mod tool_plugin;
/// Per-tool-call timeout policy + RAII timer guard.
pub mod tool_timeout;
/// Tool registry and capability matrix.
pub mod tools;
/// On-disk conversation-transcript shape + atomic JSON store.
pub mod transcript;
/// `stratum self-update` channel-manifest data shape (Phase 1 scaffold).
pub mod update_manifest;
/// whisper.cpp subprocess scaffold for Phase 5 voice-in transcription.
pub mod whisper;
/// Workspace / project discovery (`stratum.toml`, `.stratumignore`).
pub mod workspace;

pub use agent_factory::{
    default_factory_with_dispatchers, AgentFactory, AgentFactoryConfig, AgentFactoryError,
    PermissionMode,
};
pub use agent_handoff::{
    parse_handoff_marker, AgentHandoff, AgentRegistry, HandoffError, HandoffPolicy, HandoffResult,
    HandoffStep, OrdRole, ParallelPolicy, ParallelResult, RoleResult,
};
pub use agent_loop::{
    AgentLoop, AgentLoopBuildError, AgentLoopBuilder, AgentLoopConfig, TurnContext, TurnResult,
    TurnResultError,
};
pub use agent_registry_loader::{
    AgentRegistryLoadError, AgentRegistryLoader, EchoProviderResolver, LoadFailure, LoadReport,
    ProviderResolveError, ProviderResolver, SkipReason,
};
pub use agent_session::{AgentSession, SessionError};
pub use agents::{AgentBudget, AgentDef, AgentLoader};
pub use anthropic_api_judge::{synth_anthropic_body, AnthropicApiConfig, AnthropicApiJudge};
pub use budget::{BudgetCheck, BudgetTracker};
pub use budget_meter::{estimate_cost_micro_usd, BudgetMeter, BudgetMeterError, BudgetTotals};
pub use cancel::CancelToken;
pub use cancel_cascade::{CancelError, CancelReason, CascadeToken, DeadlineGuard};
pub use catalog_sync::{
    CatalogSync, CatalogSyncConfig, CatalogSyncError, SyncReport, DEFAULT_CATALOG_CHANNEL,
    DEFAULT_CATALOG_MAX_BYTES, DEFAULT_CATALOG_TIMEOUT, DEFAULT_CATALOG_URL,
};
pub use claude_cli_judge::{
    parse_verdict_line, synth_prompt, ClaudeCliJudge, JudgeError, JudgePrompt, JudgeResponse,
    JudgeVerdict,
};
pub use conversation::{
    next_state, validate_history, TurnDriver, TurnEvent, TurnFsmError, TurnOutcome, TurnState,
    TurnTransition,
};
pub use crash_report::{
    build_bundle, load_bundle, redact_log_lines, redact_path_user, write_bundle, CrashBundle,
    CrashBundleConfig, CrashBundleError, CrashEnv, CRASH_BUNDLE_SCHEMA_VERSION,
};
pub use download::{InstallReport, ModelInstaller};
pub use embedder::{
    cosine_similarity, top_k, EmbedError, Embedder, EmbeddingDim, EmbeddingVector, HashEmbedder,
    InMemoryVectorStore,
};
pub use eval_runner::{
    EvalCase, EvalLoadError, EvalReport, EvalReportError, EvalRun, EvalRunner, EvalSuite,
    EVAL_SUITE_SCHEMA_VERSION,
};
pub use event_log::{
    Event, EventClock, EventEmitter, EventLogError, EventRecord, EventSink, FixedEventClock,
    JsonlEventSink, MemoryEventSink, SystemEventClock,
};
pub use gate::{LoadedModel, MemoryGate, DEFAULT_MARGIN_MIB};
pub use i18n::{
    parse_simple_ftl, FluentArg, I18nBundle, I18nError, LocaleId, LocaleIdError, Message,
    MessageCatalog, MessageId, MessageIdError,
};
pub use injection::{fence, is_suspicious, suspicion_score, FenceSource, SUSPICION_THRESHOLD};
pub use install::{
    backup_path, load_with_migration, restore_backup, save_atomic, InstallIoError,
    InstallLoadError, InstalledToml, TierInputs, CURRENT_SCHEMA_VERSION,
};
pub use intent_router::{
    fallback_intent, Intent, IntentPattern, IntentRouter, IntentRouterError, IntentRule,
    RoutedIntent, SuggestedRole,
};
#[cfg(feature = "provider-llama-cpp")]
pub use llama_provider::LlamaCppProvider;
pub use mcp::{
    McpServeTransport, McpServerConfig, McpServerExpose, McpServerSet, McpServerStatus,
    McpTransport,
};
pub use mcp_jsonrpc::{
    ClientCapabilities, ClientInfo, McpInitializeParams, McpInitializeResult, McpJsonRpcClient,
    McpRpcError, RootsClientCapability, ServerCapabilities, ServerInfo, ToolCallResult,
    ToolContentBlock, ToolDescriptor, ToolsClientCapability,
};
pub use model_catalog::{
    ArtifactRef, ArtifactRefError, CatalogError, ModelCatalog, ModelEntry, ModelSlug,
    ModelSlugError, ModelTask, ModelTier, MODEL_CATALOG_SCHEMA_VERSION,
};
pub use model_resolver::{BlobFetcher, ModelResolver, ResolveModelError};
pub use observability::{
    format_tokens_per_second, RoleStep, RoleTimer, TurnId, TurnIdGen, TurnMetrics, TurnRecorder,
};
pub use openai::{
    audio_format_to_mime, loop_factory_from_agent_factory, LoopFactory, OpenAIChatMessage,
    OpenAIChatRequest, OpenAIChatResponse, OpenAIChoice, OpenAIContentPart, OpenAIDelta,
    OpenAIImageUrl, OpenAIInputAudio, OpenAIMessageContent, OpenAIModelEntry, OpenAIModelList,
    OpenAIServer, OpenAIServerConfig, OpenAIServerHandle, OpenAIStreamChoice, OpenAIStreamChunk,
    OpenAIUsage,
};
pub use paths::Paths;
pub use permission_prompt::{
    evaluate as evaluate_permission, request_digest, AllowAllResponder, DenyAllResponder,
    PendingPrompt, PermissionDecision, PermissionRequest, PermissionStore, PromptId, PromptIdGen,
    PromptResponder, ScriptedResponder,
};
pub use piper::{PiperError, PiperSubprocess};
pub use plan_mode::{
    enforce_plan_mode_on_request, explain_denied, filter_capability_matrix_for_plan,
    is_capability_allowed_in_plan_mode, PlanMode, PlanModeError, PlanModeGuard,
    PLAN_MODE_DENIED_CAPS,
};
pub use probe::{GpuBackend, HardwareProbe};
pub use prompt_cache::{
    fingerprint_inputs, PromptCache, PromptCacheEntry, PromptCacheError, PromptCacheKey,
    PromptHash, PromptHashError,
};
pub use prompt_template::{
    render as render_prompt_template, render_with_budget as render_prompt_template_with_budget,
    truncate_tool_results, PromptBudget, PromptContext, PromptRenderError, PromptTemplate,
    RenderedPrompt, TemplateId, TemplateIdError, ToolResultSnippet,
};
pub use prompts::{system_prompt, PromptRole};
pub use provider::{ChatHistoryTurn, EchoProvider, GenerateRequest, Provider, SamplerParams};
pub use provider_cache::{CacheError, CacheSlot, ProviderCache, ProviderKey};
pub use rag::{chunk_document, Chunk, ChunkPlan, ChunkSpan, DocumentId, RagDocument, RagIndex};
pub use rag_index_builder::{
    BuildStats, BuiltIndex, RagBuildError, RagIndexBuilder, DEFAULT_MAX_FILE_BYTES,
};
pub use rag_query::{
    split_vector_key, QueryError, QueryReport, RagQuery, RagQueryConfig, RankedPassage,
};
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
pub use serve_handler_agent::{AgentServeHandler, ServeHandlerError};
pub use serve_middleware::{
    chain as chain_serve_layers, AuthTokenHandler, LoggingHandler, RateLimitedHandler, ServeLayer,
    DEFAULT_AUTH_HEADER, SERVE_ERR_RATE_LIMITED, SERVE_ERR_UNAUTHORIZED,
};
pub use serve_protocol::{
    parse_request as parse_serve_request, render_response as render_serve_response,
    ParseRequestError, RequestId, RunTurnParams, ServeMethod, ServeRequest, ServeResponse,
    ServeResponseBody, SERVE_ERR_INTERNAL, SERVE_ERR_INVALID, SERVE_ERR_METHOD, SERVE_ERR_PARAMS,
    SERVE_ERR_PARSE,
};
pub use serve_server::{
    make_default_handler, EchoServeHandler, ServeBind, ServeConfig, ServeError, ServeHandle,
    ServeHandler, ServeServer,
};
pub use telemetry::{
    build_payload, payload_is_allowlisted, redact, AnonInstallId, AnonInstallIdError, CpuArchTag,
    OsTag, ReleaseChannel, TelemetryConfig, TelemetryError, TelemetryEventKind, TelemetryPayload,
    TELEMETRY_SCHEMA_VERSION,
};
pub use tier::Tier;
pub use tool_dispatcher_mcp::{parse_mcp_tool_id, McpToolDispatcher};
pub use tool_dispatchers::{
    base64_encode, default_dispatchers, sniff_image_mime, FsReadToolDispatcher,
    ReadAudioToolDispatcher, ShellToolDispatcher, SHELL_DEFAULT_ALLOWLIST,
};
pub use tool_invocation::{
    quick_dispatch, DenyDispatcher, DispatchError, EchoDispatcher, RegistryDispatcher,
    ToolDispatcher, ToolInvocation, ToolResult,
};
pub use tool_plugin::{
    FileSystemPluginRegistry, ToolPlugin, ToolPluginDispatcher, ToolPluginLoadError,
    ToolPluginManifest, ToolPluginRegistry, DEFAULT_PLUGIN_TIMEOUT_MS,
};
pub use tool_timeout::{
    record_outcome, run_with_timeout, ToolTimeoutError, ToolTimeoutGuard, ToolTimeoutPolicy,
};
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
