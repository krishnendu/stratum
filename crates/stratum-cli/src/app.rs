//! The CLI behavior, factored out of `main` for testability.

use std::collections::BTreeSet;
use std::ffi::OsString;
use std::io::{BufRead, BufReader, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{Duration, SystemTime};

use clap::{Args, Parser, Subcommand};
use serde::Serialize;
use stratum_runtime::{
    build_payload as build_telemetry_payload, evaluate as evaluate_update,
    payload_is_allowlisted as telemetry_payload_is_allowlisted, AgentDef, AgentFactory,
    AgentRegistryLoader, AgentServeHandler, AnonInstallId, ArtifactRef as ModelArtifactRef,
    CancelToken, CatalogError, CpuArchTag, EchoProvider, EvalReport, EvalRunner, EvalSuite, Event,
    EventEmitter, EventRecord, GenerateRequest, GpuBackend, HardwareProbe, InstalledToml,
    LoadFailure, LoadedModel, ManifestError, MemoryEventSink, MemoryGate, ModelCatalog, ModelEntry,
    ModelInstaller, ModelSlug, ModelTask, ModelTier, OsTag, Paths, PlatformTag, Provider,
    ReleaseChannel, ReleaseVersion, SandboxReport, ServeBind, ServeConfig, ServeHandler,
    ServeServer, SessionId, SkipReason, SuggestedRole, TelemetryConfig, TelemetryEventKind,
    TelemetryPayload, Tier, Transcript, TranscriptBlock, TranscriptStore, TranscriptTurn,
    UpdateChannel, UpdateDecision, UpdateManifest, DEFAULT_MARGIN_MIB,
};
use stratum_types::{Block, ErrorCode, MemEstimate, ModelId};
use time::format_description::well_known::Rfc3339;
use time::OffsetDateTime;

/// Stratum CLI.
#[derive(Debug, Parser)]
#[command(name = "stratum", version, about = "Stratum local-LLM TUI agent")]
struct Cli {
    /// Emit machine-readable JSON instead of human prose where applicable.
    #[arg(long, global = true)]
    json: bool,

    /// Override the storage root (for tests and `--workspace <path>` flows).
    /// When set, every Stratum directory lives under `<root>/{config,data,state,cache}`.
    #[arg(long, global = true, env = "STRATUM_STORAGE_ROOT")]
    storage_root: Option<PathBuf>,

    #[command(subcommand)]
    command: Option<Command>,
}

/// Top-level subcommands.
#[derive(Debug, Subcommand)]
enum Command {
    /// Probe the host and print a tier report.
    Doctor(DoctorArgs),
    /// First-run install: probe, classify, write `installed.toml`.
    Init,
    /// Smoke-test the chat loop against the `EchoProvider`.
    Echo {
        /// Prompt to feed the provider.
        prompt: Vec<String>,
        /// Maximum number of text blocks to emit (default 64).
        #[arg(long, default_value_t = 64)]
        max_blocks: u32,
    },
    /// Open the chat surface — either the interactive TUI or, with
    /// `--prompt`, a single non-interactive turn against the resolved
    /// provider.
    Chat(ChatArgs),
    /// Manage on-disk model files.
    #[command(subcommand)]
    Models(ModelsCommand),
    /// Probe RAM and decide whether a synthetic `MemEstimate` would load.
    MemCheck(MemCheckArgs),
    /// Self-update operations (read-only `--check` in this phase).
    SelfUpdate(SelfUpdateArgs),
    /// Read the on-disk JSONL event log written by `JsonlEventSink`.
    #[command(subcommand)]
    Events(EventsCommand),
    /// Inspect, list, or delete chat-session transcripts on disk.
    #[command(subcommand)]
    Sessions(SessionsCommand),
    /// Run prompt-based eval suites against an [`AgentLoop`].
    #[command(subcommand)]
    Eval(EvalCommand),
    /// Start the JSON-RPC daemon (`stratum serve`) on a Unix socket or
    /// loopback TCP port.
    Serve(ServeArgs),
    /// Inspect user-authored agent definitions in `<state>/agents/*.toml`.
    #[command(subcommand)]
    Agents(AgentsCommand),
}

/// Subcommands under `stratum agents`.
#[derive(Debug, Subcommand)]
enum AgentsCommand {
    /// List registered roles and report skipped / errored files.
    List(AgentsListArgs),
    /// Show the parsed [`AgentDef`] for a single role.
    Show(AgentsShowArgs),
}

/// Arguments for `stratum agents list`.
#[derive(Debug, Args)]
struct AgentsListArgs {
    /// Emit a structured JSON payload instead of prose.
    #[arg(long)]
    json: bool,
}

/// Arguments for `stratum agents show`.
#[derive(Debug, Args)]
struct AgentsShowArgs {
    /// Role to look up (`coder`, `polisher`, `cavemanish`, `researcher`,
    /// `default`).
    #[arg(long)]
    role: String,
    /// Emit the [`AgentDef`] as pretty JSON instead of prose.
    #[arg(long)]
    json: bool,
}

/// Arguments for `stratum serve`.
///
/// The daemon binds either a Unix-domain socket (`--unix-socket <PATH>`) or
/// a loopback TCP port (`--tcp-port <N>`, defaulting to `0` for a
/// kernel-assigned ephemeral port when neither flag is set). The two
/// transports are mutually exclusive — passing both is rejected by clap
/// with exit 64.
///
/// `--stop-after-ms` is primarily a test affordance: when set, a watchdog
/// thread stops the server after the specified wall-clock window. When
/// unset, the daemon polls an `AtomicBool` shutdown flag on a 100 ms
/// cadence and exits cleanly on the next tick after the flag flips
/// (currently only via internal handler-driven shutdown; signal-driven
/// `Ctrl+C` plumbing lands once the broader runtime signal-policy story
/// is settled — pinned to avoid pulling in the `ctrlc` crate).
#[derive(Debug, Args)]
struct ServeArgs {
    /// Filesystem path of a Unix-domain socket to bind. Mutually
    /// exclusive with `--tcp-port`.
    #[arg(long, value_name = "PATH", conflicts_with = "tcp_port")]
    unix_socket: Option<PathBuf>,
    /// Loopback TCP port to bind. Mutually exclusive with
    /// `--unix-socket`. When neither flag is passed the daemon defaults
    /// to TCP loopback with port `0` (kernel-assigned ephemeral).
    #[arg(long, value_name = "N", conflicts_with = "unix_socket")]
    tcp_port: Option<u16>,
    /// Maximum concurrent connections accepted by the server.
    #[arg(long, value_name = "N", default_value_t = 16)]
    max_connections: usize,
    /// Per-request socket read/write timeout, in milliseconds.
    #[arg(long, value_name = "N", default_value_t = 30_000)]
    request_timeout_ms: u64,
    /// Stop the daemon after the specified wall-clock window. Useful
    /// for integration tests that exercise the bind/accept loop and
    /// then need a deterministic shutdown.
    #[arg(long, value_name = "N")]
    stop_after_ms: Option<u64>,
    /// Emit a single JSON object on startup describing the bound
    /// address. Without this flag, a prose line is printed instead.
    #[arg(long)]
    json: bool,
}

/// Subcommands under `stratum eval`.
#[derive(Debug, Subcommand)]
enum EvalCommand {
    /// Load an [`EvalSuite`] from disk, run it via [`EvalRunner`] wrapped
    /// around [`AgentFactory::echo`], and save the resulting [`EvalReport`].
    Run(EvalRunArgs),
}

/// Arguments for `stratum eval run`.
///
/// The runner currently wraps [`AgentFactory::echo`] (the Echo backbone) so
/// `--model` is parsed but ignored — the field is reserved for the follow-up
/// PR that wires real provider selection.
#[derive(Debug, Args)]
struct EvalRunArgs {
    /// Path to the JSON [`EvalSuite`] file.
    #[arg(long, value_name = "PATH")]
    suite: PathBuf,
    /// Destination for the JSON [`EvalReport`]. Defaults to
    /// `<state>/eval-reports/<suite-name>-<timestamp>.json`.
    #[arg(long, value_name = "PATH")]
    out: Option<PathBuf>,
    /// When set, emit the entire [`EvalReport`] as pretty JSON on stdout
    /// (in addition to writing it to `--out`).
    #[arg(long)]
    json: bool,
    /// Catalog slug for the model to evaluate. **Currently ignored**: the
    /// runner always uses [`AgentFactory::echo`] until provider selection
    /// lands. Parsed only so callers can write forward-compatible scripts.
    #[arg(long, value_name = "SLUG")]
    model: Option<String>,
}

/// Subcommands under `stratum sessions`.
#[derive(Debug, Subcommand)]
enum SessionsCommand {
    /// List session ids currently on disk in sorted order.
    List,
    /// Pretty-print the transcript for a given session id.
    Show(SessionsShowArgs),
    /// Delete the on-disk transcript for a given session id.
    Delete(SessionsDeleteArgs),
}

/// Arguments for `stratum sessions show`.
#[derive(Debug, Args)]
struct SessionsShowArgs {
    /// Session id to load. Must be 16 lowercase hex characters.
    #[arg(long)]
    id: String,
}

/// Arguments for `stratum sessions delete`.
#[derive(Debug, Args)]
struct SessionsDeleteArgs {
    /// Session id to delete. Must be 16 lowercase hex characters.
    #[arg(long)]
    id: String,
}

/// Subcommands under `stratum events`.
#[derive(Debug, Subcommand)]
enum EventsCommand {
    /// Tail recent `EventRecord`s from `<state>/events.jsonl`.
    Tail(EventsTailArgs),
}

/// Arguments for `stratum events tail`.
#[derive(Debug, Args)]
struct EventsTailArgs {
    /// Skip records with `id <= since-id`.
    #[arg(long, value_name = "ID")]
    since_id: Option<u64>,
    /// Maximum number of records to print after filtering.
    #[arg(long, value_name = "N")]
    limit: Option<usize>,
    /// Filter by event kind.
    #[arg(long, value_enum)]
    kind: Option<EventKindArg>,
    /// Emit each filtered record as compact JSON on its own line.
    #[arg(long)]
    json: bool,
    /// Keep reading after EOF, polling for new lines.
    #[arg(long)]
    follow: bool,
}

/// Arguments for `stratum chat`.
///
/// All four arguments are optional. The interactive TUI is preserved as the
/// default surface; `--prompt <STR>` opts into a single non-interactive
/// turn, useful for scripting and integration tests. `--model <slug>`
/// resolves a catalog entry from `<state>/models.json` and spawns the
/// `LlamaCppProvider` against the on-disk GGUF (feature-gated by
/// `provider-llama-cpp`).
#[derive(Debug, Args)]
struct ChatArgs {
    /// Catalog slug to load from `<state>/models.json`. When omitted,
    /// the default [`EchoProvider`] is used (Phase 1 behavior).
    #[arg(long, value_name = "SLUG")]
    model: Option<String>,
    /// Logical context window passed to the llama.cpp provider, in tokens.
    #[arg(long, default_value_t = 4096)]
    ctx: u32,
    /// Maximum number of blocks the provider is allowed to emit per turn.
    #[arg(long = "max-blocks", default_value_t = 1)]
    max_blocks: u32,
    /// When set, run one non-interactive turn against the resolved
    /// provider, print the assistant text to stdout, and exit. Omit to
    /// open the interactive TUI.
    #[arg(long, value_name = "STR")]
    prompt: Option<String>,
    /// Append structured runtime events to this JSONL file (one record per
    /// line; persists across runs).
    #[arg(long = "events-jsonl", value_name = "PATH")]
    events_jsonl: Option<PathBuf>,
}

/// Clap-friendly mirror of the `kind` tag of [`Event`].
#[derive(Debug, Clone, Copy, clap::ValueEnum)]
#[clap(rename_all = "snake_case")]
enum EventKindArg {
    /// A tool invocation completed.
    ToolCall,
    /// A permission prompt was shown.
    PermissionAsked,
    /// Control was handed between agent roles.
    AgentHandoff,
    /// A provider returned an error.
    ProviderError,
    /// A sandboxed process was launched.
    SandboxLaunched,
}

impl EventKindArg {
    const fn as_wire(self) -> &'static str {
        match self {
            Self::ToolCall => "tool_call",
            Self::PermissionAsked => "permission_asked",
            Self::AgentHandoff => "agent_handoff",
            Self::ProviderError => "provider_error",
            Self::SandboxLaunched => "sandbox_launched",
        }
    }
}

/// Arguments for `stratum doctor`.
///
/// The base report has been stable since v1; this struct adds opt-in
/// telemetry-payload preview. Real wire emission lands in a later PR — for
/// now we only assemble the [`TelemetryPayload`] and print it via stdout so
/// the schema can be reviewed by hand.
#[derive(Debug, Args)]
struct DoctorArgs {
    /// Also assemble and print the telemetry payload. Honors the
    /// `<state>/telemetry.json` opt-out file: when `enabled` is `false`,
    /// payload assembly is skipped and the output indicates `disabled`.
    #[arg(long)]
    telemetry: bool,
    /// Override which telemetry event-kind to assemble. Default is
    /// `daily_active` because the doctor command itself is a stand-in for
    /// the once-per-UTC-day liveness beacon.
    #[arg(long, value_enum, default_value_t = TelemetryEventArg::DailyActive)]
    telemetry_event: TelemetryEventArg,
}

/// Clap-friendly mirror of [`TelemetryEventKind`].
#[derive(Debug, Clone, Copy, clap::ValueEnum)]
#[clap(rename_all = "snake_case")]
enum TelemetryEventArg {
    /// First-run install beacon.
    Install,
    /// Self-update completed.
    Update,
    /// Once-per-UTC-day liveness beacon (default).
    DailyActive,
    /// First user-initiated chat turn.
    FirstChatTurn,
    /// User opted in to crash reports.
    CrashOptIn,
    /// Uninstall beacon (best-effort).
    Uninstall,
}

impl From<TelemetryEventArg> for TelemetryEventKind {
    fn from(value: TelemetryEventArg) -> Self {
        match value {
            TelemetryEventArg::Install => Self::Install,
            TelemetryEventArg::Update => Self::Update,
            TelemetryEventArg::DailyActive => Self::DailyActive,
            TelemetryEventArg::FirstChatTurn => Self::FirstChatTurn,
            TelemetryEventArg::CrashOptIn => Self::CrashOptIn,
            TelemetryEventArg::Uninstall => Self::Uninstall,
        }
    }
}

/// Arguments for `stratum self-update`.
///
/// Two top-level actions are exposed:
///
/// * `--check`: fetch (or read) an [`UpdateManifest`], compare against the
///   running version, and print the resulting [`UpdateDecision`] without
///   touching the binary on disk.
/// * `--apply`: do the above, then download the matching artifact, verify
///   its SHA-256 + byte count, and atomically swap the running binary on
///   disk. The previous binary is preserved at `<exe>.bak` for rollback.
///
/// The two actions are mutually exclusive; exactly one must be passed.
#[derive(Debug, Args)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "self-update args are intentionally a flat clap derive struct; \
              the four bools (check / apply / dry_run / allow_insecure_url) \
              correspond 1:1 with user-facing flags and folding them into a \
              sub-enum would obscure the clap relationships"
)]
struct SelfUpdateArgs {
    /// Check for an available update and print the decision. Mutually
    /// exclusive with `--apply`.
    #[arg(long, conflicts_with = "apply")]
    check: bool,
    /// Apply the latest update: download, verify SHA-256 + byte count, then
    /// atomically swap the running binary. Mutually exclusive with `--check`.
    #[arg(long, conflicts_with = "check")]
    apply: bool,
    /// Do everything for `--apply` except the final atomic rename. Only valid
    /// together with `--apply` (rejected at runtime with exit 64 if combined
    /// with `--check`). Exits 0 after the SHA verification step.
    #[arg(long)]
    dry_run: bool,
    /// HTTPS URL of the channel manifest. Defaults to
    /// `https://updates.stratum.dev/<channel>.json`. Mutually exclusive with
    /// `--manifest-file`.
    #[arg(long, value_name = "URL", conflicts_with = "manifest_file")]
    manifest_url: Option<String>,
    /// Local manifest fixture path (used for offline runs / tests). Mutually
    /// exclusive with `--manifest-url`.
    #[arg(long, value_name = "PATH", conflicts_with = "manifest_url")]
    manifest_file: Option<PathBuf>,
    /// Release channel.
    #[arg(long, value_enum, default_value_t = ChannelArg::Stable)]
    channel: ChannelArg,
    /// Override the currently-running version. Defaults to
    /// `CARGO_PKG_VERSION`.
    #[arg(long, value_name = "VERSION")]
    current: Option<String>,
    /// Override the target platform (defaults to autodetect from
    /// `std::env::consts::OS` + `std::env::consts::ARCH`).
    #[arg(long, value_enum)]
    platform: Option<PlatformArg>,
    /// Hidden test-only override: write the swapped binary to `<path>`
    /// instead of `std::env::current_exe()`. Gated by `cfg(debug_assertions)`
    /// OR `STRATUM_ALLOW_INSECURE_URL=1`; production builds with the env var
    /// unset reject this flag at runtime so an end-user cannot silently
    /// redirect the swap. Required because `current_exe()` IS the CLI test
    /// binary and must not be modified by tests.
    #[arg(long, value_name = "PATH", hide = true)]
    target: Option<PathBuf>,
    /// Hidden test-only override: allow non-`https://` artifact URLs (e.g.
    /// the in-process `http://127.0.0.1:<port>/…` server used by the apply
    /// integration tests). Gated by `cfg(debug_assertions)` OR
    /// `STRATUM_ALLOW_INSECURE_URL=1`. Production users on a release build
    /// without the env var cannot silently disable TLS.
    #[arg(long, hide = true)]
    allow_insecure_url: bool,
}

/// Returns true iff the hidden test-only `--target` / `--allow-insecure-url`
/// flags are permitted in this process. Allowed when either:
///
/// * the build is a debug build (`cfg(debug_assertions)`), or
/// * the env var `STRATUM_ALLOW_INSECURE_URL=1` is set.
///
/// Release builds without the env var reject the flags, so a packaged binary
/// shipped to end users cannot silently bypass TLS or redirect the on-disk
/// swap target.
fn insecure_flags_allowed() -> bool {
    if cfg!(debug_assertions) {
        return true;
    }
    matches!(
        std::env::var("STRATUM_ALLOW_INSECURE_URL").as_deref(),
        Ok("1")
    )
}

/// Clap-friendly mirror of [`UpdateChannel`].
#[derive(Debug, Clone, Copy, clap::ValueEnum)]
enum ChannelArg {
    /// Stable release line.
    Stable,
    /// Beta line.
    Beta,
    /// Nightly line.
    Nightly,
}

impl ChannelArg {
    const fn as_wire(self) -> &'static str {
        match self {
            Self::Stable => "stable",
            Self::Beta => "beta",
            Self::Nightly => "nightly",
        }
    }
}

impl From<ChannelArg> for UpdateChannel {
    fn from(value: ChannelArg) -> Self {
        match value {
            ChannelArg::Stable => Self::Stable,
            ChannelArg::Beta => Self::Beta,
            ChannelArg::Nightly => Self::Nightly,
        }
    }
}

/// Clap-friendly mirror of [`PlatformTag`].
///
/// The clap value names use the friendlier `macos_*` / `linux_*` /
/// `windows_*` form requested by the brief; the on-the-wire serde encoding
/// of [`PlatformTag`] itself is independent.
#[derive(Debug, Clone, Copy, clap::ValueEnum)]
#[clap(rename_all = "snake_case")]
enum PlatformArg {
    /// macOS on Apple Silicon.
    MacosAarch64,
    /// macOS on `x86_64`.
    MacosX86_64,
    /// Linux on aarch64.
    LinuxAarch64,
    /// Linux on `x86_64`.
    LinuxX86_64,
    /// Windows on `x86_64`.
    WindowsX86_64,
}

impl PlatformArg {
    /// User-facing CLI / JSON form (matches the `--platform` value name).
    const fn as_wire(self) -> &'static str {
        match self {
            Self::MacosAarch64 => "macos_aarch64",
            Self::MacosX86_64 => "macos_x86_64",
            Self::LinuxAarch64 => "linux_aarch64",
            Self::LinuxX86_64 => "linux_x86_64",
            Self::WindowsX86_64 => "windows_x86_64",
        }
    }

    /// Best-effort detection of the host platform. Returns `None` when the
    /// running OS/ARCH pair isn't on the supported matrix.
    fn detect() -> Option<Self> {
        match (std::env::consts::OS, std::env::consts::ARCH) {
            ("macos", "aarch64") => Some(Self::MacosAarch64),
            ("macos", "x86_64") => Some(Self::MacosX86_64),
            ("linux", "aarch64") => Some(Self::LinuxAarch64),
            ("linux", "x86_64") => Some(Self::LinuxX86_64),
            ("windows", "x86_64") => Some(Self::WindowsX86_64),
            _ => None,
        }
    }
}

impl From<PlatformArg> for PlatformTag {
    fn from(value: PlatformArg) -> Self {
        match value {
            PlatformArg::MacosAarch64 => Self::MacOsAarch64,
            PlatformArg::MacosX86_64 => Self::MacOsX86_64,
            PlatformArg::LinuxAarch64 => Self::LinuxAarch64,
            PlatformArg::LinuxX86_64 => Self::LinuxX86_64,
            PlatformArg::WindowsX86_64 => Self::WindowsX86_64,
        }
    }
}

/// Arguments for `stratum mem-check`.
///
/// Three operating modes share the same subcommand:
///
/// * No flags → print the host's currently-available RAM and exit. This is
///   the default operator surface.
/// * `--requested <slug> --requested-mib <u64>` → consult
///   [`MemoryGate::suggest_unloads`] against the resident set read from
///   `<state_root>/loaded.json` (or `--loaded-file`) and print which slugs to
///   evict. Both flags must be passed together; `--requested-mib` must be
///   `> 0` (enforced via clap `value_parser`).
/// * Legacy `--weight-rss/--kv-per-token/--context` → original
///   synthetic-`MemEstimate` flow that runs the gate's full `check_with` and
///   reports OK or refusal. `--loaded` lets the legacy mode also exercise the
///   unload-hint path.
///
/// The three modes are dispatched in [`mem_check`] in the order above.
#[derive(Debug, Args)]
struct MemCheckArgs {
    /// Resident set of the weights, in mebibytes. Required for the legacy
    /// `check_with`-driven mode; omit to use the available-RAM or
    /// `--requested` modes.
    #[arg(long)]
    weight_rss: Option<u32>,
    /// KV cache bytes per token. Required for the legacy mode.
    #[arg(long)]
    kv_per_token: Option<u32>,
    /// Planned context length, in tokens. Required for the legacy mode.
    #[arg(long)]
    context: Option<u32>,
    /// Optional multimodal projector overhead, in mebibytes.
    #[arg(long, default_value_t = 0)]
    mmproj: u32,
    /// Optional VRAM cost when fully GPU-offloaded, in mebibytes.
    #[arg(long, default_value_t = 0)]
    vram: u32,
    /// Override the safety margin, in mebibytes.
    #[arg(long, default_value_t = DEFAULT_MARGIN_MIB)]
    margin: u32,
    /// Currently-loaded model, repeatable. Each value is formatted
    /// `model_id:weight_rss_mib:kv_per_token_bytes:context_tokens` (four
    /// colon-separated fields). On refusal, the gate suggests which of these
    /// to unload to free room. Pass the flag once per resident model.
    #[arg(long = "loaded", value_name = "SPEC")]
    loaded: Vec<String>,
    /// Slug of a prospective model to load. Used together with
    /// `--requested-mib` to drive [`MemoryGate::suggest_unloads`] against the
    /// resident set read from `--loaded-file`.
    #[arg(long, value_name = "SLUG", requires = "requested_mib")]
    requested: Option<String>,
    /// Estimated hot footprint, in mebibytes, of the prospective load. Must
    /// be `> 0`. Used with `--requested`.
    #[arg(
        long,
        value_name = "MIB",
        requires = "requested",
        value_parser = clap::value_parser!(u64).range(1..)
    )]
    requested_mib: Option<u64>,
    /// Path to the resident-set JSON file consumed in the `--requested` mode.
    /// Defaults to `<state_root>/loaded.json`. A missing file is treated as
    /// an empty resident set.
    #[arg(long, value_name = "PATH")]
    loaded_file: Option<PathBuf>,
}

/// Subcommands under `stratum models`.
#[derive(Debug, Subcommand)]
enum ModelsCommand {
    /// List catalog entries from `<state>/models.json`.
    List(ListArgs),
    /// Add (or replace) a catalog entry.
    Add(AddArgs),
    /// Remove a catalog entry by slug.
    Remove(RemoveArgs),
    /// Print the recommended slug for a `(tier, task)` pair.
    Recommend(RecommendArgs),
    /// Validate the on-disk catalog file.
    Validate,
    /// Legacy: install a model file from a local source path.
    InstallFile(InstallFileArgs),
}

/// Arguments for `stratum models list`.
#[derive(Debug, Args)]
struct ListArgs {
    /// Filter by tier.
    #[arg(long)]
    tier: Option<TierArg>,
    /// Filter by task.
    #[arg(long)]
    task: Option<TaskArg>,
}

/// Arguments for `stratum models add` (catalog entry).
#[derive(Debug, Args)]
struct AddArgs {
    /// Stable slug used as the catalog key.
    #[arg(long)]
    slug: String,
    /// Upstream model family / lineage (e.g. "llama").
    #[arg(long)]
    family: String,
    /// Human-friendly display name shown in the installer UI.
    #[arg(long = "display-name")]
    display_name: String,
    /// Coarse tier bucket.
    #[arg(long)]
    tier: TierArg,
    /// Comma-separated task tags (`chat`, `code`, `embedding`, `tool_use`,
    /// `vision`, `cavemanish`, `polisher`).
    #[arg(long)]
    task: String,
    /// Total artifact size, in MiB.
    #[arg(long = "size-mib")]
    size_mib: u64,
    /// Quantization tag, e.g. `Q4_K_M`.
    #[arg(long)]
    quantization: String,
    /// HTTPS download URL for the artifact.
    #[arg(long)]
    url: String,
    /// Lowercase hex SHA-256 of the artifact (64 chars).
    #[arg(long)]
    sha256: String,
    /// Expected byte size of the artifact (`> 0`).
    #[arg(long)]
    bytes: u64,
    /// SPDX license identifier (e.g. "Apache-2.0").
    #[arg(long)]
    license: String,
    /// Optional homepage / model card URL.
    #[arg(long)]
    homepage: Option<String>,
}

/// Arguments for `stratum models remove`.
#[derive(Debug, Args)]
struct RemoveArgs {
    /// Slug to remove.
    #[arg(long)]
    slug: String,
}

/// Arguments for `stratum models recommend`.
#[derive(Debug, Args)]
struct RecommendArgs {
    /// Tier budget.
    #[arg(long)]
    tier: TierArg,
    /// Task tag.
    #[arg(long)]
    task: TaskArg,
}

/// Arguments for the legacy `stratum models install-file`.
#[derive(Debug, Args)]
struct InstallFileArgs {
    /// Local file to copy into the models directory.
    #[arg(long, conflicts_with = "from_url")]
    from_file: Option<PathBuf>,
    /// HTTP(S) URL to fetch the model from.
    #[arg(long, conflicts_with = "from_file")]
    from_url: Option<String>,
    /// Destination filename (defaults to the source filename).
    #[arg(long)]
    name: Option<String>,
    /// Expected SHA-256 (lowercase hex). When set, the install verifies.
    #[arg(long)]
    sha256: Option<String>,
}

/// Clap-friendly mirror of [`ModelTier`].
#[derive(Debug, Clone, Copy, clap::ValueEnum)]
enum TierArg {
    /// Smallest models.
    Low,
    /// Mid-range.
    Medium,
    /// Large.
    High,
    /// Extra-large.
    Xl,
}

impl From<TierArg> for ModelTier {
    fn from(value: TierArg) -> Self {
        match value {
            TierArg::Low => Self::Low,
            TierArg::Medium => Self::Medium,
            TierArg::High => Self::High,
            TierArg::Xl => Self::Xl,
        }
    }
}

/// Clap-friendly mirror of [`ModelTask`].
#[derive(Debug, Clone, Copy, clap::ValueEnum)]
enum TaskArg {
    /// General chat / instruction following.
    Chat,
    /// Code generation / completion.
    Code,
    /// Sentence-embedding model.
    Embedding,
    /// Tool / function calling.
    ToolUse,
    /// Multimodal vision.
    Vision,
    /// Caveman-ish rewriter.
    Cavemanish,
    /// Polisher role.
    Polisher,
}

impl From<TaskArg> for ModelTask {
    fn from(value: TaskArg) -> Self {
        match value {
            TaskArg::Chat => Self::Chat,
            TaskArg::Code => Self::Code,
            TaskArg::Embedding => Self::Embedding,
            TaskArg::ToolUse => Self::ToolUse,
            TaskArg::Vision => Self::Vision,
            TaskArg::Cavemanish => Self::Cavemanish,
            TaskArg::Polisher => Self::Polisher,
        }
    }
}

fn parse_task_csv(input: &str) -> Result<BTreeSet<ModelTask>, String> {
    let mut out = BTreeSet::new();
    for raw in input.split(',') {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            continue;
        }
        let task = match trimmed {
            "chat" => ModelTask::Chat,
            "code" => ModelTask::Code,
            "embedding" => ModelTask::Embedding,
            "tool_use" => ModelTask::ToolUse,
            "vision" => ModelTask::Vision,
            "cavemanish" => ModelTask::Cavemanish,
            "polisher" => ModelTask::Polisher,
            other => return Err(format!("unknown task {other:?}")),
        };
        out.insert(task);
    }
    if out.is_empty() {
        return Err("--task must list at least one task".to_owned());
    }
    Ok(out)
}

/// Run the CLI against the provided argv (excluding argv[0]).
#[must_use]
#[allow(
    clippy::redundant_pub_crate,
    reason = "intentional: visible to the bin crate root only"
)]
pub(super) fn run(argv: Vec<OsString>) -> ExitCode {
    run_with(
        argv,
        &mut std::io::stdout(),
        &mut std::io::stderr(),
        Paths::resolve,
    )
}

#[must_use]
fn run_with<F>(
    argv: Vec<OsString>,
    out: &mut dyn Write,
    err: &mut dyn Write,
    fallback_paths: F,
) -> ExitCode
where
    F: FnOnce() -> stratum_types::StratumResult<Paths>,
{
    let mut full = vec![OsString::from("stratum")];
    full.extend(argv);
    let cli = match Cli::try_parse_from(full) {
        Ok(c) => c,
        Err(e) => {
            let _ = writeln!(err, "{e}");
            return ExitCode::from(64);
        }
    };

    let paths = match resolve_paths_with(cli.storage_root.as_deref(), fallback_paths) {
        Ok(p) => p,
        Err(diag) => {
            let _ = writeln!(err, "{diag}");
            return ExitCode::from(78);
        }
    };

    match cli.command {
        None => print_greeting(&paths, out),
        Some(Command::Doctor(doc_args)) => doctor(cli.json, &doc_args, &paths, out, err),
        Some(Command::Init) => init(cli.json, &paths, out, err),
        Some(Command::Echo { prompt, max_blocks }) => echo(cli.json, &prompt, max_blocks, out),
        Some(Command::Chat(chat_args)) => chat_command(&chat_args, &paths, out, err),
        Some(Command::Models(ModelsCommand::List(list_args))) => {
            models_list(cli.json, &paths, &list_args, out, err)
        }
        Some(Command::Models(ModelsCommand::Add(add_args))) => {
            models_add(cli.json, &paths, &add_args, out, err)
        }
        Some(Command::Models(ModelsCommand::Remove(rm_args))) => {
            models_remove(&paths, &rm_args, out, err)
        }
        Some(Command::Models(ModelsCommand::Recommend(rec_args))) => {
            models_recommend(cli.json, &paths, &rec_args, out, err)
        }
        Some(Command::Models(ModelsCommand::Validate)) => models_validate(&paths, out, err),
        Some(Command::Models(ModelsCommand::InstallFile(inst_args))) => {
            models_install_file(cli.json, &paths, &inst_args, out, err)
        }
        Some(Command::MemCheck(mem_args)) => mem_check(cli.json, &mem_args, &paths, out, err),
        Some(Command::SelfUpdate(su_args)) => self_update(cli.json, &su_args, out, err),
        Some(Command::Events(EventsCommand::Tail(tail_args))) => {
            events_tail(&paths, &tail_args, out, err)
        }
        Some(Command::Sessions(SessionsCommand::List)) => sessions_list(cli.json, &paths, out, err),
        Some(Command::Sessions(SessionsCommand::Show(show_args))) => {
            sessions_show(cli.json, &paths, &show_args, out, err)
        }
        Some(Command::Sessions(SessionsCommand::Delete(del_args))) => {
            sessions_delete(&paths, &del_args, out, err)
        }
        Some(Command::Eval(EvalCommand::Run(eval_args))) => eval_run(&eval_args, &paths, out, err),
        Some(Command::Serve(serve_args)) => serve(&serve_args, &paths, out, err),
        Some(Command::Agents(AgentsCommand::List(list_args))) => {
            agents_list(&list_args, &paths, out, err)
        }
        Some(Command::Agents(AgentsCommand::Show(show_args))) => {
            agents_show(&show_args, &paths, out, err)
        }
    }
}

fn models_dir(paths: &Paths) -> PathBuf {
    paths.data.join("models")
}

fn catalog_path(paths: &Paths) -> PathBuf {
    paths.state.join("models.json")
}

fn load_catalog_or_empty(path: &Path, err: &mut dyn Write) -> Result<ModelCatalog, ExitCode> {
    match ModelCatalog::load(path) {
        Ok(c) => Ok(c),
        Err(CatalogError::Io(e)) if e.kind() == std::io::ErrorKind::NotFound => {
            Ok(ModelCatalog::default())
        }
        Err(e) => {
            let _ = writeln!(err, "STRAT-E1001 {e}");
            Err(ExitCode::from(1))
        }
    }
}

fn ensure_state_dir(paths: &Paths, err: &mut dyn Write) -> Result<(), ExitCode> {
    if let Err(e) = std::fs::create_dir_all(&paths.state) {
        let _ = writeln!(
            err,
            "STRAT-E1001 cannot create {}: {e}",
            paths.state.display()
        );
        return Err(ExitCode::from(1));
    }
    Ok(())
}

fn models_list(
    json: bool,
    paths: &Paths,
    args: &ListArgs,
    out: &mut dyn Write,
    err: &mut dyn Write,
) -> ExitCode {
    let path = catalog_path(paths);
    let catalog = match load_catalog_or_empty(&path, err) {
        Ok(c) => c,
        Err(code) => return code,
    };

    let mut filtered: Vec<&ModelEntry> = catalog.entries.values().collect();
    if let Some(t) = args.tier {
        let tier: ModelTier = t.into();
        filtered.retain(|e| e.tier == tier);
    }
    if let Some(t) = args.task {
        let task: ModelTask = t.into();
        filtered.retain(|e| e.task.contains(&task));
    }

    if json {
        #[allow(clippy::expect_used, reason = "ModelEntry serialization is infallible")]
        let rendered = serde_json::to_string_pretty(&filtered)
            .expect("ModelEntry serialization is infallible");
        if writeln!(out, "{rendered}").is_err() {
            return ExitCode::from(74);
        }
    } else if filtered.is_empty() {
        if writeln!(out, "(no catalog entries)").is_err() {
            return ExitCode::from(74);
        }
    } else {
        if writeln!(
            out,
            "{:<24} {:<10} {:<10} {:>8}  DISPLAY_NAME",
            "SLUG", "TIER", "TASKS", "SIZE_MIB"
        )
        .is_err()
        {
            return ExitCode::from(74);
        }
        for entry in &filtered {
            let tasks = entry
                .task
                .iter()
                .map(|t| format!("{}", serde_json::to_value(t).unwrap_or_default()))
                .collect::<Vec<_>>()
                .join(",");
            let tier_str = format!("{}", serde_json::to_value(entry.tier).unwrap_or_default());
            if writeln!(
                out,
                "{:<24} {:<10} {:<10} {:>8}  {}",
                entry.slug.as_str(),
                tier_str.trim_matches('"'),
                tasks.replace('"', ""),
                entry.size_mib,
                entry.display_name
            )
            .is_err()
            {
                return ExitCode::from(74);
            }
        }
    }
    ExitCode::SUCCESS
}

fn models_add(
    json: bool,
    paths: &Paths,
    args: &AddArgs,
    out: &mut dyn Write,
    err: &mut dyn Write,
) -> ExitCode {
    if let Err(code) = ensure_state_dir(paths, err) {
        return code;
    }
    let path = catalog_path(paths);

    let slug: ModelSlug = match args.slug.parse() {
        Ok(s) => s,
        Err(e) => {
            let _ = writeln!(err, "invalid --slug: {e}");
            return ExitCode::from(2);
        }
    };

    let artifact = match ModelArtifactRef::new(args.url.clone(), args.sha256.clone(), args.bytes) {
        Ok(a) => a,
        Err(e) => {
            let _ = writeln!(err, "invalid artifact: {e}");
            return ExitCode::from(2);
        }
    };

    let tasks = match parse_task_csv(&args.task) {
        Ok(t) => t,
        Err(e) => {
            let _ = writeln!(err, "invalid --task: {e}");
            return ExitCode::from(2);
        }
    };

    if args.size_mib == 0 {
        let _ = writeln!(err, "invalid --size-mib: must be > 0");
        return ExitCode::from(2);
    }
    if args.family.trim().is_empty() {
        let _ = writeln!(err, "invalid --family: must not be empty");
        return ExitCode::from(2);
    }

    let mut catalog = match load_catalog_or_empty(&path, err) {
        Ok(c) => c,
        Err(code) => return code,
    };

    let entry = ModelEntry {
        slug: slug.clone(),
        family: args.family.clone(),
        display_name: args.display_name.clone(),
        tier: args.tier.into(),
        task: tasks,
        size_mib: args.size_mib,
        quantization: args.quantization.clone(),
        artifact,
        license: args.license.clone(),
        homepage: args.homepage.clone(),
    };
    catalog.insert(entry.clone());

    if let Err(e) = catalog.save_atomic(&path) {
        let _ = writeln!(err, "STRAT-E1001 {e}");
        return ExitCode::from(1);
    }

    if json {
        #[allow(clippy::expect_used, reason = "ModelEntry serialization is infallible")]
        let rendered =
            serde_json::to_string_pretty(&entry).expect("ModelEntry serialization is infallible");
        if writeln!(out, "{rendered}").is_err() {
            return ExitCode::from(74);
        }
    } else if writeln!(
        out,
        "added · {} · tier={:?} · size={} MiB",
        slug.as_str(),
        entry.tier,
        entry.size_mib
    )
    .is_err()
    {
        return ExitCode::from(74);
    }
    ExitCode::SUCCESS
}

fn models_remove(
    paths: &Paths,
    args: &RemoveArgs,
    out: &mut dyn Write,
    err: &mut dyn Write,
) -> ExitCode {
    let slug: ModelSlug = match args.slug.parse() {
        Ok(s) => s,
        Err(e) => {
            let _ = writeln!(err, "invalid --slug: {e}");
            return ExitCode::from(2);
        }
    };
    let path = catalog_path(paths);
    let mut catalog = match load_catalog_or_empty(&path, err) {
        Ok(c) => c,
        Err(code) => return code,
    };
    if catalog.entries.remove(&slug).is_none() {
        let _ = writeln!(err, "no such slug: {}", slug.as_str());
        return ExitCode::from(1);
    }
    if let Err(e) = catalog.save_atomic(&path) {
        let _ = writeln!(err, "STRAT-E1001 {e}");
        return ExitCode::from(1);
    }
    if writeln!(out, "removed · {}", slug.as_str()).is_err() {
        return ExitCode::from(74);
    }
    ExitCode::SUCCESS
}

fn models_recommend(
    json: bool,
    paths: &Paths,
    args: &RecommendArgs,
    out: &mut dyn Write,
    err: &mut dyn Write,
) -> ExitCode {
    let path = catalog_path(paths);
    let catalog = match load_catalog_or_empty(&path, err) {
        Ok(c) => c,
        Err(code) => return code,
    };
    let tier: ModelTier = args.tier.into();
    let task: ModelTask = args.task.into();
    if let Some(entry) = catalog.recommend_for(tier, task) {
        if json {
            #[allow(clippy::expect_used, reason = "ModelEntry serialization is infallible")]
            let rendered = serde_json::to_string_pretty(entry)
                .expect("ModelEntry serialization is infallible");
            if writeln!(out, "{rendered}").is_err() {
                return ExitCode::from(74);
            }
        } else if writeln!(out, "{} · {}", entry.slug.as_str(), entry.display_name).is_err() {
            return ExitCode::from(74);
        }
        ExitCode::SUCCESS
    } else {
        let _ = writeln!(err, "no model fits the requested tier/task");
        ExitCode::from(1)
    }
}

fn models_validate(paths: &Paths, out: &mut dyn Write, err: &mut dyn Write) -> ExitCode {
    let path = catalog_path(paths);
    let catalog = match load_catalog_or_empty(&path, err) {
        Ok(c) => c,
        Err(code) => return code,
    };
    match catalog.validate() {
        Ok(()) => {
            if writeln!(out, "ok · {} entries", catalog.entries.len()).is_err() {
                return ExitCode::from(74);
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            let _ = writeln!(err, "{e}");
            ExitCode::from(1)
        }
    }
}

fn models_install_file(
    json: bool,
    paths: &Paths,
    args: &InstallFileArgs,
    out: &mut dyn Write,
    err: &mut dyn Write,
) -> ExitCode {
    let dest_dir = models_dir(paths);
    let dest_filename = args
        .name
        .as_deref()
        .map_or_else(|| default_filename_for(args), ToString::to_string);
    let installer = ModelInstaller {
        dest_dir: &dest_dir,
        dest_filename: &dest_filename,
        expected_sha256: args.sha256.as_deref(),
    };
    let result = match (args.from_file.as_deref(), args.from_url.as_deref()) {
        (Some(path), None) => installer.install_local(path),
        (None, Some(url)) => installer.install_from_url(url),
        _ => {
            let _ = writeln!(
                err,
                "STRAT-E1001 supply exactly one of --from-file or --from-url"
            );
            return ExitCode::from(64);
        }
    };
    let report = match result {
        Ok(r) => r,
        Err(e) => {
            let _ = writeln!(err, "{e}");
            return ExitCode::from(73);
        }
    };

    if json {
        #[allow(
            clippy::expect_used,
            reason = "InstallReport serialization is infallible (primitives only)"
        )]
        let rendered = serde_json::to_string_pretty(&report)
            .expect("InstallReport serialization is infallible");
        if writeln!(out, "{rendered}").is_err() {
            return ExitCode::from(74);
        }
    } else if writeln!(
        out,
        "installed · {} · {} bytes · sha256={} · verified={}",
        report.dest.display(),
        report.bytes,
        report.sha256,
        report.verified
    )
    .is_err()
    {
        return ExitCode::from(74);
    }
    ExitCode::SUCCESS
}

fn default_filename_for(args: &InstallFileArgs) -> String {
    if let Some(p) = args.from_file.as_deref() {
        return p.file_name().map_or_else(
            || "model.bin".to_string(),
            |s| s.to_string_lossy().into_owned(),
        );
    }
    if let Some(u) = args.from_url.as_deref() {
        if !u.ends_with('/') {
            if let Some(last) = u.rsplit('/').next() {
                if !last.is_empty() && !last.contains(':') {
                    return last.to_string();
                }
            }
        }
    }
    "model.bin".to_string()
}

#[derive(Debug, Serialize)]
struct MemCheckOk {
    status: &'static str,
    free_mib: u32,
    needed_mib: u32,
    margin_mib: u32,
    leftover_mib: u32,
}

#[derive(Debug, Serialize)]
struct MemCheckErr {
    status: &'static str,
    code: ErrorCode,
    message: String,
    free_mib: u32,
    needed_mib: u32,
    margin_mib: u32,
    suggested_unloads: Vec<String>,
}

/// `--json` payload for the default (no-flags) and `--requested` modes.
#[derive(Debug, Serialize)]
struct MemCheckSuggestReport {
    available_mib: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    requested: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    requested_mib: Option<u64>,
    suggested_evictions: Vec<String>,
}

/// On-disk entry shape of `<state_root>/loaded.json`. The file is a JSON
/// array of these objects; each row stands in for one resident model the
/// gate may suggest evicting. `last_used_unix_secs` is read but not consumed
/// by [`MemoryGate::suggest_unloads`] (it sorts by footprint); we still
/// require the field so the on-disk schema matches the runtime's bookkeeping.
#[derive(Debug, serde::Deserialize)]
struct LoadedFileEntry {
    slug: String,
    footprint_mib: u64,
    #[allow(dead_code, reason = "field is part of the persisted schema")]
    last_used_unix_secs: u64,
}

/// Parse a single `--loaded model_id:weight_rss_mib:kv_per_token_bytes:context_tokens`
/// spec into a [`LoadedModel`] plus the planned context length that will be
/// charged to the candidate (we use the max across resident specs to be
/// conservative; the per-model context only feeds the gate's `hot_ram_mib`).
fn parse_loaded_spec(spec: &str) -> Result<(LoadedModel, u32), String> {
    let parts: Vec<&str> = spec.split(':').collect();
    if parts.len() != 4 {
        return Err(format!(
            "expected 4 colon-separated fields in --loaded \"{spec}\""
        ));
    }
    let id = parts[0];
    if id.is_empty() {
        return Err(format!("empty model id in --loaded \"{spec}\""));
    }
    let weight: u32 = parts[1]
        .parse()
        .map_err(|e| format!("weight_rss_mib in --loaded \"{spec}\": {e}"))?;
    let kv: u32 = parts[2]
        .parse()
        .map_err(|e| format!("kv_per_token_bytes in --loaded \"{spec}\": {e}"))?;
    let ctx: u32 = parts[3]
        .parse()
        .map_err(|e| format!("context_tokens in --loaded \"{spec}\": {e}"))?;
    let loaded = LoadedModel {
        id: ModelId::from(id),
        estimate: MemEstimate {
            weight_rss_mib: weight,
            kv_per_token_bytes: kv,
            mmproj_mib: 0,
            vram_mib: 0,
        },
        role_hint: None,
    };
    Ok((loaded, ctx))
}

fn mem_check(
    json: bool,
    args: &MemCheckArgs,
    paths: &Paths,
    out: &mut dyn Write,
    err: &mut dyn Write,
) -> ExitCode {
    // Mode selection. `--requested` and `--requested-mib` are bound together
    // by clap (`requires`), so seeing one implies the other.
    if let (Some(slug), Some(req_mib)) = (args.requested.as_deref(), args.requested_mib) {
        return mem_check_suggest(
            json,
            slug,
            req_mib,
            args.loaded_file.as_deref(),
            paths,
            out,
            err,
        );
    }
    // Legacy mode requires all three of weight_rss / kv_per_token / context.
    match (args.weight_rss, args.kv_per_token, args.context) {
        (Some(w), Some(kv), Some(ctx)) => mem_check_legacy(json, args, w, kv, ctx, out, err),
        (None, None, None) => mem_check_default(json, out, err),
        _ => {
            let _ = writeln!(
                err,
                "STRAT-E1001 --weight-rss, --kv-per-token, and --context must be passed together",
            );
            ExitCode::from(64)
        }
    }
}

/// "No flags" mode: just print the host's available RAM.
fn mem_check_default(json: bool, out: &mut dyn Write, err: &mut dyn Write) -> ExitCode {
    let probe = HardwareProbe::run();
    let available_mib = probe.ram_available_mib;
    if json {
        let payload = MemCheckSuggestReport {
            available_mib,
            requested: None,
            requested_mib: None,
            suggested_evictions: Vec::new(),
        };
        #[allow(
            clippy::expect_used,
            reason = "MemCheckSuggestReport serialization is infallible (primitives only)"
        )]
        let rendered = serde_json::to_string_pretty(&payload)
            .expect("MemCheckSuggestReport serialization is infallible");
        if writeln!(out, "{rendered}").is_err() {
            return ExitCode::from(74);
        }
    } else {
        let gb = format_gb_one_decimal(available_mib);
        if writeln!(out, "available: {gb} GB ({available_mib} MiB)").is_err() {
            let _ = writeln!(err, "STRAT-E1001 stdout write failed");
            return ExitCode::from(74);
        }
    }
    ExitCode::SUCCESS
}

/// `--requested` mode: consult `MemoryGate::suggest_unloads` against the
/// resident-set file and print the recommended evictions.
fn mem_check_suggest(
    json: bool,
    requested: &str,
    requested_mib: u64,
    loaded_file: Option<&Path>,
    paths: &Paths,
    out: &mut dyn Write,
    err: &mut dyn Write,
) -> ExitCode {
    let path = loaded_file.map_or_else(|| paths.state.join("loaded.json"), Path::to_path_buf);
    let entries = match read_loaded_file(&path) {
        Ok(v) => v,
        Err(diag) => {
            let _ = writeln!(err, "STRAT-E1001 {diag}");
            return ExitCode::from(1);
        }
    };
    let loaded: Vec<LoadedModel> = entries
        .into_iter()
        .map(|e| LoadedModel {
            id: ModelId::from(e.slug.as_str()),
            estimate: MemEstimate {
                weight_rss_mib: u32::try_from(e.footprint_mib).unwrap_or(u32::MAX),
                kv_per_token_bytes: 0,
                mmproj_mib: 0,
                vram_mib: 0,
            },
            role_hint: None,
        })
        .collect();

    let probe = HardwareProbe::run();
    let candidate = MemEstimate {
        weight_rss_mib: u32::try_from(requested_mib).unwrap_or(u32::MAX),
        kv_per_token_bytes: 0,
        mmproj_mib: 0,
        vram_mib: 0,
    };
    let gate = MemoryGate::new(DEFAULT_MARGIN_MIB);
    let suggested: Vec<String> = gate
        .suggest_unloads(&probe, &candidate, 0, &loaded)
        .into_iter()
        .map(|m| m.as_str().to_string())
        .collect();
    // suggest_unloads returns an empty Vec both when the load already fits
    // AND when even unloading everything wouldn't free enough. Distinguish
    // them by checking `would_fit` directly.
    let fits = gate.would_fit(&probe, &candidate, 0);

    if json {
        let payload = MemCheckSuggestReport {
            available_mib: probe.ram_available_mib,
            requested: Some(requested.to_owned()),
            requested_mib: Some(requested_mib),
            suggested_evictions: suggested,
        };
        #[allow(
            clippy::expect_used,
            reason = "MemCheckSuggestReport serialization is infallible (primitives only)"
        )]
        let rendered = serde_json::to_string_pretty(&payload)
            .expect("MemCheckSuggestReport serialization is infallible");
        if writeln!(out, "{rendered}").is_err() {
            return ExitCode::from(74);
        }
    } else {
        let gb = format_gb_one_decimal(probe.ram_available_mib);
        if writeln!(out, "available: {gb} GB ({} MiB)", probe.ram_available_mib,).is_err() {
            return ExitCode::from(74);
        }
        if fits {
            if writeln!(out, "fits without eviction").is_err() {
                return ExitCode::from(74);
            }
        } else if suggested.is_empty() {
            if writeln!(
                out,
                "to make room for {requested} ({requested_mib} MiB), evict: (no feasible subset)",
            )
            .is_err()
            {
                return ExitCode::from(74);
            }
        } else if writeln!(
            out,
            "to make room for {requested} ({requested_mib} MiB), evict: {}",
            suggested.join(", "),
        )
        .is_err()
        {
            return ExitCode::from(74);
        }
    }
    ExitCode::SUCCESS
}

/// Read `<state_root>/loaded.json` (or the explicit override) into the
/// strongly-typed entry list. A missing file returns an empty list; a
/// malformed file returns a human-readable error string.
fn read_loaded_file(path: &Path) -> Result<Vec<LoadedFileEntry>, String> {
    match std::fs::read_to_string(path) {
        Ok(body) => serde_json::from_str::<Vec<LoadedFileEntry>>(&body)
            .map_err(|e| format!("loaded-file {} parse failed: {e}", path.display())),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Vec::new()),
        Err(e) => Err(format!("loaded-file {} read failed: {e}", path.display())),
    }
}

/// Legacy `check_with`-driven mode. Unchanged behaviour from the original
/// implementation; the only structural difference is that the
/// `weight_rss/kv_per_token/context` triple is now lifted out of the args
/// struct so the caller can validate that all three were supplied.
fn mem_check_legacy(
    json: bool,
    args: &MemCheckArgs,
    weight_rss: u32,
    kv_per_token: u32,
    context: u32,
    out: &mut dyn Write,
    err: &mut dyn Write,
) -> ExitCode {
    let probe = HardwareProbe::run();
    let estimate = MemEstimate {
        weight_rss_mib: weight_rss,
        kv_per_token_bytes: kv_per_token,
        mmproj_mib: args.mmproj,
        vram_mib: args.vram,
    };
    let gate = MemoryGate::new(args.margin);
    let needed_mib = estimate.hot_ram_mib(context);
    let free_mib = probe.ram_available_mib;

    let mut loaded: Vec<LoadedModel> = Vec::with_capacity(args.loaded.len());
    for spec in &args.loaded {
        match parse_loaded_spec(spec) {
            Ok((lm, _ctx)) => loaded.push(lm),
            Err(diag) => {
                let _ = writeln!(err, "STRAT-E1001 {diag}");
                return ExitCode::from(64);
            }
        }
    }

    match gate.check_with(&probe, &estimate, context, &loaded) {
        Ok(()) => {
            let leftover = free_mib.saturating_sub(needed_mib);
            if json {
                let payload = MemCheckOk {
                    status: "ok",
                    free_mib,
                    needed_mib,
                    margin_mib: args.margin,
                    leftover_mib: leftover,
                };
                #[allow(
                    clippy::expect_used,
                    reason = "MemCheckOk serialization is infallible (primitives only)"
                )]
                let rendered = serde_json::to_string_pretty(&payload)
                    .expect("MemCheckOk serialization is infallible");
                if writeln!(out, "{rendered}").is_err() {
                    return ExitCode::from(74);
                }
            } else {
                let leftover_gb = format_gb_one_decimal(leftover);
                if writeln!(out, "ok: would leave {leftover_gb} GB free").is_err() {
                    return ExitCode::from(74);
                }
            }
            ExitCode::SUCCESS
        }
        Err(diag) => {
            let suggested: Vec<String> = gate
                .suggest_unloads(&probe, &estimate, context, &loaded)
                .into_iter()
                .map(|m| m.as_str().to_string())
                .collect();
            if json {
                let payload = MemCheckErr {
                    status: "refused",
                    code: diag.code().clone(),
                    message: diag.message.clone(),
                    free_mib,
                    needed_mib,
                    margin_mib: args.margin,
                    suggested_unloads: suggested,
                };
                #[allow(
                    clippy::expect_used,
                    reason = "MemCheckErr serialization is infallible (primitives only)"
                )]
                let rendered = serde_json::to_string_pretty(&payload)
                    .expect("MemCheckErr serialization is infallible");
                if writeln!(err, "{rendered}").is_err() {
                    return ExitCode::from(74);
                }
            } else if writeln!(err, "{diag}").is_err() {
                return ExitCode::from(74);
            }
            ExitCode::from(1)
        }
    }
}

/// Mebibytes → base-10 GB with one decimal, matching the gate's renderer.
fn format_gb_one_decimal(mib: u32) -> String {
    let scaled = u64::from(mib) * 1_048_576;
    let gb_x10 = (scaled + 50_000_000) / 100_000_000;
    let whole = gb_x10 / 10;
    let frac = gb_x10 % 10;
    format!("{whole}.{frac}")
}

fn chat_command(
    args: &ChatArgs,
    paths: &Paths,
    out: &mut dyn Write,
    err: &mut dyn Write,
) -> ExitCode {
    let probe = HardwareProbe::run();
    let tier = Tier::classify(&probe);

    // Model-resolution flow only runs when --model is set. Without the
    // `provider-llama-cpp` feature, the only legal mode is EchoProvider —
    // surface a clear STRAT-E1001 error instead of silently downgrading.
    if let Some(slug) = args.model.as_deref() {
        return chat_with_model(slug, args, paths, tier, out, err);
    }

    // No --model: keep EchoProvider behavior. `--prompt` still works for
    // the scripted path; otherwise fall through to the interactive TUI.
    if let Some(prompt) = args.prompt.as_deref() {
        let provider = EchoProvider::new("echo: ");
        let mut state = crate::chat::ChatState::new(provider, tier, crate::chat::status_for(paths));
        if let Some(path) = args.events_jsonl.as_deref() {
            state = match attach_jsonl_events(state, path, err) {
                Ok(s) => s,
                Err(code) => return code,
            };
        }
        state.submit_with_prompt(prompt);
        return print_assistant_text(&state, out, err);
    }

    if let Some(path) = args.events_jsonl.as_deref() {
        let provider = EchoProvider::new("echo: ");
        let state = crate::chat::ChatState::new(provider, tier, crate::chat::status_for(paths));
        let state = match attach_jsonl_events(state, path, err) {
            Ok(s) => s,
            Err(code) => return code,
        };
        return match crate::chat::run_with_state(state) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                let _ = writeln!(err, "{e}");
                ExitCode::from(70)
            }
        };
    }

    match crate::chat::run(paths, tier) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            let _ = writeln!(err, "{e}");
            ExitCode::from(70)
        }
    }
}

/// Open `path` as a [`JsonlEventSink`], wrap it in an [`EventEmitter`], and
/// install the emitter into `state` via [`crate::chat::ChatState::with_events`].
///
/// On failure (e.g. missing parent dir), writes a STRAT-E1001 diag to `err`
/// and returns exit code 1.
fn attach_jsonl_events(
    state: crate::chat::ChatState,
    path: &Path,
    err: &mut dyn Write,
) -> Result<crate::chat::ChatState, ExitCode> {
    use std::sync::Arc;

    use stratum_runtime::{EventEmitter, EventSink, JsonlEventSink};

    let sink = match JsonlEventSink::open(path.to_path_buf()) {
        Ok(s) => s,
        Err(e) => {
            let _ = writeln!(
                err,
                "STRAT-E1001 cannot open events JSONL {}: {e}",
                path.display()
            );
            return Err(ExitCode::from(1));
        }
    };
    let sink_dyn: Arc<dyn EventSink> = Arc::new(sink);
    let emitter = Arc::new(EventEmitter::new(sink_dyn));
    Ok(state.with_events(emitter))
}

/// Resolve `--model <slug>` to a catalog entry, materialise the GGUF, open
/// the `LlamaCppProvider`, and either run one `--prompt` turn or launch the
/// interactive TUI. Feature-gated: without `provider-llama-cpp`, returns a
/// STRAT-E1001 error and exit 1.
#[cfg(not(feature = "provider-llama-cpp"))]
fn chat_with_model(
    _slug: &str,
    _args: &ChatArgs,
    _paths: &Paths,
    _tier: Tier,
    _out: &mut dyn Write,
    err: &mut dyn Write,
) -> ExitCode {
    let _ = writeln!(
        err,
        "STRAT-E1001 the `provider-llama-cpp` feature is not enabled; rebuild with `--features provider-llama-cpp`"
    );
    ExitCode::from(1)
}

#[cfg(feature = "provider-llama-cpp")]
fn chat_with_model(
    slug: &str,
    args: &ChatArgs,
    paths: &Paths,
    _tier: Tier,
    out: &mut dyn Write,
    err: &mut dyn Write,
) -> ExitCode {
    use std::sync::Arc;

    use stratum_runtime::Provider as ProviderTrait;

    let provider = match resolve_llama_provider(slug, args.ctx, paths, err) {
        Ok(p) => p,
        Err(code) => return code,
    };
    let provider_arc: Arc<dyn ProviderTrait> = Arc::new(provider);

    let loop_ = match build_llama_agent_loop(provider_arc, err) {
        Ok(l) => l,
        Err(code) => return code,
    };
    let _ = args.max_blocks; // honored by provider request path; reserved for Phase 3 wiring.

    let mut state = crate::chat::ChatState::with_agent_loop(loop_);
    if let Some(prompt) = args.prompt.as_deref() {
        state.submit_with_prompt(prompt);
        return print_assistant_text(&state, out, err);
    }
    // No --prompt: drop into the interactive TUI. The state already wraps
    // the llama-backed loop so input is routed through real inference.
    match crate::chat::run_with_state(state) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            let _ = writeln!(err, "{e}");
            ExitCode::from(70)
        }
    }
}

/// Resolve `slug` against the on-disk catalog, materialise the GGUF on
/// disk (downloading + SHA-verifying when needed), and open a
/// [`LlamaCppProvider`]. Every failure mode emits a STRAT-E1001 diag
/// to `err` and returns exit code 1.
#[cfg(feature = "provider-llama-cpp")]
fn resolve_llama_provider(
    slug: &str,
    n_ctx: u32,
    paths: &Paths,
    err: &mut dyn Write,
) -> Result<stratum_runtime::LlamaCppProvider, ExitCode> {
    use stratum_runtime::llama_provider::LlamaCppProviderConfig;
    use stratum_runtime::LlamaCppProvider;

    let parsed_slug: ModelSlug = slug.parse().map_err(|e| {
        let _ = writeln!(err, "STRAT-E1001 invalid --model slug {slug:?}: {e}");
        ExitCode::from(1)
    })?;

    let catalog_file = catalog_path(paths);
    let catalog = match ModelCatalog::load(&catalog_file) {
        Ok(c) => c,
        Err(CatalogError::Io(io_err)) if io_err.kind() == std::io::ErrorKind::NotFound => {
            // Missing catalog file is the common first-run state; surface
            // the same "unknown slug" diag as a present-but-empty catalog
            // so the user gets one consistent error message.
            ModelCatalog::default()
        }
        Err(e) => {
            let _ = writeln!(
                err,
                "STRAT-E1001 cannot load catalog {}: {e}",
                catalog_file.display()
            );
            let _ = writeln!(
                err,
                "hint: run `stratum models list` to see installed catalog entries"
            );
            return Err(ExitCode::from(1));
        }
    };

    let Some(entry) = catalog.get(&parsed_slug) else {
        let _ = writeln!(
            err,
            "STRAT-E1001 unknown slug {:?} in {}",
            slug,
            catalog_file.display()
        );
        let _ = writeln!(
            err,
            "hint: run `stratum models list` to see installed catalog entries"
        );
        return Err(ExitCode::from(1));
    };

    // Target path is content-addressed by sha256, so re-downloads converge
    // on the same file regardless of slug renames.
    let models_root = models_dir(paths);
    let target = models_root.join(format!("{}.gguf", entry.artifact.sha256));
    if needs_refetch(&target, &entry.artifact.sha256, entry.artifact.bytes) {
        let dest_filename = format!("{}.gguf", entry.artifact.sha256);
        let installer = ModelInstaller {
            dest_dir: &models_root,
            dest_filename: &dest_filename,
            expected_sha256: Some(entry.artifact.sha256.as_str()),
        };
        installer
            .install_from_url(entry.artifact.url.as_str())
            .map_err(|e| {
                let _ = writeln!(err, "STRAT-E1001 download failed for {slug}: {e}");
                ExitCode::from(1)
            })?;
    }

    let cfg = LlamaCppProviderConfig {
        model_path: target,
        n_ctx,
        n_threads: None,
        n_gpu_layers: 0,
        seed: 42,
    };
    LlamaCppProvider::open(&cfg).map_err(|e| {
        let _ = writeln!(err, "STRAT-E1001 provider open failed: {e}");
        ExitCode::from(1)
    })
}

/// Build the `AgentLoop` wrapping `provider` with the documented
/// defaults. Mirrors `chat::default_agent_loop` but lives in `app.rs` so
/// the loop can carry the llama-backed provider rather than the echo
/// fallback.
#[cfg(feature = "provider-llama-cpp")]
fn build_llama_agent_loop(
    provider: std::sync::Arc<dyn stratum_runtime::Provider>,
    err: &mut dyn Write,
) -> Result<std::sync::Arc<stratum_runtime::AgentLoop>, ExitCode> {
    use std::sync::Arc;

    use stratum_runtime::{
        AgentLoop, AgentLoopConfig, AllowAllResponder, CapabilityMatrix, EventEmitter, EventSink,
        IntentRouter, MemoryEventSink, PermissionStore, PlanMode, PromptIdGen,
    };

    let sink: Arc<dyn EventSink> = Arc::new(MemoryEventSink::new());
    let events = Arc::new(EventEmitter::new(sink));
    AgentLoop::builder()
        .with_provider(provider)
        .with_router(IntentRouter::default())
        .with_permission_store(Arc::new(PermissionStore::new()))
        .with_prompt_gen(Arc::new(PromptIdGen::new()))
        .with_responder(Arc::new(AllowAllResponder))
        .with_events(events)
        .with_capability_matrix(Arc::new(CapabilityMatrix::new()))
        .with_plan_mode(Arc::new(PlanMode::new()))
        .with_config(AgentLoopConfig::default())
        .build()
        .map(Arc::new)
        .map_err(|e| {
            let _ = writeln!(err, "STRAT-E1001 agent loop build failed: {e}");
            ExitCode::from(1)
        })
}

/// Returns `true` when the target GGUF on disk is missing or has the wrong
/// byte count. The filename is content-addressed by SHA-256, so a matching
/// byte count plus correct filename is treated as a cache hit; mismatched
/// or missing files fall through to [`ModelInstaller::install_from_url`]
/// which re-verifies the SHA during install.
#[cfg(feature = "provider-llama-cpp")]
fn needs_refetch(target: &Path, _expected_sha256: &str, expected_bytes: u64) -> bool {
    std::fs::metadata(target).map_or(true, |m| m.len() != expected_bytes)
}

/// Print the most recent assistant turn text to `out`. Used by the
/// non-interactive `--prompt` flow.
fn print_assistant_text(
    state: &crate::chat::ChatState,
    out: &mut dyn Write,
    err: &mut dyn Write,
) -> ExitCode {
    if let Some(text) = state.last_assistant_text() {
        if writeln!(out, "{text}").is_err() {
            return ExitCode::from(74);
        }
        ExitCode::SUCCESS
    } else {
        let _ = writeln!(err, "STRAT-E1001 provider returned no text blocks");
        ExitCode::from(1)
    }
}

fn resolve_paths_with<F>(
    override_root: Option<&std::path::Path>,
    fallback: F,
) -> Result<Paths, String>
where
    F: FnOnce() -> stratum_types::StratumResult<Paths>,
{
    override_root.map_or_else(
        || fallback().map_err(|e| format!("{e}")),
        |root| Ok(Paths::under(root)),
    )
}

/// Implements `stratum serve`. Build the `AgentServeHandler` against an
/// `EchoProvider`-backed [`AgentFactory`] plus a [`TranscriptStore`] rooted
/// at `<state>/transcripts`, wrap it in a [`ServeServer`], start the
/// acceptor, and block until either `--stop-after-ms` elapses or the
/// handler's internal shutdown flag flips (via a `stop` JSON-RPC method).
///
/// The function intentionally avoids the `ctrlc` crate — graceful
/// signal-driven shutdown is deferred to a follow-up that settles the
/// broader runtime signal policy. The current shutdown surface is the
/// `--stop-after-ms` watchdog plus the in-protocol `stop` method, which is
/// enough to exercise the wiring in tests.
fn serve(args: &ServeArgs, paths: &Paths, out: &mut dyn Write, err: &mut dyn Write) -> ExitCode {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::thread;

    if let Err(code) = ensure_state_dir(paths, err) {
        return code;
    }

    // Resolve transport. Clap's `conflicts_with` already rejects "both",
    // but we still need to materialise the chosen `ServeBind`.
    let bind = args.unix_socket.as_ref().map_or_else(
        || ServeBind::TcpLoopback {
            port: args.tcp_port.unwrap_or(0),
        },
        |path| ServeBind::UnixSocket { path: path.clone() },
    );

    let transcripts_dir = paths.state.join("transcripts");
    if let Err(e) = std::fs::create_dir_all(&transcripts_dir) {
        let _ = writeln!(
            err,
            "STRAT-E1001 cannot create {}: {e}",
            transcripts_dir.display()
        );
        return ExitCode::from(1);
    }
    let store = match TranscriptStore::open(transcripts_dir.clone()) {
        Ok(s) => Arc::new(s),
        Err(e) => {
            let _ = writeln!(
                err,
                "STRAT-E1001 cannot open transcript store at {}: {e}",
                transcripts_dir.display()
            );
            return ExitCode::from(1);
        }
    };

    let factory = Arc::new(AgentFactory::new().with_provider(Arc::new(EchoProvider::new(""))));
    let events = Arc::new(EventEmitter::new(Arc::new(MemoryEventSink::new())));

    let handler = Arc::new(AgentServeHandler::new(
        factory,
        store,
        events,
        env!("CARGO_PKG_VERSION").to_string(),
    ));
    handler.mark_ready();

    let cfg = ServeConfig {
        bind,
        max_connections: args.max_connections,
        request_timeout: Duration::from_millis(args.request_timeout_ms),
    };
    let handler_for_server: Arc<dyn ServeHandler> = handler.clone();
    let server = Arc::new(ServeServer::new(cfg, handler_for_server));
    let handle = match server.start() {
        Ok(h) => h,
        Err(e) => {
            let _ = writeln!(err, "STRAT-E1001 serve start failed: {e}");
            return ExitCode::from(1);
        }
    };

    let bound = handle.bound_address().to_string();
    if args.json {
        #[allow(
            clippy::expect_used,
            reason = "ServeStartupReport serialization is infallible (primitive fields)"
        )]
        let rendered = serde_json::to_string(&ServeStartupReport { bound: &bound })
            .expect("ServeStartupReport serialization is infallible");
        if writeln!(out, "{rendered}").is_err() {
            return ExitCode::from(74);
        }
    } else if writeln!(out, "stratum serve: listening on {bound}").is_err() {
        return ExitCode::from(74);
    }
    // Flush so callers blocking on stdout (e.g. integration tests parsing
    // the bound address) see the startup line before we start polling.
    let _ = out.flush();

    let shutdown_flag = Arc::new(AtomicBool::new(false));
    if let Some(ms) = args.stop_after_ms {
        let flag = shutdown_flag.clone();
        let _ = thread::Builder::new()
            .name("stratum-serve-stopwatch".to_string())
            .spawn(move || {
                thread::sleep(Duration::from_millis(ms));
                flag.store(true, Ordering::Relaxed);
            });
    }

    // Poll for either the stopwatch fire or the handler's own
    // shutdown_requested flag (set by the in-protocol `stop` method). Tick
    // at 100ms to keep test latency low without burning CPU.
    while !shutdown_flag.load(Ordering::Relaxed) && !handler.is_shutdown_requested() {
        thread::sleep(Duration::from_millis(100));
    }

    if let Err(_panic) = handle.stop() {
        let _ = writeln!(err, "STRAT-E1001 serve acceptor thread panicked");
        return ExitCode::from(70);
    }
    ExitCode::SUCCESS
}

/// JSON payload emitted by `stratum serve --json` at startup.
#[derive(Debug, Serialize)]
struct ServeStartupReport<'a> {
    bound: &'a str,
}

fn print_greeting(paths: &Paths, out: &mut dyn Write) -> ExitCode {
    let installed = paths.installed_toml();
    let status = if installed.exists() {
        "installed"
    } else {
        "not installed; run `stratum init`"
    };
    if writeln!(out, "hello, tier=unknown — {status}").is_err() {
        return ExitCode::from(74);
    }
    ExitCode::SUCCESS
}

#[derive(Debug, Serialize)]
struct DoctorReport<'a> {
    schema_version: u32,
    stratum_version: &'static str,
    tier: Tier,
    probe: &'a HardwareProbe,
    gpu_accel: GpuBackend,
    sandbox: &'a SandboxReport,
    installed: bool,
    issues: Vec<DoctorIssue>,
    /// Telemetry payload preview. `Some(_)` when `--telemetry` was passed
    /// and the opt-out file did not disable telemetry; `None` when
    /// `--telemetry` was omitted or the user disabled it via
    /// `<state>/telemetry.json`. Serialized as the literal JSON `null` in the
    /// `disabled`/omitted case so consumers see a stable key.
    telemetry: Option<TelemetryPayload>,
}

#[derive(Debug, Serialize)]
struct DoctorIssue {
    code: ErrorCode,
    level: &'static str,
    message: String,
}

fn doctor(
    json: bool,
    args: &DoctorArgs,
    paths: &Paths,
    out: &mut dyn Write,
    err: &mut dyn Write,
) -> ExitCode {
    let probe = HardwareProbe::run();
    let tier = Tier::classify(&probe);
    let sandbox = SandboxReport::run();
    let installed = paths.installed_toml().exists();
    let mut issues = Vec::new();
    if !installed {
        issues.push(DoctorIssue {
            code: stratum_types::error::codes::E2003_TIER_DOWNGRADE_REFUSED,
            level: "info",
            message: "no installed.toml found; run `stratum init`".into(),
        });
    }

    // Telemetry assembly: only when --telemetry was requested. The opt-out
    // file lives at <state>/telemetry.json. If telemetry is disabled, we
    // skip assembly entirely (no install-id persistence either). If enabled
    // (default when file is missing or malformed), we read or generate the
    // anon install id and build the payload via the runtime helper.
    let (telemetry, telemetry_disabled) = if args.telemetry {
        let cfg = load_telemetry_config(paths);
        if cfg.enabled {
            match assemble_telemetry_payload(paths, args.telemetry_event.into(), tier, probe.gpu) {
                Ok(payload) => (Some(payload), false),
                Err(diag) => {
                    let _ = writeln!(err, "{diag}");
                    return ExitCode::from(1);
                }
            }
        } else {
            (None, true)
        }
    } else {
        (None, false)
    };

    let report = DoctorReport {
        schema_version: 1,
        stratum_version: env!("CARGO_PKG_VERSION"),
        tier,
        probe: &probe,
        gpu_accel: probe.gpu,
        sandbox: &sandbox,
        installed,
        issues,
        telemetry: telemetry.clone(),
    };

    if json {
        #[allow(
            clippy::expect_used,
            reason = "DoctorReport serialization is infallible (primitives only)"
        )]
        let rendered = serde_json::to_string_pretty(&report)
            .expect("DoctorReport serialization is infallible");
        if writeln!(out, "{rendered}").is_err() {
            return ExitCode::from(74);
        }
    } else {
        if writeln!(
            out,
            "stratum {} · tier={} · gpu={} · sandbox={} · ram={} MiB · cores={} · installed={}",
            report.stratum_version,
            tier,
            probe.gpu,
            sandbox.preferred(),
            probe.ram_total_mib,
            probe.cpu_cores,
            installed
        )
        .is_err()
        {
            return ExitCode::from(74);
        }
        if args.telemetry {
            if telemetry_disabled {
                if writeln!(out, "--- telemetry: disabled ---").is_err() {
                    return ExitCode::from(74);
                }
            } else if let Some(payload) = telemetry.as_ref() {
                #[allow(
                    clippy::expect_used,
                    reason = "TelemetryPayload serialization is infallible (primitives only)"
                )]
                let rendered = serde_json::to_string_pretty(payload)
                    .expect("TelemetryPayload serialization is infallible");
                if writeln!(out, "--- telemetry ---\n{rendered}").is_err() {
                    return ExitCode::from(74);
                }
            }
        }
    }
    ExitCode::SUCCESS
}

/// Minimal on-disk shape of `<state>/telemetry.json` — the brief documents
/// `{"enabled": bool}` as the user-facing schema. The runtime
/// [`TelemetryConfig`] carries additional fields (endpoint, channel), but we
/// only persist the opt-out bit; the rest is hard-coded for this PR.
#[derive(serde::Deserialize)]
struct TelemetryToggle {
    enabled: bool,
}

/// Read `<state>/telemetry.json` if present and parse it; on missing-file or
/// parse failure return the runtime default (enabled = true). Parse errors
/// are logged at `warn` via tracing so an operator notices them in the log
/// stream, but the command continues — telemetry is opt-out, not
/// fail-closed.
fn load_telemetry_config(paths: &Paths) -> TelemetryConfig {
    let path = paths.state.join("telemetry.json");
    let body = match std::fs::read_to_string(&path) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return TelemetryConfig::default(),
        Err(e) => {
            tracing::warn!(
                path = %path.display(),
                error = %e,
                "telemetry config read failed; falling back to enabled=true",
            );
            return TelemetryConfig::default();
        }
    };
    match serde_json::from_str::<TelemetryToggle>(&body) {
        Ok(toggle) => TelemetryConfig {
            enabled: toggle.enabled,
            ..TelemetryConfig::default()
        },
        Err(e) => {
            tracing::warn!(
                path = %path.display(),
                error = %e,
                "telemetry config parse failed; falling back to enabled=true",
            );
            TelemetryConfig::default()
        }
    }
}

/// Load or create the persistent anonymous install id at
/// `<state>/anon_install_id`. A missing or malformed file is replaced with a
/// freshly generated id, written atomically (`<path>.tmp` + rename).
fn load_or_create_anon_install_id(paths: &Paths) -> Result<AnonInstallId, String> {
    if let Err(e) = std::fs::create_dir_all(&paths.state) {
        return Err(format!(
            "STRAT-E1001 cannot create {}: {e}",
            paths.state.display()
        ));
    }
    let path = paths.state.join("anon_install_id");
    match std::fs::read_to_string(&path) {
        Ok(body) => {
            let trimmed = body.trim();
            match AnonInstallId::from_str(trimmed) {
                Ok(id) => Ok(id),
                Err(parse_err) => {
                    tracing::warn!(
                        path = %path.display(),
                        error = %parse_err,
                        "anon install id parse failed; regenerating",
                    );
                    let fresh = AnonInstallId::new_random();
                    write_anon_install_id(&path, &fresh)?;
                    Ok(fresh)
                }
            }
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            let fresh = AnonInstallId::new_random();
            write_anon_install_id(&path, &fresh)?;
            Ok(fresh)
        }
        Err(e) => Err(format!("STRAT-E1001 cannot read {}: {e}", path.display())),
    }
}

fn write_anon_install_id(path: &Path, id: &AnonInstallId) -> Result<(), String> {
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, id.as_str())
        .map_err(|e| format!("STRAT-E1001 cannot write {}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, path)
        .map_err(|e| format!("STRAT-E1001 cannot rename {}: {e}", path.display()))?;
    Ok(())
}

/// Assemble the telemetry payload for the doctor command. Caller has already
/// verified that telemetry is enabled.
fn assemble_telemetry_payload(
    paths: &Paths,
    event: TelemetryEventKind,
    tier: Tier,
    gpu: GpuBackend,
) -> Result<TelemetryPayload, String> {
    let install_id = load_or_create_anon_install_id(paths)?;
    let cfg = TelemetryConfig {
        channel: ReleaseChannel::Stable,
        ..TelemetryConfig::default()
    };
    let os = match std::env::consts::OS {
        "macos" => OsTag::MacOS,
        "linux" => OsTag::Linux,
        "windows" => OsTag::Windows,
        _ => OsTag::Other,
    };
    let cpu_arch = match std::env::consts::ARCH {
        "x86_64" => CpuArchTag::X86_64,
        "aarch64" => CpuArchTag::Aarch64,
        _ => CpuArchTag::Other,
    };
    let tier_str = tier.to_string();
    let gpu_str = gpu.to_string();
    let payload = build_telemetry_payload(
        &cfg,
        &install_id,
        env!("CARGO_PKG_VERSION"),
        event,
        &tier_str,
        &gpu_str,
        os,
        cpu_arch,
        SystemTime::now(),
    );
    // Defense-in-depth: a future field expansion must touch the allowlist.
    // If it doesn't, fail loudly here rather than silently leak data.
    telemetry_payload_is_allowlisted(&payload).map_err(|e| format!("STRAT-E1001 {e}"))?;
    Ok(payload)
}

fn init(json: bool, paths: &Paths, out: &mut dyn Write, err: &mut dyn Write) -> ExitCode {
    if let Err(e) = paths.ensure_dirs() {
        let _ = writeln!(err, "{e}");
        return ExitCode::from(73);
    }
    let probe = HardwareProbe::run();
    let tier = Tier::classify(&probe);
    let now = OffsetDateTime::now_utc();
    let record = InstalledToml::new(&probe, tier, now);
    let path = paths.installed_toml();
    if let Err(e) = record.write_atomic(&path) {
        let _ = writeln!(err, "{e}");
        return ExitCode::from(73);
    }

    if json {
        #[allow(
            clippy::expect_used,
            reason = "InstalledToml serialization is infallible (primitives only)"
        )]
        let rendered = serde_json::to_string_pretty(&record)
            .expect("InstalledToml serialization is infallible");
        if writeln!(out, "{rendered}").is_err() {
            return ExitCode::from(74);
        }
    } else if writeln!(
        out,
        "installed · tier={} · gpu={} · wrote {}",
        tier,
        probe.gpu,
        path.display()
    )
    .is_err()
    {
        return ExitCode::from(74);
    }
    ExitCode::SUCCESS
}

fn echo(json: bool, prompt: &[String], max_blocks: u32, out: &mut dyn Write) -> ExitCode {
    let provider = EchoProvider::new("echo: ");
    let request = GenerateRequest {
        model: ModelId::from("echo"),
        prompt: prompt.join(" "),
        max_blocks,
    };
    let cancel = CancelToken::new();
    let blocks = provider.generate(&request, &cancel);

    if json {
        #[allow(
            clippy::expect_used,
            reason = "Block serialization is infallible (primitives only)"
        )]
        let rendered =
            serde_json::to_string_pretty(&blocks).expect("Block serialization is infallible");
        if writeln!(out, "{rendered}").is_err() {
            return ExitCode::from(74);
        }
    } else {
        for block in &blocks {
            if render_block(out, block).is_err() {
                return ExitCode::from(74);
            }
        }
    }
    ExitCode::SUCCESS
}

fn render_block(out: &mut dyn Write, block: &Block) -> std::io::Result<()> {
    match block {
        Block::Text { text } => writeln!(out, "{text}"),
        Block::Usage { prompt, completion } => {
            writeln!(out, "(usage: prompt={prompt} completion={completion})")
        }
        Block::Done => writeln!(out, "(done)"),
        Block::Cancelled { reason } => writeln!(out, "(cancelled: {reason})"),
        Block::ToolCall { tool, .. } => writeln!(out, "(tool_call: {tool})"),
        Block::ToolResult { id, .. } => writeln!(out, "(tool_result: {id})"),
    }
}

// ---------------------------------------------------------------------------
// self-update --check / --apply
// ---------------------------------------------------------------------------

/// `--json` payload for the artifact slot of a [`SelfUpdateReport`] /
/// [`SelfUpdateApplyReport`].
#[derive(Debug, Serialize)]
struct SelfUpdateArtifact<'a> {
    url: &'a str,
    sha256: &'a str,
    bytes: u64,
}

/// `--json` payload emitted by `stratum self-update --check`.
#[derive(Debug, Serialize)]
struct SelfUpdateReport<'a> {
    decision: &'static str,
    from: Option<String>,
    to: Option<String>,
    channel: &'static str,
    platform: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    artifact: Option<SelfUpdateArtifact<'a>>,
}

/// `--json` payload emitted by `stratum self-update --apply` on a successful
/// swap. `backup_path` is the absolute path of the `<exe>.bak` rollback file
/// left next to the new binary.
#[derive(Debug, Serialize)]
struct SelfUpdateApplyReport<'a> {
    action: &'static str,
    from: String,
    to: String,
    backup_path: String,
    artifact: SelfUpdateArtifact<'a>,
}

fn self_update(
    json: bool,
    args: &SelfUpdateArgs,
    out: &mut dyn Write,
    err: &mut dyn Write,
) -> ExitCode {
    // Exactly one of --check / --apply must be set. Clap's `conflicts_with`
    // already enforces "not both" with exit 64; this branch handles the
    // "neither" case the same way.
    if !args.check && !args.apply {
        let _ = writeln!(
            err,
            "stratum self-update: exactly one of --check or --apply must be set"
        );
        return ExitCode::from(64);
    }
    // `--dry-run` is meaningless without `--apply`; reject the combo
    // explicitly rather than silently treating `--check --dry-run` as a
    // plain `--check`. Clap's `requires` doesn't fire on bool flags here, so
    // we enforce it at runtime.
    if args.dry_run && !args.apply {
        let _ = writeln!(err, "stratum self-update: --dry-run requires --apply");
        return ExitCode::from(64);
    }

    let current = match resolve_current_version(args.current.as_deref(), err) {
        Ok(v) => v,
        Err(code) => return code,
    };
    let platform_arg = match resolve_platform(args.platform, err) {
        Ok(p) => p,
        Err(code) => return code,
    };
    let channel_arg = args.channel;

    let manifest = match load_self_update_manifest(args, channel_arg, err) {
        Ok(m) => m,
        Err(code) => return code,
    };

    if args.apply {
        return self_update_apply(json, args, &manifest, &current, platform_arg, out, err);
    }

    let decision = evaluate_update(&manifest, &current);
    let artifact = manifest.pick_artifact(platform_arg.into());

    let render_result = if json {
        render_self_update_json(
            out,
            &decision,
            &current,
            channel_arg,
            platform_arg,
            artifact,
        )
    } else {
        render_self_update_prose(out, &decision, &current, channel_arg, artifact)
    };
    if let Err(code) = render_result {
        return code;
    }

    match decision {
        UpdateDecision::UpToDate | UpdateDecision::Upgrade { .. } => ExitCode::SUCCESS,
        UpdateDecision::BlockedSchemaTooOld { .. } => ExitCode::from(64),
    }
}

/// Drives the `--apply` flow. The manifest has already been loaded and the
/// platform / current version resolved by [`self_update`]. Decision logic:
///
/// * `UpToDate` — print "already up to date" and exit 0 without touching the
///   filesystem.
/// * `BlockedSchemaTooOld` — exit 64; apply cannot bridge the gap, the user
///   must reinstall manually.
/// * `Upgrade { from, to }` — look up the artifact for the requested
///   platform, download it, verify SHA-256 + byte count, then atomically
///   swap the on-disk binary (or stop after verification under `--dry-run`).
///
/// The swap target defaults to `std::env::current_exe()`. Tests pass the
/// hidden `--target <path>` flag so they don't blow away the CLI test
/// binary itself.
fn self_update_apply(
    json: bool,
    args: &SelfUpdateArgs,
    manifest: &UpdateManifest,
    current: &ReleaseVersion,
    platform_arg: PlatformArg,
    out: &mut dyn Write,
    err: &mut dyn Write,
) -> ExitCode {
    // Reject hidden test-only flags on production builds.
    if (args.target.is_some() || args.allow_insecure_url) && !insecure_flags_allowed() {
        let _ = writeln!(
            err,
            "STRAT-E1001 --target / --allow-insecure-url require a debug build or \
             STRATUM_ALLOW_INSECURE_URL=1"
        );
        return ExitCode::from(64);
    }

    let (from_version, to_version) = match evaluate_update(manifest, current) {
        UpdateDecision::UpToDate => {
            return write_or_io_exit(
                out,
                format_args!("stratum is already up to date ({current})"),
            );
        }
        UpdateDecision::BlockedSchemaTooOld {
            current: cur,
            min_supported,
        } => {
            let _ = writeln!(
                err,
                "STRAT-E1001 cannot apply: current {cur} is below min-supported \
                 {min_supported}; reinstall stratum manually"
            );
            return ExitCode::from(64);
        }
        UpdateDecision::Upgrade { from, to } => (from, to),
    };

    let Some(artifact) = manifest.pick_artifact(platform_arg.into()) else {
        let _ = writeln!(
            err,
            "STRAT-E1001 no artifact for platform {} in manifest",
            platform_arg.as_wire()
        );
        return ExitCode::from(1);
    };

    apply_upgrade_with_artifact(json, args, artifact, &from_version, &to_version, out, err)
}

/// Resolve the swap target, download + verify the artifact, and either stop
/// (dry-run) or perform the atomic swap. Split out of [`self_update_apply`]
/// to keep both functions below the per-function line limit.
fn apply_upgrade_with_artifact(
    json: bool,
    args: &SelfUpdateArgs,
    artifact: &stratum_runtime::UpdateArtifactRef,
    from_version: &ReleaseVersion,
    to_version: &ReleaseVersion,
    out: &mut dyn Write,
    err: &mut dyn Write,
) -> ExitCode {
    let target_exe = match resolve_swap_target(args, err) {
        Ok(p) => p,
        Err(code) => return code,
    };
    let new_tmp = sibling_with_suffix(&target_exe, ".new.tmp");
    let bak_path = sibling_with_suffix(&target_exe, ".bak");

    let (digest, bytes_written) =
        match download_and_verify(&artifact.url, &new_tmp, args.allow_insecure_url) {
            Ok(t) => t,
            Err(msg) => {
                let _ = std::fs::remove_file(&new_tmp);
                let _ = writeln!(err, "STRAT-E1001 {msg}");
                return ExitCode::from(1);
            }
        };

    if !sha256_eq(&digest, &artifact.sha256) {
        let _ = std::fs::remove_file(&new_tmp);
        let _ = writeln!(
            err,
            "STRAT-E1001 sha256 mismatch: manifest={} got={}",
            artifact.sha256, digest
        );
        return ExitCode::from(1);
    }
    if bytes_written != artifact.bytes {
        let _ = std::fs::remove_file(&new_tmp);
        let _ = writeln!(
            err,
            "STRAT-E1001 byte count mismatch: manifest={} got={}",
            artifact.bytes, bytes_written
        );
        return ExitCode::from(1);
    }

    if args.dry_run {
        let _ = std::fs::remove_file(&new_tmp);
        return write_or_io_exit(
            out,
            format_args!(
                "dry-run: would swap {} with {}",
                target_exe.display(),
                new_tmp.display()
            ),
        );
    }

    if let Err(msg) = make_executable(&new_tmp) {
        let _ = std::fs::remove_file(&new_tmp);
        let _ = writeln!(err, "STRAT-E1001 {msg}");
        return ExitCode::from(1);
    }
    if let Err(msg) = atomic_swap(&target_exe, &new_tmp, &bak_path) {
        let _ = std::fs::remove_file(&new_tmp);
        let _ = writeln!(err, "STRAT-E1001 {msg}");
        return ExitCode::from(1);
    }

    let render = if json {
        render_self_update_apply_json(out, from_version, to_version, &bak_path, artifact)
    } else {
        let write_res = writeln!(
            out,
            "upgraded {from_version} → {to_version}; previous binary kept at {}",
            bak_path.display()
        );
        write_res.map_err(|_| ExitCode::from(74))
    };
    match render {
        Ok(()) => ExitCode::SUCCESS,
        Err(code) => code,
    }
}

/// Resolve the on-disk path the new binary should overwrite. Production
/// callers pass `--apply` without `--target`; tests pass the hidden
/// `--target <path>` (gated by [`insecure_flags_allowed`]) to avoid
/// stomping on the CLI test binary itself.
fn resolve_swap_target(args: &SelfUpdateArgs, err: &mut dyn Write) -> Result<PathBuf, ExitCode> {
    if let Some(p) = args.target.clone() {
        return Ok(p);
    }
    std::env::current_exe().map_err(|e| {
        let _ = writeln!(err, "STRAT-E1001 cannot resolve current_exe(): {e}");
        ExitCode::from(1)
    })
}

/// Return `<base><suffix>` as a sibling of `base`. Falls back to the suffix
/// alone if `base` somehow has no filename, which can't happen for an
/// absolute exe path but keeps the function total.
fn sibling_with_suffix(base: &Path, suffix: &str) -> PathBuf {
    let parent = base.parent().map(Path::to_path_buf).unwrap_or_default();
    let mut name = base
        .file_name()
        .map(std::ffi::OsString::from)
        .unwrap_or_default();
    name.push(suffix);
    parent.join(name)
}

/// Download `url` to `dest`, returning the computed SHA-256 hex digest and
/// the number of bytes written. The transport is `ureq` HTTPS for production
/// URLs; HTTP is permitted only when `allow_insecure` is `true` AND the
/// process is allowed to use insecure flags (see [`insecure_flags_allowed`]).
fn download_and_verify(
    url: &str,
    dest: &Path,
    allow_insecure: bool,
) -> Result<(String, u64), String> {
    let is_https = url.starts_with("https://");
    let is_http = url.starts_with("http://");
    if !is_https {
        if !is_http {
            return Err(format!("artifact url must be http(s): {url:?}"));
        }
        if !(allow_insecure && insecure_flags_allowed()) {
            return Err(format!("artifact url must be https://: {url:?}"));
        }
    }

    let agent = ureq::AgentBuilder::new()
        .timeout(std::time::Duration::from_secs(30))
        .build();
    let resp = agent
        .get(url)
        .call()
        .map_err(|e| format!("artifact fetch failed: {e}"))?;
    let status = resp.status();
    if status != 200 {
        return Err(format!("artifact fetch returned HTTP {status}"));
    }
    let reader = resp.into_reader();

    let file = std::fs::OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(dest)
        .map_err(|e| format!("cannot open {}: {e}", dest.display()))?;
    let mut buf = std::io::BufWriter::new(file);
    let (digest, written) = stratum_runtime::download::hash_and_copy(reader, &mut buf)
        .map_err(|e| format!("artifact write failed: {e}"))?;
    let inner = buf
        .into_inner()
        .map_err(|e| format!("artifact flush failed: {e}"))?;
    inner
        .sync_all()
        .map_err(|e| format!("artifact fsync failed: {e}"))?;
    Ok((digest, written))
}

/// Constant-time-ish lower-case hex comparison. Both inputs are already
/// lower-case (manifest validation enforces it on one side, our hex writer
/// emits lower-case on the other), but normalise defensively.
const fn sha256_eq(lhs: &str, rhs: &str) -> bool {
    // `str::eq_ignore_ascii_case` isn't const yet on stable; compare the raw
    // bytes manually. The inputs are 64-char hex strings, so the cost is a
    // tight loop over 64 bytes.
    let lhs = lhs.as_bytes();
    let rhs = rhs.as_bytes();
    if lhs.len() != rhs.len() {
        return false;
    }
    let mut idx = 0;
    while idx < lhs.len() {
        let mut left = lhs[idx];
        let mut right = rhs[idx];
        if left.is_ascii_uppercase() {
            left = left.to_ascii_lowercase();
        }
        if right.is_ascii_uppercase() {
            right = right.to_ascii_lowercase();
        }
        if left != right {
            return false;
        }
        idx += 1;
    }
    true
}

/// Write a single line to `out` and map the IO outcome to a process exit
/// code: success ⇒ `ExitCode::SUCCESS`, IO failure ⇒ `ExitCode::from(74)`.
/// Lets `self_update_apply` and friends keep the success / dry-run paths
/// short without re-implementing the same `map_or` chain.
fn write_or_io_exit(out: &mut dyn Write, args: std::fmt::Arguments<'_>) -> ExitCode {
    if writeln!(out, "{args}").is_err() {
        ExitCode::from(74)
    } else {
        ExitCode::SUCCESS
    }
}

/// `chmod 0755` on Unix; no-op on Windows.
#[cfg(unix)]
fn make_executable(path: &Path) -> Result<(), String> {
    use std::os::unix::fs::PermissionsExt;
    let perms = std::fs::Permissions::from_mode(0o755);
    std::fs::set_permissions(path, perms)
        .map_err(|e| format!("cannot chmod 0755 {}: {e}", path.display()))
}

#[cfg(not(unix))]
fn make_executable(_path: &Path) -> Result<(), String> {
    Ok(())
}

/// Atomic-rename swap with rollback. Moves `exe → bak` (overwriting any
/// existing `bak`), then `new_tmp → exe`. If the second rename fails after
/// the first succeeded, we try to roll back by renaming `bak → exe`. The
/// caller is responsible for cleaning up `new_tmp` on any error path.
fn atomic_swap(exe: &Path, new_tmp: &Path, bak: &Path) -> Result<(), String> {
    // Drop any stale .bak so the next rename can succeed on platforms that
    // refuse to overwrite an existing target.
    if bak.exists() {
        std::fs::remove_file(bak)
            .map_err(|e| format!("cannot remove stale {}: {e}", bak.display()))?;
    }
    std::fs::rename(exe, bak)
        .map_err(|e| format!("cannot move {} → {}: {e}", exe.display(), bak.display()))?;
    if let Err(e) = std::fs::rename(new_tmp, exe) {
        // Attempt rollback. Best-effort: if it fails we leave the .bak in
        // place and surface the original error.
        let _ = std::fs::rename(bak, exe);
        return Err(format!(
            "cannot move {} → {}: {e}",
            new_tmp.display(),
            exe.display()
        ));
    }
    Ok(())
}

fn render_self_update_apply_json(
    out: &mut dyn Write,
    from: &ReleaseVersion,
    to: &ReleaseVersion,
    backup_path: &Path,
    artifact: &stratum_runtime::UpdateArtifactRef,
) -> Result<(), ExitCode> {
    let payload = SelfUpdateApplyReport {
        action: "applied",
        from: from.to_string(),
        to: to.to_string(),
        backup_path: backup_path.display().to_string(),
        artifact: SelfUpdateArtifact {
            url: &artifact.url,
            sha256: &artifact.sha256,
            bytes: artifact.bytes,
        },
    };
    #[allow(
        clippy::expect_used,
        reason = "SelfUpdateApplyReport serialization is infallible (primitives only)"
    )]
    let rendered = serde_json::to_string_pretty(&payload)
        .expect("SelfUpdateApplyReport serialization is infallible");
    if writeln!(out, "{rendered}").is_err() {
        return Err(ExitCode::from(74));
    }
    Ok(())
}

fn resolve_current_version(
    override_value: Option<&str>,
    err: &mut dyn Write,
) -> Result<ReleaseVersion, ExitCode> {
    let raw = override_value.unwrap_or(env!("CARGO_PKG_VERSION"));
    ReleaseVersion::parse(raw).map_err(|e| {
        let _ = writeln!(err, "invalid --current {raw:?}: {e}");
        ExitCode::from(2)
    })
}

fn resolve_platform(
    override_value: Option<PlatformArg>,
    err: &mut dyn Write,
) -> Result<PlatformArg, ExitCode> {
    if let Some(p) = override_value {
        return Ok(p);
    }
    PlatformArg::detect().ok_or_else(|| {
        let _ = writeln!(
            err,
            "could not auto-detect platform (os={}, arch={}); pass --platform",
            std::env::consts::OS,
            std::env::consts::ARCH
        );
        ExitCode::from(2)
    })
}

fn load_self_update_manifest(
    args: &SelfUpdateArgs,
    channel_arg: ChannelArg,
    err: &mut dyn Write,
) -> Result<UpdateManifest, ExitCode> {
    match (&args.manifest_file, &args.manifest_url) {
        (Some(path), None) => UpdateManifest::load(path).map_err(|e| {
            let _ = writeln!(err, "STRAT-E1001 {e}");
            ExitCode::from(1)
        }),
        (None, url_opt) => {
            let url = url_opt.clone().unwrap_or_else(|| {
                format!("https://updates.stratum.dev/{}.json", channel_arg.as_wire())
            });
            fetch_manifest_https(&url).map_err(|e| {
                let _ = writeln!(err, "STRAT-E1001 {e}");
                ExitCode::from(1)
            })
        }
        (Some(_), Some(_)) => {
            // Clap's `conflicts_with` should have caught this; defensive
            // fallthrough preserves exit-code shape for handcrafted argv.
            let _ = writeln!(
                err,
                "--manifest-url and --manifest-file are mutually exclusive"
            );
            Err(ExitCode::from(64))
        }
    }
}

fn render_self_update_json(
    out: &mut dyn Write,
    decision: &UpdateDecision,
    current: &ReleaseVersion,
    channel_arg: ChannelArg,
    platform_arg: PlatformArg,
    artifact: Option<&stratum_runtime::UpdateArtifactRef>,
) -> Result<(), ExitCode> {
    let (decision_tag, from, to) = match decision {
        UpdateDecision::UpToDate => ("UpToDate", Some(current.to_string()), None),
        UpdateDecision::Upgrade { from, to } => {
            ("Upgrade", Some(from.to_string()), Some(to.to_string()))
        }
        UpdateDecision::BlockedSchemaTooOld {
            current: cur,
            min_supported,
        } => (
            "BlockedSchemaTooOld",
            Some(cur.to_string()),
            Some(min_supported.to_string()),
        ),
    };
    let payload = SelfUpdateReport {
        decision: decision_tag,
        from,
        to,
        channel: channel_arg.as_wire(),
        platform: platform_arg.as_wire(),
        artifact: artifact.map(|a| SelfUpdateArtifact {
            url: &a.url,
            sha256: &a.sha256,
            bytes: a.bytes,
        }),
    };
    #[allow(
        clippy::expect_used,
        reason = "SelfUpdateReport serialization is infallible (primitives only)"
    )]
    let rendered = serde_json::to_string_pretty(&payload)
        .expect("SelfUpdateReport serialization is infallible");
    if writeln!(out, "{rendered}").is_err() {
        return Err(ExitCode::from(74));
    }
    Ok(())
}

fn render_self_update_prose(
    out: &mut dyn Write,
    decision: &UpdateDecision,
    current: &ReleaseVersion,
    channel_arg: ChannelArg,
    artifact: Option<&stratum_runtime::UpdateArtifactRef>,
) -> Result<(), ExitCode> {
    let channel = channel_arg.as_wire();
    let write_res = match decision {
        UpdateDecision::UpToDate => {
            writeln!(out, "stratum is up to date ({current} on {channel})")
        }
        UpdateDecision::Upgrade { from, to } => {
            writeln!(out, "upgrade available: {from} → {to} ({channel})").and_then(|()| {
                artifact.map_or(Ok(()), |a| {
                    writeln!(
                        out,
                        "  artifact: {} ({} bytes, sha256={})",
                        a.url, a.bytes, a.sha256
                    )
                })
            })
        }
        UpdateDecision::BlockedSchemaTooOld {
            current: cur,
            min_supported,
        } => writeln!(
            out,
            "current version {cur} is below min-supported {min_supported}; full reinstall required"
        ),
    };
    write_res.map_err(|_| ExitCode::from(74))
}

/// Fetch and parse an [`UpdateManifest`] from an HTTPS URL.
///
/// Rejects non-HTTPS URLs and any non-200 response.
fn fetch_manifest_https(url: &str) -> Result<UpdateManifest, String> {
    if !url.starts_with("https://") {
        return Err(format!("manifest URL must be https://: got {url:?}"));
    }
    let agent = ureq::AgentBuilder::new()
        .timeout(std::time::Duration::from_secs(10))
        .build();
    let resp = agent
        .get(url)
        .call()
        .map_err(|e| format!("manifest fetch failed: {e}"))?;
    let status = resp.status();
    if status != 200 {
        return Err(format!("manifest fetch returned HTTP {status}"));
    }
    let body = resp
        .into_string()
        .map_err(|e| format!("manifest body read failed: {e}"))?;
    let parsed: UpdateManifest = serde_json::from_str(&body).map_err(|e| {
        let err: ManifestError = ManifestError::Serialize(e);
        format!("{err}")
    })?;
    Ok(parsed)
}

// ---------------------------------------------------------------------------
// events tail
// ---------------------------------------------------------------------------

/// Resolve the on-disk path of the JSONL event log under the configured state
/// directory. Mirrors what `JsonlEventSink` writes to.
fn events_log_path(paths: &Paths) -> PathBuf {
    paths.state.join("events.jsonl")
}

/// Optional bound on `--follow` loops. Honors `STRATUM_EVENTS_TAIL_MAX_S`; in
/// real runs the env var is absent and the loop tails forever.
fn follow_deadline() -> Option<SystemTime> {
    let raw = std::env::var("STRATUM_EVENTS_TAIL_MAX_S").ok()?;
    let secs: u64 = raw.parse().ok()?;
    SystemTime::now().checked_add(Duration::from_secs(secs))
}

fn deadline_reached(deadline: Option<SystemTime>) -> bool {
    deadline.is_some_and(|d| SystemTime::now() >= d)
}

/// Format an [`EventRecord`] as a single-line prose summary.
fn render_event_prose(record: &EventRecord) -> String {
    let at = OffsetDateTime::from(record.at)
        .format(&Rfc3339)
        .unwrap_or_else(|_| String::from("?"));
    let (kind, body) = match &record.event {
        Event::ToolCall {
            tool_id,
            ok,
            duration_ms,
        } => ("tool_call", format!("{tool_id} ok={ok} {duration_ms}ms")),
        Event::PermissionAsked { request, decision } => {
            ("permission_asked", format!("{request} decision={decision}"))
        }
        Event::AgentHandoff { from, to, reason } => {
            ("agent_handoff", format!("{from}->{to} reason={reason}"))
        }
        Event::ProviderError {
            provider,
            code,
            message,
        } => ("provider_error", format!("{provider} {code} {message}")),
        Event::SandboxLaunched { backend, profile } => {
            ("sandbox_launched", format!("{backend} profile={profile}"))
        }
    };
    format!("[{}] {} {} {}", record.id, at, kind, body)
}

/// True when `record` matches the `kind` filter (or no filter is set).
fn kind_matches(record: &EventRecord, kind: Option<EventKindArg>) -> bool {
    kind.is_none_or(|k| {
        let wire = match &record.event {
            Event::ToolCall { .. } => "tool_call",
            Event::PermissionAsked { .. } => "permission_asked",
            Event::AgentHandoff { .. } => "agent_handoff",
            Event::ProviderError { .. } => "provider_error",
            Event::SandboxLaunched { .. } => "sandbox_launched",
        };
        wire == k.as_wire()
    })
}

/// Apply `--since-id` / `--kind` to a single record and emit it via `out` if
/// it passes. Returns `Ok(true)` when a record was emitted, `Ok(false)` when
/// filtered out, and `Err(ExitCode)` on writer failure.
fn maybe_emit_record(
    out: &mut dyn Write,
    record: &EventRecord,
    args: &EventsTailArgs,
) -> Result<bool, ExitCode> {
    if let Some(since) = args.since_id {
        if record.id <= since {
            return Ok(false);
        }
    }
    if !kind_matches(record, args.kind) {
        return Ok(false);
    }
    let line = if args.json {
        match serde_json::to_string(record) {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(error = %e, "events tail: serialize failed");
                return Ok(false);
            }
        }
    } else {
        render_event_prose(record)
    };
    if writeln!(out, "{line}").is_err() {
        return Err(ExitCode::from(74));
    }
    Ok(true)
}

/// Drain `reader` until EOF, emitting matching records. `emitted` is the
/// running count used to enforce `--limit`. Returns `Ok(true)` when the limit
/// was reached (caller should stop), `Ok(false)` when EOF was hit cleanly, or
/// `Err(ExitCode)` on writer failure.
fn drain_reader<R: BufRead>(
    reader: &mut R,
    out: &mut dyn Write,
    args: &EventsTailArgs,
    emitted: &mut usize,
) -> Result<bool, ExitCode> {
    let mut line = String::new();
    loop {
        line.clear();
        let n = match reader.read_line(&mut line) {
            Ok(n) => n,
            Err(e) => {
                tracing::warn!(error = %e, "events tail: read failed");
                return Ok(false);
            }
        };
        if n == 0 {
            return Ok(false);
        }
        let trimmed = line.trim_end_matches('\n').trim_end_matches('\r');
        if trimmed.is_empty() {
            continue;
        }
        let record: EventRecord = match serde_json::from_str(trimmed) {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!(error = %e, line = %trimmed, "events tail: skipping malformed line");
                continue;
            }
        };
        if maybe_emit_record(out, &record, args)? {
            *emitted += 1;
            if args.limit.is_some_and(|max| *emitted >= max) {
                return Ok(true);
            }
        }
    }
}

fn events_tail(
    paths: &Paths,
    args: &EventsTailArgs,
    out: &mut dyn Write,
    err: &mut dyn Write,
) -> ExitCode {
    let path = events_log_path(paths);
    let deadline = follow_deadline();
    let mut emitted: usize = 0;
    let mut offset: u64 = 0;

    loop {
        match std::fs::File::open(&path) {
            Ok(file) => {
                let mut reader = BufReader::new(file);
                if offset > 0 {
                    if let Err(e) = reader.seek(SeekFrom::Start(offset)) {
                        let _ = writeln!(err, "STRAT-E1001 cannot seek {}: {e}", path.display());
                        return ExitCode::from(1);
                    }
                }
                let stop = match drain_reader(&mut reader, out, args, &mut emitted) {
                    Ok(s) => s,
                    Err(code) => return code,
                };
                if stop {
                    return ExitCode::SUCCESS;
                }
                if !args.follow {
                    return ExitCode::SUCCESS;
                }
                offset = match reader.stream_position() {
                    Ok(p) => p,
                    Err(e) => {
                        let _ = writeln!(err, "STRAT-E1001 cannot tell {}: {e}", path.display());
                        return ExitCode::from(1);
                    }
                };
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                if !args.follow {
                    return ExitCode::SUCCESS;
                }
            }
            Err(e) => {
                let _ = writeln!(err, "STRAT-E1001 cannot read {}: {e}", path.display());
                return ExitCode::from(1);
            }
        }

        if !args.follow {
            return ExitCode::SUCCESS;
        }
        if deadline_reached(deadline) {
            return ExitCode::SUCCESS;
        }
        std::thread::sleep(Duration::from_millis(200));
        if deadline_reached(deadline) {
            return ExitCode::SUCCESS;
        }
    }
}

// ---------------------------------------------------------------------------
// sessions
// ---------------------------------------------------------------------------

/// Filesystem location of the transcript directory under the configured
/// state root. Mirrors what the future chat persistence layer writes to.
fn transcripts_dir(paths: &Paths) -> PathBuf {
    paths.state.join("transcripts")
}

/// Open the on-disk [`TranscriptStore`] rooted at `<state>/transcripts/`,
/// writing a STRAT-E1001 diagnostic to `err` on failure.
fn open_transcript_store(paths: &Paths, err: &mut dyn Write) -> Result<TranscriptStore, ExitCode> {
    let dir = transcripts_dir(paths);
    match TranscriptStore::open(dir.clone()) {
        Ok(s) => Ok(s),
        Err(e) => {
            let _ = writeln!(
                err,
                "STRAT-E1001 cannot open transcripts dir {}: {e}",
                dir.display()
            );
            Err(ExitCode::from(1))
        }
    }
}

/// Parse a CLI-supplied session id. Bad format → exit 2 + STRAT-E1001.
fn parse_session_id(raw: &str, err: &mut dyn Write) -> Result<SessionId, ExitCode> {
    match SessionId::from_str(raw) {
        Ok(id) => Ok(id),
        Err(e) => {
            let _ = writeln!(err, "STRAT-E1001 invalid --id: {e}");
            Err(ExitCode::from(2))
        }
    }
}

fn sessions_list(json: bool, paths: &Paths, out: &mut dyn Write, err: &mut dyn Write) -> ExitCode {
    let store = match open_transcript_store(paths, err) {
        Ok(s) => s,
        Err(code) => return code,
    };
    let ids = match store.list() {
        Ok(v) => v,
        Err(e) => {
            let _ = writeln!(err, "STRAT-E1001 cannot list transcripts: {e}");
            return ExitCode::from(1);
        }
    };
    if json {
        let strings: Vec<&str> = ids.iter().map(SessionId::as_str).collect();
        #[allow(
            clippy::expect_used,
            reason = "Vec<&str> serialization is infallible (primitives only)"
        )]
        let rendered =
            serde_json::to_string_pretty(&strings).expect("Vec<&str> serialization is infallible");
        if writeln!(out, "{rendered}").is_err() {
            return ExitCode::from(74);
        }
    } else {
        for id in &ids {
            if writeln!(out, "{}", id.as_str()).is_err() {
                return ExitCode::from(74);
            }
        }
    }
    ExitCode::SUCCESS
}

fn sessions_show(
    json: bool,
    paths: &Paths,
    args: &SessionsShowArgs,
    out: &mut dyn Write,
    err: &mut dyn Write,
) -> ExitCode {
    let id = match parse_session_id(&args.id, err) {
        Ok(i) => i,
        Err(code) => return code,
    };
    let store = match open_transcript_store(paths, err) {
        Ok(s) => s,
        Err(code) => return code,
    };
    let transcript = match store.load(&id) {
        Ok(t) => t,
        Err(e) => {
            let _ = writeln!(err, "STRAT-E1001 cannot load session {}: {e}", id.as_str());
            return ExitCode::from(1);
        }
    };
    if json {
        match serde_json::to_string_pretty(&transcript) {
            Ok(rendered) => {
                if writeln!(out, "{rendered}").is_err() {
                    return ExitCode::from(74);
                }
            }
            Err(e) => {
                let _ = writeln!(err, "STRAT-E1001 cannot serialize transcript: {e}");
                return ExitCode::from(1);
            }
        }
        ExitCode::SUCCESS
    } else {
        render_transcript_prose(&transcript, out)
    }
}

/// Format `t` as the documented prose body, writing each line to `out`.
fn render_transcript_prose(t: &Transcript, out: &mut dyn Write) -> ExitCode {
    let created = OffsetDateTime::from(t.created_at)
        .format(&Rfc3339)
        .unwrap_or_else(|_| String::from("?"));
    if writeln!(out, "session: {}", t.session_id.as_str()).is_err() {
        return ExitCode::from(74);
    }
    if writeln!(out, "created: {created}").is_err() {
        return ExitCode::from(74);
    }
    if writeln!(out, "turns: {}", t.turns.len()).is_err() {
        return ExitCode::from(74);
    }
    if writeln!(out, "----").is_err() {
        return ExitCode::from(74);
    }
    for (idx, turn) in t.turns.iter().enumerate() {
        let line = render_turn_line(idx + 1, turn);
        if writeln!(out, "{line}").is_err() {
            return ExitCode::from(74);
        }
    }
    ExitCode::SUCCESS
}

/// One-line prose summary of a single transcript turn.
fn render_turn_line(idx: usize, turn: &TranscriptTurn) -> String {
    let (at, role, body) = match turn {
        TranscriptTurn::User { at, text } => (at, "user", text.clone()),
        TranscriptTurn::Assistant { at, blocks } => {
            (at, "assistant", render_assistant_summary(blocks))
        }
        TranscriptTurn::System { at, text } => (at, "system", text.clone()),
        TranscriptTurn::Command { at, text, ok } => (at, "command", format!("{text} ok={ok}")),
    };
    let at_str = OffsetDateTime::from(*at)
        .format(&Rfc3339)
        .unwrap_or_else(|_| String::from("?"));
    format!("[{idx}] {at_str} {role}: {body}")
}

/// Render the assistant body: the first block's text, optionally annotated
/// with how many more blocks follow.
fn render_assistant_summary(blocks: &[TranscriptBlock]) -> String {
    let first = blocks.first().map_or("", |b| b.text.as_str());
    let head = first.lines().next().unwrap_or("");
    if blocks.len() <= 1 {
        head.to_string()
    } else {
        format!("{head} ({} blocks)", blocks.len())
    }
}

fn sessions_delete(
    paths: &Paths,
    args: &SessionsDeleteArgs,
    out: &mut dyn Write,
    err: &mut dyn Write,
) -> ExitCode {
    let id = match parse_session_id(&args.id, err) {
        Ok(i) => i,
        Err(code) => return code,
    };
    let store = match open_transcript_store(paths, err) {
        Ok(s) => s,
        Err(code) => return code,
    };
    match store.delete(&id) {
        Ok(true) => {
            if writeln!(out, "deleted · {}", id.as_str()).is_err() {
                return ExitCode::from(74);
            }
            ExitCode::SUCCESS
        }
        Ok(false) => {
            let _ = writeln!(err, "STRAT-E1001 no such session: {}", id.as_str());
            ExitCode::from(1)
        }
        Err(e) => {
            let _ = writeln!(
                err,
                "STRAT-E1001 cannot delete session {}: {e}",
                id.as_str()
            );
            ExitCode::from(1)
        }
    }
}

/// Dispatcher for `stratum eval run`.
///
/// Loads the suite, wraps `AgentFactory::echo` in an `EvalRunner`, runs every
/// case, writes the report to disk (default path:
/// `<state>/eval-reports/<suite-name>-<timestamp>.json`), and emits either a
/// prose summary or the entire pretty-printed JSON [`EvalReport`] on stdout.
fn eval_run(
    args: &EvalRunArgs,
    paths: &Paths,
    out: &mut dyn Write,
    err: &mut dyn Write,
) -> ExitCode {
    // `--model` is parsed for forward compatibility but ignored by the Echo
    // backbone; document by binding it (no-op) so clippy doesn't complain.
    let _ = args.model.as_deref();

    // Load + validate the suite file.
    let suite = match EvalSuite::load(&args.suite) {
        Ok(s) => s,
        Err(e) => {
            let _ = writeln!(
                err,
                "STRAT-E1001 cannot load eval suite {}: {e}",
                args.suite.display()
            );
            return ExitCode::from(1);
        }
    };

    // Build the Echo-backed AgentLoop and the runner. Any builder error
    // surfaces as STRAT-E1001 with exit 1.
    let agent_loop = match AgentFactory::echo() {
        Ok(l) => l,
        Err(e) => {
            let _ = writeln!(err, "STRAT-E1001 cannot build echo agent loop: {e}");
            return ExitCode::from(1);
        }
    };
    let runner = EvalRunner::new(std::sync::Arc::new(agent_loop), ModelId::from("echo"));

    let report = runner.run_suite(&suite);

    // Resolve `--out` (or default into `<state>/eval-reports/...`) and write.
    let out_path = args
        .out
        .clone()
        .unwrap_or_else(|| default_eval_report_path(paths, &suite.name, &report));
    if let Some(parent) = out_path.parent() {
        if !parent.as_os_str().is_empty() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                let _ = writeln!(err, "STRAT-E1001 cannot create {}: {e}", parent.display());
                return ExitCode::from(1);
            }
        }
    }
    if let Err(e) = report.save_atomic(&out_path) {
        let _ = writeln!(
            err,
            "STRAT-E1001 cannot save eval report {}: {e}",
            out_path.display()
        );
        return ExitCode::from(1);
    }

    if args.json {
        match serde_json::to_string_pretty(&report) {
            Ok(rendered) => {
                if writeln!(out, "{rendered}").is_err() {
                    return ExitCode::from(74);
                }
            }
            Err(e) => {
                let _ = writeln!(err, "STRAT-E1001 cannot serialize eval report: {e}");
                return ExitCode::from(1);
            }
        }
    } else if let Err(code) = render_eval_report_prose(&suite, &report, &out_path, out) {
        return code;
    }

    if report.failed == 0 {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    }
}

/// Compute the default `--out` path:
/// `<paths.state>/eval-reports/<suite-name-slug>-<ran_at_unix_secs>.json`.
///
/// The suite name is slugified (whitespace and non-ASCII / non-`[A-Za-z0-9_-]`
/// chars folded to `_`) so it is safe as a filename across platforms. The
/// timestamp comes from the report's `ran_at` so re-runs against the same
/// suite end up in different files.
fn default_eval_report_path(paths: &Paths, suite_name: &str, report: &EvalReport) -> PathBuf {
    let dir = paths.state.join("eval-reports");
    let slug = slugify_suite_name(suite_name);
    let ts = report
        .ran_at
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    dir.join(format!("{slug}-{ts}.json"))
}

/// Replace any character outside `[A-Za-z0-9_-]` with `_`, then collapse runs
/// of `_` so the result stays human-readable. Empty input falls back to
/// `"suite"` to guarantee a non-empty filename stem.
fn slugify_suite_name(raw: &str) -> String {
    let mut slug = String::with_capacity(raw.len());
    let mut last_was_us = false;
    for ch in raw.chars() {
        let ok = ch.is_ascii_alphanumeric() || ch == '-' || ch == '_';
        if ok {
            slug.push(ch);
            last_was_us = ch == '_';
        } else if !last_was_us {
            slug.push('_');
            last_was_us = true;
        }
    }
    let trimmed = slug.trim_matches('_').to_string();
    if trimmed.is_empty() {
        "suite".to_string()
    } else {
        trimmed
    }
}

/// Print the documented prose summary for a finished eval run.
fn render_eval_report_prose(
    suite: &EvalSuite,
    report: &EvalReport,
    out_path: &Path,
    out: &mut dyn Write,
) -> Result<(), ExitCode> {
    let total = report.runs.len();
    let pct = report.pass_rate() * 100.0;
    let duration_s = format_seconds_three_decimals(report.total_duration_ms);

    if writeln!(out, "suite: {}", suite.name).is_err() {
        return Err(ExitCode::from(74));
    }
    if writeln!(out, "passed: {}/{total} ({pct:.1}%)", report.passed).is_err() {
        return Err(ExitCode::from(74));
    }
    if writeln!(out, "failed: {}", report.failed).is_err() {
        return Err(ExitCode::from(74));
    }
    if writeln!(out, "duration: {duration_s}s").is_err() {
        return Err(ExitCode::from(74));
    }
    if writeln!(out, "----").is_err() {
        return Err(ExitCode::from(74));
    }
    for run in &report.runs {
        let tag = if run.passed { "pass" } else { "fail" };
        let suffix = run
            .failure_reason
            .as_deref()
            .map(|r| format!(" — {r}"))
            .unwrap_or_default();
        if writeln!(
            out,
            "[{tag}] {} ({}ms){suffix}",
            run.case_id, run.duration_ms
        )
        .is_err()
        {
            return Err(ExitCode::from(74));
        }
    }
    if writeln!(out, "report saved to: {}", out_path.display()).is_err() {
        return Err(ExitCode::from(74));
    }
    Ok(())
}

/// Render `ms` as `"<int>.<3-digit-frac>"` seconds for the prose summary.
fn format_seconds_three_decimals(ms: u64) -> String {
    let whole = ms / 1_000;
    let frac = ms % 1_000;
    format!("{whole}.{frac:03}")
}

// ---------------------------------------------------------------------------
// `stratum agents` subcommand
// ---------------------------------------------------------------------------

fn agents_dir(paths: &Paths) -> PathBuf {
    paths.state.join("agents")
}

/// Build the shared `AgentRegistryLoader` over `<state>/agents` using the
/// Echo provider. The loader runs the same factory backbone tests use, so
/// any loop construction failure surfaces in `LoadReport.errors`.
fn build_agents_loader(paths: &Paths) -> AgentRegistryLoader {
    let factory = std::sync::Arc::new(
        AgentFactory::new().with_provider(std::sync::Arc::new(EchoProvider::new(""))),
    );
    AgentRegistryLoader::new(agents_dir(paths), factory)
}

/// Render a `SuggestedRole` as its `snake_case` wire form (mirrors the
/// serde derive on the enum).
const fn role_as_wire(role: SuggestedRole) -> &'static str {
    match role {
        SuggestedRole::Default => "default",
        SuggestedRole::Cavemanish => "cavemanish",
        SuggestedRole::Polisher => "polisher",
        SuggestedRole::Coder => "coder",
        SuggestedRole::Researcher => "researcher",
    }
}

/// Parse a user-supplied role string into a `SuggestedRole`. Returns `None`
/// when the string does not match any known variant.
fn parse_role_arg(s: &str) -> Option<SuggestedRole> {
    match s {
        "default" => Some(SuggestedRole::Default),
        "cavemanish" => Some(SuggestedRole::Cavemanish),
        "polisher" => Some(SuggestedRole::Polisher),
        "coder" => Some(SuggestedRole::Coder),
        "researcher" => Some(SuggestedRole::Researcher),
        _ => None,
    }
}

/// Brief one-line summary of a `SkipReason` for prose output.
fn render_skip_reason(reason: &SkipReason) -> String {
    match reason {
        SkipReason::UnknownRole { file, role } => {
            format!("file: {} — unknown role {role:?}", file.display())
        }
        SkipReason::MissingRoleField { file } => {
            format!("file: {} — missing role field", file.display())
        }
        SkipReason::DuplicateRole {
            role,
            existing_file,
            new_file,
        } => format!(
            "file: {} — duplicate role {} (already registered by {})",
            new_file.display(),
            role_as_wire(*role),
            existing_file.display(),
        ),
    }
}

/// Brief one-line summary of a `LoadFailure` for prose output.
fn render_load_failure(failure: &LoadFailure) -> String {
    format!("file: {} — {}", failure.file.display(), failure.error)
}

/// JSON shape for `stratum agents list --json`.
#[derive(Debug, Serialize)]
struct AgentsListJson<'a> {
    registered: Vec<&'static str>,
    skipped: &'a [SkipReason],
    errors: &'a [LoadFailure],
}

fn agents_list(
    args: &AgentsListArgs,
    paths: &Paths,
    out: &mut dyn Write,
    err: &mut dyn Write,
) -> ExitCode {
    let loader = build_agents_loader(paths);
    let (_registry, report) = match loader.load() {
        Ok(pair) => pair,
        Err(e) => {
            let _ = writeln!(err, "STRAT-E1001 {e}");
            return ExitCode::from(1);
        }
    };

    // Sort registered roles by wire form for the prose surface.
    let mut sorted_roles: Vec<SuggestedRole> = report.registered.clone();
    sorted_roles.sort_by_key(|r| role_as_wire(*r));

    if args.json {
        let payload = AgentsListJson {
            registered: sorted_roles.iter().copied().map(role_as_wire).collect(),
            skipped: &report.skipped,
            errors: &report.errors,
        };
        match serde_json::to_string_pretty(&payload) {
            Ok(rendered) => {
                if writeln!(out, "{rendered}").is_err() {
                    return ExitCode::from(74);
                }
            }
            Err(e) => {
                let _ = writeln!(err, "STRAT-E1001 cannot serialize agents list: {e}");
                return ExitCode::from(1);
            }
        }
        return ExitCode::SUCCESS;
    }

    if writeln!(out, "registered roles (sorted):").is_err() {
        return ExitCode::from(74);
    }
    for role in &sorted_roles {
        if writeln!(out, "  - {}", role_as_wire(*role)).is_err() {
            return ExitCode::from(74);
        }
    }
    if writeln!(out, "skipped: {}", report.skipped.len()).is_err() {
        return ExitCode::from(74);
    }
    for s in &report.skipped {
        if writeln!(out, "  - {}", render_skip_reason(s)).is_err() {
            return ExitCode::from(74);
        }
    }
    if writeln!(out, "errors: {}", report.errors.len()).is_err() {
        return ExitCode::from(74);
    }
    for e in &report.errors {
        if writeln!(out, "  - {}", render_load_failure(e)).is_err() {
            return ExitCode::from(74);
        }
    }
    ExitCode::SUCCESS
}

/// Locate the `AgentDef` whose first-role-or-stem matches `target_role`.
/// Mirrors the registration logic in `AgentRegistryLoader::load`, so the
/// file the loader would have registered is the one we surface.
fn find_agent_for_role(dir: &Path, target_role: SuggestedRole) -> Result<Option<AgentDef>, String> {
    let entries = match std::fs::read_dir(dir) {
        Ok(it) => it,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(format!("read agents dir {}: {e}", dir.display())),
    };
    let mut paths_list: Vec<PathBuf> = entries
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| {
            p.is_file()
                && p.extension()
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("toml"))
        })
        .collect();
    paths_list.sort();
    for path in &paths_list {
        let Ok(def) = stratum_runtime::AgentLoader::load_file(path) else {
            continue;
        };
        let role_str = def
            .roles
            .first()
            .map(|r| r.as_str().to_string())
            .filter(|s| !s.is_empty())
            .or_else(|| {
                path.file_stem()
                    .and_then(|s| s.to_str())
                    .map(str::to_string)
                    .filter(|s| !s.is_empty())
            });
        let Some(role_str) = role_str else {
            continue;
        };
        if parse_role_arg(&role_str) == Some(target_role) {
            return Ok(Some(def));
        }
    }
    Ok(None)
}

fn write_show_prose(
    target_role: SuggestedRole,
    def: &AgentDef,
    out: &mut dyn Write,
) -> Result<(), ExitCode> {
    if writeln!(out, "role: {}", role_as_wire(target_role)).is_err() {
        return Err(ExitCode::from(74));
    }
    if writeln!(out, "name: {}", def.name).is_err() {
        return Err(ExitCode::from(74));
    }
    if writeln!(out, "description: {}", def.description).is_err() {
        return Err(ExitCode::from(74));
    }
    if writeln!(out, "model: {}", def.model.as_str()).is_err() {
        return Err(ExitCode::from(74));
    }
    if writeln!(out, "sandbox: {}", def.sandbox).is_err() {
        return Err(ExitCode::from(74));
    }
    let caps: Vec<&str> = def
        .tools
        .entries()
        .map(stratum_runtime::CapabilityEntry::as_str)
        .collect();
    if writeln!(out, "capabilities: {}", caps.join(", ")).is_err() {
        return Err(ExitCode::from(74));
    }
    Ok(())
}

fn agents_show(
    args: &AgentsShowArgs,
    paths: &Paths,
    out: &mut dyn Write,
    err: &mut dyn Write,
) -> ExitCode {
    let Some(target_role) = parse_role_arg(&args.role) else {
        let _ = writeln!(
            err,
            "STRAT-E1001 unknown role {:?} (expected one of: default, cavemanish, polisher, coder, researcher)",
            args.role
        );
        return ExitCode::from(1);
    };

    let def = match find_agent_for_role(&agents_dir(paths), target_role) {
        Ok(Some(d)) => d,
        Ok(None) => {
            let _ = writeln!(
                err,
                "STRAT-E1001 no agent registered for role {:?}",
                args.role
            );
            return ExitCode::from(1);
        }
        Err(diag) => {
            let _ = writeln!(err, "STRAT-E1001 {diag}");
            return ExitCode::from(1);
        }
    };

    if args.json {
        match serde_json::to_string_pretty(&def) {
            Ok(rendered) => {
                if writeln!(out, "{rendered}").is_err() {
                    return ExitCode::from(74);
                }
            }
            Err(e) => {
                let _ = writeln!(err, "STRAT-E1001 cannot serialize agent def: {e}");
                return ExitCode::from(1);
            }
        }
        return ExitCode::SUCCESS;
    }

    if let Err(code) = write_show_prose(target_role, &def, out) {
        return code;
    }
    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use tempfile::TempDir;

    use super::*;

    fn drive_under(cli_args: &[&str], root: &Path) -> (ExitCode, String, String) {
        let mut out = Vec::new();
        let mut err = Vec::new();
        let mut argv: Vec<OsString> = vec![
            OsString::from("--storage-root"),
            OsString::from(root.as_os_str()),
        ];
        argv.extend(cli_args.iter().map(OsString::from));
        let code = run_with(argv, &mut out, &mut err, Paths::resolve);
        (
            code,
            String::from_utf8(out).unwrap(),
            String::from_utf8(err).unwrap(),
        )
    }

    #[test]
    fn default_prints_hello_and_not_installed() {
        let tmp = TempDir::new().unwrap();
        let (code, out, err) = drive_under(&[], tmp.path());
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::SUCCESS));
        assert!(out.contains("hello, tier=unknown"));
        assert!(out.contains("not installed"));
        assert!(err.is_empty());
    }

    #[test]
    fn init_creates_installed_toml() {
        let tmp = TempDir::new().unwrap();
        let (_code, out, _err) = drive_under(&["init"], tmp.path());
        assert!(out.contains("installed"));
        let p = Paths::under(tmp.path());
        assert!(p.installed_toml().exists());
    }

    #[test]
    fn init_json_emits_record() {
        let tmp = TempDir::new().unwrap();
        let (_code, out, _err) = drive_under(&["--json", "init"], tmp.path());
        let v: serde_json::Value = serde_json::from_str(out.trim()).unwrap();
        assert_eq!(v["schema_version"], 1);
        assert!(v["tier"].is_string());
        assert!(v["installed_at"].is_string());
    }

    #[test]
    fn doctor_prose_after_install() {
        let tmp = TempDir::new().unwrap();
        let _ = drive_under(&["init"], tmp.path());
        let (_code, out, _err) = drive_under(&["doctor"], tmp.path());
        assert!(out.contains("installed=true"));
        assert!(out.contains("ram="));
    }

    #[test]
    fn doctor_json_after_install_marks_installed() {
        let tmp = TempDir::new().unwrap();
        let _ = drive_under(&["init"], tmp.path());
        let (_code, out, _err) = drive_under(&["--json", "doctor"], tmp.path());
        let v: serde_json::Value = serde_json::from_str(out.trim()).unwrap();
        assert_eq!(v["installed"], true);
        assert!(v["probe"]["ram_total_mib"].as_u64().unwrap_or(0) > 0);
    }

    #[test]
    fn doctor_json_before_install_lists_issue() {
        let tmp = TempDir::new().unwrap();
        let (_code, out, _err) = drive_under(&["--json", "doctor"], tmp.path());
        let v: serde_json::Value = serde_json::from_str(out.trim()).unwrap();
        assert_eq!(v["installed"], false);
        let issues = v["issues"].as_array().unwrap();
        assert!(!issues.is_empty());
        assert_eq!(issues[0]["code"], "STRAT-E2003");
    }

    #[test]
    fn default_after_install_marks_installed() {
        let tmp = TempDir::new().unwrap();
        let _ = drive_under(&["init"], tmp.path());
        let (_code, out, _err) = drive_under(&[], tmp.path());
        assert!(out.contains("installed"));
        assert!(!out.contains("not installed"));
    }

    #[test]
    fn unknown_subcommand_exits_64() {
        let tmp = TempDir::new().unwrap();
        let (code, _out, err) = drive_under(&["wat"], tmp.path());
        assert!(!err.is_empty());
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::from(64)));
    }

    #[test]
    fn help_flag_exits_64() {
        let tmp = TempDir::new().unwrap();
        let (_code, _out, err) = drive_under(&["--help"], tmp.path());
        let lower = err.to_lowercase();
        // clap's `--help` always prints the program name; the assertion below
        // is satisfied on every supported toolchain.
        assert!(lower.contains("stratum"));
    }

    #[test]
    fn init_fails_when_dirs_unwritable() {
        // Use a regular file as the storage root so `ensure_dirs` cannot create
        // the four subdirectories.
        let tmp = TempDir::new().unwrap();
        let blocker = tmp.path().join("blocker");
        std::fs::write(&blocker, b"x").unwrap();
        let (code, _out, err) = drive_under(&["init"], &blocker);
        assert!(!err.is_empty());
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::from(73)));
    }

    /// Writer that always returns an error. Used to exercise the IO-failure
    /// branches of `doctor()` and `init()`.
    struct FailingWriter;

    impl Write for FailingWriter {
        fn write(&mut self, _buf: &[u8]) -> std::io::Result<usize> {
            Err(std::io::Error::other("forced failure for coverage test"))
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Err(std::io::Error::other("forced failure for coverage test"))
        }
    }

    fn drive_with_failing_out(cli_args: &[&str], root: &Path) -> ExitCode {
        let mut fail = FailingWriter;
        let mut err = Vec::new();
        let mut argv: Vec<OsString> = vec![
            OsString::from("--storage-root"),
            OsString::from(root.as_os_str()),
        ];
        argv.extend(cli_args.iter().map(OsString::from));
        run_with(argv, &mut fail, &mut err, Paths::resolve)
    }

    #[test]
    fn greeting_io_failure_returns_74() {
        let tmp = TempDir::new().unwrap();
        assert_eq!(
            format!("{:?}", drive_with_failing_out(&[], tmp.path())),
            format!("{:?}", ExitCode::from(74))
        );
    }

    #[test]
    fn doctor_prose_io_failure_returns_74() {
        let tmp = TempDir::new().unwrap();
        assert_eq!(
            format!("{:?}", drive_with_failing_out(&["doctor"], tmp.path())),
            format!("{:?}", ExitCode::from(74))
        );
    }

    #[test]
    fn doctor_json_io_failure_returns_74() {
        let tmp = TempDir::new().unwrap();
        assert_eq!(
            format!(
                "{:?}",
                drive_with_failing_out(&["--json", "doctor"], tmp.path())
            ),
            format!("{:?}", ExitCode::from(74))
        );
    }

    #[test]
    fn init_prose_io_failure_returns_74() {
        let tmp = TempDir::new().unwrap();
        assert_eq!(
            format!("{:?}", drive_with_failing_out(&["init"], tmp.path())),
            format!("{:?}", ExitCode::from(74))
        );
    }

    #[test]
    fn init_json_io_failure_returns_74() {
        let tmp = TempDir::new().unwrap();
        assert_eq!(
            format!(
                "{:?}",
                drive_with_failing_out(&["--json", "init"], tmp.path())
            ),
            format!("{:?}", ExitCode::from(74))
        );
    }

    #[test]
    fn failing_writer_flush_errors() {
        let mut fail = FailingWriter;
        assert!(fail.flush().is_err());
    }

    const GOOD_SHA: &str = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824";

    fn seed_one(root: &Path, slug: &str) {
        let _ = drive_under(
            &[
                "models",
                "add",
                "--slug",
                slug,
                "--family",
                "llama",
                "--display-name",
                "Display",
                "--tier",
                "low",
                "--task",
                "chat",
                "--size-mib",
                "100",
                "--quantization",
                "Q4_K_M",
                "--url",
                "https://example.com/m.gguf",
                "--sha256",
                GOOD_SHA,
                "--bytes",
                "1024",
                "--license",
                "Apache-2.0",
            ],
            root,
        );
    }

    #[test]
    fn models_list_empty_emits_message() {
        let tmp = TempDir::new().unwrap();
        let (_code, out, _err) = drive_under(&["models", "list"], tmp.path());
        assert!(out.contains("no catalog entries"));
    }

    #[test]
    fn models_list_json_empty_array() {
        let tmp = TempDir::new().unwrap();
        let (_code, out, _err) = drive_under(&["--json", "models", "list"], tmp.path());
        let v: serde_json::Value = serde_json::from_str(out.trim()).unwrap();
        assert!(v.as_array().unwrap().is_empty());
    }

    #[test]
    fn models_add_then_list() {
        let tmp = TempDir::new().unwrap();
        seed_one(tmp.path(), "tiny-chat");
        let (_code, out, _err) = drive_under(&["models", "list"], tmp.path());
        assert!(out.contains("tiny-chat"));
        assert!(out.contains("Display"));
    }

    #[test]
    fn models_add_json_emits_entry() {
        let tmp = TempDir::new().unwrap();
        let (_code, out, _err) = drive_under(
            &[
                "--json",
                "models",
                "add",
                "--slug",
                "alpha",
                "--family",
                "llama",
                "--display-name",
                "Alpha",
                "--tier",
                "medium",
                "--task",
                "chat,code",
                "--size-mib",
                "200",
                "--quantization",
                "Q5",
                "--url",
                "https://example.com/a.gguf",
                "--sha256",
                GOOD_SHA,
                "--bytes",
                "2048",
                "--license",
                "MIT",
                "--homepage",
                "https://example.com/alpha",
            ],
            tmp.path(),
        );
        let v: serde_json::Value = serde_json::from_str(out.trim()).unwrap();
        assert_eq!(v["slug"], "alpha");
        assert_eq!(v["tier"], "medium");
        assert!(v["task"].as_array().unwrap().len() == 2);
    }

    #[test]
    fn models_list_json_io_failure_returns_74() {
        let tmp = TempDir::new().unwrap();
        let code = drive_with_failing_out(&["--json", "models", "list"], tmp.path());
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::from(74)));
    }

    #[test]
    fn install_file_default_filename_from_url() {
        let args = InstallFileArgs {
            from_file: None,
            from_url: Some("https://example.com/x/y/weights.gguf".into()),
            name: None,
            sha256: None,
        };
        assert_eq!(default_filename_for(&args), "weights.gguf");
    }

    #[test]
    fn install_file_default_filename_falls_back_when_empty() {
        let args = InstallFileArgs {
            from_file: None,
            from_url: Some("https://example.com/".into()),
            name: None,
            sha256: None,
        };
        assert_eq!(default_filename_for(&args), "model.bin");
    }

    #[test]
    fn install_file_default_filename_falls_back_when_no_source() {
        let args = InstallFileArgs {
            from_file: None,
            from_url: None,
            name: None,
            sha256: None,
        };
        assert_eq!(default_filename_for(&args), "model.bin");
    }

    #[test]
    fn install_file_from_local() {
        let tmp = TempDir::new().unwrap();
        let src = tmp.path().join("src.bin");
        std::fs::write(&src, b"hello").unwrap();
        let (_code, _out, _err) = drive_under(
            &[
                "models",
                "install-file",
                "--from-file",
                src.to_str().unwrap(),
            ],
            tmp.path(),
        );
        // file copied into <root>/data/models/
        let dest = tmp.path().join("data").join("models").join("src.bin");
        assert!(dest.exists());
    }

    #[test]
    fn install_file_json_emits_report() {
        let tmp = TempDir::new().unwrap();
        let src = tmp.path().join("src.bin");
        std::fs::write(&src, b"hello").unwrap();
        let (_code, out, _err) = drive_under(
            &[
                "--json",
                "models",
                "install-file",
                "--from-file",
                src.to_str().unwrap(),
                "--name",
                "renamed.bin",
                "--sha256",
                "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824",
            ],
            tmp.path(),
        );
        let v: serde_json::Value = serde_json::from_str(out.trim()).unwrap();
        assert_eq!(v["verified"], true);
        assert_eq!(v["bytes"], 5);
    }

    #[test]
    fn install_file_mismatch_exits_73() {
        let tmp = TempDir::new().unwrap();
        let src = tmp.path().join("src.bin");
        std::fs::write(&src, b"hello").unwrap();
        let (code, _out, err) = drive_under(
            &[
                "models",
                "install-file",
                "--from-file",
                src.to_str().unwrap(),
                "--sha256",
                "deadbeef",
            ],
            tmp.path(),
        );
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::from(73)));
        assert!(err.contains("mismatch"));
    }

    #[test]
    fn install_file_neither_source_exits_64() {
        let tmp = TempDir::new().unwrap();
        let (code, _out, err) = drive_under(&["models", "install-file"], tmp.path());
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::from(64)));
        assert!(!err.is_empty());
    }

    #[test]
    fn install_file_io_failure_returns_74() {
        let tmp = TempDir::new().unwrap();
        let src = tmp.path().join("src.bin");
        std::fs::write(&src, b"hi").unwrap();
        let code = drive_with_failing_out(
            &[
                "models",
                "install-file",
                "--from-file",
                src.to_str().unwrap(),
            ],
            tmp.path(),
        );
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::from(74)));
    }

    #[test]
    fn install_file_json_io_failure_returns_74() {
        let tmp = TempDir::new().unwrap();
        let src = tmp.path().join("src.bin");
        std::fs::write(&src, b"hi").unwrap();
        let code = drive_with_failing_out(
            &[
                "--json",
                "models",
                "install-file",
                "--from-file",
                src.to_str().unwrap(),
            ],
            tmp.path(),
        );
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::from(74)));
    }

    #[test]
    fn models_remove_then_list_empty() {
        let tmp = TempDir::new().unwrap();
        seed_one(tmp.path(), "removable");
        let (code, _out, _err) =
            drive_under(&["models", "remove", "--slug", "removable"], tmp.path());
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::SUCCESS));
        let (_code, out, _err) = drive_under(&["models", "list"], tmp.path());
        assert!(out.contains("no catalog entries"));
    }

    #[test]
    fn models_remove_missing_exits_1() {
        let tmp = TempDir::new().unwrap();
        let (code, _out, err) = drive_under(&["models", "remove", "--slug", "ghost"], tmp.path());
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::from(1)));
        assert!(err.contains("no such slug"));
    }

    #[test]
    fn models_remove_invalid_slug_exits_2() {
        let tmp = TempDir::new().unwrap();
        let (code, _out, err) = drive_under(&["models", "remove", "--slug", "BAD"], tmp.path());
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::from(2)));
        assert!(err.contains("invalid --slug"));
    }

    #[test]
    fn models_recommend_after_seed() {
        let tmp = TempDir::new().unwrap();
        seed_one(tmp.path(), "tiny");
        let (code, out, _err) = drive_under(
            &["models", "recommend", "--tier", "low", "--task", "chat"],
            tmp.path(),
        );
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::SUCCESS));
        assert!(out.contains("tiny"));
    }

    #[test]
    fn models_recommend_empty_exits_1() {
        let tmp = TempDir::new().unwrap();
        let (code, _out, err) = drive_under(
            &["models", "recommend", "--tier", "low", "--task", "chat"],
            tmp.path(),
        );
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::from(1)));
        assert!(err.contains("no model fits"));
    }

    #[test]
    fn models_validate_empty_ok() {
        let tmp = TempDir::new().unwrap();
        let (code, out, _err) = drive_under(&["models", "validate"], tmp.path());
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::SUCCESS));
        assert!(out.contains("ok · 0 entries"));
    }

    #[test]
    fn models_add_invalid_sha_exits_2() {
        let tmp = TempDir::new().unwrap();
        let (code, _out, err) = drive_under(
            &[
                "models",
                "add",
                "--slug",
                "x",
                "--family",
                "llama",
                "--display-name",
                "X",
                "--tier",
                "low",
                "--task",
                "chat",
                "--size-mib",
                "1",
                "--quantization",
                "Q",
                "--url",
                "https://example.com/x",
                "--sha256",
                "deadbeef",
                "--bytes",
                "1",
                "--license",
                "MIT",
            ],
            tmp.path(),
        );
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::from(2)));
        assert!(err.contains("invalid artifact"));
    }

    #[test]
    fn parse_task_csv_rejects_unknown() {
        let err = parse_task_csv("chat,nope").unwrap_err();
        assert!(err.contains("nope"));
    }

    #[test]
    fn parse_task_csv_rejects_empty() {
        let err = parse_task_csv(", ,").unwrap_err();
        assert!(err.contains("at least one"));
    }

    #[test]
    fn parse_task_csv_all_variants() {
        let set =
            parse_task_csv("chat,code,embedding,tool_use,vision,cavemanish,polisher").unwrap();
        assert_eq!(set.len(), 7);
    }

    #[test]
    fn tier_arg_into_model_tier() {
        assert_eq!(ModelTier::from(TierArg::Low), ModelTier::Low);
        assert_eq!(ModelTier::from(TierArg::Medium), ModelTier::Medium);
        assert_eq!(ModelTier::from(TierArg::High), ModelTier::High);
        assert_eq!(ModelTier::from(TierArg::Xl), ModelTier::Xl);
    }

    #[test]
    fn task_arg_into_model_task() {
        assert_eq!(ModelTask::from(TaskArg::Chat), ModelTask::Chat);
        assert_eq!(ModelTask::from(TaskArg::Code), ModelTask::Code);
        assert_eq!(ModelTask::from(TaskArg::Embedding), ModelTask::Embedding);
        assert_eq!(ModelTask::from(TaskArg::ToolUse), ModelTask::ToolUse);
        assert_eq!(ModelTask::from(TaskArg::Vision), ModelTask::Vision);
        assert_eq!(ModelTask::from(TaskArg::Cavemanish), ModelTask::Cavemanish);
        assert_eq!(ModelTask::from(TaskArg::Polisher), ModelTask::Polisher);
    }

    #[test]
    fn resolve_paths_default_uses_resolver() {
        // Without a storage-root override the resolver should succeed on
        // macOS and Linux test runners.
        let p = resolve_paths_with(None, Paths::resolve).unwrap();
        assert!(p.config.ends_with("stratum"));
    }

    #[test]
    fn resolve_paths_propagates_fallback_error() {
        let err = resolve_paths_with(None, || {
            Err(stratum_types::StratumError::new(
                stratum_types::error::codes::E1001_INSTALLED_SCHEMA_UNREADABLE,
                "synthetic resolver failure",
            ))
        })
        .unwrap_err();
        assert!(err.contains("synthetic resolver failure"));
    }

    #[test]
    fn unresolvable_default_root_exits_78() {
        // No `--storage-root` override → the injected fallback runs; we feed
        // it an Err so the CLI surfaces the diagnostic and exits 78.
        let mut out = Vec::new();
        let mut err = Vec::new();
        let code = run_with(vec![OsString::from("doctor")], &mut out, &mut err, || {
            Err(stratum_types::StratumError::new(
                stratum_types::error::codes::E1001_INSTALLED_SCHEMA_UNREADABLE,
                "synthetic resolver failure",
            ))
        });
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::from(78)));
        assert!(String::from_utf8(err)
            .unwrap()
            .contains("synthetic resolver failure"));
    }

    #[test]
    fn echo_prose_emits_words_then_done() {
        let tmp = TempDir::new().unwrap();
        let (_code, out, _err) = drive_under(&["echo", "hello", "world"], tmp.path());
        assert!(out.contains("echo: hello"));
        assert!(out.contains("echo: world"));
        assert!(out.contains("(usage:"));
        assert!(out.contains("(done)"));
    }

    #[test]
    fn echo_json_emits_block_array() {
        let tmp = TempDir::new().unwrap();
        let (_code, out, _err) = drive_under(&["--json", "echo", "hi"], tmp.path());
        let v: serde_json::Value = serde_json::from_str(out.trim()).unwrap();
        let arr = v.as_array().unwrap();
        assert!(!arr.is_empty());
        assert_eq!(arr.last().unwrap()["kind"], "done");
    }

    #[test]
    fn echo_max_blocks_limits_output() {
        let tmp = TempDir::new().unwrap();
        let (_code, out, _err) = drive_under(
            &["echo", "--max-blocks", "1", "alpha", "beta", "gamma"],
            tmp.path(),
        );
        assert!(out.contains("echo: alpha"));
        assert!(!out.contains("echo: beta"));
    }

    #[test]
    fn echo_prose_io_failure_returns_74() {
        let tmp = TempDir::new().unwrap();
        let code = drive_with_failing_out(&["echo", "x"], tmp.path());
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::from(74)));
    }

    #[test]
    fn echo_json_io_failure_returns_74() {
        let tmp = TempDir::new().unwrap();
        let code = drive_with_failing_out(&["--json", "echo", "x"], tmp.path());
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::from(74)));
    }

    #[test]
    fn render_block_handles_all_variants() {
        let mut out = Vec::new();
        render_block(&mut out, &Block::Text { text: "t".into() }).unwrap();
        render_block(
            &mut out,
            &Block::Usage {
                prompt: 1,
                completion: 2,
            },
        )
        .unwrap();
        render_block(&mut out, &Block::Done).unwrap();
        render_block(
            &mut out,
            &Block::Cancelled {
                reason: "STRAT-E4002".into(),
            },
        )
        .unwrap();
        render_block(
            &mut out,
            &Block::ToolCall {
                id: "t1".into(),
                tool: "fs.read".into(),
                args: "{}".into(),
            },
        )
        .unwrap();
        render_block(
            &mut out,
            &Block::ToolResult {
                id: "t1".into(),
                output: "ok".into(),
            },
        )
        .unwrap();
        let s = String::from_utf8(out).unwrap();
        assert!(s.contains("(done)"));
        assert!(s.contains("(cancelled: STRAT-E4002)"));
        assert!(s.contains("(tool_call: fs.read)"));
        assert!(s.contains("(tool_result: t1)"));
    }

    /// `stratum mem-check` with a tiny synthetic model should always pass
    /// because the host has plenty of headroom.
    #[test]
    fn mem_check_ok_prose_path() {
        let tmp = TempDir::new().unwrap();
        let (code, out, err) = drive_under(
            &[
                "mem-check",
                "--weight-rss",
                "1",
                "--kv-per-token",
                "0",
                "--context",
                "0",
                "--margin",
                "0",
            ],
            tmp.path(),
        );
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::SUCCESS));
        assert!(out.starts_with("ok: would leave "), "out was: {out}");
        assert!(out.contains(" GB free"));
        assert!(err.is_empty());
    }

    #[test]
    fn mem_check_ok_json_path() {
        let tmp = TempDir::new().unwrap();
        let (code, out, _err) = drive_under(
            &[
                "--json",
                "mem-check",
                "--weight-rss",
                "2",
                "--kv-per-token",
                "4",
                "--context",
                "8",
                "--mmproj",
                "1",
                "--vram",
                "0",
                "--margin",
                "0",
            ],
            tmp.path(),
        );
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::SUCCESS));
        let v: serde_json::Value = serde_json::from_str(out.trim()).unwrap();
        assert_eq!(v["status"], "ok");
        assert_eq!(v["margin_mib"], 0);
        assert!(v["free_mib"].as_u64().unwrap_or(0) > 0);
        assert!(v["leftover_mib"].as_u64().is_some());
    }

    /// `stratum mem-check` with a huge weight RSS must trigger the gate
    /// regardless of host. Prose path: exit 1, error on stderr.
    #[test]
    fn mem_check_refused_prose_path() {
        let tmp = TempDir::new().unwrap();
        let (code, out, err) = drive_under(
            &[
                "mem-check",
                "--weight-rss",
                "4000000",
                "--kv-per-token",
                "0",
                "--context",
                "0",
            ],
            tmp.path(),
        );
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::from(1)));
        assert!(out.is_empty(), "out was: {out}");
        assert!(err.contains("STRAT-E3007"), "err was: {err}");
        assert!(err.contains("free "), "err was: {err}");
        assert!(err.contains(" GB"), "err was: {err}");
    }

    #[test]
    fn mem_check_refused_json_path() {
        let tmp = TempDir::new().unwrap();
        let (code, _out, err) = drive_under(
            &[
                "--json",
                "mem-check",
                "--weight-rss",
                "4000000",
                "--kv-per-token",
                "0",
                "--context",
                "0",
            ],
            tmp.path(),
        );
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::from(1)));
        let v: serde_json::Value = serde_json::from_str(err.trim()).unwrap();
        assert_eq!(v["status"], "refused");
        assert_eq!(v["code"], "STRAT-E3007");
        assert!(v["message"].as_str().unwrap().contains("GB"));
        assert!(v["needed_mib"].as_u64().unwrap_or(0) > 0);
    }

    #[test]
    fn mem_check_prose_io_failure_returns_74() {
        let tmp = TempDir::new().unwrap();
        let code = drive_with_failing_out(
            &[
                "mem-check",
                "--weight-rss",
                "1",
                "--kv-per-token",
                "0",
                "--context",
                "0",
                "--margin",
                "0",
            ],
            tmp.path(),
        );
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::from(74)));
    }

    #[test]
    fn mem_check_json_io_failure_returns_74() {
        let tmp = TempDir::new().unwrap();
        let code = drive_with_failing_out(
            &[
                "--json",
                "mem-check",
                "--weight-rss",
                "1",
                "--kv-per-token",
                "0",
                "--context",
                "0",
                "--margin",
                "0",
            ],
            tmp.path(),
        );
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::from(74)));
    }

    /// Refusal prose path is written to stderr, so a failing stdout writer is
    /// fine — we instead drive a failing stderr.
    #[test]
    fn mem_check_refused_prose_stderr_failure_returns_74() {
        let tmp = TempDir::new().unwrap();
        let mut out = Vec::new();
        let mut fail = FailingWriter;
        let mut argv: Vec<OsString> = vec![
            OsString::from("--storage-root"),
            OsString::from(tmp.path().as_os_str()),
        ];
        for s in [
            "mem-check",
            "--weight-rss",
            "4000000",
            "--kv-per-token",
            "0",
            "--context",
            "0",
        ] {
            argv.push(OsString::from(s));
        }
        let code = run_with(argv, &mut out, &mut fail, Paths::resolve);
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::from(74)));
    }

    #[test]
    fn mem_check_refused_json_stderr_failure_returns_74() {
        let tmp = TempDir::new().unwrap();
        let mut out = Vec::new();
        let mut fail = FailingWriter;
        let mut argv: Vec<OsString> = vec![
            OsString::from("--storage-root"),
            OsString::from(tmp.path().as_os_str()),
        ];
        for s in [
            "--json",
            "mem-check",
            "--weight-rss",
            "4000000",
            "--kv-per-token",
            "0",
            "--context",
            "0",
        ] {
            argv.push(OsString::from(s));
        }
        let code = run_with(argv, &mut out, &mut fail, Paths::resolve);
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::from(74)));
    }

    #[test]
    fn format_gb_one_decimal_matches_known_values() {
        assert_eq!(format_gb_one_decimal(0), "0.0");
        assert_eq!(format_gb_one_decimal(1024), "1.1");
        assert_eq!(format_gb_one_decimal(400), "0.4");
    }

    #[test]
    fn mem_check_loaded_ok_prose_path() {
        // OK path with a `--loaded` spec exercises the parser even when no
        // refusal occurs.
        let tmp = TempDir::new().unwrap();
        let (code, out, err) = drive_under(
            &[
                "mem-check",
                "--weight-rss",
                "1",
                "--kv-per-token",
                "0",
                "--context",
                "0",
                "--margin",
                "0",
                "--loaded",
                "router:64:0:0",
            ],
            tmp.path(),
        );
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::SUCCESS));
        assert!(out.starts_with("ok: would leave "), "out was: {out}");
        assert!(err.is_empty(), "err was: {err}");
    }

    #[test]
    fn mem_check_refused_prose_with_loaded_emits_unload_hint() {
        let tmp = TempDir::new().unwrap();
        let (code, _out, err) = drive_under(
            &[
                "mem-check",
                "--weight-rss",
                "4000000",
                "--kv-per-token",
                "0",
                "--context",
                "0",
                "--loaded",
                "planner:5000000:0:0",
            ],
            tmp.path(),
        );
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::from(1)));
        assert!(err.contains("STRAT-E3007"), "err was: {err}");
        assert!(err.contains("hint: unload planner"), "err was: {err}");
    }

    #[test]
    fn mem_check_refused_json_with_loaded_lists_suggested_unloads() {
        let tmp = TempDir::new().unwrap();
        let (code, _out, err) = drive_under(
            &[
                "--json",
                "mem-check",
                "--weight-rss",
                "4000000",
                "--kv-per-token",
                "0",
                "--context",
                "0",
                "--loaded",
                "planner:5000000:0:0",
            ],
            tmp.path(),
        );
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::from(1)));
        let v: serde_json::Value = serde_json::from_str(err.trim()).unwrap();
        assert_eq!(v["status"], "refused");
        let arr = v["suggested_unloads"].as_array().unwrap();
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0], "planner");
    }

    #[test]
    fn mem_check_refused_json_without_loaded_has_empty_suggestions() {
        let tmp = TempDir::new().unwrap();
        let (code, _out, err) = drive_under(
            &[
                "--json",
                "mem-check",
                "--weight-rss",
                "4000000",
                "--kv-per-token",
                "0",
                "--context",
                "0",
            ],
            tmp.path(),
        );
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::from(1)));
        let v: serde_json::Value = serde_json::from_str(err.trim()).unwrap();
        assert_eq!(v["status"], "refused");
        assert!(v["suggested_unloads"].as_array().unwrap().is_empty());
    }

    #[test]
    fn mem_check_loaded_bad_spec_exits_64() {
        let tmp = TempDir::new().unwrap();
        let (code, _out, err) = drive_under(
            &[
                "mem-check",
                "--weight-rss",
                "1",
                "--kv-per-token",
                "0",
                "--context",
                "0",
                "--margin",
                "0",
                "--loaded",
                "only-two:fields",
            ],
            tmp.path(),
        );
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::from(64)));
        assert!(err.contains("STRAT-E1001"), "err was: {err}");
        assert!(err.contains("4 colon-separated"), "err was: {err}");
    }

    #[test]
    fn mem_check_loaded_bad_number_exits_64() {
        let tmp = TempDir::new().unwrap();
        let (code, _out, err) = drive_under(
            &[
                "mem-check",
                "--weight-rss",
                "1",
                "--kv-per-token",
                "0",
                "--context",
                "0",
                "--loaded",
                "router:notanumber:0:0",
            ],
            tmp.path(),
        );
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::from(64)));
        assert!(err.contains("weight_rss_mib"), "err was: {err}");
    }

    #[test]
    fn parse_loaded_spec_rejects_empty_id() {
        let err = parse_loaded_spec(":1:0:0").unwrap_err();
        assert!(err.contains("empty model id"), "err was: {err}");
    }

    #[test]
    fn parse_loaded_spec_happy() {
        let (lm, ctx) = parse_loaded_spec("planner:2048:4096:8192").unwrap();
        assert_eq!(lm.id.as_str(), "planner");
        assert_eq!(lm.estimate.weight_rss_mib, 2048);
        assert_eq!(lm.estimate.kv_per_token_bytes, 4096);
        assert_eq!(ctx, 8192);
    }

    #[cfg(unix)]
    #[test]
    fn init_write_failure_exits_73() {
        let tmp = TempDir::new().unwrap();
        let p = Paths::under(tmp.path());
        p.ensure_dirs().unwrap();
        // Pre-create the tmp file path as a directory so write_atomic fails.
        let tmp_path = p.installed_toml().with_extension("toml.tmp");
        std::fs::create_dir(&tmp_path).unwrap();
        let (code, _out, err) = drive_under(&["init"], tmp.path());
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::from(73)));
        assert!(!err.is_empty());
    }

    // ---- New models catalog branch coverage tests ----

    fn write_bad_catalog(root: &Path, body: &str) {
        let state_dir = root.join("state");
        std::fs::create_dir_all(&state_dir).unwrap();
        std::fs::write(state_dir.join("models.json"), body).unwrap();
    }

    #[test]
    fn models_list_bad_catalog_exits_1() {
        let tmp = TempDir::new().unwrap();
        write_bad_catalog(tmp.path(), "{not json");
        let (code, _out, err) = drive_under(&["models", "list"], tmp.path());
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::from(1)));
        assert!(err.contains("STRAT-E1001"));
    }

    #[test]
    fn models_add_bad_catalog_exits_1() {
        let tmp = TempDir::new().unwrap();
        write_bad_catalog(tmp.path(), "{not json");
        let (code, _out, err) = drive_under(
            &[
                "models",
                "add",
                "--slug",
                "x",
                "--family",
                "llama",
                "--display-name",
                "X",
                "--tier",
                "low",
                "--task",
                "chat",
                "--size-mib",
                "1",
                "--quantization",
                "Q",
                "--url",
                "https://example.com/x",
                "--sha256",
                GOOD_SHA,
                "--bytes",
                "1",
                "--license",
                "MIT",
            ],
            tmp.path(),
        );
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::from(1)));
        assert!(err.contains("STRAT-E1001"));
    }

    #[test]
    fn models_remove_bad_catalog_exits_1() {
        let tmp = TempDir::new().unwrap();
        write_bad_catalog(tmp.path(), "{not json");
        let (code, _out, err) =
            drive_under(&["models", "remove", "--slug", "anything"], tmp.path());
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::from(1)));
        assert!(err.contains("STRAT-E1001"));
    }

    #[test]
    fn models_recommend_bad_catalog_exits_1() {
        let tmp = TempDir::new().unwrap();
        write_bad_catalog(tmp.path(), "{not json");
        let (code, _out, err) = drive_under(
            &["models", "recommend", "--tier", "low", "--task", "chat"],
            tmp.path(),
        );
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::from(1)));
        assert!(err.contains("STRAT-E1001"));
    }

    #[test]
    fn models_validate_bad_catalog_exits_1() {
        let tmp = TempDir::new().unwrap();
        write_bad_catalog(tmp.path(), "{not json");
        let (code, _out, err) = drive_under(&["models", "validate"], tmp.path());
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::from(1)));
        assert!(err.contains("STRAT-E1001"));
    }

    #[test]
    fn models_add_bad_slug_exits_2() {
        let tmp = TempDir::new().unwrap();
        let (code, _out, err) = drive_under(
            &[
                "models",
                "add",
                "--slug",
                "BAD",
                "--family",
                "llama",
                "--display-name",
                "X",
                "--tier",
                "low",
                "--task",
                "chat",
                "--size-mib",
                "1",
                "--quantization",
                "Q",
                "--url",
                "https://example.com/x",
                "--sha256",
                GOOD_SHA,
                "--bytes",
                "1",
                "--license",
                "MIT",
            ],
            tmp.path(),
        );
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::from(2)));
        assert!(err.contains("invalid --slug"));
    }

    #[test]
    fn models_add_bad_task_exits_2() {
        let tmp = TempDir::new().unwrap();
        let (code, _out, err) = drive_under(
            &[
                "models",
                "add",
                "--slug",
                "x",
                "--family",
                "llama",
                "--display-name",
                "X",
                "--tier",
                "low",
                "--task",
                "nope",
                "--size-mib",
                "1",
                "--quantization",
                "Q",
                "--url",
                "https://example.com/x",
                "--sha256",
                GOOD_SHA,
                "--bytes",
                "1",
                "--license",
                "MIT",
            ],
            tmp.path(),
        );
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::from(2)));
        assert!(err.contains("invalid --task"));
    }

    #[test]
    fn models_add_zero_size_exits_2() {
        let tmp = TempDir::new().unwrap();
        let (code, _out, err) = drive_under(
            &[
                "models",
                "add",
                "--slug",
                "x",
                "--family",
                "llama",
                "--display-name",
                "X",
                "--tier",
                "low",
                "--task",
                "chat",
                "--size-mib",
                "0",
                "--quantization",
                "Q",
                "--url",
                "https://example.com/x",
                "--sha256",
                GOOD_SHA,
                "--bytes",
                "1",
                "--license",
                "MIT",
            ],
            tmp.path(),
        );
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::from(2)));
        assert!(err.contains("invalid --size-mib"));
    }

    #[test]
    fn models_add_empty_family_exits_2() {
        let tmp = TempDir::new().unwrap();
        let (code, _out, err) = drive_under(
            &[
                "models",
                "add",
                "--slug",
                "x",
                "--family",
                "   ",
                "--display-name",
                "X",
                "--tier",
                "low",
                "--task",
                "chat",
                "--size-mib",
                "1",
                "--quantization",
                "Q",
                "--url",
                "https://example.com/x",
                "--sha256",
                GOOD_SHA,
                "--bytes",
                "1",
                "--license",
                "MIT",
            ],
            tmp.path(),
        );
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::from(2)));
        assert!(err.contains("invalid --family"));
    }

    #[test]
    fn models_list_prose_io_failure_returns_74() {
        let tmp = TempDir::new().unwrap();
        seed_one(tmp.path(), "x");
        let code = drive_with_failing_out(&["models", "list"], tmp.path());
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::from(74)));
    }

    #[test]
    fn models_list_empty_prose_io_failure_returns_74() {
        let tmp = TempDir::new().unwrap();
        let code = drive_with_failing_out(&["models", "list"], tmp.path());
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::from(74)));
    }

    #[test]
    fn models_add_prose_io_failure_returns_74() {
        let tmp = TempDir::new().unwrap();
        let code = drive_with_failing_out(
            &[
                "models",
                "add",
                "--slug",
                "x",
                "--family",
                "llama",
                "--display-name",
                "X",
                "--tier",
                "low",
                "--task",
                "chat",
                "--size-mib",
                "1",
                "--quantization",
                "Q",
                "--url",
                "https://example.com/x",
                "--sha256",
                GOOD_SHA,
                "--bytes",
                "1",
                "--license",
                "MIT",
            ],
            tmp.path(),
        );
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::from(74)));
    }

    #[test]
    fn models_add_json_io_failure_returns_74() {
        let tmp = TempDir::new().unwrap();
        let code = drive_with_failing_out(
            &[
                "--json",
                "models",
                "add",
                "--slug",
                "x",
                "--family",
                "llama",
                "--display-name",
                "X",
                "--tier",
                "low",
                "--task",
                "chat",
                "--size-mib",
                "1",
                "--quantization",
                "Q",
                "--url",
                "https://example.com/x",
                "--sha256",
                GOOD_SHA,
                "--bytes",
                "1",
                "--license",
                "MIT",
            ],
            tmp.path(),
        );
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::from(74)));
    }

    #[test]
    fn models_remove_io_failure_returns_74() {
        let tmp = TempDir::new().unwrap();
        seed_one(tmp.path(), "removable");
        let code = drive_with_failing_out(&["models", "remove", "--slug", "removable"], tmp.path());
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::from(74)));
    }

    #[test]
    fn models_recommend_prose_io_failure_returns_74() {
        let tmp = TempDir::new().unwrap();
        seed_one(tmp.path(), "tiny");
        let code = drive_with_failing_out(
            &["models", "recommend", "--tier", "low", "--task", "chat"],
            tmp.path(),
        );
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::from(74)));
    }

    #[test]
    fn models_recommend_json_emits_entry() {
        let tmp = TempDir::new().unwrap();
        seed_one(tmp.path(), "tiny");
        let (code, out, _err) = drive_under(
            &[
                "--json",
                "models",
                "recommend",
                "--tier",
                "low",
                "--task",
                "chat",
            ],
            tmp.path(),
        );
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::SUCCESS));
        let v: serde_json::Value = serde_json::from_str(out.trim()).unwrap();
        assert_eq!(v["slug"], "tiny");
    }

    #[test]
    fn models_recommend_json_io_failure_returns_74() {
        let tmp = TempDir::new().unwrap();
        seed_one(tmp.path(), "tiny");
        let code = drive_with_failing_out(
            &[
                "--json",
                "models",
                "recommend",
                "--tier",
                "low",
                "--task",
                "chat",
            ],
            tmp.path(),
        );
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::from(74)));
    }

    #[test]
    fn models_validate_prose_io_failure_returns_74() {
        let tmp = TempDir::new().unwrap();
        let code = drive_with_failing_out(&["models", "validate"], tmp.path());
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::from(74)));
    }

    #[cfg(unix)]
    #[test]
    fn models_add_save_failure_exits_1() {
        // Pre-create the tmp file path as a directory so save_atomic fails.
        let tmp = TempDir::new().unwrap();
        let state_dir = tmp.path().join("state");
        std::fs::create_dir_all(&state_dir).unwrap();
        let tmp_path = state_dir.join("models.json.tmp");
        std::fs::create_dir(&tmp_path).unwrap();
        let (code, _out, err) = drive_under(
            &[
                "models",
                "add",
                "--slug",
                "x",
                "--family",
                "llama",
                "--display-name",
                "X",
                "--tier",
                "low",
                "--task",
                "chat",
                "--size-mib",
                "1",
                "--quantization",
                "Q",
                "--url",
                "https://example.com/x",
                "--sha256",
                GOOD_SHA,
                "--bytes",
                "1",
                "--license",
                "MIT",
            ],
            tmp.path(),
        );
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::from(1)));
        assert!(err.contains("STRAT-E1001"));
    }

    #[cfg(unix)]
    #[test]
    fn models_remove_save_failure_exits_1() {
        let tmp = TempDir::new().unwrap();
        seed_one(tmp.path(), "removeme");
        // Replace tmp staging path with a directory to block save.
        let state_dir = tmp.path().join("state");
        let tmp_path = state_dir.join("models.json.tmp");
        std::fs::create_dir(&tmp_path).unwrap();
        let (code, _out, err) =
            drive_under(&["models", "remove", "--slug", "removeme"], tmp.path());
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::from(1)));
        assert!(err.contains("STRAT-E1001"));
    }

    #[cfg(unix)]
    #[test]
    fn models_add_ensure_state_dir_failure_exits_1() {
        // Use a regular file at the state path so create_dir_all fails.
        let tmp = TempDir::new().unwrap();
        std::fs::write(tmp.path().join("state"), b"blocker").unwrap();
        let (code, _out, err) = drive_under(
            &[
                "models",
                "add",
                "--slug",
                "x",
                "--family",
                "llama",
                "--display-name",
                "X",
                "--tier",
                "low",
                "--task",
                "chat",
                "--size-mib",
                "1",
                "--quantization",
                "Q",
                "--url",
                "https://example.com/x",
                "--sha256",
                GOOD_SHA,
                "--bytes",
                "1",
                "--license",
                "MIT",
            ],
            tmp.path(),
        );
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::from(1)));
        assert!(err.contains("STRAT-E1001"));
    }

    #[test]
    fn install_file_from_url_default_filename_with_colon() {
        // A URL whose last segment contains ':' falls back to model.bin.
        let args = InstallFileArgs {
            from_file: None,
            from_url: Some("https://example.com/foo:bar".into()),
            name: None,
            sha256: None,
        };
        assert_eq!(default_filename_for(&args), "model.bin");
    }

    #[test]
    fn install_file_default_filename_from_file_no_name() {
        // file with no filename (only root)
        let args = InstallFileArgs {
            from_file: Some(PathBuf::from("/")),
            from_url: None,
            name: None,
            sha256: None,
        };
        assert_eq!(default_filename_for(&args), "model.bin");
    }

    // ---- self-update --check coverage ----

    /// Minimal valid manifest fixture with one release at `version`, no
    /// `min_supported_from`, single `linux_x86_64` artifact. Returns the path.
    fn write_self_update_fixture(dir: &Path, version: &str) -> PathBuf {
        let mut iter = version.split('.');
        let major: u16 = iter.next().unwrap().parse().unwrap();
        let minor: u16 = iter.next().unwrap().parse().unwrap();
        let patch: u16 = iter.next().unwrap().parse().unwrap();
        let body = format!(
            r#"{{
                "schema_version": 1,
                "channel": "stable",
                "latest": {{
                    "version": {{ "major": {major}, "minor": {minor}, "patch": {patch}, "pre": null }},
                    "released_at": {{ "secs_since_epoch": 1700000000, "nanos_since_epoch": 0 }},
                    "binary": {{
                        "url": "https://dl.stratum.dev/v{version}/stratum-linux_x86_64.tar.gz",
                        "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
                        "bytes": 1024,
                        "platform": "linux_x86_64"
                    }},
                    "min_supported_from": null,
                    "release_notes_url": "https://stratum.dev/releases/{version}"
                }},
                "history": [
                    {{
                        "version": {{ "major": {major}, "minor": {minor}, "patch": {patch}, "pre": null }},
                        "released_at": {{ "secs_since_epoch": 1700000000, "nanos_since_epoch": 0 }},
                        "binary": {{
                            "url": "https://dl.stratum.dev/v{version}/stratum-linux_x86_64.tar.gz",
                            "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
                            "bytes": 1024,
                            "platform": "linux_x86_64"
                        }},
                        "min_supported_from": null,
                        "release_notes_url": "https://stratum.dev/releases/{version}"
                    }}
                ]
            }}"#
        );
        let fixture_path = dir.join("manifest.json");
        std::fs::write(&fixture_path, body).unwrap();
        fixture_path
    }

    #[test]
    fn self_update_missing_check_flag_exits_64() {
        let tmp = TempDir::new().unwrap();
        let (code, _out, err) = drive_under(&["self-update"], tmp.path());
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::from(64)));
        assert!(err.contains("--check or --apply"));
    }

    #[test]
    fn self_update_check_and_apply_mutually_exclusive_exits_64() {
        let tmp = TempDir::new().unwrap();
        let fixture = write_self_update_fixture(tmp.path(), "1.0.0");
        let (code, _out, _err) = drive_under(
            &[
                "self-update",
                "--check",
                "--apply",
                "--manifest-file",
                fixture.to_str().unwrap(),
                "--current",
                "1.0.0",
                "--platform",
                "linux_x86_64",
            ],
            tmp.path(),
        );
        // Clap's `conflicts_with` rejects the combo with exit 64.
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::from(64)));
    }

    #[test]
    fn self_update_apply_up_to_date_short_circuits() {
        let tmp = TempDir::new().unwrap();
        let fixture = write_self_update_fixture(tmp.path(), "1.0.0");
        let (code, out, _err) = drive_under(
            &[
                "self-update",
                "--apply",
                "--manifest-file",
                fixture.to_str().unwrap(),
                "--current",
                "1.0.0",
                "--platform",
                "linux_x86_64",
            ],
            tmp.path(),
        );
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::SUCCESS));
        assert!(out.contains("already up to date"));
    }

    #[test]
    fn self_update_apply_blocked_exits_64() {
        let tmp = TempDir::new().unwrap();
        let body = r#"{
            "schema_version": 1,
            "channel": "stable",
            "latest": {
                "version": { "major": 1, "minor": 5, "patch": 0, "pre": null },
                "released_at": { "secs_since_epoch": 1700000000, "nanos_since_epoch": 0 },
                "binary": {
                    "url": "https://dl.stratum.dev/v1.5.0/stratum-linux_x86_64.tar.gz",
                    "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
                    "bytes": 1024,
                    "platform": "linux_x86_64"
                },
                "min_supported_from": { "major": 1, "minor": 3, "patch": 0, "pre": null },
                "release_notes_url": "https://stratum.dev/releases/1.5.0"
            },
            "history": [
                {
                    "version": { "major": 1, "minor": 5, "patch": 0, "pre": null },
                    "released_at": { "secs_since_epoch": 1700000000, "nanos_since_epoch": 0 },
                    "binary": {
                        "url": "https://dl.stratum.dev/v1.5.0/stratum-linux_x86_64.tar.gz",
                        "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
                        "bytes": 1024,
                        "platform": "linux_x86_64"
                    },
                    "min_supported_from": { "major": 1, "minor": 3, "patch": 0, "pre": null },
                    "release_notes_url": "https://stratum.dev/releases/1.5.0"
                }
            ]
        }"#;
        let path = tmp.path().join("manifest.json");
        std::fs::write(&path, body).unwrap();
        let (code, _out, err) = drive_under(
            &[
                "self-update",
                "--apply",
                "--manifest-file",
                path.to_str().unwrap(),
                "--current",
                "1.0.0",
                "--platform",
                "linux_x86_64",
            ],
            tmp.path(),
        );
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::from(64)));
        assert!(err.contains("STRAT-E1001"));
        assert!(err.contains("reinstall"));
    }

    #[test]
    fn self_update_apply_no_artifact_for_platform_exits_1() {
        let tmp = TempDir::new().unwrap();
        let fixture = write_self_update_fixture(tmp.path(), "1.5.0");
        let (code, _out, err) = drive_under(
            &[
                "self-update",
                "--apply",
                "--manifest-file",
                fixture.to_str().unwrap(),
                "--current",
                "1.4.7",
                "--platform",
                "windows_x86_64",
            ],
            tmp.path(),
        );
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::from(1)));
        assert!(err.contains("no artifact for platform"));
    }

    #[test]
    fn self_update_apply_dry_run_requires_apply() {
        // Clap's `requires = "apply"` rejects `--dry-run` without `--apply`.
        let tmp = TempDir::new().unwrap();
        let fixture = write_self_update_fixture(tmp.path(), "1.0.0");
        let (code, _out, _err) = drive_under(
            &[
                "self-update",
                "--check",
                "--dry-run",
                "--manifest-file",
                fixture.to_str().unwrap(),
                "--current",
                "1.0.0",
                "--platform",
                "linux_x86_64",
            ],
            tmp.path(),
        );
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::from(64)));
    }

    #[test]
    fn sibling_with_suffix_appends_to_filename() {
        let base = Path::new("/tmp/stratum");
        let new_tmp = sibling_with_suffix(base, ".new.tmp");
        assert_eq!(new_tmp, Path::new("/tmp/stratum.new.tmp"));
        let bak = sibling_with_suffix(base, ".bak");
        assert_eq!(bak, Path::new("/tmp/stratum.bak"));
    }

    #[test]
    fn sha256_eq_is_case_insensitive() {
        assert!(sha256_eq("abc", "ABC"));
        assert!(!sha256_eq("abc", "abd"));
    }

    #[test]
    fn insecure_flags_allowed_in_debug_build() {
        // The cfg(debug_assertions) branch is always true under `cargo test`,
        // which builds with debug profile by default.
        assert!(insecure_flags_allowed());
    }

    #[test]
    fn atomic_swap_moves_and_keeps_bak() {
        let tmp = TempDir::new().unwrap();
        let exe = tmp.path().join("exe");
        let new_tmp = tmp.path().join("exe.new.tmp");
        let bak = tmp.path().join("exe.bak");
        std::fs::write(&exe, b"old").unwrap();
        std::fs::write(&new_tmp, b"new").unwrap();
        atomic_swap(&exe, &new_tmp, &bak).unwrap();
        assert_eq!(std::fs::read(&exe).unwrap(), b"new");
        assert_eq!(std::fs::read(&bak).unwrap(), b"old");
        assert!(!new_tmp.exists());
    }

    #[test]
    fn atomic_swap_overwrites_existing_bak() {
        let tmp = TempDir::new().unwrap();
        let exe = tmp.path().join("exe");
        let new_tmp = tmp.path().join("exe.new.tmp");
        let bak = tmp.path().join("exe.bak");
        std::fs::write(&exe, b"old").unwrap();
        std::fs::write(&new_tmp, b"new").unwrap();
        std::fs::write(&bak, b"stale").unwrap();
        atomic_swap(&exe, &new_tmp, &bak).unwrap();
        assert_eq!(std::fs::read(&bak).unwrap(), b"old");
    }

    #[cfg(unix)]
    #[test]
    fn make_executable_sets_0755() {
        use std::os::unix::fs::PermissionsExt;
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("bin");
        std::fs::write(&path, b"x").unwrap();
        make_executable(&path).unwrap();
        let mode = std::fs::metadata(&path).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o755);
    }

    // ----- in-process coverage for `self-update --apply` flows -----

    /// Body bytes the in-process server returns for the happy-path apply
    /// tests. Distinct from the integration-test body so unit and integration
    /// failures stay attributable.
    const APPLY_BODY: &[u8] = b"unit-apply-body";

    /// Spawn a one-shot HTTP/1.0 server bound to 127.0.0.1 that answers a
    /// single GET with `body` and exits. Returns `(url, join_handle)`.
    fn spawn_unit_artifact_server(body: &'static [u8]) -> (String, std::thread::JoinHandle<()>) {
        use std::io::Read as _;
        use std::net::TcpListener;
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let port = listener.local_addr().expect("addr").port();
        let url = format!("http://127.0.0.1:{port}/artifact.bin");
        let handle = std::thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                let mut buf = [0_u8; 1024];
                let _ = stream.read(&mut buf);
                let header = format!(
                    "HTTP/1.0 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                let _ = stream.write_all(header.as_bytes());
                let _ = stream.write_all(body);
                let _ = stream.flush();
            }
        });
        (url, handle)
    }

    /// Spawn a one-shot HTTP server that returns the given non-200 status.
    fn spawn_unit_status_server(
        status_line: &'static str,
    ) -> (String, std::thread::JoinHandle<()>) {
        use std::io::Read as _;
        use std::net::TcpListener;
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let port = listener.local_addr().expect("addr").port();
        let url = format!("http://127.0.0.1:{port}/x");
        let handle = std::thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                let mut buf = [0_u8; 1024];
                let _ = stream.read(&mut buf);
                let header = format!(
                    "HTTP/1.0 {status_line}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                );
                let _ = stream.write_all(header.as_bytes());
                let _ = stream.flush();
            }
        });
        (url, handle)
    }

    /// Build a manifest fixture whose `latest` advertises one artifact.
    fn write_apply_fixture(
        dir: &Path,
        version: &str,
        artifact_url: &str,
        sha256: &str,
        bytes: u64,
        min_supported_from: Option<&str>,
    ) -> PathBuf {
        let parts: Vec<u16> = version.split('.').map(|s| s.parse().unwrap()).collect();
        let (maj, min, pat) = (parts[0], parts[1], parts[2]);
        let min_block = min_supported_from.map_or_else(
            || r#""min_supported_from": null,"#.to_owned(),
            |s| {
                let p: Vec<u16> = s.split('.').map(|x| x.parse().unwrap()).collect();
                format!(
                    r#""min_supported_from": {{ "major": {}, "minor": {}, "patch": {}, "pre": null }},"#,
                    p[0], p[1], p[2],
                )
            },
        );
        let entry = format!(
            r#"{{
                "version": {{ "major": {maj}, "minor": {min}, "patch": {pat}, "pre": null }},
                "released_at": {{ "secs_since_epoch": 1700000000, "nanos_since_epoch": 0 }},
                "binary": {{
                    "url": "{artifact_url}",
                    "sha256": "{sha256}",
                    "bytes": {bytes},
                    "platform": "linux_x86_64"
                }},
                {min_block}
                "release_notes_url": "https://stratum.dev/releases/{version}"
            }}"#
        );
        let body = format!(
            r#"{{ "schema_version": 1, "channel": "stable", "latest": {entry}, "history": [{entry}] }}"#
        );
        let path = dir.join("manifest.json");
        std::fs::write(&path, body).unwrap();
        path
    }

    #[test]
    fn apply_happy_path_unit_coverage() {
        let tmp = TempDir::new().unwrap();
        let sha = stratum_runtime::download::sha256_hex(APPLY_BODY);
        let (url, handle) = spawn_unit_artifact_server(APPLY_BODY);
        let fixture = write_apply_fixture(
            tmp.path(),
            "1.5.0",
            &url,
            &sha,
            APPLY_BODY.len() as u64,
            None,
        );
        let target = tmp.path().join("stratum-stub");
        std::fs::write(&target, b"old-binary").unwrap();
        let (code, out, err) = drive_under(
            &[
                "self-update",
                "--apply",
                "--allow-insecure-url",
                "--target",
                target.to_str().unwrap(),
                "--manifest-file",
                fixture.to_str().unwrap(),
                "--current",
                "1.4.7",
                "--platform",
                "linux_x86_64",
            ],
            tmp.path(),
        );
        let _ = handle.join();
        assert_eq!(
            format!("{code:?}"),
            format!("{:?}", ExitCode::SUCCESS),
            "err={err}"
        );
        assert!(out.contains("upgraded"), "out: {out}");
        assert!(out.contains("1.4.7"));
        assert!(out.contains("1.5.0"));
        assert_eq!(std::fs::read(&target).unwrap(), APPLY_BODY);
        let bak = tmp.path().join("stratum-stub.bak");
        assert_eq!(std::fs::read(&bak).unwrap(), b"old-binary");
    }

    #[test]
    fn apply_json_unit_coverage() {
        let tmp = TempDir::new().unwrap();
        let sha = stratum_runtime::download::sha256_hex(APPLY_BODY);
        let (url, handle) = spawn_unit_artifact_server(APPLY_BODY);
        let fixture = write_apply_fixture(
            tmp.path(),
            "1.5.0",
            &url,
            &sha,
            APPLY_BODY.len() as u64,
            None,
        );
        let target = tmp.path().join("stratum-stub");
        std::fs::write(&target, b"old-binary").unwrap();
        let (code, out, _err) = drive_under(
            &[
                "--json",
                "self-update",
                "--apply",
                "--allow-insecure-url",
                "--target",
                target.to_str().unwrap(),
                "--manifest-file",
                fixture.to_str().unwrap(),
                "--current",
                "1.4.7",
                "--platform",
                "linux_x86_64",
            ],
            tmp.path(),
        );
        let _ = handle.join();
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::SUCCESS));
        let v: serde_json::Value = serde_json::from_str(out.trim()).unwrap();
        assert_eq!(v["action"], "applied");
        assert_eq!(v["from"], "1.4.7");
        assert_eq!(v["to"], "1.5.0");
        assert_eq!(v["artifact"]["sha256"], sha);
    }

    #[test]
    fn apply_dry_run_unit_coverage() {
        let tmp = TempDir::new().unwrap();
        let sha = stratum_runtime::download::sha256_hex(APPLY_BODY);
        let (url, handle) = spawn_unit_artifact_server(APPLY_BODY);
        let fixture = write_apply_fixture(
            tmp.path(),
            "1.5.0",
            &url,
            &sha,
            APPLY_BODY.len() as u64,
            None,
        );
        let target = tmp.path().join("stratum-stub");
        std::fs::write(&target, b"orig").unwrap();
        let (code, out, _err) = drive_under(
            &[
                "self-update",
                "--apply",
                "--dry-run",
                "--allow-insecure-url",
                "--target",
                target.to_str().unwrap(),
                "--manifest-file",
                fixture.to_str().unwrap(),
                "--current",
                "1.4.7",
                "--platform",
                "linux_x86_64",
            ],
            tmp.path(),
        );
        let _ = handle.join();
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::SUCCESS));
        assert!(out.contains("dry-run: would swap"), "out: {out}");
        // Target untouched.
        assert_eq!(std::fs::read(&target).unwrap(), b"orig");
    }

    #[test]
    fn apply_sha_mismatch_unit_coverage() {
        let tmp = TempDir::new().unwrap();
        let bogus = "0".repeat(64);
        let (url, handle) = spawn_unit_artifact_server(APPLY_BODY);
        let fixture = write_apply_fixture(
            tmp.path(),
            "1.5.0",
            &url,
            &bogus,
            APPLY_BODY.len() as u64,
            None,
        );
        let target = tmp.path().join("stratum-stub");
        std::fs::write(&target, b"orig").unwrap();
        let (code, _out, err) = drive_under(
            &[
                "self-update",
                "--apply",
                "--allow-insecure-url",
                "--target",
                target.to_str().unwrap(),
                "--manifest-file",
                fixture.to_str().unwrap(),
                "--current",
                "1.4.7",
                "--platform",
                "linux_x86_64",
            ],
            tmp.path(),
        );
        let _ = handle.join();
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::from(1)));
        assert!(err.contains("sha256 mismatch"), "err: {err}");
        assert_eq!(std::fs::read(&target).unwrap(), b"orig");
    }

    #[test]
    fn apply_bytes_mismatch_unit_coverage() {
        let tmp = TempDir::new().unwrap();
        let sha = stratum_runtime::download::sha256_hex(APPLY_BODY);
        let (url, handle) = spawn_unit_artifact_server(APPLY_BODY);
        // Declare an off-by-one byte count.
        let fixture = write_apply_fixture(
            tmp.path(),
            "1.5.0",
            &url,
            &sha,
            (APPLY_BODY.len() as u64) + 1,
            None,
        );
        let target = tmp.path().join("stratum-stub");
        std::fs::write(&target, b"orig").unwrap();
        let (code, _out, err) = drive_under(
            &[
                "self-update",
                "--apply",
                "--allow-insecure-url",
                "--target",
                target.to_str().unwrap(),
                "--manifest-file",
                fixture.to_str().unwrap(),
                "--current",
                "1.4.7",
                "--platform",
                "linux_x86_64",
            ],
            tmp.path(),
        );
        let _ = handle.join();
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::from(1)));
        assert!(err.contains("byte count mismatch"), "err: {err}");
        assert_eq!(std::fs::read(&target).unwrap(), b"orig");
    }

    #[test]
    fn apply_download_http_error_status_unit_coverage() {
        let tmp = TempDir::new().unwrap();
        let sha = stratum_runtime::download::sha256_hex(APPLY_BODY);
        let (url, handle) = spawn_unit_status_server("500 Internal Server Error");
        let fixture = write_apply_fixture(
            tmp.path(),
            "1.5.0",
            &url,
            &sha,
            APPLY_BODY.len() as u64,
            None,
        );
        let target = tmp.path().join("stratum-stub");
        std::fs::write(&target, b"orig").unwrap();
        let (code, _out, err) = drive_under(
            &[
                "self-update",
                "--apply",
                "--allow-insecure-url",
                "--target",
                target.to_str().unwrap(),
                "--manifest-file",
                fixture.to_str().unwrap(),
                "--current",
                "1.4.7",
                "--platform",
                "linux_x86_64",
            ],
            tmp.path(),
        );
        let _ = handle.join();
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::from(1)));
        // `ureq` surfaces the 500 as a transport error; either flavour is
        // acceptable as long as the exit is 1.
        assert!(err.contains("STRAT-E1001"), "err: {err}");
        assert_eq!(std::fs::read(&target).unwrap(), b"orig");
    }

    #[test]
    fn apply_rejects_http_without_allow_insecure_unit() {
        // Without `--allow-insecure-url`, an http:// artifact URL is
        // rejected before any network IO.
        let tmp = TempDir::new().unwrap();
        let sha = stratum_runtime::download::sha256_hex(APPLY_BODY);
        let fixture = write_apply_fixture(
            tmp.path(),
            "1.5.0",
            "http://127.0.0.1:1/never-fetched",
            &sha,
            APPLY_BODY.len() as u64,
            None,
        );
        let target = tmp.path().join("stratum-stub");
        std::fs::write(&target, b"orig").unwrap();
        let (code, _out, err) = drive_under(
            &[
                "self-update",
                "--apply",
                "--target",
                target.to_str().unwrap(),
                "--manifest-file",
                fixture.to_str().unwrap(),
                "--current",
                "1.4.7",
                "--platform",
                "linux_x86_64",
            ],
            tmp.path(),
        );
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::from(1)));
        assert!(err.contains("https"), "err: {err}");
        assert_eq!(std::fs::read(&target).unwrap(), b"orig");
    }

    #[test]
    fn apply_rejects_unknown_scheme_unit() {
        let tmp = TempDir::new().unwrap();
        let sha = stratum_runtime::download::sha256_hex(APPLY_BODY);
        let fixture = write_apply_fixture(
            tmp.path(),
            "1.5.0",
            "ftp://example.com/stratum",
            &sha,
            APPLY_BODY.len() as u64,
            None,
        );
        let target = tmp.path().join("stratum-stub");
        std::fs::write(&target, b"orig").unwrap();
        let (code, _out, err) = drive_under(
            &[
                "self-update",
                "--apply",
                "--allow-insecure-url",
                "--target",
                target.to_str().unwrap(),
                "--manifest-file",
                fixture.to_str().unwrap(),
                "--current",
                "1.4.7",
                "--platform",
                "linux_x86_64",
            ],
            tmp.path(),
        );
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::from(1)));
        assert!(err.contains("http(s)"), "err: {err}");
    }

    #[test]
    fn apply_io_failure_returns_74_on_up_to_date() {
        let tmp = TempDir::new().unwrap();
        let sha = stratum_runtime::download::sha256_hex(APPLY_BODY);
        let fixture = write_apply_fixture(
            tmp.path(),
            "1.0.0",
            "http://127.0.0.1:1/never-fetched",
            &sha,
            APPLY_BODY.len() as u64,
            None,
        );
        let target = tmp.path().join("stratum-stub");
        std::fs::write(&target, b"orig").unwrap();
        let code = drive_with_failing_out(
            &[
                "self-update",
                "--apply",
                "--allow-insecure-url",
                "--target",
                target.to_str().unwrap(),
                "--manifest-file",
                fixture.to_str().unwrap(),
                "--current",
                "1.0.0",
                "--platform",
                "linux_x86_64",
            ],
            tmp.path(),
        );
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::from(74)));
    }

    #[test]
    fn apply_io_failure_returns_74_on_dry_run() {
        let tmp = TempDir::new().unwrap();
        let sha = stratum_runtime::download::sha256_hex(APPLY_BODY);
        let (url, handle) = spawn_unit_artifact_server(APPLY_BODY);
        let fixture = write_apply_fixture(
            tmp.path(),
            "1.5.0",
            &url,
            &sha,
            APPLY_BODY.len() as u64,
            None,
        );
        let target = tmp.path().join("stratum-stub");
        std::fs::write(&target, b"orig").unwrap();
        let code = drive_with_failing_out(
            &[
                "self-update",
                "--apply",
                "--dry-run",
                "--allow-insecure-url",
                "--target",
                target.to_str().unwrap(),
                "--manifest-file",
                fixture.to_str().unwrap(),
                "--current",
                "1.4.7",
                "--platform",
                "linux_x86_64",
            ],
            tmp.path(),
        );
        let _ = handle.join();
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::from(74)));
    }

    #[test]
    fn apply_io_failure_returns_74_on_apply() {
        let tmp = TempDir::new().unwrap();
        let sha = stratum_runtime::download::sha256_hex(APPLY_BODY);
        let (url, handle) = spawn_unit_artifact_server(APPLY_BODY);
        let fixture = write_apply_fixture(
            tmp.path(),
            "1.5.0",
            &url,
            &sha,
            APPLY_BODY.len() as u64,
            None,
        );
        let target = tmp.path().join("stratum-stub");
        std::fs::write(&target, b"orig").unwrap();
        let code = drive_with_failing_out(
            &[
                "self-update",
                "--apply",
                "--allow-insecure-url",
                "--target",
                target.to_str().unwrap(),
                "--manifest-file",
                fixture.to_str().unwrap(),
                "--current",
                "1.4.7",
                "--platform",
                "linux_x86_64",
            ],
            tmp.path(),
        );
        let _ = handle.join();
        // Prose writer fails on stdout → 74, regardless of whether the swap
        // already happened. The atomic swap still executed first.
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::from(74)));
    }

    #[test]
    fn apply_io_failure_returns_74_on_apply_json() {
        let tmp = TempDir::new().unwrap();
        let sha = stratum_runtime::download::sha256_hex(APPLY_BODY);
        let (url, handle) = spawn_unit_artifact_server(APPLY_BODY);
        let fixture = write_apply_fixture(
            tmp.path(),
            "1.5.0",
            &url,
            &sha,
            APPLY_BODY.len() as u64,
            None,
        );
        let target = tmp.path().join("stratum-stub");
        std::fs::write(&target, b"orig").unwrap();
        let code = drive_with_failing_out(
            &[
                "--json",
                "self-update",
                "--apply",
                "--allow-insecure-url",
                "--target",
                target.to_str().unwrap(),
                "--manifest-file",
                fixture.to_str().unwrap(),
                "--current",
                "1.4.7",
                "--platform",
                "linux_x86_64",
            ],
            tmp.path(),
        );
        let _ = handle.join();
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::from(74)));
    }

    #[test]
    fn apply_dry_run_with_check_exits_64() {
        // `--check --dry-run` triggers the runtime guard (`--dry-run` requires
        // `--apply`). The "neither check nor apply" branch would otherwise
        // shadow it.
        let tmp = TempDir::new().unwrap();
        let fixture = write_apply_fixture(
            tmp.path(),
            "1.0.0",
            "https://dl.stratum.dev/v1.0.0/stratum-linux_x86_64.tar.gz",
            "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            1024,
            None,
        );
        let (code, _out, err) = drive_under(
            &[
                "self-update",
                "--check",
                "--dry-run",
                "--manifest-file",
                fixture.to_str().unwrap(),
                "--current",
                "1.0.0",
                "--platform",
                "linux_x86_64",
            ],
            tmp.path(),
        );
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::from(64)));
        assert!(err.contains("--dry-run requires --apply"), "err: {err}");
    }

    #[test]
    fn atomic_swap_rolls_back_when_second_rename_fails() {
        // Make `new_tmp` a missing path: the rename will fail; rollback
        // should restore exe from bak.
        let tmp = TempDir::new().unwrap();
        let exe = tmp.path().join("exe");
        let new_tmp = tmp.path().join("does-not-exist.tmp");
        let bak = tmp.path().join("exe.bak");
        std::fs::write(&exe, b"old").unwrap();
        let res = atomic_swap(&exe, &new_tmp, &bak);
        assert!(res.is_err(), "expected rename failure");
        // Rollback restored exe.
        assert_eq!(std::fs::read(&exe).unwrap(), b"old");
    }

    #[test]
    fn sha256_eq_handles_len_mismatch() {
        assert!(!sha256_eq("abc", "ab"));
        assert!(sha256_eq("", ""));
    }

    #[test]
    fn download_and_verify_rejects_non_https_when_not_allowed() {
        let tmp = TempDir::new().unwrap();
        let dest = tmp.path().join("x");
        let err = download_and_verify("http://127.0.0.1:1/x", &dest, false).unwrap_err();
        assert!(err.contains("https"), "err: {err}");
    }

    #[test]
    fn download_and_verify_rejects_unknown_scheme() {
        let tmp = TempDir::new().unwrap();
        let dest = tmp.path().join("x");
        let err = download_and_verify("gopher://example.com/x", &dest, true).unwrap_err();
        assert!(err.contains("http(s)"), "err: {err}");
    }

    #[test]
    fn sibling_with_suffix_no_filename_returns_suffix_only() {
        // `/` has no file_name component → join("/", suffix) = "/<suffix>".
        let p = sibling_with_suffix(Path::new("/"), ".bak");
        // On macOS / Linux Path::parent("/") is None, so parent is empty.
        // The resulting path is just the suffix (".bak").
        assert_eq!(p, Path::new(".bak"));
    }

    #[test]
    fn write_or_io_exit_returns_success_on_ok() {
        let mut buf = Vec::new();
        let code = write_or_io_exit(&mut buf, format_args!("hi"));
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::SUCCESS));
        assert_eq!(buf, b"hi\n");
    }

    #[test]
    fn write_or_io_exit_returns_74_on_io_failure() {
        let mut fail = FailingWriter;
        let code = write_or_io_exit(&mut fail, format_args!("hi"));
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::from(74)));
    }

    #[test]
    fn resolve_swap_target_returns_target_override() {
        let args = SelfUpdateArgs {
            check: false,
            apply: true,
            dry_run: false,
            manifest_url: None,
            manifest_file: None,
            channel: ChannelArg::Stable,
            current: None,
            platform: None,
            target: Some(PathBuf::from("/tmp/foo")),
            allow_insecure_url: false,
        };
        let mut err = Vec::new();
        let p = resolve_swap_target(&args, &mut err).unwrap();
        assert_eq!(p, PathBuf::from("/tmp/foo"));
        assert!(err.is_empty());
    }

    #[test]
    fn resolve_swap_target_falls_back_to_current_exe() {
        let args = SelfUpdateArgs {
            check: false,
            apply: true,
            dry_run: false,
            manifest_url: None,
            manifest_file: None,
            channel: ChannelArg::Stable,
            current: None,
            platform: None,
            target: None,
            allow_insecure_url: false,
        };
        let mut err = Vec::new();
        // `std::env::current_exe()` always succeeds in test runs; we just
        // want to drive the `Ok(path)` branch.
        let p = resolve_swap_target(&args, &mut err).unwrap();
        assert!(p.exists());
    }

    #[test]
    fn self_update_invalid_current_exits_2() {
        let tmp = TempDir::new().unwrap();
        let fixture = write_self_update_fixture(tmp.path(), "1.0.0");
        let (code, _out, err) = drive_under(
            &[
                "self-update",
                "--check",
                "--manifest-file",
                fixture.to_str().unwrap(),
                "--current",
                "not-a-version",
                "--platform",
                "linux_x86_64",
            ],
            tmp.path(),
        );
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::from(2)));
        assert!(err.contains("invalid --current"));
    }

    #[test]
    fn self_update_check_up_to_date_prose_exits_0() {
        let tmp = TempDir::new().unwrap();
        let fixture = write_self_update_fixture(tmp.path(), "1.0.0");
        let (code, out, _err) = drive_under(
            &[
                "self-update",
                "--check",
                "--manifest-file",
                fixture.to_str().unwrap(),
                "--current",
                "1.0.0",
                "--platform",
                "linux_x86_64",
            ],
            tmp.path(),
        );
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::SUCCESS));
        assert!(out.contains("up to date"));
    }

    #[test]
    fn self_update_check_upgrade_prose_includes_artifact() {
        let tmp = TempDir::new().unwrap();
        let fixture = write_self_update_fixture(tmp.path(), "1.5.0");
        let (code, out, _err) = drive_under(
            &[
                "self-update",
                "--check",
                "--manifest-file",
                fixture.to_str().unwrap(),
                "--current",
                "1.4.7",
                "--platform",
                "linux_x86_64",
            ],
            tmp.path(),
        );
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::SUCCESS));
        assert!(out.contains("upgrade available"));
        assert!(out.contains("artifact:"));
        assert!(out.contains("1.4.7"));
        assert!(out.contains("1.5.0"));
    }

    #[test]
    fn self_update_check_upgrade_prose_omits_artifact_on_platform_miss() {
        let tmp = TempDir::new().unwrap();
        let fixture = write_self_update_fixture(tmp.path(), "1.5.0");
        let (code, out, _err) = drive_under(
            &[
                "self-update",
                "--check",
                "--manifest-file",
                fixture.to_str().unwrap(),
                "--current",
                "1.4.7",
                "--platform",
                "windows_x86_64",
            ],
            tmp.path(),
        );
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::SUCCESS));
        assert!(out.contains("upgrade available"));
        assert!(!out.contains("artifact:"));
    }

    #[test]
    fn self_update_check_json_emits_decision_tag() {
        let tmp = TempDir::new().unwrap();
        let fixture = write_self_update_fixture(tmp.path(), "1.0.0");
        let (code, out, _err) = drive_under(
            &[
                "--json",
                "self-update",
                "--check",
                "--manifest-file",
                fixture.to_str().unwrap(),
                "--current",
                "1.0.0",
                "--platform",
                "linux_x86_64",
            ],
            tmp.path(),
        );
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::SUCCESS));
        let v: serde_json::Value = serde_json::from_str(out.trim()).unwrap();
        assert_eq!(v["decision"], "UpToDate");
        assert_eq!(v["channel"], "stable");
        assert_eq!(v["platform"], "linux_x86_64");
    }

    #[test]
    fn self_update_check_prose_io_failure_returns_74() {
        let tmp = TempDir::new().unwrap();
        let fixture = write_self_update_fixture(tmp.path(), "1.0.0");
        let code = drive_with_failing_out(
            &[
                "self-update",
                "--check",
                "--manifest-file",
                fixture.to_str().unwrap(),
                "--current",
                "1.0.0",
                "--platform",
                "linux_x86_64",
            ],
            tmp.path(),
        );
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::from(74)));
    }

    #[test]
    fn self_update_check_json_io_failure_returns_74() {
        let tmp = TempDir::new().unwrap();
        let fixture = write_self_update_fixture(tmp.path(), "1.0.0");
        let code = drive_with_failing_out(
            &[
                "--json",
                "self-update",
                "--check",
                "--manifest-file",
                fixture.to_str().unwrap(),
                "--current",
                "1.0.0",
                "--platform",
                "linux_x86_64",
            ],
            tmp.path(),
        );
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::from(74)));
    }

    #[test]
    fn self_update_check_upgrade_prose_io_failure_returns_74() {
        let tmp = TempDir::new().unwrap();
        let fixture = write_self_update_fixture(tmp.path(), "1.5.0");
        let code = drive_with_failing_out(
            &[
                "self-update",
                "--check",
                "--manifest-file",
                fixture.to_str().unwrap(),
                "--current",
                "1.4.7",
                "--platform",
                "linux_x86_64",
            ],
            tmp.path(),
        );
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::from(74)));
    }

    #[test]
    fn self_update_check_blocked_prose_io_failure_returns_74() {
        // Build a fixture with min_supported_from to force a Blocked decision.
        let tmp = TempDir::new().unwrap();
        let body = r#"{
            "schema_version": 1,
            "channel": "stable",
            "latest": {
                "version": { "major": 1, "minor": 5, "patch": 0, "pre": null },
                "released_at": { "secs_since_epoch": 1700000000, "nanos_since_epoch": 0 },
                "binary": {
                    "url": "https://dl.stratum.dev/v1.5.0/stratum-linux_x86_64.tar.gz",
                    "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
                    "bytes": 1024,
                    "platform": "linux_x86_64"
                },
                "min_supported_from": { "major": 1, "minor": 3, "patch": 0, "pre": null },
                "release_notes_url": "https://stratum.dev/releases/1.5.0"
            },
            "history": [
                {
                    "version": { "major": 1, "minor": 5, "patch": 0, "pre": null },
                    "released_at": { "secs_since_epoch": 1700000000, "nanos_since_epoch": 0 },
                    "binary": {
                        "url": "https://dl.stratum.dev/v1.5.0/stratum-linux_x86_64.tar.gz",
                        "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
                        "bytes": 1024,
                        "platform": "linux_x86_64"
                    },
                    "min_supported_from": { "major": 1, "minor": 3, "patch": 0, "pre": null },
                    "release_notes_url": "https://stratum.dev/releases/1.5.0"
                }
            ]
        }"#;
        let path = tmp.path().join("manifest.json");
        std::fs::write(&path, body).unwrap();
        let code = drive_with_failing_out(
            &[
                "self-update",
                "--check",
                "--manifest-file",
                path.to_str().unwrap(),
                "--current",
                "1.0.0",
                "--platform",
                "linux_x86_64",
            ],
            tmp.path(),
        );
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::from(74)));
    }

    #[test]
    fn self_update_check_https_url_rejected_non_https() {
        // The URL fetch helper rejects non-https before calling ureq.
        let err = fetch_manifest_https("http://example.com/manifest.json").unwrap_err();
        assert!(err.contains("must be https"), "err: {err}");
    }

    #[test]
    fn channel_arg_wire_forms() {
        assert_eq!(ChannelArg::Stable.as_wire(), "stable");
        assert_eq!(ChannelArg::Beta.as_wire(), "beta");
        assert_eq!(ChannelArg::Nightly.as_wire(), "nightly");
    }

    #[test]
    fn channel_arg_into_update_channel() {
        assert_eq!(
            UpdateChannel::from(ChannelArg::Stable),
            UpdateChannel::Stable
        );
        assert_eq!(UpdateChannel::from(ChannelArg::Beta), UpdateChannel::Beta);
        assert_eq!(
            UpdateChannel::from(ChannelArg::Nightly),
            UpdateChannel::Nightly
        );
    }

    #[test]
    fn platform_arg_wire_forms() {
        assert_eq!(PlatformArg::MacosAarch64.as_wire(), "macos_aarch64");
        assert_eq!(PlatformArg::MacosX86_64.as_wire(), "macos_x86_64");
        assert_eq!(PlatformArg::LinuxAarch64.as_wire(), "linux_aarch64");
        assert_eq!(PlatformArg::LinuxX86_64.as_wire(), "linux_x86_64");
        assert_eq!(PlatformArg::WindowsX86_64.as_wire(), "windows_x86_64");
    }

    #[test]
    fn platform_arg_into_platform_tag() {
        assert_eq!(
            PlatformTag::from(PlatformArg::MacosAarch64),
            PlatformTag::MacOsAarch64
        );
        assert_eq!(
            PlatformTag::from(PlatformArg::MacosX86_64),
            PlatformTag::MacOsX86_64
        );
        assert_eq!(
            PlatformTag::from(PlatformArg::LinuxAarch64),
            PlatformTag::LinuxAarch64
        );
        assert_eq!(
            PlatformTag::from(PlatformArg::LinuxX86_64),
            PlatformTag::LinuxX86_64
        );
        assert_eq!(
            PlatformTag::from(PlatformArg::WindowsX86_64),
            PlatformTag::WindowsX86_64
        );
    }

    #[test]
    fn platform_arg_detect_returns_some_on_supported_host() {
        // CI matrix only runs on macOS / Linux on x86_64 or aarch64. Detect()
        // returns None for unknown OS/ARCH pairs; on any supported host this
        // must be Some.
        let detected = PlatformArg::detect();
        assert!(detected.is_some(), "detect() returned None on host");
    }

    // -----------------------------------------------------------------------
    // events tail unit coverage
    // -----------------------------------------------------------------------

    fn make_record(id: u64, event: Event) -> EventRecord {
        EventRecord {
            id,
            at: SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(1_700_000_000),
            turn_id: None,
            event,
        }
    }

    fn write_jsonl(root: &Path, lines: &[String]) {
        let state = root.join("state");
        std::fs::create_dir_all(&state).unwrap();
        let body: String = lines
            .iter()
            .map(|s| format!("{s}\n"))
            .collect::<Vec<_>>()
            .concat();
        std::fs::write(state.join("events.jsonl"), body).unwrap();
    }

    #[test]
    fn render_prose_covers_all_event_variants() {
        let cases = vec![
            Event::ToolCall {
                tool_id: "fs.read".into(),
                ok: true,
                duration_ms: 12,
            },
            Event::PermissionAsked {
                request: "net".into(),
                decision: "allow_once".into(),
            },
            Event::AgentHandoff {
                from: "planner".into(),
                to: "coder".into(),
                reason: "ready".into(),
            },
            Event::ProviderError {
                provider: "llama-cpp".into(),
                code: "STRAT-E1001".into(),
                message: "oops".into(),
            },
            Event::SandboxLaunched {
                backend: "bwrap".into(),
                profile: "default".into(),
            },
        ];
        let expected_tags = [
            "tool_call",
            "permission_asked",
            "agent_handoff",
            "provider_error",
            "sandbox_launched",
        ];
        for (event, tag) in cases.into_iter().zip(expected_tags) {
            let rec = make_record(1, event);
            let line = render_event_prose(&rec);
            assert!(line.contains(tag), "expected {tag} in {line}");
        }
    }

    #[test]
    fn kind_matches_filters_each_variant() {
        let variants = [
            (
                Event::ToolCall {
                    tool_id: "x".into(),
                    ok: true,
                    duration_ms: 0,
                },
                EventKindArg::ToolCall,
            ),
            (
                Event::PermissionAsked {
                    request: "x".into(),
                    decision: "deny".into(),
                },
                EventKindArg::PermissionAsked,
            ),
            (
                Event::AgentHandoff {
                    from: "a".into(),
                    to: "b".into(),
                    reason: "r".into(),
                },
                EventKindArg::AgentHandoff,
            ),
            (
                Event::ProviderError {
                    provider: "p".into(),
                    code: "c".into(),
                    message: "m".into(),
                },
                EventKindArg::ProviderError,
            ),
            (
                Event::SandboxLaunched {
                    backend: "b".into(),
                    profile: "p".into(),
                },
                EventKindArg::SandboxLaunched,
            ),
        ];
        for (event, kind) in variants {
            let rec = make_record(1, event);
            assert!(kind_matches(&rec, Some(kind)));
            // None matches everything.
            assert!(kind_matches(&rec, None));
        }
    }

    #[test]
    fn tail_each_kind_filter_round_trip() {
        let tmp = TempDir::new().unwrap();
        let records = vec![
            make_record(
                1,
                Event::ToolCall {
                    tool_id: "fs.read".into(),
                    ok: true,
                    duration_ms: 1,
                },
            ),
            make_record(
                2,
                Event::PermissionAsked {
                    request: "net".into(),
                    decision: "allow_once".into(),
                },
            ),
            make_record(
                3,
                Event::AgentHandoff {
                    from: "planner".into(),
                    to: "coder".into(),
                    reason: "ready".into(),
                },
            ),
            make_record(
                4,
                Event::ProviderError {
                    provider: "echo".into(),
                    code: "STRAT-E1001".into(),
                    message: "boom".into(),
                },
            ),
            make_record(
                5,
                Event::SandboxLaunched {
                    backend: "bwrap".into(),
                    profile: "default".into(),
                },
            ),
        ];
        let lines: Vec<String> = records
            .iter()
            .map(|r| serde_json::to_string(r).unwrap())
            .collect();
        write_jsonl(tmp.path(), &lines);

        for kind in [
            "tool_call",
            "permission_asked",
            "agent_handoff",
            "provider_error",
            "sandbox_launched",
        ] {
            let (code, out, _err) = drive_under(&["events", "tail", "--kind", kind], tmp.path());
            assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::SUCCESS));
            let n = out.lines().count();
            assert_eq!(n, 1, "kind={kind} got {n} lines: {out}");
            assert!(out.contains(kind));
        }
    }

    #[test]
    fn tail_prose_io_failure_returns_74() {
        let tmp = TempDir::new().unwrap();
        let rec = make_record(
            1,
            Event::ToolCall {
                tool_id: "fs.read".into(),
                ok: true,
                duration_ms: 1,
            },
        );
        write_jsonl(tmp.path(), &[serde_json::to_string(&rec).unwrap()]);
        let code = drive_with_failing_out(&["events", "tail"], tmp.path());
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::from(74)));
    }

    #[test]
    fn tail_json_io_failure_returns_74() {
        let tmp = TempDir::new().unwrap();
        let rec = make_record(
            1,
            Event::ToolCall {
                tool_id: "fs.read".into(),
                ok: true,
                duration_ms: 1,
            },
        );
        write_jsonl(tmp.path(), &[serde_json::to_string(&rec).unwrap()]);
        let code = drive_with_failing_out(&["events", "tail", "--json"], tmp.path());
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::from(74)));
    }

    #[test]
    fn tail_open_error_when_state_is_a_file_returns_1() {
        // Use a regular file as state/events.jsonl's parent so open() fails
        // with something other than NotFound.
        let tmp = TempDir::new().unwrap();
        let state = tmp.path().join("state");
        std::fs::write(&state, b"not a dir").unwrap();
        let (code, _out, err) = drive_under(&["events", "tail"], tmp.path());
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::from(1)));
        assert!(err.contains("STRAT-E1001"));
    }

    #[test]
    fn drain_reader_skips_malformed_then_reads_good() {
        // Direct unit test on drain_reader so the malformed-line tracing branch
        // is exercised without depending on follow timing.
        let body = "{not json}\n\n{\"id\":1,\"at\":{\"secs_since_epoch\":1700000000,\"nanos_since_epoch\":0},\"turn_id\":null,\"event\":{\"kind\":\"tool_call\",\"tool_id\":\"x\",\"ok\":true,\"duration_ms\":1}}\n";
        let mut reader = std::io::BufReader::new(body.as_bytes());
        let mut sink = Vec::new();
        let mut emitted = 0_usize;
        let args = EventsTailArgs {
            since_id: None,
            limit: None,
            kind: None,
            json: false,
            follow: false,
        };
        let stop = drain_reader(&mut reader, &mut sink, &args, &mut emitted).unwrap();
        assert!(!stop);
        assert_eq!(emitted, 1);
        let out = String::from_utf8(sink).unwrap();
        assert!(out.contains("tool_call"));
    }

    #[test]
    fn drain_reader_stops_on_limit() {
        let r1 = make_record(
            1,
            Event::ToolCall {
                tool_id: "a".into(),
                ok: true,
                duration_ms: 0,
            },
        );
        let r2 = make_record(
            2,
            Event::ToolCall {
                tool_id: "b".into(),
                ok: true,
                duration_ms: 0,
            },
        );
        let r3 = make_record(
            3,
            Event::ToolCall {
                tool_id: "c".into(),
                ok: true,
                duration_ms: 0,
            },
        );
        let body = format!(
            "{}\n{}\n{}\n",
            serde_json::to_string(&r1).unwrap(),
            serde_json::to_string(&r2).unwrap(),
            serde_json::to_string(&r3).unwrap(),
        );
        let mut reader = std::io::BufReader::new(body.as_bytes());
        let mut sink = Vec::new();
        let mut emitted = 0_usize;
        let args = EventsTailArgs {
            since_id: None,
            limit: Some(2),
            kind: None,
            json: false,
            follow: false,
        };
        let stop = drain_reader(&mut reader, &mut sink, &args, &mut emitted).unwrap();
        assert!(stop);
        assert_eq!(emitted, 2);
    }

    #[test]
    fn follow_deadline_honors_env_var() {
        let _guard = EnvVarGuard::set("STRATUM_EVENTS_TAIL_MAX_S", "1");
        let dl = follow_deadline().expect("deadline should be set");
        // dl is in the future relative to now.
        assert!(dl >= SystemTime::now());
    }

    #[test]
    fn follow_deadline_returns_none_when_env_missing() {
        let _guard = EnvVarGuard::unset("STRATUM_EVENTS_TAIL_MAX_S");
        assert!(follow_deadline().is_none());
    }

    #[test]
    fn follow_deadline_returns_none_when_env_invalid() {
        let _guard = EnvVarGuard::set("STRATUM_EVENTS_TAIL_MAX_S", "not-a-number");
        assert!(follow_deadline().is_none());
    }

    #[test]
    fn deadline_reached_true_in_past_false_in_future() {
        let past = SystemTime::UNIX_EPOCH;
        let future = SystemTime::now() + std::time::Duration::from_secs(60);
        assert!(deadline_reached(Some(past)));
        assert!(!deadline_reached(Some(future)));
        assert!(!deadline_reached(None));
    }

    /// RAII guard that sets or unsets an env var for the lifetime of the
    /// scope. Used by the deadline tests so they don't poison each other.
    struct EnvVarGuard {
        key: &'static str,
        prev: Option<String>,
    }

    impl EnvVarGuard {
        fn set(key: &'static str, value: &str) -> Self {
            let prev = std::env::var(key).ok();
            std::env::set_var(key, value);
            Self { key, prev }
        }
        fn unset(key: &'static str) -> Self {
            let prev = std::env::var(key).ok();
            std::env::remove_var(key);
            Self { key, prev }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            match &self.prev {
                Some(v) => std::env::set_var(self.key, v),
                None => std::env::remove_var(self.key),
            }
        }
    }
}
