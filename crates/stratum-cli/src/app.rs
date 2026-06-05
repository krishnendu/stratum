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
    payload_is_allowlisted as telemetry_payload_is_allowlisted, AnonInstallId,
    ArtifactRef as ModelArtifactRef, CancelToken, CatalogError, CpuArchTag, EchoProvider, Event,
    EventRecord, GenerateRequest, GpuBackend, HardwareProbe, InstalledToml, LoadedModel,
    ManifestError, MemoryGate, ModelCatalog, ModelEntry, ModelInstaller, ModelSlug, ModelTask,
    ModelTier, OsTag, Paths, PlatformTag, Provider, ReleaseChannel, ReleaseVersion, SandboxReport,
    TelemetryConfig, TelemetryEventKind, TelemetryPayload, Tier, UpdateChannel, UpdateDecision,
    UpdateManifest, DEFAULT_MARGIN_MIB,
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
    /// Open the interactive `EchoProvider`-backed chat TUI.
    Chat,
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
/// This phase exposes only the read-only `--check` action: fetch (or read) an
/// [`UpdateManifest`], compare against the running version, and print the
/// resulting [`UpdateDecision`]. The actual atomic binary swap lands in a
/// later PR.
#[derive(Debug, Args)]
struct SelfUpdateArgs {
    /// Check for an available update and print the decision. Required in this
    /// phase — no other actions are exposed yet.
    #[arg(long)]
    check: bool,
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

/// Arguments for `stratum mem-check`. The values describe the model that
/// would be loaded; the gate compares them against the live `HardwareProbe`.
#[derive(Debug, Args)]
struct MemCheckArgs {
    /// Resident set of the weights, in mebibytes.
    #[arg(long)]
    weight_rss: u32,
    /// KV cache bytes per token.
    #[arg(long)]
    kv_per_token: u32,
    /// Planned context length, in tokens.
    #[arg(long)]
    context: u32,
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
        Some(Command::Chat) => chat_command(&paths, err),
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
        Some(Command::MemCheck(mem_args)) => mem_check(cli.json, &mem_args, out, err),
        Some(Command::SelfUpdate(su_args)) => self_update(cli.json, &su_args, out, err),
        Some(Command::Events(EventsCommand::Tail(tail_args))) => {
            events_tail(&paths, &tail_args, out, err)
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
    out: &mut dyn Write,
    err: &mut dyn Write,
) -> ExitCode {
    let probe = HardwareProbe::run();
    let estimate = MemEstimate {
        weight_rss_mib: args.weight_rss,
        kv_per_token_bytes: args.kv_per_token,
        mmproj_mib: args.mmproj,
        vram_mib: args.vram,
    };
    let gate = MemoryGate::new(args.margin);
    let needed_mib = estimate.hot_ram_mib(args.context);
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

    match gate.check_with(&probe, &estimate, args.context, &loaded) {
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
                .suggest_unloads(&probe, &estimate, args.context, &loaded)
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

fn chat_command(paths: &Paths, err: &mut dyn Write) -> ExitCode {
    let probe = HardwareProbe::run();
    let tier = Tier::classify(&probe);
    match crate::chat::run(paths, tier) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            let _ = writeln!(err, "{e}");
            ExitCode::from(70)
        }
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
// self-update --check
// ---------------------------------------------------------------------------

/// `--json` payload for the artifact slot of a [`SelfUpdateReport`].
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

fn self_update(
    json: bool,
    args: &SelfUpdateArgs,
    out: &mut dyn Write,
    err: &mut dyn Write,
) -> ExitCode {
    if !args.check {
        let _ = writeln!(
            err,
            "stratum self-update: --check is required (no other actions in this phase)"
        );
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
        assert!(err.contains("--check is required"));
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
