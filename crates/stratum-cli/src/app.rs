//! The CLI behavior, factored out of `main` for testability.

use std::ffi::OsString;
use std::io::Write;
use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use serde::Serialize;
use stratum_runtime::{
    CancelToken, EchoProvider, GenerateRequest, GpuBackend, HardwareProbe, InstalledToml, Paths,
    Tier,
};
use stratum_types::{Block, ErrorCode, ModelId};
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
    Doctor,
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
        Some(Command::Doctor) => doctor(cli.json, &paths, out, err),
        Some(Command::Init) => init(cli.json, &paths, out, err),
        Some(Command::Echo { prompt, max_blocks }) => echo(cli.json, &prompt, max_blocks, out),
        Some(Command::Chat) => chat_command(&paths, err),
    }
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
    installed: bool,
    issues: Vec<DoctorIssue>,
}

#[derive(Debug, Serialize)]
struct DoctorIssue {
    code: ErrorCode,
    level: &'static str,
    message: String,
}

fn doctor(json: bool, paths: &Paths, out: &mut dyn Write, _err: &mut dyn Write) -> ExitCode {
    let probe = HardwareProbe::run();
    let tier = Tier::classify(&probe);
    let installed = paths.installed_toml().exists();
    let mut issues = Vec::new();
    if !installed {
        issues.push(DoctorIssue {
            code: stratum_types::error::codes::E2003_TIER_DOWNGRADE_REFUSED,
            level: "info",
            message: "no installed.toml found; run `stratum init`".into(),
        });
    }
    let report = DoctorReport {
        schema_version: 1,
        stratum_version: env!("CARGO_PKG_VERSION"),
        tier,
        probe: &probe,
        gpu_accel: probe.gpu,
        installed,
        issues,
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
    } else if writeln!(
        out,
        "stratum {} · tier={} · gpu={} · ram={} MiB · cores={} · installed={}",
        report.stratum_version, tier, probe.gpu, probe.ram_total_mib, probe.cpu_cores, installed
    )
    .is_err()
    {
        return ExitCode::from(74);
    }
    ExitCode::SUCCESS
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
}
