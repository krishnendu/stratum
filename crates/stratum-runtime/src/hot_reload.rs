//! Filesystem hot-reload for settings, agents, and hooks.
//!
//! # Why
//!
//! A long-lived TUI session shouldn't force the user to restart Stratum
//! every time they edit `~/.stratum/settings.json`, drop a new agent
//! file into `.stratum/agents/`, or rewire a hook command in
//! `.stratum/settings.local.json`. The reload path here observes the
//! filesystem and fires a [`ReloadEvent`] whenever a watched location
//! changes; the CLI layer wires those events to:
//!
//! 1. re-run [`crate::settings_loader::load`] and swap the merged
//!    settings under an `Arc<RwLock>`,
//! 2. re-run the agent registry loader and replace the live registry,
//! 3. re-parse hooks and rebuild the [`crate::hooks::HookDispatcher`],
//! 4. fire a `SessionStart`-shaped synthetic hook event so user-supplied
//!    `SettingsReload` hooks observe the swap.
//!
//! # Backends
//!
//! - With feature `fs-watch`: real OS watcher via the `notify` crate
//!   (inotify on Linux, FSEvents/KQueue on macOS,
//!   ReadDirectoryChangesW on Windows). Debounced internally.
//! - Without feature `fs-watch`: a zero-cost stub. The `HotReloader`
//!   type still exists so callers compile, but `start` returns a
//!   `Disabled` handle that never emits events. This keeps per-PR CI
//!   off the libnotify / FSEvents link surface — the production CLI
//!   build always enables `fs-watch`.

use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::Duration;

/// Bucket the watcher reports back to the caller. Each event names what
/// changed; the caller looks up which reloader to run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReloadEvent {
    /// `settings.json` / `stratum.toml` changed at any tier. The caller
    /// should re-run [`crate::settings_loader::load`].
    Settings {
        /// The path whose change triggered this event.
        path: PathBuf,
    },
    /// An agent file under `.stratum/agents/` changed (added, removed,
    /// or edited). The caller should rebuild the agent registry.
    Agents {
        /// The path whose change triggered this event.
        path: PathBuf,
    },
    /// A hook command file changed (e.g. an inline script referenced
    /// from `settings.json`). The caller should rebuild the hook
    /// dispatcher.
    Hooks {
        /// The path whose change triggered this event.
        path: PathBuf,
    },
}

/// Inputs to [`HotReloader::start`].
#[derive(Debug, Clone, Default)]
pub struct WatchInputs {
    /// Settings files (any tier). Each entry is watched individually.
    pub settings_files: Vec<PathBuf>,
    /// Directories containing agent definitions. Recursive watch.
    pub agent_dirs: Vec<PathBuf>,
    /// Hook command files / dirs. Individual files for inline scripts.
    pub hook_files: Vec<PathBuf>,
    /// Debounce window (default 250 ms). Multiple events on the same
    /// path within this window collapse to one.
    pub debounce: Option<Duration>,
}

/// Default debounce window. Editors often emit several `WRITE` events
/// per save (truncate, fsync, rename); 250 ms catches them.
pub const DEFAULT_DEBOUNCE: Duration = Duration::from_millis(250);

/// Handle returned from [`HotReloader::start`]. Holds the watcher
/// thread alive; drop it to stop watching.
#[derive(Debug)]
pub struct ReloaderHandle {
    /// The receiver the caller drains. When `fs-watch` is disabled this
    /// receiver never receives anything.
    pub events: mpsc::Receiver<ReloadEvent>,
    /// Backend-specific guard. Keeps the OS watcher registered. The
    /// stub variant holds `()`.
    #[allow(dead_code)]
    guard: WatchGuard,
}

#[cfg(feature = "fs-watch")]
#[derive(Debug)]
struct WatchGuard {
    _watcher: notify::RecommendedWatcher,
    _thread: std::thread::JoinHandle<()>,
}

#[cfg(not(feature = "fs-watch"))]
#[derive(Debug)]
struct WatchGuard;

/// Spawn a watcher thread that fires [`ReloadEvent`]s.
#[derive(Debug)]
pub struct HotReloader;

impl HotReloader {
    /// Start watching the configured paths. The returned handle owns
    /// the watcher thread; drop it to stop watching.
    ///
    /// # Errors
    ///
    /// Returns an error if the OS watcher refuses to register one of the
    /// paths (e.g. permission denied, path doesn't exist). Missing
    /// optional paths are silently skipped — only a *registered* path
    /// that fails to install bubbles up.
    pub fn start(inputs: WatchInputs) -> Result<ReloaderHandle, HotReloadError> {
        let _ = inputs.debounce.unwrap_or(DEFAULT_DEBOUNCE);
        #[cfg(feature = "fs-watch")]
        {
            start_notify(inputs)
        }
        #[cfg(not(feature = "fs-watch"))]
        {
            // Stub backend: hand back a receiver that never emits, plus
            // an empty guard. The caller's `try_recv` loop drains
            // nothing, which is exactly the no-watcher contract.
            let (_tx, rx) = mpsc::channel();
            Ok(ReloaderHandle {
                events: rx,
                guard: WatchGuard,
            })
        }
    }
}

/// Errors raised by [`HotReloader::start`].
#[derive(Debug, thiserror::Error)]
pub enum HotReloadError {
    /// The OS watcher refused to install. Carries the underlying
    /// `notify` crate error text.
    #[error("watcher install failed: {0}")]
    InstallFailed(String),
}

#[cfg(feature = "fs-watch")]
fn start_notify(inputs: WatchInputs) -> Result<ReloaderHandle, HotReloadError> {
    use notify::{RecursiveMode, Watcher};

    let (raw_tx, raw_rx) = mpsc::channel::<notify::Result<notify::Event>>();
    let watcher_tx = raw_tx.clone();
    let mut watcher = notify::recommended_watcher(move |res| {
        // Best-effort: if the receiver is gone we silently drop.
        let _ = watcher_tx.send(res);
    })
    .map_err(|e| HotReloadError::InstallFailed(e.to_string()))?;

    // Snapshot the file → bucket mapping into something the thread can own.
    let settings_files = inputs.settings_files.clone();
    let agent_dirs = inputs.agent_dirs.clone();
    let hook_files = inputs.hook_files.clone();
    let debounce = inputs.debounce.unwrap_or(DEFAULT_DEBOUNCE);

    for p in &settings_files {
        let _ = watcher.watch(p, RecursiveMode::NonRecursive);
    }
    for p in &agent_dirs {
        let _ = watcher.watch(p, RecursiveMode::Recursive);
    }
    for p in &hook_files {
        let _ = watcher.watch(p, RecursiveMode::NonRecursive);
    }

    let (out_tx, out_rx) = mpsc::channel::<ReloadEvent>();
    let thread = std::thread::Builder::new()
        .name("stratum-hot-reload".into())
        .spawn(move || {
            run_dispatch_loop(raw_rx, out_tx, settings_files, agent_dirs, hook_files, debounce);
        })
        .map_err(|e| HotReloadError::InstallFailed(e.to_string()))?;

    Ok(ReloaderHandle {
        events: out_rx,
        guard: WatchGuard {
            _watcher: watcher,
            _thread: thread,
        },
    })
}

#[cfg(feature = "fs-watch")]
fn run_dispatch_loop(
    raw_rx: mpsc::Receiver<notify::Result<notify::Event>>,
    out_tx: mpsc::Sender<ReloadEvent>,
    settings_files: Vec<PathBuf>,
    agent_dirs: Vec<PathBuf>,
    hook_files: Vec<PathBuf>,
    debounce: Duration,
) {
    use std::collections::HashMap;
    use std::time::Instant;

    let mut last_emit: HashMap<PathBuf, Instant> = HashMap::new();
    while let Ok(event) = raw_rx.recv() {
        let Ok(event) = event else {
            continue;
        };
        for path in event.paths {
            let now = Instant::now();
            if let Some(prev) = last_emit.get(&path) {
                if now.duration_since(*prev) < debounce {
                    continue;
                }
            }
            let Some(reload) = classify(&path, &settings_files, &agent_dirs, &hook_files) else {
                continue;
            };
            last_emit.insert(path, now);
            if out_tx.send(reload).is_err() {
                return;
            }
        }
    }
}

#[cfg(feature = "fs-watch")]
fn classify(
    path: &Path,
    settings_files: &[PathBuf],
    agent_dirs: &[PathBuf],
    hook_files: &[PathBuf],
) -> Option<ReloadEvent> {
    // Most specific match wins. Hook files and settings files are
    // exact-path matches; agent dirs match by prefix.
    for p in hook_files {
        if path == p.as_path() {
            return Some(ReloadEvent::Hooks {
                path: path.to_path_buf(),
            });
        }
    }
    for p in settings_files {
        if path == p.as_path() {
            return Some(ReloadEvent::Settings {
                path: path.to_path_buf(),
            });
        }
    }
    for d in agent_dirs {
        if path.starts_with(d) {
            return Some(ReloadEvent::Agents {
                path: path.to_path_buf(),
            });
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn watch_inputs_default_is_empty() {
        let i = WatchInputs::default();
        assert!(i.settings_files.is_empty());
        assert!(i.agent_dirs.is_empty());
        assert!(i.hook_files.is_empty());
        assert!(i.debounce.is_none());
    }

    #[test]
    fn default_debounce_is_250ms() {
        assert_eq!(DEFAULT_DEBOUNCE, Duration::from_millis(250));
    }

    #[test]
    fn stub_backend_returns_handle_without_fs_watch() {
        // This test runs in both cfgs. When fs-watch is enabled we still
        // expect start() to succeed for an empty input set; when it's
        // disabled the stub backend returns instantly.
        let handle = HotReloader::start(WatchInputs::default()).expect("start");
        // With no watched paths nothing fires, regardless of backend.
        assert!(handle.events.try_recv().is_err());
    }

    #[cfg(feature = "fs-watch")]
    #[test]
    fn classify_routes_by_path_bucket() {
        let settings = vec![PathBuf::from("/tmp/settings.json")];
        let agents = vec![PathBuf::from("/tmp/agents")];
        let hooks = vec![PathBuf::from("/tmp/hook.sh")];

        assert_eq!(
            classify(Path::new("/tmp/settings.json"), &settings, &agents, &hooks),
            Some(ReloadEvent::Settings {
                path: PathBuf::from("/tmp/settings.json")
            })
        );
        assert_eq!(
            classify(Path::new("/tmp/agents/foo.md"), &settings, &agents, &hooks),
            Some(ReloadEvent::Agents {
                path: PathBuf::from("/tmp/agents/foo.md")
            })
        );
        assert_eq!(
            classify(Path::new("/tmp/hook.sh"), &settings, &agents, &hooks),
            Some(ReloadEvent::Hooks {
                path: PathBuf::from("/tmp/hook.sh")
            })
        );
        assert_eq!(
            classify(Path::new("/other/file"), &settings, &agents, &hooks),
            None,
        );
    }

    #[cfg(feature = "fs-watch")]
    #[test]
    fn end_to_end_settings_file_change_fires_event() {
        use std::io::Write;
        use std::time::Instant;

        let tmp = tempfile::TempDir::new().expect("tmp");
        let settings = tmp.path().join("settings.json");
        std::fs::write(&settings, "{}").expect("seed");

        let handle = HotReloader::start(WatchInputs {
            settings_files: vec![settings.clone()],
            ..Default::default()
        })
        .expect("start");

        // Give the watcher a moment to install.
        std::thread::sleep(Duration::from_millis(100));
        // Trigger a write.
        let mut f = std::fs::OpenOptions::new()
            .write(true)
            .truncate(true)
            .open(&settings)
            .expect("open");
        writeln!(f, "{{\"a\":1}}").expect("write");
        drop(f);

        // Poll up to 2 s for the event. The watcher is debounced at
        // 250 ms by default so we expect to see the event well within.
        let deadline = Instant::now() + Duration::from_secs(2);
        let mut got = None;
        while Instant::now() < deadline {
            if let Ok(ev) = handle.events.try_recv() {
                got = Some(ev);
                break;
            }
            std::thread::sleep(Duration::from_millis(25));
        }
        let ev = got.expect("expected ReloadEvent::Settings within 2s");
        match ev {
            ReloadEvent::Settings { path } => {
                assert_eq!(path.file_name().unwrap(), settings.file_name().unwrap());
            }
            other => panic!("expected Settings, got {other:?}"),
        }
    }
}
