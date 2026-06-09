//! MCP (Model Context Protocol) client + server data shapes.
//!
//! Phase 3 (data only) — the real protocol (JSON-RPC over stdio / HTTP,
//! spec-version handshake, streaming tool outputs) lands in Phase 6. This
//! module pins the workspace `stratum.toml` shape and the namespace-prefixed
//! tool entries so the global capability matrix can intersect them today.
//!
//! Per `plan/33-mcp-and-external-tools.md` §2-3.
//!
//! ## Lock order
//!
//! When multiple `McpStdioSession` mutexes need to be held at once they
//! must be acquired in the following order, top down, to avoid deadlock:
//!
//! 1. `stdout_buffer` (paired with its `Condvar`).
//! 2. `stdin` (the `Mutex<Option<ChildStdin>>` backing `take_stdin` /
//!    `write_line`).
//! 3. `child` (the `Mutex<Option<Child>>` backing `shutdown`).
//!
//! The `state` and `stderr_lines` mutexes are leaves and can be acquired
//! from any depth without imposing further ordering constraints.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime};

use serde::{Deserialize, Serialize};

use crate::tools::{CapabilityEntry, CapabilityMatrix};

/// Transport an upstream MCP server speaks. Mirrors the `[[mcp.servers]]`
/// `transport = "stdio" | "http"` discriminator from §2.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "transport", rename_all = "snake_case")]
pub enum McpTransport {
    /// Spawn a long-lived subprocess and speak JSON-RPC over its stdio.
    Stdio {
        /// Executable to spawn.
        command: String,
        /// Argument vector. Defaults to empty when absent.
        #[serde(default)]
        args: Vec<String>,
        /// Extra environment variables merged onto the workspace `secrets`
        /// inherited env. Sorted, so the serialized form is deterministic.
        #[serde(default)]
        env: BTreeMap<String, String>,
    },
    /// Connect to a remote MCP endpoint over HTTP.
    Http {
        /// Endpoint URL (validated by the live client, not this data shape).
        url: String,
        /// Optional `keyring://...` URI carrying a bearer token. `None`
        /// means the endpoint is unauthenticated (rare; usually a local
        /// sidecar).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        bearer_token_uri: Option<String>,
    },
}

/// One configured upstream MCP server.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpServerConfig {
    /// Logical name used to key the server in [`McpServerSet`] and to
    /// build the `mcp.<name>.<verb>` capability prefix.
    pub name: String,
    /// Transport-specific connection details.
    #[serde(flatten)]
    pub transport: McpTransport,
    /// Tool keywords (without the `mcp.<server>.` prefix) the user
    /// explicitly allows. Intersected with the global capability matrix.
    #[serde(default)]
    pub allow: Vec<String>,
    /// Tool keywords the user explicitly denies. The denial wins over
    /// `allow`; the live client enforces the rule.
    #[serde(default)]
    pub deny: Vec<String>,
}

/// Live state of one MCP server, used by the `/mcp list` palette.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum McpServerStatus {
    /// Connected and responding to JSON-RPC.
    Live,
    /// Configured but not currently spawned (idle eviction or never used).
    Dormant,
    /// Last spawn or call failed; `reason` carries the human-readable
    /// detail surfaced in the palette. Encoded as a struct variant
    /// because the enum is internally tagged (`#[serde(tag = "state")]`),
    /// which forbids newtype-of-primitive variants.
    Failed {
        /// Human-readable failure reason.
        reason: String,
    },
}

/// Keyed registry of [`McpServerConfig`] entries.
///
/// Keyed by `McpServerConfig::name`; iteration is sorted by that key so
/// CLI / TUI rendering is deterministic.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct McpServerSet(BTreeMap<String, McpServerConfig>);

impl McpServerSet {
    /// Empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register `config`. If a server with the same name was already
    /// present the previous entry is returned (caller decides whether to
    /// surface a warning).
    pub fn insert(&mut self, config: McpServerConfig) -> Option<McpServerConfig> {
        self.0.insert(config.name.clone(), config)
    }

    /// Borrow a configured server by its logical name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&McpServerConfig> {
        self.0.get(name)
    }

    /// Drop a configured server. Returns the removed entry.
    pub fn remove(&mut self, name: &str) -> Option<McpServerConfig> {
        self.0.remove(name)
    }

    /// Count of registered servers.
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Is the registry empty?
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Walk every `(name, config)` pair in alphabetical order by name.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &McpServerConfig)> {
        self.0.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Translate the named server's `allow` list into a `CapabilityMatrix`
    /// of `mcp.<server>.<verb>` entries. Returns an empty matrix when the
    /// server is unknown or has no allow entries.
    #[must_use]
    pub fn effective_capabilities(&self, server_name: &str) -> CapabilityMatrix {
        let Some(server) = self.0.get(server_name) else {
            return CapabilityMatrix::new();
        };
        CapabilityMatrix::from_entries(
            server
                .allow
                .iter()
                .map(|verb| CapabilityEntry::new(format!("mcp.{server_name}.{verb}"))),
        )
    }
}

/// Transport Stratum's own MCP server listens on. Mirrors the
/// `[mcp_server]` `transport = "stdio" | "http"` discriminator from §3.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "transport", rename_all = "snake_case")]
pub enum McpServeTransport {
    /// Stdio sidecar invoked by a single local client (Claude Desktop,
    /// Zed, Cursor).
    Stdio,
    /// HTTP listener; the optional `token_uri` points at the keyring
    /// entry that carries the bearer the listener must accept.
    Http {
        /// `keyring://...` URI for the listener's bearer token. `None`
        /// only makes sense when `allow_any_client = true`; this module
        /// only encodes the shape and leaves the enforcement to the live
        /// server.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        token_uri: Option<String>,
    },
}

/// `[mcp_server]` table from `stratum.toml` (§3).
///
/// Stratum exposes a curated subset of its tool registry to outside MCP
/// clients. The whole feature is **off by default**; this shape only
/// carries the configuration — it does not start a listener.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpServerExpose {
    /// Master switch. Wizard never flips this implicitly.
    pub enabled: bool,
    /// How external clients reach this Stratum instance.
    #[serde(flatten)]
    pub transport: McpServeTransport,
    /// Global capability names exposed to clients (e.g. `fs.read`,
    /// `git.diff`). Serialized in sorted order — the underlying
    /// `BTreeSet` guarantees it.
    #[serde(default)]
    pub expose: BTreeSet<String>,
    /// If `true`, the listener skips bearer-token auth. Only sensible
    /// for stdio; the live server enforces the policy.
    #[serde(default)]
    pub allow_any_client: bool,
}

/// Default ceiling on the number of stderr lines a [`McpStdioSession`]
/// retains in memory. The drain thread drops the oldest line once this
/// many lines accumulate. Exposed as `pub(crate)` so the tests can read
/// it without freezing the literal in assertions.
pub(crate) const STDERR_TAIL_CAP: usize = 200;

/// Configuration for a spawned MCP stdio child process.
///
/// This is the runtime-facing twin of [`McpTransport::Stdio`]: the static
/// config carries strings (because TOML can't type `PathBuf` or
/// `Duration`), while this struct carries the parsed, typed values that
/// [`McpStdioSession::spawn`] actually needs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpStdioConfig {
    /// Absolute path (or `PATH`-resolvable name) of the child executable.
    pub command: PathBuf,
    /// Argument vector passed to the child after `command`.
    #[serde(default)]
    pub args: Vec<String>,
    /// Extra environment variables merged on top of the parent
    /// environment. Sorted for deterministic serialization.
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    /// Working directory for the child. `None` inherits the parent's.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workdir: Option<PathBuf>,
    /// Upper bound on how long [`McpStdioSession::do_handshake`] waits
    /// for the child's first stdout line before giving up.
    #[serde(default = "default_init_timeout")]
    pub init_timeout: Duration,
}

const fn default_init_timeout() -> Duration {
    Duration::from_secs(10)
}

impl Default for McpStdioConfig {
    fn default() -> Self {
        Self {
            command: PathBuf::new(),
            args: Vec::new(),
            env: BTreeMap::new(),
            workdir: None,
            init_timeout: default_init_timeout(),
        }
    }
}

/// Lifecycle state of a [`McpStdioSession`].
///
/// The skeleton transitions are `NotStarted → Initializing → Ready →
/// Closed { … }`. The `Closed` variant also captures whatever exit code
/// the kernel handed back (when one was available).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum McpStdioState {
    /// Constructed but `spawn` has not yet returned. Reserved for a
    /// future builder API; [`McpStdioSession::spawn`] never leaves the
    /// session here.
    NotStarted,
    /// Child has been forked. `do_handshake` has not yet completed.
    Initializing,
    /// First stdout line was read; the session is considered live.
    Ready,
    /// Child has exited (or been killed). `code` is `None` when the
    /// platform did not surface a numeric exit status.
    Closed {
        /// Exit code observed at shutdown. `None` when the child was
        /// signaled or the platform reported no code.
        code: Option<i32>,
    },
}

/// Hand-rolled JSON-RPC stdio session against an upstream MCP server.
///
/// **Skeleton state.** The real client speaks JSON-RPC 2.0 with
/// `initialize` + `notifications/initialized` and routes `tools/call`
/// requests; this scaffold only proves out the lifecycle: spawn the
/// child, drain its stderr, treat ANY first stdout line as a successful
/// handshake, and tear it down cleanly. The full protocol lands in
/// Phase 6 alongside the matching server side.
#[derive(Debug)]
pub struct McpStdioSession {
    child: Mutex<Option<Child>>,
    name: String,
    started_at: SystemTime,
    stderr_lines: Arc<Mutex<Vec<String>>>,
    state: Mutex<McpStdioState>,
    /// One-shot receiver fed by the stdout reader thread with the first
    /// line the child writes. `None` until `spawn` populates it.
    stdout_rx: Mutex<Option<mpsc::Receiver<String>>>,
    /// Rolling buffer of stdout lines pushed by the drain thread, paired
    /// with a `Condvar` so [`Self::recv_line`] can park instead of busy
    /// poll. Lines land here in addition to the one-shot mpsc above so
    /// the legacy handshake reader keeps working byte-for-byte.
    stdout_buffer: Arc<(Mutex<VecDeque<String>>, Condvar)>,
    /// Child's stdin handle. Wrapped in a `Mutex<Option<…>>` so it can
    /// either be consumed by [`Self::take_stdin`] (caller takes
    /// ownership) or driven by [`Self::write_line`] (the session keeps
    /// the handle). After `take_stdin`, `write_line` errors with
    /// [`McpSessionError::StdinAlreadyTaken`].
    stdin: Mutex<Option<ChildStdin>>,
    init_timeout: Duration,
}

/// Errors emitted by [`McpStdioSession`] lifecycle methods.
///
/// Deliberately *not* allocated a new `STRAT-E…` code: the skeleton
/// surface is internal and the full client (Phase 6) folds these into
/// the protocol-level error codes already reserved for MCP.
#[derive(Debug)]
pub enum McpSessionError {
    /// `std::process::Command::spawn` itself failed (executable missing,
    /// permission denied, …).
    Spawn(std::io::Error),
    /// The child did not write its first stdout line within
    /// `init_timeout`.
    HandshakeTimeout {
        /// Elapsed window that was waited.
        after: Duration,
    },
    /// The stdout pipe yielded an IO error during the handshake.
    HandshakeIo(std::io::Error),
    /// `Child::kill` / `wait` failed during shutdown.
    Shutdown(std::io::Error),
    /// A lifecycle method was called from an inappropriate state — e.g.
    /// `do_handshake` after `shutdown`.
    BadState {
        /// State the session was actually in.
        current: McpStdioState,
        /// Human-readable label of the state(s) the call expected.
        expected: String,
    },
    /// [`McpStdioSession::take_stdin`] was called a second time, or
    /// [`McpStdioSession::write_line`] was called after `take_stdin`
    /// already moved ownership of the handle to an external caller.
    StdinAlreadyTaken,
    /// [`McpStdioSession::recv_line`] observed the stdout drain thread
    /// disconnect (the child closed its stdout / exited) before a line
    /// arrived.
    RecvLineClosed,
    /// [`McpStdioSession::write_line`] failed to write the bytes to the
    /// child's stdin pipe.
    WriteFailed(std::io::Error),
}

impl fmt::Display for McpSessionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Spawn(err) => write!(f, "mcp stdio spawn failed: {err}"),
            Self::HandshakeTimeout { after } => {
                write!(
                    f,
                    "mcp stdio handshake timed out after {} ms",
                    after.as_millis()
                )
            }
            Self::HandshakeIo(err) => write!(f, "mcp stdio handshake io error: {err}"),
            Self::Shutdown(err) => write!(f, "mcp stdio shutdown failed: {err}"),
            Self::BadState { current, expected } => {
                write!(
                    f,
                    "mcp stdio session in unexpected state {current:?}, expected {expected}"
                )
            }
            Self::StdinAlreadyTaken => {
                write!(f, "mcp stdio stdin handle already taken")
            }
            Self::RecvLineClosed => {
                write!(f, "mcp stdio recv_line: stdout pipe closed")
            }
            Self::WriteFailed(err) => write!(f, "mcp stdio write_line failed: {err}"),
        }
    }
}

impl std::error::Error for McpSessionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Spawn(err)
            | Self::HandshakeIo(err)
            | Self::Shutdown(err)
            | Self::WriteFailed(err) => Some(err),
            Self::HandshakeTimeout { .. }
            | Self::BadState { .. }
            | Self::StdinAlreadyTaken
            | Self::RecvLineClosed => None,
        }
    }
}

impl McpStdioSession {
    /// Spawn the configured child process and hand back a live session
    /// that has been transitioned to [`McpStdioState::Initializing`].
    ///
    /// The handshake itself is deferred to [`Self::do_handshake`] so
    /// callers can interleave timeouts or other readiness checks.
    ///
    /// # Errors
    /// Returns [`McpSessionError::Spawn`] when the child cannot be
    /// forked (missing executable, EACCES, …).
    pub fn spawn(name: String, cfg: &McpStdioConfig) -> Result<Self, McpSessionError> {
        let mut command = Command::new(&cfg.command);
        command
            .args(&cfg.args)
            .envs(&cfg.env)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        if let Some(dir) = cfg.workdir.as_ref() {
            command.current_dir(dir);
        }
        let mut child = command.spawn().map_err(McpSessionError::Spawn)?;
        let stdin_handle = child.stdin.take();

        // Wire the stdout reader: a background thread reads every line
        // the child writes. The first line is mirrored into a one-shot
        // mpsc so the legacy handshake reader keeps working byte-for-
        // byte; every line (including the first) is also appended to
        // the rolling `stdout_buffer`, paired with a `Condvar` so
        // `recv_line` can park instead of busy-poll. EOF / IO failure
        // drops the sender and notifies the condvar, surfacing as
        // `RecvTimeoutError::Disconnected` in the handshake and as
        // `RecvLineClosed` in `recv_line`.
        let (stdout_tx, stdout_rx) = mpsc::channel::<String>();
        let stdout_buffer: Arc<(Mutex<VecDeque<String>>, Condvar)> =
            Arc::new((Mutex::new(VecDeque::<String>::new()), Condvar::new()));
        if let Some(stdout) = child.stdout.take() {
            let buffer = Arc::clone(&stdout_buffer);
            thread::spawn(move || {
                let mut reader = BufReader::new(stdout);
                let mut first = true;
                loop {
                    let mut line = String::new();
                    match reader.read_line(&mut line) {
                        Ok(0) | Err(_) => break,
                        Ok(_) => {
                            if first {
                                let _ = stdout_tx.send(line.clone());
                                first = false;
                            }
                            let (lock, cvar) = &*buffer;
                            if let Ok(mut guard) = lock.lock() {
                                guard.push_back(line);
                                cvar.notify_all();
                            }
                        }
                    }
                }
                // Drop the mpsc sender explicitly to surface EOF to any
                // handshake still parked on it, and wake any `recv_line`
                // parked on the condvar so it can observe the closed
                // pipe.
                drop(stdout_tx);
                let (lock, cvar) = &*buffer;
                if lock.lock().is_ok() {
                    cvar.notify_all();
                }
            });
        }

        let stderr_lines = Arc::new(Mutex::new(Vec::<String>::new()));
        if let Some(stderr) = child.stderr.take() {
            let sink = Arc::clone(&stderr_lines);
            thread::spawn(move || {
                let reader = BufReader::new(stderr);
                for line in reader.lines().map_while(Result::ok) {
                    if let Ok(mut guard) = sink.lock() {
                        if guard.len() >= STDERR_TAIL_CAP {
                            guard.remove(0);
                        }
                        guard.push(line);
                    }
                }
            });
        }

        Ok(Self {
            child: Mutex::new(Some(child)),
            name,
            started_at: SystemTime::now(),
            stderr_lines,
            state: Mutex::new(McpStdioState::Initializing),
            stdout_rx: Mutex::new(Some(stdout_rx)),
            stdout_buffer,
            stdin: Mutex::new(stdin_handle),
            init_timeout: cfg.init_timeout,
        })
    }

    /// Execute the JSON-RPC `initialize` handshake.
    ///
    /// Skeleton implementation: waits for the first stdout line from
    /// the child (up to `init_timeout`) and treats it unconditionally
    /// as a successful handshake. The real implementation parses the
    /// JSON-RPC response and validates the protocol version.
    ///
    /// # Errors
    /// - [`McpSessionError::BadState`] if called outside
    ///   [`McpStdioState::Initializing`].
    /// - [`McpSessionError::HandshakeTimeout`] if the child writes
    ///   nothing within `init_timeout`.
    /// - [`McpSessionError::HandshakeIo`] on any IO error from stdout.
    pub fn do_handshake(&self) -> Result<(), McpSessionError> {
        let init_timeout = self.init_timeout;
        let current = self.state();
        if !matches!(current, McpStdioState::Initializing) {
            return Err(McpSessionError::BadState {
                current,
                expected: "initializing".to_owned(),
            });
        }

        // The state check above ensures `stdout_rx` is still populated:
        // it only flips to `None` after a successful handshake (which
        // transitions state to `Ready` and trips the check above) or an
        // error path (which transitions to `Closed`, ditto).
        let rx = self
            .stdout_rx
            .lock()
            .ok()
            .and_then(|mut guard| guard.take());
        let Some(rx) = rx else {
            self.set_state(McpStdioState::Closed { code: None });
            return Err(McpSessionError::HandshakeIo(std::io::Error::other(
                "stdout pipe already consumed",
            )));
        };

        let outcome = rx.recv_timeout(init_timeout);
        match outcome {
            Ok(_line) => {
                self.set_state(McpStdioState::Ready);
                Ok(())
            }
            Err(err) => {
                self.set_state(McpStdioState::Closed { code: None });
                Err(match err {
                    mpsc::RecvTimeoutError::Timeout => McpSessionError::HandshakeTimeout {
                        after: init_timeout,
                    },
                    mpsc::RecvTimeoutError::Disconnected => {
                        McpSessionError::HandshakeIo(std::io::Error::new(
                            std::io::ErrorKind::UnexpectedEof,
                            "stdout reader disconnected",
                        ))
                    }
                })
            }
        }
    }

    fn set_state(&self, new_state: McpStdioState) {
        if let Ok(mut guard) = self.state.lock() {
            *guard = new_state;
        }
    }

    /// Kill the child and wait for it to exit. Idempotent: a second
    /// call when the session is already closed returns the cached exit
    /// code.
    ///
    /// # Errors
    /// [`McpSessionError::Shutdown`] when `Child::kill` or
    /// `Child::wait` returns an IO error.
    pub fn shutdown(&self) -> Result<Option<i32>, McpSessionError> {
        // Fast path: already closed.
        if let McpStdioState::Closed { code } = self.state() {
            return Ok(code);
        }

        let child_opt = self.child.lock().ok().and_then(|mut g| g.take());
        let Some(mut child) = child_opt else {
            self.set_state(McpStdioState::Closed { code: None });
            return Ok(None);
        };

        // Best-effort kill: `Child::kill` returns `InvalidInput` when
        // the child has already exited, which is a no-op for us.
        if let Err(err) = child.kill() {
            if err.kind() != std::io::ErrorKind::InvalidInput {
                return Err(McpSessionError::Shutdown(err));
            }
        }
        let status = child.wait().map_err(McpSessionError::Shutdown)?;
        let code = status.code();
        self.set_state(McpStdioState::Closed { code });
        Ok(code)
    }

    /// `true` once the handshake has completed successfully and the
    /// child has not yet been shut down.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        matches!(self.state(), McpStdioState::Ready)
    }

    /// Snapshot of the current lifecycle state.
    ///
    /// If the underlying mutex is poisoned (only possible after a panic
    /// in another thread holding the guard), the session is treated as
    /// `Closed { code: None }`.
    #[must_use]
    pub fn state(&self) -> McpStdioState {
        self.state
            .lock()
            .map(|g| g.clone())
            .unwrap_or(McpStdioState::Closed { code: None })
    }

    /// Last `max_lines` lines drained from the child's stderr,
    /// oldest first. Capped at [`STDERR_TAIL_CAP`].
    #[must_use]
    pub fn stderr_tail(&self, max_lines: usize) -> Vec<String> {
        let snapshot: Vec<String> = self
            .stderr_lines
            .lock()
            .map(|guard| guard.clone())
            .unwrap_or_default();
        let take = snapshot.len().min(max_lines);
        snapshot[snapshot.len().saturating_sub(take)..].to_vec()
    }

    /// Logical server name this session was spawned for.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Wall-clock instant `spawn` returned successfully.
    #[must_use]
    pub const fn started_at(&self) -> SystemTime {
        self.started_at
    }

    /// Take ownership of the child's stdin handle. One-shot: the second
    /// call (or any call after [`Self::write_line`]'s handle has been
    /// otherwise consumed) returns [`McpSessionError::StdinAlreadyTaken`].
    ///
    /// After this call the session no longer owns stdin, so subsequent
    /// [`Self::write_line`] invocations will also error with
    /// `StdinAlreadyTaken`.
    ///
    /// # Errors
    /// [`McpSessionError::StdinAlreadyTaken`] when the handle has
    /// already been taken or was never available (child closed stdin or
    /// session was constructed without a stdin pipe).
    pub fn take_stdin(&self) -> Result<ChildStdin, McpSessionError> {
        let mut guard = self
            .stdin
            .lock()
            .map_err(|_| McpSessionError::StdinAlreadyTaken)?;
        guard.take().ok_or(McpSessionError::StdinAlreadyTaken)
    }

    /// Block up to `timeout` for the next stdout line. Returns
    /// `Ok(Some(line))` with the raw bytes (trailing `\n` preserved),
    /// `Ok(None)` when the timeout elapsed with the buffer empty, and
    /// [`McpSessionError::RecvLineClosed`] when the drain thread has
    /// exited (child closed its stdout / process died) and no buffered
    /// line is left to hand back.
    ///
    /// # Errors
    /// - [`McpSessionError::RecvLineClosed`] if the underlying stdout
    ///   pipe has been closed and the buffer is empty.
    pub fn recv_line(&self, timeout: Duration) -> Result<Option<String>, McpSessionError> {
        let deadline = Instant::now().checked_add(timeout);
        let (lock, cvar) = &*self.stdout_buffer;
        let mut guard = lock.lock().map_err(|_| McpSessionError::RecvLineClosed)?;
        loop {
            if let Some(line) = guard.pop_front() {
                return Ok(Some(line));
            }
            // The drain thread only holds a weak reference via `Arc`;
            // once it exits its clone is dropped. When this `Arc` is
            // the last strong reference the writer has gone away.
            if Arc::strong_count(&self.stdout_buffer) == 1 {
                return Err(McpSessionError::RecvLineClosed);
            }
            let now = Instant::now();
            let remaining = match deadline {
                Some(d) if d > now => d - now,
                Some(_) => return Ok(None),
                None => Duration::from_millis(10),
            };
            let (next_guard, result) = cvar
                .wait_timeout(guard, remaining)
                .map_err(|_| McpSessionError::RecvLineClosed)?;
            guard = next_guard;
            if result.timed_out() && guard.is_empty() {
                if Arc::strong_count(&self.stdout_buffer) == 1 {
                    return Err(McpSessionError::RecvLineClosed);
                }
                return Ok(None);
            }
        }
    }

    /// Drain every currently-buffered stdout line and return them in
    /// arrival order. Returns an empty `Vec` when nothing is pending.
    /// Lines are removed from the internal buffer; subsequent
    /// [`Self::recv_line`] calls will only see lines that arrive after
    /// this call.
    #[must_use]
    pub fn pending_lines(&self) -> Vec<String> {
        let (lock, _) = &*self.stdout_buffer;
        let Ok(mut guard) = lock.lock() else {
            return Vec::new();
        };
        guard.drain(..).collect()
    }

    /// `true` while the child process is still running. Uses
    /// `Child::try_wait` under the hood; a poisoned mutex or any
    /// `try_wait` error is reported as "not alive".
    #[must_use]
    pub fn is_alive(&self) -> bool {
        let Ok(mut guard) = self.child.lock() else {
            return false;
        };
        let Some(child) = guard.as_mut() else {
            return false;
        };
        matches!(child.try_wait(), Ok(None))
    }

    /// Write `line` (followed by a single `\n`) to the child's stdin
    /// and flush. Only usable while the session still owns stdin: once
    /// [`Self::take_stdin`] has handed ownership to an external caller
    /// this method errors with [`McpSessionError::StdinAlreadyTaken`].
    ///
    /// # Errors
    /// - [`McpSessionError::StdinAlreadyTaken`] if stdin has already
    ///   been taken (or was never available).
    /// - [`McpSessionError::WriteFailed`] if the write or flush returns
    ///   an IO error.
    // The stdin mutex must be held for the full write+flush so two
    // concurrent callers cannot interleave bytes into the JSON-RPC
    // framing — that's the whole point of the lock — so the tightening
    // lint doesn't apply here.
    #[allow(clippy::significant_drop_tightening)]
    pub fn write_line(&self, line: &str) -> Result<(), McpSessionError> {
        let mut guard = self
            .stdin
            .lock()
            .map_err(|_| McpSessionError::StdinAlreadyTaken)?;
        let stdin = guard.as_mut().ok_or(McpSessionError::StdinAlreadyTaken)?;
        stdin
            .write_all(line.as_bytes())
            .and_then(|()| stdin.write_all(b"\n"))
            .and_then(|()| stdin.flush())
            .map_err(McpSessionError::WriteFailed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stdio_cfg() -> McpServerConfig {
        McpServerConfig {
            name: "filesystem".into(),
            transport: McpTransport::Stdio {
                command: "uvx".into(),
                args: vec!["mcp-server-filesystem".into(), "--root".into(), ".".into()],
                env: BTreeMap::new(),
            },
            allow: vec!["read".into(), "list".into()],
            deny: vec!["write".into()],
        }
    }

    fn http_cfg(token: Option<&str>) -> McpServerConfig {
        McpServerConfig {
            name: "linear".into(),
            transport: McpTransport::Http {
                url: "https://mcp.linear.app".into(),
                bearer_token_uri: token.map(str::to_owned),
            },
            allow: vec!["issue.read".into()],
            deny: vec![],
        }
    }

    #[test]
    fn stdio_transport_roundtrips_via_json() {
        let cfg = stdio_cfg();
        let json = serde_json::to_string(&cfg).expect("serialize");
        let back: McpServerConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(cfg, back);
    }

    #[test]
    fn http_transport_with_bearer_roundtrips() {
        let cfg = http_cfg(Some("keyring://linear/personal"));
        let json = serde_json::to_string(&cfg).expect("serialize");
        let back: McpServerConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(cfg, back);
        assert!(json.contains("keyring://linear/personal"));
    }

    #[test]
    fn http_transport_without_bearer_skips_field() {
        let cfg = http_cfg(None);
        let json = serde_json::to_string(&cfg).expect("serialize");
        assert!(!json.contains("bearer_token_uri"));
        let back: McpServerConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(cfg, back);
    }

    #[test]
    fn server_config_parses_stdio_toml() {
        let toml = r#"
name = "filesystem"
transport = "stdio"
command = "uvx"
args = ["mcp-server-filesystem", "--root", "."]
allow = ["read", "list", "search"]
deny = ["write"]
"#;
        let cfg: McpServerConfig = toml_edit::de::from_str(toml).expect("parse");
        assert_eq!(cfg.name, "filesystem");
        match cfg.transport {
            McpTransport::Stdio {
                ref command,
                ref args,
                ref env,
            } => {
                assert_eq!(command, "uvx");
                assert_eq!(args.len(), 3);
                assert!(env.is_empty());
            }
            McpTransport::Http { .. } => panic!("expected stdio"),
        }
        assert_eq!(cfg.allow, vec!["read", "list", "search"]);
        assert_eq!(cfg.deny, vec!["write"]);
    }

    #[test]
    fn server_config_parses_http_toml() {
        let toml = r#"
name = "linear"
transport = "http"
url = "https://mcp.linear.app"
bearer_token_uri = "keyring://linear/personal"
allow = ["issue.read"]
"#;
        let cfg: McpServerConfig = toml_edit::de::from_str(toml).expect("parse");
        match cfg.transport {
            McpTransport::Http {
                ref url,
                ref bearer_token_uri,
            } => {
                assert_eq!(url, "https://mcp.linear.app");
                assert_eq!(
                    bearer_token_uri.as_deref(),
                    Some("keyring://linear/personal")
                );
            }
            McpTransport::Stdio { .. } => panic!("expected http"),
        }
        assert!(cfg.deny.is_empty());
    }

    #[test]
    fn server_set_insert_returns_prior() {
        let mut set = McpServerSet::new();
        assert!(set.is_empty());
        assert!(set.insert(stdio_cfg()).is_none());
        let mut renamed = stdio_cfg();
        renamed.allow = vec!["read".into()];
        let prior = set.insert(renamed).expect("prior config");
        assert_eq!(prior.allow, vec!["read".to_owned(), "list".to_owned()]);
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn server_set_get_and_remove() {
        let mut set = McpServerSet::new();
        set.insert(stdio_cfg());
        assert!(set.get("filesystem").is_some());
        assert!(set.get("missing").is_none());
        let removed = set.remove("filesystem").expect("removed");
        assert_eq!(removed.name, "filesystem");
        assert!(set.is_empty());
        assert!(set.remove("filesystem").is_none());
    }

    #[test]
    fn effective_capabilities_prefixes_allow_entries() {
        let mut set = McpServerSet::new();
        set.insert(McpServerConfig {
            name: "fs".into(),
            transport: McpTransport::Stdio {
                command: "mcp-fs".into(),
                args: vec![],
                env: BTreeMap::new(),
            },
            allow: vec!["read".into()],
            deny: vec![],
        });
        let matrix = set.effective_capabilities("fs");
        assert_eq!(matrix.len(), 1);
        assert!(matrix.allows("mcp.fs.read", None));
        let names: Vec<&str> = matrix.entries().map(CapabilityEntry::as_str).collect();
        assert_eq!(names, vec!["mcp.fs.read"]);
    }

    #[test]
    fn effective_capabilities_empty_allow_is_empty_matrix() {
        let mut set = McpServerSet::new();
        set.insert(McpServerConfig {
            name: "fs".into(),
            transport: McpTransport::Stdio {
                command: "mcp-fs".into(),
                args: vec![],
                env: BTreeMap::new(),
            },
            allow: vec![],
            deny: vec![],
        });
        assert!(set.effective_capabilities("fs").is_empty());
    }

    #[test]
    fn effective_capabilities_unknown_server_is_empty_matrix() {
        let set = McpServerSet::new();
        assert!(set.effective_capabilities("nope").is_empty());
    }

    #[test]
    fn iter_walks_servers_alphabetically() {
        let mut set = McpServerSet::new();
        set.insert(McpServerConfig {
            name: "zeta".into(),
            transport: McpTransport::Stdio {
                command: "z".into(),
                args: vec![],
                env: BTreeMap::new(),
            },
            allow: vec![],
            deny: vec![],
        });
        set.insert(McpServerConfig {
            name: "alpha".into(),
            transport: McpTransport::Stdio {
                command: "a".into(),
                args: vec![],
                env: BTreeMap::new(),
            },
            allow: vec![],
            deny: vec![],
        });
        set.insert(McpServerConfig {
            name: "mid".into(),
            transport: McpTransport::Stdio {
                command: "m".into(),
                args: vec![],
                env: BTreeMap::new(),
            },
            allow: vec![],
            deny: vec![],
        });
        let names: Vec<&str> = set.iter().map(|(name, _)| name).collect();
        assert_eq!(names, vec!["alpha", "mid", "zeta"]);
    }

    #[test]
    fn status_failed_roundtrips_through_serde() {
        let status = McpServerStatus::Failed {
            reason: "connection refused".into(),
        };
        let json = serde_json::to_string(&status).expect("serialize");
        // Internally-tagged: the discriminator is `state`.
        assert!(json.contains("\"state\""));
        let back: McpServerStatus = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(status, back);
    }

    #[test]
    fn status_live_and_dormant_roundtrip() {
        for s in [McpServerStatus::Live, McpServerStatus::Dormant] {
            let json = serde_json::to_string(&s).expect("serialize");
            let back: McpServerStatus = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(s, back);
        }
    }

    #[test]
    fn server_expose_http_with_token_roundtrips() {
        let cfg = McpServerExpose {
            enabled: true,
            transport: McpServeTransport::Http {
                token_uri: Some("keyring://stratum/mcp-serve-token".into()),
            },
            expose: BTreeSet::from(["fs.read".to_owned(), "git.diff".to_owned()]),
            allow_any_client: false,
        };
        let toml = toml_edit::ser::to_string(&cfg).expect("serialize");
        let back: McpServerExpose = toml_edit::de::from_str(&toml).expect("deserialize");
        assert_eq!(cfg, back);
    }

    #[test]
    fn server_expose_stdio_allow_any_client_optional_token() {
        let cfg = McpServerExpose {
            enabled: true,
            transport: McpServeTransport::Stdio,
            expose: BTreeSet::from(["rag.search".to_owned()]),
            allow_any_client: true,
        };
        let toml = toml_edit::ser::to_string(&cfg).expect("serialize");
        // Stdio variant has no `token_uri` field at all.
        assert!(!toml.contains("token_uri"));
        let back: McpServerExpose = toml_edit::de::from_str(&toml).expect("deserialize");
        assert_eq!(cfg, back);
    }

    #[test]
    fn server_expose_serializes_expose_sorted() {
        let cfg = McpServerExpose {
            enabled: true,
            transport: McpServeTransport::Stdio,
            expose: BTreeSet::from([
                "git.diff".to_owned(),
                "fs.read".to_owned(),
                "rag.search".to_owned(),
            ]),
            allow_any_client: false,
        };
        let json = serde_json::to_string(&cfg).expect("serialize");
        // Sorted: fs.read < git.diff < rag.search.
        let fs_idx = json.find("fs.read").expect("fs.read present");
        let git_idx = json.find("git.diff").expect("git.diff present");
        let rag_idx = json.find("rag.search").expect("rag.search present");
        assert!(fs_idx < git_idx);
        assert!(git_idx < rag_idx);
    }

    #[test]
    fn stdio_config_default_init_timeout_is_ten_seconds() {
        let cfg = McpStdioConfig::default();
        assert_eq!(cfg.init_timeout, Duration::from_secs(10));
        assert!(cfg.args.is_empty());
        assert!(cfg.env.is_empty());
        assert!(cfg.workdir.is_none());
        assert_eq!(cfg.command, PathBuf::new());
    }

    #[test]
    fn stdio_config_roundtrips_via_json() {
        let mut env = BTreeMap::new();
        env.insert("RUST_LOG".to_owned(), "info".to_owned());
        let cfg = McpStdioConfig {
            command: PathBuf::from("/usr/bin/mcp-fs"),
            args: vec!["--root".into(), "/tmp".into()],
            env,
            workdir: Some(PathBuf::from("/tmp")),
            init_timeout: Duration::from_millis(500),
        };
        let json = serde_json::to_string(&cfg).expect("serialize");
        let back: McpStdioConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(cfg, back);
    }

    #[test]
    fn stdio_config_omits_workdir_when_none() {
        let cfg = McpStdioConfig::default();
        let json = serde_json::to_string(&cfg).expect("serialize");
        assert!(!json.contains("workdir"));
    }

    #[test]
    fn stdio_state_roundtrips_each_variant() {
        let variants = [
            McpStdioState::NotStarted,
            McpStdioState::Initializing,
            McpStdioState::Ready,
            McpStdioState::Closed { code: Some(0) },
            McpStdioState::Closed { code: None },
        ];
        for variant in variants {
            let json = serde_json::to_string(&variant).expect("serialize");
            let back: McpStdioState = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(variant, back);
        }
    }

    #[test]
    fn stdio_state_is_clone_eq_debug() {
        let a = McpStdioState::Closed { code: Some(7) };
        let b = a.clone();
        assert_eq!(a, b);
        let debug = format!("{a:?}");
        assert!(debug.contains("Closed"));
        assert!(debug.contains('7'));
    }

    #[cfg(unix)]
    fn echo_hello_cfg() -> McpStdioConfig {
        McpStdioConfig {
            command: PathBuf::from("/bin/sh"),
            args: vec!["-c".into(), "echo hello".into()],
            env: BTreeMap::new(),
            workdir: None,
            init_timeout: Duration::from_secs(2),
        }
    }

    #[cfg(unix)]
    #[test]
    fn spawn_succeeds_against_echo_script() {
        let cfg = echo_hello_cfg();
        let session = McpStdioSession::spawn("echo".into(), &cfg).expect("spawn ok");
        assert_eq!(session.name(), "echo");
        // State immediately after spawn is `Initializing`.
        assert!(matches!(session.state(), McpStdioState::Initializing));
        assert!(!session.is_ready());
        // Wall clock was set.
        let _ = session.started_at();
        // Cleanup.
        let _ = session.shutdown();
    }

    #[cfg(unix)]
    #[test]
    fn do_handshake_transitions_to_ready() {
        let cfg = echo_hello_cfg();
        let session = McpStdioSession::spawn("echo".into(), &cfg).expect("spawn ok");
        session.do_handshake().expect("handshake ok");
        assert!(matches!(session.state(), McpStdioState::Ready));
        assert!(session.is_ready());
        let _ = session.shutdown();
    }

    #[cfg(unix)]
    #[test]
    fn shutdown_returns_exit_code_and_marks_closed() {
        let cfg = echo_hello_cfg();
        let session = McpStdioSession::spawn("echo".into(), &cfg).expect("spawn ok");
        session.do_handshake().expect("handshake ok");
        // Give the child a moment to exit on its own.
        std::thread::sleep(Duration::from_millis(50));
        let code = session.shutdown().expect("shutdown ok");
        // /bin/sh -c "echo hello" exits 0 when it completes naturally;
        // when we killed it first the code may be None. Accept either.
        assert!(code == Some(0) || code.is_none());
        assert!(matches!(session.state(), McpStdioState::Closed { .. }));
        // Idempotent.
        let again = session.shutdown().expect("second shutdown ok");
        assert_eq!(again, code);
    }

    #[cfg(unix)]
    #[test]
    fn spawn_with_nonexistent_command_errors() {
        let cfg = McpStdioConfig {
            command: PathBuf::from("/this/path/definitely/does/not/exist/xyz"),
            args: vec![],
            env: BTreeMap::new(),
            workdir: None,
            init_timeout: Duration::from_millis(100),
        };
        let err = McpStdioSession::spawn("missing".into(), &cfg).expect_err("must fail");
        assert!(matches!(err, McpSessionError::Spawn(_)));
        // Display smoke.
        let msg = format!("{err}");
        assert!(msg.contains("spawn failed"));
    }

    #[cfg(unix)]
    #[test]
    fn do_handshake_times_out_when_child_is_silent() {
        let cfg = McpStdioConfig {
            command: PathBuf::from("/bin/sh"),
            args: vec!["-c".into(), "sleep 5".into()],
            env: BTreeMap::new(),
            workdir: None,
            init_timeout: Duration::from_millis(100),
        };
        let session = McpStdioSession::spawn("sleeper".into(), &cfg).expect("spawn ok");
        let err = session.do_handshake().expect_err("must time out");
        assert!(matches!(err, McpSessionError::HandshakeTimeout { .. }));
        // State transitioned to Closed.
        assert!(matches!(session.state(), McpStdioState::Closed { .. }));
        let _ = session.shutdown();
    }

    #[cfg(unix)]
    #[test]
    fn stderr_tail_captures_lines() {
        let cfg = McpStdioConfig {
            command: PathBuf::from("/bin/sh"),
            args: vec!["-c".into(), "echo oops >&2; echo hello".into()],
            env: BTreeMap::new(),
            workdir: None,
            init_timeout: Duration::from_secs(2),
        };
        let session = McpStdioSession::spawn("err".into(), &cfg).expect("spawn ok");
        session.do_handshake().expect("handshake ok");
        // Allow the drain thread to land the stderr line.
        std::thread::sleep(Duration::from_millis(150));
        let tail = session.stderr_tail(10);
        assert!(
            tail.iter().any(|line| line.contains("oops")),
            "expected oops in stderr tail, got {tail:?}"
        );
        // Asking for 0 yields an empty slice without panicking.
        assert!(session.stderr_tail(0).is_empty());
        let _ = session.shutdown();
    }

    #[cfg(unix)]
    #[test]
    fn is_ready_tracks_state() {
        let cfg = echo_hello_cfg();
        let session = McpStdioSession::spawn("rdy".into(), &cfg).expect("spawn ok");
        assert!(!session.is_ready());
        session.do_handshake().expect("handshake ok");
        assert!(session.is_ready());
        let _ = session.shutdown();
        assert!(!session.is_ready());
    }

    #[cfg(unix)]
    #[test]
    fn do_handshake_rejects_bad_state() {
        let cfg = echo_hello_cfg();
        let session = McpStdioSession::spawn("twice".into(), &cfg).expect("spawn ok");
        session.do_handshake().expect("first handshake");
        let err = session.do_handshake().expect_err("second must fail");
        assert!(matches!(err, McpSessionError::BadState { .. }));
        let msg = format!("{err}");
        assert!(msg.contains("unexpected state"));
        let _ = session.shutdown();
    }

    #[test]
    fn session_error_display_smoke() {
        let variants = [
            McpSessionError::Spawn(std::io::Error::other("nope")),
            McpSessionError::HandshakeTimeout {
                after: Duration::from_millis(50),
            },
            McpSessionError::HandshakeIo(std::io::Error::other("io")),
            McpSessionError::Shutdown(std::io::Error::other("kill")),
            McpSessionError::BadState {
                current: McpStdioState::Ready,
                expected: "initializing".into(),
            },
            McpSessionError::StdinAlreadyTaken,
            McpSessionError::RecvLineClosed,
            McpSessionError::WriteFailed(std::io::Error::other("pipe")),
        ];
        for err in variants {
            let msg = format!("{err}");
            assert!(!msg.is_empty(), "error must have a display");
            // std::error::Error::source should not panic.
            let _ = std::error::Error::source(&err);
        }
    }

    #[test]
    fn new_error_variants_display_have_expected_substrings() {
        assert!(format!("{}", McpSessionError::StdinAlreadyTaken)
            .contains("stdin handle already taken"));
        assert!(format!("{}", McpSessionError::RecvLineClosed).contains("stdout pipe closed"));
        let write_err = McpSessionError::WriteFailed(std::io::Error::other("boom"));
        let msg = format!("{write_err}");
        assert!(msg.contains("write_line failed"));
        assert!(msg.contains("boom"));
        // Source plumbing: WriteFailed surfaces the inner io::Error.
        assert!(std::error::Error::source(&write_err).is_some());
        assert!(std::error::Error::source(&McpSessionError::StdinAlreadyTaken).is_none());
        assert!(std::error::Error::source(&McpSessionError::RecvLineClosed).is_none());
    }

    #[test]
    fn stderr_tail_cap_constant_is_two_hundred() {
        assert_eq!(STDERR_TAIL_CAP, 200);
    }

    #[cfg(unix)]
    #[test]
    fn spawn_honors_workdir() {
        let tmp = tempfile::tempdir().expect("tmp");
        let cfg = McpStdioConfig {
            command: PathBuf::from("/bin/sh"),
            args: vec!["-c".into(), "pwd".into()],
            env: BTreeMap::new(),
            workdir: Some(tmp.path().to_path_buf()),
            init_timeout: Duration::from_secs(2),
        };
        let session = McpStdioSession::spawn("pwd".into(), &cfg).expect("spawn ok");
        session.do_handshake().expect("handshake ok");
        let _ = session.shutdown();
    }

    #[cfg(unix)]
    #[test]
    fn handshake_after_silent_child_exit_disconnects() {
        // Child exits immediately without writing anything. The stdout
        // reader thread closes its sender, surfacing as `Disconnected`.
        let cfg = McpStdioConfig {
            command: PathBuf::from("/bin/sh"),
            args: vec!["-c".into(), "exit 0".into()],
            env: BTreeMap::new(),
            workdir: None,
            init_timeout: Duration::from_secs(2),
        };
        let session = McpStdioSession::spawn("silent".into(), &cfg).expect("spawn ok");
        let err = session.do_handshake().expect_err("must error");
        assert!(matches!(err, McpSessionError::HandshakeIo(_)));
        assert!(matches!(session.state(), McpStdioState::Closed { .. }));
        let _ = session.shutdown();
    }

    #[cfg(unix)]
    #[test]
    fn shutdown_returns_cached_code_when_already_closed_via_handshake_failure() {
        let cfg = McpStdioConfig {
            command: PathBuf::from("/bin/sh"),
            args: vec!["-c".into(), "sleep 5".into()],
            env: BTreeMap::new(),
            workdir: None,
            init_timeout: Duration::from_millis(50),
        };
        let session = McpStdioSession::spawn("sleeper".into(), &cfg).expect("spawn ok");
        let _ = session.do_handshake();
        // First shutdown after handshake-timeout reaps the child.
        let first = session.shutdown().expect("shutdown ok");
        // Cached path on the second call.
        let second = session.shutdown().expect("idempotent");
        assert_eq!(first, second);
    }

    #[test]
    fn state_helper_returns_closed_when_no_change() {
        // The `state()` helper is exercised indirectly above; this is a
        // pure-data check that the variant returned for the canonical
        // `Closed { code: None }` round-trips through clone+eq.
        let s = McpStdioState::Closed { code: None };
        assert_eq!(s, s.clone());
    }

    #[cfg(unix)]
    fn sleep_forever_cfg() -> McpStdioConfig {
        McpStdioConfig {
            command: PathBuf::from("/bin/sh"),
            args: vec!["-c".into(), "sleep 30".into()],
            env: BTreeMap::new(),
            workdir: None,
            init_timeout: Duration::from_millis(100),
        }
    }

    #[cfg(unix)]
    #[test]
    fn take_stdin_is_one_shot() {
        let session = McpStdioSession::spawn("once".into(), &sleep_forever_cfg()).expect("spawn");
        assert!(session.take_stdin().is_ok(), "first take must succeed");
        let err = session.take_stdin().expect_err("second take must fail");
        assert!(matches!(err, McpSessionError::StdinAlreadyTaken));
        let _ = session.shutdown();
    }

    #[cfg(unix)]
    #[test]
    fn take_stdin_after_shutdown_then_taken_errors() {
        let session = McpStdioSession::spawn("post".into(), &sleep_forever_cfg()).expect("spawn");
        let _stdin = session.take_stdin().expect("first take");
        let _ = session.shutdown();
        // Stdin was moved out before shutdown; a second take after the
        // child is gone must report StdinAlreadyTaken.
        let err = session.take_stdin().expect_err("must error");
        assert!(matches!(err, McpSessionError::StdinAlreadyTaken));
    }

    #[cfg(unix)]
    #[test]
    fn recv_line_times_out_when_child_is_silent() {
        let session = McpStdioSession::spawn("quiet".into(), &sleep_forever_cfg()).expect("spawn");
        let line = session
            .recv_line(Duration::from_millis(50))
            .expect("timeout returns Ok(None)");
        assert!(line.is_none(), "expected no line, got {line:?}");
        let _ = session.shutdown();
    }

    #[cfg(unix)]
    #[test]
    fn recv_line_returns_some_after_child_writes() {
        let cfg = McpStdioConfig {
            command: PathBuf::from("/bin/sh"),
            args: vec!["-c".into(), "echo hello".into()],
            env: BTreeMap::new(),
            workdir: None,
            init_timeout: Duration::from_secs(2),
        };
        let session = McpStdioSession::spawn("hello".into(), &cfg).expect("spawn");
        let line = session
            .recv_line(Duration::from_secs(2))
            .expect("recv_line ok")
            .expect("some line");
        assert!(line.contains("hello"), "got {line:?}");
        let _ = session.shutdown();
    }

    #[cfg(unix)]
    #[test]
    fn pending_lines_drains_buffered_lines() {
        let cfg = McpStdioConfig {
            command: PathBuf::from("/bin/sh"),
            args: vec!["-c".into(), "echo one; echo two; echo three".into()],
            env: BTreeMap::new(),
            workdir: None,
            init_timeout: Duration::from_secs(2),
        };
        let session = McpStdioSession::spawn("three".into(), &cfg).expect("spawn");
        // Wait for the drain thread to land all three lines.
        std::thread::sleep(Duration::from_millis(200));
        let drained = session.pending_lines();
        assert!(
            drained.len() >= 3,
            "expected >=3 lines, got {}: {drained:?}",
            drained.len()
        );
        // Buffer is empty after drain.
        let again = session.pending_lines();
        assert!(
            again.is_empty(),
            "expected empty after drain, got {again:?}"
        );
        let _ = session.shutdown();
    }

    #[cfg(unix)]
    #[test]
    fn pending_lines_empty_when_buffer_empty() {
        let session = McpStdioSession::spawn("idle".into(), &sleep_forever_cfg()).expect("spawn");
        assert!(session.pending_lines().is_empty());
        let _ = session.shutdown();
    }

    #[cfg(unix)]
    #[test]
    fn is_alive_true_after_spawn() {
        let session = McpStdioSession::spawn("alive".into(), &sleep_forever_cfg()).expect("spawn");
        assert!(session.is_alive());
        let _ = session.shutdown();
    }

    #[cfg(unix)]
    #[test]
    fn is_alive_false_after_shutdown() {
        let session = McpStdioSession::spawn("dying".into(), &sleep_forever_cfg()).expect("spawn");
        let _ = session.shutdown();
        assert!(!session.is_alive());
    }

    #[cfg(unix)]
    #[test]
    fn write_line_succeeds_before_take_stdin() {
        // A `cat` child echoes whatever we write back out.
        let cfg = McpStdioConfig {
            command: PathBuf::from("/bin/sh"),
            args: vec!["-c".into(), "cat".into()],
            env: BTreeMap::new(),
            workdir: None,
            init_timeout: Duration::from_millis(100),
        };
        let session = McpStdioSession::spawn("cat".into(), &cfg).expect("spawn");
        session.write_line("hello").expect("write ok");
        let _ = session.shutdown();
    }

    #[cfg(unix)]
    #[test]
    fn write_line_errors_after_take_stdin() {
        let session = McpStdioSession::spawn("moved".into(), &sleep_forever_cfg()).expect("spawn");
        let _stdin = session.take_stdin().expect("take ok");
        let err = session.write_line("nope").expect_err("must error");
        assert!(matches!(err, McpSessionError::StdinAlreadyTaken));
        let _ = session.shutdown();
    }

    #[cfg(unix)]
    #[test]
    fn end_to_end_write_then_recv_echo() {
        // `cat` mirrors stdin to stdout line-by-line — the cheapest
        // imitation of a JSON-RPC peer that responds to every request.
        let cfg = McpStdioConfig {
            command: PathBuf::from("/bin/sh"),
            args: vec!["-c".into(), "cat".into()],
            env: BTreeMap::new(),
            workdir: None,
            init_timeout: Duration::from_secs(2),
        };
        let session = McpStdioSession::spawn("echo-e2e".into(), &cfg).expect("spawn");
        session.write_line("hi").expect("write ok");
        let line = session
            .recv_line(Duration::from_secs(2))
            .expect("recv_line ok")
            .expect("got line");
        assert!(line.contains("hi"), "expected echo of 'hi', got {line:?}");
        let _ = session.shutdown();
    }

    #[cfg(unix)]
    #[test]
    fn recv_line_returns_closed_after_child_exit() {
        // Child exits immediately with no output. The drain thread
        // closes the buffer; recv_line should return RecvLineClosed.
        let cfg = McpStdioConfig {
            command: PathBuf::from("/bin/sh"),
            args: vec!["-c".into(), "exit 0".into()],
            env: BTreeMap::new(),
            workdir: None,
            init_timeout: Duration::from_secs(2),
        };
        let session = McpStdioSession::spawn("departed".into(), &cfg).expect("spawn");
        // Let the drain thread observe EOF.
        std::thread::sleep(Duration::from_millis(150));
        let err = session
            .recv_line(Duration::from_millis(200))
            .expect_err("must error");
        assert!(matches!(err, McpSessionError::RecvLineClosed));
        let _ = session.shutdown();
    }

    #[test]
    fn stdio_transport_env_roundtrips() {
        let mut env = BTreeMap::new();
        env.insert("RAG_INDEX".to_owned(), "/var/rag".to_owned());
        env.insert("LOG".to_owned(), "info".to_owned());
        let cfg = McpServerConfig {
            name: "rag".into(),
            transport: McpTransport::Stdio {
                command: "stratum-mcp-rag".into(),
                args: vec![],
                env,
            },
            allow: vec!["search".into()],
            deny: vec![],
        };
        let json = serde_json::to_string(&cfg).expect("serialize");
        // BTreeMap sorts: LOG < RAG_INDEX.
        let log_idx = json.find("LOG").expect("LOG present");
        let rag_idx = json.find("RAG_INDEX").expect("RAG_INDEX present");
        assert!(log_idx < rag_idx);
        let back: McpServerConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(cfg, back);
    }
}
