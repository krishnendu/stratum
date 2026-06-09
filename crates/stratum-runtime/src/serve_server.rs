//! Synchronous `stratum serve` JSON-RPC dispatch server.
//!
//! Phase 6 scaffold — pairs with [`crate::serve_protocol`] (which owns
//! the wire-format types and `SERVE_ERR_*` constants) to give the
//! `stratum serve` CLI a real socket-listening backend. Single-process,
//! one-thread-per-connection, no async runtime. The acceptor polls a
//! shutdown flag on a 50 ms cadence so the server can be torn down
//! cleanly from tests and the CLI signal handler.
//!
//! ## Binding modes
//!
//! [`ServeBind`] covers both transports used by `stratum serve`:
//!
//! - [`ServeBind::UnixSocket`] for the default `~/.local/share/stratum/serve.sock`
//!   IPC path on Unix.
//! - [`ServeBind::TcpLoopback`] for the loopback-bound TCP fallback used
//!   on Windows and inside CI matrices where the temp dir doesn't allow
//!   abstract sockets. Port `0` asks the kernel for an ephemeral port —
//!   the actual address lives on [`ServeHandle::bound_address`].
//!
//! ## Handler trait
//!
//! [`ServeHandler`] is the single seam between the wire-level loop and
//! the agent runtime. The default implementation
//! ([`EchoServeHandler`] via [`make_default_handler`]) just echoes the
//! method back — wiring a real `AgentLoop`-backed handler lands in a
//! follow-up PR.
//!
//! ## Error code policy
//!
//! All `SERVE_ERR_*` numeric codes are defined in
//! [`crate::serve_protocol`]; this module only emits responses through
//! that surface and ships its own internal [`ServeError`] for
//! bind/accept failures that never cross the wire.

// xtask-check-error-codes: ignore-file
//
// Reason: this module surfaces failures via `serve_protocol`'s
// `SERVE_ERR_*` sentinels (mirroring JSON-RPC reserved codes) and a
// local `ServeError` enum. Both are scoped to the `stratum serve`
// scaffold and predate the catalog `STRAT-E####` entry for the daemon.
// No `STRAT-E####` literals appear in this file.

use std::fmt;
use std::io::{BufRead, BufReader, ErrorKind, Read, Write};
use std::net::{TcpListener, TcpStream};
#[cfg(unix)]
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::serve_protocol::{
    parse_request, render_response, RequestId, ServeRequest, ServeResponse, SERVE_ERR_PARSE,
};

/// How long an acceptor sleeps between non-blocking poll attempts.
const ACCEPT_POLL_INTERVAL: Duration = Duration::from_millis(50);

/// Configuration for a [`ServeServer`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ServeConfig {
    /// Where to listen.
    pub bind: ServeBind,
    /// Maximum concurrent connections accepted. Currently advisory — the
    /// per-connection thread model relies on the OS to throttle.
    pub max_connections: usize,
    /// Per-request read timeout applied to each accepted socket.
    pub request_timeout: Duration,
}

impl Default for ServeConfig {
    fn default() -> Self {
        Self {
            bind: ServeBind::TcpLoopback { port: 0 },
            max_connections: 16,
            request_timeout: Duration::from_secs(30),
        }
    }
}

/// Socket binding strategy for [`ServeServer`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ServeBind {
    /// Bind a Unix-domain socket at `path`.
    UnixSocket {
        /// Filesystem path of the listening socket.
        path: PathBuf,
    },
    /// Bind a TCP listener on `127.0.0.1:<port>`. Port `0` requests an
    /// ephemeral port; the resolved address is on
    /// [`ServeHandle::bound_address`].
    TcpLoopback {
        /// Loopback TCP port; `0` for ephemeral.
        port: u16,
    },
}

/// Dispatch surface implemented by callers — one method per inbound JSON-RPC line.
pub trait ServeHandler: Send + Sync {
    /// Handle one parsed [`ServeRequest`] and return its response.
    fn handle(&self, req: ServeRequest) -> ServeResponse;
}

/// Synchronous JSON-RPC server bound to either a Unix or TCP loopback socket.
pub struct ServeServer {
    cfg: ServeConfig,
    handler: Arc<dyn ServeHandler>,
    shutdown: Arc<AtomicBool>,
    bound_address: Mutex<Option<String>>,
}

impl fmt::Debug for ServeServer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let addr = self.bound_address.lock().ok().and_then(|g| g.clone());
        f.debug_struct("ServeServer")
            .field("cfg", &self.cfg)
            .field("bound_address", &addr)
            .field("shutdown", &self.shutdown.load(Ordering::Relaxed))
            .finish_non_exhaustive()
    }
}

impl ServeServer {
    /// Build a new [`ServeServer`] — no socket is bound yet.
    #[must_use]
    pub fn new(cfg: ServeConfig, handler: Arc<dyn ServeHandler>) -> Self {
        Self {
            cfg,
            handler,
            shutdown: Arc::new(AtomicBool::new(false)),
            bound_address: Mutex::new(None),
        }
    }

    /// Returns the resolved socket address once [`start`] has bound it.
    pub fn bound_address(&self) -> Option<String> {
        self.bound_address.lock().ok().and_then(|g| g.clone())
    }

    /// Start the acceptor loop on a dedicated thread.
    ///
    /// # Errors
    ///
    /// Returns [`ServeError::Bind`] when the underlying listener cannot
    /// be created, or [`ServeError::AlreadyStarted`] when called twice
    /// on the same server.
    pub fn start(self: Arc<Self>) -> Result<ServeHandle, ServeError> {
        {
            let guard = self
                .bound_address
                .lock()
                .map_err(|_| ServeError::AlreadyStarted)?;
            if guard.is_some() {
                return Err(ServeError::AlreadyStarted);
            }
        }
        match &self.cfg.bind.clone() {
            #[cfg(unix)]
            ServeBind::UnixSocket { path } => self.start_unix(path.clone()),
            #[cfg(not(unix))]
            ServeBind::UnixSocket { .. } => Err(ServeError::Bind(std::io::Error::new(
                ErrorKind::Unsupported,
                "unix-domain sockets are not supported on this platform",
            ))),
            ServeBind::TcpLoopback { port } => self.start_tcp(*port),
        }
    }

    fn start_tcp(self: Arc<Self>, port: u16) -> Result<ServeHandle, ServeError> {
        let listener = TcpListener::bind(("127.0.0.1", port)).map_err(ServeError::Bind)?;
        let bound = listener.local_addr().map_err(ServeError::Bind)?.to_string();
        listener.set_nonblocking(true).map_err(ServeError::Bind)?;
        if let Ok(mut guard) = self.bound_address.lock() {
            *guard = Some(bound.clone());
        }
        let shutdown = self.shutdown.clone();
        let handler = self.handler.clone();
        let timeout = self.cfg.request_timeout;
        let acceptor = thread::Builder::new()
            .name("stratum-serve-acceptor".to_string())
            .spawn(move || {
                while !shutdown.load(Ordering::Relaxed) {
                    match listener.accept() {
                        Ok((stream, _peer)) => {
                            let h = handler.clone();
                            let _ = thread::Builder::new()
                                .name("stratum-serve-conn".to_string())
                                .spawn(move || {
                                    handle_tcp_connection(stream, h.as_ref(), timeout);
                                });
                        }
                        Err(err) if err.kind() == ErrorKind::WouldBlock => {
                            thread::sleep(ACCEPT_POLL_INTERVAL);
                        }
                        Err(_) => {
                            thread::sleep(ACCEPT_POLL_INTERVAL);
                        }
                    }
                }
            })
            .map_err(ServeError::Bind)?;
        Ok(ServeHandle {
            acceptor: Some(acceptor),
            shutdown: self.shutdown.clone(),
            bound_address: bound,
            unix_socket_path: None,
        })
    }

    #[cfg(unix)]
    fn start_unix(self: Arc<Self>, path: PathBuf) -> Result<ServeHandle, ServeError> {
        // Best-effort: remove any stale socket file from a prior unclean shutdown.
        let _ = std::fs::remove_file(&path);
        let listener = UnixListener::bind(&path).map_err(ServeError::Bind)?;
        let bound = path.display().to_string();
        listener.set_nonblocking(true).map_err(ServeError::Bind)?;
        if let Ok(mut guard) = self.bound_address.lock() {
            *guard = Some(bound.clone());
        }
        let shutdown = self.shutdown.clone();
        let handler = self.handler.clone();
        let timeout = self.cfg.request_timeout;
        let acceptor = thread::Builder::new()
            .name("stratum-serve-acceptor".to_string())
            .spawn(move || {
                while !shutdown.load(Ordering::Relaxed) {
                    match listener.accept() {
                        Ok((stream, _peer)) => {
                            let h = handler.clone();
                            let _ = thread::Builder::new()
                                .name("stratum-serve-conn".to_string())
                                .spawn(move || {
                                    handle_unix_connection(stream, h.as_ref(), timeout);
                                });
                        }
                        Err(err) if err.kind() == ErrorKind::WouldBlock => {
                            thread::sleep(ACCEPT_POLL_INTERVAL);
                        }
                        Err(_) => {
                            thread::sleep(ACCEPT_POLL_INTERVAL);
                        }
                    }
                }
            })
            .map_err(ServeError::Bind)?;
        Ok(ServeHandle {
            acceptor: Some(acceptor),
            shutdown: self.shutdown.clone(),
            bound_address: bound,
            unix_socket_path: Some(path),
        })
    }
}

fn handle_tcp_connection(stream: TcpStream, handler: &dyn ServeHandler, timeout: Duration) {
    // Accepted sockets can inherit the listener's nonblocking flag on some
    // platforms; force blocking so `BufReader::read_line` doesn't surface
    // an immediate `WouldBlock`.
    let _ = stream.set_nonblocking(false);
    let _ = stream.set_read_timeout(Some(timeout));
    let _ = stream.set_write_timeout(Some(timeout));
    let Ok(writer) = stream.try_clone() else {
        return;
    };
    handle_connection(stream, writer, handler);
}

#[cfg(unix)]
fn handle_unix_connection(stream: UnixStream, handler: &dyn ServeHandler, timeout: Duration) {
    let _ = stream.set_nonblocking(false);
    let _ = stream.set_read_timeout(Some(timeout));
    let _ = stream.set_write_timeout(Some(timeout));
    let Ok(writer) = stream.try_clone() else {
        return;
    };
    handle_connection(stream, writer, handler);
}

fn handle_connection<R, W>(reader: R, mut writer: W, handler: &dyn ServeHandler)
where
    R: Read,
    W: Write,
{
    let mut buf = BufReader::new(reader);
    let mut line = String::new();
    loop {
        line.clear();
        match buf.read_line(&mut line) {
            Ok(0) | Err(_) => return,
            Ok(_) => {}
        }
        if line.trim().is_empty() {
            continue;
        }
        let resp = match parse_request(&line) {
            Ok(req) => handler.handle(req),
            Err(err) => ServeResponse::err(RequestId::Num(0), SERVE_ERR_PARSE, err.to_string()),
        };
        let mut payload = render_response(&resp);
        payload.push('\n');
        if writer.write_all(payload.as_bytes()).is_err() {
            return;
        }
        if writer.flush().is_err() {
            return;
        }
    }
}

/// RAII handle returned by [`ServeServer::start`].
#[derive(Debug)]
pub struct ServeHandle {
    acceptor: Option<JoinHandle<()>>,
    shutdown: Arc<AtomicBool>,
    bound_address: String,
    unix_socket_path: Option<PathBuf>,
}

impl ServeHandle {
    /// Resolved socket address (Unix path or `host:port`).
    #[must_use]
    pub fn bound_address(&self) -> &str {
        &self.bound_address
    }

    /// Signal shutdown and join the acceptor thread.
    ///
    /// Calling `stop` more than once is a no-op on the second call —
    /// the second call sees the join handle already taken.
    ///
    /// # Errors
    ///
    /// Surfaces any panic from the acceptor thread via
    /// [`std::thread::Result`].
    pub fn stop(mut self) -> std::thread::Result<()> {
        self.shutdown.store(true, Ordering::Relaxed);
        if let Some(handle) = self.acceptor.take() {
            handle.join()?;
        }
        if let Some(path) = self.unix_socket_path.take() {
            let _ = std::fs::remove_file(path);
        }
        Ok(())
    }
}

impl Drop for ServeHandle {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
        if let Some(handle) = self.acceptor.take() {
            let _ = handle.join();
        }
        if let Some(path) = self.unix_socket_path.take() {
            let _ = std::fs::remove_file(path);
        }
    }
}

/// Errors surfaced by [`ServeServer::start`].
#[derive(Debug)]
pub enum ServeError {
    /// Listener bind failed.
    Bind(std::io::Error),
    /// Acceptor returned an unrecoverable error.
    Accept(std::io::Error),
    /// Generic IO error.
    Io(std::io::Error),
    /// `start` was called twice on the same [`ServeServer`].
    AlreadyStarted,
}

impl fmt::Display for ServeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bind(err) => write!(f, "serve server bind failed: {err}"),
            Self::Accept(err) => write!(f, "serve server accept failed: {err}"),
            Self::Io(err) => write!(f, "serve server io error: {err}"),
            Self::AlreadyStarted => f.write_str("serve server already started"),
        }
    }
}

impl std::error::Error for ServeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Bind(err) | Self::Accept(err) | Self::Io(err) => Some(err),
            Self::AlreadyStarted => None,
        }
    }
}

/// Default echo handler used for tests and the smoke fixture in the CLI.
#[derive(Debug, Default, Clone, Copy)]
pub struct EchoServeHandler;

impl ServeHandler for EchoServeHandler {
    fn handle(&self, req: ServeRequest) -> ServeResponse {
        ServeResponse::ok(req.id, json!({"echo": req.method.as_str()}))
    }
}

/// Production handler factory — currently wraps [`EchoServeHandler`].
///
/// Wiring a real `AgentLoop`-backed handler lands in a follow-up PR.
#[must_use]
pub fn make_default_handler() -> Arc<dyn ServeHandler> {
    Arc::new(EchoServeHandler)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader, Write};
    use std::net::TcpStream;
    use std::sync::Mutex as StdMutex;
    use std::time::Instant;

    fn send_and_recv(addr: &str, line: &str) -> String {
        let mut stream = TcpStream::connect(addr).expect("connect");
        stream
            .set_read_timeout(Some(Duration::from_secs(2)))
            .expect("read timeout");
        stream.write_all(line.as_bytes()).expect("write");
        if !line.ends_with('\n') {
            stream.write_all(b"\n").expect("write nl");
        }
        let mut reader = BufReader::new(stream);
        let mut resp = String::new();
        reader.read_line(&mut resp).expect("read line");
        resp
    }

    fn start_echo() -> (Arc<ServeServer>, ServeHandle) {
        let cfg = ServeConfig {
            bind: ServeBind::TcpLoopback { port: 0 },
            max_connections: 16,
            request_timeout: Duration::from_secs(2),
        };
        let srv = Arc::new(ServeServer::new(cfg, make_default_handler()));
        let handle = srv.clone().start().expect("start");
        (srv, handle)
    }

    #[test]
    fn config_default_values() {
        let cfg = ServeConfig::default();
        assert_eq!(cfg.bind, ServeBind::TcpLoopback { port: 0 });
        assert_eq!(cfg.max_connections, 16);
        assert_eq!(cfg.request_timeout, Duration::from_secs(30));
    }

    #[test]
    fn serve_bind_serde_roundtrip_tcp() {
        let b = ServeBind::TcpLoopback { port: 1234 };
        let s = serde_json::to_string(&b).expect("ser");
        assert!(s.contains("tcp_loopback"));
        let back: ServeBind = serde_json::from_str(&s).expect("de");
        assert_eq!(b, back);
    }

    #[test]
    fn serve_bind_serde_roundtrip_unix() {
        let b = ServeBind::UnixSocket {
            path: PathBuf::from("/tmp/x.sock"),
        };
        let s = serde_json::to_string(&b).expect("ser");
        assert!(s.contains("unix_socket"));
        let back: ServeBind = serde_json::from_str(&s).expect("de");
        assert_eq!(b, back);
    }

    #[test]
    fn start_then_stop_succeeds() {
        let (_srv, handle) = start_echo();
        assert!(!handle.bound_address().is_empty());
        handle.stop().expect("stop");
    }

    #[test]
    fn ping_roundtrip() {
        let (srv, handle) = start_echo();
        let addr = srv.bound_address().expect("bound");
        let line = r#"{"jsonrpc":"2.0","id":1,"method":"ping","params":{}}"#;
        let resp = send_and_recv(&addr, line);
        let v: serde_json::Value = serde_json::from_str(&resp).expect("json");
        assert_eq!(v["jsonrpc"], "2.0");
        assert_eq!(v["id"], 1);
        assert_eq!(v["result"]["echo"], "ping");
        handle.stop().expect("stop");
    }

    #[test]
    fn malformed_json_returns_parse_error() {
        let (srv, handle) = start_echo();
        let addr = srv.bound_address().expect("bound");
        let resp = send_and_recv(&addr, "{not json");
        let v: serde_json::Value = serde_json::from_str(&resp).expect("json");
        assert_eq!(v["id"], 0);
        assert_eq!(v["error"]["code"], SERVE_ERR_PARSE);
        handle.stop().expect("stop");
    }

    #[test]
    fn bad_jsonrpc_version_returns_error() {
        let (srv, handle) = start_echo();
        let addr = srv.bound_address().expect("bound");
        let line = r#"{"jsonrpc":"1.0","id":1,"method":"ping"}"#;
        let resp = send_and_recv(&addr, line);
        let v: serde_json::Value = serde_json::from_str(&resp).expect("json");
        assert_eq!(v["error"]["code"], SERVE_ERR_PARSE);
        handle.stop().expect("stop");
    }

    #[test]
    fn multiple_sequential_requests_one_connection() {
        let (srv, handle) = start_echo();
        let addr = srv.bound_address().expect("bound");
        for i in 1..=5 {
            let line = format!(r#"{{"jsonrpc":"2.0","id":{i},"method":"ping","params":{{}}}}"#);
            let resp = send_and_recv(&addr, &line);
            let v: serde_json::Value = serde_json::from_str(&resp).expect("json");
            assert_eq!(v["id"], i);
        }
        handle.stop().expect("stop");
    }

    #[test]
    fn concurrent_clients_each_get_responses() {
        let (srv, handle) = start_echo();
        let addr = srv.bound_address().expect("bound");
        let errors: Arc<StdMutex<Vec<String>>> = Arc::new(StdMutex::new(Vec::new()));
        let mut joins = Vec::new();
        for c in 0..4 {
            let addr = addr.clone();
            let errors = errors.clone();
            joins.push(thread::spawn(move || {
                for i in 0..5 {
                    let id = c * 10 + i;
                    let line =
                        format!(r#"{{"jsonrpc":"2.0","id":{id},"method":"ping","params":{{}}}}"#);
                    let resp = send_and_recv(&addr, &line);
                    let v: serde_json::Value = match serde_json::from_str(&resp) {
                        Ok(v) => v,
                        Err(e) => {
                            errors.lock().expect("lock").push(format!("decode {e}"));
                            return;
                        }
                    };
                    if v["id"] != id {
                        errors
                            .lock()
                            .expect("lock")
                            .push(format!("id mismatch {} != {id}", v["id"]));
                    }
                }
            }));
        }
        for j in joins {
            j.join().expect("client thread");
        }
        let errs = errors.lock().expect("lock").clone();
        assert!(errs.is_empty(), "errors: {errs:?}");
        handle.stop().expect("stop");
    }

    #[test]
    fn bound_address_non_empty_after_start() {
        let (srv, handle) = start_echo();
        let addr = srv.bound_address().expect("bound");
        assert!(!addr.is_empty());
        assert!(!handle.bound_address().is_empty());
        handle.stop().expect("stop");
    }

    #[test]
    fn shutdown_refuses_new_connections_after_stop() {
        let (srv, handle) = start_echo();
        let addr = srv.bound_address().expect("bound");
        handle.stop().expect("stop");
        // Best-effort: after stop, either connect fails or the connection
        // is immediately closed with no response. Both outcomes prove the
        // acceptor is gone.
        let result = TcpStream::connect(&addr).and_then(|mut s| {
            s.set_read_timeout(Some(Duration::from_millis(300)))?;
            s.write_all(b"{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"ping\"}\n")?;
            let mut buf = [0u8; 64];
            s.read(&mut buf)
        });
        if let Ok(n) = result {
            assert_eq!(n, 0, "expected no response after stop");
        }
    }

    #[test]
    fn start_twice_returns_already_started() {
        let cfg = ServeConfig {
            bind: ServeBind::TcpLoopback { port: 0 },
            ..ServeConfig::default()
        };
        let srv = Arc::new(ServeServer::new(cfg, make_default_handler()));
        let h = srv.clone().start().expect("start");
        let err = srv.start().expect_err("second start");
        assert!(matches!(err, ServeError::AlreadyStarted));
        h.stop().expect("stop");
    }

    #[test]
    fn serve_error_display_smoke() {
        use std::error::Error;
        let e = ServeError::Bind(std::io::Error::other("x"));
        assert!(e.to_string().contains("bind"));
        let e = ServeError::Accept(std::io::Error::other("x"));
        assert!(e.to_string().contains("accept"));
        let e = ServeError::Io(std::io::Error::other("x"));
        assert!(e.to_string().contains("io"));
        let e = ServeError::AlreadyStarted;
        assert!(e.to_string().contains("already"));
        assert!(ServeError::Bind(std::io::Error::other("x"))
            .source()
            .is_some());
        assert!(ServeError::AlreadyStarted.source().is_none());
    }

    #[test]
    fn echo_handler_returns_method_echo() {
        let h = EchoServeHandler;
        let req = ServeRequest {
            id: RequestId::Num(7),
            method: crate::serve_protocol::ServeMethod::Ping,
            params: serde_json::Value::Null,
        };
        let resp = h.handle(req);
        assert_eq!(resp.id, RequestId::Num(7));
        if let crate::serve_protocol::ServeResponseBody::Ok(v) = resp.body {
            assert_eq!(v["echo"], "ping");
        } else {
            panic!("expected ok response");
        }
    }

    struct ScriptedHandler {
        replies: StdMutex<Vec<serde_json::Value>>,
    }

    impl ServeHandler for ScriptedHandler {
        fn handle(&self, req: ServeRequest) -> ServeResponse {
            let popped = {
                let mut q = self.replies.lock().expect("lock");
                q.pop()
            };
            let v = popped.unwrap_or_else(|| json!({"default": true}));
            ServeResponse::ok(req.id, v)
        }
    }

    #[test]
    fn scripted_handler_dispatches() {
        let handler = Arc::new(ScriptedHandler {
            replies: StdMutex::new(vec![json!({"k": "v"})]),
        });
        let cfg = ServeConfig {
            bind: ServeBind::TcpLoopback { port: 0 },
            max_connections: 4,
            request_timeout: Duration::from_secs(2),
        };
        let srv = Arc::new(ServeServer::new(cfg, handler));
        let handle = srv.clone().start().expect("start");
        let addr = srv.bound_address().expect("bound");
        let resp = send_and_recv(
            &addr,
            r#"{"jsonrpc":"2.0","id":99,"method":"custom","params":{}}"#,
        );
        let v: serde_json::Value = serde_json::from_str(&resp).expect("json");
        assert_eq!(v["id"], 99);
        assert_eq!(v["result"]["k"], "v");
        handle.stop().expect("stop");
    }

    #[test]
    fn read_timeout_drops_idle_connection() {
        let cfg = ServeConfig {
            bind: ServeBind::TcpLoopback { port: 0 },
            max_connections: 4,
            request_timeout: Duration::from_millis(200),
        };
        let srv = Arc::new(ServeServer::new(cfg, make_default_handler()));
        let handle = srv.clone().start().expect("start");
        let addr = srv.bound_address().expect("bound");
        let mut stream = TcpStream::connect(&addr).expect("connect");
        stream
            .set_read_timeout(Some(Duration::from_secs(2)))
            .expect("rto");
        let start = Instant::now();
        let mut buf = [0u8; 64];
        // Server-side read times out (~200ms), server closes -> client sees EOF (0).
        // We allow up to a 2s wall to cover slow CI.
        let n = stream.read(&mut buf).unwrap_or(0);
        assert_eq!(n, 0, "expected EOF from server-side timeout, got {n} bytes");
        assert!(
            start.elapsed() < Duration::from_secs(2),
            "took too long: {:?}",
            start.elapsed()
        );
        handle.stop().expect("stop");
    }

    #[test]
    fn server_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ServeServer>();
        assert_send_sync::<ServeHandle>();
        assert_send_sync::<EchoServeHandler>();
    }

    #[test]
    fn multiple_servers_concurrently_on_different_ports() {
        let mut handles = Vec::new();
        let mut servers = Vec::new();
        for _ in 0..3 {
            let cfg = ServeConfig {
                bind: ServeBind::TcpLoopback { port: 0 },
                max_connections: 4,
                request_timeout: Duration::from_secs(2),
            };
            let srv = Arc::new(ServeServer::new(cfg, make_default_handler()));
            let h = srv.clone().start().expect("start");
            servers.push(srv);
            handles.push(h);
        }
        // Each must have a unique non-empty address.
        let addrs: Vec<String> = servers
            .iter()
            .map(|s| s.bound_address().expect("bound"))
            .collect();
        for a in &addrs {
            assert!(!a.is_empty());
        }
        assert_eq!(
            addrs.iter().collect::<std::collections::HashSet<_>>().len(),
            3
        );
        // And each responds.
        for a in &addrs {
            let resp = send_and_recv(a, r#"{"jsonrpc":"2.0","id":1,"method":"ping"}"#);
            let v: serde_json::Value = serde_json::from_str(&resp).expect("json");
            assert_eq!(v["id"], 1);
        }
        for h in handles {
            h.stop().expect("stop");
        }
    }

    #[test]
    fn debug_impl_smoke() {
        let cfg = ServeConfig::default();
        let srv = ServeServer::new(cfg, make_default_handler());
        let s = format!("{srv:?}");
        assert!(s.contains("ServeServer"));
    }

    #[test]
    fn handle_drop_signals_shutdown_without_explicit_stop() {
        let (_srv, handle) = start_echo();
        let shutdown = handle.shutdown.clone();
        drop(handle);
        assert!(shutdown.load(Ordering::Relaxed));
    }

    #[cfg(unix)]
    #[test]
    fn unix_socket_bind_and_roundtrip() {
        use std::os::unix::net::UnixStream;
        let tmp = tempfile::tempdir().expect("tmp");
        let path = tmp.path().join("serve.sock");
        let cfg = ServeConfig {
            bind: ServeBind::UnixSocket { path: path.clone() },
            max_connections: 4,
            request_timeout: Duration::from_secs(2),
        };
        let srv = Arc::new(ServeServer::new(cfg, make_default_handler()));
        let handle = srv.start().expect("start");
        assert_eq!(handle.bound_address(), path.display().to_string());
        let mut stream = UnixStream::connect(&path).expect("connect");
        stream
            .set_read_timeout(Some(Duration::from_secs(2)))
            .expect("rto");
        stream
            .write_all(br#"{"jsonrpc":"2.0","id":42,"method":"ping","params":{}}"#)
            .expect("w");
        stream.write_all(b"\n").expect("nl");
        let mut reader = BufReader::new(stream);
        let mut resp = String::new();
        reader.read_line(&mut resp).expect("read");
        let v: serde_json::Value = serde_json::from_str(&resp).expect("json");
        assert_eq!(v["id"], 42);
        assert_eq!(v["result"]["echo"], "ping");
        handle.stop().expect("stop");
        // Socket file cleaned up.
        assert!(!path.exists());
    }

    #[test]
    fn bind_failure_returns_error_on_busy_port() {
        // Hold a port, then try to bind a second server on the same port.
        let occupant = TcpListener::bind(("127.0.0.1", 0)).expect("occupy");
        let port = occupant.local_addr().expect("addr").port();
        let cfg = ServeConfig {
            bind: ServeBind::TcpLoopback { port },
            max_connections: 1,
            request_timeout: Duration::from_secs(1),
        };
        let srv = Arc::new(ServeServer::new(cfg, make_default_handler()));
        let err = srv.start().expect_err("expect bind failure");
        assert!(matches!(err, ServeError::Bind(_)));
        drop(occupant);
    }

    #[test]
    fn handler_handles_other_method_through_socket() {
        let (srv, handle) = start_echo();
        let addr = srv.bound_address().expect("bound");
        let resp = send_and_recv(
            &addr,
            r#"{"jsonrpc":"2.0","id":1,"method":"unknown_method","params":{}}"#,
        );
        let v: serde_json::Value = serde_json::from_str(&resp).expect("json");
        assert_eq!(v["result"]["echo"], "unknown_method");
        handle.stop().expect("stop");
    }

    #[test]
    fn blank_lines_between_requests_are_skipped() {
        let (srv, handle) = start_echo();
        let addr = srv.bound_address().expect("bound");
        let mut stream = TcpStream::connect(&addr).expect("connect");
        stream
            .set_read_timeout(Some(Duration::from_secs(2)))
            .expect("rto");
        stream.write_all(b"\n\n").expect("blank");
        stream
            .write_all(br#"{"jsonrpc":"2.0","id":2,"method":"ping","params":{}}"#)
            .expect("payload");
        stream.write_all(b"\n").expect("nl");
        let mut reader = BufReader::new(stream);
        let mut resp = String::new();
        reader.read_line(&mut resp).expect("read");
        let v: serde_json::Value = serde_json::from_str(&resp).expect("json");
        assert_eq!(v["id"], 2);
        handle.stop().expect("stop");
    }

    #[cfg(unix)]
    #[test]
    fn unix_socket_cleaned_on_drop() {
        let tmp = tempfile::tempdir().expect("tmp");
        let path = tmp.path().join("drop.sock");
        let cfg = ServeConfig {
            bind: ServeBind::UnixSocket { path: path.clone() },
            max_connections: 4,
            request_timeout: Duration::from_secs(2),
        };
        let srv = Arc::new(ServeServer::new(cfg, make_default_handler()));
        let handle = srv.start().expect("start");
        assert!(path.exists());
        drop(handle);
        // Drop runs cleanup; give the OS a brief beat.
        let mut waited = Duration::ZERO;
        while path.exists() && waited < Duration::from_secs(1) {
            thread::sleep(Duration::from_millis(20));
            waited += Duration::from_millis(20);
        }
        assert!(!path.exists(), "socket file not cleaned");
    }
}
