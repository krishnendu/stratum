//! `AgentServeHandler` — first real [`ServeHandler`] implementation that
//! wires a [`crate::agent_session::AgentSession`] behind every JSON-RPC
//! `run_turn` request on the `stratum serve` daemon socket.
//!
//! ## Why this exists
//!
//! [`crate::serve_server::ServeServer`] (Phase 6) is wire-only — it just
//! parses one JSON-RPC line, hands the [`crate::serve_protocol::ServeRequest`]
//! to a [`crate::serve_server::ServeHandler`], and writes back the
//! response. Until this module landed the only implementation was the
//! stub [`crate::serve_server::EchoServeHandler`].
//!
//! `AgentServeHandler` composes existing types — it owns:
//!
//! * a shared [`crate::agent_factory::AgentFactory`] used to build one
//!   [`crate::agent_loop::AgentLoop`] per session on demand,
//! * an [`crate::transcript::TranscriptStore`] passed through to each
//!   session for atomic on-disk persistence,
//! * an [`crate::event_log::EventEmitter`] handed to each session,
//! * a session map keyed by [`crate::transcript::SessionId`].
//!
//! ## Method dispatch
//!
//! | [`ServeMethod`]            | Behavior                                                                 |
//! |---------------------------|--------------------------------------------------------------------------|
//! | [`ServeMethod::Ping`]      | Returns `{"now_ms": <unix-millis>}`.                                     |
//! | [`ServeMethod::RunTurn`]   | Parses [`RunTurnParams`]; runs one turn through the resolved session.    |
//! | [`ServeMethod::Cancel`]    | Cancels the session resolved from `params.session_id` or the request id. |
//! | `Other("health")`         | Returns `{"version", "uptime_secs", "ready"}`.                            |
//! | `Other("list_models")`    | Returns `{"slugs": []}` — placeholder until a `ModelCatalog` is injected.|
//! | `Other("stop")`           | Sets the internal shutdown flag; returns `{}`. The server outer loop     |
//! |                           | reads [`AgentServeHandler::is_shutdown_requested`] to break.             |
//! | Any other                 | Returns [`SERVE_ERR_INTERNAL`] `"not implemented"`.                       |
//!
//! ## Session id derivation
//!
//! [`RunTurnParams`] carries an optional `session_id` string. The handler
//! uses, in order of precedence:
//!
//! 1. `params.session_id` when it parses as a valid [`SessionId`].
//! 2. Otherwise the request id:
//!    * [`RequestId::Str(s)`] when `s` parses as a [`SessionId`].
//!    * Otherwise a stable derivation `format!("rpc-{}", n)` for the
//!      numeric variant, hashed into 16 lowercase hex chars.
//!
//! The derivation is documented here, fully deterministic, and covered
//! by tests so peers can predict which session they will be talking to
//! when they reuse a request id.

// xtask-check-error-codes: ignore-file
//
// Reason: this module routes errors through `serve_protocol`'s existing
// `SERVE_ERR_*` JSON-RPC sentinels and the local [`ServeHandlerError`]
// enum (which never crosses the wire). No `STRAT-E####` literals appear
// here — promotion to the catalog happens once the `stratum serve` CLI
// surface stabilizes alongside the rest of `serve_protocol`.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use serde::Serialize;
use serde_json::{json, Value};

use stratum_types::ModelId;

use crate::agent_factory::AgentFactory;
use crate::agent_session::{AgentSession, SessionError};
use crate::event_log::EventEmitter;
use crate::serve_protocol::{
    RequestId, RunTurnParams, ServeMethod, ServeRequest, ServeResponse, SERVE_ERR_INTERNAL,
    SERVE_ERR_PARAMS,
};
use crate::serve_server::ServeHandler;
use crate::transcript::{SessionId, TranscriptStore};

/// Default [`ModelId`] used for sessions when [`RunTurnParams`] does not
/// carry one. Matches the in-process `AgentFactory::echo` provider tag so
/// the scaffold path is exercised end-to-end in tests.
const DEFAULT_MODEL_TAG: &str = "echo";

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors raised by [`AgentServeHandler::get_or_create_session`].
///
/// These never cross the wire — handler-internal failures are mapped onto
/// [`SERVE_ERR_INTERNAL`] before being returned to the JSON-RPC peer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ServeHandlerError {
    /// Opening (or resuming) an [`AgentSession`] failed at the store layer.
    SessionOpen(String),
    /// [`AgentFactory::build`] surfaced an error.
    Factory(String),
}

impl fmt::Display for ServeHandlerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SessionOpen(msg) => write!(f, "agent serve handler: session open failed: {msg}"),
            Self::Factory(msg) => write!(f, "agent serve handler: factory build failed: {msg}"),
        }
    }
}

impl Error for ServeHandlerError {}

// ---------------------------------------------------------------------------
// RunTurnResult on-the-wire shape
// ---------------------------------------------------------------------------

/// JSON shape returned by a successful `run_turn` dispatch.
#[derive(Debug, Clone, Serialize)]
struct RunTurnResult {
    turn_id: u64,
    outcome: String,
    blocks: Value,
    session_id: String,
}

// ---------------------------------------------------------------------------
// AgentServeHandler
// ---------------------------------------------------------------------------

/// First production [`ServeHandler`] — wraps real
/// [`AgentSession`]s behind the `stratum serve` JSON-RPC socket.
///
/// See the module docs for the dispatch table.
pub struct AgentServeHandler {
    sessions: Mutex<BTreeMap<SessionId, Arc<AgentSession>>>,
    factory: Arc<AgentFactory>,
    store: Arc<TranscriptStore>,
    events: Arc<EventEmitter>,
    started_at: Instant,
    version: String,
    ready: AtomicBool,
    shutdown_requested: AtomicBool,
}

impl fmt::Debug for AgentServeHandler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let session_count = self.sessions.lock().ok().map_or(0, |g| g.len());
        f.debug_struct("AgentServeHandler")
            .field("version", &self.version)
            .field("ready", &self.ready.load(Ordering::Relaxed))
            .field(
                "shutdown_requested",
                &self.shutdown_requested.load(Ordering::Relaxed),
            )
            .field("session_count", &session_count)
            .finish_non_exhaustive()
    }
}

impl AgentServeHandler {
    /// Build a fresh handler. `ready` starts false; the caller flips it
    /// once the daemon has finished its warm-up.
    #[must_use]
    pub fn new(
        factory: Arc<AgentFactory>,
        store: Arc<TranscriptStore>,
        events: Arc<EventEmitter>,
        version: String,
    ) -> Self {
        Self {
            sessions: Mutex::new(BTreeMap::new()),
            factory,
            store,
            events,
            started_at: Instant::now(),
            version,
            ready: AtomicBool::new(false),
            shutdown_requested: AtomicBool::new(false),
        }
    }

    /// Flip the ready flag to `true`. Idempotent.
    pub fn mark_ready(&self) {
        self.ready.store(true, Ordering::Relaxed);
    }

    /// Returns `true` once [`AgentServeHandler::mark_ready`] has fired.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.ready.load(Ordering::Relaxed)
    }

    /// Read by the [`crate::serve_server::ServeServer`] outer loop so the
    /// `stop` JSON-RPC method can break cleanly.
    #[must_use]
    pub fn is_shutdown_requested(&self) -> bool {
        self.shutdown_requested.load(Ordering::Relaxed)
    }

    /// Build (or reuse) the session for `id`. Each newly built session
    /// gets its own [`crate::agent_loop::AgentLoop`] cloned from the
    /// factory — sessions never share a loop, so cancellation tokens are
    /// independent.
    ///
    /// # Errors
    ///
    /// * [`ServeHandlerError::Factory`] when [`AgentFactory::build`] fails.
    /// * [`ServeHandlerError::SessionOpen`] when [`AgentSession::open`]
    ///   fails at the transcript-store layer.
    pub fn get_or_create_session(
        &self,
        id: &SessionId,
        model: ModelId,
    ) -> Result<Arc<AgentSession>, ServeHandlerError> {
        let mut guard = self.sessions.lock().map_err(|_| {
            ServeHandlerError::SessionOpen("session map mutex poisoned".to_string())
        })?;
        if let Some(existing) = guard.get(id) {
            let cloned = existing.clone();
            drop(guard);
            return Ok(cloned);
        }
        let loop_ = (*self.factory)
            .clone()
            .build()
            .map_err(|e| ServeHandlerError::Factory(e.to_string()))?;
        let session = AgentSession::open(
            id.clone(),
            Arc::new(loop_),
            self.store.clone(),
            self.events.clone(),
            model,
        )
        .map_err(|e| ServeHandlerError::SessionOpen(e.to_string()))?;
        let arc = Arc::new(session);
        guard.insert(id.clone(), arc.clone());
        drop(guard);
        Ok(arc)
    }

    fn handle_ping(id: RequestId) -> ServeResponse {
        let raw = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0u128, |d| d.as_millis());
        let now_ms = u64::try_from(raw).unwrap_or(u64::MAX);
        ServeResponse::ok(id, json!({ "now_ms": now_ms }))
    }

    fn handle_health(&self, id: RequestId) -> ServeResponse {
        let uptime_secs = self.started_at.elapsed().as_secs();
        ServeResponse::ok(
            id,
            json!({
                "version": self.version,
                "uptime_secs": uptime_secs,
                "ready": self.is_ready(),
            }),
        )
    }

    fn handle_list_models(id: RequestId) -> ServeResponse {
        // TODO: inject a `ModelCatalog` so this returns real slugs.
        ServeResponse::ok(id, json!({ "slugs": [] }))
    }

    fn handle_stop(&self, id: RequestId) -> ServeResponse {
        self.shutdown_requested.store(true, Ordering::Relaxed);
        ServeResponse::ok(id, json!({}))
    }

    fn handle_run_turn(&self, id: RequestId, params: Value) -> ServeResponse {
        let parsed: RunTurnParams = match serde_json::from_value(params) {
            Ok(p) => p,
            Err(_) => return ServeResponse::err(id, SERVE_ERR_PARAMS, "bad params"),
        };
        let session_id = resolve_session_id(parsed.session_id.as_deref(), &id);
        let session =
            match self.get_or_create_session(&session_id, ModelId::from(DEFAULT_MODEL_TAG)) {
                Ok(s) => s,
                Err(e) => return ServeResponse::err(id, SERVE_ERR_INTERNAL, e.to_string()),
            };
        match session.next_turn(&parsed.prompt) {
            Ok(result) => {
                let body = RunTurnResult {
                    turn_id: result.turn_id.0,
                    outcome: format!("{:?}", result.outcome),
                    blocks: serde_json::to_value(&result.blocks).unwrap_or(Value::Null),
                    session_id: session_id.as_str().to_string(),
                };
                let body_value = serde_json::to_value(&body).unwrap_or(Value::Null);
                ServeResponse::ok(id, body_value)
            }
            Err(SessionError::Cancelled) => {
                ServeResponse::err(id, SERVE_ERR_INTERNAL, "session cancelled")
            }
            Err(other) => ServeResponse::err(id, SERVE_ERR_INTERNAL, other.to_string()),
        }
    }

    fn handle_cancel(&self, id: RequestId, params: &Value) -> ServeResponse {
        // `params` is best-effort: the caller may pass `{"session_id": "..."}`
        // or no body at all. When absent we fall back to the request id.
        let extracted = params
            .as_object()
            .and_then(|m| m.get("session_id"))
            .and_then(Value::as_str)
            .map(str::to_string);
        let session_id = resolve_session_id(extracted.as_deref(), &id);
        if let Ok(guard) = self.sessions.lock() {
            if let Some(session) = guard.get(&session_id) {
                session.cancel();
            }
        }
        ServeResponse::ok(id, json!({}))
    }
}

impl ServeHandler for AgentServeHandler {
    fn handle(&self, req: ServeRequest) -> ServeResponse {
        match &req.method {
            ServeMethod::Ping => Self::handle_ping(req.id),
            ServeMethod::RunTurn => self.handle_run_turn(req.id, req.params),
            ServeMethod::Cancel => self.handle_cancel(req.id, &req.params),
            ServeMethod::Other(name) => match name.as_str() {
                "health" => self.handle_health(req.id),
                "list_models" => Self::handle_list_models(req.id),
                "stop" => self.handle_stop(req.id),
                _ => ServeResponse::err(req.id, SERVE_ERR_INTERNAL, "not implemented"),
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Resolve the [`SessionId`] for an incoming request.
///
/// Precedence:
///
/// 1. `explicit` (the param's `session_id`) when it parses cleanly.
/// 2. The request id, mapped per the rule documented at the module level:
///    * [`RequestId::Str`] when the inner string is a valid session id.
///    * Numeric and otherwise: `format!("rpc-{n}")` → fold into 16
///      lowercase hex chars via [`fold_to_session_hex`].
fn resolve_session_id(explicit: Option<&str>, req_id: &RequestId) -> SessionId {
    if let Some(raw) = explicit {
        if let Ok(sid) = SessionId::from_str(raw) {
            return sid;
        }
    }
    if let RequestId::Str(s) = req_id {
        if let Ok(sid) = SessionId::from_str(s) {
            return sid;
        }
    }
    let seed = match req_id {
        RequestId::Num(n) => format!("rpc-{n}"),
        RequestId::Str(s) => format!("rpc-str-{s}"),
        RequestId::Null => "rpc-null".to_string(),
    };
    SessionId::from_str(&fold_to_session_hex(&seed)).unwrap_or_else(|_| SessionId::new_random())
}

/// Fold an arbitrary string into 16 lowercase hex chars via a
/// dependency-free FNV-1a 64-bit hash. Deterministic.
fn fold_to_session_hex(seed: &str) -> String {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in seed.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
    }
    let bytes = hash.to_be_bytes();
    let mut out = String::with_capacity(16);
    for byte in bytes {
        let hi = (byte >> 4) & 0xF;
        let lo = byte & 0xF;
        out.push(hex_digit(hi));
        out.push(hex_digit(lo));
    }
    out
}

const fn hex_digit(n: u8) -> char {
    match n {
        0 => '0',
        1 => '1',
        2 => '2',
        3 => '3',
        4 => '4',
        5 => '5',
        6 => '6',
        7 => '7',
        8 => '8',
        9 => '9',
        10 => 'a',
        11 => 'b',
        12 => 'c',
        13 => 'd',
        14 => 'e',
        _ => 'f',
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::agent_factory::AgentFactory;
    use crate::event_log::{EventEmitter, MemoryEventSink};
    use crate::provider::EchoProvider;
    use crate::serve_protocol::{ServeMethod, ServeRequest, ServeResponseBody};
    use crate::transcript::TranscriptStore;
    use std::sync::Arc;
    use std::thread;
    use tempfile::TempDir;

    fn fresh_handler() -> (AgentServeHandler, TempDir) {
        let tmp = TempDir::new().unwrap();
        let store = Arc::new(TranscriptStore::open(tmp.path().to_path_buf()).unwrap());
        let events = Arc::new(EventEmitter::new(Arc::new(MemoryEventSink::new())));
        let factory = Arc::new(AgentFactory::new().with_provider(Arc::new(EchoProvider::new(""))));
        let handler = AgentServeHandler::new(factory, store, events, "0.0.0-test".to_string());
        (handler, tmp)
    }

    fn req(id: i64, method: ServeMethod, params: Value) -> ServeRequest {
        ServeRequest {
            id: RequestId::Num(id),
            method,
            params,
        }
    }

    fn ok_body(resp: &ServeResponse) -> Value {
        match &resp.body {
            ServeResponseBody::Ok(v) => v.clone(),
            ServeResponseBody::Err { code, message } => {
                panic!("expected ok, got err {code} {message}")
            }
        }
    }

    fn err_code(resp: &ServeResponse) -> i32 {
        match &resp.body {
            ServeResponseBody::Err { code, .. } => *code,
            ServeResponseBody::Ok(v) => panic!("expected err, got ok {v}"),
        }
    }

    #[test]
    fn new_starts_with_ready_false() {
        let (h, _tmp) = fresh_handler();
        assert!(!h.is_ready());
    }

    #[test]
    fn mark_ready_flips_ready_true() {
        let (h, _tmp) = fresh_handler();
        h.mark_ready();
        assert!(h.is_ready());
        // Idempotent.
        h.mark_ready();
        assert!(h.is_ready());
    }

    #[test]
    fn is_shutdown_requested_initially_false() {
        let (h, _tmp) = fresh_handler();
        assert!(!h.is_shutdown_requested());
    }

    #[test]
    fn handle_ping_returns_now_ms() {
        let (h, _tmp) = fresh_handler();
        let resp = h.handle(req(1, ServeMethod::Ping, Value::Null));
        assert_eq!(resp.id, RequestId::Num(1));
        let body = ok_body(&resp);
        assert!(body.get("now_ms").and_then(Value::as_u64).is_some());
    }

    #[test]
    fn handle_health_returns_version_uptime_ready() {
        let (h, _tmp) = fresh_handler();
        let resp = h.handle(req(
            2,
            ServeMethod::Other("health".to_string()),
            Value::Null,
        ));
        let body = ok_body(&resp);
        assert_eq!(body["version"], "0.0.0-test");
        assert!(body.get("uptime_secs").and_then(Value::as_u64).is_some());
        assert_eq!(body["ready"], false);
    }

    #[test]
    fn handle_health_reflects_post_mark_ready() {
        let (h, _tmp) = fresh_handler();
        h.mark_ready();
        let resp = h.handle(req(
            3,
            ServeMethod::Other("health".to_string()),
            Value::Null,
        ));
        let body = ok_body(&resp);
        assert_eq!(body["ready"], true);
    }

    #[test]
    fn handle_list_models_returns_empty_slugs() {
        let (h, _tmp) = fresh_handler();
        let resp = h.handle(req(
            4,
            ServeMethod::Other("list_models".to_string()),
            Value::Null,
        ));
        let body = ok_body(&resp);
        assert_eq!(body["slugs"], json!([]));
    }

    #[test]
    fn handle_run_turn_valid_params_returns_success() {
        let (h, _tmp) = fresh_handler();
        let params = json!({"prompt": "hello"});
        let resp = h.handle(req(10, ServeMethod::RunTurn, params));
        let body = ok_body(&resp);
        assert!(body.get("turn_id").is_some());
        assert!(body.get("blocks").is_some());
        assert!(body.get("session_id").is_some());
    }

    #[test]
    fn handle_run_turn_malformed_params_returns_invalid_params() {
        let (h, _tmp) = fresh_handler();
        // `prompt` is required by RunTurnParams.
        let params = json!({"nope": true});
        let resp = h.handle(req(11, ServeMethod::RunTurn, params));
        assert_eq!(err_code(&resp), SERVE_ERR_PARAMS);
    }

    #[test]
    fn handle_run_turn_with_echo_provider_outcome_is_success() {
        let (h, _tmp) = fresh_handler();
        let params = json!({"prompt": "hello"});
        let resp = h.handle(req(12, ServeMethod::RunTurn, params));
        let body = ok_body(&resp);
        assert_eq!(body["outcome"], "Success");
    }

    #[test]
    fn handle_run_turn_reuses_session_for_same_request_id() {
        let (h, _tmp) = fresh_handler();
        let params = json!({"prompt": "one"});
        let r1 = h.handle(req(20, ServeMethod::RunTurn, params.clone()));
        let r2 = h.handle(req(20, ServeMethod::RunTurn, params));
        let b1 = ok_body(&r1);
        let b2 = ok_body(&r2);
        // Same derived session id.
        assert_eq!(b1["session_id"], b2["session_id"]);
        // Turn id increments inside the same session.
        let t1 = b1["turn_id"].as_u64().unwrap();
        let t2 = b2["turn_id"].as_u64().unwrap();
        assert_eq!(t2, t1 + 1);
    }

    #[test]
    fn handle_cancel_returns_ok_empty() {
        let (h, _tmp) = fresh_handler();
        // First create a session.
        let _ = h.handle(req(30, ServeMethod::RunTurn, json!({"prompt": "x"})));
        // Now cancel via the same request id.
        let resp = h.handle(req(30, ServeMethod::Cancel, Value::Null));
        let body = ok_body(&resp);
        assert_eq!(body, json!({}));
    }

    #[test]
    fn handle_cancel_then_run_turn_returns_error() {
        let (h, _tmp) = fresh_handler();
        let _ = h.handle(req(31, ServeMethod::RunTurn, json!({"prompt": "x"})));
        let _ = h.handle(req(31, ServeMethod::Cancel, Value::Null));
        let resp = h.handle(req(31, ServeMethod::RunTurn, json!({"prompt": "y"})));
        assert_eq!(err_code(&resp), SERVE_ERR_INTERNAL);
    }

    #[test]
    fn handle_stop_flips_shutdown_requested() {
        let (h, _tmp) = fresh_handler();
        assert!(!h.is_shutdown_requested());
        let resp = h.handle(req(40, ServeMethod::Other("stop".to_string()), Value::Null));
        let body = ok_body(&resp);
        assert_eq!(body, json!({}));
        assert!(h.is_shutdown_requested());
    }

    #[test]
    fn handle_unknown_method_returns_internal_not_implemented() {
        let (h, _tmp) = fresh_handler();
        let resp = h.handle(req(
            50,
            ServeMethod::Other("agent_loop_config_get".to_string()),
            Value::Null,
        ));
        assert_eq!(err_code(&resp), SERVE_ERR_INTERNAL);
        if let ServeResponseBody::Err { message, .. } = &resp.body {
            assert!(message.contains("not implemented"));
        }
    }

    #[test]
    fn response_id_matches_request_id_for_every_method() {
        let (h, _tmp) = fresh_handler();
        for (method, params) in [
            (ServeMethod::Ping, Value::Null),
            (ServeMethod::RunTurn, json!({"prompt": "x"})),
            (ServeMethod::Cancel, Value::Null),
            (ServeMethod::Other("health".to_string()), Value::Null),
            (ServeMethod::Other("list_models".to_string()), Value::Null),
            (ServeMethod::Other("stop".to_string()), Value::Null),
            (ServeMethod::Other("nope".to_string()), Value::Null),
        ] {
            let resp = h.handle(req(77, method, params));
            assert_eq!(resp.id, RequestId::Num(77));
        }
    }

    #[test]
    fn serve_handler_error_display_smoke() {
        let a = ServeHandlerError::SessionOpen("io".to_string());
        assert!(a.to_string().contains("session open failed"));
        let b = ServeHandlerError::Factory("missing".to_string());
        assert!(b.to_string().contains("factory build failed"));
    }

    #[test]
    fn concurrent_ping_all_succeed() {
        let (h, _tmp) = fresh_handler();
        let h = Arc::new(h);
        let mut handles = Vec::new();
        for tid in 0..4 {
            let h = h.clone();
            handles.push(thread::spawn(move || {
                let mut oks = 0;
                for i in 0..5 {
                    let resp = h.handle(req(tid * 100 + i, ServeMethod::Ping, Value::Null));
                    if matches!(resp.body, ServeResponseBody::Ok(_)) {
                        oks += 1;
                    }
                }
                oks
            }));
        }
        let mut total = 0;
        for j in handles {
            total += j.join().unwrap();
        }
        assert_eq!(total, 20);
    }

    #[test]
    fn get_or_create_session_returns_same_arc_for_same_id() {
        let (h, _tmp) = fresh_handler();
        let id = SessionId::from_str("0123456789abcdef").unwrap();
        let a = h.get_or_create_session(&id, ModelId::from("echo")).unwrap();
        let b = h.get_or_create_session(&id, ModelId::from("echo")).unwrap();
        assert!(Arc::ptr_eq(&a, &b));
    }

    #[test]
    fn get_or_create_session_distinct_ids_yield_distinct_sessions() {
        let (h, _tmp) = fresh_handler();
        let id_a = SessionId::from_str("aaaaaaaaaaaaaaaa").unwrap();
        let id_b = SessionId::from_str("bbbbbbbbbbbbbbbb").unwrap();
        let a = h
            .get_or_create_session(&id_a, ModelId::from("echo"))
            .unwrap();
        let b = h
            .get_or_create_session(&id_b, ModelId::from("echo"))
            .unwrap();
        assert!(!Arc::ptr_eq(&a, &b));
    }

    #[test]
    fn get_or_create_session_factory_error_propagates() {
        let tmp = TempDir::new().unwrap();
        let store = Arc::new(TranscriptStore::open(tmp.path().to_path_buf()).unwrap());
        let events = Arc::new(EventEmitter::new(Arc::new(MemoryEventSink::new())));
        // No provider on the factory → build must fail with MissingProvider.
        let factory = Arc::new(AgentFactory::new());
        let h = AgentServeHandler::new(factory, store, events, "v".to_string());
        let id = SessionId::from_str("0123456789abcdef").unwrap();
        let err = h
            .get_or_create_session(&id, ModelId::from("echo"))
            .unwrap_err();
        assert!(matches!(err, ServeHandlerError::Factory(_)));
        assert!(err.to_string().contains("factory build failed"));
    }

    #[test]
    fn resolve_session_id_prefers_explicit_param() {
        let explicit = "0123456789abcdef";
        let sid = resolve_session_id(Some(explicit), &RequestId::Num(99));
        assert_eq!(sid.as_str(), explicit);
    }

    #[test]
    fn resolve_session_id_falls_back_to_string_request_id() {
        let sid = resolve_session_id(None, &RequestId::Str("aaaaaaaaaaaaaaaa".to_string()));
        assert_eq!(sid.as_str(), "aaaaaaaaaaaaaaaa");
    }

    #[test]
    fn resolve_session_id_derives_from_numeric_id_is_deterministic() {
        let a = resolve_session_id(None, &RequestId::Num(7));
        let b = resolve_session_id(None, &RequestId::Num(7));
        assert_eq!(a.as_str(), b.as_str());
        assert_eq!(a.as_str().len(), 16);
    }

    #[test]
    fn resolve_session_id_derives_from_null_id_is_deterministic() {
        let a = resolve_session_id(None, &RequestId::Null);
        let b = resolve_session_id(None, &RequestId::Null);
        assert_eq!(a.as_str(), b.as_str());
    }

    #[test]
    fn resolve_session_id_invalid_explicit_falls_through() {
        // "not-hex!" is not a valid SessionId; we fall through to the
        // string request id which IS valid.
        let sid = resolve_session_id(
            Some("not-hex!"),
            &RequestId::Str("ffffffffeeeeeeee".to_string()),
        );
        assert_eq!(sid.as_str(), "ffffffffeeeeeeee");
    }

    #[test]
    fn fold_to_session_hex_yields_16_chars() {
        let h = fold_to_session_hex("hello");
        assert_eq!(h.len(), 16);
        assert!(h
            .chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_uppercase()));
    }

    #[test]
    fn handler_debug_smoke() {
        let (h, _tmp) = fresh_handler();
        let rendered = format!("{h:?}");
        assert!(rendered.contains("AgentServeHandler"));
        assert!(rendered.contains("ready"));
    }
}
