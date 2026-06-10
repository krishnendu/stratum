//! Composable middleware layers wrapping a [`ServeHandler`].
//!
//! Phase 6 scaffold — sits between [`crate::serve_server`] (the socket
//! loop) and concrete handlers like
//! [`crate::serve_handler_agent::AgentServeHandler`]. Each layer takes an
//! `Arc<dyn ServeHandler>` and produces a new `Arc<dyn ServeHandler>`,
//! so layers compose via the [`chain`] helper:
//!
//! ```text
//! chain(echo, vec![auth_layer, rate_layer, log_layer])
//!   == Auth(Rate(Log(echo)))
//! ```
//!
//! ## Layers
//!
//! - [`RateLimitedHandler`] — wraps a [`crate::rate_limit::TokenBucket`].
//!   Rejects with [`SERVE_ERR_RATE_LIMITED`] when the bucket is empty.
//! - [`AuthTokenHandler`] — checks `req.params["auth"]` against an
//!   allow-list. Rejects with [`SERVE_ERR_UNAUTHORIZED`] on mismatch.
//!   JSON-RPC has no transport headers, so the convention is to embed
//!   the bearer in `params`. The `header_name` field is preserved as
//!   metadata for a future HTTP-fronted variant.
//! - [`LoggingHandler`] — emits a paired [`Event::AgentHandoff`] before
//!   and after the inner handler runs.
//!
//! ## Error code policy
//!
//! Both new sentinels stay inside the JSON-RPC reserved range
//! `[-32099, -32000]` (the "implementation-defined server errors"
//! sub-block), matching the convention already established in
//! [`crate::serve_protocol`]. They'll promote to `STRAT-E####` once
//! the `stratum serve` CLI surface stabilises.

// xtask-check-error-codes: ignore-file
//
// Reason: this module surfaces failures via local `SERVE_ERR_*`
// sentinels (mirroring the JSON-RPC 2.0 reserved range) rather than
// catalog `STRAT-E####` entries, matching `serve_protocol.rs`.

use std::collections::BTreeSet;
use std::fmt;
use std::sync::Arc;

use serde_json::Value;

use crate::event_log::{Event, EventEmitter};
use crate::rate_limit::{Clock, SystemClock, TokenBucket};
use crate::serve_protocol::{ServeRequest, ServeResponse, ServeResponseBody};
use crate::serve_server::ServeHandler;

/// Caller exceeded the configured rate limit for this socket.
///
/// Lives in the JSON-RPC "implementation-defined server error" range
/// (`-32099..=-32000`).
pub const SERVE_ERR_RATE_LIMITED: i32 = -32_006;

/// Caller failed bearer-token auth.
///
/// Lives in the JSON-RPC "implementation-defined server error" range
/// (`-32099..=-32000`).
pub const SERVE_ERR_UNAUTHORIZED: i32 = -32_007;

/// Default header name used by [`AuthTokenHandler`].
///
/// Since JSON-RPC has no concept of transport headers, the token is
/// actually read from `req.params["auth"]`. The header name is kept
/// as metadata so a future HTTP-fronted wrapper can use the same
/// constant.
pub const DEFAULT_AUTH_HEADER: &str = "Authorization";

/// Wraps a [`ServeHandler`] with a [`TokenBucket`] rate limiter.
///
/// Every inbound [`ServeRequest`] costs one token. When the bucket is
/// empty the request is rejected with [`SERVE_ERR_RATE_LIMITED`] and
/// the inner handler is **not** called.
pub struct RateLimitedHandler<C: Clock + 'static = SystemClock> {
    inner: Arc<dyn ServeHandler>,
    bucket: TokenBucket<C>,
}

impl RateLimitedHandler<SystemClock> {
    /// Build a [`RateLimitedHandler`] backed by the real monotonic clock.
    #[must_use]
    pub fn new(inner: Arc<dyn ServeHandler>, capacity: u32, refill_per_sec: f64) -> Self {
        Self {
            inner,
            bucket: TokenBucket::new(capacity, refill_per_sec),
        }
    }
}

impl<C: Clock + 'static> RateLimitedHandler<C> {
    /// Build a [`RateLimitedHandler`] with a caller-supplied clock.
    pub fn with_clock(
        inner: Arc<dyn ServeHandler>,
        capacity: u32,
        refill_per_sec: f64,
        clock: C,
    ) -> Self {
        Self {
            inner,
            bucket: TokenBucket::with_clock(capacity, refill_per_sec, clock),
        }
    }

    /// Tokens currently available — useful for asserting layer ordering
    /// in tests.
    #[must_use]
    pub fn available(&self) -> u32 {
        self.bucket.available()
    }
}

impl<C: Clock + 'static> fmt::Debug for RateLimitedHandler<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RateLimitedHandler")
            .field("available", &self.bucket.available())
            .finish_non_exhaustive()
    }
}

impl<C: Clock + 'static> ServeHandler for RateLimitedHandler<C> {
    fn handle(&self, req: ServeRequest) -> ServeResponse {
        if self.bucket.try_acquire(1).is_err() {
            return ServeResponse::err(req.id, SERVE_ERR_RATE_LIMITED, "rate limit exceeded");
        }
        self.inner.handle(req)
    }
}

/// Wraps a [`ServeHandler`] with a bearer-token allow-list.
///
/// The token is read from `req.params["auth"]` because JSON-RPC has no
/// native transport headers. [`header_name`](Self::header_name) is
/// metadata for a future HTTP-fronted wrapper.
pub struct AuthTokenHandler {
    inner: Arc<dyn ServeHandler>,
    tokens: BTreeSet<String>,
    header_name: String,
}

impl AuthTokenHandler {
    /// Build an [`AuthTokenHandler`] with the given token allow-list and
    /// the default header name ([`DEFAULT_AUTH_HEADER`]).
    #[must_use]
    pub fn new(inner: Arc<dyn ServeHandler>, tokens: BTreeSet<String>) -> Self {
        Self {
            inner,
            tokens,
            header_name: DEFAULT_AUTH_HEADER.to_string(),
        }
    }

    /// Builder: override the header name surfaced to a future HTTP
    /// front-end. Has no effect on the JSON-RPC path (which reads
    /// `params["auth"]`).
    #[must_use]
    pub fn with_header_name(mut self, name: impl Into<String>) -> Self {
        self.header_name = name.into();
        self
    }

    /// Currently configured header name.
    #[must_use]
    pub fn header_name(&self) -> &str {
        &self.header_name
    }
}

impl fmt::Debug for AuthTokenHandler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AuthTokenHandler")
            .field("tokens", &self.tokens.len())
            .field("header_name", &self.header_name)
            .finish_non_exhaustive()
    }
}

impl ServeHandler for AuthTokenHandler {
    fn handle(&self, req: ServeRequest) -> ServeResponse {
        let supplied = req
            .params
            .as_object()
            .and_then(|obj| obj.get("auth"))
            .and_then(Value::as_str);
        match supplied {
            Some(token) if self.tokens.contains(token) => self.inner.handle(req),
            _ => ServeResponse::err(req.id, SERVE_ERR_UNAUTHORIZED, "unauthorized"),
        }
    }
}

/// Wraps a [`ServeHandler`] with a paired [`Event::AgentHandoff`]
/// before/after every call.
///
/// We deliberately reuse [`Event::AgentHandoff`] because the event log
/// has no `Serve`-specific variant yet — adding one would touch the
/// catalog. The `from` field is always `"serve"`; the `to` field
/// carries the JSON-RPC method name; the `reason` field is `"incoming"`
/// on the pre-event and one of `"ok"` / `"err:<code>"` on the post-event.
pub struct LoggingHandler {
    inner: Arc<dyn ServeHandler>,
    events: Arc<EventEmitter>,
}

impl LoggingHandler {
    /// Build a [`LoggingHandler`] that emits to `events`.
    #[must_use]
    pub fn new(inner: Arc<dyn ServeHandler>, events: Arc<EventEmitter>) -> Self {
        Self { inner, events }
    }
}

impl fmt::Debug for LoggingHandler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LoggingHandler")
            .field("events", &self.events)
            .finish_non_exhaustive()
    }
}

impl ServeHandler for LoggingHandler {
    fn handle(&self, req: ServeRequest) -> ServeResponse {
        let method = req.method.as_str().to_string();
        self.events.emit(
            Event::AgentHandoff {
                from: "serve".to_string(),
                to: method.clone(),
                reason: "incoming".to_string(),
            },
            None,
        );
        let resp = self.inner.handle(req);
        let reason = match &resp.body {
            ServeResponseBody::Ok(_) => "ok".to_string(),
            ServeResponseBody::Err { code, .. } => format!("err:{code}"),
        };
        self.events.emit(
            Event::AgentHandoff {
                from: "serve".to_string(),
                to: method,
                reason,
            },
            None,
        );
        resp
    }
}

/// A boxed `Arc<dyn ServeHandler>` → `Arc<dyn ServeHandler>` adapter.
///
/// Used by [`chain`] to compose layers without naming concrete types.
pub type ServeLayer = Box<dyn Fn(Arc<dyn ServeHandler>) -> Arc<dyn ServeHandler>>;

/// Compose `layers` around `handler`.
///
/// Layers are applied **outermost-first**: the first element of `layers`
/// becomes the outermost wrapper that sees a request before any other
/// layer. Concretely, `chain(h, [a, b])` returns `a(b(h))` — `a` runs
/// first on the inbound path.
///
/// An empty `layers` vector returns `handler` unchanged.
#[must_use]
pub fn chain(handler: Arc<dyn ServeHandler>, layers: Vec<ServeLayer>) -> Arc<dyn ServeHandler> {
    let mut current = handler;
    for layer in layers.into_iter().rev() {
        current = layer(current);
    }
    current
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex as StdMutex;
    use std::time::{Duration, Instant};

    use serde_json::json;

    use crate::event_log::{EventEmitter, EventSink, MemoryEventSink};
    use crate::rate_limit::ManualClock;
    use crate::serve_protocol::{RequestId, ServeMethod};
    use crate::serve_server::EchoServeHandler;

    fn echo() -> Arc<dyn ServeHandler> {
        Arc::new(EchoServeHandler)
    }

    fn req(id: i64, method: &str, params: Value) -> ServeRequest {
        ServeRequest {
            id: RequestId::Num(id),
            method: ServeMethod::from_raw(method),
            params,
        }
    }

    fn err_code(resp: &ServeResponse) -> Option<i32> {
        match &resp.body {
            ServeResponseBody::Err { code, .. } => Some(*code),
            ServeResponseBody::Ok(_) => None,
        }
    }

    fn is_ok(resp: &ServeResponse) -> bool {
        matches!(resp.body, ServeResponseBody::Ok(_))
    }

    /// Counts how many times `handle` is invoked. Lets composed-layer
    /// tests prove the inner handler never ran.
    struct CountingHandler {
        calls: StdMutex<u32>,
    }

    impl CountingHandler {
        fn new() -> Self {
            Self {
                calls: StdMutex::new(0),
            }
        }

        fn calls(&self) -> u32 {
            self.calls.lock().map_or(0, |g| *g)
        }
    }

    impl ServeHandler for CountingHandler {
        fn handle(&self, req: ServeRequest) -> ServeResponse {
            if let Ok(mut g) = self.calls.lock() {
                *g += 1;
            }
            ServeResponse::ok(req.id, json!({"counted": true}))
        }
    }

    #[test]
    fn rate_limited_initial_budget_equals_capacity_and_allows_first_request() {
        let h = RateLimitedHandler::new(echo(), 3, 0.0);
        assert_eq!(h.available(), 3);
        let resp = h.handle(req(1, "ping", json!({})));
        assert!(is_ok(&resp));
        assert_eq!(h.available(), 2);
    }

    #[test]
    fn rate_limited_blocks_after_budget_exhausted() {
        let h = RateLimitedHandler::new(echo(), 2, 0.0);
        assert!(is_ok(&h.handle(req(1, "ping", json!({})))));
        assert!(is_ok(&h.handle(req(2, "ping", json!({})))));
        let resp = h.handle(req(3, "ping", json!({})));
        assert_eq!(err_code(&resp), Some(SERVE_ERR_RATE_LIMITED));
        assert_eq!(resp.id, RequestId::Num(3));
    }

    #[test]
    fn rate_limited_refills_over_time_with_manual_clock() {
        let clock = ManualClock::new(Instant::now());
        let h = RateLimitedHandler::with_clock(echo(), 2, 2.0, clock.clone());
        // Drain.
        assert!(is_ok(&h.handle(req(1, "ping", json!({})))));
        assert!(is_ok(&h.handle(req(2, "ping", json!({})))));
        assert_eq!(
            err_code(&h.handle(req(3, "ping", json!({})))),
            Some(SERVE_ERR_RATE_LIMITED)
        );
        // 1s * 2/s == 2 tokens refilled.
        clock.advance(Duration::from_secs(1));
        assert!(is_ok(&h.handle(req(4, "ping", json!({})))));
    }

    #[test]
    fn auth_token_empty_set_rejects_everything() {
        let h = AuthTokenHandler::new(echo(), BTreeSet::new());
        let resp = h.handle(req(1, "ping", json!({"auth": "anything"})));
        assert_eq!(err_code(&resp), Some(SERVE_ERR_UNAUTHORIZED));
    }

    #[test]
    fn auth_token_allows_matching_token() {
        let mut tokens = BTreeSet::new();
        tokens.insert("secret".to_string());
        let h = AuthTokenHandler::new(echo(), tokens);
        let resp = h.handle(req(1, "ping", json!({"auth": "secret"})));
        assert!(is_ok(&resp));
    }

    #[test]
    fn auth_token_rejects_mismatched_token() {
        let mut tokens = BTreeSet::new();
        tokens.insert("secret".to_string());
        let h = AuthTokenHandler::new(echo(), tokens);
        let resp = h.handle(req(1, "ping", json!({"auth": "nope"})));
        assert_eq!(err_code(&resp), Some(SERVE_ERR_UNAUTHORIZED));
    }

    #[test]
    fn auth_token_rejects_missing_auth_field() {
        let mut tokens = BTreeSet::new();
        tokens.insert("secret".to_string());
        let h = AuthTokenHandler::new(echo(), tokens);
        let resp = h.handle(req(1, "ping", json!({})));
        assert_eq!(err_code(&resp), Some(SERVE_ERR_UNAUTHORIZED));
    }

    #[test]
    fn auth_token_default_header_name_is_authorization() {
        let h = AuthTokenHandler::new(echo(), BTreeSet::new());
        assert_eq!(h.header_name(), DEFAULT_AUTH_HEADER);
        assert_eq!(h.header_name(), "Authorization");
    }

    #[test]
    fn auth_token_with_header_name_round_trip() {
        let h = AuthTokenHandler::new(echo(), BTreeSet::new()).with_header_name("X-Stratum-Auth");
        assert_eq!(h.header_name(), "X-Stratum-Auth");
    }

    #[test]
    fn logging_handler_emits_exactly_two_events_per_call() {
        let sink = Arc::new(MemoryEventSink::new());
        let emitter = Arc::new(EventEmitter::new(sink.clone() as Arc<dyn EventSink>));
        let h = LoggingHandler::new(echo(), emitter);
        let _ = h.handle(req(1, "ping", json!({})));
        assert_eq!(sink.snapshot().len(), 2);
    }

    #[test]
    fn logging_handler_events_reach_underlying_sink_with_correct_shape() {
        let sink = Arc::new(MemoryEventSink::new());
        let emitter = Arc::new(EventEmitter::new(sink.clone() as Arc<dyn EventSink>));
        let h = LoggingHandler::new(echo(), emitter);
        let _ = h.handle(req(7, "run_turn", json!({"prompt": "hi"})));
        let snap = sink.snapshot();
        assert_eq!(snap.len(), 2);
        match &snap[0].event {
            Event::AgentHandoff { from, to, reason } => {
                assert_eq!(from, "serve");
                assert_eq!(to, "run_turn");
                assert_eq!(reason, "incoming");
            }
            other => panic!("expected AgentHandoff, got {other:?}"),
        }
        match &snap[1].event {
            Event::AgentHandoff { from, to, reason } => {
                assert_eq!(from, "serve");
                assert_eq!(to, "run_turn");
                assert_eq!(reason, "ok");
            }
            other => panic!("expected AgentHandoff, got {other:?}"),
        }
    }

    #[test]
    fn logging_handler_records_err_code_on_inner_failure() {
        struct AlwaysFail;
        impl ServeHandler for AlwaysFail {
            fn handle(&self, req: ServeRequest) -> ServeResponse {
                ServeResponse::err(req.id, -32_099, "boom")
            }
        }
        let sink = Arc::new(MemoryEventSink::new());
        let emitter = Arc::new(EventEmitter::new(sink.clone() as Arc<dyn EventSink>));
        let h = LoggingHandler::new(Arc::new(AlwaysFail), emitter);
        let _ = h.handle(req(1, "ping", json!({})));
        let snap = sink.snapshot();
        assert_eq!(snap.len(), 2);
        match &snap[1].event {
            Event::AgentHandoff { reason, .. } => assert_eq!(reason, "err:-32099"),
            other => panic!("expected AgentHandoff, got {other:?}"),
        }
    }

    #[test]
    fn chain_composes_layers_outermost_first() {
        // Pre-event is emitted by the outer layer, then the auth check
        // runs, then echo. With layers = [Logging, Auth], a successful
        // request emits 2 events (Logging wraps Auth wraps echo).
        let sink = Arc::new(MemoryEventSink::new());
        let emitter = Arc::new(EventEmitter::new(sink.clone() as Arc<dyn EventSink>));
        let mut tokens = BTreeSet::new();
        tokens.insert("ok".to_string());
        let layers: Vec<ServeLayer> = vec![
            Box::new(move |inner| Arc::new(LoggingHandler::new(inner, emitter.clone()))),
            Box::new(move |inner| Arc::new(AuthTokenHandler::new(inner, tokens.clone()))),
        ];
        let composed = chain(echo(), layers);
        // Happy path: auth passes, echo replies, both log events fire.
        let resp = composed.handle(req(1, "ping", json!({"auth": "ok"})));
        assert!(is_ok(&resp));
        assert_eq!(sink.snapshot().len(), 2);
    }

    #[test]
    fn chain_with_zero_layers_returns_handler_unchanged() {
        let original = echo();
        let returned = chain(original.clone(), Vec::new());
        // Arc::ptr_eq confirms it's the same allocation.
        assert!(Arc::ptr_eq(&original, &returned));
    }

    #[test]
    fn composed_handler_happy_path_auth_rate_echo() {
        let mut tokens = BTreeSet::new();
        tokens.insert("ok".to_string());
        let layers: Vec<ServeLayer> = vec![
            {
                let tokens = tokens.clone();
                Box::new(move |inner| Arc::new(AuthTokenHandler::new(inner, tokens.clone())))
            },
            Box::new(|inner| Arc::new(RateLimitedHandler::new(inner, 5, 0.0))),
        ];
        let composed = chain(echo(), layers);
        let resp = composed.handle(req(1, "ping", json!({"auth": "ok"})));
        assert!(is_ok(&resp));
    }

    #[test]
    fn composed_handler_rejects_unauth_before_rate_limit_sees_it() {
        let counting = Arc::new(CountingHandler::new());
        let counting_dyn: Arc<dyn ServeHandler> = counting.clone();
        // Build the rate limiter manually so we can inspect tokens
        // after the run.
        let rate = Arc::new(RateLimitedHandler::new(counting_dyn, 2, 0.0));
        let rate_dyn: Arc<dyn ServeHandler> = rate.clone();
        let mut tokens = BTreeSet::new();
        tokens.insert("ok".to_string());
        let auth = Arc::new(AuthTokenHandler::new(rate_dyn, tokens));
        // Send an unauthorized request.
        let resp = auth.handle(req(1, "ping", json!({"auth": "nope"})));
        assert_eq!(err_code(&resp), Some(SERVE_ERR_UNAUTHORIZED));
        // The rate-limit bucket still has its full budget because the
        // request was rejected upstream.
        assert_eq!(rate.available(), 2);
        // And the innermost handler never saw the request.
        assert_eq!(counting.calls(), 0);
    }

    #[test]
    fn rate_limited_handler_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<RateLimitedHandler<SystemClock>>();
        assert_send_sync::<AuthTokenHandler>();
        assert_send_sync::<LoggingHandler>();
    }

    #[test]
    fn handlers_preserve_request_id_on_rejection() {
        // RateLimitedHandler.
        let h = RateLimitedHandler::new(echo(), 1, 0.0);
        let _ = h.handle(req(1, "ping", json!({})));
        let resp = h.handle(req(99, "ping", json!({})));
        assert_eq!(resp.id, RequestId::Num(99));
        // AuthTokenHandler.
        let h = AuthTokenHandler::new(echo(), BTreeSet::new());
        let resp = h.handle(req(7, "ping", json!({})));
        assert_eq!(resp.id, RequestId::Num(7));
    }

    #[test]
    fn serve_err_constants_fall_in_implementation_defined_range() {
        for code in [SERVE_ERR_RATE_LIMITED, SERVE_ERR_UNAUTHORIZED] {
            assert!(
                (-32_099..=-32_000).contains(&code),
                "code {code} out of JSON-RPC implementation-defined range"
            );
        }
        // And they aren't equal to each other.
        assert_ne!(SERVE_ERR_RATE_LIMITED, SERVE_ERR_UNAUTHORIZED);
    }

    #[test]
    fn rate_limited_with_clock_constructor_works() {
        let clock = ManualClock::new(Instant::now());
        let h = RateLimitedHandler::with_clock(echo(), 1, 0.0, clock);
        assert_eq!(h.available(), 1);
        assert!(is_ok(&h.handle(req(1, "ping", json!({})))));
        assert_eq!(
            err_code(&h.handle(req(2, "ping", json!({})))),
            Some(SERVE_ERR_RATE_LIMITED)
        );
    }

    #[test]
    fn auth_token_with_null_params_rejects() {
        let mut tokens = BTreeSet::new();
        tokens.insert("ok".to_string());
        let h = AuthTokenHandler::new(echo(), tokens);
        let resp = h.handle(req(1, "ping", Value::Null));
        assert_eq!(err_code(&resp), Some(SERVE_ERR_UNAUTHORIZED));
    }

    #[test]
    fn auth_token_with_non_string_auth_field_rejects() {
        let mut tokens = BTreeSet::new();
        tokens.insert("ok".to_string());
        let h = AuthTokenHandler::new(echo(), tokens);
        let resp = h.handle(req(1, "ping", json!({"auth": 42})));
        assert_eq!(err_code(&resp), Some(SERVE_ERR_UNAUTHORIZED));
    }

    #[test]
    fn debug_impls_smoke() {
        let rate = RateLimitedHandler::new(echo(), 4, 0.0);
        let s = format!("{rate:?}");
        assert!(s.contains("RateLimitedHandler"));
        assert!(s.contains("available"));

        let mut tokens = BTreeSet::new();
        tokens.insert("a".to_string());
        let auth = AuthTokenHandler::new(echo(), tokens).with_header_name("X-Auth");
        let s = format!("{auth:?}");
        assert!(s.contains("AuthTokenHandler"));
        assert!(s.contains("X-Auth"));

        let sink = Arc::new(MemoryEventSink::new());
        let emitter = Arc::new(EventEmitter::new(sink as Arc<dyn EventSink>));
        let logging = LoggingHandler::new(echo(), emitter);
        let s = format!("{logging:?}");
        assert!(s.contains("LoggingHandler"));
    }

    #[test]
    fn err_code_helper_returns_none_for_ok_response() {
        // Exercises the `Ok` arm of the test-local `err_code` helper.
        let resp = ServeResponse::ok(RequestId::Num(1), json!({}));
        assert!(err_code(&resp).is_none());
    }

    #[test]
    fn counting_handler_increments_on_call() {
        // Exercises CountingHandler's `handle` path on its own, so
        // future test additions don't lose coverage if the composed
        // test changes.
        let h = CountingHandler::new();
        let resp = h.handle(req(1, "ping", json!({})));
        assert!(is_ok(&resp));
        assert_eq!(h.calls(), 1);
    }
}
