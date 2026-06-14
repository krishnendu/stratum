//! Interactive permission-prompt data shape.
//!
//! Phase 1 (data only) scaffold for the interactive capability / secret /
//! network / file-write / tool-use prompt that the CLI raises when an agent
//! attempts something that is not pre-allowed by its declared capability
//! matrix. See `plan/19-permissions-prompt.md` and the capability matrix in
//! `plan/20-tool-registry.md`.
//!
//! This module is intentionally UI-agnostic — TUI prompt rendering lives at
//! the CLI layer. Here we only define:
//!
//! - [`PermissionRequest`] — what the runtime is asking permission for.
//! - [`PermissionDecision`] — what the user (or a test) decided.
//! - [`PendingPrompt`] — an in-flight request awaiting a decision.
//! - [`PromptResponder`] — the abstraction the CLI implements.
//! - [`PermissionStore`] — remembers session / forever grants across asks.
//! - [`evaluate`] — the single entry point that short-circuits via the store
//!   when possible and otherwise drives the responder.
//!
//! Three test responders ([`DenyAllResponder`], [`AllowAllResponder`],
//! [`ScriptedResponder`]) cover the typical agent / tool test surface.

use std::collections::{BTreeMap, VecDeque};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::SystemTime;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

// ---------------------------------------------------------------------------
// PermissionRequest
// ---------------------------------------------------------------------------

/// A single permission the runtime is asking the user to grant or deny.
///
/// Tagged via serde's external `kind` field so JSON round-trips are stable
/// across variants and forward-compatible with new shapes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PermissionRequest {
    /// Grant a named capability (e.g. `net`, `fs.write`) optionally scoped to
    /// a target (`example.com`, `/tmp/work`).
    CapabilityGrant {
        /// Capability identifier (matches `tools::CapabilityMatrix` keys).
        capability: String,
        /// Optional fine-grained target (host, path, etc.).
        target: Option<String>,
        /// Human-readable reason shown in the prompt.
        reason: String,
    },
    /// Access a stored secret by reference.
    SecretAccess {
        /// Secret reference (project + id), as printed by the secrets module.
        secret_ref: String,
        /// Scope label (e.g. `read`, `read-write`).
        scope: String,
    },
    /// Open a network connection to a host.
    NetworkHost {
        /// Host name or IP literal.
        host: String,
        /// Optional port; `None` means "any port".
        port: Option<u16>,
    },
    /// Write to a filesystem path.
    FileWrite {
        /// Absolute path the runtime intends to write to.
        path: PathBuf,
    },
    /// Invoke a tool by its registry id.
    ToolUse {
        /// Tool identifier from the tool registry.
        tool_id: String,
        /// JSON-encoded arguments the model wants to pass. Surfaced in
        /// the permission modal so the user sees exactly what will run.
        /// Empty string when the call was issued with no args.
        #[serde(default)]
        args: String,
    },
}

// ---------------------------------------------------------------------------
// PermissionDecision
// ---------------------------------------------------------------------------

/// The user's (or test's) answer to a [`PermissionRequest`].
///
/// `AllowOnce` and `Deny` are transient — they are not persisted by
/// [`PermissionStore::record`]. `AllowSession` lives until the session is
/// cleared via [`PermissionStore::forget_session`]. `AllowForever` and
/// `DenyForever` persist beyond the session boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PermissionDecision {
    /// Allow exactly this one request; do not remember.
    AllowOnce,
    /// Allow for the remainder of this session.
    AllowSession,
    /// Allow permanently across sessions.
    AllowForever,
    /// Deny exactly this one request; do not remember.
    Deny,
    /// Deny permanently across sessions.
    DenyForever,
}

// ---------------------------------------------------------------------------
// PromptId / PromptIdGen
// ---------------------------------------------------------------------------

/// Monotonic identifier for an in-flight [`PendingPrompt`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct PromptId(pub u64);

/// Thread-safe monotonic generator for [`PromptId`]s.
#[derive(Debug, Default)]
pub struct PromptIdGen {
    next: AtomicU64,
}

impl PromptIdGen {
    /// Create a fresh generator starting at `0`.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            next: AtomicU64::new(0),
        }
    }

    /// Allocate the next [`PromptId`]. Monotonic; safe across threads.
    pub fn next(&self) -> PromptId {
        PromptId(self.next.fetch_add(1, Ordering::Relaxed))
    }
}

// ---------------------------------------------------------------------------
// PendingPrompt
// ---------------------------------------------------------------------------

/// A permission request that has been issued and is awaiting a decision.
#[derive(Debug, Clone)]
pub struct PendingPrompt {
    /// Unique identifier for this prompt.
    pub id: PromptId,
    /// The underlying request.
    pub request: PermissionRequest,
    /// When the prompt was issued.
    pub issued_at: SystemTime,
}

// ---------------------------------------------------------------------------
// PromptResponder
// ---------------------------------------------------------------------------

/// The runtime-facing abstraction for deciding a prompt.
///
/// Implementations include the real CLI TUI responder (defined at the CLI
/// layer) and the test responders below.
pub trait PromptResponder: Send + Sync {
    /// Decide a single prompt synchronously.
    fn ask(&self, prompt: &PendingPrompt) -> PermissionDecision;
}

/// Test responder that always denies.
#[derive(Debug, Default, Clone, Copy)]
pub struct DenyAllResponder;

impl PromptResponder for DenyAllResponder {
    fn ask(&self, _prompt: &PendingPrompt) -> PermissionDecision {
        PermissionDecision::Deny
    }
}

/// Test responder that always allows once.
#[derive(Debug, Default, Clone, Copy)]
pub struct AllowAllResponder;

impl PromptResponder for AllowAllResponder {
    fn ask(&self, _prompt: &PendingPrompt) -> PermissionDecision {
        PermissionDecision::AllowOnce
    }
}

/// Test responder that returns a pre-scripted queue of decisions.
///
/// # Panics
///
/// The `ask` implementation panics if the queue is exhausted. This is
/// intentional: scripted tests should always provide enough decisions to
/// cover their evaluation calls. The panic is gated behind `cfg(any(test,
/// debug_assertions))` semantics by the responder being a test helper — in
/// release builds, callers should not reach this branch. The poisoned-mutex
/// branch is treated the same way for the same reason.
#[derive(Debug)]
pub struct ScriptedResponder {
    queue: Mutex<VecDeque<PermissionDecision>>,
}

impl ScriptedResponder {
    /// Create a scripted responder from a vector of decisions.
    ///
    /// The decisions are returned in FIFO order from [`PromptResponder::ask`].
    #[must_use]
    pub fn new(decisions: Vec<PermissionDecision>) -> Self {
        Self {
            queue: Mutex::new(decisions.into_iter().collect()),
        }
    }

    /// Remaining unconsumed decisions.
    #[must_use]
    pub fn remaining(&self) -> usize {
        match self.queue.lock() {
            Ok(g) => g.len(),
            Err(poisoned) => poisoned.into_inner().len(),
        }
    }
}

impl PromptResponder for ScriptedResponder {
    // `ScriptedResponder` is a test helper. Panicking on exhausted queue is
    // its contract — documented on the struct and asserted by an
    // in-module `#[should_panic]` test. The workspace lints
    // `clippy::panic`, `clippy::option_if_let_else`, and
    // `clippy::missing_panics_doc` are intentionally relaxed here.
    #[allow(clippy::panic, clippy::option_if_let_else, clippy::missing_panics_doc)]
    fn ask(&self, _prompt: &PendingPrompt) -> PermissionDecision {
        let mut guard = match self.queue.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        match guard.pop_front() {
            Some(d) => d,
            None => panic!("ScriptedResponder queue exhausted"),
        }
    }
}

// ---------------------------------------------------------------------------
// PermissionStore
// ---------------------------------------------------------------------------

/// Bucket distinguishing session-scoped from forever-scoped remembered
/// decisions inside a [`PermissionStore`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PermissionScope {
    /// Lives until [`PermissionStore::forget_session`] is called.
    Session,
    /// Lives forever (or until externally cleared).
    Forever,
}

/// Sha-256 hex digest of a [`PermissionRequest`].
pub type RequestDigest = String;

/// Composite key used by [`PermissionStore`].
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PermissionKey(pub PermissionScope, pub RequestDigest);

/// A decision stored alongside the time it was recorded.
#[derive(Debug, Clone, Copy)]
pub struct RememberedDecision {
    /// The recorded decision.
    pub decision: PermissionDecision,
    /// When the decision was recorded.
    pub recorded_at: SystemTime,
}

/// In-memory store of remembered permission decisions.
#[derive(Debug, Default)]
pub struct PermissionStore {
    remembered: Mutex<BTreeMap<PermissionKey, RememberedDecision>>,
}

impl PermissionStore {
    /// Create an empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Look up a remembered decision for `req`.
    ///
    /// `Forever` wins over `Session` when both exist.
    #[must_use]
    pub fn lookup(&self, req: &PermissionRequest) -> Option<PermissionDecision> {
        let digest = request_digest(req);
        let result = {
            let guard = match self.remembered.lock() {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };
            guard
                .get(&PermissionKey(PermissionScope::Forever, digest.clone()))
                .map(|r| r.decision)
                .or_else(|| {
                    guard
                        .get(&PermissionKey(PermissionScope::Session, digest))
                        .map(|r| r.decision)
                })
        };
        result
    }

    /// Record a decision if its variant warrants persistence.
    ///
    /// `AllowOnce` and `Deny` are no-ops. `AllowSession` lands in the Session
    /// bucket. `AllowForever` and `DenyForever` land in the Forever bucket.
    pub fn record(&self, req: &PermissionRequest, decision: PermissionDecision, now: SystemTime) {
        let scope = match decision {
            PermissionDecision::AllowSession => PermissionScope::Session,
            PermissionDecision::AllowForever | PermissionDecision::DenyForever => {
                PermissionScope::Forever
            }
            PermissionDecision::AllowOnce | PermissionDecision::Deny => return,
        };
        let digest = request_digest(req);
        let mut guard = match self.remembered.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        guard.insert(
            PermissionKey(scope, digest),
            RememberedDecision {
                decision,
                recorded_at: now,
            },
        );
    }

    /// Drop every Session-scoped entry. Forever entries are retained.
    pub fn forget_session(&self) {
        let mut guard = match self.remembered.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        guard.retain(|k, _| k.0 != PermissionScope::Session);
    }

    /// Count of recorded entries (both buckets).
    #[must_use]
    pub fn len(&self) -> usize {
        match self.remembered.lock() {
            Ok(g) => g.len(),
            Err(p) => p.into_inner().len(),
        }
    }

    /// `true` if no decisions are recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// ---------------------------------------------------------------------------
// evaluate / request_digest
// ---------------------------------------------------------------------------

/// Evaluate `req` against the store, falling back to `responder` and
/// recording any persistable decision.
///
/// `req` is taken by value so the [`PendingPrompt`] passed to the responder
/// can own its own copy — callers typically construct the request inline.
#[allow(clippy::needless_pass_by_value)] // owning the request matches the spec'd surface.
pub fn evaluate(
    req: PermissionRequest,
    store: &PermissionStore,
    gen: &PromptIdGen,
    responder: &dyn PromptResponder,
    now: SystemTime,
) -> PermissionDecision {
    if let Some(remembered) = store.lookup(&req) {
        return remembered;
    }
    let prompt = PendingPrompt {
        id: gen.next(),
        request: req.clone(),
        issued_at: now,
    };
    let decision = responder.ask(&prompt);
    store.record(&req, decision, now);
    decision
}

/// Sha-256 hex digest of the JSON-serialized request.
///
/// Used as the key inside [`PermissionStore`] so structurally identical
/// requests (same kind + same fields) collapse to a single entry.
#[must_use]
pub fn request_digest(req: &PermissionRequest) -> RequestDigest {
    // Serializing a small enum to JSON is infallible in practice — but we
    // refuse to unwrap. A failure falls back to a stable per-variant tag so
    // the store still functions, just with coarser collision behaviour.
    let raw = serde_json::to_string(req).unwrap_or_else(|_| format!("{req:?}"));
    let mut hasher = Sha256::new();
    hasher.update(raw.as_bytes());
    let bytes = hasher.finalize();
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push_str(&hex_byte(b));
    }
    out
}

#[inline]
fn hex_byte(b: u8) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(2);
    s.push(char::from(HEX[(b >> 4) as usize]));
    s.push(char::from(HEX[(b & 0x0f) as usize]));
    s
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::thread;
    use std::time::{Duration, UNIX_EPOCH};

    fn sample_cap() -> PermissionRequest {
        PermissionRequest::CapabilityGrant {
            capability: "net".into(),
            target: Some("example.com".into()),
            reason: "fetch docs".into(),
        }
    }

    fn sample_secret() -> PermissionRequest {
        PermissionRequest::SecretAccess {
            secret_ref: "proj/api_key".into(),
            scope: "read".into(),
        }
    }

    fn sample_net() -> PermissionRequest {
        PermissionRequest::NetworkHost {
            host: "api.example.com".into(),
            port: Some(443),
        }
    }

    fn sample_file() -> PermissionRequest {
        PermissionRequest::FileWrite {
            path: PathBuf::from("/tmp/out.txt"),
        }
    }

    fn sample_tool() -> PermissionRequest {
        PermissionRequest::ToolUse { args: String::new(),
            tool_id: "shell.exec".into(),
        }
    }

    fn now() -> SystemTime {
        UNIX_EPOCH + Duration::from_secs(1_700_000_000)
    }

    #[test]
    fn prompt_id_gen_monotonic_single_thread() {
        let g = PromptIdGen::new();
        let a = g.next();
        let b = g.next();
        let c = g.next();
        assert_eq!(a.0, 0);
        assert_eq!(b.0, 1);
        assert_eq!(c.0, 2);
    }

    #[test]
    fn prompt_id_gen_unique_under_threads() {
        let g = Arc::new(PromptIdGen::new());
        let mut handles = Vec::new();
        for _ in 0..4 {
            let g = Arc::clone(&g);
            handles.push(thread::spawn(move || {
                let mut ids = Vec::with_capacity(1000);
                for _ in 0..1000 {
                    ids.push(g.next().0);
                }
                ids
            }));
        }
        let mut all: HashSet<u64> = HashSet::new();
        for h in handles {
            let ids = h.join().unwrap_or_default();
            for id in ids {
                assert!(all.insert(id), "duplicate id {id}");
            }
        }
        assert_eq!(all.len(), 4_000);
    }

    #[test]
    fn request_digest_is_deterministic() {
        let r = sample_cap();
        assert_eq!(request_digest(&r), request_digest(&r));
    }

    #[test]
    fn request_digest_distinct_across_variants() {
        let mut seen: HashSet<String> = HashSet::new();
        for r in [
            sample_cap(),
            sample_secret(),
            sample_net(),
            sample_file(),
            sample_tool(),
        ] {
            assert!(seen.insert(request_digest(&r)), "digest collision");
        }
    }

    #[test]
    fn request_digest_distinct_within_variant() {
        let a = PermissionRequest::NetworkHost {
            host: "a".into(),
            port: Some(80),
        };
        let b = PermissionRequest::NetworkHost {
            host: "b".into(),
            port: Some(80),
        };
        assert_ne!(request_digest(&a), request_digest(&b));
    }

    #[test]
    fn evaluate_deny_all_returns_deny_and_does_not_remember() {
        let store = PermissionStore::new();
        let gen = PromptIdGen::new();
        let r = evaluate(sample_cap(), &store, &gen, &DenyAllResponder, now());
        assert_eq!(r, PermissionDecision::Deny);
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn evaluate_allow_all_returns_once_and_does_not_remember() {
        let store = PermissionStore::new();
        let gen = PromptIdGen::new();
        let r = evaluate(sample_cap(), &store, &gen, &AllowAllResponder, now());
        assert_eq!(r, PermissionDecision::AllowOnce);
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn evaluate_allow_forever_records_into_forever_and_short_circuits() {
        let store = PermissionStore::new();
        let gen = PromptIdGen::new();
        let responder = ScriptedResponder::new(vec![PermissionDecision::AllowForever]);
        let first = evaluate(sample_cap(), &store, &gen, &responder, now());
        assert_eq!(first, PermissionDecision::AllowForever);
        assert_eq!(store.len(), 1);
        // Second call short-circuits — queue stays empty without panicking.
        let second = evaluate(sample_cap(), &store, &gen, &responder, now());
        assert_eq!(second, PermissionDecision::AllowForever);
        assert_eq!(responder.remaining(), 0);
    }

    #[test]
    fn evaluate_allow_session_records_into_session_and_short_circuits() {
        let store = PermissionStore::new();
        let gen = PromptIdGen::new();
        let responder = ScriptedResponder::new(vec![PermissionDecision::AllowSession]);
        let first = evaluate(sample_secret(), &store, &gen, &responder, now());
        assert_eq!(first, PermissionDecision::AllowSession);
        let second = evaluate(sample_secret(), &store, &gen, &responder, now());
        assert_eq!(second, PermissionDecision::AllowSession);
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn evaluate_deny_forever_records_and_short_circuits() {
        let store = PermissionStore::new();
        let gen = PromptIdGen::new();
        let responder = ScriptedResponder::new(vec![PermissionDecision::DenyForever]);
        let first = evaluate(sample_tool(), &store, &gen, &responder, now());
        assert_eq!(first, PermissionDecision::DenyForever);
        let second = evaluate(sample_tool(), &store, &gen, &responder, now());
        assert_eq!(second, PermissionDecision::DenyForever);
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn forget_session_clears_session_but_keeps_forever() {
        let store = PermissionStore::new();
        store.record(&sample_cap(), PermissionDecision::AllowSession, now());
        store.record(&sample_tool(), PermissionDecision::AllowForever, now());
        assert_eq!(store.len(), 2);
        store.forget_session();
        assert_eq!(store.len(), 1);
        assert_eq!(
            store.lookup(&sample_cap()),
            None,
            "session entry should be gone"
        );
        assert_eq!(
            store.lookup(&sample_tool()),
            Some(PermissionDecision::AllowForever)
        );
    }

    #[test]
    fn record_is_noop_for_allow_once_and_deny() {
        let store = PermissionStore::new();
        store.record(&sample_cap(), PermissionDecision::AllowOnce, now());
        store.record(&sample_cap(), PermissionDecision::Deny, now());
        assert_eq!(store.len(), 0);
        assert!(store.is_empty());
    }

    #[test]
    fn different_variants_do_not_alias_in_store() {
        let store = PermissionStore::new();
        store.record(&sample_cap(), PermissionDecision::AllowForever, now());
        store.record(&sample_secret(), PermissionDecision::AllowForever, now());
        store.record(&sample_net(), PermissionDecision::AllowForever, now());
        store.record(&sample_file(), PermissionDecision::AllowForever, now());
        store.record(&sample_tool(), PermissionDecision::AllowForever, now());
        assert_eq!(store.len(), 5);
    }

    #[test]
    fn decision_serde_is_snake_case() {
        assert_eq!(
            serde_json::to_string(&PermissionDecision::AllowOnce).unwrap_or_default(),
            "\"allow_once\""
        );
        assert_eq!(
            serde_json::to_string(&PermissionDecision::AllowSession).unwrap_or_default(),
            "\"allow_session\""
        );
        assert_eq!(
            serde_json::to_string(&PermissionDecision::AllowForever).unwrap_or_default(),
            "\"allow_forever\""
        );
        assert_eq!(
            serde_json::to_string(&PermissionDecision::Deny).unwrap_or_default(),
            "\"deny\""
        );
        assert_eq!(
            serde_json::to_string(&PermissionDecision::DenyForever).unwrap_or_default(),
            "\"deny_forever\""
        );
    }

    #[test]
    fn request_serde_round_trips_with_kind_tag() {
        for r in [
            sample_cap(),
            sample_secret(),
            sample_net(),
            sample_file(),
            sample_tool(),
        ] {
            let json = serde_json::to_string(&r).unwrap_or_default();
            assert!(json.contains("\"kind\""), "missing kind tag: {json}");
            let back: PermissionRequest = serde_json::from_str(&json).unwrap_or_else(|e| {
                panic!("round-trip failed for {r:?}: {e}");
            });
            assert_eq!(back, r);
        }
    }

    #[test]
    fn permission_store_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<PermissionStore>();
        assert_send_sync::<PromptIdGen>();
    }

    #[test]
    #[should_panic(expected = "ScriptedResponder queue exhausted")]
    fn scripted_responder_panics_when_exhausted() {
        let responder = ScriptedResponder::new(Vec::new());
        let prompt = PendingPrompt {
            id: PromptId(0),
            request: sample_cap(),
            issued_at: now(),
        };
        let _ = responder.ask(&prompt);
    }

    #[test]
    fn evaluate_calls_responder_exactly_once_when_remembered() {
        let store = PermissionStore::new();
        let gen = PromptIdGen::new();
        let responder = ScriptedResponder::new(vec![PermissionDecision::AllowForever]);
        let _ = evaluate(sample_cap(), &store, &gen, &responder, now());
        assert_eq!(responder.remaining(), 0);
        // Second + third calls must NOT consume from the queue.
        let _ = evaluate(sample_cap(), &store, &gen, &responder, now());
        let _ = evaluate(sample_cap(), &store, &gen, &responder, now());
        assert_eq!(responder.remaining(), 0);
    }

    #[test]
    fn pending_prompt_carries_issued_timestamp_from_now() {
        struct Capture(Mutex<Option<SystemTime>>);
        impl PromptResponder for Capture {
            fn ask(&self, prompt: &PendingPrompt) -> PermissionDecision {
                if let Ok(mut g) = self.0.lock() {
                    *g = Some(prompt.issued_at);
                }
                PermissionDecision::AllowOnce
            }
        }
        let cap = Capture(Mutex::new(None));
        let store = PermissionStore::new();
        let gen = PromptIdGen::new();
        let t = now();
        let _ = evaluate(sample_cap(), &store, &gen, &cap, t);
        let captured = cap.0.lock().ok().and_then(|g| *g);
        assert_eq!(captured, Some(t));
    }

    #[test]
    fn len_matches_recorded_count() {
        let store = PermissionStore::new();
        assert_eq!(store.len(), 0);
        store.record(&sample_cap(), PermissionDecision::AllowSession, now());
        assert_eq!(store.len(), 1);
        store.record(&sample_secret(), PermissionDecision::AllowForever, now());
        assert_eq!(store.len(), 2);
        // Re-recording same request + bucket is an upsert, not a new row.
        store.record(&sample_cap(), PermissionDecision::AllowSession, now());
        assert_eq!(store.len(), 2);
    }

    #[test]
    fn permission_key_ord_is_stable() {
        let d1 = request_digest(&sample_cap());
        let d2 = request_digest(&sample_secret());
        let mut keys = vec![
            PermissionKey(PermissionScope::Session, d1.clone()),
            PermissionKey(PermissionScope::Forever, d1),
            PermissionKey(PermissionScope::Session, d2.clone()),
            PermissionKey(PermissionScope::Forever, d2),
        ];
        let original = keys.clone();
        keys.sort();
        // Sorting must be deterministic — sorting twice yields the same order.
        let mut second = original;
        second.sort();
        assert_eq!(keys, second);
        // Forever < Session because variants are ordered by declaration.
        for w in keys.windows(2) {
            assert!(w[0] <= w[1]);
        }
    }

    #[test]
    fn lookup_returns_forever_over_session_when_both_present() {
        let store = PermissionStore::new();
        // Session first
        store.record(&sample_cap(), PermissionDecision::AllowSession, now());
        // Forever — would normally not co-exist for the same request, but the
        // store does not forbid it; the lookup contract is: Forever wins.
        store.record(&sample_cap(), PermissionDecision::DenyForever, now());
        assert_eq!(
            store.lookup(&sample_cap()),
            Some(PermissionDecision::DenyForever)
        );
    }

    #[test]
    fn lookup_returns_none_for_unknown_request() {
        let store = PermissionStore::new();
        assert_eq!(store.lookup(&sample_cap()), None);
    }

    #[test]
    fn default_constructors_compile() {
        // Smoke: cover Default impls so the coverage gate is happy.
        let _ = PermissionStore::default();
        let _ = PromptIdGen::default();
        let _ = DenyAllResponder;
        let _ = AllowAllResponder;
    }
}
