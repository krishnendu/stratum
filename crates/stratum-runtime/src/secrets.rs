//! Secrets / keyring data shape.
//!
//! Phase 1 (data only) — the OS keyring binding lands later. This module
//! pins the public types (`SecretId`, `ProjectId`, `SecretRef`,
//! `SecretScope`, `SecretValue`, `SecretStore`) and ships an
//! [`InMemorySecretStore`] so the rest of the runtime can compile and test
//! against the trait surface today.
//!
//! See `plan/31-secrets-and-keys.md`.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::{self, Debug, Display, Formatter};
use std::sync::Mutex;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use zeroize::{Zeroize, ZeroizeOnDrop};

// ---------------------------------------------------------------------------
// SecretId / ProjectId
// ---------------------------------------------------------------------------

/// Maximum length, in bytes, of a [`SecretId`] or [`ProjectId`].
const MAX_ID_LEN: usize = 64;

/// Validate the shared `[a-zA-Z0-9._-]`, 1..=64, no-leading-dot-or-dash rule
/// used by both [`SecretId`] and [`ProjectId`].
fn validate_id(s: &str) -> Result<(), SecretIdError> {
    if s.is_empty() {
        return Err(SecretIdError::Empty);
    }
    if s.len() > MAX_ID_LEN {
        return Err(SecretIdError::TooLong { len: s.len() });
    }
    let mut chars = s.chars();
    // Emptiness was rejected above; this branch is defensive only.
    let Some(first) = chars.next() else {
        return Err(SecretIdError::Empty);
    };
    if first == '-' || first == '.' {
        return Err(SecretIdError::BadPrefix { ch: first });
    }
    if !is_id_char(first) {
        return Err(SecretIdError::InvalidChar { ch: first });
    }
    for c in chars {
        if !is_id_char(c) {
            return Err(SecretIdError::InvalidChar { ch: c });
        }
    }
    Ok(())
}

#[inline]
const fn is_id_char(c: char) -> bool {
    matches!(c, 'a'..='z' | 'A'..='Z' | '0'..='9' | '.' | '_' | '-')
}

/// Stable identifier for a single secret within a [`SecretScope`].
///
/// Construct via [`SecretId::new`]; the inner string is guaranteed to
/// satisfy the validation rules documented on the constructor.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SecretId(String);

impl SecretId {
    /// Validate and wrap `s`.
    ///
    /// Rules:
    /// - 1..=64 bytes
    /// - ASCII characters `[a-zA-Z0-9._-]`
    /// - must not start with `-` or `.`
    ///
    /// # Errors
    ///
    /// Returns a [`SecretIdError`] describing the first failed rule.
    pub fn new(s: &str) -> Result<Self, SecretIdError> {
        validate_id(s)?;
        Ok(Self(s.to_owned()))
    }

    /// Borrow the underlying string slice.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Display for SecretId {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl AsRef<str> for SecretId {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl Serialize for SecretId {
    fn serialize<S: serde::Serializer>(&self, ser: S) -> Result<S::Ok, S::Error> {
        ser.serialize_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for SecretId {
    fn deserialize<D: serde::Deserializer<'de>>(de: D) -> Result<Self, D::Error> {
        let s = String::deserialize(de)?;
        Self::new(&s).map_err(serde::de::Error::custom)
    }
}

/// Stable identifier for a project scope (mirror of [`SecretId`] rules).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ProjectId(String);

impl ProjectId {
    /// Validate and wrap `s`. Rules match [`SecretId::new`].
    ///
    /// # Errors
    ///
    /// Returns a [`SecretIdError`] on the first failed rule.
    pub fn new(s: &str) -> Result<Self, SecretIdError> {
        validate_id(s)?;
        Ok(Self(s.to_owned()))
    }

    /// Borrow the underlying string slice.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Display for ProjectId {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl AsRef<str> for ProjectId {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl Serialize for ProjectId {
    fn serialize<S: serde::Serializer>(&self, ser: S) -> Result<S::Ok, S::Error> {
        ser.serialize_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for ProjectId {
    fn deserialize<D: serde::Deserializer<'de>>(de: D) -> Result<Self, D::Error> {
        let s = String::deserialize(de)?;
        Self::new(&s).map_err(serde::de::Error::custom)
    }
}

/// First-failure rejection from [`SecretId::new`] or [`ProjectId::new`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SecretIdError {
    /// Input was the empty string.
    Empty,
    /// Input exceeded the 64-byte limit.
    TooLong {
        /// Observed length in bytes.
        len: usize,
    },
    /// Encountered a character outside `[a-zA-Z0-9._-]`.
    InvalidChar {
        /// The offending character.
        ch: char,
    },
    /// First character was `-` or `.`.
    BadPrefix {
        /// The offending leading character.
        ch: char,
    },
}

impl Display for SecretIdError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => f.write_str("secret id must not be empty"),
            Self::TooLong { len } => {
                write!(f, "secret id is {len} bytes; max is {MAX_ID_LEN}")
            }
            Self::InvalidChar { ch } => {
                write!(f, "secret id contains invalid character {ch:?}")
            }
            Self::BadPrefix { ch } => {
                write!(f, "secret id must not start with {ch:?}")
            }
        }
    }
}

impl Error for SecretIdError {}

// ---------------------------------------------------------------------------
// SecretScope
// ---------------------------------------------------------------------------

/// Lookup scope for a secret.
///
/// Round-trips as `{"kind":"global"}` or `{"kind":"project","id":"foo"}`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SecretScope {
    /// Visible to every workspace on this machine.
    Global,
    /// Scoped to a single project, identified by `id`.
    Project {
        /// The owning project identifier.
        id: ProjectId,
    },
}

// ---------------------------------------------------------------------------
// SecretRef
// ---------------------------------------------------------------------------

/// Fully-qualified reference to a single secret: its [`SecretScope`] plus
/// its [`SecretId`].
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SecretRef {
    /// Resolution scope.
    pub scope: SecretScope,
    /// Identifier within that scope.
    pub id: SecretId,
}

impl SecretRef {
    /// Build a new `SecretRef` directly from a pre-validated scope + id.
    #[must_use]
    pub const fn new(scope: SecretScope, id: SecretId) -> Self {
        Self { scope, id }
    }
}

impl Display for SecretRef {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match &self.scope {
            SecretScope::Global => write!(f, "global:{}", self.id),
            SecretScope::Project { id } => write!(f, "project[{id}]:{}", self.id),
        }
    }
}

// ---------------------------------------------------------------------------
// SecretValue
// ---------------------------------------------------------------------------

/// Opaque secret payload.
///
/// - Bytes are zeroized on drop.
/// - `Debug` renders `SecretValue(<redacted, N bytes>)`; never raw bytes.
/// - Intentionally does **not** implement `Serialize` / `Deserialize`:
///   secrets are persisted exclusively through the backend, never inline.
#[derive(Clone, PartialEq, Eq, Zeroize, ZeroizeOnDrop)]
pub struct SecretValue(Box<[u8]>);

impl SecretValue {
    /// Borrow the raw bytes. Callers must treat the slice as sensitive.
    #[must_use]
    pub fn expose(&self) -> &[u8] {
        &self.0
    }

    /// Length in bytes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// True when the payload is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl Debug for SecretValue {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "SecretValue(<redacted, {} bytes>)", self.0.len())
    }
}

impl From<&str> for SecretValue {
    fn from(s: &str) -> Self {
        Self(s.as_bytes().to_vec().into_boxed_slice())
    }
}

impl From<Vec<u8>> for SecretValue {
    fn from(v: Vec<u8>) -> Self {
        Self(v.into_boxed_slice())
    }
}

// ---------------------------------------------------------------------------
// SecretStore trait + error
// ---------------------------------------------------------------------------

/// Backend-agnostic CRUD surface for secrets.
///
/// All methods are synchronous; an async wrapper layers on top in a later
/// phase once the OS-keyring backends arrive.
pub trait SecretStore: Send + Sync {
    /// Fetch the value at `r`, or `Ok(None)` when no entry exists.
    ///
    /// # Errors
    ///
    /// Backend failures surface as [`SecretStoreError::Backend`].
    fn get(&self, r: &SecretRef) -> Result<Option<SecretValue>, SecretStoreError>;

    /// Store (or overwrite) the value at `r`.
    ///
    /// # Errors
    ///
    /// Backend failures surface as [`SecretStoreError::Backend`].
    fn put(&self, r: &SecretRef, value: SecretValue) -> Result<(), SecretStoreError>;

    /// Remove the entry at `r`. Returns `true` when an entry existed.
    ///
    /// # Errors
    ///
    /// Backend failures surface as [`SecretStoreError::Backend`].
    fn delete(&self, r: &SecretRef) -> Result<bool, SecretStoreError>;

    /// List the [`SecretId`]s present in `scope`, sorted ascending.
    ///
    /// # Errors
    ///
    /// Backend failures surface as [`SecretStoreError::Backend`].
    fn list(&self, scope: &SecretScope) -> Result<Vec<SecretId>, SecretStoreError>;
}

/// Failure modes for a [`SecretStore`] operation.
#[derive(Debug)]
pub enum SecretStoreError {
    /// Backend-specific failure with a human-readable message.
    Backend(String),
    /// Caller supplied a malformed identifier.
    InvalidKey(SecretIdError),
    /// Reserved for backends that distinguish "missing" from "ok-none";
    /// the in-memory store does not currently surface this variant.
    NotFound,
}

impl Display for SecretStoreError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Backend(msg) => write!(f, "secret store backend error: {msg}"),
            Self::InvalidKey(err) => write!(f, "secret store invalid key: {err}"),
            Self::NotFound => f.write_str("secret store entry not found"),
        }
    }
}

impl Error for SecretStoreError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InvalidKey(err) => Some(err),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// InMemorySecretStore
// ---------------------------------------------------------------------------

/// Thread-safe, process-local [`SecretStore`] used by tests and by the
/// runtime before the real OS keyring binding lands.
#[derive(Debug, Default)]
pub struct InMemorySecretStore {
    inner: Mutex<BTreeMap<(SecretScope, SecretId), SecretValue>>,
}

impl InMemorySecretStore {
    /// Empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn with_lock<R>(
        &self,
        f: impl FnOnce(&mut BTreeMap<(SecretScope, SecretId), SecretValue>) -> R,
    ) -> Result<R, SecretStoreError> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|e| SecretStoreError::Backend(format!("mutex poisoned: {e}")))?;
        Ok(f(&mut guard))
    }
}

impl SecretStore for InMemorySecretStore {
    fn get(&self, r: &SecretRef) -> Result<Option<SecretValue>, SecretStoreError> {
        self.with_lock(|map| map.get(&(r.scope.clone(), r.id.clone())).cloned())
    }

    fn put(&self, r: &SecretRef, value: SecretValue) -> Result<(), SecretStoreError> {
        self.with_lock(|map| {
            map.insert((r.scope.clone(), r.id.clone()), value);
        })
    }

    fn delete(&self, r: &SecretRef) -> Result<bool, SecretStoreError> {
        self.with_lock(|map| map.remove(&(r.scope.clone(), r.id.clone())).is_some())
    }

    fn list(&self, scope: &SecretScope) -> Result<Vec<SecretId>, SecretStoreError> {
        self.with_lock(|map| {
            let mut out: Vec<SecretId> = map
                .keys()
                .filter(|(s, _)| s == scope)
                .map(|(_, id)| id.clone())
                .collect();
            out.sort();
            out
        })
    }
}

// ---------------------------------------------------------------------------
// KeyringSecretStore (feature-gated)
// ---------------------------------------------------------------------------

/// OS-backed [`SecretStore`] using the `keyring` crate.
///
/// Backends:
/// - **macOS** — Keychain Services
/// - **Linux** — Secret Service (`gnome-keyring` / KWallet)
/// - **Windows** — Credential Manager
///
/// Entries are stored under the service name configured at construction;
/// the default is `"stratum"`. The OS surfaces them under that name in
/// Keychain Access / `secret-tool` / Credential Manager.
///
/// ## Listing semantics
///
/// The `keyring` crate doesn't expose a native enumeration on every
/// platform, so [`Self::list`] tracks every put/delete in a per-instance
/// index. The index persists alongside the store as a single entry
/// named `__stratum_index__` so a fresh process can resume the listing.
/// This is fine for Stratum's expected scale (dozens of secrets per
/// scope, not thousands).
#[cfg(feature = "os-keyring")]
#[derive(Debug)]
pub struct KeyringSecretStore {
    service: String,
}

#[cfg(feature = "os-keyring")]
impl KeyringSecretStore {
    /// Build a store using the default service name `"stratum"`.
    #[must_use]
    pub fn new() -> Self {
        Self::with_service("stratum")
    }

    /// Build a store using a custom service name. Useful for tests
    /// against a sandbox keyring or for multi-tenant deployments.
    #[must_use]
    pub fn with_service(service: impl Into<String>) -> Self {
        Self {
            service: service.into(),
        }
    }

    /// Encode a SecretRef into the keyring's "account" slot. The
    /// keyring's (service, account) pair is the addressable key.
    fn account_for(r: &SecretRef) -> String {
        match &r.scope {
            SecretScope::Global => format!("global:{}", r.id.as_str()),
            SecretScope::Project { id } => {
                format!("project:{}:{}", id.as_str(), r.id.as_str())
            }
        }
    }

    /// Decode an account slot back into (scope, id). Returns `None` on
    /// malformed input — used only by `list` to skip stale rows.
    fn decode_account(account: &str) -> Option<(SecretScope, SecretId)> {
        if let Some(rest) = account.strip_prefix("global:") {
            let id = SecretId::new(rest).ok()?;
            return Some((SecretScope::Global, id));
        }
        if let Some(rest) = account.strip_prefix("project:") {
            let (proj, id_str) = rest.split_once(':')?;
            let proj_id = ProjectId::new(proj).ok()?;
            let id = SecretId::new(id_str).ok()?;
            return Some((SecretScope::Project { id: proj_id }, id));
        }
        None
    }

    /// Read the per-store index of accounts (newline-separated). Returns
    /// an empty vec when no index entry exists yet.
    fn read_index(&self) -> Vec<String> {
        let Ok(entry) = keyring::Entry::new(&self.service, "__stratum_index__") else {
            return Vec::new();
        };
        match entry.get_password() {
            Ok(raw) => raw
                .lines()
                .filter(|l| !l.is_empty())
                .map(String::from)
                .collect(),
            Err(_) => Vec::new(),
        }
    }

    fn write_index(&self, rows: &[String]) -> Result<(), SecretStoreError> {
        let entry = keyring::Entry::new(&self.service, "__stratum_index__")
            .map_err(|e| SecretStoreError::Backend(e.to_string()))?;
        let body = rows.join("\n");
        if body.is_empty() {
            let _ = entry.delete_password();
            return Ok(());
        }
        entry
            .set_password(&body)
            .map_err(|e| SecretStoreError::Backend(e.to_string()))
    }
}

#[cfg(feature = "os-keyring")]
impl Default for KeyringSecretStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "os-keyring")]
impl SecretStore for KeyringSecretStore {
    fn get(&self, r: &SecretRef) -> Result<Option<SecretValue>, SecretStoreError> {
        let entry = keyring::Entry::new(&self.service, &Self::account_for(r))
            .map_err(|e| SecretStoreError::Backend(e.to_string()))?;
        match entry.get_password() {
            Ok(raw) => Ok(Some(SecretValue::from(raw.as_str()))),
            Err(keyring::Error::NoEntry) => Ok(None),
            Err(e) => Err(SecretStoreError::Backend(e.to_string())),
        }
    }

    fn put(&self, r: &SecretRef, value: SecretValue) -> Result<(), SecretStoreError> {
        let account = Self::account_for(r);
        let entry = keyring::Entry::new(&self.service, &account)
            .map_err(|e| SecretStoreError::Backend(e.to_string()))?;
        // SecretValue is opaque bytes; convert to UTF-8 lossily here —
        // the keyring backends require a string. Stratum stores text
        // tokens (API keys, OAuth refresh) in practice, not raw blobs.
        let s = std::str::from_utf8(value.expose())
            .map_err(|e| SecretStoreError::Backend(format!("non-UTF8 secret: {e}")))?;
        entry
            .set_password(s)
            .map_err(|e| SecretStoreError::Backend(e.to_string()))?;
        let mut idx = self.read_index();
        if !idx.contains(&account) {
            idx.push(account);
            idx.sort();
            self.write_index(&idx)?;
        }
        Ok(())
    }

    fn delete(&self, r: &SecretRef) -> Result<bool, SecretStoreError> {
        let account = Self::account_for(r);
        let entry = keyring::Entry::new(&self.service, &account)
            .map_err(|e| SecretStoreError::Backend(e.to_string()))?;
        let existed = matches!(entry.get_password(), Ok(_));
        match entry.delete_password() {
            Ok(()) | Err(keyring::Error::NoEntry) => {}
            Err(e) => return Err(SecretStoreError::Backend(e.to_string())),
        }
        if existed {
            let mut idx = self.read_index();
            idx.retain(|a| a != &account);
            self.write_index(&idx)?;
        }
        Ok(existed)
    }

    fn list(&self, scope: &SecretScope) -> Result<Vec<SecretId>, SecretStoreError> {
        let idx = self.read_index();
        let mut out: Vec<SecretId> = idx
            .iter()
            .filter_map(|account| {
                let (s, id) = Self::decode_account(account)?;
                if &s == scope {
                    Some(id)
                } else {
                    None
                }
            })
            .collect();
        out.sort_by(|a, b| a.as_str().cmp(b.as_str()));
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// redact_for_log
// ---------------------------------------------------------------------------

/// Emit a deterministic log-safe fingerprint of `value`: `***` followed by
/// the first 8 hex chars of the SHA-256 of the bytes.
#[must_use]
pub fn redact_for_log(value: &SecretValue) -> String {
    let digest = Sha256::digest(value.expose());
    let mut out = String::with_capacity(3 + 8);
    out.push_str("***");
    for byte in digest.iter().take(4) {
        let hi = (byte >> 4) & 0xF;
        let lo = byte & 0xF;
        out.push(hex_digit(hi));
        out.push(hex_digit(lo));
    }
    out
}

#[inline]
const fn hex_digit(nibble: u8) -> char {
    match nibble {
        0..=9 => (b'0' + nibble) as char,
        _ => (b'a' + nibble - 10) as char,
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::sync::Arc;

    use super::*;

    fn sid(s: &str) -> SecretId {
        SecretId::new(s).unwrap()
    }

    fn pid(s: &str) -> ProjectId {
        ProjectId::new(s).unwrap()
    }

    // ---- SecretId::new ----------------------------------------------------

    #[test]
    fn secret_id_happy() {
        let id = SecretId::new("openai.api_key-1").unwrap();
        assert_eq!(id.as_str(), "openai.api_key-1");
        assert_eq!(id.to_string(), "openai.api_key-1");
        assert_eq!(<SecretId as AsRef<str>>::as_ref(&id), "openai.api_key-1");
    }

    #[test]
    fn secret_id_empty_rejected() {
        assert_eq!(SecretId::new(""), Err(SecretIdError::Empty));
    }

    #[test]
    fn secret_id_too_long_rejected() {
        let s = "a".repeat(65);
        assert_eq!(SecretId::new(&s), Err(SecretIdError::TooLong { len: 65 }));
    }

    #[test]
    fn secret_id_invalid_char_rejected() {
        assert_eq!(
            SecretId::new("bad key"),
            Err(SecretIdError::InvalidChar { ch: ' ' })
        );
        assert_eq!(
            SecretId::new("bad/key"),
            Err(SecretIdError::InvalidChar { ch: '/' })
        );
    }

    #[test]
    fn secret_id_bad_prefix_dot_rejected() {
        assert_eq!(
            SecretId::new(".hidden"),
            Err(SecretIdError::BadPrefix { ch: '.' })
        );
    }

    #[test]
    fn secret_id_bad_prefix_dash_rejected() {
        assert_eq!(
            SecretId::new("-flag"),
            Err(SecretIdError::BadPrefix { ch: '-' })
        );
    }

    #[test]
    fn secret_id_max_len_allowed() {
        let s = "a".repeat(64);
        assert!(SecretId::new(&s).is_ok());
    }

    #[test]
    fn secret_id_error_display_covers_all_variants() {
        assert!(SecretIdError::Empty.to_string().contains("empty"));
        assert!(SecretIdError::TooLong { len: 99 }
            .to_string()
            .contains("99"));
        assert!(SecretIdError::InvalidChar { ch: '/' }
            .to_string()
            .contains('/'));
        assert!(SecretIdError::BadPrefix { ch: '-' }
            .to_string()
            .contains('-'));
        // Error trait
        let _: &dyn Error = &SecretIdError::Empty;
    }

    // ---- ProjectId::new ---------------------------------------------------

    #[test]
    fn project_id_mirrors_secret_id_rules() {
        assert!(ProjectId::new("proj.alpha-1").is_ok());
        assert_eq!(ProjectId::new(""), Err(SecretIdError::Empty));
        assert_eq!(
            ProjectId::new(".dot"),
            Err(SecretIdError::BadPrefix { ch: '.' })
        );
        assert_eq!(
            ProjectId::new("-dash"),
            Err(SecretIdError::BadPrefix { ch: '-' })
        );
        assert_eq!(
            ProjectId::new("space here"),
            Err(SecretIdError::InvalidChar { ch: ' ' })
        );
        let s = "a".repeat(65);
        assert_eq!(ProjectId::new(&s), Err(SecretIdError::TooLong { len: 65 }));
        let pid = ProjectId::new("ok").unwrap();
        assert_eq!(pid.as_str(), "ok");
        assert_eq!(pid.to_string(), "ok");
        assert_eq!(<ProjectId as AsRef<str>>::as_ref(&pid), "ok");
    }

    // ---- SecretRef Display ------------------------------------------------

    #[test]
    fn secret_ref_display_global() {
        let r = SecretRef::new(SecretScope::Global, sid("my-key"));
        assert_eq!(r.to_string(), "global:my-key");
    }

    #[test]
    fn secret_ref_display_project() {
        let r = SecretRef::new(SecretScope::Project { id: pid("foo") }, sid("my-key"));
        assert_eq!(r.to_string(), "project[foo]:my-key");
    }

    // ---- SecretRef serde round-trip --------------------------------------

    #[test]
    fn secret_ref_serde_global_roundtrip() {
        let r = SecretRef::new(SecretScope::Global, sid("token"));
        let json = serde_json::to_string(&r).unwrap();
        assert!(json.contains("\"kind\":\"global\""), "got {json}");
        assert!(json.contains("\"id\":\"token\""), "got {json}");
        let back: SecretRef = serde_json::from_str(&json).unwrap();
        assert_eq!(back, r);
    }

    #[test]
    fn secret_ref_serde_project_roundtrip() {
        let r = SecretRef::new(SecretScope::Project { id: pid("alpha") }, sid("token"));
        let json = serde_json::to_string(&r).unwrap();
        assert!(json.contains("\"kind\":\"project\""), "got {json}");
        assert!(json.contains("\"id\":\"alpha\""), "got {json}");
        let back: SecretRef = serde_json::from_str(&json).unwrap();
        assert_eq!(back, r);
    }

    #[test]
    fn secret_id_serde_rejects_invalid() {
        let bad = "\"-nope\"";
        let r: Result<SecretId, _> = serde_json::from_str(bad);
        assert!(r.is_err());
    }

    // ---- SecretValue ------------------------------------------------------

    #[test]
    fn secret_value_debug_redacts() {
        let v = SecretValue::from("hunter2");
        let s = format!("{v:?}");
        assert!(s.contains("redacted"), "got {s}");
        assert!(s.contains("7 bytes"), "got {s}");
        assert!(!s.contains("hunter2"), "leaked: {s}");
    }

    #[test]
    fn secret_value_expose_returns_bytes_verbatim() {
        let v = SecretValue::from("abc");
        assert_eq!(v.expose(), b"abc");
        let bytes: Vec<u8> = vec![0, 1, 2, 3, 255];
        let v2 = SecretValue::from(bytes.clone());
        assert_eq!(v2.expose(), bytes.as_slice());
        assert_eq!(v2.len(), 5);
        assert!(!v2.is_empty());
        assert!(SecretValue::from("").is_empty());
    }

    // ---- redact_for_log --------------------------------------------------

    #[test]
    fn redact_for_log_is_deterministic_and_distinguishing() {
        let a = SecretValue::from("alpha");
        let b = SecretValue::from("alpha");
        let c = SecretValue::from("beta");
        let ra = redact_for_log(&a);
        let rb = redact_for_log(&b);
        let rc = redact_for_log(&c);
        assert_eq!(ra, rb);
        assert_ne!(ra, rc);
        assert!(ra.starts_with("***"));
        assert_eq!(ra.len(), 3 + 8);
    }

    // ---- InMemorySecretStore round-trip ----------------------------------

    #[test]
    fn in_memory_store_round_trip() {
        let store = InMemorySecretStore::new();
        let r = SecretRef::new(SecretScope::Global, sid("k1"));

        // initially empty
        assert!(store.get(&r).unwrap().is_none());
        assert!(store.list(&SecretScope::Global).unwrap().is_empty());

        // put
        store.put(&r, SecretValue::from("v1")).unwrap();
        let got = store.get(&r).unwrap().unwrap();
        assert_eq!(got.expose(), b"v1");

        // list
        let listed = store.list(&SecretScope::Global).unwrap();
        assert_eq!(listed, vec![sid("k1")]);

        // delete
        assert!(store.delete(&r).unwrap());
        assert!(store.get(&r).unwrap().is_none());
        assert!(store.list(&SecretScope::Global).unwrap().is_empty());
    }

    #[test]
    fn in_memory_store_list_is_sorted_and_scope_isolated() {
        let store = InMemorySecretStore::new();
        let global = SecretScope::Global;
        let foo = SecretScope::Project { id: pid("foo") };
        let bar = SecretScope::Project { id: pid("bar") };

        store
            .put(
                &SecretRef::new(global.clone(), sid("b")),
                SecretValue::from("x"),
            )
            .unwrap();
        store
            .put(
                &SecretRef::new(global.clone(), sid("a")),
                SecretValue::from("x"),
            )
            .unwrap();
        store
            .put(
                &SecretRef::new(foo.clone(), sid("only-in-foo")),
                SecretValue::from("x"),
            )
            .unwrap();
        store
            .put(
                &SecretRef::new(bar.clone(), sid("only-in-bar")),
                SecretValue::from("x"),
            )
            .unwrap();

        assert_eq!(store.list(&global).unwrap(), vec![sid("a"), sid("b")]);
        assert_eq!(store.list(&foo).unwrap(), vec![sid("only-in-foo")]);
        assert_eq!(store.list(&bar).unwrap(), vec![sid("only-in-bar")]);
    }

    #[test]
    fn in_memory_store_delete_missing_returns_false() {
        let store = InMemorySecretStore::new();
        let r = SecretRef::new(SecretScope::Global, sid("ghost"));
        assert!(!store.delete(&r).unwrap());
    }

    #[test]
    fn in_memory_store_get_missing_returns_none() {
        let store = InMemorySecretStore::new();
        let r = SecretRef::new(SecretScope::Project { id: pid("p") }, sid("missing"));
        assert!(store.get(&r).unwrap().is_none());
    }

    #[test]
    fn in_memory_store_put_overwrites() {
        let store = InMemorySecretStore::new();
        let r = SecretRef::new(SecretScope::Global, sid("k"));
        store.put(&r, SecretValue::from("v1")).unwrap();
        store.put(&r, SecretValue::from("v2")).unwrap();
        let got = store.get(&r).unwrap().unwrap();
        assert_eq!(got.expose(), b"v2");
    }

    // ---- Trait-object Send + Sync confirmation ---------------------------

    #[test]
    fn store_is_send_sync_trait_object() {
        let store: Arc<dyn SecretStore> = Arc::new(InMemorySecretStore::new());
        let r = SecretRef::new(SecretScope::Global, sid("x"));
        store.put(&r, SecretValue::from("y")).unwrap();
        let clone = Arc::clone(&store);
        let handle = std::thread::spawn(move || {
            let got = clone.get(&r).unwrap().unwrap();
            got.expose().to_vec()
        });
        let bytes = handle.join().unwrap();
        assert_eq!(bytes, b"y");
    }

    // ---- SecretStoreError -------------------------------------------------

    #[test]
    fn secret_store_error_display_and_source() {
        let backend = SecretStoreError::Backend("boom".into());
        assert!(backend.to_string().contains("boom"));
        assert!(backend.source().is_none());

        let nf = SecretStoreError::NotFound;
        assert!(nf.to_string().contains("not found"));
        assert!(nf.source().is_none());

        let invalid = SecretStoreError::InvalidKey(SecretIdError::Empty);
        assert!(invalid.to_string().contains("invalid key"));
        assert!(invalid.source().is_some());
    }

    // ---- SecretScope ordering / hashing sanity ---------------------------

    #[test]
    fn secret_scope_serde_shapes() {
        let g = SecretScope::Global;
        let p = SecretScope::Project { id: pid("foo") };
        assert_eq!(serde_json::to_string(&g).unwrap(), r#"{"kind":"global"}"#);
        assert_eq!(
            serde_json::to_string(&p).unwrap(),
            r#"{"kind":"project","id":"foo"}"#
        );
    }
}
