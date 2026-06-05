//! Prefix prompt cache + reuse-key fingerprinting.
//!
//! See `plan/13-prompt-cache.md`. Re-tokenizing the system prompt and the
//! persistent agent header on every turn is wasteful — for a typical chat
//! turn the static prefix is far larger than the user's actual message.
//! This module gives providers a stable lookup key for the prefix and a
//! bounded LRU that stores the pre-tokenized prefix tokens so the provider
//! can skip straight to encoding the variable tail.
//!
//! # Layout
//!
//! - [`PromptHash`] — 64-character lowercase hex (sha-256) wrapper with
//!   strict construction.
//! - [`PromptCacheKey`] — `(model_slug, system_prompt_hash,
//!   agent_header_hash, ctx_size)` — the prefix is uniquely identified by
//!   these four fields. Two turns with the same key share a prefix.
//! - [`PromptCacheEntry`] — the pre-tokenized prefix: `tokens`, `n_tokens`,
//!   `created_at`, `last_used`, and a declared `bytes` footprint used for
//!   the cache's RAM budget.
//! - [`PromptCache`] — bounded LRU keyed by [`PromptCacheKey`]. Evicts by
//!   slot count **and** total byte budget. Mirrors the locking discipline
//!   of [`crate::provider_cache::ProviderCache`].
//! - [`fingerprint_inputs`] — convenience constructor that hashes the
//!   prefix inputs and assembles a [`PromptCacheKey`].
//!
//! # Locking
//!
//! Lock order, **never** reversed:
//!
//! 1. `entries`
//! 2. `lru`
//! 3. `used_bytes`
//!
//! Methods acquire only the locks they need, in that order. Poisoning is
//! recovered from rather than panicked on, matching the rest of the
//! runtime.
//!
//! # Eviction policy
//!
//! [`PromptCache::insert`] evicts oldest-first until both `len <=
//! capacity` **and** `used_bytes + entry.bytes <= max_total_bytes`. If a
//! single entry alone exceeds `max_total_bytes`, the call fails with
//! [`PromptCacheError::TooLarge`] and the cache state is unchanged.

use std::collections::{BTreeMap, VecDeque};
use std::error::Error;
use std::fmt::{self, Display, Formatter};
use std::sync::Mutex;
use std::time::SystemTime;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// 64-character lowercase hex sha-256 digest.
///
/// Used both as the digest of the system prompt and of the agent header
/// in [`PromptCacheKey`]. Construction always validates the invariant
/// (length == 64, all chars in `[0-9a-f]`).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct PromptHash(String);

impl PromptHash {
    /// Compute a fresh sha-256 over `text` and wrap the lowercase hex
    /// digest.
    #[must_use]
    pub fn from_text(text: &str) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(text.as_bytes());
        let digest = hasher.finalize();
        let mut hex = String::with_capacity(64);
        for byte in digest {
            // {:02x} is the documented lowercase-hex formatter for u8.
            let _ = fmt::write(&mut hex, format_args!("{byte:02x}"));
        }
        Self(hex)
    }

    /// Wrap an existing hex digest. Validates length and alphabet.
    ///
    /// # Errors
    ///
    /// - [`PromptHashError::WrongLength`] when `hex.len() != 64`.
    /// - [`PromptHashError::BadHex`] when any character is outside the
    ///   `[0-9a-f]` alphabet (uppercase is rejected — the cache key relies
    ///   on a canonical form).
    pub fn from_hex(hex: &str) -> Result<Self, PromptHashError> {
        if hex.len() != 64 {
            return Err(PromptHashError::WrongLength { actual: hex.len() });
        }
        for ch in hex.chars() {
            if !ch.is_ascii_digit() && !matches!(ch, 'a'..='f') {
                return Err(PromptHashError::BadHex);
            }
        }
        Ok(Self(hex.to_owned()))
    }

    /// Borrow the underlying lowercase hex string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Display for PromptHash {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Failure modes for [`PromptHash::from_hex`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PromptHashError {
    /// The candidate hex was not exactly 64 characters long.
    WrongLength {
        /// The candidate's length, for debugging.
        actual: usize,
    },
    /// The candidate contained a non-`[0-9a-f]` character.
    BadHex,
}

impl Display for PromptHashError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongLength { actual } => write!(
                f,
                "prompt hash must be 64 lowercase hex chars, got {actual}"
            ),
            Self::BadHex => f.write_str("prompt hash contains a non lowercase hex character"),
        }
    }
}

impl Error for PromptHashError {}

/// Lookup key for a cached prefix.
///
/// Two prefixes from the same `(model_slug, system_prompt_hash,
/// agent_header_hash, ctx_size)` are interchangeable — the provider may
/// reuse the cached tokens without re-tokenizing.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct PromptCacheKey {
    /// Stable model identifier, e.g. `"qwen-2.5-7b-instruct"`.
    pub model_slug: String,
    /// Hash of the system prompt text.
    pub system_prompt_hash: PromptHash,
    /// Hash of the persistent agent header text.
    pub agent_header_hash: PromptHash,
    /// Context window the prefix was tokenized for. A different `ctx_size`
    /// implies a potentially different BOS / chat-template framing, so it
    /// participates in the key.
    pub ctx_size: u32,
}

/// A live cache entry: the pre-tokenized prefix and accounting metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromptCacheEntry {
    /// The prefix tokens, ready to be fed straight into the provider's
    /// KV cache.
    pub tokens: Vec<u32>,
    /// Cached `tokens.len()` as a `u32` for the provider's bookkeeping.
    pub n_tokens: u32,
    /// When the entry was inserted.
    pub created_at: SystemTime,
    /// When the entry was last read via [`PromptCache::get`].
    pub last_used: SystemTime,
    /// Declared RAM footprint in bytes. Treated as the source of truth
    /// for budgeting — the cache does not measure it independently.
    pub bytes: u64,
}

/// Failure modes for [`PromptCache::insert`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PromptCacheError {
    /// The new entry's `bytes` alone exceeds `max_total_bytes`; no amount
    /// of eviction can make it fit.
    TooLarge {
        /// The candidate's footprint, in bytes.
        entry_bytes: u64,
        /// The cache's configured total byte budget.
        max_total_bytes: u64,
    },
    /// The key is already present. The caller must
    /// [`evict`](PromptCache::evict) the existing entry first.
    AlreadyPresent {
        /// The key that collided.
        key: PromptCacheKey,
    },
}

impl Display for PromptCacheError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooLarge {
                entry_bytes,
                max_total_bytes,
            } => write!(
                f,
                "prompt cache entry of {entry_bytes} bytes exceeds total budget {max_total_bytes} bytes"
            ),
            Self::AlreadyPresent { key } => write!(
                f,
                "prompt cache already contains entry for model {}",
                key.model_slug
            ),
        }
    }
}

impl Error for PromptCacheError {}

/// Bounded LRU cache of pre-tokenized prompt prefixes.
///
/// See the module-level docs for the locking discipline and eviction
/// semantics.
#[derive(Debug)]
pub struct PromptCache {
    entries: Mutex<BTreeMap<PromptCacheKey, PromptCacheEntry>>,
    lru: Mutex<VecDeque<PromptCacheKey>>,
    capacity: usize,
    max_total_bytes: u64,
    used_bytes: Mutex<u64>,
}

impl PromptCache {
    /// Build an empty cache with the given slot capacity and total byte
    /// budget.
    #[must_use]
    pub const fn new(capacity: usize, max_total_bytes: u64) -> Self {
        Self {
            entries: Mutex::new(BTreeMap::new()),
            lru: Mutex::new(VecDeque::new()),
            capacity,
            max_total_bytes,
            used_bytes: Mutex::new(0),
        }
    }

    /// Look up `key`. On hit, touches the LRU (moving the entry to the
    /// most-recently-used end), stamps `last_used` to now, and returns a
    /// clone of the stored entry.
    #[must_use]
    pub fn get(&self, key: &PromptCacheKey) -> Option<PromptCacheEntry> {
        let mut entries = lock(&self.entries);
        let slot = entries.get_mut(key)?;
        slot.last_used = SystemTime::now();
        let entry = slot.clone();
        drop(entries);
        let mut lru = lock(&self.lru);
        if let Some(pos) = lru.iter().position(|existing| existing == key) {
            let _ = lru.remove(pos);
        }
        lru.push_back(key.clone());
        drop(lru);
        Some(entry)
    }

    /// Insert `entry` under `key`.
    ///
    /// Evicts in LRU order (oldest first) until both `len <= capacity`
    /// **and** `used_bytes + entry.bytes <= max_total_bytes`. Returns the
    /// keys that were evicted so the caller can drop their handles
    /// outside the cache's locks.
    ///
    /// # Errors
    ///
    /// - [`PromptCacheError::TooLarge`] when `entry.bytes` alone exceeds
    ///   `max_total_bytes`.
    /// - [`PromptCacheError::AlreadyPresent`] when `key` is already in
    ///   the cache; the caller must [`evict`](Self::evict) first.
    pub fn insert(
        &self,
        key: PromptCacheKey,
        entry: PromptCacheEntry,
    ) -> Result<Vec<PromptCacheKey>, PromptCacheError> {
        if entry.bytes > self.max_total_bytes {
            return Err(PromptCacheError::TooLarge {
                entry_bytes: entry.bytes,
                max_total_bytes: self.max_total_bytes,
            });
        }

        let mut entries = lock(&self.entries);
        if entries.contains_key(&key) {
            return Err(PromptCacheError::AlreadyPresent { key });
        }
        let mut lru = lock(&self.lru);
        let mut used = lock(&self.used_bytes);

        let mut evicted = Vec::new();
        let new_bytes = entry.bytes;

        loop {
            let len_after_insert = entries.len().saturating_add(1);
            let bytes_after_insert = used.saturating_add(new_bytes);
            let over_count = len_after_insert > self.capacity;
            let over_bytes = bytes_after_insert > self.max_total_bytes;
            if !over_count && !over_bytes {
                break;
            }
            let Some(victim) = lru.pop_front() else {
                // Single-entry guard at the top ensures a freshly drained
                // cache always fits — defend without panicking.
                break;
            };
            if let Some(removed) = entries.remove(&victim) {
                *used = used.saturating_sub(removed.bytes);
                evicted.push(victim);
            }
        }

        *used = used.saturating_add(new_bytes);
        let _previous = entries.insert(key.clone(), entry);
        lru.push_back(key);

        drop(used);
        drop(lru);
        drop(entries);
        Ok(evicted)
    }

    /// Manually evict `key`, returning its entry if present.
    pub fn evict(&self, key: &PromptCacheKey) -> Option<PromptCacheEntry> {
        let mut entries = lock(&self.entries);
        let entry = entries.remove(key)?;
        let mut lru = lock(&self.lru);
        if let Some(pos) = lru.iter().position(|existing| existing == key) {
            let _ = lru.remove(pos);
        }
        let mut used = lock(&self.used_bytes);
        *used = used.saturating_sub(entry.bytes);
        drop(used);
        drop(lru);
        drop(entries);
        Some(entry)
    }

    /// Drain every entry, returning the keys in LRU order (oldest first).
    pub fn clear(&self) -> Vec<PromptCacheKey> {
        let mut entries = lock(&self.entries);
        let mut lru = lock(&self.lru);
        let mut used = lock(&self.used_bytes);
        let drained: Vec<_> = lru.drain(..).collect();
        entries.clear();
        *used = 0;
        drop(used);
        drop(lru);
        drop(entries);
        drained
    }

    /// Number of entries currently held.
    #[must_use]
    pub fn len(&self) -> usize {
        lock(&self.entries).len()
    }

    /// Whether the cache holds zero entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        lock(&self.entries).is_empty()
    }

    /// Sum of `bytes` across all currently held entries.
    #[must_use]
    pub fn used_bytes(&self) -> u64 {
        *lock(&self.used_bytes)
    }

    /// Snapshot of `(key, bytes)` pairs in LRU order, oldest first.
    #[must_use]
    pub fn snapshot(&self) -> Vec<(PromptCacheKey, u64)> {
        let entries = lock(&self.entries);
        let lru = lock(&self.lru);
        let out: Vec<_> = lru
            .iter()
            .filter_map(|key| entries.get(key).map(|entry| (key.clone(), entry.bytes)))
            .collect();
        drop(lru);
        drop(entries);
        out
    }
}

/// Build a [`PromptCacheKey`] from the raw prefix inputs.
///
/// Hashes `system` and `agent_header` independently so the cache can
/// distinguish "different system prompt, same agent header" from "same
/// system prompt, different agent header" — both common shapes during
/// agent reconfiguration.
#[must_use]
pub fn fingerprint_inputs(
    model_slug: &str,
    system: &str,
    agent_header: &str,
    ctx_size: u32,
) -> PromptCacheKey {
    PromptCacheKey {
        model_slug: model_slug.to_owned(),
        system_prompt_hash: PromptHash::from_text(system),
        agent_header_hash: PromptHash::from_text(agent_header),
        ctx_size,
    }
}

/// Lock helper that recovers from poisoning rather than panicking,
/// matching the runtime's convention.
fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Barrier};
    use std::thread;

    fn key(slug: &str, sys: &str, agent: &str, ctx: u32) -> PromptCacheKey {
        fingerprint_inputs(slug, sys, agent, ctx)
    }

    fn entry(bytes: u64, n_tokens: u32) -> PromptCacheEntry {
        let now = SystemTime::now();
        PromptCacheEntry {
            tokens: (0..n_tokens).collect(),
            n_tokens,
            created_at: now,
            last_used: now,
            bytes,
        }
    }

    #[test]
    fn prompt_hash_from_text_is_64_hex() {
        let h = PromptHash::from_text("hello");
        let s = h.as_str();
        assert_eq!(s.len(), 64);
        assert!(s
            .chars()
            .all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c)));
    }

    #[test]
    fn prompt_hash_from_hex_happy() {
        let raw = PromptHash::from_text("hello").as_str().to_owned();
        let h = PromptHash::from_hex(&raw).expect("happy");
        assert_eq!(h.as_str(), raw);
        assert_eq!(format!("{h}"), raw);
    }

    #[test]
    fn prompt_hash_from_hex_rejects_wrong_length() {
        let err = PromptHash::from_hex("abc").expect_err("too short");
        assert_eq!(err, PromptHashError::WrongLength { actual: 3 });
    }

    #[test]
    fn prompt_hash_from_hex_rejects_non_hex() {
        // 64 chars but one is uppercase / out of alphabet.
        let mut bad = "a".repeat(63);
        bad.push('Z');
        let err = PromptHash::from_hex(&bad).expect_err("bad hex");
        assert_eq!(err, PromptHashError::BadHex);
    }

    fn assert_error<E: std::error::Error>() {}

    #[test]
    fn prompt_hash_error_display_smoke() {
        let wrong = PromptHashError::WrongLength { actual: 7 };
        assert!(format!("{wrong}").contains('7'));
        let bad = PromptHashError::BadHex;
        assert!(format!("{bad}").contains("hex"));
        assert_error::<PromptHashError>();
    }

    #[test]
    fn prompt_cache_key_ord_stable() {
        let mut v = vec![
            key("b", "sys", "agent", 4096),
            key("a", "sys", "agent", 8192),
            key("a", "sys", "agent", 4096),
            key("c", "sys", "agent", 4096),
        ];
        v.sort();
        // Sorted by model_slug then ctx (other fields constant).
        assert_eq!(v[0].model_slug, "a");
        assert_eq!(v[0].ctx_size, 4096);
        assert_eq!(v[1].model_slug, "a");
        assert_eq!(v[1].ctx_size, 8192);
        assert_eq!(v[2].model_slug, "b");
        assert_eq!(v[3].model_slug, "c");
    }

    #[test]
    fn new_is_empty() {
        let c = PromptCache::new(4, 1_000_000);
        assert_eq!(c.len(), 0);
        assert!(c.is_empty());
        assert_eq!(c.used_bytes(), 0);
        assert!(c.snapshot().is_empty());
    }

    #[test]
    fn insert_happy_path() {
        let c = PromptCache::new(4, 10_000);
        let k = key("qwen", "sys", "agent", 4096);
        let evicted = c.insert(k, entry(100, 25)).expect("insert");
        assert!(evicted.is_empty());
        assert_eq!(c.len(), 1);
        assert_eq!(c.used_bytes(), 100);
    }

    #[test]
    fn get_returns_clone_of_entry() {
        let c = PromptCache::new(4, 10_000);
        let k = key("qwen", "sys", "agent", 4096);
        c.insert(k.clone(), entry(120, 30)).expect("insert");
        let got = c.get(&k).expect("hit");
        assert_eq!(got.n_tokens, 30);
        assert_eq!(got.tokens.len(), 30);
        assert_eq!(got.bytes, 120);
    }

    #[test]
    fn get_updates_lru_order() {
        let c = PromptCache::new(4, 10_000);
        let a = key("a", "sys", "agent", 4096);
        let b = key("b", "sys", "agent", 4096);
        c.insert(a.clone(), entry(10, 1)).expect("a");
        c.insert(b.clone(), entry(10, 1)).expect("b");
        assert_eq!(
            c.snapshot().into_iter().map(|(k, _)| k).collect::<Vec<_>>(),
            vec![a.clone(), b.clone()]
        );
        let _ = c.get(&a);
        assert_eq!(
            c.snapshot().into_iter().map(|(k, _)| k).collect::<Vec<_>>(),
            vec![b, a]
        );
    }

    #[test]
    fn get_missing_returns_none() {
        let c = PromptCache::new(4, 10_000);
        assert!(c.get(&key("nope", "sys", "agent", 4096)).is_none());
    }

    #[test]
    fn insert_evicts_when_over_capacity() {
        let c = PromptCache::new(2, 1_000_000);
        let a = key("a", "sys", "agent", 4096);
        let b = key("b", "sys", "agent", 4096);
        let d = key("d", "sys", "agent", 4096);
        c.insert(a.clone(), entry(10, 1)).expect("a");
        c.insert(b.clone(), entry(10, 1)).expect("b");
        let evicted = c.insert(d.clone(), entry(10, 1)).expect("d");
        assert_eq!(evicted, vec![a]);
        assert_eq!(c.len(), 2);
        let keys: Vec<_> = c.snapshot().into_iter().map(|(k, _)| k).collect();
        assert_eq!(keys, vec![b, d]);
    }

    #[test]
    fn insert_evicts_when_bytes_budget_exceeded() {
        let c = PromptCache::new(10, 100);
        let a = key("a", "sys", "agent", 4096);
        let b = key("b", "sys", "agent", 4096);
        let d = key("d", "sys", "agent", 4096);
        c.insert(a.clone(), entry(40, 1)).expect("a");
        c.insert(b.clone(), entry(40, 1)).expect("b");
        // 80 used, total 100 — inserting 30 needs to evict a (oldest).
        let evicted = c.insert(d.clone(), entry(30, 1)).expect("d");
        assert_eq!(evicted, vec![a]);
        assert_eq!(c.used_bytes(), 70);
        let keys: Vec<_> = c.snapshot().into_iter().map(|(k, _)| k).collect();
        assert_eq!(keys, vec![b, d]);
    }

    #[test]
    fn insert_oversized_returns_too_large() {
        let c = PromptCache::new(4, 100);
        let err = c
            .insert(key("a", "sys", "agent", 4096), entry(200, 1))
            .expect_err("must fail");
        assert_eq!(
            err,
            PromptCacheError::TooLarge {
                entry_bytes: 200,
                max_total_bytes: 100,
            }
        );
        assert!(c.is_empty());
        assert_eq!(c.used_bytes(), 0);
    }

    #[test]
    fn insert_duplicate_returns_already_present() {
        let c = PromptCache::new(4, 10_000);
        let k = key("a", "sys", "agent", 4096);
        c.insert(k.clone(), entry(10, 1)).expect("first");
        let err = c.insert(k.clone(), entry(10, 1)).expect_err("dup");
        assert_eq!(err, PromptCacheError::AlreadyPresent { key: k });
    }

    #[test]
    fn evict_removes_and_updates_used_bytes() {
        let c = PromptCache::new(4, 10_000);
        let k = key("a", "sys", "agent", 4096);
        c.insert(k.clone(), entry(64, 1)).expect("insert");
        assert_eq!(c.used_bytes(), 64);
        let removed = c.evict(&k).expect("evicted");
        assert_eq!(removed.bytes, 64);
        assert!(c.is_empty());
        assert_eq!(c.used_bytes(), 0);
    }

    #[test]
    fn evict_missing_returns_none() {
        let c = PromptCache::new(4, 10_000);
        assert!(c.evict(&key("nope", "sys", "agent", 4096)).is_none());
    }

    #[test]
    fn clear_drains_and_returns_all_keys() {
        let c = PromptCache::new(4, 10_000);
        let a = key("a", "sys", "agent", 4096);
        let b = key("b", "sys", "agent", 4096);
        let d = key("d", "sys", "agent", 4096);
        c.insert(a.clone(), entry(10, 1)).expect("a");
        c.insert(b.clone(), entry(10, 1)).expect("b");
        c.insert(d.clone(), entry(10, 1)).expect("d");
        let drained = c.clear();
        assert_eq!(drained, vec![a, b, d]);
        assert!(c.is_empty());
        assert_eq!(c.used_bytes(), 0);
        assert!(c.snapshot().is_empty());
    }

    #[test]
    fn snapshot_ordered_oldest_first() {
        let c = PromptCache::new(4, 10_000);
        let a = key("a", "sys", "agent", 4096);
        let b = key("b", "sys", "agent", 4096);
        let d = key("d", "sys", "agent", 4096);
        c.insert(a.clone(), entry(1, 1)).expect("a");
        c.insert(b.clone(), entry(1, 1)).expect("b");
        c.insert(d.clone(), entry(1, 1)).expect("d");
        let keys: Vec<_> = c.snapshot().into_iter().map(|(k, _)| k).collect();
        assert_eq!(keys, vec![a, b, d]);
    }

    #[test]
    fn fingerprint_inputs_deterministic() {
        let k1 = fingerprint_inputs("qwen", "sys", "agent", 4096);
        let k2 = fingerprint_inputs("qwen", "sys", "agent", 4096);
        assert_eq!(k1, k2);
    }

    #[test]
    fn fingerprint_inputs_distinct() {
        let base = fingerprint_inputs("qwen", "sys", "agent", 4096);
        let diff_model = fingerprint_inputs("llama", "sys", "agent", 4096);
        let diff_sys = fingerprint_inputs("qwen", "SYS", "agent", 4096);
        let diff_agent = fingerprint_inputs("qwen", "sys", "AGENT", 4096);
        let diff_ctx = fingerprint_inputs("qwen", "sys", "agent", 8192);
        assert_ne!(base, diff_model);
        assert_ne!(base, diff_sys);
        assert_ne!(base, diff_agent);
        assert_ne!(base, diff_ctx);
    }

    #[test]
    fn prompt_cache_error_display_smoke() {
        let too_large = PromptCacheError::TooLarge {
            entry_bytes: 8192,
            max_total_bytes: 1024,
        };
        let s = format!("{too_large}");
        assert!(s.contains("8192"));
        assert!(s.contains("1024"));
        let already = PromptCacheError::AlreadyPresent {
            key: key("qwen", "sys", "agent", 4096),
        };
        let s = format!("{already}");
        assert!(s.contains("qwen"));
        assert_error::<PromptCacheError>();
    }

    #[test]
    fn prompt_cache_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<PromptCache>();
    }

    #[test]
    fn prompt_cache_key_serde_roundtrip() {
        let k = key("qwen", "sys", "agent", 4096);
        let s = serde_json::to_string(&k).expect("ser");
        let back: PromptCacheKey = serde_json::from_str(&s).expect("de");
        assert_eq!(k, back);
    }

    #[test]
    fn prompt_cache_entry_serde_roundtrip() {
        let e = entry(128, 4);
        let s = serde_json::to_string(&e).expect("ser");
        let back: PromptCacheEntry = serde_json::from_str(&s).expect("de");
        assert_eq!(e, back);
    }

    #[test]
    fn concurrent_fuzz_no_deadlock() {
        use std::sync::atomic::{AtomicU64, Ordering};

        const THREADS: usize = 4;
        const OPS_PER_THREAD: usize = 100;

        let cache = Arc::new(PromptCache::new(4, 400));
        let barrier = Arc::new(Barrier::new(THREADS));
        let seed_counter = Arc::new(AtomicU64::new(0xc0ff_ee01));

        let mut handles = Vec::new();
        for tid in 0..THREADS {
            let c = Arc::clone(&cache);
            let b = Arc::clone(&barrier);
            let seeds = Arc::clone(&seed_counter);
            handles.push(thread::spawn(move || {
                b.wait();
                for op in 0..OPS_PER_THREAD {
                    let s = seeds.fetch_add(1, Ordering::Relaxed);
                    let bucket =
                        (s.wrapping_mul(2_654_435_761) ^ (tid as u64)).wrapping_add(op as u64);
                    let slug_idx = bucket % 6;
                    let bytes = 30 + (bucket % 60);
                    let k = fingerprint_inputs(&format!("m{slug_idx}"), "sys", "agent", 4096);
                    match bucket % 4 {
                        0 => {
                            let _ = c.insert(k, entry(bytes, 2));
                        }
                        1 => {
                            let _ = c.get(&k);
                        }
                        2 => {
                            let _ = c.evict(&k);
                        }
                        _ => {
                            let _ = c.snapshot();
                        }
                    }
                }
            }));
        }

        for h in handles {
            h.join().expect("thread join");
        }

        assert!(cache.len() <= 4);
        assert!(cache.used_bytes() <= 400);
        assert_eq!(cache.snapshot().len(), cache.len());
    }
}
