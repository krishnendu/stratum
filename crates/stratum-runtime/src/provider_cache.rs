//! Provider warm-up cache + handle pool.
//!
//! See `plan/06-providers.md` §4. The orchestrator may serve many distinct
//! `(model, variant)` pairs over a session, but only a small set fits in
//! RAM at once. This module keeps a bounded LRU of loaded
//! [`Provider`](crate::provider::Provider) handles and evicts the
//! least-recently-used entry when either the slot count or the RAM
//! footprint exceeds the configured budget.
//!
//! # Layout
//!
//! - [`ProviderCache`] — the cache itself: a `BTreeMap` of slots keyed by
//!   [`ProviderKey`] plus a `VecDeque` recording LRU order.
//! - [`ProviderKey`] — the `(model_slug, variant)` lookup key.
//! - [`CacheSlot`] — what the cache stores: an `Arc<dyn Provider>`
//!   together with its declared footprint in MiB and timestamps.
//! - [`CacheError`] — the two failure modes callers must handle.
//!
//! # Locking
//!
//! All mutations take an internal lock order, **never** in reverse:
//!
//! 1. `entries`
//! 2. `lru`
//! 3. `used_ram_mib`
//!
//! Methods that need multiple of these locks acquire them in that order
//! and drop them as soon as their critical section is done. This is
//! exercised by the `concurrent_fuzz_no_deadlock` test which spins up
//! four threads doing one hundred random insert/get/evict cycles each
//! and asserts the cache invariants hold at the end.
//!
//! # Eviction policy
//!
//! [`insert`](ProviderCache::insert) evicts (oldest-first) until both
//! `len <= capacity` **and**
//! `used_ram_mib + new_footprint <= total_ram_budget_mib`. If a single
//! new slot's footprint alone exceeds the total budget, insertion fails
//! with [`CacheError::DoesNotFit`] and no state changes.

use std::collections::{BTreeMap, VecDeque};
use std::error::Error;
use std::fmt::{self, Display, Formatter};
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

use serde::{Deserialize, Serialize};

use crate::provider::Provider;

/// Lookup key for a cached provider handle.
///
/// Two providers backing the same `(model_slug, variant)` are
/// considered interchangeable from the cache's point of view; callers
/// that need to distinguish e.g. quantization level should fold that
/// into `variant`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ProviderKey {
    /// Stable model identifier, e.g. `"qwen-2.5-7b-instruct"`.
    pub model_slug: String,
    /// Variant discriminator, e.g. `"q4_k_m"` or `"fp16"`.
    pub variant: String,
}

impl ProviderKey {
    /// Build a new key from its two components.
    #[must_use]
    pub fn new(model_slug: impl Into<String>, variant: impl Into<String>) -> Self {
        Self {
            model_slug: model_slug.into(),
            variant: variant.into(),
        }
    }
}

/// A live cache entry: the provider handle, its declared RAM footprint
/// and timestamps for loaded-at and last-used.
#[derive(Debug)]
pub struct CacheSlot {
    /// Shared, thread-safe handle to the loaded provider.
    pub handle: Arc<dyn Provider>,
    /// Declared RAM footprint in MiB. Treated as the source of truth
    /// for budgeting — the cache does not measure it independently.
    pub footprint_mib: u64,
    /// When the slot was constructed.
    pub loaded_at: SystemTime,
    /// When the slot was last handed out via
    /// [`ProviderCache::get`]. Updated under the cache's internal lock.
    pub last_used: Mutex<SystemTime>,
}

impl CacheSlot {
    /// Build a fresh slot. `loaded_at` and `last_used` are stamped to
    /// the current `SystemTime`.
    #[must_use]
    pub fn new(handle: Arc<dyn Provider>, footprint_mib: u64) -> Self {
        let now = SystemTime::now();
        Self {
            handle,
            footprint_mib,
            loaded_at: now,
            last_used: Mutex::new(now),
        }
    }
}

/// Failure modes for [`ProviderCache::insert`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CacheError {
    /// The new slot's footprint alone exceeds the total RAM budget;
    /// no amount of eviction can make it fit.
    DoesNotFit {
        /// The candidate slot's footprint, in MiB.
        footprint: u64,
        /// The cache's configured total budget, in MiB.
        budget: u64,
    },
    /// The key is already present. The caller must
    /// [`evict`](ProviderCache::evict) the existing entry first.
    AlreadyPresent {
        /// The key that collided.
        key: ProviderKey,
    },
}

impl Display for CacheError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::DoesNotFit { footprint, budget } => write!(
                f,
                "provider cache slot footprint {footprint} MiB exceeds total budget {budget} MiB"
            ),
            Self::AlreadyPresent { key } => write!(
                f,
                "provider cache already contains {}/{}",
                key.model_slug, key.variant
            ),
        }
    }
}

impl Error for CacheError {}

/// Bounded LRU cache of loaded [`Provider`] handles.
///
/// See the module-level docs for the locking discipline and eviction
/// semantics.
#[derive(Debug)]
pub struct ProviderCache {
    entries: Mutex<BTreeMap<ProviderKey, CacheSlot>>,
    lru: Mutex<VecDeque<ProviderKey>>,
    capacity: usize,
    total_ram_budget_mib: u64,
    used_ram_mib: Mutex<u64>,
}

impl ProviderCache {
    /// Build an empty cache with the given slot capacity and total RAM
    /// budget (MiB).
    #[must_use]
    pub const fn new(capacity: usize, total_ram_budget_mib: u64) -> Self {
        Self {
            entries: Mutex::new(BTreeMap::new()),
            lru: Mutex::new(VecDeque::new()),
            capacity,
            total_ram_budget_mib,
            used_ram_mib: Mutex::new(0),
        }
    }

    /// Look up `key`. On hit, touches the LRU (moving the entry to the
    /// most-recently-used end) and stamps `last_used` to now.
    #[must_use]
    pub fn get(&self, key: &ProviderKey) -> Option<Arc<dyn Provider>> {
        let handle = {
            let entries = lock(&self.entries);
            let Some(slot) = entries.get(key) else {
                drop(entries);
                return None;
            };
            if let Ok(mut stamp) = slot.last_used.lock() {
                *stamp = SystemTime::now();
            }
            let cloned = Arc::clone(&slot.handle);
            drop(entries);
            cloned
        };
        let mut lru = lock(&self.lru);
        if let Some(pos) = lru.iter().position(|existing| existing == key) {
            let _ = lru.remove(pos);
        }
        lru.push_back(key.clone());
        drop(lru);
        Some(handle)
    }

    /// Insert `slot` under `key`.
    ///
    /// Evicts in LRU order (oldest first) until both `len <= capacity`
    /// **and** `used_ram_mib + new_footprint <= total_ram_budget_mib`.
    /// Returns the keys that were evicted so the caller can drop their
    /// handles outside the cache's locks.
    ///
    /// # Errors
    ///
    /// - [`CacheError::DoesNotFit`] when the slot footprint alone
    ///   exceeds `total_ram_budget_mib`.
    /// - [`CacheError::AlreadyPresent`] when `key` is already in the
    ///   cache; the caller must [`evict`](Self::evict) first.
    pub fn insert(
        &self,
        key: ProviderKey,
        slot: CacheSlot,
    ) -> Result<Vec<ProviderKey>, CacheError> {
        if slot.footprint_mib > self.total_ram_budget_mib {
            return Err(CacheError::DoesNotFit {
                footprint: slot.footprint_mib,
                budget: self.total_ram_budget_mib,
            });
        }

        let mut entries = lock(&self.entries);
        if entries.contains_key(&key) {
            return Err(CacheError::AlreadyPresent { key });
        }
        let mut lru = lock(&self.lru);
        let mut used = lock(&self.used_ram_mib);

        let mut evicted = Vec::new();
        let new_footprint = slot.footprint_mib;

        loop {
            let len_after_insert = entries.len().saturating_add(1);
            let ram_after_insert = used.saturating_add(new_footprint);
            let over_count = len_after_insert > self.capacity;
            let over_ram = ram_after_insert > self.total_ram_budget_mib;
            if !over_count && !over_ram {
                break;
            }
            // Pop oldest from LRU.
            let Some(victim) = lru.pop_front() else {
                // Nothing left to evict but we still can't fit.
                // This is unreachable in practice — the
                // single-footprint check at the top guarantees a
                // freshly emptied cache always fits the new slot —
                // but we defend it without panicking.
                break;
            };
            if let Some(removed) = entries.remove(&victim) {
                *used = used.saturating_sub(removed.footprint_mib);
                evicted.push(victim);
            }
        }

        *used = used.saturating_add(new_footprint);
        let _previous = entries.insert(key.clone(), slot);
        lru.push_back(key);

        drop(used);
        drop(lru);
        drop(entries);
        Ok(evicted)
    }

    /// Manually evict `key`, returning its slot if present.
    pub fn evict(&self, key: &ProviderKey) -> Option<CacheSlot> {
        let mut entries = lock(&self.entries);
        let slot = entries.remove(key)?;
        let mut lru = lock(&self.lru);
        if let Some(pos) = lru.iter().position(|existing| existing == key) {
            let _ = lru.remove(pos);
        }
        let mut used = lock(&self.used_ram_mib);
        *used = used.saturating_sub(slot.footprint_mib);
        drop(used);
        drop(lru);
        drop(entries);
        Some(slot)
    }

    /// Number of slots currently held.
    #[must_use]
    pub fn len(&self) -> usize {
        lock(&self.entries).len()
    }

    /// Whether the cache holds zero slots.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        lock(&self.entries).is_empty()
    }

    /// Sum of `footprint_mib` across all currently held slots.
    #[must_use]
    pub fn used_ram_mib(&self) -> u64 {
        *lock(&self.used_ram_mib)
    }

    /// Snapshot of `(key, footprint_mib)` pairs in LRU order, oldest
    /// first.
    #[must_use]
    pub fn snapshot(&self) -> Vec<(ProviderKey, u64)> {
        let entries = lock(&self.entries);
        let lru = lock(&self.lru);
        let out: Vec<_> = lru
            .iter()
            .filter_map(|key| entries.get(key).map(|slot| (key.clone(), slot.footprint_mib)))
            .collect();
        drop(lru);
        drop(entries);
        out
    }
}

/// Lock helper that recovers from poisoning rather than panicking,
/// matching the runtime's existing convention (see `rate_limit.rs`).
fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::EchoProvider;
    use std::sync::{Arc, Barrier};
    use std::thread;

    fn echo() -> Arc<dyn Provider> {
        Arc::new(EchoProvider::new(""))
    }

    fn key(slug: &str, variant: &str) -> ProviderKey {
        ProviderKey::new(slug, variant)
    }

    #[test]
    fn new_is_empty() {
        let c = ProviderCache::new(4, 1024);
        assert_eq!(c.len(), 0);
        assert!(c.is_empty());
        assert_eq!(c.used_ram_mib(), 0);
        assert!(c.snapshot().is_empty());
    }

    #[test]
    fn insert_happy_path() {
        let c = ProviderCache::new(4, 1024);
        let evicted = c
            .insert(key("a", "v1"), CacheSlot::new(echo(), 100))
            .expect("insert");
        assert!(evicted.is_empty());
        assert_eq!(c.len(), 1);
        assert_eq!(c.used_ram_mib(), 100);
    }

    #[test]
    fn get_returns_same_arc_identity() {
        let c = ProviderCache::new(4, 1024);
        let handle = echo();
        let k = key("a", "v1");
        c.insert(k.clone(), CacheSlot::new(Arc::clone(&handle), 50))
            .expect("insert");
        let got = c.get(&k).expect("hit");
        assert!(Arc::ptr_eq(&got, &handle));
    }

    #[test]
    fn get_misses_for_unknown_key() {
        let c = ProviderCache::new(4, 1024);
        assert!(c.get(&key("nope", "v1")).is_none());
    }

    #[test]
    fn get_updates_last_used_and_lru_order() {
        let c = ProviderCache::new(4, 1024);
        let a = key("a", "v1");
        let b = key("b", "v1");
        c.insert(a.clone(), CacheSlot::new(echo(), 10))
            .expect("insert a");
        c.insert(b.clone(), CacheSlot::new(echo(), 10))
            .expect("insert b");
        // LRU before touch: [a, b]
        assert_eq!(
            c.snapshot().into_iter().map(|(k, _)| k).collect::<Vec<_>>(),
            vec![a.clone(), b.clone()]
        );
        let _ = c.get(&a);
        // LRU after touching a: [b, a]
        assert_eq!(
            c.snapshot().into_iter().map(|(k, _)| k).collect::<Vec<_>>(),
            vec![b, a]
        );
    }

    #[test]
    fn insert_oversized_returns_does_not_fit() {
        let c = ProviderCache::new(4, 100);
        let err = c
            .insert(key("a", "v1"), CacheSlot::new(echo(), 200))
            .expect_err("must fail");
        assert_eq!(
            err,
            CacheError::DoesNotFit {
                footprint: 200,
                budget: 100,
            }
        );
        assert!(c.is_empty());
        assert_eq!(c.used_ram_mib(), 0);
    }

    #[test]
    fn insert_over_capacity_evicts_lru() {
        let c = ProviderCache::new(2, 10_000);
        let a = key("a", "v1");
        let b = key("b", "v1");
        let d = key("d", "v1");
        c.insert(a.clone(), CacheSlot::new(echo(), 10)).expect("a");
        c.insert(b.clone(), CacheSlot::new(echo(), 10)).expect("b");
        let evicted = c.insert(d.clone(), CacheSlot::new(echo(), 10)).expect("d");
        assert_eq!(evicted, vec![a]);
        assert_eq!(c.len(), 2);
        let keys: Vec<_> = c.snapshot().into_iter().map(|(k, _)| k).collect();
        assert_eq!(keys, vec![b, d]);
    }

    #[test]
    fn insert_over_ram_budget_evicts() {
        let c = ProviderCache::new(10, 100);
        let a = key("a", "v1");
        let b = key("b", "v1");
        let d = key("d", "v1");
        c.insert(a.clone(), CacheSlot::new(echo(), 40)).expect("a");
        c.insert(b.clone(), CacheSlot::new(echo(), 40)).expect("b");
        // 80 used, total 100 — inserting 30 needs to evict a (oldest).
        let evicted = c.insert(d.clone(), CacheSlot::new(echo(), 30)).expect("d");
        assert_eq!(evicted, vec![a]);
        assert_eq!(c.used_ram_mib(), 70);
        let keys: Vec<_> = c.snapshot().into_iter().map(|(k, _)| k).collect();
        assert_eq!(keys, vec![b, d]);
    }

    #[test]
    fn insert_duplicate_returns_already_present() {
        let c = ProviderCache::new(4, 1024);
        let k = key("a", "v1");
        c.insert(k.clone(), CacheSlot::new(echo(), 10)).expect("a");
        let err = c
            .insert(k.clone(), CacheSlot::new(echo(), 10))
            .expect_err("dup");
        assert_eq!(err, CacheError::AlreadyPresent { key: k });
    }

    #[test]
    fn evict_removes_and_updates_ram() {
        let c = ProviderCache::new(4, 1024);
        let k = key("a", "v1");
        c.insert(k.clone(), CacheSlot::new(echo(), 64)).expect("a");
        assert_eq!(c.used_ram_mib(), 64);
        let slot = c.evict(&k).expect("evicted");
        assert_eq!(slot.footprint_mib, 64);
        assert!(c.is_empty());
        assert_eq!(c.used_ram_mib(), 0);
    }

    #[test]
    fn evict_missing_returns_none() {
        let c = ProviderCache::new(4, 1024);
        assert!(c.evict(&key("nope", "v1")).is_none());
    }

    #[test]
    fn snapshot_lru_order_oldest_first() {
        let c = ProviderCache::new(4, 1024);
        let a = key("a", "v1");
        let b = key("b", "v1");
        let d = key("d", "v1");
        c.insert(a.clone(), CacheSlot::new(echo(), 1)).expect("a");
        c.insert(b.clone(), CacheSlot::new(echo(), 1)).expect("b");
        c.insert(d.clone(), CacheSlot::new(echo(), 1)).expect("d");
        let keys: Vec<_> = c.snapshot().into_iter().map(|(k, _)| k).collect();
        assert_eq!(keys, vec![a, b, d]);
    }

    #[test]
    fn len_is_empty_used_ram_accurate_after_series() {
        let c = ProviderCache::new(4, 1024);
        assert!(c.is_empty());
        c.insert(key("a", "v1"), CacheSlot::new(echo(), 10))
            .expect("a");
        c.insert(key("b", "v1"), CacheSlot::new(echo(), 20))
            .expect("b");
        assert_eq!(c.len(), 2);
        assert_eq!(c.used_ram_mib(), 30);
        let _ = c.evict(&key("a", "v1"));
        assert_eq!(c.len(), 1);
        assert_eq!(c.used_ram_mib(), 20);
    }

    #[test]
    fn cache_error_display_smoke() {
        let does_not_fit = CacheError::DoesNotFit {
            footprint: 4096,
            budget: 1024,
        };
        let s = format!("{does_not_fit}");
        assert!(s.contains("4096"));
        assert!(s.contains("1024"));
        let already = CacheError::AlreadyPresent { key: key("m", "v") };
        let s = format!("{already}");
        assert!(s.contains("m/v"));
    }

    #[test]
    fn cache_error_is_std_error() {
        fn assert_error<E: std::error::Error>() {}
        assert_error::<CacheError>();
    }

    #[test]
    fn provider_key_serde_roundtrip() {
        let k = key("qwen-2.5-7b", "q4_k_m");
        let s = serde_json::to_string(&k).expect("ser");
        let back: ProviderKey = serde_json::from_str(&s).expect("de");
        assert_eq!(k, back);
    }

    #[test]
    fn provider_key_ord_is_stable() {
        let mut v = vec![
            key("b", "v1"),
            key("a", "v2"),
            key("a", "v1"),
            key("c", "v1"),
        ];
        v.sort();
        assert_eq!(
            v,
            vec![
                key("a", "v1"),
                key("a", "v2"),
                key("b", "v1"),
                key("c", "v1"),
            ]
        );
    }

    #[test]
    fn provider_cache_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ProviderCache>();
    }

    #[test]
    fn empty_snapshot_returns_empty() {
        let c = ProviderCache::new(4, 1024);
        assert!(c.snapshot().is_empty());
    }

    #[test]
    fn evict_all_resets_state() {
        let c = ProviderCache::new(4, 1024);
        c.insert(key("a", "v1"), CacheSlot::new(echo(), 10))
            .expect("a");
        c.insert(key("b", "v1"), CacheSlot::new(echo(), 10))
            .expect("b");
        c.insert(key("d", "v1"), CacheSlot::new(echo(), 10))
            .expect("d");
        assert_eq!(c.len(), 3);
        let _ = c.evict(&key("a", "v1"));
        let _ = c.evict(&key("b", "v1"));
        let _ = c.evict(&key("d", "v1"));
        assert!(c.is_empty());
        assert_eq!(c.used_ram_mib(), 0);
    }

    #[test]
    fn lru_touch_on_get_changes_eviction_priority() {
        let c = ProviderCache::new(2, 10_000);
        let a = key("a", "v1");
        let b = key("b", "v1");
        let d = key("d", "v1");
        c.insert(a.clone(), CacheSlot::new(echo(), 10)).expect("a");
        c.insert(b.clone(), CacheSlot::new(echo(), 10)).expect("b");
        // Touch a so b becomes the LRU victim.
        let _ = c.get(&a);
        let evicted = c.insert(d.clone(), CacheSlot::new(echo(), 10)).expect("d");
        assert_eq!(evicted, vec![b]);
        let keys: Vec<_> = c.snapshot().into_iter().map(|(k, _)| k).collect();
        assert_eq!(keys, vec![a, d]);
    }

    #[test]
    fn insert_at_exact_budget_does_not_evict() {
        let c = ProviderCache::new(4, 100);
        let a = key("a", "v1");
        c.insert(a, CacheSlot::new(echo(), 100)).expect("a");
        assert_eq!(c.used_ram_mib(), 100);
        assert_eq!(c.len(), 1);
    }

    #[test]
    fn insert_evicts_multiple_until_fits() {
        let cache = ProviderCache::new(10, 100);
        let k_a = key("a", "v1");
        let k_b = key("b", "v1");
        let k_d = key("d", "v1");
        let k_e = key("e", "v1");
        cache
            .insert(k_a.clone(), CacheSlot::new(echo(), 30))
            .expect("a");
        cache
            .insert(k_b.clone(), CacheSlot::new(echo(), 30))
            .expect("b");
        cache
            .insert(k_d.clone(), CacheSlot::new(echo(), 30))
            .expect("d");
        // 90 used out of 100. Insert 60 — must evict a (60 used, 60+60
        // = 120 > 100), then b (30 used, 30+60 = 90 <= 100, done).
        // Result: [d, e] held with used=90.
        let evicted = cache
            .insert(k_e.clone(), CacheSlot::new(echo(), 60))
            .expect("e");
        assert_eq!(evicted, vec![k_a, k_b]);
        assert_eq!(cache.used_ram_mib(), 90); // 30 + 60
        let keys: Vec<_> = cache.snapshot().into_iter().map(|(k, _)| k).collect();
        assert_eq!(keys, vec![k_d, k_e]);
    }

    #[test]
    fn concurrent_fuzz_no_deadlock() {
        use std::sync::atomic::{AtomicU64, Ordering};

        const THREADS: usize = 4;
        const OPS_PER_THREAD: usize = 100;

        let cache = Arc::new(ProviderCache::new(4, 200));
        let barrier = Arc::new(Barrier::new(THREADS));
        let seed_counter = Arc::new(AtomicU64::new(0xc0ff_ee00));

        let mut handles = Vec::new();
        for tid in 0..THREADS {
            let c = Arc::clone(&cache);
            let b = Arc::clone(&barrier);
            let seeds = Arc::clone(&seed_counter);
            handles.push(thread::spawn(move || {
                b.wait();
                for op in 0..OPS_PER_THREAD {
                    // Cheap deterministic pseudo-random pick.
                    let s = seeds.fetch_add(1, Ordering::Relaxed);
                    let bucket =
                        (s.wrapping_mul(2_654_435_761) ^ (tid as u64)).wrapping_add(op as u64);
                    let slug_idx = bucket % 6;
                    let footprint = 20 + (bucket % 60);
                    let k = ProviderKey::new(format!("m{slug_idx}"), "v");
                    match bucket % 3 {
                        0 => {
                            let _ = c.insert(k, CacheSlot::new(echo(), footprint));
                        }
                        1 => {
                            let _ = c.get(&k);
                        }
                        _ => {
                            let _ = c.evict(&k);
                        }
                    }
                }
            }));
        }

        for h in handles {
            h.join().expect("thread join");
        }

        // Final invariants.
        assert!(cache.len() <= 4);
        assert!(cache.used_ram_mib() <= 200);
        // Snapshot length matches entries length — internal LRU and
        // entries map stayed coherent.
        assert_eq!(cache.snapshot().len(), cache.len());
    }
}
