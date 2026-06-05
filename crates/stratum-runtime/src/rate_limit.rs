//! Token-bucket rate limiting primitives.
//!
//! Generic enough to reuse for inbound HTTP (the future `stratum serve`
//! daemon, per `plan/24-stratum-serve-daemon.md` §4) and for inter-agent
//! call budgets.
//!
//! Two surfaces:
//!
//! * [`TokenBucket`] — a single bucket parameterised over a [`Clock`] so
//!   tests stay deterministic via [`ManualClock`].
//! * [`KeyedRateLimiter`] — a map of buckets keyed by anything `Ord`
//!   (typically a client IP or agent id), lazily materialised from a
//!   [`TokenBucketConfig`] template.
//!
//! Deliberately omitted (deferred to a follow-up):
//!
//! * TTL / LRU eviction of idle keys from [`KeyedRateLimiter`]. The map
//!   only grows; callers that need bounded memory should layer their own
//!   reaper on top.
//! * Async / wait-for-token semantics. The current API is non-blocking;
//!   callers decide how to back off after [`RateLimitError::Insufficient`].

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Source of monotonic timestamps.
///
/// Injected into [`TokenBucket`] so tests can drive refills with a
/// [`ManualClock`] instead of a real `sleep`.
pub trait Clock: Send + Sync {
    /// Current monotonic instant.
    fn now(&self) -> Instant;
}

/// Production clock that delegates to [`Instant::now`].
#[derive(Debug, Default, Clone, Copy)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> Instant {
        Instant::now()
    }
}

/// Deterministic clock for tests.
///
/// The internal `Instant` is shared via `Arc<Mutex<_>>`, so callers can
/// hand the same clock to multiple buckets and drive them in lockstep.
#[derive(Debug, Clone)]
pub struct ManualClock {
    inner: Arc<Mutex<Instant>>,
}

impl ManualClock {
    /// Create a manual clock anchored at `start`.
    #[must_use]
    pub fn new(start: Instant) -> Self {
        Self {
            inner: Arc::new(Mutex::new(start)),
        }
    }

    /// Advance the clock by `by`.
    ///
    /// Silently no-ops on lock poisoning so tests never panic from a
    /// peer thread's unrelated failure; downstream `now()` will simply
    /// see the unchanged instant.
    pub fn advance(&self, by: Duration) {
        if let Ok(mut guard) = self.inner.lock() {
            *guard += by;
        }
    }
}

impl Clock for ManualClock {
    fn now(&self) -> Instant {
        self.inner
            .lock()
            .map_or_else(|poisoned| *poisoned.into_inner(), |g| *g)
    }
}

/// Failure modes for [`TokenBucket::try_acquire`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RateLimitError {
    /// Bucket lacked enough tokens for the request.
    Insufficient {
        /// Tokens currently in the bucket (after refill).
        available: u32,
        /// Tokens the caller asked for.
        requested: u32,
    },
    /// Caller asked for zero tokens, which is meaningless and almost
    /// always a logic bug at the call site.
    ZeroRequested,
}

impl fmt::Display for RateLimitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Insufficient {
                available,
                requested,
            } => write!(
                f,
                "rate-limit: requested {requested} tokens, only {available} available"
            ),
            Self::ZeroRequested => f.write_str("rate-limit: zero tokens requested"),
        }
    }
}

impl Error for RateLimitError {}

/// Plain-data configuration for a [`TokenBucket`].
///
/// Used as the template inside [`KeyedRateLimiter`].
#[derive(Debug, Clone, Copy)]
pub struct TokenBucketConfig {
    /// Maximum number of tokens the bucket can hold.
    pub capacity: u32,
    /// Refill rate in tokens per second.
    pub refill_per_sec: f64,
}

#[derive(Debug)]
struct Inner {
    tokens: f64,
    capacity: f64,
    refill_per_sec: f64,
    last_refill: Instant,
}

impl Inner {
    fn refill(&mut self, now: Instant) {
        // `checked_duration_since` returns `None` if `now` is before
        // `last_refill`; treat that as zero elapsed and keep `last_refill`
        // pinned so a non-monotonic clock can't leak tokens.
        let elapsed = now
            .checked_duration_since(self.last_refill)
            .unwrap_or_default();
        if elapsed == Duration::ZERO {
            return;
        }
        let added = elapsed.as_secs_f64() * self.refill_per_sec;
        self.tokens = (self.tokens + added).min(self.capacity);
        self.last_refill = now;
    }
}

/// Token-bucket rate limiter.
///
/// Construction starts the bucket full. Tokens regenerate continuously
/// at `refill_per_sec` and saturate at `capacity`.
#[derive(Debug)]
pub struct TokenBucket<C: Clock = SystemClock> {
    inner: Mutex<Inner>,
    clock: C,
}

impl TokenBucket<SystemClock> {
    /// Build a token bucket backed by the real monotonic clock.
    #[must_use]
    pub fn new(capacity: u32, refill_per_sec: f64) -> Self {
        Self::with_clock(capacity, refill_per_sec, SystemClock)
    }
}

#[allow(
    clippy::cast_precision_loss,
    reason = "capacity is a u32; precise f64 conversion is fine up to 2^53"
)]
impl<C: Clock> TokenBucket<C> {
    /// Build a token bucket with a caller-supplied clock.
    pub fn with_clock(capacity: u32, refill_per_sec: f64, clock: C) -> Self {
        let now = clock.now();
        let cap_f = f64::from(capacity);
        let inner = Inner {
            tokens: cap_f,
            capacity: cap_f,
            refill_per_sec,
            last_refill: now,
        };
        Self {
            inner: Mutex::new(inner),
            clock,
        }
    }

    /// Attempt to deduct `n` tokens.
    ///
    /// Refills based on elapsed wall time first, then deducts on success.
    /// Returns [`RateLimitError::ZeroRequested`] for `n == 0` and
    /// [`RateLimitError::Insufficient`] when the bucket is too low.
    ///
    /// # Errors
    ///
    /// Returns [`RateLimitError`] if the request is malformed or the
    /// bucket is exhausted.
    pub fn try_acquire(&self, n: u32) -> Result<(), RateLimitError> {
        if n == 0 {
            return Err(RateLimitError::ZeroRequested);
        }
        let now = self.clock.now();
        let mut guard = match self.inner.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        guard.refill(now);
        let need = f64::from(n);
        if guard.tokens + f64::EPSILON >= need {
            guard.tokens -= need;
            drop(guard);
            Ok(())
        } else {
            let available = floor_to_u32(guard.tokens);
            drop(guard);
            Err(RateLimitError::Insufficient {
                available,
                requested: n,
            })
        }
    }

    /// Tokens currently available (rounded down) after a refill pass.
    #[must_use]
    pub fn available(&self) -> u32 {
        let now = self.clock.now();
        let mut guard = match self.inner.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        guard.refill(now);
        floor_to_u32(guard.tokens)
    }
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "tokens are clamped to [0, capacity] where capacity fits in u32"
)]
fn floor_to_u32(value: f64) -> u32 {
    if value <= 0.0 {
        0
    } else {
        value.floor() as u32
    }
}

/// Per-key token-bucket limiter.
///
/// Each key gets its own bucket, lazily created from `template` on first
/// touch. The map grows monotonically — there is no built-in eviction;
/// see the module docs.
#[derive(Debug)]
pub struct KeyedRateLimiter<K: Ord + Clone + Send + Sync, C: Clock + Clone = SystemClock> {
    template: TokenBucketConfig,
    clock: C,
    buckets: Mutex<BTreeMap<K, TokenBucket<C>>>,
}

impl<K: Ord + Clone + Send + Sync> KeyedRateLimiter<K, SystemClock> {
    /// Build a keyed limiter backed by the real monotonic clock.
    #[must_use]
    pub const fn new(template: TokenBucketConfig) -> Self {
        Self::with_clock(template, SystemClock)
    }
}

impl<K: Ord + Clone + Send + Sync, C: Clock + Clone> KeyedRateLimiter<K, C> {
    /// Build a keyed limiter with a caller-supplied clock.
    pub const fn with_clock(template: TokenBucketConfig, clock: C) -> Self {
        Self {
            template,
            clock,
            buckets: Mutex::new(BTreeMap::new()),
        }
    }

    /// Attempt to deduct `n` tokens from `key`'s bucket, creating it
    /// from the template if it does not exist.
    ///
    /// # Errors
    ///
    /// Same shape as [`TokenBucket::try_acquire`].
    #[allow(
        clippy::significant_drop_tightening,
        reason = "the outer mutex must outlive the &TokenBucket borrow into the map"
    )]
    pub fn try_acquire(&self, key: K, n: u32) -> Result<(), RateLimitError> {
        // Acquire a clone of the per-key bucket handle by extracting it
        // briefly, but since `TokenBucket` is `!Clone`, we hold the outer
        // mutex only long enough to insert-if-missing, then defer the
        // actual `try_acquire` to a call against the entry under the
        // same lock. The bucket's own mutex is uncontended in this path.
        let mut guard = match self.buckets.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        let bucket = guard.entry(key).or_insert_with(|| {
            TokenBucket::with_clock(
                self.template.capacity,
                self.template.refill_per_sec,
                self.clock.clone(),
            )
        });
        bucket.try_acquire(n)
    }

    /// Number of keys currently tracked. Useful for tests and metrics.
    #[must_use]
    pub fn len(&self) -> usize {
        match self.buckets.lock() {
            Ok(g) => g.len(),
            Err(poisoned) => poisoned.into_inner().len(),
        }
    }

    /// `true` when no keys have been touched yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    fn manual() -> ManualClock {
        ManualClock::new(Instant::now())
    }

    #[test]
    fn new_bucket_starts_full() {
        let bucket: TokenBucket = TokenBucket::new(10, 1.0);
        assert_eq!(bucket.available(), 10);
    }

    #[test]
    fn try_acquire_one_from_full_bucket() {
        let bucket: TokenBucket = TokenBucket::new(10, 1.0);
        assert!(bucket.try_acquire(1).is_ok());
        assert_eq!(bucket.available(), 9);
    }

    #[test]
    fn try_acquire_over_capacity_is_insufficient() {
        let clock = manual();
        let bucket = TokenBucket::with_clock(5, 0.0, clock);
        let err = bucket
            .try_acquire(6)
            .expect_err("over-capacity request must fail");
        assert_eq!(
            err,
            RateLimitError::Insufficient {
                available: 5,
                requested: 6,
            }
        );
    }

    #[test]
    fn try_acquire_zero_is_rejected() {
        let bucket: TokenBucket = TokenBucket::new(10, 1.0);
        assert_eq!(bucket.try_acquire(0), Err(RateLimitError::ZeroRequested));
    }

    #[test]
    fn refill_restores_tokens_over_time() {
        let clock = manual();
        let bucket = TokenBucket::with_clock(10, 2.0, clock.clone());
        bucket.try_acquire(10).expect("drain");
        assert_eq!(bucket.available(), 0);
        clock.advance(Duration::from_secs(3));
        // 3s * 2/sec = 6 tokens
        assert_eq!(bucket.available(), 6);
    }

    #[test]
    fn refill_saturates_at_capacity() {
        let clock = manual();
        let bucket = TokenBucket::with_clock(5, 10.0, clock.clone());
        bucket.try_acquire(5).expect("drain");
        clock.advance(Duration::from_secs(60));
        assert_eq!(bucket.available(), 5);
    }

    #[test]
    fn zero_elapsed_refill_is_noop() {
        let clock = manual();
        let bucket = TokenBucket::with_clock(10, 5.0, clock.clone());
        bucket.try_acquire(3).expect("partial drain");
        let before = bucket.available();
        clock.advance(Duration::ZERO);
        assert_eq!(bucket.available(), before);
    }

    #[test]
    fn concurrent_acquire_totals_exactly_capacity() {
        let bucket: Arc<TokenBucket> = Arc::new(TokenBucket::with_clock(100, 0.0, SystemClock));
        let mut handles = Vec::new();
        for _ in 0..4 {
            let b = Arc::clone(&bucket);
            handles.push(thread::spawn(move || {
                let mut wins = 0_u32;
                for _ in 0..200 {
                    if b.try_acquire(1).is_ok() {
                        wins += 1;
                    }
                }
                wins
            }));
        }
        let total: u32 = handles.into_iter().map(|h| h.join().unwrap_or(0)).sum();
        assert_eq!(total, 100);
    }

    #[test]
    fn available_is_consistent_after_refill() {
        let clock = manual();
        let bucket = TokenBucket::with_clock(10, 1.0, clock.clone());
        bucket.try_acquire(10).expect("drain");
        clock.advance(Duration::from_secs(2));
        let a = bucket.available();
        let b = bucket.available();
        assert_eq!(a, b);
        assert_eq!(a, 2);
    }

    #[test]
    fn keyed_limiter_per_key_budgets_are_independent() {
        let clock = manual();
        let limiter: KeyedRateLimiter<&'static str, ManualClock> = KeyedRateLimiter::with_clock(
            TokenBucketConfig {
                capacity: 3,
                refill_per_sec: 0.0,
            },
            clock,
        );
        assert!(limiter.try_acquire("alice", 3).is_ok());
        assert!(matches!(
            limiter.try_acquire("alice", 1),
            Err(RateLimitError::Insufficient { .. })
        ));
        // Bob's bucket is untouched.
        assert!(limiter.try_acquire("bob", 3).is_ok());
    }

    #[test]
    fn keyed_limiter_initializes_new_key_from_template() {
        let clock = manual();
        let limiter: KeyedRateLimiter<u32, ManualClock> = KeyedRateLimiter::with_clock(
            TokenBucketConfig {
                capacity: 7,
                refill_per_sec: 0.0,
            },
            clock,
        );
        assert!(limiter.is_empty());
        assert!(limiter.try_acquire(42, 7).is_ok());
        assert_eq!(limiter.len(), 1);
        // Capacity exhausted on the same key.
        assert!(limiter.try_acquire(42, 1).is_err());
    }

    #[test]
    fn fractional_refill_accumulates_over_many_steps() {
        let clock = manual();
        let bucket = TokenBucket::with_clock(10, 1.0, clock.clone());
        bucket.try_acquire(10).expect("drain");
        for _ in 0..10 {
            clock.advance(Duration::from_millis(100));
            // Refill happens on the next observation; touch via available()
            // to fold partial tokens in.
            let _ = bucket.available();
        }
        // 10 * 0.1s * 1/sec ≈ 1 token, modulo f64 rounding. The bucket may
        // sit at 0.999…, which floors to 0 — but the *next* tick crosses
        // the integer boundary cleanly, and the bucket has enough to
        // satisfy a 1-token request (try_acquire allows an epsilon slack).
        assert!(
            bucket.try_acquire(1).is_ok(),
            "fractional refill should have accumulated to ~1 token"
        );
    }

    #[test]
    fn rate_limit_error_display_round_trips() {
        let e = RateLimitError::Insufficient {
            available: 2,
            requested: 5,
        };
        assert_eq!(
            e.to_string(),
            "rate-limit: requested 5 tokens, only 2 available"
        );
        assert_eq!(
            RateLimitError::ZeroRequested.to_string(),
            "rate-limit: zero tokens requested"
        );
        // `Error` impl present.
        let _: &dyn Error = &e;
    }

    #[test]
    fn system_clock_is_monotonic_over_small_sleep() {
        let clock = SystemClock;
        let a = clock.now();
        thread::sleep(Duration::from_millis(2));
        let b = clock.now();
        assert!(b >= a);
    }

    #[test]
    fn keyed_limiter_does_not_evict_on_its_own() {
        // Documents the deliberate omission: keys persist forever until a
        // future TTL-eviction follow-up lands.
        let clock = manual();
        let limiter: KeyedRateLimiter<u8, ManualClock> = KeyedRateLimiter::with_clock(
            TokenBucketConfig {
                capacity: 1,
                refill_per_sec: 1_000.0,
            },
            clock.clone(),
        );
        for k in 0..16_u8 {
            limiter.try_acquire(k, 1).expect("acquire");
        }
        clock.advance(Duration::from_secs(3_600));
        assert_eq!(limiter.len(), 16);
    }

    #[test]
    fn try_acquire_does_not_deadlock_under_contention() {
        // Spawn a producer that hammers `try_acquire` while a poller
        // hammers `available`; both go through the same mutex. If we'd
        // accidentally double-locked, the poller would never observe
        // the bucket draining.
        let bucket: Arc<TokenBucket> = Arc::new(TokenBucket::new(50, 0.0));
        let producer = {
            let b = Arc::clone(&bucket);
            thread::spawn(move || {
                for _ in 0..50 {
                    let _ = b.try_acquire(1);
                }
            })
        };
        let poller = {
            let b = Arc::clone(&bucket);
            thread::spawn(move || {
                let mut last = u32::MAX;
                for _ in 0..200 {
                    let now = b.available();
                    if now <= last {
                        last = now;
                    }
                }
                last
            })
        };
        producer.join().expect("producer join");
        let final_seen = poller.join().expect("poller join");
        assert!(final_seen <= 50);
        assert_eq!(bucket.available(), 0);
    }

    #[test]
    fn manual_clock_now_advances_after_advance() {
        let start = Instant::now();
        let clock = ManualClock::new(start);
        let a = clock.now();
        clock.advance(Duration::from_secs(5));
        let b = clock.now();
        assert_eq!(b.saturating_duration_since(a), Duration::from_secs(5));
    }

    #[test]
    fn keyed_limiter_new_uses_system_clock() {
        let limiter: KeyedRateLimiter<&'static str> = KeyedRateLimiter::new(TokenBucketConfig {
            capacity: 2,
            refill_per_sec: 1.0,
        });
        assert!(limiter.try_acquire("a", 1).is_ok());
        assert_eq!(limiter.len(), 1);
    }

    #[test]
    fn token_bucket_config_is_copy() {
        // Compile-test: ensure the trait bounds documented at the type
        // remain accurate.
        fn assert_copy<T: Copy>() {}
        assert_copy::<TokenBucketConfig>();
        let cfg = TokenBucketConfig {
            capacity: 1,
            refill_per_sec: 1.0,
        };
        let c2 = cfg;
        assert_eq!(c2.capacity, 1);
        assert!((cfg.refill_per_sec - 1.0).abs() < f64::EPSILON);
    }
}
