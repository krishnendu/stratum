//! Deterministic retry-with-backoff helper for transient errors.
//!
//! Used by the HTTP installer, MCP transports, and future provider retries.
//! See `plan/32-cancellation-and-timeouts.md` §6.
//!
//! The core entry point is [`retry`] (synchronous, uses [`SystemClock`]) and
//! its testable sibling [`retry_with_clock`] (generic over [`Clock`]). The
//! caller supplies a [`RetryClassifier`] that decides whether a particular
//! error should be retried or surfaced immediately. Backoff is exponential
//! with optional jitter ([`Jitter`]).

use std::error::Error;
use std::fmt;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

/// Backoff jitter strategy.
///
/// - `None`: no jitter — every attempt waits the exact exponential base.
/// - `Full`: uniform `[0, base]` (AWS Architecture Blog "Full Jitter").
/// - `Decorrelated`: uniform `[initial_delay, min(prev*3, max_delay)]`
///   (AWS "Decorrelated Jitter").
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Jitter {
    /// Deterministic exponential delay, no randomness.
    None,
    /// Uniform random in `[0, base]`.
    Full,
    /// Uniform random in `[initial_delay, min(prev*3, max_delay)]`.
    Decorrelated,
}

/// Retry policy: caps, base delays, growth factor, jitter strategy.
#[derive(Clone, Copy, Debug)]
pub struct RetryPolicy {
    /// Maximum total attempts (initial + retries).
    pub max_attempts: u32,
    /// Base delay used for attempt 1 and as the floor for decorrelated jitter.
    pub initial_delay: Duration,
    /// Absolute cap on any single inter-attempt sleep.
    pub max_delay: Duration,
    /// Exponential growth factor applied per attempt.
    pub backoff_multiplier: f64,
    /// Jitter strategy applied on top of the exponential base.
    pub jitter: Jitter,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 4,
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(10),
            backoff_multiplier: 2.0,
            jitter: Jitter::Full,
        }
    }
}

impl RetryPolicy {
    /// Compute the pre-attempt delay for `attempt` (0-indexed).
    ///
    /// - Attempt 0 is the initial call and returns [`Duration::ZERO`].
    /// - Attempt `n ≥ 1` uses `min(initial * multiplier^(n-1), max_delay)`,
    ///   then jitter is applied.
    /// - `prev` is the previously-slept duration; only consulted for
    ///   [`Jitter::Decorrelated`].
    #[must_use]
    pub fn delay_for(&self, attempt: u32, prev: Option<Duration>, rng: &mut SmallRng) -> Duration {
        if attempt == 0 {
            return Duration::ZERO;
        }
        let base = self.base_delay(attempt);
        match self.jitter {
            Jitter::None => base,
            Jitter::Full => {
                let nanos = u128_min(base.as_nanos(), u128::from(u64::MAX));
                let pick = if nanos == 0 {
                    0
                } else {
                    rng.random_range(0..=nanos_to_u64(nanos))
                };
                Duration::from_nanos(pick)
            }
            Jitter::Decorrelated => {
                let prev = prev.unwrap_or(self.initial_delay);
                let tripled = prev.saturating_mul(3);
                let upper = if tripled > self.max_delay {
                    self.max_delay
                } else {
                    tripled
                };
                let lo = self.initial_delay;
                let (lo_n, hi_n) = if upper < lo {
                    (lo.as_nanos(), lo.as_nanos())
                } else {
                    (lo.as_nanos(), upper.as_nanos())
                };
                let lo_u = nanos_to_u64(lo_n);
                let hi_u = nanos_to_u64(hi_n);
                let pick = if lo_u >= hi_u {
                    lo_u
                } else {
                    rng.random_range(lo_u..=hi_u)
                };
                Duration::from_nanos(pick)
            }
        }
    }

    #[allow(
        clippy::cast_precision_loss,
        reason = "multiplier^attempt may exceed f64 mantissa for huge attempt counts; saturated below"
    )]
    #[allow(
        clippy::cast_sign_loss,
        reason = "value is clamped to non-negative before cast"
    )]
    #[allow(
        clippy::cast_possible_truncation,
        reason = "value is clamped to u64::MAX before cast"
    )]
    fn base_delay(&self, attempt: u32) -> Duration {
        // attempt >= 1 guaranteed by caller.
        let exponent = i32::try_from(attempt - 1).unwrap_or(i32::MAX);
        let factor = self.backoff_multiplier.powi(exponent);
        let initial_nanos = self.initial_delay.as_nanos() as f64;
        let scaled = initial_nanos * factor;
        let max_nanos = self.max_delay.as_nanos() as f64;
        let clamped = if !scaled.is_finite() || scaled >= max_nanos {
            max_nanos
        } else if scaled < 0.0 {
            0.0
        } else {
            scaled
        };
        let as_u64 = if clamped >= (u64::MAX as f64) {
            u64::MAX
        } else {
            clamped as u64
        };
        Duration::from_nanos(as_u64)
    }
}

const fn u128_min(a: u128, b: u128) -> u128 {
    if a < b {
        a
    } else {
        b
    }
}

#[allow(
    clippy::cast_possible_truncation,
    reason = "saturated to u64::MAX before cast"
)]
const fn nanos_to_u64(n: u128) -> u64 {
    if n > u64::MAX as u128 {
        u64::MAX
    } else {
        n as u64
    }
}

/// Per-error classifier decision.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RetryDecision {
    /// The error is transient — schedule another attempt.
    Retry,
    /// The error is terminal — surface it immediately.
    Fatal,
}

/// Classifies an error as retryable or fatal.
pub trait RetryClassifier<E> {
    /// Inspect `err` and decide what to do.
    fn classify(&self, err: &E) -> RetryDecision;
}

impl<E, F> RetryClassifier<E> for F
where
    F: Fn(&E) -> RetryDecision,
{
    fn classify(&self, err: &E) -> RetryDecision {
        (self)(err)
    }
}

/// Abstract time source so callers can swap real sleeps for instant
/// recordings in tests.
pub trait Clock: Send + Sync {
    /// Current monotonic instant.
    fn now(&self) -> Instant;
    /// Block the calling thread for `dur`.
    fn sleep(&self, dur: Duration);
}

/// Real-time clock backed by `std::thread::sleep` and `Instant::now`.
#[derive(Debug, Default, Clone, Copy)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> Instant {
        Instant::now()
    }
    fn sleep(&self, dur: Duration) {
        std::thread::sleep(dur);
    }
}

/// Test clock: records sleep durations and advances a virtual `now` without
/// blocking.
#[derive(Debug)]
pub struct ManualClock {
    now: Mutex<Instant>,
    sleeps: Mutex<Vec<Duration>>,
}

impl Default for ManualClock {
    fn default() -> Self {
        Self::new()
    }
}

impl ManualClock {
    /// Construct a `ManualClock` anchored at `Instant::now()`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            now: Mutex::new(Instant::now()),
            sleeps: Mutex::new(Vec::new()),
        }
    }

    /// Snapshot of every sleep duration the clock has been asked to perform.
    #[must_use]
    pub fn sleeps(&self) -> Vec<Duration> {
        match self.sleeps.lock() {
            Ok(g) => g.clone(),
            Err(p) => p.into_inner().clone(),
        }
    }

    /// Manually advance virtual time (does not record a sleep).
    pub fn advance(&self, by: Duration) {
        if let Ok(mut g) = self.now.lock() {
            *g += by;
        }
    }
}

impl Clock for ManualClock {
    fn now(&self) -> Instant {
        match self.now.lock() {
            Ok(g) => *g,
            Err(p) => *p.into_inner(),
        }
    }
    fn sleep(&self, dur: Duration) {
        if let Ok(mut g) = self.sleeps.lock() {
            g.push(dur);
        }
        if let Ok(mut g) = self.now.lock() {
            *g += dur;
        }
    }
}

/// Newtype carrying a computed delay; useful when callers want to log
/// `DelayPlan` without confusing it with `Duration`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DelayPlan(pub Duration);

/// Outcome of a failed [`retry`] call.
#[derive(Debug)]
pub enum RetryError<E> {
    /// The classifier marked the error as terminal — no retries attempted
    /// beyond the failing one.
    Fatal(E),
    /// All `max_attempts` were used; the wrapped error is the last one seen.
    Exhausted {
        /// Total attempts performed.
        attempts: u32,
        /// Last error returned by the operation.
        last_error: E,
    },
}

impl<E: fmt::Display> fmt::Display for RetryError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Fatal(e) => write!(f, "retry aborted: fatal error: {e}"),
            Self::Exhausted {
                attempts,
                last_error,
            } => write!(f, "retry exhausted after {attempts} attempts: {last_error}"),
        }
    }
}

impl<E: fmt::Display + fmt::Debug + Error + 'static> Error for RetryError<E> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Fatal(e) | Self::Exhausted { last_error: e, .. } => Some(e),
        }
    }
}

/// Run `op` under `policy` using the real [`SystemClock`].
///
/// # Errors
/// Returns [`RetryError::Fatal`] if the classifier rejects an error or
/// [`RetryError::Exhausted`] if every attempt fails.
pub fn retry<T, E, F, C>(policy: &RetryPolicy, classifier: &C, op: F) -> Result<T, RetryError<E>>
where
    F: FnMut(u32) -> Result<T, E>,
    C: RetryClassifier<E>,
{
    retry_with_clock(policy, classifier, &SystemClock, op)
}

/// Like [`retry`] but parameterized over a [`Clock`] so tests can drive
/// sleeps deterministically.
///
/// # Errors
/// Same surface as [`retry`].
pub fn retry_with_clock<T, E, F, C, K>(
    policy: &RetryPolicy,
    classifier: &C,
    clock: &K,
    mut op: F,
) -> Result<T, RetryError<E>>
where
    F: FnMut(u32) -> Result<T, E>,
    C: RetryClassifier<E>,
    K: Clock + ?Sized,
{
    // Deterministic seed by default; callers wanting fresh randomness across
    // runs should use `retry_with_clock_seeded` with an entropy-backed RNG.
    let mut rng = SmallRng::seed_from_u64(0);
    retry_with_clock_seeded(policy, classifier, clock, &mut rng, &mut op)
}

/// Seeded variant for advanced tests and deterministic production paths.
///
/// # Errors
/// Same surface as [`retry`].
pub fn retry_with_clock_seeded<T, E, F, C, K>(
    policy: &RetryPolicy,
    classifier: &C,
    clock: &K,
    rng: &mut SmallRng,
    op: &mut F,
) -> Result<T, RetryError<E>>
where
    F: FnMut(u32) -> Result<T, E>,
    C: RetryClassifier<E>,
    K: Clock + ?Sized,
{
    let max = policy.max_attempts.max(1);
    let mut prev: Option<Duration> = None;
    // Drive `max - 1` retryable attempts, then handle the final attempt
    // separately so we can always own the trailing error for `Exhausted`.
    for attempt in 0..max.saturating_sub(1) {
        let delay = policy.delay_for(attempt, prev, rng);
        if !delay.is_zero() {
            clock.sleep(delay);
            prev = Some(delay);
        }
        match op(attempt) {
            Ok(value) => return Ok(value),
            Err(err) => match classifier.classify(&err) {
                RetryDecision::Fatal => return Err(RetryError::Fatal(err)),
                RetryDecision::Retry => {}
            },
        }
    }
    // Final attempt.
    let final_attempt = max - 1;
    let delay = policy.delay_for(final_attempt, prev, rng);
    if !delay.is_zero() {
        clock.sleep(delay);
    }
    match op(final_attempt) {
        Ok(value) => Ok(value),
        Err(err) => match classifier.classify(&err) {
            RetryDecision::Fatal => Err(RetryError::Fatal(err)),
            RetryDecision::Retry => Err(RetryError::Exhausted {
                attempts: max,
                last_error: err,
            }),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::sync::atomic::{AtomicU32, Ordering};

    fn retry_all<E>() -> impl Fn(&E) -> RetryDecision {
        |_| RetryDecision::Retry
    }

    fn never_retry<E>() -> impl Fn(&E) -> RetryDecision {
        |_| RetryDecision::Fatal
    }

    #[test]
    fn default_matches_spec() {
        let p = RetryPolicy::default();
        assert_eq!(p.max_attempts, 4);
        assert_eq!(p.initial_delay, Duration::from_millis(100));
        assert_eq!(p.max_delay, Duration::from_secs(10));
        assert!((p.backoff_multiplier - 2.0).abs() < f64::EPSILON);
        assert_eq!(p.jitter, Jitter::Full);
    }

    #[test]
    fn delay_for_attempt_zero_is_zero() {
        let p = RetryPolicy::default();
        let mut rng = SmallRng::seed_from_u64(1);
        assert_eq!(p.delay_for(0, None, &mut rng), Duration::ZERO);
        assert_eq!(
            p.delay_for(0, Some(Duration::from_secs(5)), &mut rng),
            Duration::ZERO
        );
    }

    #[test]
    fn delay_for_attempt_one_with_none_jitter_equals_initial() {
        let p = RetryPolicy {
            jitter: Jitter::None,
            ..RetryPolicy::default()
        };
        let mut rng = SmallRng::seed_from_u64(1);
        assert_eq!(p.delay_for(1, None, &mut rng), Duration::from_millis(100));
    }

    #[test]
    fn delay_for_exponential_growth_none_jitter() {
        let p = RetryPolicy {
            jitter: Jitter::None,
            ..RetryPolicy::default()
        };
        let mut rng = SmallRng::seed_from_u64(1);
        // attempt=3 -> initial * 2^(3-1) = 400ms
        assert_eq!(p.delay_for(3, None, &mut rng), Duration::from_millis(400));
    }

    #[test]
    fn delay_for_saturates_at_max_delay() {
        let p = RetryPolicy {
            jitter: Jitter::None,
            max_delay: Duration::from_secs(1),
            ..RetryPolicy::default()
        };
        let mut rng = SmallRng::seed_from_u64(1);
        // attempt=20 would be 100ms * 2^19 — far above 1s cap
        assert_eq!(p.delay_for(20, None, &mut rng), Duration::from_secs(1));
    }

    #[test]
    fn delay_for_full_jitter_in_bounds() {
        let p = RetryPolicy {
            jitter: Jitter::Full,
            ..RetryPolicy::default()
        };
        let mut rng = SmallRng::seed_from_u64(42);
        // base for attempt=4 = 100ms * 2^3 = 800ms
        let base = Duration::from_millis(800);
        for _ in 0..1000 {
            let d = p.delay_for(4, None, &mut rng);
            assert!(d <= base, "{d:?} exceeded base {base:?}");
        }
    }

    #[test]
    fn delay_for_decorrelated_in_bounds() {
        let p = RetryPolicy {
            jitter: Jitter::Decorrelated,
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(10),
            ..RetryPolicy::default()
        };
        let mut rng = SmallRng::seed_from_u64(7);
        let prev = Duration::from_millis(500);
        let upper = Duration::from_millis(1500); // min(prev*3, max_delay) = 1500ms
        for _ in 0..1000 {
            let d = p.delay_for(2, Some(prev), &mut rng);
            assert!(d >= p.initial_delay, "{d:?} below floor");
            assert!(d <= upper, "{d:?} above upper {upper:?}");
        }
    }

    #[test]
    fn decorrelated_first_attempt_uses_initial_when_no_prev() {
        let p = RetryPolicy {
            jitter: Jitter::Decorrelated,
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(10),
            ..RetryPolicy::default()
        };
        let mut rng = SmallRng::seed_from_u64(11);
        // prev=None -> floor=initial, upper=min(initial*3, max_delay)=300ms
        let upper = Duration::from_millis(300);
        for _ in 0..200 {
            let d = p.delay_for(1, None, &mut rng);
            assert!(d >= p.initial_delay);
            assert!(d <= upper);
        }
    }

    #[test]
    fn retry_succeeds_first_try_records_zero_sleeps() {
        let p = RetryPolicy::default();
        let clock = ManualClock::new();
        let classifier = retry_all::<&'static str>();
        let result: Result<u32, RetryError<&'static str>> =
            retry_with_clock(&p, &classifier, &clock, |_| Ok(7));
        assert_eq!(result.ok(), Some(7));
        assert!(clock.sleeps().is_empty());
    }

    #[test]
    fn retry_fails_twice_then_succeeds() {
        let p = RetryPolicy {
            jitter: Jitter::None,
            ..RetryPolicy::default()
        };
        let clock = ManualClock::new();
        let classifier = retry_all::<&'static str>();
        let attempts = AtomicU32::new(0);
        let result: Result<u32, RetryError<&'static str>> =
            retry_with_clock(&p, &classifier, &clock, |_| {
                let n = attempts.fetch_add(1, Ordering::SeqCst);
                if n < 2 {
                    Err("nope")
                } else {
                    Ok(99)
                }
            });
        assert_eq!(result.ok(), Some(99));
        assert_eq!(clock.sleeps().len(), 2);
    }

    #[test]
    fn retry_fatal_short_circuits_without_sleeping() {
        let p = RetryPolicy::default();
        let clock = ManualClock::new();
        let classifier = never_retry::<&'static str>();
        let result: Result<u32, RetryError<&'static str>> =
            retry_with_clock(&p, &classifier, &clock, |_| Err("boom"));
        assert!(matches!(result, Err(RetryError::Fatal("boom"))));
        assert!(clock.sleeps().is_empty());
    }

    #[test]
    fn retry_exhausts_after_max_attempts() {
        let p = RetryPolicy {
            max_attempts: 3,
            jitter: Jitter::None,
            ..RetryPolicy::default()
        };
        let clock = ManualClock::new();
        let classifier = retry_all::<&'static str>();
        let attempts = AtomicU32::new(0);
        let result: Result<u32, RetryError<&'static str>> =
            retry_with_clock(&p, &classifier, &clock, |_| {
                attempts.fetch_add(1, Ordering::SeqCst);
                Err("transient")
            });
        match result {
            Err(RetryError::Exhausted {
                attempts: a,
                last_error,
            }) => {
                assert_eq!(a, 3);
                assert_eq!(last_error, "transient");
            }
            other => unreachable!("expected Exhausted, got {other:?}"),
        }
        assert_eq!(attempts.load(Ordering::SeqCst), 3);
        // 2 sleeps between 3 attempts (no sleep before first).
        assert_eq!(clock.sleeps().len(), 2);
    }

    #[test]
    fn manual_clock_advances_on_sleep() {
        let clock = ManualClock::new();
        let t0 = clock.now();
        clock.sleep(Duration::from_millis(250));
        let t1 = clock.now();
        assert!(t1 >= t0 + Duration::from_millis(250));
        clock.advance(Duration::from_millis(100));
        assert!(clock.now() >= t1 + Duration::from_millis(100));
    }

    #[test]
    fn deterministic_seed_same_trace() {
        let p = RetryPolicy {
            jitter: Jitter::Full,
            ..RetryPolicy::default()
        };
        let run = || {
            let clock = ManualClock::new();
            let mut rng = SmallRng::seed_from_u64(12345);
            let classifier = retry_all::<&'static str>();
            let mut op = |_attempt: u32| -> Result<u32, &'static str> { Err("x") };
            let _ = retry_with_clock_seeded(&p, &classifier, &clock, &mut rng, &mut op);
            clock.sleeps()
        };
        let a = run();
        let b = run();
        assert_eq!(a, b);
        assert!(!a.is_empty());
    }

    #[test]
    fn retry_error_display_includes_inner() {
        let fatal: RetryError<&'static str> = RetryError::Fatal("boom");
        assert!(fatal.to_string().contains("boom"));
        let ex: RetryError<&'static str> = RetryError::Exhausted {
            attempts: 5,
            last_error: "still bad",
        };
        let msg = ex.to_string();
        assert!(msg.contains("still bad"));
        assert!(msg.contains('5'));
    }

    #[test]
    fn retry_error_is_send_sync_and_error() {
        fn assert_traits<T: Send + Sync + std::error::Error>() {}
        assert_traits::<RetryError<std::io::Error>>();
    }

    #[test]
    fn policy_is_send_sync_copy() {
        fn assert_traits<T: Send + Sync + Copy>() {}
        assert_traits::<RetryPolicy>();
        assert_traits::<Jitter>();
        assert_traits::<RetryDecision>();
        assert_traits::<SystemClock>();
        assert_traits::<DelayPlan>();
    }

    #[test]
    fn retry_with_real_clock_succeeds() {
        // Smoke: drives the public `retry` entry point + SystemClock once.
        let p = RetryPolicy {
            max_attempts: 1,
            ..RetryPolicy::default()
        };
        let classifier = retry_all::<&'static str>();
        let result: Result<u32, RetryError<&'static str>> = retry(&p, &classifier, |_| Ok(1));
        assert_eq!(result.ok(), Some(1));
    }

    #[test]
    fn retry_invokes_classifier_for_each_failure() {
        let p = RetryPolicy {
            max_attempts: 3,
            jitter: Jitter::None,
            ..RetryPolicy::default()
        };
        let clock = ManualClock::new();
        let calls = Cell::new(0u32);
        let classifier = |_err: &&'static str| {
            calls.set(calls.get() + 1);
            RetryDecision::Retry
        };
        let _ = retry_with_clock::<u32, &'static str, _, _, _>(&p, &classifier, &clock, |_| {
            Err("fail")
        });
        assert_eq!(calls.get(), 3);
    }

    #[test]
    fn fatal_after_one_retry_returns_fatal() {
        let p = RetryPolicy {
            max_attempts: 5,
            jitter: Jitter::None,
            ..RetryPolicy::default()
        };
        let clock = ManualClock::new();
        let attempts = AtomicU32::new(0);
        let classifier = |err: &&'static str| {
            if *err == "fatal" {
                RetryDecision::Fatal
            } else {
                RetryDecision::Retry
            }
        };
        let result: Result<u32, RetryError<&'static str>> =
            retry_with_clock(&p, &classifier, &clock, |_| {
                let n = attempts.fetch_add(1, Ordering::SeqCst);
                if n == 0 {
                    Err("transient")
                } else {
                    Err("fatal")
                }
            });
        assert!(matches!(result, Err(RetryError::Fatal("fatal"))));
        // One sleep between attempts 0 and 1; none after the Fatal.
        assert_eq!(clock.sleeps().len(), 1);
    }

    #[test]
    fn jitter_none_is_pure() {
        // Same policy, two different RNG seeds — must produce identical delay.
        let p = RetryPolicy {
            jitter: Jitter::None,
            ..RetryPolicy::default()
        };
        let mut a = SmallRng::seed_from_u64(1);
        let mut b = SmallRng::seed_from_u64(2);
        assert_eq!(p.delay_for(3, None, &mut a), p.delay_for(3, None, &mut b));
    }

    #[test]
    fn full_jitter_zero_base_returns_zero() {
        let p = RetryPolicy {
            jitter: Jitter::Full,
            initial_delay: Duration::ZERO,
            max_delay: Duration::ZERO,
            ..RetryPolicy::default()
        };
        let mut rng = SmallRng::seed_from_u64(0);
        assert_eq!(p.delay_for(1, None, &mut rng), Duration::ZERO);
    }

    #[test]
    fn decorrelated_tripled_below_cap() {
        // prev*3 < max_delay -> upper = tripled
        let p = RetryPolicy {
            jitter: Jitter::Decorrelated,
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(60),
            ..RetryPolicy::default()
        };
        let mut rng = SmallRng::seed_from_u64(3);
        let prev = Duration::from_millis(200);
        // upper = min(600ms, 60s) = 600ms
        let upper = Duration::from_millis(600);
        for _ in 0..200 {
            let d = p.delay_for(2, Some(prev), &mut rng);
            assert!(d >= p.initial_delay);
            assert!(d <= upper);
        }
    }

    #[test]
    fn decorrelated_upper_below_floor_collapses_to_floor() {
        // initial_delay > max_delay -> upper < lo path -> deterministic floor.
        let p = RetryPolicy {
            jitter: Jitter::Decorrelated,
            initial_delay: Duration::from_secs(10),
            max_delay: Duration::from_millis(100),
            ..RetryPolicy::default()
        };
        let mut rng = SmallRng::seed_from_u64(4);
        // prev=None -> prev=initial=10s -> tripled=30s -> upper=min(30s,100ms)=100ms
        // upper(100ms) < lo(10s) -> hi_n=lo_n=10s, lo_u>=hi_u -> pick=lo_u=10s.
        let d = p.delay_for(1, None, &mut rng);
        assert_eq!(d, Duration::from_secs(10));
    }

    #[test]
    fn base_delay_negative_factor_is_clamped_to_zero() {
        // Negative multiplier -> scaled < 0 -> clamped to 0.
        let p = RetryPolicy {
            jitter: Jitter::None,
            backoff_multiplier: -2.0,
            ..RetryPolicy::default()
        };
        let mut rng = SmallRng::seed_from_u64(0);
        // attempt=2 -> exponent=1 -> factor=-2.0 -> scaled<0 -> clamped to 0.
        assert_eq!(p.delay_for(2, None, &mut rng), Duration::ZERO);
    }

    #[test]
    fn base_delay_infinite_factor_saturates_at_max() {
        // Multiplier that overflows to infinity should saturate at max_delay.
        let p = RetryPolicy {
            jitter: Jitter::None,
            backoff_multiplier: f64::MAX,
            max_delay: Duration::from_secs(5),
            ..RetryPolicy::default()
        };
        let mut rng = SmallRng::seed_from_u64(0);
        // attempt=10 -> factor = f64::MAX^9 = inf
        assert_eq!(p.delay_for(10, None, &mut rng), Duration::from_secs(5));
    }

    #[test]
    fn u128_min_picks_smaller_first_arg() {
        assert_eq!(u128_min(3, 7), 3);
        assert_eq!(u128_min(9, 2), 2);
    }

    #[test]
    fn nanos_to_u64_saturates_above_max() {
        assert_eq!(nanos_to_u64(u128::from(u64::MAX) + 1), u64::MAX);
        assert_eq!(nanos_to_u64(42), 42);
    }

    #[test]
    fn manual_clock_default_works() {
        let c = ManualClock::default();
        let t0 = c.now();
        c.sleep(Duration::from_millis(1));
        assert!(c.now() > t0);
    }

    #[test]
    fn system_clock_sleeps_and_now_advances() {
        let c = SystemClock;
        let t0 = c.now();
        c.sleep(Duration::from_millis(1));
        let t1 = c.now();
        assert!(t1 >= t0);
    }

    #[test]
    fn system_clock_retry_with_real_sleep() {
        // Exercise `retry` (public) which routes through SystemClock.
        let p = RetryPolicy {
            max_attempts: 2,
            initial_delay: Duration::from_millis(1),
            max_delay: Duration::from_millis(1),
            backoff_multiplier: 1.0,
            jitter: Jitter::None,
        };
        let classifier = retry_all::<&'static str>();
        let attempts = AtomicU32::new(0);
        let result: Result<u32, RetryError<&'static str>> = retry(&p, &classifier, |_| {
            if attempts.fetch_add(1, Ordering::SeqCst) == 0 {
                Err("nope")
            } else {
                Ok(1)
            }
        });
        assert_eq!(result.ok(), Some(1));
    }

    #[test]
    fn retry_error_source_is_inner_error() {
        use std::io;
        let inner = io::Error::other("io boom");
        let fatal: RetryError<io::Error> = RetryError::Fatal(inner);
        let src = std::error::Error::source(&fatal);
        assert!(src.is_some());
        let exh: RetryError<io::Error> = RetryError::Exhausted {
            attempts: 2,
            last_error: io::Error::other("io again"),
        };
        let src = std::error::Error::source(&exh);
        assert!(src.is_some());
    }

    #[test]
    fn fatal_classifier_on_final_attempt_returns_fatal() {
        // Drive a scenario where the LAST attempt's error is classified Fatal.
        let p = RetryPolicy {
            max_attempts: 2,
            jitter: Jitter::None,
            ..RetryPolicy::default()
        };
        let clock = ManualClock::new();
        let attempts = AtomicU32::new(0);
        let classifier = |_err: &&'static str| {
            // First attempt -> Retry, second -> Fatal.
            let n = attempts.fetch_add(1, Ordering::SeqCst);
            if n == 0 {
                RetryDecision::Retry
            } else {
                RetryDecision::Fatal
            }
        };
        let result: Result<u32, RetryError<&'static str>> =
            retry_with_clock(&p, &classifier, &clock, |_| Err("err"));
        assert!(matches!(result, Err(RetryError::Fatal("err"))));
    }

    #[test]
    fn delay_plan_constructs_and_compares() {
        let a = DelayPlan(Duration::from_millis(50));
        let b = DelayPlan(Duration::from_millis(50));
        let c = DelayPlan(Duration::from_millis(60));
        assert_eq!(a, b);
        assert_ne!(a, c);
        // Trigger Debug+Clone formatters.
        let _ = format!("{a:?}");
        let _ = a;
    }

    #[test]
    fn classifier_blanket_impl_via_closure() {
        // Ensures `impl<E,F> RetryClassifier<E> for F where F: Fn(&E)->RetryDecision`
        // is exercised.
        fn takes<C: RetryClassifier<&'static str>>(c: &C, e: &&'static str) -> RetryDecision {
            c.classify(e)
        }
        let f = |_e: &&'static str| RetryDecision::Retry;
        assert_eq!(takes(&f, &"x"), RetryDecision::Retry);
    }

    #[test]
    fn manual_clock_sleeps_records_all() {
        let c = ManualClock::new();
        c.sleep(Duration::from_millis(5));
        c.sleep(Duration::from_millis(10));
        let s = c.sleeps();
        assert_eq!(s, vec![Duration::from_millis(5), Duration::from_millis(10)]);
    }

    #[test]
    fn full_jitter_huge_base_saturates_via_min_branch() {
        // base.as_nanos > u64::MAX exercises u128_min picking `b<a`.
        let p = RetryPolicy {
            jitter: Jitter::Full,
            initial_delay: Duration::from_secs(u64::MAX / 2),
            max_delay: Duration::new(u64::MAX, 0),
            backoff_multiplier: 2.0,
            ..RetryPolicy::default()
        };
        let mut rng = SmallRng::seed_from_u64(0);
        // attempt=5 with huge initial saturates; just must not panic.
        let _ = p.delay_for(5, None, &mut rng);
    }
}
