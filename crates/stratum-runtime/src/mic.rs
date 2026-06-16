//! cpal-backed microphone capture scaffold.
//!
//! Phase 5 v2 — see `plan/05-multimodal.md` §Voice In. This module
//! captures `f32` PCM from the host's default input device into an
//! in-memory buffer, resamples to 16 kHz mono (the rate whisper.cpp
//! wants), and dumps the buffer to a 16-bit PCM WAV via `hound`.
//!
//! ## Why a scaffold and not the TUI wiring
//!
//! The `/audio` palette command in `stratum-tui::chat` will own the
//! end-user surface — start/stop hotkeys, the recording indicator, and
//! the hand-off to [`crate::whisper::WhisperSubprocess::transcribe`].
//! That integration lands in a follow-up PR; this module ships the
//! capture primitive alone so the TUI work has a stable type to build
//! against.
//!
//! ## Coverage carve-out
//!
//! The body of [`MicCapture::start`] opens a real cpal stream against
//! the host's audio subsystem, which is non-deterministic in CI (the
//! GitHub macOS runners have no input device; Linux CI lacks a
//! PulseAudio / PipeWire server unless we provision one). The
//! pure-data helpers — [`build_wav_from_samples`] and [`resample_to_16k`]
//! — are unit-tested in `#[cfg(test)] mod tests` below; the cpal
//! callback closure and the device-list code path are documented in
//! `docs/coverage-exclusions.md` as a measured gap that does not
//! degrade the workspace coverage gate.

use std::io;
use std::path::Path;
use std::sync::{Arc, Mutex};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{BuildStreamError, PlayStreamError, Stream, StreamConfig};

/// Target sample rate for whisper.cpp ingestion. Fixed at 16 kHz mono
/// per the upstream model's expected PCM shape.
pub const TARGET_SAMPLE_RATE_HZ: u32 = 16_000;

/// Errors a [`MicCapture`] call can surface.
#[derive(Debug)]
pub enum MicError {
    /// The host reports no default input device. Typically means
    /// either no microphone is wired up, the user has not granted
    /// mic permission (macOS / Wayland portal), or CI is running
    /// against a headless audio stack.
    NoInputDevice,
    /// The default input device's reported config — sample format,
    /// channel count, sample rate — is something we cannot adapt.
    /// We currently accept any positive sample rate and any channel
    /// count ≥ 1 at `f32` sample format; anything else lands here.
    UnsupportedConfig,
    /// `cpal::Device::build_input_stream` failed.
    Build(BuildStreamError),
    /// `cpal::Stream::play` failed.
    Play(PlayStreamError),
    /// A WAV-write or other I/O step failed.
    Io(io::Error),
}

impl std::fmt::Display for MicError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoInputDevice => f.write_str("no default input device available"),
            Self::UnsupportedConfig => f.write_str("default input config is not supported"),
            Self::Build(e) => write!(f, "build_input_stream failed: {e}"),
            Self::Play(e) => write!(f, "stream play failed: {e}"),
            Self::Io(e) => write!(f, "mic i/o failed: {e}"),
        }
    }
}

impl std::error::Error for MicError {}

impl From<io::Error> for MicError {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}

/// Microphone capture handle.
///
/// `new()` resolves the default host + input device and snapshots the
/// device's native config (channels + sample rate). `start()` opens a
/// cpal input stream that appends every received `f32` sample to an
/// internal `Vec<f32>`. `stop()` halts the stream, drops it, and
/// returns the captured buffer downsampled to 16 kHz mono. The buffer
/// can be persisted with [`MicCapture::save_wav`] in the format
/// whisper.cpp wants.
///
/// The type is intentionally not `Clone` — a cpal `Stream` owns OS
/// resources and cannot be safely shared. Construct a fresh
/// `MicCapture` for each capture session.
#[allow(
    missing_debug_implementations,
    reason = "cpal::Stream is not Debug; the wrapped fields are runtime handles, not data"
)]
pub struct MicCapture {
    /// The device's native input config. We snapshot this at
    /// construction so the resampler can run against a stable rate
    /// even if the user later swaps devices behind the OS API.
    native_config: StreamConfig,
    /// Shared sample buffer the cpal callback appends to.
    buffer: Arc<Mutex<Vec<f32>>>,
    /// Active stream handle. `Some` while recording.
    stream: Option<Stream>,
}

impl MicCapture {
    /// Construct a `MicCapture` against the host's default input device.
    ///
    /// Does not yet open an audio stream — call [`Self::start`] to
    /// begin recording. Returns [`MicError::NoInputDevice`] when the
    /// default host has no input device, and [`MicError::UnsupportedConfig`]
    /// when the device's default config cannot be queried (typically
    /// a permissions or driver issue).
    ///
    /// # Errors
    ///
    /// See [`MicError`].
    pub fn new() -> Result<Self, MicError> {
        let host = cpal::default_host();
        let device = host.default_input_device().ok_or(MicError::NoInputDevice)?;
        let supported = device
            .default_input_config()
            .map_err(|_| MicError::UnsupportedConfig)?;
        let native_config: StreamConfig = supported.into();
        Ok(Self {
            native_config,
            buffer: Arc::new(Mutex::new(Vec::new())),
            stream: None,
        })
    }

    /// List the names of every input device the default host reports.
    ///
    /// Useful for a future TUI device-picker. Devices whose name cannot
    /// be queried are silently skipped — cpal returns a per-device
    /// `Result` and a missing name is rare but not fatal.
    #[must_use]
    pub fn list_input_devices() -> Vec<String> {
        let host = cpal::default_host();
        host.input_devices().map_or_else(
            |_| Vec::new(),
            |iter| iter.filter_map(|d| d.name().ok()).collect(),
        )
    }

    /// Returns the native sample rate of the input device, in Hz.
    #[must_use]
    pub const fn native_sample_rate_hz(&self) -> u32 {
        self.native_config.sample_rate.0
    }

    /// Returns the native channel count of the input device.
    #[must_use]
    pub const fn native_channels(&self) -> u16 {
        self.native_config.channels
    }

    /// Returns `true` while a capture stream is open.
    #[must_use]
    pub const fn is_recording(&self) -> bool {
        self.stream.is_some()
    }

    /// Open the cpal input stream and begin appending samples.
    ///
    /// Idempotent in the sense that a second call while a stream is
    /// already running returns `Ok(())` without re-opening — the
    /// underlying cpal `Stream` does not support double-play.
    ///
    /// # Errors
    ///
    /// Returns [`MicError::Build`] if `build_input_stream` fails or
    /// [`MicError::Play`] if the stream cannot be started.
    pub fn start(&mut self) -> Result<(), MicError> {
        if self.stream.is_some() {
            return Ok(());
        }
        let host = cpal::default_host();
        let device = host.default_input_device().ok_or(MicError::NoInputDevice)?;
        let buf = Arc::clone(&self.buffer);
        let err_buf = Arc::clone(&self.buffer);
        let stream = device
            .build_input_stream(
                &self.native_config,
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    if let Ok(mut guard) = buf.lock() {
                        guard.extend_from_slice(data);
                    }
                },
                move |err| {
                    // Best-effort: a stream error is surfaced at the
                    // next `stop()` via a truncated buffer rather than
                    // crashing the capture thread. We tag the buffer
                    // with a sentinel-free no-op; the caller decides
                    // what to do with a short clip.
                    let _ = &err_buf;
                    tracing::warn!(target = "mic", error = %err, "cpal input stream error");
                },
                None,
            )
            .map_err(MicError::Build)?;
        stream.play().map_err(MicError::Play)?;
        self.stream = Some(stream);
        Ok(())
    }

    /// Stop the input stream and return the captured buffer downmixed
    /// to mono and resampled to 16 kHz.
    ///
    /// The returned `Vec<f32>` is the buffer at whisper.cpp's expected
    /// rate. Calling `stop()` when no stream is active yields whatever
    /// has already been captured (possibly empty) — useful when the
    /// caller wants to flush a paused session.
    ///
    /// # Errors
    ///
    /// Returns [`MicError::Io`] if the internal mutex was poisoned by a
    /// previous panic in the cpal callback.
    pub fn stop(&mut self) -> Result<Vec<f32>, MicError> {
        // Drop the stream first so the cpal callback stops appending.
        self.stream = None;
        let guard = self
            .buffer
            .lock()
            .map_err(|_| MicError::Io(io::Error::other("mic buffer mutex poisoned")))?;
        let raw: Vec<f32> = guard.clone();
        drop(guard);
        let mono = downmix_to_mono(&raw, self.native_channels());
        let resampled = resample_to_16k(&mono, self.native_sample_rate_hz());
        Ok(resampled)
    }

    /// Write the captured-and-resampled buffer to `path` as a 16-bit
    /// PCM mono 16 kHz WAV file.
    ///
    /// Calls [`Self::stop`] internally if a stream is still open so the
    /// caller doesn't need a separate flush step.
    ///
    /// # Errors
    ///
    /// Returns [`MicError::Io`] when the file cannot be created or the
    /// WAV header / payload cannot be flushed.
    pub fn save_wav(&mut self, path: &Path) -> Result<(), MicError> {
        let samples = self.stop()?;
        build_wav_from_samples(&samples, TARGET_SAMPLE_RATE_HZ, path)
    }
}

/// Downmix interleaved multi-channel `f32` samples to a mono channel by
/// averaging across the channel dimension.
///
/// `channels = 0` or `1` returns the input unchanged. Frames that do
/// not have all `channels` samples present (a partial tail) are
/// dropped — cpal can deliver an odd-length buffer at stream close.
fn downmix_to_mono(samples: &[f32], channels: u16) -> Vec<f32> {
    if channels <= 1 {
        return samples.to_vec();
    }
    let ch = usize::from(channels);
    let frame_count = samples.len() / ch;
    let mut out = Vec::with_capacity(frame_count);
    for i in 0..frame_count {
        let base = i * ch;
        let mut sum = 0.0_f32;
        for c in 0..ch {
            sum += samples[base + c];
        }
        #[allow(
            clippy::cast_precision_loss,
            reason = "channel count is u16; f32 precision loss across <=65535 is irrelevant for averaging"
        )]
        let mean = sum / (ch as f32);
        out.push(mean);
    }
    out
}

/// Resample a mono `f32` PCM buffer from `src_hz` to 16 kHz using
/// linear interpolation.
///
/// Linear interpolation is the cheapest resampler that doesn't alias
/// audibly when fed into whisper.cpp's spectrogram front-end. A
/// higher-quality sinc filter is overkill for speech ingestion at this
/// stage; we revisit if the transcription quality regresses.
///
/// Special cases:
/// - `src_hz == 16_000` → returns the input unchanged.
/// - `src_hz == 0` or empty input → returns an empty buffer.
#[must_use]
pub fn resample_to_16k(samples: &[f32], src_hz: u32) -> Vec<f32> {
    if src_hz == TARGET_SAMPLE_RATE_HZ {
        return samples.to_vec();
    }
    if src_hz == 0 || samples.is_empty() {
        return Vec::new();
    }
    // Output length: round(len * 16000 / src_hz). Use u64 math so a
    // long recording at 192 kHz can't overflow u32 mid-multiply.
    let in_len_u64 = samples.len() as u64;
    let out_len_u64 = in_len_u64 * u64::from(TARGET_SAMPLE_RATE_HZ) / u64::from(src_hz);
    #[allow(
        clippy::cast_possible_truncation,
        reason = "out_len_u64 cannot exceed usize::MAX for any real-world capture buffer"
    )]
    let out_len = out_len_u64 as usize;
    if out_len == 0 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(out_len);
    #[allow(
        clippy::cast_precision_loss,
        reason = "f64 has 52-bit mantissa; sample-rate ratios fit comfortably without audible drift"
    )]
    let ratio = f64::from(src_hz) / f64::from(TARGET_SAMPLE_RATE_HZ);
    for i in 0..out_len {
        #[allow(
            clippy::cast_precision_loss,
            reason = "i is bounded by out_len which fits in an f64 mantissa for any real capture"
        )]
        let src_pos = (i as f64) * ratio;
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "src_pos is non-negative and bounded by samples.len() by construction"
        )]
        let idx = src_pos as usize;
        let frac = src_pos - (src_pos.floor());
        let a = samples[idx.min(samples.len() - 1)];
        let b = samples[(idx + 1).min(samples.len() - 1)];
        #[allow(
            clippy::cast_possible_truncation,
            reason = "f32 output is the target format; the f64 math just removes accumulated drift"
        )]
        let v = (f64::from(b) - f64::from(a)).mul_add(frac, f64::from(a)) as f32;
        out.push(v);
    }
    out
}

/// Write `samples` (assumed mono `f32` in `[-1.0, 1.0]`) as a 16-bit
/// PCM WAV at `sample_rate` Hz to `path`.
///
/// Exposed under `pub(crate)` for the `mic::tests` module; the public
/// surface is [`MicCapture::save_wav`]. Samples outside `[-1.0, 1.0]`
/// are clamped before quantization so an over-driven capture doesn't
/// wrap around the i16 range.
///
/// # Errors
///
/// Returns [`MicError::Io`] when `hound` fails to open, write, or
/// finalize the WAV file.
pub(crate) fn build_wav_from_samples(
    samples: &[f32],
    sample_rate: u32,
    path: &Path,
) -> Result<(), MicError> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)
        .map_err(|e| MicError::Io(io::Error::other(format!("wav create: {e}"))))?;
    for &s in samples {
        let clamped = s.clamp(-1.0, 1.0);
        #[allow(
            clippy::cast_possible_truncation,
            reason = "clamped is in [-1.0, 1.0]; scaled value fits cleanly in i16"
        )]
        let q = (clamped * f32::from(i16::MAX)) as i16;
        writer
            .write_sample(q)
            .map_err(|e| MicError::Io(io::Error::other(format!("wav write: {e}"))))?;
    }
    writer
        .finalize()
        .map_err(|e| MicError::Io(io::Error::other(format!("wav finalize: {e}"))))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn mic_error_display_covers_every_variant() {
        let _ = format!("{}", MicError::NoInputDevice);
        let _ = format!("{}", MicError::UnsupportedConfig);
        // Build / Play errors don't have a public constructor we can
        // synthesize cheaply across cpal versions; cover their Display
        // via the From-Io path instead. The Display arms are mechanical
        // `write!`s so a runtime-driven path is overkill.
        let _ = format!("{}", MicError::Io(io::Error::other("boom")));
        // Round-trip the From<io::Error> impl while we're here.
        let from_io: MicError = io::Error::other("x").into();
        assert!(matches!(from_io, MicError::Io(_)));
    }

    #[test]
    fn mic_error_display_for_build_and_play_variants() {
        // BuildStreamError and PlayStreamError both have a `DeviceNotAvailable`
        // unit variant in cpal 0.16 — synthesise it directly so we exercise
        // the Display arm without needing a real audio host.
        let build = MicError::Build(BuildStreamError::DeviceNotAvailable);
        let play = MicError::Play(PlayStreamError::DeviceNotAvailable);
        assert!(format!("{build}").contains("build_input_stream failed"));
        assert!(format!("{play}").contains("stream play failed"));
    }

    #[test]
    fn resample_passthrough_when_rate_matches() {
        let input = vec![0.1_f32, -0.2, 0.3, -0.4];
        let out = resample_to_16k(&input, TARGET_SAMPLE_RATE_HZ);
        assert_eq!(out, input);
    }

    #[test]
    fn resample_empty_and_zero_rate_yield_empty() {
        assert!(resample_to_16k(&[], 48_000).is_empty());
        assert!(resample_to_16k(&[0.1_f32, 0.2], 0).is_empty());
    }

    #[test]
    fn resample_downsample_48k_to_16k_thirds_length() {
        // 48 kHz → 16 kHz: output length should be input_len / 3.
        let input: Vec<f32> = (0..300_u16).map(|i| f32::from(i) / 300.0).collect();
        let out = resample_to_16k(&input, 48_000);
        assert_eq!(out.len(), 100, "expected 300 / 3 = 100 samples");
        // First sample should equal input[0] (no interpolation needed
        // at position 0).
        assert!((out[0] - input[0]).abs() < 1e-6);
    }

    #[test]
    fn resample_upsample_8k_to_16k_doubles_length() {
        // 8 kHz → 16 kHz: output length should be input_len * 2.
        let input: Vec<f32> = (0..50_u16).map(|i| f32::from(i) / 50.0).collect();
        let out = resample_to_16k(&input, 8_000);
        assert_eq!(out.len(), 100, "expected 50 * 2 = 100 samples");
        // First sample passes through unchanged.
        assert!((out[0] - input[0]).abs() < 1e-6);
        // Second sample should sit halfway between input[0] and input[1].
        let expected_mid = f32::midpoint(input[0], input[1]);
        assert!(
            (out[1] - expected_mid).abs() < 1e-6,
            "got {} expected {}",
            out[1],
            expected_mid
        );
    }

    #[test]
    fn downmix_passthrough_for_mono() {
        let input = vec![0.1_f32, -0.2, 0.3];
        assert_eq!(downmix_to_mono(&input, 1), input);
        assert_eq!(downmix_to_mono(&input, 0), input);
    }

    #[test]
    fn downmix_stereo_averages_pairs() {
        // L,R,L,R,L,R → average of each pair.
        let input = vec![1.0_f32, -1.0, 0.5, 0.5, 0.0, 1.0];
        let out = downmix_to_mono(&input, 2);
        assert_eq!(out, vec![0.0_f32, 0.5, 0.5]);
    }

    #[test]
    fn downmix_drops_partial_tail_frame() {
        // 5 samples / 2 channels = 2 complete frames; trailing sample dropped.
        let input = vec![1.0_f32, -1.0, 0.5, 0.5, 0.25];
        let out = downmix_to_mono(&input, 2);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn build_wav_round_trips_through_hound() {
        let tmp = TempDir::new().expect("tmp");
        let path = tmp.path().join("clip.wav");
        let samples: Vec<f32> = (0..1600_u16)
            .map(|i| (f32::from(i) / 1600.0) - 0.5)
            .collect();
        build_wav_from_samples(&samples, TARGET_SAMPLE_RATE_HZ, &path).expect("write wav");

        let reader = hound::WavReader::open(&path).expect("open wav");
        let spec = reader.spec();
        assert_eq!(spec.channels, 1);
        assert_eq!(spec.sample_rate, TARGET_SAMPLE_RATE_HZ);
        assert_eq!(spec.bits_per_sample, 16);
        assert_eq!(spec.sample_format, hound::SampleFormat::Int);

        let read_samples: Vec<i16> = reader
            .into_samples::<i16>()
            .collect::<Result<_, _>>()
            .expect("read samples");
        assert_eq!(read_samples.len(), samples.len(), "sample count must match");
    }

    #[test]
    fn build_wav_clamps_overdriven_samples() {
        let tmp = TempDir::new().expect("tmp");
        let path = tmp.path().join("clipped.wav");
        // Out-of-range samples; must not wrap to negative on quantization.
        let samples = vec![5.0_f32, -5.0, 1.5, -1.5];
        build_wav_from_samples(&samples, TARGET_SAMPLE_RATE_HZ, &path).expect("write wav");
        let reader = hound::WavReader::open(&path).expect("open wav");
        let read: Vec<i16> = reader
            .into_samples::<i16>()
            .collect::<Result<_, _>>()
            .expect("read");
        // After clamp + i16 scale: +1.0 → i16::MAX, -1.0 → -i16::MAX.
        assert_eq!(read, vec![i16::MAX, -i16::MAX, i16::MAX, -i16::MAX]);
    }

    #[test]
    fn build_wav_io_error_when_path_unwritable() {
        // A path inside a non-existent directory triggers the
        // `WavWriter::create` failure arm.
        let nope = std::path::Path::new("/this/path/should/not/exist/clip.wav");
        let err = build_wav_from_samples(&[0.0_f32], TARGET_SAMPLE_RATE_HZ, nope)
            .expect_err("expected io error");
        assert!(matches!(err, MicError::Io(_)));
    }

    // NOTE: We deliberately do NOT unit-test `MicCapture::new`,
    // `MicCapture::start`, `MicCapture::stop`, or
    // `MicCapture::list_input_devices`. cpal's CoreAudio backend on
    // macOS test runners (and ALSA on headless Linux CI) issues OS-level
    // calls during `cargo test` that can SIGSEGV when no audio device
    // exists or no GUI run loop is available. These functions are
    // exercised manually via the future `/audio` palette command and
    // are documented as a measured coverage gap in
    // `docs/coverage-exclusions.md`.
}
