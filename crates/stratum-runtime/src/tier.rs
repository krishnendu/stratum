//! Composite tier classifier per `plan/28-finalization-v2.md` §3.3.
//!
//! A box must meet **all** the "min" cells of a tier to land there; a GPU
//! presence bumps the tier up one notch (clamped at `high`). Falling below
//! the `low` floor is reported as `Tier::Low` with a separate refusal handled
//! by the caller.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::probe::{GpuBackend, HardwareProbe};

/// Classification of the host's overall capability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Tier {
    /// Resident tier only; no swap.
    Low,
    /// Resident + optional small dense swap.
    Medium,
    /// Resident + dense 7B swap.
    High,
}

impl fmt::Display for Tier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
        };
        f.write_str(s)
    }
}

impl Tier {
    /// Classify a probe into a tier. Pure function; no side effects.
    #[must_use]
    pub fn classify(probe: &HardwareProbe) -> Self {
        let base = Self::base_tier(probe);
        if has_acceleration(probe.gpu) {
            base.bumped()
        } else {
            base
        }
    }

    fn base_tier(probe: &HardwareProbe) -> Self {
        let has_simd = probe
            .cpu_features
            .iter()
            .any(|f| matches!(f.as_str(), "avx2" | "avx512f" | "neon"));
        if probe.ram_total_mib >= 16 * 1024 && has_simd {
            Self::High
        } else if probe.ram_total_mib >= 8 * 1024 && has_simd {
            Self::Medium
        } else {
            Self::Low
        }
    }

    /// One notch up; saturates at `High`.
    #[must_use]
    pub const fn bumped(self) -> Self {
        match self {
            Self::Low => Self::Medium,
            Self::Medium | Self::High => Self::High,
        }
    }
}

const fn has_acceleration(gpu: GpuBackend) -> bool {
    !matches!(gpu, GpuBackend::Cpu)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::probe::GpuBackend;

    fn probe(ram_mib: u32, features: Vec<&'static str>, gpu: GpuBackend) -> HardwareProbe {
        HardwareProbe::synthetic(ram_mib, ram_mib / 2, "aarch64", features, 8, gpu, "linux")
    }

    #[test]
    fn low_below_8gb() {
        let p = probe(4096, vec!["avx2"], GpuBackend::Cpu);
        assert_eq!(Tier::classify(&p), Tier::Low);
    }

    #[test]
    fn medium_at_12gb_no_gpu() {
        let p = probe(12 * 1024, vec!["avx2"], GpuBackend::Cpu);
        assert_eq!(Tier::classify(&p), Tier::Medium);
    }

    #[test]
    fn high_at_16gb_no_gpu() {
        let p = probe(16 * 1024, vec!["avx2"], GpuBackend::Cpu);
        assert_eq!(Tier::classify(&p), Tier::High);
    }

    #[test]
    fn gpu_bumps_12gb_to_high() {
        let p = probe(12 * 1024, vec!["neon"], GpuBackend::Metal);
        assert_eq!(Tier::classify(&p), Tier::High);
    }

    #[test]
    fn gpu_bumps_low_to_medium() {
        let p = probe(4096, vec!["neon"], GpuBackend::Metal);
        assert_eq!(Tier::classify(&p), Tier::Medium);
    }

    #[test]
    fn no_simd_drops_to_low() {
        let p = probe(16 * 1024, vec![], GpuBackend::Cpu);
        assert_eq!(Tier::classify(&p), Tier::Low);
    }

    #[test]
    fn bumped_saturates_at_high() {
        assert_eq!(Tier::High.bumped(), Tier::High);
        assert_eq!(Tier::Medium.bumped(), Tier::High);
        assert_eq!(Tier::Low.bumped(), Tier::Medium);
    }

    #[test]
    fn display_renders_lowercase() {
        assert_eq!(format!("{}", Tier::Low), "low");
        assert_eq!(format!("{}", Tier::Medium), "medium");
        assert_eq!(format!("{}", Tier::High), "high");
    }

    #[test]
    fn serde_roundtrip() {
        for t in [Tier::Low, Tier::Medium, Tier::High] {
            let s = serde_json::to_string(&t).unwrap();
            let back: Tier = serde_json::from_str(&s).unwrap();
            assert_eq!(t, back);
        }
    }

    #[test]
    fn has_acceleration_for_each_backend() {
        assert!(has_acceleration(GpuBackend::Metal));
        assert!(has_acceleration(GpuBackend::Cuda));
        assert!(has_acceleration(GpuBackend::Vulkan));
        assert!(!has_acceleration(GpuBackend::Cpu));
    }

    #[test]
    fn classify_with_avx512_high_tier() {
        let p = probe(16 * 1024, vec!["avx512f"], GpuBackend::Cpu);
        assert_eq!(Tier::classify(&p), Tier::High);
    }
}
