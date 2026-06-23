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
///
/// Desktop tiers (`Low`, `Medium`, `High`) are derived from RAM + SIMD + GPU.
/// Mobile tiers (`MobileUltraLow`, `MobileLow`, `MobileMedium`, `MobileHigh`)
/// per `plan/21-mobile-clients.md` §2 are stubbed today and pinned to
/// `MobileMedium` on iOS / Android; real device-class detection is a follow-up.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Tier {
    /// Resident tier only; no swap.
    Low,
    /// Resident + optional small dense swap.
    Medium,
    /// Resident + dense 7B swap.
    High,
    /// Mid-range phones (< 6 GB total RAM); tiny model only, rule-based routing.
    MobileUltraLow,
    /// Flagship phones (~8 GB total RAM); compact dense crew.
    MobileLow,
    /// Top-end iPhone Pro / Pixel Pro / Samsung Ultra (12-16 GB RAM).
    MobileMedium,
    /// Foldables / tablets (16 GB+); dense 7B is in reach, still no 30B `MoE`.
    MobileHigh,
}

impl fmt::Display for Tier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::MobileUltraLow => "mobile_ultra_low",
            Self::MobileLow => "mobile_low",
            Self::MobileMedium => "mobile_medium",
            Self::MobileHigh => "mobile_high",
        };
        f.write_str(s)
    }
}

impl Tier {
    /// Classify a probe into a tier. Pure function; no side effects.
    ///
    /// On iOS / Android this stubs to [`Tier::MobileMedium`] until a real
    /// device-class detector lands (tracked in `plan/21` §2).
    #[must_use]
    pub fn classify(probe: &HardwareProbe) -> Self {
        #[cfg(target_os = "ios")]
        {
            let _ = probe;
            return Self::MobileMedium;
        }
        #[cfg(target_os = "android")]
        {
            let _ = probe;
            return Self::MobileMedium;
        }
        #[cfg(not(any(target_os = "ios", target_os = "android")))]
        {
            let base = Self::base_tier(probe);
            if has_acceleration(probe.gpu) {
                base.bumped()
            } else {
                base
            }
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

    /// One notch up the desktop ladder; saturates at `High`. Mobile tiers
    /// are returned unchanged — the mobile ladder is its own world and is
    /// not bumped by desktop signals like the presence of a discrete GPU.
    #[must_use]
    pub const fn bumped(self) -> Self {
        match self {
            Self::Low => Self::Medium,
            Self::Medium | Self::High => Self::High,
            Self::MobileUltraLow => Self::MobileLow,
            Self::MobileLow => Self::MobileMedium,
            Self::MobileMedium | Self::MobileHigh => Self::MobileHigh,
        }
    }

    /// `true` if this tier is one of the mobile rungs.
    #[must_use]
    pub const fn is_mobile(self) -> bool {
        matches!(
            self,
            Self::MobileUltraLow | Self::MobileLow | Self::MobileMedium | Self::MobileHigh
        )
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
    fn serde_roundtrip() {
        for t in [Tier::Low, Tier::Medium, Tier::High] {
            let s = serde_json::to_string(&t).unwrap();
            let back: Tier = serde_json::from_str(&s).unwrap();
            assert_eq!(t, back);
        }
    }

    #[test]
    fn mobile_serde_roundtrip() {
        for t in [
            Tier::MobileUltraLow,
            Tier::MobileLow,
            Tier::MobileMedium,
            Tier::MobileHigh,
        ] {
            let s = serde_json::to_string(&t).unwrap();
            let back: Tier = serde_json::from_str(&s).unwrap();
            assert_eq!(t, back);
        }
    }

    #[test]
    fn mobile_serde_uses_snake_case() {
        // Lock in the wire format so on-disk records do not drift.
        assert_eq!(
            serde_json::to_string(&Tier::MobileUltraLow).unwrap(),
            "\"mobile_ultra_low\""
        );
        assert_eq!(
            serde_json::to_string(&Tier::MobileLow).unwrap(),
            "\"mobile_low\""
        );
        assert_eq!(
            serde_json::to_string(&Tier::MobileMedium).unwrap(),
            "\"mobile_medium\""
        );
        assert_eq!(
            serde_json::to_string(&Tier::MobileHigh).unwrap(),
            "\"mobile_high\""
        );
    }

    #[test]
    fn display_renders_all_variants() {
        // Every variant must produce a stable, snake_case identifier so
        // logs / `stratum doctor` output / on-disk records stay stable.
        assert_eq!(format!("{}", Tier::Low), "low");
        assert_eq!(format!("{}", Tier::Medium), "medium");
        assert_eq!(format!("{}", Tier::High), "high");
        assert_eq!(format!("{}", Tier::MobileUltraLow), "mobile_ultra_low");
        assert_eq!(format!("{}", Tier::MobileLow), "mobile_low");
        assert_eq!(format!("{}", Tier::MobileMedium), "mobile_medium");
        assert_eq!(format!("{}", Tier::MobileHigh), "mobile_high");
    }

    #[test]
    fn mobile_bumped_walks_mobile_ladder() {
        assert_eq!(Tier::MobileUltraLow.bumped(), Tier::MobileLow);
        assert_eq!(Tier::MobileLow.bumped(), Tier::MobileMedium);
        assert_eq!(Tier::MobileMedium.bumped(), Tier::MobileHigh);
        assert_eq!(Tier::MobileHigh.bumped(), Tier::MobileHigh);
    }

    #[test]
    fn is_mobile_partitions_variants() {
        assert!(!Tier::Low.is_mobile());
        assert!(!Tier::Medium.is_mobile());
        assert!(!Tier::High.is_mobile());
        assert!(Tier::MobileUltraLow.is_mobile());
        assert!(Tier::MobileLow.is_mobile());
        assert!(Tier::MobileMedium.is_mobile());
        assert!(Tier::MobileHigh.is_mobile());
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
