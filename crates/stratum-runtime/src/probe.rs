//! Hardware probe used by `stratum doctor`, `stratum init`, and the
//! memory-safety gate.
//!
//! The probe is deliberately deterministic: every input comes from a clearly
//! named source so tests can construct synthetic probes for the tier
//! classifier and the install record without polluting the host.

use std::fmt;

use serde::{Deserialize, Serialize};
use sysinfo::System;

/// GPU acceleration backend selected at first run.
///
/// Detection priority per `plan/18-first-run-and-system-tiers.md` §5 (v2 cuts
/// the matrix to Metal / CUDA / Vulkan / CPU; `ROCm` and `OpenCL` deferred).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GpuBackend {
    /// Apple Metal (Apple Silicon).
    Metal,
    /// NVIDIA CUDA.
    Cuda,
    /// Cross-vendor Vulkan.
    Vulkan,
    /// CPU only.
    Cpu,
}

impl fmt::Display for GpuBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            Self::Metal => "metal",
            Self::Cuda => "cuda",
            Self::Vulkan => "vulkan",
            Self::Cpu => "cpu",
        };
        f.write_str(label)
    }
}

/// Snapshot of the host capabilities relevant to tier classification.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HardwareProbe {
    /// Total physical RAM in mebibytes.
    pub ram_total_mib: u32,
    /// Currently available RAM in mebibytes (used by the safety gate at startup).
    pub ram_available_mib: u32,
    /// CPU architecture identifier, e.g. `aarch64`, `x86_64`.
    pub cpu_arch: String,
    /// Sorted list of detected SIMD features (`avx2`, `avx512f`, `neon`, …).
    pub cpu_features: Vec<String>,
    /// Number of logical CPU cores.
    pub cpu_cores: u32,
    /// Selected GPU backend.
    pub gpu: GpuBackend,
    /// Operating system identifier (`macos`, `linux`, `windows`).
    pub os: String,
}

impl HardwareProbe {
    /// Probe the live host. Reads RAM and CPU info from [`sysinfo`] and infers
    /// the GPU backend from build target + presence of the NVIDIA driver tool
    /// in `PATH`.
    #[must_use]
    pub fn run() -> Self {
        let mut sys = System::new();
        sys.refresh_memory();
        sys.refresh_cpu_all();
        let ram_total_mib = mib_from_bytes(sys.total_memory());
        let ram_available_mib = mib_from_bytes(sys.available_memory());
        let cpu_features = detect_cpu_features()
            .into_iter()
            .map(str::to_string)
            .collect();
        let cpu_cores = u32::try_from(sys.cpus().len()).unwrap_or(u32::MAX);
        let arch = std::env::consts::ARCH;
        let os = std::env::consts::OS;
        let has_nvidia = has_in_path("nvidia-smi");
        Self {
            ram_total_mib,
            ram_available_mib,
            cpu_arch: pick_cpu_arch(arch).to_string(),
            cpu_features,
            cpu_cores,
            gpu: pick_gpu(arch, os, has_nvidia),
            os: pick_os(os).to_string(),
        }
    }

    /// Construct a synthetic probe for tier-classifier tests.
    #[must_use]
    pub fn synthetic(
        ram_total_mib: u32,
        ram_available_mib: u32,
        cpu_arch: &str,
        cpu_features: Vec<&str>,
        cpu_cores: u32,
        gpu: GpuBackend,
        os: &str,
    ) -> Self {
        Self {
            ram_total_mib,
            ram_available_mib,
            cpu_arch: cpu_arch.to_string(),
            cpu_features: cpu_features.into_iter().map(str::to_string).collect(),
            cpu_cores,
            gpu,
            os: os.to_string(),
        }
    }
}

fn mib_from_bytes(bytes: u64) -> u32 {
    let mib = bytes / 1_048_576;
    u32::try_from(mib).unwrap_or(u32::MAX)
}

fn pick_cpu_arch(target_arch: &str) -> &'static str {
    match target_arch {
        "aarch64" => "aarch64",
        "x86_64" => "x86_64",
        _ => "unknown",
    }
}

fn pick_os(target_os: &str) -> &'static str {
    match target_os {
        "macos" => "macos",
        "linux" => "linux",
        "windows" => "windows",
        _ => "unknown",
    }
}

fn pick_gpu(target_arch: &str, target_os: &str, has_nvidia: bool) -> GpuBackend {
    if target_arch == "aarch64" && target_os == "macos" {
        return GpuBackend::Metal;
    }
    if has_nvidia {
        return GpuBackend::Cuda;
    }
    if matches!(target_os, "linux" | "windows") {
        return GpuBackend::Vulkan;
    }
    GpuBackend::Cpu
}

fn detect_cpu_features() -> Vec<&'static str> {
    let mut out: Vec<&'static str> = Vec::new();
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            out.push("avx2");
        }
        if std::is_x86_feature_detected!("avx512f") {
            out.push("avx512f");
        }
        if std::is_x86_feature_detected!("sse4.2") {
            out.push("sse4.2");
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            out.push("neon");
        }
    }
    out.sort_unstable();
    out
}

fn has_in_path(cmd: &str) -> bool {
    let path = std::env::var_os("PATH").unwrap_or_default();
    has_in_path_with(cmd, &path)
}

fn has_in_path_with(cmd: &str, path_env: &std::ffi::OsStr) -> bool {
    if path_env.is_empty() {
        return false;
    }
    std::env::split_paths(path_env).any(|dir| {
        let mut candidate = dir;
        candidate.push(cmd);
        candidate.is_file()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mib_conversion_basic() {
        assert_eq!(mib_from_bytes(0), 0);
        assert_eq!(mib_from_bytes(1_048_576), 1);
        assert_eq!(mib_from_bytes(2 * 1_048_576), 2);
    }

    #[test]
    fn mib_conversion_saturates() {
        assert_eq!(mib_from_bytes(u64::MAX), u32::MAX);
    }

    #[test]
    fn pick_cpu_arch_known_targets() {
        assert_eq!(pick_cpu_arch("aarch64"), "aarch64");
        assert_eq!(pick_cpu_arch("x86_64"), "x86_64");
        assert_eq!(pick_cpu_arch("riscv64"), "unknown");
    }

    #[test]
    fn pick_os_known_targets() {
        assert_eq!(pick_os("macos"), "macos");
        assert_eq!(pick_os("linux"), "linux");
        assert_eq!(pick_os("windows"), "windows");
        assert_eq!(pick_os("freebsd"), "unknown");
    }

    #[test]
    fn pick_gpu_apple_silicon_is_metal() {
        assert_eq!(pick_gpu("aarch64", "macos", false), GpuBackend::Metal);
    }

    #[test]
    fn pick_gpu_nvidia_is_cuda() {
        assert_eq!(pick_gpu("x86_64", "linux", true), GpuBackend::Cuda);
    }

    #[test]
    fn pick_gpu_linux_no_nvidia_is_vulkan() {
        assert_eq!(pick_gpu("x86_64", "linux", false), GpuBackend::Vulkan);
    }

    #[test]
    fn pick_gpu_windows_no_nvidia_is_vulkan() {
        assert_eq!(pick_gpu("x86_64", "windows", false), GpuBackend::Vulkan);
    }

    #[test]
    fn pick_gpu_intel_mac_is_cpu() {
        // Intel Mac, no NVIDIA: Metal is not Apple-Silicon-class so we fall to CPU.
        assert_eq!(pick_gpu("x86_64", "macos", false), GpuBackend::Cpu);
    }

    #[test]
    fn pick_gpu_unknown_os_is_cpu() {
        assert_eq!(pick_gpu("x86_64", "freebsd", false), GpuBackend::Cpu);
    }

    #[test]
    fn pick_gpu_nvidia_beats_vulkan_priority() {
        // On Linux with both signals, Cuda wins over Vulkan.
        assert_eq!(pick_gpu("x86_64", "linux", true), GpuBackend::Cuda);
    }

    #[test]
    fn detect_cpu_features_returns_sorted_unique() {
        let f = detect_cpu_features();
        // Each entry appears at most once and the slice is sorted.
        let mut sorted = f.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(f, sorted);
    }

    #[cfg(unix)]
    #[test]
    fn has_in_path_finds_sh_on_unix() {
        assert!(has_in_path("sh"));
    }

    #[test]
    fn has_in_path_missing_returns_false() {
        assert!(!has_in_path("definitely-not-a-real-binary-xyzzy"));
    }

    #[test]
    fn has_in_path_with_empty_returns_false() {
        let empty = std::ffi::OsString::new();
        assert!(!has_in_path_with("sh", &empty));
    }

    #[test]
    fn has_in_path_unwraps_env_default() {
        // Round-trip the live PATH through the unwrap-or-default branch.
        let _ = has_in_path("definitely-not-real-xyzzy");
    }

    #[test]
    fn probe_run_produces_sane_values() {
        let p = HardwareProbe::run();
        assert!(p.ram_total_mib > 0);
        assert!(p.ram_available_mib > 0);
        assert!(p.cpu_cores > 0);
    }

    #[test]
    fn probe_synthetic_constructor() {
        let p = HardwareProbe::synthetic(
            12_288,
            7_400,
            "aarch64",
            vec!["neon"],
            8,
            GpuBackend::Metal,
            "macos",
        );
        assert_eq!(p.ram_total_mib, 12_288);
        assert_eq!(p.gpu, GpuBackend::Metal);
        assert_eq!(p.cpu_arch, "aarch64");
    }

    #[test]
    fn probe_serde_roundtrip() {
        let p = HardwareProbe::synthetic(
            16_384,
            12_000,
            "x86_64",
            vec!["avx2", "sse4.2"],
            16,
            GpuBackend::Cuda,
            "linux",
        );
        let s = serde_json::to_string(&p).unwrap();
        let back: HardwareProbe = serde_json::from_str(&s).unwrap();
        assert_eq!(p, back);
    }

    #[test]
    fn gpu_backend_display() {
        assert_eq!(format!("{}", GpuBackend::Metal), "metal");
        assert_eq!(format!("{}", GpuBackend::Cuda), "cuda");
        assert_eq!(format!("{}", GpuBackend::Vulkan), "vulkan");
        assert_eq!(format!("{}", GpuBackend::Cpu), "cpu");
    }

    #[test]
    fn gpu_backend_serde_roundtrip() {
        for g in [
            GpuBackend::Metal,
            GpuBackend::Cuda,
            GpuBackend::Vulkan,
            GpuBackend::Cpu,
        ] {
            let s = serde_json::to_string(&g).unwrap();
            let back: GpuBackend = serde_json::from_str(&s).unwrap();
            assert_eq!(g, back);
        }
    }
}
