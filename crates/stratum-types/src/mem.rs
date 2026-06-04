//! Memory estimate carried with every `ModelHandle` and consumed by the
//! memory-safety gate (`plan/14-memory-safety-gate.md`).

use serde::{Deserialize, Serialize};

/// Per-model memory estimate. All values are mebibytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemEstimate {
    /// Resident set of the weights once loaded.
    pub weight_rss_mib: u32,
    /// KV cache size per token; multiplied by the planned context length.
    pub kv_per_token_bytes: u32,
    /// Optional multimodal projector overhead (mmproj).
    pub mmproj_mib: u32,
    /// VRAM cost when fully GPU-offloaded; `0` when CPU-only.
    pub vram_mib: u32,
}

impl MemEstimate {
    /// Total hot RAM estimate for a given context length, in mebibytes.
    #[must_use]
    pub fn hot_ram_mib(self, context_tokens: u32) -> u32 {
        let kv_bytes = u64::from(self.kv_per_token_bytes) * u64::from(context_tokens);
        let kv_mib = u32::try_from(kv_bytes / 1_048_576).unwrap_or(u32::MAX);
        self.weight_rss_mib
            .saturating_add(self.mmproj_mib)
            .saturating_add(kv_mib)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> MemEstimate {
        MemEstimate {
            weight_rss_mib: 3600,
            kv_per_token_bytes: 4096,
            mmproj_mib: 256,
            vram_mib: 0,
        }
    }

    #[test]
    fn hot_ram_with_zero_context() {
        assert_eq!(fixture().hot_ram_mib(0), 3600 + 256);
    }

    #[test]
    fn hot_ram_with_context() {
        // 8192 tokens * 4096 bytes / 1 MiB = 32 MiB.
        assert_eq!(fixture().hot_ram_mib(8192), 3600 + 256 + 32);
    }

    #[test]
    fn hot_ram_saturates_on_overflow() {
        let big = MemEstimate {
            weight_rss_mib: u32::MAX - 10,
            kv_per_token_bytes: 1,
            mmproj_mib: 100,
            vram_mib: 0,
        };
        assert_eq!(big.hot_ram_mib(0), u32::MAX);
    }

    #[test]
    fn serde_roundtrip() {
        let m = fixture();
        let s = serde_json::to_string(&m).unwrap();
        let back: MemEstimate = serde_json::from_str(&s).unwrap();
        assert_eq!(m, back);
    }
}
