//! Stable identifiers: `RoleId` and `ModelId`.

use std::fmt;

use serde::{Deserialize, Serialize};

/// Identifies an orchestration role (planner, coder, router, etc.).
///
/// See `plan/17-agent-roles.md` for the full taxonomy and `plan/15-model-switching.md`
/// for the role -> model mapping at runtime.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RoleId(String);

impl RoleId {
    /// Build a `RoleId` from any string-like value.
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// Borrow the underlying name.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for RoleId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<&str> for RoleId {
    fn from(s: &str) -> Self {
        Self::new(s)
    }
}

impl From<String> for RoleId {
    fn from(s: String) -> Self {
        Self::new(s)
    }
}

/// Stable identifier for a weight in the model registry, e.g. `gemma-4-e4b-q4_k_m`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ModelId(String);

impl ModelId {
    /// Build a `ModelId` from any string-like value.
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// Borrow the underlying name.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ModelId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<&str> for ModelId {
    fn from(s: &str) -> Self {
        Self::new(s)
    }
}

impl From<String> for ModelId {
    fn from(s: String) -> Self {
        Self::new(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn role_id_roundtrip_str() {
        let r = RoleId::from("coder");
        assert_eq!(r.as_str(), "coder");
        assert_eq!(format!("{r}"), "coder");
    }

    #[test]
    fn role_id_roundtrip_string() {
        let r = RoleId::from(String::from("polisher"));
        assert_eq!(r.as_str(), "polisher");
    }

    #[test]
    fn model_id_roundtrip_str() {
        let m = ModelId::from("gemma-4-e4b-q4_k_m");
        assert_eq!(m.as_str(), "gemma-4-e4b-q4_k_m");
        assert_eq!(format!("{m}"), "gemma-4-e4b-q4_k_m");
    }

    #[test]
    fn model_id_roundtrip_string() {
        let m = ModelId::from(String::from("qwen3-0.6b"));
        assert_eq!(m.as_str(), "qwen3-0.6b");
    }

    #[test]
    fn role_id_equality_and_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(RoleId::from("router"));
        assert!(set.contains(&RoleId::from("router")));
        assert!(!set.contains(&RoleId::from("planner")));
    }

    #[test]
    fn model_id_serde_roundtrip() {
        let m = ModelId::from("gemma-4-e4b-q4_k_m");
        let s = serde_json::to_string(&m).unwrap();
        let back: ModelId = serde_json::from_str(&s).unwrap();
        assert_eq!(m, back);
    }

    #[test]
    fn role_id_new_constructor() {
        let r = RoleId::new("critic");
        assert_eq!(r.as_str(), "critic");
    }

    #[test]
    fn model_id_new_constructor() {
        let m = ModelId::new("qwen3-coder-7b-q4_k_m");
        assert_eq!(m.as_str(), "qwen3-coder-7b-q4_k_m");
    }
}
