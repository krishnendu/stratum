//! Provider registry + role-to-provider routing table.
//!
//! Phase 2 v2: the registry is a typed map `ModelId -> Arc<dyn Provider>` and
//! `RoleId -> ModelId`. Future Dispatcher logic (cross-provider fan-out,
//! capability-based query, memory-safety-gate activation) plugs in on top of
//! this storage layer. Today's shape is minimal but covers the API every
//! later subsystem needs: register, look up by id, bind a role, resolve a
//! role to its provider, list, and capability-based filtering.

use std::collections::HashMap;
use std::sync::Arc;

use stratum_types::error::codes::{E3007_MODEL_LOAD_REFUSED, E4002_AGENT_SHADOW};
use stratum_types::{Capability, ModelId, RoleId, StratumError, StratumResult};

use crate::provider::Provider;

/// Typed registry that owns providers and the role binding map.
#[derive(Clone, Default)]
pub struct Registry {
    providers: HashMap<ModelId, Arc<dyn Provider>>,
    roles: HashMap<RoleId, ModelId>,
}

impl std::fmt::Debug for Registry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Registry")
            .field(
                "providers",
                &self
                    .providers
                    .keys()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>(),
            )
            .field(
                "roles",
                &self
                    .roles
                    .iter()
                    .map(|(r, m)| (r.to_string(), m.to_string()))
                    .collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl Registry {
    /// Fresh empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a provider under its [`Provider::id`].
    ///
    /// Returns the previously-registered provider with the same id, if any.
    pub fn register(&mut self, provider: Arc<dyn Provider>) -> Option<Arc<dyn Provider>> {
        let id = ModelId::from(provider.id().to_string());
        self.providers.insert(id, provider)
    }

    /// Look up a provider by its model id.
    #[must_use]
    pub fn get(&self, id: &ModelId) -> Option<Arc<dyn Provider>> {
        self.providers.get(id).cloned()
    }

    /// Bind a role to a model id. Subsequent [`Self::resolve`] for that role
    /// returns the bound provider.
    ///
    /// # Errors
    /// Returns [`E3007_MODEL_LOAD_REFUSED`] if the model id is not in the
    /// registry — binding a role to an unknown provider is a programming
    /// error and surfaces immediately.
    pub fn bind_role(&mut self, role: RoleId, model: ModelId) -> StratumResult<()> {
        if !self.providers.contains_key(&model) {
            return Err(StratumError::new(
                E3007_MODEL_LOAD_REFUSED,
                format!(
                    "cannot bind role {role} to unknown model {model}; register the provider first"
                ),
            ));
        }
        self.roles.insert(role, model);
        Ok(())
    }

    /// Resolve a role to the provider currently bound to it.
    ///
    /// # Errors
    /// Returns [`E4002_AGENT_SHADOW`] if the role has no binding (an agent
    /// asked for a role no provider is serving).
    pub fn resolve(&self, role: &RoleId) -> StratumResult<Arc<dyn Provider>> {
        let model = self.roles.get(role).ok_or_else(|| {
            StratumError::new(E4002_AGENT_SHADOW, format!("no binding for role {role}"))
        })?;
        self.providers.get(model).cloned().ok_or_else(|| {
            StratumError::new(
                E3007_MODEL_LOAD_REFUSED,
                format!("role {role} bound to {model} but model is no longer registered"),
            )
        })
    }

    /// List every registered model id, sorted by string order.
    #[must_use]
    pub fn list_models(&self) -> Vec<ModelId> {
        let mut ids: Vec<_> = self.providers.keys().cloned().collect();
        ids.sort_by(|a, b| a.as_str().cmp(b.as_str()));
        ids
    }

    /// List every bound role, sorted by string order.
    #[must_use]
    pub fn list_roles(&self) -> Vec<(RoleId, ModelId)> {
        let mut entries: Vec<_> = self
            .roles
            .iter()
            .map(|(r, m)| (r.clone(), m.clone()))
            .collect();
        entries.sort_by(|a, b| a.0.as_str().cmp(b.0.as_str()));
        entries
    }

    /// Filter providers by capability.
    #[must_use]
    pub fn providers_with(&self, cap: Capability) -> Vec<Arc<dyn Provider>> {
        let mut out: Vec<_> = self
            .providers
            .values()
            .filter(|p| p.capabilities().contains(&cap))
            .cloned()
            .collect();
        out.sort_by(|a, b| a.id().cmp(b.id()));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::EchoProvider;

    fn echo(id: &str) -> Arc<dyn Provider> {
        // EchoProvider's hardcoded id is "echo"; tests that need distinct
        // ids wrap it in a thin newtype.
        if id == "echo" {
            Arc::new(EchoProvider::new(""))
        } else {
            Arc::new(NamedEcho::new(id))
        }
    }

    /// Test-only Provider wrapper that lets us mint multiple distinct ids.
    #[derive(Debug)]
    struct NamedEcho {
        id: String,
        inner: EchoProvider,
    }

    impl NamedEcho {
        fn new(id: &str) -> Self {
            Self {
                id: id.to_string(),
                inner: EchoProvider::new(""),
            }
        }
    }

    impl Provider for NamedEcho {
        fn id(&self) -> &str {
            &self.id
        }

        fn capabilities(&self) -> &'static [Capability] {
            const CAPS: &[Capability] = &[Capability::Generate];
            CAPS
        }

        fn generate(
            &self,
            request: &crate::provider::GenerateRequest,
            cancel: &crate::cancel::CancelToken,
        ) -> Vec<stratum_types::Block> {
            self.inner.generate(request, cancel)
        }
    }

    #[test]
    fn registry_starts_empty() {
        let r = Registry::new();
        assert!(r.list_models().is_empty());
        assert!(r.list_roles().is_empty());
    }

    #[test]
    fn register_then_get_roundtrip() {
        let mut r = Registry::new();
        let prior = r.register(echo("echo"));
        assert!(prior.is_none());
        let got = r.get(&ModelId::from("echo")).expect("registered");
        assert_eq!(got.id(), "echo");
    }

    #[test]
    fn register_replaces_existing() {
        let mut r = Registry::new();
        r.register(echo("dupe"));
        let prior = r.register(echo("dupe"));
        assert!(prior.is_some(), "second register returns the prior");
    }

    #[test]
    fn bind_role_and_resolve() {
        let mut r = Registry::new();
        r.register(echo("echo"));
        r.bind_role(RoleId::from("main"), ModelId::from("echo"))
            .unwrap();
        let provider = r.resolve(&RoleId::from("main")).unwrap();
        assert_eq!(provider.id(), "echo");
    }

    #[test]
    fn bind_role_to_unknown_model_errors() {
        let mut r = Registry::new();
        let err = r
            .bind_role(RoleId::from("main"), ModelId::from("missing"))
            .unwrap_err();
        assert_eq!(err.code(), &E3007_MODEL_LOAD_REFUSED);
    }

    #[test]
    fn resolve_unbound_role_errors() {
        let r = Registry::new();
        let err = r.resolve(&RoleId::from("main")).unwrap_err();
        assert_eq!(err.code(), &E4002_AGENT_SHADOW);
    }

    #[test]
    fn list_models_is_sorted() {
        let mut r = Registry::new();
        r.register(echo("b"));
        r.register(echo("a"));
        r.register(echo("c"));
        let ids: Vec<_> = r
            .list_models()
            .into_iter()
            .map(|m| m.as_str().to_string())
            .collect();
        assert_eq!(ids, vec!["a", "b", "c"]);
    }

    #[test]
    fn list_roles_is_sorted() {
        let mut r = Registry::new();
        r.register(echo("e"));
        r.bind_role(RoleId::from("b"), ModelId::from("e")).unwrap();
        r.bind_role(RoleId::from("a"), ModelId::from("e")).unwrap();
        let roles: Vec<_> = r
            .list_roles()
            .into_iter()
            .map(|(r, _)| r.as_str().to_string())
            .collect();
        assert_eq!(roles, vec!["a", "b"]);
    }

    #[test]
    fn providers_with_filters_by_capability() {
        let mut r = Registry::new();
        r.register(echo("p1"));
        r.register(echo("p2"));
        let gens = r.providers_with(Capability::Generate);
        assert_eq!(gens.len(), 2);
        let embeds = r.providers_with(Capability::Embed);
        assert!(embeds.is_empty());
    }

    #[test]
    fn debug_renders_summary() {
        let mut r = Registry::new();
        r.register(echo("echo"));
        r.bind_role(RoleId::from("main"), ModelId::from("echo"))
            .unwrap();
        let s = format!("{r:?}");
        assert!(s.contains("Registry"));
        assert!(s.contains("echo"));
        assert!(s.contains("main"));
    }

    #[test]
    fn registry_clone_shares_providers() {
        let mut r = Registry::new();
        r.register(echo("echo"));
        let c = r.clone();
        // Both registries see the same provider via Arc.
        let a = r.get(&ModelId::from("echo")).unwrap();
        let b = c.get(&ModelId::from("echo")).unwrap();
        assert!(Arc::ptr_eq(&a, &b));
    }

    #[test]
    fn resolve_role_after_provider_removed() {
        let mut r = Registry::new();
        r.register(echo("echo"));
        r.bind_role(RoleId::from("main"), ModelId::from("echo"))
            .unwrap();
        // Simulate provider removal by clearing the providers map directly.
        r.providers.clear();
        let err = r.resolve(&RoleId::from("main")).unwrap_err();
        assert_eq!(err.code(), &E3007_MODEL_LOAD_REFUSED);
    }
}
