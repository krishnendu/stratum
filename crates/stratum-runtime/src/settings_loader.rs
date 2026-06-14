//! Four-tier settings loader (managed → user → project → local).
//!
//! Per `plan/30 §10`. Walks the four conventional paths, parses each
//! tier's `settings.json` (or `.toml`), merges into a single
//! [`MergedSettings`] with explicit precedence: later wins on key
//! conflicts, except for `permissions.deny` which composes by union
//! (any deny across any tier wins).
//!
//! Hot-reload (Phase 5 v2) plugs in by re-running [`load`] when the
//! `notify` watcher fires `SettingsReload`.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// One tier in the precedence stack.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SettingsTier {
    /// Org policy (managed by admin).
    Managed,
    /// User defaults (per-user config dir).
    User,
    /// Project-level (committed).
    Project,
    /// Per-checkout (gitignored).
    Local,
}

impl SettingsTier {
    fn label(self) -> &'static str {
        match self {
            Self::Managed => "managed",
            Self::User => "user",
            Self::Project => "project",
            Self::Local => "local",
        }
    }
}

/// Permissions block — three rule lists per `plan/30 §10.1`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PermissionsBlock {
    /// Tools that may dispatch without a permission prompt.
    #[serde(default)]
    pub allow: Vec<String>,
    /// Tools that may NEVER dispatch — deny is the highest-precedence tier.
    #[serde(default)]
    pub deny: Vec<String>,
    /// Tools that always trigger the permission modal.
    #[serde(default)]
    pub ask: Vec<String>,
}

/// One tier's parsed settings document.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SettingsDoc {
    /// Permission rules (plan/30 §10.1).
    #[serde(default)]
    pub permissions: PermissionsBlock,
    /// Environment variables merged into tool subprocesses.
    #[serde(default)]
    pub env: std::collections::BTreeMap<String, String>,
    /// Default theme name (`/theme`).
    #[serde(default)]
    pub theme: Option<String>,
    /// `outputStyle` per plan/30 §12.
    #[serde(default)]
    pub output_style: Option<String>,
    /// Custom statusline shell command (plan/30 §11).
    #[serde(default)]
    pub status_line: Option<String>,
    /// Vim mode in the input box (plan/38 Phase C).
    #[serde(default)]
    pub editor_mode: Option<String>,
}

/// Merged settings across all four tiers — what the runtime
/// actually consults.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MergedSettings {
    /// Combined permissions: allow + ask are union'd, deny is union'd.
    pub permissions: PermissionsBlock,
    /// env: later tier wins on key conflict.
    pub env: std::collections::BTreeMap<String, String>,
    /// theme: highest-precedence (innermost) wins.
    pub theme: Option<String>,
    /// outputStyle: highest-precedence wins.
    pub output_style: Option<String>,
    /// statusLine: highest-precedence wins.
    pub status_line: Option<String>,
    /// editorMode: highest-precedence wins.
    pub editor_mode: Option<String>,
    /// Per-tier source paths for the `/config` palette / debug UI.
    pub sources: Vec<(SettingsTier, PathBuf)>,
}

/// Inputs for the loader. Pass `None` for any tier to skip it.
#[derive(Debug, Clone, Default)]
pub struct LoaderInputs {
    /// Managed-tier file path. Typically
    /// `/Library/Application Support/Stratum/settings.json` on macOS,
    /// `/etc/stratum/settings.json` on Linux.
    pub managed: Option<PathBuf>,
    /// User-tier file path. Typically `<config>/stratum/settings.json`.
    pub user: Option<PathBuf>,
    /// Project root. Loader looks for `.stratum/settings.json`,
    /// `stratum.toml`, AND `.stratum/settings.local.json` inside it.
    pub project_root: Option<PathBuf>,
}

/// Load + merge all four tiers. Missing files are silently skipped.
/// JSON and TOML are interchangeable — extension picks parser.
#[must_use]
pub fn load(inputs: &LoaderInputs) -> MergedSettings {
    let mut merged = MergedSettings::default();
    if let Some(p) = inputs.managed.as_ref() {
        try_apply(&mut merged, SettingsTier::Managed, p);
    }
    if let Some(p) = inputs.user.as_ref() {
        try_apply(&mut merged, SettingsTier::User, p);
    }
    if let Some(root) = inputs.project_root.as_ref() {
        // Project tier: prefer `.stratum/settings.json`, fall back to
        // `stratum.toml`.
        let candidates = [
            root.join(".stratum").join("settings.json"),
            root.join("stratum.toml"),
        ];
        for c in &candidates {
            if try_apply(&mut merged, SettingsTier::Project, c) {
                break;
            }
        }
        // Local tier (always after project so it can override).
        let local_candidates = [
            root.join(".stratum").join("settings.local.json"),
            root.join(".stratum").join("local.toml"),
        ];
        for c in &local_candidates {
            if try_apply(&mut merged, SettingsTier::Local, c) {
                break;
            }
        }
    }
    merged
}

fn try_apply(merged: &mut MergedSettings, tier: SettingsTier, path: &Path) -> bool {
    let Ok(raw) = std::fs::read_to_string(path) else {
        return false;
    };
    let doc = parse_doc(path, &raw);
    let Some(doc) = doc else { return false };
    merge_into(merged, tier, path.to_path_buf(), doc);
    true
}

fn parse_doc(path: &Path, raw: &str) -> Option<SettingsDoc> {
    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
    if ext == "toml" {
        toml_edit::de::from_str::<SettingsDoc>(raw).ok()
    } else {
        // Default to JSON for `.json`, `.json5`, no-extension, anything else.
        serde_json::from_str::<SettingsDoc>(raw).ok()
    }
}

fn merge_into(
    merged: &mut MergedSettings,
    tier: SettingsTier,
    source: PathBuf,
    doc: SettingsDoc,
) {
    // Permissions: union allow + ask; union deny (deny is highest authority).
    for r in &doc.permissions.allow {
        if !merged.permissions.allow.contains(r) {
            merged.permissions.allow.push(r.clone());
        }
    }
    for r in &doc.permissions.deny {
        if !merged.permissions.deny.contains(r) {
            merged.permissions.deny.push(r.clone());
        }
    }
    for r in &doc.permissions.ask {
        if !merged.permissions.ask.contains(r) {
            merged.permissions.ask.push(r.clone());
        }
    }
    // env: later tier overrides earlier tier on key collision.
    for (k, v) in doc.env {
        merged.env.insert(k, v);
    }
    // theme / outputStyle / statusLine / editorMode: highest-precedence wins.
    if doc.theme.is_some() {
        merged.theme = doc.theme;
    }
    if doc.output_style.is_some() {
        merged.output_style = doc.output_style;
    }
    if doc.status_line.is_some() {
        merged.status_line = doc.status_line;
    }
    if doc.editor_mode.is_some() {
        merged.editor_mode = doc.editor_mode;
    }
    merged.sources.push((tier, source));
}

impl MergedSettings {
    /// Sources contributing to this merged result, in tier order.
    #[must_use]
    pub fn source_labels(&self) -> Vec<String> {
        self.sources
            .iter()
            .map(|(t, p)| format!("{}={}", t.label(), p.display()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn write_json(p: &Path, body: &str) {
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, body).unwrap();
    }

    #[test]
    fn empty_inputs_produce_empty_merged() {
        let m = load(&LoaderInputs::default());
        assert!(m.permissions.allow.is_empty());
        assert!(m.theme.is_none());
        assert!(m.sources.is_empty());
    }

    #[test]
    fn user_tier_loaded() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("settings.json");
        write_json(
            &path,
            r#"{
                "permissions": { "allow": ["fs.read"], "deny": [], "ask": [] },
                "theme": "vivid"
            }"#,
        );
        let m = load(&LoaderInputs {
            user: Some(path),
            ..Default::default()
        });
        assert!(m.permissions.allow.contains(&"fs.read".to_string()));
        assert_eq!(m.theme.as_deref(), Some("vivid"));
        assert_eq!(m.sources.len(), 1);
    }

    #[test]
    fn project_overrides_user_on_theme() {
        let tmp = TempDir::new().unwrap();
        let user_path = tmp.path().join("user.json");
        write_json(&user_path, r#"{ "theme": "vivid" }"#);
        let project_root = tmp.path().join("project");
        let project_path = project_root.join(".stratum").join("settings.json");
        write_json(&project_path, r#"{ "theme": "ocean" }"#);
        let m = load(&LoaderInputs {
            user: Some(user_path),
            project_root: Some(project_root),
            ..Default::default()
        });
        assert_eq!(m.theme.as_deref(), Some("ocean"));
    }

    #[test]
    fn deny_unions_across_tiers() {
        let tmp = TempDir::new().unwrap();
        let user_path = tmp.path().join("user.json");
        write_json(
            &user_path,
            r#"{ "permissions": { "deny": ["shell.exec(rm *)"] } }"#,
        );
        let project_root = tmp.path().join("project");
        let project_path = project_root.join(".stratum").join("settings.json");
        write_json(
            &project_path,
            r#"{ "permissions": { "deny": ["shell.exec(curl *)"] } }"#,
        );
        let m = load(&LoaderInputs {
            user: Some(user_path),
            project_root: Some(project_root),
            ..Default::default()
        });
        assert_eq!(m.permissions.deny.len(), 2);
        assert!(m.permissions.deny.iter().any(|r| r.contains("rm")));
        assert!(m.permissions.deny.iter().any(|r| r.contains("curl")));
    }

    #[test]
    fn env_overrides_per_key_by_tier() {
        let tmp = TempDir::new().unwrap();
        let user_path = tmp.path().join("user.json");
        write_json(
            &user_path,
            r#"{ "env": { "FOO": "user", "BAR": "user" } }"#,
        );
        let project_root = tmp.path().join("project");
        let project_path = project_root.join(".stratum").join("settings.json");
        write_json(
            &project_path,
            r#"{ "env": { "FOO": "project" } }"#,
        );
        let m = load(&LoaderInputs {
            user: Some(user_path),
            project_root: Some(project_root),
            ..Default::default()
        });
        assert_eq!(m.env.get("FOO").unwrap(), "project");
        assert_eq!(m.env.get("BAR").unwrap(), "user");
    }

    #[test]
    fn local_tier_overrides_project() {
        let tmp = TempDir::new().unwrap();
        let project_root = tmp.path().join("project");
        let project_path = project_root.join(".stratum").join("settings.json");
        write_json(&project_path, r#"{ "theme": "ocean" }"#);
        let local_path = project_root.join(".stratum").join("settings.local.json");
        write_json(&local_path, r#"{ "theme": "vivid" }"#);
        let m = load(&LoaderInputs {
            project_root: Some(project_root),
            ..Default::default()
        });
        assert_eq!(m.theme.as_deref(), Some("vivid"));
    }

    #[test]
    fn malformed_json_silently_skipped() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("user.json");
        std::fs::write(&path, "this is not json").unwrap();
        let m = load(&LoaderInputs {
            user: Some(path),
            ..Default::default()
        });
        // No panic; merged stays empty.
        assert!(m.sources.is_empty());
    }

    #[test]
    fn toml_form_at_project_root_works() {
        let tmp = TempDir::new().unwrap();
        let project_root = tmp.path().join("project");
        let toml_path = project_root.join("stratum.toml");
        std::fs::create_dir_all(&project_root).unwrap();
        std::fs::write(
            &toml_path,
            "theme = \"vivid\"\n[permissions]\nallow = [\"fs.read\"]\n",
        )
        .unwrap();
        let m = load(&LoaderInputs {
            project_root: Some(project_root),
            ..Default::default()
        });
        assert_eq!(m.theme.as_deref(), Some("vivid"));
        assert!(m.permissions.allow.contains(&"fs.read".to_string()));
    }

    #[test]
    fn source_labels_list_paths() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("user.json");
        write_json(&path, r#"{}"#);
        let m = load(&LoaderInputs {
            user: Some(path),
            ..Default::default()
        });
        let labels = m.source_labels();
        assert_eq!(labels.len(), 1);
        assert!(labels[0].starts_with("user="));
    }
}
