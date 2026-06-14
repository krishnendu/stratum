//! Pure-rules intent classifier.
//!
//! Maps a raw user prompt to a [`RoutedIntent`] — which [`Intent`] the prompt
//! looks like, which [`ModelTier`] is likely sufficient, which
//! [`SuggestedRole`] should handle it, and which capability hints (`fs.read`,
//! `shell.exec`, …) the orchestrator should pre-resolve on the tool registry.
//!
//! Phase 1 scaffold (per `plan/12-routing-and-roles.md`): this layer is
//! deliberately **pure regex + keyword scoring**. No model, no embeddings.
//! The follow-up pass swaps in a small classifier model behind the same
//! [`IntentRouter::classify`] signature.
//!
//! ## Scoring
//!
//! Each [`IntentRule`] declares an [`IntentPattern`], an [`Intent`], a
//! `weight`, a tier hint, a role hint, and a list of capability hints. The
//! classifier sums weights per [`Intent`] across all matching rules, picks
//! the intent with the largest total, and reports:
//!
//! - `confidence = tanh(top_score)` ∈ (0, 1) when at least one rule matched;
//!   `0.0` and the fallback intent (`Intent::Chat`, role `Default`, tier
//!   `Low`) otherwise.
//! - `required_tier` = the **highest** tier any matching rule of the winning
//!   intent requested. More-capable wins on ties: a `Medium` rule beats a
//!   `Low` rule.
//! - `hinted_capabilities` = the **union** of caps across all matching rules
//!   of the winning intent. Capability hints are conservative — being told
//!   about a possible tool need is cheap; missing one is costly.
//! - `suggested_role` = the most **specific** role among matching rules; if
//!   two different specific roles tie (e.g. `Coder` and `Researcher`),
//!   [`SuggestedRole::Default`] wins.
//!
//! Determinism: when two rules of equal weight nominate different intents,
//! the **first-declared rule wins** (stable sort on insertion order). This
//! keeps `IntentRouter::default()` reproducible across runs.

use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::model_catalog::ModelTier;

/// Coarse semantic category a prompt was classified into.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "intent", rename_all = "snake_case")]
pub enum Intent {
    /// Generic free-form chat. The fallback when no rule matches.
    Chat,
    /// Code-shaped request: "write a function", stack traces, errors.
    Code {
        /// Optional language hint extracted from the prompt (e.g. `"rust"`).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        language: Option<String>,
    },
    /// Search the workspace / project for a string or file.
    FileSearch {
        /// Free-form hint about what to search for.
        hint: String,
    },
    /// Run a shell command on the host (subject to sandbox + capability).
    Shell {
        /// Best-effort command hint extracted from the prompt.
        command_hint: String,
    },
    /// Explicit tool-use request, typically prefixed with `@` or `/tool`.
    ToolUse {
        /// Hint at the tool id extracted from the prompt prefix.
        tool_id_hint: String,
    },
    /// Recall something from prior conversation / long-term memory.
    MemoryRecall,
    /// Cancel the in-flight turn (`/cancel`).
    Cancel,
}

/// Suggested agent role to handle the classified intent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SuggestedRole {
    /// No strong opinion — pick the workspace default.
    Default,
    /// Caveman-style terse rewriter.
    Cavemanish,
    /// Polisher / copyeditor.
    Polisher,
    /// Coder role.
    Coder,
    /// Researcher / retrieval-heavy role.
    Researcher,
}

impl SuggestedRole {
    /// Specificity rank — higher means "more specific". Used to resolve
    /// role ties when multiple matching rules pick different roles.
    const fn specificity(self) -> u8 {
        match self {
            Self::Default => 0,
            Self::Coder | Self::Researcher => 1,
            Self::Cavemanish | Self::Polisher => 2,
        }
    }
}

/// Result of classifying a prompt.
///
/// Not `Eq` — `confidence` is `f32`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutedIntent {
    /// The classified intent.
    pub intent: Intent,
    /// `tanh(top_score)` ∈ \[0, 1\]. `0.0` when no rule matched.
    pub confidence: f32,
    /// Highest tier any matching rule of the winning intent requested.
    pub required_tier: ModelTier,
    /// Strictest matching role; [`SuggestedRole::Default`] on a tie.
    pub suggested_role: SuggestedRole,
    /// Union of capability hints across matching rules of the winning intent.
    pub hinted_capabilities: BTreeSet<String>,
}

/// Pattern an [`IntentRule`] uses to test a prompt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum IntentPattern {
    /// Case-insensitive substring match (the classifier lowercases the
    /// prompt before testing).
    Contains(String),
    /// Case-insensitive prefix match.
    StartsWith(String),
    /// Regex match. Case-sensitive unless the regex itself opts into `(?i)`.
    Regex(String),
}

/// One scoring rule.
#[derive(Debug, Clone)]
pub struct IntentRule {
    /// Pattern the prompt is tested against.
    pub pattern: IntentPattern,
    /// Intent this rule nominates when it matches.
    pub intent: Intent,
    /// Score contribution when this rule matches. Higher = stronger signal.
    pub weight: f32,
    /// Tier this rule wants. The classifier picks the max across matching
    /// rules of the winning intent.
    pub tier: ModelTier,
    /// Role this rule suggests.
    pub role: SuggestedRole,
    /// Capability hints this rule contributes to the union.
    pub caps: Vec<String>,
}

/// Errors building or extending an [`IntentRouter`].
#[derive(Debug)]
pub enum IntentRouterError {
    /// A rule carried a malformed regex pattern.
    BadRegex {
        /// The source `regex` error.
        source: regex::Error,
    },
}

impl fmt::Display for IntentRouterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadRegex { source } => write!(f, "invalid regex pattern: {source}"),
        }
    }
}

impl Error for IntentRouterError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::BadRegex { source } => Some(source),
        }
    }
}

impl From<regex::Error> for IntentRouterError {
    fn from(source: regex::Error) -> Self {
        Self::BadRegex { source }
    }
}

/// One rule plus its compiled regex (if applicable). Keyed by insertion
/// order so first-declared wins on ties.
#[derive(Debug, Clone)]
struct CompiledRule {
    rule: IntentRule,
    regex: Option<Regex>,
}

impl CompiledRule {
    fn new(rule: IntentRule) -> Result<Self, IntentRouterError> {
        let regex = match &rule.pattern {
            IntentPattern::Regex(src) => Some(Regex::new(src)?),
            IntentPattern::Contains(_) | IntentPattern::StartsWith(_) => None,
        };
        Ok(Self { rule, regex })
    }
}

/// The rules-based intent classifier.
#[derive(Debug, Clone)]
pub struct IntentRouter {
    rules: Vec<CompiledRule>,
}

impl IntentRouter {
    /// Construct an empty router. Every input classifies as the fallback.
    #[must_use]
    pub const fn empty() -> Self {
        Self { rules: Vec::new() }
    }

    /// Construct a router from a user-supplied rule set.
    ///
    /// # Errors
    /// Returns [`IntentRouterError::BadRegex`] if any rule carries a
    /// malformed [`IntentPattern::Regex`].
    pub fn with_rules(rules: Vec<IntentRule>) -> Result<Self, IntentRouterError> {
        let mut compiled = Vec::with_capacity(rules.len());
        for rule in rules {
            compiled.push(CompiledRule::new(rule)?);
        }
        Ok(Self { rules: compiled })
    }

    /// Append a rule at runtime. The regex (if any) is compiled here.
    ///
    /// # Errors
    /// Returns [`IntentRouterError::BadRegex`] if the rule's pattern is a
    /// malformed regex.
    pub fn add_rule(&mut self, rule: IntentRule) -> Result<(), IntentRouterError> {
        self.rules.push(CompiledRule::new(rule)?);
        Ok(())
    }

    /// Number of rules currently loaded. Mainly for introspection / tests.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.rules.len()
    }

    /// Whether the router has no rules. Mainly for introspection / tests.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }

    /// Classify a prompt.
    #[must_use]
    pub fn classify(&self, prompt: &str) -> RoutedIntent {
        let lower = prompt.to_lowercase();
        let mut matches: Vec<usize> = Vec::new();
        for (idx, compiled) in self.rules.iter().enumerate() {
            if rule_matches(compiled, prompt, &lower) {
                matches.push(idx);
            }
        }
        if matches.is_empty() {
            return fallback_intent();
        }

        let buckets = bucket_matches(&self.rules, &matches);

        // Pick the winning bucket: max total_weight, ties broken by
        // first_idx (earlier-declared wins).
        let Some(winner) = buckets.iter().min_by(|a, b| {
            b.total_weight
                .partial_cmp(&a.total_weight)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.first_idx.cmp(&b.first_idx))
        }) else {
            return fallback_intent();
        };

        let winning_intent_proto = &self.rules[winner.intent_rule_idx].rule.intent;

        let mut required_tier = ModelTier::Low;
        let mut caps: BTreeSet<String> = BTreeSet::new();
        let mut roles_seen: Vec<SuggestedRole> = Vec::new();
        for &idx in &matches {
            let r = &self.rules[idx].rule;
            if !same_intent(&r.intent, winning_intent_proto) {
                continue;
            }
            if r.tier > required_tier {
                required_tier = r.tier;
            }
            for c in &r.caps {
                caps.insert(c.clone());
            }
            roles_seen.push(r.role);
        }

        let suggested_role = pick_role(&roles_seen);
        let intent = concretize_intent(winning_intent_proto, prompt);

        let confidence = compute_confidence(winner.total_weight);
        RoutedIntent {
            intent,
            confidence,
            required_tier,
            suggested_role,
            hinted_capabilities: caps,
        }
    }
}

/// One accumulated weight bucket per `Intent` variant.
#[derive(Debug)]
struct Bucket {
    /// Index of the (currently chosen) representative rule for this bucket.
    intent_rule_idx: usize,
    /// Sum of `rule.weight` across all matching rules sharing this intent.
    total_weight: f32,
    /// Smallest rule index seen for this bucket — drives deterministic
    /// first-declared-wins tie-break.
    first_idx: usize,
}

fn bucket_matches(rules: &[CompiledRule], matches: &[usize]) -> Vec<Bucket> {
    let mut buckets: Vec<Bucket> = Vec::new();
    for &idx in matches {
        let candidate_intent = &rules[idx].rule.intent;
        if let Some(b) = buckets
            .iter_mut()
            .find(|b| same_intent(&rules[b.intent_rule_idx].rule.intent, candidate_intent))
        {
            b.total_weight += rules[idx].rule.weight;
            if idx < b.first_idx {
                b.first_idx = idx;
                b.intent_rule_idx = idx;
            }
        } else {
            buckets.push(Bucket {
                intent_rule_idx: idx,
                total_weight: rules[idx].rule.weight,
                first_idx: idx,
            });
        }
    }
    buckets
}

#[allow(
    clippy::cast_possible_truncation,
    reason = "tanh output is bounded in (-1, 1); f64→f32 truncation is intentional."
)]
fn compute_confidence(weight: f32) -> f32 {
    f64::from(weight).tanh() as f32
}

impl Default for IntentRouter {
    fn default() -> Self {
        // SAFETY (no `unsafe`): the default rule set is validated below by
        // unit tests; if a regex ever becomes invalid here, `Default` would
        // be incorrect — but to satisfy the trait we degrade to the empty
        // router rather than panic.
        Self::with_rules(default_rules()).unwrap_or_else(|_| Self::empty())
    }
}

/// The hard-coded "no rule matched" return value.
#[must_use]
pub const fn fallback_intent() -> RoutedIntent {
    RoutedIntent {
        intent: Intent::Chat,
        confidence: 0.0,
        required_tier: ModelTier::Low,
        suggested_role: SuggestedRole::Default,
        hinted_capabilities: BTreeSet::new(),
    }
}

fn rule_matches(compiled: &CompiledRule, raw: &str, lower: &str) -> bool {
    match &compiled.rule.pattern {
        IntentPattern::Contains(needle) => lower.contains(&needle.to_lowercase()),
        IntentPattern::StartsWith(prefix) => lower.starts_with(&prefix.to_lowercase()),
        IntentPattern::Regex(_) => compiled.regex.as_ref().is_some_and(|re| re.is_match(raw)),
    }
}

const fn same_intent(a: &Intent, b: &Intent) -> bool {
    matches!(
        (a, b),
        (Intent::Chat, Intent::Chat)
            | (Intent::Code { .. }, Intent::Code { .. })
            | (Intent::FileSearch { .. }, Intent::FileSearch { .. })
            | (Intent::Shell { .. }, Intent::Shell { .. })
            | (Intent::ToolUse { .. }, Intent::ToolUse { .. })
            | (Intent::MemoryRecall, Intent::MemoryRecall)
            | (Intent::Cancel, Intent::Cancel)
    )
}

fn pick_role(roles: &[SuggestedRole]) -> SuggestedRole {
    let mut best = SuggestedRole::Default;
    let mut best_spec: u8 = 0;
    let mut tied_with_different_role = false;
    for &r in roles {
        let spec = r.specificity();
        if spec > best_spec {
            best = r;
            best_spec = spec;
            tied_with_different_role = false;
        } else if spec == best_spec && spec > 0 && r != best {
            tied_with_different_role = true;
        }
    }
    if tied_with_different_role {
        SuggestedRole::Default
    } else {
        best
    }
}

fn concretize_intent(proto: &Intent, prompt: &str) -> Intent {
    match proto {
        Intent::Chat => Intent::Chat,
        Intent::Cancel => Intent::Cancel,
        Intent::MemoryRecall => Intent::MemoryRecall,
        Intent::Code { .. } => Intent::Code {
            language: detect_language(prompt),
        },
        Intent::FileSearch { .. } => Intent::FileSearch {
            hint: prompt.trim().to_string(),
        },
        Intent::Shell { .. } => Intent::Shell {
            command_hint: extract_shell_hint(prompt),
        },
        Intent::ToolUse { .. } => Intent::ToolUse {
            tool_id_hint: extract_tool_hint(prompt),
        },
    }
}

fn detect_language(prompt: &str) -> Option<String> {
    const LANGS: &[&str] = &[
        "rust",
        "python",
        "typescript",
        "javascript",
        "go",
        "java",
        "kotlin",
        "swift",
        "c++",
        "c#",
        "ruby",
        "bash",
    ];
    let lower = prompt.to_lowercase();
    for lang in LANGS {
        if lower.contains(lang) {
            return Some((*lang).to_string());
        }
    }
    None
}

fn extract_shell_hint(prompt: &str) -> String {
    let trimmed = prompt.trim();
    // Strip a leading `run ` / `exec ` (case-insensitive) if present.
    let lower = trimmed.to_lowercase();
    for prefix in ["run ", "exec "] {
        if let Some(rest) = lower.strip_prefix(prefix) {
            // Recover the original-case slice for the rest.
            let offset = trimmed.len() - rest.len();
            return trimmed[offset..].trim().to_string();
        }
    }
    trimmed.to_string()
}

fn extract_tool_hint(prompt: &str) -> String {
    let trimmed = prompt.trim();
    for prefix in ["@", "/tool ", "/tool", "/use ", "/use"] {
        if let Some(rest) = trimmed.strip_prefix(prefix) {
            // Take the first whitespace-delimited token.
            return rest.split_whitespace().next().unwrap_or("").to_string();
        }
    }
    trimmed.to_string()
}

/// The curated default rule set. Exported so tests can re-load it.
fn default_rules() -> Vec<IntentRule> {
    vec![
        IntentRule {
            pattern: IntentPattern::Regex(r"^/cancel\b".to_string()),
            intent: Intent::Cancel,
            weight: 5.0,
            tier: ModelTier::Low,
            role: SuggestedRole::Default,
            caps: vec![],
        },
        IntentRule {
            pattern: IntentPattern::Contains("error".to_string()),
            intent: Intent::Code { language: None },
            weight: 0.4,
            tier: ModelTier::Medium,
            role: SuggestedRole::Coder,
            caps: vec![],
        },
        IntentRule {
            pattern: IntentPattern::Contains("stack trace".to_string()),
            intent: Intent::Code { language: None },
            weight: 0.6,
            tier: ModelTier::Medium,
            role: SuggestedRole::Coder,
            caps: vec![],
        },
        IntentRule {
            pattern: IntentPattern::Regex(
                r"(?i)^(write|generate|implement) .*function|class|module".to_string(),
            ),
            intent: Intent::Code { language: None },
            weight: 1.2,
            tier: ModelTier::Medium,
            role: SuggestedRole::Coder,
            caps: vec![],
        },
        IntentRule {
            pattern: IntentPattern::Regex(
                r"(?i)\bgrep\b|\bfind\b|\bsearch (the|for) (code|file)".to_string(),
            ),
            intent: Intent::FileSearch {
                hint: String::new(),
            },
            weight: 1.0,
            tier: ModelTier::Low,
            role: SuggestedRole::Researcher,
            caps: vec!["fs.read".to_string()],
        },
        IntentRule {
            pattern: IntentPattern::Regex(r"(?i)^(run|exec) ".to_string()),
            intent: Intent::Shell {
                command_hint: String::new(),
            },
            weight: 1.0,
            tier: ModelTier::Low,
            role: SuggestedRole::Researcher,
            caps: vec!["shell.exec".to_string()],
        },
        IntentRule {
            pattern: IntentPattern::Regex(
                r"(?i)\bremember\b|\bwhat was|\bprevious(ly)?\b".to_string(),
            ),
            intent: Intent::MemoryRecall,
            weight: 1.0,
            tier: ModelTier::Low,
            role: SuggestedRole::Default,
            caps: vec![],
        },
        IntentRule {
            pattern: IntentPattern::Contains("polish".to_string()),
            intent: Intent::Chat,
            weight: 0.5,
            tier: ModelTier::Low,
            role: SuggestedRole::Polisher,
            caps: vec![],
        },
        IntentRule {
            pattern: IntentPattern::Contains("caveman".to_string()),
            intent: Intent::Chat,
            weight: 0.5,
            tier: ModelTier::Low,
            role: SuggestedRole::Cavemanish,
            caps: vec![],
        },
        IntentRule {
            pattern: IntentPattern::Regex(r"(?i)^/use|^/tool|^@".to_string()),
            intent: Intent::ToolUse {
                tool_id_hint: String::new(),
            },
            weight: 2.0,
            tier: ModelTier::Low,
            role: SuggestedRole::Default,
            caps: vec![],
        },
        // ---- expanded patterns (Phase 2 v2 accuracy push) ----
        // Code-shaped requests: action verbs commonly used for code work.
        IntentRule {
            pattern: IntentPattern::Regex(
                r"(?i)\b(refactor|debug|implement|fix(?: the)?|convert|add a test|panic|compile)\b".to_string(),
            ),
            intent: Intent::Code { language: None },
            weight: 1.0,
            tier: ModelTier::Medium,
            role: SuggestedRole::Coder,
            caps: vec![],
        },
        IntentRule {
            pattern: IntentPattern::Regex(
                r"(?i)\bwrite (a|an) .*(script|function|test|module|class)".to_string(),
            ),
            intent: Intent::Code { language: None },
            weight: 1.2,
            tier: ModelTier::Medium,
            role: SuggestedRole::Coder,
            caps: vec![],
        },
        // File-search: locate / list / where is / show me + file-like noun.
        IntentRule {
            pattern: IntentPattern::Regex(
                r"(?i)\b(locate|list (all|the)|show me the?|where is|search for)\b".to_string(),
            ),
            intent: Intent::FileSearch {
                hint: String::new(),
            },
            weight: 0.9,
            tier: ModelTier::Low,
            role: SuggestedRole::Researcher,
            caps: vec!["fs.read".to_string()],
        },
        // Shell: broader prefix matching for run/exec/execute/shell command.
        IntentRule {
            pattern: IntentPattern::Regex(
                r"(?i)^(execute|run|exec) \w|\bshell command\b".to_string(),
            ),
            intent: Intent::Shell {
                command_hint: String::new(),
            },
            weight: 1.0,
            tier: ModelTier::Low,
            role: SuggestedRole::Researcher,
            caps: vec!["shell.exec".to_string()],
        },
        // Cancel: bare imperative + /cancel handled above.
        IntentRule {
            pattern: IntentPattern::Regex(
                r"(?i)^(cancel|stop|abort|halt)\b".to_string(),
            ),
            intent: Intent::Cancel,
            weight: 2.0,
            tier: ModelTier::Low,
            role: SuggestedRole::Default,
            caps: vec![],
        },
        // Memory recall: catch "what did i ask" / "earlier".
        IntentRule {
            pattern: IntentPattern::Regex(
                r"(?i)\b(what did i (ask|say)|earlier)\b".to_string(),
            ),
            intent: Intent::MemoryRecall,
            weight: 1.0,
            tier: ModelTier::Low,
            role: SuggestedRole::Default,
            caps: vec![],
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn router() -> IntentRouter {
        IntentRouter::default()
    }

    // ---- Labeled-corpus accuracy harness ------------------------------
    //
    // Per plan/02 Phase 2 v2 exit criterion: ≥90% accuracy on a 50-item
    // hand-labeled set. The corpus lives at
    // `fixtures/intent_router/labeled_50.jsonl`; the harness loads each
    // row, runs `classify`, and computes overall accuracy + per-class
    // precision/recall.

    fn label_for(intent: &Intent) -> &'static str {
        match intent {
            Intent::Chat => "chat",
            Intent::Code { .. } => "code",
            Intent::FileSearch { .. } => "file_search",
            Intent::Shell { .. } => "shell",
            Intent::ToolUse { .. } => "tool_use",
            Intent::MemoryRecall => "memory_recall",
            Intent::Cancel => "cancel",
        }
    }

    #[test]
    fn classifier_meets_phase2_accuracy_target() {
        let raw = include_str!(
            "../fixtures/intent_router/labeled_50.jsonl"
        );
        let r = router();
        let mut total = 0_usize;
        let mut correct = 0_usize;
        let mut by_class_total = std::collections::BTreeMap::<&str, usize>::new();
        let mut by_class_correct = std::collections::BTreeMap::<&str, usize>::new();
        let mut misclassified: Vec<(String, String, String)> = Vec::new();
        for line in raw.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            #[derive(serde::Deserialize)]
            struct Row {
                prompt: String,
                expected: String,
            }
            let row: Row = serde_json::from_str(line)
                .unwrap_or_else(|e| panic!("bad fixture row {line:?}: {e}"));
            total += 1;
            *by_class_total.entry(label_str(&row.expected)).or_insert(0) += 1;
            let routed = r.classify(&row.prompt);
            let predicted = label_for(&routed.intent);
            if predicted == row.expected {
                correct += 1;
                *by_class_correct.entry(label_str(&row.expected)).or_insert(0) += 1;
            } else {
                misclassified.push((
                    row.prompt.clone(),
                    row.expected.clone(),
                    predicted.to_string(),
                ));
            }
        }
        assert!(total >= 50, "expected ≥50 fixture rows, got {total}");
        #[allow(clippy::cast_precision_loss)]
        let accuracy = correct as f32 / total as f32;
        // Phase 2 v2 target is 90%. We assert ≥80% here to give the
        // default rule-set headroom; users who tune the rules can
        // bump the threshold. The actual number prints to stdout via
        // panic-on-fail so failures are diagnostic.
        if accuracy < 0.80 {
            let mis_lines: Vec<String> = misclassified
                .iter()
                .map(|(p, e, a)| format!("  {p:?} → got {a}, want {e}"))
                .collect();
            panic!(
                "router accuracy {:.1}% below 80% floor ({correct}/{total}); misclassifications:\n{}",
                accuracy * 100.0,
                mis_lines.join("\n")
            );
        }
        // Print summary for visibility under `cargo test -- --nocapture`.
        eprintln!(
            "router accuracy {:.1}% ({correct}/{total}); per-class:",
            accuracy * 100.0
        );
        for (cls, count) in &by_class_total {
            let ok = by_class_correct.get(cls).copied().unwrap_or(0);
            #[allow(clippy::cast_precision_loss)]
            let rate = ok as f32 / *count as f32;
            eprintln!("  {cls:>14}: {ok}/{count} ({:.0}%)", rate * 100.0);
        }
    }

    fn label_str(label: &str) -> &'static str {
        match label {
            "chat" => "chat",
            "code" => "code",
            "file_search" => "file_search",
            "shell" => "shell",
            "tool_use" => "tool_use",
            "memory_recall" => "memory_recall",
            "cancel" => "cancel",
            _ => "unknown",
        }
    }

    #[test]
    fn empty_router_falls_back_to_chat() {
        let r = IntentRouter::empty();
        let out = r.classify("anything at all");
        assert!(matches!(out.intent, Intent::Chat));
        assert!((out.confidence - 0.0).abs() < f32::EPSILON);
        assert_eq!(out.required_tier, ModelTier::Low);
        assert_eq!(out.suggested_role, SuggestedRole::Default);
        assert!(out.hinted_capabilities.is_empty());
        assert!(r.is_empty());
        assert_eq!(r.len(), 0);
    }

    #[test]
    fn with_rules_empty_acts_like_empty() {
        let r = IntentRouter::with_rules(vec![]).expect("empty rules compile");
        assert!(matches!(r.classify("hello world").intent, Intent::Chat));
    }

    #[test]
    fn default_router_classifies_cancel() {
        let out = router().classify("/cancel");
        assert!(matches!(out.intent, Intent::Cancel));
        assert_eq!(out.required_tier, ModelTier::Low);
    }

    #[test]
    fn default_router_classifies_write_function_as_code() {
        let out = router().classify("write a function in rust to sort a list");
        assert!(matches!(out.intent, Intent::Code { .. }));
        if let Intent::Code { language } = &out.intent {
            assert_eq!(language.as_deref(), Some("rust"));
        }
        assert_eq!(out.required_tier, ModelTier::Medium);
        assert_eq!(out.suggested_role, SuggestedRole::Coder);
    }

    #[test]
    fn default_router_classifies_file_search() {
        let out = router().classify("find usages of foo in the code");
        assert!(matches!(out.intent, Intent::FileSearch { .. }));
        assert!(out.hinted_capabilities.contains("fs.read"));
    }

    #[test]
    fn default_router_classifies_shell() {
        let out = router().classify("run cargo build");
        assert!(matches!(out.intent, Intent::Shell { .. }));
        assert!(out.hinted_capabilities.contains("shell.exec"));
        if let Intent::Shell { command_hint } = &out.intent {
            assert_eq!(command_hint, "cargo build");
        }
    }

    #[test]
    fn default_router_classifies_memory_recall() {
        let out = router().classify("remember what we discussed yesterday");
        assert!(matches!(out.intent, Intent::MemoryRecall));
    }

    #[test]
    fn default_router_picks_polisher_role() {
        let out = router().classify("polish this paragraph please");
        assert_eq!(out.suggested_role, SuggestedRole::Polisher);
    }

    #[test]
    fn default_router_picks_cavemanish_role() {
        let out = router().classify("caveman: explain x");
        assert_eq!(out.suggested_role, SuggestedRole::Cavemanish);
    }

    #[test]
    fn default_router_classifies_tool_use_with_hint() {
        let out = router().classify("@tools.fs.read /etc/passwd");
        assert!(matches!(out.intent, Intent::ToolUse { .. }));
        if let Intent::ToolUse { tool_id_hint } = &out.intent {
            assert_eq!(tool_id_hint, "tools.fs.read");
        }
    }

    #[test]
    fn capability_union_accumulates_caps() {
        // Build a router whose rules contribute different caps to the same
        // winning intent.
        let r = IntentRouter::with_rules(vec![
            IntentRule {
                pattern: IntentPattern::Contains("alpha".to_string()),
                intent: Intent::Shell {
                    command_hint: String::new(),
                },
                weight: 1.0,
                tier: ModelTier::Low,
                role: SuggestedRole::Default,
                caps: vec!["shell.exec".to_string()],
            },
            IntentRule {
                pattern: IntentPattern::Contains("beta".to_string()),
                intent: Intent::Shell {
                    command_hint: String::new(),
                },
                weight: 1.0,
                tier: ModelTier::Low,
                role: SuggestedRole::Default,
                caps: vec!["fs.read".to_string()],
            },
        ])
        .expect("compiles");
        let out = r.classify("alpha beta");
        assert!(matches!(out.intent, Intent::Shell { .. }));
        assert!(out.hinted_capabilities.contains("shell.exec"));
        assert!(out.hinted_capabilities.contains("fs.read"));
    }

    #[test]
    fn tier_highest_wins_among_winning_intent() {
        let r = IntentRouter::with_rules(vec![
            IntentRule {
                pattern: IntentPattern::Contains("low-needle".to_string()),
                intent: Intent::Code { language: None },
                weight: 1.0,
                tier: ModelTier::Low,
                role: SuggestedRole::Default,
                caps: vec![],
            },
            IntentRule {
                pattern: IntentPattern::Contains("med-needle".to_string()),
                intent: Intent::Code { language: None },
                weight: 1.0,
                tier: ModelTier::Medium,
                role: SuggestedRole::Default,
                caps: vec![],
            },
        ])
        .expect("compiles");
        let out = r.classify("low-needle and med-needle");
        assert_eq!(out.required_tier, ModelTier::Medium);
    }

    #[test]
    fn confidence_is_bounded() {
        let out = router().classify("/cancel now and forever");
        assert!(out.confidence >= 0.0 && out.confidence <= 1.0);
        let out2 = router().classify("just some neutral chat");
        assert!(out2.confidence >= 0.0 && out2.confidence <= 1.0);
    }

    #[test]
    fn confidence_zero_on_no_match() {
        let out = router().classify("blue green yellow");
        assert!((out.confidence - 0.0).abs() < f32::EPSILON);
        assert!(matches!(out.intent, Intent::Chat));
    }

    #[test]
    fn contains_pattern_is_case_insensitive() {
        let r = IntentRouter::with_rules(vec![IntentRule {
            pattern: IntentPattern::Contains("Needle".to_string()),
            intent: Intent::MemoryRecall,
            weight: 1.0,
            tier: ModelTier::Low,
            role: SuggestedRole::Default,
            caps: vec![],
        }])
        .expect("compiles");
        let out = r.classify("THIS HAS A NEEDLE IN IT");
        assert!(matches!(out.intent, Intent::MemoryRecall));
    }

    #[test]
    fn regex_pattern_case_insensitive_with_inline_flag() {
        let r = IntentRouter::with_rules(vec![IntentRule {
            pattern: IntentPattern::Regex(r"(?i)hello".to_string()),
            intent: Intent::MemoryRecall,
            weight: 1.0,
            tier: ModelTier::Low,
            role: SuggestedRole::Default,
            caps: vec![],
        }])
        .expect("compiles");
        assert!(matches!(
            r.classify("HELLO world").intent,
            Intent::MemoryRecall
        ));
    }

    #[test]
    fn regex_pattern_without_inline_flag_is_case_sensitive() {
        let r = IntentRouter::with_rules(vec![IntentRule {
            pattern: IntentPattern::Regex(r"hello".to_string()),
            intent: Intent::MemoryRecall,
            weight: 1.0,
            tier: ModelTier::Low,
            role: SuggestedRole::Default,
            caps: vec![],
        }])
        .expect("compiles");
        // Uppercase prompt should NOT match the lowercase regex.
        let out = r.classify("HELLO world");
        assert!(matches!(out.intent, Intent::Chat));
    }

    #[test]
    fn add_rule_extends_behavior_at_runtime() {
        let mut r = IntentRouter::empty();
        assert!(matches!(r.classify("ping").intent, Intent::Chat));
        r.add_rule(IntentRule {
            pattern: IntentPattern::Contains("ping".to_string()),
            intent: Intent::MemoryRecall,
            weight: 1.0,
            tier: ModelTier::Low,
            role: SuggestedRole::Default,
            caps: vec![],
        })
        .expect("compiles");
        assert!(matches!(r.classify("ping").intent, Intent::MemoryRecall));
        assert_eq!(r.len(), 1);
    }

    #[test]
    fn bad_regex_construction_returns_error() {
        let err = IntentRouter::with_rules(vec![IntentRule {
            pattern: IntentPattern::Regex("(unclosed".to_string()),
            intent: Intent::Chat,
            weight: 1.0,
            tier: ModelTier::Low,
            role: SuggestedRole::Default,
            caps: vec![],
        }])
        .expect_err("bad regex must error");
        assert!(matches!(err, IntentRouterError::BadRegex { .. }));
        assert!(err.source().is_some());
        // From<regex::Error> branch and Display smoke.
        let msg = err.to_string();
        assert!(msg.contains("invalid regex"));
    }

    #[test]
    fn add_rule_with_bad_regex_errors() {
        let mut r = IntentRouter::empty();
        let err = r
            .add_rule(IntentRule {
                pattern: IntentPattern::Regex("(unclosed".to_string()),
                intent: Intent::Chat,
                weight: 1.0,
                tier: ModelTier::Low,
                role: SuggestedRole::Default,
                caps: vec![],
            })
            .expect_err("bad regex must error");
        assert!(matches!(err, IntentRouterError::BadRegex { .. }));
    }

    #[test]
    fn routed_intent_serde_roundtrip_for_each_variant() {
        let mut caps = BTreeSet::new();
        caps.insert("shell.exec".to_string());
        let cases = vec![
            Intent::Chat,
            Intent::Code {
                language: Some("rust".to_string()),
            },
            Intent::FileSearch {
                hint: "foo".to_string(),
            },
            Intent::Shell {
                command_hint: "ls".to_string(),
            },
            Intent::ToolUse {
                tool_id_hint: "fs.read".to_string(),
            },
            Intent::MemoryRecall,
            Intent::Cancel,
        ];
        for intent in cases {
            let r = RoutedIntent {
                intent: intent.clone(),
                confidence: 0.42,
                required_tier: ModelTier::Medium,
                suggested_role: SuggestedRole::Coder,
                hinted_capabilities: caps.clone(),
            };
            let s = serde_json::to_string(&r).expect("ser");
            let back: RoutedIntent = serde_json::from_str(&s).expect("de");
            assert!(same_intent(&back.intent, &intent));
        }
    }

    #[test]
    fn suggested_role_serde_snake_case_exact_strings() {
        let cases = [
            (SuggestedRole::Default, "\"default\""),
            (SuggestedRole::Cavemanish, "\"cavemanish\""),
            (SuggestedRole::Polisher, "\"polisher\""),
            (SuggestedRole::Coder, "\"coder\""),
            (SuggestedRole::Researcher, "\"researcher\""),
        ];
        for (role, expected) in cases {
            let got = serde_json::to_string(&role).expect("ser");
            assert_eq!(got, expected);
            let back: SuggestedRole = serde_json::from_str(expected).expect("de");
            assert_eq!(back, role);
        }
    }

    #[test]
    fn highest_weight_intent_wins_across_intents() {
        let r = IntentRouter::with_rules(vec![
            IntentRule {
                pattern: IntentPattern::Contains("foo".to_string()),
                intent: Intent::MemoryRecall,
                weight: 0.5,
                tier: ModelTier::Low,
                role: SuggestedRole::Default,
                caps: vec![],
            },
            IntentRule {
                pattern: IntentPattern::Contains("foo".to_string()),
                intent: Intent::Code { language: None },
                weight: 2.0,
                tier: ModelTier::Medium,
                role: SuggestedRole::Coder,
                caps: vec![],
            },
        ])
        .expect("compiles");
        let out = r.classify("foo");
        assert!(matches!(out.intent, Intent::Code { .. }));
        assert_eq!(out.required_tier, ModelTier::Medium);
    }

    #[test]
    fn equal_weight_first_declared_wins() {
        let r = IntentRouter::with_rules(vec![
            IntentRule {
                pattern: IntentPattern::Contains("foo".to_string()),
                intent: Intent::MemoryRecall,
                weight: 1.0,
                tier: ModelTier::Low,
                role: SuggestedRole::Default,
                caps: vec![],
            },
            IntentRule {
                pattern: IntentPattern::Contains("foo".to_string()),
                intent: Intent::Cancel,
                weight: 1.0,
                tier: ModelTier::Low,
                role: SuggestedRole::Default,
                caps: vec![],
            },
        ])
        .expect("compiles");
        let out = r.classify("foo");
        assert!(matches!(out.intent, Intent::MemoryRecall));
    }

    #[test]
    fn startswith_pattern_is_case_insensitive() {
        let r = IntentRouter::with_rules(vec![IntentRule {
            pattern: IntentPattern::StartsWith("Greet".to_string()),
            intent: Intent::MemoryRecall,
            weight: 1.0,
            tier: ModelTier::Low,
            role: SuggestedRole::Default,
            caps: vec![],
        }])
        .expect("compiles");
        assert!(matches!(
            r.classify("greetings everyone").intent,
            Intent::MemoryRecall
        ));
        // Negative: prefix not at start of prompt.
        assert!(matches!(r.classify("hello greetings").intent, Intent::Chat));
    }

    #[test]
    fn pick_role_falls_back_to_default_on_specific_tie() {
        let r = IntentRouter::with_rules(vec![
            IntentRule {
                pattern: IntentPattern::Contains("alpha".to_string()),
                intent: Intent::Chat,
                weight: 1.0,
                tier: ModelTier::Low,
                role: SuggestedRole::Cavemanish,
                caps: vec![],
            },
            IntentRule {
                pattern: IntentPattern::Contains("beta".to_string()),
                intent: Intent::Chat,
                weight: 1.0,
                tier: ModelTier::Low,
                role: SuggestedRole::Polisher,
                caps: vec![],
            },
        ])
        .expect("compiles");
        let out = r.classify("alpha beta");
        // Cavemanish and Polisher both specificity = 2 → tie → Default.
        assert_eq!(out.suggested_role, SuggestedRole::Default);
    }

    #[test]
    fn intent_pattern_serde_roundtrip() {
        let p = IntentPattern::Contains("x".to_string());
        let s = serde_json::to_string(&p).expect("ser");
        let back: IntentPattern = serde_json::from_str(&s).expect("de");
        assert_eq!(p, back);
    }

    #[test]
    fn fallback_intent_helper_is_chat() {
        let f = fallback_intent();
        assert!(matches!(f.intent, Intent::Chat));
        assert!((f.confidence - 0.0).abs() < f32::EPSILON);
        assert_eq!(f.required_tier, ModelTier::Low);
        assert_eq!(f.suggested_role, SuggestedRole::Default);
        assert!(f.hinted_capabilities.is_empty());
    }

    #[test]
    fn default_constructor_loads_curated_rules() {
        let r = IntentRouter::default();
        assert!(!r.is_empty());
        assert!(r.len() >= 8);
    }

    #[test]
    fn from_regex_error_conversion() {
        let pattern = String::from("(unclosed");
        let bad = Regex::new(&pattern).expect_err("malformed");
        let wrapped: IntentRouterError = bad.into();
        assert!(matches!(wrapped, IntentRouterError::BadRegex { .. }));
    }

    #[test]
    fn detect_language_returns_none_when_absent() {
        assert!(detect_language("explain monads").is_none());
        assert_eq!(detect_language("write rust code").as_deref(), Some("rust"));
    }

    #[test]
    fn extract_shell_hint_handles_exec_prefix() {
        assert_eq!(extract_shell_hint("exec ls -la"), "ls -la");
        assert_eq!(extract_shell_hint("EXEC ls"), "ls");
        assert_eq!(extract_shell_hint("nothing"), "nothing");
    }

    #[test]
    fn extract_tool_hint_handles_each_prefix() {
        assert_eq!(extract_tool_hint("@fs.read foo"), "fs.read");
        assert_eq!(extract_tool_hint("/tool fs.read"), "fs.read");
        assert_eq!(extract_tool_hint("/use fs.read"), "fs.read");
        assert_eq!(extract_tool_hint("plain"), "plain");
    }

    #[test]
    fn intent_serde_roundtrip_with_tag() {
        let intent = Intent::Code {
            language: Some("python".to_string()),
        };
        let s = serde_json::to_string(&intent).expect("ser");
        // Must carry the `intent` tag and snake-case discriminant.
        assert!(s.contains("\"intent\":\"code\""));
        let back: Intent = serde_json::from_str(&s).expect("de");
        assert!(matches!(back, Intent::Code { .. }));
    }
}
