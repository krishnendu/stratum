//! Fluent-style i18n catalog + lookup (Phase 1 scaffold).
//!
//! See the "i18n via Fluent" item in `plan/35-governance.md` (project
//! governance memo). This module is the in-process catalog + interpolation
//! that the eventual real Fluent loader will plug into; no `.ftl` files ship
//! today.
//!
//! ## Scope of this pass
//!
//! - Strongly-typed [`LocaleId`] and [`MessageId`] newtypes with validation.
//! - A [`Message`] = id + body template, where placeholders use a tiny
//!   Fluent-ish `{ $name }` syntax (no selectors, no terms, no functions).
//! - A [`MessageCatalog`] per locale, and an [`I18nBundle`] that owns one or
//!   more catalogs and a fallback locale.
//! - [`I18nBundle::lookup`] tries the requested locale, then the fallback,
//!   and interpolates `args` of type [`FluentArg`].
//! - [`parse_simple_ftl`] — extremely small subset parser, just enough to
//!   round-trip a catalog the Phase-2 Fluent loader will eventually replace.
//!
//! Out of scope (deferred to the real loader):
//!
//! - Multi-line message bodies.
//! - Fluent selectors, terms, attributes, functions.
//! - Plural rules / CLDR formatting.
//!
//! No new `STRAT-E####` codes are introduced; all failure modes are returned
//! as [`I18nError`] variants and are not user-facing strings yet.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::{self, Debug, Display, Formatter};
use std::str::FromStr;

use regex::Regex;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// LocaleId
// ---------------------------------------------------------------------------

/// BCP-47-ish locale tag.
///
/// Accepts a small subset matching
/// `^[a-z]{2,3}(-[A-Z][a-z]{3})?(-[A-Z]{2})?$` — e.g. `en`, `en-US`,
/// `zh-Hans-CN`. The real Fluent loader may broaden this; the validation
/// here is conservative on purpose so we don't lock in something the loader
/// would reject later.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LocaleId(String);

impl LocaleId {
    /// Validate and wrap `s`.
    ///
    /// # Errors
    ///
    /// Returns [`LocaleIdError::Empty`] for the empty string or
    /// [`LocaleIdError::BadFormat`] for anything outside the documented
    /// subset.
    pub fn new(s: &str) -> Result<Self, LocaleIdError> {
        if s.is_empty() {
            return Err(LocaleIdError::Empty);
        }
        // The pattern is a static literal known to compile; on the
        // theoretically-impossible compile failure we treat the input as
        // `BadFormat` so we never panic, satisfying the workspace
        // no-`unwrap` rule outside test code.
        let matched = Regex::new(LOCALE_PATTERN)
            .map(|re| re.is_match(s))
            .unwrap_or(false);
        if !matched {
            return Err(LocaleIdError::BadFormat {
                input: s.to_owned(),
            });
        }
        Ok(Self(s.to_owned()))
    }

    /// Borrow the underlying tag as a `&str`.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl FromStr for LocaleId {
    type Err = LocaleIdError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::new(s)
    }
}

impl Display for LocaleId {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl AsRef<str> for LocaleId {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl Serialize for LocaleId {
    fn serialize<S: serde::Serializer>(&self, ser: S) -> Result<S::Ok, S::Error> {
        ser.serialize_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for LocaleId {
    fn deserialize<D: serde::Deserializer<'de>>(de: D) -> Result<Self, D::Error> {
        let s = String::deserialize(de)?;
        Self::new(&s).map_err(serde::de::Error::custom)
    }
}

/// The BCP-47-ish subset accepted by [`LocaleId`].
const LOCALE_PATTERN: &str = r"^[a-z]{2,3}(-[A-Z][a-z]{3})?(-[A-Z]{2})?$";

/// First-failure rejection from [`LocaleId::new`] / [`LocaleId::from_str`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LocaleIdError {
    /// Input was the empty string.
    Empty,
    /// Input did not match the documented BCP-47 subset.
    BadFormat {
        /// The rejected input.
        input: String,
    },
}

impl Display for LocaleIdError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => f.write_str("locale id must not be empty"),
            Self::BadFormat { input } => {
                write!(f, "locale id {input:?} does not match BCP-47 subset")
            }
        }
    }
}

impl Error for LocaleIdError {}

// ---------------------------------------------------------------------------
// MessageId
// ---------------------------------------------------------------------------

/// Maximum length, in bytes, of a [`MessageId`].
const MAX_MESSAGE_ID_LEN: usize = 128;

/// Stable identifier for a single message within a [`MessageCatalog`].
///
/// Rules:
/// - 1..=128 bytes
/// - ASCII `[a-zA-Z0-9._-]`
/// - must not start with `-`, `.`, or `_`
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MessageId(String);

impl MessageId {
    /// Validate and wrap `s`. See the type-level docs for the rules.
    ///
    /// # Errors
    ///
    /// Returns the first failing [`MessageIdError`].
    pub fn new(s: &str) -> Result<Self, MessageIdError> {
        if s.is_empty() {
            return Err(MessageIdError::Empty);
        }
        if s.len() > MAX_MESSAGE_ID_LEN {
            return Err(MessageIdError::TooLong { len: s.len() });
        }
        let mut chars = s.chars().enumerate();
        for (idx, c) in chars.by_ref() {
            if idx == 0 && (c == '-' || c == '.' || c == '_') {
                return Err(MessageIdError::BadPrefix { ch: c });
            }
            if !is_message_id_char(c) {
                return Err(MessageIdError::InvalidChar { ch: c });
            }
        }
        Ok(Self(s.to_owned()))
    }

    /// Borrow the underlying id as a `&str`.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[inline]
const fn is_message_id_char(c: char) -> bool {
    matches!(c, 'a'..='z' | 'A'..='Z' | '0'..='9' | '.' | '_' | '-')
}

impl FromStr for MessageId {
    type Err = MessageIdError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::new(s)
    }
}

impl Display for MessageId {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl AsRef<str> for MessageId {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl Serialize for MessageId {
    fn serialize<S: serde::Serializer>(&self, ser: S) -> Result<S::Ok, S::Error> {
        ser.serialize_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for MessageId {
    fn deserialize<D: serde::Deserializer<'de>>(de: D) -> Result<Self, D::Error> {
        let s = String::deserialize(de)?;
        Self::new(&s).map_err(serde::de::Error::custom)
    }
}

/// First-failure rejection from [`MessageId::new`] / [`MessageId::from_str`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MessageIdError {
    /// Input was the empty string.
    Empty,
    /// Input exceeded the 128-byte limit.
    TooLong {
        /// Observed length in bytes.
        len: usize,
    },
    /// Encountered a character outside `[a-zA-Z0-9._-]`.
    InvalidChar {
        /// The offending character.
        ch: char,
    },
    /// First character was `-`, `.`, or `_`.
    BadPrefix {
        /// The offending leading character.
        ch: char,
    },
}

impl Display for MessageIdError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => f.write_str("message id must not be empty"),
            Self::TooLong { len } => {
                write!(f, "message id is {len} bytes; max is {MAX_MESSAGE_ID_LEN}")
            }
            Self::InvalidChar { ch } => {
                write!(f, "message id contains invalid character {ch:?}")
            }
            Self::BadPrefix { ch } => {
                write!(f, "message id must not start with {ch:?}")
            }
        }
    }
}

impl Error for MessageIdError {}

// ---------------------------------------------------------------------------
// Message + MessageCatalog
// ---------------------------------------------------------------------------

/// A single localized message: id + template body.
///
/// Placeholders in `body` use the tiny Fluent-ish `{ $name }` syntax. The
/// loader is intentionally permissive about surrounding whitespace inside the
/// braces — `{$name}`, `{ $name }`, and `{  $name  }` are all equivalent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Message {
    /// The id this message is keyed by inside its catalog.
    pub id: MessageId,
    /// The template body; see the type-level docs for placeholder syntax.
    pub body: String,
}

/// A collection of messages for a single [`LocaleId`].
///
/// Catalogs are deliberately serializable so the Phase-2 loader can persist
/// them or hand them to the TUI layer without round-tripping through the
/// `.ftl` text form.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MessageCatalog {
    /// The locale this catalog supplies.
    pub locale: LocaleId,
    /// Messages keyed by id.
    pub messages: BTreeMap<MessageId, Message>,
}

// ---------------------------------------------------------------------------
// FluentArg
// ---------------------------------------------------------------------------

/// A value substituted into a `{ $name }` placeholder.
///
/// Floats are stringified with four fractional digits and bools as
/// `"true"` / `"false"` — both choices match what the eventual Fluent
/// loader will do by default, so call sites today won't need to change when
/// it lands.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FluentArg {
    /// A string value, inserted verbatim.
    Str(String),
    /// An integer value.
    Int(i64),
    /// A floating-point value, formatted as `{:.4}`.
    Float(f64),
    /// A boolean value, formatted as `true` / `false`.
    Bool(bool),
}

impl Display for FluentArg {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Str(s) => f.write_str(s),
            Self::Int(i) => write!(f, "{i}"),
            Self::Float(v) => write!(f, "{v:.4}"),
            Self::Bool(b) => f.write_str(if *b { "true" } else { "false" }),
        }
    }
}

// ---------------------------------------------------------------------------
// I18nBundle
// ---------------------------------------------------------------------------

/// A collection of [`MessageCatalog`]s plus a fallback locale.
///
/// The lookup order is always `(requested, fallback)`. Catalogs are stored
/// in a `BTreeMap` so [`loaded_locales`](Self::loaded_locales) can return
/// them sorted without an extra `sort` pass.
#[derive(Debug, Clone)]
pub struct I18nBundle {
    catalogs: BTreeMap<LocaleId, MessageCatalog>,
    fallback: LocaleId,
}

impl I18nBundle {
    /// Build an empty bundle keyed by `fallback`.
    #[must_use]
    pub const fn new(fallback: LocaleId) -> Self {
        Self {
            catalogs: BTreeMap::new(),
            fallback,
        }
    }

    /// Install (or replace) the catalog for its declared locale.
    pub fn insert_catalog(&mut self, catalog: MessageCatalog) {
        self.catalogs.insert(catalog.locale.clone(), catalog);
    }

    /// Return all loaded locales, sorted.
    #[must_use]
    pub fn loaded_locales(&self) -> Vec<LocaleId> {
        self.catalogs.keys().cloned().collect()
    }

    /// Check whether a message id is loaded for `locale`.
    #[must_use]
    pub fn has_message(&self, locale: &LocaleId, id: &MessageId) -> bool {
        self.catalogs
            .get(locale)
            .is_some_and(|cat| cat.messages.contains_key(id))
    }

    /// Borrow the fallback locale.
    #[must_use]
    pub const fn fallback_locale(&self) -> &LocaleId {
        &self.fallback
    }

    /// Look up `id` in `locale` (falling back to `self.fallback`) and
    /// interpolate `args` into its body.
    ///
    /// # Errors
    ///
    /// - [`I18nError::MissingMessage`] if neither catalog has `id`.
    /// - [`I18nError::MissingArg`] if the body references a placeholder
    ///   that isn't in `args`.
    /// - [`I18nError::BadTemplate`] for mismatched / nested `{` `}`.
    pub fn lookup(
        &self,
        locale: &LocaleId,
        id: &MessageId,
        args: &BTreeMap<String, FluentArg>,
    ) -> Result<String, I18nError> {
        let mut tried: Vec<LocaleId> = Vec::with_capacity(2);
        tried.push(locale.clone());

        let message = self
            .catalogs
            .get(locale)
            .and_then(|cat| cat.messages.get(id))
            .or_else(|| {
                if locale == &self.fallback {
                    None
                } else {
                    tried.push(self.fallback.clone());
                    self.catalogs
                        .get(&self.fallback)
                        .and_then(|cat| cat.messages.get(id))
                }
            });

        let Some(msg) = message else {
            return Err(I18nError::MissingMessage {
                id: id.clone(),
                locales_tried: tried,
            });
        };

        interpolate(&msg.body, args)
    }
}

/// Render `template` by substituting each `{ $name }` with the matching
/// [`FluentArg`].
///
/// Plain `{` or `}` outside of a placeholder is treated as a template error;
/// the eventual Fluent loader allows escaping via doubling, but this minimal
/// parser is strict.
fn interpolate(template: &str, args: &BTreeMap<String, FluentArg>) -> Result<String, I18nError> {
    use std::fmt::Write as _;

    let mut out = String::with_capacity(template.len());
    let mut chars = template.chars();
    while let Some(c) = chars.next() {
        match c {
            '{' => {
                // Read until '}', expecting "$name" (with optional whitespace).
                let mut inner = String::new();
                let mut closed = false;
                for ic in chars.by_ref() {
                    if ic == '}' {
                        closed = true;
                        break;
                    }
                    if ic == '{' {
                        return Err(I18nError::BadTemplate(format!(
                            "nested '{{' in template: {template:?}"
                        )));
                    }
                    inner.push(ic);
                }
                if !closed {
                    return Err(I18nError::BadTemplate(format!(
                        "unclosed '{{' in template: {template:?}"
                    )));
                }
                let trimmed = inner.trim();
                let Some(name) = trimmed.strip_prefix('$') else {
                    return Err(I18nError::BadTemplate(format!(
                        "placeholder {trimmed:?} must start with '$' in template: {template:?}"
                    )));
                };
                let name = name.trim();
                if name.is_empty() {
                    return Err(I18nError::BadTemplate(format!(
                        "empty placeholder name in template: {template:?}"
                    )));
                }
                let Some(val) = args.get(name) else {
                    return Err(I18nError::MissingArg {
                        name: name.to_owned(),
                    });
                };
                // Writing into a `String` is infallible; the `Result` is
                // here only to satisfy the trait signature.
                let _ = write!(out, "{val}");
            }
            '}' => {
                return Err(I18nError::BadTemplate(format!(
                    "unexpected '}}' in template: {template:?}"
                )));
            }
            other => out.push(other),
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// I18nError
// ---------------------------------------------------------------------------

/// Failure surface for [`I18nBundle::lookup`] and [`parse_simple_ftl`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum I18nError {
    /// Neither the requested locale nor the fallback had `id`.
    MissingMessage {
        /// The id that was looked up.
        id: MessageId,
        /// Locales that were tried, in order.
        locales_tried: Vec<LocaleId>,
    },
    /// The template referenced `{ $name }` but `args` had no `name` entry.
    MissingArg {
        /// The missing placeholder name.
        name: String,
    },
    /// The template body was malformed (mismatched braces, empty placeholder).
    BadTemplate(String),
    /// A locale string in an `.ftl` header failed [`LocaleId::new`].
    BadLocale(LocaleIdError),
    /// A message id in an `.ftl` line failed [`MessageId::new`].
    BadMessageId(MessageIdError),
}

impl Display for I18nError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingMessage { id, locales_tried } => {
                let locales: Vec<&str> = locales_tried.iter().map(LocaleId::as_str).collect();
                write!(
                    f,
                    "message {id:?} not found in any of: [{}]",
                    locales.join(", ")
                )
            }
            Self::MissingArg { name } => {
                write!(f, "missing arg for placeholder {name:?}")
            }
            Self::BadTemplate(msg) => write!(f, "bad template: {msg}"),
            Self::BadLocale(err) => write!(f, "bad locale in catalog header: {err}"),
            Self::BadMessageId(err) => write!(f, "bad message id in catalog: {err}"),
        }
    }
}

impl Error for I18nError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::BadLocale(err) => Some(err),
            Self::BadMessageId(err) => Some(err),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// parse_simple_ftl
// ---------------------------------------------------------------------------

/// Parse a minimal `.ftl`-style document into a [`MessageCatalog`].
///
/// Grammar:
///
/// - The first non-blank line **must** be `# locale: <tag>` and supplies the
///   catalog's [`LocaleId`].
/// - Any other line starting with `#` is a comment and is ignored.
/// - Blank lines are ignored.
/// - All other lines must be of the form `id = body`. The `id` is validated
///   via [`MessageId::new`]; the `body` is the remainder of the line after
///   the first `=`, with surrounding whitespace trimmed.
///
/// Multi-line bodies are **not** supported in this minimal subset; the
/// real Fluent loader (landing later) will replace this parser entirely.
///
/// # Errors
///
/// - [`I18nError::BadTemplate`] for a missing locale header, a missing `=`,
///   or a duplicate id.
/// - [`I18nError::BadLocale`] if the header tag fails [`LocaleId::new`].
/// - [`I18nError::BadMessageId`] if a line's id fails [`MessageId::new`].
pub fn parse_simple_ftl(s: &str) -> Result<MessageCatalog, I18nError> {
    let mut locale: Option<LocaleId> = None;
    let mut messages: BTreeMap<MessageId, Message> = BTreeMap::new();

    for raw in s.lines() {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(rest) = line.strip_prefix('#') {
            // Header is "# locale: <tag>"; other comments are ignored.
            let rest = rest.trim();
            if let Some(tag) = rest.strip_prefix("locale:") {
                if locale.is_some() {
                    // Duplicate header: keep the first, ignore later ones.
                    continue;
                }
                let tag = tag.trim();
                let parsed = LocaleId::new(tag).map_err(I18nError::BadLocale)?;
                locale = Some(parsed);
            }
            continue;
        }
        if locale.is_none() {
            return Err(I18nError::BadTemplate(
                "missing required '# locale: <tag>' header before first message".to_owned(),
            ));
        }
        let Some((id_part, body_part)) = line.split_once('=') else {
            return Err(I18nError::BadTemplate(format!(
                "missing '=' in line: {line:?}"
            )));
        };
        let id = MessageId::new(id_part.trim()).map_err(I18nError::BadMessageId)?;
        let body = body_part.trim().to_owned();
        if messages.contains_key(&id) {
            return Err(I18nError::BadTemplate(format!(
                "duplicate message id: {id}"
            )));
        }
        messages.insert(id.clone(), Message { id, body });
    }

    let Some(locale) = locale else {
        return Err(I18nError::BadTemplate(
            "missing required '# locale: <tag>' header".to_owned(),
        ));
    };

    Ok(MessageCatalog { locale, messages })
}

/// Default English message catalog — parses the bundled `en.ftl`
/// shipped under `crates/stratum-runtime/i18n/`.
///
/// # Errors
/// Returns `Err` only if the bundled file is malformed (which would
/// be a compile-time-detectable bug — CI runs this in the i18n tests).
pub fn default_en_catalog() -> Result<MessageCatalog, I18nError> {
    parse_simple_ftl(BUNDLED_EN_FTL)
}

/// Build the default [`I18nBundle`] — en as both active and fallback.
/// CLI callers use this until a `--locale` flag lands.
///
/// # Errors
/// Same as [`default_en_catalog`].
pub fn default_bundle() -> Result<I18nBundle, I18nError> {
    let cat = default_en_catalog()?;
    let mut bundle = I18nBundle::new(cat.locale.clone());
    bundle.insert_catalog(cat);
    Ok(bundle)
}

/// Bundled English catalog text — pulled in at compile time so the
/// runtime ships with at least one locale even on minimal builds.
const BUNDLED_EN_FTL: &str = include_str!("../i18n/en.ftl");

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn lid(s: &str) -> LocaleId {
        LocaleId::new(s).expect("valid locale id in test")
    }

    fn mid(s: &str) -> MessageId {
        MessageId::new(s).expect("valid message id in test")
    }

    // -- LocaleId ---------------------------------------------------------

    #[test]
    fn locale_two_letter_ok() {
        assert_eq!(LocaleId::from_str("en").unwrap().as_str(), "en");
    }

    #[test]
    fn locale_region_ok() {
        assert_eq!(LocaleId::from_str("en-US").unwrap().as_str(), "en-US");
    }

    #[test]
    fn locale_script_region_ok() {
        assert_eq!(
            LocaleId::from_str("zh-Hans-CN").unwrap().as_str(),
            "zh-Hans-CN"
        );
    }

    #[test]
    fn locale_three_letter_ok() {
        assert_eq!(LocaleId::from_str("haw").unwrap().as_str(), "haw");
    }

    #[test]
    fn locale_empty_rejected() {
        assert_eq!(LocaleId::from_str(""), Err(LocaleIdError::Empty));
    }

    #[test]
    fn locale_uppercase_lang_rejected() {
        let err = LocaleId::from_str("ENG").unwrap_err();
        assert!(matches!(err, LocaleIdError::BadFormat { ref input } if input == "ENG"));
    }

    #[test]
    fn locale_underscore_rejected() {
        let err = LocaleId::from_str("en_US").unwrap_err();
        assert!(matches!(err, LocaleIdError::BadFormat { .. }));
    }

    #[test]
    fn locale_lower_region_rejected() {
        // `us` is lowercase, the region rule wants two uppercase letters.
        assert!(LocaleId::from_str("en-us").is_err());
    }

    #[test]
    fn locale_display_round_trip() {
        let lid = LocaleId::from_str("en-US").unwrap();
        assert_eq!(format!("{lid}"), "en-US");
    }

    #[test]
    fn locale_serde_round_trip() {
        let lid = LocaleId::from_str("en-US").unwrap();
        let json = serde_json::to_string(&lid).unwrap();
        assert_eq!(json, "\"en-US\"");
        let back: LocaleId = serde_json::from_str(&json).unwrap();
        assert_eq!(back, lid);
    }

    #[test]
    fn locale_serde_rejects_bad() {
        let err = serde_json::from_str::<LocaleId>("\"en_US\"").unwrap_err();
        assert!(err.to_string().contains("BCP-47"));
    }

    #[test]
    fn locale_ord_is_lexicographic() {
        let mut v = vec![lid("fr-FR"), lid("en"), lid("en-US")];
        v.sort();
        assert_eq!(v, vec![lid("en"), lid("en-US"), lid("fr-FR")]);
    }

    // -- MessageId --------------------------------------------------------

    #[test]
    fn message_id_ok() {
        assert_eq!(MessageId::from_str("greet").unwrap().as_str(), "greet");
        assert_eq!(
            MessageId::from_str("g.r-e_t1").unwrap().as_str(),
            "g.r-e_t1"
        );
    }

    #[test]
    fn message_id_empty_rejected() {
        assert_eq!(MessageId::from_str(""), Err(MessageIdError::Empty));
    }

    #[test]
    fn message_id_too_long_rejected() {
        let s = "a".repeat(MAX_MESSAGE_ID_LEN + 1);
        let err = MessageId::from_str(&s).unwrap_err();
        assert!(matches!(err, MessageIdError::TooLong { len } if len == MAX_MESSAGE_ID_LEN + 1));
    }

    #[test]
    fn message_id_invalid_char_rejected() {
        let err = MessageId::from_str("a!b").unwrap_err();
        assert!(matches!(err, MessageIdError::InvalidChar { ch: '!' }));
    }

    #[test]
    fn message_id_first_char_invalid_rejected() {
        // First char is neither a `BadPrefix` symbol nor a valid id char.
        let err = MessageId::from_str("!ab").unwrap_err();
        assert!(matches!(err, MessageIdError::InvalidChar { ch: '!' }));
    }

    #[test]
    fn locale_id_as_ref_str() {
        let l = lid("en-US");
        let s: &str = l.as_ref();
        assert_eq!(s, "en-US");
    }

    #[test]
    fn message_id_as_ref_str() {
        let m = mid("hello");
        let s: &str = m.as_ref();
        assert_eq!(s, "hello");
    }

    #[test]
    fn message_id_bad_prefix_rejected() {
        for c in ['-', '.', '_'] {
            let s = format!("{c}id");
            let err = MessageId::from_str(&s).unwrap_err();
            assert!(matches!(err, MessageIdError::BadPrefix { ch } if ch == c));
        }
    }

    #[test]
    fn message_id_serde_round_trip() {
        let id = MessageId::new("greet").unwrap();
        let json = serde_json::to_string(&id).unwrap();
        assert_eq!(json, "\"greet\"");
        let back: MessageId = serde_json::from_str(&json).unwrap();
        assert_eq!(back, id);
    }

    // -- Message / Catalog serde -----------------------------------------

    #[test]
    fn message_serde_round_trip() {
        let m = Message {
            id: mid("hello"),
            body: "Hello, { $name }!".to_owned(),
        };
        let json = serde_json::to_string(&m).unwrap();
        let back: Message = serde_json::from_str(&json).unwrap();
        assert_eq!(back, m);
    }

    #[test]
    fn catalog_serde_round_trip() {
        let mut messages = BTreeMap::new();
        messages.insert(
            mid("hi"),
            Message {
                id: mid("hi"),
                body: "Hi!".to_owned(),
            },
        );
        let cat = MessageCatalog {
            locale: lid("en"),
            messages,
        };
        let json = serde_json::to_string(&cat).unwrap();
        let back: MessageCatalog = serde_json::from_str(&json).unwrap();
        assert_eq!(back, cat);
    }

    // -- I18nBundle lookup -----------------------------------------------

    fn sample_bundle() -> I18nBundle {
        let mut en = BTreeMap::new();
        en.insert(
            mid("hello"),
            Message {
                id: mid("hello"),
                body: "Hello, { $name }!".to_owned(),
            },
        );
        en.insert(
            mid("only_en"),
            Message {
                id: mid("only_en"),
                body: "english only".to_owned(),
            },
        );
        let mut fr = BTreeMap::new();
        fr.insert(
            mid("hello"),
            Message {
                id: mid("hello"),
                body: "Bonjour, { $name } !".to_owned(),
            },
        );

        let mut bundle = I18nBundle::new(lid("en"));
        bundle.insert_catalog(MessageCatalog {
            locale: lid("en"),
            messages: en,
        });
        bundle.insert_catalog(MessageCatalog {
            locale: lid("fr"),
            messages: fr,
        });
        bundle
    }

    #[test]
    fn lookup_primary_locale() {
        let b = sample_bundle();
        let mut args = BTreeMap::new();
        args.insert("name".to_owned(), FluentArg::Str("Ada".to_owned()));
        let s = b.lookup(&lid("fr"), &mid("hello"), &args).unwrap();
        assert_eq!(s, "Bonjour, Ada !");
    }

    #[test]
    fn lookup_falls_back() {
        let b = sample_bundle();
        let args = BTreeMap::new();
        let s = b.lookup(&lid("fr"), &mid("only_en"), &args).unwrap();
        assert_eq!(s, "english only");
    }

    #[test]
    fn lookup_missing_lists_locales() {
        let b = sample_bundle();
        let args = BTreeMap::new();
        let err = b.lookup(&lid("fr"), &mid("nope"), &args).unwrap_err();
        let I18nError::MissingMessage { id, locales_tried } = err else {
            unreachable!("expected MissingMessage");
        };
        assert_eq!(id, mid("nope"));
        assert_eq!(locales_tried, vec![lid("fr"), lid("en")]);
    }

    #[test]
    fn lookup_missing_when_primary_is_fallback() {
        let b = sample_bundle();
        let args = BTreeMap::new();
        let err = b.lookup(&lid("en"), &mid("nope"), &args).unwrap_err();
        let I18nError::MissingMessage { locales_tried, .. } = err else {
            unreachable!("expected MissingMessage");
        };
        assert_eq!(locales_tried, vec![lid("en")]);
    }

    #[test]
    fn lookup_interpolates_named_arg() {
        let b = sample_bundle();
        let mut args = BTreeMap::new();
        args.insert("name".to_owned(), FluentArg::Str("Grace".to_owned()));
        let s = b.lookup(&lid("en"), &mid("hello"), &args).unwrap();
        assert_eq!(s, "Hello, Grace!");
    }

    #[test]
    fn lookup_missing_arg_reports_name() {
        let b = sample_bundle();
        let args = BTreeMap::new();
        let err = b.lookup(&lid("en"), &mid("hello"), &args).unwrap_err();
        assert!(matches!(err, I18nError::MissingArg { ref name } if name == "name"));
    }

    #[test]
    fn lookup_bad_template_unclosed() {
        let mut bundle = I18nBundle::new(lid("en"));
        let mut messages = BTreeMap::new();
        messages.insert(
            mid("bad"),
            Message {
                id: mid("bad"),
                body: "broken { $name".to_owned(),
            },
        );
        bundle.insert_catalog(MessageCatalog {
            locale: lid("en"),
            messages,
        });
        let mut args = BTreeMap::new();
        args.insert("name".to_owned(), FluentArg::Str("x".to_owned()));
        let err = bundle.lookup(&lid("en"), &mid("bad"), &args).unwrap_err();
        assert!(matches!(err, I18nError::BadTemplate(_)));
    }

    #[test]
    fn lookup_bad_template_stray_close() {
        let mut bundle = I18nBundle::new(lid("en"));
        let mut messages = BTreeMap::new();
        messages.insert(
            mid("bad"),
            Message {
                id: mid("bad"),
                body: "stray } here".to_owned(),
            },
        );
        bundle.insert_catalog(MessageCatalog {
            locale: lid("en"),
            messages,
        });
        let args = BTreeMap::new();
        let err = bundle.lookup(&lid("en"), &mid("bad"), &args).unwrap_err();
        assert!(matches!(err, I18nError::BadTemplate(_)));
    }

    #[test]
    fn lookup_bad_template_nested() {
        let mut bundle = I18nBundle::new(lid("en"));
        let mut messages = BTreeMap::new();
        messages.insert(
            mid("bad"),
            Message {
                id: mid("bad"),
                body: "{ { $name } }".to_owned(),
            },
        );
        bundle.insert_catalog(MessageCatalog {
            locale: lid("en"),
            messages,
        });
        let mut args = BTreeMap::new();
        args.insert("name".to_owned(), FluentArg::Str("x".to_owned()));
        let err = bundle.lookup(&lid("en"), &mid("bad"), &args).unwrap_err();
        assert!(matches!(err, I18nError::BadTemplate(_)));
    }

    #[test]
    fn lookup_bad_template_no_dollar() {
        let mut bundle = I18nBundle::new(lid("en"));
        let mut messages = BTreeMap::new();
        messages.insert(
            mid("bad"),
            Message {
                id: mid("bad"),
                body: "{ name }".to_owned(),
            },
        );
        bundle.insert_catalog(MessageCatalog {
            locale: lid("en"),
            messages,
        });
        let args = BTreeMap::new();
        let err = bundle.lookup(&lid("en"), &mid("bad"), &args).unwrap_err();
        assert!(matches!(err, I18nError::BadTemplate(_)));
    }

    #[test]
    fn lookup_bad_template_empty_name() {
        let mut bundle = I18nBundle::new(lid("en"));
        let mut messages = BTreeMap::new();
        messages.insert(
            mid("bad"),
            Message {
                id: mid("bad"),
                body: "{ $ }".to_owned(),
            },
        );
        bundle.insert_catalog(MessageCatalog {
            locale: lid("en"),
            messages,
        });
        let args = BTreeMap::new();
        let err = bundle.lookup(&lid("en"), &mid("bad"), &args).unwrap_err();
        assert!(matches!(err, I18nError::BadTemplate(_)));
    }

    // -- FluentArg formatting --------------------------------------------

    fn render(body: &str, args: &BTreeMap<String, FluentArg>) -> String {
        let mut bundle = I18nBundle::new(lid("en"));
        let mut messages = BTreeMap::new();
        messages.insert(
            mid("m"),
            Message {
                id: mid("m"),
                body: body.to_owned(),
            },
        );
        bundle.insert_catalog(MessageCatalog {
            locale: lid("en"),
            messages,
        });
        bundle.lookup(&lid("en"), &mid("m"), args).unwrap()
    }

    #[test]
    fn lookup_formats_int() {
        let mut args = BTreeMap::new();
        args.insert("n".to_owned(), FluentArg::Int(-42));
        assert_eq!(render("n={ $n }", &args), "n=-42");
    }

    #[test]
    fn lookup_formats_float_4dp() {
        let mut args = BTreeMap::new();
        // Intentionally not `std::f64::consts::PI` — we want to verify the
        // 4-decimal formatting on an arbitrary float, not the math constant.
        #[allow(clippy::approx_constant)]
        let v = 3.14159_f64;
        args.insert("v".to_owned(), FluentArg::Float(v));
        assert_eq!(render("v={ $v }", &args), "v=3.1416");
    }

    #[test]
    fn lookup_formats_bool() {
        let mut args = BTreeMap::new();
        args.insert("t".to_owned(), FluentArg::Bool(true));
        args.insert("f".to_owned(), FluentArg::Bool(false));
        assert_eq!(render("{ $t }/{ $f }", &args), "true/false");
    }

    #[test]
    fn fluent_arg_serde_round_trip() {
        let v = FluentArg::Int(7);
        let json = serde_json::to_string(&v).unwrap();
        let back: FluentArg = serde_json::from_str(&json).unwrap();
        assert!(matches!(back, FluentArg::Int(7)));
    }

    // -- Bundle helpers --------------------------------------------------

    #[test]
    fn loaded_locales_sorted() {
        let mut bundle = I18nBundle::new(lid("en"));
        for tag in ["fr", "de", "en", "ja"] {
            bundle.insert_catalog(MessageCatalog {
                locale: lid(tag),
                messages: BTreeMap::new(),
            });
        }
        assert_eq!(
            bundle.loaded_locales(),
            vec![lid("de"), lid("en"), lid("fr"), lid("ja")]
        );
    }

    #[test]
    fn insert_catalog_overwrites() {
        let mut bundle = I18nBundle::new(lid("en"));
        let mut m1 = BTreeMap::new();
        m1.insert(
            mid("a"),
            Message {
                id: mid("a"),
                body: "old".to_owned(),
            },
        );
        bundle.insert_catalog(MessageCatalog {
            locale: lid("en"),
            messages: m1,
        });
        let mut m2 = BTreeMap::new();
        m2.insert(
            mid("a"),
            Message {
                id: mid("a"),
                body: "new".to_owned(),
            },
        );
        bundle.insert_catalog(MessageCatalog {
            locale: lid("en"),
            messages: m2,
        });
        let s = bundle
            .lookup(&lid("en"), &mid("a"), &BTreeMap::new())
            .unwrap();
        assert_eq!(s, "new");
    }

    #[test]
    fn has_message_true_false() {
        let b = sample_bundle();
        assert!(b.has_message(&lid("en"), &mid("hello")));
        assert!(!b.has_message(&lid("en"), &mid("nope")));
        assert!(!b.has_message(&lid("de"), &mid("hello")));
    }

    #[test]
    fn fallback_locale_accessor() {
        let b = sample_bundle();
        assert_eq!(b.fallback_locale(), &lid("en"));
    }

    // -- parse_simple_ftl ------------------------------------------------

    #[test]
    fn parse_ftl_happy() {
        let src = "\
# locale: en-US
# A friendly greeting.
hello = Hello, { $name }!

bye = Bye!
n.count = { $n }
";
        let cat = parse_simple_ftl(src).unwrap();
        assert_eq!(cat.locale, lid("en-US"));
        assert_eq!(cat.messages.len(), 3);
        assert_eq!(
            cat.messages.get(&mid("hello")).unwrap().body,
            "Hello, { $name }!"
        );
        assert_eq!(cat.messages.get(&mid("bye")).unwrap().body, "Bye!");
        assert_eq!(cat.messages.get(&mid("n.count")).unwrap().body, "{ $n }");
    }

    #[test]
    fn parse_ftl_no_header_rejected() {
        let src = "hello = Hi!";
        let err = parse_simple_ftl(src).unwrap_err();
        assert!(matches!(err, I18nError::BadTemplate(_)));
    }

    #[test]
    fn parse_ftl_no_header_only_comments_rejected() {
        let src = "# just a comment\n# another\n";
        let err = parse_simple_ftl(src).unwrap_err();
        assert!(matches!(err, I18nError::BadTemplate(_)));
    }

    #[test]
    fn parse_ftl_ignores_comments_and_blanks() {
        let src = "\n# locale: en\n\n# greeting follows\nhello = Hi!\n\n";
        let cat = parse_simple_ftl(src).unwrap();
        assert_eq!(cat.messages.len(), 1);
        assert_eq!(cat.locale, lid("en"));
    }

    #[test]
    fn parse_ftl_bad_message_id() {
        let src = "# locale: en\n-bad = nope\n";
        let err = parse_simple_ftl(src).unwrap_err();
        assert!(matches!(err, I18nError::BadMessageId(_)));
    }

    #[test]
    fn parse_ftl_bad_locale() {
        let src = "# locale: ENG\nhello = Hi!\n";
        let err = parse_simple_ftl(src).unwrap_err();
        assert!(matches!(err, I18nError::BadLocale(_)));
    }

    #[test]
    fn parse_ftl_missing_equals() {
        let src = "# locale: en\nhello world\n";
        let err = parse_simple_ftl(src).unwrap_err();
        assert!(matches!(err, I18nError::BadTemplate(_)));
    }

    #[test]
    fn parse_ftl_duplicate_id() {
        let src = "# locale: en\nhello = One\nhello = Two\n";
        let err = parse_simple_ftl(src).unwrap_err();
        assert!(matches!(err, I18nError::BadTemplate(_)));
    }

    #[test]
    fn parse_ftl_second_locale_header_ignored() {
        let src = "# locale: en\n# locale: fr\nhello = Hi!\n";
        let cat = parse_simple_ftl(src).unwrap();
        assert_eq!(cat.locale, lid("en"));
    }

    // -- I18nError Display smoke -----------------------------------------

    #[test]
    fn i18n_error_display_smoke() {
        let e = I18nError::MissingMessage {
            id: mid("x"),
            locales_tried: vec![lid("fr"), lid("en")],
        };
        assert!(e.to_string().contains("fr"));
        assert!(e.to_string().contains("en"));

        let e = I18nError::MissingArg {
            name: "name".to_owned(),
        };
        assert!(e.to_string().contains("name"));

        let e = I18nError::BadTemplate("boom".to_owned());
        assert!(e.to_string().contains("boom"));

        let e = I18nError::BadLocale(LocaleIdError::Empty);
        assert!(e.to_string().contains("locale"));
        assert!(e.source().is_some());

        let e = I18nError::BadMessageId(MessageIdError::Empty);
        assert!(e.to_string().contains("message id"));
        assert!(e.source().is_some());

        // Errors without a source.
        let e = I18nError::BadTemplate("x".to_owned());
        assert!(e.source().is_none());
    }

    #[test]
    fn locale_id_error_display_smoke() {
        assert!(LocaleIdError::Empty.to_string().contains("empty"));
        let s = LocaleIdError::BadFormat {
            input: "x_y".to_owned(),
        }
        .to_string();
        assert!(s.contains("x_y"));
    }

    #[test]
    fn message_id_error_display_smoke() {
        assert!(MessageIdError::Empty.to_string().contains("empty"));
        assert!(MessageIdError::TooLong { len: 200 }
            .to_string()
            .contains("200"));
        assert!(MessageIdError::InvalidChar { ch: '!' }
            .to_string()
            .contains('!'));
        assert!(MessageIdError::BadPrefix { ch: '-' }
            .to_string()
            .contains('-'));
    }

    #[test]
    fn fluent_arg_display_smoke() {
        assert_eq!(FluentArg::Str("hi".to_owned()).to_string(), "hi");
        assert_eq!(FluentArg::Int(0).to_string(), "0");
        assert_eq!(FluentArg::Float(1.0).to_string(), "1.0000");
        assert_eq!(FluentArg::Bool(true).to_string(), "true");
        assert_eq!(FluentArg::Bool(false).to_string(), "false");
    }

    // ---- Bundled en.ftl regression tests --------------------------------

    #[test]
    fn bundled_en_catalog_parses() {
        let cat = default_en_catalog().expect("bundled en.ftl must parse");
        assert_eq!(cat.locale.as_str(), "en");
        assert!(cat.messages.len() >= 20);
    }

    #[test]
    fn bundled_en_catalog_has_known_message_ids() {
        let cat = default_en_catalog().unwrap();
        for id in [
            "stratum-greeting",
            "stratum-tagline",
            "tool-unknown",
            "turn-thinking",
            "cmd-compact-done",
            "memory-saved",
            "err-provider-no-text",
        ] {
            let mid = MessageId::new(id).unwrap();
            assert!(cat.messages.contains_key(&mid), "missing id: {id}");
        }
    }

    #[test]
    fn default_bundle_resolves_at_least_one_message() {
        let bundle = default_bundle().unwrap();
        let id = MessageId::new("stratum-greeting").unwrap();
        let resolved = bundle
            .lookup(
                &LocaleId::new("en").unwrap(),
                &id,
                &std::collections::BTreeMap::default(),
            )
            .unwrap();
        assert!(!resolved.is_empty());
    }
}
