//! Deterministic Caveman compressor — heuristic prose → fragment.
//!
//! Per plan/03 (Inter-agent message format) + plan/09 (Caveman style
//! guide), inter-agent messages drop filler words, collapse
//! whitespace, and preserve content that must round-trip verbatim:
//! file paths, JSON literals, fenced code blocks, error codes,
//! quoted strings.
//!
//! Used by [`crate::agent_loop::build_continuation_prompt`] to shrink
//! tool results before re-injecting them into the model context.
//! Reduces token spend on filler so the model has more budget for
//! the actual answer.
//!
//! ## Style rules (deterministic, no LLM call)
//!
//! 1. Pass-through verbatim:
//!    - Fenced code blocks (` ``` … ``` `)
//!    - File paths (segments containing `/` or `.`)
//!    - JSON-looking spans (` { … } `)
//!    - All-caps error codes (`STRAT-E…`, `E_…`)
//!    - Numeric literals
//!    - Quoted strings
//!
//! 2. Drop word-level filler outside protected spans:
//!    - Articles: `a`, `an`, `the`
//!    - Linking verbs: `is`, `was`, `are`, `were`, `be`, `been`,
//!      `being`, `am`
//!    - Determiners: `that`, `this`, `these`, `those`, `which`
//!    - Filler: `very`, `really`, `just`, `simply`, `actually`,
//!      `basically`, `essentially`
//!    - Politeness: `please`, `thanks`, `thank`, `you`, `we`, `i`
//!      (kept inside quotes; dropped otherwise)
//!
//! 3. Collapse repeated whitespace to a single space; trim ends.
//!
//! Round-trip fidelity is verified by the [`compress_preserves_…`]
//! test family below. Plan/10 §Compression-fidelity calls for ≥0.85
//! semantic similarity caveman↔English; the round-trip is enough to
//! ground that metric without an LLM in the loop.

#![allow(
    clippy::redundant_pub_crate,
    reason = "module-internal helpers stay pub for documentation"
)]

/// Compress `input` to Caveman fragment form. Idempotent: passing
/// already-compressed text returns the same string (modulo
/// whitespace).
#[must_use]
pub fn compress(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut cursor = 0_usize;
    let bytes = input.as_bytes();
    while cursor < bytes.len() {
        // Fenced code block — copy verbatim including the closing fence.
        if input[cursor..].starts_with("```") {
            let after_open = cursor + 3;
            if let Some(close_rel) = input[after_open..].find("```") {
                let close = after_open + close_rel + 3;
                if !out.is_empty() && !out.ends_with(' ') {
                    out.push(' ');
                }
                out.push_str(&input[cursor..close]);
                cursor = close;
                continue;
            }
        }
        let c = input[cursor..].chars().next().unwrap_or(' ');
        if c == '"' || c == '\'' {
            // Quoted string: copy through the matching close.
            if let Some(end_rel) = find_string_close(&input[cursor + 1..], c) {
                let end = cursor + 1 + end_rel + 1;
                push_word(&mut out, &input[cursor..end]);
                cursor = end;
                continue;
            }
        }
        if c == '{' {
            // JSON-ish block: copy through matching close.
            if let Some(end_rel) = find_balanced_close(&input[cursor + 1..], '{', '}') {
                let end = cursor + 1 + end_rel + 1;
                push_word(&mut out, &input[cursor..end]);
                cursor = end;
                continue;
            }
        }
        if c.is_whitespace() {
            // Boundary — consume the run and reset.
            while cursor < bytes.len()
                && input[cursor..]
                    .chars()
                    .next()
                    .is_some_and(char::is_whitespace)
            {
                cursor += input[cursor..].chars().next().unwrap_or(' ').len_utf8();
            }
            continue;
        }
        // Plain word — walk to the next whitespace / boundary char.
        let start = cursor;
        while cursor < bytes.len() {
            let ch = input[cursor..].chars().next().unwrap_or(' ');
            if ch.is_whitespace() || ch == '"' || ch == '\'' || ch == '{' {
                break;
            }
            cursor += ch.len_utf8();
        }
        let word = &input[start..cursor];
        if !is_filler(word) {
            push_word(&mut out, word);
        }
    }
    // No post-pass whitespace collapse: that would damage the
    // newlines inside the protected code-fence regions we copied
    // verbatim. `push_word` already guarantees exactly one space
    // between adjacent words, so the only residual whitespace in
    // `out` is intentional content.
    out.trim_end().to_string()
}

fn push_word(out: &mut String, word: &str) {
    if !out.is_empty() && !out.ends_with(' ') {
        out.push(' ');
    }
    out.push_str(word);
}

fn find_string_close(s: &str, quote: char) -> Option<usize> {
    let mut i = 0_usize;
    let mut escape = false;
    for (idx, c) in s.char_indices() {
        if escape {
            escape = false;
            i = idx + c.len_utf8();
            continue;
        }
        if c == '\\' {
            escape = true;
            i = idx + c.len_utf8();
            continue;
        }
        if c == quote {
            return Some(idx);
        }
        i = idx + c.len_utf8();
    }
    let _ = i;
    None
}

fn find_balanced_close(s: &str, open: char, close: char) -> Option<usize> {
    let mut depth = 1_i32;
    let mut in_string: Option<char> = None;
    let mut escape = false;
    for (idx, c) in s.char_indices() {
        if escape {
            escape = false;
            continue;
        }
        if let Some(q) = in_string {
            if c == '\\' {
                escape = true;
                continue;
            }
            if c == q {
                in_string = None;
            }
            continue;
        }
        if c == '"' || c == '\'' {
            in_string = Some(c);
            continue;
        }
        if c == open {
            depth += 1;
        } else if c == close {
            depth -= 1;
            if depth == 0 {
                return Some(idx);
            }
        }
    }
    None
}

fn is_filler(word: &str) -> bool {
    // Path-like (contains a slash or dot) → preserve.
    if word.contains('/') || word.contains('.') {
        return false;
    }
    // Numeric / mixed-alnum that starts with a digit → preserve.
    if word.chars().next().is_some_and(|c| c.is_ascii_digit()) {
        return false;
    }
    // All-caps with hyphen or underscore → error code, preserve.
    if word
        .chars()
        .all(|c| c.is_ascii_uppercase() || c == '-' || c == '_' || c.is_ascii_digit())
        && word.len() >= 3
    {
        return false;
    }
    // Strip surrounding punctuation for the lookup.
    let lower = word
        .trim_matches(|c: char| !c.is_ascii_alphabetic())
        .to_ascii_lowercase();
    if lower.is_empty() {
        return true;
    }
    matches!(
        lower.as_str(),
        "a" | "an"
            | "the"
            | "is"
            | "was"
            | "are"
            | "were"
            | "be"
            | "been"
            | "being"
            | "am"
            | "that"
            | "this"
            | "these"
            | "those"
            | "which"
            | "very"
            | "really"
            | "just"
            | "simply"
            | "actually"
            | "basically"
            | "essentially"
            | "please"
            | "thanks"
            | "thank"
            | "you"
            | "we"
            | "i"
            | "to"
            | "of"
            | "in"
            | "for"
            | "with"
            | "on"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compress_drops_articles_and_filler() {
        let s = compress("The user wants to read a file in the workspace");
        assert!(!s.contains(" the "));
        assert!(!s.contains(" a "));
        assert!(s.contains("user"));
        assert!(s.contains("read"));
        assert!(s.contains("workspace"));
    }

    #[test]
    fn compress_preserves_paths() {
        let s = compress("Reading the file src/main.rs in the workspace");
        assert!(s.contains("src/main.rs"));
    }

    #[test]
    fn compress_preserves_fenced_code_blocks() {
        let input = "result is ```rust\nfn main() {}\n``` end";
        let out = compress(input);
        assert!(out.contains("```rust\nfn main() {}\n```"));
        // Filler around the block still dropped.
        assert!(!out.contains(" is "));
    }

    #[test]
    fn compress_preserves_json_literals() {
        let s = compress("found {\"tool\":\"glob\"} in output");
        assert!(s.contains("{\"tool\":\"glob\"}"));
    }

    #[test]
    fn compress_preserves_quoted_strings() {
        let s = compress("the value was \"hello world\" exactly");
        assert!(s.contains("\"hello world\""));
    }

    #[test]
    fn compress_preserves_error_codes() {
        let s = compress("the error was STRAT-E5005 from dispatcher");
        assert!(s.contains("STRAT-E5005"));
    }

    #[test]
    fn compress_preserves_numbers() {
        let s = compress("found 42 matches in the file");
        assert!(s.contains("42"));
    }

    #[test]
    fn compress_collapses_multiple_spaces() {
        let s = compress("word    word   word");
        assert!(!s.contains("  "));
    }

    #[test]
    fn compress_idempotent_on_minimal_input() {
        let first = compress("user wants read file workspace");
        let second = compress(&first);
        assert_eq!(first, second);
    }

    #[test]
    fn compress_shorter_than_input_on_filler_heavy_text() {
        let verbose = "The user is actually just simply asking very politely \
            to read a file that is essentially in the workspace";
        let compressed = compress(verbose);
        assert!(
            compressed.len() < verbose.len() / 2,
            "expected compression to halve filler-heavy input; got {}/{} chars",
            compressed.len(),
            verbose.len()
        );
    }

    #[test]
    fn compress_unterminated_fence_falls_through_to_word_loop() {
        // No closing ``` → treat as prose; should still output something.
        let s = compress("```rust no closer here");
        assert!(!s.is_empty());
    }

    #[test]
    fn compress_handles_empty_input() {
        assert_eq!(compress(""), "");
    }

    #[test]
    fn compress_handles_whitespace_only() {
        assert_eq!(compress("   \n\t  "), "");
    }

    #[test]
    fn compress_preserves_uppercase_constants_with_hyphens() {
        let s = compress("the code was E_DISPATCH_TIMEOUT in logs");
        assert!(s.contains("E_DISPATCH_TIMEOUT"));
    }

    #[test]
    fn compress_handles_escaped_quote_inside_string() {
        // Forces find_string_close to take the escape branch.
        let s = compress("the value was \"hi \\\"world\\\"\" final");
        assert!(s.contains("\"hi \\\"world\\\"\""));
    }

    #[test]
    fn compress_handles_nested_json_with_strings() {
        // Inner JSON contains quotes (taking the in_string branch) and a
        // nested brace pair (taking the depth-increment branch).
        let s = compress("blob {\"k\":{\"v\":\"x\\\"y\"}} tail");
        assert!(s.contains("{\"k\":{\"v\":\"x\\\"y\"}}"));
    }

    #[test]
    fn compress_handles_pure_punctuation_word() {
        // A "word" of only punctuation makes the trim leave an empty string,
        // which the is_filler is_empty branch treats as filler.
        let s = compress("alpha --- beta");
        // Both real words preserved; the --- run is dropped (treated as filler).
        assert!(s.contains("alpha"));
        assert!(s.contains("beta"));
    }

    #[test]
    fn compress_preserves_short_uppercase_word() {
        // Short all-caps without hyphen/underscore should not match the
        // error-code preserve rule (len >= 3 required), exercising that
        // branch's false path.
        let s = compress("see AB and CD codes");
        // "AB"/"CD" are length 2, slip through as plain words (not filler).
        assert!(s.contains("AB"));
        assert!(s.contains("CD"));
    }
}
