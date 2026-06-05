//! RAG (retrieval-augmented generation) index data shape.
//!
//! This module is the stable surface the orchestrator and `stratum-tools`
//! can target today, before any real vector store lands. Phase 4+ will swap
//! the storage layer for LanceDB / sqlite-vss and embeddings for
//! Arctic-Embed-L; the types declared here are the contract those backends
//! will continue to honor.
//!
//! See `plan/01-architecture.md` (RAG row of the cold tier) and
//! `plan/30-workspace-and-project.md` §5 (per-project RAG scope).
//!
//! ## Scope of this pass
//!
//! - Pure data shape: documents, chunks, plan, in-memory index.
//! - Substring-only `search` fallback. Real embedding retrieval lands later.
//! - The public surface is infallible — failure modes are represented by an
//!   empty `Vec` or a `None`, so no new `STRAT-E…` codes are needed.
//!
//! ## Chunk span representation
//!
//! `Chunk::span` is a [`ChunkSpan`] `{ start, end }` byte-offset pair into
//! the parent document's UTF-8 text. A struct (rather than
//! `std::ops::Range`) is used so the type round-trips cleanly through
//! `serde` without a custom adapter.

use std::collections::HashMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Stable identifier for a document in the index.
///
/// Callers pick the id (typically a hash of the source path or a UUID); the
/// index treats it as opaque and uses it for `insert` / `remove` / `get`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DocumentId(pub String);

impl DocumentId {
    /// Construct a `DocumentId` from anything string-like.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Borrow the underlying id as a `&str`.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// A document tracked by the RAG index.
///
/// `source_path` is `None` for in-memory / synthetic documents (e.g. chat
/// transcripts injected for retrieval). `indexed_at` is an RFC 3339 string,
/// kept as a `String` to keep this type cheap to clone and serialize without
/// pulling `time` types into the public surface.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RagDocument {
    /// Stable id of this document within the index.
    pub id: DocumentId,
    /// Optional on-disk path the document was read from.
    pub source_path: Option<PathBuf>,
    /// Full UTF-8 text of the document.
    pub text: String,
    /// MIME type (e.g. `text/markdown`, `text/plain`).
    pub mime: String,
    /// RFC 3339 timestamp the document was indexed at.
    pub indexed_at: String,
}

/// Byte-offset span of a chunk within its parent document's `text`.
///
/// `start` is inclusive, `end` is exclusive — the same convention as
/// `std::ops::Range<usize>`. Both offsets always fall on a UTF-8 codepoint
/// boundary when produced by [`chunk_document`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChunkSpan {
    /// Inclusive start byte offset.
    pub start: usize,
    /// Exclusive end byte offset.
    pub end: usize,
}

/// A chunk of a document — the unit the retrieval layer scores against.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Chunk {
    /// Id of the document this chunk was sliced from.
    pub document_id: DocumentId,
    /// 0-based position of this chunk within its document.
    pub ordinal: u32,
    /// The chunk's text (a slice of the parent document's `text`).
    pub text: String,
    /// Byte-offset span into the parent document.
    pub span: ChunkSpan,
}

/// Knobs for slicing a document into chunks.
///
/// Defaults mirror the cold-tier bullet in `plan/01-architecture.md`:
/// 512-byte windows with 64-byte overlap. "Char" here is interpreted as
/// "byte" — UTF-8 boundaries are honored, see [`chunk_document`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChunkPlan {
    /// Maximum chunk size in bytes.
    pub max_chars: usize,
    /// Overlap between adjacent chunks in bytes.
    pub overlap_chars: usize,
}

impl Default for ChunkPlan {
    fn default() -> Self {
        Self {
            max_chars: 512,
            overlap_chars: 64,
        }
    }
}

/// Slide over `doc.text` producing UTF-8-safe overlapping chunks.
///
/// The stride is `plan.max_chars - plan.overlap_chars`. Empty input
/// produces an empty `Vec` (we never emit a zero-length chunk). When
/// `overlap_chars` is `>= max_chars`, the stride is clamped to 1 to
/// guarantee forward progress.
///
/// Chunk boundaries are always at UTF-8 codepoint boundaries: if the target
/// byte offset would land inside a multi-byte sequence, the boundary is
/// pulled back to the nearest preceding codepoint start.
#[must_use]
pub fn chunk_document(plan: &ChunkPlan, doc: &RagDocument) -> Vec<Chunk> {
    if doc.text.is_empty() || plan.max_chars == 0 {
        return Vec::new();
    }

    let stride = plan.max_chars.saturating_sub(plan.overlap_chars).max(1);

    let total = doc.text.len();
    let mut chunks: Vec<Chunk> = Vec::new();
    let mut ordinal: u32 = 0;
    let mut start: usize = 0;

    while start < total {
        let raw_end = start.saturating_add(plan.max_chars).min(total);
        let end = floor_char_boundary(&doc.text, raw_end);

        // Clamping to a codepoint boundary can yield `end <= start` when a
        // single codepoint is wider than `max_chars`. Bump forward to the
        // next codepoint so we always make progress.
        let safe_end = if end <= start {
            ceil_char_boundary(&doc.text, start.saturating_add(1)).min(total)
        } else {
            end
        };

        let slice = &doc.text[start..safe_end];
        chunks.push(Chunk {
            document_id: doc.id.clone(),
            ordinal,
            text: slice.to_string(),
            span: ChunkSpan {
                start,
                end: safe_end,
            },
        });
        ordinal = ordinal.saturating_add(1);

        if safe_end >= total {
            break;
        }

        let next_start = start.saturating_add(stride);
        let aligned = floor_char_boundary(&doc.text, next_start);
        start = if aligned <= start {
            ceil_char_boundary(&doc.text, start.saturating_add(1))
        } else {
            aligned
        };
    }

    chunks
}

/// Round `idx` down to the nearest UTF-8 codepoint boundary in `s`.
fn floor_char_boundary(s: &str, idx: usize) -> usize {
    let mut i = idx.min(s.len());
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

/// Round `idx` up to the nearest UTF-8 codepoint boundary in `s`.
fn ceil_char_boundary(s: &str, idx: usize) -> usize {
    let len = s.len();
    let mut i = idx.min(len);
    while i < len && !s.is_char_boundary(i) {
        i += 1;
    }
    i
}

/// In-memory RAG index.
///
/// This is the today-surface for tests, the orchestrator, and
/// `stratum-tools`. Phase 4+ swaps the backing store for `LanceDB` or
/// sqlite-vss; the public method set is intended to stay stable across
/// that migration.
#[derive(Debug, Default)]
pub struct RagIndex {
    documents: HashMap<DocumentId, RagDocument>,
    chunks: Vec<Chunk>,
    plan: ChunkPlan,
}

impl RagIndex {
    /// Construct an empty index using [`ChunkPlan::default`].
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct an empty index with a custom chunk plan.
    #[must_use]
    pub fn with_plan(plan: ChunkPlan) -> Self {
        Self {
            documents: HashMap::new(),
            chunks: Vec::new(),
            plan,
        }
    }

    /// Insert (or replace) a document, re-chunking it under the index's
    /// current plan. Returns the document's id for caller convenience.
    pub fn insert(&mut self, doc: RagDocument) -> DocumentId {
        let id = doc.id.clone();
        self.drop_chunks_for(&id);
        let new_chunks = chunk_document(&self.plan, &doc);
        self.chunks.extend(new_chunks);
        self.documents.insert(id.clone(), doc);
        id
    }

    /// Remove a document and all of its chunks. Returns whether the document
    /// existed prior to the call.
    pub fn remove(&mut self, id: &DocumentId) -> bool {
        let existed = self.documents.remove(id).is_some();
        self.drop_chunks_for(id);
        existed
    }

    /// Borrow a document by id.
    #[must_use]
    pub fn get(&self, id: &DocumentId) -> Option<&RagDocument> {
        self.documents.get(id)
    }

    /// Return all chunks for `id` in ascending `ordinal` order.
    #[must_use]
    pub fn chunks_for(&self, id: &DocumentId) -> Vec<&Chunk> {
        let mut out: Vec<&Chunk> = self
            .chunks
            .iter()
            .filter(|c| &c.document_id == id)
            .collect();
        out.sort_by_key(|c| c.ordinal);
        out
    }

    /// Substring-only search across all chunks. Returns up to `limit`
    /// matches in insertion order. An empty `needle` matches every chunk,
    /// which is the documented behavior callers can lean on for "give me
    /// everything you have" enumeration without a separate API.
    ///
    /// Real embedding retrieval lands in Phase 4+.
    #[must_use]
    pub fn search(&self, needle: &str, limit: usize) -> Vec<&Chunk> {
        if limit == 0 {
            return Vec::new();
        }
        self.chunks
            .iter()
            .filter(|c| needle.is_empty() || c.text.contains(needle))
            .take(limit)
            .collect()
    }

    /// Number of documents in the index.
    #[must_use]
    pub fn len(&self) -> usize {
        self.documents.len()
    }

    /// True iff no documents are in the index.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.documents.is_empty()
    }

    /// Iterate over `(id, document)` pairs. Iteration order is the
    /// `HashMap` order — callers that need a deterministic order should
    /// sort the result.
    pub fn iter(&self) -> impl Iterator<Item = (&DocumentId, &RagDocument)> {
        self.documents.iter()
    }

    fn drop_chunks_for(&mut self, id: &DocumentId) {
        self.chunks.retain(|c| &c.document_id != id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn doc(id: &str, text: &str) -> RagDocument {
        RagDocument {
            id: DocumentId::new(id),
            source_path: None,
            text: text.to_string(),
            mime: "text/plain".to_string(),
            indexed_at: "2026-06-05T00:00:00Z".to_string(),
        }
    }

    #[test]
    fn chunk_plan_default_is_512_64() {
        let p = ChunkPlan::default();
        assert_eq!(p.max_chars, 512);
        assert_eq!(p.overlap_chars, 64);
    }

    #[test]
    fn chunk_document_short_text_one_chunk() {
        let plan = ChunkPlan::default();
        let d = doc("a", "hello world");
        let chunks = chunk_document(&plan, &d);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].ordinal, 0);
        assert_eq!(chunks[0].text, "hello world");
        assert_eq!(chunks[0].span, ChunkSpan { start: 0, end: 11 });
        assert_eq!(chunks[0].document_id, DocumentId::new("a"));
    }

    #[test]
    fn chunk_document_long_text_overlaps_and_rejoins() {
        let plan = ChunkPlan {
            max_chars: 10,
            overlap_chars: 3,
        };
        let text: String = (0u8..50).map(|i| (b'a' + (i % 26)) as char).collect();
        let d = doc("long", &text);
        let chunks = chunk_document(&plan, &d);
        assert!(chunks.len() >= 2);

        // Stride = 10 - 3 = 7. Spans should advance by 7 bytes until the tail.
        for pair in chunks.windows(2) {
            let prev = &pair[0];
            let next = &pair[1];
            assert_eq!(next.span.start, prev.span.start + 7);
            // Adjacent full-width chunks must overlap by exactly 3 bytes.
            if next.span.end - next.span.start == plan.max_chars {
                assert_eq!(prev.span.end - next.span.start, 3);
            }
        }

        // Joining chunk[0] + chunk[i][overlap..] for i>0 reproduces the
        // original document text.
        let mut joined = String::new();
        for (i, c) in chunks.iter().enumerate() {
            if i == 0 {
                joined.push_str(&c.text);
            } else {
                let prev_end = chunks[i - 1].span.end;
                if c.span.end > prev_end {
                    let tail_start = prev_end - c.span.start;
                    joined.push_str(&c.text[tail_start..]);
                }
            }
        }
        assert_eq!(joined, text);
    }

    #[test]
    fn chunk_document_utf8_safe() {
        // Multi-byte codepoints sized so that naive byte-window splits
        // would land mid-codepoint.
        let text = "héllo wörld 🌍 done";
        let plan = ChunkPlan {
            max_chars: 4,
            overlap_chars: 1,
        };
        let d = doc("u", text);
        let chunks = chunk_document(&plan, &d);
        assert!(!chunks.is_empty());
        for c in &chunks {
            assert!(text.is_char_boundary(c.span.start));
            assert!(text.is_char_boundary(c.span.end));
            assert_eq!(&text[c.span.start..c.span.end], c.text.as_str());
        }
        // Last chunk reaches the document end.
        let last = chunks.last().expect("at least one chunk");
        assert_eq!(last.span.end, text.len());
    }

    #[test]
    fn chunk_document_empty_text_yields_no_chunks() {
        let plan = ChunkPlan::default();
        let d = doc("e", "");
        assert!(chunk_document(&plan, &d).is_empty());
    }

    #[test]
    fn chunk_document_zero_max_yields_no_chunks() {
        let plan = ChunkPlan {
            max_chars: 0,
            overlap_chars: 0,
        };
        let d = doc("z", "not empty");
        assert!(chunk_document(&plan, &d).is_empty());
    }

    #[test]
    fn chunk_document_codepoint_wider_than_max_chars() {
        // 🌍 is 4 bytes — wider than max_chars=2. The slider must still
        // emit it as a single chunk and reach the end.
        let plan = ChunkPlan {
            max_chars: 2,
            overlap_chars: 0,
        };
        let text = "🌍ab";
        let d = doc("w", text);
        let chunks = chunk_document(&plan, &d);
        assert!(!chunks.is_empty());
        // First chunk holds the wide codepoint in its entirety.
        assert_eq!(chunks[0].text, "🌍");
        assert_eq!(chunks[0].span, ChunkSpan { start: 0, end: 4 });
        let last = chunks.last().expect("at least one chunk");
        assert_eq!(last.span.end, text.len());
    }

    #[test]
    fn chunk_document_overlap_geq_max_makes_progress() {
        // Stride would be zero — implementation must clamp it to 1.
        let plan = ChunkPlan {
            max_chars: 4,
            overlap_chars: 8,
        };
        let d = doc("p", "abcdefghij");
        let chunks = chunk_document(&plan, &d);
        assert!(!chunks.is_empty());
        let last = chunks.last().expect("at least one chunk");
        assert_eq!(last.span.end, d.text.len());
    }

    #[test]
    fn index_insert_registers_doc_and_chunks() {
        let mut idx = RagIndex::with_plan(ChunkPlan {
            max_chars: 5,
            overlap_chars: 1,
        });
        let id = idx.insert(doc("a", "hello world here"));
        assert_eq!(id, DocumentId::new("a"));
        assert!(idx.get(&id).is_some());
        let chunks = idx.chunks_for(&id);
        assert!(!chunks.is_empty());
        assert_eq!(chunks[0].ordinal, 0);
    }

    #[test]
    fn index_insert_overwrites_same_id() {
        let mut idx = RagIndex::with_plan(ChunkPlan {
            max_chars: 4,
            overlap_chars: 0,
        });
        idx.insert(doc("a", "first text"));
        let first_chunks = idx.chunks_for(&DocumentId::new("a")).len();
        idx.insert(doc("a", "x"));
        let second_chunks = idx.chunks_for(&DocumentId::new("a"));
        assert_eq!(idx.len(), 1);
        assert_ne!(first_chunks, second_chunks.len());
        assert_eq!(second_chunks.len(), 1);
        assert_eq!(second_chunks[0].text, "x");
        let stored = idx.get(&DocumentId::new("a")).expect("doc present");
        assert_eq!(stored.text, "x");
    }

    #[test]
    fn index_remove_deletes_doc_and_chunks() {
        let mut idx = RagIndex::new();
        let id = idx.insert(doc("a", "hello"));
        assert!(idx.remove(&id));
        assert!(!idx.remove(&id));
        assert!(idx.get(&id).is_none());
        assert!(idx.chunks_for(&id).is_empty());
    }

    #[test]
    fn index_get_returns_none_after_removal() {
        let mut idx = RagIndex::new();
        let id = idx.insert(doc("a", "hello"));
        assert!(idx.get(&id).is_some());
        idx.remove(&id);
        assert!(idx.get(&id).is_none());
    }

    #[test]
    fn index_chunks_for_is_ordinal_sorted() {
        let mut idx = RagIndex::with_plan(ChunkPlan {
            max_chars: 3,
            overlap_chars: 1,
        });
        let id = idx.insert(doc("a", "abcdefghij"));
        let chunks = idx.chunks_for(&id);
        assert!(chunks.len() > 1);
        let ordinals: Vec<u32> = chunks.iter().map(|c| c.ordinal).collect();
        let mut sorted = ordinals.clone();
        sorted.sort_unstable();
        assert_eq!(ordinals, sorted);
        assert_eq!(ordinals[0], 0);
    }

    #[test]
    fn index_search_matches_substring_and_empty_needle_returns_all() {
        let mut idx = RagIndex::with_plan(ChunkPlan {
            max_chars: 50,
            overlap_chars: 0,
        });
        idx.insert(doc("a", "the quick brown fox"));
        idx.insert(doc("b", "jumps over the lazy dog"));

        let hits = idx.search("quick", 10);
        assert_eq!(hits.len(), 1);
        assert!(hits[0].text.contains("quick"));

        let all = idx.search("", 10);
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn index_search_respects_limit() {
        let mut idx = RagIndex::with_plan(ChunkPlan {
            max_chars: 4,
            overlap_chars: 0,
        });
        idx.insert(doc("a", "aaaaaaaaaaaaaaaa"));
        let chunks_total = idx.chunks_for(&DocumentId::new("a")).len();
        assert!(chunks_total >= 3);

        let limited = idx.search("a", 2);
        assert_eq!(limited.len(), 2);

        let zero = idx.search("a", 0);
        assert!(zero.is_empty());
    }

    #[test]
    fn index_len_and_is_empty_reflect_doc_count() {
        let mut idx = RagIndex::new();
        assert!(idx.is_empty());
        assert_eq!(idx.len(), 0);
        idx.insert(doc("a", "hi"));
        idx.insert(doc("b", "yo"));
        assert!(!idx.is_empty());
        assert_eq!(idx.len(), 2);
        idx.remove(&DocumentId::new("a"));
        assert_eq!(idx.len(), 1);
    }

    #[test]
    fn index_iter_yields_all_docs_sorted_by_id() {
        let mut index = RagIndex::new();
        index.insert(doc("c", "third"));
        index.insert(doc("a", "first"));
        index.insert(doc("b", "second"));

        let mut ids: Vec<&DocumentId> = index.iter().map(|(doc_id, _)| doc_id).collect();
        ids.sort_by(|x, y| x.as_str().cmp(y.as_str()));
        assert_eq!(
            ids,
            vec![
                &DocumentId::new("a"),
                &DocumentId::new("b"),
                &DocumentId::new("c"),
            ]
        );
    }

    #[test]
    fn rag_document_serde_roundtrip() {
        let d = RagDocument {
            id: DocumentId::new("a"),
            source_path: Some(PathBuf::from("/tmp/x.md")),
            text: "hi".to_string(),
            mime: "text/markdown".to_string(),
            indexed_at: "2026-06-05T00:00:00Z".to_string(),
        };
        let json = serde_json::to_string(&d).expect("serialize doc");
        let back: RagDocument = serde_json::from_str(&json).expect("deserialize doc");
        assert_eq!(d, back);
    }

    #[test]
    fn chunk_serde_roundtrip() {
        let c = Chunk {
            document_id: DocumentId::new("a"),
            ordinal: 7,
            text: "hello".to_string(),
            span: ChunkSpan { start: 0, end: 5 },
        };
        let json = serde_json::to_string(&c).expect("serialize chunk");
        let back: Chunk = serde_json::from_str(&json).expect("deserialize chunk");
        assert_eq!(c, back);
    }

    #[test]
    fn chunk_plan_serde_roundtrip() {
        let p = ChunkPlan::default();
        let json = serde_json::to_string(&p).expect("serialize plan");
        let back: ChunkPlan = serde_json::from_str(&json).expect("deserialize plan");
        assert_eq!(p, back);
    }

    #[test]
    fn document_id_as_str_matches_inner() {
        let id = DocumentId::new("abc");
        assert_eq!(id.as_str(), "abc");
    }
}
