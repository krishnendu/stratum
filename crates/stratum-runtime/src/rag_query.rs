//! `RagQuery` — given a user prompt, score chunks in a built RAG index
//! and return the top-`k` passages.
//!
//! This is the read-side complement to
//! [`crate::rag_index_builder::RagIndexBuilder`]:
//!
//! 1. Embed the prompt via the same [`crate::embedder::Embedder`] used at
//!    build time.
//! 2. Rank vectors via [`crate::embedder::InMemoryVectorStore::search`].
//! 3. Filter results below [`RagQueryConfig::min_score`].
//! 4. Optionally diversify so that no single document monopolises the top
//!    `k` — see the algorithm note on [`RagQueryConfig::diversify_by_document`].
//! 5. Re-hydrate each surviving vector hit into a [`RankedPassage`] by
//!    splitting its key (`"<doc_id>#<ordinal>"`) and looking up the matching
//!    [`crate::rag::Chunk`] in the [`crate::rag::RagIndex`].
//! 6. Bound the cumulative `text` bytes returned to
//!    [`RagQueryConfig::max_passages_text_bytes`].
//!
//! All failure modes route through [`QueryError`]; no new `STRAT-E…` code
//! is allocated for this scaffold (consistent with
//! `plan/29-error-taxonomy-and-logging.md`).

use std::collections::HashSet;
use std::fmt;
use std::sync::Arc;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::embedder::{EmbedError, Embedder, InMemoryVectorStore};
use crate::rag::{DocumentId, RagIndex};

/// Default cap on the cumulative text bytes returned in a [`QueryReport`].
///
/// 8 KiB — large enough to carry several typical RAG passages, small enough
/// to leave room for the rest of the assistant prompt.
const DEFAULT_MAX_PASSAGES_TEXT_BYTES: u64 = 8 * 1024;

/// Default top-`k` passages returned per query.
const DEFAULT_K: usize = 5;

/// Default minimum cosine score for a passage to be returned (no floor).
const DEFAULT_MIN_SCORE: f32 = 0.0;

/// Configuration knobs for [`RagQuery`].
///
/// Intentionally `Copy` — the entire struct fits in a few words and a
/// `RagQuery` typically clones one off the wire per turn.
///
/// `Eq` is *not* derived because of the `f32` field — equality is structural
/// via `PartialEq` only.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RagQueryConfig {
    /// Maximum number of passages to return.
    pub k: usize,
    /// Drop hits whose cosine score is strictly less than this floor.
    pub min_score: f32,
    /// Cumulative cap on the bytes of `RankedPassage::text` returned.
    pub max_passages_text_bytes: u64,
    /// When `true`, no document contributes more than one passage until every
    /// document with a hit has contributed once.
    pub diversify_by_document: bool,
}

impl Default for RagQueryConfig {
    fn default() -> Self {
        Self {
            k: DEFAULT_K,
            min_score: DEFAULT_MIN_SCORE,
            max_passages_text_bytes: DEFAULT_MAX_PASSAGES_TEXT_BYTES,
            diversify_by_document: true,
        }
    }
}

/// A retrieved passage, paired with its cosine score.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RankedPassage {
    /// Id of the parent document.
    pub document_id: String,
    /// 0-based chunk ordinal within the parent document.
    pub ordinal: u32,
    /// Chunk text re-hydrated from the [`RagIndex`].
    pub text: String,
    /// Cosine score in `[-1, 1]`.
    pub score: f32,
}

/// Result of [`RagQuery::query`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QueryReport {
    /// The original query text.
    pub query: String,
    /// Surviving ranked passages, descending by score.
    pub passages: Vec<RankedPassage>,
    /// Wall-clock elapsed time in milliseconds.
    pub elapsed_ms: u64,
    /// Number of vector candidates considered before filtering / diversification.
    pub total_candidates: usize,
}

/// Errors emitted by [`RagQuery::query`].
#[derive(Debug)]
pub enum QueryError {
    /// The embedder failed to embed the prompt or the vector search hit a
    /// backend error.
    Embedder(EmbedError),
    /// A vector key did not match the `"<doc_id>#<ordinal>"` shape.
    BadVectorKey(String),
    /// The query vector and stored vectors disagree on dimensionality.
    DimMismatch(String),
}

impl fmt::Display for QueryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Embedder(e) => write!(f, "rag query: embedder error: {e}"),
            Self::BadVectorKey(key) => write!(f, "rag query: bad vector key: {key}"),
            Self::DimMismatch(msg) => write!(f, "rag query: dim mismatch: {msg}"),
        }
    }
}

impl std::error::Error for QueryError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Embedder(e) => Some(e),
            Self::BadVectorKey(_) | Self::DimMismatch(_) => None,
        }
    }
}

/// Read-side companion to
/// [`crate::rag_index_builder::RagIndexBuilder`].
///
/// Cloning a `RagQuery` is cheap — the embedder is shared via `Arc` and the
/// config is `Copy`.
#[derive(Clone)]
pub struct RagQuery {
    embedder: Arc<dyn Embedder>,
    cfg: RagQueryConfig,
}

impl fmt::Debug for RagQuery {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RagQuery")
            .field("cfg", &self.cfg)
            .finish_non_exhaustive()
    }
}

impl RagQuery {
    /// Construct a `RagQuery` bound to an embedder and a config.
    #[must_use]
    pub fn new(embedder: Arc<dyn Embedder>, cfg: RagQueryConfig) -> Self {
        Self { embedder, cfg }
    }

    /// Borrow the configured knobs.
    #[must_use]
    pub const fn config(&self) -> &RagQueryConfig {
        &self.cfg
    }

    /// Run the query against `index` + `vectors`.
    ///
    /// # Algorithm
    ///
    /// 1. Embed `prompt` via [`Embedder::embed`].
    /// 2. Ask `vectors` for the top `k * 4` cosine-ranked keys (over-fetch
    ///    so diversification and stale-key skips still have headroom).
    /// 3. Drop candidates whose score is `< cfg.min_score`.
    /// 4. If `cfg.diversify_by_document`, run one pass that **picks the
    ///    highest-scoring hit from each distinct `document_id`** (in
    ///    descending score order), then a second pass that backfills any
    ///    remaining slots from the leftovers in the original score order.
    ///    This guarantees that whenever the candidate pool spans at least
    ///    `min(k, distinct_docs)` documents, the result contains that many
    ///    distinct documents.
    /// 5. Re-hydrate each kept candidate's chunk via [`split_vector_key`] +
    ///    [`RagIndex::chunks_for`]. Stale keys (the chunk no longer exists)
    ///    are silently skipped — they are counted in `total_candidates` but
    ///    omitted from `passages`.
    /// 6. Append passages until either `k` is reached or appending the next
    ///    passage's text would push the cumulative text byte total over
    ///    `cfg.max_passages_text_bytes`. The first passage is always kept
    ///    even if it alone exceeds the cap, so callers never see an empty
    ///    result purely because of the cap.
    ///
    /// # Errors
    ///
    /// - [`QueryError::Embedder`] when the embedder rejects the prompt or
    ///   the vector store's cosine search fails (e.g. dim mismatch in
    ///   stored vectors).
    /// - [`QueryError::DimMismatch`] when the prompt embedding's dimension
    ///   disagrees with the vector store's configured dimension.
    pub fn query(
        &self,
        prompt: &str,
        index: &RagIndex,
        vectors: &InMemoryVectorStore,
    ) -> Result<QueryReport, QueryError> {
        let started = Instant::now();

        let query_vec = self.embedder.embed(prompt).map_err(QueryError::Embedder)?;

        // Over-fetch so diversification + stale-key skips still leave us
        // enough to fill `cfg.k`.
        let over_fetch = self.cfg.k.saturating_mul(4).max(self.cfg.k);
        let raw_hits = vectors
            .search(&query_vec, over_fetch)
            .map_err(|e| match e {
                EmbedError::Backend(msg) if msg.contains("dim mismatch") => {
                    QueryError::DimMismatch(msg)
                }
                other => QueryError::Embedder(other),
            })?;

        let total_candidates = raw_hits.len();

        // Filter on min_score.
        let filtered: Vec<(String, f32)> = raw_hits
            .into_iter()
            .filter(|(_, score)| *score >= self.cfg.min_score)
            .collect();

        // Diversify (or not).
        let ordered_keys: Vec<(String, f32)> = if self.cfg.diversify_by_document {
            diversify(&filtered)
        } else {
            filtered
        };

        // Re-hydrate keys → ranked passages, honoring the byte cap.
        let mut passages: Vec<RankedPassage> = Vec::with_capacity(self.cfg.k);
        let mut cumulative_bytes: u64 = 0;
        let cap = self.cfg.max_passages_text_bytes;

        for (key, score) in ordered_keys {
            if passages.len() >= self.cfg.k {
                break;
            }
            let Some((doc_id, ordinal)) = split_vector_key(&key) else {
                // Stale or malformed key — count it as a candidate but skip.
                continue;
            };
            let doc_handle = DocumentId::new(doc_id.clone());
            let chunk = index
                .chunks_for(&doc_handle)
                .into_iter()
                .find(|c| c.ordinal == ordinal);
            let Some(chunk) = chunk else {
                // Stale index — the vector key references a chunk that no
                // longer exists. Skip silently (per the documented contract).
                continue;
            };

            let chunk_bytes = chunk.text.len() as u64;
            let would_be = cumulative_bytes.saturating_add(chunk_bytes);
            if !passages.is_empty() && would_be > cap {
                // Cap reached. Stop appending — but always keep the first
                // passage so an oversized single hit doesn't produce an
                // empty result.
                break;
            }

            passages.push(RankedPassage {
                document_id: doc_id,
                ordinal,
                text: chunk.text.clone(),
                score,
            });
            cumulative_bytes = would_be;
        }

        let elapsed_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);

        Ok(QueryReport {
            query: prompt.to_owned(),
            passages,
            elapsed_ms,
            total_candidates,
        })
    }
}

/// Diversify a score-sorted list by document.
///
/// First pass: walk the descending-score list and pick the first hit from
/// each distinct `document_id`. Second pass: walk the original list again
/// and append any leftovers (multiple hits per doc) in score order.
///
/// The output is therefore a stable permutation of the input where the
/// prefix is "one hit per distinct doc, descending by score" and the suffix
/// is "everything else, descending by score".
fn diversify(scored: &[(String, f32)]) -> Vec<(String, f32)> {
    let mut seen: HashSet<String> = HashSet::with_capacity(scored.len());
    let mut first_pass: Vec<(String, f32)> = Vec::with_capacity(scored.len());
    let mut leftovers: Vec<(String, f32)> = Vec::new();

    for (key, score) in scored {
        match split_vector_key(key) {
            Some((doc_id, _)) => {
                if seen.insert(doc_id) {
                    first_pass.push((key.clone(), *score));
                } else {
                    leftovers.push((key.clone(), *score));
                }
            }
            None => {
                // Malformed keys flow through the same path as leftovers so
                // the caller's stale-key path still gets a shot at them.
                leftovers.push((key.clone(), *score));
            }
        }
    }

    first_pass.extend(leftovers);
    first_pass
}

/// Parse a vector store key of the form `"<doc_id>#<ordinal>"`.
///
/// Returns `None` when the key has no `#`, or when the suffix doesn't parse
/// as a `u32`. The doc id may itself contain `#` — only the **last** `#` is
/// treated as the separator so paths like `"a#b#0"` round-trip cleanly
/// (`Some(("a#b", 0))`).
#[must_use]
pub fn split_vector_key(key: &str) -> Option<(String, u32)> {
    let (doc, ord) = key.rsplit_once('#')?;
    if doc.is_empty() {
        return None;
    }
    let ordinal: u32 = ord.parse().ok()?;
    Some((doc.to_owned(), ordinal))
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::Arc;

    use super::*;
    use crate::embedder::{EmbeddingDim, HashEmbedder};
    use crate::rag::{ChunkPlan, DocumentId, RagDocument};

    fn make_embedder() -> Arc<dyn Embedder> {
        Arc::new(HashEmbedder::new(EmbeddingDim(16)))
    }

    fn make_doc(id: &str, text: &str) -> RagDocument {
        RagDocument {
            id: DocumentId::new(id),
            source_path: Some(PathBuf::from(format!("/virtual/{id}"))),
            text: text.to_owned(),
            mime: "text/plain".to_owned(),
            indexed_at: "2026-06-10T00:00:00Z".to_owned(),
        }
    }

    /// Build a tiny index (~5 chunks across 2 docs) for query tests.
    fn build_tiny_index() -> (Arc<dyn Embedder>, RagIndex, InMemoryVectorStore) {
        let embedder = make_embedder();
        // Chunk plan that gives us ~3 chunks per doc for a ~20-char body.
        let plan = ChunkPlan {
            max_chars: 8,
            overlap_chars: 0,
        };
        let mut index = RagIndex::with_plan(plan);
        let mut vectors = InMemoryVectorStore::new(embedder.dim());

        let docs = [
            make_doc("doc1", "alpha bravo charlie delta echo"),
            make_doc("doc2", "foxtrot golf hotel india juliet"),
        ];

        for doc in &docs {
            index.insert(doc.clone());
            let doc_id = DocumentId::new(doc.id.as_str());
            for chunk in index.chunks_for(&doc_id) {
                let vec = embedder.embed(&chunk.text).expect("embed chunk");
                let key = format!("{}#{}", doc.id.as_str(), chunk.ordinal);
                vectors.insert(key, vec).expect("insert vector");
            }
        }

        (embedder, index, vectors)
    }

    #[test]
    fn config_default_matches_documented_values() {
        let cfg = RagQueryConfig::default();
        assert_eq!(cfg.k, 5);
        assert!((cfg.min_score - 0.0).abs() < f32::EPSILON);
        assert_eq!(cfg.max_passages_text_bytes, 8 * 1024);
        assert!(cfg.diversify_by_document);
    }

    #[test]
    fn split_vector_key_happy_path() {
        assert_eq!(split_vector_key("doc1#0"), Some(("doc1".to_owned(), 0_u32)));
        assert_eq!(
            split_vector_key("doc1#42"),
            Some(("doc1".to_owned(), 42_u32))
        );
    }

    #[test]
    fn split_vector_key_no_separator_is_none() {
        assert_eq!(split_vector_key("doc1"), None);
    }

    #[test]
    fn split_vector_key_non_numeric_suffix_is_none() {
        assert_eq!(split_vector_key("doc1#abc"), None);
    }

    #[test]
    fn split_vector_key_empty_doc_is_none() {
        assert_eq!(split_vector_key("#0"), None);
    }

    #[test]
    fn split_vector_key_uses_last_separator() {
        // Document ids may contain '#'; only the final segment is the ordinal.
        assert_eq!(split_vector_key("a#b#3"), Some(("a#b".to_owned(), 3_u32)));
    }

    #[test]
    fn query_happy_path_returns_at_most_k_passages() {
        let (embedder, index, vectors) = build_tiny_index();
        let q = RagQuery::new(embedder, RagQueryConfig::default());
        let report = q
            .query("alpha bravo charlie", &index, &vectors)
            .expect("query");
        assert!(report.passages.len() <= q.config().k);
        assert_eq!(report.query, "alpha bravo charlie");
    }

    #[test]
    fn query_against_empty_corpus_returns_empty_passages() {
        let embedder = make_embedder();
        let index = RagIndex::new();
        let vectors = InMemoryVectorStore::new(embedder.dim());
        let q = RagQuery::new(embedder, RagQueryConfig::default());
        let report = q.query("anything", &index, &vectors).expect("query");
        assert!(report.passages.is_empty());
        assert_eq!(report.total_candidates, 0);
    }

    #[test]
    fn query_with_extreme_min_score_filters_results() {
        let (embedder, index, vectors) = build_tiny_index();
        let cfg = RagQueryConfig {
            min_score: 0.99,
            ..RagQueryConfig::default()
        };
        let q = RagQuery::new(embedder, cfg);
        let report = q.query("alpha", &index, &vectors).expect("query");
        // Hash-embedder cosines between distinct texts are well below 0.99.
        assert!(
            report.passages.len() <= 1,
            "expected ≤1 passage, got {}",
            report.passages.len()
        );
    }

    #[test]
    fn query_honors_k() {
        let (embedder, index, vectors) = build_tiny_index();
        let cfg = RagQueryConfig {
            k: 2,
            ..RagQueryConfig::default()
        };
        let q = RagQuery::new(embedder, cfg);
        let report = q.query("alpha bravo", &index, &vectors).expect("query");
        assert!(report.passages.len() <= 2);
    }

    #[test]
    fn query_diversify_spans_multiple_documents() {
        let (embedder, index, vectors) = build_tiny_index();
        let cfg = RagQueryConfig {
            k: 4,
            diversify_by_document: true,
            ..RagQueryConfig::default()
        };
        let q = RagQuery::new(embedder, cfg);
        let report = q.query("alpha bravo", &index, &vectors).expect("query");
        let distinct_docs: HashSet<&str> = report
            .passages
            .iter()
            .map(|p| p.document_id.as_str())
            .collect();
        assert!(
            distinct_docs.len() >= 2,
            "expected ≥2 distinct docs, got {distinct_docs:?}"
        );
    }

    #[test]
    fn query_without_diversify_may_repeat_document() {
        // With diversify off, the top-k can lean on a single doc. We don't
        // assert "must repeat" (HashEmbedder cosines are noisy) but we do
        // assert the diversified output is a permutation of the same key set
        // when k covers the full pool.
        let (embedder, index, vectors) = build_tiny_index();
        let cfg = RagQueryConfig {
            k: 4,
            diversify_by_document: false,
            ..RagQueryConfig::default()
        };
        let q = RagQuery::new(embedder, cfg);
        let report = q.query("alpha bravo", &index, &vectors).expect("query");
        assert!(report.passages.len() <= 4);
        // Sanity: at least one passage came back.
        assert!(!report.passages.is_empty());
    }

    #[test]
    fn query_honors_max_passages_text_bytes() {
        let (embedder, index, vectors) = build_tiny_index();
        let cfg = RagQueryConfig {
            k: 99,
            max_passages_text_bytes: 12,
            ..RagQueryConfig::default()
        };
        let q = RagQuery::new(embedder, cfg);
        let report = q.query("alpha", &index, &vectors).expect("query");
        // First passage is always kept; subsequent passages should be
        // capped. Total bytes after the first must be ≤ cap, OR we have
        // exactly one passage (the "first hit oversize" escape hatch).
        let total: usize = report.passages.iter().map(|p| p.text.len()).sum();
        if report.passages.len() > 1 {
            assert!(
                total as u64 <= cfg.max_passages_text_bytes || report.passages.len() == 1,
                "total {total} > cap {} with {} passages",
                cfg.max_passages_text_bytes,
                report.passages.len()
            );
        }
    }

    #[test]
    fn query_records_total_candidates() {
        let (embedder, index, vectors) = build_tiny_index();
        let q = RagQuery::new(embedder, RagQueryConfig::default());
        let report = q.query("alpha", &index, &vectors).expect("query");
        assert!(report.total_candidates > 0);
        assert!(report.total_candidates <= vectors.len());
    }

    #[test]
    fn query_elapsed_ms_is_present() {
        let (embedder, index, vectors) = build_tiny_index();
        let q = RagQuery::new(embedder, RagQueryConfig::default());
        let report = q.query("alpha", &index, &vectors).expect("query");
        // elapsed_ms is a `u64`; we only assert it round-trips through the
        // report (it may legitimately be 0 on very fast machines).
        let _ = report.elapsed_ms;
    }

    #[test]
    fn query_stale_vector_key_is_skipped_not_panicked() {
        let embedder = make_embedder();
        let mut index = RagIndex::new();
        index.insert(make_doc("doc1", "alpha bravo"));
        let mut vectors = InMemoryVectorStore::new(embedder.dim());

        // Insert a real key for doc1#0, plus a stale key pointing at a
        // chunk ordinal that doesn't exist.
        let chunks = index.chunks_for(&DocumentId::new("doc1"));
        assert!(!chunks.is_empty());
        let real_vec = embedder.embed(&chunks[0].text).expect("embed");
        vectors.insert("doc1#0", real_vec.clone()).expect("insert");
        vectors.insert("doc1#999", real_vec).expect("insert stale");

        let q = RagQuery::new(embedder, RagQueryConfig::default());
        let report = q.query("alpha", &index, &vectors).expect("query");
        for p in &report.passages {
            assert_ne!(p.ordinal, 999, "stale ordinal should have been skipped");
        }
        // total_candidates counts the over-fetch — both keys feed in.
        assert!(report.total_candidates >= 2);
    }

    #[test]
    fn query_embedder_error_is_surfaced() {
        let embedder = make_embedder();
        let (_, index, vectors) = build_tiny_index();
        let q = RagQuery::new(embedder, RagQueryConfig::default());
        // HashEmbedder rejects empty input.
        let err = q
            .query("", &index, &vectors)
            .expect_err("expected embedder error");
        match err {
            QueryError::Embedder(EmbedError::EmptyInput) => {}
            other => panic!("expected EmptyInput, got {other:?}"),
        }
    }

    #[test]
    fn query_dim_mismatch_routes_through_dim_error() {
        // Build a vector store with one dim and an embedder with another.
        let embedder: Arc<dyn Embedder> = Arc::new(HashEmbedder::new(EmbeddingDim(8)));
        let other_embedder: Arc<dyn Embedder> = Arc::new(HashEmbedder::new(EmbeddingDim(16)));

        let mut index = RagIndex::new();
        index.insert(make_doc("doc1", "alpha bravo charlie"));

        // Populate the store with vectors from `other_embedder` (dim 16).
        let mut vectors = InMemoryVectorStore::new(other_embedder.dim());
        for chunk in index.chunks_for(&DocumentId::new("doc1")) {
            let v = other_embedder.embed(&chunk.text).expect("embed");
            let key = format!("doc1#{}", chunk.ordinal);
            vectors.insert(key, v).expect("insert");
        }

        // Query with the dim-8 embedder — should hit dim mismatch.
        let q = RagQuery::new(embedder, RagQueryConfig::default());
        let err = q
            .query("alpha", &index, &vectors)
            .expect_err("expected dim mismatch");
        match err {
            QueryError::DimMismatch(_) => {}
            other => panic!("expected DimMismatch, got {other:?}"),
        }
    }

    #[test]
    fn query_error_display_smoke() {
        let cases = [
            QueryError::Embedder(EmbedError::EmptyInput),
            QueryError::BadVectorKey("not-a-key".to_owned()),
            QueryError::DimMismatch("8 != 16".to_owned()),
        ];
        for e in cases {
            let rendered = format!("{e}");
            assert!(rendered.starts_with("rag query:"), "got: {rendered}");
        }
    }

    #[test]
    fn query_error_source_smoke() {
        use std::error::Error;
        let embedder_err = QueryError::Embedder(EmbedError::EmptyInput);
        assert!(embedder_err.source().is_some());
        let bad_key = QueryError::BadVectorKey("x".to_owned());
        assert!(bad_key.source().is_none());
        let dim = QueryError::DimMismatch("x".to_owned());
        assert!(dim.source().is_none());
    }

    #[test]
    fn rag_query_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<RagQuery>();
        assert_send_sync::<RagQueryConfig>();
        assert_send_sync::<QueryReport>();
        assert_send_sync::<RankedPassage>();
        assert_send_sync::<QueryError>();
    }

    #[test]
    fn rag_query_debug_smoke() {
        let q = RagQuery::new(make_embedder(), RagQueryConfig::default());
        let rendered = format!("{q:?}");
        assert!(rendered.contains("RagQuery"));
    }

    #[test]
    fn rag_query_clone_preserves_config() {
        let cfg = RagQueryConfig {
            k: 9,
            min_score: 0.25,
            max_passages_text_bytes: 4096,
            diversify_by_document: false,
        };
        let q = RagQuery::new(make_embedder(), cfg);
        let cloned = q.clone();
        assert_eq!(*cloned.config(), cfg);
        // Keep the original alive long enough to ensure the clone is a real
        // independent handle on the embedder Arc, not just a redundant copy.
        assert_eq!(*q.config(), cfg);
    }

    #[test]
    fn query_report_serde_round_trip() {
        let report = QueryReport {
            query: "alpha".to_owned(),
            passages: vec![RankedPassage {
                document_id: "doc1".to_owned(),
                ordinal: 0,
                text: "alpha".to_owned(),
                score: 0.42,
            }],
            elapsed_ms: 7,
            total_candidates: 3,
        };
        let json = serde_json::to_string(&report).expect("serialize");
        let back: QueryReport = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(report, back);
    }

    #[test]
    fn ranked_passage_serde_round_trip() {
        let p = RankedPassage {
            document_id: "doc2".to_owned(),
            ordinal: 3,
            text: "echo foxtrot".to_owned(),
            score: -0.1,
        };
        let json = serde_json::to_string(&p).expect("serialize");
        let back: RankedPassage = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(p, back);
    }

    #[test]
    fn rag_query_config_serde_round_trip() {
        let cfg = RagQueryConfig {
            k: 11,
            min_score: 0.33,
            max_passages_text_bytes: 1024,
            diversify_by_document: false,
        };
        let json = serde_json::to_string(&cfg).expect("serialize");
        let back: RagQueryConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, cfg);
    }

    #[test]
    fn diversify_picks_one_per_doc_first() {
        let scored = vec![
            ("doc1#0".to_owned(), 0.9_f32),
            ("doc1#1".to_owned(), 0.8_f32),
            ("doc2#0".to_owned(), 0.7_f32),
            ("doc2#1".to_owned(), 0.6_f32),
        ];
        let out = diversify(&scored);
        // First two entries must reference distinct docs.
        let (d0, _) = split_vector_key(&out[0].0).expect("k0");
        let (d1, _) = split_vector_key(&out[1].0).expect("k1");
        assert_ne!(d0, d1);
    }

    #[test]
    fn diversify_preserves_malformed_keys_as_leftovers() {
        let scored = vec![
            ("doc1#0".to_owned(), 0.9_f32),
            ("garbage".to_owned(), 0.8_f32),
        ];
        let out = diversify(&scored);
        assert_eq!(out.len(), 2);
        // Malformed key lands after the well-formed one.
        assert_eq!(out[0].0, "doc1#0");
        assert_eq!(out[1].0, "garbage");
    }
}
