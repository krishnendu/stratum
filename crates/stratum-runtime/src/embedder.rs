//! Embedder trait + in-memory stub.
//!
//! Phase 1 scaffold for the RAG embedding surface defined in
//! `plan/05-models-and-installer.md` §5 and `plan/14-rag.md`. The real ML
//! weights (Arctic-Embed-L / nomic-embed-text / etc.) land in Phase 4+; this
//! module provides the stable abstraction the orchestrator can target today.
//!
//! ## What's here
//!
//! - [`EmbeddingVector`] — newtype around `Vec<f32>` so the public surface
//!   isn't a structural type.
//! - [`EmbeddingDim`] — strongly typed embedding dimension; serializes
//!   transparently as a `u32` so on-disk indexes round-trip cleanly when the
//!   real backend lands.
//! - [`Embedder`] — trait every concrete backend implements. The
//!   `embed_batch` default impl makes single-row backends "batchable" for
//!   free; HF-style backends will override for real batching.
//! - [`HashEmbedder`] — deterministic SHA-256-based fake. Useful as a unit
//!   test fixture and as the "fallback" backend if no model is installed.
//! - [`cosine_similarity`] / [`top_k`] — utilities the orchestrator uses to
//!   rank chunks once they're embedded.
//! - [`InMemoryVectorStore`] — `BTreeMap`-backed store; the real one will be
//!   LanceDB / sqlite-vss, but the call sites can target this surface today.
//!
//! ## What's not here
//!
//! No real ML weights, no tokenizer, no on-disk index. The error enum
//! deliberately uses generic `Backend(String)` rather than introducing new
//! `STRAT-E…` codes — once the real backend lands it'll likely fold into the
//! existing model-install / probe codes.

use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// A dense embedding vector.
///
/// Newtype over `Vec<f32>` so callers depend on the named type rather than the
/// structural one. Cosine-similarity utilities in this module operate on
/// `&EmbeddingVector` directly.
///
/// `Eq` is intentionally *not* derived — `f32` doesn't implement `Eq` because
/// of `NaN`. Equality is structural via `PartialEq` only.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingVector(Vec<f32>);

impl EmbeddingVector {
    /// Wrap an owned `Vec<f32>`.
    #[must_use]
    pub const fn new(v: Vec<f32>) -> Self {
        Self(v)
    }

    /// Borrow the underlying values.
    #[must_use]
    pub fn as_slice(&self) -> &[f32] {
        &self.0
    }

    /// Number of dimensions.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.0.len()
    }

    /// `true` if the vector has zero dimensions.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// Embedding dimensionality, e.g. `EmbeddingDim(384)`.
///
/// Serializes transparently as a `u32` so on-disk indexes don't carry a
/// wrapper object.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct EmbeddingDim(pub u32);

/// Errors emitted by [`Embedder`] backends and the utility functions in this
/// module.
///
/// Deliberately *not* allocated to a new `STRAT-E…` code — Phase 1 keeps the
/// surface generic; the real backend will fold these into the model-install
/// and probe error codes when it lands.
#[derive(Debug)]
pub enum EmbedError {
    /// The input text was empty.
    EmptyInput,
    /// The configured backend model isn't installed or isn't loadable.
    ModelUnavailable(String),
    /// A backend-specific failure (dim mismatch, zero vector, IO, …).
    Backend(String),
}

impl fmt::Display for EmbedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput => f.write_str("embedder input was empty"),
            Self::ModelUnavailable(msg) => write!(f, "embedder model unavailable: {msg}"),
            Self::Backend(msg) => write!(f, "embedder backend error: {msg}"),
        }
    }
}

impl std::error::Error for EmbedError {}

/// Backend trait every embedder implements.
///
/// Implementations must be `Send + Sync` so a single embedder handle can be
/// shared across the worker pool. `embed_batch` has a default that fans out
/// to repeated `embed` calls — real backends will override it for true batched
/// inference.
pub trait Embedder: Send + Sync {
    /// Dimensionality of the vectors this backend produces.
    fn dim(&self) -> EmbeddingDim;

    /// Embed a single piece of text.
    ///
    /// # Errors
    /// Returns [`EmbedError::EmptyInput`] when `text` is empty and a backend
    /// error if the model fails to embed.
    fn embed(&self, text: &str) -> Result<EmbeddingVector, EmbedError>;

    /// Embed a batch of texts. Default impl falls through to repeated
    /// [`embed`](Self::embed); real backends will override.
    ///
    /// # Errors
    /// Returns the first per-row failure encountered.
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<EmbeddingVector>, EmbedError> {
        texts.iter().map(|t| self.embed(t)).collect()
    }
}

/// Deterministic SHA-256-based fake embedder.
///
/// Useful as a test fixture and as the "no model installed" fallback so the
/// orchestrator's retrieval code can be exercised before the real backend
/// lands. The output is L2-normalized so cosine on hash embeddings is
/// well-defined.
#[derive(Debug, Clone)]
pub struct HashEmbedder {
    dim: EmbeddingDim,
}

impl HashEmbedder {
    /// Construct a `HashEmbedder` producing vectors of size `dim.0`.
    #[must_use]
    pub const fn new(dim: EmbeddingDim) -> Self {
        Self { dim }
    }
}

impl Embedder for HashEmbedder {
    fn dim(&self) -> EmbeddingDim {
        self.dim
    }

    fn embed(&self, text: &str) -> Result<EmbeddingVector, EmbedError> {
        if text.is_empty() {
            return Err(EmbedError::EmptyInput);
        }

        let target_len = self.dim.0 as usize;
        let mut out: Vec<f32> = Vec::with_capacity(target_len);

        // Build a deterministic byte stream by repeatedly hashing
        // (`SHA-256(text || counter)`) and consuming four-byte chunks. This
        // keeps the output strictly a function of `text` (and `dim`) without
        // requiring an HKDF dependency.
        let mut counter: u32 = 0;
        while out.len() < target_len {
            let mut hasher = Sha256::new();
            hasher.update(text.as_bytes());
            hasher.update(counter.to_le_bytes());
            let digest = hasher.finalize();

            for chunk in digest.chunks_exact(4) {
                if out.len() == target_len {
                    break;
                }
                // `chunks_exact(4)` guarantees a 4-byte slice; the array
                // conversion is infallible here.
                let bytes: [u8; 4] = match chunk.try_into() {
                    Ok(b) => b,
                    Err(_) => continue,
                };
                let raw = u32::from_le_bytes(bytes);
                #[allow(
                    clippy::cast_precision_loss,
                    reason = "u32 → f32 for normalized [0,1) noise; precision loss is intentional"
                )]
                let scaled = raw as f32 / u32::MAX as f32;
                out.push(scaled);
            }

            counter = counter.wrapping_add(1);
        }

        // L2-normalize so cosine on hash embeddings is meaningful.
        let norm_sq: f32 = out.iter().map(|x| x * x).sum();
        let norm = norm_sq.sqrt();
        if norm > 0.0 {
            for x in &mut out {
                *x /= norm;
            }
        }

        Ok(EmbeddingVector::new(out))
    }
}

/// Cosine similarity between two embedding vectors, in `[-1, 1]`.
///
/// # Errors
/// - [`EmbedError::Backend`] (`"dim mismatch"`) when the vectors differ in
///   length.
/// - [`EmbedError::Backend`] (`"zero vector"`) when either vector has zero
///   L2 norm — cosine isn't defined in that case.
pub fn cosine_similarity(a: &EmbeddingVector, b: &EmbeddingVector) -> Result<f32, EmbedError> {
    if a.len() != b.len() {
        return Err(EmbedError::Backend("dim mismatch".to_owned()));
    }

    let mut dot = 0.0_f32;
    let mut left_sq = 0.0_f32;
    let mut right_sq = 0.0_f32;
    for (x, y) in a.as_slice().iter().zip(b.as_slice().iter()) {
        dot += x * y;
        left_sq += x * x;
        right_sq += y * y;
    }

    let left_norm = left_sq.sqrt();
    let right_norm = right_sq.sqrt();
    #[allow(
        clippy::float_cmp,
        reason = "exact zero check on accumulated squared norm; not a precision-sensitive comparison"
    )]
    let either_zero = left_norm == 0.0 || right_norm == 0.0;
    if either_zero {
        return Err(EmbedError::Backend("zero vector".to_owned()));
    }

    Ok(dot / (left_norm * right_norm))
}

/// Return the `k` highest-cosine entries in `corpus`, sorted descending.
///
/// Returns an empty `Vec` when `corpus` is empty or `k == 0`. When
/// `k > corpus.len()` the result is clamped to `corpus.len()`.
///
/// # Errors
/// Surfaces the first [`cosine_similarity`] error encountered (typically dim
/// mismatch).
pub fn top_k(
    query: &EmbeddingVector,
    corpus: &[(String, EmbeddingVector)],
    k: usize,
) -> Result<Vec<(String, f32)>, EmbedError> {
    if k == 0 || corpus.is_empty() {
        return Ok(Vec::new());
    }

    let mut scored: Vec<(String, f32)> = Vec::with_capacity(corpus.len());
    for (id, vec) in corpus {
        let score = cosine_similarity(query, vec)?;
        scored.push((id.clone(), score));
    }

    // Descending by score. NaN shouldn't occur because cosine_similarity
    // rejects zero vectors; fall back to `Ordering::Equal` defensively.
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(k);
    Ok(scored)
}

/// `BTreeMap`-backed vector store. Phase 1 stub for the `LanceDB` / sqlite-vss
/// surface; the real backend will mirror this API.
#[derive(Debug, Clone)]
pub struct InMemoryVectorStore {
    vectors: BTreeMap<String, EmbeddingVector>,
    dim: EmbeddingDim,
}

impl InMemoryVectorStore {
    /// Construct an empty store that only accepts vectors of dimension `dim`.
    #[must_use]
    pub const fn new(dim: EmbeddingDim) -> Self {
        Self {
            vectors: BTreeMap::new(),
            dim,
        }
    }

    /// Insert (or replace) a vector. Returns the previous vector if any.
    ///
    /// # Errors
    /// [`EmbedError::Backend`] (`"dim mismatch"`) when `vec.len() != dim.0`.
    pub fn insert(
        &mut self,
        id: impl Into<String>,
        vec: EmbeddingVector,
    ) -> Result<Option<EmbeddingVector>, EmbedError> {
        if vec.len() != self.dim.0 as usize {
            return Err(EmbedError::Backend("dim mismatch".to_owned()));
        }
        Ok(self.vectors.insert(id.into(), vec))
    }

    /// Borrow the stored vector for `id`, if present.
    #[must_use]
    pub fn get(&self, id: &str) -> Option<&EmbeddingVector> {
        self.vectors.get(id)
    }

    /// Remove and return the vector for `id`.
    pub fn remove(&mut self, id: &str) -> Option<EmbeddingVector> {
        self.vectors.remove(id)
    }

    /// Number of stored vectors.
    #[must_use]
    #[allow(
        clippy::missing_const_for_fn,
        reason = "BTreeMap::len isn't const fn yet on stable"
    )]
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// `true` if no vectors are stored.
    #[must_use]
    #[allow(
        clippy::missing_const_for_fn,
        reason = "BTreeMap::is_empty isn't const fn yet on stable"
    )]
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Rank stored vectors against `query` and return the top `k`.
    ///
    /// # Errors
    /// Surfaces the first ranking error (typically dim mismatch with the
    /// query).
    pub fn search(
        &self,
        query: &EmbeddingVector,
        k: usize,
    ) -> Result<Vec<(String, f32)>, EmbedError> {
        let corpus: Vec<(String, EmbeddingVector)> = self
            .vectors
            .iter()
            .map(|(id, v)| (id.clone(), v.clone()))
            .collect();
        top_k(query, &corpus, k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn embedding_vector_new_len_is_empty() {
        let v = EmbeddingVector::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(v.len(), 3);
        assert!(!v.is_empty());
        assert_eq!(v.as_slice(), &[1.0, 2.0, 3.0]);

        let empty = EmbeddingVector::new(vec![]);
        assert_eq!(empty.len(), 0);
        assert!(empty.is_empty());
    }

    #[test]
    fn embedding_dim_serde_transparent() {
        let dim = EmbeddingDim(384);
        let json = serde_json::to_string(&dim).expect("serialize");
        assert_eq!(json, "384");
        let back: EmbeddingDim = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, dim);
    }

    #[test]
    fn hash_embedder_produces_correct_dim() {
        let e = HashEmbedder::new(EmbeddingDim(16));
        let v = e.embed("hello").expect("embed");
        assert_eq!(v.len(), 16);
        assert_eq!(e.dim(), EmbeddingDim(16));
    }

    #[test]
    fn hash_embedder_is_deterministic() {
        let e = HashEmbedder::new(EmbeddingDim(32));
        let a = e.embed("the quick brown fox").expect("embed");
        let b = e.embed("the quick brown fox").expect("embed");
        assert_eq!(a, b);
    }

    #[test]
    fn hash_embedder_distinct_inputs_distinct_outputs() {
        let e = HashEmbedder::new(EmbeddingDim(32));
        let a = e.embed("alpha").expect("embed");
        let b = e.embed("beta").expect("embed");
        assert_ne!(a, b);
    }

    #[test]
    fn hash_embedder_empty_input_errors() {
        let e = HashEmbedder::new(EmbeddingDim(8));
        match e.embed("") {
            Err(EmbedError::EmptyInput) => {}
            other => panic!("expected EmptyInput, got {other:?}"),
        }
    }

    #[test]
    fn hash_embedder_output_is_l2_normalized() {
        let e = HashEmbedder::new(EmbeddingDim(64));
        let v = e.embed("normalize me please").expect("embed");
        let norm_sq: f32 = v.as_slice().iter().map(|x| x * x).sum();
        let norm = norm_sq.sqrt();
        assert!(
            (norm - 1.0).abs() < 1.0e-5,
            "expected unit norm, got {norm}"
        );
    }

    #[test]
    fn cosine_identical_vectors_is_one() {
        let v = EmbeddingVector::new(vec![0.5, 0.5, 0.5, 0.5]);
        let s = cosine_similarity(&v, &v).expect("cosine");
        assert!((s - 1.0).abs() < 1.0e-6, "expected 1.0, got {s}");
    }

    #[test]
    fn cosine_negated_vectors_is_minus_one() {
        let a = EmbeddingVector::new(vec![1.0, 2.0, 3.0]);
        let b = EmbeddingVector::new(vec![-1.0, -2.0, -3.0]);
        let s = cosine_similarity(&a, &b).expect("cosine");
        assert!((s + 1.0).abs() < 1.0e-6, "expected -1.0, got {s}");
    }

    #[test]
    fn cosine_dim_mismatch_errors() {
        let a = EmbeddingVector::new(vec![1.0, 2.0]);
        let b = EmbeddingVector::new(vec![1.0, 2.0, 3.0]);
        match cosine_similarity(&a, &b) {
            Err(EmbedError::Backend(msg)) => assert!(msg.contains("dim mismatch")),
            other => panic!("expected dim mismatch Backend error, got {other:?}"),
        }
    }

    #[test]
    fn cosine_zero_vector_errors() {
        let a = EmbeddingVector::new(vec![0.0, 0.0, 0.0]);
        let b = EmbeddingVector::new(vec![1.0, 2.0, 3.0]);
        match cosine_similarity(&a, &b) {
            Err(EmbedError::Backend(msg)) => assert!(msg.contains("zero vector")),
            other => panic!("expected zero vector Backend error, got {other:?}"),
        }
    }

    #[test]
    fn top_k_empty_corpus_returns_empty() {
        let q = EmbeddingVector::new(vec![1.0, 0.0]);
        let out = top_k(&q, &[], 5).expect("top_k");
        assert!(out.is_empty());
    }

    #[test]
    fn top_k_zero_k_returns_empty() {
        let q = EmbeddingVector::new(vec![1.0, 0.0]);
        let corpus = vec![("a".to_owned(), EmbeddingVector::new(vec![1.0, 0.0]))];
        let out = top_k(&q, &corpus, 0).expect("top_k");
        assert!(out.is_empty());
    }

    #[test]
    fn top_k_clamps_k_larger_than_corpus() {
        let q = EmbeddingVector::new(vec![1.0, 0.0]);
        let corpus = vec![
            ("a".to_owned(), EmbeddingVector::new(vec![1.0, 0.0])),
            ("b".to_owned(), EmbeddingVector::new(vec![0.0, 1.0])),
        ];
        let out = top_k(&q, &corpus, 99).expect("top_k");
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn top_k_sorts_descending() {
        let q = EmbeddingVector::new(vec![1.0, 0.0]);
        let corpus = vec![
            (
                "orthogonal".to_owned(),
                EmbeddingVector::new(vec![0.0, 1.0]),
            ),
            ("aligned".to_owned(), EmbeddingVector::new(vec![1.0, 0.0])),
            ("partial".to_owned(), EmbeddingVector::new(vec![0.8, 0.6])),
        ];
        let out = top_k(&q, &corpus, 3).expect("top_k");
        assert_eq!(out[0].0, "aligned");
        assert_eq!(out[1].0, "partial");
        assert_eq!(out[2].0, "orthogonal");
        // Strictly descending.
        assert!(out[0].1 >= out[1].1);
        assert!(out[1].1 >= out[2].1);
    }

    #[test]
    fn in_memory_store_insert_get_round_trip() {
        let mut store = InMemoryVectorStore::new(EmbeddingDim(3));
        let v = EmbeddingVector::new(vec![1.0, 2.0, 3.0]);
        let prior = store.insert("doc1", v.clone()).expect("insert");
        assert!(prior.is_none());
        assert_eq!(store.len(), 1);
        assert!(!store.is_empty());
        assert_eq!(store.get("doc1"), Some(&v));
    }

    #[test]
    fn in_memory_store_rejects_dim_mismatch() {
        let mut store = InMemoryVectorStore::new(EmbeddingDim(3));
        let v = EmbeddingVector::new(vec![1.0, 2.0]);
        match store.insert("bad", v) {
            Err(EmbedError::Backend(msg)) => assert!(msg.contains("dim mismatch")),
            other => panic!("expected dim mismatch, got {other:?}"),
        }
    }

    #[test]
    fn in_memory_store_remove_returns_prior() {
        let mut store = InMemoryVectorStore::new(EmbeddingDim(2));
        let v = EmbeddingVector::new(vec![0.6, 0.8]);
        store.insert("k", v.clone()).expect("insert");
        let prior = store.remove("k");
        assert_eq!(prior, Some(v));
        assert!(store.is_empty());
        assert!(store.remove("k").is_none());
    }

    #[test]
    fn in_memory_store_search_ranks_results() {
        let mut store = InMemoryVectorStore::new(EmbeddingDim(2));
        store
            .insert("aligned", EmbeddingVector::new(vec![1.0, 0.0]))
            .expect("insert");
        store
            .insert("orthogonal", EmbeddingVector::new(vec![0.0, 1.0]))
            .expect("insert");
        let q = EmbeddingVector::new(vec![1.0, 0.0]);
        let out = store.search(&q, 2).expect("search");
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].0, "aligned");
    }

    #[test]
    fn embed_batch_default_returns_one_per_input() {
        let e = HashEmbedder::new(EmbeddingDim(8));
        let texts = ["alpha", "beta", "gamma"];
        let refs: Vec<&str> = texts.to_vec();
        let out = e.embed_batch(&refs).expect("batch");
        assert_eq!(out.len(), 3);
        for v in &out {
            assert_eq!(v.len(), 8);
        }
    }

    #[test]
    fn embed_batch_propagates_first_error() {
        let e = HashEmbedder::new(EmbeddingDim(4));
        let refs = vec!["ok", ""];
        match e.embed_batch(&refs) {
            Err(EmbedError::EmptyInput) => {}
            other => panic!("expected EmptyInput, got {other:?}"),
        }
    }

    #[test]
    fn embed_error_display_smoke() {
        let cases = [
            EmbedError::EmptyInput,
            EmbedError::ModelUnavailable("no weights".to_owned()),
            EmbedError::Backend("bad math".to_owned()),
        ];
        for e in cases {
            let rendered = format!("{e}");
            assert!(!rendered.is_empty());
        }
    }

    #[test]
    fn hash_embedder_is_send_sync() {
        assert_send_sync::<HashEmbedder>();
    }
}
