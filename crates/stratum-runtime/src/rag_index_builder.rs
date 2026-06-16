//! `RagIndexBuilder` — workspace tree → [`RagIndex`] + [`InMemoryVectorStore`].
//!
//! Phase 1 scaffold. This is the first composition of the four already-landed
//! RAG primitives:
//!
//! - [`crate::workspace::Workspace`] for ignore filtering.
//! - [`crate::rag::chunk_document`] for chunk planning.
//! - [`crate::embedder::Embedder`] for embedding.
//! - [`crate::rag::RagIndex`] + [`crate::embedder::InMemoryVectorStore`] as
//!   the populated outputs.
//!
//! The real workspace walker (Phase 4+) will recurse with proper depth
//! controls; this pass implements a deterministic **depth-1** walk so the
//! composition surface is testable without committing to a recursive walker
//! design.
//!
//! ## Vector-store key shape
//!
//! Each chunk lands in [`InMemoryVectorStore`] under the key
//! `format!("{}#{}", chunk.document_id.0, chunk.ordinal)`. This matches the
//! `document#ordinal` shape used by the orchestrator's retrieval layer when
//! it joins a vector hit back to its parent [`Chunk`].
//!
//! ## Error policy
//!
//! All failure modes route through [`RagBuildError`], which wraps the
//! existing [`crate::workspace::WorkspaceError`] / [`crate::embedder::EmbedError`]
//! / [`std::io::Error`] enums without allocating a new `STRAT-E…` code —
//! per `plan/29-error-taxonomy-and-logging.md`, scaffold modules ship local
//! `enum` variants until a stable catalog code is needed.

use std::collections::BTreeSet;
use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::embedder::{EmbedError, Embedder, InMemoryVectorStore};
use crate::rag::{chunk_document, ChunkPlan, DocumentId, RagDocument, RagIndex};
use crate::workspace::{Workspace, WorkspaceError};

/// Default maximum file size considered for indexing (1 MiB).
pub const DEFAULT_MAX_FILE_BYTES: u64 = 1 << 20;

/// Default set of file extensions that are eligible for indexing.
///
/// Lowercase, no leading dot. Files outside this set are skipped and
/// counted under [`BuildStats::files_skipped_extension`].
const DEFAULT_EXTENSIONS: &[&str] = &["rs", "md", "txt", "toml", "yml", "json"];

/// Builder that walks a [`Workspace`] tree and populates a [`RagIndex`] +
/// [`InMemoryVectorStore`] under a shared [`ChunkPlan`].
///
/// Defaults:
/// - `chunk_plan` = [`ChunkPlan::default`]
/// - `max_file_bytes` = [`DEFAULT_MAX_FILE_BYTES`] (1 MiB)
/// - `allowed_extensions` = `{rs, md, txt, toml, yml, json}`
pub struct RagIndexBuilder {
    workspace: Arc<Workspace>,
    embedder: Arc<dyn Embedder>,
    chunk_plan: ChunkPlan,
    max_file_bytes: u64,
    allowed_extensions: BTreeSet<String>,
}

impl fmt::Debug for RagIndexBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RagIndexBuilder")
            .field("workspace_root", &self.workspace.root)
            .field("chunk_plan", &self.chunk_plan)
            .field("max_file_bytes", &self.max_file_bytes)
            .field("allowed_extensions", &self.allowed_extensions)
            .finish_non_exhaustive()
    }
}

impl RagIndexBuilder {
    /// Construct a builder bound to a workspace + embedder, with defaults.
    #[must_use]
    pub fn new(workspace: Arc<Workspace>, embedder: Arc<dyn Embedder>) -> Self {
        let allowed_extensions: BTreeSet<String> =
            DEFAULT_EXTENSIONS.iter().map(|s| (*s).to_owned()).collect();
        Self {
            workspace,
            embedder,
            chunk_plan: ChunkPlan::default(),
            max_file_bytes: DEFAULT_MAX_FILE_BYTES,
            allowed_extensions,
        }
    }

    /// Override the chunk plan.
    #[must_use]
    pub const fn with_chunk_plan(mut self, plan: ChunkPlan) -> Self {
        self.chunk_plan = plan;
        self
    }

    /// Override the per-file size cap. Files larger than this are skipped.
    #[must_use]
    pub const fn with_max_file_bytes(mut self, bytes: u64) -> Self {
        self.max_file_bytes = bytes;
        self
    }

    /// Override the allowed-extension set. Pass extensions without the
    /// leading dot, lowercase (e.g. `"rs"`, `"md"`).
    #[must_use]
    pub fn with_extensions(mut self, extensions: BTreeSet<String>) -> Self {
        self.allowed_extensions = extensions;
        self
    }

    /// Borrow the chunk plan the builder will use.
    #[must_use]
    pub const fn chunk_plan(&self) -> &ChunkPlan {
        &self.chunk_plan
    }

    /// Borrow the file-size cap (bytes).
    #[must_use]
    pub const fn max_file_bytes(&self) -> u64 {
        self.max_file_bytes
    }

    /// Borrow the allowed-extension set.
    #[must_use]
    pub const fn allowed_extensions(&self) -> &BTreeSet<String> {
        &self.allowed_extensions
    }

    /// Walk the supplied paths and emit a populated [`BuiltIndex`].
    ///
    /// For each input path:
    /// 1. Stat the path. If it is a directory, walk it at **depth 1**
    ///    (entries directly inside, not recursive — recursive walking is a
    ///    follow-up PR). The directory itself is not indexed.
    /// 2. Skip ignored paths (via [`Workspace::is_ignored`]). Counts under
    ///    [`BuildStats::files_skipped_ignored`].
    /// 3. Skip files whose extension isn't in [`Self::allowed_extensions`].
    ///    Counts under [`BuildStats::files_skipped_extension`].
    /// 4. Skip files larger than [`Self::max_file_bytes`]. Counts under
    ///    [`BuildStats::files_skipped_size`].
    /// 5. Read the file as UTF-8 (lossy → `from_utf8_lossy`), insert it as
    ///    a [`RagDocument`] keyed by `path.display()`, chunk it via
    ///    [`chunk_document`], embed each chunk and insert into the vector
    ///    store under `"<doc_id>#<ordinal>"`.
    ///
    /// Input is sorted before walking so the walk is deterministic.
    ///
    /// # Errors
    ///
    /// Returns [`RagBuildError`] on the first non-recoverable failure
    /// (workspace ignore lookup, vector store dim mismatch, IO). Per-chunk
    /// embedder failures are *not* fatal; they are counted under
    /// [`BuildStats::embed_failures`] and the build continues.
    pub fn build<I>(&self, paths: I) -> Result<BuiltIndex, RagBuildError>
    where
        I: IntoIterator<Item = PathBuf>,
    {
        let started = Instant::now();
        let mut stats = BuildStats::default();

        let mut rag = RagIndex::with_plan(self.chunk_plan);
        let mut vectors = InMemoryVectorStore::new(self.embedder.dim());

        // Sort the input for deterministic ordering.
        let mut sorted: Vec<PathBuf> = paths.into_iter().collect();
        sorted.sort();

        // Expand directories at depth 1.
        let mut leaves: Vec<PathBuf> = Vec::new();
        for p in sorted {
            match std::fs::metadata(&p) {
                Ok(meta) if meta.is_dir() => {
                    let mut children: Vec<PathBuf> = Vec::new();
                    let read_dir = std::fs::read_dir(&p).map_err(RagBuildError::Io)?;
                    for entry in read_dir {
                        let entry = entry.map_err(RagBuildError::Io)?;
                        children.push(entry.path());
                    }
                    children.sort();
                    for c in children {
                        // Only file children — depth-1 walk does not recurse.
                        match std::fs::metadata(&c) {
                            Ok(m) if m.is_file() => leaves.push(c),
                            Ok(_) => {
                                // Sub-directories are silently dropped at
                                // depth-1; future recursive walker handles them.
                            }
                            Err(e) => return Err(RagBuildError::Io(e)),
                        }
                    }
                }
                Ok(meta) if meta.is_file() => leaves.push(p),
                Ok(_) => {
                    // Other file types (symlinks to non-files, sockets, …)
                    // are not indexable at this scaffold layer.
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    return Err(RagBuildError::Io(e));
                }
                Err(e) => return Err(RagBuildError::Io(e)),
            }
        }

        for path in leaves {
            stats.files_scanned = stats.files_scanned.saturating_add(1);

            // Ignore filter.
            match self.workspace.is_ignored(&path) {
                Ok(true) => {
                    stats.files_skipped_ignored = stats.files_skipped_ignored.saturating_add(1);
                    continue;
                }
                Ok(false) => {}
                Err(e) => return Err(RagBuildError::Workspace(e)),
            }

            // Extension filter.
            if !self.extension_allowed(&path) {
                stats.files_skipped_extension = stats.files_skipped_extension.saturating_add(1);
                continue;
            }

            // Size filter.
            let meta = std::fs::metadata(&path).map_err(RagBuildError::Io)?;
            if meta.len() > self.max_file_bytes {
                stats.files_skipped_size = stats.files_skipped_size.saturating_add(1);
                continue;
            }

            // Read the file (lossy UTF-8).
            let bytes = std::fs::read(&path).map_err(RagBuildError::Io)?;
            let text = String::from_utf8_lossy(&bytes).into_owned();

            let doc_id = DocumentId::new(path.display().to_string());
            let doc = RagDocument {
                id: doc_id.clone(),
                source_path: Some(path.clone()),
                text,
                mime: "text/plain".to_owned(),
                indexed_at: rfc3339_now(),
            };

            // Insert (re-chunks under the index's plan, which matches ours).
            rag.insert(doc.clone());
            stats.files_indexed = stats.files_indexed.saturating_add(1);

            let chunks = chunk_document(&self.chunk_plan, &doc);
            for chunk in chunks {
                stats.chunks_emitted = stats.chunks_emitted.saturating_add(1);
                match self.embedder.embed(&chunk.text) {
                    Ok(vec) => {
                        let key = format!("{}#{}", chunk.document_id.0, chunk.ordinal);
                        vectors.insert(key, vec).map_err(|e| match e {
                            EmbedError::Backend(msg) => RagBuildError::DimMismatch(msg),
                            other => RagBuildError::Embedder(other),
                        })?;
                    }
                    Err(_) => {
                        stats.embed_failures = stats.embed_failures.saturating_add(1);
                    }
                }
            }
        }

        let elapsed = started.elapsed();
        stats.elapsed_ms = u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX);

        Ok(BuiltIndex {
            rag,
            vectors,
            stats,
        })
    }

    fn extension_allowed(&self, path: &std::path::Path) -> bool {
        path.extension()
            .and_then(|s| s.to_str())
            .is_some_and(|ext| self.allowed_extensions.contains(&ext.to_ascii_lowercase()))
    }
}

/// RFC 3339 timestamp for "now". Falls back to an empty string when the
/// system clock is before the Unix epoch, which is treated as unindexed
/// metadata rather than a fatal error.
fn rfc3339_now() -> String {
    let now = SystemTime::now();
    now.duration_since(UNIX_EPOCH).map_or_else(
        |_| String::new(),
        |d| format!("1970-01-01T00:00:00Z+{}s", d.as_secs()),
    )
}

/// Output of [`RagIndexBuilder::build`].
#[derive(Debug)]
pub struct BuiltIndex {
    /// Populated RAG index.
    pub rag: RagIndex,
    /// Populated vector store, keyed by `"<document_id>#<ordinal>"`.
    pub vectors: InMemoryVectorStore,
    /// Build statistics.
    pub stats: BuildStats,
}

/// Per-build counters.
///
/// `serde` round-trips cleanly so callers can emit these as JSON events.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BuildStats {
    /// Number of leaf files visited (after depth-1 expansion).
    pub files_scanned: u64,
    /// Number of files successfully indexed.
    pub files_indexed: u64,
    /// Files skipped because they exceeded the size cap.
    pub files_skipped_size: u64,
    /// Files skipped because their extension wasn't allowed.
    pub files_skipped_extension: u64,
    /// Files skipped because the workspace ignore rules matched.
    pub files_skipped_ignored: u64,
    /// Total chunks fed into the embedder.
    pub chunks_emitted: u64,
    /// Embedder calls that returned an error (build continued).
    pub embed_failures: u64,
    /// Wall-clock elapsed time in milliseconds.
    pub elapsed_ms: u64,
}

/// Errors returned by [`RagIndexBuilder::build`].
#[derive(Debug)]
pub enum RagBuildError {
    /// A workspace-level failure (ignore lookup, path outside root, …).
    Workspace(WorkspaceError),
    /// A non-recoverable embedder failure surfaced by the vector store.
    Embedder(EmbedError),
    /// IO failure while statting or reading a path.
    Io(std::io::Error),
    /// The embedder produced a vector whose dimension doesn't match the
    /// store (carries the vector store's own message).
    DimMismatch(String),
}

impl fmt::Display for RagBuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Workspace(e) => write!(f, "rag build: workspace error: {e}"),
            Self::Embedder(e) => write!(f, "rag build: embedder error: {e}"),
            Self::Io(e) => write!(f, "rag build: io error: {e}"),
            Self::DimMismatch(msg) => write!(f, "rag build: dim mismatch: {msg}"),
        }
    }
}

impl std::error::Error for RagBuildError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Workspace(e) => Some(e),
            Self::Embedder(e) => Some(e),
            Self::Io(e) => Some(e),
            Self::DimMismatch(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use tempfile::TempDir;

    use super::*;
    use crate::embedder::{EmbeddingDim, HashEmbedder};
    use crate::workspace::{Workspace, IGNORE_FILE};

    fn make_workspace(tmp: &TempDir) -> Arc<Workspace> {
        let ws = Workspace::load(tmp.path()).expect("load workspace");
        Arc::new(ws)
    }

    fn make_embedder() -> Arc<dyn Embedder> {
        Arc::new(HashEmbedder::new(EmbeddingDim(16)))
    }

    fn write_file(root: &Path, name: &str, body: &str) -> PathBuf {
        let p = root.join(name);
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent).expect("mkdir parent");
        }
        std::fs::write(&p, body).expect("write file");
        p
    }

    #[test]
    fn new_uses_documented_defaults() {
        let tmp = TempDir::new().unwrap();
        let ws = make_workspace(&tmp);
        let emb = make_embedder();
        let b = RagIndexBuilder::new(ws, emb);

        assert_eq!(b.chunk_plan(), &ChunkPlan::default());
        assert_eq!(b.max_file_bytes(), DEFAULT_MAX_FILE_BYTES);

        let expected: BTreeSet<String> =
            DEFAULT_EXTENSIONS.iter().map(|s| (*s).to_owned()).collect();
        assert_eq!(b.allowed_extensions(), &expected);
    }

    #[test]
    fn with_chunk_plan_round_trip() {
        let tmp = TempDir::new().unwrap();
        let plan = ChunkPlan {
            max_chars: 7,
            overlap_chars: 2,
        };
        let b = RagIndexBuilder::new(make_workspace(&tmp), make_embedder()).with_chunk_plan(plan);
        assert_eq!(b.chunk_plan(), &plan);
    }

    #[test]
    fn with_max_file_bytes_round_trip() {
        let tmp = TempDir::new().unwrap();
        let b =
            RagIndexBuilder::new(make_workspace(&tmp), make_embedder()).with_max_file_bytes(4242);
        assert_eq!(b.max_file_bytes(), 4242);
    }

    #[test]
    fn with_extensions_round_trip() {
        let tmp = TempDir::new().unwrap();
        let exts: BTreeSet<String> = ["rs".to_owned(), "lock".to_owned()].into_iter().collect();
        let b = RagIndexBuilder::new(make_workspace(&tmp), make_embedder())
            .with_extensions(exts.clone());
        assert_eq!(b.allowed_extensions(), &exts);
    }

    #[test]
    fn build_on_empty_input_is_empty() {
        let tmp = TempDir::new().unwrap();
        let b = RagIndexBuilder::new(make_workspace(&tmp), make_embedder());
        let built = b.build(Vec::<PathBuf>::new()).expect("build empty");
        assert!(built.rag.is_empty());
        assert!(built.vectors.is_empty());
        assert_eq!(
            built.stats,
            BuildStats {
                elapsed_ms: built.stats.elapsed_ms,
                ..BuildStats::default()
            }
        );
    }

    #[test]
    fn build_indexes_single_markdown_file() {
        let tmp = TempDir::new().unwrap();
        let path = write_file(tmp.path(), "notes.md", "hello rag world");
        let b = RagIndexBuilder::new(make_workspace(&tmp), make_embedder());
        let built = b.build(vec![path]).expect("build");
        assert_eq!(built.rag.len(), 1);
        assert!(!built.vectors.is_empty());
        assert_eq!(built.stats.files_indexed, 1);
        assert_eq!(built.stats.files_scanned, 1);
    }

    #[test]
    fn build_skips_disallowed_extension() {
        let tmp = TempDir::new().unwrap();
        let kept = write_file(tmp.path(), "keep.md", "kept body");
        let dropped = write_file(tmp.path(), "image.pdf", "binary-ish");
        let b = RagIndexBuilder::new(make_workspace(&tmp), make_embedder());
        let built = b.build(vec![kept, dropped]).expect("build");
        assert_eq!(built.stats.files_indexed, 1);
        assert_eq!(built.stats.files_skipped_extension, 1);
    }

    #[test]
    fn build_skips_files_over_size_cap() {
        let tmp = TempDir::new().unwrap();
        let big = write_file(tmp.path(), "big.txt", "0123456789abcdef");
        let b = RagIndexBuilder::new(make_workspace(&tmp), make_embedder()).with_max_file_bytes(4);
        let built = b.build(vec![big]).expect("build");
        assert_eq!(built.stats.files_skipped_size, 1);
        assert_eq!(built.stats.files_indexed, 0);
    }

    #[test]
    fn build_skips_ignored_files() {
        let tmp = TempDir::new().unwrap();
        std::fs::write(tmp.path().join(IGNORE_FILE), "secret.md\n").expect("ignore");
        let kept = write_file(tmp.path(), "keep.md", "kept");
        let dropped = write_file(tmp.path(), "secret.md", "shh");
        let b = RagIndexBuilder::new(make_workspace(&tmp), make_embedder());
        let built = b.build(vec![kept, dropped]).expect("build");
        assert_eq!(built.stats.files_indexed, 1);
        assert_eq!(built.stats.files_skipped_ignored, 1);
    }

    #[test]
    fn build_expands_directory_at_depth_one() {
        let tmp = TempDir::new().unwrap();
        let _a = write_file(tmp.path(), "sub/a.md", "alpha");
        let _b = write_file(tmp.path(), "sub/b.md", "beta");
        // A nested directory under sub/ that should NOT be descended into.
        std::fs::create_dir_all(tmp.path().join("sub/deeper")).expect("mkdir deeper");
        let _c = write_file(tmp.path(), "sub/deeper/c.md", "gamma");

        let b = RagIndexBuilder::new(make_workspace(&tmp), make_embedder());
        let built = b.build(vec![tmp.path().join("sub")]).expect("build");

        // Depth-1: two files indexed, the nested one is skipped (not descended).
        assert_eq!(built.stats.files_indexed, 2);
        assert_eq!(built.rag.len(), 2);
    }

    #[test]
    fn build_chunks_long_file_into_multiple_pieces() {
        let tmp = TempDir::new().unwrap();
        let long_body = "abcdefghijklmnopqrstuvwxyz".repeat(40); // 1040 bytes
        let p = write_file(tmp.path(), "long.txt", &long_body);
        let b = RagIndexBuilder::new(make_workspace(&tmp), make_embedder()).with_chunk_plan(
            ChunkPlan {
                max_chars: 32,
                overlap_chars: 4,
            },
        );
        let built = b.build(vec![p]).expect("build");
        assert!(
            built.stats.chunks_emitted > 1,
            "expected >1 chunk emitted, got {}",
            built.stats.chunks_emitted
        );
    }

    #[test]
    fn build_emits_one_vector_per_chunk() {
        let tmp = TempDir::new().unwrap();
        let p = write_file(tmp.path(), "n.md", "abcdefghij");
        let b = RagIndexBuilder::new(make_workspace(&tmp), make_embedder()).with_chunk_plan(
            ChunkPlan {
                max_chars: 3,
                overlap_chars: 1,
            },
        );
        let built = b.build(vec![p]).expect("build");
        assert!(built.stats.chunks_emitted >= 4);
        assert_eq!(built.vectors.len() as u64, built.stats.chunks_emitted);
    }

    #[test]
    fn build_empty_file_produces_no_chunks() {
        let tmp = TempDir::new().unwrap();
        let p = write_file(tmp.path(), "empty.md", "");
        let b = RagIndexBuilder::new(make_workspace(&tmp), make_embedder());
        let built = b.build(vec![p]).expect("build");
        assert_eq!(built.stats.files_indexed, 1);
        assert_eq!(built.stats.chunks_emitted, 0);
        assert!(built.vectors.is_empty());
    }

    #[test]
    fn build_stats_serde_round_trip() {
        let stats = BuildStats {
            files_scanned: 5,
            files_indexed: 3,
            files_skipped_size: 1,
            files_skipped_extension: 1,
            files_skipped_ignored: 0,
            chunks_emitted: 7,
            embed_failures: 0,
            elapsed_ms: 42,
        };
        let json = serde_json::to_string(&stats).expect("serialize stats");
        let back: BuildStats = serde_json::from_str(&json).expect("deserialize stats");
        assert_eq!(stats, back);
    }

    #[test]
    fn build_stats_default_and_eq() {
        let a = BuildStats::default();
        let b = BuildStats::default();
        assert_eq!(a, b);
        assert_eq!(a.files_scanned, 0);
        assert_eq!(a.chunks_emitted, 0);
    }

    #[test]
    fn rag_build_error_display_smoke() {
        use std::error::Error;
        let cases: Vec<RagBuildError> = vec![
            RagBuildError::Workspace(WorkspaceError::IgnoreParse("bad".into())),
            RagBuildError::Embedder(EmbedError::EmptyInput),
            RagBuildError::Io(std::io::Error::other("boom")),
            RagBuildError::DimMismatch("mismatch".into()),
        ];
        for e in &cases {
            let rendered = format!("{e}");
            assert!(rendered.starts_with("rag build:"), "got: {rendered}");
        }
        // source() returns Some for the wrapping variants, None for DimMismatch.
        assert!(cases[0].source().is_some());
        assert!(cases[1].source().is_some());
        assert!(cases[2].source().is_some());
        assert!(cases[3].source().is_none());
    }

    #[test]
    fn build_walks_inputs_in_sorted_order() {
        let tmp = TempDir::new().unwrap();
        let a = write_file(tmp.path(), "a.md", "alpha");
        let b = write_file(tmp.path(), "b.md", "beta");
        let c = write_file(tmp.path(), "c.md", "gamma");
        let builder = RagIndexBuilder::new(make_workspace(&tmp), make_embedder());
        // Pass in reverse order — the builder should sort.
        let built = builder.build(vec![c, b, a]).expect("build");
        assert_eq!(built.rag.len(), 3);
        // Iter order isn't guaranteed by RagIndex, but stats reflect all
        // three were indexed deterministically.
        assert_eq!(built.stats.files_indexed, 3);
    }

    #[test]
    fn rag_index_builder_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<RagIndexBuilder>();
        assert_send_sync::<BuiltIndex>();
        assert_send_sync::<BuildStats>();
        assert_send_sync::<RagBuildError>();
    }

    #[test]
    fn build_propagates_path_outside_workspace_as_workspace_error() {
        // Workspace::is_ignored returns PathOutsideWorkspace for absolute
        // paths not under root.
        let tmp = TempDir::new().unwrap();
        let other = TempDir::new().unwrap();
        let stray = write_file(other.path(), "x.md", "out of bounds");
        let b = RagIndexBuilder::new(make_workspace(&tmp), make_embedder());
        let err = b.build(vec![stray]).expect_err("expected workspace error");
        match err {
            RagBuildError::Workspace(WorkspaceError::PathOutsideWorkspace { .. }) => {}
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn build_uses_supplied_chunk_plan_in_index() {
        // The RagIndex inside BuiltIndex is constructed with the same plan,
        // so re-chunking on insert produces chunks consistent with the
        // builder's plan.
        let tmp = TempDir::new().unwrap();
        let p = write_file(tmp.path(), "n.md", "abcdefghijklmnop");
        let plan = ChunkPlan {
            max_chars: 4,
            overlap_chars: 0,
        };
        let b = RagIndexBuilder::new(make_workspace(&tmp), make_embedder()).with_chunk_plan(plan);
        let built = b.build(vec![p]).expect("build");
        let doc_id = built
            .rag
            .iter()
            .map(|(id, _)| id.clone())
            .next()
            .expect("one doc");
        let chunks = built.rag.chunks_for(&doc_id);
        // 16 bytes / 4 per chunk, no overlap → 4 chunks.
        assert_eq!(chunks.len(), 4);
    }

    #[test]
    fn build_skips_non_file_non_dir_entries_silently() {
        // Pass a path that does not exist at stat time — expect Io error.
        let tmp = TempDir::new().unwrap();
        let missing = tmp.path().join("missing.md");
        let b = RagIndexBuilder::new(make_workspace(&tmp), make_embedder());
        let err = b.build(vec![missing]).expect_err("expected io error");
        match err {
            RagBuildError::Io(e) => assert_eq!(e.kind(), std::io::ErrorKind::NotFound),
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn rag_index_builder_debug_includes_fields() {
        let tmp = TempDir::new().unwrap();
        let b = RagIndexBuilder::new(make_workspace(&tmp), make_embedder())
            .with_max_file_bytes(123)
            .with_chunk_plan(ChunkPlan {
                max_chars: 5,
                overlap_chars: 1,
            });
        let dbg = format!("{b:?}");
        assert!(dbg.contains("RagIndexBuilder"));
        assert!(dbg.contains("max_file_bytes"));
        assert!(dbg.contains("123"));
        assert!(dbg.contains("chunk_plan"));
        assert!(dbg.contains("allowed_extensions"));
    }

    #[test]
    fn build_silently_drops_nested_subdirectories_at_depth_one() {
        // The depth-1 walker should hit the Ok(_) (non-file) arm when it
        // encounters a sub-directory, exercising the silent-drop branch.
        let tmp = TempDir::new().unwrap();
        // Sub directory with one nested sub-dir (no leaves matching).
        std::fs::create_dir_all(tmp.path().join("sub/nested")).unwrap();
        let b = RagIndexBuilder::new(make_workspace(&tmp), make_embedder());
        let built = b.build(vec![tmp.path().join("sub")]).expect("build");
        // Nested sub-dir silently dropped → 0 files indexed.
        assert_eq!(built.stats.files_indexed, 0);
    }
}
