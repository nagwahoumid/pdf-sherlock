"""
app/index_store.py - Hybrid (Dense + BM25)

Holds the in-memory search state for PDF Sherlock:
- Loads the chunk table (Parquet) with one row per text chunk.
- Encodes chunks with a Sentence-Transformers model.
- Builds (or loads) a FAISS IndexHNSWSQ (HNSW + 8-bit Scalar Quantization) over normalized vectors.
- Builds (or opens) a disk-backed Tantivy inverted index for BM25 retrieval.
- Executes top-k vector search and hydrates results back to chunk metadata.

Design notes

• Scalable Cosine Similarity:
  We use FAISS IndexHNSWSQ. By L2-normalising BOTH corpus vectors and query
  vectors, the HNSW graph search efficiently approximates nearest neighbors 
  based on cosine similarity. The 8-bit scalar quantisation (SQ8) significantly 
  compresses the memory footprint of the vectors.

• Persistence:
  To avoid recomputing embeddings on every run, we persist the FAISS index
  to disk (data/index.faiss). If it exists (and rebuild=False), we load it.

• Dtypes & memory:
  FAISS expects float32 arrays. Sentence-Transformers returns float32 by
  default; SQ8 automatically handles the quantisation during index training.

• Safety & UX:
  - If there are fewer than k vectors, FAISS will pad with -1 labels; we
    filter those out.
  - If the chunk table is empty or missing, we raise a helpful error.

• Lexical index (BM25) via Tantivy:
  Earlier revisions used ``rank_bm25.BM25Okapi``, which keeps the entire
  posting list and per-document token arrays in Python memory. At the
  ~6,600-chunk scale of the dissertation corpus this was comfortably the
  largest single consumer of RAM in the process. We now use Tantivy, the
  Rust search engine, via its Python wrapper. Tantivy writes a segment-based
  inverted index to ``data/tantivy_index/`` and memory-maps it at query
  time, so RAM usage is effectively independent of the corpus size. The
  schema stores just enough metadata (``row_idx`` as an unsigned integer,
  ``chunk_id`` as text) to map each search hit back to a row in ``self.df``
  so that the existing RRF / weighted-sum fusion math (which operates on
  DataFrame row indices) is unchanged.

Adds:
- Retrieval-tuned dense model option (MS MARCO MiniLM).
- Disk-backed BM25 lexical index via Tantivy.
- Fusion: RRF (default) or weighted-sum with alpha.

"""

from __future__ import annotations

# -----------------------------------------------------------------------------
# Threading / OpenMP guards (MUST run before FAISS / torch / tantivy imports)
# -----------------------------------------------------------------------------
#
# On Apple Silicon (ARM64) we observed a SIGSEGV in libomp.dylib whenever
# FAISS' OpenMP pool and Tantivy's Rust thread pool initialised concurrently
# during IndexStore startup. Two OpenMP runtimes getting loaded into the same
# process (one shipped by faiss-cpu, one by the Hugging Face tokenizers shared
# library) is a well known crash on macOS.
#
# The following environment variables mitigate the issue by:
#   * ``OMP_NUM_THREADS=1`` - force the OpenMP runtime used by FAISS to stick
#     to a single thread so it cannot race against Tantivy's Rayon threads.
#   * ``TOKENIZERS_PARALLELISM=false`` - tell the HF tokenizers library not
#     to spin up its own thread pool (which otherwise dead-locks on fork and
#     can double-register OpenMP).
#   * ``KMP_DUPLICATE_LIB_OK=TRUE`` - allow the process to continue if a
#     second copy of libomp is loaded instead of aborting. This is the safety
#     net when ``OMP_NUM_THREADS`` alone is not enough.
#
# These MUST be set before importing faiss / torch / sentence_transformers /
# tantivy because those libraries read them only at import time.
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import re
import shutil
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import math

import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import tantivy  # disk-backed lexical index (Rust-backed, low RAM)


# -----------------------------------------------------------------------------
# Tantivy helpers
# -----------------------------------------------------------------------------

# Characters that have special meaning in Tantivy's query parser. For a
# natural-language search box we want a bag-of-terms behaviour, so we replace
# these with spaces before calling ``parse_query``. The two offenders that
# matter most in practice are ``:`` (which Tantivy interprets as
# ``field:term`` and rejects when the left-hand side is not a known field)
# and ``"``/``\`` (which can open unterminated phrase queries or escape
# sequences).
_TANTIVY_QUERY_SPECIALS = re.compile(r'[:"\\]')

# Writer heap size for building the Tantivy index. 50 MB is the lower end of
# the range Tantivy accepts; it is plenty for our few-thousand-chunk corpus
# and keeps the peak RAM footprint of ingestion modest.
_TANTIVY_WRITER_HEAP_BYTES = 50 * 1024 * 1024


def _sanitize_tantivy_query(query: str) -> str:
    """Replace Tantivy query-syntax specials with spaces for bag-of-terms search."""
    return _TANTIVY_QUERY_SPECIALS.sub(" ", query).strip()


def _build_tantivy_schema() -> tantivy.Schema:
    """
    Build the schema used for the lexical index.

    Fields
    ------
    text : text, indexed, not stored
        The chunk text, tokenised with Tantivy's default analyzer. This is
        the only field we actually search against.
    chunk_id : text, stored
        The per-chunk UUID that we also carry in the Parquet table. Stored so
        hits can be inspected or joined back to other artefacts by id.
    row_idx : u64, stored + indexed + fast
        The 0-based row index of the chunk in ``self.df``. Stored so each hit
        can be mapped directly back to a row without needing another lookup;
        ``fast`` speeds up access from the searcher.
    """
    sb = tantivy.SchemaBuilder()
    sb.add_text_field("text", stored=False)
    sb.add_text_field("chunk_id", stored=True)
    sb.add_unsigned_field("row_idx", stored=True, indexed=True, fast=True)
    return sb.build()


class IndexStore:
    """
    Wrapper around:
      - chunk table (Parquet)
      - dense encoder + FAISS IndexHNSWSQ (cosine via L2-normalization)
      - Tantivy disk-backed inverted index for BM25
      - fusion strategy to combine both

    Parameters
    ----------
    chunks_path        : Path
    index_path         : FAISS index path
    tantivy_index_path : Tantivy index directory (defaults next to
                         ``index_path``)
    model_name         : dense model (default: MS MARCO MiniLM for retrieval)
    batch_size         : encode batch size
    device             : "cpu" | "cuda" | None (auto)
    rebuild            : force both FAISS and Tantivy rebuilds
    fusion             : "rrf" | "wsum" | "dense" | "bm25"
    alpha              : weight for dense scores in "wsum" (0..1)
    k_rrf              : RRF constant (typical 60)
    topn_dense         : how many dense hits to consider before fusion
    topn_bm25          : how many bm25 hits to consider before fusion
    """

    def __init__(
        self,
        chunks_path: str | Path = "data/chunks.parquet",
        index_path: str | Path = "data/index.faiss",
        tantivy_index_path: str | Path | None = None,
        # Retrieval-tuned default (great for short queries); your old model still works:
        model_name: str = "sentence-transformers/msmarco-MiniLM-L-6-v3",
        batch_size: int = 64,
        device: Optional[str] = None,
        rebuild: bool = False,
        # Hybrid knobs (GOOD TO EXPERIMENT)
        fusion: str = "rrf",  # "rrf" | "wsum" | "dense" | "bm25"
        alpha: float = 0.5,  # used only in "wsum"
        k_rrf: int = 60,  # RRF constant
        topn_dense: int = 50,
        topn_bm25: int = 200,
    ) -> None:
        self.chunks_path = Path(chunks_path)
        self.index_path = Path(index_path)
        # Default the Tantivy directory next to the FAISS index so all derived
        # artefacts live in the same data/ folder without further config.
        self.tantivy_index_path = Path(
            tantivy_index_path
            if tantivy_index_path is not None
            else self.index_path.parent / "tantivy_index"
        )
        self.model_name = model_name
        self.batch_size = int(batch_size)
        self.device = device
        self.rebuild = rebuild

        self.fusion = fusion
        self.alpha = float(alpha)
        self.k_rrf = int(k_rrf)
        self.topn_dense = int(topn_dense)
        self.topn_bm25 = int(topn_bm25)

        # Re-entrant lock that guards concurrent access to the mutable shared
        # state (self.df, self.index, self.tantivy_index). It is held
        # exclusively during rebuilds and acquired by readers (search) to make
        # sure they never observe a half-swapped state.
        #
        # Python's stdlib does not ship a reader-writer lock, so we use an
        # RLock. For a local app with modest search QPS this is more than
        # enough; if we ever need true read concurrency we can swap this for
        # a third-party RWLock without changing the call sites.
        self._lock = threading.RLock()

        # 1) Load chunk table
        self.df = self._load_chunks(self.chunks_path)
        if "text" not in self.df.columns:
            raise ValueError(
                f"Chunk table at {self.chunks_path} must include a 'text' column."
            )

        # 2) Init encoder
        self.model = SentenceTransformer(self.model_name, device=self.device)

        # 3) Build or load FAISS index
        if self.index_path.exists() and not self.rebuild:
            self.index = faiss.read_index(str(self.index_path))
            if self.index.ntotal != len(self.df):
                # If dimensions mismatch, rebuild to align with current df
                self._build_and_persist_index()
        else:
            self._build_and_persist_index()

        # 4) Build or open the Tantivy (BM25) lexical index
        self.tantivy_index: tantivy.Index = self._open_or_build_tantivy()

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------
    def _load_chunks(self, path: Path) -> pd.DataFrame:
        """
        Load the chunks Parquet file.

        Returns
        -------
        pd.DataFrame
            The chunk table. Raises FileNotFoundError if the path does not exist.
        """
        if not path.exists():
            raise FileNotFoundError(
                f"Chunk table not found at {path}. "
                f"Run ingest to generate 'data/chunks.parquet'."
            )
        df = pd.read_parquet(path)
        if df.empty:
            raise ValueError(f"Chunk table at {path} is empty. Ingest some PDFs first.")
        return df

    def _encode_corpus(self) -> np.ndarray:
        """
        Encode the full corpus (self.df['text']) into a float32 NumPy array,
        then L2-normalize in-place.

        Returns
        -------
        np.ndarray, shape (N, D)
            L2-normalized embeddings for all chunks.
        """
        texts: List[str] = self.df["text"].astype(str).tolist()
        # Sentence-Transformers returns float32 by default; ensure numpy array.
        embs = self.model.encode(
            texts,
            convert_to_numpy=True,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=False,  # we explicitly normalize with FAISS below
        )
        # Make sure dtype is float32 for FAISS
        if embs.dtype != np.float32:
            embs = embs.astype(np.float32, copy=False)

        # Critical: L2-normalize corpus vectors so inner product ≈ cosine
        faiss.normalize_L2(embs)
        return embs

    def _build_and_persist_index(self) -> None:
        """
        Build a fresh FAISS index using HNSW + SQ8 (8-bit scalar quantization)
        and persist it to disk.

        This combines:
          - HNSW: Hierarchical Navigable Small World graph for O(log N) search.
          - SQ8: 8-bit scalar quantization for ~4x memory reduction.
        """
        embs = self._encode_corpus()
        dim = int(embs.shape[1])

        # HNSW parameters (tunable)
        M = 32  # number of bi-directional links per node
        ef_construction = 200  # build-time search width
        ef_search = 64  # query-time search width

        # Create HNSW + SQ8 index
        # IndexHNSWSQ combines HNSW graph with scalar quantization
        self.index = faiss.IndexHNSWSQ(dim, faiss.ScalarQuantizer.QT_8bit, M)
        self.index.hnsw.efConstruction = ef_construction
        self.index.hnsw.efSearch = ef_search

        # SQ8 requires training (computes min/max per dimension for quantization)
        self.index.train(embs)
        self.index.add(embs)

        # Persist to disk
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))

    def index_memory_usage(self) -> int:
        """
        Estimate the memory footprint of the FAISS index in bytes.

        FAISS does not expose a direct "how big is this index in RAM" API for
        every index type, so we approximate by serializing the index to a
        temporary file on disk, reading its size, and deleting the file. The
        serialized size is a close proxy for the in-memory footprint for the
        index types we use (HNSW + SQ8) and is good enough for diagnostics.
        """
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(delete=False, suffix=".faiss") as f:
            faiss.write_index(self.index, f.name)
            size = os.path.getsize(f.name)
            os.unlink(f.name)
        return size

    def _encode_query(self, query: str) -> np.ndarray:
        q = self.model.encode(
            [query],
            convert_to_numpy=True,
            batch_size=1,
            show_progress_bar=False,
            normalize_embeddings=False,
        ).astype(np.float32, copy=False)
        faiss.normalize_L2(q)
        return q

    # ---- Fusion utilities ----
    @staticmethod
    def _minmax_norm(arr: np.ndarray) -> np.ndarray:
        if arr.size == 0:
            return arr
        a_min = float(arr.min())
        a_max = float(arr.max())
        if math.isclose(a_min, a_max):
            return np.zeros_like(arr)
        return (arr - a_min) / (a_max - a_min)

    def _dense_topn(self, q: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
        distances, labels = self.index.search(q, n)
        lbls = labels[0]
        dists = distances[0]
        mask = lbls != -1
        return lbls[mask], dists[mask]

    # ---- Tantivy (BM25) lexical index ----
    def _open_or_build_tantivy(self) -> tantivy.Index:
        """
        Open the on-disk Tantivy index, rebuilding it if necessary.

        Triggers a rebuild when any of the following is true:
          * ``self.rebuild`` was requested by the caller;
          * the index directory does not exist yet (cold start);
          * the existing index has a different document count from
            ``self.df`` (stale after a re-ingestion).
        """
        idx_dir = self.tantivy_index_path
        needs_build = self.rebuild or not (idx_dir.exists() and (idx_dir / "meta.json").exists())

        if not needs_build:
            try:
                index = tantivy.Index.open(str(idx_dir))
                index.reload()
                searcher = index.searcher()
                if searcher.num_docs != len(self.df):
                    needs_build = True
                else:
                    return index
            except Exception:
                # Corrupt / incompatible on-disk index -> rebuild from scratch.
                needs_build = True

        return self._build_tantivy_index()

    def _build_tantivy_index(self) -> tantivy.Index:
        """
        (Re)build the Tantivy inverted index from ``self.df`` and persist it
        to ``self.tantivy_index_path``.

        The directory is wiped and recreated before indexing so stale segments
        from previous runs cannot leak into the new index. Documents carry
        both the opaque ``chunk_id`` and the DataFrame ``row_idx`` so that
        query hits can be mapped back to rows without another lookup.
        """
        idx_dir = self.tantivy_index_path
        if idx_dir.exists():
            shutil.rmtree(idx_dir)
        idx_dir.mkdir(parents=True, exist_ok=True)

        schema = _build_tantivy_schema()
        index = tantivy.Index(schema, path=str(idx_dir))
        writer = index.writer(heap_size=_TANTIVY_WRITER_HEAP_BYTES)

        texts = self.df["text"].astype(str).tolist()
        chunk_ids = (
            self.df["chunk_id"].astype(str).tolist()
            if "chunk_id" in self.df.columns
            else [str(i) for i in range(len(self.df))]
        )
        for row_idx, (text, chunk_id) in enumerate(zip(texts, chunk_ids)):
            doc = tantivy.Document()
            doc.add_text("text", text)
            doc.add_text("chunk_id", chunk_id)
            doc.add_unsigned("row_idx", int(row_idx))
            writer.add_document(doc)

        writer.commit()
        writer.wait_merging_threads()
        index.reload()
        return index

    def _bm25_topn(self, query: str, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (row_indices, scores) of the top-n BM25 hits from Tantivy.

        The Tantivy stored field ``row_idx`` lets us emit DataFrame row
        indices directly, so the downstream fusion helpers (``_fuse_rrf``,
        ``_fuse_wsum``) continue to operate on the same integer row-index
        space as the FAISS dense hits — no additional mapping required.

        Robustness: the Tantivy query parser rejects some punctuation common
        in natural-language queries (notably ``:``), so we replace query
        specials with spaces before parsing. Any remaining parse error is
        caught and treated as "no lexical hits" so the dense stream can
        still answer the request.
        """
        m = len(self.df)
        if m == 0 or not query.strip():
            return np.empty(0, dtype=int), np.empty(0, dtype=np.float32)

        n = max(1, min(n, m))
        safe_query = _sanitize_tantivy_query(query)
        if not safe_query:
            return np.empty(0, dtype=int), np.empty(0, dtype=np.float32)

        try:
            parsed = self.tantivy_index.parse_query(safe_query, ["text"])
        except ValueError:
            return np.empty(0, dtype=int), np.empty(0, dtype=np.float32)

        searcher = self.tantivy_index.searcher()
        result = searcher.search(parsed, n)

        row_indices: List[int] = []
        scores: List[float] = []
        for score, doc_address in result.hits:
            doc = searcher.doc(doc_address)
            row_idx_field = doc.get_first("row_idx")
            if row_idx_field is None:
                continue
            row_indices.append(int(row_idx_field))
            scores.append(float(score))

        return (
            np.asarray(row_indices, dtype=np.int64),
            np.asarray(scores, dtype=np.float32),
        )

    def _fuse_rrf(
        self,
        vec_idx: np.ndarray,
        vec_scores: np.ndarray,
        bm_idx: np.ndarray,
        bm_scores: np.ndarray,
        k: int,
    ) -> List[int]:
        """Reciprocal Rank Fusion (rank-based, not magnitude-based)."""
        # Build ranks
        rrf = {}
        # dense ranks
        for rank, i in enumerate(vec_idx, start=1):
            rrf[int(i)] = rrf.get(int(i), 0.0) + 1.0 / (self.k_rrf + rank)
        # bm25 ranks
        for rank, i in enumerate(bm_idx, start=1):
            rrf[int(i)] = rrf.get(int(i), 0.0) + 1.0 / (self.k_rrf + rank)
        # top-k by fused score
        merged = sorted(rrf.items(), key=lambda x: x[1], reverse=True)
        return [i for i, _ in merged[:k]]

    def _fuse_wsum(
        self,
        vec_idx: np.ndarray,
        vec_scores: np.ndarray,
        bm_idx: np.ndarray,
        bm_scores: np.ndarray,
        k: int,
    ) -> List[int]:
        """Weighted sum over min-max normalized scores."""
        fused = {}
        # normalize to [0,1] each stream
        vnorm = self._minmax_norm(vec_scores)
        bnorm = self._minmax_norm(bm_scores)
        # dense
        for i, s in zip(vec_idx, vnorm, strict=False):
            fused[int(i)] = fused.get(int(i), 0.0) + self.alpha * float(s)
        # bm25
        for i, s in zip(bm_idx, bnorm, strict=False):
            fused[int(i)] = fused.get(int(i), 0.0) + (1.0 - self.alpha) * float(s)
        merged = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        return [i for i, _ in merged[:k]]

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def search(
        self, query: str, k: int = 5, mode: str | None = None
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search:
          - encode query -> dense topN (skipped when mode='bm25')
          - tokenize query -> BM25 topN (skipped when mode='dense')
          - fuse results (rrf | wsum | dense | bm25) -> top-k
          - hydrate to rows

        mode: Override fusion strategy per-request. If None, uses self.fusion.
        For efficiency: bm25 skips dense encoding/FAISS; dense skips BM25.

        Thread safety: acquires ``self._lock`` for the whole call so readers
        never observe a half-swapped state while an admin rebuild is in
        progress.
        """
        query = (query or "").strip()
        if not query:
            return []
        k = int(max(1, min(k, 100)))

        with self._lock:
            if self.index.ntotal == 0:
                return []

            mode = (mode or self.fusion).lower()
            vec_idx: np.ndarray = np.empty(0, dtype=np.int64)
            vec_scores: np.ndarray = np.empty(0, dtype=np.float32)
            bm_idx: np.ndarray = np.empty(0, dtype=np.int64)
            bm_scores: np.ndarray = np.empty(0, dtype=np.float32)

            # Dense stream: only when needed (dense, rrf, wsum)
            if mode in ("dense", "rrf", "wsum"):
                q = self._encode_query(query)
                vec_idx, vec_scores = self._dense_topn(q, max(k, self.topn_dense))

            # BM25 stream: only when needed (bm25, rrf, wsum)
            if mode in ("bm25", "rrf", "wsum"):
                bm_idx, bm_scores = self._bm25_topn(query, max(k, self.topn_bm25))

            # Choose fusion
            if mode == "dense":
                final_idx = list(vec_idx[:k])
            elif mode == "bm25":
                final_idx = list(bm_idx[:k])
            elif mode == "wsum":
                final_idx = self._fuse_wsum(vec_idx, vec_scores, bm_idx, bm_scores, k)
            else:  # "rrf" default
                final_idx = self._fuse_rrf(vec_idx, vec_scores, bm_idx, bm_scores, k)

            # Map back to DataFrame rows
            vec_idx_list = vec_idx.tolist()
            bm_idx_list = bm_idx.tolist()
            rows = []
            for i_row in final_idx:
                r = self.df.iloc[int(i_row)]
                # Use dense score if available, else BM25 score, else 0
                if i_row in vec_idx_list:
                    score = float(vec_scores[vec_idx_list.index(i_row)])
                elif i_row in bm_idx_list:
                    score = float(bm_scores[bm_idx_list.index(i_row)])
                else:
                    score = 0.0
                rows.append(
                    {
                        "doc_id": str(r.get("doc_id", "")),
                        "page": int(r.get("page_no", 0)),
                        "score": score,
                        "snippet": str(r.get("text", "")),
                        "chunk_id": str(r.get("chunk_id", "")),
                        "path": (
                            str(r.get("path", ""))
                            if "path" in self.df.columns
                            else None
                        ),
                    }
                )
            return rows
