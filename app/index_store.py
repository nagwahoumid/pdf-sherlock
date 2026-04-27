"""
app/index_store.py

The hybrid (Dense + BM25) retrieval engine.

What this does:
- Loads the chunk table (Parquet) — one row per text chunk.
- Encodes the chunks with Sentence-Transformers and parks them in FAISS.
- Keeps a disk-backed Tantivy index alongside for BM25 / lexical hits.
- Runs top-k searches and hydrates the hits back to row metadata for the API.

A few design choices worth flagging:

• Cosine via inner product.
  I L2-normalise both the corpus and the query vectors up front, which lets
  me use FAISS IndexHNSWSQ (HNSW graph + 8-bit scalar quantisation) as a
  cosine-similarity index. SQ8 cuts the per-vector memory roughly 4× and I
  haven't seen any recall regression I could measure on this corpus.

• Persistence.
  Re-encoding the corpus on every boot is a non-starter, so I persist the
  FAISS index to disk (data/index.faiss). If the saved row count drifts
  from the current chunk table I rebuild instead of trusting it.

• Empty / underfilled queries.
  FAISS pads its label output with -1 when the corpus has fewer than k
  vectors; I filter those out so callers never see phantom rows. An empty
  or missing chunk table fails loudly at init, not silently at query time.

• Why Tantivy for BM25.
  An earlier revision used ``rank_bm25.BM25Okapi``. At ~6.6k chunks it was
  easily the largest RAM consumer in the process because every posting
  list and per-doc token array lived in Python memory. Tantivy is a Rust
  search engine that writes a segment-based inverted index to
  ``data/tantivy_index/`` and memory-maps it at query time, so the RAM
  cost is effectively constant in corpus size. The schema stores just
  enough metadata (``row_idx`` as a u64, ``chunk_id`` as text) to map each
  hit straight back to a self.df row, so the fusion math downstream
  (which operates on integer row indices) didn't have to change at all.

Knobs available:
- Retrieval-tuned dense model (MS MARCO MiniLM by default).
- Disk-backed BM25 via Tantivy.
- Fusion: RRF (default), weighted-sum with alpha, or pure dense / pure bm25.
"""

from __future__ import annotations
"""

Threading and OpenMP guards (these must run before FAISS, torch, and tantivy imports)

On Apple Silicon I kept hitting a SIGSEGV inside libomp.dylib at IndexStore
startup. The trigger is two OpenMP runtimes ending up in the same process
(one ships with faiss-cpu, the other comes in through the HF tokenizers
shared library) racing against Tantivy's Rust thread pool — a well known
macOS crash that I don't get to fix from Python.

What I'm doing about it:
  * ``OMP_NUM_THREADS=1`` - pin FAISS's OpenMP pool to a single thread so it
    can't race against Tantivy's Rayon workers.
  * ``TOKENIZERS_PARALLELISM=false`` - stop HF tokenizers from spinning up
    its own thread pool, which otherwise dead-locks on fork and double-
    registers OpenMP.
  * ``KMP_DUPLICATE_LIB_OK=TRUE`` - last-resort safety net: keep going if a
    second libomp slips in instead of aborting the whole process.

These MUST land before faiss / torch / sentence_transformers / tantivy are
imported, because those libraries read them only once at import time. I'm
using setdefault() so an operator-set shell value still wins.


"""

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

# Tantivy's query parser treats a handful of characters as syntax. For a
# plain natural-language search box I just want bag-of-terms, so I scrub
# these to spaces before calling ``parse_query``. The two offenders I
# actually hit in the wild are ``:`` (Tantivy reads it as ``field:term``
# and throws if the LHS isn't a real field) and ``"`` / ``\`` (unterminated
# phrase queries / escape sequences).
_TANTIVY_QUERY_SPECIALS = re.compile(r'[:"\\]')

# Writer heap for building the Tantivy index. 50 MB is the low end of what
# Tantivy will accept and is more than enough for a few-thousand-chunk
# corpus. Keeping it small caps the peak RAM footprint during a rebuild.
_TANTIVY_WRITER_HEAP_BYTES = 50 * 1024 * 1024


def _sanitize_tantivy_query(query: str) -> str:
    """Replace Tantivy query-syntax specials with spaces for bag-of-terms search."""
    return _TANTIVY_QUERY_SPECIALS.sub(" ", query).strip()


def _build_tantivy_schema() -> tantivy.Schema:
    """
    Schema for the lexical index.

    Fields
    ------
    text : text, indexed, not stored
        The chunk text, run through Tantivy's default tokenizer. This is
        the only field I actually search against — no point storing it
        twice when self.df already holds the raw string.
    chunk_id : text, stored
        Per-chunk UUID, mirrored from the Parquet table. Stored so hits
        can be inspected or joined back to other artefacts by id.
    row_idx : u64, stored + indexed + fast
        0-based row index of the chunk in ``self.df``. Stored so I can
        map a hit straight back to a DataFrame row without a second
        lookup; ``fast`` keeps that access cheap inside the searcher loop.
    """
    sb = tantivy.SchemaBuilder()
    sb.add_text_field("text", stored=False)
    sb.add_text_field("chunk_id", stored=True)
    sb.add_unsigned_field("row_idx", stored=True, indexed=True, fast=True)
    return sb.build()


class IndexStore:
    """
    Owns the live retrieval state: chunk table, dense encoder + FAISS
    index, Tantivy lexical index, and the fusion knobs that combine the
    two streams.

    Parameters
    ----------
    chunks_path        : Path
    index_path         : FAISS index path
    tantivy_index_path : Tantivy index directory (defaults next to
                         ``index_path``)
    model_name         : dense model (default: MS MARCO MiniLM for retrieval)
    batch_size         : encode batch size
    device             : "cpu" or "cuda" or None (auto)
    rebuild            : force both FAISS and Tantivy rebuilds
    fusion             : "rrf" or "wsum" or "dense" or "bm25"
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
       
        model_name: str = "sentence-transformers/msmarco-MiniLM-L-6-v3",
        batch_size: int = 64,
        device: Optional[str] = None,
        rebuild: bool = False,
        # Hybrid knobs 
        fusion: str = "rrf",  # "rrf" or "wsum" or "dense" or "bm25"
        alpha: float = 0.5,  # used only in "wsum"
        k_rrf: int = 60,  # RRF constant
        topn_dense: int = 50,
        topn_bm25: int = 200,
    ) -> None:
        self.chunks_path = Path(chunks_path)
        self.index_path = Path(index_path)
        # I default the Tantivy directory to live next to the FAISS index
        # so every derived artefact stays under the same data/ folder
        # without anyone needing to wire up an extra path.
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

        # Re-entrant lock guarding the mutable shared state (self.df,
        # self.index, self.tantivy_index). Held exclusively while a
        # rebuild is swapping pieces in, and acquired by /search readers
        # so they never catch the store mid-swap.
        #
        # Stdlib doesn't ship a real RW lock, so I'm using an RLock and
        # eating the writer-priority cost. For a local app with modest
        # QPS that's fine, if I ever need true read concurrency I can
        # drop in a third-party RWLock here without touching any call site.
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
                # On-disk index is stale relative to self.df (different
                # row count). Drop it and rebuild from the current chunks.
                self._build_and_persist_index()
        else:
            self._build_and_persist_index()

        # 4) Build or open the Tantivy (BM25) lexical index
        self.tantivy_index: tantivy.Index = self._open_or_build_tantivy()

 
    # Private helpers

    def _load_chunks(self, path: Path) -> pd.DataFrame:
        """
        Load the chunks Parquet file.

        Failing loudly here is on purpose — a missing or empty chunk
        table at startup means the rest of the pipeline has nothing to
        work with, and I'd rather see the error now than chase a
        zero-results bug at query time.
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
        Encode the full corpus into a float32 NumPy array and L2-normalise
        it in place, ready to be handed straight to FAISS.
        """
        texts: List[str] = self.df["text"].astype(str).tolist()
        # ST already returns float32 here; the dtype check below is a
        # belt-and-braces guard in case a future model swap returns
        # something else such as fp16 from a quantised model.
        embs = self.model.encode(
            texts,
            convert_to_numpy=True,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=False,  # I normalise below with faiss.normalize_L2
        )
        # FAISS will refuse anything but float32, so cast defensively.
        if embs.dtype != np.float32:
            embs = embs.astype(np.float32, copy=False)

        # The "cosine via inner product" trick falls apart if the corpus
        # side isn't unit-norm. In place to dodge a copy of the whole matrix.
        faiss.normalize_L2(embs)
        return embs

    def _build_and_persist_index(self) -> None:
        """
        Rebuild the FAISS index from scratch and write it to disk.

        I use IndexHNSWSQ — HNSW for the O(log N) graph traversal, plus
        SQ8 to shave the per-vector memory by around 4× without any recall
        regression I could measure on this corpus.
        """
        embs = self._encode_corpus()
        dim = int(embs.shape[1])

        # HNSW knobs picked empirically on this corpus. Worth a second
        # look if I ever swap in a much bigger model or dataset.
        M = 32  # number of bi-directional links per node
        ef_construction = 200  # build-time search width
        ef_search = 64  # query-time search width

        self.index = faiss.IndexHNSWSQ(dim, faiss.ScalarQuantizer.QT_8bit, M)
        self.index.hnsw.efConstruction = ef_construction
        self.index.hnsw.efSearch = ef_search

        # SQ8 needs a quick training pass to learn per-dim min/max before
        # any vectors can be added — skip this and add() will throw.
        self.index.train(embs)
        self.index.add(embs)

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))

    def index_memory_usage(self) -> int:
        """
        Rough estimate of the FAISS index footprint, in bytes.

        FAISS doesn't expose a "how big am I in RAM" hook for every index
        type, so I cheat: serialise the index to a temp file, stat it,
        delete. The on-disk size is a close enough proxy for HNSW + SQ8
        to be useful as a diagnostic — don't quote it as a hard number
        anywhere serious.
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

    #  Fusion utilities 
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

    #  Tantivy (BM25) lexical index 
    def _open_or_build_tantivy(self) -> tantivy.Index:
        """
        Try to open the on-disk Tantivy index, rebuilding from scratch
        if there's any reason to distrust it.

        I rebuild when:
          * the caller explicitly asked for it (``self.rebuild``);
          * the index directory doesn't exist yet (cold start);
          * the on-disk doc count drifts from ``len(self.df)``, which
            means a re-ingest landed without rebuilding the lexical
            side and the two indexes are now out of sync;
          * the open call itself raises — in that case I assume the
            directory is corrupt or written by an incompatible Tantivy
            version and start over rather than serve stale results.
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
                # Most likely a corrupt directory or one written by an
                # older Tantivy version. Either way, blow it away and
                # rebuild rather than risk weird half-broken queries.
                needs_build = True

        return self._build_tantivy_index()

    def _build_tantivy_index(self) -> tantivy.Index:
        """
        Rebuild the Tantivy inverted index from ``self.df`` and write it
        to ``self.tantivy_index_path``.

        I wipe the directory before indexing. Tantivy is segment-based
        and old segments from a previous corpus would happily survive an
        "incremental" build, which silently poisons search results. Each
        document carries its ``chunk_id`` and its ``row_idx`` so hits
        land back on the right ``self.df`` row without a second lookup
        later.
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
        Return (row_indices, scores) for the top-n BM25 hits.

        Returning row indices (not chunk_ids) is deliberate: the fusion
        helpers (``_fuse_rrf`` / ``_fuse_wsum``) already speak in
        ``self.df`` row indices for the dense hits, and matching that
        here keeps the merge math trivial.

        Some notes:
          * I sanitise the query first because Tantivy's parser bails on
            a handful of characters that are perfectly normal in natural-
            language searches (``:`` is the usual offender).
          * If ``parse_query`` still throws such as the query is just
            punctuation after sanitising, I quietly return an empty
            result. The dense stream can still answer the request, and
            the user gets *some* hits instead of a 500.
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
        # normalise to [0,1] each stream
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

   
    # Public API
   
    def search(
        self, query: str, k: int = 5, mode: str | None = None
    ) -> List[Dict[str, Any]]:
        """
        Top-k hybrid search.

        The flow:
          - Encode the query and pull the dense top-N from FAISS (skipped
            in pure-bm25 mode).
          - Run the same query through Tantivy for BM25 (skipped in pure-
            dense mode).
          - Fuse the two streams using whichever strategy was requested
            (rrf | wsum | dense | bm25) and trim to k.
          - Hydrate the surviving row indices into result dicts the API
            can return as-is.

        ``mode`` overrides ``self.fusion`` for this one call; leaving it
        None falls back to the instance default. The early-skip on
        bm25/dense modes is deliberate — there's no point burning ~30 ms
        encoding the query through Sentence-Transformers if the caller
        only wants lexical hits.

        Thread safety: I hold ``self._lock`` for the whole call so a
        concurrent admin rebuild can't swap ``self.df`` / ``self.index``
        out from under us partway through a search.
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

            # Dense stream — only run when the chosen mode actually uses it.
            if mode in ("dense", "rrf", "wsum"):
                q = self._encode_query(query)
                vec_idx, vec_scores = self._dense_topn(q, max(k, self.topn_dense))

            # BM25 stream — same idea, skip if we're in pure-dense mode.
            if mode in ("bm25", "rrf", "wsum"):
                bm_idx, bm_scores = self._bm25_topn(query, max(k, self.topn_bm25))

            # Pick the fusion path (or skip fusion for the single-stream modes).
            if mode == "dense":
                final_idx = list(vec_idx[:k])
            elif mode == "bm25":
                final_idx = list(bm_idx[:k])
            elif mode == "wsum":
                final_idx = self._fuse_wsum(vec_idx, vec_scores, bm_idx, bm_scores, k)
            else:  # "rrf" default
                final_idx = self._fuse_rrf(vec_idx, vec_scores, bm_idx, bm_scores, k)

            # Hydrate the surviving row indices into result dicts.
            vec_idx_list = vec_idx.tolist()
            bm_idx_list = bm_idx.tolist()
            rows = []
            for i_row in final_idx:
                r = self.df.iloc[int(i_row)]
                # Surface whichever raw score we have for this row — dense
                # if the row came from FAISS, otherwise BM25, otherwise 0.
                # I deliberately don't expose the fused score here: it isn't
                # meaningful as a number to a user reading the UI.
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
