# tests/test_index_store.py
from __future__ import annotations

import numpy as np
from app.index_store import IndexStore

def test_dense_search_returns_hits(store: IndexStore):
    hits = store.search("search api", k=3)
    assert 1 <= len(hits) <= 3
    # dict shape
    h0 = hits[0]
    for key in ("doc_id", "page", "score", "snippet", "chunk_id"):
        assert key in h0

def test_bm25_topn_bounds(store: IndexStore, monkeypatch):
    """
    Ask for more BM25 hits than corpus size; should NOT crash (earlier bug),
    and should return at most the corpus size.
    """
    m = store.df.shape[0]
    # Force very large topn_bm25
    store.topn_bm25 = 10_000
    hits = store.search("extract", k=5)
    assert len(hits) <= m

def test_hybrid_lifts_short_queries(store: IndexStore):
    """
    For the tiny corpus, the word 'extract' appears in the first row text.
    With hybrid (RRF default), ensure that row is in the top results.
    """
    hits = store.search("extract", k=3)
    doc_ids = [h["doc_id"] for h in hits]
    assert "docA" in doc_ids  # row containing 'Extract text per page...'
