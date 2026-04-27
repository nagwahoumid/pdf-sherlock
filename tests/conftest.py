# tests/conftest.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any
import types

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

# Import your modules
import app.api as api_mod
from app.index_store import IndexStore


# ----------------------------- Fake encoder (fast) -----------------------------

@dataclass
class FakeEncoder:
    name_or_path: str = "tests/FakeEncoder"

    def encode(self, texts: List[str], convert_to_numpy=True, batch_size=64,
               show_progress_bar=False, normalize_embeddings=False):
        """
        Deterministic 'embeddings': simple bag-of-words counts over a tiny vocab,
        projected to fixed length. Good enough to test search logic without downloads.
        """
        if isinstance(texts, str):
            texts = [texts]
        vocab = ["extract", "search", "api", "pdf", "chunk", "bm25", "dense", "page"]
        embs = []
        for t in texts:
            t_low = (t or "").lower()
            vec = [t_low.count(tok) for tok in vocab]
            # pad to 16 dims so FAISS index shape is consistent
            vec = np.array(vec + [0] * (16 - len(vec)), dtype=np.float32)
            embs.append(vec)
        return np.vstack(embs) if convert_to_numpy else embs


# ----------------------------- Temp corpus / parquet ---------------------------

@pytest.fixture
def tiny_corpus_df() -> pd.DataFrame:
    rows = [
        dict(doc_id="docA", page_no=1, text="Extract text per page using pymupdf", chunk_id="A1", path="/abs/A.pdf"),
        dict(doc_id="docA", page_no=2, text="Chunk the page into overlapping windows", chunk_id="A2", path="/abs/A.pdf"),
        dict(doc_id="docB", page_no=3, text="Search API returns top k snippets", chunk_id="B3", path="/abs/B.pdf"),
        dict(doc_id="docB", page_no=4, text="BM25 lexical baseline and dense vectors", chunk_id="B4", path="/abs/B.pdf"),
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def temp_workspace(tmp_path, tiny_corpus_df):
    """Creates an isolated workspace with parquet + index paths."""
    root = tmp_path
    data_dir = root / "data"
    data_dir.mkdir()
    chunks_path = data_dir / "chunks.parquet"
    tiny_corpus_df.to_parquet(chunks_path)
    index_path = data_dir / "index.faiss"
    return dict(root=root, chunks_path=chunks_path, index_path=index_path)


# ----------------------------- IndexStore with fakes ---------------------------

@pytest.fixture
def store(temp_workspace, monkeypatch):
    # Patch SentenceTransformer used inside IndexStore to our FakeEncoder
    import app.index_store as idx_mod
    monkeypatch.setattr(idx_mod, "SentenceTransformer", lambda *a, **k: FakeEncoder())

    s = IndexStore(
        chunks_path=temp_workspace["chunks_path"],
        index_path=temp_workspace["index_path"],
        model_name="tests/FakeEncoder",
        rebuild=True,              # ensure we build fresh per test
        fusion="rrf",              # default hybrid mode
        topn_dense=10,
        topn_bm25=10,
    )
    return s


# ----------------------------- FastAPI TestClient ------------------------------

@pytest.fixture
def client(store, monkeypatch) -> TestClient:
    """
    Provide a TestClient where the app's store is replaced by our test store.
    We let FastAPI start, then override app.state.store.
    """
    app = api_mod.app
    with TestClient(app) as tc:
        app.state.store = store
        yield tc
