"""


app/api.py

FastAPI backend

What this does:
- Serves a simple HTTP API over the local PDF vector index.
- Manages shared resources, like the IndexStore, on startup or shutdown.
- Keeps inputs validated and responses predictable for the UI/CLI.

"""

from __future__ import annotations
"""

Threading and OpenMP guards (these must run before FAISS, torch, and tantivy imports)

See app/index_store.py for the full context. I'm setting these here because 
api.py is our main entry point such as `uvicorn app.api:app`. This guarantees 
the guards are applied even if a future refactor accidentally pulls in 
FAISS and Tantivy before `index_store` runs. 
Using setdefault() so any operator-set shell values still win.


"""


import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

# Rebuilds can take several seconds on larger corpora, so I dispatch the
# heavy work to FastAPI BackgroundTasks and return 202 responses.


from fastapi import FastAPI, HTTPException, Request, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from pathlib import Path as FSPath

from .index_store import IndexStore  # loads chunk table, FAISS and encoder
from .ingest import ingest_dir, IngestConfig  # reuse ingest pipeline
from .models import (
    SearchRequest,
    SearchResponse,
    AdminIngestRequest,
)

logger = logging.getLogger(__name__)

# Lifespan — recommended way to manage startup andshutdown in FastAPI.
# Everything placed here runs before the app starts serving and after it stops.

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialise heavy andshared resources once (encoder, FAISS index, chunks table).
    # This avoids reloading models on every request and keeps latency low.

    # Experiment with the params here, I think there are some libraries that helps you with this like wandb, optuna, etc
    store = IndexStore(
        fusion="rrf",  # or "wsum", "dense", "bm25"
        alpha=0.6,  # used only when fusion="wsum"
        topn_dense=50,
        topn_bm25=200,
        model_name="sentence-transformers/msmarco-MiniLM-L-6-v3",  # retrieval-tuned
    )
    app.state.store = store
    try:
        yield
    finally:
        # Place for cleanup if needed like close DB connections.
        # in-memory FAISS and DataFrame require no explicit teardown.
        app.state.store = None


# Create the FastAPI app with the lifespan handler.
app = FastAPI(title="PDF Sherlock", version="0.1.0", lifespan=lifespan)

# If you don't need it, you can comment this out.
# FastAPI / Starlette CORS middleware reference.                               # :contentReference[oaicite:3]{index=3}
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # limit this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# Helpers

def _get_store(request: Request) -> IndexStore:
    """
    Retrieve the shared IndexStore from app state.
    Raises a 503 if the store isn't ready (e.g., during startup).
    """
    store: Optional[IndexStore] = getattr(request.app.state, "store", None)
    if store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Index not ready. Try again shortly.",
        )
    return store



# Routes

@app.get("/")
def health(request: Request):
    """
    Health probe used by tests and local dev.
    Returns whether the app is OK and some minimal stats if available.
    """
    ok = True
    # Stats are best-effort so they won't fail the health check if missing.
    num_chunks = num_docs = None
    store = getattr(request.app.state, "store", None)
    if store is not None and getattr(store, "df", None) is not None:
        num_chunks = int(store.df.shape[0])
        num_docs = int(store.df["doc_id"].nunique())
    return {"ok": ok, "num_docs": num_docs, "num_chunks": num_chunks}


@app.get("/version")
def version(request: Request):
    """
    Returns app version and the encoder model identifier for debugging.
    """
    store = _get_store(request)
    model_name = getattr(store, "model", None)
    model_name = getattr(model_name, "name_or_path", None) or "unknown"
    return {"app_version": request.app.version, "encoder": model_name}


@app.get("/stats")
def stats(request: Request):
    """
    Lightweight corpus statistics to verify ingest worked as expected.
    """
    store = _get_store(request)
    df = store.df
    return {
        "num_docs": int(df["doc_id"].nunique()),
        "num_chunks": int(df.shape[0]),
        "example_doc_ids": sorted(list(df["doc_id"].dropna().unique()))[:5],
    }


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest, request: Request):
    """
    Vector search endpoint.

    The flow is pretty straightforward:
    1. Sanity-check the inputs (don't bother searching for empty strings).
    2. Encode the query and hit the FAISS index. (here we normalise both the query 
       and corpus vectors ahead of time, which lets us use a fast inner-product 
       search to get cosine similarity).
    3. Take the raw hit IDs and "hydrate" them by pulling the actual text snippets 
       and metadata out of our Pandas DataFrame.

    Payload:
        - query: The actual search string.
        - k: Number of hits to return (hardcapped at 50 so we don't choke the UI).
        - mode: Lets the client override the default search strategy on the fly 
                (dense, bm25, rrf, wsum).
    """
    store = _get_store(request)

    # Basic input hygiene, trim whitespace. Blank queries return empty set.
    query = (req.query or "").strip()
    if not query:
        return SearchResponse(results=[], took_ms=0.0)

    t0 = time.time()
    # list[dict] with fields matching SearchHit
    hits = store.search(query, req.k, mode=req.mode)
    took_ms = (time.time() - t0) * 1000.0

    # Pydantic will coerce dicts to SearchHit automatically via response_model.  # :contentReference[oaicite:6]{index=6}
    return SearchResponse(results=hits, took_ms=round(took_ms, 3))


@app.get("/open-page/{doc_id}/{page}")
def open_page(doc_id: str, page: int, request: Request):
    """
    Resolve a document + page number to a local file path.

    UI/CLI can use the path to open a PDF viewer at that page (OS-dependent).
    If the (doc_id, page) pair doesn't exist, respond with 404 for clarity.
    """
    store = _get_store(request)
    row = store.df.query("doc_id == @doc_id & page_no == @page").head(1)
    if row.empty:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No entry for doc_id={doc_id!r} page={page}",
        )
    return {"path": row["path"].item(), "page": page}


def _run_ingest_rebuild(store: IndexStore, req: AdminIngestRequest) -> None:
    """
    Background worker that ingests PDFs and rebuilds FAISS + Tantivy in place.

    Runs outside the request/response cycle, so any exception raised here
    would otherwise be silently eaten by BackgroundTasks. We catch and log
    everything to make sure a failing rebuild cannot crash the server.

    The critical section where ``store.df``, ``store.index`` and the Tantivy
    handle are swapped is protected by ``store._lock`` so concurrent
    ``/search`` calls never observe a half-built index.
    """
    t0 = time.time()
    try:
        cfg = IngestConfig(
            mode=req.mode,
            sort=req.sort,
            chunk_size=req.size,
            chunk_overlap=req.overlap,
            min_chars=req.min_chars,
            recurse=req.recurse,
            compression=None if req.compression == "none" else req.compression,
        )

        # Ingest is CPU/IO heavy but doesn't touch shared state, so I run it outside the lock to keep /search responsive for as long as possible.
        ingest_dir(FSPath(req.pdfs_dir), out=store.chunks_path, cfg=cfg)

        # Load the new chunk table outside the lock as well, it only becomes visible once swapped in below.
        new_df = pd.read_parquet(store.chunks_path)
        if new_df.empty:
            logger.error(
                "admin_ingest_rebuild: chunk table is empty after ingest "
                "(pdfs_dir=%s). Aborting rebuild; keeping previous index.",
                req.pdfs_dir,
            )
            return

        # Critical section: swap DF + FAISS + Tantivy atomically from the point of view of readers.
        with store._lock:
            store.df = new_df
            store._build_and_persist_index()  # FAISS (dense)
            # Tantivy wipes the on-disk index dir and rebuilds from the freshly swapped-in self.df.
            store.tantivy_index = store._build_tantivy_index()

        took_ms = (time.time() - t0) * 1000.0
        logger.info(
            "admin_ingest_rebuild: completed in %.1f ms "
            "(num_docs=%d, num_chunks=%d)",
            took_ms,
            int(store.df["doc_id"].nunique()),
            int(store.df.shape[0]),
        )
    except (Exception, SystemExit):
        # Never let a background failure propagate — log with traceback and
        # leave the existing index untouched. We catch SystemExit too because
        # ingest_dir raises it when it cannot find any PDFs, which must not
        # terminate the server.
        logger.exception("admin_ingest_rebuild: background rebuild failed")


@app.post(
    "/admin/ingest-rebuild",
    status_code=status.HTTP_202_ACCEPTED,
)
def admin_ingest_rebuild(
    req: AdminIngestRequest,
    request: Request,
    background_tasks: BackgroundTasks,
):
    """
    Kick off an ingest and FAISS/BM25 rebuild in the background.

    Returns immediately with HTTP 202 so the caller such as the Streamlit UI
    does not block for the full duration of the rebuild. Progress and any
    errors are written to the server log.
    """
    store = _get_store(request)
    background_tasks.add_task(_run_ingest_rebuild, store, req)
    return {
        "status": "accepted",
        "message": "Rebuild started in background",
    }
