from pydantic import BaseModel, Field
from typing import List, Optional


# -----------------------------------------------------------------------------
# Pydantic models (request/response schemas)
# FastAPI uses these models to validate inputs and to produce OpenAPI & docs.    # :contentReference[oaicite:4]{index=4}
# -----------------------------------------------------------------------------
class SearchRequest(BaseModel):
    """Request body for /search."""

    query: str = Field(..., description="Natural language search query.")
    k: int = Field(5, ge=1, le=50, description="How many results to return (1–50).")
    mode: Optional[str] = Field(
        None,
        description="Override fusion mode: dense, bm25, rrf, wsum",
    )


class SearchHit(BaseModel):
    """One search result (a chunk mapped back to doc/page)."""

    doc_id: str
    page: int
    score: float
    snippet: str
    chunk_id: str
    path: Optional[str] = None  # local file path if available


class SearchResponse(BaseModel):
    """Response for /search."""

    results: List[SearchHit] = []
    took_ms: float = Field(..., description="Server-side elapsed time in milliseconds.")


# Admin: ingest and rebuild the index


class AdminIngestRequest(BaseModel):
    pdfs_dir: str = Field(..., description="Directory containing PDFs to ingest")
    # optional knobs; defaults mirror your IngestConfig
    mode: str = "text"
    # Match app/ingest.py's IngestConfig default: PyMuPDF sort=True produces
    # cleaner visual reading order (top-left -> bottom-right) on multi-column
    # academic PDFs, which is the primary corpus for PDF Sherlock.
    sort: bool = True
    size: int = 350
    overlap: int = 50
    min_chars: int = 20
    recurse: bool = True
    compression: str = "snappy"


class AdminIngestResponse(BaseModel):
    ok: bool
    num_docs: int
    num_chunks: int
    took_ms: float
