# PDF Sherlock

Interactive PDF retrieval stack that ingests local PDFs, chunks them into semantic vectors, and serves FastAPI + Streamlit tooling for natural-language search.

## Highlights


- **Vector retrieval API** – FastAPI service with lifespan-managed shared state, typed schemas, and endpoints for health, stats, semantic search, and page resolution (`app/api.py:42`, `app/api.py:116`, `app/api.py:157`, `app/api.py:183`).
- **IndexStore** – Thin wrapper around a Sentence-Transformers encoder, FAISS `IndexFlatIP`, and the chunk metadata DataFrame (`app/index_store.py:42`, `app/index_store.py:182`).
- **Ingestion CLI** – PyMuPDF-based pipeline that walks PDFs, extracts per-page text, chunks words with overlap, and writes a Parquet table for indexing (`app/ingest.py:48`, `app/ingest.py:77`, `app/ingest.py:121`).
- **Streamlit UI** – Simple front-end for uploads plus search cards backed by the API (`ui/app.py:32`, `ui/app.py:88`).
- **Smoke tests** – FastAPI TestClient coverage for health and search flows (`tests/test_api.py:4`).

## Repository Layout

| Path | Purpose |
| --- | --- |
| `app/` | Backend code: FastAPI app (`api.py`), ingest helpers, FAISS store. |
| `ui/` | Streamlit client (`app.py`) for upload + semantic search UX. |
| `data/` | Generated artifacts: `chunks.parquet`, `index.faiss`, optional `pdfs/`. |
| `tests/` | Pytest suite targeting the API surface. |
| `requirements.txt` | Runtime + dev dependencies, split by concern (`requirements.txt:1`). |
| `pyproject.toml` | Tooling configuration for Black, Ruff, and pytest. |

## Prerequisites

- Python 3.10+ (FAISS wheels ship for CPython on Windows/macOS/Linux).
- A working C++ redistributable (required by FAISS wheels, typically bundled).
- Optional GPU: set `device="cuda"` when you customize `IndexStore` instantiation.

## Quickstart

```bash
# 1. Create and activate a virtualenv
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Add PDFs under data/pdfs (or any folder you ingest)
mkdir -p data/pdfs
cp /path/to/*.pdf data/pdfs/
```

## Build the Chunk Table & Index

1. **Ingest PDFs**

   ```bash
   python -m app.ingest data/pdfs --out data/chunks.parquet
   ```

   Helpful flags (`python -m app.ingest --help`):
   - `--mode blocks|words` to adjust PyMuPDF extraction (`app/ingest.py:48`).
   - `--size` / `--overlap` to tune chunk windows (`app/ingest.py:77`).
   - `--no-recurse` to disable nested-folder traversal.
   - `--compression gzip|brotli|none` for the Parquet artifact.

2. **(Re)build FAISS index**

   `IndexStore` auto-builds `data/index.faiss` if it does not exist or if you set `rebuild=True` (`app/index_store.py:42`). To rebuild manually, delete `data/index.faiss` and restart the API.

## Run the FastAPI Service

```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
```

- Health probe: `curl http://127.0.0.1:8000/`
- OpenAPI docs: `http://127.0.0.1:8000/docs`
- Sample search:

  ```bash
  curl -s http://127.0.0.1:8000/search \
       -H "content-type: application/json" \
       -d '{"query": "binary search", "k": 5}' | jq .
  ```

## Run the Streamlit UI

```bash
streamlit run ui/app.py --server.port 8501
```

- Configure the API base URL in the sidebar (defaults to `http://127.0.0.1:8000`).
- Uploading files via the sidebar writes them to `data/pdfs/` and reminds you to rerun ingest (`ui/app.py:32`).
- Result cards include “Open page” buttons that call `GET /open-page/{doc_id}/{page}` (`ui/app.py:88`).

## Testing & Linting

```bash
pytest          # runs tests in tests/
ruff check app ui tests
black app ui tests
```

Pytest uses the configuration in `pyproject.toml`.

## API Surface

| Method | Path | Description |
| --- | --- | --- |
| `GET /` | Health & basic corpus counts. |
| `GET /version` | App version + encoder identifier. |
| `GET /stats` | Doc/chunk counts + sample doc IDs. |
| `POST /search` | Vector search with `{"query": str, "k": int}` body, returns scored snippets (`app/api.py:157`). |
| `GET /open-page/{doc_id}/{page}` | Resolves local path + page for UI deep links (`app/api.py:183`). |

Responses use Pydantic models for validation and automatic docs.

## Troubleshooting

- **`data/chunks.parquet` missing** – Run the ingest CLI first; the API raises a clear error otherwise (`app/index_store.py:63`).
- **Empty results** – Check ingest stats for skipped/empty pages; adjust `--mode` or OCR PDFs before ingesting.
- **FAISS row mismatch** – Delete `data/index.faiss` and restart to rebuild against the latest chunk table (`app/index_store.py:132`).
- **Large PDFs** – Increase Streamlit’s upload limit via `.streamlit/config.toml` (see sidebar note in `ui/app.py`).

## Next Steps

1. Gate the API with authentication and tighten `allow_origins` before deploying.
2. Add an admin endpoint or CLI toggle to rebuild the FAISS index without restarting.
3. Extend tests with fixture data for deterministic search assertions.

