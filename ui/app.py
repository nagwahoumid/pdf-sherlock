"""
ui/app.py

Minimal Streamlit UI for PDF Sherlock.

Layout:
- Sidebar: PDF upload helper (saves to ./data/pdfs) + ingest instructions.
- Main: Search form -> calls FastAPI /search and renders results.

Notes:
- st.file_uploader defaults to a 200 MB per-file limit; configure via server.maxUploadSize
  in .streamlit/config.toml. See Streamlit docs for details.
- st.number_input is used for 'k' (bounded).
- st.cache_data caches identical searches briefly for a snappier UX.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List

import requests
import streamlit as st
import pandas as pd


# -----------------------------------------------------------------------------
# App title and API base URL (editable so I can point to a remote server)
# -----------------------------------------------------------------------------
st.set_page_config(page_title="PDF Sherlock", page_icon=":)")
st.title("PDF Sherlock")

API_BASE = st.sidebar.text_input("API base URL", value="http://127.0.0.1:8000")


# -----------------------------------------------------------------------------
# Sidebar: Upload PDFs (saves locally) + ingest instructions
# -----------------------------------------------------------------------------
st.sidebar.subheader("Upload PDFs")

uploaded = st.sidebar.file_uploader(
    "Drop one or more PDFs",
    type=["pdf"],
    accept_multiple_files=True,
    help="Files are saved to ./data/pdfs then ingested into the index."
)

saved = []
if uploaded:
    save_dir = Path("data/pdfs")
    save_dir.mkdir(parents=True, exist_ok=True)
    for uf in uploaded:
        out_path = save_dir / uf.name
        with open(out_path, "wb") as f:
            f.write(uf.getbuffer())
        saved.append(str(out_path))
    st.sidebar.success(f"Saved {len(saved)} file(s) to {save_dir}")

# Optional cache clear for fresh results
if st.sidebar.button("Clear search cache"):
    st.cache_data.clear()
    st.success("Cache cleared.")

# One click ingest + rebuild (calls your admin endpoint)
if st.sidebar.button("Ingest & Rebuild now", type="primary", help="Run ingest, rebuild FAISS+BM25, and reload."):
    with st.spinner("Ingesting PDFs and rebuilding the index…"):
        try:
            payload = {
                "pdfs_dir": "data/pdfs",  # where we saved uploads
                "mode": "text",
                "sort": False,
                "size": 350,
                "overlap": 50,
                "min_chars": 20,
                "recurse": True,
                "compression": "snappy",
            }
            resp = requests.post(f"{API_BASE.rstrip('/')}/admin/ingest-rebuild", json=payload, timeout=600)
            resp.raise_for_status()
            info = resp.json()
            st.cache_data.clear()
            st.success(
                f"Rebuilt ✅  docs={info.get('num_docs')}  chunks={info.get('num_chunks')}  "
                f"took={info.get('took_ms')} ms"
            )
            # Fresh UI after rebuild
            st.rerun()
        except requests.HTTPError as e:
            st.error(f"Ingest/rebuild failed: {e.response.text}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")


st.sidebar.caption(
    "Tip: If you prefer manual steps, run "
    "`python -m app.ingest data/pdfs` then restart the API."
)


# -----------------------------------------------------------------------------
# Helpers: small wrappers around the HTTP API
# -----------------------------------------------------------------------------
def _search(base: str, query: str, k: int, mode: str) -> Dict[str, Any]:
    """Call POST /search and return the JSON dict (or raise for status)."""
    url = f"{base.rstrip('/')}/search"
    resp = requests.post(
        url,
        json={"query": query, "k": int(k), "mode": mode},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def _open_page(base: str, doc_id: str, page: int) -> Dict[str, Any]:
    """Call GET /open-page/{doc_id}/{page} and return path/page (or raise)."""
    url = f"{base.rstrip('/')}/open-page/{doc_id}/{int(page)}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


# Cache identical searches (same base, query, k, mode) for a short TTL
@st.cache_data(ttl=30)  # seconds
def cached_search(base: str, query: str, k: int, mode: str) -> Dict[str, Any]:
    return _search(base, query, k, mode)


# -----------------------------------------------------------------------------
# Retrieval-mode selector metadata
#
# Backend values match the /search API's ``mode`` field. The display labels
# are kept short but descriptive so non-engineering readers of the
# dissertation demo can tell the modes apart at a glance. Ordering here is
# also the display order in the selectbox: RRF first because it's the
# headline system in the dissertation.
# -----------------------------------------------------------------------------
_MODE_OPTIONS: List[tuple[str, str]] = [
    ("rrf",   "Hybrid RRF (Default)"),
    ("dense", "Dense (Semantic)"),
    ("bm25",  "BM25 (Lexical)"),
    ("wsum",  "Hybrid Weighted"),
]
_MODE_VALUES = [value for value, _label in _MODE_OPTIONS]
_MODE_LABELS = {value: label for value, label in _MODE_OPTIONS}


# -----------------------------------------------------------------------------
# Callbacks
#
# Wiring the "Open page" action through ``on_click`` instead of
# ``if st.button(...):`` has two benefits:
#
#   1. Callbacks run on Streamlit's *pre-rerun* pass, before any widget in
#      the main body has been re-rendered. That sidesteps the classic
#      "nested button state" problem where a button inside an expander only
#      registers its click on the *next* rerun, so the user has to click it
#      twice to see a result.
#   2. Errors raised inside the click handler no longer need to be caught in
#      the middle of the render loop, which keeps the loop body small and
#      predictable.
#
# We surface all user-visible feedback via ``st.toast`` so the notification
# floats over the results instead of pushing the layout around and causing a
# visual jump every time someone opens a PDF.
# -----------------------------------------------------------------------------
def open_pdf_callback(doc_id: str, page: int) -> None:
    """Resolve a hit's PDF path via the API and launch the OS default viewer."""
    try:
        res = _open_page(API_BASE, doc_id, int(page))
        pdf_path = res.get("path")
        resolved_page = res.get("page")
        st.toast(f"Opening: {pdf_path} (page {resolved_page})")

        # Hand the PDF to the OS so it opens in whatever viewer the user has
        # set as default (Preview on macOS, Acrobat / Edge on Windows,
        # Evince/Okular on Linux). Streamlit runs server-side, so this
        # launches the viewer on the *server* machine — which is the local
        # laptop in the common dev/dissertation-demo setup.
        if pdf_path:
            if sys.platform == "darwin":
                subprocess.run(["open", pdf_path], check=False)
            elif sys.platform == "win32":
                os.startfile(pdf_path)  # type: ignore[attr-defined]
            else:
                subprocess.run(["xdg-open", pdf_path], check=False)
    except requests.HTTPError as e:
        st.toast(f"Open-page failed: {e.response.text}", icon="⚠️")
    except Exception as e:
        st.toast(f"Open-page error: {e}", icon="⚠️")


# -----------------------------------------------------------------------------
# Main: search form
# -----------------------------------------------------------------------------
st.subheader("Search")

with st.form(key="search-form", clear_on_submit=False):
    query = st.text_input("Query", placeholder="e.g., binary search")

    # Put ``k`` and the retrieval-mode selector on one row so the form stays
    # compact and the two primary knobs live side-by-side.
    col_k, col_mode = st.columns(2)
    with col_k:
        k = st.number_input(
            "k (top results)", min_value=1, max_value=50, value=5, step=1
        )
    with col_mode:
        mode = st.selectbox(
            "Retrieval Mode",
            options=_MODE_VALUES,
            index=0,
            format_func=lambda value: _MODE_LABELS[value],
            help=(
                "Choose the retrieval backend: Hybrid RRF fuses BM25 + dense "
                "via Reciprocal Rank Fusion, Dense uses FAISS sentence "
                "embeddings, BM25 is pure lexical, and Hybrid Weighted does "
                "score-weighted fusion."
            ),
        )

    submitted = st.form_submit_button("Search")

if submitted:
    if not query.strip():
        st.warning("Please enter a non-empty query.")
    else:
        with st.spinner("Searching…"):
            try:
                data = cached_search(API_BASE, query.strip(), int(k), mode)
                hits: List[Dict[str, Any]] = data.get("results", [])
                took_ms = data.get("took_ms", None)

                # Header with quick stats (includes the chosen retrieval mode
                # so the user can tell which backend produced the results).
                st.success(
                    f"Found {len(hits)} result(s) via {_MODE_LABELS[mode]}"
                    f"{f' in {took_ms:.1f} ms' if took_ms is not None else ''}."
                )

                # Cards + optional table
                for i, h in enumerate(hits, start=1):
                    with st.expander(
                        f"{i}. {h.get('doc_id','?')} — page {h.get('page','?')} (score {h.get('score',0):.3f})",
                        expanded=(i == 1),
                    ):
                        st.write(h.get("snippet", ""))
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.button(
                                "Open page",
                                key=f"open-{i}",
                                on_click=open_pdf_callback,
                                args=(
                                    h.get("doc_id", ""),
                                    int(h.get("page", 1)),
                                ),
                            )
                        with c2:
                            st.caption(f"Chunk: `{h.get('chunk_id','')}`")
                        with c3:
                            if h.get("path"):
                                st.caption(f"File: `{h['path']}`")

                if hits:
                    df = pd.DataFrame(hits)
                    st.dataframe(df[["doc_id", "page", "score", "snippet"]])

                # Reproduce with curl (Bash/Zsh)
                payload = {"query": query.strip(), "k": int(k), "mode": mode}
                curl_cmd = (
                    f"curl -s {API_BASE.rstrip('/')}/search "
                    "-H 'content-type: application/json' "
                    f"-d '{json.dumps(payload)}' | jq ."
                )
                st.caption("Reproduce via curl (Bash/Zsh):")
                st.code(curl_cmd, language="bash")

                # PowerShell variant (Windows)
                ps_cmd = (
                    f"$body = '{json.dumps(payload)}'\n"
                    f"Invoke-WebRequest -UseBasicParsing -Uri '{API_BASE.rstrip('/')}/search' "
                    "-Method POST -ContentType 'application/json' -Body $body | "
                    "Select-Object -Expand Content"
                )
                st.caption("Reproduce via PowerShell:")
                st.code(ps_cmd, language="powershell")

            except requests.HTTPError as e:
                st.error(f"API error {e.response.status_code}: {e.response.text}")
            except requests.ConnectionError:
                st.error("Could not reach the API. Is it running at the URL above?")
            except Exception as e:
                st.error(f"Unexpected error: {e}")
