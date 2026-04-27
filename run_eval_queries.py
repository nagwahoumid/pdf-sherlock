"""
run_eval_queries.py — automated run-file generation for PDF Sherlock.

Hits the local FastAPI search endpoint with a fixed panel of 25 academic
queries (IR / NLP / RAG / dense encoders / Tantivy territory) in three modes
(BM25, dense, RRF) and writes per-mode CSV run files into ``eval/`` so they
can be fed straight into ``eval/make_qrels_from_runs.py`` to produce a
relevance-grading skeleton for the dissertation's evaluation methodology.

Query design
------------
The 25 queries are split across three difficulty tiers so the run files
exercise complementary parts of the retrieval stack:

1. Factoid / keyword  (q1-q8) — short, termly, BM25-friendly questions whose
   answer is typically a single formula, parameter name, or definition.
2. Conceptual         (q9-q16) — medium-length questions that probe
   understanding of a technique; dense retrieval should earn its keep here.
3. Broad / expert     (q17-q25) — open-ended survey-style prompts where the
   top-k recall of the hybrid engine (RRF over BM25 + dense) should matter
   most.

Run file schema
---------------
Each row represents one retrieved hit:

    query_id, query, doc_id, page, score

The ``query`` column is included because ``eval/make_qrels_from_runs.py``
(the downstream grading-skeleton tool referenced in the dissertation
methodology) requires it. The four columns the user explicitly asked for
(``query_id``, ``doc_id``, ``page``, ``score``) are all present.

Usage
-----
Start the API first (in another terminal):

    uvicorn app.api:app --reload

Then run this script:

    python run_eval_queries.py
    python run_eval_queries.py --url http://127.0.0.1:8000 --k 10 \\
        --out-dir eval --modes bm25 dense rrf
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

try:
    import requests
except ImportError as exc:  # pragma: no cover - executed only when missing
    raise SystemExit(
        "The 'requests' package is required. Install it with:\n"
        "    pip install requests\n"
        "or re-run: pip install -r requirements.txt"
    ) from exc


# -----------------------------------------------------------------------------
# Query panel
# -----------------------------------------------------------------------------

# Each entry is (query_id, category, query_text). Query IDs are contiguous
# q1..q25 so they sort numerically in the output CSVs; the category column is
# kept on the dataclass only so operators can filter/print by difficulty tier
# if desired.


@dataclass(frozen=True)
class EvalQuery:
    query_id: str
    category: str
    query: str


# --- 8 factoid / keyword queries --------------------------------------------
_FACTOID_QUERIES: Sequence[EvalQuery] = (
    EvalQuery("q1", "factoid", "What is the formula for Reciprocal Rank Fusion?"),
    EvalQuery("q2", "factoid", "BM25 term frequency saturation parameter k1"),
    EvalQuery("q3", "factoid", "What is the default value of the b parameter in BM25?"),
    EvalQuery("q4", "factoid", "Definition of nDCG@10"),
    EvalQuery("q5", "factoid", "FAISS HNSW M parameter meaning"),
    EvalQuery("q6", "factoid", "What does IDF stand for in information retrieval?"),
    EvalQuery("q7", "factoid", "Cosine similarity formula for sentence embeddings"),
    EvalQuery("q8", "factoid", "Number of layers in BERT-base"),
)

# --- 8 conceptual queries ---------------------------------------------------
_CONCEPTUAL_QUERIES: Sequence[EvalQuery] = (
    EvalQuery("q9", "conceptual", "How does chunk size affect retrieval accuracy?"),
    EvalQuery("q10", "conceptual", "Mitigating hallucinations in large language models"),
    EvalQuery("q11", "conceptual", "Why do dense retrievers underperform on out-of-domain data?"),
    EvalQuery("q12", "conceptual", "Trade-offs between recall and precision in hybrid search"),
    EvalQuery("q13", "conceptual", "How does query expansion improve sparse retrieval?"),
    EvalQuery("q14", "conceptual", "The role of negative sampling in training dense encoders"),
    EvalQuery("q15", "conceptual", "Impact of tokenization on multilingual retrieval"),
    EvalQuery("q16", "conceptual", "Why use late interaction models like ColBERT?"),
)

# --- 9 broad / expert queries -----------------------------------------------
_BROAD_QUERIES: Sequence[EvalQuery] = (
    EvalQuery("q17", "broad", "Challenges in offline document search"),
    EvalQuery("q18", "broad", "Cross-encoder vs Bi-encoder architectures"),
    EvalQuery("q19", "broad", "Evaluating retrieval systems beyond the BEIR benchmark"),
    EvalQuery("q20", "broad", "Scaling dense retrieval to billions of documents"),
    EvalQuery("q21", "broad", "Open problems in retrieval-augmented generation"),
    EvalQuery("q22", "broad", "Comparing learned sparse retrieval (SPLADE) with dense vectors"),
    EvalQuery("q23", "broad", "Best practices for building a local vector search engine"),
    EvalQuery("q24", "broad", "Role of re-ranking in modern search pipelines"),
    EvalQuery("q25", "broad", "How are inverted indexes implemented in Rust search engines?"),
)

DEFAULT_QUERIES: Sequence[EvalQuery] = (
    *_FACTOID_QUERIES,
    *_CONCEPTUAL_QUERIES,
    *_BROAD_QUERIES,
)
assert len(DEFAULT_QUERIES) == 25


# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------

DEFAULT_URL = "http://127.0.0.1:8000"
DEFAULT_K = 10
DEFAULT_OUT_DIR = Path("eval")
DEFAULT_MODES: Sequence[str] = ("bm25", "dense", "rrf")
# Filename suffix for each mode. RRF = "hybrid" in the dissertation prose.
_MODE_FILENAMES = {
    "bm25": "results_bm25.csv",
    "dense": "results_dense.csv",
    "rrf": "results_hybrid.csv",
    "wsum": "results_wsum.csv",
}
_RUN_COLUMNS = ["query_id", "query", "doc_id", "page", "score"]


# -----------------------------------------------------------------------------
# Core helpers
# -----------------------------------------------------------------------------


def _load_queries(path: Path | None) -> List[EvalQuery]:
    """
    Load queries from JSON if provided, otherwise fall back to the embedded
    25-query panel.

    JSON schema: a list of objects with ``query_id``, ``category``, ``query``.
    """
    if path is None:
        return list(DEFAULT_QUERIES)
    with path.open(encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, list):
        raise SystemExit(
            f"Queries file {path} must contain a JSON list of "
            "{query_id, category, query} objects."
        )
    return [
        EvalQuery(
            query_id=str(item["query_id"]),
            category=str(item.get("category", "")),
            query=str(item["query"]),
        )
        for item in raw
    ]


def _search(
    session: requests.Session,
    base_url: str,
    query: str,
    k: int,
    mode: str,
    timeout: float,
) -> List[dict]:
    """POST to /search and return the ``results`` list (possibly empty)."""
    resp = session.post(
        f"{base_url.rstrip('/')}/search",
        json={"query": query, "k": k, "mode": mode},
        timeout=timeout,
    )
    resp.raise_for_status()
    payload = resp.json()
    return list(payload.get("results", []))


def _write_run_csv(rows: Iterable[dict], path: Path) -> int:
    """Write ``rows`` to ``path`` using the fixed run-file schema. Returns row count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_RUN_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in _RUN_COLUMNS})
            n += 1
    return n


def run_evaluation(
    queries: Sequence[EvalQuery],
    base_url: str = DEFAULT_URL,
    k: int = DEFAULT_K,
    out_dir: Path = DEFAULT_OUT_DIR,
    modes: Sequence[str] = DEFAULT_MODES,
    timeout: float = 30.0,
) -> dict[str, Path]:
    """
    For each mode, run every query and write a CSV run file.

    Returns a dict ``{mode: output_path}``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}
    with requests.Session() as session:
        # Sanity check: warn loudly if the API isn't up rather than hammering
        # it with 25 * N_MODES failing POSTs.
        try:
            session.get(f"{base_url.rstrip('/')}/", timeout=5.0).raise_for_status()
        except requests.RequestException as exc:
            raise SystemExit(
                f"Could not reach PDF Sherlock API at {base_url}: {exc}\n"
                "Start it with: uvicorn app.api:app --reload"
            ) from exc

        for mode in modes:
            if mode not in _MODE_FILENAMES:
                print(
                    f"[eval] WARNING: unknown mode {mode!r}; writing to "
                    f"results_{mode}.csv",
                    file=sys.stderr,
                )
            out_path = out_dir / _MODE_FILENAMES.get(mode, f"results_{mode}.csv")
            print(f"\n[eval] mode={mode}  ->  {out_path}")

            rows: List[dict] = []
            t0 = time.time()
            for i, q in enumerate(queries, start=1):
                try:
                    hits = _search(session, base_url, q.query, k, mode, timeout)
                except requests.RequestException as exc:
                    print(
                        f"  [{i:2d}/{len(queries)}] {q.query_id} FAILED: {exc!r}",
                        file=sys.stderr,
                    )
                    continue

                for hit in hits:
                    rows.append(
                        {
                            "query_id": q.query_id,
                            "query": q.query,
                            "doc_id": hit.get("doc_id", ""),
                            "page": hit.get("page", ""),
                            "score": hit.get("score", ""),
                        }
                    )
                print(
                    f"  [{i:2d}/{len(queries)}] {q.query_id} "
                    f"({q.category:<10s}) -> {len(hits):2d} hits"
                )

            n = _write_run_csv(rows, out_path)
            print(
                f"[eval] wrote {n} rows for mode={mode} "
                f"in {time.time() - t0:.2f}s"
            )
            written[mode] = out_path

    return written


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run the 25-query evaluation panel against the local PDF "
            "Sherlock API and write per-mode CSV run files."
        )
    )
    p.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=f"Base URL of the PDF Sherlock API (default: {DEFAULT_URL})",
    )
    p.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        help=f"Number of hits per query (default: {DEFAULT_K})",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Directory to write run CSVs into (default: {DEFAULT_OUT_DIR})",
    )
    p.add_argument(
        "--modes",
        nargs="+",
        default=list(DEFAULT_MODES),
        help=(
            "Fusion modes to evaluate (default: bm25 dense rrf). "
            "rrf is written as results_hybrid.csv."
        ),
    )
    p.add_argument(
        "--queries-file",
        type=Path,
        default=None,
        help=(
            "Optional JSON file with a custom query list; schema is a list of "
            "{query_id, category, query}. Defaults to the embedded 25-query panel."
        ),
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Per-request HTTP timeout in seconds (default: 30).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    queries = _load_queries(args.queries_file)
    print(
        f"[eval] loaded {len(queries)} queries; "
        f"url={args.url}; k={args.k}; modes={args.modes}"
    )
    written = run_evaluation(
        queries=queries,
        base_url=args.url,
        k=args.k,
        out_dir=args.out_dir,
        modes=tuple(args.modes),
        timeout=args.timeout,
    )
    print("\n[eval] done. Output files:")
    for mode, path in written.items():
        print(f"  {mode:5s} -> {path}")


if __name__ == "__main__":
    main()
