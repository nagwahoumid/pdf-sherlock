#!/usr/bin/env python3
"""
eval.py — Comparative Evaluation for PDF Sherlock Retrieval

Evaluates POST /search across three retrieval modes (bm25, dense, rrf) against
a ground-truth qrels file. Computes Precision@k, MRR, and latency per mode.
Writes per-mode result CSVs and prints a Summary Comparison Table for
dissertation results.

Usage:
  python eval.py --api-base http://127.0.0.1:8000 --k 10
  python eval.py --qrels eval/qrels.csv --out-dir eval --repeats 3

Output files:
  - results_bm25.csv
  - results_dense.csv
  - results_hybrid.csv  (rrf mode)
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    # Used for paired significance tests on per-query MRR. scipy is optional
    # at import time so the rest of the script still works on a box without
    # it; the tests themselves gracefully skip when HAS_SCIPY is False.
    from scipy.stats import wilcoxon
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

import pandas as pd


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

MODES = ["bm25", "dense", "rrf"]
MODE_OUTPUT_FILES = {
    "bm25": "results_bm25.csv",
    "dense": "results_dense.csv",
    "rrf": "results_hybrid.csv",
}
MODE_DISPLAY = {
    "bm25": "BM25",
    "dense": "Dense",
    "rrf": "RRF (Hybrid)",
}


# -----------------------------------------------------------------------------
# Ground truth and query loading
# -----------------------------------------------------------------------------

def load_qrels(path: Path, level: str) -> tuple[dict[str, str], dict[str, dict[str, int]]]:
    """
    Load qrels CSV and build:
    - queries: query_id -> query string
    - ground_truth: query_id -> { item_id -> relevance }
    item_id is doc_id for level "doc", or "doc_id::p{page}" for level "page".
    """
    df = pd.read_csv(path)
    required = {"query_id", "query", "doc_id", "relevance"}
    if not required.issubset(df.columns):
        raise SystemExit(f"qrels must have columns: {required}. Found: {list(df.columns)}")

    queries: dict[str, str] = {}
    ground_truth: dict[str, dict[str, int]] = {}

    for _, row in df.iterrows():
        qid = str(row["query_id"]).strip()
        query_str = str(row["query"]).strip()
        doc_id = str(row["doc_id"]).strip()
        raw_page = row.get("page")
        try:
            page = int(raw_page) if pd.notna(raw_page) and str(raw_page).strip() != "" else None
        except (ValueError, TypeError):
            page = None
        rel = int(row["relevance"]) if pd.notna(row["relevance"]) else 0

        if qid not in queries:
            queries[qid] = query_str
        if qid not in ground_truth:
            ground_truth[qid] = {}

        if level == "doc":
            key = doc_id
            ground_truth[qid][key] = max(ground_truth[qid].get(key, 0), rel)
        else:
            key = f"{doc_id}::p{page}" if page is not None else doc_id
            ground_truth[qid][key] = max(ground_truth[qid].get(key, 0), rel)

    return queries, ground_truth


def ranked_list_from_results(results: list, level: str) -> list[str]:
    """Build ordered list of item_ids from API results. Deduplicates by item_id."""
    seen: set[str] = set()
    out: list[str] = []
    for h in results:
        doc_id = str(h.get("doc_id", "")).strip()
        page = h.get("page")
        try:
            p = int(page) if page is not None else None
        except (ValueError, TypeError):
            p = None
        if level == "doc":
            item_id = doc_id
        else:
            item_id = f"{doc_id}::p{p}" if p is not None else doc_id
        if item_id not in seen:
            seen.add(item_id)
            out.append(item_id)
    return out


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def precision_at_k(retrieved: list[str], gt: dict[str, int], k: int) -> float:
    """# of relevant in top-k / k. Relevant = relevance >= 1."""
    top_k = retrieved[:k]
    n_relevant = sum(1 for i in top_k if gt.get(i, 0) >= 1)
    return n_relevant / k if k else 0.0


def mrr(retrieved: list[str], gt: dict[str, int]) -> float:
    """Reciprocal rank of first relevant (1/rank); 0 if none."""
    for rank, item_id in enumerate(retrieved, start=1):
        if gt.get(item_id, 0) >= 1:
            return 1.0 / rank
    return 0.0


def recall_at_k(retrieved: list[str], gt: dict[str, int], k: int) -> float:
    """
    Recall@k = |relevant ∩ top-k| / |relevant|.

    Counts any item with ``gt[item] >= 1`` as relevant, so the metric is
    compatible with graded-relevance qrels (e.g. PDF Sherlock's 0/2 scale).
    Queries with zero relevant items in the ground truth contribute 0.0;
    if you want them omitted, filter upstream.
    """
    relevant = {item for item, rel in gt.items() if rel >= 1}
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant)


def dcg_at_k(retrieved: list[str], gt: dict[str, int], k: int) -> float:
    """
    Discounted Cumulative Gain at k (Järvelin & Kekäläinen, 2002).

    DCG@k = Σ_{i=1..k} rel_i / log2(i + 1)

    We use the linear-gain form so that graded relevance values (0 / 2 in
    PDF Sherlock's qrels) flow through unchanged — a relevance-2 hit at
    rank 1 contributes ``2 / log2(2) = 2``, and at rank 2 contributes
    ``2 / log2(3) ≈ 1.26``. Missing items are treated as ``rel = 0``.
    """
    if k <= 0:
        return 0.0
    score = 0.0
    for i, item in enumerate(retrieved[:k], start=1):
        rel = gt.get(item, 0)
        if rel:
            score += rel / math.log2(i + 1)
    return score


def ndcg_at_k(retrieved: list[str], gt: dict[str, int], k: int) -> float:
    """
    Normalised Discounted Cumulative Gain at k.

    NDCG@k = DCG@k / IDCG@k, where IDCG@k is the DCG of the ideal ranking
    (all relevant items in descending relevance order, truncated to k).

    With PDF Sherlock's 0/2 graded qrels this correctly rewards a system
    for putting relevance-2 hits above relevance-0 hits: both the numerator
    and the denominator use the same graded-gain formula, so the result
    is in [0, 1] regardless of whether the qrels are binary or graded.
    Queries with no relevant items in the ground truth return 0.0.
    """
    if k <= 0:
        return 0.0
    dcg = dcg_at_k(retrieved, gt, k)
    ideal_rels = sorted(gt.values(), reverse=True)[:k]
    idcg = 0.0
    for i, rel in enumerate(ideal_rels, start=1):
        if rel:
            idcg += rel / math.log2(i + 1)
    return dcg / idcg if idcg > 0 else 0.0


# -----------------------------------------------------------------------------
# API call
# -----------------------------------------------------------------------------

def call_search(
    api_base: str,
    query: str,
    k: int,
    mode: str,
    timeout: int,
    verify_ssl: bool,
) -> tuple[list, float]:
    """
    POST /search with {"query": "...", "k": k, "mode": mode}.
    Returns (results list, latency_ms). Measures wall-clock latency.
    """
    url = f"{api_base.rstrip('/')}/search"
    payload = {"query": query, "k": k, "mode": mode}
    start = time.perf_counter()
    try:
        if HAS_REQUESTS:
            r = requests.post(
                url,
                json=payload,
                timeout=timeout,
                verify=verify_ssl,
            )
            r.raise_for_status()
            data = r.json()
        else:
            import json
            import urllib.request
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        results = data.get("results", [])
        return results, elapsed_ms
    except Exception as e:
        if HAS_REQUESTS and isinstance(e, requests.exceptions.ConnectionError):
            raise SystemExit(
                "Cannot reach the API. Is the FastAPI server running at "
                f"{api_base}? Error: {e}"
            ) from e
        if HAS_REQUESTS and isinstance(e, requests.exceptions.Timeout):
            raise SystemExit(f"Request timed out after {timeout}s. Error: {e}") from e
        raise SystemExit(f"Search request failed: {e}") from e


# -----------------------------------------------------------------------------
# Evaluation per mode
# -----------------------------------------------------------------------------

def run_mode(
    mode: str,
    api_base: str,
    k: int,
    queries: dict[str, str],
    ground_truth: dict[str, dict[str, int]],
    level: str,
    repeats: int,
    timeout: int,
    verify_ssl: bool,
) -> tuple[list[dict], float, float, float, float, float]:
    """
    Run all queries for one mode.

    Returns
    -------
    (rows, mean_pk, mean_mrr, mean_recall, mean_ndcg, mean_latency_ms)

    Each row in ``rows`` also carries per-query ``Recall@k`` and ``NDCG@k``
    values so downstream analysis (e.g. paired significance tests in
    ``run_comparative_evaluation``) can pull them straight out of the run
    without re-evaluating the API.
    """
    rows: list[dict] = []
    sum_pk = 0.0
    sum_mrr = 0.0
    sum_recall = 0.0
    sum_ndcg = 0.0
    all_latencies: list[float] = []

    for qid in sorted(queries.keys()):
        query_str = queries[qid]
        gt = ground_truth[qid]
        latencies: list[float] = []
        all_retrieved: list[list[str]] = []

        for _ in range(repeats):
            results, elapsed_ms = call_search(
                api_base, query_str, k, mode, timeout, verify_ssl
            )
            latencies.append(elapsed_ms)
            all_latencies.append(elapsed_ms)
            retrieved = ranked_list_from_results(results, level)
            all_retrieved.append(retrieved)

        retrieved = all_retrieved[0]
        mean_lat = sum(latencies) / len(latencies) if latencies else 0.0

        pk = precision_at_k(retrieved, gt, k)
        mrr_val = mrr(retrieved, gt)
        recall_val = recall_at_k(retrieved, gt, k)
        ndcg_val = ndcg_at_k(retrieved, gt, k)

        sum_pk += pk
        sum_mrr += mrr_val
        sum_recall += recall_val
        sum_ndcg += ndcg_val

        rows.append({
            "query_id": qid,
            "query": query_str,
            "P@k": round(pk, 4),
            "MRR": round(mrr_val, 4),
            "Recall@k": round(recall_val, 4),
            "NDCG@k": round(ndcg_val, 4),
            "latency_mean_ms": round(mean_lat, 2),
        })

    n_q = len(queries)
    mean_pk = sum_pk / n_q if n_q else 0.0
    mean_mrr = sum_mrr / n_q if n_q else 0.0
    mean_recall = sum_recall / n_q if n_q else 0.0
    mean_ndcg = sum_ndcg / n_q if n_q else 0.0
    mean_latency = sum(all_latencies) / len(all_latencies) if all_latencies else 0.0

    return rows, mean_pk, mean_mrr, mean_recall, mean_ndcg, mean_latency


# -----------------------------------------------------------------------------
# Main: loop modes, save CSVs, print summary table
# -----------------------------------------------------------------------------

def run_comparative_evaluation(
    api_base: str,
    k: int,
    qrels_path: Path,
    out_dir: Path,
    level: str,
    repeats: int,
    timeout: int,
    verify_ssl: bool,
) -> None:
    queries, ground_truth = load_qrels(qrels_path, level)
    if not queries:
        raise SystemExit("No queries found in qrels.")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_data: list[dict] = []
    # Per-mode, ordered-by-query-id MRR vectors for the paired Wilcoxon tests.
    # Keyed by mode (not display name) so the significance block below can
    # look up "bm25", "dense", "rrf" directly.
    per_query_mrr: dict[str, list[float]] = {}

    for mode in MODES:
        rows, mean_pk, mean_mrr, mean_recall, mean_ndcg, mean_latency = run_mode(
            mode=mode,
            api_base=api_base,
            k=k,
            queries=queries,
            ground_truth=ground_truth,
            level=level,
            repeats=repeats,
            timeout=timeout,
            verify_ssl=verify_ssl,
        )

        out_file = MODE_OUTPUT_FILES[mode]
        out_path = out_dir / out_file
        fieldnames = [
            "query_id",
            "query",
            "P@k",
            "MRR",
            "Recall@k",
            "NDCG@k",
            "latency_mean_ms",
        ]
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

        summary_data.append({
            "Mode": MODE_DISPLAY[mode],
            "Mean P@k": mean_pk,
            "Mean MRR": mean_mrr,
            "Mean Recall@k": mean_recall,
            "Mean NDCG@k": mean_ndcg,
            "Mean Latency (ms)": mean_latency,
        })
        # ``rows`` is already in sorted-query-id order (see run_mode), so the
        # resulting list pairs up element-wise across modes for Wilcoxon.
        per_query_mrr[mode] = [float(r["MRR"]) for r in rows]
        print(f"  [{mode}] Completed — {out_path}")

    # -------------------------------------------------------------------------
    # Paired significance tests (Wilcoxon signed-rank on per-query MRR)
    # -------------------------------------------------------------------------
    # We test two directional dissertation claims:
    #     Hybrid (RRF) MRR  vs  BM25 MRR
    #     Hybrid (RRF) MRR  vs  Dense MRR
    # The Wilcoxon test is non-parametric and paired, which is the right tool
    # for comparing two rankings produced by different systems over the same
    # set of queries. See scipy.stats.wilcoxon for the implementation.
    significance_lines: list[str] = []
    if not HAS_SCIPY:
        significance_lines.append(
            "  (scipy not installed — skipping Wilcoxon tests. "
            "Run: pip install scipy)"
        )
    else:
        def _wilcoxon_line(label: str, a: list[float], b: list[float]) -> str:
            if len(a) != len(b) or len(a) < 2:
                return f"  {label}: n/a (need >=2 paired observations)"
            try:
                stat, pval = wilcoxon(a, b)
            except ValueError as exc:
                # Raised when every paired difference is zero — i.e. the two
                # systems produced identical per-query MRR. Report that
                # explicitly rather than pretending we have a meaningful p.
                return f"  {label}: identical per-query MRR ({exc})"
            return (
                f"  {label}: W={stat:.4f}, p={pval:.4g}"
                f"  (n={len(a)})"
            )

        if "rrf" in per_query_mrr and "bm25" in per_query_mrr:
            significance_lines.append(
                _wilcoxon_line(
                    "Hybrid (RRF) vs BM25 [MRR]",
                    per_query_mrr["rrf"],
                    per_query_mrr["bm25"],
                )
            )
        if "rrf" in per_query_mrr and "dense" in per_query_mrr:
            significance_lines.append(
                _wilcoxon_line(
                    "Hybrid (RRF) vs Dense [MRR]",
                    per_query_mrr["rrf"],
                    per_query_mrr["dense"],
                )
            )

    # -------------------------------------------------------------------------
    # Summary Comparison Table
    # -------------------------------------------------------------------------
    print()
    print("=" * 92)
    print("  COMPARATIVE EVALUATION — Summary")
    print("=" * 92)
    print()
    print(f"  Queries: {len(queries)}  |  k: {k}  |  Repeats: {repeats}")
    print()
    print("-" * 92)
    print(
        f"  {'Mode':<16}  {'Mean P@k':>10}  {'Mean MRR':>10}  "
        f"{'Mean Recall@k':>14}  {'Mean NDCG@k':>12}  {'Mean Latency (ms)':>18}"
    )
    print("-" * 92)
    for row in summary_data:
        print(
            f"  {row['Mode']:<16}  "
            f"{row['Mean P@k']:>10.4f}  "
            f"{row['Mean MRR']:>10.4f}  "
            f"{row['Mean Recall@k']:>14.4f}  "
            f"{row['Mean NDCG@k']:>12.4f}  "
            f"{row['Mean Latency (ms)']:>18.2f}"
        )
    print("-" * 92)
    print()
    print("  Paired significance (Wilcoxon signed-rank on per-query MRR):")
    for line in significance_lines:
        print(line)
    print()
    print("  Output files:")
    for mode in MODES:
        print(f"    • {out_dir / MODE_OUTPUT_FILES[mode]}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Comparative evaluation of bm25, dense, and rrf retrieval modes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--api-base",
        default="http://127.0.0.1:8000",
        help="Base URL of the FastAPI server",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of results to request and evaluate at",
    )
    parser.add_argument(
        "--qrels",
        type=Path,
        default=Path("eval/qrels.csv"),
        help="Path to qrels CSV",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("eval"),
        help="Directory for output CSVs (results_bm25.csv, etc.)",
    )
    parser.add_argument(
        "--level",
        choices=("doc", "page"),
        default="doc",
        help="Evaluation granularity: doc or page",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of times to repeat each query for latency stats",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Request timeout in seconds",
    )
    parser.add_argument(
        "--no-verify-ssl",
        action="store_true",
        help="Disable SSL verification for the API",
    )
    args = parser.parse_args()

    if not args.qrels.exists():
        raise SystemExit(f"Qrels file not found: {args.qrels}")

    run_comparative_evaluation(
        api_base=args.api_base,
        k=args.k,
        qrels_path=args.qrels,
        out_dir=args.out_dir,
        level=args.level,
        repeats=args.repeats,
        timeout=args.timeout,
        verify_ssl=not args.no_verify_ssl,
    )


if __name__ == "__main__":
    main()
