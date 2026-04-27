#!/usr/bin/env python3
"""
make_qrels_from_runs.py — Build a skeleton qrels.csv from a run file for manual labeling.

Input run file format: query_id, query, doc_id, page, score
(one row per retrieved hit; multiple rows per query_id).
Output: qrels.csv with columns query_id, query, doc_id, page, relevance
where relevance is left blank for you to fill in (e.g. 0, 1, 2).

Usage:
  python eval/make_qrels_from_runs.py runs.csv -o eval/qrels_skeleton.csv
  python eval/make_qrels_from_runs.py runs.csv  # writes eval/qrels_from_runs.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create skeleton qrels CSV from a run file for manual relevance labeling.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "run_file",
        type=Path,
        help="Run file with columns: query_id, query, doc_id, page, score",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output qrels path (default: eval/qrels_from_runs.csv)",
    )
    args = parser.parse_args()

    if not args.run_file.exists():
        raise SystemExit(f"Run file not found: {args.run_file}")

    out = args.output or Path("eval/qrels_from_runs.csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    # Expected run columns; page and score may have different names in practice
    run_columns = ["query_id", "query", "doc_id", "page", "score"]
    qrels_columns = ["query_id", "query", "doc_id", "page", "relevance"]

    seen: set[tuple[str, str, str, str]] = set()
    rows: list[dict[str, str]] = []

    with open(args.run_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise SystemExit("Run file has no header.")
        # Normalize column names (strip whitespace, case-insensitive match)
        fieldnames = [c.strip() for c in reader.fieldnames]
        col_map = {c.lower(): c for c in fieldnames}
        # Map required run columns to actual names
        def get_col(name: str) -> str:
            key = name.lower()
            if key in col_map:
                return col_map[key]
            # Try without underscore
            alt = name.replace("_", "").lower()
            for fn in fieldnames:
                if fn.replace("_", "").lower() == alt:
                    return fn
            raise SystemExit(
                f"Run file must have a column for '{name}'. Found: {fieldnames}"
            )
        qid_col = get_col("query_id")
        query_col = get_col("query")
        doc_col = get_col("doc_id")
        page_col = None
        for fn in fieldnames:
            if fn.lower() in ("page", "page_no"):
                page_col = fn
                break

        for r in reader:
            # Use original key if we mapped
            qid = r.get(qid_col, r.get("query_id", "")).strip()
            query = r.get(query_col, r.get("query", "")).strip()
            doc_id = r.get(doc_col, r.get("doc_id", "")).strip()
            page_val = ""
            if page_col:
                p = r.get(page_col, "")
                if p is not None and str(p).strip() != "":
                    page_val = str(p).strip()
            key = (qid, query, doc_id, page_val)
            if key in seen:
                continue
            seen.add(key)
            rows.append({
                "query_id": qid,
                "query": query,
                "doc_id": doc_id,
                "page": page_val,
                "relevance": "",  # blank for manual labeling
            })

    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=qrels_columns)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out}. Fill the 'relevance' column (0/1/2...) and save as qrels.csv for eval.py.")


if __name__ == "__main__":
    main()
