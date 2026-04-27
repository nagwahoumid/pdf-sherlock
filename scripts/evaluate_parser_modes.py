"""
scripts/evaluate_parser_modes.py

Benchmark a handful of PyMuPDF text-extraction configurations on the PDFs in
a directory, without touching the production ingestion pipeline.

Why does this matter for retrieval quality?
-------------------------------------------
Downstream dense + BM25 retrieval is only as good as the text it sees. If the
parser returns words in the wrong reading order (two-column papers are the
classic offender), chunks end up mixing unrelated sentences, embeddings get
noisier, and BM25 term proximity goes out the window. Before we change the
default parser in app/ingest.py, this script gives us cheap, quantitative
signal on three candidate configurations:

    1) mode="text",   sort=False  -- current default (creator's internal order)
    2) mode="text",   sort=True   -- reorder top-left to bottom-right
    3) mode="blocks", sort=True   -- block-positioned extraction, reading-ordered

The script is intentionally read-only: it does NOT modify ingest.py, does not
rebuild any index, and just writes a CSV plus a printed summary.

Usage
-----
    python scripts/evaluate_parser_modes.py \
        --pdfs-dir data/pdfs \
        --out data/parser_eval.csv \
        --sample-pages 5
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import fitz  # PyMuPDF
import pandas as pd

# Each tuple is (mode, sort). Keep this small and explicit rather than a full
# cross-product -- we only want the three configurations the reviewer asked
# about so the CSV stays readable.
CONFIGS: List[Tuple[str, bool]] = [
    ("text", False),
    ("text", True),
    ("blocks", True),
]


def _iter_pdfs(pdfs_dir: Path) -> Iterable[Path]:
    """Yield PDF paths under pdfs_dir (recursive), sorted for determinism."""
    return sorted(p for p in pdfs_dir.rglob("*.pdf") if p.is_file())


# Matches any character that isn't alphanumeric, dash, underscore, or dot.
# Used to keep sample-text filenames portable across macOS / Linux / Windows.
_UNSAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_stem(name: str) -> str:
    """
    Turn an arbitrary PDF stem into something safe to use in a filename.

    Replaces runs of unsafe characters with a single underscore and trims
    leading/trailing underscores so we don't produce paths like
    "__foo__.txt".
    """
    cleaned = _UNSAFE_FILENAME_CHARS.sub("_", name).strip("_")
    return cleaned or "untitled"


def _sample_text_path(sample_dir: Path, pdf_path: Path, mode: str, sort: bool) -> Path:
    """
    Build the per-(pdf, config) sample-text path, e.g.
    ``sample_dir/papername__mode-text__sort-false.txt``.
    """
    stem = _safe_stem(pdf_path.stem)
    sort_str = "true" if sort else "false"
    return sample_dir / f"{stem}__mode-{mode}__sort-{sort_str}.txt"


def _extract_page_text(page: "fitz.Page", mode: str, sort: bool) -> str:
    """
    Run page.get_text for the given mode and coerce the result to a single
    string so we can measure character counts uniformly.

    For mode="blocks", get_text returns a list of tuples where index 4 is
    conventionally the text payload, but older/newer PyMuPDF releases have
    shipped slightly different tuple layouts. We defensively only pick up
    element [4] if it's actually a string.
    """
    if mode == "blocks":
        blocks = page.get_text("blocks", sort=sort)
        parts: List[str] = []
        for block in blocks:
            if len(block) > 4 and isinstance(block[4], str):
                parts.append(block[4])
        return "\n".join(parts)

    # All other modes that return a string (text, words rendered as string, etc.)
    result = page.get_text(mode, sort=sort)
    return result if isinstance(result, str) else str(result or "")


def _evaluate_one(
    pdf_path: Path,
    mode: str,
    sort: bool,
    sample_pages: int,
    sample_text_dir: Optional[Path] = None,
) -> dict:
    """
    Time extraction for a single (pdf, config) pair, capturing per-PDF stats.

    Returns a dict with one row's worth of data. On failure, the "error"
    column is populated and the numeric columns are filled with zero / None
    so the overall DataFrame shape stays consistent.

    If ``sample_text_dir`` is provided, the extracted text for every sampled
    page is ALSO appended to a per-(pdf, config) ``.txt`` file inside that
    directory. The text written to disk is produced by the exact same
    ``_extract_page_text`` call that feeds the CSV metrics, so manual
    inspection always matches the numbers.
    """
    row = {
        "file": pdf_path.name,
        "mode": mode,
        "sort": sort,
        "pages_sampled": 0,
        "total_chars": 0,
        "empty_pages": 0,
        "extraction_ms": 0.0,
        "avg_chars_per_page": 0.0,
        "error": None,
    }

    # Collected in-memory first so we don't leave a half-written .txt file on
    # disk if extraction blows up halfway through.
    sample_chunks: List[str] = []

    try:
        t0 = time.perf_counter()
        total_chars = 0
        empty_pages = 0
        pages_sampled = 0

        # Open inside the try block -- a corrupt/encrypted PDF will raise here
        # and we want that to surface as an "error" row, not a crash.
        with fitz.open(pdf_path) as doc:
            # sample_pages <= 0 is treated as "no cap" to make the CLI flexible.
            max_pages = len(doc) if sample_pages <= 0 else min(len(doc), sample_pages)
            for i in range(max_pages):
                page = doc[i]
                text = _extract_page_text(page, mode=mode, sort=sort)
                if not text.strip():
                    empty_pages += 1
                total_chars += len(text)
                pages_sampled += 1

                # 1-based page numbers are what the rest of the app uses
                # (see app/ingest.py::page_texts), so keep them consistent
                # here too.
                if sample_text_dir is not None:
                    sample_chunks.append(f"----- Page {i + 1} -----\n{text}\n")

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        avg = (total_chars / pages_sampled) if pages_sampled else 0.0

        row.update(
            {
                "pages_sampled": pages_sampled,
                "total_chars": total_chars,
                "empty_pages": empty_pages,
                "extraction_ms": round(elapsed_ms, 3),
                "avg_chars_per_page": round(avg, 2),
            }
        )
    except Exception as exc:
        # Stash the message so the reviewer can grep the CSV for problem PDFs.
        row["error"] = f"{type(exc).__name__}: {exc}"

    # Only write the sample file if extraction succeeded AND the user asked
    # for it. A failed PDF has no meaningful text to inspect.
    if sample_text_dir is not None and row["error"] is None:
        try:
            out_path = _sample_text_path(sample_text_dir, pdf_path, mode, sort)
            header = (
                f"PDF: {pdf_path.name}\n"
                f"Mode: {mode}\n"
                f"Sort: {sort}\n"
                f"Pages sampled: {row['pages_sampled']}\n"
                f"{'=' * 72}\n\n"
            )
            out_path.write_text(header + "\n".join(sample_chunks), encoding="utf-8")
        except OSError as exc:
            # Don't lose the CSV row over a filesystem hiccup -- just record
            # the write failure alongside any successful extraction metrics.
            row["error"] = f"sample_text_write_failed: {exc}"

    return row


def _print_summary(df: pd.DataFrame) -> None:
    """Print a compact per-config summary to stdout."""
    if df.empty:
        print("No rows collected -- nothing to summarise.")
        return

    # Only aggregate successful rows; broken PDFs would skew the means.
    ok = df[df["error"].isna()]
    if ok.empty:
        print("All rows had errors -- see the 'error' column in the CSV.")
        return

    grouped = (
        ok.groupby(["mode", "sort"], as_index=False)
        .agg(
            pdfs=("file", "count"),
            avg_extraction_ms=("extraction_ms", "mean"),
            avg_empty_pages=("empty_pages", "mean"),
            avg_chars_per_page=("avg_chars_per_page", "mean"),
        )
        .round(2)
    )

    print("\nParser configuration summary (successful PDFs only):")
    print(grouped.to_string(index=False))

    n_errors = int(df["error"].notna().sum())
    if n_errors:
        print(f"\n{n_errors} row(s) had extraction errors -- inspect the CSV.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare PyMuPDF parser configurations on a directory of PDFs.",
    )
    parser.add_argument(
        "--pdfs-dir",
        type=Path,
        required=True,
        help="Directory containing PDF files (searched recursively).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/parser_eval.csv"),
        help="CSV output path (default: data/parser_eval.csv).",
    )
    parser.add_argument(
        "--sample-pages",
        type=int,
        default=5,
        help="Max pages per PDF to extract (default: 5; use 0 for no cap).",
    )
    parser.add_argument(
        "--sample-text-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory to dump extracted sample text (one .txt per "
            "PDF x config) for manual reading-order inspection."
        ),
    )
    args = parser.parse_args()

    pdfs_dir: Path = args.pdfs_dir.expanduser().resolve()
    if not pdfs_dir.exists() or not pdfs_dir.is_dir():
        raise SystemExit(f"--pdfs-dir not found or not a directory: {pdfs_dir}")

    sample_text_dir: Optional[Path] = None
    if args.sample_text_dir is not None:
        sample_text_dir = args.sample_text_dir.expanduser().resolve()
        sample_text_dir.mkdir(parents=True, exist_ok=True)

    pdfs = list(_iter_pdfs(pdfs_dir))
    if not pdfs:
        raise SystemExit(f"No PDFs found under {pdfs_dir}")

    print(f"Evaluating {len(pdfs)} PDF(s) across {len(CONFIGS)} configuration(s)...")
    if sample_text_dir is not None:
        print(f"Sample text will be written to {sample_text_dir}")

    rows: List[dict] = []
    for pdf in pdfs:
        for mode, sort in CONFIGS:
            rows.append(
                _evaluate_one(
                    pdf,
                    mode=mode,
                    sort=sort,
                    sample_pages=args.sample_pages,
                    sample_text_dir=sample_text_dir,
                )
            )

    df = pd.DataFrame(rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} row(s) to {args.out}")

    _print_summary(df)


if __name__ == "__main__":
    main()
