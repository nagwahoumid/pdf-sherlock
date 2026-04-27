"""
Nagwa Houmid El Amrani
Date : 26/03/2026

app/ingest.py

This module or file provides a pipeline for extracting textual content from PDF documents,
segmenting it into overlapping chunks, and persisting the result as a compressed
Parquet table. The output serves as the primary input for the downstream search
index (IndexStore).

Design and Implementation Decisions:

Text extraction is performed using PyMuPDF (fitz). The library offers several
extraction modes, including "text", "blocks", and "words". The default mode,
"text", returns a continuous string representation of the page content. For
documents with complex layouts such as multi‑column academic papers, enabling
the sort=True parameter instructs PyMuPDF to reorder text fragments
spatially (top to bottom and left to right), which substantially improves reading
order fidelity. Alternative modes such as "blocks" or "words" are available for
specialised use cases and can be configured via the IngestConfig dataclass.

Pages that consist entirely of raster images scanned documents typically
yield empty strings when queried with get_text(). In the current
implementation, such pages are skipped and tallied in the ingestion statistics.
Future extensions may incorporate optical character recognition (OCR) to recover
text from these pages.

Chunking is layout aware. Instead of slicing a page's text at arbitrary word
boundaries, PDF Sherlock queries PyMuPDF in "blocks" mode to obtain the list of
paragraph-like text blocks that the parser already identifies during layout
analysis (get_text("blocks", sort=cfg.sort) returns one tuple per block, whose
fifth element is the text and whose seventh element is the block type; only
text blocks with block_type == 0 are retained). The chunker then concatenates
whole blocks together until the accumulated word count approaches or slightly
exceeds cfg.chunk_size, so sentences are never split mid-paragraph. To preserve
semantic context across boundaries, the tail of each chunk (roughly
cfg.chunk_overlap words, always aligned to whole blocks) is repeated as the
head of the next chunk. The net effect is a document decomposition that
respects the visual structure of the page while still producing chunks of a
size suitable for dense retrieval.

The processed data is serialised to the Parquet format, a columnar storage
layout that provides efficient read performance and native integration with the
PyData ecosystem. Compression is applied via the Snappy codec (default) to
reduce on‑disk footprint, alternative codecs such as Zstandard or Gzip may be
selected through the configuration object.

Output Schema:

The generated Parquet table contains one row per chunk with the following
columns:

    chunk_id : str
        Universally unique identifier (UUID4) for the chunk.
    doc_id : str
        Stable document identifier equal to the file stem (the filename
        without its extension).
    page_no : int
        1 based page number from which the chunk originates.
    text : str
        The textual content of the chunk (one or more concatenated blocks,
        separated by a blank line).
    start : int
        Index (inclusive) of the first layout block on the page that is
        represented in this chunk.
    end : int
        Index (exclusive) of the block one past the last layout block included
        in this chunk. Consecutive chunks on the same page typically overlap in
        this index range, mirroring the overlap in their text.
    path : str
        Absolute filesystem path to the source PDF.

Command‑Line Interface:

The module may be invoked directly to ingest a directory of PDFs:

    python -m app.ingest /path/to/pdfs --out data/chunks.parquet

Optional flags allow overriding the default chunking and extraction parameters.
Run with --help for a complete list of options.

"""

from __future__ import (
    annotations,
)  # this is for type hints to behave correctly 

import argparse  # this is for command line arguments
from dataclasses import dataclass  
from pathlib import Path
from typing import Generator, Iterable, List, Tuple
import uuid

import fitz  # (fitz is the PyMuPDF library)
import pandas as pd

# ------------------------------- configuration -------------------------------


@dataclass(frozen=True)
class IngestConfig:
    """Static knobs for ingestion."""

    mode: str = "blocks"  # PyMuPDF get_text mode; "blocks" enables layout aware chunking
    sort: bool = True  # PyMuPDF visual reading order (top-left -> bottom-right)
    chunk_size: int = 350  # target word count per chunk (whole blocks, may slightly exceed)
    chunk_overlap: int = 50  # approx word overlap (aligned to whole blocks)
    min_chars: int = 20  # skip very short chunks
    recurse: bool = True  # traverse subfolders
    compression: str = (
        "snappy"  # parquet compression ("snappy", "gzip", "brotli", None)
    )


# ---------------------------------------------------------------
# Below are the core helpers which are the functions that are used to extract
# layout blocks from a PDF and group them into overlapping chunks.

# PDF Sherlock uses PyMuPDF "blocks" extraction with sort=True by default
# because it exposes the parser's own paragraph-level layout analysis. The
# parser mode evaluation (see scripts/evaluate_parser_modes.py) showed that
# sort=True gives the cleanest overall reading order for sampled academic PDFs
# and that grouping whole blocks avoids the mid-sentence splits produced by a
# naive word-window chunker.


# Block type 0 in PyMuPDF's "blocks" mode denotes a text block; type 1 denotes
# an image block. Only text blocks are meaningful for retrieval.
_TEXT_BLOCK_TYPE = 0


def page_blocks(
    pdf_path: Path, sort: bool = True
) -> Generator[Tuple[int, List[str]], None, None]:
    """
    Yield (1 based page number, list_of_block_texts) for each page in a PDF.

    The text of each block is taken from element 4 of the block tuple returned
    by PyMuPDF's page.get_text("blocks", sort=sort); element 6 is the block
    type, and only text blocks (block_type == 0) are kept. Empty or whitespace
    only blocks are discarded so downstream code can rely on non trivial text.

    Parameters
    ----------
    pdf_path : Path
        Path to a PDF file.
    sort : bool
        If True, PyMuPDF returns blocks in visual reading order (top-left to
        bottom-right), which generally improves reading order fidelity on
        multi-column academic papers.

    Yields
    ------
    (int, list[str])
        page number (1-based) and the list of paragraph-level text blocks on
        that page (may be empty on scanned or image-only pages).
    """
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc, start=1):
            raw_blocks = page.get_text("blocks", sort=sort)
            texts: List[str] = []
            for b in raw_blocks:
                # b = (x0, y0, x1, y1, text, block_no, block_type)
                if len(b) < 7:
                    continue
                block_type = b[6]
                if block_type != _TEXT_BLOCK_TYPE:
                    continue
                text = (b[4] or "").strip()
                if not text:
                    continue
                texts.append(text)
            yield i, texts


def chunk_blocks(
    blocks: List[str], size: int = 350, overlap: int = 50
) -> Generator[Tuple[str, int, int], None, None]:
    """
    Layout-aware chunker: group whole paragraph-level blocks into chunks.

    The algorithm walks forward through the list of blocks, accumulating them
    into the current chunk until the running word count reaches or slightly
    exceeds ``size``. Blocks are never split mid-sentence. Once a chunk is
    emitted, the next chunk is seeded with the trailing block(s) of the
    previous chunk whose combined word count is at least ``overlap`` (aligned
    to whole blocks), so semantic context carries over the boundary. A single
    block larger than ``size`` is emitted as its own chunk rather than being
    cut.

    Parameters
    ----------
    blocks : list[str]
        Paragraph-level text blocks in reading order (as produced by
        :func:`page_blocks`).
    size : int
        Target minimum word count per chunk.
    overlap : int
        Approximate number of words to repeat from the end of a chunk at the
        start of the next chunk. Rounded up to the nearest whole block.

    Yields
    ------
    (str, int, int)
        ``(chunk_text, start_block_idx, end_block_idx)`` where
        ``chunk_text`` is the blocks joined by a blank line, and the indices
        refer to positions in ``blocks`` (``end`` is exclusive). Consecutive
        chunks may have overlapping block index ranges.
    """
    if not blocks:
        return

    word_counts: List[int] = [len(b.split()) for b in blocks]
    n = len(blocks)

    size = max(1, size)
    overlap = max(0, overlap)

    i = 0
    while i < n:
        # Grow the window of whole blocks until we reach the target size
        # or run out of blocks. Always include at least one block so that
        # oversized blocks are not dropped or split.
        current_words = 0
        j = i
        while j < n and (j == i or current_words < size):
            current_words += word_counts[j]
            j += 1

        chunk_text = "\n\n".join(blocks[i:j])
        yield chunk_text, i, j

        if j >= n:
            break

        # Determine the overlap slice: walk back from j until we have gathered
        # roughly `overlap` words worth of trailing blocks. Always advance by
        # at least one block to guarantee forward progress.
        back_words = 0
        k = j
        while k > i + 1 and back_words < overlap:
            k -= 1
            back_words += word_counts[k]
        if k <= i:
            k = i + 1
        i = k


def _doc_id_for(path: Path) -> str:
    """
    Return a stable doc_id equal to the file name without its extension.

    Using the plain stem keeps doc_ids human-readable and, crucially, stable
    across re-ingestions: neither file size nor modification time influences
    the identifier, so evaluation artefacts such as eval/qrels.csv remain
    valid when the corpus is re-indexed. This assumes filenames within the
    corpus are unique, which is the case for PDF Sherlock's curated set.
    """
    return path.stem


def _pdf_iter(root: Path, recurse: bool = True) -> Iterable[Path]:
    """Iterate PDFs under a directory (optionally recursively)."""
    pattern = "**/*.pdf" if recurse else "*.pdf"
    yield from sorted(root.glob(pattern))


# -------------------------------------------------------------------
#This is the main function that is used to ingest the PDFs into the Parquet file.

def ingest_dir(
    pdfs_dir: Path,
    out: Path | str = "data/chunks.parquet",
    cfg: IngestConfig = IngestConfig(),
) -> Path:
    """
    Walk a folder of PDFs, extract per-page text, chunk into overlapping windows,
    and write a Parquet file with one row per chunk.

    Columns
    -------
    chunk_id : str (uuid4)
    doc_id   : str (equal to the file stem; stable across re-ingestions)
    page_no  : int (1-based)
    text     : str (one or more concatenated layout blocks)
    start    : int (first block index on the page, inclusive)
    end      : int (one past the last block index on the page, exclusive)
    path     : str (absolute file path)

    Returns
    -------
    Path to the written Parquet file.
    """
    pdfs_dir = Path(pdfs_dir).expanduser().resolve()
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    stats = {"files": 0, "pages": 0, "empty_pages": 0, "chunks": 0}

    for p in _pdf_iter(pdfs_dir, recurse=cfg.recurse):
        stats["files"] += 1
        doc_id = _doc_id_for(p)

        for page_no, blocks in page_blocks(p, sort=cfg.sort):
            stats["pages"] += 1

            # Skip pages with no usable text blocks (common in scanned PDFs
            # or pages that are purely figures).
            if not blocks:
                stats["empty_pages"] += 1
                continue

            for text, start, end in chunk_blocks(
                blocks, size=cfg.chunk_size, overlap=cfg.chunk_overlap
            ):
                if len(text) < cfg.min_chars:
                    continue
                rows.append(
                    {
                        "chunk_id": str(uuid.uuid4()),
                        "doc_id": doc_id,
                        "page_no": int(page_no),
                        "text": text,
                        "start": int(start),
                        "end": int(end),
                        "path": str(p),
                    }
                )
                stats["chunks"] += 1

    if not rows:
        raise SystemExit(
            f"No chunks produced. Checked {stats['files']} files / {stats['pages']} pages "
            f"({stats['empty_pages']} empty). Are the PDFs text-based?"
        )

    df = pd.DataFrame(rows)
    # Tip: specify compression (snappy default) and let pandas auto-pick engine (pyarrow/fastparquet).  # to_parquet
    df.to_parquet(
        out_path, compression=cfg.compression
    )  # requires pyarrow or fastparquet installed
    print(
        f"[ingest] wrote {stats['chunks']} chunks from {stats['files']} files "
        f"({stats['pages']} pages, {stats['empty_pages']} empty) -> {out_path}"
    )
    return out_path


# ------------------------------------------------------------------------
#This is the CLI (Command Line Interface) that is used to ingest the PDFs into the Parquet file.


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingest PDFs into a Parquet chunk table.")
    p.add_argument("pdfs_dir", type=Path, help="Directory containing PDFs")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("data/chunks.parquet"),
        help="Output Parquet path",
    )
    p.add_argument(
        "--sort",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "PyMuPDF sort=True requests visual reading order "
            "(roughly top-left to bottom-right) when extracting layout "
            "blocks. This is the default; pass --no-sort to fall back to "
            "the PDF creator's internal block order."
        ),
    )
    p.add_argument(
        "--size",
        type=int,
        default=350,
        help="Target chunk size in words (whole blocks, may slightly exceed)",
    )
    p.add_argument(
        "--overlap",
        type=int,
        default=50,
        help="Approximate word overlap between chunks (aligned to whole blocks)",
    )
    p.add_argument(
        "--min-chars",
        type=int,
        default=20,
        help="Drop chunks shorter than this many characters",
    )
    p.add_argument(
        "--no-recurse", action="store_true", help="Do not search subfolders for PDFs"
    )
    p.add_argument(
        "--compression",
        default="snappy",
        choices=["snappy", "gzip", "brotli", "none"],
        help="Parquet compression (default: snappy)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = IngestConfig(
        sort=args.sort,
        chunk_size=args.size,
        chunk_overlap=args.overlap,
        min_chars=args.min_chars,
        recurse=not args.no_recurse,
        compression=None if args.compression == "none" else args.compression,
    )
    ingest_dir(args.pdfs_dir, out=args.out, cfg=cfg)


if __name__ == "__main__":
    main()
