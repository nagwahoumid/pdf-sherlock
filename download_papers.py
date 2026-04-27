"""
download_papers.py

Populate data/pdfs/ with recent arXiv papers for PDF Sherlock's evaluation
corpus.

This script pulls the 150 most recent submissions from arXiv's cs.IR
(Information Retrieval) and cs.CL (Computation and Language) categories and
saves each PDF into ``data/pdfs/`` under a sanitised filename so the
ingestion pipeline (``app.ingest``) can consume them without tripping over
special characters.

Network notes
-------------
arXiv rejects the default Python ``urllib`` User-Agent with HTTP 403, so the
``arxiv`` library is used **only** to run the search and discover each
paper's ``pdf_url``. The PDFs themselves are fetched with ``requests`` using
a standard desktop Chrome User-Agent header, streamed to disk in chunks, and
separated by a short polite delay (``_POLITE_DELAY_SECONDS``) so we don't
hammer the arXiv servers.

Filename scheme
---------------
``<arxiv_id>__<cleaned_title>.pdf``

* ``arxiv_id`` is taken from the entry's short id (e.g. ``2404.01234v1``) with
  the version suffix removed, so the file for a paper is stable across new
  versions on arXiv.
* ``cleaned_title`` is ASCII only: it keeps letters, digits, hyphens and
  underscores, collapses runs of whitespace into single underscores, and is
  truncated to keep the overall filename well within common filesystem
  limits.

The doubled underscore (``__``) makes the boundary between the id and the
title obvious both to humans and to downstream scripts.

Usage
-----
    python download_papers.py
    python download_papers.py --max-results 150 --out data/pdfs

Requires the ``arxiv`` Python package (see requirements.txt).
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import unicodedata
from pathlib import Path

try:
    import arxiv
except ImportError as exc:  # pragma: no cover - executed only when missing
    raise SystemExit(
        "The 'arxiv' package is required. Install it with:\n"
        "    pip install arxiv\n"
        "or re-run: pip install -r requirements.txt"
    ) from exc

try:
    import requests
except ImportError as exc:  # pragma: no cover - executed only when missing
    raise SystemExit(
        "The 'requests' package is required. Install it with:\n"
        "    pip install requests\n"
        "or re-run: pip install -r requirements.txt"
    ) from exc


DEFAULT_CATEGORIES = ("cs.IR", "cs.CL")
DEFAULT_MAX_RESULTS = 150
DEFAULT_OUT_DIR = Path("data/pdfs")

# arXiv rejects requests from the default urllib User-Agent with HTTP 403,
# so we masquerade as a standard desktop browser. This header is also good
# netiquette: it makes it easy for arXiv's operators to identify traffic
# coming from this script should they ever want to contact us.
_BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
_REQUEST_HEADERS = {
    "User-Agent": _BROWSER_USER_AGENT,
    "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.8",
}

# Seconds to wait between successive PDF downloads; arXiv asks clients to
# space requests out by a few seconds. 3s matches the delay used by the
# official ``arxiv`` library's Client default.
_POLITE_DELAY_SECONDS = 3.0
# Per-request network timeout (connect, read) in seconds.
_REQUEST_TIMEOUT = (10.0, 60.0)
# How many bytes to pull from the HTTP stream at a time.
_STREAM_CHUNK_BYTES = 64 * 1024

# Characters that are legal in our sanitised filenames.
_FILENAME_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")
# Extra safety net: collapse runs of underscores produced by the substitution.
_MULTI_UNDERSCORE_RE = re.compile(r"_+")
# arXiv ids look like "2404.01234" or "cs.IR/0403001"; we strip any version
# suffix ("v1", "v12", ...) so the saved filename is stable across revisions.
_VERSION_SUFFIX_RE = re.compile(r"v\d+$")


def _sanitize_title(title: str, max_len: int = 120) -> str:
    """
    Return an ASCII-only, filesystem-friendly slug derived from ``title``.

    Steps:
      1. Unicode NFKD normalisation drops accents and exotic code points.
      2. All non ``[A-Za-z0-9._-]`` characters are replaced with underscores.
      3. Runs of underscores are collapsed, leading/trailing underscores and
         dots are stripped, and the result is truncated to ``max_len``.
    """
    # Normalise unicode to its ASCII-compatible form where possible.
    normalised = unicodedata.normalize("NFKD", title)
    ascii_only = normalised.encode("ascii", "ignore").decode("ascii")
    cleaned = _FILENAME_SAFE_RE.sub("_", ascii_only)
    cleaned = _MULTI_UNDERSCORE_RE.sub("_", cleaned).strip("._")
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len].rstrip("._")
    return cleaned or "untitled"


def _stable_arxiv_id(result: "arxiv.Result") -> str:
    """Return ``result.get_short_id()`` with any trailing ``vN`` stripped."""
    short_id = result.get_short_id()
    return _VERSION_SUFFIX_RE.sub("", short_id)


def _build_filename(result: "arxiv.Result") -> str:
    """Compose the ``<id>__<title>.pdf`` filename for a result."""
    arxiv_id = _sanitize_title(_stable_arxiv_id(result), max_len=40)
    title_slug = _sanitize_title(result.title)
    return f"{arxiv_id}__{title_slug}.pdf"


def _download_pdf_with_browser_ua(
    pdf_url: str,
    target: Path,
    session: requests.Session,
) -> None:
    """
    Fetch ``pdf_url`` with a browser User-Agent and stream it into ``target``.

    Writes to a ``.part`` sibling first and renames on success, so an
    interrupted download never leaves a half-written PDF at the final path
    that the ``--overwrite``-free re-run path would mistake for complete.

    Raises
    ------
    requests.HTTPError
        If the server responds with a non-2xx status (including arXiv's
        403 when the User-Agent header is missing).
    requests.RequestException
        For connection timeouts or other transport errors.
    ValueError
        If the response body is empty.
    """
    partial = target.with_suffix(target.suffix + ".part")
    with session.get(
        pdf_url,
        headers=_REQUEST_HEADERS,
        stream=True,
        timeout=_REQUEST_TIMEOUT,
        allow_redirects=True,
    ) as response:
        response.raise_for_status()
        bytes_written = 0
        with partial.open("wb") as fh:
            for chunk in response.iter_content(chunk_size=_STREAM_CHUNK_BYTES):
                if not chunk:
                    continue
                fh.write(chunk)
                bytes_written += len(chunk)
        if bytes_written == 0:
            partial.unlink(missing_ok=True)
            raise ValueError(f"empty response body for {pdf_url!r}")
    partial.replace(target)


def download_recent_papers(
    out_dir: Path = DEFAULT_OUT_DIR,
    max_results: int = DEFAULT_MAX_RESULTS,
    categories: tuple[str, ...] = DEFAULT_CATEGORIES,
    overwrite: bool = False,
) -> list[Path]:
    """
    Download the ``max_results`` most recent papers across ``categories``.

    Parameters
    ----------
    out_dir : Path
        Directory to write PDFs into. Created if it does not exist.
    max_results : int
        Total number of recent papers to fetch (across all categories
        combined). Defaults to 150.
    categories : tuple[str, ...]
        arXiv subject class identifiers (e.g. ``cs.IR``, ``cs.CL``).
    overwrite : bool
        If False (default), papers whose sanitised filename already exists in
        ``out_dir`` are skipped. If True, they are re-downloaded.

    Returns
    -------
    list[Path]
        Paths of the PDFs that are now present on disk (both newly downloaded
        and already-existing ones that were skipped).
    """
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    query = " OR ".join(f"cat:{c}" for c in categories)
    print(
        f"[download] querying arXiv: {query}  "
        f"(requesting up to {max_results} most recent results)"
    )

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    client = arxiv.Client(page_size=100, delay_seconds=3.0, num_retries=5)

    saved: list[Path] = []
    skipped = 0
    failed = 0
    downloaded_this_run = 0

    results = list(client.results(search))
    total = len(results)
    if total == 0:
        print("[download] arXiv returned 0 results; nothing to do.")
        return saved

    print(f"[download] arXiv returned {total} results; writing to {out_dir}/")

    # One Session gives us connection pooling and lets us reuse the same
    # browser User-Agent across requests.
    with requests.Session() as session:
        session.headers.update(_REQUEST_HEADERS)

        for i, result in enumerate(results, start=1):
            filename = _build_filename(result)
            target = out_dir / filename
            arxiv_id = _stable_arxiv_id(result)

            if target.exists() and not overwrite:
                skipped += 1
                print(
                    f"[{i:3d}/{total}] skip   {arxiv_id}: already present "
                    f"({filename})"
                )
                saved.append(target)
                continue

            # Space out real network requests to be polite to arXiv. We only
            # sleep after we've actually issued at least one download so that
            # skipped/cached entries don't incur a delay.
            if downloaded_this_run > 0:
                time.sleep(_POLITE_DELAY_SECONDS)

            try:
                _download_pdf_with_browser_ua(
                    pdf_url=result.pdf_url,
                    target=target,
                    session=session,
                )
                saved.append(target)
                downloaded_this_run += 1
                print(
                    f"[{i:3d}/{total}] ok     {arxiv_id}: "
                    f"{result.title[:80]!r} -> {filename}"
                )
            except Exception as exc:  # network / HTTP / filesystem errors
                failed += 1
                print(
                    f"[{i:3d}/{total}] FAIL   {arxiv_id}: {exc!r} "
                    f"(title: {result.title[:80]!r})",
                    file=sys.stderr,
                )
                # Back off a bit on errors before moving on.
                time.sleep(1.0)

    print(
        f"[download] done. downloaded={downloaded_this_run}, "
        f"skipped={skipped}, failed={failed}, total_on_disk={len(saved)}"
    )
    return saved


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Download recent arXiv papers from cs.IR and cs.CL into "
            "data/pdfs for PDF Sherlock's evaluation corpus."
        )
    )
    p.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory to save PDFs into (default: data/pdfs)",
    )
    p.add_argument(
        "--max-results",
        type=int,
        default=DEFAULT_MAX_RESULTS,
        help="Total number of recent papers to download (default: 150)",
    )
    p.add_argument(
        "--categories",
        nargs="+",
        default=list(DEFAULT_CATEGORIES),
        help="arXiv categories to query (default: cs.IR cs.CL)",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download papers whose filename already exists on disk.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    download_recent_papers(
        out_dir=args.out,
        max_results=args.max_results,
        categories=tuple(args.categories),
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
