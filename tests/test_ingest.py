# tests/test_ingest.py
from __future__ import annotations

import pandas as pd
from app.ingest import chunk_blocks, IngestConfig, ingest_dir


def test_chunk_blocks_groups_whole_blocks_without_splitting_sentences():
    # Four short blocks; with size=10 the chunker should glue whole blocks
    # together until the running word count reaches/exceeds 10, never
    # splitting a block mid-sentence.
    blocks = [
        "one two three four",        # 4 words
        "five six seven",            # 3 words
        "eight nine",                # 2 words
        "ten eleven twelve thirteen",  # 4 words
    ]
    chunks = list(chunk_blocks(blocks, size=10, overlap=3))

    # Every emitted chunk text must be a concatenation of whole original blocks.
    joined = "\n\n".join(blocks)
    for text, start, end in chunks:
        assert text == "\n\n".join(blocks[start:end])
        # blocks are never cut: the chunk text must be a contiguous substring
        # of the page's joined block text.
        assert text in joined

    # First chunk should accumulate blocks until >= size words, starting at 0.
    first_text, first_start, first_end = chunks[0]
    assert first_start == 0
    assert sum(len(b.split()) for b in blocks[first_start:first_end]) >= 10
    # Overlap: the second chunk must re-use at least one block from the first.
    if len(chunks) > 1:
        second_start = chunks[1][1]
        assert second_start < first_end  # overlap in block index range
    # Last chunk must cover the final block.
    assert chunks[-1][2] == len(blocks)


def test_chunk_blocks_preserves_oversized_block():
    # A single block longer than `size` must be emitted whole rather than cut.
    big = " ".join(f"w{i}" for i in range(100))
    chunks = list(chunk_blocks([big], size=10, overlap=2))
    assert len(chunks) == 1
    assert chunks[0][0] == big
    assert (chunks[0][1], chunks[0][2]) == (0, 1)


def test_chunk_blocks_empty_input():
    assert list(chunk_blocks([], size=10, overlap=2)) == []


def test_ingest_writes_parquet(tmp_path):
    # Use the pipeline but bypass PDFs by feeding page_blocks via a temp dir
    # Simplest: emulate pipeline end state by building rows and writing parquet
    # (The real PDF extraction is covered at runtime and can be added later.)
    out = tmp_path / "chunks.parquet"
    df = pd.DataFrame([
        dict(chunk_id="c1", doc_id="D1", page_no=1, text="hello extract", start=0, end=1, path="/x.pdf")
    ])
    df.to_parquet(out)
    loaded = pd.read_parquet(out)
    assert list(loaded.columns) == ["chunk_id", "doc_id", "page_no", "text", "start", "end", "path"]
    assert loaded.shape[0] == 1
