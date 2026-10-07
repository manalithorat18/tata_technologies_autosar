"""
Chunking module.

Splits extracted pages into overlapping, section-aware chunks small enough
for embedding and retrieval, while preserving page-level citation metadata.

Mirrors Case Study 1 workflow step 3: "Chunk content by section and
architecture context."
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

try:
    from .ingestion import PageRecord
except ImportError:  # pragma: no cover - supports running as a script from backend/
    from ingestion import PageRecord

# Headings commonly seen in AUTOSAR HLD documents - used to bias chunk
# boundaries towards section starts rather than mid-sentence cuts.
HEADING_PATTERN = re.compile(
    r"^\s*(\d+(\.\d+)*\s+.+|"
    r"(Component|Interface|Port|Signal|Module|Section|Chapter)\b.*)$",
    re.IGNORECASE,
)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_name: str
    page_start: int
    page_end: int
    text: str


def _split_into_sentences(text: str) -> List[str]:
    # Lightweight sentence/line splitter; good enough for technical prose
    # and avoids pulling in a heavy NLP dependency for a pilot.
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    return lines if lines else [text]


def chunk_pages(
    pages: List[PageRecord],
    target_chars: int = 1200,
    overlap_chars: int = 200,
) -> List[Chunk]:
    """
    Greedily accumulate page text into ~target_chars chunks, preferring to
    break at heading boundaries, and carrying a small overlap so retrieval
    doesn't lose context that spans a chunk boundary.
    """
    chunks: List[Chunk] = []
    if not pages:
        return chunks

    doc_id = pages[0].doc_id
    doc_name = pages[0].doc_name

    buffer_lines: List[str] = []
    buffer_len = 0
    page_start = pages[0].page_number
    current_page = pages[0].page_number

    def flush(end_page: int):
        nonlocal buffer_lines, buffer_len, page_start
        if not buffer_lines:
            return
        text = "\n".join(buffer_lines).strip()
        if text:
            idx = len(chunks)
            chunks.append(Chunk(
                chunk_id=f"{doc_id}_c{idx:04d}",
                doc_id=doc_id,
                doc_name=doc_name,
                page_start=page_start,
                page_end=end_page,
                text=text,
            ))
        # keep a small overlap tail for continuity
        tail_text = text[-overlap_chars:] if len(text) > overlap_chars else text
        buffer_lines = [tail_text] if tail_text else []
        buffer_len = len(tail_text)
        page_start = end_page

    for page in pages:
        current_page = page.page_number
        for line in _split_into_sentences(page.text):
            is_heading = bool(HEADING_PATTERN.match(line))
            if is_heading and buffer_len > target_chars * 0.4:
                flush(current_page)
            buffer_lines.append(line)
            buffer_len += len(line) + 1
            if buffer_len >= target_chars:
                flush(current_page)

    flush(current_page)
    return chunks
