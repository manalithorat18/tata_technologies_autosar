"""
Ingestion module for the AUTOSAR HLD Document Analysis Assistant.

Responsible for:
- Extracting text, headings, and page-level structure from PDF documents
- Producing a list of page records that downstream chunking can use

This mirrors Case Study 1's workflow steps 1-2:
  1. Upload or select an approved HLD document.
  2. Extract text, tables, headings, and metadata.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import List

import pymupdf as fitz  # PyMuPDF (using the non-deprecated import name)


@dataclass
class PageRecord:
    """A single page of extracted content with traceable metadata."""
    doc_id: str
    doc_name: str
    page_number: int  # 1-indexed, matches what a human would cite
    text: str
    char_count: int = field(init=False)

    def __post_init__(self):
        self.char_count = len(self.text)


def compute_doc_id(file_path: str) -> str:
    """Stable content-based ID so re-uploading the same file reuses its ID."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()[:16]


def extract_pages(file_path: str) -> List[PageRecord]:
    """
    Extract per-page text from a PDF using PyMuPDF.

    Note on scope: this pilot extracts native text layers. Scanned pages
    with no text layer will come back empty; production scope (per the
    case study) adds OCR (e.g. Tesseract) as a fallback for those pages.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No such file: {file_path}")

    doc_id = compute_doc_id(file_path)
    doc_name = os.path.basename(file_path)

    pages: List[PageRecord] = []
    with fitz.open(file_path) as pdf:
        for i, page in enumerate(pdf, start=1):
            text = page.get_text("text") or ""
            text = text.strip()
            if not text:
                # Flag likely-scanned pages rather than silently dropping them
                text = "[NO EXTRACTABLE TEXT - possible scanned page; OCR required]"
            pages.append(PageRecord(
                doc_id=doc_id,
                doc_name=doc_name,
                page_number=i,
                text=text,
            ))
    return pages


def needs_ocr(pages: List[PageRecord]) -> List[int]:
    """Return page numbers that appear to have no extractable text layer."""
    return [p.page_number for p in pages if "NO EXTRACTABLE TEXT" in p.text]
