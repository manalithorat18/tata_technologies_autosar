"""
Answer composition module.

Mirrors Case Study 1 workflow step 6: "Generate grounded responses with
page or section citations."

PILOT NOTE ON GENERATION:
This sandbox has no access to a hosted or local LLM (no GPU, no model
weights, no external inference API configured). So this pilot uses an
*extractive* composer: it returns the most relevant retrieved passages,
ranked and cited by page, with simple confidence scoring based on
retrieval similarity - rather than a free-text LLM-generated summary.

This keeps every word of the answer directly traceable to source text
(there is nothing to hallucinate), which is a reasonable, conservative
placeholder for a first pilot. The `compose_answer()` function is the
single integration point: swap its body for a call to your approved LLM
(local Llama/Mistral/Qwen via vLLM/Ollama, etc.) using the same
`retrieved_chunks` as the RAG context, and nothing upstream changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

try:
    from .vectorstore import RetrievedChunk
except ImportError:  # pragma: no cover - supports running as a script from backend/
    from vectorstore import RetrievedChunk

# Below this similarity score, we don't have much confidence the corpus
# actually contains an answer to the question.
LOW_CONFIDENCE_THRESHOLD = 0.08


@dataclass
class Citation:
    doc_name: str
    page_start: int
    page_end: int
    score: float


@dataclass
class ComposedAnswer:
    query: str
    summary: str
    citations: List[Citation]
    confidence: str  # "high" | "medium" | "low"
    raw_chunks: List[RetrievedChunk]


def _confidence_label(top_score: float) -> str:
    if top_score >= 0.35:
        return "high"
    if top_score >= LOW_CONFIDENCE_THRESHOLD:
        return "medium"
    return "low"


def compose_answer(query: str, retrieved_chunks: List[RetrievedChunk]) -> ComposedAnswer:
    if not retrieved_chunks:
        return ComposedAnswer(
            query=query,
            summary=(
                "No relevant content was found in the ingested document(s) for this "
                "question. Try rephrasing, or confirm the right document is loaded."
            ),
            citations=[],
            confidence="low",
            raw_chunks=[],
        )

    top_score = retrieved_chunks[0].score
    confidence = _confidence_label(top_score)

    lines = []
    citations = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        page_ref = (
            f"p.{chunk.page_start}" if chunk.page_start == chunk.page_end
            else f"pp.{chunk.page_start}-{chunk.page_end}"
        )
        snippet = chunk.text.strip().replace("\n", " ")
        if len(snippet) > 400:
            snippet = snippet[:400].rsplit(" ", 1)[0] + " …"
        lines.append(f"[{i}] ({chunk.doc_name}, {page_ref}) {snippet}")
        citations.append(Citation(
            doc_name=chunk.doc_name,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            score=chunk.score,
        ))

    header = (
        "Most relevant passages found (extractive - not LLM-generated; see note below):\n"
    )
    summary = header + "\n\n".join(lines)

    if confidence == "low":
        summary += (
            "\n\nNote: retrieval similarity is low for this query. Treat these "
            "passages as tentative and verify against the source document."
        )

    return ComposedAnswer(
        query=query,
        summary=summary,
        citations=citations,
        confidence=confidence,
        raw_chunks=retrieved_chunks,
    )
