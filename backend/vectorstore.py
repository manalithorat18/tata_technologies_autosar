"""
Embedding + local vector store module.

Mirrors Case Study 1 / common architecture:
  "Create embeddings and save them in the local vector database."
  Embeddings: BGE, E5, Sentence Transformers, or equivalent
  Vector database: FAISS or ChromaDB

PILOT NOTE ON EMBEDDINGS:
This sandbox environment has tight local disk space, so this pilot uses a
TF-IDF vectorizer (scikit-learn) as a lightweight, dependency-light stand-in
for a neural embedding model. It is a legitimate "or equivalent" for a
proof-of-concept: it requires no GPU, no multi-GB model download, and still
demonstrates the full ingest -> embed -> index -> retrieve -> cite pipeline.

For a production pilot, swap `TfidfEmbedder` for a real sentence embedding
model (e.g. BAAI/bge-small-en-v1.5 or intfloat/e5-small-v2 via
sentence-transformers) behind the same `embed()` interface -- nothing else
in the pipeline needs to change.
"""

from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass
from typing import List, Optional, Tuple

import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from .chunking import Chunk
except ImportError:  # pragma: no cover - supports running as a script from backend/
    from chunking import Chunk


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    doc_name: str
    page_start: int
    page_end: int
    text: str
    score: float


class TfidfEmbedder:
    """Lightweight embedder used for this pilot (see module docstring)."""

    def __init__(self, max_features: int = 4096):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            stop_words="english",
        )
        self._fitted = False

    def fit(self, texts: List[str]):
        self.vectorizer.fit(texts)
        self._fitted = True

    def embed(self, texts: List[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Embedder must be fit before use")
        vecs = self.vectorizer.transform(texts).toarray().astype("float32")
        # L2-normalize so inner product == cosine similarity
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.vectorizer, f)

    @classmethod
    def load(cls, path: str) -> "TfidfEmbedder":
        obj = cls()
        with open(path, "rb") as f:
            obj.vectorizer = pickle.load(f)
        obj._fitted = True
        return obj


class ProjectVectorStore:
    """
    A project-scoped local vector collection, backed by FAISS, with a
    parallel metadata store for citations (doc name, page range, text).

    Each "project" gets its own index directory, matching the case study's
    requirement to "separate project collections and enforce role-based
    access to prevent cross-program data exposure."
    """

    def __init__(self, project_dir: str):
        self.project_dir = project_dir
        os.makedirs(project_dir, exist_ok=True)
        self.index_path = os.path.join(project_dir, "index.faiss")
        self.meta_path = os.path.join(project_dir, "meta.jsonl")
        self.embedder_path = os.path.join(project_dir, "embedder.pkl")

        self.embedder: Optional[TfidfEmbedder] = None
        self.index: Optional[faiss.Index] = None
        self.metadata: List[dict] = []

        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self._load()

    def _load(self):
        self.index = faiss.read_index(self.index_path)
        self.metadata = []
        with open(self.meta_path, "r", encoding="utf-8") as f:
            for line in f:
                self.metadata.append(json.loads(line))
        if os.path.exists(self.embedder_path):
            self.embedder = TfidfEmbedder.load(self.embedder_path)

    def build(self, chunks: List[Chunk]):
        """(Re)build the index from scratch for this project's chunks."""
        texts = [c.text for c in chunks]
        self.embedder = TfidfEmbedder()
        self.embedder.fit(texts)
        vectors = self.embedder.embed(texts)

        dim = vectors.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # cosine sim via normalized IP
        self.index.add(vectors)

        self.metadata = [
            {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "doc_name": c.doc_name,
                "page_start": c.page_start,
                "page_end": c.page_end,
                "text": c.text,
            }
            for c in chunks
        ]
        self._persist()

    def add(self, chunks: List[Chunk]):
        """Add chunks to an existing index (re-fits embedder on full corpus,
        since TF-IDF vocabularies are corpus-dependent - a real embedding
        model would not need this refit step)."""
        existing_texts = [m["text"] for m in self.metadata]
        existing_chunks_as_dicts = self.metadata
        all_texts = existing_texts + [c.text for c in chunks]

        self.embedder = TfidfEmbedder()
        self.embedder.fit(all_texts)
        vectors = self.embedder.embed(all_texts)

        dim = vectors.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(vectors)

        new_meta = [
            {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "doc_name": c.doc_name,
                "page_start": c.page_start,
                "page_end": c.page_end,
                "text": c.text,
            }
            for c in chunks
        ]
        self.metadata = existing_chunks_as_dicts + new_meta
        self._persist()

    def _persist(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            for m in self.metadata:
                f.write(json.dumps(m) + "\n")
        self.embedder.save(self.embedder_path)

    def is_ready(self) -> bool:
        return self.index is not None and len(self.metadata) > 0

    def search(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        if not self.is_ready():
            return []
        qvec = self.embedder.embed([query])
        scores, idxs = self.index.search(qvec, min(top_k, len(self.metadata)))
        results: List[RetrievedChunk] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue
            m = self.metadata[idx]
            results.append(RetrievedChunk(
                chunk_id=m["chunk_id"],
                doc_id=m["doc_id"],
                doc_name=m["doc_name"],
                page_start=m["page_start"],
                page_end=m["page_end"],
                text=m["text"],
                score=float(score),
            ))
        return results
