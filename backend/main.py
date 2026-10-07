"""
FastAPI application services for the AUTOSAR HLD Document Analysis Assistant.

Implements the reference processing flow's application/API layer:
  authentication (stubbed for pilot) | ingestion trigger | retrieval | audit

Endpoints:
  POST /projects/{project}/upload   - upload + ingest a PDF into a project
  GET  /projects/{project}/documents - list ingested documents
  POST /projects/{project}/query    - ask a question, get a cited answer
  GET  /health                      - liveness check

Run with:
  uvicorn main:app --reload --port 8000   (from the backend/ directory)
"""

from __future__ import annotations

import os
import shutil
from datetime import datetime, timezone
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

try:
    from .answer import compose_answer
    from .chunking import chunk_pages
    from .ingestion import extract_pages, needs_ocr
    from .vectorstore import ProjectVectorStore
except ImportError:  # pragma: no cover - supports running as a script from backend/
    from answer import compose_answer
    from chunking import chunk_pages
    from ingestion import extract_pages, needs_ocr
    from vectorstore import ProjectVectorStore

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
INDEX_DIR = os.path.join(DATA_DIR, "index")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

# Simple in-memory audit log for the pilot. Production scope (per the case
# study) persists this in SQLite/PostgreSQL with user identity and project ID.
AUDIT_LOG: List[dict] = []

app = FastAPI(
    title="AUTOSAR HLD Document Analysis Assistant",
    description="Pilot RAG service for AUTOSAR HLD document search and Q&A.",
    version="0.1.0",
)


def _project_index_dir(project: str) -> str:
    safe = "".join(c for c in project if c.isalnum() or c in ("-", "_")) or "default"
    return os.path.join(INDEX_DIR, safe)


def _log(event: str, **fields):
    AUDIT_LOG.append({
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **fields,
    })


class DocumentInfo(BaseModel):
    doc_name: str
    doc_id: str
    pages: int
    chunks: int
    ocr_flagged_pages: List[int]


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class CitationOut(BaseModel):
    doc_name: str
    page_start: int
    page_end: int
    score: float


class QueryResponse(BaseModel):
    query: str
    answer: str
    confidence: str
    citations: List[CitationOut]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/projects/{project}/upload", response_model=DocumentInfo)
async def upload_document(project: str, file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF ingestion is supported in this pilot.")

    project_dir = _project_index_dir(project)
    os.makedirs(project_dir, exist_ok=True)

    project_upload_dir = os.path.join(UPLOAD_DIR, os.path.basename(project_dir))
    os.makedirs(project_upload_dir, exist_ok=True)
    save_path = os.path.join(project_upload_dir, file.filename)
    with open(save_path, "wb") as out:
        shutil.copyfileobj(file.file, out)

    try:
        pages = extract_pages(save_path)
    except Exception as e:
        raise HTTPException(500, f"Failed to extract PDF content: {e}")

    ocr_pages = needs_ocr(pages)
    chunks = chunk_pages(pages)

    if not chunks:
        raise HTTPException(422, "No text could be extracted from this document.")

    store = ProjectVectorStore(project_dir)
    if store.is_ready():
        store.add(chunks)
    else:
        store.build(chunks)

    _log("upload", project=project, doc_name=file.filename,
         pages=len(pages), chunks=len(chunks), ocr_flagged=ocr_pages)

    return DocumentInfo(
        doc_name=file.filename,
        doc_id=pages[0].doc_id,
        pages=len(pages),
        chunks=len(chunks),
        ocr_flagged_pages=ocr_pages,
    )


@app.get("/projects/{project}/documents")
def list_documents(project: str):
    project_dir = _project_index_dir(project)
    store = ProjectVectorStore(project_dir)
    if not store.is_ready():
        return {"documents": []}
    seen = {}
    for m in store.metadata:
        d = seen.setdefault(m["doc_name"], {"doc_name": m["doc_name"], "chunks": 0})
        d["chunks"] += 1
    return {"documents": list(seen.values())}


@app.post("/projects/{project}/query", response_model=QueryResponse)
def query_project(project: str, req: QueryRequest):
    project_dir = _project_index_dir(project)
    store = ProjectVectorStore(project_dir)
    if not store.is_ready():
        raise HTTPException(404, "No ingested documents in this project yet.")

    retrieved = store.search(req.question, top_k=req.top_k)
    composed = compose_answer(req.question, retrieved)

    _log("query", project=project, question=req.question,
         confidence=composed.confidence, n_results=len(retrieved))

    return QueryResponse(
        query=composed.query,
        answer=composed.summary,
        confidence=composed.confidence,
        citations=[
            CitationOut(
                doc_name=c.doc_name,
                page_start=c.page_start,
                page_end=c.page_end,
                score=round(c.score, 4),
            ) for c in composed.citations
        ],
    )


@app.get("/projects/{project}/audit")
def get_audit_log(project: str):
    return {"events": [e for e in AUDIT_LOG if e.get("project") == project]}
