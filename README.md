# AUTOSAR HLD Document Analysis Assistant — Pilot

A working pilot implementation of **Case Study 1** from the *Automotive
Engineering AI: Standard Project Case Study Document*: a RAG-based assistant
that lets engineers upload AUTOSAR HLD PDFs and ask natural-language
questions, getting answers grounded in **cited page passages**.

This is the recommended starting project in the source document because it
has low decision risk, minimal deterministic-rule complexity, and produces a
reusable architecture knowledge base for later HARA/TARA/UDS work.

## What's implemented

| Layer | This pilot uses | Matches doc's recommendation? |
|---|---|---|
| Ingestion | PyMuPDF text extraction, per page | ✅ |
| OCR fallback | Not implemented — scanned pages are flagged, not processed | ⚠️ documented gap |
| Chunking | Heading-aware, ~1200 char chunks with overlap | ✅ |
| Embeddings | **TF-IDF (scikit-learn)** | ⚠️ stand-in — see below |
| Vector store | FAISS, one isolated index per project | ✅ |
| API layer | FastAPI | ✅ |
| UI | Streamlit | ✅ |
| Answer generation | **Extractive** (ranked cited passages) | ⚠️ stand-in — see below |
| Citations & confidence | Every answer carries page numbers + a similarity-based confidence label | ✅ |

### Why TF-IDF instead of a neural embedding model, and extractive instead of an LLM

The build environment for this pilot has very limited local disk space, and
a real embedding model (`sentence-transformers`, which pulls in PyTorch) and
a local LLM both need several GB. Rather than skip the retrieval/generation
steps, this pilot uses lightweight, dependency-free stand-ins so you can see
and test the **full pipeline** end to end:

- `backend/vectorstore.py` → `TfidfEmbedder` stands in for BGE/E5/Sentence-Transformers.
- `backend/answer.py` → `compose_answer()` returns ranked, cited source passages
  instead of an LLM-written summary, so there's nothing to hallucinate.

Both are isolated behind small, clean interfaces so you can swap them for
the real thing without touching ingestion, chunking, the API, or the UI:

```python
# vectorstore.py — replace TfidfEmbedder with e.g.:
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("BAAI/bge-small-en-v1.5")
vectors = model.encode(texts, normalize_embeddings=True)

# answer.py — replace compose_answer()'s body with a call to your
# approved local LLM (Ollama, vLLM, etc.), passing retrieved_chunks
# as the RAG context, same as the case study's reference architecture.
```

TF-IDF works reasonably for queries whose exact words appear in the source
text, but it has no notion of synonyms or semantic similarity — expect
noticeably better retrieval once a real embedding model is swapped in.

## Project structure

```
autosar_hld_assistant/
├── backend/
│   ├── ingestion.py     # PDF → per-page text (PyMuPDF)
│   ├── chunking.py      # Page text → heading-aware chunks
│   ├── vectorstore.py   # Embedding + FAISS index, per-project isolation
│   ├── answer.py        # Cited, confidence-scored answer composition
│   └── main.py          # FastAPI app: upload / documents / query / audit
├── frontend/
│   └── app.py           # Streamlit UI
├── data/
│   ├── uploads/         # Saved source PDFs, per project
│   └── index/           # FAISS indexes + metadata, per project
├── requirements.txt
└── README.md
```

## Running it

```bash
pip install -r requirements.txt

# Terminal 1 — backend
cd backend
uvicorn main:app --reload --port 8000

# Terminal 2 — frontend
cd frontend
streamlit run app.py
```

Then open the Streamlit URL, upload an HLD PDF in the sidebar, and ask
questions in the main panel. Each answer shows a confidence label
(🟢 high / 🟡 medium / 🔴 low) and the exact source pages it drew from.

### API only (no UI)

```bash
curl -F "file=@your_hld.pdf" http://localhost:8000/projects/demo/upload
curl -X POST http://localhost:8000/projects/demo/query \
     -H "Content-Type: application/json" \
     -d '{"question": "What signals does the braking component publish?"}'
```

## Tested against

This pilot was tested end-to-end against the uploaded
*Automotive Engineering AI Project Case Studies* PDF (26 pages): ingestion
extracted all 26 pages with no OCR gaps, chunking produced 39 chunks, and
querying returned cited, page-accurate passages with confidence scoring.

## Governance notes carried over from the case study

- Every response includes page-level citations and a confidence indicator.
- Documents and indexes are isolated per project (`data/index/<project>/`).
- An in-memory audit log records uploads and queries (`main.py: AUDIT_LOG`);
  production scope should persist this to SQLite/PostgreSQL with real user
  identity, per the reference architecture.
- This tool assists engineers; it does not auto-approve architecture
  decisions or modify source documents.

## Known gaps vs. full Case Study 1 scope (next iterations)

1. OCR for scanned pages (Tesseract) — currently just flagged, not processed.
2. Swap TF-IDF → a real embedding model once GPU/disk budget allows.
3. Swap extractive answers → LLM-generated summaries via an approved local LLM.
4. Document comparison / revision-diff feature (case study section 6).
5. Role-based access control and authenticated users (currently open/local).
6. Persistent structured storage (SQLite/PostgreSQL) for audit/review data.
