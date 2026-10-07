"""
Streamlit UI for the AUTOSAR HLD Document Analysis Assistant.

Talks to the FastAPI backend (see backend/main.py) over HTTP.
Run with:  streamlit run app.py   (after starting the backend with uvicorn)
"""

import os

import requests
import streamlit as st

BACKEND_URL = os.environ.get("HLD_BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="AUTOSAR HLD Assistant", page_icon="🚗", layout="wide")

st.title("🚗 AUTOSAR HLD Document Analysis Assistant")
st.caption(
    "Pilot RAG application — Case Study 1. Upload an HLD document, then ask "
    "questions and get answers grounded in cited page passages."
)

with st.sidebar:
    st.header("Project")
    project = st.text_input("Project name", value="default-project")
    st.caption("Documents are isolated per project (no cross-program exposure).")

    st.divider()
    st.header("Upload a document")
    uploaded = st.file_uploader("HLD PDF", type=["pdf"])
    if uploaded is not None and st.button("Ingest document", type="primary"):
        with st.spinner("Extracting, chunking, and indexing…"):
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/projects/{project}/upload",
                    files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
                    timeout=120,
                )
                if resp.status_code == 200:
                    info = resp.json()
                    st.success(f"Ingested '{info['doc_name']}': {info['pages']} pages, {info['chunks']} chunks.")
                    if info["ocr_flagged_pages"]:
                        st.warning(
                            f"Pages with no extractable text (likely scanned, need OCR): "
                            f"{info['ocr_flagged_pages']}"
                        )
                else:
                    st.error(f"Upload failed: {resp.status_code} — {resp.text}")
            except requests.exceptions.ConnectionError:
                st.error(
                    "Could not reach the backend. Start it first with:\n"
                    "`uvicorn main:app --reload --port 8000` from the backend/ directory."
                )

    st.divider()
    st.header("Ingested documents")
    try:
        docs_resp = requests.get(f"{BACKEND_URL}/projects/{project}/documents", timeout=10)
        if docs_resp.status_code == 200:
            docs = docs_resp.json().get("documents", [])
            if docs:
                for d in docs:
                    st.write(f"📄 **{d['doc_name']}** — {d['chunks']} chunks")
            else:
                st.info("No documents ingested yet.")
    except requests.exceptions.ConnectionError:
        st.info("Backend not running yet.")

st.divider()
st.subheader("Ask a question about this project's architecture")

top_k = st.slider("Number of source passages to retrieve", 1, 10, 5)
question = st.text_area(
    "Question",
    placeholder="e.g. What signals does the braking component publish? "
                "What is the recommended vector database for a pilot?",
    height=80,
)

if st.button("Ask", type="primary") and question.strip():
    with st.spinner("Retrieving relevant sections and composing a cited answer…"):
        try:
            resp = requests.post(
                f"{BACKEND_URL}/projects/{project}/query",
                json={"question": question, "top_k": top_k},
                timeout=60,
            )
            if resp.status_code == 200:
                result = resp.json()

                conf = result["confidence"]
                conf_color = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(conf, "")
                st.markdown(f"**Confidence:** {conf_color} {conf.capitalize()}")

                st.markdown("### Answer")
                st.write(result["answer"])

                st.markdown("### Citations")
                for c in result["citations"]:
                    page_ref = (
                        f"p.{c['page_start']}" if c["page_start"] == c["page_end"]
                        else f"pp.{c['page_start']}–{c['page_end']}"
                    )
                    st.write(f"- **{c['doc_name']}**, {page_ref} (similarity: {c['score']:.3f})")
            elif resp.status_code == 404:
                st.warning("No documents have been ingested into this project yet. Upload one first.")
            else:
                st.error(f"Query failed: {resp.status_code} — {resp.text}")
        except requests.exceptions.ConnectionError:
            st.error(
                "Could not reach the backend. Start it first with:\n"
                "`uvicorn main:app --reload --port 8000` from the backend/ directory."
            )

st.divider()
with st.expander("ℹ️ About this pilot"):
    st.markdown(
        """
This is a **pilot-scope** implementation of Case Study 1 (AUTOSAR HLD Document
Analysis Assistant) from the reference solution document:

- **Ingestion:** PyMuPDF text extraction, per page (OCR fallback not yet wired in)
- **Chunking:** heading-aware, ~1200-character chunks with overlap
- **Embeddings:** TF-IDF (scikit-learn) as a lightweight stand-in for a neural
  embedding model — see `backend/vectorstore.py` for the swap-in point for
  BGE / E5 / Sentence-Transformers
- **Vector store:** FAISS, one project-isolated index per project
- **Answer generation:** extractive (cited passages ranked by similarity) —
  see `backend/answer.py` for the swap-in point for a local LLM
- **Governance:** every answer carries page-level citations and a confidence
  label; this tool assists but does not replace engineering review.
        """
    )
