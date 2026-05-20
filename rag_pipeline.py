import re
import faiss
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# ── Load models (cached by Streamlit) ──────────────────────────────────────
@st_cache_resource
def load_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@st_cache_resource
def load_llm():
    return pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=256)

# We import st inside functions to allow this file to be tested standalone
def get_models():
    import streamlit as st
    @st.cache_resource
    def _load_embedder():
        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    @st.cache_resource
    def _load_llm():
        return pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=256)
    return _load_embedder(), _load_llm()


# ── PDF Text Extraction ─────────────────────────────────────────────────────
def extract_text_from_pdf(uploaded_file):
    """Extract all text from an uploaded PDF file."""
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text.strip()


# ── Chunking ────────────────────────────────────────────────────────────────
def chunk_text(text, chunk_size=400, overlap=50):
    """
    Split text into overlapping chunks by word count.
    chunk_size: target words per chunk
    overlap:    words shared between consecutive chunks
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


# ── FAISS Index ─────────────────────────────────────────────────────────────
def build_index(chunks, embedder):
    """Embed chunks and build a FAISS flat-L2 index."""
    embeddings = embedder.encode(chunks, show_progress_bar=False)
    embeddings = np.array(embeddings, dtype="float32")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embeddings


def search_index(query, index, chunks, embedder, top_k=4):
    """Return the top_k most relevant chunks for a query."""
    query_vec = embedder.encode([query], show_progress_bar=False)
    query_vec = np.array(query_vec, dtype="float32")
    distances, indices = index.search(query_vec, top_k)
    results = [chunks[i] for i in indices[0] if i < len(chunks)]
    return results


# ── Answer Generation ───────────────────────────────────────────────────────
def generate_answer(question, context_chunks, llm):
    """
    Build a prompt from retrieved chunks and generate an answer using flan-t5.
    """
    context = "\n\n".join(context_chunks)
    # flan-t5 works best with explicit instruction-style prompts
    prompt = (
        f"Read the following document excerpts and answer the question accurately.\n\n"
        f"Document:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )
    # Truncate prompt to ~1500 words to stay within model limits
    words = prompt.split()
    if len(words) > 1500:
        prompt = " ".join(words[:1500])

    result = llm(prompt)
    answer = result[0]["generated_text"].strip()
    return answer if answer else "I could not find a clear answer in the document. Try rephrasing your question."
