import faiss
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer


def get_models(st):
    @st.cache_resource
    def _load_embedder():
        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _load_embedder(), None


def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text.strip()


def chunk_text(text, chunk_size=400, overlap=50):
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


def build_index(chunks, embedder):
    embeddings = embedder.encode(chunks, show_progress_bar=False)
    embeddings = np.array(embeddings, dtype="float32")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embeddings


def search_index(query, index, chunks, embedder, top_k=4):
    query_vec = embedder.encode([query], show_progress_bar=False)
    query_vec = np.array(query_vec, dtype="float32")
    distances, indices = index.search(query_vec, top_k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]


def generate_answer(question, context_chunks, llm):
    if not context_chunks:
        return "I could not find a relevant answer. Please try rephrasing your question."

    # Find the most relevant sentences across all retrieved chunks
    all_sentences = []
    for chunk in context_chunks:
        sentences = [s.strip() for s in chunk.replace("\n", ". ").split(".") if len(s.strip()) > 40]
        all_sentences.extend(sentences)

    if not all_sentences:
        return context_chunks[0][:500]

    # Return the top 3 most relevant sentences joined together
    return " ".join(all_sentences[:3])
