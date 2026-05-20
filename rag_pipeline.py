import faiss
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import pipeline


def get_models(st):
    @st.cache_resource
    def _load_embedder():
        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    @st.cache_resource
    def _load_llm():
        return pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=256)

    return _load_embedder(), _load_llm()


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
    results = [chunks[i] for i in indices[0] if i < len(chunks)]
    return results


def generate_answer(question, context_chunks, llm):
    context = "\n\n".join(context_chunks)
    prompt = (
        f"Read the following document excerpts and answer the question accurately.\n\n"
        f"Document:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )
    words = prompt.split()
    if len(words) > 1500:
        prompt = " ".join(words[:1500])

    result = llm(prompt)
    answer = result[0]["generated_text"].strip()
    return answer if answer else "I could not find a clear answer in the document. Try rephrasing your question."
