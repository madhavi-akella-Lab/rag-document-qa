# 🤖 LLM-Based Document Q&A System (RAG Chatbot)

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![LangChain](https://img.shields.io/badge/LangChain-0.1+-green)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-412991?logo=openai)
![FAISS](https://img.shields.io/badge/Vector_DB-FAISS-orange)
![Streamlit](https://img.shields.io/badge/Deployed-Streamlit_Cloud-FF4B4B?logo=streamlit)

> **Ask questions about any PDF document and get accurate, context-aware answers powered by Retrieval-Augmented Generation (RAG).**

🔗 **[Live Demo](https://your-app-link.streamlit.app)** | 📄 **[Project Write-up](#how-it-works)**

---

## 📌 What This Project Does

Upload any PDF — a research paper, financial report, policy document, or product manual — and ask it questions in plain English. The system retrieves the most relevant sections and generates precise answers grounded in the document content, not hallucinated from general knowledge.

**Example queries this handles well:**
- *"What are the key risks mentioned in this report?"*
- *"Summarize the methodology section"*
- *"What does the document say about compliance requirements?"*

---

## 🏗️ Architecture

```
PDF Upload
    │
    ▼
Text Extraction (PyPDF2)
    │
    ▼
Chunking with Overlap ──► Embedding Generation (OpenAI)
    │                              │
    ▼                              ▼
FAISS Vector Index ◄──── Stored Embeddings
    │
    ▼
User Query ──► Query Embedding ──► Similarity Search
                                        │
                                        ▼
                              Top-K Relevant Chunks
                                        │
                                        ▼
                              LangChain Prompt + GPT-4
                                        │
                                        ▼
                              Context-Aware Answer
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| LLM | OpenAI GPT-4 via API |
| Orchestration | LangChain |
| Vector Store | FAISS (Facebook AI Similarity Search) |
| Embeddings | OpenAI `text-embedding-ada-002` |
| PDF Parsing | PyPDF2 |
| Frontend | Streamlit |
| Deployment | Streamlit Cloud |
| Secret Management | Streamlit Secrets (no keys in code) |

---

## ✨ Key Features

- 📄 **Multi-page PDF support** — handles documents of any length
- 🔍 **Semantic search** — finds relevant content even when exact words differ
- 🧩 **Chunk overlap strategy** — prevents context loss at chunk boundaries
- 💬 **Conversational responses** — answers in natural language with source grounding
- 🔐 **Secure API key handling** — keys managed via environment variables, never hardcoded
- ⚡ **Fast retrieval** — FAISS in-memory index for sub-second similarity search

---

## 🚀 How to Run Locally

### Prerequisites
- Python 3.10+
- OpenAI API key

### Setup

```bash
# Clone the repo
git clone https://github.com/madhavi-akella-Lab/rag-document-qa
cd rag-document-qa

# Install dependencies
pip install -r requirements.txt

# Set your API key
export OPENAI_API_KEY=your_key_here

# Run the app
streamlit run app.py
```

---

## 📁 Project Structure

```
rag-document-qa/
│
├── app.py                  # Streamlit UI and main app logic
├── rag_pipeline.py         # Core RAG pipeline (chunking, embedding, retrieval)
├── vector_store.py         # FAISS index creation and search
├── requirements.txt
└── README.md
```

---

## 💡 How It Works

### 1. Document Ingestion
When a PDF is uploaded, the text is extracted page by page using PyPDF2. The full text is then split into overlapping chunks (default: 500 tokens, 50-token overlap) to ensure no context is lost at boundaries.

### 2. Embedding & Indexing
Each chunk is converted into a dense vector using OpenAI's `text-embedding-ada-002` model. These vectors are stored in a FAISS index for fast similarity search.

### 3. Query Processing
When the user submits a question, the query is also embedded and the top-K most semantically similar chunks are retrieved from the FAISS index.

### 4. Answer Generation
The retrieved chunks are injected into a LangChain prompt template alongside the user's question. GPT-4 generates a grounded, context-aware answer based only on the retrieved content.

---

## 📊 Performance Notes

- Chunk size of 500 tokens with 50-token overlap tested optimal for 10–50 page documents
- Top-5 chunk retrieval balances context richness with prompt length
- Average response latency: ~2–3 seconds for typical queries

---

## 🔮 Future Enhancements

- [ ] Multi-document support (query across multiple PDFs)
- [ ] Chat history / memory for follow-up questions
- [ ] Swap FAISS for Pinecone for persistent cloud vector storage
- [ ] Add source citation (highlight which page the answer came from)
- [ ] Support for DOCX and TXT file types

---

## 👩‍💻 About the Author

**Madhavi Akella** — Data & AI Engineer | Databricks Generative AI Engineer Associate

Building production-grade data and AI systems with 7+ years of enterprise experience.

🔗 [LinkedIn](https://linkedin.com/in/madhavi-akella-Lab-2b8213114) | 🌐 [Portfolio](https://madhavi-akella-Lab.dev)
