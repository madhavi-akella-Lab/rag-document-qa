import streamlit as st
from rag_pipeline import (
    extract_text_from_pdf,
    chunk_text,
    build_index,
    search_index,
    generate_answer,
    get_models,
)

st.set_page_config(
    page_title="Document Q&A — RAG Chatbot",
    page_icon="🤖",
    layout="centered",
)

st.markdown("""
<style>
    .answer-box {
        background: #f0f7ff;
        border-left: 4px solid #2d7dd2;
        border-radius: 8px;
        padding: 16px 20px;
        margin-top: 12px;
        font-size: 15px;
        line-height: 1.7;
        color: #1a1a2e;
    }
    .context-box {
        background: #f9f9f9;
        border: 1px solid #e0e0e0;
        border-radius: 6px;
        padding: 12px 16px;
        font-size: 13px;
        color: #555;
        line-height: 1.6;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🤖 Document Q&A — RAG Chatbot")
st.markdown("**Upload any PDF and ask questions about it in plain English.**")
st.markdown("*Powered by Sentence Transformers + FAISS — 100% free, no API key needed.*")
st.divider()

with st.spinner("Loading AI models (first load takes ~30 seconds)..."):
    embedder, llm = get_models(st)
st.success("✅ Models ready!")

uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"],
    help="Research papers, reports, policy docs, manuals — any text-based PDF works."
)

if uploaded_file:
    with st.spinner("Reading and indexing your document..."):
        raw_text = extract_text_from_pdf(uploaded_file)

        if not raw_text or len(raw_text.strip()) < 100:
            st.error("Could not extract text from this PDF. It may be a scanned image. Please try a text-based PDF.")
            st.stop()

        chunks = chunk_text(raw_text, chunk_size=400, overlap=50)
        index, _ = build_index(chunks, embedder)

    st.success(f"✅ Document indexed! ({len(chunks)} sections found)")
    st.divider()

    st.subheader("Ask a question")

    suggestions = [
        "What is the main topic of this document?",
        "What are the key findings or conclusions?",
        "Summarize the most important points.",
        "What recommendations are made?",
    ]

    st.markdown("**Try one of these:**")
    cols = st.columns(2)
    selected_suggestion = None
    for i, suggestion in enumerate(suggestions):
        if cols[i % 2].button(suggestion, key=f"sug_{i}"):
            selected_suggestion = suggestion

    question = st.text_input(
        "Or type your own question:",
        value=selected_suggestion if selected_suggestion else "",
        placeholder="e.g. What are the key risks mentioned?",
    )

    if question:
        with st.spinner("Searching document..."):
            relevant_chunks = search_index(question, index, chunks, embedder, top_k=4)
            answer = generate_answer(question, relevant_chunks, llm)

        st.markdown("### 💡 Answer")
        st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

        with st.expander("📄 View source sections used to answer"):
            for i, chunk in enumerate(relevant_chunks, 1):
                st.markdown(f'<div class="context-box"><b>Section {i}:</b><br>{chunk}</div>',
                            unsafe_allow_html=True)

    st.divider()
    st.markdown(
        "Built by [Madhavi Akella](https://linkedin.com/in/madhavi-akella-2b8213114) · "
        "[GitHub](https://github.com/madhavi-akella-lab) · "
        "Powered by 🤗 Hugging Face + FAISS"
    )

else:
    st.info("👆 Upload a PDF above to get started.")
    st.markdown("""
    **What this app does:**
    - 📄 Reads your PDF and splits it into searchable sections
    - 🔍 Uses semantic search to find the most relevant content
    - 💡 Returns the best matching answer from your document

    **Works great for:**
    - Research papers
    - Financial reports
    - Policy documents
    - Product manuals
    """)
