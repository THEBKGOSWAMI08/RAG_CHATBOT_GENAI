import os
import sys
import tempfile
import streamlit as st
from dotenv import load_dotenv

# Add project root to path so src imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

load_dotenv()

from src.pipeline.rag_pipeline import RAGPipeline

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="RAG Q&A System", layout="wide")
st.title("RAG-based Question Answering System")

# Initialize pipeline in session state
if "pipeline" not in st.session_state:
    st.session_state.pipeline = RAGPipeline(groq_api_key=GROQ_API_KEY)
    st.session_state.ingested = False

# --- Sidebar: Upload Documents ---
with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF, TXT, or DOCX files",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
    )

    if st.button("Ingest Documents") and uploaded_files:
        total_chunks = 0
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=os.path.splitext(uploaded_file.name)[1]
            ) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            chunks = st.session_state.pipeline.ingest(tmp_path)
            total_chunks += chunks
            os.unlink(tmp_path)

        st.session_state.ingested = True
        st.success(f"Ingested {total_chunks} chunks from {len(uploaded_files)} file(s).")

    if st.button("Reset Knowledge Base"):
        st.session_state.pipeline.reset()
        st.session_state.ingested = False
        st.info("Knowledge base cleared.")

# --- Main: Query ---
st.header("Ask a Question")
query = st.text_input("Enter your question:")

use_reranker = st.checkbox("Use reranker", value=True)

if st.button("Get Answer") and query:
    if not st.session_state.ingested:
        st.warning("Please upload and ingest documents first.")
    else:
        with st.spinner("Searching and generating answer..."):
            answer = st.session_state.pipeline.query(query, use_reranker=use_reranker)
        st.subheader("Answer")
        st.write(answer)
