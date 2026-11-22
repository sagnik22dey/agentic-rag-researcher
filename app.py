"""Streamlit Web UI for the Autonomous Fact-Checked Research Agent."""

import os
import streamlit as st
from pathlib import Path
from langchain_core.documents import Document

from src.config import BASE_DIR, DATA_DIR, CHROMA_DIR
from src.vectorstore import index_documents, load_file_content
from src.workflow import stream_agentic_research

st.set_page_config(
    page_title="Agentic RAG Researcher",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 700; margin-bottom: 0.2rem; }
    .subtitle { color: #6c757d; font-size: 1.05rem; margin-bottom: 1.5rem; }
    .step-box { background: #f8f9fa; border-left: 4px solid #4f46e5; padding: 0.6rem 1rem; margin-bottom: 0.5rem; border-radius: 4px; font-family: monospace; font-size: 0.9rem; }
    .badge-grounded { background-color: #dcfce7; color: #166534; padding: 0.2rem 0.6rem; border-radius: 12px; font-weight: 600; font-size: 0.85rem; }
    .badge-web { background-color: #e0f2fe; color: #075985; padding: 0.2rem 0.6rem; border-radius: 12px; font-weight: 600; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Configuration")

    provider = st.selectbox(
        "LLM Provider",
        options=["ollama", "groq", "gemini"],
        index=0,
        help="Select local Ollama (zero API keys needed) or high-speed cloud providers."
    )
    os.environ["LLM_PROVIDER"] = provider

    if provider == "ollama":
        st.info("Ensure Ollama is running locally (e.g. `ollama run llama3.2:3b`).")
        model_name = st.text_input("Ollama Model", value="llama3.2:3b")
        embed_name = st.text_input("Embedding Model", value="nomic-embed-text")
        os.environ["OLLAMA_MODEL"] = model_name
        os.environ["OLLAMA_EMBED_MODEL"] = embed_name
    elif provider == "groq":
        groq_key = st.text_input("Groq API Key", type="password", value=os.getenv("GROQ_API_KEY", ""))
        os.environ["GROQ_API_KEY"] = groq_key
        model_name = st.text_input("Groq Model", value="llama-3.3-70b-versatile")
        os.environ["GROQ_MODEL"] = model_name
    elif provider == "gemini":
        gemini_key = st.text_input("Gemini API Key", type="password", value=os.getenv("GEMINI_API_KEY", ""))
        os.environ["GEMINI_API_KEY"] = gemini_key
        model_name = st.text_input("Gemini Model", value="gemini-1.5-flash")
        os.environ["GEMINI_MODEL"] = model_name

    st.markdown("---")
    st.subheader("📚 Knowledge Base Ingestion")

    uploaded_files = st.file_uploader("Upload Documents (PDF or TXT)", accept_multiple_files=True, type=["pdf", "txt", "md"])

    if st.button("Index Uploaded Files", use_container_width=True):
        if uploaded_files:
            docs_to_index = []
            for uploaded in uploaded_files:
                save_path = DATA_DIR / "sample_docs" / uploaded.name
                with open(save_path, "wb") as f:
                    f.write(uploaded.getbuffer())
                docs_to_index.extend(load_file_content(str(save_path)))
            if docs_to_index:
                index_documents(docs_to_index)
                st.success(f"Indexed {len(docs_to_index)} document segments into ChromaDB!")
        else:
            st.warning("Please upload at least one file first.")

    if st.button("Load Built-in Sample Knowledge", use_container_width=True):
        sample_path = DATA_DIR / "sample_docs" / "autonomous_agents_guide.txt"
        if sample_path.exists():
            docs = load_file_content(str(sample_path))
            index_documents(docs)
            st.success("Successfully indexed 'Autonomous AI Agents Technical Architecture' into ChromaDB!")
        else:
            st.error("Sample document not found.")

st.markdown('<div class="main-title">🧠 Autonomous Fact-Checked Research Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Agentic RAG with dynamic query routing, relevance grading, web search fallback, and hallucination checks.</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("📄 Test Local RAG", help="Ask about autonomous agent architecture from indexed documents"):
        st.session_state["query_input"] = "What are the four explicit stages of Corrective RAG (CRAG)?"
with col2:
    if st.button("🌐 Test Web Fallback", help="Ask about real-time or recent events requiring DuckDuckGo"):
        st.session_state["query_input"] = "What are the latest updates on quantum computing breakthroughs this year?"
with col3:
    if st.button("💬 Test Direct Reasoning", help="Ask conversational or logic query bypassing retrieval"):
        st.session_state["query_input"] = "Explain the difference between short-term and working memory in humans vs AI."

user_query = st.text_input(
    "Enter research question or topic:",
    value=st.session_state.get("query_input", ""),
    placeholder="e.g., How does the self-correction loop in CRAG prevent hallucinations?"
)

if st.button("🚀 Run Agentic Research", type="primary", use_container_width=True) and user_query:
    st.markdown("### 🔄 Agent Execution Trace")
    step_container = st.container()
    result_container = st.empty()

    final_state = {}
    with st.spinner("Agentic pipeline reasoning in progress..."):
        try:
            for step_update in stream_agentic_research(user_query):
                for node_name, node_output in step_update.items():
                    final_state.update(node_output)
                    with step_container:
                        for step_msg in node_output.get("steps", []):
                            st.markdown(f'<div class="step-box">⚡ {step_msg}</div>', unsafe_allow_html=True)
        except Exception as err:
            st.error(f"Execution Error: {str(err)}")

    if final_state.get("generation"):
        st.markdown("---")
        st.markdown("### 📝 Synthesized Research Report")

        h_grade = final_state.get("hallucination_grade", "grounded")
        badge_class = "badge-grounded" if h_grade == "grounded" else "badge-web"
        st.markdown(f'Verification Status: <span class="{badge_class}">{h_grade.upper()}</span>', unsafe_allow_html=True)

        st.markdown(final_state["generation"])

        docs = final_state.get("documents", [])
        if docs:
            with st.expander(f"🔍 Inspected Sources & References ({len(docs)} retrieved)"):
                for i, doc in enumerate(docs, start=1):
                    src = doc.metadata.get("source", "Unknown")
                    title = doc.metadata.get("title", f"Document #{i}")
                    st.markdown(f"**[{i}] {title}** (`{src}`)")
                    st.text(doc.page_content[:400] + ("..." if len(doc.page_content) > 400 else ""))
