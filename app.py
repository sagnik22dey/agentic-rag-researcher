"""Streamlit Web UI for the Autonomous Fact-Checked Research Agent."""

import os
import urllib.request
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
    .subtitle { color: var(--text-color, #94a3b8); opacity: 0.85; font-size: 1.05rem; margin-bottom: 1.5rem; }
    .step-box {
        background-color: var(--secondary-background-color, #1e293b);
        color: var(--text-color, #f8fafc);
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-left: 4px solid #4f46e5;
        padding: 0.75rem 1.1rem;
        margin-bottom: 0.6rem;
        border-radius: 6px;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        font-size: 0.92rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
    }
    .badge-grounded {
        background-color: rgba(34, 197, 94, 0.15);
        color: #22c55e;
        border: 1px solid rgba(34, 197, 94, 0.4);
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-web {
        background-color: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.4);
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
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
        try:
            with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=1) as response:
                if response.status == 200:
                    st.success("🟢 Ollama server is active")
        except Exception:
            st.error("🔴 Ollama server is offline. Please start Ollama (`ollama serve`).")
        model_name = st.text_input("Ollama Model", value=os.getenv("OLLAMA_MODEL", "llama3.2:1b"))
        embed_name = st.text_input("Embedding Model", value=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"))
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
    displayed_steps = set()
    with st.spinner("Agentic pipeline reasoning in progress..."):
        try:
            for step_update in stream_agentic_research(user_query):
                for node_name, node_output in step_update.items():
                    final_state.update(node_output)
                    for step_msg in node_output.get("steps", []):
                        if step_msg not in displayed_steps:
                            displayed_steps.add(step_msg)
                            with step_container:
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
