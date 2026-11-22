"""Vector store management, document ingestion, and retrieval module."""

import os
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

from src.config import CHROMA_DIR, get_embeddings


def get_default_splitter() -> RecursiveCharacterTextSplitter:
    """Return configured RecursiveCharacterTextSplitter."""
    return RecursiveCharacterTextSplitter(
        chunk_size=750,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )


def load_file_content(file_path: str) -> List[Document]:
    """Load text or PDF file into LangChain Documents."""
    path = Path(file_path)
    if not path.exists():
        return []

    if path.suffix.lower() == ".pdf":
        loader = PyPDFLoader(str(path))
        return loader.load()
    else:
        loader = TextLoader(str(path), encoding="utf-8")
        return loader.load()


def index_documents(documents: List[Document], persist_directory: str = str(CHROMA_DIR)) -> Chroma:
    """Split documents, compute embeddings, and store them in ChromaDB."""
    splitter = get_default_splitter()
    chunked_docs = splitter.split_documents(documents)
    embeddings = get_embeddings()

    os.makedirs(persist_directory, exist_ok=True)
    vectorstore = Chroma.from_documents(
        documents=chunked_docs,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    return vectorstore


def get_vectorstore(persist_directory: str = str(CHROMA_DIR)) -> Chroma:
    """Load an existing Chroma vector store from disk."""
    embeddings = get_embeddings()
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )


def retrieve_documents(query: str, k: int = 4, persist_directory: str = str(CHROMA_DIR)) -> List[Document]:
    """Query the vector store and return top-k matching documents."""
    try:
        store = get_vectorstore(persist_directory=persist_directory)
        retriever = store.as_retriever(search_kwargs={"k": k})
        return retriever.invoke(query)
    except Exception:
        return []
