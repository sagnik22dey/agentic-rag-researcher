"""Integration tests for tools, document parsing, and state flow."""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.tools import search_web
from src.vectorstore import load_file_content, get_default_splitter


def test_document_loader():
    """Verify that sample docs load and split properly."""
    sample_file = BASE_DIR / "data" / "sample_docs" / "autonomous_agents_guide.txt"
    assert sample_file.exists(), f"Sample doc not found at {sample_file}"

    docs = load_file_content(str(sample_file))
    assert len(docs) > 0, "Loader returned empty list"

    splitter = get_default_splitter()
    chunks = splitter.split_documents(docs)
    assert len(chunks) >= 3, f"Expected at least 3 chunks, got {len(chunks)}"
    print(f"PASS: Document loader successfully loaded and split {len(chunks)} chunks.")


def test_duckduckgo_search():
    """Verify that duckduckgo web search returns formatted documents."""
    results = search_web("artificial intelligence autonomous agents", max_results=2)
    assert isinstance(results, list), "Expected list of documents"
    assert len(results) > 0, "Search returned 0 results"
    assert results[0].page_content, "Document content is empty"
    print(f"PASS: Web search returned {len(results)} valid documents.")


if __name__ == "__main__":
    test_document_loader()
    test_duckduckgo_search()
    print("All component tests passed successfully!")
