"""End-to-end integration test verifying indexing and LangGraph execution."""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.vectorstore import load_file_content, index_documents
from src.workflow import run_agentic_research


def test_full_pipeline():
    """Verify document ingestion and full agentic research graph execution."""
    doc_path = BASE_DIR / "data" / "sample_docs" / "autonomous_agents_guide.txt"
    docs = load_file_content(str(doc_path))
    index_documents(docs)
    print("PASS: ChromaDB indexing completed.")

    question = "What are the four explicit stages of Corrective RAG (CRAG)?"
    result = run_agentic_research(question)

    assert result.get("generation"), "Generation output is empty"
    assert len(result.get("steps", [])) > 0, "No audit steps recorded"

    print("\n--- Pipeline Execution Trace ---")
    for step in result["steps"]:
        print(f"  * {step}")

    print("\n--- Final Agent Response ---")
    print(result["generation"])
    print(f"\nHallucination Status: {result.get('hallucination_grade')}")
    print("\nPASS: End-to-end pipeline test executed successfully!")


if __name__ == "__main__":
    test_full_pipeline()
