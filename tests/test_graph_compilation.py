"""Unit test to verify LangGraph assembly and structure."""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.workflow import build_agentic_rag_graph


def test_graph_structure():
    """Verify state graph nodes, edges, and compilation."""
    app = build_agentic_rag_graph()
    nodes = set(app.get_graph().nodes.keys())

    expected_nodes = {
        "route_question",
        "retrieve",
        "grade_documents",
        "transform_query",
        "web_search",
        "generate",
        "direct_answer",
        "grade_hallucination"
    }

    assert expected_nodes.issubset(nodes), f"Missing nodes: {expected_nodes - nodes}"
    print(f"PASS: LangGraph compiled successfully with {len(nodes)} active graph nodes.")


if __name__ == "__main__":
    test_graph_structure()
