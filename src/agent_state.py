"""State definitions for the LangGraph agentic RAG workflow."""

from typing import List, Optional
from typing_extensions import TypedDict
from langchain_core.documents import Document


class GraphState(TypedDict):
    """Represents the mutable state schema passed through each graph node."""
    question: str
    generation: str
    documents: List[Document]
    search_query: str
    web_search_needed: bool
    router_decision: str
    hallucination_grade: str
    answer_grade: str
    steps: List[str]
