"""LangGraph compilation and execution pipeline for Agentic RAG."""

from typing import Dict, Any, Generator
from langgraph.graph import StateGraph, START, END

from src.agent_state import GraphState
from src.agent_nodes import (
    route_question_node,
    retrieve_node,
    grade_documents_node,
    transform_query_node,
    web_search_node,
    generate_node,
    direct_answer_node,
    grade_hallucination_node
)


def decide_route(state: GraphState) -> str:
    """Evaluate router output and direct to the next node."""
    decision = state.get("router_decision", "vectorstore")
    if decision == "direct":
        return "direct_answer"
    elif decision == "websearch":
        return "transform_query"
    return "retrieve"


def decide_after_grading(state: GraphState) -> str:
    """Determine whether to proceed to generation or fallback to web search."""
    if state.get("web_search_needed", False):
        return "transform_query"
    return "generate"


def build_agentic_rag_graph():
    """Assemble and compile the StateGraph for the research agent."""
    builder = StateGraph(GraphState)

    builder.add_node("route_question", route_question_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("grade_documents", grade_documents_node)
    builder.add_node("transform_query", transform_query_node)
    builder.add_node("web_search", web_search_node)
    builder.add_node("generate", generate_node)
    builder.add_node("direct_answer", direct_answer_node)
    builder.add_node("grade_hallucination", grade_hallucination_node)

    builder.add_edge(START, "route_question")

    builder.add_conditional_edges(
        "route_question",
        decide_route,
        {
            "direct_answer": "direct_answer",
            "transform_query": "transform_query",
            "retrieve": "retrieve"
        }
    )

    builder.add_edge("retrieve", "grade_documents")

    builder.add_conditional_edges(
        "grade_documents",
        decide_after_grading,
        {
            "transform_query": "transform_query",
            "generate": "generate"
        }
    )

    builder.add_edge("transform_query", "web_search")
    builder.add_edge("web_search", "generate")
    builder.add_edge("generate", "grade_hallucination")
    builder.add_edge("grade_hallucination", END)
    builder.add_edge("direct_answer", END)

    return builder.compile()


def run_agentic_research(question: str) -> Dict[str, Any]:
    """Execute the full workflow and return final state dict."""
    app = build_agentic_rag_graph()
    initial_state: GraphState = {
        "question": question,
        "generation": "",
        "documents": [],
        "search_query": "",
        "web_search_needed": False,
        "router_decision": "",
        "hallucination_grade": "",
        "answer_grade": "",
        "steps": []
    }
    return app.invoke(initial_state)


def stream_agentic_research(question: str) -> Generator[Dict[str, Any], None, None]:
    """Stream graph node updates progressively for real-time visualization."""
    app = build_agentic_rag_graph()
    initial_state: GraphState = {
        "question": question,
        "generation": "",
        "documents": [],
        "search_query": "",
        "web_search_needed": False,
        "router_decision": "",
        "hallucination_grade": "",
        "answer_grade": "",
        "steps": []
    }
    for output in app.stream(initial_state):
        yield output
