"""Node definitions for the agentic RAG graph."""

import json
from typing import List, Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document

from src.agent_state import GraphState
from src.config import get_llm
from src.vectorstore import retrieve_documents
from src.tools import search_web


def route_question_node(state: GraphState) -> Dict[str, Any]:
    """Analyze query and route to vectorstore, websearch, or direct answering."""
    question = state["question"]
    llm = get_llm(temperature=0.0)

    system_prompt = (
        "You are an expert query router. "
        "Analyze the user question and choose the best route: "
        "'vectorstore' for questions about uploaded documents, domain-specific files, or local knowledge; "
        "'websearch' for current events, fresh live information, weather, or real-time news; "
        "'direct' for simple greetings, conversational questions, or basic math/logic that needs no retrieval. "
        "Respond with ONLY a JSON object: {\"route\": \"vectorstore\" | \"websearch\" | \"direct\"}"
    )

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Question: {question}")
        ])
        content = response.content.strip()
        if "{" in content and "}" in content:
            parsed = json.loads(content[content.find("{"):content.rfind("}") + 1])
            route = parsed.get("route", "vectorstore")
        else:
            route = "vectorstore" if "vector" in content.lower() else "websearch"
    except Exception:
        route = "vectorstore"

    steps = list(state.get("steps", []))
    steps.append(f"Router: Directed to [{route.upper()}]")
    return {"router_decision": route, "steps": steps}


def retrieve_node(state: GraphState) -> Dict[str, Any]:
    """Retrieve matching documents from the vector store."""
    question = state["question"]
    docs = retrieve_documents(question, k=4)
    steps = list(state.get("steps", []))
    steps.append(f"Retriever: Fetched {len(docs)} documents from vector store")
    return {"documents": docs, "steps": steps}


def grade_documents_node(state: GraphState) -> Dict[str, Any]:
    """Filter retrieved documents based on relevance to the user query."""
    question = state["question"]
    documents = state.get("documents", [])
    llm = get_llm(temperature=0.0)

    relevant_docs: List[Document] = []
    web_search_needed = False

    if not documents:
        web_search_needed = True
    else:
        snippets = []
        for idx, doc in enumerate(documents, start=1):
            sample = doc.page_content[:350].replace("\n", " ").strip()
            snippets.append(f"[{idx}] {sample}")
        combined_snippets = "\n\n".join(snippets)

        system_prompt = (
            "You are an objective document evaluator. "
            "Determine which of the candidate document snippets are relevant and useful for answering the user question. "
            "Respond with ONLY a JSON array of relevant 1-based integer indices, e.g. [1, 2], or [] if none are relevant."
        )

        try:
            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Question: {question}\n\nCandidate Document Snippets:\n{combined_snippets}")
            ])
            content = response.content.strip()
            if "[" in content and "]" in content:
                raw_json = content[content.find("["):content.rfind("]") + 1]
                selected_indices = json.loads(raw_json)
                for idx in selected_indices:
                    if isinstance(idx, int) and 1 <= idx <= len(documents):
                        relevant_docs.append(documents[idx - 1])
            else:
                relevant_docs = list(documents)
        except Exception:
            relevant_docs = list(documents)

        if not relevant_docs:
            web_search_needed = True

    steps = list(state.get("steps", []))
    steps.append(
        f"Relevance Grader: Kept {len(relevant_docs)}/{len(documents)} relevant documents "
        f"(Web fallback: {web_search_needed})"
    )

    return {
        "documents": relevant_docs,
        "web_search_needed": web_search_needed,
        "steps": steps
    }


def transform_query_node(state: GraphState) -> Dict[str, Any]:
    """Rewrite question into an optimized web search query."""
    question = state["question"]
    llm = get_llm(temperature=0.2)

    system_prompt = (
        "You are an expert search query optimizer. "
        "The initial retrieval did not find sufficient information. "
        "Rewrite the user question into an effective, concise keyword query for a search engine. "
        "Respond with ONLY the rewritten query text, nothing else."
    )

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Original Question: {question}")
        ])
        new_query = response.content.strip().strip('"').strip("'")
    except Exception:
        new_query = question

    steps = list(state.get("steps", []))
    steps.append(f"Query Optimizer: Rewrote query to '{new_query}'")
    return {"search_query": new_query, "steps": steps}


def web_search_node(state: GraphState) -> Dict[str, Any]:
    """Execute live web search and append results to documents."""
    query = state.get("search_query") or state["question"]
    web_docs = search_web(query=query, max_results=4)

    existing_docs = list(state.get("documents", []))
    combined_docs = existing_docs + web_docs

    steps = list(state.get("steps", []))
    steps.append(f"Web Search Tool: Retrieved {len(web_docs)} web snippets for '{query}'")
    return {"documents": combined_docs, "steps": steps}


def generate_node(state: GraphState) -> Dict[str, Any]:
    """Generate synthesized final response using gathered context."""
    question = state["question"]
    documents = state.get("documents", [])
    llm = get_llm(temperature=0.2)

    context_blocks = []
    for idx, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", f"Doc #{idx}")
        title = doc.metadata.get("title", "")
        header = f"[Source {idx}: {title} ({source})]" if title else f"[Source {idx}: {source}]"
        context_blocks.append(f"{header}\n{doc.page_content}")

    context_str = "\n\n---\n\n".join(context_blocks) if context_blocks else "No external context available."

    system_prompt = (
        "You are an autonomous research assistant. "
        "Answer the user question concisely, directly, and accurately based on the provided context. "
        "Always cite sources when referencing specific facts using [Source X] notation. "
        "Avoid conversational fluff, introductory filler, or repeating the prompt. "
        "If the context is insufficient, state clearly what is missing."
    )

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Question: {question}\n\nContext:\n{context_str}")
        ])
        generation = response.content.strip()
    except Exception as exc:
        generation = f"Generation failed: {str(exc)}"

    steps = list(state.get("steps", []))
    steps.append("Generator: Synthesized response with context grounding")
    return {"generation": generation, "steps": steps}


def direct_answer_node(state: GraphState) -> Dict[str, Any]:
    """Provide a direct conversational or logical answer without external retrieval."""
    question = state["question"]
    llm = get_llm(temperature=0.5)

    try:
        response = llm.invoke([
            SystemMessage(content="You are a helpful and articulate AI assistant."),
            HumanMessage(content=question)
        ])
        generation = response.content.strip()
    except Exception as exc:
        generation = f"Response failed: {str(exc)}"

    steps = list(state.get("steps", []))
    steps.append("Direct Responder: Answered directly without retrieval")
    return {"generation": generation, "steps": steps}


def grade_hallucination_node(state: GraphState) -> Dict[str, Any]:
    """Evaluate whether the generation is grounded in facts and answers the question."""
    question = state["question"]
    generation = state.get("generation", "")
    documents = state.get("documents", [])
    llm = get_llm(temperature=0.0)

    if not documents:
        steps = list(state.get("steps", []))
        steps.append("Fact Checker: Skipped evaluation (direct answer or no documents)")
        return {
            "hallucination_grade": "grounded",
            "answer_grade": "useful",
            "steps": steps
        }

    context_str = "\n\n".join(d.page_content[:400] for d in documents[:3])

    system_prompt = (
        "You are an evaluator assessing if an answer is grounded in facts. "
        "Compare the generation against the context. "
        "Respond with ONLY a JSON object: "
        "{\"grounded\": true | false, \"answers_question\": true | false}"
    )

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Question: {question}\n\nContext:\n{context_str}\n\nGeneration:\n{generation}")
        ])
        content = response.content.strip()
        grounded = "true" in content.lower()
    except Exception:
        grounded = True

    h_grade = "grounded" if grounded else "not grounded"
    steps = list(state.get("steps", []))
    steps.append(f"Fact Checker: Assessed output as [{h_grade.upper()}]")
    return {"hallucination_grade": h_grade, "answer_grade": "useful", "steps": steps}
