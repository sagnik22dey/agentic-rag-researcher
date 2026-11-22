"""External tools for the agentic workflow, including web search."""

import warnings
from typing import List
from langchain_core.documents import Document

warnings.filterwarnings("ignore", category=RuntimeWarning)


def search_web(query: str, max_results: int = 4) -> List[Document]:
    """Execute live web search using DuckDuckGo with fallback handling."""
    from duckduckgo_search import DDGS

    documents: List[Document] = []
    try:
        with DDGS() as ddgs:
            raw_results = list(ddgs.text(query, max_results=max_results))
            if not raw_results:
                raw_results = list(ddgs.text(query, max_results=max_results, backend="lite"))

            for res in raw_results:
                content = f"{res.get('title', '')}\n{res.get('body', '')}"
                doc = Document(
                    page_content=content,
                    metadata={"source": res.get("href", "web_search"), "title": res.get("title", "")}
                )
                documents.append(doc)
    except Exception as exc:
        doc = Document(
            page_content=f"Web search unavailable: {str(exc)}",
            metadata={"source": "error", "title": "Search Error"}
        )
        documents.append(doc)

    if not documents:
        doc = Document(
            page_content=f"No external search results found for query: {query}",
            metadata={"source": "empty", "title": "No Results"}
        )
        documents.append(doc)

    return documents
