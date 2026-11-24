"""Configuration module for LLM provider, embeddings, and system settings."""

import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "data" / "chroma_db"

load_dotenv(BASE_DIR / ".env")

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")


def get_llm(temperature: float = 0.0):
    """Instantiate and return the configured Chat LLM based on environment settings."""
    provider = os.getenv("LLM_PROVIDER", LLM_PROVIDER).lower()
    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            api_key=os.getenv("GROQ_API_KEY", GROQ_API_KEY),
            model_name=os.getenv("GROQ_MODEL", GROQ_MODEL),
            temperature=temperature
        )
    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            google_api_key=os.getenv("GEMINI_API_KEY", GEMINI_API_KEY),
            model=os.getenv("GEMINI_MODEL", GEMINI_MODEL),
            temperature=temperature
        )
    else:
        from langchain_ollama import ChatOllama
        return ChatOllama(
            base_url=os.getenv("OLLAMA_BASE_URL", OLLAMA_BASE_URL),
            model=os.getenv("OLLAMA_MODEL", OLLAMA_MODEL),
            temperature=temperature,
            keep_alive="1h",
            num_ctx=4096
        )


def get_embeddings():
    """Instantiate and return the configured embedding model."""
    provider = os.getenv("LLM_PROVIDER", LLM_PROVIDER).lower()
    gemini_key = os.getenv("GEMINI_API_KEY", GEMINI_API_KEY)
    if provider == "gemini" and gemini_key:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(
            google_api_key=gemini_key,
            model="models/text-embedding-004"
        )
    else:
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(
            base_url=os.getenv("OLLAMA_BASE_URL", OLLAMA_BASE_URL),
            model=os.getenv("OLLAMA_EMBED_MODEL", OLLAMA_EMBED_MODEL)
        )
