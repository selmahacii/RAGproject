"""
RAG Project - Production-Grade Retrieval-Augmented Generation Pipeline

This package provides a complete RAG implementation with:
- Multi-format document ingestion (PDF, Web, CSV)
- Intelligent chunking with overlap
- Local embeddings with sentence-transformers
- ChromaDB vector storage
- FlashRank re-ranking
- ZhipuAI LLM integration
- RAGAs evaluation
"""

__version__ = "1.0.0"
__author__ = "RAG Project"

from .config import settings, get_settings

__all__ = ["settings", "get_settings"]
