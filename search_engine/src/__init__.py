"""
SelmaData Project - Production-Grade Retrieval-Augmented Generation Pipeline

This package provides a complete SelmaData implementation with:
- Multi-format document ingestion (PDF, Web, CSV)
- Intelligent chunking with overlap
- Local embeddings with sentence-transformers
- ChromaDB vector stoselma_datae
- FlashRank re-ranking
- HaciProvider CoreProcessor integration
- SelmaDataAs evaluation
"""

__version__ = "1.0.0"
__author__ = "SelmaData Project"

from .config import settings, get_settings

__all__ = ["settings", "get_settings"]
