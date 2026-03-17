"""
Embeddings and Vector Store Module (Step 4)

This module handles the transformation of text chunks into vector representations
and manages their persistence in a local ChromaDB vector store.
"""

import os
from typing import List, Optional, Any
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from loguru import logger

# ============================================
# STEP 4: Embeddings & Vector Store
# ============================================

def get_embeddings(provider: str = "local") -> Any:
    """
    Factory function to get embedding models.
    
    local=free/80ms/good multilingual
    openai=$0.02/1M tokens/faster
    """
    if provider == "local":
        # Using a proven multilingual model for local, free, and robust retrieval
        # normalize=True ensures cosine similarity works correctly
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            encode_kwargs={'normalize_embeddings': True}
        )
    elif provider == "openai":
        # Faster API-based embeddings with high dimension configurability
        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            dimensions=512
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def build_vectorstore(chunks: List[Document], persist_dir: str = "./chroma_db", provider: str = "local") -> Chroma:
    """
    Initialize a Chroma vector store from documents and persist it.
    
    Why HNSW:Space='cosine'? 
    Cosine similarity is superior to L2 (Euclidean) for text embeddings 
    as it focuses on the direction (semantic meaning) rather than magnitude.
    """
    embeddings = get_embeddings(provider=provider)
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_metadata={"hnsw:space": "cosine"}
    )
    
    # Requirement: Print count of indexed vectors
    count = vectorstore._collection.count()
    print(f"✅ Indexed {count} vectors into ChromaDB at {persist_dir}")
    
    return vectorstore


def load_vectorstore(persist_dir: str = "./chroma_db", provider: str = "local") -> Chroma:
    """
    Load an existing Chroma store from disk. 
    Crucial for production to avoid re-indexing on every restart.
    """
    embeddings = get_embeddings(provider=provider)
    
    if not os.path.exists(persist_dir):
        logger.warning(f"Persistence directory {persist_dir} does not exist. Creating new store.")
    
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )
    
    return vectorstore

if __name__ == "__main__":
    print("Embeddings & Vector Store module ready.")
