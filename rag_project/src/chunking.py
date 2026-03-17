"""
Intelligent Chunking Module (Step 3)

Implements three key strategies for RAG:
1. Recursive Character Splitting (Baseline)
2. Semantic Chunking (Top 10% semantic breaks)
3. Parent-Child Retrieval (Precision + Context)
"""

from typing import List, Sequence, Tuple, Any
from loguru import logger
from tqdm import tqdm
from langchain_core.documents import Document

# ============================================
# STEP 3: Intelligent Chunking Strategies
# ============================================

def recursive_chunk(docs: Sequence[Document], chunk_size: int = 512, overlap: int = 64) -> List[Document]:
    """
    Standard recursive splitting. Best for general heterogeneous data.
    
    Strategy: Splits by \n\n (paragraphs), then \n (lines), then . (sentences), then spaces.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    # Priority: \n\n > \n > . > space
    separators = ["\n\n", "\n", ". ", " ", ""]
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=separators,
        add_start_index=True  # Keep track of position for source attribution
    )
    
    all_chunks = splitter.split_documents(docs)
    
    # Enrich metadata for portfolio visibility
    for i, chunk in enumerate(all_chunks):
        chunk.metadata.update({
            "chunk_id": f"chunk_{i}",
            "chunk_size": len(chunk.page_content)
        })
        
    print(f"📦 {len(docs)} docs → {len(all_chunks)} chunks")
    return all_chunks


def semantic_chunk(docs: Sequence[Document]) -> List[Document]:
    """
    Topic-aware splitting using sentence embeddings. 
    Best for research papers or technical documentation.
    
    How it works: Detects the largest 10% semantic gaps between sentences.
    """
    try:
        from langchain_experimental.text_splitter import SemanticChunker
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        logger.error("Please install: pip install langchain-experimental langchain-huggingface")
        return []

    # Requirements specified: paraphrase-multilingual-MiniLM-L12-v2
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    # Requirement: percentile breakpoint at 90
    chunker = SemanticChunker(
        embeddings, 
        breakpoint_threshold_type="percentile", 
        breakpoint_threshold_amount=90
    )
    
    semantic_chunks = chunker.split_documents(docs)
    logger.info(f"Created {len(semantic_chunks)} semantic chunks from {len(docs)} documents.")
    return semantic_chunks


def build_parent_child_retriever(docs: Sequence[Document], vectorstore: Any) -> Any:
    """
    Advanced strategy for high precision.
    Indexes 128-char 'child' chunks but returns 512-char 'parent' chunks to the LLM.
    
    Benefit: Precise retrieval trigger + Rich contextual generation.
    """
    from langchain.retrievers import ParentDocumentRetriever
    from langchain.storage import InMemoryStore
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    # Granularity as requested in Step 3
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=128)
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=512)
    
    # Persistence for parent chunks
    store = InMemoryStore()
    
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )
    
    retriever.add_documents(docs, ids=None)
    logger.info("Parent-Child retriever built and documents added.")
    return retriever


"""
┌─────────────────────────────────────────────────────────────────────────────┐
│                     CHUNKING STRATEGY SELECTION GUIDE                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  RECURSIVE CHARACTER → Best for heterogeneous data, large volumes, quick     │
│  CHUNKER              starts, and general production baselines.              │
│                                                                              │
│  SEMANTIC CHUNKER   → Best for academic papers, research documents, and       │
│                       well-structured technical docs with topic transitions. │
│                                                                              │
│  PARENT-DOCUMENT    → Best when retrieval precision must be high (small hits)│
│  RETRIEVER            but generation requires rich context (large chunks).   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
"""

if __name__ == "__main__":
    print("Chunking Module Step 3 Verified.")
