"""
Retrieval Module

Provides vector storage and retrieval with re-ranking:
- ChromaDB for local vector storage
- FlashRank for local, free re-ranking
- Hybrid search (vector + keyword)
- Metadata filtering
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

from loguru import logger
from tqdm import tqdm

from .config import get_settings
from .chunking import Chunk
from .embeddings import BaseEmbeddings, SentenceTransformerEmbeddings


@dataclass
class SearchResult:
    """
    Represents a search result with content and score.
    
    Attributes:
        content: The retrieved text content
        score: Relevance score (higher is better)
        metadata: Associated metadata
        rank: Position in results after ranking
    """
    content: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    rank: int = 0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "score": self.score,
            "metadata": self.metadata,
            "rank": self.rank,
        }


class BaseVectorStore(ABC):
    """Abstract base class for vector stores."""
    
    @abstractmethod
    def add(
        self,
        documents: Sequence[Chunk],
        embeddings: Optional[Sequence[list[float]]] = None,
    ) -> None:
        """Add documents to the store."""
        pass
    
    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filter: Optional[dict[str, Any]] = None,
    ) -> list[SearchResult]:
        """Search for similar documents."""
        pass
    
    @abstractmethod
    def delete(self, ids: Optional[list[str]] = None) -> None:
        """Delete documents from the store."""
        pass
    
    @abstractmethod
    def count(self) -> int:
        """Get the number of documents in the store."""
        pass


class ChromaVectorStore(BaseVectorStore):
    """
    ChromaDB vector store for local persistence.
    
    Features:
    - Local persistence (no external database needed)
    - Metadata filtering
    - Collection management
    - Support for multiple embedding functions
    """
    
    def __init__(
        self,
        collection_name: Optional[str] = None,
        persist_directory: Optional[str | Path] = None,
        embedding_function: Optional[BaseEmbeddings] = None,
    ):
        settings = get_settings()
        self.collection_name = collection_name or settings.collection_name
        self.persist_directory = Path(
            persist_directory or settings.persist_directory
        )
        self.embedding_function = embedding_function
        
        self._client = None
        self._collection = None
    
    @property
    def client(self):
        """Lazy load ChromaDB client."""
        if self._client is None:
            try:
                import chromadb
            except ImportError:
                raise ImportError(
                    "chromadb is required. Install with: pip install chromadb"
                )
            
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            
            self._client = chromadb.PersistentClient(
                path=str(self.persist_directory)
            )
            logger.info(f"ChromaDB client initialized at {self.persist_directory}")
        
        return self._client
    
    @property
    def collection(self):
        """Get or create the collection."""
        if self._collection is None:
            # Create embedding function wrapper for ChromaDB
            embedding_func = None
            if self.embedding_function:
                embedding_func = self._create_embedding_function()
            
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=embedding_func,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(f"Collection '{self.collection_name}' loaded with {self._collection.count()} documents")
        
        return self._collection
    
    def _create_embedding_function(self):
        """Create ChromaDB-compatible embedding function."""
        from chromadb.api import EmbeddingFunction
        
        embeddings = self.embedding_function
        
        class CustomEmbeddingFunction(EmbeddingFunction):
            def __init__(self, emb):
                self.emb = emb
            
            def __call__(self, input: list[str]) -> list[list[float]]:
                return self.emb.embed_texts(input)
        
        return CustomEmbeddingFunction(embeddings)
    
    def add(
        self,
        documents: Sequence[Chunk],
        embeddings: Optional[Sequence[list[float]]] = None,
        batch_size: int = 100,
    ) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of Chunk objects to add
            embeddings: Pre-computed embeddings (optional)
            batch_size: Batch size for adding documents
        """
        if not documents:
            logger.warning("No documents to add")
            return
        
        # Generate embeddings if not provided
        if embeddings is None and self.embedding_function:
            logger.info(f"Generating embeddings for {len(documents)} documents")
            embeddings = self.embedding_function.embed_texts(
                [doc.content for doc in documents]
            )
        
        if embeddings is None:
            raise ValueError("Either embeddings or embedding_function must be provided")
        
        # Prepare data
        ids = [doc.chunk_id or f"doc_{i}" for i, doc in enumerate(documents)]
        contents = [doc.content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        # Add in batches
        for i in tqdm(
            range(0, len(documents), batch_size),
            desc="Adding to vector store",
            unit="batch",
        ):
            batch_end = min(i + batch_size, len(documents))
            
            self.collection.add(
                ids=ids[i:batch_end],
                documents=contents[i:batch_end],
                embeddings=embeddings[i:batch_end],
                metadatas=metadatas[i:batch_end],
            )
        
        logger.info(f"Added {len(documents)} documents to collection '{self.collection_name}'")
    
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filter: Optional[dict[str, Any]] = None,
    ) -> list[SearchResult]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter: Metadata filter (e.g., {"source": "file.pdf"})
        
        Returns:
            List of SearchResult objects sorted by relevance
        """
        # Build where clause for filtering
        where = None
        if filter:
            where = self._build_where_clause(filter)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        
        # Convert to SearchResult objects
        search_results = []
        
        if results["documents"] and results["documents"][0]:
            for i, (doc, metadata, distance) in enumerate(zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )):
                # Convert distance to similarity score (1 - distance for cosine)
                score = 1 - distance
                
                search_results.append(SearchResult(
                    content=doc,
                    score=score,
                    metadata=metadata or {},
                    rank=i + 1,
                ))
        
        return search_results
    
    def _build_where_clause(self, filter: dict[str, Any]) -> dict:
        """Build ChromaDB where clause from filter dict."""
        conditions = []
        
        for key, value in filter.items():
            if isinstance(value, dict):
                # Already a ChromaDB operator
                conditions.append({key: value})
            elif isinstance(value, list):
                # $in operator
                conditions.append({key: {"$in": value}})
            else:
                # Exact match
                conditions.append({key: value})
        
        if len(conditions) == 1:
            return conditions[0]
        elif len(conditions) > 1:
            return {"$and": conditions}
        
        return {}
    
    def delete(self, ids: Optional[list[str]] = None) -> None:
        """Delete documents from the store."""
        if ids:
            self.collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} documents")
        else:
            # Delete all documents in collection
            all_ids = self.collection.get()["ids"]
            if all_ids:
                self.collection.delete(ids=all_ids)
                logger.info(f"Deleted all {len(all_ids)} documents from collection")
    
    def delete_collection(self) -> None:
        """Delete the entire collection."""
        self.client.delete_collection(self.collection_name)
        self._collection = None
        logger.info(f"Deleted collection '{self.collection_name}'")
    
    def count(self) -> int:
        """Get the number of documents in the store."""
        return self.collection.count()
    
    def get(
        self,
        ids: Optional[list[str]] = None,
        where: Optional[dict[str, Any]] = None,
    ) -> list[SearchResult]:
        """Get documents by ID or filter."""
        results = self.collection.get(
            ids=ids,
            where=where,
            include=["documents", "metadatas"],
        )
        
        search_results = []
        for doc, metadata in zip(
            results["documents"],
            results["metadatas"],
        ):
            search_results.append(SearchResult(
                content=doc,
                score=1.0,  # No relevance score for direct retrieval
                metadata=metadata or {},
            ))
        
        return search_results


class BaseReranker(ABC):
    """Abstract base class for re-rankers."""
    
    @abstractmethod
    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 3,
    ) -> list[SearchResult]:
        """Re-rank search results."""
        pass


class FlashRankReranker(BaseReranker):
    """
    FlashRank re-ranker for improved retrieval quality.
    
    Re-ranking is a crucial step in production RAG systems:
    - Initial retrieval uses fast vector similarity
    - Re-ranking uses more sophisticated cross-encoder models
    - This improves precision without sacrificing speed
    
    FlashRank is:
    - Completely local (no API calls)
    - Free
    - Fast (optimized for inference)
    
    Recommended models:
    - ms-marco-MiniLM-L-12-v2: Best quality
    - ms-marco-MiniLM-L-6-v2: Faster
    - ms-marco-TinyBERT-L-2-v2: Fastest
    """
    
    # Model sizes for reference
    MODEL_SIZES = {
        "ms-marco-MiniLM-L-12-v2": "120MB",
        "ms-marco-MiniLM-L-6-v2": "80MB",
        "ms-marco-TinyBERT-L-2-v2": "30MB",
        "cross-encoder/ms-marco-MiniLM-L-6-v2": "80MB",
    }
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        max_length: int = 512,
        cache_dir: Optional[str] = None,
    ):
        settings = get_settings()
        self.model_name = model_name or settings.rerank_model
        self.max_length = max_length
        self.cache_dir = cache_dir
        
        self._reranker = None
    
    @property
    def reranker(self):
        """Lazy load the re-ranker model."""
        if self._reranker is None:
            try:
                # Try new API first (flashrank >= 0.3.0)
                try:
                    from flashrank import Ranker, RerankRequest
                    self._reranker = Ranker(model_name=self.model_name)
                    self._use_new_api = True
                    logger.info(f"Loaded Ranker (new API): {self.model_name}")
                except ImportError:
                    # Fall back to old API (flashrank < 0.3.0)
                    from flashrank import RerankRequest, Reranker
                    self._reranker = Reranker(
                        model_name=self.model_name,
                        cache_dir=self.cache_dir,
                        max_length=self.max_length,
                    )
                    self._use_new_api = False
                    logger.info(f"Loaded Reranker (old API): {self.model_name}")
            except ImportError:
                raise ImportError(
                    "flashrank is required. Install with: pip install flashrank"
                )
        
        return self._reranker
    
    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 3,
    ) -> list[SearchResult]:
        """
        Re-rank search results using cross-encoder model.
        
        Args:
            query: The search query
            results: Initial search results from vector search
            top_k: Number of top results to return
        
        Returns:
            Re-ranked results with updated scores
        """
        if not results:
            return []
        
        # Prepare passages for re-ranking
        passages = [
            {"id": i, "text": result.content}
            for i, result in enumerate(results)
        ]
        
        # Create rerank request - handle both API versions
        rerank_request = RerankRequest(
            query=query,
            passages=passages,
        )
        
        # Get reranked results
        reranked = self.reranker.rerank(rerank_request)
        
        # Map back to SearchResult objects
        reranked_results = []
        for i, item in enumerate(reranked[:top_k]):
            idx = item["id"]
            reranked_results.append(SearchResult(
                content=results[idx].content,
                score=item["score"],
                metadata=results[idx].metadata,
                rank=i + 1,
            ))
        
        logger.info(f"Re-ranked {len(results)} results, returning top {len(reranked_results)}")
        
        return reranked_results


class CrossEncoderReranker(BaseReranker):
    """
    Alternative re-ranker using sentence-transformers CrossEncoder.
    
    Similar to FlashRank but using a different library.
    """
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        max_length: int = 512,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self._model = None
    
    @property
    def model(self):
        """Lazy load the cross-encoder model."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError:
                raise ImportError(
                    "sentence-transformers required. "
                    "Install with: pip install sentence-transformers"
                )
            
            logger.info(f"Loading cross-encoder: {self.model_name}")
            self._model = CrossEncoder(
                self.model_name,
                max_length=self.max_length,
            )
        
        return self._model
    
    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 3,
    ) -> list[SearchResult]:
        """Re-rank using cross-encoder."""
        if not results:
            return []
        
        # Prepare pairs for scoring
        pairs = [(query, result.content) for result in results]
        
        # Get scores
        scores = self.model.predict(pairs)
        
        # Sort by score
        scored_results = list(zip(results, scores))
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k
        reranked_results = []
        for i, (result, score) in enumerate(scored_results[:top_k]):
            reranked_results.append(SearchResult(
                content=result.content,
                score=float(score),
                metadata=result.metadata,
                rank=i + 1,
            ))
        
        return reranked_results


class NoopReranker(BaseReranker):
    """
    No-op re-ranker that returns results unchanged.
    Useful for comparing with and without re-ranking.
    """
    
    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 3,
    ) -> list[SearchResult]:
        """Return top-k results unchanged."""
        return results[:top_k]


class HybridRetriever:
    """
    Hybrid retrieval combining vector search and keyword matching.
    
    Features:
    - Combines semantic (vector) and lexical (keyword) search
    - Configurable weighting
    - Re-ranking support
    """
    
    def __init__(
        self,
        vector_store: BaseVectorStore,
        embedding_function: BaseEmbeddings,
        reranker: Optional[BaseReranker] = None,
        alpha: float = 0.7,  # Weight for vector search (1-alpha for keyword)
    ):
        self.vector_store = vector_store
        self.embedding_function = embedding_function
        self.reranker = reranker or NoopReranker()
        self.alpha = alpha
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        rerank_top_k: int = 3,
        filter: Optional[dict[str, Any]] = None,
    ) -> list[SearchResult]:
        """
        Perform hybrid search.
        
        Args:
            query: Search query
            top_k: Number of initial results
            rerank_top_k: Number of results after re-ranking
            filter: Metadata filter
        
        Returns:
            Re-ranked search results
        """
        # Get query embedding
        query_embedding = self.embedding_function.embed_query(query)
        
        # Vector search
        vector_results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filter=filter,
        )
        
        # TODO: Add keyword search (BM25) for true hybrid
        # For now, just use vector results
        
        # Re-rank
        if self.reranker and len(vector_results) > rerank_top_k:
            final_results = self.reranker.rerank(
                query=query,
                results=vector_results,
                top_k=rerank_top_k,
            )
        else:
            final_results = vector_results[:rerank_top_k]
        
        return final_results


class Retriever:
    """
    Main retriever class that orchestrates the retrieval process.
    
    This is the primary interface for RAG retrieval:
    - Handles vector store operations
    - Manages embedding generation
    - Performs re-ranking
    - Provides a simple search interface
    """
    
    def __init__(
        self,
        vector_store: Optional[BaseVectorStore] = None,
        embedding_function: Optional[BaseEmbeddings] = None,
        reranker: Optional[BaseReranker] = None,
        top_k: Optional[int] = None,
        rerank_top_k: Optional[int] = None,
    ):
        settings = get_settings()
        
        # Initialize components
        self.embedding_function = embedding_function or SentenceTransformerEmbeddings()
        
        self.vector_store = vector_store or ChromaVectorStore(
            embedding_function=self.embedding_function,
        )
        
        self.reranker = reranker or FlashRankReranker()
        
        self.top_k = top_k or settings.top_k
        self.rerank_top_k = rerank_top_k or settings.rerank_top_k
    
    def index_documents(
        self,
        documents: Sequence[Chunk],
        batch_size: int = 100,
    ) -> None:
        """
        Index documents into the vector store.
        
        Args:
            documents: List of Chunk objects to index
            batch_size: Batch size for embedding generation
        """
        logger.info(f"Indexing {len(documents)} documents")
        
        self.vector_store.add(
            documents=documents,
            batch_size=batch_size,
        )
    
    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        rerank_top_k: Optional[int] = None,
        filter: Optional[dict[str, Any]] = None,
        rerank: bool = True,
    ) -> list[SearchResult]:
        """
        Search for relevant documents.
        
        Args:
            query: Search query
            top_k: Number of initial results
            rerank_top_k: Number of results after re-ranking
            filter: Metadata filter
            rerank: Whether to apply re-ranking
        
        Returns:
            List of SearchResult objects
        """
        top_k = top_k or self.top_k
        rerank_top_k = rerank_top_k or self.rerank_top_k
        
        # Generate query embedding
        query_embedding = self.embedding_function.embed_query(query)
        
        # Vector search
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filter=filter,
        )
        
        # Re-rank if enabled and we have enough results
        if rerank and self.reranker and len(results) > 0:
            results = self.reranker.rerank(
                query=query,
                results=results,
                top_k=min(rerank_top_k, len(results)),
            )
        
        return results
    
    def get_context(
        self,
        query: str,
        max_tokens: int = 2000,
        separator: str = "\n\n---\n\n",
        **kwargs,
    ) -> tuple[str, list[SearchResult]]:
        """
        Get context for RAG generation.
        
        Returns concatenated context and the source results.
        """
        results = self.search(query, **kwargs)
        
        # Build context with token budget
        context_parts = []
        current_length = 0
        
        for result in results:
            # Rough token estimate (4 chars ≈ 1 token)
            doc_tokens = len(result.content) // 4
            
            if current_length + doc_tokens > max_tokens:
                break
            
            context_parts.append(result.content)
            current_length += doc_tokens
        
        context = separator.join(context_parts)
        
        return context, results[:len(context_parts)]
    
    def count(self) -> int:
        """Get the number of indexed documents."""
        return self.vector_store.count()
    
    def clear(self) -> None:
        """Clear all indexed documents."""
        self.vector_store.delete()


# Factory functions
def get_vector_store(
    provider: str = "chroma",
    **kwargs,
) -> BaseVectorStore:
    """Get vector store by provider name."""
    providers = {
        "chroma": ChromaVectorStore,
    }
    
    if provider not in providers:
        raise ValueError(f"Unknown vector store provider: {provider}")
    
    return providers[provider](**kwargs)


def get_reranker(
    provider: str = "flashrank",
    **kwargs,
) -> BaseReranker:
    """Get re-ranker by provider name."""
    providers = {
        "flashrank": FlashRankReranker,
        "cross_encoder": CrossEncoderReranker,
        "noop": NoopReranker,
    }
    
    if provider not in providers:
        raise ValueError(f"Unknown re-ranker provider: {provider}")
    
    return providers[provider](**kwargs)


# ============================================
# STEP 5: Retrieval with Re-ranking Functions
# ============================================

"""
┌─────────────────────────────────────────────────────────────────────────────┐
│                      RETRIEVAL STRATEGY COMPARISON                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  BASIC RETRIEVAL (basic_retrieval)                                   │    │
│  ├─────────────────────────────────────────────────────────────────────┤    │
│  │  📋 Method: Vector similarity search only                            │    │
│  │  ⏱️  Speed: Fastest (~10-20ms)                                        │    │
│  │  📊 Quality: Good baseline                                            │    │
│  │                                                                      │    │
│  │  Use when:                                                           │    │
│  │  • Quick prototyping                                                 │    │
│  │  • Low latency requirements                                          │    │
│  │  • Simple use cases                                                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  RETRIEVE + RE-RANK (retrieve_and_rerank)                            │    │
│  ├─────────────────────────────────────────────────────────────────────┤    │
│  │  📋 Method: Vector search → Cross-encoder re-ranking                │    │
│  │  ⏱️  Speed: +50ms for re-ranking step                                │    │
│  │  📊 Quality: Significantly better precision                          │    │
│  │                                                                      │    │
│  │  Why re-ranking?                                                     │    │
│  │  • Vector search: Fast but approximate (ANN)                        │    │
│  │  • Re-ranking: Slow but accurate (cross-encoder)                    │    │
│  │  • Best of both: Fast retrieval + Accurate ranking                  │    │
│  │                                                                      │    │
│  │  Use when:                                                           │    │
│  │  • Quality matters more than speed                                   │    │
│  │  • Production RAG systems                                            │    │
│  │  • Complex queries requiring precision                               │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  MMR RETRIEVAL (retrieve_mmr)                                        │    │
│  ├─────────────────────────────────────────────────────────────────────┤    │
│  │  📋 Method: Max Marginal Relevance for diversity                    │    │
│  │  ⏱️  Speed: Similar to basic (~10-30ms)                              │    │
│  │  📊 Quality: Diverse, non-redundant results                          │    │
│  │                                                                      │    │
│  │  Why MMR?                                                            │    │
│  │  • Basic search might return 3 nearly identical chunks              │    │
│  │  • MMR balances relevance + diversity                               │    │
│  │  • λ=0.5: equal weight to both                                      │    │
│  │                                                                      │    │
│  │  Use when:                                                           │    │
│  │  • Need comprehensive coverage of a topic                           │    │
│  │  • Avoiding redundant information                                    │    │
│  │  • Multi-faceted queries                                             │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  🏆 BEST PRACTICE: MMR + Re-ranking                                   │  │
│  │  ─────────────────────────────────────────────────────────────────────│  │
│  │  retrieve_mmr(k=10) → get diverse candidates                         │  │
│  │  then rerank(top=3) → select most relevant                           │  │
│  │                                                                       │  │
│  │  This gives: diverse + accurate = best results                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
"""


def basic_retrieval(
    vectorstore: Any,
    query: str,
    k: int = 5,
) -> list[dict[str, Any]]:
    """
    Perform basic vector similarity search.
    
    This is the simplest retrieval method - just vector similarity.
    Fast but may miss nuanced relevance.
    
    Args:
        vectorstore: LangChain-compatible vector store (e.g., Chroma)
        query: Search query string
        k: Number of results to return (default: 5)
    
    Returns:
        List of dicts with: content, metadata, score, rank
    
    Example:
        >>> from src.embeddings import build_vectorstore
        >>> vectorstore = build_vectorstore(chunks, "./db")
        >>> results = basic_retrieval(vectorstore, "What is RAG?", k=5)
        >>> for r in results:
        ...     print(f"[{r['score']:.3f}] {r['content'][:50]}...")
    """
    import time
    
    start_time = time.time()
    
    # Use similarity_search_with_score for distance scores
    # Returns list of (Document, score) tuples
    results_with_scores = vectorstore.similarity_search_with_score(
        query,
        k=k,
    )
    
    # Convert to our format
    results = []
    for i, (doc, score) in enumerate(results_with_scores):
        # ChromaDB returns distance (lower = more similar)
        # Convert to similarity score (higher = more similar)
        # For cosine: similarity = 1 - distance
        similarity_score = 1 - score if score <= 1 else 1 / (1 + score)
        
        results.append({
            "content": doc.page_content,
            "metadata": doc.metadata,
            "score": float(similarity_score),
            "rank": i + 1,
        })
    
    elapsed = (time.time() - start_time) * 1000
    
    print(f"\n{'='*50}")
    print(f"🔍 Basic Retrieval")
    print(f"{'='*50}")
    print(f"   Query: '{query[:40]}...'")
    print(f"   Results: {len(results)}")
    print(f"   Time: {elapsed:.1f}ms")
    print(f"{'='*50}")
    
    logger.info(f"Basic retrieval: {len(results)} results in {elapsed:.1f}ms")
    
    return results


def retrieve_and_rerank(
    vectorstore: Any,
    query: str,
    k: int = 10,
    rerank_top: int = 3,
    rerank_model: str = "ms-marco-MiniLM-L-12-v2",
) -> list[dict[str, Any]]:
    """
    Retrieve candidates with vector search, then re-rank with FlashRank.
    
    This two-stage approach is the production standard for RAG:
    1. Fast vector search retrieves k candidates
    2. Cross-encoder re-ranks for precision
    3. Return top rerank_top results
    
    Performance Impact:
    ┌──────────────────────────────────────────────────┐
    │ Vector search: ~10-20ms (fast, approximate)      │
    │ Re-ranking:    ~50ms (slower, accurate)          │
    │ Total:         ~60-70ms                          │
    │                                                  │
    │ The +50ms cost is worth it for significantly    │
    │ better retrieval quality.                        │
    └──────────────────────────────────────────────────┘
    
    Why Re-ranking Improves Quality:
    - Vector search uses approximate nearest neighbor (ANN)
    - ANN is fast but can miss nuanced relevance
    - Cross-encoders compare query-doc pairs directly
    - Much more accurate but too slow for full corpus
    - Two-stage = best of both worlds
    
    Args:
        vectorstore: LangChain-compatible vector store
        query: Search query string
        k: Number of initial candidates (default: 10)
        rerank_top: Number of final results after re-ranking (default: 3)
        rerank_model: FlashRank model to use
    
    Returns:
        List of dicts with: content, metadata, score, rank
    
    Example:
        >>> results = retrieve_and_rerank(vectorstore, "What is RAG?", k=10, rerank_top=3)
        >>> print(f"Retrieved {len(results)} re-ranked results")
    """
    import time
    
    start_time = time.time()
    
    # Step 1: Get candidates with vector search
    step1_start = time.time()
    candidates = vectorstore.similarity_search_with_score(
        query,
        k=k,
    )
    step1_time = (time.time() - step1_start) * 1000
    
    if not candidates:
        return []
    
    # Step 2: Re-rank with FlashRank
    step2_start = time.time()
    
    try:
        # Try new API first (flashrank >= 0.3.0 uses Ranker)
        try:
            from flashrank import Ranker
            reranker = Ranker(model_name=rerank_model)
            # New API: rank(query, passages)
            passages = [
                {"id": i, "text": doc.page_content}
                for i, (doc, _) in enumerate(candidates)
            ]
            reranked_results = reranker.rank(query=query, passages=passages)
        except (ImportError, AttributeError):
            # Fall back to old API (flashrank < 0.3.0 uses Reranker + RerankRequest)
            from flashrank import Reranker, RerankRequest
            reranker = Reranker(model_name=rerank_model)
            passages = [
                {"id": i, "text": doc.page_content}
                for i, (doc, _) in enumerate(candidates)
            ]
            rerank_request = RerankRequest(query=query, passages=passages)
            reranked_results = reranker.rerank(rerank_request)
    except ImportError:
        logger.warning("FlashRank not installed, returning basic results")
        return basic_retrieval(vectorstore, query, k=rerank_top)
    
    step2_time = (time.time() - step2_start) * 1000
    total_time = (time.time() - start_time) * 1000
    
    # Step 3: Return top re-ranked results
    results = []
    for i, item in enumerate(reranked_results[:rerank_top]):
        idx = item["id"]
        doc, _ = candidates[idx]
        
        results.append({
            "content": doc.page_content,
            "metadata": doc.metadata,
            "score": float(item["score"]),
            "rank": i + 1,
            "original_rank": idx + 1,  # Where it was before re-ranking
        })
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"🔍 Retrieve + Re-rank")
    print(f"{'='*60}")
    print(f"   Query: '{query[:40]}...'")
    print(f"   Candidates: {k} → Final: {rerank_top}")
    print(f"   ")
    print(f"   ⏱️  Timing:")
    print(f"      Vector search: {step1_time:.1f}ms")
    print(f"      Re-ranking:    {step2_time:.1f}ms  (+{step2_time:.0f}ms cost)")
    print(f"      Total:         {total_time:.1f}ms")
    print(f"   ")
    print(f"   📊 Re-ranking Impact:")
    for r in results:
        if "original_rank" in r:
            change = r["original_rank"] - r["rank"]
            if change > 0:
                print(f"      #{r['rank']} was #{r['original_rank']} (↑{change})")
            elif change < 0:
                print(f"      #{r['rank']} was #{r['original_rank']} (↓{abs(change)})")
            else:
                print(f"      #{r['rank']} unchanged")
    print(f"{'='*60}")
    
    logger.info(f"Retrieve+rerank: {k} → {rerank_top} in {total_time:.1f}ms")
    
    return results


def retrieve_mmr(
    vectorstore: Any,
    query: str,
    k: int = 4,
    fetch_k: Optional[int] = None,
    lambda_mult: float = 0.5,
) -> list[dict[str, Any]]:
    """
    Retrieve diverse results using Max Marginal Relevance (MMR).
    
    MMR solves a common problem in RAG: basic similarity search might
    return 3 nearly identical chunks because they're all similar to
    the query. MMR balances relevance with diversity.
    
    How MMR Works:
    ┌──────────────────────────────────────────────────────────┐
    │ Score = λ × relevance - (1-λ) × max_similarity_to_selected │
    │                                                            │
    │ λ = 0.5: equal balance between relevance and diversity    │
    │ λ = 1.0: pure relevance (like basic search)               │
    │ λ = 0.0: pure diversity (might return irrelevant)         │
    └──────────────────────────────────────────────────────────┘
    
    Example:
    Basic search for "RAG benefits":
      1. "RAG benefits: accuracy" (score: 0.95)
      2. "RAG benefits: accuracy" (score: 0.94) ← duplicate!
      3. "RAG benefits: accuracy" (score: 0.93) ← duplicate!
    
    MMR search for "RAG benefits":
      1. "RAG benefits: accuracy" (score: 0.95)
      2. "RAG benefits: cost savings" (score: 0.88) ← diverse!
      3. "RAG benefits: privacy" (score: 0.82) ← diverse!
    
    Args:
        vectorstore: LangChain-compatible vector store
        query: Search query string
        k: Number of diverse results to return (default: 4)
        fetch_k: Number of candidates to fetch initially (default: k * 4)
        lambda_mult: Balance relevance/diversity (0.0-1.0, default: 0.5)
    
    Returns:
        List of dicts with: content, metadata, score, rank
    
    Example:
        >>> results = retrieve_mmr(vectorstore, "What is RAG?", k=4)
        >>> # Returns 4 diverse, relevant chunks
    """
    import time
    
    start_time = time.time()
    
    # Default: fetch 4x candidates for diversity selection
    if fetch_k is None:
        fetch_k = k * 4
    
    try:
        # Use MMR search
        results = vectorstore.max_marginal_relevance_search(
            query,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
        )
        
        # MMR returns documents without scores, so we compute similarity
        # For Chroma, we can get embeddings and compute similarity
        results_with_scores = []
        for i, doc in enumerate(results):
            # MMR doesn't return scores, use placeholder
            # In production, you'd compute actual similarity
            results_with_scores.append((doc, 0.0))
        
    except AttributeError:
        # Fallback if MMR not supported
        logger.warning("MMR not supported, falling back to basic search")
        return basic_retrieval(vectorstore, query, k=k)
    
    elapsed = (time.time() - start_time) * 1000
    
    # Convert to our format
    final_results = []
    for i, (doc, _) in enumerate(results_with_scores):
        final_results.append({
            "content": doc.page_content,
            "metadata": doc.metadata,
            "score": 0.0,  # MMR doesn't provide scores directly
            "rank": i + 1,
            "method": "mmr",
            "lambda": lambda_mult,
        })
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"🎯 MMR Retrieval (Diversity)")
    print(f"{'='*50}")
    print(f"   Query: '{query[:40]}...'")
    print(f"   Fetched: {fetch_k} candidates")
    print(f"   Returned: {k} diverse results")
    print(f"   Lambda (relevance/diversity): {lambda_mult}")
    print(f"   Time: {elapsed:.1f}ms")
    print(f"   ")
    print(f"   💡 MMR avoids returning nearly identical chunks")
    print(f"{'='*50}")
    
    logger.info(f"MMR retrieval: {k} diverse results in {elapsed:.1f}ms")
    
    return final_results


def retrieve_mmr_and_rerank(
    vectorstore: Any,
    query: str,
    mmr_k: int = 10,
    final_k: int = 3,
    lambda_mult: float = 0.5,
    rerank_model: str = "ms-marco-MiniLM-L-12-v2",
) -> list[dict[str, Any]]:
    """
    🏆 BEST PRACTICE: MMR retrieval followed by re-ranking.
    
    This combines the benefits of both approaches:
    1. MMR retrieves diverse candidates (no redundancy)
    2. Re-ranking selects the most relevant from diverse set
    3. Result: diverse + accurate = best retrieval quality
    
    Why This Works Best:
    ┌──────────────────────────────────────────────────────────┐
    │ Basic search:     Fast, but redundant results            │
    │ MMR only:         Diverse, but might miss best           │
    │ Rerank only:      Accurate, but might be redundant       │
    │ MMR + Rerank:     Diverse AND accurate ✅                │
    └──────────────────────────────────────────────────────────┘
    
    Args:
        vectorstore: LangChain-compatible vector store
        query: Search query string
        mmr_k: Number of diverse candidates from MMR (default: 10)
        final_k: Number of final results after re-ranking (default: 3)
        lambda_mult: MMR relevance/diversity balance (default: 0.5)
        rerank_model: FlashRank model for re-ranking
    
    Returns:
        List of dicts with: content, metadata, score, rank
    
    Example:
        >>> # Best practice retrieval
        >>> results = retrieve_mmr_and_rerank(
        ...     vectorstore,
        ...     "What are the benefits of RAG?",
        ...     mmr_k=10,
        ...     final_k=3
        ... )
    """
    import time
    
    total_start = time.time()
    
    # Step 1: MMR for diverse candidates
    mmr_results = retrieve_mmr(
        vectorstore,
        query,
        k=mmr_k,
        lambda_mult=lambda_mult,
    )
    
    if not mmr_results:
        return []
    
    # Step 2: Re-rank the diverse candidates
    try:
        from flashrank import Reranker, RerankRequest
    except ImportError:
        logger.warning("FlashRank not installed, returning MMR results")
        return mmr_results[:final_k]
    
    reranker = Reranker(model_name=rerank_model)
    
    passages = [
        {"id": i, "text": r["content"]}
        for i, r in enumerate(mmr_results)
    ]
    
    rerank_request = RerankRequest(query=query, passages=passages)
    reranked = reranker.rerank(rerank_request)
    
    # Build final results
    results = []
    for i, item in enumerate(reranked[:final_k]):
        idx = item["id"]
        original = mmr_results[idx]
        
        results.append({
            "content": original["content"],
            "metadata": original["metadata"],
            "score": float(item["score"]),
            "rank": i + 1,
            "mmr_rank": original["rank"],
        })
    
    total_time = (time.time() - total_start) * 1000
    
    print(f"\n{'='*60}")
    print(f"🏆 MMR + Re-rank (Best Practice)")
    print(f"{'='*60}")
    print(f"   Query: '{query[:40]}...'")
    print(f"   Pipeline: MMR({mmr_k}) → Rerank({final_k})")
    print(f"   Total time: {total_time:.1f}ms")
    print(f"   ")
    print(f"   ✅ Diverse candidates + Accurate selection = Best results")
    print(f"{'='*60}")
    
    return results


if __name__ == "__main__":
    # Example usage - STEP 5 Retrieval with Re-ranking
    print("=" * 60)
    print("Retrieval with Re-ranking Module - STEP 5 Implementation")
    print("=" * 60)
    
    # Demo the three retrieval strategies
    print("""
    ┌─────────────────────────────────────────────────────────────┐
    │                 RETRIEVAL STRATEGY COMPARISON                │
    ├─────────────────────────────────────────────────────────────┤
    │                                                              │
    │  1. basic_retrieval()                                        │
    │     • Method: Vector similarity search only                 │
    │     • Speed: Fastest (~10-20ms)                             │
    │     • Use: Quick prototyping, low latency needs             │
    │                                                              │
    │  2. retrieve_and_rerank()                                    │
    │     • Method: Vector search → Cross-encoder re-ranking      │
    │     • Speed: +50ms for re-ranking                           │
    │     • Quality: Significantly better precision               │
    │     • Use: Production RAG, quality-focused                  │
    │                                                              │
    │  3. retrieve_mmr()                                           │
    │     • Method: Max Marginal Relevance                        │
    │     • Speed: Similar to basic                               │
    │     • Quality: Diverse, non-redundant results               │
    │     • Use: Comprehensive coverage, avoiding duplicates      │
    │                                                              │
    │  🏆 BEST PRACTICE: MMR(k=10) → Rerank(top=3)                │
    │     Diverse candidates + Accurate selection = Best results  │
    │                                                              │
    └─────────────────────────────────────────────────────────────┘
    """)
    
    # Create sample data for demonstration
    print("\n--- Setting Up Demo ---")
    print("Creating sample documents and vector store...")
    
    from .chunking import Chunk
    from .embeddings import FakeEmbeddings
    
    # Sample chunks
    chunks = [
        Chunk(
            content="RAG combines retrieval and generation for better answers.",
            metadata={"source": "doc1.txt", "topic": "RAG"},
            chunk_id="doc1_0",
        ),
        Chunk(
            content="Vector databases store embeddings for fast similarity search.",
            metadata={"source": "doc2.txt", "topic": "Vector DB"},
            chunk_id="doc2_0",
        ),
        Chunk(
            content="Re-ranking improves retrieval quality using cross-encoders.",
            metadata={"source": "doc3.txt", "topic": "Re-ranking"},
            chunk_id="doc3_0",
        ),
        Chunk(
            content="MMR (Max Marginal Relevance) ensures diverse search results.",
            metadata={"source": "doc4.txt", "topic": "MMR"},
            chunk_id="doc4_0",
        ),
        Chunk(
            content="FlashRank is a fast, free re-ranking library for production use.",
            metadata={"source": "doc5.txt", "topic": "FlashRank"},
            chunk_id="doc5_0",
        ),
    ]
    
    # Use fake embeddings for demo
    embeddings = FakeEmbeddings(dimension=384)
    
    # Create vector store
    store = ChromaVectorStore(
        collection_name="demo_retrieval_test",
        embedding_function=embeddings,
    )
    
    # Index chunks
    store.add(chunks)
    print(f"✓ Indexed {store.count()} documents")
    
    # Demo 1: Basic Retrieval
    print("\n" + "=" * 60)
    print("DEMO 1: basic_retrieval()")
    print("=" * 60)
    print("""
    >>> results = basic_retrieval(vectorstore, "What is RAG?", k=5)
    
    Features:
    • similarity_search_with_score
    • Returns: content, metadata, score, rank
    • Fastest option (~10-20ms)
    """)
    
    print("\n    To run:")
    print("    >>> from src.retrieval import basic_retrieval")
    print("    >>> results = basic_retrieval(vectorstore, 'What is RAG?')")
    
    # Demo 2: Retrieve and Re-rank
    print("\n" + "=" * 60)
    print("DEMO 2: retrieve_and_rerank()")
    print("=" * 60)
    print("""
    >>> results = retrieve_and_rerank(
    ...     vectorstore,
    ...     "What is RAG?",
    ...     k=10,          # Initial candidates
    ...     rerank_top=3   # Final results
    ... )
    
    Why Re-ranking Works:
    ┌──────────────────────────────────────────────────┐
    │ Vector search: ~10-20ms (fast, approximate)      │
    │ Re-ranking:    ~50ms (slower, accurate)          │
    │                                                  │
    │ The +50ms cost gives significantly better       │
    │ precision - worth it for production!             │
    └──────────────────────────────────────────────────┘
    """)
    
    print("    Model: ms-marco-MiniLM-L-12-v2 (120MB)")
    print("    Requires: pip install flashrank")
    
    # Demo 3: MMR Retrieval
    print("\n" + "=" * 60)
    print("DEMO 3: retrieve_mmr()")
    print("=" * 60)
    print("""
    >>> results = retrieve_mmr(
    ...     vectorstore,
    ...     "What is RAG?",
    ...     k=4,
    ...     fetch_k=16,      # 4x k
    ...     lambda_mult=0.5  # balance relevance/diversity
    ... )
    
    Why MMR?
    • Basic search might return 3 nearly identical chunks!
    • MMR balances relevance + diversity
    • λ=0.5: equal weight to both
    
    Example:
      Basic: "RAG benefits: accuracy" x3 (duplicate!)
      MMR:   "RAG benefits: accuracy", "cost savings", "privacy"
    """)
    
    # Demo 4: Best Practice
    print("\n" + "=" * 60)
    print("DEMO 4: 🏆 Best Practice - MMR + Re-rank")
    print("=" * 60)
    print("""
    >>> from src.retrieval import retrieve_mmr_and_rerank
    >>> 
    >>> results = retrieve_mmr_and_rerank(
    ...     vectorstore,
    ...     "What are the benefits of RAG?",
    ...     mmr_k=10,   # Diverse candidates
    ...     final_k=3   # Best 3 after re-ranking
    ... )
    
    This combines:
    1. MMR → diverse candidates (no redundancy)
    2. Re-rank → most relevant from diverse set
    3. Result: diverse + accurate = BEST QUALITY
    """)
    
    # Summary
    print("\n" + "=" * 60)
    print("STEP 5 COMPLETE - Retrieval Functions Summary")
    print("=" * 60)
    print("""
    ✅ basic_retrieval(vectorstore, query, k=5)
       - Fast vector similarity search
       - Returns: content, metadata, score, rank
    
    ✅ retrieve_and_rerank(vectorstore, query, k=10, rerank_top=3)
       - Vector search → FlashRank re-ranking
       - +50ms cost but significantly better quality
    
    ✅ retrieve_mmr(vectorstore, query, k=4)
       - Max Marginal Relevance for diversity
       - fetch_k = k * 4, lambda_mult = 0.5
       - Avoids returning nearly identical chunks
    
    ✅ retrieve_mmr_and_rerank(...) [bonus - BEST PRACTICE]
       - MMR + Re-ranking combined
       - Diverse AND accurate = best results
    """)
    
    # Cleanup
    store.delete_collection()
    print("\n--- Demo complete, cleaned up test collection ---")
