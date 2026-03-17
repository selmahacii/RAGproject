"""
Embeddings Module

Provides embedding generation using sentence-transformers:
- Local, free embeddings (no API calls)
- Support for multiple models (BGE, all-MiniLM, etc.)
- Batch processing with progress bars
- Caching for efficiency
"""

import hashlib
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
from loguru import logger
from tqdm import tqdm

from .config import get_settings
from .chunking import Chunk


class BaseEmbeddings(ABC):
    """Abstract base class for embedding models."""
    
    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts."""
        pass
    
    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """Embed a single query."""
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Get embedding dimension."""
        pass


class SentenceTransformerEmbeddings(BaseEmbeddings):
    """
    Local embeddings using Sentence Transformers.
    
    This is the recommended approach for RAG:
    - No API costs
    - Fast inference
    - Privacy (data stays local)
    - Multiple model options
    
    Recommended models:
    - BAAI/bge-m3: Best overall performance, multilingual
    - BAAI/bge-large-en-v1.5: Great for English
    - sentence-transformers/all-MiniLM-L6-v2: Fast, lightweight
    - intfloat/e5-large-v2: Strong retrieval performance
    """
    
    # Popular embedding models with their dimensions
    MODEL_DIMENSIONS = {
        "BAAI/bge-m3": 1024,
        "BAAI/bge-large-en-v1.5": 1024,
        "BAAI/bge-base-en-v1.5": 768,
        "BAAI/bge-small-en-v1.5": 384,
        "sentence-transformers/all-MiniLM-L6-v2": 384,
        "sentence-transformers/all-mpnet-base-v2": 768,
        "intfloat/e5-large-v2": 1024,
        "intfloat/e5-base-v2": 768,
        "intfloat/e5-small-v2": 384,
        "intfloat/multilingual-e5-large": 1024,
    }
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        cache_folder: Optional[str] = None,
        normalize_embeddings: bool = True,
        batch_size: int = 32,
        show_progress_bar: bool = True,
    ):
        settings = get_settings()
        self.model_name = model_name or settings.embedding_model
        self.device = device
        self.cache_folder = cache_folder
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar
        
        self._model = None
        self._dimension = self.MODEL_DIMENSIONS.get(self.model_name)
    
    @property
    def model(self):
        """Lazy load the model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required. "
                    "Install with: pip install sentence-transformers"
                )
            
            logger.info(f"Loading embedding model: {self.model_name}")
            
            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
                cache_folder=self.cache_folder,
            )
            
            # Update dimension from model if not known
            if self._dimension is None:
                self._dimension = self._model.get_sentence_embedding_dimension()
            
            logger.info(f"Embedding dimension: {self._dimension}")
        
        return self._model
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        if self._dimension is None:
            _ = self.model  # Force model loading
        return self._dimension
    
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of texts.
        
        Uses batch processing for efficiency with progress bar.
        """
        if not texts:
            return []
        
        logger.info(f"Embedding {len(texts)} texts with {self.model_name}")
        
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
        )
        
        return embeddings.tolist()
    
    def embed_query(self, query: str) -> list[float]:
        """
        Embed a single query.
        
        For queries, we might want different handling (e.g., adding
        "query: " prefix for E5 models).
        """
        # Some models like E5 expect a prefix for queries
        if "e5" in self.model_name.lower():
            query = f"query: {query}"
        
        embedding = self.model.encode(
            query,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
        )
        
        return embedding.tolist()
    
    def embed_documents(
        self,
        documents: Sequence[Chunk],
        include_content: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Embed document chunks and return with metadata.
        
        Args:
            documents: List of Chunk objects
            include_content: Whether to include content in output
        
        Returns:
            List of dicts with embedding and metadata
        """
        texts = [doc.content for doc in documents]
        embeddings = self.embed_texts(texts)
        
        results = []
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            result = {
                "id": doc.chunk_id or f"chunk_{i}",
                "embedding": embedding,
                "metadata": doc.metadata,
            }
            if include_content:
                result["content"] = doc.content
            results.append(result)
        
        return results


class CachedEmbeddings(BaseEmbeddings):
    """
    Wrapper that adds caching to any embedding model.
    
    Caches embeddings to disk to avoid recomputing.
    Useful during development and testing.
    """
    
    def __init__(
        self,
        embeddings: BaseEmbeddings,
        cache_dir: str = "./data/embedding_cache",
        use_checksum: bool = True,
    ):
        self.embeddings = embeddings
        self.cache_dir = Path(cache_dir)
        self.use_checksum = use_checksum
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Use model name in cache file
        model_hash = hashlib.md5(
            embeddings.model_name.encode()
        ).hexdigest()[:8]
        self.cache_file = self.cache_dir / f"embeddings_{model_hash}.json"
        
        self._cache = self._load_cache()
    
    def _load_cache(self) -> dict[str, list[float]]:
        """Load cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        return {}
    
    def _save_cache(self) -> None:
        """Save cache to disk."""
        try:
            with open(self.cache_file, "w") as f:
                json.dump(self._cache, f)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        if self.use_checksum:
            return hashlib.md5(text.encode()).hexdigest()
        return text
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self.embeddings.dimension
    
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed texts with caching."""
        results = []
        uncached_indices = []
        uncached_texts = []
        
        # Check cache for each text
        for i, text in enumerate(texts):
            key = self._get_cache_key(text)
            if key in self._cache:
                results.append(self._cache[key])
            else:
                results.append(None)
                uncached_indices.append(i)
                uncached_texts.append(text)
        
        # Embed uncached texts
        if uncached_texts:
            logger.info(f"Computing {len(uncached_texts)} new embeddings")
            new_embeddings = self.embeddings.embed_texts(uncached_texts)
            
            # Update results and cache
            for idx, text, embedding in zip(
                uncached_indices, uncached_texts, new_embeddings
            ):
                results[idx] = embedding
                self._cache[self._get_cache_key(text)] = embedding
            
            self._save_cache()
        
        return results
    
    def embed_query(self, query: str) -> list[float]:
        """Embed query with caching."""
        key = self._get_cache_key(query)
        
        if key in self._cache:
            return self._cache[key]
        
        embedding = self.embeddings.embed_query(query)
        self._cache[key] = embedding
        self._save_cache()
        
        return embedding


class OpenAIEmbeddings(BaseEmbeddings):
    """
    OpenAI embeddings (optional, for comparison).
    
    Note: This requires an OpenAI API key and incurs costs.
    For production RAG, local embeddings are recommended.
    """
    
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
    ):
        self.model = model
        self.api_key = api_key
        
        # Dimensions for OpenAI models
        self._dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimensions.get(self.model, 1536)
    
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using OpenAI API."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required: pip install openai")
        
        client = OpenAI(api_key=self.api_key)
        
        response = client.embeddings.create(
            model=self.model,
            input=texts,
        )
        
        return [item.embedding for item in response.data]
    
    def embed_query(self, query: str) -> list[float]:
        """Embed query using OpenAI API."""
        return self.embed_texts([query])[0]


class ZhipuAIEmbeddings(BaseEmbeddings):
    """
    ZhipuAI embeddings using their API.
    
    Supports GLM embedding models for Chinese and English.
    """
    
    def __init__(
        self,
        model: str = "embedding-3",
        api_key: Optional[str] = None,
    ):
        self.model = model
        self._api_key = api_key
        self._client = None
        
        # Dimensions for ZhipuAI models
        self._dimensions = {
            "embedding-3": 1024,
            "embedding-2": 1024,
        }
    
    @property
    def client(self):
        """Lazy load ZhipuAI client."""
        if self._client is None:
            try:
                from zhipuai import ZhipuAI
            except ImportError:
                raise ImportError(
                    "zhipuai package required: pip install zhipuai"
                )
            
            settings = get_settings()
            api_key = self._api_key or settings.get_api_key()
            self._client = ZhipuAI(api_key=api_key)
        
        return self._client
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimensions.get(self.model, 1024)
    
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using ZhipuAI API."""
        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
        )
        
        return [item.embedding for item in response.data]
    
    def embed_query(self, query: str) -> list[float]:
        """Embed query using ZhipuAI API."""
        return self.embed_texts([query])[0]


class FakeEmbeddings(BaseEmbeddings):
    """
    Fake embeddings for testing.
    
    Generates random embeddings without loading any model.
    Useful for quick testing and development.
    """
    
    def __init__(self, dimension: int = 384):
        self._dimension = dimension
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate fake embeddings."""
        np.random.seed(42)  # Reproducibility
        return np.random.randn(len(texts), self._dimension).tolist()
    
    def embed_query(self, query: str) -> list[float]:
        """Generate fake query embedding."""
        np.random.seed(43)
        return np.random.randn(self._dimension).tolist()


# Utility functions
def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    a_array = np.array(a)
    b_array = np.array(b)
    
    return float(
        np.dot(a_array, b_array) / 
        (np.linalg.norm(a_array) * np.linalg.norm(b_array))
    )


def compute_similarities(
    query_embedding: list[float],
    doc_embeddings: list[list[float]],
    top_k: int = 5,
) -> list[tuple[int, float]]:
    """
    Compute similarities between query and documents.
    
    Returns indices and scores of top-k most similar documents.
    """
    query_array = np.array(query_embedding)
    doc_array = np.array(doc_embeddings)
    
    # Normalize for cosine similarity
    query_norm = query_array / np.linalg.norm(query_array)
    doc_norms = doc_array / np.linalg.norm(doc_array, axis=1, keepdims=True)
    
    # Compute similarities
    similarities = np.dot(doc_norms, query_norm)
    
    # Get top-k indices
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    return [(int(i), float(similarities[i])) for i in top_indices]


# Factory function (legacy, kept for backward compatibility)
def get_embeddings_legacy(
    provider: str = "sentence_transformer",
    **kwargs,
) -> BaseEmbeddings:
    """
    Get embeddings by provider name (legacy version).
    
    Args:
        provider: Embedding provider ("sentence_transformer", "openai", "zhipuai", "fake")
        **kwargs: Additional arguments for the embedding model
    
    Returns:
        BaseEmbeddings instance
    """
    providers = {
        "sentence_transformer": SentenceTransformerEmbeddings,
        "openai": OpenAIEmbeddings,
        "zhipuai": ZhipuAIEmbeddings,
        "fake": FakeEmbeddings,
    }
    
    if provider not in providers:
        raise ValueError(f"Unknown embedding provider: {provider}")
    
    embeddings = providers[provider](**kwargs)
    
    # Optionally wrap with caching
    if kwargs.get("use_cache", False) and provider != "fake":
        cache_dir = kwargs.get("cache_dir", "./data/embedding_cache")
        embeddings = CachedEmbeddings(embeddings, cache_dir=cache_dir)
    
    return embeddings


# ============================================
# STEP 4: Embeddings & Vector Store Functions
# ============================================

"""
┌─────────────────────────────────────────────────────────────────────────────┐
│                    EMBEDDING PROVIDER COMPARISON                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  LOCAL (HuggingFaceEmbeddings)                                      │    │
│  ├─────────────────────────────────────────────────────────────────────┤    │
│  │  💰 Cost: FREE                                                      │    │
│  │  ⏱️  Speed: ~80ms per query (CPU), ~20ms (GPU)                       │    │
│  │  🌍 Multilingual: Excellent (paraphrase-multilingual-MiniLM-L12-v2) │    │
│  │  🔒 Privacy: 100% local, data never leaves your machine            │    │
│  │  📦 Model: paraphrase-multilingual-MiniLM-L12-v2 (384 dims)        │    │
│  │                                                                      │    │
│  │  Best for: Development, privacy-sensitive apps, unlimited usage    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  OPENAI (OpenAIEmbeddings)                                          │    │
│  ├─────────────────────────────────────────────────────────────────────┤    │
│  │  💰 Cost: $0.02 per 1M tokens                                       │    │
│  │  ⏱️  Speed: ~50ms per query (API latency)                            │    │
│  │  🌍 Multilingual: Good (optimized for English)                      │    │
│  │  🔒 Privacy: Data sent to OpenAI servers                            │    │
│  │  📦 Model: text-embedding-3-small (512 dims configurable)           │    │
│  │                                                                      │    │
│  │  Best for: Production scale, consistent quality, no GPU needed     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  Why cosine similarity?                                               │  │
│  │  ─────────────────────────────────────────────────────────────────────│  │
│  │  • Measures angle between vectors (direction), not magnitude          │  │
│  │  • More intuitive for text: "similar meaning" ≈ "similar direction"   │  │
│  │  • L2 distance can be skewed by vector length (document length)       │  │
│  │  • Normalized embeddings make cosine = dot product (faster)           │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
"""


def get_embeddings(
    provider: str = "local",
    model_name: Optional[str] = None,
    **kwargs,
) -> BaseEmbeddings:
    """
    Get embeddings by provider with sensible defaults for RAG.
    
    This function provides a simple interface to get embedding models
    optimized for different use cases.
    
    Provider Comparison:
    ┌─────────────┬────────┬───────────┬─────────────────────────────────┐
    │ Provider    │ Cost   │ Speed     │ Best For                        │
    ├─────────────┼────────┼───────────┼─────────────────────────────────┤
    │ local       │ FREE   │ ~80ms     │ Dev, privacy, unlimited usage   │
    │ openai      │ $0.02/1M│ ~50ms    │ Production, scale, no GPU       │
    └─────────────┴────────┴───────────┴─────────────────────────────────┘
    
    Args:
        provider: "local" (HuggingFace) or "openai" (OpenAI API)
        model_name: Override default model (optional)
        **kwargs: Additional provider-specific arguments
    
    Returns:
        BaseEmbeddings instance ready for use
    
    Examples:
        >>> # Local embeddings (free, multilingual)
        >>> embeddings = get_embeddings("local")
        >>> vectors = embeddings.embed_texts(["Hello world", "你好世界"])
        
        >>> # OpenAI embeddings (faster, paid)
        >>> embeddings = get_embeddings("openai")
        >>> vectors = embeddings.embed_texts(["Hello world"])
    """
    if provider == "local":
        # Default: paraphrase-multilingual-MiniLM-L12-v2
        # - 384 dimensions
        # - 50+ languages supported
        # - Fast inference
        # - FREE
        default_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        
        return SentenceTransformerEmbeddings(
            model_name=model_name or default_model,
            normalize_embeddings=True,  # Normalize for cosine similarity
            **kwargs,
        )
    
    elif provider == "openai":
        # Default: text-embedding-3-small
        # - 512 dimensions (configurable)
        # - Fast API
        # - $0.02 per 1M tokens
        default_model = "text-embedding-3-small"
        
        return OpenAIEmbeddings(
            model=model_name or default_model,
            **kwargs,
        )
    
    elif provider == "zhipuai":
        # Default: embedding-3
        # - Good for Chinese
        default_model = "embedding-3"
        
        return ZhipuAIEmbeddings(
            model=model_name or default_model,
            **kwargs,
        )
    
    elif provider == "fake":
        # For testing
        return FakeEmbeddings(dimension=kwargs.get("dimension", 384))
    
    else:
        raise ValueError(
            f"Unknown provider: {provider}. "
            f"Choose from: 'local', 'openai', 'zhipuai', 'fake'"
        )


def build_vectorstore(
    chunks: Sequence[Chunk],
    persist_dir: str = "./chroma_db",
    collection_name: str = "rag_collection",
    embeddings: Optional[BaseEmbeddings] = None,
    embeddings_provider: str = "local",
) -> Any:
    """
    Build a Chroma vector store from document chunks.
    
    This function creates a persistent vector store using ChromaDB,
    which stores embeddings for efficient similarity search.
    
    Why ChromaDB?
    - Local persistence (no external database needed)
    - HNSW indexing for fast ANN search
    - Metadata filtering support
    - Easy to use with LangChain
    
    Why Cosine Similarity?
    - Better for text than L2 distance
    - Measures semantic similarity (direction, not magnitude)
    - Less affected by document length
    - Works well with normalized embeddings
    
    Args:
        chunks: List of Chunk objects to index
        persist_dir: Directory to store ChromaDB data
        collection_name: Name for the ChromaDB collection
        embeddings: Embedding model instance (optional)
        embeddings_provider: Provider if embeddings not provided ("local" or "openai")
    
    Returns:
        Chroma vectorstore instance
    
    Example:
        >>> from src.chunking import recursive_chunk
        >>> from src.ingestion import Document
        >>> 
        >>> docs = [Document(content="RAG is...", metadata={})]
        >>> chunks = recursive_chunk(docs)
        >>> vectorstore = build_vectorstore(chunks, persist_dir="./my_db")
        >>> 
        >>> # Search
        >>> results = vectorstore.similarity_search("What is RAG?", k=3)
    """
    try:
        from langchain_community.vectorstores import Chroma
        from langchain_core.documents import Document as LCDocument
    except ImportError as e:
        logger.error(f"Required packages not installed: {e}")
        raise ImportError(
            "Install with: pip install langchain-community langchain-core chromadb"
        )
    
    if not chunks:
        raise ValueError("No chunks provided to index")
    
    # Get embeddings if not provided
    if embeddings is None:
        embeddings = get_embeddings(embeddings_provider)
    
    # Ensure persist directory exists
    persist_path = Path(persist_dir)
    persist_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Building vector store with {len(chunks)} chunks")
    logger.info(f"Persist directory: {persist_dir}")
    logger.info(f"Collection: {collection_name}")
    
    # Convert chunks to LangChain documents
    lc_docs = [
        LCDocument(
            page_content=chunk.content,
            metadata=chunk.metadata,
        )
        for chunk in chunks
    ]
    
    # Build the vector store
    # Using cosine similarity (better for text than L2)
    vectorstore = Chroma.from_documents(
        documents=lc_docs,
        embedding=embeddings,
        persist_directory=str(persist_path),
        collection_name=collection_name,
        collection_metadata={
            "hnsw:space": "cosine",  # Cosine similarity for text
            "hnsw:construction_ef": 200,  # Higher = better recall, slower build
            "hnsw:M": 16,  # Connections per node
        },
    )
    
    # Get count
    count = vectorstore._collection.count()
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"📦 Vector Store Built Successfully")
    print(f"{'='*60}")
    print(f"   Chunks indexed: {count}")
    print(f"   Persist directory: {persist_dir}")
    print(f"   Collection: {collection_name}")
    print(f"   Similarity metric: cosine")
    print(f"   Embedding provider: {embeddings_provider}")
    print(f"{'='*60}")
    
    logger.info(f"Vector store built: {count} vectors indexed")
    
    return vectorstore


def load_vectorstore(
    persist_dir: str = "./chroma_db",
    collection_name: str = "rag_collection",
    embeddings: Optional[BaseEmbeddings] = None,
    embeddings_provider: str = "local",
) -> Any:
    """
    Load an existing Chroma vector store from disk.
    
    This function loads a previously created vector store, avoiding
    the need to re-index documents every time you restart.
    
    Important:
    - The embedding model MUST be the same one used during indexing
    - Different models produce different embedding spaces
    - Mixing models will result in poor retrieval quality
    
    Args:
        persist_dir: Directory where ChromaDB data is stored
        collection_name: Name of the ChromaDB collection
        embeddings: Embedding model instance (optional)
        embeddings_provider: Provider if embeddings not provided
    
    Returns:
        Chroma vectorstore instance ready for search
    
    Raises:
        FileNotFoundError: If persist_dir doesn't exist
        ValueError: If collection is empty or doesn't exist
    
    Example:
        >>> # Load existing vectorstore
        >>> vectorstore = load_vectorstore("./my_db")
        >>> 
        >>> # Search
        >>> results = vectorstore.similarity_search("What is RAG?", k=3)
        >>> for doc in results:
        ...     print(doc.page_content[:100])
    """
    try:
        from langchain_community.vectorstores import Chroma
    except ImportError as e:
        logger.error(f"Required packages not installed: {e}")
        raise ImportError(
            "Install with: pip install langchain-community chromadb"
        )
    
    persist_path = Path(persist_dir)
    
    if not persist_path.exists():
        raise FileNotFoundError(
            f"Vector store not found at {persist_dir}. "
            f"Run build_vectorstore() first to create it."
        )
    
    # Get embeddings if not provided
    if embeddings is None:
        embeddings = get_embeddings(embeddings_provider)
    
    logger.info(f"Loading vector store from {persist_dir}")
    
    try:
        # Load the existing vector store
        vectorstore = Chroma(
            persist_directory=str(persist_path),
            embedding_function=embeddings,
            collection_name=collection_name,
        )
        
        # Get count
        count = vectorstore._collection.count()
        
        if count == 0:
            raise ValueError(
                f"Collection '{collection_name}' is empty. "
                f"Run build_vectorstore() to add documents."
            )
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"📂 Vector Store Loaded Successfully")
        print(f"{'='*60}")
        print(f"   Vectors in store: {count}")
        print(f"   Persist directory: {persist_dir}")
        print(f"   Collection: {collection_name}")
        print(f"   Embedding provider: {embeddings_provider}")
        print(f"{'='*60}")
        
        logger.info(f"Vector store loaded: {count} vectors")
        
        return vectorstore
        
    except Exception as e:
        logger.error(f"Failed to load vector store: {e}")
        raise


def get_or_create_vectorstore(
    chunks: Optional[Sequence[Chunk]] = None,
    persist_dir: str = "./chroma_db",
    collection_name: str = "rag_collection",
    embeddings: Optional[BaseEmbeddings] = None,
    embeddings_provider: str = "local",
    force_rebuild: bool = False,
) -> Any:
    """
    Get existing vectorstore or create new one if it doesn't exist.
    
    This is a convenience function that handles both cases:
    - If vectorstore exists: load it
    - If not: build it from chunks
    
    Args:
        chunks: Chunks to index (required if building new)
        persist_dir: Directory for ChromaDB data
        collection_name: Collection name
        embeddings: Embedding model instance
        embeddings_provider: Provider for embeddings
        force_rebuild: Force rebuild even if exists
    
    Returns:
        Chroma vectorstore instance
    
    Example:
        >>> # First run: builds from chunks
        >>> vs = get_or_create_vectorstore(chunks, "./my_db")
        >>> 
        >>> # Later runs: loads from disk
        >>> vs = get_or_create_vectorstore(persist_dir="./my_db")
    """
    persist_path = Path(persist_dir)
    
    # Check if we should load existing
    if persist_path.exists() and not force_rebuild:
        try:
            return load_vectorstore(
                persist_dir=persist_dir,
                collection_name=collection_name,
                embeddings=embeddings,
                embeddings_provider=embeddings_provider,
            )
        except (FileNotFoundError, ValueError):
            pass  # Fall through to build
    
    # Build new
    if chunks is None:
        raise ValueError(
            "No existing vectorstore found and no chunks provided. "
            "Pass chunks to build a new vectorstore."
        )
    
    return build_vectorstore(
        chunks=chunks,
        persist_dir=persist_dir,
        collection_name=collection_name,
        embeddings=embeddings,
        embeddings_provider=embeddings_provider,
    )


if __name__ == "__main__":
    # Example usage - STEP 4 Implementation
    print("=" * 60)
    print("Embeddings & Vector Store Module - STEP 4 Implementation")
    print("=" * 60)
    
    # Demo 1: get_embeddings()
    print("\n" + "=" * 60)
    print("DEMO 1: get_embeddings() - Provider Selection")
    print("=" * 60)
    
    print("""
    Provider Comparison:
    ┌─────────────┬────────┬───────────┬─────────────────────────────────┐
    │ Provider    │ Cost   │ Speed     │ Best For                        │
    ├─────────────┼────────┼───────────┼─────────────────────────────────┤
    │ local       │ FREE   │ ~80ms     │ Dev, privacy, unlimited usage   │
    │ openai      │ $0.02/1M│ ~50ms    │ Production, scale, no GPU       │
    └─────────────┴────────┴───────────┴─────────────────────────────────┘
    
    Local: paraphrase-multilingual-MiniLM-L12-v2 (384 dims)
           - 50+ languages
           - normalize=True for cosine similarity
    
    OpenAI: text-embedding-3-small (512 dims)
            - Fast API
            - Paid but affordable
    """)
    
    # Demo with fake embeddings (no model loading)
    print("\n--- Using Fake Embeddings for Demo ---")
    embeddings = get_embeddings("fake", dimension=384)
    
    texts = [
        "RAG combines retrieval and generation.",
        "Embeddings represent text as dense vectors.",
        "Vector databases enable fast similarity search.",
        "你好世界",  # Chinese
        "Bonjour le monde",  # French
    ]
    
    print(f"\nEmbedding {len(texts)} texts...")
    vectors = embeddings.embed_texts(texts)
    
    print(f"✓ Embedding dimension: {embeddings.dimension}")
    print(f"✓ Number of vectors: {len(vectors)}")
    print(f"✓ Vector shape: {len(vectors[0])}")
    
    # Test similarity
    query = "What is RAG?"
    query_vec = embeddings.embed_query(query)
    
    print(f"\n--- Similarity Test ---")
    print(f"Query: '{query}'")
    print("\nSimilarities:")
    for i, text in enumerate(texts):
        sim = cosine_similarity(query_vec, vectors[i])
        print(f"  [{sim:.3f}] {text[:40]}...")
    
    # Demo 2: build_vectorstore()
    print("\n" + "=" * 60)
    print("DEMO 2: build_vectorstore() - Create Vector Store")
    print("=" * 60)
    
    print("""
    Why ChromaDB?
    - Local persistence (no external database)
    - HNSW indexing for fast ANN search
    - Metadata filtering support
    
    Why Cosine Similarity?
    - Better for text than L2 distance
    - Measures semantic similarity (direction, not magnitude)
    - Less affected by document length
    """)
    
    print("    To build a vector store:")
    print("    >>> from src.chunking import recursive_chunk")
    print("    >>> from src.ingestion import Document")
    print("    >>> ")
    print("    >>> docs = [Document(content='RAG is...', metadata={})]")
    print("    >>> chunks = recursive_chunk(docs)")
    print("    >>> vectorstore = build_vectorstore(chunks, './my_db')")
    print("    >>> ")
    print("    >>> # Search")
    print("    >>> results = vectorstore.similarity_search('What is RAG?', k=3)")
    
    # Demo 3: load_vectorstore()
    print("\n" + "=" * 60)
    print("DEMO 3: load_vectorstore() - Load Existing Store")
    print("=" * 60)
    
    print("""
    Load existing vectorstore to avoid re-indexing:
    
    >>> # First run: build
    >>> vectorstore = build_vectorstore(chunks, "./my_db")
    
    >>> # Later runs: load (fast!)
    >>> vectorstore = load_vectorstore("./my_db")
    
    Important:
    - Use the SAME embedding model for loading and building
    - Different models = different embedding spaces
    - Mixing models = poor retrieval quality
    """)
    
    # Summary
    print("\n" + "=" * 60)
    print("STEP 4 COMPLETE - Key Functions")
    print("=" * 60)
    print("""
    ✅ get_embeddings(provider="local")
       - "local": FREE, 80ms, multilingual
       - "openai": $0.02/1M, 50ms, faster
    
    ✅ build_vectorstore(chunks, persist_dir="./chroma_db")
       - Creates persistent ChromaDB
       - Uses cosine similarity (better for text)
       - Prints count of indexed vectors
    
    ✅ load_vectorstore(persist_dir="./chroma_db")
       - Loads existing vector store
       - Avoids re-indexing
       - Must use same embedding model
    
    ✅ get_or_create_vectorstore(...)  [bonus]
       - Auto-detects: load or build
    """)
