"""
RAG Pipeline Module

Orchestrates the complete RAG workflow:
Document Loading → Chunking → Embedding → Indexing → Retrieval → Re-ranking → Generation

This is the main entry point for the RAG system.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence, Union

from loguru import logger
from tqdm import tqdm

from .config import Settings, get_settings
from .chunking import (
    BaseChunker,
    Chunk,
    RecursiveCharacterChunker,
    SemanticChunker,
    get_chunker,
)
from .embeddings import (
    BaseEmbeddings,
    SentenceTransformerEmbeddings,
    get_embeddings,
)
from .ingestion import (
    BaseLoader,
    Document,
    DirectoryLoader,
    PDFLoader,
    WebPageLoader,
    load_documents,
)
from .llm import (
    BaseLLM,
    FakeLLM,
    Message,
    RAGGenerator,
    RAGPrompts,
    ZhipuAILLM,
    get_llm,
)
from .retrieval import (
    BaseReranker,
    BaseVectorStore,
    ChromaVectorStore,
    FlashRankReranker,
    Retriever,
    SearchResult,
    get_reranker,
    get_vector_store,
)


@dataclass
class RAGResponse:
    """
    Complete RAG response with answer and metadata.
    
    Attributes:
        answer: Generated answer text
        sources: Retrieved source documents
        context: Context used for generation
        query: Original query
        metadata: Additional metadata (timing, etc.)
    """
    answer: str
    sources: list[SearchResult]
    context: str
    query: str
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "answer": self.answer,
            "sources": [s.to_dict() for s in self.sources],
            "context": self.context,
            "query": self.query,
            "metadata": self.metadata,
        }


@dataclass
class IndexingResult:
    """
    Result of document indexing operation.
    
    Attributes:
        documents_loaded: Number of documents loaded
        chunks_created: Number of chunks created
        indexing_time: Time taken for indexing
        errors: List of any errors encountered
    """
    documents_loaded: int
    chunks_created: int
    indexing_time: float
    errors: list[str] = field(default_factory=list)


class RAGPipeline:
    """
    Complete RAG Pipeline that orchestrates all components.
    
    This is the main class for using the RAG system:
    
    ```python
    # Initialize
    pipeline = RAGPipeline()
    
    # Index documents
    pipeline.index_from_directory("./data/raw")
    
    # Query
    response = pipeline.query("What is RAG?")
    print(response.answer)
    ```
    
    Features:
    - Document ingestion from multiple sources
    - Intelligent chunking with overlap
    - Local embeddings (free, private)
    - ChromaDB vector storage
    - FlashRank re-ranking
    - ZhipuAI LLM generation
    - Streaming support
    """
    
    def __init__(
        self,
        settings: Optional[Settings] = None,
        embeddings: Optional[BaseEmbeddings] = None,
        vector_store: Optional[BaseVectorStore] = None,
        chunker: Optional[BaseChunker] = None,
        reranker: Optional[BaseReranker] = None,
        llm: Optional[BaseLLM] = None,
    ):
        """
        Initialize RAG pipeline with optional component overrides.
        
        Args:
            settings: Configuration settings
            embeddings: Custom embedding model
            vector_store: Custom vector store
            chunker: Custom chunker
            reranker: Custom reranker
            llm: Custom LLM
        """
        self.settings = settings or get_settings()
        
        # Initialize components with defaults
        self.embeddings = embeddings or SentenceTransformerEmbeddings(
            model_name=self.settings.embedding_model,
        )
        
        self.vector_store = vector_store or ChromaVectorStore(
            collection_name=self.settings.collection_name,
            persist_directory=self.settings.persist_directory,
            embedding_function=self.embeddings,
        )
        
        self.chunker = chunker or RecursiveCharacterChunker(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        
        self.reranker = reranker or FlashRankReranker(
            model_name=self.settings.rerank_model,
        )
        
        self.llm = llm
        
        self._retriever = None
        self._generator = None
    
    @property
    def retriever(self) -> Retriever:
        """Get or create the retriever."""
        if self._retriever is None:
            self._retriever = Retriever(
                vector_store=self.vector_store,
                embedding_function=self.embeddings,
                reranker=self.reranker,
                top_k=self.settings.top_k,
                rerank_top_k=self.settings.rerank_top_k,
            )
        return self._retriever
    
    @property
    def generator(self) -> RAGGenerator:
        """Get or create the generator."""
        if self._generator is None:
            self._generator = RAGGenerator(
                llm=self._get_llm(),
                max_context_tokens=3000,
            )
        return self._generator
    
    def _get_llm(self) -> BaseLLM:
        """Get or create the LLM."""
        if self.llm is None:
            # Try to use ZhipuAI, fall back to fake if not configured
            try:
                self.llm = ZhipuAILLM(model=self.settings.llm_model)
            except ValueError:
                logger.warning("ZhipuAI not configured, using fake LLM for testing")
                self.llm = FakeLLM()
        return self.llm
    
    # ==========================================
    # Indexing Methods
    # ==========================================
    
    def index_documents(
        self,
        documents: Sequence[Document],
        show_progress: bool = True,
    ) -> IndexingResult:
        """
        Index documents into the vector store.
        
        Args:
            documents: List of Document objects
            show_progress: Show progress bar
        
        Returns:
            IndexingResult with statistics
        """
        start_time = time.time()
        errors = []
        
        logger.info(f"Indexing {len(documents)} documents")
        
        # Chunk documents
        chunks = self.chunker.chunk(documents)
        
        if not chunks:
            return IndexingResult(
                documents_loaded=len(documents),
                chunks_created=0,
                indexing_time=time.time() - start_time,
                errors=["No chunks created from documents"],
            )
        
        # Index into vector store
        try:
            self.vector_store.add(chunks)
        except Exception as e:
            errors.append(str(e))
            logger.error(f"Indexing error: {e}")
        
        result = IndexingResult(
            documents_loaded=len(documents),
            chunks_created=len(chunks),
            indexing_time=time.time() - start_time,
            errors=errors,
        )
        
        logger.info(
            f"Indexing complete: {result.documents_loaded} docs, "
            f"{result.chunks_created} chunks in {result.indexing_time:.2f}s"
        )
        
        return result
    
    def index_from_directory(
        self,
        directory: Union[str, Path],
        glob_pattern: str = "**/*",
        recursive: bool = True,
    ) -> IndexingResult:
        """
        Index all documents from a directory.
        
        Args:
            directory: Directory path
            glob_pattern: File pattern to match
            recursive: Search recursively
        
        Returns:
            IndexingResult with statistics
        """
        directory = Path(directory)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        loader = DirectoryLoader(
            directory=directory,
            glob_pattern=glob_pattern,
            recursive=recursive,
        )
        
        documents = loader.load()
        
        return self.index_documents(documents)
    
    def index_from_file(
        self,
        file_path: Union[str, Path],
        **kwargs,
    ) -> IndexingResult:
        """
        Index a single file.
        
        Args:
            file_path: Path to file
            **kwargs: Additional arguments for the loader
        
        Returns:
            IndexingResult with statistics
        """
        documents = load_documents(file_path, **kwargs)
        return self.index_documents(documents)
    
    def index_from_url(
        self,
        url: str,
        **kwargs,
    ) -> IndexingResult:
        """
        Index content from a URL.
        
        Args:
            url: URL to fetch
            **kwargs: Additional arguments for WebPageLoader
        
        Returns:
            IndexingResult with statistics
        """
        loader = WebPageLoader(url=url, **kwargs)
        documents = loader.load()
        return self.index_documents(documents)
    
    def index_from_text(
        self,
        text: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> IndexingResult:
        """
        Index raw text content.
        
        Args:
            text: Text content to index
            metadata: Optional metadata
        
        Returns:
            IndexingResult with statistics
        """
        doc = Document(
            content=text,
            metadata=metadata or {"source": "text_input"},
        )
        return self.index_documents([doc])
    
    # ==========================================
    # Query Methods
    # ==========================================
    
    def query(
        self,
        query: str,
        top_k: Optional[int] = None,
        rerank_top_k: Optional[int] = None,
        filter: Optional[dict[str, Any]] = None,
        stream: bool = False,
    ) -> RAGResponse:
        """
        Query the RAG system.
        
        Args:
            query: User question
            top_k: Number of initial retrieval results
            rerank_top_k: Number of results after re-ranking
            filter: Metadata filter
            stream: Whether to stream the response
        
        Returns:
            RAGResponse with answer and sources
        """
        start_time = time.time()
        
        logger.info(f"Query: {query[:50]}...")
        
        # Retrieve context
        context, sources = self.retriever.get_context(
            query=query,
            top_k=top_k or self.settings.top_k,
            rerank_top_k=rerank_top_k or self.settings.rerank_top_k,
            filter=filter,
        )
        
        retrieval_time = time.time() - start_time
        
        # Generate answer
        generation_start = time.time()
        
        if stream:
            # Return iterator for streaming
            answer_iter = self.generator.generate_answer(
                query=query,
                context=context,
                sources=sources,
                stream=True,
            )
            return answer_iter
        else:
            answer = self.generator.generate_answer(
                query=query,
                context=context,
                sources=sources,
                stream=False,
            )
        
        generation_time = time.time() - generation_start
        total_time = time.time() - start_time
        
        return RAGResponse(
            answer=answer,
            sources=sources,
            context=context,
            query=query,
            metadata={
                "retrieval_time": retrieval_time,
                "generation_time": generation_time,
                "total_time": total_time,
                "num_sources": len(sources),
                "context_length": len(context),
            },
        )
    
    def stream_query(
        self,
        query: str,
        top_k: Optional[int] = None,
        rerank_top_k: Optional[int] = None,
        filter: Optional[dict[str, Any]] = None,
    ) -> Iterator[str]:
        """
        Query with streaming response.
        
        Yields answer tokens as they're generated.
        """
        response = self.query(
            query=query,
            top_k=top_k,
            rerank_top_k=rerank_top_k,
            filter=filter,
            stream=True,
        )
        
        for token in response:
            yield token
    
    # ==========================================
    # Utility Methods
    # ==========================================
    
    def count_documents(self) -> int:
        """Get the number of indexed documents."""
        return self.vector_store.count()
    
    def clear_index(self) -> None:
        """Clear all indexed documents."""
        self.vector_store.delete()
        logger.info("Index cleared")
    
    def get_stats(self) -> dict[str, Any]:
        """Get pipeline statistics."""
        return {
            "documents_indexed": self.count_documents(),
            "collection_name": self.settings.collection_name,
            "embedding_model": self.settings.embedding_model,
            "llm_model": self.settings.llm_model,
            "chunk_size": self.settings.chunk_size,
            "chunk_overlap": self.settings.chunk_overlap,
            "top_k": self.settings.top_k,
            "rerank_top_k": self.settings.rerank_top_k,
        }


class RAGPipelineBuilder:
    """
    Builder pattern for creating RAG pipelines with custom components.
    
    ```python
    pipeline = (
        RAGPipelineBuilder()
        .with_embeddings("sentence_transformer", model_name="BAAI/bge-m3")
        .with_chunker("recursive", chunk_size=512)
        .with_llm("zhipuai", model="glm-4-flash")
        .build()
    )
    ```
    """
    
    def __init__(self):
        self._settings = None
        self._embeddings = None
        self._vector_store = None
        self._chunker = None
        self._reranker = None
        self._llm = None
    
    def with_settings(self, settings: Settings) -> "RAGPipelineBuilder":
        """Use custom settings."""
        self._settings = settings
        return self
    
    def with_embeddings(
        self,
        provider: str = "sentence_transformer",
        **kwargs,
    ) -> "RAGPipelineBuilder":
        """Set embedding provider."""
        self._embeddings = get_embeddings(provider, **kwargs)
        return self
    
    def with_vector_store(
        self,
        provider: str = "chroma",
        **kwargs,
    ) -> "RAGPipelineBuilder":
        """Set vector store provider."""
        self._vector_store = get_vector_store(provider, **kwargs)
        return self
    
    def with_chunker(
        self,
        strategy: str = "recursive",
        **kwargs,
    ) -> "RAGPipelineBuilder":
        """Set chunking strategy."""
        self._chunker = get_chunker(strategy, **kwargs)
        return self
    
    def with_reranker(
        self,
        provider: str = "flashrank",
        **kwargs,
    ) -> "RAGPipelineBuilder":
        """Set reranker provider."""
        self._reranker = get_reranker(provider, **kwargs)
        return self
    
    def with_llm(
        self,
        provider: str = "zhipuai",
        **kwargs,
    ) -> "RAGPipelineBuilder":
        """Set LLM provider."""
        self._llm = get_llm(provider, **kwargs)
        return self
    
    def build(self) -> RAGPipeline:
        """Build the RAG pipeline."""
        return RAGPipeline(
            settings=self._settings,
            embeddings=self._embeddings,
            vector_store=self._vector_store,
            chunker=self._chunker,
            reranker=self._reranker,
            llm=self._llm,
        )


# Convenience function
def create_pipeline(**kwargs) -> RAGPipeline:
    """
    Create a RAG pipeline with optional configuration.
    
    Args:
        **kwargs: Configuration overrides
    
    Returns:
        RAGPipeline instance
    """
    return RAGPipeline(**kwargs)


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("RAG Pipeline Demo")
    print("=" * 60)
    
    from .embeddings import FakeEmbeddings
    from .llm import FakeLLM
    
    # Create pipeline with fake components for testing
    pipeline = RAGPipeline(
        embeddings=FakeEmbeddings(dimension=384),
        llm=FakeLLM(responses=[
            "RAG是一种结合检索和生成的AI技术，可以提高回答的准确性。",
        ]),
    )
    
    # Index some sample text
    print("\n--- Indexing Sample Documents ---")
    
    sample_docs = [
        Document(
            content="RAG（Retrieval-Augmented Generation）是一种将检索和生成结合的AI技术。"
                   "它首先从知识库中检索相关文档，然后将这些文档作为上下文输入到语言模型中生成回答。",
            metadata={"source": "rag_intro.txt", "topic": "RAG"},
        ),
        Document(
            content="向量数据库是RAG系统的核心组件。它存储文本的嵌入向量，"
                   "并支持高效的相似度搜索。常用的向量数据库包括Chroma、Pinecone和Milvus。",
            metadata={"source": "vector_db.txt", "topic": "Vector Database"},
        ),
        Document(
            content="Re-ranking是提高检索质量的关键技术。它使用更精确的模型对初步检索结果重新排序，"
                   "确保最相关的文档排在前面。FlashRank是一个高效的本地re-ranking工具。",
            metadata={"source": "reranking.txt", "topic": "Re-ranking"},
        ),
    ]
    
    result = pipeline.index_documents(sample_docs)
    print(f"Loaded: {result.documents_loaded} documents")
    print(f"Created: {result.chunks_created} chunks")
    print(f"Time: {result.indexing_time:.2f}s")
    
    # Query the pipeline
    print("\n--- Querying ---")
    
    response = pipeline.query("什么是RAG？")
    
    print(f"\nQuestion: {response.query}")
    print(f"\nAnswer: {response.answer}")
    print(f"\nSources: {len(response.sources)}")
    print(f"Time: {response.metadata['total_time']:.2f}s")
    
    # Print pipeline stats
    print("\n--- Pipeline Stats ---")
    stats = pipeline.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Clean up
    pipeline.clear_index()
    print("\n--- Demo Complete ---")
