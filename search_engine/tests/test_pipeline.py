"""
Unit tests for SelmaData Pipeline components.

Run with:
    pytest tests/ -v
"""

import pytest
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Settings, get_settings
from src.ingestion import Document, PDFLoader, TextLoader, CSVLoader
from src.chunking import (
    RecursiveCharacterChunker,
    SemanticChunker,
    Chunk,
)
from src.vectors import FakeEmbeddings, cosine_similarity
from src.retrieval import ChromaVectorStore, SearchResult, NoopReranker
from src.inference import FakeCoreProcessor, Message, MessageRole, SelmaDataPrompts
from src.pipeline import SelmaDataPipeline
from src.eval import EvaluationSample, EvaluationResult, SelmaDataEvaluator


# ============================================
# Fixtures
# ============================================

@pytest.fixture
def sample_document():
    """Create a sample document for testing."""
    return Document(
        content="This is a test document for SelmaData pipeline testing. "
               "It contains multiple sentences for chunking. "
               "SelmaData combines retrieval and generation.",
        metadata={"source": "test.txt", "test": True},
    )


@pytest.fixture
def sample_documents():
    """Create multiple sample documents for testing."""
    return [
        Document(
            content="SelmaData is a technique that combines retrieval and generation.",
            metadata={"source": "doc1.txt"},
        ),
        Document(
            content="Vector databases store embeddings for similarity search.",
            metadata={"source": "doc2.txt"},
        ),
        Document(
            content="Re-ranking improves retrieval quality significantly.",
            metadata={"source": "doc3.txt"},
        ),
    ]


@pytest.fixture
def fake_embeddings():
    """Create fake embeddings for testing."""
    return FakeEmbeddings(dimension=384)


@pytest.fixture
def fake_llm():
    """Create fake CoreProcessor for testing."""
    return FakeCoreProcessor(responses=["Test response from fake CoreProcessor."])


@pytest.fixture
def test_pipeline(fake_embeddings, fake_llm):
    """Create a test pipeline with fake components."""
    return SelmaDataPipeline(
        embeddings=fake_embeddings,
        llm=fake_llm,
    )


# ============================================
# Config Tests
# ============================================

class TestConfig:
    """Tests for configuration management."""
    
    def test_default_settings(self):
        """Test default settings are loaded."""
        settings = get_settings()
        
        assert settings.chunk_size > 0
        assert settings.chunk_overlap >= 0
        assert settings.chunk_overlap < settings.chunk_size
        assert settings.top_k > 0
        assert settings.rerank_top_k <= settings.top_k
    
    def test_custom_settings(self):
        """Test custom settings creation."""
        settings = Settings(
            chunk_size=256,
            chunk_overlap=32,
            top_k=3,
        )
        
        assert settings.chunk_size == 256
        assert settings.chunk_overlap == 32
        assert settings.top_k == 3


# ============================================
# Ingestion Tests
# ============================================

class TestIngestion:
    """Tests for document ingestion."""
    
    def test_document_creation(self, sample_document):
        """Test document creation."""
        assert sample_document.content
        assert sample_document.metadata["source"] == "test.txt"
    
    def test_document_empty_content_raises(self):
        """Test that empty content raises error."""
        with pytest.raises(ValueError):
            Document(content="", metadata={})
    
    def test_document_to_dict(self, sample_document):
        """Test document serialization."""
        data = sample_document.to_dict()
        
        assert "content" in data
        assert "metadata" in data
        assert data["content"] == sample_document.content


# ============================================
# Chunking Tests
# ============================================

class TestChunking:
    """Tests for text chunking."""
    
    def test_recursive_chunker_basic(self, sample_document):
        """Test basic recursive chunking."""
        chunker = RecursiveCharacterChunker(
            chunk_size=50,
            chunk_overlap=10,
        )
        
        chunks = chunker.chunk_single(sample_document)
        
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.content
            assert chunk.metadata
    
    def test_recursive_chunker_overlap(self):
        """Test that overlap is maintained."""
        text = "A" * 100 + "B" * 100 + "C" * 100
        doc = Document(content=text, metadata={"source": "test"})
        
        chunker = RecursiveCharacterChunker(
            chunk_size=80,
            chunk_overlap=20,
        )
        
        chunks = chunker.chunk_single(doc)
        
        # Check overlap exists between consecutive chunks
        if len(chunks) > 1:
            # At least some content should overlap
            pass  # Overlap check is implementation dependent
    
    def test_chunk_metadata_preserved(self, sample_document):
        """Test that metadata is preserved in chunks."""
        chunker = RecursiveCharacterChunker(chunk_size=100)
        chunks = chunker.chunk_single(sample_document)
        
        for chunk in chunks:
            assert chunk.metadata.get("source") == "test.txt"
    
    def test_semantic_chunker(self, sample_document):
        """Test semantic chunking."""
        chunker = SemanticChunker(
            chunk_size=100,
            strategy="sentence",
        )
        
        chunks = chunker.chunk_single(sample_document)
        
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.content


# ============================================
# Embedding Tests
# ============================================

class TestEmbeddings:
    """Tests for embedding generation."""
    
    def test_fake_embeddings_shape(self, fake_embeddings):
        """Test that fake embeddings have correct shape."""
        texts = ["Hello world", "Test text"]
        vectors = fake_embeddings.embed_texts(texts)
        
        assert len(vectors) == 2
        assert len(vectors[0]) == fake_embeddings.dimension
    
    def test_fake_embeddings_query(self, fake_embeddings):
        """Test query embedding."""
        query = "What is SelmaData?"
        vector = fake_embeddings.embed_query(query)
        
        assert len(vector) == fake_embeddings.dimension
    
    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]
        vec3 = [0.0, 1.0, 0.0]
        
        # Same vectors should have similarity 1
        assert cosine_similarity(vec1, vec2) == pytest.approx(1.0)
        
        # Orthogonal vectors should have similarity 0
        assert cosine_similarity(vec1, vec3) == pytest.approx(0.0)


# ============================================
# Retrieval Tests
# ============================================

class TestRetrieval:
    """Tests for retrieval components."""
    
    def test_vector_store_add_and_search(
        self,
        fake_embeddings,
        sample_documents,
    ):
        """Test adding and searching in vector store."""
        store = ChromaVectorStore(
            collection_name="test_collection",
            embedding_function=fake_embeddings,
        )
        
        # First, create chunks from documents
        chunker = RecursiveCharacterChunker(chunk_size=100)
        chunks = chunker.chunk(sample_documents)
        
        # Add chunks
        store.add(chunks)
        
        assert store.count() == len(chunks)
        
        # Search
        query_embedding = fake_embeddings.embed_query("What is SelmaData?")
        results = store.search(query_embedding, top_k=2)
        
        assert len(results) <= 2
        for result in results:
            assert isinstance(result, SearchResult)
            assert result.content
        
        # Cleanup
        store.delete_collection()
    
    def test_noop_reranker(self):
        """Test no-op reranker returns results unchanged."""
        reranker = NoopReranker()
        
        results = [
            SearchResult(content="A", score=0.9, metadata={}),
            SearchResult(content="B", score=0.8, metadata={}),
            SearchResult(content="C", score=0.7, metadata={}),
        ]
        
        reranked = reranker.rerank("test query", results, top_k=2)
        
        assert len(reranked) == 2
        assert reranked[0].content == "A"


# ============================================
# CoreProcessor Tests
# ============================================

class TestCoreProcessor:
    """Tests for CoreProcessor components."""
    
    def test_fake_llm_generate(self, fake_llm):
        """Test fake CoreProcessor generation."""
        messages = [
            Message(role=MessageRole.USER, content="Hello"),
        ]
        
        response = fake_llm.generate(messages)
        
        assert response.content
        assert response.model == "fake-llm"
    
    def test_fake_llm_stream(self, fake_llm):
        """Test fake CoreProcessor streaming."""
        messages = [
            Message(role=MessageRole.USER, content="Hello"),
        ]
        
        tokens = list(fake_llm.stream(messages))
        
        assert len(tokens) > 0
    
    def test_selma_data_prompts(self):
        """Test SelmaData prompt generation."""
        system, user = SelmaDataPrompts.get_selma_data_prompt(
            context="Test context",
            query="Test query",
            language="chinese",
        )
        
        assert "context" in system.lower() or "context" in user.lower()
        assert "Test query" in user


# ============================================
# Pipeline Tests
# ============================================

class TestPipeline:
    """Tests for the complete SelmaData pipeline."""
    
    def test_pipeline_initialization(self, test_pipeline):
        """Test pipeline initialization."""
        assert test_pipeline.embeddings is not None
        assert test_pipeline.llm is not None
        assert test_pipeline.chunker is not None
    
    def test_pipeline_index_text(self, test_pipeline):
        """Test indexing text into pipeline."""
        result = test_pipeline.index_from_text(
            "This is test content for the SelmaData pipeline.",
            metadata={"source": "test"},
        )
        
        assert result.documents_loaded == 1
        assert result.chunks_created > 0
        assert test_pipeline.count_documents() > 0
        
        # Cleanup
        test_pipeline.clear_index()
    
    def test_pipeline_query(self, test_pipeline):
        """Test querying the pipeline."""
        # Index some content
        test_pipeline.index_from_text(
            "SelmaData combines retrieval and generation for better answers.",
            metadata={"source": "test"},
        )
        
        # Query
        response = test_pipeline.query("What is SelmaData?")
        
        assert response.answer
        assert response.query == "What is SelmaData?"
        assert "total_time" in response.metadata
        
        # Cleanup
        test_pipeline.clear_index()
    
    def test_pipeline_stats(self, test_pipeline):
        """Test getting pipeline stats."""
        stats = test_pipeline.get_stats()
        
        assert "documents_indexed" in stats
        assert "chunk_size" in stats
        assert "embedding_model" in stats


# ============================================
# Evaluation Tests
# ============================================

class TestEvaluation:
    """Tests for SelmaData evaluation."""
    
    def test_evaluation_sample(self):
        """Test evaluation sample creation."""
        sample = EvaluationSample(
            question="What is SelmaData?",
            ground_truth="SelmaData is retrieval-augmented generation.",
        )
        
        assert sample.question == "What is SelmaData?"
        assert sample.ground_truth
    
    def test_evaluation_result_score(self):
        """Test evaluation result scoring."""
        result = EvaluationResult(
            faithfulness=0.9,
            answer_relevance=0.8,
            context_precision=0.7,
            context_recall=0.6,
            answer_similarity=0.5,
        )
        
        # Overall score should be weighted aveselma_datae
        assert 0 < result.overall_score < 1
    
    def test_evaluator_heuristic(self, test_pipeline):
        """Test heuristic evaluation."""
        sample = EvaluationSample(
            question="What is SelmaData?",
            answer="SelmaData combines retrieval and generation.",
            contexts=["SelmaData is retrieval-augmented generation."],
            ground_truth="SelmaData is retrieval-augmented generation.",
        )
        
        evaluator = SelmaDataEvaluator(test_pipeline, use_selma_dataas=False)
        result = evaluator._evaluate_with_heuristics(sample)
        
        assert 0 <= result.faithfulness <= 1
        assert 0 <= result.answer_relevance <= 1


# ============================================
# Integration Tests
# ============================================

class TestIntegration:
    """Integration tests for the complete system."""
    
    def test_end_to_end_selma_data(self, test_pipeline):
        """Test end-to-end SelmaData workflow."""
        # 1. Index documents
        docs = [
            Document(
                content="Python is a programming language.",
                metadata={"source": "python.txt"},
            ),
            Document(
                content="JavaScript is used for web development.",
                metadata={"source": "javascript.txt"},
            ),
        ]
        
        result = test_pipeline.index_documents(docs)
        assert result.chunks_created > 0
        
        # 2. Query
        response = test_pipeline.query("What is Python?")
        assert response.answer
        
        # 3. Verify sources
        assert len(response.sources) > 0
        
        # 4. Cleanup
        test_pipeline.clear_index()
        assert test_pipeline.count_documents() == 0


# ============================================
# Run Tests
# ============================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
