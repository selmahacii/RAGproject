"""
Configuration management using Pydantic Settings.
Loads environment variables and provides type-safe configuration.
"""

from pathlib import Path
from typing import Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    Uses pydantic-settings for type validation and .env file support.
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # ===========================================
    # API Keys
    # ===========================================
    zhipuai_api_key: str = Field(
        default="",
        description="ZhipuAI API key for GLM models",
    )
    
    # ===========================================
    # Model Configuration
    # ===========================================
    llm_model: str = Field(
        default="glm-4-flash",
        description="LLM model to use for generation",
    )
    embedding_model: str = Field(
        default="BAAI/bge-m3",
        description="Sentence transformer model for embeddings",
    )
    rerank_model: str = Field(
        default="ms-marco-MiniLM-L-12-v2",
        description="FlashRank model for re-ranking",
    )
    
    # ===========================================
    # Chunking Configuration
    # ===========================================
    chunk_size: int = Field(
        default=512,
        ge=100,
        le=4096,
        description="Size of text chunks in characters",
    )
    chunk_overlap: int = Field(
        default=64,
        ge=0,
        description="Overlap between consecutive chunks",
    )
    
    @field_validator("chunk_overlap")
    @classmethod
    def validate_chunk_overlap(cls, v: int, info) -> int:
        """Ensure overlap is less than chunk size."""
        chunk_size = info.data.get("chunk_size", 512)
        if v >= chunk_size:
            raise ValueError(f"chunk_overlap ({v}) must be less than chunk_size ({chunk_size})")
        return v
    
    # ===========================================
    # Retrieval Configuration
    # ===========================================
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of documents to retrieve initially",
    )
    rerank_top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of documents after re-ranking",
    )
    
    @field_validator("rerank_top_k")
    @classmethod
    def validate_rerank_top_k(cls, v: int, info) -> int:
        """Ensure rerank_top_k is not greater than top_k."""
        top_k = info.data.get("top_k", 5)
        if v > top_k:
            raise ValueError(f"rerank_top_k ({v}) cannot be greater than top_k ({top_k})")
        return v
    
    # ===========================================
    # Vector Store Configuration
    # ===========================================
    collection_name: str = Field(
        default="rag_demo",
        description="ChromaDB collection name",
    )
    persist_directory: Path = Field(
        default=Path("./data/chroma_db"),
        description="Directory to persist ChromaDB data",
    )
    
    # ===========================================
    # Evaluation Configuration
    # ===========================================
    eval_sample_size: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of samples for RAGAs evaluation",
    )
    
    # ===========================================
    # API Configuration
    # ===========================================
    api_host: str = Field(
        default="0.0.0.0",
        description="FastAPI server host",
    )
    api_port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="FastAPI server port",
    )
    
    # ===========================================
    # Streamlit Configuration
    # ===========================================
    streamlit_port: int = Field(
        default=8501,
        ge=1,
        le=65535,
        description="Streamlit server port",
    )
    
    # ===========================================
    # Paths
    # ===========================================
    @property
    def project_root(self) -> Path:
        """Get the project root directory."""
        return Path(__file__).parent.parent
    
    @property
    def data_dir(self) -> Path:
        """Get the data directory."""
        return self.project_root / "data"
    
    @property
    def raw_data_dir(self) -> Path:
        """Get the raw data directory."""
        return self.data_dir / "raw"
    
    @property
    def processed_data_dir(self) -> Path:
        """Get the processed data directory."""
        return self.data_dir / "processed"
    
    def ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        directories = [
            self.raw_data_dir,
            self.processed_data_dir,
            self.persist_directory,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_api_key(self) -> str:
        """Get ZhipuAI API key, raising error if not set."""
        if not self.zhipuai_api_key or self.zhipuai_api_key == "your_zhipuai_api_key_here":
            raise ValueError(
                "ZhipuAI API key not configured. "
                "Set ZHIPUAI_API_KEY in .env file or environment variable."
            )
        return self.zhipuai_api_key


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings


if __name__ == "__main__":
    # Print current configuration for debugging
    print("=" * 50)
    print("RAG Project Configuration")
    print("=" * 50)
    s = get_settings()
    print(f"LLM Model: {s.llm_model}")
    print(f"Embedding Model: {s.embedding_model}")
    print(f"Rerank Model: {s.rerank_model}")
    print(f"Chunk Size: {s.chunk_size}")
    print(f"Chunk Overlap: {s.chunk_overlap}")
    print(f"Top K: {s.top_k}")
    print(f"Rerank Top K: {s.rerank_top_k}")
    print(f"Collection Name: {s.collection_name}")
    print(f"Persist Directory: {s.persist_directory}")
    print(f"API Key Configured: {'Yes' if s.zhipuai_api_key else 'No'}")
    print("=" * 50)
