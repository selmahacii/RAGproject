#!/usr/bin/env python3
"""
SelmaData Pipeline Entry Point

A simple function to run the complete SelmaData pipeline:
    1. Load PDFs from folder
    2. Chunk with recursive strategy
    3. Build vectorstore
    4. Retrieve with re-ranking (k=10, rerank_top=3)
    5. Generate answer with Provider
    6. Return result dict

Usage:
    from main import run_selma_data_pipeline
    
    result = run_selma_data_pipeline("What is SelmaData?", data_folder="./data/raw")
    print(result["answer"])
    print(result["sources"])
"""

import sys
import time
from pathlib import Path
from typing import Any, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger
from src.config import get_settings

# Configure structured logging
logger.remove()  # Remove default handler
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
)
logger.add(
    "logs/selma_data_pipeline_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="7 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
    level="DEBUG",
)


def run_selma_data_pipeline(
    question: str,
    data_folder: str = "./data/raw",
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    k: int = 10,
    rerank_top: int = 3,
    model: str = "glm-4-flash",
    persist_dir: str = "./data/chroma_db",
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Run the complete SelmaData pipeline with a single function call.
    
    This function:
    1. Loads PDFs from folder
    2. Chunks with recursive strategy
    3. Builds vectorstore
    4. Retrieves with re-ranking (k=10, rerank_top=3)
    5. Generates answer with Provider
    6. Returns result dict
    
    Args:
        question: The question to answer
        data_folder: Folder containing PDF documents (default: "./data/raw")
        chunk_size: Size of each chunk in characters (default: 512)
        chunk_overlap: Overlap between chunks (default: 64)
        k: Number of candidates to retrieve (default: 10)
        rerank_top: Number of top results after re-ranking (default: 3)
        model: Provider model to use (default: "glm-4-flash")
        persist_dir: Directory to persist vector store (default: "./data/chroma_db")
        verbose: Print progress information (default: True)
    
    Returns:
        Dictionary containing:
        - answer: Generated answer text
        - sources: List of source documents used
        - tokens_used: Token usage statistics
        - context: Full context used for generation
        - timing: Timing breakdown
        - chunks_indexed: Number of chunks in the index
    
    Example:
        >>> result = run_selma_data_pipeline("What is SelmaData?", data_folder="./docs")
        >>> print(result["answer"])
        >>> print(f"Used {result['tokens_used']['total_tokens']} tokens")
        >>> for source in result["sources"]:
        ...     print(f"Source: {source['source']}")
    
    Example output:
        ═══════════════════════════════════════════════════════════
                         SelmaData PIPELINE RESULT
        ═══════════════════════════════════════════════════════════
        
        Question: What is SelmaData?
        
        Answer:
        SelmaData (Retrieval-Augmented Generation) is a technique that combines
        retrieval and generation. It retrieves relevant documents from a
        knowledge base and uses them as context for the CoreProcessor to generate
        accurate answers.
        
        ─────────────────────────────────────────────────────────────
        Sources (3):
          [1] selma_data_intro.pdf (Score: 0.892)
          [2] vector_databases.pdf (Score: 0.845)
          [3] reranking.pdf (Score: 0.823)
        
        Tokens: 450 prompt + 180 completion = 630 total
        
        Timing:
          - Ingestion:    1.2s
          - Retrieval:    0.3s
          - Generation:   2.1s
          - Total:        3.6s
        ═══════════════════════════════════════════════════════════
    """
    start_time = time.time()
    timing = {}
    
    # =========================================================
    # STEP 1: Load PDFs from folder
    # =========================================================
    step_start = time.time()
    
    if verbose:
        print("\n" + "=" * 60)
        print("🚀 SelmaData PIPELINE")
        print("=" * 60)
        print(f"\n📥 Step 1: Loading PDFs from {data_folder}")
    
    data_path = Path(data_folder)
    if not data_path.exists():
        raise FileNotFoundError(f"Data folder not found: {data_folder}")
    
    # Import ingestion module
    from src.ingestion import load_pdfs
    
    # Load PDFs
    documents = load_pdfs(data_path)
    
    if not documents:
        raise ValueError(f"No documents found in {data_folder}")
    
    timing["ingestion"] = time.time() - step_start
    
    if verbose:
        print(f"   ✅ Loaded {len(documents)} documents")
    
    # =========================================================
    # STEP 2: Chunk with recursive strategy
    # =========================================================
    step_start = time.time()
    
    if verbose:
        print(f"\n✂️  Step 2: Chunking (size={chunk_size}, overlap={chunk_overlap})")
    
    from src.chunking import recursive_chunk
    
    chunks = recursive_chunk(documents, chunk_size=chunk_size, overlap=chunk_overlap)
    
    timing["chunking"] = time.time() - step_start
    
    if verbose:
        print(f"   ✅ Created {len(chunks)} chunks")
    
    # =========================================================
    # STEP 3: Build vectorstore
    # =========================================================
    step_start = time.time()
    
    if verbose:
        print(f"\n🧠 Step 3: Building vectorstore")
    
    from src.vectors import get_embeddings, build_vectorstore, get_or_create_vectorstore
    
    # Get embeddings (local BGE-M3 by default)
    embeddings = get_embeddings(provider="local")
    
    # Build or load vectorstore
    vectorstore = get_or_create_vectorstore(
        chunks=chunks,
        persist_dir=persist_dir,
        embeddings=embeddings,
    )
    
    timing["indexing"] = time.time() - step_start
    
    if verbose:
        print(f"   ✅ Vectorstore ready ({vectorstore._collection.count()} vectors)")
    
    # =========================================================
    # STEP 4: Retrieve with re-ranking
    # =========================================================
    step_start = time.time()
    
    if verbose:
        print(f"\n🔍 Step 4: Retrieving (k={k}, rerank_top={rerank_top})")
    
    from src.retrieval import retrieve_and_rerank
    
    # Retrieve and re-rank
    results = retrieve_and_rerank(
        vectorstore,
        question,
        k=k,
        rerank_top=rerank_top,
    )
    
    timing["retrieval"] = time.time() - step_start
    
    if verbose:
        print(f"   ✅ Retrieved {len(results)} relevant chunks")
    
    # =========================================================
    # STEP 5: Generate answer with Provider
    # =========================================================
    step_start = time.time()
    
    if verbose:
        print(f"\n🤖 Step 5: Generating answer with {model}")
    
    from src.inference import generate_answer
    
    # Convert results to chunk format for generate_answer
    context_chunks = [
        {
            "content": r["content"],
            "metadata": r["metadata"],
        }
        for r in results
    ]
    
    # Generate answer
    gen_result = generate_answer(question, context_chunks, model=model)
    
    timing["generation"] = time.time() - step_start
    timing["total"] = time.time() - start_time
    
    if verbose:
        print(f"   ✅ Generated answer ({gen_result['tokens_used']['total_tokens']} tokens)")
    
    # =========================================================
    # STEP 6: Build result and print summary
    # =========================================================
    
    result = {
        "answer": gen_result["answer"],
        "sources": gen_result["sources"],
        "tokens_used": gen_result["tokens_used"],
        "context_chunks": results,
        "timing": timing,
        "chunks_indexed": len(chunks),
        "model": gen_result["model"],
        "question": question,
    }
    
    if verbose:
        print_result_summary(result)
    
    return result


def print_result_summary(result: dict[str, Any]) -> None:
    """Print a formatted summary of the SelmaData result."""
    
    print("\n" + "=" * 60)
    print("📊 SelmaData PIPELINE RESULT")
    print("=" * 60)
    
    print(f"\n❓ Question: {result['question']}")
    
    print(f"\n📝 Answer:")
    print("-" * 40)
    print(result["answer"])
    
    # Sources
    print(f"\n📚 Sources ({len(result['sources'])}):")
    for i, source in enumerate(result["sources"], 1):
        score = source.get("score", 0)
        source_name = source.get("source", "unknown")
        page = source.get("page")
        page_str = f" (p.{page})" if page else ""
        print(f"   [{i}] {source_name}{page_str} - Score: {score:.3f}")
    
    # Tokens
    tokens = result["tokens_used"]
    print(f"\n📊 Tokens:")
    print(f"   Prompt: {tokens['prompt_tokens']}")
    print(f"   Completion: {tokens['completion_tokens']}")
    print(f"   Total: {tokens['total_tokens']}")
    
    # Timing
    timing = result["timing"]
    print(f"\n⏱️  Timing:")
    print(f"   Ingestion:  {timing.get('ingestion', 0):.2f}s")
    print(f"   Chunking:   {timing.get('chunking', 0):.2f}s")
    print(f"   Indexing:   {timing.get('indexing', 0):.2f}s")
    print(f"   Retrieval:  {timing.get('retrieval', 0):.2f}s")
    print(f"   Generation: {timing.get('generation', 0):.2f}s")
    print(f"   ─────────────────────")
    print(f"   Total:      {timing.get('total', 0):.2f}s")
    
    print("\n" + "=" * 60)


def interactive_mode():
    """Run SelmaData pipeline in interactive Q&A mode."""
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + " " * 15 + "🔍 SelmaData Pipeline Interactive Mode" + " " * 12 + "║")
    print("╚" + "═" * 58 + "╝")
    
    print("""
    This mode allows you to ask multiple questions.
    
    Usage:
    - Type your question and press Enter
    - Type 'quit' or 'exit' to stop
    - Type 'clear' to clear the index
    
    First, let's index your documents...
    """)
    
    data_folder = input("📁 Enter data folder path [./data/raw]: ").strip() or "./data/raw"
    
    # Initial indexing
    print(f"\n📂 Indexing documents from {data_folder}...")
    
    try:
        from src.ingestion import load_pdfs
        from src.chunking import recursive_chunk
        from src.vectors import get_embeddings, build_vectorstore
        from src.retrieval import retrieve_and_rerank
        from src.inference import generate_answer
        
        # Load and index
        documents = load_pdfs(Path(data_folder))
        if not documents:
            print(f"❌ No documents found in {data_folder}")
            return
        
        chunks = recursive_chunk(documents)
        embeddings = get_embeddings(provider="local")
        vectorstore = build_vectorstore(chunks, persist_dir="./data/chroma_db")
        
        print(f"✅ Ready! Indexed {len(chunks)} chunks from {len(documents)} documents")
        
        # Interactive loop
        while True:
            print("\n" + "-" * 40)
            question = input("❓ Your question: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ["quit", "exit", "q"]:
                print("👋 Goodbye!")
                break
            
            if question.lower() == "clear":
                import shutil
                chroma_path = Path("./data/chroma_db")
                if chroma_path.exists():
                    shutil.rmtree(chroma_path)
                print("🗑️ Index cleared!")
                continue
            
            # Process question
            print("\n🔍 Retrieving context...")
            results = retrieve_and_rerank(vectorstore, question, k=10, rerank_top=3)
            
            context_chunks = [
                {"content": r["content"], "metadata": r["metadata"]}
                for r in results
            ]
            
            print("🤖 Generating answer...")
            gen_result = generate_answer(question, context_chunks)
            
            print("\n📝 Answer:")
            print(gen_result["answer"])
            
            print(f"\n📚 Sources: {len(gen_result['sources'])}")
            for i, s in enumerate(gen_result["sources"], 1):
                print(f"   [{i}] {s.get('source', 'unknown')}")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        raise


# =========================================================
# CLI Entry Point
# =========================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="SelmaData Pipeline - Ask questions about your documents"
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="Question to ask (omit for interactive mode)",
    )
    parser.add_argument(
        "--data", "-d",
        default="./data/raw",
        help="Data folder containing PDFs (default: ./data/raw)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of candidates to retrieve (default: 10)",
    )
    parser.add_argument(
        "--rerank-top",
        type=int,
        default=3,
        help="Number of results after re-ranking (default: 3)",
    )
    parser.add_argument(
        "--model", "-m",
        default="glm-4-flash",
        help="Provider model to use (default: glm-4-flash)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress detailed output",
    )
    
    args = parser.parse_args()
    
    if args.question:
        # Single question mode
        result = run_selma_data_pipeline(
            question=args.question,
            data_folder=args.data,
            k=args.k,
            rerank_top=args.rerank_top,
            model=args.model,
            verbose=not args.quiet,
        )
    else:
        # Interactive mode
        interactive_mode()
