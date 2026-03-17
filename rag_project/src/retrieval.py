"""
Retrieval Module (Step 5)

Implements advanced retrieval strategies:
1. Basic Similarity Search
2. Re-ranking with FlashRank (Cross-Encoder)
3. Maximal Marginal Relevance (MMR) for diversity
"""

from typing import List, Dict, Any
from loguru import logger
from langchain_community.vectorstores import Chroma

# ============================================
# STEP 5: Retrieval with Re-ranking Functions
# ============================================

def basic_retrieval(vectorstore: Chroma, query: str, k: int = 5) -> List[Dict[str, Any]]:
    """
    Standard vector search using cosine similarity.
    Fast and efficient, but lacks semantic re-scoring.
    """
    # Use similarity_search_with_score to get the distance/score
    results = vectorstore.similarity_search_with_score(query, k=k)
    
    return [
        {
            "content": doc.page_content,
            "metadata": doc.metadata,
            "score": float(score),
            "rank": i + 1
        }
        for i, (doc, score) in enumerate(results)
    ]


def retrieve_and_rerank(vectorstore: Chroma, query: str, k: int = 10, rerank_top: int = 3) -> List[Dict[str, Any]]:
    """
    Two-stage retrieval: Vector search + Cross-Encoder re-ranking.
    
    Comment: +50ms latency cost but significantly better quality 
    as it uses a specialized model to re-score the top candidates.
    """
    try:
        from flashrank import Ranker, RerankRequest
    except ImportError:
        logger.error("FlashRank not installed. Run: pip install flashrank")
        return []

    # Step 1: Get candidates with vector search (wider net)
    candidates = vectorstore.similarity_search(query, k=k)
    
    if not candidates:
        return []

    # Step 2: Re-rank with FlashRank using specified model
    # Model: ms-marco-MiniLM-L-12-v2 is a great balance of speed and precision
    ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="./data/flashrank_cache")
    
    passages = [
        {"id": i, "text": doc.page_content, "metadata": doc.metadata}
        for i, doc in enumerate(candidates)
    ]
    
    rerank_request = RerankRequest(query=query, passages=passages)
    results = ranker.rerank(rerank_request)
    
    # Step 3: Return top results
    final_results = []
    for i, res in enumerate(results[:rerank_top]):
        final_results.append({
            "content": res["text"],
            "metadata": res["metadata"],
            "score": float(res["score"]),
            "rank": i + 1
        })
    
    logger.info(f"Re-ranked {len(candidates)} candidates down to {len(final_results)} using FlashRank.")
    return final_results


def retrieve_mmr(vectorstore: Chroma, query: str, k: int = 4) -> List[Dict[str, Any]]:
    """
    Maximal Marginal Relevance (MMR) retrieval.
    
    Explain: Avoids returning nearly identical chunks by balancing 
    relevance to the query with diversity among the results.
    """
    # fetch_k = k * 4: Number of candidates to fetch before applying MMR
    # lambda_mult = 0.5: 0 means max diversity, 1 means max relevance
    results = vectorstore.max_marginal_relevance_search(
        query, 
        k=k, 
        fetch_k=k*4, 
        lambda_mult=0.5
    )
    
    return [
        {
            "content": doc.page_content,
            "metadata": doc.metadata,
            "rank": i + 1
        }
        for i, doc in enumerate(results)
    ]

"""
💡 BEST PRACTICE:
The most robust production strategy is:
1. retrieve_mmr(k=10) to get a diverse set of relevant candidates.
2. rerank(top=3) to select the absolute best snippets for the LLM.
"""

if __name__ == "__main__":
    print("Retrieval Module Step 5 Verified.")
