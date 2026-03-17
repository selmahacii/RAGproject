"""
Evaluation Module (Step 7)

Uses RAGAs to evaluate factual accuracy, relevancy, and retrieval quality.
"""

import pandas as pd
from typing import List, Dict, Any
from loguru import logger

# ============================================
# STEP 7: Evaluation with RAGAs
# ============================================

"""
RAGAs Metrics targets for production:
- faithfulness: Target > 0.85 (Is the answer grounded in context?)
- answer_relevancy: Target > 0.80 (Does it address the question?)
- context_precision: Target > 0.75 (Are retrieved chunks useful?)
- context_recall: Target > 0.80 (Were all useful chunks retrieved?)
"""

def build_test_dataset(questions: List[str], ground_truths: List[str], rag_pipeline=None) -> List[Dict[str, Any]]:
    """
    Build a test dataset of 20-50 Q/A pairs.
    Each item contains: question, answer, contexts, ground_truth.
    """
    dataset = []
    logger.info(f"Building dataset for {len(questions)} questions...")
    
    for q, gt in zip(questions, ground_truths):
        # In a real scenario, we'd run the RAG pipeline here
        # For this step, we show the structure required by Ragas
        sample = {
            "question": q,
            "answer": "", # To be filled by RAG
            "contexts": [], # To be filled by Retrieval
            "ground_truth": gt
        }
        
        if rag_pipeline:
            # Simulated pipeline call
            res = rag_pipeline.query(q)
            sample["answer"] = res["answer"]
            sample["contexts"] = [c["content"] for c in res["sources"]]
            
        dataset.append(sample)
    
    return dataset


def evaluate_rag(dataset: List[Dict[str, Any]]):
    """
    Run evaluation using RAGAs metrics.
    Requires an OpenAI-compatible API key for the 'critics' models.
    """
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from datasets import Dataset
    except ImportError:
        logger.error("RAGAs not installed. Run: pip install ragas datasets")
        return None

    # Transform list to HuggingFace Dataset
    hf_dataset = Dataset.from_list(dataset)
    
    # Run evaluation
    result = evaluate(
        hf_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
    )
    
    return result


def generate_report(result, output_path: str = "eval_report.csv"):
    """
    Analyze results and generate a production report.
    """
    # 1. Convert to pandas
    df = result.to_pandas()
    
    # 2. Print summary
    print("\n--- RAG Evaluation Summary ---")
    print(df.describe())
    
    # 3. Identify problematic samples (faithfulness < 0.7)
    hallucinations = df[df["faithfulness"] < 0.7]
    if not hallucinations.empty:
        print(f"\n⚠️ WARNING: Found {len(hallucinations)} potential hallucinations (faithfulness < 0.7)")
        print(hallucinations[["question", "faithfulness"]])
    
    # 4. Save to CSV
    df.to_csv(output_path, index=False)
    print(f"\n✅ Report saved to {output_path}")
    
    # Suggestions based on performance
    print("\n--- Debugging Guide ---")
    if result["faithfulness"] < 0.85:
        print("💡 Low faithfulness → stricter system prompt, lower temperature")
    if result["answer_relevancy"] < 0.80:
        print("💡 Low answer_relevancy → rewrite prompt, add instructions")
    if result["context_precision"] < 0.75:
        print("💡 Low context_precision → better chunking, add re-ranking")
    if result["context_recall"] < 0.80:
        print("💡 Low context_recall → increase k, improve embeddings")


def compare_strategies(results_a, results_b, name_a="recursive", name_b="semantic"):
    """
    Compare two RAG strategies side by side.
    """
    score_a = pd.Series(results_a).rename(name_a)
    score_b = pd.Series(results_b).rename(name_b)
    
    comparison = pd.concat([score_a, score_b], axis=1)
    print(f"\n--- Strategy Comparison: {name_a} vs {name_b} ---")
    print(comparison)
    return comparison

if __name__ == "__main__":
    print("Evaluation Module Step 7 Ready.")
