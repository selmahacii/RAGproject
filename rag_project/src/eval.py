"""
RAG Evaluation Module with RAGAs

Provides comprehensive evaluation metrics for RAG systems:
- Faithfulness: Is the answer grounded in context? Target > 0.85
- Answer Relevancy: Does the answer address the question? Target > 0.80
- Context Precision: Are retrieved chunks useful? Target > 0.75
- Context Recall: Were all useful chunks retrieved? Target > 0.80

This module follows RAGAs best practices for evaluation.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from .config import get_settings


# ============================================
# STEP 7: Evaluation with RAGAs
# ============================================

"""
┌─────────────────────────────────────────────────────────────────────────────┐
│                        RAGAs EVALUATION METRICS GUIDE                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  FAITHFULNESS (忠实度)                                               │    │
│  ├─────────────────────────────────────────────────────────────────────┤    │
│  │  Question: Is the answer grounded in the retrieved context?          │    │
│  │  Target:    > 0.85                                                   │    │
│  │                                                                      │    │
│  │  What it measures:                                                   │    │
│  │  - Does the LLM hallucinate information?                            │    │
│  │  - Are all claims in the answer supported by context?               │    │
│  │                                                                      │    │
│  │  ┌─────────────────────────────────────────────────────────────┐    │    │
│  │  │  IF LOW FAITHFULNESS (< 0.85):                               │    │    │
│  │  │                                                               │    │    │
│  │  │  1. STRICHER SYSTEM PROMPT                                   │    │    │
│  │  │     "Answer ONLY from context. Never add external info."     │    │    │
│  │  │                                                               │    │    │
│  │  │  2. LOWER TEMPERATURE                                        │    │    │
│  │  │     temperature: 0.1 → 0.0 (more deterministic)              │    │    │
│  │  │                                                               │    │    │
│  │  │  3. ADD EXPLICIT CONSTRAINTS                                 │    │    │
│  │  │     "If info not in context, say 'I don't know'"            │    │    │
│  │  │                                                               │    │    │
│  │  │  4. CHECK CHUNK QUALITY                                      │    │    │
│  │  │     Too small chunks may lack context for grounding          │    │    │
│  │  └─────────────────────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  ANSWER RELEVANCY (答案相关性)                                       │    │
│  ├─────────────────────────────────────────────────────────────────────┤    │
│  │  Question: Does the answer address the user's question?             │    │
│  │  Target:    > 0.80                                                   │    │
│  │                                                                      │    │
│  │  What it measures:                                                   │    │
│  │  - Is the answer on-topic?                                          │    │
│  │  - Does it actually answer what was asked?                          │    │
│  │                                                                      │    │
│  │  ┌─────────────────────────────────────────────────────────────┐    │    │
│  │  │  IF LOW ANSWER RELEVANCY (< 0.80):                           │    │    │
│  │  │                                                               │    │    │
│  │  │  1. REWRITE SYSTEM PROMPT                                    │    │    │
│  │  │     Add: "Focus on directly answering the question"         │    │    │
│  │  │                                                               │    │    │
│  │  │  2. ADD INSTRUCTIONS FOR HANDLING OFF-TOPIC CONTEXT          │    │    │
│  │  │     "If context doesn't help, explain what you found"       │    │    │
│  │  │                                                               │    │    │
│  │  │  3. IMPROVE QUERY UNDERSTANDING                              │    │    │
│  │  │     Consider query rewriting/expansion before retrieval      │    │    │
│  │  │                                                               │    │    │
│  │  │  4. CHECK IF RETRIEVED CONTEXT IS RELEVANT                   │    │    │
│  │  │     If context_precision is also low, fix retrieval first    │    │    │
│  │  └─────────────────────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  CONTEXT PRECISION (上下文精确度)                                    │    │
│  ├─────────────────────────────────────────────────────────────────────┤    │
│  │  Question: Are the retrieved chunks actually useful?                │    │
│  │  Target:    > 0.75                                                   │    │
│  │                                                                      │    │
│  │  What it measures:                                                   │    │
│  │  - Signal-to-noise ratio in retrieved context                       │    │
│  │  - How much of the retrieved context is actually relevant?          │    │
│  │                                                                      │    │
│  │  ┌─────────────────────────────────────────────────────────────┐    │    │
│  │  │  IF LOW CONTEXT PRECISION (< 0.75):                          │    │    │
│  │  │                                                               │    │    │
│  │  │  1. BETTER CHUNKING STRATEGY                                 │    │    │
│  │  │     - Increase chunk_size (512 → 1024) for more context     │    │    │
│  │  │     - Use semantic chunking for better boundaries           │    │    │
│  │  │     - Try parent-document retriever for richer context      │    │    │
│  │  │                                                               │    │    │
│  │  │  2. ADD RE-RANKING                                           │    │    │
│  │  │     - Use FlashRank to filter out irrelevant chunks         │    │    │
│  │  │     - Retrieve k=10, rerank to top 3                        │    │    │
│  │  │                                                               │    │    │
│  │  │  3. IMPROVE EMBEDDINGS                                       │    │    │
│  │  │     - Use better embedding model (bge-m3, gte-large)        │    │    │
│  │  │     - Consider domain-specific fine-tuning                  │    │    │
│  │  │                                                               │    │    │
│  │  │  4. REDUCE TOP_K                                             │    │    │
│  │  │     - If retrieving too many noisy chunks, reduce k         │    │    │
│  │  └─────────────────────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  CONTEXT RECALL (上下文召回率)                                       │    │
│  ├─────────────────────────────────────────────────────────────────────┤    │
│  │  Question: Were all useful chunks retrieved?                        │    │
│  │  Target:    > 0.80                                                   │    │
│  │                                                                      │    │
│  │  What it measures:                                                   │    │
│  │  - Did we miss important information?                               │    │
│  │  - Coverage of relevant documents in the corpus                     │    │
│  │                                                                      │    │
│  │  ┌─────────────────────────────────────────────────────────────┐    │    │
│  │  │  IF LOW CONTEXT RECALL (< 0.80):                             │    │    │
│  │  │                                                               │    │    │
│  │  │  1. INCREASE K (retrieval count)                             │    │    │
│  │  │     - top_k: 5 → 10 → 20                                     │    │    │
│  │  │     - More candidates = better chance of finding all        │    │    │
│  │  │                                                               │    │    │
│  │  │  2. IMPROVE EMBEDDINGS                                       │    │    │
│  │  │     - Better model captures semantic meaning better         │    │    │
│  │  │     - Try multilingual models for mixed language docs       │    │    │
│  │  │                                                               │    │    │
│  │  │  3. USE MMR FOR DIVERSITY                                    │    │    │
│  │  │     - retrieve_mmr ensures diverse results                  │    │    │
│  │  │     - Avoids returning near-duplicate chunks                │    │    │
│  │  │                                                               │    │    │
│  │  │  4. HYBRID SEARCH (VECTOR + KEYWORD)                         │    │    │
│  │  │     - Some queries benefit from keyword matching            │    │    │
│  │  │     - Combine BM25 + vector search                          │    │    │
│  │  └─────────────────────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
"""

# Metric targets for production RAG systems
METRIC_TARGETS = {
    "faithfulness": 0.85,      # Answer grounded in context
    "answer_relevancy": 0.80,  # Answer addresses question
    "context_precision": 0.75, # Retrieved chunks are useful
    "context_recall": 0.80,    # All useful chunks retrieved
}


@dataclass
class EvaluationSample:
    """
    A single evaluation sample for RAGAs.
    
    Required fields for RAGAs:
    - question: The user query
    - answer: Generated answer from RAG system
    - contexts: List of retrieved context strings
    - ground_truth: Expected/human-verified answer (for reference-based metrics)
    
    Attributes:
        question: The question to evaluate
        answer: Generated answer (filled by RAG system)
        contexts: Retrieved contexts (filled by retrieval)
        ground_truth: Expected answer for evaluation
        metadata: Additional metadata (source, difficulty, etc.)
    """
    question: str
    answer: str = ""
    contexts: list[str] = field(default_factory=list)
    ground_truth: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for RAGAs Dataset."""
        return {
            "question": self.question,
            "answer": self.answer,
            "contexts": self.contexts,
            "ground_truth": self.ground_truth,
        }
    
    def is_complete(self) -> bool:
        """Check if sample has all required fields."""
        return bool(self.question and self.answer and self.contexts and self.ground_truth)


@dataclass
class EvaluationResult:
    """
    Result of RAGAs evaluation.
    
    Contains per-sample and aggregate metrics.
    """
    # Core RAGAs metrics
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    
    # Metadata
    num_samples: int = 0
    evaluation_time: float = 0.0
    samples: list[dict[str, Any]] = field(default_factory=list)
    
    def __str__(self) -> str:
        """Pretty print the result."""
        lines = [
            "=" * 60,
            "           RAGAs EVALUATION RESULTS",
            "=" * 60,
            "",
            f"  Metric              Score    Target    Status",
            "  " + "-" * 52,
        ]
        
        metrics = [
            ("Faithfulness", self.faithfulness, METRIC_TARGETS["faithfulness"]),
            ("Answer Relevancy", self.answer_relevancy, METRIC_TARGETS["answer_relevancy"]),
            ("Context Precision", self.context_precision, METRIC_TARGETS["context_precision"]),
            ("Context Recall", self.context_recall, METRIC_TARGETS["context_recall"]),
        ]
        
        for name, score, target in metrics:
            status = "✅ PASS" if score >= target else "❌ FAIL"
            lines.append(f"  {name:<20} {score:.3f}    >{target:.2f}    {status}")
        
        # Overall
        avg_score = np.mean([m[1] for m in metrics])
        avg_target = np.mean([m[2] for m in metrics])
        overall_status = "✅ PASS" if avg_score >= avg_target else "❌ FAIL"
        
        lines.extend([
            "  " + "-" * 52,
            f"  {'Overall Average':<20} {avg_score:.3f}    >{avg_target:.2f}    {overall_status}",
            "",
            f"  Samples evaluated: {self.num_samples}",
            f"  Evaluation time:   {self.evaluation_time:.1f}s",
            "=" * 60,
        ])
        
        return "\n".join(lines)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        if not self.samples:
            return pd.DataFrame()
        
        return pd.DataFrame(self.samples)


def build_test_dataset(
    questions: list[str],
    ground_truths: list[str],
    rag_pipeline: Optional[Any] = None,
    contexts_list: Optional[list[list[str]]] = None,
    answers: Optional[list[str]] = None,
) -> list[EvaluationSample]:
    """
    Build a test dataset for RAGAs evaluation.
    
    For a production evaluation, you need 20-50 Q/A pairs covering:
    - Different question types (factual, analytical, comparative)
    - Different difficulty levels (easy, medium, hard)
    - Different document sections
    
    Args:
        questions: List of test questions
        ground_truths: Expected answers for each question
        rag_pipeline: RAG pipeline to generate answers and retrieve contexts
        contexts_list: Pre-retrieved contexts (if not using pipeline)
        answers: Pre-generated answers (if not using pipeline)
    
    Returns:
        List of EvaluationSample ready for RAGAs
    
    Example:
        >>> # Manual test dataset creation
        >>> questions = [
        ...     "What is RAG?",
        ...     "How does vector search work?",
        ...     "What are the benefits of re-ranking?",
        ... ]
        >>> ground_truths = [
        ...     "RAG combines retrieval and generation for accurate answers.",
        ...     "Vector search finds similar documents using embeddings.",
        ...     "Re-ranking improves precision by re-scoring candidates.",
        ... ]
        >>> samples = build_test_dataset(questions, ground_truths, pipeline)
    
    Example for large-scale evaluation (20-50 pairs):
        >>> # Create diverse test set
        >>> test_questions = [
        ...     # Factual questions (what, when, who)
        ...     "What is the main purpose of RAG?",
        ...     "When should I use semantic chunking?",
        ...     
        ...     # Analytical questions (how, why)
        ...     "How does re-ranking improve retrieval quality?",
        ...     "Why is chunk overlap important?",
        ...     
        ...     # Comparative questions
        ...     "What's the difference between vector search and keyword search?",
        ...     "Compare recursive vs semantic chunking.",
        ...     
        ...     # Edge cases
        ...     "What if the answer is not in the documents?",
        ...     "How to handle multi-hop questions?",
        ... ]
        >>> # ... create 20-50 diverse questions
    """
    if len(questions) != len(ground_truths):
        raise ValueError(
            f"Mismatch: {len(questions)} questions but {len(ground_truths)} ground truths"
        )
    
    samples = []
    
    for i, (question, ground_truth) in enumerate(zip(questions, ground_truths)):
        sample = EvaluationSample(
            question=question,
            ground_truth=ground_truth,
            metadata={"sample_id": i},
        )
        
        # If RAG pipeline provided, run query
        if rag_pipeline is not None:
            try:
                response = rag_pipeline.query(question)
                sample.answer = response.get("answer", "")
                sample.contexts = [s.get("content", "") for s in response.get("sources", [])]
            except Exception as e:
                logger.warning(f"Failed to get RAG response for sample {i}: {e}")
        else:
            # Use provided contexts and answers
            if contexts_list and i < len(contexts_list):
                sample.contexts = contexts_list[i]
            if answers and i < len(answers):
                sample.answer = answers[i]
        
        samples.append(sample)
    
    logger.info(f"Built test dataset with {len(samples)} samples")
    return samples


def evaluate_with_ragas(
    samples: list[EvaluationSample],
    llm: Optional[Any] = None,
    embeddings: Optional[Any] = None,
    metrics: Optional[list[str]] = None,
) -> EvaluationResult:
    """
    Evaluate samples using RAGAs library.
    
    RAGAs uses an LLM to judge the quality of RAG outputs.
    It requires an OpenAI-compatible LLM for evaluation.
    
    Args:
        samples: List of completed EvaluationSample
        llm: LangChain LLM for RAGAs (default: uses config)
        embeddings: LangChain embeddings for some metrics
        metrics: Specific metrics to compute (default: all 4)
    
    Returns:
        EvaluationResult with aggregate and per-sample scores
    
    Example:
        >>> samples = build_test_dataset(questions, ground_truths, pipeline)
        >>> result = evaluate_with_ragas(samples)
        >>> print(result)
    """
    start_time = time.time()
    
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
    except ImportError as e:
        logger.error(f"RAGAs not installed: {e}")
        logger.error("Install with: pip install ragas datasets")
        return _evaluate_with_heuristics(samples)
    
    # Check if samples are complete
    complete_samples = [s for s in samples if s.is_complete()]
    if not complete_samples:
        logger.error("No complete samples for evaluation")
        return EvaluationResult(num_samples=len(samples))
    
    if len(complete_samples) < len(samples):
        logger.warning(
            f"Only {len(complete_samples)}/{len(samples)} samples are complete"
        )
    
    # Build RAGAs dataset
    data = {
        "question": [s.question for s in complete_samples],
        "answer": [s.answer for s in complete_samples],
        "contexts": [s.contexts for s in complete_samples],
        "ground_truth": [s.ground_truth for s in complete_samples],
    }
    dataset = Dataset.from_dict(data)
    
    # Select metrics
    all_metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    
    if metrics:
        metric_map = {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
            "context_recall": context_recall,
        }
        selected_metrics = [metric_map[m] for m in metrics if m in metric_map]
    else:
        selected_metrics = all_metrics
    
    # Set up LLM for RAGAs if provided
    llm_for_ragas = llm
    if llm_for_ragas is None:
        # Try to create from settings
        settings = get_settings()
        try:
            from langchain_openai import ChatOpenAI
            # RAGAs works best with OpenAI models
            # For ZhipuAI, you'd need an OpenAI-compatible endpoint
            llm_for_ragas = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0,
            )
        except ImportError:
            logger.warning("langchain_openai not installed, RAGAs may not work")
    
    logger.info(f"Running RAGAs evaluation on {len(complete_samples)} samples")
    
    try:
        # Run RAGAs evaluation
        eval_result = evaluate(
            dataset,
            metrics=selected_metrics,
            llm=llm_for_ragas,
        )
        
        # Convert to DataFrame
        result_df = eval_result.to_pandas()
        
        # Calculate aggregate scores
        result = EvaluationResult(
            faithfulness=result_df["faithfulness"].mean() if "faithfulness" in result_df else 0.0,
            answer_relevancy=result_df["answer_relevancy"].mean() if "answer_relevancy" in result_df else 0.0,
            context_precision=result_df["context_precision"].mean() if "context_precision" in result_df else 0.0,
            context_recall=result_df["context_recall"].mean() if "context_recall" in result_df else 0.0,
            num_samples=len(complete_samples),
            evaluation_time=time.time() - start_time,
            samples=result_df.to_dict("records"),
        )
        
        logger.info(f"RAGAs evaluation complete: {result.evaluation_time:.1f}s")
        return result
    
    except Exception as e:
        logger.error(f"RAGAs evaluation failed: {e}")
        return _evaluate_with_heuristics(samples)


def _evaluate_with_heuristics(samples: list[EvaluationSample]) -> EvaluationResult:
    """
    Fallback evaluation using simple heuristics.
    
    Used when RAGAs is not available. These are rough estimates
    and should not be used for production evaluation.
    """
    logger.warning("Using heuristic evaluation (RAGAs not available)")
    
    results = []
    for sample in samples:
        if not sample.is_complete():
            continue
        
        # Faithfulness: word overlap between answer and contexts
        answer_words = set(sample.answer.lower().split())
        context_words = set()
        for ctx in sample.contexts:
            context_words.update(ctx.lower().split())
        faithfulness = len(answer_words & context_words) / max(len(answer_words), 1)
        
        # Answer relevancy: word overlap between question and answer
        question_words = set(sample.question.lower().split())
        answer_relevancy = len(question_words & answer_words) / max(len(question_words), 1)
        
        # Context precision: heuristic based on number of contexts
        context_precision = min(1.0, 3.0 / max(len(sample.contexts), 1))
        
        # Context recall: overlap between ground truth and contexts
        gt_words = set(sample.ground_truth.lower().split())
        context_recall = len(context_words & gt_words) / max(len(gt_words), 1)
        
        results.append({
            "question": sample.question,
            "faithfulness": min(1.0, faithfulness),
            "answer_relevancy": min(1.0, answer_relevancy),
            "context_precision": context_precision,
            "context_recall": context_recall,
        })
    
    if not results:
        return EvaluationResult(num_samples=len(samples))
    
    df = pd.DataFrame(results)
    
    return EvaluationResult(
        faithfulness=df["faithfulness"].mean(),
        answer_relevancy=df["answer_relevancy"].mean(),
        context_precision=df["context_precision"].mean(),
        context_recall=df["context_recall"].mean(),
        num_samples=len(results),
        evaluation_time=0.0,
        samples=results,
    )


def generate_report(
    result: EvaluationResult,
    output_path: str = "eval_report.csv",
    include_summary: bool = True,
) -> pd.DataFrame:
    """
    Generate evaluation report from RAGAs results.
    
    This function:
    1. Converts results to pandas DataFrame
    2. Prints describe() summary statistics
    3. Identifies problematic samples (faithfulness < 0.7)
    4. Saves report to CSV
    
    Args:
        result: EvaluationResult from evaluate_with_ragas()
        output_path: Path to save CSV report
        include_summary: Whether to print summary statistics
    
    Returns:
        pandas DataFrame with detailed results
    
    Example:
        >>> result = evaluate_with_ragas(samples)
        >>> df = generate_report(result, "eval_report.csv")
        >>> # This prints summary and saves CSV
    """
    df = result.to_dataframe()
    
    if df.empty:
        logger.warning("No samples in evaluation result")
        return df
    
    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if include_summary:
        print("\n" + "=" * 60)
        print("           EVALUATION REPORT SUMMARY")
        print("=" * 60)
        
        # Print aggregate results
        print(result)
        
        # Print describe() for numeric columns
        print("\n--- Statistical Summary (describe) ---")
        metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        existing_cols = [c for c in metric_cols if c in df.columns]
        
        if existing_cols:
            print(df[existing_cols].describe().round(3).to_string())
        
        # Identify problematic samples
        print("\n--- Problematic Samples Analysis ---")
        
        # Low faithfulness samples (< 0.7)
        if "faithfulness" in df.columns:
            low_faith = df[df["faithfulness"] < 0.7]
            if len(low_faith) > 0:
                print(f"\n⚠️  {len(low_faith)} samples with faithfulness < 0.7:")
                print("   These answers may contain hallucinations.")
                print("   Questions:")
                for i, row in low_faith.iterrows():
                    question_preview = row.get("question", "")[:50]
                    score = row.get("faithfulness", 0)
                    print(f"   - [{score:.2f}] {question_preview}...")
            else:
                print("\n✅ No samples with critically low faithfulness (< 0.7)")
        
        # Low answer relevancy samples (< 0.7)
        if "answer_relevancy" in df.columns:
            low_rel = df[df["answer_relevancy"] < 0.7]
            if len(low_rel) > 0:
                print(f"\n⚠️  {len(low_rel)} samples with answer_relevancy < 0.7:")
                print("   These answers may not address the questions.")
        
        # Suggestions for improvement
        print("\n--- Improvement Suggestions ---")
        
        suggestions = []
        
        if result.faithfulness < METRIC_TARGETS["faithfulness"]:
            suggestions.append(
                "• Low faithfulness → Use stricter system prompt, lower temperature (0.1→0.0)"
            )
        
        if result.answer_relevancy < METRIC_TARGETS["answer_relevancy"]:
            suggestions.append(
                "• Low answer_relevancy → Rewrite prompt, add explicit answer instructions"
            )
        
        if result.context_precision < METRIC_TARGETS["context_precision"]:
            suggestions.append(
                "• Low context_precision → Better chunking, add re-ranking (FlashRank)"
            )
        
        if result.context_recall < METRIC_TARGETS["context_recall"]:
            suggestions.append(
                "• Low context_recall → Increase k, improve embeddings, use MMR"
            )
        
        if suggestions:
            print("\n".join(suggestions))
        else:
            print("✅ All metrics meet targets!")
        
        print("\n" + "=" * 60)
    
    # Add metadata columns
    df["eval_timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    df["num_samples"] = result.num_samples
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"\n📊 Report saved to: {output_path}")
    
    return df


def compare_strategies(
    results_dict: dict[str, EvaluationResult],
) -> pd.DataFrame:
    """
    Compare evaluation results from different RAG strategies side by side.
    
    This is useful for A/B testing different configurations:
    - Recursive chunking vs Semantic chunking
    - With re-ranking vs Without re-ranking
    - Different embedding models
    
    Args:
        results_dict: Dict mapping strategy name to EvaluationResult
                     e.g., {"recursive": result1, "semantic": result2}
    
    Returns:
        pandas DataFrame comparing mean scores
    
    Example:
        >>> # Compare recursive vs semantic chunking
        >>> recursive_result = evaluate_with_ragas(samples_recursive)
        >>> semantic_result = evaluate_with_ragas(samples_semantic)
        >>> 
        >>> comparison = compare_strategies({
        ...     "recursive_chunking": recursive_result,
        ...     "semantic_chunking": semantic_result,
        ... })
        >>> print(comparison)
        
        Output:
        ┌──────────────────┬───────────────┬─────────────────┐
        │ metric           │ recursive     │ semantic        │
        ├──────────────────┼───────────────┼─────────────────┤
        │ faithfulness     │ 0.823         │ 0.856           │
        │ answer_relevancy │ 0.791         │ 0.812           │
        │ context_precision│ 0.702         │ 0.789           │
        │ context_recall   │ 0.756         │ 0.823           │
        └──────────────────┴───────────────┴─────────────────┘
    """
    comparison_data = []
    
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    
    for metric in metrics:
        row = {"metric": metric}
        for strategy_name, result in results_dict.items():
            row[strategy_name] = getattr(result, metric, 0.0)
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    df = df.set_index("metric")
    
    # Calculate which strategy is better for each metric
    if len(results_dict) == 2:
        strategies = list(results_dict.keys())
        df["winner"] = df.apply(
            lambda row: strategies[0] if row[strategies[0]] > row[strategies[1]] 
            else strategies[1] if row[strategies[1]] > row[strategies[0]] 
            else "tie",
            axis=1
        )
    
    # Print comparison
    print("\n" + "=" * 70)
    print("           STRATEGY COMPARISON")
    print("=" * 70)
    
    # Format for display
    display_df = df.round(3)
    print(display_df.to_string())
    
    # Summary
    print("\n--- Summary ---")
    for strategy_name, result in results_dict.items():
        avg_score = np.mean([
            result.faithfulness,
            result.answer_relevancy,
            result.context_precision,
            result.context_recall,
        ])
        print(f"  {strategy_name}: avg = {avg_score:.3f}")
    
    print("=" * 70)
    
    return df


# ============================================
# Backward Compatibility Classes
# ============================================

class RAGEvaluator:
    """
    RAG Evaluator class for backward compatibility.
    
    Usage:
        evaluator = RAGEvaluator(rag_pipeline)
        samples = build_test_dataset(questions, ground_truths, rag_pipeline)
        result = evaluator.evaluate(samples)
    """
    
    def __init__(
        self,
        pipeline: Optional[Any] = None,
        use_ragas: bool = True,
    ):
        self.pipeline = pipeline
        self.use_ragas = use_ragas
    
    def evaluate(
        self,
        samples: list[EvaluationSample],
        show_progress: bool = True,
    ) -> EvaluationResult:
        """Evaluate samples and return results."""
        
        # If pipeline provided, fill in answers and contexts
        if self.pipeline is not None:
            for sample in tqdm(samples, desc="Running RAG", disable=not show_progress):
                try:
                    response = self.pipeline.query(sample.question)
                    sample.answer = response.get("answer", "")
                    sample.contexts = [s.get("content", "") for s in response.get("sources", [])]
                except Exception as e:
                    logger.warning(f"Failed to process sample: {e}")
        
        # Run evaluation
        if self.use_ragas:
            return evaluate_with_ragas(samples)
        else:
            return _evaluate_with_heuristics(samples)
    
    def generate_report(
        self,
        result: EvaluationResult,
        output_path: str = "eval_report.csv",
    ) -> pd.DataFrame:
        """Generate evaluation report."""
        return generate_report(result, output_path)


class TestCaseGenerator:
    """
    Generate test cases for RAG evaluation.
    
    Creates synthetic question-answer pairs from documents.
    """
    
    def __init__(self, llm: Optional[Any] = None):
        self.llm = llm
    
    def generate_from_documents(
        self,
        documents: Sequence[Any],
        num_questions_per_doc: int = 2,
    ) -> list[EvaluationSample]:
        """Generate test samples from documents."""
        samples = []
        
        for doc in tqdm(documents, desc="Generating test cases"):
            content = doc.content if hasattr(doc, 'content') else str(doc)
            metadata = doc.metadata if hasattr(doc, 'metadata') else {}
            
            # Simple question generation (use LLM for better results)
            sentences = content.replace("\n", " ").split(". ")
            sentences = [s.strip() for s in sentences if len(s.strip()) > 30]
            
            for i, sentence in enumerate(sentences[:num_questions_per_doc]):
                # Create question from sentence
                words = sentence.split()[:8]
                question = f"What is mentioned about {' '.join(words[:5])}?"
                
                samples.append(EvaluationSample(
                    question=question,
                    ground_truth=sentence,
                    metadata={"source": metadata.get("source", "unknown")},
                ))
        
        logger.info(f"Generated {len(samples)} test samples")
        return samples


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("RAG Evaluation Module - RAGAs Integration")
    print("=" * 60)
    
    # Create sample evaluation data
    sample_data = [
        {
            "question": "What is RAG?",
            "answer": "RAG (Retrieval-Augmented Generation) is a technique that combines retrieval and generation for accurate answers.",
            "contexts": ["RAG combines retrieval and generation.", "It uses vector databases for retrieval."],
            "ground_truth": "RAG is a technique combining retrieval with generation.",
        },
        {
            "question": "How does vector search work?",
            "answer": "Vector search finds similar documents using embeddings and cosine similarity.",
            "contexts": ["Vector search uses embeddings to represent documents.", "Similarity is measured with cosine distance."],
            "ground_truth": "Vector search uses embeddings and similarity metrics.",
        },
    ]
    
    # Create samples
    samples = [
        EvaluationSample(
            question=d["question"],
            answer=d["answer"],
            contexts=d["contexts"],
            ground_truth=d["ground_truth"],
        )
        for d in sample_data
    ]
    
    # Run heuristic evaluation (RAGAs requires OpenAI)
    result = _evaluate_with_heuristics(samples)
    
    print(result)
    
    # Generate report
    df = generate_report(result, "eval_report.csv")
    
    print("\n--- Sample DataFrame ---")
    print(df[["question", "faithfulness", "answer_relevancy"]].to_string())
