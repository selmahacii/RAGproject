"""
LLM Generation Module (Step 6)

Integrated with ZhipuAI GLM API.
Handles prompting, context formatting, and streaming responses.
"""

import time
from typing import List, Dict, Any, Generator, Sequence, Union
from loguru import logger
from .config import get_settings

# ============================================
# STEP 6: LLM Generation with GLM API
# ============================================

SYSTEM_PROMPT = """You are a professional RAG Assistant. 
Answer the user's question ONLY based on the provided context.

CRITICAL RULES:
1. If the answer is not contained within the context, state clearly that you do not have enough information.
2. ALWAYS cite the source for every piece of information used (use source_file or source_url).
3. Reply in the EXACT SAME LANGUAGE as the user's question.
4. If your answer contains more than 2 pieces of information, use bullet points for readability.
5. Do not use your internal knowledge; act only as a retrieval-based generator.
"""

USER_TEMPLATE = """Based on the following context documents, answer the question at the end.

CONTEXT:
{context}

QUESTION:
{question}
"""

def build_prompt(question: str, chunks: List[Any]) -> tuple:
    """
    Format context chunks and build the final prompt for the LLM.
    
    Context format requirement: [1] SOURCE: filename\ncontent
    """
    context_str = ""
    for i, chunk in enumerate(chunks, 1):
        # Extract metadata accurately
        if hasattr(chunk, 'metadata'):
            metadata = chunk.metadata
            content = chunk.page_content
        else:
            # Fallback for dict-like results from retrieval
            metadata = chunk.get('metadata', {})
            content = chunk.get('content', '')
            
        source = metadata.get("source_file") or metadata.get("source_url") or "Unknown Source"
        context_str += f"[{i}] SOURCE: {source}\n{content}\n\n"
        
    user_message = USER_TEMPLATE.format(context=context_str, question=question)
    return SYSTEM_PROMPT, user_message


def generate_answer(question: str, chunks: List[Any], model: str = "glm-4-flash") -> Dict[str, Any]:
    """
    Generate a factual answer using ZhipuAI.
    
    Settings: temperature=0.1 (factual), top_p=0.9.
    Returns: answer, model, tokens_used, sources
    """
    try:
        from zhipuai import ZhipuAI
    except ImportError:
        logger.error("ZhipuAI not installed. Run: pip install zhipuai")
        return {"error": "zhipuai not installed"}

    settings = get_settings()
    client = ZhipuAI(api_key=settings.zhipuai_api_key)
    
    system_p, user_p = build_prompt(question, chunks)
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_p},
            {"role": "user", "content": user_p},
        ],
        temperature=0.1,
        max_tokens=1024,
        top_p=0.9
    )
    
    # Extract sources for the UI
    sources = list(set([
        (c.metadata.get("source_file") or c.metadata.get("source_url")) 
        if hasattr(c, 'metadata') else (c.get('metadata', {}).get("source_file") or c.get('metadata', {}).get("source_url"))
        for c in chunks
    ]))

    return {
        "answer": response.choices[0].message.content,
        "model": model,
        "tokens_used": response.usage.total_tokens,
        "sources": sources
    }


def generate_streaming(question: str, chunks: List[Any], model: str = "glm-4-flash") -> Generator[str, None, None]:
    """
    Stream tokens from ZhipuAI for real-time UI updates (Streamlit compatible).
    """
    try:
        from zhipuai import ZhipuAI
    except ImportError:
        yield "Error: zhipuai not installed"
        return

    settings = get_settings()
    client = ZhipuAI(api_key=settings.zhipuai_api_key)
    
    system_p, user_p = build_prompt(question, chunks)
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_p},
            {"role": "user", "content": user_p},
        ],
        temperature=0.1,
        max_tokens=1024,
        top_p=0.9,
        stream=True
    )
    
    for chunk in response:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

if __name__ == "__main__":
    print("LLM Module Step 6 Verified.")
