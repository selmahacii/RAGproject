"""
LLM Generation Module with GLM API

Provides LLM integration for RAG generation:
- ZhipuAI GLM-4 integration
- Structured prompts for RAG
- Streaming support for Streamlit
- Source attribution
"""

import time
from dataclasses import dataclass, field
from typing import Any, Generator, Optional, Sequence, Union

from loguru import logger

from .config import get_settings
from .chunking import Chunk


# ============================================
# STEP 6: LLM Generation with GLM API
# ============================================

# SYSTEM_PROMPT: Instructions for the LLM to follow during RAG generation
# This ensures accurate, grounded responses with proper source attribution
SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based ONLY on the provided context.

CRITICAL RULES:
1. Answer ONLY using information from the provided context
2. If the answer is not in the context, say clearly: "I cannot find this information in the provided documents."
3. ALWAYS cite your sources using [SOURCE: filename] or [SOURCE: URL] format
4. Reply in the SAME LANGUAGE as the question (Chinese question → Chinese answer, etc.)
5. If providing more than 2 pieces of information, use bullet points for clarity
6. Be concise and accurate - do not add information not in the context

Remember: Your job is to be a faithful interpreter of the provided documents, not to add external knowledge."""

# USER_TEMPLATE: Structured template for RAG queries
# Format: [N] SOURCE: filename/URL followed by content
USER_TEMPLATE = """Based on the following context, please answer the question.

CONTEXT:
{context}

QUESTION:
{question}

Please provide your answer with source citations."""


@dataclass
class GenerationResult:
    """
    Result from LLM generation.
    
    Attributes:
        answer: The generated answer text
        model: Model used for generation
        tokens_used: Token usage statistics
        sources: List of source documents used
        latency_ms: Generation time in milliseconds
    """
    answer: str
    model: str
    tokens_used: dict[str, int] = field(default_factory=dict)
    sources: list[dict[str, Any]] = field(default_factory=list)
    latency_ms: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "answer": self.answer,
            "model": self.model,
            "tokens_used": self.tokens_used,
            "sources": self.sources,
            "latency_ms": self.latency_ms,
        }


def build_prompt(
    question: str,
    chunks: Sequence[Union[Chunk, dict[str, Any]]],
) -> tuple[str, str]:
    """
    Build system and user prompts for RAG generation.
    
    Formats context chunks with source attribution:
    [1] SOURCE: filename.pdf
    content here...
    
    [2] SOURCE: https://example.com
    content here...
    
    Args:
        question: User's question
        chunks: List of Chunk objects or dicts with 'content' and 'metadata'
    
    Returns:
        Tuple of (system_prompt, user_message)
    
    Example:
        >>> chunks = [Chunk(content="RAG is...", metadata={"source": "doc.pdf"})]
        >>> system, user = build_prompt("What is RAG?", chunks)
        >>> print(user)  # Shows formatted context with [1] SOURCE: doc.pdf
    """
    # Format each chunk with source attribution
    context_parts = []
    
    for i, chunk in enumerate(chunks, 1):
        # Handle both Chunk objects and dicts
        if isinstance(chunk, Chunk):
            content = chunk.content
            metadata = chunk.metadata
        else:
            content = chunk.get("content", "")
            metadata = chunk.get("metadata", {})
        
        # Extract source name (prefer source_file, then source, then source_url)
        source_name = (
            metadata.get("source_file") or
            metadata.get("source") or
            metadata.get("source_url") or
            "unknown"
        )
        
        # Format: [N] SOURCE: filename\ncontent
        context_parts.append(f"[{i}] SOURCE: {source_name}\n{content}")
    
    # Join all context parts
    formatted_context = "\n\n".join(context_parts)
    
    # Build user message from template
    user_message = USER_TEMPLATE.format(
        context=formatted_context,
        question=question,
    )
    
    return SYSTEM_PROMPT, user_message


def generate_answer(
    question: str,
    chunks: Sequence[Union[Chunk, dict[str, Any]]],
    model: str = "glm-4-flash",
) -> dict[str, Any]:
    """
    Generate answer using ZhipuAI GLM model.
    
    Uses conservative settings for factual, grounded responses:
    - temperature=0.1: Low creativity, high consistency
    - max_tokens=1024: Sufficient for detailed answers
    - top_p=0.9: Standard nucleus sampling
    
    Args:
        question: User's question
        chunks: Retrieved context chunks
        model: GLM model to use (default: glm-4-flash)
               Options: glm-4, glm-4-flash, glm-4-plus, glm-4-air
    
    Returns:
        Dict containing:
        - answer: Generated answer text
        - model: Model used
        - tokens_used: {prompt_tokens, completion_tokens, total_tokens}
        - sources: List of unique sources cited
        - latency_ms: Generation time in milliseconds
    
    Example:
        >>> result = generate_answer("What is RAG?", chunks, model="glm-4-flash")
        >>> print(result["answer"])
        >>> print(f"Used {result['tokens_used']['total_tokens']} tokens")
    """
    start_time = time.time()
    
    # Build prompts
    system_prompt, user_message = build_prompt(question, chunks)
    
    # Get API key
    settings = get_settings()
    api_key = settings.get_api_key()
    
    # Initialize ZhipuAI client
    try:
        from zhipuai import ZhipuAI
    except ImportError:
        raise ImportError(
            "zhipuai is required. Install with: pip install zhipuai"
        )
    
    client = ZhipuAI(api_key=api_key)
    
    # Build messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    
    logger.info(f"Generating answer with {model}")
    logger.debug(f"Question: {question[:100]}...")
    
    try:
        # Call API with conservative settings for factual responses
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.1,      # Low temperature = factual, not creative
            max_tokens=1024,      # Sufficient for detailed answers
            top_p=0.9,            # Standard nucleus sampling
        )
        
        # Extract response data
        answer = response.choices[0].message.content
        tokens_used = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
        
        # Extract unique sources from chunks
        sources = []
        seen_sources = set()
        for chunk in chunks:
            if isinstance(chunk, Chunk):
                metadata = chunk.metadata
            else:
                metadata = chunk.get("metadata", {})
            
            source = (
                metadata.get("source_file") or
                metadata.get("source") or
                metadata.get("source_url") or
                "unknown"
            )
            
            if source not in seen_sources:
                seen_sources.add(source)
                sources.append({
                    "source": source,
                    "page": metadata.get("page"),
                    "chunk_index": metadata.get("chunk_index"),
                })
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Build result
        result = GenerationResult(
            answer=answer,
            model=response.model,
            tokens_used=tokens_used,
            sources=sources,
            latency_ms=latency_ms,
        )
        
        # Log summary
        print(f"\n{'='*60}")
        print(f"🤖 LLM Generation Complete")
        print(f"{'='*60}")
        print(f"   Model: {response.model}")
        print(f"   Tokens: {tokens_used['total_tokens']} (prompt: {tokens_used['prompt_tokens']}, completion: {tokens_used['completion_tokens']})")
        print(f"   Sources: {len(sources)}")
        print(f"   Latency: {latency_ms:.0f}ms")
        print(f"{'='*60}")
        
        logger.info(f"Generated answer: {tokens_used['total_tokens']} tokens in {latency_ms:.0f}ms")
        
        return result.to_dict()
    
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        raise


def generate_streaming(
    question: str,
    chunks: Sequence[Union[Chunk, dict[str, Any]]],
    model: str = "glm-4-flash",
) -> Generator[str, None, None]:
    """
    Generate answer with streaming output.
    
    This generator function yields each token as it's generated,
    making it compatible with Streamlit's st.write_stream() for
    real-time response display.
    
    Args:
        question: User's question
        chunks: Retrieved context chunks
        model: GLM model to use (default: glm-4-flash)
    
    Yields:
        Each token/chunk of the generated answer as it arrives
    
    Example (Streamlit):
        >>> # In Streamlit app
        >>> response = generate_streaming(question, chunks)
        >>> st.write_stream(response)  # Real-time streaming display
    
    Example (Manual iteration):
        >>> for token in generate_streaming(question, chunks):
        ...     print(token, end="", flush=True)
    """
    # Build prompts
    system_prompt, user_message = build_prompt(question, chunks)
    
    # Get API key
    settings = get_settings()
    api_key = settings.get_api_key()
    
    # Initialize ZhipuAI client
    try:
        from zhipuai import ZhipuAI
    except ImportError:
        raise ImportError(
            "zhipuai is required. Install with: pip install zhipuai"
        )
    
    client = ZhipuAI(api_key=api_key)
    
    # Build messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    
    logger.info(f"Streaming answer with {model}")
    
    try:
        # Call API with streaming enabled
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.1,
            max_tokens=1024,
            top_p=0.9,
            stream=True,  # Enable streaming
        )
        
        # Yield each token as it arrives
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    except Exception as e:
        logger.error(f"Streaming generation failed: {e}")
        raise


# ============================================
# Additional Helper Classes (Backward Compatibility)
# ============================================

class MessageRole:
    """Message role enum for backward compatibility."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    """Chat message for backward compatibility."""
    role: str
    content: str
    
    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class ChatResponse:
    """LLM chat response for backward compatibility."""
    content: str
    model: str
    usage: dict[str, int] = field(default_factory=dict)
    finish_reason: str = "stop"


class ZhipuAILLM:
    """
    ZhipuAI LLM wrapper for backward compatibility.
    
    Supports GLM models:
    - glm-4: Most capable model
    - glm-4-flash: Faster, cheaper (recommended for RAG)
    - glm-4-plus: Enhanced reasoning
    - glm-4-air: Lightweight model
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        top_p: float = 0.9,
    ):
        settings = get_settings()
        self.model = model or settings.llm_model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self._client = None
    
    @property
    def client(self):
        """Lazy load ZhipuAI client."""
        if self._client is None:
            try:
                from zhipuai import ZhipuAI
            except ImportError:
                raise ImportError("zhipuai required: pip install zhipuai")
            
            settings = get_settings()
            api_key = self.api_key or settings.get_api_key()
            self._client = ZhipuAI(api_key=api_key)
            logger.info(f"ZhipuAI client initialized with model: {self.model}")
        
        return self._client
    
    def generate(
        self,
        messages: Sequence[Message],
        **kwargs,
    ) -> ChatResponse:
        """Generate response from messages."""
        api_messages = [msg.to_dict() for msg in messages]
        
        params = {
            "model": kwargs.get("model", self.model),
            "messages": api_messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "top_p": kwargs.get("top_p", self.top_p),
        }
        
        response = self.client.chat.completions.create(**params)
        
        return ChatResponse(
            content=response.choices[0].message.content,
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            finish_reason=response.choices[0].finish_reason,
        )
    
    def stream(
        self,
        messages: Sequence[Message],
        **kwargs,
    ) -> Generator[str, None, None]:
        """Stream response from messages."""
        api_messages = [msg.to_dict() for msg in messages]
        
        params = {
            "model": kwargs.get("model", self.model),
            "messages": api_messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "top_p": kwargs.get("top_p", self.top_p),
            "stream": True,
        }
        
        response = self.client.chat.completions.create(**params)
        
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class RAGPrompts:
    """RAG prompt templates for backward compatibility."""
    
    DEFAULT_SYSTEM = SYSTEM_PROMPT
    
    CHINESE_SYSTEM = """你是一个专业的AI助手，请仅基于提供的上下文信息回答问题。

关键规则：
1. 只使用上下文中的信息回答
2. 如果上下文中没有答案，请明确说明
3. 必须引用来源：[来源: 文件名]
4. 用与问题相同的语言回答
5. 如果信息超过2条，使用列表格式"""
    
    ENGLISH_SYSTEM = SYSTEM_PROMPT
    
    @classmethod
    def get_rag_prompt(
        cls,
        context: str,
        query: str,
        language: str = "chinese",
    ) -> tuple[str, str]:
        """Get system and user prompts for RAG."""
        if language.lower() == "chinese":
            system_prompt = cls.CHINESE_SYSTEM
            user_prompt = f"""参考资料：
{context}

问题：{query}

请基于以上资料回答，并注明来源。"""
        else:
            system_prompt = cls.ENGLISH_SYSTEM
            user_prompt = f"""Context:
{context}

Question: {query}

Please answer based on the context and cite your sources."""
        
        return system_prompt, user_prompt


class RAGGenerator:
    """RAG generator combining retrieval and LLM for backward compatibility."""
    
    def __init__(
        self,
        llm: Optional[ZhipuAILLM] = None,
        language: str = "chinese",
    ):
        self.llm = llm or ZhipuAILLM()
        self.language = language
    
    def generate_answer(
        self,
        query: str,
        context: str,
        sources: Optional[list] = None,
        stream: bool = False,
    ) -> Union[str, Generator[str, None, None]]:
        """Generate answer based on context."""
        system_prompt, user_prompt = RAGPrompts.get_rag_prompt(
            context=context,
            query=query,
            language=self.language,
        )
        
        messages = [
            Message(role=MessageRole.SYSTEM, content=system_prompt),
            Message(role=MessageRole.USER, content=user_prompt),
        ]
        
        if stream:
            return self.llm.stream(messages)
        else:
            response = self.llm.generate(messages)
            return response.content


def get_llm(provider: str = "zhipuai", **kwargs) -> ZhipuAILLM:
    """Get LLM instance by provider name."""
    if provider == "zhipuai":
        return ZhipuAILLM(**kwargs)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("LLM Module - GLM API Integration")
    print("=" * 60)
    
    # Test prompt building
    test_chunks = [
        Chunk(
            content="RAG (Retrieval-Augmented Generation) is a technique that combines retrieval with generation.",
            metadata={"source": "rag_intro.pdf", "page": 1},
        ),
        Chunk(
            content="Vector databases store embeddings for fast similarity search.",
            metadata={"source": "vector_db.pdf", "page": 5},
        ),
    ]
    
    system, user = build_prompt("What is RAG?", test_chunks)
    
    print("\n--- System Prompt (first 200 chars) ---")
    print(system[:200] + "...")
    
    print("\n--- User Message ---")
    print(user)
    
    print("\n--- To generate answer ---")
    print("result = generate_answer('What is RAG?', test_chunks)")
    print("print(result['answer'])")
    
    print("\n--- For streaming (Streamlit) ---")
    print("for token in generate_streaming(question, chunks):")
    print("    st.write(token)")
