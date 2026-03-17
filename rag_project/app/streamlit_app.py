"""
RAG Demo - Streamlit Chat Interface

A clean, modern chat interface for RAG:
- Multi-PDF upload and indexing
- Real-time streaming responses
- Source attribution with expanders
- Session state management

Run with:
    streamlit run app/streamlit_app.py
"""

import os
import sys
import time
import tempfile
from pathlib import Path
from typing import Any, Generator, Optional

import streamlit as st

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_settings


# ============================================
# Page Configuration (MUST BE FIRST)
# ============================================

st.set_page_config(
    page_title="RAG Demo",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================
# Custom CSS
# ============================================

st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
    }
    .source-card {
        background-color: #f8f9fa;
        border-left: 3px solid #4CAF50;
        padding: 0.5rem 1rem;
        margin: 0.5rem 0;
        border-radius: 0 5px 5px 0;
    }
    .indexing-status {
        padding: 1rem;
        background-color: #e3f2fd;
        border-radius: 10px;
        margin: 1rem 0;
    }
    div[data-testid="stSidebar"] {
        background-color: #fafafa;
    }
</style>
""", unsafe_allow_html=True)


# ============================================
# Session State Management
# ============================================

def init_session_state():
    """Initialize session state variables."""
    # Core state
    if "messages" not in st.session_state:
        st.session_state.messages = []  # Chat history: [{"role": "user/assistant", "content": "..."}]
    
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None  # ChromaDB vector store
    
    if "indexed_chunks" not in st.session_state:
        st.session_state.indexed_chunks = 0  # Number of indexed chunks
    
    if "indexed_files" not in st.session_state:
        st.session_state.indexed_files = []  # List of indexed file names
    
    if "embeddings" not in st.session_state:
        st.session_state.embeddings = None  # Embedding model
    
    if "llm_client" not in st.session_state:
        st.session_state.llm_client = None  # LLM client


# ============================================
# Lazy Loading Components
# ============================================

@st.cache_resource
def get_embeddings():
    """Get or create embedding model (cached)."""
    try:
        from src.embeddings import SentenceTransformerEmbeddings
        model = SentenceTransformerEmbeddings(model_name="BAAI/bge-m3")
        return model
    except Exception as e:
        st.error(f"Failed to load embedding model: {e}")
        return None


@st.cache_resource
def get_llm_client():
    """Get or create LLM client (cached)."""
    try:
        from zhipuai import ZhipuAI
        settings = get_settings()
        api_key = settings.get_api_key()
        return ZhipuAI(api_key=api_key)
    except Exception as e:
        st.error(f"Failed to initialize LLM: {e}")
        return None


# ============================================
# Document Processing
# ============================================

def process_pdfs(uploaded_files: list) -> tuple[int, list[str]]:
    """
    Process uploaded PDF files and index them.
    
    Returns:
        Tuple of (number of chunks, list of file names)
    """
    from src.ingestion import Document
    from src.chunking import recursive_chunk
    from src.embeddings import build_vectorstore
    
    all_chunks = []
    file_names = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"Processing: {uploaded_file.name}")
        
        try:
            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name
            
            # Load PDF
            from src.ingestion import load_pdfs
            docs = load_pdfs(Path(tmp_path).parent)
            
            # Filter to only this file
            docs = [d for d in docs if uploaded_file.name in d.metadata.get("source", "")]
            
            if not docs:
                # If no docs found, try loading directly
                from langchain_community.document_loaders import PyPDFLoader
                loader = PyPDFLoader(tmp_path)
                pages = loader.load()
                docs = [
                    Document(
                        content=page.page_content,
                        metadata={**page.metadata, "source": uploaded_file.name}
                    )
                    for page in pages
                ]
            
            # Chunk documents
            chunks = recursive_chunk(docs, chunk_size=512, overlap=64)
            all_chunks.extend(chunks)
            file_names.append(uploaded_file.name)
            
            # Clean up temp file
            os.unlink(tmp_path)
            
        except Exception as e:
            st.warning(f"Error processing {uploaded_file.name}: {e}")
        
        progress_bar.progress((i + 1) / len(uploaded_files))
    
    status_text.text("Building vector index...")
    
    # Build or update vectorstore
    if all_chunks:
        embeddings = get_embeddings()
        if embeddings is None:
            return 0, []
        
        vectorstore = build_vectorstore(all_chunks, persist_dir="./data/chroma_db")
        st.session_state.vectorstore = vectorstore
        st.session_state.embeddings = embeddings
    
    progress_bar.progress(1.0)
    status_text.empty()
    
    return len(all_chunks), file_names


def retrieve_context(query: str, k: int = 5) -> list[dict]:
    """
    Retrieve relevant context for a query.
    
    Returns:
        List of dicts with 'content', 'score', 'metadata'
    """
    from src.retrieval import retrieve_and_rerank
    
    if st.session_state.vectorstore is None:
        return []
    
    try:
        # Use retrieve and rerank for better results
        results = retrieve_and_rerank(
            st.session_state.vectorstore,
            query,
            k=k,
            rerank_top=3,
        )
        return results
    except Exception as e:
        # Fallback to basic retrieval
        try:
            docs = st.session_state.vectorstore.similarity_search_with_score(query, k=k)
            return [
                {
                    "content": doc.page_content,
                    "score": float(1 - score),  # Convert distance to similarity
                    "metadata": doc.metadata,
                }
                for doc, score in docs
            ]
        except Exception as e2:
            st.error(f"Retrieval error: {e2}")
            return []


def generate_streaming_answer(
    query: str,
    contexts: list[dict],
) -> Generator[str, None, None]:
    """
    Generate streaming answer from LLM.
    
    Yields tokens as they are generated.
    """
    from src.llm import build_prompt
    
    if st.session_state.llm_client is None:
        st.session_state.llm_client = get_llm_client()
    
    if st.session_state.llm_client is None:
        yield "Error: LLM client not initialized. Please check your API key."
        return
    
    # Build prompt
    system_prompt, user_message = build_prompt(query, contexts)
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    
    try:
        response = st.session_state.llm_client.chat.completions.create(
            model="glm-4-flash",
            messages=messages,
            temperature=0.1,
            max_tokens=1024,
            top_p=0.9,
            stream=True,
        )
        
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    except Exception as e:
        yield f"Error generating answer: {e}"


# ============================================
# Sidebar
# ============================================

def render_sidebar():
    """Render the sidebar with document upload and indexing."""
    with st.sidebar:
        st.markdown("## 📄 Document Index")
        st.markdown("---")
        
        # File uploader for multiple PDFs
        uploaded_files = st.file_uploader(
            "Upload PDF documents",
            type=["pdf"],
            accept_multiple_files=True,
            help="Select one or more PDF files to index",
        )
        
        # Index button
        if st.button("📥 Index Documents", use_container_width=True, type="primary"):
            if not uploaded_files:
                st.warning("Please upload at least one PDF file.")
            else:
                # Show spinner during indexing
                with st.spinner("Indexing documents..."):
                    num_chunks, file_names = process_pdfs(uploaded_files)
                
                if num_chunks > 0:
                    st.session_state.indexed_chunks = num_chunks
                    st.session_state.indexed_files.extend(file_names)
                    st.success(f"✅ Indexed {num_chunks} chunks from {len(file_names)} files!")
                else:
                    st.error("Failed to index documents.")
        
        st.markdown("---")
        
        # Show indexed files
        st.markdown("### 📚 Indexed Files")
        if st.session_state.indexed_files:
            for fname in st.session_state.indexed_files:
                st.markdown(f"- {fname}")
            st.metric("Total Chunks", st.session_state.indexed_chunks)
        else:
            st.info("No documents indexed yet.")
        
        st.markdown("---")
        
        # Settings
        st.markdown("### ⚙️ Settings")
        
        retrieval_k = st.slider(
            "Retrieval count (k)",
            min_value=1,
            max_value=10,
            value=5,
            help="Number of documents to retrieve",
        )
        st.session_state.retrieval_k = retrieval_k
        
        # Clear button
        if st.button("🗑️ Clear Index", use_container_width=True):
            st.session_state.vectorstore = None
            st.session_state.indexed_chunks = 0
            st.session_state.indexed_files = []
            st.session_state.messages = []
            
            # Clear ChromaDB
            import shutil
            chroma_path = Path("./data/chroma_db")
            if chroma_path.exists():
                shutil.rmtree(chroma_path)
            
            st.success("Index cleared!")
            st.rerun()


# ============================================
# Main Chat Interface
# ============================================

def render_chat():
    """Render the main chat interface."""
    st.markdown("## 💬 Chat with your Documents")
    
    # Check if documents are indexed
    if st.session_state.vectorstore is None:
        st.info("👆 Please upload and index PDF documents from the sidebar to start chatting.")
        
        # Show demo prompt
        st.markdown("""
        ### How to use:
        1. Upload PDF files using the sidebar
        2. Click "Index Documents"
        3. Ask questions about your documents
        
        The RAG system will:
        - Retrieve relevant passages from your documents
        - Generate accurate answers with source citations
        - Stream responses in real-time
        """)
        return
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources if available
            if "sources" in message and message["sources"]:
                with st.expander("📚 Sources"):
                    for i, source in enumerate(message["sources"]):
                        st.markdown(f"""
                        <div class="source-card">
                            <strong>[{i+1}] {source.get('metadata', {}).get('source', 'Unknown')}</strong><br>
                            <small>Score: {source.get('score', 0):.3f}</small><br>
                            {source.get('content', '')[:200]}...
                        </div>
                        """, unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Add to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Get context
        with st.status("Retrieving context...", expanded=False) as status:
            contexts = retrieve_context(prompt, k=st.session_state.get("retrieval_k", 5))
            status.update(label=f"Retrieved {len(contexts)} relevant passages", state="complete")
        
        # Generate and stream answer
        with st.chat_message("assistant"):
            # Use st.write_stream for cleaner streaming (Streamlit 1.30+)
            try:
                # Try st.write_stream (recommended method)
                full_response = st.write_stream(generate_streaming_answer(prompt, contexts))
            except AttributeError:
                # Fallback for older Streamlit versions
                response_placeholder = st.empty()
                full_response = ""
                
                for token in generate_streaming_answer(prompt, contexts):
                    full_response += token
                    response_placeholder.markdown(full_response + "▌")
                
                response_placeholder.markdown(full_response)
            
            # Show sources in expander
            if contexts:
                with st.expander("📚 Sources"):
                    for i, source in enumerate(contexts):
                        source_name = source.get('metadata', {}).get('source', 'Unknown')
                        score = source.get('score', 0)
                        content = source.get('content', '')
                        
                        st.markdown(f"""
                        <div class="source-card">
                            <strong>[{i+1}] {source_name}</strong><br>
                            <small>Relevance Score: {score:.3f}</small>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        with st.container():
                            st.text(content[:500] + ("..." if len(content) > 500 else ""))
                        
                        if i < len(contexts) - 1:
                            st.markdown("---")
        
        # Add to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "sources": contexts,
        })


# ============================================
# Main Application
# ============================================

def main():
    """Main application entry point."""
    # Initialize session state
    init_session_state()
    
    # Initialize embeddings if not set
    if st.session_state.embeddings is None:
        st.session_state.embeddings = get_embeddings()
    
    # Header
    st.title("🔍 RAG Demo")
    st.markdown("Chat with your PDF documents using Retrieval-Augmented Generation")
    st.markdown("---")
    
    # Render sidebar
    render_sidebar()
    
    # Render main chat
    render_chat()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888; font-size: 0.9rem;">
        Powered by ZhipuAI GLM-4 | ChromaDB | FlashRank | BGE-M3 Embeddings
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
