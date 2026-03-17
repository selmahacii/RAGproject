"""
Professional RAG Interface (Light/Clarity Edition)
A minimalist, high-fidelity light theme for professional deployment.
"""

import streamlit as st
import os
import tempfile
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion import load_pdfs
from src.chunking import recursive_chunk
from src.embeddings import build_vectorstore
from src.retrieval import retrieve_and_rerank
from src.llm import generate_streaming

# ============================================
# 1. Professional Page Configuration
# ============================================
st.set_page_config(
    page_title="RAG Intelligence Center",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Clarity" Light Theme Logic
st.markdown("""
<style>
    /* Premium Typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #ffffff;
    }

    /* Minimalist Sidebar */
    [data-testid="stSidebar"] {
        background-color: #fcfcfd;
        border-right: 1px solid #f0f1f3;
        padding-top: 2rem;
    }

    /* Elegant Chat Container */
    .stChatMessage {
        background-color: transparent !important;
        border-bottom: 1px solid #f3f4f6;
        padding: 2.5rem 0 !important;
        max-width: 850px;
        margin: 0 auto;
    }
    
    .stChatMessage[data-testid="stChatMessageUser"] {
        background-color: #f9fafb !important;
        border-radius: 12px;
        border: none;
        padding: 1.5rem !important;
        margin-bottom: 1rem;
    }

    /* Custom Header */
    .app-title {
        font-size: 2rem;
        font-weight: 700;
        color: #111827;
        letter-spacing: -0.025em;
        margin-bottom: 0.2rem;
    }
    .app-subtitle {
        font-size: 1rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }

    /* Source Cards (Minimalist) */
    .source-pill {
        display: inline-block;
        background-color: #f3f4f6;
        color: #374151;
        font-size: 0.75rem;
        font-weight: 500;
        padding: 4px 10px;
        border-radius: 6px;
        margin-right: 8px;
        margin-bottom: 8px;
        border: 1px solid #e5e7eb;
    }
    
    .source-detail {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 1rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    
    /* Input Styling */
    .stChatInputContainer {
        border-top: 1px solid #f3f4f6 !important;
        padding-top: 1rem !important;
        background-color: #ffffff !important;
    }
    
    /* Utility Buttons */
    .stButton>button {
        border-radius: 6px;
        font-size: 0.875rem;
        text-transform: none;
        border: 1px solid #e5e7eb;
        background-color: #ffffff;
        color: #374151;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        border-color: #3b82f6;
        color: #3b82f6;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# 2. State & Session Management
# ============================================
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# ============================================
# 3. Knowledge Dashboard (Sidebar)
# ============================================
with st.sidebar:
    st.markdown("<div style='font-size:1.2rem; font-weight:700;'>Data Workspace</div>", unsafe_allow_html=True)
    st.markdown("<div style='color:#6b7280; font-size:0.85rem; margin-bottom:1.5rem;'>Manage your knowledge repository</div>", unsafe_allow_html=True)
    
    # Upload Zone
    files = st.file_uploader("Upload reference documents", type="pdf", accept_multiple_files=True, label_visibility="collapsed")
    
    if st.button("Index Repository", type="primary", use_container_width=True):
        if files:
            with st.status("Analyzing Repository...", expanded=False) as status:
                with tempfile.TemporaryDirectory() as d:
                    for f in files:
                        with open(os.path.join(d, f.name), "wb") as out:
                            out.write(f.getbuffer())
                    
                    # RAG Pipeline Call
                    docs = load_pdfs(d)
                    chunks = recursive_chunk(docs)
                    st.session_state.vectorstore = build_vectorstore(chunks)
                status.update(label=f"Successfully indexed {len(chunks)} nodes.", state="complete")
        else:
            st.warning("No files detected.")

    st.markdown("---")
    
    # System Telemetry
    st.caption("SYSTEM STATUS")
    if st.session_state.vectorstore:
        st.success("Database Active")
        st.metric("Contextual Nodes", st.session_state.vectorstore._collection.count())
    else:
        st.info("Database Idle")

    # Global Controls
    st.markdown("---")
    res_depth = st.select_slider("Search Depth", options=[3, 5, 7, 10], value=5)
    if st.button("Purge System State", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# ============================================
# 4. Expert Chat Interface (Main Area)
# ============================================
st.markdown("<div class='app-title'>RAG Intelligence Assistant</div>", unsafe_allow_html=True)
st.markdown("<div class='app-subtitle'>High-precision context retrieval & language generation.</div>", unsafe_allow_html=True)

# Empty State Logic
if not st.session_state.messages:
    st.markdown("""
    <div style='padding: 3rem; background-color: #f9fafb; border-radius: 16px; border: 1px dashed #e5e7eb; text-align: center;'>
        <div style='font-size: 1.5rem; margin-bottom: 1rem;'>👋 Welcome to the Intelligence Center</div>
        <div style='color: #6b7280; font-size: 0.95rem; max-width: 500px; margin: 0 auto;'>
            To begin, upload your documents in the left workspace. 
            The AI will use these documents as its exclusive source of truth.
        </div>
    </div>
    """, unsafe_allow_html=True)

# Render Chat Thread
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            cols = st.columns([0.1, 0.9])
            with cols[1]:
                unique_sources = list(set([s["file"] for s in msg["sources"]]))
                for s_name in unique_sources:
                    st.markdown(f"<span class='source-pill'>Source: {s_name}</span>", unsafe_allow_html=True)
                with st.expander("Review Verified Evidence"):
                    for i, s in enumerate(msg["sources"], 1):
                        st.markdown(f"""
                        <div class='source-detail'>
                            <div style='font-weight:600; font-size:0.8rem; color:#3b82f6; margin-bottom:5px;'>EVIDENCE NODE #{i}</div>
                            <div style='font-size:0.9rem; color:#374151;'>{s['text'][:500]}...</div>
                        </div>
                        """, unsafe_allow_html=True)

# Query Execution
if prompt := st.chat_input("Enter your command or question..."):
    # Log User Intent
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Agent Processing
    if st.session_state.vectorstore:
        with st.chat_message("assistant"):
            # 1. Retrieval
            with st.spinner("Extracting contextual evidence..."):
                raw_results = retrieve_and_rerank(st.session_state.vectorstore, prompt, k=res_depth)
                evidence = [{"file": r["metadata"].get("source_file", "Unknown"), "text": r["content"]} for r in raw_results]
            
            # 2. Generation & Streaming
            full_response = st.write_stream(generate_streaming(prompt, raw_results))
            
            # 3. Evidence Display
            unique_files = list(set([e["file"] for e in evidence]))
            for f_name in unique_files:
                st.markdown(f"<span class='source-pill'>Source: {f_name}</span>", unsafe_allow_html=True)
            
            with st.expander("Review Verified Evidence"):
                for i, e in enumerate(evidence, 1):
                    st.markdown(f"""
                    <div class='source-detail'>
                        <div style='font-weight:600; font-size:0.8rem; color:#3b82f6; margin-bottom:5px;'>EVIDENCE NODE #{i}</div>
                        <div style='font-size:0.9rem; color:#374151;'>{e['text'][:500]}...</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.session_state.messages.append({
                "role": "assistant", 
                "content": full_response, 
                "sources": evidence
            })
    else:
        st.info("System Ready. Please provide contextual data (PDFs) to enable retrieval mode.")

# Footer Section
st.markdown("<div style='margin-top:5rem; border-top:1px solid #f3f4f6; padding-top:2rem; text-align:center;'>", unsafe_allow_html=True)
st.caption("Secure Intelligence Center • v1.0 • Built for Antigravity")
st.markdown("</div>", unsafe_allow_html=True)
