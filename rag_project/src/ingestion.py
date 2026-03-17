"""
Document Ingestion Module (Step 2)

Supports loading documents from:
- PDF (recursively)
- Web pages (with semantic tag filtering)
- CSV (converting rows to documents)
"""

import os
from pathlib import Path
from typing import Any, List, Optional
from loguru import logger
from tqdm import tqdm
import pandas as pd
from langchain_core.documents import Document as LCDocument

# Step 2 Implementation

def load_pdfs(folder: str) -> List[LCDocument]:
    """
    Recursively load all PDF files from a folder using PyPDFLoader.
    
    Args:
        folder: Path to the directory containing PDFs
    
    Returns:
        List of LangChain Document objects with enriched metadata
    """
    from langchain_community.document_loaders import PyPDFLoader
    
    folder_path = Path(folder)
    if not folder_path.exists():
        logger.error(f"Folder not found: {folder}")
        return []
    
    all_docs = []
    pdf_files = list(folder_path.rglob("*.pdf"))
    
    logger.info(f"Scanning {folder}... Found {len(pdf_files)} PDF files.")
    
    for pdf_path_raw in tqdm(pdf_files, desc="Loading PDFs"):
        pdf_path = Path(pdf_path_raw)
        try:
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()
            
            # Enrich metadata
            file_size = pdf_path.stat().st_size
            for page in pages:
                page.metadata.update({
                    "source_file": pdf_path.name,
                    "file_size": file_size,
                    "page_number": page.metadata.get("page", 0) + 1,
                    "source_type": "pdf"
                })
                all_docs.append(page)
                
            logger.info(f"Successfully loaded: {pdf_path.name}")
        except Exception as e:
            logger.error(f"Error loading {str(pdf_path)}: {e}")
            continue
            
    return all_docs


def load_web_pages(urls: List[str]) -> List[LCDocument]:
    """
    Load web pages using WebBaseLoader with BeautifulSoup filtering.
    
    Args:
        urls: List of URLs to scrape
        
    Returns:
        List of Document objects with cleaned content
    """
    from langchain_community.document_loaders import WebBaseLoader
    from bs4 import SoupStrainer
    
    # Only extract relevant semantic tags as requested in Step 2
    strainer = SoupStrainer(["article", "main", "content", "post-body"])
    
    all_docs = []
    for url in tqdm(urls, desc="Scraping Web"):
        try:
            loader = WebBaseLoader(
                web_path=url,
                bs_kwargs={"parse_only": strainer}
            )
            docs = loader.load()
            for doc in docs:
                doc.metadata.update({
                    "source_url": url,
                    "source_type": "web"
                })
                all_docs.append(doc)
            logger.info(f"Successfully scraped: {url}")
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            continue
            
    return all_docs


def load_csv(path: str, text_col: str, meta_cols: Optional[List[str]] = None) -> List[LCDocument]:
    """
    Load data from a CSV file using pandas and convert rows to Documents.
    
    Args:
        path: Path to the CSV file
        text_col: The column containing the document text
        meta_cols: Optional list of columns to include as metadata
        
    Returns:
        List of Document objects
    """
    if meta_cols is None:
        meta_cols = []
        
    try:
        df = pd.read_csv(path)
        docs = []
        
        for idx, row in df.iterrows():
            content = str(row[text_col])
            metadata = {col: row[col] for col in meta_cols if col in df.columns}
            metadata["source"] = f"{Path(path).name}_row_{idx}"
            metadata["source_type"] = "csv"
            
            docs.append(LCDocument(
                page_content=content,
                metadata=metadata
            ))
            
        logger.info(f"Loaded {len(docs)} rows from {path}")
        return docs
    except Exception as e:
        logger.error(f"Error loading CSV {path}: {e}")
        return []

if __name__ == "__main__":
    # Quick test
    print("Ingestion Module Loaded.")
