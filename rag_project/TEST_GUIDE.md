# RAG Performance Test Guide

Use these resources to verify if your RAG pipeline is working correctly across different document types.

## 🧪 Phase 1: Local PDF Test (Factual Accuracy)
Download these technical papers and upload them to the **Control Terminal**:
1.  **[Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)** (Transformers Paper)
    *   **Question to ask**: "Explain the Scaled Dot-Product Attention formula."
    *   **What to check**: Ensure the assistant cites `transformer.pdf` and mentions $\sqrt{d_k}$.
2.  **[RAG for LLMs: A Survey](https://arxiv.org/pdf/2312.10997.pdf)**
    *   **Question to ask**: "What is the difference between Naive RAG and Advanced RAG?"
    *   **What to check**: Check if it mentions "Pre-retrieval" and "Post-retrieval" strategies.

---

## 🌐 Phase 2: Web Scraping Test (Real-time Scraping)
Add these URLs if you implement the `load_web_pages` feature in the UI later:
1.  **Python Documentation**: `https://docs.python.org/3/whatsnew/3.12.html`
    *   **Question**: "What are the core changes in Python 3.12 regarding type hinting?"

---

## 📊 Phase 3: CSV Data Test (Structured Data)
Create a `test.csv` with:
```csv
product_name,description,price
Antigravity-X1,Quantum processor with 128 qubits,5000
Lumina-S2,Holographic display 8K,1200
```
*   **Question**: "Which product is cheaper and by how much?"

---

## 📉 Phase 4: Stress Testing (Hallucination Check)
Ask a question that is **NOT** in your documents:
*   **Question**: "Who won the World Cup in 2030?"
*   **Target Response**: The assistant **MUST** say: "I am sorry, but the provided context does not contain information about the 2030 World Cup."
*   **If it answers**: Your `SYSTEM_PROMPT` grounding is too weak (Lower temperature to 0.1).

---

## 📈 Metric Verification (RAGAs Tab)
After asking 5-10 questions:
1.  Switch to the **Evaluation** tab.
2.  Check if **Faithfulness** is $> 0.85$.
3.  If **Context Recall** is low, increase **Retrieval Depth (k)** to 10.
