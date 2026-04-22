# Hybrid RAG  
This project implements an advanced Retrieval-Augmented Generation (RAG) system.  
Unlike basic RAG systems that rely only on vector search, this project combines:  
  
Dense retrieval (semantic search)  
Sparse retrieval (BM25 keyword search)  
Cross-encoder re-ranking  

# Architecture  

```text
User Query
   ↓
Query Rewriting
   ↓
 ┌───────────────┬────────────────┐
 │ Dense Search  │ Sparse Search  │
 │ (FAISS)       │ (BM25)         │
 └──────┬────────┴───────┬────────┘
        ↓                ↓
        → Score Fusion →
              ↓
        Re-Ranker (Cross Encoder)
              ↓
        Top-K Context
              ↓
        LLM Generator
              ↓
         Final Answer
```
  
# Tech Stack  

Python  
FAISS (vector similarity search)  
BM25 (rank_bm25)  
HuggingFace Transformers  
Sentence Transformers  
PyTorch  

# Key Components  

1. Sparse Retrieval (BM25)  
- Keyword-based retrieval  
- Strong for exact matches (e.g., function names, APIs)  
2. Dense Retrieval (Embeddings + FAISS)  
- Semantic search using embeddings  
- Captures meaning beyond exact words  
3. Score Fusion  
- Combines BM25 and dense scores:  
final_score = α * dense + (1 - α) * bm25  
Default: α = 0.6 (semantic > keyword)  
4. Re-Ranking (Cross-Encoder)  
- Uses a transformer model to score (query, document) pairs  
- Produces relevance ordering  
5. Query Rewriting  
- Expands/improves user queries to boost retrieval quality  
6. LLM Generation  
- Generates final answers using retrieved context  
- Constrained to avoid hallucinations  

# Project Structure  

hybrid-rag/  
│  
├── data/  
│   └── raw/                  # PDF documents  
│  
├── src/  
│   ├── ingestion/  
│   │   ├── __init__.py  
│   │   ├── pdf_loader.py  
│   │   └── chunker.py  
│   │  
│   ├── retrieval/  
│   │   ├── __init__.py  
│   │   ├── sparse.py         # BM25  
│   │   ├── dense.py          # FAISS  
│   │   ├── fusion.py         # Score fusion  
│   │   └── reranker.py       # Cross-encoder  
│   │  
│   ├── query/  
│   │   ├── __init__.py  
│   │   └── rewrite.py  
│   │  
│   └── rag/  
│       ├── __init__.py  
│       ├── pipeline.py  
│       └── llm.py  
│  
├── main.py  
├── requirements.txt  
└── README.md  

# How It Works  

Load and chunk PDF documents  
Build:  
BM25 index (sparse)  
FAISS index (dense)  
Process query:  
Rewrite query  
Retrieve from both systems  
Fuse scores  
Re-rank results  
Generate final answer using LLM  

# Example  
Query:  How to start a FastAPI server?  
  
System Behavior:  
  
BM25 → finds exact matches ("FastAPI", "server")  
Dense → finds semantic matches ("run app", "launch API")  
Re-ranker → selects most relevant context  
LLM → generates final answer  