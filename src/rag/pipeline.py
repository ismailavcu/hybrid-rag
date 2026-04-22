from ..query.rewrite import rewrite_query
from ..retrieval.fusion import fuse
from ..ingestion.pdf_loader import load_pdf
from ..ingestion.chunker import chunk_text
from ..retrieval.dense import DenseRetriever
from ..retrieval.sparse import BM25Retriever
from ..retrieval.reranker import Reranker
from .llm import llm

def rag_pipeline(query, PDF_PATH): #query = "how to use dependency injection in fastapi"

    docs = load_pdf(PDF_PATH)
    chunks = chunk_text(docs)

    bm25 = BM25Retriever(chunks)
    dense = DenseRetriever(chunks)
    reranker = Reranker()


    rewritten = rewrite_query(query)

    bm25_res = bm25.search(rewritten, top_k=10)
    dense_res = dense.search(rewritten, top_k=10)

    fused = fuse(bm25_res, dense_res)

    reranked = reranker.rerank(query, fused)

    top_docs = [doc for doc, _ in reranked[:5]]

    answer = llm(query, top_docs)

    return answer, top_docs