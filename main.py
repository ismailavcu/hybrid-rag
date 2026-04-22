from src.ingestion.pdf_loader import load_pdf
from src.ingestion.chunker import chunk_text
from src.retrieval.sparse import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.reranker import Reranker
from src.rag.pipeline import rag_pipeline

PDF_PATH = "data/raw/fastapi-contrib-readthedocs-io-en-latest.pdf"

if __name__ == "__main__":

    query = "how to use dependency injection in fastapi"

    answer, docs = rag_pipeline(query, PDF_PATH)

    print("\n=== ANSWER ===\n")
    print(answer)

    print("\n=== SOURCES ===\n")
    for d in docs:
        print("-", d[:500])


