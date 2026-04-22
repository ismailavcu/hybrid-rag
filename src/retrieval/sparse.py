"""
sparse (lexical) retrieval system using the BM25 ranking function
"""

from rank_bm25 import BM25Okapi

class BM25Retriever:
    def __init__(self, documents):
        self.tokenized_docs = [doc.split() for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_docs)  # BM25Okapi builds an internal representation based on term frequencies, document frequencies, and document lengths
        self.documents = documents

    def search(self, query, top_k=5):
        tokenized_query = query.split()   #  the query is tokenized the same way and ..
        scores = self.bm25.get_scores(tokenized_query)   # .. passed to get_scores, which computes a BM25 relevance score for every document independently

        ranked = sorted(  # sorts all pairs in descending order of score, and returns the top_k results
            list(zip(self.documents, scores)),
            key=lambda x: x[1],
            reverse=True
        )

        return ranked[:top_k]
    

if __name__ == "__main__":
    # python -m src.retrieval.sparse

    from src.ingestion.chunker import chunk_text
    from src.ingestion.pdf_loader import load_pdf

    file_path = r"data/raw/fastapi-contrib-readthedocs-io-en-latest.pdf"
    texts = load_pdf(file_path)
    chunks = chunk_text(texts)


    bm25 = BM25Retriever(chunks)

    results = bm25.search("fastapi start server")

    print("type of results: ", type(results))

    for r in results:
        print("- ", r[::][:300])
        print()