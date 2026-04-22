"""
This file implements a dense (semantic) retrieval system using vector embeddings and a similarity search index.

"""


from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class DenseRetriever:
    def __init__(self, documents):
        self.model = SentenceTransformer("intfloat/e5-base") 
        self.documents = documents

        self.embeddings = self.model.encode(documents, show_progress_bar=True)  # the model intfloat/e5-base converts each document chunk into a high-dimensional vector embedding, where semantic meaning (not just keywords) is encoded numerically.
        
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(self.embeddings)) # FAISS index (IndexFlatL2) is created, which stores all vectors and enables efficient nearest-neighbor search using L2 distance (Euclidean distance).

    def search(self, query, top_k=5):
        query_vec = self.model.encode([query]) # In search, the query is also encoded into the same embedding space, ensuring compatibility with document vectors.
        distances, indices = self.index.search(query_vec, top_k) # FAISS computes distances between the query vector and all stored document vectors, returning the indices of the closest matches.

        return [(self.documents[i], distances[0][idx]) for idx, i in enumerate(indices[0])] # Smaller distance = higher semantic similarity. This row maps these indices back to the original documents and returns (document, distance) pairs for the top_k results.
    
if __name__ == "__main__":
    # python -m src.retrieval.dense

    from src.ingestion.chunker import chunk_text
    from src.ingestion.pdf_loader import load_pdf

    file_path = r"data/raw/fastapi-contrib-readthedocs-io-en-latest.pdf"
    texts = load_pdf(file_path)
    chunks = chunk_text(texts)

    dense = DenseRetriever(chunks)
    results = dense.search("fastapi start server")

    print("type of results: ", type(results))
    for r in results:
        print(r[::][:300])
        print()