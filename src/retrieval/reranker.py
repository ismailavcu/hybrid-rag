"""
About reranker.py file:
More precise filtering step applied after fusion.
Instead of relying on rough similarity scores, we pass the query and each candidate document into
a cross-encoder model (like a transformer), which evaluates them together and assigns a more accurate relevance score.
This step is slower but much smarter, so we only apply it to the top ~10–20 results, and it significantly improves
the final top results by fixing mistakes made in earlier retrieval stages.
"""


from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class Reranker:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-base") #
        self.model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-base") # specifically trained as a cross-encoder. behaves like a relevance scoring function

    def rerank(self, query, docs, top_k=5):
        pairs = [(query, doc) for doc, _ in docs] # cross-encoder processes the (query – candidate document) pair jointly.

        inputs = self.tokenizer(  # The tokenizer encodes these pairs into a single input sequence per pair (PyTorch tensors)
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        with torch.no_grad():
            scores = self.model(**inputs).logits.squeeze() # The model performs a forward pass and outputs a single relevance logit per pair, representing how well that document answers the query.
            # The logits are treated as relevance scores: higher = more relevant

        # We map these scores back to the original documents, sort them in descending order, and return the top_k results
        scored_docs = list(zip([doc for doc, _ in docs], scores.tolist()))
        ranked = sorted(scored_docs, key=lambda x: x[1], reverse=True)

        return ranked[:top_k]