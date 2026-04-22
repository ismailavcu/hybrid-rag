"""
About fusion.py file:
Combine results from two different retrieval systems—BM25 (keyword-based) and dense embeddings (semantic).
Each returns its own relevance scores, but those scores aren’t directly comparable, so we first normalize them,
then compute a weighted sum (e.g., 60% dense, 40% BM25). This creates a single unified ranking that benefits from
both exact keyword matching (useful for things like function names) and semantic understanding (useful for vague or conceptual queries).
BM25 is strong for exact matches (e.g., FastAPI, function names)
Dense is strong for meaning (e.g., “start server” ≈ “run app”)
"""


def normalize(scores): # Since their raw scores are on different scales (BM25 arbitrary positive values, FAISS L2 distances where lower is better), we first apply min–max normalization to each score list so they lie in [0,1].
    min_s = min(scores) 
    max_s = max(scores)
    return [(s - min_s) / (max_s - min_s + 1e-8) for s in scores]

def fuse(bm25_results, dense_results, alpha=0.6):
    bm25_docs, bm25_scores = zip(*bm25_results) # unpack
    dense_docs, dense_scores = zip(*dense_results) # unpack

    # normalize both score sets
    bm25_scores = normalize(bm25_scores)
    dense_scores = normalize(dense_scores)

    
    score_dict = {} # each document accumulates a final score in score_dict

    # weighted linear combination:
    # Look up the current score of this document in score_dict
    # If it doesn’t exist yet, start from 0 -> get(doc, 0)
    # Add the BM25 contribution, scaled by (1 - alpha)
    # Store the updated value back in the dictionary
    # final = 0.4*BM25 score + 0.6*Dense score (if alpha=0.6)
    for doc, score in zip(bm25_docs, bm25_scores):
        score_dict[doc] = score_dict.get(doc, 0) + (1 - alpha) * score

    for doc, score in zip(dense_docs, dense_scores):
        score_dict[doc] = score_dict.get(doc, 0) + alpha * score

    ranked = sorted(score_dict.items(), key=lambda x: x[1], reverse=True) # Finally, we sort all documents by the fused score and return the top results.

    return ranked[:10]