"""
Semantic search – ranks posts by cosine similarity to a query embedding.
Also handles edge cases: empty queries, very short inputs, and
non-English text (MiniLM-L6-v2 supports 100+ languages, so this
usually just works without special handling).
"""

import numpy as np
from ..services.embeddings import get_embeddings, embed_query
from ..core.data_loader import get_posts


def _cosine_similarity(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    """Batch cosine similarity between one query and many documents."""
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    doc_norms = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-10)
    return doc_norms @ query_norm


def search(query: str, top_k: int = 20, subreddit: str = None) -> dict:
    """
    Semantic search over all posts. Returns ranked results with scores.

    Edge cases we handle here:
    - Empty query → friendly message, no crash
    - Very short query (1-2 chars) → still try, but flag it
    - Non-English → MiniLM handles it, we just pass it through
    """
    # guard: empty query
    if not query or not query.strip():
        return {
            "query": query,
            "results": [],
            "total": 0,
            "message": "Please enter a search query to find relevant posts.",
        }

    query = query.strip()
    is_short = len(query) <= 2

    # embed the query
    query_vec = embed_query(query)
    embeddings, post_ids = get_embeddings()
    posts = get_posts()

    # build a lookup from id to post
    post_map = {p["id"]: p for p in posts}

    # compute similarities
    scores = _cosine_similarity(query_vec, embeddings)

    # optional subreddit filter
    if subreddit:
        for i, pid in enumerate(post_ids):
            if post_map.get(pid, {}).get("subreddit", "").lower() != subreddit.lower():
                scores[i] = -1.0

    # rank and pick top_k
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        pid = post_ids[idx]
        post = post_map.get(pid)
        if post and scores[idx] > 0:
            results.append({
                **post,
                "relevance_score": round(float(scores[idx]), 4),
            })

    msg = None
    if is_short:
        msg = f"Your query \"{query}\" is very short. Results may be less precise."
    if not results:
        msg = f"No results found for \"{query}\". Try a different query or broader terms."

    return {
        "query": query,
        "results": results,
        "total": len(results),
        "message": msg,
    }
