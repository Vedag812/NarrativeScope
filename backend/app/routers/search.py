"""
Search router – semantic search + chatbot endpoint.
"""

from fastapi import APIRouter, Query
from ..services.search import search
from ..services.genai import suggest_related_queries

router = APIRouter(prefix="/api", tags=["search"])


@router.get("/search")
async def search_posts(
    q: str = Query("", description="Search query (semantic, not keyword-only)"),
    top_k: int = Query(20, ge=1, le=100),
    subreddit: str = Query(None, description="Filter by subreddit"),
):
    """
    Semantic search over posts. Handles empty, short, and
    non-English queries without crashing.
    """
    results = search(q, top_k=top_k, subreddit=subreddit)

    # suggest follow-up queries if we have results
    related = []
    if results["results"] and q.strip():
        try:
            related = await suggest_related_queries(q, results["results"])
        except Exception:
            related = []

    results["related_queries"] = related
    return results
