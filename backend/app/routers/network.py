"""
Network router – graph construction and influence analysis.
"""

from fastapi import APIRouter, Query
from ..services.network import build_network

router = APIRouter(prefix="/api", tags=["network"])


@router.get("/network")
async def get_network(
    graph_type: str = Query("account", description="account, domain, or hashtag"),
    q: str = Query(None, description="Filter posts by keyword"),
    remove_top: bool = Query(False, description="Remove highest PageRank node for resilience test"),
    max_nodes: int = Query(150, ge=10, le=500),
):
    """
    Build a network graph with influence metrics.
    Supports resilience testing by removing the top influencer node.
    """
    return build_network(
        graph_type=graph_type,
        query=q,
        remove_top_node=remove_top,
        max_nodes=max_nodes,
    )
