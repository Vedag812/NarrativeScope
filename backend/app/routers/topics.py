"""
Topics router – clustering and TF Projector export.
"""

from fastapi import APIRouter, Query
from fastapi.responses import PlainTextResponse
from ..services.topics import cluster_posts, export_projector_data
from ..services.genai import generate_topic_labels

router = APIRouter(prefix="/api", tags=["topics"])


@router.get("/topics")
async def get_topics(
    n_clusters: int = Query(8, ge=1, le=200, description="Number of topic clusters"),
):
    """
    Cluster posts by topic. The cluster count is tunable –
    we handle extreme values by clamping without crashing.
    """
    data = cluster_posts(n_clusters=n_clusters)

    # try to get AI-generated labels
    try:
        data["clusters"] = await generate_topic_labels(data["clusters"])
    except Exception:
        pass  # we already have TF-IDF labels as fallback

    return data


@router.get("/topics/projector/vectors.tsv")
async def get_projector_vectors():
    """TSV file of embedding vectors for TensorFlow Projector."""
    data = export_projector_data()
    return PlainTextResponse(data["vectors_tsv"], media_type="text/tab-separated-values")


@router.get("/topics/projector/metadata.tsv")
async def get_projector_metadata():
    """TSV file of post metadata for TensorFlow Projector."""
    data = export_projector_data()
    return PlainTextResponse(data["metadata_tsv"], media_type="text/tab-separated-values")
