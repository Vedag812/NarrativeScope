"""
Embedding service – wraps sentence-transformers so the rest of the app
doesn't need to know about model loading, caching, or batching.

We compute embeddings once at startup and keep them in a numpy array.
For 8.8k posts with 384-dim vectors that's about 13 MB – totally fine.
"""

import logging
import pickle
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from ..core.config import EMBEDDING_MODEL, EMBEDDING_DIM, CACHE_DIR

logger = logging.getLogger(__name__)

_model: SentenceTransformer | None = None
_embeddings: np.ndarray | None = None
_post_ids: list[str] = []


def _cache_path() -> Path:
    return CACHE_DIR / "embeddings.pkl"


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def compute_embeddings(posts: list[dict], force: bool = False) -> np.ndarray:
    """
    Embed all posts. Results are cached to disk so restarts are fast.
    """
    global _embeddings, _post_ids

    cache = _cache_path()

    # try loading from cache first
    if not force and cache.exists():
        try:
            with open(cache, "rb") as f:
                cached = pickle.load(f)
            if cached["count"] == len(posts):
                _embeddings = cached["vectors"]
                _post_ids = cached["ids"]
                logger.info(f"Loaded {len(_post_ids)} embeddings from cache")
                return _embeddings
        except Exception as e:
            logger.warning(f"Cache load failed, recomputing: {e}")

    # compute fresh embeddings
    model = get_model()
    texts = [p["text"][:512] for p in posts]  # truncate long posts
    ids = [p["id"] for p in posts]

    logger.info(f"Computing embeddings for {len(texts)} posts...")
    vectors = model.encode(texts, show_progress_bar=True, batch_size=64)
    _embeddings = np.array(vectors, dtype=np.float32)
    _post_ids = ids

    # save to cache
    with open(cache, "wb") as f:
        pickle.dump({
            "vectors": _embeddings,
            "ids": _post_ids,
            "count": len(posts),
        }, f)

    logger.info(f"Embeddings computed and cached ({_embeddings.shape})")
    return _embeddings


def get_embeddings() -> tuple[np.ndarray, list[str]]:
    """Return the precomputed embeddings and their corresponding post IDs."""
    if _embeddings is None:
        raise RuntimeError("Embeddings not computed yet – call compute_embeddings first")
    return _embeddings, _post_ids


def embed_query(text: str) -> np.ndarray:
    """Embed a single search query. Returns a 1-D vector."""
    model = get_model()
    return model.encode(text, show_progress_bar=False)
