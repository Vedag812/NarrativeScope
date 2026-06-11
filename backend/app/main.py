"""
NarrativeScope – FastAPI application entry point.

On startup we load the dataset and precompute embeddings so the first
request doesn't take forever. Everything after that is served from memory.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from .core.config import ALLOWED_ORIGINS
from .core.data_loader import load_posts
from .services.embeddings import compute_embeddings
from .routers import search, timeseries, topics, network, dashboard

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up the data and embeddings before accepting requests."""
    logger.info("Starting up NarrativeScope backend...")
    posts = load_posts()
    logger.info(f"Loaded {len(posts)} posts")
    compute_embeddings(posts)
    logger.info("Embeddings ready – accepting requests")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="NarrativeScope API",
    description="Investigative dashboard API for analyzing political narratives on Reddit",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS – permit the frontend to talk to us
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# compress responses – the network graphs can be chunky
app.add_middleware(GZipMiddleware, minimum_size=1000)

# mount routers
app.include_router(search.router)
app.include_router(timeseries.router)
app.include_router(topics.router)
app.include_router(network.router)
app.include_router(dashboard.router)


@app.get("/")
async def root():
    return {
        "name": "NarrativeScope API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    from .core.data_loader import get_posts
    posts = get_posts()
    return {
        "status": "healthy",
        "posts_loaded": len(posts),
    }
