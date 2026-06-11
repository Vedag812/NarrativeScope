"""
App-wide settings. Reads from .env so we can swap API keys
between local dev and HuggingFace Spaces without touching code.
"""

import os
from pathlib import Path

# paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent   # backend/
DATA_DIR = BASE_DIR / "data"
DATA_FILE = DATA_DIR / "data.jsonl"
CACHE_DIR = DATA_DIR / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

# gemini config – supports comma-separated keys for rotation
_raw_keys = os.getenv("GEMINI_API_KEYS", os.getenv("GEMINI_API_KEY", ""))
GEMINI_API_KEYS: list[str] = [k.strip() for k in _raw_keys.split(",") if k.strip()]

# CORS – add your vercel domain here when deploying
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://localhost:3000"
).split(",")

# embedding model – small and fast, runs fine on a free CPU instance
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# sensible defaults for clustering
DEFAULT_N_CLUSTERS = 8
MAX_CLUSTERS = 50   # hard cap so users can't OOM us
MIN_CLUSTERS = 2
