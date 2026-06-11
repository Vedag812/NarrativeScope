"""
Loads the Reddit JSONL dump and normalises it into a clean list of dicts.
Skips malformed rows instead of crashing – we log a count at the end
so we know if something went wrong without blocking startup.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Optional

from .config import DATA_FILE

logger = logging.getLogger(__name__)

# we keep the normalised posts in memory – 8.8k records is ~15 MB tops
_posts: list[dict] = []


def _extract_hashtags(title: str, body: str) -> list[str]:
    """Pull anything that looks like a hashtag from text."""
    import re
    combined = f"{title} {body}"
    return re.findall(r"#(\w+)", combined)


def _extract_urls(text: str) -> list[str]:
    """Grab http(s) URLs from selftext."""
    import re
    return re.findall(r"https?://[^\s\)\"'>]+", text)


def _normalise(raw: dict) -> Optional[dict]:
    """
    Take a raw Reddit submission object and flatten it into
    the fields we actually care about. Returns None if the row
    is too broken to use.
    """
    d = raw.get("data", raw)

    # need at least an id and some text to work with
    post_id = d.get("id") or d.get("name")
    title = d.get("title", "")
    if not post_id or not title:
        return None

    body = d.get("selftext", "") or ""
    created_utc = d.get("created_utc", 0)

    try:
        created_dt = datetime.fromtimestamp(created_utc, tz=timezone.utc)
    except (ValueError, OSError):
        return None

    url = d.get("url", "")
    domain = d.get("domain", "")
    is_self = d.get("is_self", True)

    return {
        "id": post_id,
        "title": title,
        "body": body,
        "text": f"{title}. {body}".strip(),  # combined for embedding
        "author": d.get("author", "[deleted]"),
        "subreddit": d.get("subreddit", "unknown"),
        "score": d.get("score", 0),
        "upvote_ratio": d.get("upvote_ratio", 0.5),
        "num_comments": d.get("num_comments", 0),
        "created_utc": created_utc,
        "created_dt": created_dt.isoformat(),
        "created_date": created_dt.strftime("%Y-%m-%d"),
        "url": url,
        "domain": domain,
        "is_self": is_self,
        "permalink": d.get("permalink", ""),
        "hashtags": _extract_hashtags(title, body),
        "mentioned_urls": _extract_urls(body) if is_self else [url] if url else [],
        "over_18": d.get("over_18", False),
        "flair": d.get("link_flair_text") or "",
    }


def load_posts(force_reload: bool = False) -> list[dict]:
    """
    Load and cache posts in memory. Idempotent – calling it twice
    just returns the same list unless you pass force_reload.
    """
    global _posts
    if _posts and not force_reload:
        return _posts

    if not DATA_FILE.exists():
        logger.error(f"Dataset not found at {DATA_FILE}")
        return []

    loaded, skipped = [], 0
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
                post = _normalise(raw)
                if post:
                    loaded.append(post)
                else:
                    skipped += 1
            except json.JSONDecodeError:
                skipped += 1

    _posts = loaded
    logger.info(f"Loaded {len(loaded)} posts, skipped {skipped} malformed rows")
    return _posts


def get_posts() -> list[dict]:
    """Grab the cached posts – loads them first if needed."""
    if not _posts:
        load_posts()
    return _posts
