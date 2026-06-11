"""
GenAI service – uses Google Gemini for dynamic plain-language summaries.
Falls back to rule-based templates when the API is unavailable or rate-limited.

We rotate through multiple API keys if provided (comma-separated in env)
to avoid hitting free-tier limits too quickly.
"""

import logging
import random
from typing import Optional

from ..core.config import GEMINI_API_KEYS

logger = logging.getLogger(__name__)

_client = None
_key_index = 0


def _get_client():
    """Lazy-init the Gemini client, cycling through available keys."""
    global _client, _key_index

    if not GEMINI_API_KEYS:
        return None

    try:
        import google.generativeai as genai
        key = GEMINI_API_KEYS[_key_index % len(GEMINI_API_KEYS)]
        genai.configure(api_key=key)
        _client = genai.GenerativeModel("gemini-2.0-flash")
        return _client
    except Exception as e:
        logger.warning(f"Gemini client init failed: {e}")
        return None


def _rotate_key():
    """Move to the next API key after a failure."""
    global _key_index
    _key_index += 1


def _rule_based_timeseries_summary(series_data: list[dict], query: str = None) -> str:
    """
    Fallback summary when Gemini isn't available.
    Covers the basics: total posts, peak, trend direction.
    """
    if not series_data:
        return "No data available for this time period."

    total = sum(d["count"] for d in series_data)
    peak = max(series_data, key=lambda x: x["count"])
    first_half = sum(d["count"] for d in series_data[:len(series_data)//2])
    second_half = total - first_half

    trend = "increasing" if second_half > first_half * 1.1 else \
            "decreasing" if second_half < first_half * 0.9 else "relatively stable"

    topic = f" about '{query}'" if query else ""
    return (
        f"Over this period, there were {total:,} posts{topic}. "
        f"Activity peaked on {peak['date']} with {peak['count']} posts. "
        f"The overall trend is {trend}."
    )


async def generate_timeseries_summary(
    series_data: list[dict],
    query: str = None,
    subreddit: str = None,
) -> str:
    """
    Ask Gemini for a plain-language summary of a time-series.
    Falls back to rule-based if anything goes wrong.
    """
    fallback = _rule_based_timeseries_summary(series_data, query)

    client = _get_client()
    if not client:
        return fallback

    # build a compact data summary for the prompt
    total = sum(d["count"] for d in series_data)
    peak = max(series_data, key=lambda x: x["count"])
    date_range = f"{series_data[0]['date']} to {series_data[-1]['date']}"

    context = f"Total posts: {total}. Date range: {date_range}. Peak: {peak['date']} ({peak['count']} posts)."
    if query:
        context += f" Search filter: '{query}'."
    if subreddit:
        context += f" Subreddit: r/{subreddit}."

    # sample some data points to keep the prompt short
    sample_size = min(15, len(series_data))
    step = max(1, len(series_data) // sample_size)
    sampled = series_data[::step]
    data_str = ", ".join(f"{d['date']}: {d['count']}" for d in sampled)

    prompt = f"""You are a data analyst writing a brief summary for a non-technical reader.
Summarize this social media activity trend in 2-3 sentences. Be specific about dates and numbers.
Do NOT use markdown formatting. Write plain text only.

Context: {context}
Data points: {data_str}

Write a clear, concise summary:"""

    try:
        response = client.generate_content(prompt)
        text = response.text.strip()
        if text and len(text) > 20:
            return text
        return fallback
    except Exception as e:
        logger.warning(f"Gemini call failed, using fallback: {e}")
        _rotate_key()
        return fallback


async def generate_topic_labels(clusters: list[dict]) -> list[dict]:
    """
    Ask Gemini to generate human-readable names for topic clusters
    based on their keywords and representative posts.
    """
    client = _get_client()
    if not client:
        return clusters  # just use the TF-IDF labels we already have

    for cluster in clusters:
        keywords = ", ".join(cluster.get("keywords", [])[:5])
        titles = " | ".join(
            p["title"][:60] for p in cluster.get("representative_posts", [])[:3]
        )

        prompt = f"""Name this topic cluster in 3-5 words. Be specific and descriptive.

Keywords: {keywords}
Sample titles: {titles}

Respond with ONLY the topic name, nothing else:"""

        try:
            response = client.generate_content(prompt)
            label = response.text.strip().strip('"').strip("'")
            if label and len(label) < 60:
                cluster["ai_label"] = label
        except Exception as e:
            logger.warning(f"Gemini label generation failed: {e}")
            _rotate_key()

    return clusters


async def suggest_related_queries(query: str, results: list[dict]) -> list[str]:
    """
    After returning search results, suggest 2-3 follow-up queries.
    Nice-to-have for product thinking.
    """
    client = _get_client()
    if not client:
        return []

    titles = " | ".join(r.get("title", "")[:50] for r in results[:5])

    prompt = f"""A user searched for: "{query}"
They found posts about: {titles}

Suggest exactly 3 related search queries they might want to explore next.
Each query should be 3-6 words. Return one per line, no numbering or bullets:"""

    try:
        response = client.generate_content(prompt)
        lines = [l.strip() for l in response.text.strip().split("\n") if l.strip()]
        return lines[:3]
    except Exception as e:
        logger.warning(f"Related query suggestion failed: {e}")
        _rotate_key()
        return []
