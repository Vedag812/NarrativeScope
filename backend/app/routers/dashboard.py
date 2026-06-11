"""
Dashboard router – overview stats and dataset summary.
"""

from collections import Counter
from fastapi import APIRouter
from ..core.data_loader import get_posts

router = APIRouter(prefix="/api", tags=["dashboard"])


# news source reliability tiers based on independent fact-check ratings
# this is the editorial angle that makes our dashboard tell a story
SOURCE_RELIABILITY = {
    "apnews.com": "high",
    "reuters.com": "high",
    "bbc.com": "high",
    "bbc.co.uk": "high",
    "npr.org": "high",
    "nytimes.com": "mainstream",
    "washingtonpost.com": "mainstream",
    "theguardian.com": "mainstream",
    "thehill.com": "mainstream",
    "politico.com": "mainstream",
    "nbcnews.com": "mainstream",
    "cnn.com": "mainstream",
    "abcnews.go.com": "mainstream",
    "cbsnews.com": "mainstream",
    "newsweek.com": "mainstream",
    "ft.com": "mainstream",
    "foxnews.com": "mixed",
    "nypost.com": "mixed",
    "dailymail.co.uk": "mixed",
    "breitbart.com": "low",
    "dailywire.com": "low",
    "townhall.com": "low",
    "redstate.com": "low",
    "thefederalist.com": "low",
    "infowars.com": "low",
    "gateway pundit.com": "low",
}


@router.get("/dashboard/overview")
async def dashboard_overview():
    """
    High-level dataset summary for the dashboard landing page.
    Includes stats, subreddit breakdown, source reliability analysis,
    and top contributors.
    """
    posts = get_posts()

    if not posts:
        return {"error": "No data loaded"}

    # basic stats
    total = len(posts)
    dates = [p["created_date"] for p in posts]
    authors = Counter(p["author"] for p in posts if p["author"] != "[deleted]")
    subreddits = Counter(p["subreddit"] for p in posts)

    # domain / source analysis
    domains = Counter()
    reliability_counts = Counter()
    source_by_subreddit = {}

    for p in posts:
        domain = p.get("domain", "")
        if domain.startswith("self.") or domain in ("i.redd.it", "v.redd.it", "reddit.com", ""):
            continue
        domains[domain] += 1
        tier = SOURCE_RELIABILITY.get(domain, "unrated")
        reliability_counts[tier] += 1

        # track which subreddits share which reliability tiers
        sub = p["subreddit"]
        if sub not in source_by_subreddit:
            source_by_subreddit[sub] = Counter()
        source_by_subreddit[sub][tier] += 1

    # engagement stats
    scores = [p["score"] for p in posts]
    comments = [p["num_comments"] for p in posts]

    return {
        "total_posts": total,
        "date_range": {"start": min(dates), "end": max(dates)},
        "unique_authors": len(authors),
        "subreddit_breakdown": dict(subreddits.most_common()),
        "top_authors": [
            {"author": a, "posts": c} for a, c in authors.most_common(15)
        ],
        "top_domains": [
            {"domain": d, "count": c, "reliability": SOURCE_RELIABILITY.get(d, "unrated")}
            for d, c in domains.most_common(20)
        ],
        "source_reliability": dict(reliability_counts),
        "source_by_subreddit": {
            sub: dict(counts) for sub, counts in source_by_subreddit.items()
        },
        "engagement": {
            "avg_score": round(sum(scores) / len(scores), 1),
            "max_score": max(scores),
            "avg_comments": round(sum(comments) / len(comments), 1),
            "max_comments": max(comments),
        },
    }
