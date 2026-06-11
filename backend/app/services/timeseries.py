"""
Time-series aggregation – buckets posts by hour/day/week and
optionally filters by query, subreddit, or author. The frontend
draws the chart, we just ship the numbers.
"""

from collections import defaultdict
from datetime import datetime, timezone
from ..core.data_loader import get_posts


# key events during the dataset period (Jul 2024 – Feb 2025)
# these get overlaid on the time-series chart as vertical markers
TIMELINE_EVENTS = [
    {"date": "2024-07-21", "label": "Biden drops out of race", "type": "politics"},
    {"date": "2024-07-22", "label": "Harris endorsed by Biden", "type": "politics"},
    {"date": "2024-08-19", "label": "DNC Convention begins", "type": "politics"},
    {"date": "2024-09-10", "label": "Trump-Harris debate (ABC)", "type": "debate"},
    {"date": "2024-10-01", "label": "VP debate: Vance vs Walz", "type": "debate"},
    {"date": "2024-11-05", "label": "Election Day", "type": "election"},
    {"date": "2024-11-06", "label": "Trump declared winner", "type": "election"},
    {"date": "2024-12-17", "label": "Electoral College votes", "type": "politics"},
    {"date": "2025-01-06", "label": "Jan 6 anniversary", "type": "politics"},
    {"date": "2025-01-20", "label": "Inauguration Day", "type": "politics"},
]


def _bucket_key(ts: float, granularity: str) -> str:
    """Turn a unix timestamp into a date string at the right granularity."""
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    if granularity == "hour":
        return dt.strftime("%Y-%m-%d %H:00")
    elif granularity == "week":
        # ISO week start (monday)
        iso = dt.isocalendar()
        return f"{iso.year}-W{iso.week:02d}"
    else:  # default: day
        return dt.strftime("%Y-%m-%d")


def aggregate_timeseries(
    query: str = None,
    subreddit: str = None,
    author: str = None,
    granularity: str = "day",
    group_by: str = None,
) -> dict:
    """
    Bucket posts into time bins. Optionally filter and group.
    Returns data shaped for a line/area chart.
    """
    posts = get_posts()

    # apply filters
    filtered = posts
    if query:
        q = query.lower()
        filtered = [p for p in filtered if q in p["text"].lower()]
    if subreddit:
        filtered = [p for p in filtered if p["subreddit"].lower() == subreddit.lower()]
    if author:
        filtered = [p for p in filtered if p["author"].lower() == author.lower()]

    if not filtered:
        return {
            "series": [],
            "events": TIMELINE_EVENTS,
            "total_posts": 0,
            "message": "No posts match your filters.",
        }

    # bucket the posts
    if group_by and group_by in ("subreddit", "author", "domain"):
        # grouped time-series: one line per group
        grouped: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for p in filtered:
            key = _bucket_key(p["created_utc"], granularity)
            group_val = p.get(group_by, "unknown")
            grouped[group_val][key] += 1

        # flatten into series format
        series = []
        for group_name, buckets in sorted(grouped.items(), key=lambda x: -sum(x[1].values()))[:10]:
            data_points = [{"date": k, "count": v} for k, v in sorted(buckets.items())]
            series.append({"name": group_name, "data": data_points})
    else:
        # simple time-series: one line
        buckets: dict[str, int] = defaultdict(int)
        for p in filtered:
            key = _bucket_key(p["created_utc"], granularity)
            buckets[key] += 1

        data_points = [{"date": k, "count": v} for k, v in sorted(buckets.items())]
        series = [{"name": "all", "data": data_points}]

    # filter events to the dataset's date range
    dates = [p["created_date"] for p in filtered]
    min_date, max_date = min(dates), max(dates)
    relevant_events = [
        e for e in TIMELINE_EVENTS
        if min_date <= e["date"] <= max_date
    ]

    return {
        "series": series,
        "events": relevant_events,
        "total_posts": len(filtered),
        "granularity": granularity,
        "filters": {"query": query, "subreddit": subreddit, "author": author},
    }
