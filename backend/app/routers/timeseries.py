"""
Time-series router – aggregated post counts over time with GenAI summaries.
"""

from fastapi import APIRouter, Query
from ..services.timeseries import aggregate_timeseries
from ..services.genai import generate_timeseries_summary

router = APIRouter(prefix="/api", tags=["timeseries"])


@router.get("/timeseries")
async def get_timeseries(
    q: str = Query(None, description="Filter posts by keyword"),
    subreddit: str = Query(None),
    author: str = Query(None),
    granularity: str = Query("day", regex="^(hour|day|week)$"),
    group_by: str = Query(None, description="Group by: subreddit, author, domain"),
):
    """
    Time-series data with optional GenAI summary.
    Each response includes a plain-language summary so non-technical
    readers can understand the trend without reading the chart.
    """
    data = aggregate_timeseries(
        query=q, subreddit=subreddit, author=author,
        granularity=granularity, group_by=group_by,
    )

    # generate a plain-language summary for the main series
    summary = ""
    if data["series"]:
        # use the first (or only) series for the summary
        main_series = data["series"][0]["data"]
        if main_series:
            try:
                summary = await generate_timeseries_summary(
                    main_series, query=q, subreddit=subreddit
                )
            except Exception:
                summary = "Summary generation unavailable."

    data["summary"] = summary
    return data
