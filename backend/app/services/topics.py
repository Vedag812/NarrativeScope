"""
Topic clustering – KMeans on sentence-transformer embeddings,
with TF-IDF keyword extraction for human-readable labels.

The number of clusters is a tunable parameter exposed to the frontend.
We clamp it to sane bounds so extreme values don't crash the app.
"""

import logging
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

from ..core.config import DEFAULT_N_CLUSTERS, MAX_CLUSTERS, MIN_CLUSTERS
from ..core.data_loader import get_posts
from ..services.embeddings import get_embeddings

logger = logging.getLogger(__name__)


def cluster_posts(n_clusters: int = DEFAULT_N_CLUSTERS) -> dict:
    """
    Cluster all posts and return cluster metadata.

    Handles extreme values:
    - n_clusters < 2 → clamp to 2
    - n_clusters > num_posts → cap at num_posts
    - n_clusters > MAX_CLUSTERS → cap at MAX_CLUSTERS
    """
    posts = get_posts()
    embeddings, post_ids = get_embeddings()
    n_posts = len(posts)

    # clamp cluster count to something reasonable
    original_n = n_clusters
    n_clusters = max(MIN_CLUSTERS, min(n_clusters, MAX_CLUSTERS, n_posts))

    clamped_msg = None
    if n_clusters != original_n:
        clamped_msg = (
            f"Requested {original_n} clusters, adjusted to {n_clusters} "
            f"(valid range: {MIN_CLUSTERS}-{min(MAX_CLUSTERS, n_posts)})"
        )
        logger.info(clamped_msg)

    # run KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    # build a map from post_id to post for quick lookups
    post_map = {p["id"]: p for p in posts}

    # extract keywords per cluster using TF-IDF
    cluster_texts = {}
    cluster_posts_map = {}
    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        ids_in_cluster = [post_ids[i] for i in range(len(post_ids)) if mask[i]]
        posts_in_cluster = [post_map[pid] for pid in ids_in_cluster if pid in post_map]
        cluster_posts_map[cluster_id] = posts_in_cluster
        cluster_texts[cluster_id] = " ".join(p["text"][:200] for p in posts_in_cluster)

    # TF-IDF for keyword extraction
    all_cluster_texts = [cluster_texts[i] for i in range(n_clusters)]
    tfidf = TfidfVectorizer(max_features=1000, stop_words="english", max_df=0.8)

    try:
        tfidf_matrix = tfidf.fit_transform(all_cluster_texts)
        feature_names = tfidf.get_feature_names_out()
    except ValueError:
        # fallback if text is too short or empty
        feature_names = []
        tfidf_matrix = None

    clusters = []
    for cluster_id in range(n_clusters):
        posts_in_cluster = cluster_posts_map[cluster_id]

        # top keywords from TF-IDF
        keywords = []
        if tfidf_matrix is not None and len(feature_names) > 0:
            row = tfidf_matrix[cluster_id].toarray().flatten()
            top_indices = row.argsort()[-8:][::-1]
            keywords = [feature_names[i] for i in top_indices if row[i] > 0]

        # subreddit breakdown
        sub_counts = Counter(p["subreddit"] for p in posts_in_cluster)

        # pick 3 representative posts (highest score)
        rep_posts = sorted(posts_in_cluster, key=lambda x: x["score"], reverse=True)[:3]

        clusters.append({
            "id": cluster_id,
            "size": len(posts_in_cluster),
            "keywords": keywords,
            "label": ", ".join(keywords[:3]) if keywords else f"Cluster {cluster_id}",
            "subreddit_breakdown": dict(sub_counts.most_common(5)),
            "representative_posts": [
                {
                    "id": p["id"],
                    "title": p["title"],
                    "subreddit": p["subreddit"],
                    "score": p["score"],
                    "author": p["author"],
                }
                for p in rep_posts
            ],
        })

    return {
        "n_clusters": n_clusters,
        "clusters": clusters,
        "total_posts": n_posts,
        "message": clamped_msg,
    }


def export_projector_data() -> dict:
    """
    Export embeddings + metadata as TSV strings for TensorFlow Projector.
    The frontend can link to https://projector.tensorflow.org/ with these files.
    """
    embeddings, post_ids = get_embeddings()
    posts = get_posts()
    post_map = {p["id"]: p for p in posts}

    # vectors TSV: one row per post, tab-separated floats
    vectors_lines = []
    metadata_lines = ["id\ttitle\tsubreddit\tauthor\tscore"]  # header

    for i, pid in enumerate(post_ids):
        post = post_map.get(pid)
        if not post:
            continue
        vec_str = "\t".join(f"{v:.6f}" for v in embeddings[i])
        vectors_lines.append(vec_str)

        # escape tabs in text fields
        title = post["title"][:80].replace("\t", " ").replace("\n", " ")
        metadata_lines.append(
            f"{pid}\t{title}\t{post['subreddit']}\t{post['author']}\t{post['score']}"
        )

    return {
        "vectors_tsv": "\n".join(vectors_lines),
        "metadata_tsv": "\n".join(metadata_lines),
    }
