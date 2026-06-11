"""
Network analysis – builds graphs from post data and computes
influence metrics. Supports different graph types:

- account:   author ↔ subreddit edges (who posts where)
- hashtag:   co-occurrence of hashtags within posts
- domain:    which domains get shared across subreddits
- url:       which URLs get shared by which authors

Each graph includes PageRank, betweenness centrality, and
Louvain community detection. We also support resilience analysis
(removing the top node and seeing how the graph changes).
"""

import logging
from collections import defaultdict, Counter
import networkx as nx

try:
    from community import community_louvain
    HAS_LOUVAIN = True
except ImportError:
    HAS_LOUVAIN = False

from ..core.data_loader import get_posts

logger = logging.getLogger(__name__)


def _build_account_graph(posts: list[dict]) -> nx.Graph:
    """Author ↔ subreddit bipartite graph, weighted by post count."""
    G = nx.Graph()
    edge_weights = defaultdict(int)

    for p in posts:
        author = p["author"]
        sub = f"r/{p['subreddit']}"
        if author == "[deleted]":
            continue
        edge_weights[(author, sub)] += 1

    for (a, s), weight in edge_weights.items():
        G.add_node(a, node_type="author")
        G.add_node(s, node_type="subreddit")
        G.add_edge(a, s, weight=weight)

    return G


def _build_domain_graph(posts: list[dict]) -> nx.Graph:
    """Domain ↔ subreddit bipartite graph – which sources feed which communities."""
    G = nx.Graph()
    edge_weights = defaultdict(int)

    for p in posts:
        domain = p.get("domain", "")
        sub = f"r/{p['subreddit']}"
        if not domain or domain.startswith("self.") or domain in ("i.redd.it", "v.redd.it", "reddit.com"):
            continue
        edge_weights[(domain, sub)] += 1

    for (d, s), weight in edge_weights.items():
        G.add_node(d, node_type="domain")
        G.add_node(s, node_type="subreddit")
        G.add_edge(d, s, weight=weight)

    return G


def _build_hashtag_graph(posts: list[dict]) -> nx.Graph:
    """Hashtag co-occurrence graph – tags that show up together in posts."""
    G = nx.Graph()
    edge_weights = defaultdict(int)

    for p in posts:
        tags = p.get("hashtags", [])
        if len(tags) < 2:
            continue
        for i, t1 in enumerate(tags):
            for t2 in tags[i + 1:]:
                edge_weights[(t1.lower(), t2.lower())] += 1

    for (t1, t2), weight in edge_weights.items():
        G.add_node(t1, node_type="hashtag")
        G.add_node(t2, node_type="hashtag")
        G.add_edge(t1, t2, weight=weight)

    return G


def _compute_metrics(G: nx.Graph) -> dict:
    """PageRank, betweenness, and Louvain communities for a graph."""
    if G.number_of_nodes() == 0:
        return {"pagerank": {}, "betweenness": {}, "communities": {}, "num_components": 0}

    # PageRank – works on disconnected graphs just fine
    try:
        pr = nx.pagerank(G, weight="weight")
    except Exception:
        pr = {n: 0.0 for n in G.nodes()}

    # betweenness centrality – can be slow on large graphs, so sample
    try:
        if G.number_of_nodes() > 500:
            bc = nx.betweenness_centrality(G, k=min(100, G.number_of_nodes()))
        else:
            bc = nx.betweenness_centrality(G)
    except Exception:
        bc = {n: 0.0 for n in G.nodes()}

    # Louvain community detection
    communities = {}
    if HAS_LOUVAIN:
        try:
            communities = community_louvain.best_partition(G)
        except Exception:
            communities = {n: 0 for n in G.nodes()}
    else:
        # fallback: connected components as communities
        for i, comp in enumerate(nx.connected_components(G)):
            for node in comp:
                communities[node] = i

    num_components = nx.number_connected_components(G)

    return {
        "pagerank": pr,
        "betweenness": bc,
        "communities": communities,
        "num_components": num_components,
    }


def build_network(
    graph_type: str = "account",
    query: str = None,
    remove_top_node: bool = False,
    max_nodes: int = 150,
) -> dict:
    """
    Build a network graph with metrics. Handles:
    - Different graph types
    - Optional query filtering
    - Top-node removal for resilience testing
    - Disconnected components (doesn't crash)
    - Max node cap so the frontend doesn't choke
    """
    posts = get_posts()

    # optional text filter
    if query:
        q = query.lower()
        posts = [p for p in posts if q in p["text"].lower()]

    if not posts:
        return {
            "nodes": [], "edges": [], "metrics": {},
            "message": "No posts match your query.",
            "graph_type": graph_type,
        }

    # build the right kind of graph
    builders = {
        "account": _build_account_graph,
        "domain": _build_domain_graph,
        "hashtag": _build_hashtag_graph,
    }
    builder = builders.get(graph_type, _build_account_graph)
    G = builder(posts)

    if G.number_of_nodes() == 0:
        return {
            "nodes": [], "edges": [], "metrics": {},
            "message": "Not enough data to build a network for this query.",
            "graph_type": graph_type,
        }

    # compute metrics before any node removal
    metrics = _compute_metrics(G)

    # resilience test: remove the node with highest PageRank
    removed_node = None
    if remove_top_node and metrics["pagerank"]:
        top_node = max(metrics["pagerank"], key=metrics["pagerank"].get)
        removed_node = top_node
        G.remove_node(top_node)
        # recompute metrics after removal
        metrics = _compute_metrics(G)
        metrics["removed_node"] = removed_node

    # trim to max_nodes by keeping highest PageRank nodes
    if G.number_of_nodes() > max_nodes:
        pr = metrics["pagerank"]
        keep = sorted(pr.keys(), key=lambda n: pr.get(n, 0), reverse=True)[:max_nodes]
        G = G.subgraph(keep).copy()
        metrics = _compute_metrics(G)

    # serialize for the frontend
    nodes = []
    for n in G.nodes():
        nodes.append({
            "id": n,
            "type": G.nodes[n].get("node_type", "unknown"),
            "pagerank": round(metrics["pagerank"].get(n, 0), 6),
            "betweenness": round(metrics["betweenness"].get(n, 0), 6),
            "community": metrics["communities"].get(n, 0),
            "degree": G.degree(n),
        })

    edges = []
    for u, v, data in G.edges(data=True):
        edges.append({
            "source": u,
            "target": v,
            "weight": data.get("weight", 1),
        })

    # community summary
    comm_counts = Counter(metrics["communities"].values())

    return {
        "nodes": nodes,
        "edges": edges,
        "graph_type": graph_type,
        "num_nodes": len(nodes),
        "num_edges": len(edges),
        "num_components": metrics["num_components"],
        "num_communities": len(comm_counts),
        "removed_node": removed_node,
        "top_influencers": sorted(nodes, key=lambda x: x["pagerank"], reverse=True)[:10],
    }
