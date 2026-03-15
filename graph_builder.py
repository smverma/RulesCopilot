"""
graph_builder.py
Builds a bipartite NetworkX graph (Rules ↔ Features) and renders it with
PyVis for interactive HTML display inside Streamlit.
"""

from __future__ import annotations

import os
import networkx as nx
from pyvis.network import Network


def build_rule_feature_graph(rules: list[dict]) -> nx.Graph:
    """
    Construct a bipartite NetworkX graph where:
      - One partition contains Rule nodes
      - Other partition contains Feature nodes
      - Edges connect each Rule to the features it references

    Each node carries a ``type`` attribute ("rule" or "feature").

    Parameters
    ----------
    rules:
        List of rule dicts.  Each dict must have ``name`` and ``features``
        keys; ``features`` is a list of dicts with at minimum a ``feature``
        key.

    Returns
    -------
    nx.Graph
    """
    G = nx.Graph()

    for rule in rules:
        rule_id = rule["name"]
        G.add_node(rule_id, type="rule")

        for feat in rule.get("features", []):
            feature_id = feat.get("feature", "unknown")
            if not G.has_node(feature_id):
                G.add_node(feature_id, type="feature")
            G.add_edge(rule_id, feature_id)

    return G


def render_graph_html(G: nx.Graph, output_path: str = "/tmp/rule_graph.html") -> str:
    """
    Render the NetworkX graph *G* to an interactive PyVis HTML file.

    - Rule nodes: blue (#4A90D9)
    - Feature nodes: orange (#F5A623)
    - Supports zoom and drag out of the box via PyVis

    Parameters
    ----------
    G:
        A NetworkX graph produced by :func:`build_rule_feature_graph`.
    output_path:
        Where to write the HTML file.

    Returns
    -------
    str
        Path to the generated HTML file.
    """
    net = Network(
        height="600px",
        width="100%",
        bgcolor="#1a1a2e",
        font_color="#ffffff",
        notebook=False,
    )

    # Physics for nicer layout
    net.barnes_hut(
        gravity=-8000,
        central_gravity=0.3,
        spring_length=150,
        spring_strength=0.05,
        damping=0.09,
    )

    for node, attrs in G.nodes(data=True):
        node_type = attrs.get("type", "feature")
        if node_type == "rule":
            net.add_node(
                node,
                label=node,
                color="#4A90D9",
                size=25,
                title=f"Rule: {node}",
                shape="dot",
            )
        else:
            net.add_node(
                node,
                label=node,
                color="#F5A623",
                size=18,
                title=f"Feature: {node}",
                shape="diamond",
            )

    for source, target in G.edges():
        net.add_edge(source, target, color="#888888", width=1.5)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    net.save_graph(output_path)
    return output_path


def get_graph_stats(G: nx.Graph) -> dict:
    """Return basic statistics about the bipartite graph."""
    rule_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "rule"]
    feature_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "feature"]
    return {
        "total_nodes": G.number_of_nodes(),
        "total_edges": G.number_of_edges(),
        "rule_nodes": len(rule_nodes),
        "feature_nodes": len(feature_nodes),
    }
