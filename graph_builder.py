"""
graph_builder.py
Builds and visualises a rule–feature bipartite graph using NetworkX and PyVis.
"""

from __future__ import annotations

import os
import tempfile
from typing import List, Tuple

import networkx as nx


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_bipartite_graph(rules: List[dict]) -> nx.Graph:
    """Build a bipartite graph connecting Rule nodes to Feature nodes.

    Parameters
    ----------
    rules:
        List of rule dicts.  Each dict must contain:
        - ``name``     (str)  – rule identifier used as node label
        - ``features`` (list) – list of feature condition dicts, each with a
                                ``feature`` key.

    Returns
    -------
    networkx.Graph
        Bipartite graph.  Nodes carry a ``bipartite`` attribute:
        - 0 → rule node
        - 1 → feature node
    """
    G = nx.Graph()

    for rule in rules:
        rule_name = rule["name"]
        G.add_node(rule_name, bipartite=0, node_type="rule")

        for cond in rule.get("features", []):
            feature = cond.get("feature", "").strip()
            if not feature:
                continue
            if feature not in G:
                G.add_node(feature, bipartite=1, node_type="feature")
            G.add_edge(rule_name, feature)

    return G


def get_graph_data(G: nx.Graph) -> Tuple[List[dict], List[dict]]:
    """Return ``(nodes, edges)`` lists suitable for JSON serialisation.

    Each node dict has keys ``id`` and ``type``.
    Each edge dict has keys ``source`` and ``target``.
    """
    nodes = [
        {"id": n, "type": data.get("node_type", "unknown")}
        for n, data in G.nodes(data=True)
    ]
    edges = [{"source": u, "target": v} for u, v in G.edges()]
    return nodes, edges


# ---------------------------------------------------------------------------
# PyVis visualisation
# ---------------------------------------------------------------------------

def render_pyvis_graph(G: nx.Graph, height: str = "600px") -> str:
    """Render *G* with PyVis and return the HTML string.

    Rule nodes are blue; feature nodes are orange.
    The returned HTML can be embedded directly in a Streamlit ``components.html``
    call.
    """
    try:
        from pyvis.network import Network
    except ImportError:
        return "<p>PyVis is not installed. Run: pip install pyvis</p>"

    net = Network(
        height=height,
        width="100%",
        bgcolor="#0e1117",
        font_color="white",
        notebook=False,
        directed=False,
        cdn_resources="in_line",
    )
    net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=200)

    for node, data in G.nodes(data=True):
        if data.get("node_type") == "rule":
            net.add_node(
                node,
                label=node,
                color="#4A90D9",
                size=20,
                title=f"Rule: {node}",
                shape="dot",
                font={"size": 14, "color": "white"},
            )
        else:
            net.add_node(
                node,
                label=node,
                color="#F5A623",
                size=14,
                title=f"Feature: {node}",
                shape="diamond",
                font={"size": 12, "color": "white"},
            )

    for u, v in G.edges():
        net.add_edge(u, v, color="#888888", width=1.5)

    # Write to a temp file then read back the HTML
    tmp = tempfile.NamedTemporaryFile(
        suffix=".html", delete=False, mode="w", encoding="utf-8"
    )
    tmp_path = tmp.name
    tmp.close()

    try:
        net.save_graph(tmp_path)
        with open(tmp_path, "r", encoding="utf-8") as fh:
            html_content = fh.read()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    return html_content
