"""
app.py – Rule Copilot AI  |  Streamlit Dashboard

Sections
--------
1. Rule Upload        – Upload a CSV file; display parsed rules.
2. Rule Intelligence  – Metrics + tables for duplicates, overlaps, conflicts.
3. Graph Visualisation – Interactive bipartite rule-feature graph.
"""

from __future__ import annotations

import io
import os
import traceback
from typing import List

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# ---------------------------------------------------------------------------
# Page config  (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Rule Copilot AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Local module imports (after page config so Streamlit initialises cleanly)
# ---------------------------------------------------------------------------
from rule_parser import parse_rule
from embedding_service import generate_embeddings_for_rules
from similarity_engine import detect_duplicate_rules, detect_overlapping_rules
from conflict_detector import detect_rule_conflicts
from graph_builder import build_bipartite_graph, render_pyvis_graph

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_CSV_PATH = os.path.join(os.path.dirname(__file__), "sample_rules.csv")


def _load_sample_rules() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_CSV_PATH)


def _parse_and_enrich(df: pd.DataFrame, status_placeholder) -> List[dict]:
    """Parse every rule text and return enriched rule list."""
    rules: List[dict] = []
    total = len(df)

    for idx, row in df.iterrows():
        name = str(row["rule_name"]).strip()
        text = str(row["rule_text"]).strip()
        status_placeholder.info(f"Parsing rule {idx + 1}/{total}: **{name}** …")
        try:
            parsed = parse_rule(text)
        except Exception:
            parsed = {"features": [], "action": "unknown"}

        rules.append(
            {
                "id": idx + 1,
                "name": name,
                "text": text,
                "features": parsed.get("features", []),
                "action": parsed.get("action", "unknown"),
                "embedding": [],
            }
        )

    status_placeholder.info("Generating embeddings …")
    generate_embeddings_for_rules(rules)
    status_placeholder.empty()
    return rules


def _rules_to_df(rules: List[dict]) -> pd.DataFrame:
    rows = []
    for r in rules:
        feature_names = ", ".join(
            c.get("feature", "") for c in r.get("features", [])
        )
        rows.append(
            {
                "Rule Name": r["name"],
                "Action": r["action"],
                "Features": feature_names,
                "Rule Text": r["text"],
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Sidebar – API key & controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.image(
        "https://img.icons8.com/color/96/000000/artificial-intelligence.png",
        width=64,
    )
    st.title("Rule Copilot AI")
    st.markdown("*Fraud Rule Intelligence Platform*")
    st.divider()

    gemini_key = st.text_input(
        "Gemini API Key",
        type="password",
        help="Optional – if empty, TF-IDF fallback embeddings are used.",
        placeholder="AIza…",
    )
    if gemini_key:
        os.environ["GEMINI_API_KEY"] = gemini_key

    st.divider()
    st.markdown(
        """
        **How it works**
        1. Upload your rule CSV (or use the demo).
        2. Rules are parsed by Gemini (or a heuristic fallback).
        3. Embeddings power duplicate/overlap/conflict detection.
        4. Explore the interactive rule-feature graph.
        """
    )
    st.divider()
    st.caption("Rule Copilot AI · MVP")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "rules" not in st.session_state:
    st.session_state["rules"] = []
if "analysis_done" not in st.session_state:
    st.session_state["analysis_done"] = False
if "raw_df" not in st.session_state:
    st.session_state["raw_df"] = None

# ---------------------------------------------------------------------------
# Main header
# ---------------------------------------------------------------------------

st.markdown(
    """
    <h1 style='text-align:center; color:#4A90D9;'>🛡️ Rule Copilot AI</h1>
    <p style='text-align:center; color:#aaa; font-size:1.1em;'>
        Fraud Rule Intelligence Platform &nbsp;|&nbsp;
        Detect duplicates · overlaps · conflicts · visualise dependencies
    </p>
    """,
    unsafe_allow_html=True,
)
st.divider()

# ===========================================================================
# SECTION 1 – Rule Upload
# ===========================================================================

st.header("📂 Section 1 · Rule Upload")

upload_col, demo_col = st.columns([3, 1])
with upload_col:
    uploaded_file = st.file_uploader(
        "Upload a CSV file with columns `rule_name` and `rule_text`",
        type=["csv"],
        help="Each row should have a rule name and the plain-English rule text.",
    )
with demo_col:
    st.markdown("<br>", unsafe_allow_html=True)
    use_demo = st.button("▶ Load Demo Rules", use_container_width=True)

raw_df: pd.DataFrame | None = None

if uploaded_file is not None:
    try:
        raw_df = pd.read_csv(uploaded_file)
        if "rule_name" not in raw_df.columns or "rule_text" not in raw_df.columns:
            st.error("CSV must contain columns `rule_name` and `rule_text`.")
            raw_df = None
        else:
            st.session_state["raw_df"] = raw_df
    except Exception as exc:
        st.error(f"Failed to read CSV: {exc}")
elif use_demo:
    raw_df = _load_sample_rules()
    st.session_state["raw_df"] = raw_df
    st.success("Demo rules loaded.")
elif st.session_state["raw_df"] is not None:
    raw_df = st.session_state["raw_df"]

if raw_df is not None:
    st.subheader(f"Uploaded Rules ({len(raw_df)} rows)")
    st.dataframe(raw_df, use_container_width=True)

    analyse_btn = st.button("🔍 Analyse Rules", type="primary", use_container_width=True)
    if analyse_btn:
        status = st.empty()
        with st.spinner("Analysing rules …"):
            try:
                rules = _parse_and_enrich(raw_df, status)
                st.session_state["rules"] = rules
                st.session_state["analysis_done"] = True
                st.success(f"✅ Analysis complete – {len(rules)} rules processed.")
            except Exception:
                st.error("An error occurred during analysis.")
                st.code(traceback.format_exc())

# ===========================================================================
# SECTION 2 – Rule Intelligence Summary
# ===========================================================================

st.divider()
st.header("📊 Section 2 · Rule Intelligence Summary")

rules: List[dict] = st.session_state.get("rules", [])

if not rules:
    st.info("Upload and analyse rules in Section 1 to see the intelligence summary.")
else:
    duplicates = detect_duplicate_rules(rules)
    overlaps = detect_overlapping_rules(rules)
    conflicts_result = detect_rule_conflicts(rules)
    conflicts = conflicts_result.get("conflicts", [])

    # ── Metrics row ────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("📋 Total Rules", len(rules))
    m2.metric("🔁 Duplicate Rules", len(duplicates))
    m3.metric("🔀 Overlapping Rules", len(overlaps))
    m4.metric("⚠️ Conflicting Rules", len(conflicts))

    st.markdown("---")

    # ── Duplicate rules ────────────────────────────────────────────────────
    with st.expander(f"🔁 Duplicate Rules ({len(duplicates)})", expanded=len(duplicates) > 0):
        if duplicates:
            st.dataframe(
                pd.DataFrame(duplicates)[["rule1", "rule2", "similarity"]].rename(
                    columns={"rule1": "Rule 1", "rule2": "Rule 2", "similarity": "Similarity"}
                ),
                use_container_width=True,
            )
        else:
            st.success("No duplicate rules detected.")

    # ── Overlapping rules ──────────────────────────────────────────────────
    with st.expander(f"🔀 Overlapping Rules ({len(overlaps)})", expanded=len(overlaps) > 0):
        if overlaps:
            st.dataframe(
                pd.DataFrame(overlaps)[["rule1", "rule2", "similarity"]].rename(
                    columns={"rule1": "Rule 1", "rule2": "Rule 2", "similarity": "Similarity"}
                ),
                use_container_width=True,
            )
        else:
            st.success("No overlapping rules detected.")

    # ── Conflicting rules ──────────────────────────────────────────────────
    with st.expander(f"⚠️ Conflicting Rules ({len(conflicts)})", expanded=len(conflicts) > 0):
        if conflicts:
            st.dataframe(
                pd.DataFrame(conflicts)[
                    ["rule1", "rule2", "action1", "action2", "similarity"]
                ].rename(
                    columns={
                        "rule1": "Rule 1",
                        "rule2": "Rule 2",
                        "action1": "Action 1",
                        "action2": "Action 2",
                        "similarity": "Similarity",
                    }
                ),
                use_container_width=True,
            )
        else:
            st.success("No conflicting rules detected.")

    # ── Parsed rules table ─────────────────────────────────────────────────
    with st.expander("📋 All Parsed Rules", expanded=False):
        st.dataframe(_rules_to_df(rules), use_container_width=True)

# ===========================================================================
# SECTION 3 – Graph Visualisation
# ===========================================================================

st.divider()
st.header("🕸️ Section 3 · Rule-Feature Graph")

if not rules:
    st.info("Analyse rules first (Section 1) to see the graph.")
else:
    with st.spinner("Building graph …"):
        G = build_bipartite_graph(rules)

    st.markdown(
        f"""
        **Graph stats** &nbsp;·&nbsp;
        Nodes: **{G.number_of_nodes()}** &nbsp;·&nbsp;
        Edges: **{G.number_of_edges()}** &nbsp;·&nbsp;
        🔵 Blue = Rules &nbsp;·&nbsp; 🟠 Orange = Features
        """
    )

    graph_height = st.slider(
        "Graph height (px)", min_value=400, max_value=1000, value=600, step=50
    )

    with st.spinner("Rendering graph …"):
        html_content = render_pyvis_graph(G, height=f"{graph_height}px")

    components.html(html_content, height=graph_height + 20, scrolling=False)

    with st.expander("📄 Raw Graph Data (JSON)", expanded=False):
        from graph_builder import get_graph_data
        nodes, edges = get_graph_data(G)
        import json
        st.json({"nodes": nodes, "edges": edges})
