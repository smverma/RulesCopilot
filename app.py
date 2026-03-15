"""
app.py
RuleCopilotAI – Streamlit dashboard for fraud rule intelligence.

Sections
--------
1. Rule Upload          – Upload / preview a rules CSV.
2. Rule Intelligence    – Metrics + duplicate / overlap / conflict tables.
3. Graph Visualisation  – Interactive bipartite rule-feature graph.
"""

from __future__ import annotations

import io
import os
import streamlit as st
import pandas as pd

from rule_parser import parse_rule
from embedding_service import generate_embedding
from similarity_engine import detect_duplicate_rules, detect_overlapping_rules
from conflict_detector import detect_rule_conflicts
from graph_builder import build_rule_feature_graph, render_graph_html, get_graph_stats

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RuleCopilotAI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        color: #4A90D9;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #aaaaaa;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #1e1e2f;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        text-align: center;
        border: 1px solid #333355;
    }
    .section-title {
        font-size: 1.4rem;
        font-weight: 600;
        color: #F5A623;
        border-bottom: 2px solid #F5A623;
        padding-bottom: 0.3rem;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<div class="main-header">🛡️ RuleCopilotAI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Fraud Rule Intelligence Platform – detect duplicates, overlaps, conflicts and visualise rule graphs.</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar – API key + info
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image(
        "https://img.shields.io/badge/Powered%20by-Gemini%20AI-blue?style=for-the-badge&logo=google",
        use_container_width=True,
    )
    st.markdown("### ⚙️ Configuration")
    api_key_input = st.text_input(
        "Gemini API Key (optional)",
        type="password",
        value=os.environ.get("GEMINI_API_KEY", ""),
        help="Provide your Gemini API key to enable AI-powered parsing and embeddings. "
             "The app works in demo mode without a key.",
    )
    if api_key_input:
        os.environ["GEMINI_API_KEY"] = api_key_input
        # Reload modules that read the env var at import time
        import importlib
        import rule_parser as _rp
        import embedding_service as _es

        importlib.reload(_rp)
        importlib.reload(_es)

    st.markdown("---")
    st.markdown(
        """
        **Legend**
        - 🔵 Blue nodes → Rules
        - 🟠 Orange nodes → Features
        - Similarity ≥ 0.90 → Duplicate
        - 0.80 – 0.89 → Overlap
        - Similar + different action → Conflict
        """
    )

# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------
if "rules" not in st.session_state:
    st.session_state["rules"] = []


def _process_rules(df: pd.DataFrame) -> list[dict]:
    """Parse + embed every rule in *df* and return a list of rule dicts."""
    rules: list[dict] = []
    progress = st.progress(0, text="Analysing rules…")
    total = len(df)

    for idx, row in df.iterrows():
        rule_name = str(row["rule_name"]).strip()
        rule_text = str(row["rule_text"]).strip()

        parsed = parse_rule(rule_text)
        embedding = generate_embedding(rule_text)

        rules.append(
            {
                "id": idx + 1,
                "name": rule_name,
                "text": rule_text,
                "features": parsed.get("features", []),
                "action": parsed.get("action", "unknown"),
                "embedding": embedding,
            }
        )
        progress.progress((idx + 1) / total, text=f"Processing {rule_name}…")

    progress.empty()
    return rules


# ===========================================================================
# SECTION 1 – Rule Upload
# ===========================================================================
st.markdown('<div class="section-title">📂 Section 1 – Rule Upload</div>', unsafe_allow_html=True)

upload_col, sample_col = st.columns([3, 1])

with upload_col:
    uploaded_file = st.file_uploader(
        "Upload your rules CSV (columns: rule_name, rule_text)",
        type=["csv"],
        help="The CSV must have at least two columns: rule_name and rule_text.",
    )

with sample_col:
    st.markdown("##### Or use sample data")
    if st.button("🎲 Load Sample Rules", use_container_width=True):
        sample_path = os.path.join(os.path.dirname(__file__), "sample_rules.csv")
        if os.path.exists(sample_path):
            with open(sample_path, "rb") as f:
                uploaded_file = io.BytesIO(f.read())
                uploaded_file.name = "sample_rules.csv"  # type: ignore[attr-defined]
        else:
            st.error("sample_rules.csv not found.")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        df.columns = [c.strip().lower() for c in df.columns]

        if "rule_name" not in df.columns or "rule_text" not in df.columns:
            st.error("CSV must contain columns: **rule_name** and **rule_text**")
        else:
            st.success(f"✅ Loaded **{len(df)}** rules.")
            st.dataframe(df, use_container_width=True, height=250)

            if st.button("🚀 Analyse Rules", type="primary", use_container_width=True):
                with st.spinner("Running rule intelligence engine…"):
                    st.session_state["rules"] = _process_rules(df)
                st.success("Analysis complete! Scroll down to view results.")
    except Exception as exc:
        st.error(f"Failed to read CSV: {exc}")

# ===========================================================================
# SECTION 2 – Rule Intelligence Summary
# ===========================================================================
st.markdown("---")
st.markdown(
    '<div class="section-title">🧠 Section 2 – Rule Intelligence Summary</div>',
    unsafe_allow_html=True,
)

rules: list[dict] = st.session_state.get("rules", [])

if not rules:
    st.info("Upload and analyse a rules CSV in Section 1 to see intelligence results here.")
else:
    duplicates = detect_duplicate_rules(rules)
    overlapping = detect_overlapping_rules(rules)
    conflicts_result = detect_rule_conflicts(rules)
    conflicts = conflicts_result.get("conflicts", [])

    # --- Metrics ---
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("📋 Total Rules", len(rules))
    m2.metric("🔁 Duplicate Pairs", len(duplicates))
    m3.metric("🔀 Overlapping Pairs", len(overlapping))
    m4.metric("⚠️ Conflicting Pairs", len(conflicts))

    st.markdown("---")

    # --- Duplicate rules table ---
    with st.expander(f"🔁 Duplicate Rules ({len(duplicates)} pairs)", expanded=bool(duplicates)):
        if duplicates:
            st.dataframe(pd.DataFrame(duplicates), use_container_width=True)
        else:
            st.success("No duplicate rules detected.")

    # --- Overlapping rules table ---
    with st.expander(f"🔀 Overlapping Rules ({len(overlapping)} pairs)", expanded=bool(overlapping)):
        if overlapping:
            st.dataframe(pd.DataFrame(overlapping), use_container_width=True)
        else:
            st.success("No overlapping rules detected.")

    # --- Conflicting rules table ---
    with st.expander(f"⚠️ Conflicting Rules ({len(conflicts)} pairs)", expanded=bool(conflicts)):
        if conflicts:
            st.dataframe(pd.DataFrame(conflicts), use_container_width=True)
        else:
            st.success("No conflicting rules detected.")

    # --- Parsed rule details ---
    with st.expander("🔍 Parsed Rule Details", expanded=False):
        parsed_rows = []
        for r in rules:
            feature_str = ", ".join(
                f"{f['feature']} {f['operator']} {f['value']}"
                for f in r.get("features", [])
            )
            parsed_rows.append(
                {
                    "Rule Name": r["name"],
                    "Action": r["action"],
                    "Features Detected": feature_str or "—",
                    "Rule Text": r["text"],
                }
            )
        st.dataframe(pd.DataFrame(parsed_rows), use_container_width=True)

# ===========================================================================
# SECTION 3 – Graph Visualisation
# ===========================================================================
st.markdown("---")
st.markdown(
    '<div class="section-title">📊 Section 3 – Rule-Feature Bipartite Graph</div>',
    unsafe_allow_html=True,
)

if not rules:
    st.info("Analyse rules in Section 1 to generate the graph.")
else:
    G = build_rule_feature_graph(rules)
    stats = get_graph_stats(G)

    gs1, gs2, gs3, gs4 = st.columns(4)
    gs1.metric("🔵 Rule Nodes", stats["rule_nodes"])
    gs2.metric("🟠 Feature Nodes", stats["feature_nodes"])
    gs3.metric("🔗 Total Edges", stats["total_edges"])
    gs4.metric("🕸️ Total Nodes", stats["total_nodes"])

    html_path = render_graph_html(G, output_path="/tmp/rule_graph.html")

    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    st.components.v1.html(html_content, height=650, scrolling=False)

    st.caption(
        "🔵 Blue nodes = Rules  |  🟠 Orange nodes = Features  |  Drag nodes to rearrange."
    )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<center><small>RuleCopilotAI · Fraud Rule Intelligence Platform · "
    "Powered by Google Gemini &amp; NetworkX</small></center>",
    unsafe_allow_html=True,
)
