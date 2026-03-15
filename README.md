# Rule Copilot AI 🛡️

**Fraud Rule Intelligence Platform**

Rule Copilot AI helps fraud teams analyse large rule libraries and automatically detect:

- 🔁 **Duplicate rules** – near-identical rules wasting evaluation cycles
- 🔀 **Overlapping rules** – rules with partially shared conditions
- ⚠️ **Conflicting rules** – similar conditions but contradictory actions
- 🕸️ **Rule-feature dependencies** – visualised as a bipartite graph

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | [Streamlit](https://streamlit.io) |
| AI (parsing + embeddings) | [Google Gemini API](https://ai.google.dev/) |
| Graph processing | [NetworkX](https://networkx.org/) |
| Graph visualisation | [PyVis](https://pyvis.readthedocs.io/) |
| Similarity | cosine similarity via scikit-learn |
| Storage | In-memory Python objects |

---

## Project Structure

```
RulesCopilot/
├── app.py                 # Streamlit dashboard (3 sections)
├── rule_parser.py         # Gemini-based rule parser (+ heuristic fallback)
├── embedding_service.py   # Gemini embeddings (+ TF-IDF fallback)
├── similarity_engine.py   # Duplicate & overlap detection
├── conflict_detector.py   # Conflict detection
├── graph_builder.py       # NetworkX graph + PyVis renderer
├── sample_rules.csv       # 15-rule demo dataset
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. (Optional) Set your Gemini API key

```bash
export GEMINI_API_KEY="AIza..."
```

If no key is provided, the app uses a TF-IDF fallback for embeddings and a
heuristic parser for rule parsing – the demo still works end-to-end.

### 3. Run

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Using the App

### Section 1 – Rule Upload
Upload a CSV with columns `rule_name` and `rule_text`, or click **Load Demo Rules**
to use the built-in sample dataset.  
Click **Analyse Rules** to parse and embed all rules.

### Section 2 – Rule Intelligence Summary
Displays four metrics and expandable tables for:
- Duplicate rules (similarity ≥ 0.90)
- Overlapping rules (similarity 0.80–0.90)
- Conflicting rules (similarity ≥ 0.85, different actions)

### Section 3 – Graph Visualisation
Interactive bipartite graph built with PyVis:
- 🔵 Blue nodes = Rules
- 🟠 Orange nodes = Features  
- Drag nodes, zoom, and pan to explore dependencies.

---

## CSV Format

```csv
rule_name,rule_text
Rule1,If transaction amount > 5000 and device is new decline transaction
Rule2,If amount > 4800 and device is new decline transaction
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `GEMINI_API_KEY` | Google Gemini API key (optional) |
