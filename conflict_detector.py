"""
conflict_detector.py
Detects conflicting fraud rules: rules with similar conditions but different actions.
"""

from __future__ import annotations

from typing import List

from similarity_engine import get_similarity_matrix

CONFLICT_SIMILARITY_THRESHOLD = 0.85


def detect_rule_conflicts(rules: List[dict]) -> dict:
    """Detect rules that are semantically similar but have different actions.

    Two rules conflict when:
    - Their embedding cosine similarity >= CONFLICT_SIMILARITY_THRESHOLD
    - They carry different ``action`` values

    Parameters
    ----------
    rules:
        List of rule dicts, each with keys ``name``, ``action``, and
        ``embedding``.

    Returns
    -------
    dict
        ``{"conflicts": [{"rule1": <name>, "rule2": <name>, "similarity": <float>,
                          "action1": <str>, "action2": <str>}, ...]}``
    """
    if len(rules) < 2:
        return {"conflicts": []}

    sim_matrix = get_similarity_matrix(rules)
    n = len(rules)
    conflicts = []

    for i in range(n):
        for j in range(i + 1, n):
            score = float(sim_matrix[i, j])
            if score >= CONFLICT_SIMILARITY_THRESHOLD:
                action_i = (rules[i].get("action") or "").lower().strip()
                action_j = (rules[j].get("action") or "").lower().strip()
                if action_i != action_j:
                    conflicts.append(
                        {
                            "rule1": rules[i]["name"],
                            "rule2": rules[j]["name"],
                            "similarity": round(score, 4),
                            "action1": action_i,
                            "action2": action_j,
                        }
                    )

    return {"conflicts": conflicts}
