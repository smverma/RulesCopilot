"""
conflict_detector.py
Identifies conflicting rules: pairs whose conditions are highly similar but
whose actions differ.
"""

from __future__ import annotations

from similarity_engine import compute_similarity_matrix

CONFLICT_THRESHOLD = 0.85


def detect_rule_conflicts(rules: list[dict]) -> dict:
    """
    Detect pairs of rules that are likely in conflict.

    Two rules conflict when:
      - Cosine similarity of their embeddings >= CONFLICT_THRESHOLD
      - Their ``action`` fields are different

    Returns:
        {
            "conflicts": [
                {"rule1": <name>, "rule2": <name>, "similarity": <float>,
                 "action1": <str>, "action2": <str>}
            ]
        }
    """
    if not rules:
        return {"conflicts": []}

    sim_matrix = compute_similarity_matrix(rules)
    conflicts: list[dict] = []

    for i in range(len(rules)):
        for j in range(i + 1, len(rules)):
            sim = float(sim_matrix[i, j])
            action_i = rules[i].get("action", "unknown")
            action_j = rules[j].get("action", "unknown")

            if sim >= CONFLICT_THRESHOLD and action_i != action_j:
                conflicts.append(
                    {
                        "rule1": rules[i]["name"],
                        "rule2": rules[j]["name"],
                        "similarity": round(sim, 4),
                        "action1": action_i,
                        "action2": action_j,
                    }
                )

    return {"conflicts": conflicts}
