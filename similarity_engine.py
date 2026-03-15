"""
similarity_engine.py
Detects duplicate and overlapping fraud rules via cosine similarity on embeddings.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Similarity thresholds
DUPLICATE_THRESHOLD = 0.90
OVERLAP_LOWER = 0.80
OVERLAP_UPPER = DUPLICATE_THRESHOLD  # exclusive upper bound for overlaps


def _similarity_matrix(rules: List[dict]) -> np.ndarray:
    """Return an (N x N) cosine-similarity matrix for the rule embeddings.

    Requires at least 2 rules; callers should guard with ``len(rules) < 2``.
    """
    embeddings = np.array([r["embedding"] for r in rules], dtype=float)
    return cosine_similarity(embeddings)


def _pairs_above_threshold(
    sim_matrix: np.ndarray,
    low: float,
    high: float,
    rules: List[dict],
) -> List[Tuple[dict, dict, float]]:
    """Return unique (rule_a, rule_b, similarity) triples where low <= sim < high."""
    n = sim_matrix.shape[0]
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            score = float(sim_matrix[i, j])
            if low <= score < high:
                pairs.append((rules[i], rules[j], round(score, 4)))
    return pairs


def detect_duplicate_rules(rules: List[dict]) -> List[dict]:
    """Return pairs of rules whose embedding similarity is >= DUPLICATE_THRESHOLD.

    Each result entry is a dict with keys:
        rule1, rule2, similarity
    """
    if len(rules) < 2:
        return []

    sim_matrix = _similarity_matrix(rules)
    pairs = _pairs_above_threshold(
        sim_matrix, DUPLICATE_THRESHOLD, float("inf"), rules
    )
    return [
        {"rule1": a["name"], "rule2": b["name"], "similarity": score}
        for a, b, score in pairs
    ]


def detect_overlapping_rules(rules: List[dict]) -> List[dict]:
    """Return pairs of rules whose similarity is in [OVERLAP_LOWER, OVERLAP_UPPER).

    Each result entry is a dict with keys:
        rule1, rule2, similarity
    """
    if len(rules) < 2:
        return []

    sim_matrix = _similarity_matrix(rules)
    pairs = _pairs_above_threshold(
        sim_matrix, OVERLAP_LOWER, OVERLAP_UPPER, rules
    )
    return [
        {"rule1": a["name"], "rule2": b["name"], "similarity": score}
        for a, b, score in pairs
    ]


def get_similarity_matrix(rules: List[dict]) -> np.ndarray:
    """Expose the full similarity matrix (useful for conflict detection)."""
    return _similarity_matrix(rules)
