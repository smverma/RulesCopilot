"""
similarity_engine.py
Computes pairwise cosine similarity between rule embeddings and classifies
rule pairs as duplicates or overlapping.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Thresholds (inclusive lower bound, exclusive upper bound for overlap)
DUPLICATE_THRESHOLD = 0.90
OVERLAP_LOWER = 0.80
OVERLAP_UPPER = DUPLICATE_THRESHOLD  # 0.80 <= sim < 0.90  → overlap


def compute_similarity_matrix(rules: list[dict]) -> np.ndarray:
    """
    Build an (N × N) cosine-similarity matrix from the embeddings stored in
    each rule dict.  Each rule must have an ``embedding`` key.

    Returns a numpy float64 array.
    """
    if not rules:
        return np.array([])

    embeddings = np.array([r["embedding"] for r in rules], dtype=float)
    # cosine_similarity returns shape (N, N)
    return cosine_similarity(embeddings)


def detect_duplicate_rules(rules: list[dict]) -> list[dict]:
    """
    Return a list of duplicate-pair dicts:
        {"rule1": <name>, "rule2": <name>, "similarity": <float>}

    A pair is a duplicate when cosine similarity >= DUPLICATE_THRESHOLD.
    Each pair is reported only once (upper triangle of the matrix).
    """
    sim_matrix = compute_similarity_matrix(rules)
    duplicates: list[dict] = []

    for i in range(len(rules)):
        for j in range(i + 1, len(rules)):
            sim = float(sim_matrix[i, j])
            if sim >= DUPLICATE_THRESHOLD:
                duplicates.append(
                    {
                        "rule1": rules[i]["name"],
                        "rule2": rules[j]["name"],
                        "similarity": round(sim, 4),
                    }
                )

    return duplicates


def detect_overlapping_rules(rules: list[dict]) -> list[dict]:
    """
    Return a list of overlapping-pair dicts:
        {"rule1": <name>, "rule2": <name>, "similarity": <float>}

    A pair overlaps when OVERLAP_LOWER <= cosine similarity < OVERLAP_UPPER.
    Each pair is reported only once (upper triangle of the matrix).
    """
    sim_matrix = compute_similarity_matrix(rules)
    overlapping: list[dict] = []

    for i in range(len(rules)):
        for j in range(i + 1, len(rules)):
            sim = float(sim_matrix[i, j])
            if OVERLAP_LOWER <= sim < OVERLAP_UPPER:
                overlapping.append(
                    {
                        "rule1": rules[i]["name"],
                        "rule2": rules[j]["name"],
                        "similarity": round(sim, 4),
                    }
                )

    return overlapping
