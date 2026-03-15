"""
embedding_service.py
Generates and caches rule embeddings using the Google Gemini Embedding API.
Falls back to TF-IDF vectors when the API is unavailable.
"""

from __future__ import annotations

import os
from typing import List

import numpy as np

# ---------------------------------------------------------------------------
# Gemini embedding helpers
# ---------------------------------------------------------------------------

def _gemini_embed(text: str) -> List[float] | None:
    """Return a Gemini embedding for *text*, or None on failure."""
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return None
    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        result = client.models.embed_content(
            model="models/text-embedding-004",
            contents=text,
        )
        return result.embeddings[0].values
    except Exception:
        return None


# ---------------------------------------------------------------------------
# TF-IDF fallback
# ---------------------------------------------------------------------------

_tfidf_vectorizer = None
_tfidf_corpus: List[str] = []
_tfidf_matrix = None


def _tfidf_embed_all(texts: List[str]) -> np.ndarray:
    """Return a TF-IDF matrix for *texts* (one row per text)."""
    global _tfidf_vectorizer, _tfidf_corpus, _tfidf_matrix

    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    matrix = vectorizer.fit_transform(texts).toarray()

    _tfidf_vectorizer = vectorizer
    _tfidf_corpus = texts
    _tfidf_matrix = matrix
    return matrix


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_embedding(rule_text: str) -> List[float]:
    """Generate an embedding vector for a single rule text.

    Tries the Gemini API first; falls back to a zero-vector placeholder.
    The preferred entry-point for batch processing is
    :func:`generate_embeddings_for_rules`.
    """
    vec = _gemini_embed(rule_text)
    if vec is not None:
        return vec
    # Return a placeholder; callers should use generate_embeddings_for_rules
    # for proper TF-IDF fallback that takes all texts into account.
    return []


def generate_embeddings_for_rules(rules: List[dict]) -> List[dict]:
    """Attach an ``embedding`` vector to each rule dict in *rules*.

    Modifies *rules* in-place and returns the same list.

    Each rule dict must contain at least ``text`` (str).

    Strategy:
    1. Try to obtain Gemini embeddings one-by-one (API call per rule).
    2. If any embedding is empty (API unavailable / no key), fall back to
       computing TF-IDF vectors for the whole corpus so that cosine
       similarity is meaningful.
    """
    texts = [r["text"] for r in rules]

    # Attempt Gemini embeddings
    embeddings = [_gemini_embed(t) for t in texts]

    if all(e is not None for e in embeddings):
        for rule, emb in zip(rules, embeddings):
            rule["embedding"] = emb
        return rules

    # Fallback: TF-IDF for the whole corpus
    matrix = _tfidf_embed_all(texts)
    for rule, row in zip(rules, matrix):
        rule["embedding"] = row.tolist()

    return rules
