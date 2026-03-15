"""
embedding_service.py
Generates text embeddings for fraud rules using the Google Gemini embedding API.
Falls back to a TF-IDF-style numpy vector when no API key is available.
"""

import os
import hashlib
import numpy as np
from google import genai

_API_KEY = os.environ.get("GEMINI_API_KEY", "")
_client: genai.Client | None = None
if _API_KEY:
    _client = genai.Client(api_key=_API_KEY)

_EMBEDDING_MODEL = "text-embedding-004"

# In-memory cache: rule_text hash → embedding vector
_cache: dict[str, list[float]] = {}


def generate_embedding(rule_text: str) -> list[float]:
    """
    Return a float embedding vector for *rule_text*.
    Results are cached by content hash to avoid redundant API calls.
    """
    key = hashlib.sha256(rule_text.encode()).hexdigest()
    if key in _cache:
        return _cache[key]

    if _API_KEY and _client is not None:
        try:
            result = _client.models.embed_content(
                model=_EMBEDDING_MODEL,
                contents=rule_text,
            )
            embedding = result.embeddings[0].values
            _cache[key] = list(embedding)
            return list(embedding)
        except Exception:  # noqa: BLE001
            pass  # Fall through to fallback

    embedding = _fallback_embedding(rule_text)
    _cache[key] = embedding
    return embedding


# ---------------------------------------------------------------------------
# Fallback: deterministic bag-of-words vector (no external dependency)
# ---------------------------------------------------------------------------

# Vocabulary of fraud-domain tokens used to build the fallback vector
_VOCAB = [
    "amount", "transaction", "device", "new", "old", "country", "high", "risk",
    "velocity", "minutes", "block", "decline", "approve", "review", "flag",
    "card", "prepaid", "account", "age", "ip", "billing", "user", "days",
    "greater", "less", "equal", "than", "and", "or", "if", "in",
    "5000", "4800", "4000", "3000", "2000", "1000", "500", "3", "10", "7", "1",
]

_VOCAB_INDEX = {word: idx for idx, word in enumerate(_VOCAB)}


def _fallback_embedding(rule_text: str) -> list[float]:
    """
    Create a simple bag-of-words embedding over *_VOCAB*.
    Values are normalized to unit length.
    """
    tokens = rule_text.lower().split()
    vec = np.zeros(len(_VOCAB), dtype=float)
    for token in tokens:
        # Strip punctuation
        clean = token.strip(".,;:!?\"'()")
        if clean in _VOCAB_INDEX:
            vec[_VOCAB_INDEX[clean]] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


def clear_cache() -> None:
    """Clear the in-memory embedding cache (useful for testing)."""
    _cache.clear()
