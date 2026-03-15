"""
rule_parser.py
Parses fraud rule text into structured JSON using the Google Gemini API.
"""

import json
import os
import re


def _get_client():
    """Return a configured Gemini client, or None if no API key is set."""
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None
    from google import genai

    return genai.Client(api_key=api_key)


_PARSE_PROMPT_TEMPLATE = """Convert the following fraud rule into structured JSON.

Rule:
"{rule_text}"

Output ONLY valid JSON in exactly this format (no markdown, no explanation):
{{
  "features": [
    {{"feature": "<feature_name>", "operator": "<operator>", "value": <value>}}
  ],
  "action": "<action>"
}}

Rules for the JSON:
- "features" is a list of conditions extracted from the rule.
- "action" must be one of: "decline", "approve", "block", "review", "flag", "allow".
- Numeric values should be numbers, not strings.
- Feature names should use snake_case (e.g. transaction_amount, device_age, ip_country).
"""


def parse_rule(rule_text: str) -> dict:
    """Parse a single fraud rule text into structured JSON via Gemini.

    Parameters
    ----------
    rule_text:
        The natural-language fraud rule to parse.

    Returns
    -------
    dict
        Parsed rule with ``features`` list and ``action`` string.
        Returns a fallback dict on API or parsing errors.
    """
    client = _get_client()
    if client is None:
        return _fallback_parse(rule_text)

    prompt = _PARSE_PROMPT_TEMPLATE.format(rule_text=rule_text)

    try:
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt,
        )
        raw = response.text.strip()

        # Strip markdown code fences if present
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        parsed = json.loads(raw)
        return parsed
    except Exception:
        # Return a best-effort fallback so callers always receive a dict
        return _fallback_parse(rule_text)


# ---------------------------------------------------------------------------
# Fallback parser (no API required) – used when Gemini is unavailable
# ---------------------------------------------------------------------------

_ACTION_KEYWORDS = ["decline", "approve", "block", "review", "flag", "allow"]

_OPERATOR_MAP = {
    ">=": ">=",
    "<=": "<=",
    ">": ">",
    "<": "<",
    "!=": "!=",
    "==": "=",
    "=": "=",
    "is not": "!=",
    "is": "=",
}

_FEATURE_ALIASES = {
    "transaction amount": "transaction_amount",
    "amount": "amount",
    "device age": "device_age",
    "device_age": "device_age",
    "device": "device",
    "velocity": "velocity",
    "country": "country",
    "ip_country": "ip_country",
    "billing_country": "billing_country",
    "card_type": "card_type",
    "user_age": "user_age",
    "device_fingerprint": "device_fingerprint",
    "email_domain": "email_domain",
}


_LEADING_STOP_WORDS = re.compile(
    r"^\s*(if|when|and|or|where|then|also)\s+", re.IGNORECASE
)

# Stop-word tokens that must not appear inside a cleaned feature name
_INNER_STOP_WORDS = {"and", "or", "if", "when", "then", "in", "for"}


def _clean_feature(raw: str) -> str:
    """Strip leading stop-words and normalise to snake_case."""
    # Iteratively strip leading stop-words
    cleaned = raw.strip()
    while True:
        new = _LEADING_STOP_WORDS.sub("", cleaned).strip()
        if new == cleaned:
            break
        cleaned = new
    cleaned = cleaned.lower()
    # Try alias lookup first
    if cleaned in _FEATURE_ALIASES:
        return _FEATURE_ALIASES[cleaned]
    return cleaned.replace(" ", "_")


def _fallback_parse(rule_text: str) -> dict:
    """Heuristic rule parser that does not call any external API."""
    text_lower = rule_text.lower()

    action = "decline"
    for kw in _ACTION_KEYWORDS:
        if kw in text_lower:
            action = kw
            break

    features = []
    seen_features: set = set()

    # Two-pass approach:
    # Pass 1 – numeric comparisons:  <feature> {>=|<=|>|<} <number>
    #   Feature names: single snake_case token OR a two-word phrase of letters (e.g. "transaction amount")
    # Pass 2 – equality comparisons: <feature> {is|==} <word>
    numeric_pattern = re.compile(
        r"\b([a-zA-Z][a-zA-Z_]*(?:\s+[a-zA-Z][a-zA-Z_]*)?)\s*(>=|<=|>|<)\s*(\d+(?:\.\d+)?)\b",
        re.IGNORECASE,
    )
    equality_pattern = re.compile(
        r"\b([a-zA-Z][a-zA-Z_]{0,30}(?:\s+[a-zA-Z][a-zA-Z_]{0,30})?)\s+(?:is not|is|==)\s+([a-zA-Z]\w*)\b",
        re.IGNORECASE,
    )

    def _add_feature(raw_feature, op, raw_value):
        feature_name = _clean_feature(raw_feature)
        # Filter noise: too short, or contains stop-words indicating a bad match
        tokens = feature_name.replace("_", " ").split()
        if not feature_name or len(feature_name) < 2:
            return
        if any(t in _INNER_STOP_WORDS for t in tokens[1:]):
            # Stop-word mid-name usually means the regex overshot
            return
        if feature_name in seen_features:
            return
        seen_features.add(feature_name)
        # Normalise operator
        op_norm = _OPERATOR_MAP.get(op.lower(), op.lower())
        # Cast value
        try:
            value: object = int(raw_value)
        except (ValueError, TypeError):
            try:
                value = float(raw_value)
            except (ValueError, TypeError):
                value = raw_value
        features.append({"feature": feature_name, "operator": op_norm, "value": value})

    for m in numeric_pattern.finditer(rule_text):
        _add_feature(m.group(1), m.group(2), m.group(3))

    for m in equality_pattern.finditer(rule_text):
        _add_feature(m.group(1), "=", m.group(2))

    return {"features": features, "action": action}
