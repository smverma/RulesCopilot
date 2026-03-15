"""
rule_parser.py
Parses fraud rule text into structured JSON using the Google Gemini API.
"""

import json
import re
import os
from google import genai
from google.genai import types

# Configure the Gemini client once at module load time.
_API_KEY = os.environ.get("GEMINI_API_KEY", "")
_client: genai.Client | None = None
if _API_KEY:
    _client = genai.Client(api_key=_API_KEY)

_MODEL_NAME = "gemini-2.0-flash"

_PARSE_PROMPT_TEMPLATE = """Convert the following fraud rule into structured JSON.

Rule:
"{rule_text}"

Output ONLY valid JSON in exactly this format (no markdown, no explanation):
{{
  "features": [
    {{"feature": "<feature_name>", "operator": "<operator>", "value": "<value>"}}
  ],
  "action": "<action>"
}}

Rules for the JSON:
- "features" is a list of conditions found in the rule.
- "operator" can be ">", "<", ">=", "<=", "=", "!=", "in", "not in".
- "action" is one of: "decline", "approve", "block", "review", "flag".
- Use snake_case for feature names (e.g. "transaction_amount", "device_age").
- If a value is numeric, use a number type; otherwise use a string.
"""


def parse_rule(rule_text: str) -> dict:
    """
    Send *rule_text* to Gemini and return a structured dict with keys:
      - features: list of {feature, operator, value}
      - action: str
    Falls back to an empty structure if the API call fails or the key is missing.
    """
    if not _API_KEY or _client is None:
        return _fallback_parse(rule_text)

    try:
        prompt = _PARSE_PROMPT_TEMPLATE.format(rule_text=rule_text)
        response = _client.models.generate_content(
            model=_MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            ),
        )
        raw = response.text.strip()

        # Strip optional markdown code fences
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        parsed = json.loads(raw)
        return {
            "features": parsed.get("features", []),
            "action": parsed.get("action", "unknown"),
        }
    except Exception:  # noqa: BLE001
        return _fallback_parse(rule_text)


# ---------------------------------------------------------------------------
# Lightweight regex-based fallback (used when no API key is present)
# ---------------------------------------------------------------------------

_ACTION_KEYWORDS = ["decline", "approve", "block", "review", "flag"]

_FEATURE_PATTERNS = [
    (r"amount\s*([><=!]+)\s*([\d,]+)", "transaction_amount"),
    (r"velocity\s*([><=!]+)\s*(\d+)", "velocity"),
    (r"device(?:_age)?\s+is\s+(new|old)", "device_age"),
    (r"device_age\s*([<>=!]+)\s*(\d+)", "device_age"),
    (r"country\s+is\s+(high\s+risk|low\s+risk|\w+)", "country"),
    (r"ip_country\s*([!=]+)\s*billing_country", "ip_country"),
    (r"card_type\s+is\s+(\w+)", "card_type"),
    (r"account\s+age\s*([<>=!]+)\s*(\d+)", "account_age"),
    (r"user\s+account\s+age\s*([<>=!]+)\s*(\d+)", "account_age"),
]


def _fallback_parse(rule_text: str) -> dict:
    """Simple regex-based parser used when Gemini is unavailable."""
    text_lower = rule_text.lower()

    action = "unknown"
    for kw in _ACTION_KEYWORDS:
        if kw in text_lower:
            action = kw
            break

    features = []
    for pattern, feature_name in _FEATURE_PATTERNS:
        match = re.search(pattern, text_lower)
        if match:
            groups = match.groups()
            if len(groups) == 2:
                operator, value = groups
                # Try to coerce value to a number
                try:
                    value = float(value.replace(",", ""))
                    if value.is_integer():
                        value = int(value)
                except (ValueError, AttributeError):
                    pass
                features.append(
                    {"feature": feature_name, "operator": operator.strip(), "value": value}
                )
            elif len(groups) == 1:
                features.append(
                    {"feature": feature_name, "operator": "=", "value": groups[0].strip()}
                )

    return {"features": features, "action": action}
