from __future__ import annotations

import json
import re
from typing import Any


def extract_first_json(text: str) -> tuple[bool, Any | None, str]:
    stripped = text.strip()
    if not stripped:
        return False, None, "empty_text"
    try:
        return True, json.loads(stripped), "ok"
    except Exception:
        pass

    # Try to extract JSON object from fenced blocks or plain text wrappers.
    json_like = re.search(r"\{[\s\S]*\}", text)
    if not json_like:
        return False, None, "no_json_found"
    try:
        return True, json.loads(json_like.group(0)), "recovered"
    except Exception as exc:
        return False, None, f"json_recover_failed:{exc}"


def validate_schema(payload: Any, schema: dict[str, Any]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if not isinstance(schema, dict):
        return True, errors

    if schema.get("type") == "object":
        if not isinstance(payload, dict):
            return False, ["expected_object"]
        required = schema.get("required", [])
        for key in required:
            if key not in payload:
                errors.append(f"missing_required:{key}")
        properties = schema.get("properties", {})
        for key, spec in properties.items():
            if key in payload and "type" in spec:
                if not _validate_type(payload[key], spec["type"]):
                    errors.append(f"type_mismatch:{key}")
    return len(errors) == 0, errors


def _validate_type(value: Any, expected: str) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "number":
        return isinstance(value, (int, float))
    if expected == "integer":
        return isinstance(value, int)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    return True

