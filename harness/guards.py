from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any


@dataclass
class GuardDecision:
    allow: bool
    reason: str | None = None


_SAFE_INPUT_LIMIT = 3000
_SENSITIVE_FIELDS = (
    "api_key",
    "apikey",
    "auth",
    "authorization",
    "bearer",
    "password",
    "secret",
    "private_key",
    "token",
)

_INPUT_PATTERNS = (
    r"ignore previous instructions",
    r"ignore\\s+previous\\s+instructions",
    r"disregard\\s+safety",
    r"act as (?:an?|the) assistant",
    r"system prompt",
    r"system\\s+prompt",
    r"reveal hidden",
    r"override (?:the )?policy",
    r"bypass (?:policy|safety|controls|guard|restriction)",
    r"prompt injection",
    r"jailbreak",
    r"\brole\\s*:\\s*system\b",
    r"\brole\\s*:\\s*assistant\b",
    r"you are (?:not|now|free)\b",
    r"you are not",
    r"you are now",
    r"this is a security test",
)

_COMMAND_ABUSE_PATTERNS = (
    r"\b(?:python|bash|sh|powershell|pwsh|cmd|wget|curl|nc|ssh|scp|rsync|python3)\b",
    r"\$\(",
    r"&&",
    r"\|\|",
    r";",
    r"`",
    r"\b(?:sudo|su|doas|rm|rmdir|dd|mkfs|shutdown|reboot|chmod|chown|sudoers|del|format|netcat|nc)\b",
    r"/bin/\\w+",
    r"\b(?:eval|exec)\b",
)

_OUTPUT_OUTPUT_FORBIDDEN_PATTERNS = (
    r"\b(secret|api[_-]?key|passwd|authorization|bearer|private[_-]?key)\b",
    r"\b(password|token)\s*[:=]\s*['\"]?[^\s'\";,|&]{4,}",
)

_HARMLESS_TOOL_ARGUMENTS = {
    "calculator": (
        r"^[0-9+\-*/().\s]+$",
    ),
}


def _compile(patterns: tuple[str, ...]) -> tuple[re.Pattern, ...]:
    return tuple(re.compile(pattern, re.IGNORECASE) for pattern in patterns)


_COMPILED_INPUT_PATTERNS = _compile(_INPUT_PATTERNS)
_COMPILED_COMMAND_PATTERNS = _compile(_COMMAND_ABUSE_PATTERNS)


def _redact_sensitive_string(text: str, fields: tuple[str, ...] | None = None) -> str:
    fields = fields or _SENSITIVE_FIELDS
    lowered = text.lower()
    if not lowered:
        return text

    for field in fields:
        marker = f"{field}="
        if marker in lowered:
            idx = lowered.index(marker)
            start = idx + len(marker)
            tail = text[start:]
            end = len(tail)
            for delimiter in (" ", "\n", ";", "&", "|", ",", "\"", "'"):
                pos = tail.find(delimiter)
                if pos != -1:
                    end = min(end, pos)
            replacement = f"{field}=[REDACTED]"
            text = text[:idx] + replacement + tail[end:]
            lowered = text.lower()
            continue

        pattern = re.compile(rf'"?{re.escape(field)}"?\s*[:=]\s*["\']?[^"\',\s;|&`$()]+', re.IGNORECASE)
        text = pattern.sub(f'"{field}":"[REDACTED]"', text)
        lowered = text.lower()
    return text


def split_trusted_untrusted(payload: str) -> tuple[str, str]:
    cleaned = sanitize_text(payload)
    if "\n\n" in cleaned:
        trusted, _, untrusted = cleaned.partition("\n\n")
    elif "::USER::" in cleaned:
        trusted, _, untrusted = cleaned.partition("::USER::")
    else:
        trusted, untrusted = "", cleaned
    return trusted, untrusted


def sanitize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    cleaned = text.strip()
    if len(cleaned) > _SAFE_INPUT_LIMIT:
        cleaned = cleaned[:3000]
    return cleaned


def check_tool_request(arguments: dict[str, Any]) -> GuardDecision:
    if not isinstance(arguments, dict):
        return GuardDecision(False, reason="tool_arguments_non_dict")

    payload = sanitize_text(json.dumps(arguments, ensure_ascii=False, sort_keys=True))
    lowered = payload.lower()
    if _matches_abuse(lowered):
        return GuardDecision(False, reason="blocked_tool_pattern:command_abuse")

    payload = sanitize_text(str(arguments))
    lowered = payload.lower()
    for pattern in _COMPILED_INPUT_PATTERNS:
        if pattern.search(lowered):
            return GuardDecision(False, reason=f"blocked_tool_pattern:{pattern.pattern}")
    return GuardDecision(True, None)


def check_tool_request_with_tool(
    arguments: dict[str, Any],
    tool_name: str | None = None,
) -> GuardDecision:
    if not isinstance(arguments, dict):
        return GuardDecision(False, reason="tool_arguments_non_dict")

    if tool_name in _HARMLESS_TOOL_ARGUMENTS:
        pattern = _HARMLESS_TOOL_ARGUMENTS.get(tool_name)
        if not isinstance(pattern, tuple):
            pattern = ("",)
        if not re.match(pattern[0], str(arguments.get("expression", "")) if tool_name == "calculator" else sanitize_text(str(arguments))):
            return GuardDecision(False, reason="tool_expression_invalid" if tool_name == "calculator" else "tool_arguments_denied")
        expression = str(arguments.get("expression", "")).strip()
        if not expression:
            return GuardDecision(False, reason="tool_missing_argument:expression")
        if not re.match(r"^[0-9+\-*/().\s]+$", expression):
            return GuardDecision(False, reason="tool_expression_invalid")
        return GuardDecision(True, None)

    if tool_name == "run_tests":
        scope = arguments.get("scope")
        if scope is not None and not isinstance(scope, (str, type(None))):
            return GuardDecision(False, reason="tool_argument_type_invalid")
        if isinstance(scope, str) and scope and any(ch in scope for ch in (";", "&", "|", "`")):
            return GuardDecision(False, reason="tool_arguments_command_like")
        return GuardDecision(True, None)

    return check_tool_request(arguments)


def _matches_abuse(value: str) -> bool:
    for pattern in _COMPILED_COMMAND_PATTERNS:
        if pattern.search(value):
            return True
    return False


def check_tool_output(output: Any) -> GuardDecision:
    text = output if isinstance(output, str) else json.dumps(output, ensure_ascii=False, default=str)
    if len(text) > 2000:
        return GuardDecision(False, reason="tool_output_too_large")
    lowered = sanitize_text(str(text)).lower()
    for pattern in _COMPILED_INPUT_PATTERNS:
        if pattern.search(lowered):
            return GuardDecision(False, reason=f"tool_output_blocked:{pattern.pattern}")
    for pattern in _OUTPUT_OUTPUT_FORBIDDEN_PATTERNS:
        if re.search(pattern, lowered, re.IGNORECASE):
            return GuardDecision(False, reason="tool_output_blocked:forbidden_content")
    return GuardDecision(True, None)


def check_input_text(text: str) -> GuardDecision:
    lowered = sanitize_text(text).lower()
    for pattern in _COMPILED_INPUT_PATTERNS:
        if pattern.search(lowered):
            return GuardDecision(False, reason=f"blocked_input_pattern:{pattern.pattern}")
    if _matches_abuse(lowered):
        return GuardDecision(False, reason="blocked_input_pattern:command_abuse")
    return GuardDecision(True, None)


def redact_sensitive_args(text: Any, fields: list[str] | None = None) -> Any:
    if fields is None:
        fields = [str(field) for field in _SENSITIVE_FIELDS]

    if isinstance(text, dict):
        payload = {}
        for key, value in text.items():
            if isinstance(key, str) and key.lower() in {x.lower() for x in fields}:
                payload[key] = "[REDACTED]"
            else:
                payload[key] = redact_sensitive_args(value, list(fields))
        return payload

    if isinstance(text, list):
        return [redact_sensitive_args(item, list(fields)) for item in text]

    if not isinstance(text, str):
        return str(text)
    return _redact_sensitive_string(text, tuple(fields))


def check_output_text(text: str) -> GuardDecision:
    lowered = sanitize_text(text).lower()
    if _matches_abuse(lowered):
        return GuardDecision(False, reason="unsafe_output_detected")
    if "i cannot" in lowered and "assist" in lowered:
        return GuardDecision(True, None)
    if "unsafe" in lowered:
        return GuardDecision(False, reason="unsafe_output_detected")
    return GuardDecision(True, None)
