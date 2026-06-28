from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GuardDecision:
    allow: bool
    reason: str | None = None


_INJECTION_PATTERNS = [
    "ignore previous instructions",
    "jailbreak",
    "system prompt",
    "reveal hidden",
    "override policy",
    "sudo",
    "drop table",
]


def check_input_text(text: str) -> GuardDecision:
    lowered = text.lower()
    for pattern in _INJECTION_PATTERNS:
        if pattern in lowered:
            return GuardDecision(False, reason=f"blocked_input_pattern:{pattern}")
    return GuardDecision(True, None)


def check_output_text(text: str) -> GuardDecision:
    lowered = text.lower()
    if "i cannot" in lowered and "assist" in lowered:
        return GuardDecision(True, None)
    if "unsafe" in lowered:
        return GuardDecision(False, reason="unsafe_output_detected")
    return GuardDecision(True, None)

