"""Routing layer for production runtime with confidence metadata."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any


class Route(str, Enum):
    DIRECT = "direct"
    GROUNDED_QA = "grounded_qa"
    TOOL_REQUIRED = "tool_required"
    STRUCTURED_EXTRACTION = "structured_extraction"
    CODE_OR_DATA = "code_or_data"
    SIDE_EFFECTING_ACTION = "side_effecting_action"
    HIGH_RISK = "high_risk"


@dataclass
class RoutePolicy:
    route: Route
    use_retrieval: bool
    use_tools: bool
    strict_schema: bool
    require_verification: bool
    allow_reasoning: bool
    thinking_budget: str
    max_model_calls: int
    temperature: float
    max_new_tokens: int
    max_tool_calls_per_turn: int = 0
    allowed_tools: tuple[str, ...] = ()
    require_confirmation: bool = False
    cite_evidence: bool = False
    output_schema_required: bool = False
    confidence: float = 1.0
    confidence_gap: float = 0.0
    required_evidence_fields: tuple[str, ...] = ()
    hard_fail_errors: tuple[str, ...] = ()
    tool_sandbox_required: tuple[str, ...] = ()
    next_action: str = "proceed"
    route_metadata: dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.route_metadata is None:
            self.route_metadata = {}

    def as_dict(self) -> dict[str, object]:
        return {
            "route": self.route.value,
            "use_retrieval": self.use_retrieval,
            "use_tools": self.use_tools,
            "strict_schema": self.strict_schema,
            "require_verification": self.require_verification,
            "allow_reasoning": self.allow_reasoning,
            "thinking_budget": self.thinking_budget,
            "max_model_calls": self.max_model_calls,
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
            "max_tool_calls_per_turn": self.max_tool_calls_per_turn,
            "allowed_tools": list(self.allowed_tools),
            "require_confirmation": self.require_confirmation,
            "cite_evidence": self.cite_evidence,
            "output_schema_required": self.output_schema_required,
            "confidence": self.confidence,
            "confidence_gap": self.confidence_gap,
            "required_evidence_fields": list(self.required_evidence_fields),
            "hard_fail_errors": list(self.hard_fail_errors),
            "tool_sandbox_required": list(self.tool_sandbox_required),
            "next_action": self.next_action,
            "route_metadata": self.route_metadata,
        }


_LOW_RISK_RE = [
    re.compile(r"\b(summarize|rewrite|improve|shorten|expand|translate|reformat|grammar)\b", re.I),
]

_TOOL_RE = [
    re.compile(r"\b(calculate|compute|math|ratio|sum|difference|equation|calculator|python|execute|run)\b", re.I),
    re.compile(r"\b(look up|check|query|sql|db|database|read file|write file|patch|edit)\b", re.I),
]

_GROUND_RE = [
    re.compile(
        r"\b(policy|manual|doc|documentation|what does|latest|recent|version|evidence|sources|cite)\b",
        re.I,
    ),
]

_REASON_RE = [
    re.compile(
        r"\b(analyze|analysis|compare|tradeoff|migration|design|architecture|evaluate|plan|optimi[sz]e|recommend)\b",
        re.I,
    ),
]

_SIDE_RE = [
    re.compile(r"\b(cancel|refund|billing|delete|remove|transfer|provision|provisioning|deploy|ship|publish|mutate)\b", re.I),
]

_HIGH_RISK_RE = [
    re.compile(r"\b(legal|medical|financial|jailbreak|exploit|hack|bypass|evade|bypasses)\b", re.I),
]


def _last_user_text(messages: list[dict[str, str]]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return str(msg.get("content", ""))
    return ""


def _extract_evidence_fields(route_meta: dict[str, Any] | None) -> list[str]:
    if not isinstance(route_meta, dict):
        return []
    validator = route_meta.get("validator") or route_meta.get("manifests")
    if not isinstance(validator, dict):
        return []

    raw = validator.get("required_evidence_fields")
    if isinstance(raw, list):
        return [str(item) for item in raw]

    legacy = validator.get("validator_fields")
    if isinstance(legacy, list):
        return [str(item) for item in legacy]
    return []


def _extract_thresholds(route_meta: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(route_meta, dict):
        return {}
    policy = route_meta.get("policy")
    if isinstance(policy, dict):
        if isinstance(policy.get("thresholds"), dict):
            return dict(policy["thresholds"])
        # fallback: permit route-level threshold shorthand for compatibility
        if isinstance(route_meta.get("thresholds"), dict):
            return dict(route_meta["thresholds"])
    return dict(route_meta.get("thresholds", {})) if isinstance(route_meta.get("thresholds"), dict) else {}


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _extract_tool_sandbox_required(route_meta: dict[str, Any] | None) -> list[str]:
    if not isinstance(route_meta, dict):
        return []

    candidates: list[Any] = []
    policy = route_meta.get("policy")
    if isinstance(policy, dict):
        sandbox = policy.get("tool_sandbox_required")
        if sandbox is not None:
            candidates.append(sandbox)

    direct = route_meta.get("tool_sandbox_required")
    if direct is not None:
        candidates.append(direct)

    for candidate in candidates:
        if isinstance(candidate, str):
            return [candidate] if candidate.strip() else []
        if isinstance(candidate, (list, tuple, set)):
            return [str(item) for item in candidate if isinstance(item, str) and item.strip()]
    return []


def _extract_hard_fail_fields(route_meta: dict[str, Any] | None) -> list[str]:
    if not isinstance(route_meta, dict):
        return []
    validator = route_meta.get("validator") or route_meta.get("manifests")
    if not isinstance(validator, dict):
        return []
    raw = validator.get("hard_fail_errors")
    if isinstance(raw, list):
        return [str(item) for item in raw]
    return []


def _resolve_next_action(
    route: Route,
    confidence: float,
    confidence_gap: float,
    route_meta: dict[str, Any] | None,
    feature_level: str,
    advanced_router_enabled: bool,
) -> str:
    thresholds = (
        _extract_thresholds(route_meta)
        if advanced_router_enabled and feature_level == "hardening"
        else {}
    )
    ask_confidence = _to_float(
        thresholds.get("ask_confidence", thresholds.get("min_confidence")),
        0.50 if route == Route.DIRECT else 0.60,
    )
    ask_gap = _to_float(
        thresholds.get("ask_gap", thresholds.get("ambiguity_gap", thresholds.get("confidence_gap"))),
        0.08,
    )

    if feature_level == "hardening":
        ask_confidence = max(ask_confidence, 0.60)
        ask_gap = max(ask_gap, 0.06)

    if confidence < ask_confidence:
        return "ask_clarification"
    if feature_level == "hardening" and confidence_gap < ask_gap:
        return "ask_clarification"
    if route == Route.HIGH_RISK:
        return "ask_clarification"
    return "proceed"


def _resolve_route_metadata(
    route: str,
    route_manifest: dict[str, Any] | None,
    route_overrides: dict[str, Any] | None,
) -> dict[str, Any]:
    route_data: dict[str, Any] = {}
    if isinstance(route_manifest, dict):
        manifest_obj = route_manifest.get(route)
        if isinstance(manifest_obj, dict):
            route_data.update(manifest_obj)
        elif hasattr(manifest_obj, "model_dump"):
            route_data.update(manifest_obj.model_dump())
    if isinstance(route_overrides, dict) and isinstance(route_overrides.get(route), dict):
        route_data.update(route_overrides.get(route, {}))
    return route_data


def _apply_policy_overrides(
    route: Route,
    confidence: float,
    confidence_gap: float,
    route_meta: dict[str, Any],
    advanced_router_enabled: bool,
) -> tuple[float, float]:
    if not advanced_router_enabled:
        return confidence, confidence_gap

    policy = route_meta.get("policy")
    if isinstance(policy, dict):
        if "confidence" in policy:
            confidence = float(policy["confidence"])
        if "confidence_gap" in policy:
            confidence_gap = float(policy["confidence_gap"])

    thresholds = _extract_thresholds(route_meta)
    if thresholds:
        if "min_confidence" in thresholds:
            confidence = max(confidence, _to_float(thresholds["min_confidence"], 0.0))
        if "max_confidence" in thresholds:
            confidence = min(confidence, _to_float(thresholds["max_confidence"], 1.0))
        if "confidence" in thresholds:
            confidence = _to_float(thresholds["confidence"], confidence)
        if "confidence_gap" in thresholds:
            confidence_gap = _to_float(thresholds["confidence_gap"], confidence_gap)
    return confidence, confidence_gap

def classify_route(
    messages: list[dict[str, str]],
    response_schema: dict | None = None,
    route_override: str | None = None,
    route_manifest: dict[str, Any] | None = None,
    route_overrides: dict[str, Any] | None = None,
    feature_level: str = "basic",
    advanced_router_enabled: bool = False,
) -> RoutePolicy:
    user_text = _last_user_text(messages).strip().lower()

    tool_match_count = sum(1 for pattern in _TOOL_RE if pattern.search(user_text))
    grounded_match_count = sum(1 for pattern in _GROUND_RE if pattern.search(user_text))
    reason_match_count = sum(1 for pattern in _REASON_RE if pattern.search(user_text))
    low_risk_count = sum(1 for pattern in _LOW_RISK_RE if pattern.search(user_text))
    side_match_count = sum(1 for pattern in _SIDE_RE if pattern.search(user_text))
    high_risk_count = sum(1 for pattern in _HIGH_RISK_RE if pattern.search(user_text))

    if route_override:
        try:
            route = Route(route_override)
        except ValueError:
            route = None
    else:
        route = None

    if route is None:
        if response_schema is not None:
            route = Route.STRUCTURED_EXTRACTION
            confidence = 0.98
            confidence_gap = 0.0
        elif high_risk_count:
            route = Route.HIGH_RISK
            confidence = 0.8 + min(0.19, 0.02 * high_risk_count)
            confidence_gap = 0.04
        elif side_match_count:
            route = Route.SIDE_EFFECTING_ACTION
            confidence = 0.78 + min(0.2, 0.02 * side_match_count)
            confidence_gap = 0.06
        elif tool_match_count:
            route = Route.TOOL_REQUIRED
            confidence = 0.72 + min(0.23, 0.03 * tool_match_count)
            confidence_gap = 0.05
        elif grounded_match_count:
            route = Route.GROUNDED_QA
            confidence = 0.69 + min(0.24, 0.03 * grounded_match_count)
            confidence_gap = 0.08
        elif reason_match_count:
            route = Route.CODE_OR_DATA
            confidence = 0.66 + min(0.28, 0.025 * reason_match_count)
            confidence_gap = 0.07
        elif low_risk_count:
            route = Route.DIRECT
            confidence = 0.6 + min(0.3, 0.03 * low_risk_count)
            confidence_gap = 0.10
        else:
            route = Route.DIRECT
            confidence = 0.52
            confidence_gap = 0.08
    else:
        confidence = 0.75
        confidence_gap = 0.0

    if feature_level == "hardening":
        confidence = max(0.0, min(0.99, confidence))

    route_meta = _resolve_route_metadata(route.value, route_manifest, route_overrides)
    confidence, confidence_gap = _apply_policy_overrides(
        route,
        confidence,
        confidence_gap,
        route_meta,
        advanced_router_enabled=advanced_router_enabled,
    )
    required_evidence = _extract_evidence_fields(route_meta)
    hard_fail = _extract_hard_fail_fields(route_meta)
    sandbox_required = tuple(_extract_tool_sandbox_required(route_meta))
    next_action = _resolve_next_action(
        route,
        confidence,
        confidence_gap,
        route_meta,
        feature_level,
        advanced_router_enabled=advanced_router_enabled,
    )
    confidence = round(confidence, 3)
    confidence_gap = round(confidence_gap, 3)

    if route == Route.STRUCTURED_EXTRACTION:
        return RoutePolicy(
            route=route,
            use_retrieval=False,
            use_tools=False,
            strict_schema=True,
            require_verification=False,
            allow_reasoning=False,
            thinking_budget="low",
            max_model_calls=1,
            temperature=0.0,
            max_new_tokens=512,
            output_schema_required=True,
            allowed_tools=(),
            max_tool_calls_per_turn=0,
            confidence=confidence,
            confidence_gap=confidence_gap,
            required_evidence_fields=tuple(required_evidence),
            hard_fail_errors=tuple(hard_fail),
            tool_sandbox_required=(),
            next_action=next_action,
            route_metadata=route_meta,
        )

    if route == Route.GROUNDED_QA:
        return RoutePolicy(
            route=route,
            use_retrieval=True,
            use_tools=False,
            strict_schema=True,
            require_verification=True,
            allow_reasoning=False,
            thinking_budget="medium",
            max_model_calls=2,
            temperature=0.2,
            max_new_tokens=900,
            max_tool_calls_per_turn=0,
            allowed_tools=(),
            cite_evidence=True,
            confidence=confidence,
            confidence_gap=confidence_gap,
            required_evidence_fields=tuple(required_evidence),
            hard_fail_errors=tuple(hard_fail),
            tool_sandbox_required=(),
            next_action=next_action,
            route_metadata=route_meta,
        )

    if route == Route.SIDE_EFFECTING_ACTION:
        return RoutePolicy(
            route=route,
            use_retrieval=True,
            use_tools=True,
            strict_schema=False,
            require_verification=True,
            allow_reasoning=True,
            thinking_budget="high",
            max_model_calls=4,
            temperature=0.0,
            max_new_tokens=900,
            max_tool_calls_per_turn=2,
            allowed_tools=("calculator",),
            require_confirmation=True,
            cite_evidence=True,
            confidence=confidence,
            confidence_gap=confidence_gap,
            required_evidence_fields=tuple(required_evidence),
            hard_fail_errors=tuple(hard_fail),
            tool_sandbox_required=sandbox_required if any(item == "calculator" for item in sandbox_required) else (),
            next_action=next_action,
            route_metadata=route_meta,
        )

    if route == Route.TOOL_REQUIRED:
        allowed_tools = ("calculator", "time_now", "run_tests")
        return RoutePolicy(
            route=route,
            use_retrieval=False,
            use_tools=True,
            strict_schema=True,
            require_verification=False,
            allow_reasoning=False,
            thinking_budget="low",
            max_model_calls=3,
            temperature=0.0,
            max_new_tokens=900,
            max_tool_calls_per_turn=3,
            allowed_tools=allowed_tools,
            confidence=confidence,
            confidence_gap=confidence_gap,
            required_evidence_fields=tuple(required_evidence),
            hard_fail_errors=tuple(hard_fail),
            tool_sandbox_required=sandbox_required,
            next_action=next_action,
            route_metadata=route_meta,
        )

    if route == Route.CODE_OR_DATA:
        return RoutePolicy(
            route=route,
            use_retrieval=True,
            use_tools=True,
            strict_schema=False,
            require_verification=True,
            allow_reasoning=True,
            thinking_budget="high",
            max_model_calls=3,
            temperature=0.1,
            max_new_tokens=1200,
            max_tool_calls_per_turn=4,
            allowed_tools=("calculator", "time_now"),
            cite_evidence=True,
            confidence=confidence,
            confidence_gap=confidence_gap,
            required_evidence_fields=tuple(required_evidence),
            hard_fail_errors=tuple(hard_fail),
            tool_sandbox_required=sandbox_required,
            next_action=next_action,
            route_metadata=route_meta,
        )

    if route == Route.HIGH_RISK:
        return RoutePolicy(
            route=route,
            use_retrieval=True,
            use_tools=True,
            strict_schema=False,
            require_verification=True,
            allow_reasoning=True,
            thinking_budget="high",
            max_model_calls=4,
            temperature=0.0,
            max_new_tokens=1200,
            max_tool_calls_per_turn=2,
            allowed_tools=("calculator", "time_now"),
            require_confirmation=True,
            cite_evidence=True,
            confidence=confidence,
            confidence_gap=confidence_gap,
            required_evidence_fields=tuple(required_evidence),
            hard_fail_errors=tuple(hard_fail),
            tool_sandbox_required=sandbox_required,
            next_action=next_action,
            route_metadata=route_meta,
        )

    return RoutePolicy(
        route=Route.DIRECT,
        use_retrieval=False,
        use_tools=False,
        strict_schema=False,
        require_verification=False,
        allow_reasoning=False,
        thinking_budget="low",
        max_model_calls=1,
        temperature=0.3,
        max_new_tokens=700,
        max_tool_calls_per_turn=0,
        allowed_tools=(),
        confidence=confidence,
        confidence_gap=confidence_gap,
        required_evidence_fields=tuple(required_evidence),
        hard_fail_errors=tuple(hard_fail),
        tool_sandbox_required=sandbox_required,
        next_action=next_action,
        route_metadata=route_meta,
    )
