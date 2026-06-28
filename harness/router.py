from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


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
        }


_LOW_RISK_RE = [
    re.compile(r"\b(summarize|rewrite|improve|shorten|expand|translate|reformat|grammar)\b", re.I),
]

_TOOL_RE = [
    re.compile(r"\b(calculate|compute|math|ratio|sum|difference|equation|calculator|python|execute|run)\b", re.I),
    re.compile(r"\b(look up|check|query|sql|db|database|read file|write file|patch|edit)\b", re.I),
]

_GROUND_RE = [
    re.compile(r"\b(policy|manual|doc|documentation|which|what does|latest|recent|version|evidence|sources|cite)\b", re.I),
]

_REASON_RE = [
    re.compile(r"\b(analyze|analysis|compare|tradeoff|migration|design|architecture|evaluate|plan|optimi[sz]e|recommend)\b", re.I),
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


def classify_route(messages: list[dict[str, str]], response_schema: dict | None = None, route_override: str | None = None) -> RoutePolicy:
    user_text = _last_user_text(messages).strip().lower()

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
        elif any(pattern.search(user_text) for pattern in _HIGH_RISK_RE):
            route = Route.HIGH_RISK
        elif any(pattern.search(user_text) for pattern in _SIDE_RE):
            route = Route.SIDE_EFFECTING_ACTION
        elif any(pattern.search(user_text) for pattern in _TOOL_RE):
            route = Route.TOOL_REQUIRED
        elif any(pattern.search(user_text) for pattern in _GROUND_RE):
            route = Route.GROUNDED_QA
        elif any(pattern.search(user_text) for pattern in _REASON_RE):
            route = Route.CODE_OR_DATA
        elif any(pattern.search(user_text) for pattern in _LOW_RISK_RE):
            route = Route.DIRECT
        else:
            route = Route.DIRECT


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
            strict_schema=True,
            output_schema_required=True,
            allowed_tools=(),
            max_tool_calls_per_turn=0,
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
        )
    if route == Route.TOOL_REQUIRED:
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
            allowed_tools=("calculator", "time_now"),
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
    )
