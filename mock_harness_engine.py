#!/usr/bin/env python3
"""Mock AI Harness Engine in "teaching mode".

This file is intentionally explicit and verbose.
It simulates the full Mermaid policy flow in small callable pieces and prints
every phase + sub-step as it runs.

Design principle:
1. Every phase is decomposed into micro-steps.
2. Every micro-step has a matching method.
3. Every method call that advances the workflow is logged and printed.
4. No real network or external LLM/tool calls are performed.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml
except Exception as exc:  # pragma: no cover
    # The script must fail early and clearly when YAML parsing is unavailable.
    raise SystemExit("PyYAML is required for this mock to run: pip install pyyaml") from exc


# ----------------------------------------------------------------------------
# Mock data and static defaults used by the teaching harness.
# ----------------------------------------------------------------------------

MOCK_MATCHUPS = [
    {
        "date": "2026-06-16",
        "home": "Texas Rangers",
        "away": "Houston Astros",
        "score": "8-5",
        "status": "final",
        "source": "MockSportsIndexService",
    },
    {
        "date": "2026-06-09",
        "home": "Houston Astros",
        "away": "Texas Rangers",
        "score": "5-3",
        "status": "final",
        "source": "MockSportsIndexService",
    },
]

DEFAULT_MANIFEST_PATH = "mock_harness_route.yaml"
CORE_PHASES = [
    "inspect_query",
    "route_classify",
    "permission_check",
    "extract_slots",
    "build_plan",
    "execute_tools",
    "summarize",
    "validate",
]
HELP_TEXT = """
Mock AI Harness Engine - teaching help section
=============================================

The simulator mirrors a policy runtime with explicit sub-steps.

1) bootstrap phase
   - initialize run state and trace container
   - validate manifest path
   - read manifest file from disk
   - parse YAML
   - validate manifest mapping
   - load budgets
   - build route catalog

2) main entry and input wiring
   - CLI input capture (query + manifest path)
   - run entry parse_request

3) inspect_query phase
   - read raw query
   - normalize query and tokenize
   - build deterministic profile
   - optionally run mock structured intent parser (LLM sub-step)
   - merge deterministic + LLM profiles

2) route_classify phase
   - iterate all routes
   - compute heuristic score (phrases, intents, negatives)
   - compute semantic score (token overlap stub)
   - combine scores
   - sort + top-k
   - detect ambiguity / low confidence
   - optionally rerank with mock LLM branch
   - apply fallback route if needed
   - decide next action

3) decision fan-out
   - ask_clarification branch
   - meta_route branch
   - permission_check branch

4) permission_check phase
   - collect required permissions
   - run-tests branch (stub path)
   - add execute_tools if tools are needed
   - detect missing permissions

5) extract_slots phase
   - load slot extractor prompt
   - check if mock LLM extractor should run
   - run LLM extractor or deterministic fallback
   - merge and normalize slots
   - check required slots

6) build_plan phase
   - enumerate route tools
   - construct ordered plan
   - branch to summarize if no plan

7) execute_tools phase
   - per-call loop
   - budget check before each call
   - dispatch mock tool
   - collect output/success/errors
   - iterate until no more calls

8) summarize phase
   - route-specific finalization
   - optional mock LLM summarizer
   - deterministic fallback summarizer

9) validate phase
   - validator presence check
   - required evidence check (box score + score)
   - pass/fail branch
   - escalate to ask_clarification on failure

10) final report
   - status / final_response
   - selected route and next action
   - evidence + budgets
   - full trace list
"""


@dataclass
class TraceEvent:
    """One tiny trace row for one micro-step.

    Every row is a user-visible teaching artifact.
    """

    ts: str
    phase: str
    step: str
    actor: str
    action: str
    input: Any
    output: Any
    next_action: str
    explanation: str
    tool_calls_so_far: int
    attempt: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts": self.ts,
            "phase": self.phase,
            "step": self.step,
            "actor": self.actor,
            "action": self.action,
            "input": self.input,
            "output": self.output,
            "next_action": self.next_action,
            "explanation": self.explanation,
            "tool_calls_so_far": self.tool_calls_so_far,
            "attempt": self.attempt,
        }


class MockHarnessEngine:
    """Micro-stepped mock harness.

    This class intentionally mirrors a control-plane harness:
    parse -> classify -> check permissions -> extract slots -> plan -> execute -> summarize -> validate -> final.
    """

    def __init__(self, manifest_path: str):
        # 1) Keep the path for trace visibility.
        self.manifest_path = manifest_path

        # 2) Persist trace events as the only source of output truth.
        self.phase_trace: List[TraceEvent] = []

        # 2b) Counter starts at zero before first bootstrap trace emit.
        self.tool_calls_used = 0

        # 3) Shared runtime state used by phase methods.
        self.phase_state: Dict[str, Any] = {
            "attempt": 1,
            "manifest_path": manifest_path,
            "route": None,
            "query_profile": {},
            "slots": {},
            "plan": [],
            "tool_outputs": [],
            "evidence": {},
            "missing_permissions": [],
            "validation": {"passed": False, "missing_fields": []},
            "final_response": None,
            "final_status": "in_progress",
            "next_action": "",
        }

        # 4) Emit bootstrap sub-steps before any main pipeline phase.
        self._emit(
            phase="bootstrap",
            step="bootstrap.init_state",
            actor="HarnessRuntime",
            action="initialize run state and trace container",
            inp={"manifest_path": manifest_path},
            out={"attempt": self.phase_state["attempt"]},
            next_action="bootstrap.load_manifest_path",
            explanation="Everything starts with local runtime state and output tracing setup.",
        )

        # 5) Load manifest, parse YAML, and normalize.
        self.manifest = self._load_manifest(manifest_path)
        self._emit(
            phase="bootstrap",
            step="bootstrap.manifest_loaded",
            actor="HarnessRuntime",
            action="manifest loaded and normalized",
            inp={"manifest_path": manifest_path},
            out={"manifest_keys": sorted(list(self.manifest.keys()))},
            next_action="bootstrap.load_budgets",
            explanation="Manifest keys are now known; routing and validation can begin.",
        )

        # 6) Read budgets from manifest for traceable guard rails.
        self.budgets = self._build_budgets()
        self._emit(
            phase="bootstrap",
            step="bootstrap.budgets_ready",
            actor="HarnessRuntime",
            action="set runtime budgets from manifest",
            inp={"manifest_budgets": self.manifest.get("budgets", {})},
            out={"budgets": self.budgets},
            next_action="bootstrap.load_route_catalog",
            explanation="Tool-call/attempt limits are fixed before route scoring.",
        )

        # 8) Build route catalog once.
        self.route_catalog = self._build_route_catalog()
        self._emit(
            phase="bootstrap",
            step="bootstrap.route_catalog_ready",
            actor="PolicyRouter",
            action="prepare route catalog for scoring stage",
            inp={"catalog_size": len(self.route_catalog)},
            out={"route_ids": [route.get("id") for route in self.route_catalog]},
            next_action="main.parse_request",
            explanation="Routes are ready before the first main phase runs.",
        )

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _utc_now() -> str:
        """Return an ISO timestamp used by every trace row."""
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _safe_json(value: Any) -> str:
        """Serialize any object safely for human-readable printing."""
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        except TypeError:
            return json.dumps(str(value), ensure_ascii=False)

    def _emit(
        self,
        phase: str,
        step: str,
        actor: str,
        action: str,
        inp: Any,
        out: Any,
        next_action: str = "",
        explanation: str = "",
    ) -> None:
        """Record and print one micro-step trace row.

        The print layout is intentionally repetitive so beginners can visually map
        each step to the Mermaid node names.
        """
        event = TraceEvent(
            ts=self._utc_now(),
            phase=phase,
            step=step,
            actor=actor,
            action=action,
            input=inp,
            output=out,
            next_action=next_action,
            explanation=explanation,
            tool_calls_so_far=self.tool_calls_used,
            attempt=self.phase_state["attempt"],
        )
        self.phase_trace.append(event)

        print(f"[{event.ts}] phase={phase} step={step} actor={actor}")
        print(f"  action: {action}")
        if inp is not None:
            print(f"  input : {self._safe_json(inp)}")
        print(f"  output: {self._safe_json(out)}")
        if next_action:
            print(f"  next  : {next_action}")
        if explanation:
            print(f"  why   : {explanation}")
        print("-" * 90)

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize spacing/case for deterministic matching."""
        return re.sub(r"\s+", " ", text.strip().lower())

    @staticmethod
    def _tokens(text: str) -> List[str]:
        """Split into word tokens and remove punctuation."""
        cleaned = re.sub(r"[^\w]+", " ", text.lower(), flags=re.UNICODE)
        return [token for token in cleaned.split() if token]

    @staticmethod
    def _token_set(text: str) -> set[str]:
        """Return a unique token set for quick membership operations."""
        return set(MockHarnessEngine._tokens(text))

    @staticmethod
    def _read_file_text(path: str) -> str:
        """Load text from a local file path."""
        return Path(path).read_text(encoding="utf-8")

    @staticmethod
    def _contains_any_token(tokens: set[str], terms: List[str]) -> bool:
        """Helper for token-based phrase checks."""
        normalized = {t.lower() for t in tokens}
        return any(term.lower() in normalized for term in terms)

    # ------------------------------------------------------------------
    # Manifest and route loading
    # ------------------------------------------------------------------

    def _load_manifest(self, path: str) -> Dict[str, Any]:
        """Load the YAML manifest and enforce a dict shape."""
        path_exists = Path(path).exists()
        self._emit(
            phase="bootstrap",
            step="bootstrap.validate_manifest_path",
            actor="HarnessRuntime",
            action="check manifest file exists",
            inp={"manifest_path": path},
            out={"exists": bool(path_exists)},
            next_action="bootstrap.read_manifest",
            explanation="The loader must prove the manifest path is valid before parse.",
        )
        if not path_exists:
            raise SystemExit(f"Manifest file missing: {path}")

        self._emit(
            phase="bootstrap",
            step="bootstrap.read_manifest",
            actor="HarnessRuntime",
            action="read manifest file bytes",
            inp={"manifest_path": path},
            out={"file_size_bytes": Path(path).stat().st_size},
            next_action="bootstrap.parse_manifest_yaml",
            explanation="File read happens before YAML parsing and validation.",
        )
        raw = self._read_file_text(path)

        self._emit(
            phase="bootstrap",
            step="bootstrap.parse_manifest_yaml",
            actor="HarnessRuntime",
            action="parse YAML payload",
            inp={"manifest_path": path},
            out={"raw_preview": raw[:120]},
            next_action="bootstrap.normalize_manifest",
            explanation="The YAML text is converted to native data structures.",
        )
        loaded = yaml.safe_load(raw)
        if not isinstance(loaded, dict):
            raise RuntimeError("Manifest must be a YAML mapping.")

        self._emit(
            phase="bootstrap",
            step="bootstrap.normalize_manifest",
            actor="HarnessRuntime",
            action="validate manifest structure",
            inp={"is_mapping": isinstance(loaded, dict)},
            out={"manifest_route": loaded.get("route", "<missing>")},
            next_action="bootstrap.manifest_loaded",
            explanation="Normalization guarantees downstream readers can safely index manifest fields.",
        )
        return loaded

    def _build_budgets(self) -> Dict[str, int]:
        """Read budgets from manifest and fill defaults."""
        manifest_budgets = self.manifest.get("budgets", {}) if isinstance(self.manifest.get("budgets", {}), dict) else {}
        return {
            "reasoning_steps_max": int(manifest_budgets.get("reasoning_steps_max", 12)),
            "tool_calls_max": int(manifest_budgets.get("tool_calls_max", 8)),
            "patch_attempt_max": int(manifest_budgets.get("patch_attempt_max", 0)),
            "retries_max": int(manifest_budgets.get("retries_max", 3)),
        }

    def _build_route_catalog(self) -> List[Dict[str, Any]]:
        """Build the route catalog used by route_classify.

        If the manifest declares a route_catalog list, use that.
        Otherwise create a deterministic teaching catalog.
        """
        declared = self.manifest.get("route_catalog")
        if isinstance(declared, list) and declared:
            return declared

        # Explicit sports fallback route derived from the mock manifest.
        manifest_route = self.manifest.get("route", "sports_score_lookup")
        manifest_tools = list((self.manifest.get("actors", {}).get("tools", {}) or {}).keys()) or [
            "lookup_team_matchups",
            "fetch_latest_box_score",
        ]
        manifest_required_perms = self.manifest.get("permissions", {}).get("required", [])
        if not isinstance(manifest_required_perms, list):
            manifest_required_perms = []

        sports_route = {
            "id": manifest_route,
            "title": "sports score lookup",
            "description": "Return a final score result for team vs team lookup queries.",
            "phrase_terms": ["lookup", "final", "who", "won", "score", "result", "matchup"],
            "intent_terms": ["score", "final", "played", "mlb", "baseball", "game"],
            "negative_terms": ["deploy", "email", "password"],
            "required_permissions": list(
                dict.fromkeys(
                    [
                        *manifest_required_perms,
                        "inspect_query",
                        "route_classify",
                        "permission_check",
                        "extract_slots",
                        "build_plan",
                        "summarize",
                        "validate",
                        "execute_tools",
                    ]
                )
            ),
            "required_slots": ["team_a", "team_b"],
            "tools": manifest_tools,
            "validator_required_fields": ["box_score", "score"],
        }
        ask_route = {
            "id": "ask_clarification",
            "title": "ask clarification",
            "description": "Collect missing information and stop before tool execution.",
            "required_permissions": [],
            "tools": [],
            "validator_required_fields": [],
        }
        meta_route = {
            "id": "meta_route",
            "title": "temporary plan route",
            "description": "Generate temporary route plan and re-enter flow.",
            "required_permissions": [],
            "tools": [],
            "validator_required_fields": [],
        }
        fallback_route = {
            "id": "fallback_unknown",
            "title": "generic fallback",
            "description": "Generic fallback when no route is a clean match.",
            "required_permissions": [],
            "tools": [],
            "validator_required_fields": [],
        }
        return [sports_route, ask_route, meta_route, fallback_route]

    def _get_route(self, route_id: str) -> Optional[Dict[str, Any]]:
        """Find a route by id from the current catalog."""
        for route in self.route_catalog:
            if route.get("id") == route_id:
                return route
        return None

    # ------------------------------------------------------------------
    # Phase 1: inspect_query
    # ------------------------------------------------------------------

    def _inspect_query_read_raw(self, query: str) -> str:
        """Sub-step: read raw query string."""
        raw_query = str(query)
        self._emit(
            phase="inspect_query",
            step="inspect_query.read_raw_query",
            actor="InputReader",
            action="capture raw user query",
            inp={"query_from_cli": raw_query},
            out={"raw_query": raw_query},
            next_action="inspect_query.normalize",
            explanation="Every run starts by saving the exact user text.",
        )
        return raw_query

    def _inspect_query_normalize(self, raw_query: str) -> Tuple[str, List[str], set[str]]:
        """Sub-step: normalize, tokenize."""
        normalized = self._normalize(raw_query)
        tokens = self._tokens(raw_query)
        token_set = set(tokens)
        self._emit(
            phase="inspect_query",
            step="inspect_query.normalize_and_tokenize",
            actor="InputNormalizer",
            action="normalize query text and generate tokens",
            inp={"raw_query": raw_query},
            out={"normalized": normalized, "tokens": tokens, "token_set": sorted(token_set)},
            next_action="inspect_query.build_profile",
            explanation="Lower/canonical tokens make deterministic matching stable.",
        )
        return normalized, tokens, token_set

    def _inspect_query_build_deterministic_profile(self, raw_query: str, normalized: str, tokens: List[str], token_set: set[str]) -> Dict[str, Any]:
        """Sub-step: build deterministic profile from keyword presence."""
        profile = {
            "query": raw_query,
            "raw_query": raw_query,
            "normalized_query": normalized,
            "language_hint": "en",
            "token_count": len(tokens),
            "has_score_term": "score" in token_set or "result" in token_set,
            "has_final_term": "final" in token_set,
            "has_team_hints": ("rangers" in token_set and "astros" in token_set),
            "wants_output": "score" in token_set or "result" in token_set or "lookup" in token_set,
            "wants_final": "final" in token_set,
            "query_length": len(raw_query),
        }
        self._emit(
            phase="inspect_query",
            step="inspect_query.deterministic_profile",
            actor="InputProfiler",
            action="build deterministic profile",
            inp={"tokens": tokens, "normalized": normalized},
            out=profile,
            next_action="inspect_query.llm_profile_check",
            explanation="Deterministic profile is always available and never relies on LLM.",
        )
        return profile

    def _inspect_query_llm_profile_enabled(self) -> bool:
        """Sub-step: decide whether mock structured parser should run."""
        enabled = bool(self.manifest.get("prompts", {}).get("inspect_query"))
        self._emit(
            phase="inspect_query",
            step="inspect_query.llm_profile_enabled",
            actor="PolicyRouter",
            action="check if inspect_query prompt exists",
            inp={"has_prompt": enabled},
            out={"enabled": enabled},
            next_action="inspect_query.structured_intent",
            explanation="If no prompt is available, deterministic path only is used.",
        )
        return enabled

    def _inspect_query_llm_profile(self, query: str, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Sub-step: call the mock LLM structured intent parser."""
        self.tool_calls_used += 1
        parsed = {
            "intent": "sports_score_lookup" if profile.get("has_score_term") and profile.get("has_team_hints") else "unknown",
            "teams_detected": ["Texas Rangers", "Houston Astros"] if profile.get("has_team_hints") else [],
            "sport_hint": "mlb" if profile.get("has_team_hints") or ("mlb" in profile.get("normalized_query", "")) else None,
            "needs_final": profile.get("has_final_term", False),
            "wants_score": profile.get("has_score_term", False),
            "raw_query": query,
            "language_hint": "en",
            "confidence": 0.94 if profile.get("has_team_hints") else 0.40,
        }
        self._emit(
            phase="inspect_query",
            step="inspect_query.structured_intent_json",
            actor="MockLLM",
            action="parse intent JSON from query",
            inp={
                "model": "mock-llm",
                "prompt": (self.manifest.get("prompts", {}) or {}).get("inspect_query", ""),
                "query": query,
            },
            out=parsed,
            next_action="inspect_query.merge_profiles",
            explanation="Mock structured parser returns route_hint, team hints, and confidence.",
        )
        return parsed

    def _inspect_query_skip_llm(self) -> Dict[str, Any]:
        """Sub-step: stub when LLM intent parser is disabled."""
        skipped = {
            "status": "skipped",
            "reason": "inspect_query prompt missing in manifest",
            "intent": "unknown",
            "teams_detected": [],
            "sport_hint": None,
            "needs_final": False,
            "wants_score": False,
            "raw_query": None,
            "language_hint": "en",
            "confidence": 0.0,
        }
        self._emit(
            phase="inspect_query",
            step="inspect_query.structured_intent_skipped",
            actor="PolicyRouter",
            action="stub non-LLM path",
            inp={"inspect_query_prompt_present": False},
            out=skipped,
            next_action="inspect_query.merge_profiles",
            explanation="Deterministic route still continues without mock LLM parsing.",
        )
        return skipped

    def _inspect_query_merge_profiles(self, deterministic: Dict[str, Any], llm_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Sub-step: merge deterministic + mock LLM profile dictionaries."""
        merged = dict(deterministic)
        merged["llm_intent_hint"] = llm_profile.get("intent")
        merged["llm_confidence"] = llm_profile.get("confidence", 0.0)
        merged["teams_detected"] = llm_profile.get("teams_detected", deterministic.get("teams_detected", []))
        if llm_profile.get("teams_detected"):
            merged["team_a"] = llm_profile["teams_detected"][0]
            merged["team_b"] = llm_profile["teams_detected"][1] if len(llm_profile["teams_detected"]) > 1 else None
        self._emit(
            phase="inspect_query",
            step="inspect_query.merge_profiles",
            actor="PolicyRouter",
            action="merge deterministic + LLM profile",
            inp={"deterministic": deterministic, "llm_profile": llm_profile},
            out=merged,
            next_action="route_classify",
            explanation="This merged profile drives every downstream phase.",
        )
        return merged

    def _phase_inspect_query(self, query: str) -> Dict[str, Any]:
        """Run all inspect_query sub-steps and return merged profile."""
        raw = self._inspect_query_read_raw(query)
        normalized, tokens, token_set = self._inspect_query_normalize(raw)
        deterministic_profile = self._inspect_query_build_deterministic_profile(raw, normalized, tokens, token_set)
        if self._inspect_query_llm_profile_enabled():
            llm_profile = self._inspect_query_llm_profile(raw, deterministic_profile)
        else:
            llm_profile = self._inspect_query_skip_llm()
        merged = self._inspect_query_merge_profiles(deterministic_profile, llm_profile)
        self.phase_state["query_profile"] = merged
        return merged

    # ------------------------------------------------------------------
    # Phase 2: route_classify
    # ------------------------------------------------------------------

    def _route_classify_iterate_routes(self, query_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Sub-step: iterate route list and score each candidate."""
        candidates: List[Dict[str, Any]] = []
        for route in self.route_catalog:
            heuristic = self._route_classify_heuristic_score(route, query_profile)
            semantic = self._route_classify_semantic_score(route, query_profile)
            combined = self._route_classify_combined_score(route, heuristic, semantic)
            candidates.append(
                {
                    "route_id": route.get("id"),
                    "title": route.get("title", ""),
                    "heuristic": heuristic,
                    "semantic": semantic,
                    "combined": combined,
                    "route": route,
                }
            )
        sorted_candidates = sorted(candidates, key=lambda item: item["combined"], reverse=True)
        self._emit(
            phase="route_classify",
            step="route_classify.iterate_and_score",
            actor="LayeredRouter",
            action="compute candidate score for every route",
            inp={"route_count": len(self.route_catalog)},
            out=sorted_candidates,
            next_action="route_classify.top_k",
            explanation="The route list is fully iterated to keep scoring deterministic and explainable.",
        )
        return sorted_candidates

    def _route_classify_heuristic_score(self, route: Dict[str, Any], query_profile: Dict[str, Any]) -> float:
        """Sub-step: heuristic phrase and intent scoring."""
        query = query_profile.get("normalized_query", "")
        token_set = self._token_set(query)
        phrase_terms = [term.lower() for term in route.get("phrase_terms", [])]
        intent_terms = [term.lower() for term in route.get("intent_terms", [])]
        negative_terms = [term.lower() for term in route.get("negative_terms", [])]

        phrase_hit = sum(1 for term in phrase_terms if term in token_set)
        intent_hit = sum(1 for term in intent_terms if term in token_set)
        negative_hit = sum(1 for term in negative_terms if term in token_set)

        score = float(phrase_hit * 0.35 + intent_hit * 0.25 - negative_hit * 0.30)
        bounded = max(0.0, min(score, 1.0))
        self._emit(
            phase="route_classify",
            step="route_classify.heuristic_score",
            actor="LayeredRouter",
            action="phrase + intent matching score",
            inp={"route": route.get("id"), "query": query_profile.get("normalized_query")},
            out={"heuristic_raw": score, "heuristic_clamped": bounded},
            next_action="route_classify.semantic_score",
            explanation="Phrase matches raise score; explicit negative terms lower score.",
        )
        return bounded

    def _route_classify_semantic_score(self, route: Dict[str, Any], query_profile: Dict[str, Any]) -> float:
        """Sub-step: lightweight semantic-style similarity stub."""
        route_text = f"{route.get('title', '')} {route.get('description', '')} {route.get('id', '')}".lower()
        query_text = query_profile.get("normalized_query", "")
        query_ngrams = set(self._word_ngrams(query_text, n=3))
        route_ngrams = set(self._word_ngrams(route_text, n=3))
        if not query_ngrams or not route_ngrams:
            score = 0.0
        else:
            score = len(query_ngrams.intersection(route_ngrams)) / max(len(route_ngrams), 1)
        self._emit(
            phase="route_classify",
            step="route_classify.semantic_score",
            actor="LayeredRouter",
            action="semantic overlap (n-gram overlap stub)",
            inp={"route": route.get("id"), "route_text": route_text},
            out={"semantic_score": round(score, 4)},
            next_action="route_classify.combine_scores",
            explanation="This is a deterministic approximation for teaching instead of a real embedding model.",
        )
        return float(min(max(score, 0.0), 1.0))

    @staticmethod
    def _word_ngrams(text: str, n: int = 3) -> List[str]:
        """Split text into character n-grams as a simple semantic helper."""
        cleaned = re.sub(r"\s+", " ", text.strip().lower())
        return [cleaned[i : i + n] for i in range(max(len(cleaned) - n + 1, 0))]

    def _route_classify_combined_score(self, route: Dict[str, Any], heuristic: float, semantic: float) -> float:
        """Sub-step: combine heuristic + semantic and apply route penalties."""
        route_id = route.get("id", "")
        penalty = 0.10 if route_id == "fallback_unknown" else 0.0
        combined = (heuristic * 0.65) + (semantic * 0.35) - penalty
        self._emit(
            phase="route_classify",
            step="route_classify.combined_score",
            actor="LayeredRouter",
            action="combine candidate scores",
            inp={"route": route_id, "heuristic": heuristic, "semantic": semantic, "penalty": penalty},
            out={"combined": round(combined, 4)},
            next_action="route_classify.sort_top_k",
            explanation="Combined score is one deterministic number used for ranking.",
        )
        return round(max(0.0, min(combined, 1.0)), 4)

    def _route_classify_top_k(self, ranked: List[Dict[str, Any]], k: int = 3) -> List[Dict[str, Any]]:
        """Sub-step: sort and return top-k candidates."""
        top_k = ranked[:k]
        self._emit(
            phase="route_classify",
            step="route_classify.sort_top_k",
            actor="LayeredRouter",
            action="sort by combined score and keep top-k",
            inp={"k": k},
            out=top_k,
            next_action="route_classify.ambiguity_check",
            explanation="Top candidates are still available for rerank + fallback logic.",
        )
        return top_k

    def _route_classify_ambiguity(self, top_k: List[Dict[str, Any]]) -> Tuple[bool, bool]:
        """Sub-step: detect low confidence or ambiguity."""
        if not top_k:
            return True, True
        top_score = float(top_k[0].get("combined", 0.0))
        second_score = float(top_k[1].get("combined", 0.0)) if len(top_k) > 1 else 0.0
        low_confidence = top_score < 0.35
        ambiguous = abs(top_score - second_score) < 0.10
        self._emit(
            phase="route_classify",
            step="route_classify.ambiguity_check",
            actor="LayeredRouter",
            action="check low-confidence + ambiguity",
            inp={"top1": top_score, "top2": second_score},
            out={"low_confidence": low_confidence, "ambiguous": ambiguous},
            next_action="route_classify.llm_rerank_or_accept",
            explanation="Both low confidence and ambiguity gates are evaluated here.",
        )
        return low_confidence, ambiguous

    def _route_classify_llm_fallback(self, query: str, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Sub-step: mock LLM reranker for ambiguous cases."""
        self.tool_calls_used += 1
        if not candidates:
            decision = {}
        elif len(candidates) == 1:
            decision = candidates[0]
        else:
            # Deterministic mock rerank: keep first route, but add an explicit reason.
            decision = candidates[0]
            decision = {**decision, "rerank_reason": "top route kept in stable deterministic tie-break", "source": "mock_llm_rerank"}
        self._emit(
            phase="route_classify",
            step="route_classify.route_classifier_llm",
            actor="MockLLMRouter",
            action="rerank ambiguous candidates",
            inp={"query": query, "candidate_ids": [c.get("route_id") for c in candidates]},
            out=decision,
            next_action="route_classify.apply_fallback",
            explanation="Mock rerank only runs when ambiguous + low-confidence path requested it.",
        )
        return decision

    def _route_classify_apply_fallback(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        """Sub-step: if no candidate, switch to fallback_unknown."""
        chosen = dict(candidate)
        if not chosen:
            fallback_route = self._get_route("fallback_unknown")
            self._emit(
                phase="route_classify",
                step="route_classify.apply_fallback",
                actor="LayeredRouter",
                action="replace empty decision with fallback_unknown",
                inp={"candidate": candidate},
                out={"route_id": "fallback_unknown", "reason": "no candidate above threshold"},
                next_action="route_classify.decide_next_action",
                explanation="A safe fallback route is required for robustness.",
            )
            return {
                "route_id": "fallback_unknown",
                "combined": 0.20,
                "route": fallback_route or {"id": "fallback_unknown", "required_permissions": [], "tools": []},
                "source": "fallback",
            }
        self._emit(
            phase="route_classify",
            step="route_classify.apply_fallback",
            actor="LayeredRouter",
            action="accept top candidate",
            inp={"candidate": candidate.get("route_id")},
            out=chosen,
            next_action="route_classify.decide_next_action",
            explanation="Top candidate was good enough; no fallback applied.",
        )
        return chosen

    def _route_classify_decide_next_action(self, route_id: str, low_conf: bool, ambiguous: bool) -> str:
        """Sub-step: next-action decision from route and confidence."""
        if route_id in {"ask_clarification", "fallback_unknown"}:
            decision = "ask_clarification"
        elif route_id == "meta_route":
            decision = "meta_route"
        elif low_conf and ambiguous:
            # If both conditions exist, demonstrate the reroute path.
            decision = "meta_route"
        elif low_conf:
            decision = "ask_clarification"
        else:
            decision = "permission_check"
        self._emit(
            phase="route_classify",
            step="route_classify.next_action_decision",
            actor="LayeredRouter",
            action="decide next action",
            inp={"route_id": route_id, "low_confidence": low_conf, "ambiguous": ambiguous},
            out={"next_action": decision},
            next_action=decision,
            explanation="This is the decision branch that moves the policy to the next phase.",
        )
        return decision

    def _phase_route_classify(self, query_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Run all route_classify sub-steps and return the chosen route decision."""
        ranked = self._route_classify_iterate_routes(query_profile)
        top_k = self._route_classify_top_k(ranked, k=3)
        low_conf, ambiguous = self._route_classify_ambiguity(top_k)

        if (low_conf or ambiguous) and self.manifest.get("prompts", {}).get("route_classify"):
            chosen = self._route_classify_llm_fallback(query_profile.get("raw_query", ""), top_k)
        elif top_k:
            chosen = top_k[0]
            self._emit(
                phase="route_classify",
                step="route_classify.skip_llm_fallback",
                actor="LayeredRouter",
                action="skip mock LLM rerank",
                inp={"low_confidence": low_conf, "ambiguous": ambiguous},
                out={"reason": "confidence route", "chosen": chosen.get("route_id")},
                next_action="route_classify.apply_fallback",
                explanation="LLM rerank is only called on weak/ambiguous scoring in this teaching path.",
            )
        else:
            self._emit(
                phase="route_classify",
                step="route_classify.no_scored_candidates",
                actor="LayeredRouter",
                action="handle empty candidate list",
                inp={"route_count": len(self.route_catalog)},
                out={"reason": "ranked list empty"},
                next_action="route_classify.apply_fallback",
                explanation="No route candidate means fallback route must be selected.",
            )
            chosen = {}

        final_choice = self._route_classify_apply_fallback(chosen)
        next_action = self._route_classify_decide_next_action(
            route_id=final_choice.get("route_id"),
            low_conf=low_conf,
            ambiguous=ambiguous,
        )
        decision = {
            "route_id": final_choice.get("route_id"),
            "route": self._get_route(final_choice.get("route_id")) or final_choice.get("route", {}),
            "next_action": next_action,
            "candidates_top_k": top_k,
            "low_confidence": low_conf,
            "ambiguous": ambiguous,
            "source": final_choice.get("source", "scoring"),
        }
        self.phase_state["route"] = decision["route_id"]
        self.phase_state["next_action"] = next_action
        self._emit(
            phase="route_classify",
            step="route_classify.log_final_decision",
            actor="LayeredRouter",
            action="write final route + next action",
            inp={"candidates_top_k": [c.get("route_id") for c in top_k]},
            out=decision,
            next_action=next_action,
            explanation="Decision object is used by every later phase.",
        )
        return decision

    # ------------------------------------------------------------------
    # Decision branches
    # ------------------------------------------------------------------

    def _phase_ask_clarification(self, reason: str, detail: Dict[str, Any]) -> Dict[str, Any]:
        """Sub-step block for ask_clarification outcomes."""
        payload = {
            "missing_information": detail.get("missing", []),
            "reason": reason,
            "raw_profile": self.phase_state.get("query_profile", {}),
            "route": self.phase_state.get("route"),
            "detail": detail,
        }
        self._emit(
            phase="ask_clarification",
            step="ask_clarification.build_payload",
            actor="Clarifier",
            action="build clarification payload",
            inp={"reason": reason},
            out=payload,
            next_action="final",
            explanation="When confidence is too low, or required data is missing, harness asks for clarification.",
        )
        self._emit(
            phase="ask_clarification",
            step="ask_clarification.set_status",
            actor="Clarifier",
            action="set final status",
            inp={"status_in": self.phase_state.get("final_status")},
            out={"status": "clarification_required", "next_action": "final"},
            next_action="final",
            explanation="No tool execution happens after this branch.",
        )
        self.phase_state["final_status"] = "clarification_required"
        self.phase_state["final_response"] = "Clarification requested. I need a clearer route, required slots, or permission."
        return {
            "next_action": "final",
            "status": "clarification_required",
            "response": self.phase_state["final_response"],
            "payload": payload,
        }

    def _phase_meta_route(self, base_decision: Dict[str, Any], inspect_state: Dict[str, Any]) -> Dict[str, Any]:
        """Sub-step for temporary route planning path."""
        self._emit(
            phase="meta_route",
            step="meta_route.emit_temporary_plan",
            actor="LayeredRouter",
            action="LLM proposes temporary next route",
            inp={"query": inspect_state.get("query", "")},
            out={"temporary_next_route": "sports_score_lookup", "plan_reason": "query appears about score lookup"},
            next_action="meta_route.validate_next_route",
            explanation="Meta route is a stubbed branch that can re-enter permission check with new route id.",
        )

        proposed_next_route = "sports_score_lookup"
        known = self._get_route(proposed_next_route)
        if known:
            self._emit(
                phase="meta_route",
                step="meta_route.next_route_known",
                actor="LayeredRouter",
                action="assign proposed next_route",
                inp={"candidate": proposed_next_route},
                out={"route_id": proposed_next_route, "known": True},
                next_action="permission_check",
                explanation="The proposed route maps to a known route in the catalog.",
            )
            self.phase_state["route"] = proposed_next_route
            return {
                "route_id": proposed_next_route,
                "route": known,
                "next_action": "permission_check",
                "meta_plan": {"temporary_plan": "use sports_score_lookup"},
            }

        self._emit(
            phase="meta_route",
            step="meta_route.next_route_unknown",
            actor="LayeredRouter",
            action="reject unknown temporary route",
            inp={"candidate": proposed_next_route},
            out={"route_id": "unknown", "known": False},
            next_action="ask_clarification",
            explanation="Unknown routes cannot continue without clarification.",
        )
        return {
            "route_id": "unknown",
            "route": None,
            "next_action": "ask_clarification",
            "reason": "meta route candidate missing from catalog",
        }

    # ------------------------------------------------------------------
    # Phase 3: permission_check
    # ------------------------------------------------------------------

    def _permission_collect_core(self, route: Dict[str, Any]) -> List[str]:
        """Sub-step: collect mandatory runtime permissions for this route."""
        required_core = CORE_PHASES.copy()
        route_required = list(route.get("required_permissions", []) or [])
        required = list(dict.fromkeys(required_core + route_required))
        self._emit(
            phase="permission_check",
            step="permission_check.collect_required_permissions",
            actor="PermissionGuard",
            action="collect required permissions",
            inp={"route_required": route_required},
            out={"required": required},
            next_action="permission_check.check_run_tests_branch",
            explanation="Core and route-specific permissions are always merged.",
        )
        return required

    def _permission_run_tests_check(self, route_id: str) -> bool:
        """Sub-step: check branch where run_tests route adds execute_tools."""
        is_run_tests = route_id == "run_tests"
        self._emit(
            phase="permission_check",
            step="permission_check.run_tests_route_check",
            actor="PermissionGuard",
            action="route equals run_tests?",
            inp={"route_id": route_id},
            out={"is_run_tests": is_run_tests},
            next_action="permission_check.add_execute_tools",
            explanation="A run_tests route may require execute_tools even when no tool list exists.",
        )
        return is_run_tests

    def _permission_add_execute_tools(self, required: List[str], route: Dict[str, Any]) -> List[str]:
        """Sub-step: add execute_tools when tool calls are part of the route."""
        needs_tools = bool(route.get("tools"))
        updated = list(required)
        if needs_tools and "execute_tools" not in updated:
            updated.append("execute_tools")
        self._emit(
            phase="permission_check",
            step="permission_check.add_execute_tools",
            actor="PermissionGuard",
            action="ensure execute_tools if tools exist",
            inp={"needs_tools": needs_tools, "current_permissions": required},
            out={"updated_permissions": updated},
            next_action="permission_check.check_missing",
            explanation="Permission list is normalized with execute_tools whenever a route has tools.",
        )
        return updated

    def _permission_missing(self, required: List[str], manifest_permissions: Dict[str, Any]) -> List[str]:
        """Sub-step: compute missing permissions from allow/deny policy."""
        allow = set(manifest_permissions.get("allow", []))
        deny = set(manifest_permissions.get("deny", []))
        missing = [perm for perm in required if (perm in deny or (allow and perm not in allow))]
        self._emit(
            phase="permission_check",
            step="permission_check.missing_permissions",
            actor="PermissionGuard",
            action="intersect required with denied/unsupported perms",
            inp={"required": required, "allow": sorted(list(allow)), "deny": sorted(list(deny))},
            out={"missing_permissions": missing},
            next_action="permission_check.set_next",
            explanation="A permission that is denied or not explicitly allowed is considered missing.",
        )
        return missing

    def _phase_permission_check(self, route: Dict[str, Any]) -> Dict[str, Any]:
        """Run permission_check phase including all mermaid sub-steps."""
        route_id = route.get("id")
        required = self._permission_collect_core(route)
        run_tests_needed = self._permission_run_tests_check(route_id or "")
        required = self._permission_add_execute_tools(required, route)
        missing = self._permission_missing(required, self.manifest.get("permissions", {}))
        if run_tests_needed:
            self._emit(
                phase="permission_check",
                step="permission_check.run_tests_branch.added_execute",
                actor="PermissionGuard",
                action="run_tests special branch applied",
                inp={"run_tests_needed": run_tests_needed},
                out={"status": "execute_tools present"},
                next_action="permission_check.eval_next",
                explanation="This branch demonstrates the decision path shown in the Mermaid graph.",
            )
        else:
            self._emit(
                phase="permission_check",
                step="permission_check.run_tests_branch.skipped",
                actor="PermissionGuard",
                action="run_tests special branch skipped",
                inp={"run_tests_needed": run_tests_needed},
                out={"status": "no-op"},
                next_action="permission_check.eval_next",
                explanation="run_tests branch only applies when route_id == run_tests.",
            )

        if missing:
            self._emit(
                phase="permission_check",
                step="permission_check.blocked",
                actor="PermissionGuard",
                action="route blocked due to permission mismatch",
                inp={"missing_permissions": missing},
                out={"next_action": "ask_clarification"},
                next_action="ask_clarification",
                explanation="Any missing permissions must escalate before execution.",
            )
            self.phase_state["missing_permissions"] = missing
            return {"next_action": "ask_clarification", "allowed": False, "missing_permissions": missing}

        self._emit(
            phase="permission_check",
            step="permission_check.allowed",
            actor="PermissionGuard",
            action="all required permissions satisfied",
            inp={"required": required},
            out={"allowed": True},
            next_action="extract_slots",
            explanation="No blockers in manifest permission model.",
        )
        self.phase_state["missing_permissions"] = []
        return {"next_action": "extract_slots", "allowed": True, "required_permissions": required}

    # ------------------------------------------------------------------
    # Phase 4: extract_slots
    # ------------------------------------------------------------------

    def _extract_slots_load_prompt(self) -> str:
        """Sub-step: load slot extractor prompt from manifest."""
        prompt = (self.manifest.get("prompts") or {}).get(
            "extract_slots",
            "You are a deterministic slot extractor stub. Use regex-like checks only.",
        )
        self._emit(
            phase="extract_slots",
            step="extract_slots.load_prompt",
            actor="SlotExtractor",
            action="load slot extraction prompt",
            inp={"present": bool(prompt)},
            out={"prompt": prompt},
            next_action="extract_slots.llm_gate",
            explanation="A specific prompt switch controls whether mock LLM extractor runs.",
        )
        return prompt

    def _extract_slots_llm_enabled(self) -> bool:
        """Sub-step: decide if mock LLM slot extraction runs."""
        enabled = "extract_slots" in self.manifest.get("prompts", {})
        self._emit(
            phase="extract_slots",
            step="extract_slots.llm_slot_enabled",
            actor="SlotExtractor",
            action="determine slot extractor mode",
            inp={"prompt_loaded": enabled},
            out={"enabled": enabled},
            next_action="extract_slots.llm_or_deterministic",
            explanation="LLM slot extractor can be toggled via manifest.",
        )
        return enabled

    def _extract_slots_llm(self, query: str, route: Dict[str, Any]) -> Dict[str, Any]:
        """Sub-step: mock LLM slot extraction."""
        self.tool_calls_used += 1
        tokens = self._token_set(query)
        extracted = {
            "team_a": "Texas Rangers" if "rangers" in tokens else None,
            "team_b": "Houston Astros" if "astros" in tokens else None,
            "route": route.get("id"),
            "source": "mock_llm",
        }
        self._emit(
            phase="extract_slots",
            step="extract_slots.llm_parse_slots",
            actor="MockLLM",
            action="extract slots from query",
            inp={"query": query, "route": route.get("id")},
            out=extracted,
            next_action="extract_slots.merge_and_normalize",
            explanation="Slot extraction is still deterministic in this mock.",
        )
        return extracted

    def _extract_slots_deterministic(self, query: str, route: Dict[str, Any]) -> Dict[str, Any]:
        """Sub-step: deterministic regex fallback for missing LLM path."""
        tokens = self._token_set(query)
        fallback = {
            "team_a": "Texas Rangers" if "rangers" in tokens else None,
            "team_b": "Houston Astros" if "astros" in tokens else None,
            "route": route.get("id"),
            "source": "deterministic",
        }
        self._emit(
            phase="extract_slots",
            step="extract_slots.deterministic_parse_slots",
            actor="SlotExtractor",
            action="extract slots from token heuristics",
            inp={"query": query},
            out=fallback,
            next_action="extract_slots.merge_and_normalize",
            explanation="Regex-like fallback ensures slots can still be produced without LLM.",
        )
        return fallback

    def _extract_slots_merge(self, llm_slots: Dict[str, Any], deterministic_slots: Dict[str, Any]) -> Dict[str, Any]:
        """Sub-step: normalize merged slot output."""
        merged = {**deterministic_slots, **{k: v for k, v in llm_slots.items() if v is not None}}
        # If one side fills only one slot, keep both by forcing canonical normalization below.
        merged["team_a"] = merged.get("team_a")
        merged["team_b"] = merged.get("team_b")
        self._emit(
            phase="extract_slots",
            step="extract_slots.merge_slots",
            actor="SlotExtractor",
            action="merge llm + deterministic slot outputs",
            inp={"llm_slots": llm_slots, "deterministic_slots": deterministic_slots},
            out={"merged": merged},
            next_action="extract_slots.check_required",
            explanation="LLM values are preferred when available, deterministic values fill gaps.",
        )
        return merged

    def _extract_slots_missing_check(self, route: Dict[str, Any], slots: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Sub-step: verify required slots are present."""
        required_slots = [s for s in route.get("required_slots", []) if s]
        missing = [slot for slot in required_slots if not slots.get(slot)]
        self._emit(
            phase="extract_slots",
            step="extract_slots.check_required_slots",
            actor="SlotExtractor",
            action="compare required slots vs extracted slots",
            inp={"required_slots": required_slots},
            out={"missing_slots": missing},
            next_action="extract_slots.next",
            explanation="Missing required slots force clarifying question path.",
        )
        return (len(missing) > 0), missing

    def _phase_extract_slots(self, route: Dict[str, Any], inspect_state: Dict[str, Any]) -> Dict[str, Any]:
        """Run slot extraction and emit branch result."""
        _ = self._extract_slots_load_prompt()
        use_llm = self._extract_slots_llm_enabled()
        query = inspect_state.get("raw_query") or inspect_state.get("query", "")
        if use_llm:
            llm_slots = self._extract_slots_llm(query, route)
            deterministic_slots = self._extract_slots_deterministic(query, route)
        else:
            llm_slots = {}
            deterministic_slots = self._extract_slots_deterministic(query, route)
            self._emit(
                phase="extract_slots",
                step="extract_slots.llm_disabled",
                actor="SlotExtractor",
                action="skip llm slot branch",
                inp={"llm_enabled": False},
                out={"llm_slots": {}},
                next_action="extract_slots.merge_slots",
                explanation="LLM slot branch is disabled, deterministic path is authoritative.",
            )
        slots = self._extract_slots_merge(llm_slots, deterministic_slots)
        self.phase_state["slots"] = slots
        missing_any, missing = self._extract_slots_missing_check(route, slots)
        if missing_any:
            self._emit(
                phase="extract_slots",
                step="extract_slots.route_to_ask_clarification",
                actor="SlotExtractor",
                action="required slots missing -> ask clarification",
                inp={"missing_slots": missing},
                out={"next_action": "ask_clarification"},
                next_action="ask_clarification",
                explanation="Route cannot proceed to build_plan without required slots.",
            )
            return {"next_action": "ask_clarification", "slots": slots, "missing": missing}

        self._emit(
            phase="extract_slots",
            step="extract_slots.to_build_plan",
            actor="SlotExtractor",
            action="all required slots present",
            inp={"slots": slots},
            out={"next_action": "build_plan"},
            next_action="build_plan",
            explanation="Slot extraction completed successfully.",
        )
        return {"next_action": "build_plan", "slots": slots, "missing": []}

    # ------------------------------------------------------------------
    # Phase 5: build_plan
    # ------------------------------------------------------------------

    def _build_plan_enumerate_tools(self, route: Dict[str, Any]) -> List[str]:
        """Sub-step: enumerate configured tools."""
        tools = list(route.get("tools", []) or [])
        self._emit(
            phase="build_plan",
            step="build_plan.enumerate_tools",
            actor="Planner",
            action="read route tool list",
            inp={"route_id": route.get("id")},
            out={"tools": tools},
            next_action="build_plan.construct_calls",
            explanation="Plan is derived directly from route tooling declarations.",
        )
        return tools

    def _build_plan_construct_calls(self, route: Dict[str, Any], slots: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Sub-step: build ordered tool calls."""
        calls = []
        for tool_name in self._build_plan_enumerate_tools(route):
            calls.append({
                "id": tool_name,
                "args": {
                    "route": route.get("id"),
                    "team_a": slots.get("team_a"),
                    "team_b": slots.get("team_b"),
                },
                "attempt_index": len(calls),
            })
        self._emit(
            phase="build_plan",
            step="build_plan.construct_ordered_plan",
            actor="Planner",
            action="order tool calls deterministically",
            inp={"slots": slots},
            out={"plan": calls},
            next_action="build_plan.check_empty",
            explanation="Call order is stable: as declared in manifest route.tools.",
        )
        return calls

    def _build_plan_is_empty(self, plan: List[Dict[str, Any]]) -> bool:
        """Sub-step: detect empty plan and route to summarize."""
        empty = len(plan) == 0
        self._emit(
            phase="build_plan",
            step="build_plan.check_empty_plan",
            actor="Planner",
            action="if plan empty -> summarize directly",
            inp={"plan_size": len(plan)},
            out={"plan_empty": empty},
            next_action="summarize" if empty else "execute_tools",
            explanation="Routes with no tools go to summarize directly.",
        )
        return empty

    def _phase_build_plan(self, route: Dict[str, Any], slots: Dict[str, Any]) -> Dict[str, Any]:
        """Run planning phase and decide execute vs summarize branch."""
        plan = self._build_plan_construct_calls(route, slots)
        self.phase_state["plan"] = plan
        if self._build_plan_is_empty(plan):
            self._emit(
                phase="build_plan",
                step="build_plan.route_to_summarize",
                actor="Planner",
                action="no tools required",
                inp={"plan": plan},
                out={"next_action": "summarize"},
                next_action="summarize",
                explanation="This route can respond immediately from route logic.",
            )
            return {"next_action": "summarize", "plan": plan, "planned": False}

        self._emit(
            phase="build_plan",
            step="build_plan.route_to_execute_tools",
            actor="Planner",
            action="tools exist -> execute",
            inp={"plan": plan},
            out={"next_action": "execute_tools"},
            next_action="execute_tools",
            explanation="One or more tools means tool execution phase is required.",
        )
        return {"next_action": "execute_tools", "plan": plan, "planned": True}

    # ------------------------------------------------------------------
    # Phase 6: execute_tools
    # ------------------------------------------------------------------

    def _execute_tools_budget_ok(self) -> bool:
        """Sub-step: check tool budget before each dispatch."""
        remaining = self.budgets.get("tool_calls_max", 8)
        ok = self.tool_calls_used < remaining
        self._emit(
            phase="execute_tools",
            step="execute_tools.budget_check",
            actor="ToolRunner",
            action="compare tool_calls_used vs budget",
            inp={"tool_calls_used": self.tool_calls_used, "tool_calls_max": remaining},
            out={"allowed": ok},
            next_action="execute_tools.dispatch_or_fail",
            explanation="In-memory budget prevents infinite loops and enforces trace limits.",
        )
        return ok

    def _tool_lookup_team_matchups(self, call: Dict[str, Any]) -> Dict[str, Any]:
        """Sub-step mock tool: find matchups for provided teams."""
        team_a = (call.get("args") or {}).get("team_a")
        team_b = (call.get("args") or {}).get("team_b")
        rows = []
        for row in MOCK_MATCHUPS:
            names = {row["home"], row["away"]}
            if team_a in names and team_b in names:
                rows.append(row)
        self.tool_calls_used += 1
        out = {"status": "ok", "matches": rows, "count": len(rows)}
        self._emit(
            phase="execute_tools",
            step="execute_tools.run_lookup_team_matchups",
            actor="MockToolRunner",
            action="query stub sports index",
            inp={"call": call},
            out=out,
            next_action="execute_tools.collect_output",
            explanation="This mock tool returns historical match rows for the team pair.",
        )
        return out

    def _tool_fetch_latest_box_score(self, call: Dict[str, Any], evidence: Dict[str, Any]) -> Dict[str, Any]:
        """Sub-step mock tool: fetch final score from previous matcher context."""
        latest = None
        matches = evidence.get("lookup_team_matchups", {}).get("matches", [])
        if matches:
            latest = sorted(matches, key=lambda r: r["date"])[-1]
        elif MOCK_MATCHUPS:
            latest = sorted(MOCK_MATCHUPS, key=lambda r: r["date"])[-1]
        out = {
            "status": "ok" if latest else "missing",
            "box_score": latest,
            "score": latest.get("score") if latest else None,
            "home": latest.get("home") if latest else None,
            "away": latest.get("away") if latest else None,
            "date": latest.get("date") if latest else None,
        }
        self.tool_calls_used += 1
        self._emit(
            phase="execute_tools",
            step="execute_tools.run_fetch_latest_box_score",
            actor="MockToolRunner",
            action="fetch mock latest box score",
            inp={"call": call},
            out=out,
            next_action="execute_tools.collect_output",
            explanation="This mock tool extracts final score from selected mock matchups.",
        )
        return out

    def _execute_tool_dispatcher(self, call: Dict[str, Any], evidence: Dict[str, Any]) -> Dict[str, Any]:
        """Sub-step: route one call to the matching mock tool."""
        tool_name = call.get("id")
        if tool_name == "lookup_team_matchups":
            return self._tool_lookup_team_matchups(call)
        if tool_name == "fetch_latest_box_score":
            return self._tool_fetch_latest_box_score(call, evidence)
        unknown = {"status": "skipped", "reason": f"unknown tool: {tool_name}"}
        self.tool_calls_used += 1
        self._emit(
            phase="execute_tools",
            step="execute_tools.dispatch_unknown_tool",
            actor="ToolRunner",
            action="handle unsupported tool name",
            inp={"tool_name": tool_name},
            out=unknown,
            next_action="execute_tools.collect_output",
            explanation="Unknown tools are surfaced explicitly for clarity.",
        )
        return unknown

    def _execute_tools_collect(self, call_index: int, call: Dict[str, Any], evidence: Dict[str, Any]) -> Dict[str, Any]:
        """Sub-step: collect per-tool output into evidence."""
        output = self._execute_tool_dispatcher(call, evidence)
        evidence_key = call.get("id")
        evidence[evidence_key] = output
        result = {
            "call_index": call_index,
            "tool_id": call.get("id"),
            "success": output.get("status") == "ok",
            "output": output,
        }
        self._emit(
            phase="execute_tools",
            step="execute_tools.collect_output",
            actor="ToolRunner",
            action="store execution output in evidence",
            inp={"call": call},
            out=result,
            next_action="execute_tools.next_call_check",
            explanation="Every execution result is appended to evidence and trace.",
        )
        self.phase_state["tool_outputs"].append(result)
        return result

    def _phase_execute_tools(self, plan: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run execute_tools loop and return collect evidence."""
        evidence = dict(self.phase_state["evidence"])
        self._emit(
            phase="execute_tools",
            step="execute_tools.loop_enter",
            actor="ToolRunner",
            action="enter for-loop over planned calls",
            inp={"plan_count": len(plan)},
            out={"status": "starting"},
            next_action="execute_tools.budget_check",
            explanation="This phase explicitly shows each loop iteration and decision.",
        )

        if not plan:
            self._emit(
                phase="execute_tools",
                step="execute_tools.no_planned_calls",
                actor="ToolRunner",
                action="skip execution when no calls",
                inp={"plan": []},
                out={"next_action": "summarize"},
                next_action="summarize",
                explanation="This is the loop-no-op branch.",
            )
            self.phase_state["evidence"] = evidence
            return {"next_action": "summarize", "results": [], "evidence": evidence}

        for index, call in enumerate(plan):
            if not self._execute_tools_budget_ok():
                self._emit(
                    phase="execute_tools",
                    step="execute_tools.budget_exceeded",
                    actor="ToolRunner",
                    action="stop execution due to budget",
                    inp={"attempt_index": index},
                    out={"reason": "tool_calls_budget_exceeded"},
                    next_action="summarize",
                    explanation="Tool budget limit intentionally blocks further calls.",
                )
                self.phase_state["evidence"] = evidence
                return {"next_action": "summarize", "results": self.phase_state["tool_outputs"], "evidence": evidence, "budget_exceeded": True}

            self._emit(
                phase="execute_tools",
                step="execute_tools.iteration_dispatch",
                actor="ToolRunner",
                action="dispatch this call",
                inp={"call_index": index, "call": call},
                out={"allowed": True},
                next_action="execute_tools.collect_output",
                explanation="Each iteration executes exactly one tool with its index.",
            )
            self._emit(
                phase="execute_tools",
                step="execute_tools.more_calls_check",
                actor="ToolRunner",
                action="check if more calls remain",
                inp={"current_index": index, "plan_total": len(plan)},
                out={"more_calls_after_this": index < len(plan) - 1},
                next_action="execute_tools.iteration_dispatch" if index < len(plan) - 1 else "execute_tools.done",
                explanation="The branch is explicit for loop bookkeeping.",
            )
            self._execute_tools_collect(index, call, evidence)

        self._emit(
            phase="execute_tools",
            step="execute_tools.done",
            actor="ToolRunner",
            action="all calls completed",
            inp={"calls_executed": len(plan)},
            out={"next_action": "summarize"},
            next_action="summarize",
            explanation="Loop completion triggers summarize phase.",
        )
        self.phase_state["evidence"] = evidence
        return {"next_action": "summarize", "results": self.phase_state["tool_outputs"], "evidence": evidence}

    # ------------------------------------------------------------------
    # Phase 7: summarize
    # ------------------------------------------------------------------

    def _summarize_llm_enabled(self) -> bool:
        """Sub-step: check whether mock LLM summarize should run."""
        enabled = bool(self.manifest.get("prompts", {}).get("summarize"))
        self._emit(
            phase="summarize",
            step="summarize.llm_enabled",
            actor="Summarizer",
            action="check summarize prompt availability",
            inp={"has_summarize_prompt": enabled},
            out={"enabled": enabled},
            next_action="summarize.route_specific_or_fallback",
            explanation="Prompt presence decides whether mock LLM path is visible.",
        )
        return enabled

    def _summarize_llm(self, route_id: str, inspect_state: Dict[str, Any], evidence: Dict[str, Any]) -> str:
        """Sub-step: mock LLM summary path."""
        self.tool_calls_used += 1
        slots = inspect_state.get("teams_detected") or []
        if isinstance(slots, list) and len(slots) == 2:
            team_text = f"{slots[0]} vs {slots[1]}"
        else:
            team_text = inspect_state.get("query", "")
        out = f"LLM-style summary for {route_id}: final answer prepared for {team_text}."
        self._emit(
            phase="summarize",
            step="summarize.llm_summarizer",
            actor="MockLLM",
            action="generate final response text",
            inp={"route_id": route_id, "evidence": list(evidence.keys())},
            out={"response": out},
            next_action="validate",
            explanation="Mock summary text is produced deterministically from route + evidence keys.",
        )
        return out

    def _summarize_deterministic(self, route_id: str, evidence: Dict[str, Any], slots: Dict[str, Any]) -> str:
        """Sub-step: deterministic fallback summarizer."""
        if route_id == "sports_score_lookup":
            box = evidence.get("fetch_latest_box_score", {})
            score = box.get("score")
            home = box.get("home")
            away = box.get("away")
            date = box.get("date")
            if score and home and away:
                response = f"{home} vs {away} final score on {date}: {score}."
            elif slots:
                response = f"Could not find result for {slots.get('team_a')} vs {slots.get('team_b')}."
            else:
                response = "No score found for requested matchup."
        else:
            response = f"Route {route_id} completed with evidence keys: {', '.join(sorted(evidence.keys())) or 'none'}."

        self._emit(
            phase="summarize",
            step="summarize.deterministic_fallback",
            actor="Summarizer",
            action="build deterministic response",
            inp={"route_id": route_id},
            out={"response": response},
            next_action="validate",
            explanation="Fallback does not call LLM and is easy to inspect.",
        )
        return response

    def _phase_summarize(self, route: Dict[str, Any], inspect_state: Dict[str, Any], plan_state: Dict[str, Any]) -> Dict[str, Any]:
        """Run summarize phase; choose LLM path only when enabled."""
        route_id = route.get("id")
        evidence = self.phase_state.get("evidence", {})
        if self._summarize_llm_enabled():
            if route_id in {"sports_score_lookup", "summarize_doc", "run_tests", "bug_repair", "llm_answer", "ask_clarification"}:
                response = self._summarize_llm(route_id, inspect_state, evidence)
            else:
                response = self._summarize_llm(route_id, inspect_state, evidence)
        else:
            response = self._summarize_deterministic(route_id, evidence, self.phase_state.get("slots", {}))

        self._emit(
            phase="summarize",
            step="summarize.complete",
            actor="Summarizer",
            action="final response emitted",
            inp={"route_id": route_id, "plan_state": plan_state},
            out={"response": response},
            next_action="validate",
            explanation="A response exists before validator because even validation failures still need evidence.",
        )
        self.phase_state["final_response"] = response
        return {"next_action": "validate", "response": response}

    # ------------------------------------------------------------------
    # Phase 8: validate
    # ------------------------------------------------------------------

    def _validate_manifest_configured(self) -> bool:
        """Sub-step: verify whether validator definitions exist."""
        configured = bool(self.manifest.get("validators"))
        self._emit(
            phase="validate",
            step="validate.check_validator_config",
            actor="RuleValidator",
            action="check if route/manifest validator exists",
            inp={"validators_present": configured},
            out={"configured": configured},
            next_action="validate.required_fields" if configured else "validate.pass",
            explanation="Flow path changes depending on whether validators are configured.",
        )
        return configured

    def _validate_required_fields(self, evidence: Dict[str, Any], required_fields: List[str]) -> Tuple[bool, List[str]]:
        """Sub-step: check required evidence fields."""
        missing = [field for field in required_fields if field not in evidence.get("fetch_latest_box_score", {}) and field not in evidence]
        self._emit(
            phase="validate",
            step="validate.check_required_keys",
            actor="RuleValidator",
            action="validate required evidence keys",
            inp={"required_fields": required_fields},
            out={"missing_fields": missing},
            next_action="validate.fail" if missing else "validate.pass",
            explanation="Missing field detection is strict and explicit.",
        )
        return (len(missing) == 0), missing

    def _validate_fail_to_ask(self, missing: List[str]) -> Dict[str, Any]:
        """Sub-step: validation fail branch -> ask clarification."""
        self._emit(
            phase="validate",
            step="validate.fail",
            actor="RuleValidator",
            action="mark invalid and branch to ask_clarification",
            inp={"missing_fields": missing},
            out={"next_action": "ask_clarification"},
            next_action="ask_clarification",
            explanation="If required evidence is missing, flow escalates for clarification.",
        )
        self.phase_state["validation"] = {"passed": False, "missing_fields": missing}
        return {"status": "failed", "next_action": "ask_clarification", "missing_fields": missing}

    def _validate_pass(self) -> Dict[str, Any]:
        """Sub-step: validation success path."""
        self._emit(
            phase="validate",
            step="validate.pass",
            actor="RuleValidator",
            action="validation checks passed",
            inp={},
            out={"next_action": "final_report"},
            next_action="final_report",
            explanation="Evidence satisfies declared validator requirements.",
        )
        self.phase_state["validation"] = {"passed": True, "missing_fields": []}
        return {"status": "passed", "next_action": "final_report", "missing_fields": []}

    def _phase_validate(self, route: Dict[str, Any]) -> Dict[str, Any]:
        """Run validation phase after summarization."""
        if self._validate_manifest_configured():
            required_fields = route.get("validator_required_fields", []) or []
            passed, missing = self._validate_required_fields(self.phase_state.get("evidence", {}), required_fields)
            if not passed:
                return self._validate_fail_to_ask(missing)
            return self._validate_pass()

        # No validator branch in manifest -> treat as pass with pass-through state.
        self._emit(
            phase="validate",
            step="validate.no_validator",
            actor="RuleValidator",
            action="skip route-level field checks",
            inp={"route_id": route.get("id")},
            out={"status": "ok"},
            next_action="final_report",
            explanation="No validator configuration means this path is considered pass.",
        )
        self.phase_state["validation"] = {"passed": True, "missing_fields": []}
        return {"status": "passed", "next_action": "final_report", "missing_fields": []}

    # ------------------------------------------------------------------
    # Final assembly and report
    # ------------------------------------------------------------------

    def _phase_final(self) -> Dict[str, Any]:
        """Assemble final response object for the mock harness run."""
        status = self.phase_state.get("final_status") or ("completed" if self.phase_state["validation"].get("passed") else "failed")
        final = {
            "status": status,
            "route": self.phase_state.get("route"),
            "next_action": self.phase_state.get("next_action"),
            "final_response": self.phase_state.get("final_response"),
            "query_profile": self.phase_state.get("query_profile"),
            "slots": self.phase_state.get("slots"),
            "plan": self.phase_state.get("plan"),
            "tool_calls_used": self.tool_calls_used,
            "validation": self.phase_state.get("validation"),
            "missing_permissions": self.phase_state.get("missing_permissions"),
            "trace": [event.to_dict() for event in self.phase_trace],
            "budgets": self.budgets,
            "attempt": self.phase_state.get("attempt"),
            "attempts_available": self.budgets.get("retries_max", 3),
            "evidence": self.phase_state.get("evidence"),
        }
        self._emit(
            phase="final",
            step="final.report",
            actor="TraceBuilder",
            action="emit final structured report",
            inp={"status": status},
            out={"final_status": status, "trace_count": len(self.phase_trace)},
            next_action="done",
            explanation="This final object is the one that can be consumed by UI/docs/tests.",
        )
        return final

    def run(self, query: str) -> Dict[str, Any]:
        """Main orchestrator for all sub-step phases."""
        self._emit(
            phase="main",
            step="main.receive_inputs",
            actor="CLIAdapter",
            action="capture incoming arguments",
            inp={"query": query, "manifest_path": self.manifest_path},
            out={"query_len": len(query)},
            next_action="main.parse_request",
            explanation="The first real action is receiving and validating caller inputs.",
        )
        self._emit(
            phase="main",
            step="main.parse_request",
            actor="HarnessRuntime",
            action="start mock harness",
            inp={"query": query, "manifest_path": self.manifest_path},
            out={"attempt": self.phase_state["attempt"]},
            next_action="inspect_query",
            explanation="Every run begins in the inspect_query phase.",
        )

        inspect_state = self._phase_inspect_query(query)
        route_decision = self._phase_route_classify(inspect_state)

        # Decision fan-out section:
        if route_decision["next_action"] == "ask_clarification":
            self.phase_state["next_action"] = "ask_clarification"
            ask_state = self._phase_ask_clarification("route_classify", route_decision)
            self.phase_state["final_status"] = "clarification_required"
            return {"status": ask_state["status"], "final_report": self._phase_final()}
        elif route_decision["next_action"] == "meta_route":
            self.phase_state["next_action"] = "meta_route"
            meta_state = self._phase_meta_route(route_decision, inspect_state)
            if meta_state.get("next_action") == "ask_clarification":
                self.phase_state["route"] = meta_state.get("route_id", self.phase_state.get("route"))
                ask_state = self._phase_ask_clarification("meta_route", meta_state)
                self.phase_state["final_status"] = "clarification_required"
                return {"status": ask_state["status"], "final_report": self._phase_final()}
            route_decision["route_id"] = meta_state.get("route_id", route_decision["route_id"])
            route_decision["route"] = self._get_route(route_decision["route_id"]) or {}
            self._emit(
                phase="decision",
                step="decision.meta_route_to_permission",
                actor="PolicyRouter",
                action="meta_route resolved and passed route forward",
                inp={"meta_state": meta_state},
                out={"route_id": route_decision["route_id"], "next": "permission_check"},
                next_action="permission_check",
                explanation="Meta branch resolves route and falls back into regular pipeline.",
            )
            route = route_decision["route"]
        else:
            self._emit(
                phase="decision",
                step="decision.meta_route_skipped",
                actor="PolicyRouter",
                action="skip meta_route branch",
                inp={"next_action": route_decision["next_action"]},
                out={"skipped_branch": "meta_route"},
                next_action="permission_check",
                explanation="Executed path did not request meta route.",
            )
            self._emit(
                phase="decision",
                step="decision.ask_clarification_skipped",
                actor="PolicyRouter",
                action="skip ask_clarification branch",
                inp={"next_action": route_decision["next_action"]},
                out={"skipped_branch": "ask_clarification"},
                next_action="permission_check",
                explanation="Executed path did not request immediate clarification.",
            )
            route = route_decision["route"]

        perm_state = self._phase_permission_check(route)
        if perm_state["next_action"] == "ask_clarification":
            self._emit(
                phase="permission_check",
                step="permission_check.to_ask_clarification",
                actor="PolicyRouter",
                action="permission phase blocked, escalate",
                inp={"missing_permissions": perm_state.get("missing_permissions")},
                out={"next_action": "ask_clarification"},
                next_action="ask_clarification",
                explanation="No tool execution should start until permissions pass.",
            )
            ask_state = self._phase_ask_clarification("permission_check", perm_state)
            self.phase_state["final_status"] = "clarification_required"
            return {"status": ask_state["status"], "final_report": self._phase_final()}

        slots_state = self._phase_extract_slots(route, inspect_state)
        if slots_state["next_action"] == "ask_clarification":
            self._emit(
                phase="extract_slots",
                step="extract_slots.to_ask_clarification",
                actor="PolicyRouter",
                action="required slots missing",
                inp={"missing": slots_state.get("missing")},
                out={"next_action": "ask_clarification"},
                next_action="ask_clarification",
                explanation="Missing slots are escalated before plan generation.",
            )
            ask_state = self._phase_ask_clarification("extract_slots", slots_state)
            self.phase_state["final_status"] = "clarification_required"
            return {"status": ask_state["status"], "final_report": self._phase_final()}

        # Emit explicit skip path for the 'plan is empty' branch before normal run.
        plan_state = self._phase_build_plan(route, slots_state.get("slots", {}))
        if plan_state["next_action"] == "summarize":
            self._emit(
                phase="build_plan",
                step="build_plan.execute_tools_skipped",
                actor="PolicyRouter",
                action="log that execute_tools branch is not taken",
                inp={"plan": plan_state.get("plan")},
                out={"next_action": "summarize"},
                next_action="summarize",
                explanation="This is the explicit empty-plan branch from Mermaid.",
            )
            self.phase_state["evidence"] = {"box_score": None, "score": None}
            summarize_state = self._phase_summarize(route, inspect_state, plan_state)
            validation_state = self._phase_validate(route)
            self.phase_state["final_status"] = "completed" if validation_state.get("status") == "passed" else "failed"
            self.phase_state["next_action"] = validation_state.get("next_action", "final")
            return self._phase_final()

        self._emit(
            phase="build_plan",
            step="build_plan.plan_to_execute_tools",
            actor="PolicyRouter",
            action="plan non-empty, continue",
            inp={"plan_size": len(plan_state.get("plan", []))},
            out={"next_action": "execute_tools"},
            next_action="execute_tools",
            explanation="At least one planned call means execute_tools runs.",
        )
        execute_state = self._phase_execute_tools(plan_state.get("plan", []))
        summarize_state = self._phase_summarize(route, inspect_state, execute_state)
        validation_state = self._phase_validate(route)

        self.phase_state["next_action"] = validation_state.get("next_action", "final")
        if validation_state.get("status") != "passed":
            ask_state = self._phase_ask_clarification("validate", validation_state)
            self.phase_state["final_status"] = "clarification_required"
            return {"status": ask_state["status"], "final_report": self._phase_final()}
        self.phase_state["final_status"] = "completed"
        final = self._phase_final()
        return final


def print_help_section() -> None:
    """Print the detailed inline help text."""
    print(HELP_TEXT)


def parse_args() -> argparse.Namespace:
    """Build CLI parser for the teaching mock harness."""
    parser = argparse.ArgumentParser(
        description=(
            "Mock AI Harness Engine (teaching mode). "
            "Prints every Mermaid step/sub-step while simulating route-classify + policy flow."
        )
    )
    parser.add_argument("query", nargs="?", default="", help="User query for the mock harness.")
    parser.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST_PATH,
        help="Path to route manifest YAML (default: mock_harness_route.yaml).",
    )
    parser.add_argument(
        "--help-section",
        action="store_true",
        help="Print detailed mock harness flow help and exit.",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Also print final report as pretty JSON at the end.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    if args.help_section:
        print_help_section()
        return
    if not args.query:
        raise SystemExit("Error: query argument is required unless --help-section is used.")

    engine = MockHarnessEngine(args.manifest)
    final_report = engine.run(args.query)
    if args.print_json:
        print("--- FINAL JSON REPORT ---")
        print(json.dumps(final_report, ensure_ascii=False, indent=2, default=str))
    else:
        print("--- FINAL REPORT ---")
        print(json.dumps(final_report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
