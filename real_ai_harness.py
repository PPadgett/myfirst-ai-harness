#!/usr/bin/env python3
"""Generic route-driven harness.

This engine demonstrates a realistic policy pipeline:
- inspect
- classify route (heuristics + semantic + optional LLM rerank)
- permissions
- slot extraction
- plan
- execute tools
- summarize
- validate
- output full phase trace

It does not contain hardcoded query-topic dictionaries.
"""

from __future__ import annotations

import argparse
import sys
import json
import math
import os
import re
import subprocess
import shutil
from pathlib import Path
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import httpx
import yaml
from harness.tools import execute_tool
from harness.evidence import Claim
from harness.guards import (
    check_input_text,
    check_tool_output,
    check_tool_request_with_tool,
    sanitize_text,
    split_trusted_untrusted,
    redact_sensitive_args,
)


HELP_TEXT = """
Real AI Harness Engine - inline help and trace walkthrough
========================================================

This engine can be run in two modes:
1) normal execution mode with live logs
2) full help mode using `--help-section`

The trace is designed to be read linearly, where each phase is a step and each
important decision inside that phase is a sub-step. Every call to `_log()` prints:
- phase: current pipeline step
- actor: component owning the operation
- action: what this phase did
- input: what it received
- output: what it returned
- next: which phase runs next
- why: rationale for the transition (if provided)

Command arguments:
- --query: user request text (empty request is blocked and returns clarification-required output)
- --manifest: route definition YAML (default real_harness_routes.yaml)
- --no-llm: disable all LLM calls
- --no-network: disable remote calls and treat LLM as unavailable
- --max-tool-calls: override manifest tool budget
- --help-section: print this explanation and exit

Pipeline phases and sub-step map:

This manifest currently includes example high-level workflow routes like:
- llm_answer (informational questions)
- summarize_doc
- package_payload
- run_tests
- bug_repair
- schedule_email
- ask_clarification
- meta_route
- fallback_unknown

bootstrap
- Capture constructor inputs (manifest path and flags)
- Initialize phase trace and counters
- Load YAML manifest and build route map/signatures
- Resolve budgets, permission model, and LLM runtime settings
- Emit bootstrap trace event

parse_request (implemented by run() + phase_inspect_query)
- Receive raw query in run()
- Normalize and tokenize input
- Build deterministic profile fields (language hint + flags)
- Optionally enrich profile via LLM (`enable_query_inspect_llm`)

route_classify
- Score each route using heuristic bundles
- Score each route semantically via character n-gram cosine
- Combine scores and select top-k candidates
- Optional LLM rerank when low-confidence or ambiguous
- Force universal fallback when no route is confident

permission_check
- Build required permissions list for selected route
- Add run-time required permissions (inspect/route/extract/plan/validate...)
- Add run_tests special-case execute permission
- Compare required permissions with allow/deny model
- Emit clarification flow when blocked

extract_slots
- Apply route-specific prompt extraction if available
- Apply deterministic extraction patterns by route
- Validate required slots and list missing fields
- Route to clarification when mandatory slots absent

build_plan
- Convert selected route tools to executable plan rows
- Add route-specific default arguments for each tool
- Return empty plan for non-tool routes

execute_tools
- Iterate plan rows and enforce per-run tool budget
- Dispatch known handlers (`run_tests`)
- Increment tool call counter per call
- Record structured tool output or failure reason

summarize
- Build route-specific final response
- Prefer LLM summarizer when available
- Fall back to deterministic defaults

validate
- Read validator field list for route
- Validate evidence fields from tool output where needed
- Return ok/fail and missing fields when relevant

final report
- Return status / next_action / route / tool usage
- Include full trace so every phase event can be reviewed
"""

@dataclass
class TraceEvent:
    ts: str
    phase: str
    actor: str
    action: str
    input: Any
    output: Any
    next_action: str
    tool_calls_so_far: int
    attempt: int


class RealHarnessEngine:
    """Real harness pipeline with logged phases and optional LLM usage."""

    def __init__(
        self,
        manifest_path: str,
        force_disable_llm: bool = False,
        no_network: bool = False,
        max_tool_calls_override: Optional[int] = None,
    ) -> None:
        # 1) Keep constructor inputs for traceability/debugging.
        self.manifest_path = manifest_path
        self.force_disable_llm = force_disable_llm
        self.no_network = no_network

        # 2) Initialize tracing counters before any log call.
        self.phase_trace: List[TraceEvent] = []
        self.tool_calls_used = 0
        self.attempt = 1

        # 3) Load manifest and read policy structures.
        self.manifest = self._load_manifest(manifest_path)
        self.routes = self.manifest.get("routes", [])
        self.route_map = {str(r.get("id", "")): r for r in self.routes}

        self.budgets = self.manifest.get("budgets", {})
        self.llm_cfg = self.manifest.get("llm", {})
        self.router_cfg = self.manifest.get("router", {})
        self.permissions = self.manifest.get("permissions", {})

        # 4) Build runtime budget and overrides.
        self.max_tool_calls = max_tool_calls_override if max_tool_calls_override is not None else int(
            self.budgets.get("tool_calls_max", 8)
        )

        # 5) Materialize allow/deny permission model.
        self.allowed = set(self.permissions.get("allow_default", []))
        self.denied = set(self.permissions.get("deny_default", []))

        # 6) Precompute semantic n-gram signature map used in routing.
        self.semantic_n = int(self.router_cfg.get("semantic_ngram_n", 3))
        self.route_signatures: Dict[str, Counter[str]] = {}
        self._build_route_signatures()

        # 7) Resolve LLM runtime flags and call target.
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.llm_enabled = bool(self.openai_api_key and (not force_disable_llm) and (not self.no_network))

        self._log(
            "bootstrap",
            "HarnessRuntime",
            "bootstrap engine with manifest + permissions + budgets",
            {
                "manifest_path": manifest_path,
                "force_disable_llm": force_disable_llm,
                "no_network": no_network,
                "max_tool_calls_override": max_tool_calls_override,
            },
            {
                "routes": len(self.routes),
                "semantic_n": self.semantic_n,
                "tool_calls_max": self.max_tool_calls,
            },
            "inspect_query",
            explanation="Runtime is ready and moves to inspect_query.",
        )
        self.state_dir = Path(os.getenv("HARNESS_STATE_DIR", "state"))
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.tool_sandbox_mode = os.getenv("HARNESS_TOOL_SANDBOX", "off").lower()
        self.sandbox_image = os.getenv("HARNESS_TOOL_SANDBOX_IMAGE", "python:3.12-slim")
        self.sandbox_timeout = int(os.getenv("HARNESS_TOOL_SANDBOX_TIMEOUT", "120"))
        self.require_evidence = os.getenv("HARNESS_REQUIRE_EVIDENCE", "0") in {"1", "true", "TRUE", "yes", "on"}
        self.feature_level = os.getenv("HARNESS_FEATURE_LEVEL", "basic").lower()
        self.checkpoint_files: list[str] = []
        self.current_run_id: str | None = None

    def _checkpoint(
        self,
        phase: str,
        run_id: str,
        route_id: str,
        attempt: int,
        status: str = "ok",
        next_action: str | None = None,
        error_code: str | None = None,
        evidence_refs: list[str] | None = None,
        payload: Any = None,
        route_metadata: dict[str, Any] | None = None,
    ) -> str:
        timestamp = datetime.now(timezone.utc).isoformat()
        filename = f"{run_id}-a{attempt}-{phase}.json"
        target = self.state_dir / filename
        entry = {
            "phase": phase,
            "route_id": route_id,
            "attempt": attempt,
            "status": status,
            "next_action": next_action,
            "error_code": error_code,
            "evidence_refs": evidence_refs or [],
            "timestamp": timestamp,
            "route_metadata": redact_sensitive_args(route_metadata or {}),
            "payload": redact_sensitive_args(payload or {}),
        }
        target.write_text(json.dumps(entry, indent=2), encoding="utf-8")
        self.checkpoint_files.append(str(target))
        return str(target)

    @staticmethod
    def _load_manifest(path: str) -> Dict[str, Any]:
        # Load and validate the manifest file once during bootstrap.
        cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(cfg, dict):
            raise RuntimeError("Manifest must be YAML mapping")
        return cfg

    @staticmethod
    def _utc_now() -> str:
        # Return one UTC timestamp string per trace event.
        return datetime.now(timezone.utc).isoformat()

    def _log(
        self,
        phase: str,
        actor: str,
        action: str,
        inp: Any,
        out: Any,
        next_action: str,
        explanation: Optional[str] = None,
    ) -> None:
        event = TraceEvent(
            ts=self._utc_now(),
            phase=phase,
            actor=actor,
            action=action,
            input=inp,
            output=out,
            next_action=next_action,
            tool_calls_so_far=self.tool_calls_used,
            attempt=self.attempt,
        )
        self.phase_trace.append(event)
        print(f"[{event.ts}] phase={event.phase} actor={event.actor}")
        print(f"  action: {event.action}")
        if inp is not None:
            print(f"  input : {json.dumps(inp, ensure_ascii=False)}")
        print(f"  output: {json.dumps(out, ensure_ascii=False)}")
        if next_action:
            print(f"  next  : {next_action}")
        if explanation:
            print(f"  why   : {explanation}")
        print("-" * 90)

    @staticmethod
    def _safe_json(payload: str) -> Optional[Dict[str, Any]]:
        # Parse only the first valid JSON object from a model response.
        payload = payload.strip()
        start = payload.find("{")
        end = payload.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            parsed = json.loads(payload[start : end + 1])
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None

    def _run_id(self) -> str:
        if not self.current_run_id:
            self.current_run_id = str(uuid.uuid4())
        return self.current_run_id

    def _build_evidence_rows(
        self,
        route_id: str,
        final_response: str,
        tool_outputs: List[Dict[str, Any]],
    ) -> list[Dict[str, Any]]:
        run_id = self._run_id()
        evidence_rows: list[Dict[str, Any]] = []
        for index, output in enumerate(tool_outputs):
            evidence_rows.append(
                {
                    "evidence_id": f"{run_id}-tool-{index}",
                    "route_id": route_id,
                    "source": "tool",
                    "record": redact_sensitive_args(output),
                }
            )
        evidence_rows.append(
            {
                "evidence_id": f"{run_id}-final",
                "route_id": route_id,
                "source": "final_response",
                "record": {
                    "final_response": redact_sensitive_args(final_response),
                    "answer": redact_sensitive_args(final_response),
                },
            }
        )
        return evidence_rows

    def _build_claims(self, route_id: str, final_response: str, evidence: list[Dict[str, Any]], valid: bool) -> list[dict[str, Any]]:
        evidence_ids = [row["evidence_id"] for row in evidence]
        if not evidence_ids:
            evidence_ids = []
        claim = Claim(
            claim_id=f"{self._run_id()}-claim-0",
            route_id=route_id,
            statement=redact_sensitive_args(str(final_response)[:260] if isinstance(final_response, str) else str(final_response)),
            evidence_ids=evidence_ids if valid and self.require_evidence else [],
            status="verified" if valid else "unverified",
        )
        return [claim.to_dict()]

    def _common_response_meta(self) -> dict[str, Any]:
        return {
            "run_id": self.current_run_id,
            "checkpoint_id": self.checkpoint_files[-1] if self.checkpoint_files else None,
            "trace_checkpoint_count": len(self.checkpoint_files),
        }

    def _route_metadata(self, route_id: str) -> dict[str, Any]:
        route = self.route_map.get(route_id, {})
        if not route:
            return {}
        return {
            "title": route.get("title"),
            "description": route.get("description"),
            "tools": route.get("tools", []),
            "required_permissions": route.get("required_permissions", []),
            "required_slots": route.get("required_slots", {}),
            "validator": route.get("validator") or route.get("manifests"),
        }

    def _route_sandbox_tools(self, route_id: str) -> set[str]:
        route = self.route_map.get(route_id, {})
        policy = route.get("policy", {})
        candidates = policy.get("tool_sandbox_required")
        sandbox_tools: set[str] = set()
        if isinstance(candidates, str):
            if candidates.strip():
                sandbox_tools.add(candidates.strip())
        elif isinstance(candidates, (list, tuple, set)):
            for item in candidates:
                if isinstance(item, str) and item.strip():
                    sandbox_tools.add(item.strip())
        return sandbox_tools

    def _final_payload(
        self,
        status: str,
        next_action: str,
        final_response: Optional[str],
        route_id: str,
        tool_outputs: list[Dict[str, Any]] | None = None,
        validation: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        responses = final_response if final_response is not None else None
        outputs = tool_outputs or []
        validation = validation or {"ok": status == "ok", "missing_fields": [], "failed_route_ids": []}
        evidence = self._build_evidence_rows(route_id, responses or "", outputs) if status != "blocked" and status != "validation_failed" else []
        claims = self._build_claims(route_id, responses or "", evidence, bool(validation.get("ok", False))
                                ) if evidence else []
        payload = {
            "status": status,
            "next_action": next_action,
            "final_response": responses,
            "tool_calls_used": self.tool_calls_used,
            "route_id": route_id,
            "route_metadata": self._route_metadata(route_id),
            "budgets": self.budgets,
            "trace": [e.__dict__ for e in self.phase_trace],
            "trace_phase_count": len(self.phase_trace),
            "trace_checkpoint_count": self._common_response_meta().get("trace_checkpoint_count"),
            "run_id": self._common_response_meta().get("run_id"),
            "checkpoint_id": self._common_response_meta().get("checkpoint_id"),
            "evidence": evidence,
            "claims": claims,
            "validation": validation,
        }
        if extra:
            payload.update(extra)
        return payload

    @staticmethod
    def _route_validator(route: Dict[str, Any] | None) -> Dict[str, Any]:
        if not isinstance(route, dict):
            return {}
        validator = route.get("validator")
        if isinstance(validator, dict):
            return validator

        manifests = route.get("manifests")
        if isinstance(manifests, dict):
            fields = manifests.get("validator_fields") or []
            hard_fail = manifests.get("hard_fail_errors", [])
            if isinstance(fields, list) or isinstance(hard_fail, list):
                return {
                    "required_evidence_fields": list(fields) if isinstance(fields, list) else [],
                    "hard_fail_errors": list(hard_fail) if isinstance(hard_fail, list) else [],
                }
        return {}

    def _llm_json(self, system_prompt: str, user_prompt: str, max_tokens: int = 500) -> Optional[Dict[str, Any]]:
        # Ask the LLM only when enabled; return raw structured JSON or None.
        if not self.llm_enabled:
            return None
        payload = {
            "model": self.llm_cfg.get("model", "gpt-4.1-mini"),
            "temperature": float(self.llm_cfg.get("temperature", 0.0)),
            "max_tokens": int(max_tokens),
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json",
        }
        try:
            with httpx.Client(timeout=int(self.llm_cfg.get("timeout_seconds", 30))) as client:
                resp = client.post(f"{self.openai_base}/chat/completions", json=payload, headers=headers)
            resp.raise_for_status()
            body = resp.json()
            raw = body.get("choices", [{}])[0].get("message", {}).get("content", "")
            return self._safe_json(raw)
        except Exception:
            return None

    @staticmethod
    def _normalize(text: str) -> str:
        # Canonicalize whitespace and case for deterministic checks.
        return re.sub(r"\s+", " ", text.strip().lower())

    @staticmethod
    def _tokens(text: str) -> List[str]:
        # Basic tokenization used by phrase/term matching heuristics.
        cleaned = re.sub(r"[^\w]+", " ", text.lower(), flags=re.UNICODE)
        return [t for t in cleaned.split() if t]

    @staticmethod
    def _char_ngrams(text: str, n: int) -> Counter[str]:
        # Build cheap semantic vectors for cosine similarity routing.
        src = re.sub(r"\s+", " ", text.lower()).strip()
        if not src:
            return Counter()
        if len(src) < n:
            return Counter([src])
        return Counter(src[i : i + n] for i in range(len(src) - n + 1))

    @staticmethod
    def _cosine(a: Counter[str], b: Counter[str]) -> float:
        # Compute cosine similarity over n-gram frequency vectors.
        if not a or not b:
            return 0.0
        keys = set(a) | set(b)
        dot = 0.0
        na = 0.0
        nb = 0.0
        for k in keys:
            av = float(a.get(k, 0.0))
            bv = float(b.get(k, 0.0))
            dot += av * bv
            na += av * av
            nb += bv * bv
        denom = math.sqrt(na) * math.sqrt(nb)
        return dot / denom if denom else 0.0

    @staticmethod
    def _guess_language(text: str) -> str:
        # Minimal heuristic for non-English input detection in this demo engine.
        if not text:
            return "en"
        return "non_english_detected" if any(ord(ch) > 127 for ch in text) else "en"

    def _build_route_signatures(self) -> None:
        # Precompute route signatures so route scoring is fast and repeatable.
        for route in self.routes:
            rid = str(route.get("id", ""))
            src = " ".join(
                [
                    str(route.get("title", "")),
                    str(route.get("description", "")),
                    " ".join(route.get("phrase_terms", []) or []),
                    " ".join(route.get("intent_terms", []) or []),
                    " ".join(route.get("entity_terms", []) or []),
                ]
            ).lower()
            self.route_signatures[rid] = self._char_ngrams(src, self.semantic_n)

    def _heuristic_score(self, route: Dict[str, Any], norm_query: str, tokens: set[str]) -> Tuple[float, Dict[str, Any]]:
        # Route-level heuristic score computed from configured route metadata.
        score = 0.0
        reasons = {
            "phrases": [],
            "intents": [],
            "entities": [],
            "optional": [],
            "required_bundles": [],
            "negatives": [],
            "bundle_hits": 0,
            "bundle_total": len(route.get("required_bundles", []) or []),
        }

        phrase_w = float(route.get("phrase_weight", 1.5))
        intent_w = float(route.get("intent_weight", 1.0))
        entity_w = float(route.get("entity_weight", 0.8))
        optional_w = float(route.get("optional_weight", 0.4))
        required_w = float(route.get("required_weight", 1.2))
        miss_penalty = float(route.get("penalty_for_missing_required_bundle", 0.7))

        for phrase in route.get("phrase_terms", []) or []:
            p = str(phrase).lower()
            if p in norm_query:
                score += phrase_w
                reasons["phrases"].append(p)

        for term in route.get("intent_terms", []) or []:
            t = str(term).lower()
            if t in tokens:
                score += intent_w
                reasons["intents"].append(t)

        for term in route.get("entity_terms", []) or []:
            t = str(term).lower()
            if t in tokens:
                score += entity_w
                reasons["entities"].append(t)

        for term in route.get("optional_terms", []) or []:
            t = str(term).lower()
            if t in tokens:
                score += optional_w
                reasons["optional"].append(t)

        for bundle in route.get("required_bundles", []) or []:
            req = [str(x).lower() for x in (bundle or [])]
            if req and all(x in tokens for x in req):
                score += required_w
                reasons["required_bundles"].append(req)
                reasons["bundle_hits"] += 1

        if reasons["bundle_total"] and reasons["bundle_hits"] == 0:
            score -= miss_penalty

        for bad in route.get("negative_terms", []) or []:
            b = str(bad).lower()
            if b in tokens:
                score -= 1.0
                reasons["negatives"].append(b)

        return score, reasons

    def _semantic_score(self, route_id: str, norm_query: str) -> float:
        # Route semantic score from character n-gram cosine match.
        q = self._char_ngrams(norm_query, self.semantic_n)
        return self._cosine(q, self.route_signatures.get(route_id, Counter()))

    def _route_classifier_llm(self, query: str, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Secondary route classifier used only when heuristics are weak or ambiguous.
        if not self.llm_enabled:
            return {"selected_route": "", "confidence": 0.0}

        short_desc = [
            {
                "id": c["route_id"],
                "title": c["title"],
                "description": c["description"],
            }
            for c in candidates
        ]
        system = (
            "You are a strict route router. Pick exactly one route ID from known routes.\n"
            "Return strict JSON: {\"selected_route\":string, \"confidence\":0.0-1.0, \"reason\":string}.\n"
            f"Known routes: {json.dumps(short_desc, ensure_ascii=False)}"
        )
        out = self._llm_json(system, query, max_tokens=240)
        if not isinstance(out, dict):
            return {"selected_route": "", "confidence": 0.0}

        return {
            "selected_route": str(out.get("selected_route", "")).strip(),
            "confidence": float(out.get("confidence", 0.0)),
            "reason": str(out.get("reason", "")),
        }

    def phase_inspect_query(self, query: str) -> Dict[str, Any]:
        # First inspect phase: normalize query, derive signals, then optionally enrich with LLM.
        self._log(
            "parse_request",
            "InputInspector",
            "start request parse",
            {"query": query},
            {"step": "normalization"},
            "inspect_query",
            explanation="Raw user input is prepared before any routing decisions are made.",
        )

        norm = self._normalize(query)
        self._log(
            "inspect_query",
            "InputInspector",
            "normalize query",
            {"raw_query": query},
            {"normalized_query": norm},
            "inspect_query",
            explanation="Whitespace and casing are normalized so matching rules are stable.",
        )

        tokens = set(self._tokens(norm))
        self._log(
            "inspect_query",
            "InputInspector",
            "tokenize query",
            {"normalized_query": norm},
            {"token_count": len(tokens)},
            "inspect_query",
            explanation="A token set feeds heuristic term and required-bundle checks.",
        )

        deterministic = {
            "query": query,
            "language_hint": self._guess_language(query),
            "has_question_mark": "?" in query,
            "quoted_text": True if re.search(r'"[^"]+"', query) else False,
            "token_count": len(tokens),
        }
        self._log(
            "inspect_query",
            "InputInspector",
            "build deterministic fields",
            {"normalized_query": norm},
            {k: deterministic[k] for k in ["language_hint", "has_question_mark", "quoted_text", "token_count"]},
            "inspect_query",
            explanation="These fields never require network and are always available.",
        )

        profile = deterministic
        if self.llm_enabled and self.llm_cfg.get("enable_query_inspect_llm", True):
            self._log(
                "inspect_query",
                "InputInspector",
                "query inspection LLM requested",
                {"query": query},
                {"llm_enabled": True},
                "inspect_query",
                explanation="LLM overlays `intent_hint`, `wants_action`, and `needs_output` when configured.",
            )
            system = (
                "You extract a compact intent profile from the user request. "
                "Return strict JSON with keys: intent_hint, language_hint, wants_action, needs_output. "
                "No extra keys."
            )
            llm_profile = self._llm_json(system, query)
            if isinstance(llm_profile, dict):
                profile = {
                    "query": query,
                    "language_hint": llm_profile.get("language_hint", deterministic["language_hint"]),
                    "intent_hint": llm_profile.get("intent_hint"),
                    "wants_action": bool(llm_profile.get("wants_action", deterministic["has_question_mark"])),
                    "needs_output": bool(llm_profile.get("needs_output", True)),
                    "token_count": deterministic["token_count"],
                    "quoted_text": deterministic["quoted_text"],
                }
            else:
                llm_profile = {}
            self._log(
                "inspect_query",
                "InputInspector",
                "LLM profile merge result",
                {"intent_hint": llm_profile.get("intent_hint")},
                {"profile_keys": sorted(profile.keys())},
                "inspect_query",
                explanation="Merged fields become part of the same `query_profile` payload.",
            )

        profile["normalized_query"] = norm
        profile["token_set"] = tokens
        profile_snapshot = {k: profile[k] for k in profile if k != "token_set"}

        self._log(
            "inspect_query",
            "InputInspector",
            "extract request profile (deterministic + optional LLM)",
            {"query": query},
            {"query_profile": profile_snapshot},
            "route_classify",
            explanation="Profile is complete and routing can begin.",
        )
        return {"query_profile": profile}

    def phase_route_classify(self, query_profile: Dict[str, Any]) -> Dict[str, Any]:
        # Route selection phase: combine deterministic scoring and optional LLM rerank.
        norm = str(query_profile.get("normalized_query", ""))
        tokens = set(query_profile.get("token_set", []))
        lang = str(query_profile.get("language_hint", "en"))
        self._log(
            "route_classify",
            "LayeredRouter",
            "load query representation for scoring",
            {"query": query_profile.get("query"), "language_hint": lang},
            {"token_count": len(tokens)},
            "route_classify",
            explanation="Router receives parsed profile and computes scores per route.",
        )

        scored: List[Dict[str, Any]] = []
        for route in self.routes:
            rid = str(route.get("id", ""))
            h, reasons = self._heuristic_score(route, norm, tokens)
            s = self._semantic_score(rid, norm)
            base_h = h if h > 0 else 0.0001
            combined = float(self.router_cfg.get("heuristic_weight", 0.58)) * base_h + float(self.router_cfg.get("semantic_weight", 0.34)) * s
            scored.append(
                {
                    "route_id": rid,
                    "title": route.get("title", ""),
                    "description": route.get("description", ""),
                    "combined": combined,
                    "heuristic": h,
                    "semantic": s,
                    "reasons": reasons,
                }
            )

        scored.sort(key=lambda x: x["combined"], reverse=True)
        self._log(
            "route_classify",
            "LayeredRouter",
            "score all routes and select top candidates",
            {"route_total": len(scored)},
            {"top_route": scored[0]["route_id"] if scored else "none"},
            "route_classify",
            explanation="Heuristic and semantic scores are combined before confidence checks.",
        )
        candidates = scored[: int(self.router_cfg.get("candidate_top_k", 5))]
        top = candidates[0] if candidates else {"route_id": "fallback_unknown", "combined": 0.0}
        conf = float(top.get("combined", 0.0))
        second = candidates[1]["combined"] if len(candidates) > 1 else 0.0
        margin = conf - second
        self._log(
            "route_classify",
            "LayeredRouter",
            "calculate top-2 confidence gap",
            {"candidates": [c["route_id"] for c in candidates]},
            {"top": conf, "second": second, "margin": margin},
            "route_classify",
            explanation="Small gap indicates ambiguity and can trigger fallback behavior.",
        )

        direct_thr = float(self.router_cfg.get("direct_confidence_min", 0.52))
        ambiguity_margin = float(self.router_cfg.get("ambiguity_margin", 0.10))
        ask_thr = float(self.router_cfg.get("ask_clarification_threshold", 0.18))

        selected = top["route_id"] if top else "fallback_unknown"
        universal_route_id = next(
            (rid for rid, route in self.route_map.items() if route.get("is_universal_fallback")),
            None,
        )
        if (lang != "en") or conf < direct_thr or margin < ambiguity_margin:
            self._log(
                "route_classify",
                "LayeredRouter",
                "fallback condition hit -> optional LLM re-check",
                {"selected_route": selected, "conf": conf, "margin": margin},
                {"language_ok": lang == "en"},
                "route_classify",
                explanation="When non-English/uncertain, ask LLM only if configured.",
            )
            llm_route = self._route_classifier_llm(query_profile.get("query", ""), candidates)
            rr = llm_route.get("selected_route", "")
            rc = float(llm_route.get("confidence", 0.0))
            if rr in self.route_map and rc >= ask_thr:
                selected = rr
                conf = max(conf, rc)

        if selected == "fallback_unknown" and universal_route_id:
            selected = universal_route_id

        selected_route_config = self.route_map.get(str(selected), {})
        universal_route = bool(selected_route_config.get("is_universal_fallback", False))

        if selected == "fallback_unknown":
            selected = "ask_clarification"
            next_action = "ask_clarification"
        elif not universal_route and conf < ask_thr:
            selected = "ask_clarification"
            next_action = "ask_clarification"
        elif not universal_route and (conf < direct_thr or margin < ambiguity_margin):
            selected = "meta_route"
            next_action = "permission_check"
        else:
            next_action = "permission_check"

        out = {
            "selected_route": selected,
            "score": conf,
            "candidates": candidates,
            "next_action": next_action,
        }

        self._log(
            "route_classify",
            "LayeredRouter",
            "heuristic + semantic + optional LLM fallback",
            {"query_profile": {k: query_profile.get(k) for k in ["query", "language_hint"] if k in query_profile}},
            out,
            next_action,
            explanation="Router resolves final route and transitions to next phase.",
        )
        return out

    def phase_permission_check(self, route_id: str) -> Dict[str, Any]:
        # Permission phase validates that the current policy allows every required step.
        route = self.route_map.get(route_id, {})
        required = set(route.get("required_permissions", []))
        required.update(
            {
                "inspect_query",
                "route_classify",
                "permission_check",
                "extract_slots",
                "build_plan",
                "summarize",
                "validate",
            }
        )
        if route_id == "run_tests":
            required.add("execute_tools")

        missing = sorted(p for p in required if p in self.denied or p not in self.allowed)
        self._log(
            "permission_check",
            "PolicyGate",
            "build required permission set",
            {"route": route_id},
            {"required_permissions": sorted(required)},
            "ask_clarification" if missing else "extract_slots",
            explanation="Permission list includes route-specific and pipeline safety permissions.",
        )
        if missing:
            self._log(
                "permission_check",
                "PolicyGate",
                "permission validation",
                {"required": sorted(required)},
                {"allowed": False, "missing_permissions": missing},
                "ask_clarification",
                explanation="Missing permissions move flow to ask_clarification.",
            )
            return {"allowed": False, "missing_permissions": missing, "next_action": "ask_clarification"}

        self._log(
            "permission_check",
            "PolicyGate",
            "permission validation",
            {"required": sorted(required)},
            {"allowed": True},
            "extract_slots",
            explanation="No permission block found, continue to slot extraction.",
        )
        return {"allowed": True, "missing_permissions": [], "next_action": "extract_slots"}

    def phase_extract_slots(self, route_id: str, query_profile: Dict[str, Any]) -> Dict[str, Any]:
        # Slot extraction maps route-specific arguments from query or LLM output.
        route = self.route_map.get(route_id, {})
        query = str(query_profile.get("query", ""))
        slots: Dict[str, Any] = {}
        self._log(
            "extract_slots",
            "SlotExtractor",
            "start slot extraction",
            {"route": route_id},
            {"query": query},
            "extract_slots",
            explanation="Collect required inputs for the selected route before planning.",
        )

        if self.llm_enabled and self.llm_cfg.get("enable_slot_llm", True):
            prompt = route.get("prompts", {}).get("slot_extractor")
            if prompt:
                llm_slots = self._llm_json(prompt, query)
                if isinstance(llm_slots, dict):
                    slots.update({k: v for k, v in llm_slots.items() if v is not None})

        if route_id == "llm_answer":
            slots.setdefault("question", query)
            slots.setdefault("language_guess", query_profile.get("language_hint", "en"))

        elif route_id == "summarize_doc":
            if not slots.get("source_text"):
                m = re.search(r'"([^"]+)"', query)
                slots["source_text"] = m.group(1) if m else query
            slots.setdefault("target_length", "normal")

        elif route_id == "run_tests":
            if not slots.get("scope"):
                m = re.search(r"pytest\s+([^\n]+)", query)
                slots["scope"] = m.group(1).strip() if m else None

        elif route_id == "bug_repair":
            slots.setdefault("bug_text", query)

        elif route_id == "schedule_email":
            if not slots.get("to"):
                m = re.search(r"\bto\s+([\w.+-]+@[\w.-]+)", query, re.IGNORECASE)
                slots["to"] = m.group(1) if m else None
            if not slots.get("subject"):
                m = re.search(r"subject\s*: ?([^\n]+)", query, re.IGNORECASE)
                slots["subject"] = m.group(1).strip() if m else None
            if not slots.get("body"):
                m = re.search(r"body\s*: ?([^\n]+)", query, re.IGNORECASE)
                if m:
                    slots["body"] = m.group(1).strip()
                elif " body " in self._normalize(query):
                    parts = self._normalize(query).split(" body ", 1)
                    if len(parts) > 1:
                        slots["body"] = parts[1].strip()

        elif route_id == "package_payload":
            if not slots.get("package_name"):
                m = re.search(
                    r"\b(?:package|project)\s+(?:named|name|called)\s+([A-Za-z0-9_\-]+)",
                    query,
                    re.IGNORECASE,
                )
                if m:
                    slots["package_name"] = m.group(1).strip().strip('"').strip("'")
            if not slots.get("version"):
                m = re.search(r"\bversion\s*[:=]?\s*([0-9]+(?:\.[0-9]+){0,2})\b", query, re.IGNORECASE)
                if m:
                    slots["version"] = m.group(1).strip()
            if not slots.get("description"):
                m = re.search(r"\bdescription\s*[:=]?\s*\"([^\"]+)\"", query, re.IGNORECASE)
                if m:
                    slots["description"] = m.group(1).strip()
                else:
                    m = re.search(r"package to build for\s+([A-Za-z0-9_\- ]+)", query, re.IGNORECASE)
                    if m:
                        slots["description"] = m.group(1).strip()

            if not slots.get("dependencies"):
                m = re.search(r"\bdependencies?\b[:=]?\s*([^\n]+)", query, re.IGNORECASE)
                if m:
                    raw = m.group(1)
                    items = [x.strip() for x in re.split(r"[;,]", raw) if x.strip()]
                    slots["dependencies"] = [x for x in items if x]

            if not slots.get("scripts"):
                m = re.search(r"\bscripts?\b[:=]?\s*([^\n]+)", query, re.IGNORECASE)
                if m:
                    raw = m.group(1)
                    scripts = [x.strip() for x in re.split(r"[;,]", raw) if x.strip()]
                    slots["scripts"] = scripts

            if not slots.get("files"):
                m = re.search(r"\bfiles?\b[:=]?\s*([^\n]+)", query, re.IGNORECASE)
                if m:
                    raw = m.group(1)
                    slots["files"] = [x.strip() for x in re.split(r"[;,]", raw) if x.strip()]

            if not slots.get("tests"):
                m = re.search(r"\btests?\b[:=]?\s*([^\n]+)", query, re.IGNORECASE)
                if m:
                    raw = m.group(1)
                    slots["tests"] = [x.strip() for x in re.split(r"[;,]", raw) if x.strip()]

            if not slots.get("entry_points"):
                m = re.search(r"entry\s*points?\s*[:=]?\s*([^\n]+)", query, re.IGNORECASE)
                if m:
                    raw = m.group(1)
                    ep = {}
                    for piece in [x.strip() for x in raw.split(",") if x.strip()]:
                        if ":" in piece:
                            left, right = piece.split(":", 1)
                            ep[left.strip()] = right.strip()
                    if ep:
                        slots["entry_points"] = ep

        required_slots = route.get("required_slots", {}) or {}
        missing = []
        for slot_name, meta in required_slots.items():
            if bool(meta.get("required", False)) and not slots.get(slot_name):
                missing.append(slot_name)
                self._log(
                    "extract_slots",
                    "SlotExtractor",
                    "required slot missing",
                    {"slot_name": slot_name, "route": route_id},
                    {"found_slots": list(slots.keys())},
                    "ask_clarification",
                    explanation="Route requires this slot before we can build a safe plan.",
                )

        next_action = "build_plan" if not missing else "ask_clarification"
        self._log(
            "extract_slots",
            "SlotExtractor",
            f"extract slots for {route_id}",
            {"route": route_id, "query": query},
            {"slots": slots, "missing": missing},
            next_action,
            explanation="If any required slot is missing, ask for clarification instead of executing.",
        )
        return {"slots": slots, "missing": missing, "next_action": next_action}

    def phase_build_plan(self, route_id: str, slots: Dict[str, Any]) -> Dict[str, Any]:
        # Build an execution plan from route tools, resolving concrete arguments from slots.
        route = self.route_map.get(route_id, {})
        plan: List[Dict[str, Any]] = []
        for tool in route.get("tools", []) or []:
            arguments: Dict[str, Any] = {}
            if tool == "run_tests":
                arguments = {"scope": slots.get("scope")}
            call = {"name": tool, "arguments": arguments}
            plan.append(call)
            self._log(
                "build_plan",
                "Planner",
                "append tool call",
                {"route": route_id, "tool": call["name"]},
                call,
                "build_plan",
                explanation="Each supported tool contributes a single callable plan row.",
            )

        self._log(
            "build_plan",
            "Planner",
            "build route-specific tool plan",
            {"route": route_id, "slots": slots},
            {"plan": plan},
            "execute_tools" if plan else "summarize",
            explanation="Empty plan means route can summarize directly without tool execution.",
        )
        return {"plan": plan, "next_action": "execute_tools" if plan else "summarize"}

    def _tool_run_tests(self, args: Dict[str, Any]) -> Dict[str, Any]:
        # Execute pytest in a centralized registry-backed tool path.
        result = execute_tool("run_tests", args if isinstance(args, dict) else {})
        return self._normalize_tool_result("run_tests", args if isinstance(args, dict) else {}, result)

    def _tool_run_sandbox(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        # Execute declared heavy tools in Docker sandbox mode.
        payload = args if isinstance(args, dict) else {}
        start = self._utc_now()
        if name != "run_tests":
            return {
                "tool": name,
                "success": False,
                "arguments": redact_sensitive_args(payload),
                "error": "tool_sandbox_exec_error",
                "error_code": "tool_sandbox_exec_error",
                "sandbox": "docker",
                "started_at": start,
                "finished_at": self._utc_now(),
            }

        if shutil.which("docker") is None:
            return {
                "tool": name,
                "success": False,
                "arguments": redact_sensitive_args(payload),
                "error": "tool_sandbox_unavailable",
                "error_code": "tool_sandbox_unavailable",
                "sandbox": "docker",
                "started_at": start,
                "finished_at": self._utc_now(),
            }

        scope = str(payload.get("scope", "")).strip() if isinstance(payload.get("scope"), str) else ""
        cmd = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{Path.cwd()}:/workspace",
            "-w",
            "/workspace",
            self.sandbox_image,
            "pytest",
        ]
        if scope:
            cmd.append(scope)

        try:
            proc = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=max(5, self.sandbox_timeout))
        except subprocess.TimeoutExpired as exc:
            return {
                "tool": name,
                "success": False,
                "arguments": redact_sensitive_args(payload),
                "error": "tool_sandbox_timeout",
                "error_code": "tool_sandbox_timeout",
                "sandbox": "docker",
                "command": cmd,
                "stdout": str(exc.stdout or ""),
                "stderr": str(exc.stderr or ""),
                "started_at": start,
                "finished_at": self._utc_now(),
            }
        except OSError as exc:
            return {
                "tool": name,
                "success": False,
                "arguments": redact_sensitive_args(payload),
                "error": "tool_sandbox_exec_error",
                "error_code": "tool_sandbox_exec_error",
                "sandbox": "docker",
                "command": cmd,
                "error_detail": str(exc),
                "started_at": start,
                "finished_at": self._utc_now(),
            }

        payload_out = {
            "sandbox": "docker",
            "command": cmd,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
        if proc.returncode == 0:
            return {
                "tool": name,
                "success": True,
                "arguments": redact_sensitive_args(payload),
                "error": None,
                "error_code": None,
                "sandbox": "docker",
                "output": payload_out,
                "started_at": start,
                "finished_at": self._utc_now(),
            }
        return {
            "tool": name,
            "success": False,
            "arguments": redact_sensitive_args(payload),
            "error": "tool_return_nonzero",
            "error_code": "tool_return_nonzero",
            "sandbox": "docker",
            "output": payload_out,
            "started_at": start,
            "finished_at": self._utc_now(),
        }

    def _tool_run_generic(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        # Execute any declared tool through the registry for typed outcomes.
        result = execute_tool(name, args if isinstance(args, dict) else {})
        return self._normalize_tool_result(name, args if isinstance(args, dict) else {}, result)

    def _normalize_tool_result(self, name: str, args: Dict[str, Any], result: Any) -> Dict[str, Any]:
        # Normalize registry result payload for stable checkpoint/error behavior.
        normalized: Dict[str, Any] = {
            "tool": name,
            "success": bool(result.success),
            "arguments": redact_sensitive_args(args),
            "error": result.error,
            "error_code": result.error_code,
        }
        if result.started_at:
            normalized["started_at"] = result.started_at
        if result.finished_at:
            normalized["finished_at"] = result.finished_at
        if result.sandbox:
            normalized["sandbox"] = result.sandbox
        if isinstance(result.output, dict):
            output_payload = redact_sensitive_args(result.output)
            normalized["output"] = output_payload
            for output_key, output_value in output_payload.items():
                if output_key not in normalized:
                    normalized[output_key] = output_value
        elif result.output is not None:
            normalized["output"] = redact_sensitive_args(result.output)
        return normalized

    def _guard_tool_output(self, out: Dict[str, Any]) -> Dict[str, Any]:
        result = check_tool_output(json.dumps(out, ensure_ascii=False, default=str))
        if result.allow:
            return out
        out = dict(out)
        out["success"] = False
        out["error"] = result.reason or "tool_output_blocked"
        out["error_code"] = "tool_output_blocked"
        return out

    def phase_execute_tools(self, route_id: str, plan: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Dispatch each plan row and collect output objects, tracking call budget.
        outputs: List[Dict[str, Any]] = []
        budget_exhausted = False
        sandbox_tools = set(self._route_sandbox_tools(route_id))
        for index, call in enumerate(plan):
            if self.tool_calls_used >= self.max_tool_calls:
                budget_exhausted = True
                self._log(
                    "execute_tools",
                    "ToolExecutor",
                    "tool budget guard",
                    {"planned_tool": call.get("name")},
                    {"tool_calls_so_far": self.tool_calls_used, "max_tool_calls": self.max_tool_calls},
                    "summarize",
                    explanation="Stop dispatching when tool budget is exceeded.",
                )
                outputs.append(
                    {
                        "tool": "tool_budget_guard",
                        "success": False,
                        "error": "tool_calls_budget_exceeded",
                        "error_code": "tool_calls_budget_exceeded",
                        "branch": "tool_budget_guard",
                        "tool_calls_limit": self.max_tool_calls,
                        "plan_index": index,
                        "planned_tool_count": len(plan),
                    }
                )
                break

            name = str(call.get("name", ""))
            args = call.get("arguments", {}) if isinstance(call.get("arguments", {}), dict) else {}
            if not args:
                args = {}
            gate = check_tool_request_with_tool(args, tool_name=name)
            if not gate.allow:
                outputs.append(
                    {
                        "tool": name,
                        "success": False,
                        "arguments": redact_sensitive_args(args),
                        "error": gate.reason or "tool_request_blocked",
                        "error_code": "tool_request_blocked",
                        "started_at": self._utc_now(),
                        "finished_at": self._utc_now(),
                        "tool_budget_blocked": True,
                    }
                )
                self._log(
                    "execute_tools",
                    "ToolExecutor",
                    "tool request blocked",
                    {"name": name},
                    {"error": gate.reason},
                    "summarize",
                    explanation="Pre-tool guard blocked execution before dispatch.",
                )
                self.tool_calls_used += 1
                continue
            self._log(
                "execute_tools",
                "ToolExecutor",
                "dispatch tool",
                {"name": name, "args": args},
                {"status": "pending"},
                "execute_tools",
                explanation="Call the next tool in plan order.",
            )

            is_sandboxed = self.tool_sandbox_mode == "docker" and (name == "run_tests" or name in sandbox_tools)
            if is_sandboxed:
                out = self._tool_run_sandbox(name, args)
            elif name == "run_tests":
                out = self._tool_run_tests(args)
            else:
                out = self._tool_run_generic(name, args)

            if out.get("error_code") not in {None, "tool_not_allowed", "unknown_tool"}:
                out = self._guard_tool_output(out)

            self.tool_calls_used += 1
            self._log(
                "execute_tools",
                "ToolExecutor",
                "tool call completed",
                {"name": name},
                out,
                "summarize",
                explanation="Append execution output and continue until all plan entries are processed.",
            )
            outputs.append(out)

        execute_status = "ok"
        error_codes = {str(item.get("error_code") or "") for item in outputs}
        has_failure = any(not bool(item.get("success", False)) for item in outputs)
        if budget_exhausted:
            execute_status = "tool_budget_exhausted"
        elif has_failure:
            execute_status = "partial_fail"
        if not outputs:
            outputs.append(
                {
                    "tool": "none",
                    "success": True,
                    "arguments": {},
                    "error": None,
                    "error_code": None,
                    "started_at": self._utc_now(),
                    "finished_at": self._utc_now(),
                    "tool_outputs": [],
                }
            )

        self._checkpoint(
            phase="execute_tools",
            run_id=self.current_run_id or str(uuid.uuid4()),
            route_id=route_id,
            attempt=self.attempt,
            status=execute_status,
            next_action="summarize",
            error_code=(
                "tool_budget_exhausted"
                if budget_exhausted
                else (
                    "tool_output_blocked"
                    if "tool_output_blocked" in error_codes
                    else ("tool_return_nonzero" if "tool_return_nonzero" in error_codes else None)
                )
            ),
            payload={
                "tool_outputs": outputs,
                "tool_calls_used": self.tool_calls_used,
                "max_tool_calls": self.max_tool_calls,
                "tool_budget_guarded": budget_exhausted,
                "tool_budget_limit": self.max_tool_calls,
                "planned_tool_count": len(plan),
                "error_codes": sorted(code for code in error_codes if code),
            },
        )

        self._log(
            "execute_tools",
            "ToolExecutor",
            "execute planned tools",
            {"plan": plan},
            {"tool_outputs": outputs},
            "summarize",
            explanation="Consolidate all per-tool outputs before summarization.",
        )
        return {"tool_outputs": outputs, "next_action": "summarize"}

    def _summarize(self, route_id: str, slots: Dict[str, Any], tool_outputs: List[Dict[str, Any]]) -> str:
        # Convert internal route state into a human-facing final response.
        route = self.route_map.get(route_id, {})
        prompt = route.get("prompts", {}).get("summarizer")

        if route_id == "llm_answer":
            q = slots.get("question")
            if self.llm_enabled and prompt:
                out = self._llm_json(prompt, str(q))
                if isinstance(out, dict) and out.get("summary"):
                    return str(out.get("summary"))
            return "I can answer that, but need additional context or explicit request format."

        if route_id == "summarize_doc":
            text = str(slots.get("source_text", ""))
            if self.llm_enabled and prompt:
                out = self._llm_json(prompt, text, max_tokens=240)
                if isinstance(out, dict) and out.get("summary"):
                    return str(out.get("summary"))
            compact = " ".join(text.split())
            return compact[:700] + ("..." if len(compact) > 700 else "")

        if route_id == "run_tests":
            if not tool_outputs:
                return "No tests were run."
            t = tool_outputs[0]
            if t.get("returncode") is None and isinstance(t.get("output"), dict):
                nested = t.get("output") or {}
                returncode = nested.get("returncode")
                if returncode is not None and "returncode" not in t:
                    t = dict(t)
                    t["returncode"] = returncode
            if t.get("success"):
                return "Run tests passed."
            return f"Run tests failed: rc={t.get('returncode')}."

        if route_id == "bug_repair":
            bug = slots.get("bug_text", "")
            if self.llm_enabled and prompt:
                out = self._llm_json(prompt, str(bug), max_tokens=260)
                if isinstance(out, dict) and out.get("summary"):
                    return str(out.get("summary"))
            return "I need stack trace and code context to propose a concrete repair plan."

        if route_id == "schedule_email":
            if slots.get("to") and slots.get("subject") and slots.get("body"):
                return f"Prepared email draft to {slots.get('to')} with subject '{slots.get('subject')}'."
            return "I need recipient, subject, and body before drafting or scheduling email."

        if route_id == "package_payload":
            payload = {
                "name": slots.get("package_name", ""),
                "version": slots.get("version") or "0.1.0",
                "description": slots.get("description", ""),
                "dependencies": slots.get("dependencies", []) if isinstance(slots.get("dependencies"), list) else [],
                "scripts": slots.get("scripts", []) if isinstance(slots.get("scripts"), list) else [],
                "entry_points": (
                    slots.get("entry_points")
                    if isinstance(slots.get("entry_points"), dict)
                    else {}
                ),
                "files": slots.get("files", []) if isinstance(slots.get("files"), list) else [],
                "tests": slots.get("tests", []) if isinstance(slots.get("tests"), list) else [],
            }

            if self.llm_enabled and prompt:
                out = self._llm_json(
                    prompt,
                    json.dumps({"slots": slots}, ensure_ascii=False),
                    max_tokens=320,
                )
                if isinstance(out, dict):
                    if isinstance(out.get("summary"), str):
                        return out.get("summary")
                    if isinstance(out.get("package_payload"), dict):
                        return json.dumps(out.get("package_payload"), ensure_ascii=False)

            return json.dumps({"package_payload": payload}, ensure_ascii=False)

        if route_id in {"ask_clarification", "meta_route"}:
            return "Clarification required before taking action."

        return "Completed."

    def phase_summarize(self, route_id: str, slots: Dict[str, Any], tool_outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Build the final response text and transition into validation.
        final_response = self._summarize(route_id, slots, tool_outputs)
        self._log(
            "summarize",
            "Responder",
            "build final response text",
            {"route": route_id},
            {"final_response": final_response},
            "validate",
            explanation="Final response now contains route-specific answer and will be validated.",
        )
        return {"final_response": final_response, "next_action": "validate"}

    def phase_validate(self, route_id: str, final_response: str, tool_outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Validate final evidence/format before marking the run as successful.
        route_meta = self.route_map.get(route_id, {})
        validator = self._route_validator(route_meta)
        required = [str(x) for x in validator.get("required_evidence_fields", [])]
        hard_fail = [str(x) for x in validator.get("hard_fail_errors", [])]
        hard_fail_set = set(hard_fail)
        hard_fail_set.add("unknown_tool")

        evidence_rows: list[Dict[str, Any]] = []
        for row in tool_outputs:
            if isinstance(row, dict):
                evidence_rows.append(row)
                output = row.get("output")
                if isinstance(output, dict):
                    evidence_rows.append(output)

        if final_response is not None:
            evidence_rows.append({"final_response": redact_sensitive_args(final_response)})

        missing: list[str] = []
        for field in required:
            found = False
            for row in evidence_rows:
                if row.get(field) not in (None, "", []):
                    found = True
                    break
            if not found:
                missing.append(field)

        tool_error_codes = []
        for row in tool_outputs:
            if isinstance(row, dict):
                error_code = row.get("error_code")
                if isinstance(error_code, str) and error_code:
                    tool_error_codes.append(error_code)
        hard_fail_from_tools = any(code in hard_fail_set for code in tool_error_codes)

        if required:
            if self.require_evidence:
                ok = not missing
            else:
                ok = not hard_fail_from_tools
        else:
            ok = not hard_fail_from_tools

        next_action = "report" if ok else "ask_clarification"
        status = "ok" if ok else "failed"
        detail = {"required": required, "found": evidence_rows, "missing_fields": missing}
        if not ok:
            reason = "missing_required_evidence" if (self.require_evidence and missing) else "insufficient_fields"
            if self.require_evidence and hard_fail:
                reason = next((x for x in hard_fail if "missing" in x or "required" in x), reason)
            self._log(
                "validate",
                "PolicyValidator",
                "required fields check failed",
                {"route": route_id, "required": required},
                {"ok": False, "missing_fields": missing, "failed_route_ids": [route_id]},
                "ask_clarification",
                explanation="Route validation failed due to missing required evidence fields.",
            )
            return {
                "ok": False,
                "next_action": next_action,
                "missing_fields": missing,
                "failed_route_ids": [route_id],
                "reason": reason,
                "validation": {"status": status, "detail": detail},
            }

        self._log(
            "validate",
            "PolicyValidator",
            "required fields check passed",
            {"route": route_id, "required": required},
            {"ok": True},
            "report",
            explanation="All required evidence fields are present.",
        )
        return {
            "ok": True,
            "next_action": next_action,
            "missing_fields": [],
            "failed_route_ids": [],
            "validation": {"status": status, "detail": detail},
        }

    def phase_meta_route(self, query: str) -> Dict[str, Any]:
        # Meta route tries one more pass at selecting a concrete route from LLM.
        if not self.llm_enabled:
            return {"route_id": "ask_clarification", "next_action": "ask_clarification", "reason": "llm_disabled"}

        prompt = (
            "Return strict JSON with keys temporary_plan, reason, and next_route (optional). "
            "Use an existing route id from known routes if possible."
        )
        meta = self._llm_json(prompt, query, max_tokens=240)
        if not isinstance(meta, dict):
            return {"route_id": "ask_clarification", "next_action": "ask_clarification", "reason": "invalid_llm_response"}

        nr = str(meta.get("next_route", "")).strip()
        if nr in self.route_map:
            return {"route_id": nr, "next_action": "permission_check", "temporary_plan": meta.get("temporary_plan")}
        return {"route_id": "ask_clarification", "next_action": "ask_clarification", "temporary_plan": meta}

    def run(self, query: str) -> Dict[str, Any]:
        # Top-level orchestration for one request, including all phase transitions.
        self.phase_trace = []
        self.tool_calls_used = 0
        self.attempt = 1
        self.current_run_id = str(uuid.uuid4())
        run_id = self.current_run_id
        self.checkpoint_files = []
        trusted_input, untrusted_input = split_trusted_untrusted(query)
        sanitized_query = sanitize_text(untrusted_input or trusted_input or query)

        self._log(
            "main",
            "HarnessRuntime",
            "run() invoked",
            {"query": sanitized_query},
            {"attempt": self.attempt},
            "inspect_query",
            explanation="Every run starts from main and proceeds through inspect_query next.",
        )
        self._checkpoint(
            phase="bootstrap",
            run_id=run_id,
            route_id="bootstrap",
            attempt=self.attempt,
            status="ok",
            next_action="inspect_query",
            payload={"query": sanitized_query},
        )

        input_decision = check_input_text(sanitized_query)
        if not input_decision.allow:
            self._log(
                "main",
                "HarnessRuntime",
                "input blocked by policy",
                {"query": sanitized_query},
                {"reason": input_decision.reason},
                "stop",
                explanation="Input guard triggered before inspection/symbolic routing.",
            )
            self._checkpoint(
                phase="input_check",
                run_id=run_id,
                route_id="bootstrap",
                attempt=self.attempt,
                status="blocked",
                next_action="ask_clarification",
                error_code=input_decision.reason or "input_blocked",
                payload={"query": sanitized_query},
            )
            return self._final_payload(
                status="blocked",
                next_action="ask_clarification",
                final_response="Input blocked by policy.",
                route_id="bootstrap",
                extra={
                    "validation": {"ok": False, "missing_fields": [], "failed_route_ids": ["bootstrap"]},
                    "guard": {"input": {"allow": False, "reason": input_decision.reason}},
                },
            )

        if not sanitized_query.strip():
            self._log(
                "main",
                "HarnessRuntime",
                "reject empty input",
                {"query": sanitized_query},
                {"status": "blocked"},
                "stop",
                explanation="Empty query is not executable and needs clarification.",
            )
            self._checkpoint(
                phase="input_check",
                run_id=run_id,
                route_id="bootstrap",
                attempt=self.attempt,
                status="blocked",
                next_action="ask_clarification",
                error_code="empty_input",
                payload={"query": sanitized_query},
            )
            return self._final_payload(
                status="blocked",
                next_action="ask_clarification",
                final_response=None,
                route_id="bootstrap",
            )

        inspect = self.phase_inspect_query(sanitized_query)
        self._checkpoint(
            phase="inspect_query",
            run_id=run_id,
            route_id="bootstrap",
            attempt=self.attempt,
            status="ok",
            next_action="route_classify",
            payload={"query_profile": inspect.get("query_profile")},
        )
        self._log(
            "main",
            "HarnessRuntime",
            "inspect_query returned profile",
            {"query": query},
            {"query_profile_keys": list(inspect["query_profile"].keys())},
            "route_classify",
            explanation="Parsed profile now feeds route classification.",
        )
        qprof = inspect["query_profile"]
        route_result = self.phase_route_classify(qprof)
        route_id = route_result["selected_route"]
        self._checkpoint(
            phase="route_classify",
            run_id=run_id,
            route_id=route_id,
            attempt=self.attempt,
            status="ok",
            next_action=route_result.get("next_action"),
            payload={
                "selected_route": route_id,
                "score": route_result.get("score"),
                "next_action": route_result.get("next_action"),
            },
        )
        self._log(
            "main",
            "HarnessRuntime",
            "route selection complete",
            {"query": query},
            {"route_id": route_id, "next_action": route_result.get("next_action")},
            route_result.get("next_action", "inspect_query"),
            explanation="Continue only when route is clear; otherwise request clarification.",
        )

        if route_id == "meta_route":
            self._log(
                "main",
                "HarnessRuntime",
                "enter meta route expansion",
                {"route_id": route_id},
                {"status": "rerouting"},
                "permission_check",
                explanation="Meta route lets LLM choose a better concrete route.",
            )
            meta = self.phase_meta_route(query)
            self._checkpoint(
                phase="meta_route",
                run_id=run_id,
                route_id="meta_route",
                attempt=self.attempt,
                status="ok" if meta.get("route_id") != "ask_clarification" else "blocked",
                next_action="permission_check" if meta.get("route_id") != "ask_clarification" else "ask_clarification",
                payload={"route_id": meta.get("route_id"), "reason": meta.get("reason", "")},
            )
            if meta.get("route_id") == "ask_clarification":
                self._log(
                    "main",
                    "HarnessRuntime",
                    "meta route unresolved",
                    {"query": query},
                    {"reason": meta.get("reason")},
                    "stop",
                    explanation="Unable to resolve a concrete route, so ask user for clarity.",
                )
                self._checkpoint(
                    phase="route_classify",
                    run_id=run_id,
                    route_id="meta_route",
                    attempt=self.attempt,
                    status="blocked",
                    error_code="meta_route_unresolved",
                    payload={"reason": meta.get("reason")},
                )
                return self._final_payload(
                    status="clarification_required",
                    next_action="ask_clarification",
                    final_response=meta,
                    route_id="meta_route",
                    validation={"ok": False, "failed_route_ids": ["meta_route"]},
                    extra={"meta": meta},
                )
            route_id = meta.get("route_id")
            self._log(
                "main",
                "HarnessRuntime",
                "meta route produced concrete route",
                {"query": query},
                {"route_id": route_id},
                "permission_check",
                explanation="Rerouted to a concrete route; proceed with normal policy flow.",
            )

        if route_id not in self.route_map:
            route_id = "fallback_unknown"
            self._log(
                "main",
                "HarnessRuntime",
                "fallback to unknown route",
                {"query": query},
                {"route_id": route_id},
                "ask_clarification",
                explanation="Unknown route ids are not executable; request clarification.",
            )
            self._checkpoint(
                phase="route_classify",
                run_id=run_id,
                route_id=route_id,
                attempt=self.attempt,
                status="blocked",
                error_code="unknown_route",
                payload={"route_id": route_id},
            )

        if route_result.get("next_action") == "ask_clarification":
            self._log(
                "ask_clarification",
                "PolicyRoute",
                "low-confidence route mapping",
                {"query": query},
                {"route": route_id, "score": route_result.get("score")},
                "stop",
                explanation="Confidence was too low or ambiguous for autonomous execution.",
            )
            self._checkpoint(
                phase="route_classify",
                run_id=run_id,
                route_id=route_id,
                attempt=self.attempt,
                status="needs_clarification",
                payload={"score": route_result.get("score")},
            )
            return self._final_payload(
                status="clarification_required",
                next_action="ask_clarification",
                final_response="Please rephrase with an explicit task and constraints.",
                route_id=route_id,
                validation={"ok": False, "failed_route_ids": [route_id]},
            )

        perm = self.phase_permission_check(route_id)
        if not perm.get("allowed", False):
            self._log(
                "main",
                "HarnessRuntime",
                "blocked by permissions",
                {"route_id": route_id},
                {"missing_permissions": perm.get("missing_permissions", [])},
                "stop",
                explanation="Policy denies route execution until required permissions are granted.",
            )
            self._checkpoint(
                phase="permission_check",
                run_id=run_id,
                route_id=route_id,
                attempt=self.attempt,
                status="blocked",
                error_code="missing_permissions",
                payload={"missing_permissions": perm.get("missing_permissions", [])},
            )
            return self._final_payload(
                status="blocked",
                next_action="ask_clarification",
                final_response={"missing_permissions": perm.get("missing_permissions", [])},
                route_id=route_id,
            )

        slots_result = self.phase_extract_slots(route_id, qprof)
        self._checkpoint(
            phase="extract_slots",
            run_id=run_id,
            route_id=route_id,
            attempt=self.attempt,
            status="ok" if slots_result.get("next_action") != "ask_clarification" else "needs_clarification",
            payload={"slots": slots_result.get("slots"), "missing": slots_result.get("missing")},
        )
        self._log(
            "main",
            "HarnessRuntime",
            "slot extraction done",
            {"route": route_id},
            {"slots_found": list(slots_result.get("slots", {}).keys()), "missing": slots_result.get("missing")},
            slots_result.get("next_action"),
            explanation="Slots are required for planning; missing slots require clarification.",
        )
        if slots_result.get("next_action") == "ask_clarification":
            self._log(
                "main",
                "HarnessRuntime",
                "slots missing",
                {"route": route_id},
                {"missing_slots": slots_result.get("missing", [])},
                "stop",
                explanation="Cannot execute until required slots are provided.",
            )
            self._checkpoint(
                phase="extract_slots",
                run_id=run_id,
                route_id=route_id,
                attempt=self.attempt,
                status="needs_clarification",
                error_code="missing_slots",
                payload={"missing_slots": slots_result.get("missing", [])},
            )
            return self._final_payload(
                status="clarification_required",
                next_action="ask_clarification",
                final_response={"missing_slots": slots_result.get("missing", [])},
                route_id=route_id,
                validation={
                    "ok": False,
                    "failed_route_ids": [route_id],
                    "missing_fields": slots_result.get("missing", []),
                },
            )

        plan_result = self.phase_build_plan(route_id, slots_result.get("slots", {}))
        self._checkpoint(
            phase="build_plan",
            run_id=run_id,
            route_id=route_id,
            attempt=self.attempt,
            status="ok",
            payload={"plan": plan_result.get("plan", [])},
        )
        if plan_result.get("next_action") == "execute_tools":
            self._log(
                "main",
                "HarnessRuntime",
                "plan ready, executing tools",
                {"route": route_id},
                {"plan_len": len(plan_result.get("plan", []))},
                "execute_tools",
                explanation="Plan is non-empty, so runtime executes each tool.",
            )
            exec_result = self.phase_execute_tools(route_id, plan_result.get("plan", []))
            tool_outputs = exec_result.get("tool_outputs", [])
            self._checkpoint(
                phase="execute_tools",
                run_id=run_id,
                route_id=route_id,
                attempt=self.attempt,
                status="ok",
                payload={"tool_outputs": tool_outputs},
            )
        else:
            self._log(
                "main",
                "HarnessRuntime",
                "plan empty, skip execute_tools",
                {"route": route_id},
                {"plan_next_action": plan_result.get("next_action")},
                "summarize",
                explanation="Some routes are informational and do not require tool calls.",
            )
            self._checkpoint(
                phase="execute_tools",
                run_id=run_id,
                route_id=route_id,
                attempt=self.attempt,
                status="skipped",
                payload={"plan_next_action": plan_result.get("next_action")},
            )
            tool_outputs = []

        summary_result = self.phase_summarize(route_id, slots_result.get("slots", {}), tool_outputs)
        self._checkpoint(
            phase="summarize",
            run_id=run_id,
            route_id=route_id,
            attempt=self.attempt,
            status="ok",
            payload={"tool_output_count": len(tool_outputs)},
        )
        final_response = summary_result.get("final_response", "")
        validation = self.phase_validate(route_id, final_response, tool_outputs)
        self._checkpoint(
            phase="validate",
            run_id=run_id,
            route_id=route_id,
            attempt=self.attempt,
            status="ok" if validation.get("ok") else "failed",
            payload=validation,
        )
        self._log(
            "main",
            "HarnessRuntime",
            "validation completed",
            {"route": route_id, "tool_outputs": len(tool_outputs)},
            {"validation_ok": validation.get("ok", False)},
            validation.get("next_action", "final"),
            explanation="Validation decides whether run becomes ok or asks for clarification.",
        )

        if not validation.get("ok", False):
            self._log(
                "main",
                "HarnessRuntime",
                "final validation failed",
                {"route": route_id},
                {"validation": validation},
                "stop",
                explanation="Cannot mark run successful without required evidence.",
            )
            self._checkpoint(
                phase="final_report",
                run_id=run_id,
                route_id=route_id,
                attempt=self.attempt,
                status="failed",
                payload=validation,
            )
            return self._final_payload(
                status="validation_failed",
                next_action=validation.get("next_action", "ask_clarification"),
                final_response=None,
                route_id=route_id,
                tool_outputs=tool_outputs,
                validation=validation,
            )

        self._log(
            "main",
            "HarnessRuntime",
            "run completed successfully",
            {"route": route_id},
            {"status": "ok"},
            "final_report",
            explanation="All required phases executed and validation passed.",
        )
        self._checkpoint(
            phase="final_report",
            run_id=run_id,
            route_id=route_id,
            attempt=self.attempt,
            status="ok",
            payload={"status": "ok", "route_id": route_id},
        )
        return self._final_payload(
            status="ok",
            next_action="report",
            final_response=final_response,
            route_id=route_id,
            tool_outputs=tool_outputs,
            validation=validation,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the generic harness engine demo")
    parser.add_argument("--query", default="", help="User request")
    parser.add_argument("--manifest", default="real_harness_routes.yaml")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM calls")
    parser.add_argument("--no-network", action="store_true", help="Disable network calls")
    parser.add_argument("--max-tool-calls", type=int, default=None, help="Tool call budget override")
    parser.add_argument("--help-section", action="store_true", help="Print the inline help section and exit")
    args = parser.parse_args()

    if args.help_section:
        print(HELP_TEXT.strip())
        return

    engine = RealHarnessEngine(
        manifest_path=args.manifest,
        force_disable_llm=args.no_llm,
        no_network=args.no_network,
        max_tool_calls_override=args.max_tool_calls,
    )
    result = engine.run(args.query)
    print("\n=== HARNESS FINAL REPORT ===")
    payload = json.dumps(result, indent=2, ensure_ascii=False)
    try:
        print(payload)
    except UnicodeEncodeError:
        sys.stdout.buffer.write(payload.encode("utf-8"))
        sys.stdout.buffer.write(b"\n")


if __name__ == "__main__":
    main()
