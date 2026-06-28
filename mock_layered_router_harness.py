#!/usr/bin/env python3
"""
Layered harness router demo.

This script shows a practical route-selection pipeline:
1) Fast heuristics (keyword + phrase + required terms)
2) Semantic matching (character n-gram similarity on route text)
3) Mock LLM fallback when confidence is low / ambiguous

If no route is clear, it routes to:
- ask_clarification (low confidence)
- meta_route (enough signal but unclear intent)

No external tools, no network, no LLM API calls.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import yaml


# ---- Shared helpers ---------------------------------------------------------


def load_manifest(path: str) -> Dict[str, Any]:
    """Load YAML manifest from disk."""
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def now_utc() -> str:
    """Return an ISO-8601 UTC timestamp for trace logging."""
    return datetime.now(timezone.utc).isoformat()


def normalize_text(text: str) -> str:
    """Normalize whitespace and lowercase query text."""
    return re.sub(r"\s+", " ", text.lower().strip())


def tokenize(text: str) -> List[str]:
    """
    Lightweight tokenizer used by both heuristic and semantic steps.
    Keeps numbers and apostrophe words; strips punctuation.
    """
    return re.findall(r"[a-z0-9']+", text.lower())


def build_ngrams(text: str, n: int) -> List[str]:
    """
    Char n-gram builder used for semantic fallback.
    This is a tiny local similarity signal; no embeddings library required.
    """
    cleaned = re.sub(r"\s+", " ", text.lower()).strip()
    if len(cleaned) < n:
        return [cleaned] if cleaned else []
    return [cleaned[i : i + n] for i in range(len(cleaned) - n + 1)]


def cosine_similarity(a: Counter, b: Counter) -> float:
    """
    Cosine similarity between sparse token counters.
    Used instead of full TF-IDF/embedding model for self-contained demo.
    """
    if not a or not b:
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    keys = set(a) | set(b)
    for k in keys:
        av = a.get(k, 0.0)
        bv = b.get(k, 0.0)
        dot += av * bv
        norm_a += av * av
        norm_b += bv * bv
    denom = math.sqrt(norm_a) * math.sqrt(norm_b)
    if denom == 0:
        return 0.0
    return dot / denom


# ---- Data models ------------------------------------------------------------


@dataclass
class RouteCandidate:
    route_id: str
    heuristic_score: float
    semantic_score: float
    combined_score: float
    reasons: Dict[str, Any]
    manifest: str


# ---- Router ----------------------------------------------------------------


class LayeredRouter:
    """
    Route router that combines:
    - explicit heuristic signals
    - semantic similarity fallback
    - mock LLM decision fallback

    All scoring outputs are fully traceable.
    """

    def __init__(self, registry_path: str):
        cfg = load_manifest(registry_path)
        self.routes = cfg["routes"]
        self.router_cfg = cfg["router"]
        self.trace: List[Dict[str, Any]] = []

        # Precompute cheap route "embeddings" (character n-grams) for semantic layer.
        self.route_text_index: Dict[str, List[str]] = {}
        for route in self.routes:
            # Merge every semantic text source used for route retrieval.
            source_text = " ".join(
                [
                    route.get("title", ""),
                    route.get("description", ""),
                    " ".join(route.get("intents", [])),
                    " ".join(route.get("examples", [])),
                    " ".join(route.get("entity_terms", [])),
                    " ".join(route.get("optional_terms", [])),
                ]
            )
            self.route_text_index[route["id"]] = build_ngrams(
                source_text,
                int(self.router_cfg.get("semantic_ngram_n", 3)),
            )

    def _trace(self, phase: str, actor: str, action: str, inp: Any, out: Any) -> None:
        """Record a trace event for explanation and replayability."""
        event = {
            "ts": now_utc(),
            "phase": phase,
            "actor": actor,
            "action": action,
            "input": inp,
            "output": out,
        }
        self.trace.append(event)
        print(f"[{event['ts']}] {phase} | {actor} | {action}")
        print(f"  input : {json.dumps(inp, ensure_ascii=False)}")
        print(f"  output: {json.dumps(out, ensure_ascii=False)}")
        print("-" * 100)

    def _heuristic_score_route(self, route: Dict[str, Any], text: str, tokens: List[str]) -> Tuple[float, Dict[str, Any]]:
        """
        Fast deterministic scoring layer.

        This is the 'cheap' stage:
        - phrase hits
        - intent token hits
        - required bundles (must be co-occuring tokens)
        - penalties on negative words
        """
        score = 0.0
        token_set = set(tokens)
        reasons: Dict[str, Any] = {
            "matched_phrases": [],
            "matched_intents": [],
            "matched_entities": [],
            "matched_required_bundles": [],
            "matched_negatives": [],
        }

        phrase_weight = float(route.get("phrase_weight", 2.0))
        intent_weight = float(route.get("intent_weight", 1.0))
        entity_weight = float(route.get("entity_weight", 0.8))
        optional_weight = float(route.get("optional_weight", 0.4))
        required_weight = float(route.get("required_weight", 1.4))
        req_penalty = float(route.get("penalty_for_missing_required_bundle", 0.8))

        # phrase-level matches are high-precision signals
        for phrase in route.get("phrase_terms", []):
            phrase_l = phrase.lower()
            if phrase_l in text:
                score += phrase_weight
                reasons["matched_phrases"].append(phrase_l)

        # intent/entity token matches are medium-precision
        for term in route.get("intent_terms", []):
            term_l = term.lower()
            if term_l in token_set:
                score += intent_weight
                reasons["matched_intents"].append(term_l)

        for ent in route.get("entity_terms", []):
            ent_l = ent.lower()
            if ent_l in token_set:
                score += entity_weight
                reasons["matched_entities"].append(ent_l)

        # optional terms can support ties and close calls
        for opt in route.get("optional_terms", []):
            if opt.lower() in token_set:
                score += optional_weight

        # required bundles reduce ambiguity because they represent intent shape.
        bundle_hits = 0
        for raw_bundle in route.get("required_bundles", []):
            bundle = [part.lower() for part in raw_bundle]
            if all(part in token_set for part in bundle):
                score += required_weight
                reasons["matched_required_bundles"].append(bundle)
                bundle_hits += 1

        if route.get("required_bundles") and bundle_hits == 0:
            score -= req_penalty

        # discourage routes that clearly conflict
        for neg in route.get("negative_terms", []):
            if neg.lower() in token_set:
                score -= 1.0
                reasons["matched_negatives"].append(neg.lower())

        reasons["bundle_hits"] = bundle_hits
        reasons["bundle_total"] = len(route.get("required_bundles", []))
        return score, reasons

    def _semantic_score_route(self, route_id: str, text: str) -> float:
        """
        "Light semantic" stage without external embeddings.
        Route text signature is compared to query with char n-gram cosine.
        """
        q = Counter(build_ngrams(text, int(self.router_cfg.get("semantic_ngram_n", 3))))
        route_signature = Counter(self.route_text_index[route_id])
        return cosine_similarity(q, route_signature)

    def _mock_llm_rerank(self, query: str, candidates: List[RouteCandidate]) -> Dict[str, Any]:
        """
        Mock LLM fallback. In a real harness this is where you'd call a model:
        - model_prompt: "Given query + candidate routes, pick one route+confidence"
        - schema: route, confidence, reason
        Here we emulate the same with deterministic logic.
        """
        if not candidates:
            return {
                "selected_route": "ask_clarification",
                "confidence": 0.0,
                "reason": "no candidates available",
            }

        # Structured request envelope to make the mechanism explicit.
        prompt = (
            "Classify this user query into one route. "
            "Return JSON: {\"route\", \"confidence\", \"reason\"}. "
            "Allowed routes: " + ", ".join(c.route_id for c in candidates)
        )
        self._trace(
            phase="route_llm_fallback",
            actor="MockLLM",
            action="select route from candidates",
            inp={
                "query": query,
                "prompt": prompt,
                "candidate_scores": {c.route_id: c.combined_score for c in candidates},
            },
            out={},
        )

        # Deterministic mock decision:
        # - boost sports if both team names exist
        q = normalize_text(query)
        if ("rangers" in q and "astros" in q) and "score" in q:
            return {
                "selected_route": "sports_score_lookup",
                "confidence": 0.91,
                "reason": "strong lexical evidence for sports matchup + score",
            }
        if "bug" in q or "error" in q or "traceback" in q:
            return {
                "selected_route": "bug_repair",
                "confidence": 0.88,
                "reason": "explicit defect/failure language",
            }
        if "test" in q:
            return {
                "selected_route": "run_tests",
                "confidence": 0.87,
                "reason": "contains explicit testing directives",
            }

        # Otherwise choose best combined score, but keep reason explicit.
        best = candidates[0]
        return {
            "selected_route": best.route_id,
            "confidence": best.combined_score,
            "reason": "best-scoring candidate under fallback policy",
        }

    def classify(self, query: str) -> Dict[str, Any]:
        """
        Run the 3-layer policy router and return a decision record.
        Returns explicit route, confidence, and trace context used later by engine.
        """
        normalized = normalize_text(query)
        tokens = tokenize(normalized)

        self._trace(
            phase="route_input",
            actor="PolicyRouter",
            action="start route resolution",
            inp={"query": query},
            out={"normalized_query": normalized, "tokens": tokens},
        )

        # Layer 1: heuristic candidate recall.
        raw_scores: List[Tuple[Dict[str, Any], float, Dict[str, Any]]] = []
        for route in self.routes:
            h_score, reasons = self._heuristic_score_route(route, normalized, tokens)
            raw_scores.append((route, h_score, reasons))

        max_h = max((s for _, s, _ in raw_scores), default=0.0) or 1.0
        k = int(self.router_cfg.get("candidate_top_k", 3))
        threshold_ratio = float(self.router_cfg.get("candidate_threshold_ratio", 0.3))
        candidates_raw = [x for x in raw_scores if x[1] >= max_h * threshold_ratio]
        if not candidates_raw:
            candidates_raw = sorted(raw_scores, key=lambda x: x[1], reverse=True)[:k]

        layer1 = [
            {
                "route_id": route["id"],
                "heuristic_score": round(h, 3),
                "base_confidence": round(route.get("confidence_base", 0.0), 3),
            }
            for route, h, _ in candidates_raw
        ]
        self._trace(
            phase="heuristic_layer",
            actor="PolicyRouter",
            action="recall candidates",
            inp={"candidate_threshold_ratio": threshold_ratio, "max_heuristic_score": round(max_h, 3)},
            out=layer1,
        )

        # Layer 2: semantic rerank.
        route_candidates: List[RouteCandidate] = []
        for route, h_score, reasons in candidates_raw:
            s_score = self._semantic_score_route(route["id"], normalized)
            combined = (
                float(self.router_cfg.get("heuristic_weight", 0.6)) * (h_score / max_h)
                + float(self.router_cfg.get("semantic_weight", 0.3)) * s_score
            )
            combined = round(combined, 3)
            route_candidates.append(
                RouteCandidate(
                    route_id=route["id"],
                    heuristic_score=round(h_score / max_h, 3),
                    semantic_score=round(s_score, 3),
                    combined_score=combined,
                    reasons=reasons,
                    manifest=route.get("manifest", ""),
                )
            )

        route_candidates = sorted(route_candidates, key=lambda rc: rc.combined_score, reverse=True)
        top_k = int(self.router_cfg.get("combined_top_k", 3))
        route_candidates = route_candidates[:top_k]
        self._trace(
            phase="semantic_layer",
            actor="PolicyRouter",
            action="semantic rerank",
            inp={
                "layer1_candidates": [x["route_id"] for x in layer1],
            },
            out=[
                {
                    "route_id": rc.route_id,
                    "heuristic_norm": rc.heuristic_score,
                    "semantic_score": rc.semantic_score,
                    "combined": rc.combined_score,
                }
                for rc in route_candidates
            ],
        )

        # Layer 3: confidence decision + fallback branch.
        best = route_candidates[0] if route_candidates else None
        second_score = route_candidates[1].combined_score if len(route_candidates) > 1 else 0.0

        direct_min = float(self.router_cfg.get("direct_confidence_min", 0.58))
        ambiguity_margin = float(self.router_cfg.get("ambiguity_margin", 0.10))
        ask_max = float(self.router_cfg.get("ask_clarification_max", 0.22))
        meta_min = float(self.router_cfg.get("meta_route_min", 0.22))

        decision = {
            "selected_route": None,
            "confidence": 0.0,
            "path": "blocked",
            "reason": "",
            "candidates": [rc.route_id for rc in route_candidates],
            "route_scores": {rc.route_id: rc.combined_score for rc in route_candidates},
            "routing_trace": {},
        }

        if best is None:
            decision["reason"] = "no route candidates"
            return decision

        direct_ok = best.combined_score >= direct_min
        ambiguous = (best.combined_score - second_score) < ambiguity_margin

        # Very low confidence => ask for clarification.
        if best.combined_score < ask_max:
            decision.update(
                {
                    "selected_route": "ask_clarification",
                    "confidence": best.combined_score,
                    "path": "ask_clarification_direct",
                    "reason": "top route score is too weak",
                }
            )
            self._trace(
                phase="decision_layer",
                actor="PolicyRouter",
                action="final decision: ask_clarification",
                inp={"top": best.route_id, "score": best.combined_score, "threshold": ask_max},
                out=decision,
            )
            return decision

        # Ambiguous but enough signal => ask mock LLM to disambiguate or choose meta route.
        if ambiguous and self.router_cfg.get("use_mock_llm_fallback", True):
            llm = self._mock_llm_rerank(query, route_candidates)
            fallback_route = llm["selected_route"]
            fallback_conf = float(llm["confidence"])
            reason = llm["reason"]
            if fallback_conf < meta_min:
                fallback_route = "ask_clarification"
            if fallback_route == "meta_route":
                selected = "meta_route"
            else:
                selected = fallback_route
            decision.update(
                {
                    "selected_route": selected,
                    "confidence": fallback_conf,
                    "path": "llm_fallback",
                    "reason": reason,
                }
            )
            self._trace(
                phase="decision_layer",
                actor="PolicyRouter",
                action="final decision: llm fallback",
                inp={"top_candidates": [rc.route_id for rc in route_candidates]},
                out=decision,
            )
            return decision

        # Direct path when route is clear enough.
        if direct_ok:
            decision.update(
                {
                    "selected_route": best.route_id,
                    "confidence": best.combined_score,
                    "path": "direct",
                    "reason": "confidence and margin are sufficient",
                }
            )
            self._trace(
                phase="decision_layer",
                actor="PolicyRouter",
                action="final decision: direct",
                inp={"top": best.route_id, "score": best.combined_score},
                out=decision,
            )
            return decision

        # Ambiguous but not low enough for direct "ask" => route to meta path.
        if best.combined_score >= meta_min:
            decision.update(
                {
                    "selected_route": "meta_route",
                    "confidence": best.combined_score,
                    "path": "meta_route",
                    "reason": "ambiguous classification around enough signal",
                }
            )
            self._trace(
                phase="decision_layer",
                actor="PolicyRouter",
                action="final decision: meta_route",
                inp={"best": best.route_id, "second": second_score},
                out=decision,
            )
            return decision

        # Catch-all: no clear signal.
        decision.update(
            {
                "selected_route": "ask_clarification",
                "confidence": best.combined_score,
                "path": "ask_clarification_fallback",
                "reason": "no confidence or ambiguity policy match",
            }
        )
        self._trace(
            phase="decision_layer",
            actor="PolicyRouter",
            action="final decision: ask_clarification",
            inp={"best": best.route_id, "second": second_score},
            out=decision,
        )
        return decision


# ---- Engine ----------------------------------------------------------------


class MockHarnessEngine:
    """
    Thin harness runner that consumes route decision and executes:
    - concrete route handlers
    - or fallback handlers (ask_clarification/meta_route)
    """

    def __init__(self, route_registry_path: str, harness_route_path: str):
        self.router = LayeredRouter(route_registry_path)
        self.harness_route_cfg = load_manifest(harness_route_path)
        self.execution_trace: List[Dict[str, Any]] = []

    def _trace(self, phase: str, actor: str, action: str, inp: Any, out: Any) -> None:
        event = {
            "ts": now_utc(),
            "phase": phase,
            "actor": actor,
            "action": action,
            "input": inp,
            "output": out,
        }
        self.execution_trace.append(event)
        print(f"[{event['ts']}] {phase} | {actor} | {action}")
        print(f"  input : {json.dumps(inp, ensure_ascii=False)}")
        print(f"  output: {json.dumps(out, ensure_ascii=False)}")
        print("-" * 100)

    def _simulate_sports_lookup(self, query: str) -> Dict[str, Any]:
        """
        Simulated tool call chain:
        1) lookup_team_matchups
        2) fetch_latest_box_score
        """
        q = normalize_text(query)
        talks_to_index = self.harness_route_cfg["actors"]["tools"]["lookup_team_matchups"]["talks_to"]
        self._trace(
            "execute_route",
            "ToolRouter",
            "tool call: lookup_team_matchups",
            {"talks_to": talks_to_index, "query": q},
            {"status": "invoked"},
        )
        score = "not_found"
        if "rangers" in q and "astros" in q:
            score = "Rangers 3 - Astros 5 (mock latest final)"
        return {"route": "sports_score_lookup", "answer": f"Latest mock result: {score}"}

    def _simulate_bug_repair(self, query: str) -> Dict[str, Any]:
        """
        Simulated code-repair pipeline:
        inspect -> plan -> patch -> validate.
        """
        self._trace(
            "execute_route",
            "Planner",
            "bug_repair: inspect failure",
            {"query": query},
            {"status": "inspected", "failure_category": "mock"},
        )
        self._trace(
            "execute_route",
            "ToolRouter",
            "bug_repair: patch",
            {"tool": "edit_file"},
            {"status": "simulated", "files": ["app.py"]},
        )
        self._trace(
            "execute_route",
            "ToolRouter",
            "bug_repair: validate",
            {"tool": "pytest", "scope": "focused"},
            {"status": "simulated", "result": "pass"},
        )
        return {"route": "bug_repair", "answer": "Simulated fix complete and tests pass (focused path)."}

    def _simulate_run_tests(self, query: str) -> Dict[str, Any]:
        """
        Simulated multi-stage test policy:
        focused first, broader second.
        """
        self._trace(
            "execute_route",
            "ToolRouter",
            "run_tests: focused",
            {"tool": "pytest", "scope": "focused"},
            {"status": "simulated", "result": "pass"},
        )
        self._trace(
            "execute_route",
            "ToolRouter",
            "run_tests: broader",
            {"tool": "pytest", "scope": "broader"},
            {"status": "simulated", "result": "pass"},
        )
        return {"route": "run_tests", "answer": "Simulated focused + broader tests passed."}

    def _simulate_summarize_doc(self, query: str) -> Dict[str, Any]:
        self._trace(
            "execute_route",
            "ToolRouter",
            "summarizer",
            {"tool": "summarizer", "query": normalize_text(query)},
            {"status": "simulated", "summary": "This is a mock one-paragraph summary."},
        )
        return {"route": "summarize_doc", "answer": "Mock summary generated."}

    def _simulate_schedule_email(self, query: str) -> Dict[str, Any]:
        self._trace(
            "execute_route",
            "ToolRouter",
            "schedule_email: compose + route",
            {"tool": "draft_email", "query": normalize_text(query)},
            {"status": "simulated", "draft": "Draft drafted and queued."},
        )
        return {"route": "schedule_email", "answer": "Mock email drafted."}

    def _ask_clarification(self, query: str, reason: str) -> Dict[str, Any]:
        self._trace(
            "execute_route",
            "Planner",
            "ask_clarification",
            {"query": query},
            {"next_action": "ask_user", "reason": reason},
        )
        return {
            "route": "ask_clarification",
            "answer": f"Need clarification: {reason}",
        }

    def _build_meta_plan(self, decision: Dict[str, Any], query: str) -> Dict[str, Any]:
        """
        Simulated meta route proposal:
        an LLM could suggest a temporary plan when no built-in route is clear.
        """
        allowed = [r["id"] for r in self.router.routes if r["id"] != "ask_clarification"]
        prompt = {
            "user_query": query,
            "available_routes": allowed,
            "decision_context": {
                "selected_by_router": decision["selected_route"],
                "confidence": decision["confidence"],
                "path": decision["path"],
            },
        }
        self._trace(
            "meta_route",
            "MockLLM",
            "propose temporary route plan",
            {"prompt": prompt},
            {},
        )

        # For the demo, produce a deterministic temporary route plan.
        plan = {
            "route_id": "meta_temporary_route",
            "goal": "bridge unknown intent by decomposing to known tasks",
            "steps": [
                "extract intent spans",
                "map spans to existing tasks",
                "if split intent -> ask clarification",
                "if single clear intent -> execute mapped route",
            ],
            "target_routes": allowed,
        }
        self._trace(
            "meta_route",
            "Planner",
            "validate temporary route schema",
            {"schema": {"required_fields": ["route_id", "goal", "steps", "target_routes"]}},
            {"valid": True},
        )
        return plan

    def _run_meta_route(self, query: str, decision: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the temporary plan path. In real harnesses this would recurse with
        tighter policies or request human approval before execution.
        """
        plan = self._build_meta_plan(decision, query)
        mapped = "run_tests" if "test" in query.lower() else None
        if mapped:
            self._trace(
                "meta_route",
                "Engine",
                "meta route chooses mapped known route",
                {"mapped_route": mapped},
                {"status": "executing"},
            )
            # For this demo, execute mapped route directly.
            result = self._simulate_run_tests(query)
            return {"route": "meta_route", "answer": f"Meta route mapped to {mapped}: {result['answer']}", "plan": plan}

        self._trace(
            "meta_route",
            "Engine",
            "meta route requires clarifying question",
            {"decision": decision},
            {"next_action": "ask_user"},
        )
        return {
            "route": "meta_route",
            "answer": "Meta route could not safely map; asking user for constraints.",
            "plan": plan,
        }

    def run(self, query: str) -> Dict[str, Any]:
        """
        End-to-end entrypoint for the mock harness.
        Returns a full structured envelope + trace for every phase.
        """
        decision = self.router.classify(query)
        selected = decision["selected_route"] or "ask_clarification"

        self._trace(
            "engine",
            "Engine",
            "route dispatch",
            {"selected_route": selected, "decision": decision},
            {"status": "dispatch"},
        )

        # Route execution switch.
        if selected == "sports_score_lookup":
            output = self._simulate_sports_lookup(query)
        elif selected == "bug_repair":
            output = self._simulate_bug_repair(query)
        elif selected == "run_tests":
            output = self._simulate_run_tests(query)
        elif selected == "summarize_doc":
            output = self._simulate_summarize_doc(query)
        elif selected == "schedule_email":
            output = self._simulate_schedule_email(query)
        elif selected == "meta_route":
            output = self._run_meta_route(query, decision)
        else:
            output = self._ask_clarification(query, decision.get("reason", "no clear route"))

        final = {
            "query": query,
            "decision": decision,
            "route_result": output,
            "router_trace": self.router.trace,
            "engine_trace": self.execution_trace,
        }
        self._trace(
            "engine",
            "Engine",
            "final envelope",
            {"query": query},
            final,
        )
        return final


def print_help_sections() -> None:
    """
    Human-readable help section that explains the full three-layer flow.
    Keep this near main for on-demand learning.
    """
    text = """
Layered policy flow in this mock:

1) POLICY ROUTE SELECTION
   - Input: free-form user query.
   - Layer 1: heuristic scoring from route dictionary terms.
   - Layer 2: local semantic scoring via char n-gram cosine similarity.
   - Layer 3: mock LLM rerank only when ambiguous.

2) FALLBACK PATHS
   - ask_clarification: for very low confidence.
   - meta_route: for medium confidence but ambiguous intents.
   - meta_route returns a temporary plan and validates it before action.

3) ROUTE EXECUTION
   - Known routes run simulated handler functions.
   - Every route call is logged as an explicit actor + phase trace.

Who says what:
   - PolicyRouter: does candidate recall + ranking.
   - MockLLM: only used in fallback phase.
   - Planner: route dispatch and decision logging.
   - ToolRouter: simulated tool calls inside route handlers.
"""
    print(text)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a dry-run harness-style policy router and execution demo.",
    )
    parser.add_argument(
        "--query",
        required=True,
        help="User query to classify and run in the mock harness.",
    )
    parser.add_argument(
        "--route-registry",
        default="mock_route_registry.yaml",
        help="Route registry YAML path.",
    )
    parser.add_argument(
        "--route-manifest",
        default="mock_harness_route.yaml",
        help="Route manifest YAML path used by simulated route handlers.",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Print detailed flow explanation before running.",
    )
    args = parser.parse_args()

    if args.explain:
        print_help_sections()

    engine = MockHarnessEngine(args.route_registry, args.route_manifest)
    result = engine.run(args.query)

    print("\n=== HARNESS FINAL REPORT ===")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

