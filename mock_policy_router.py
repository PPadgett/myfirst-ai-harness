#!/usr/bin/env python3
"""
Hybrid route/router demo for a harness-like policy selection layer.

This is a dry-run model of how a real harness chooses one policy route:
1) deterministic preprocessing
2) keyword/regex recall stage (fast)
3) vector-style similarity stage (fallback/re-rank)
4) ambiguity/confidence check
5) optional fallback to a mock LLM reranker

No external network calls are made.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import yaml


def read_yaml(path: str) -> Dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


STOP_WORDS = {
    "a",
    "an",
    "the",
    "is",
    "it",
    "for",
    "to",
    "of",
    "and",
    "or",
    "in",
    "on",
    "at",
    "by",
    "with",
    "from",
    "into",
    "that",
    "this",
    "as",
    "be",
    "are",
    "i",
    "you",
    "we",
    "they",
    "my",
    "me",
    "our",
    "your",
    "their",
    "about",
    "what",
    "when",
    "where",
    "who",
    "whom",
    "how",
    "why",
    "will",
    "would",
    "should",
    "could",
    "can",
    "not",
    "have",
    "has",
    "had",
    "will",
    "do",
    "did",
    "does",
    "so",
}


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    return re.sub(r"\s+", " ", text)


def tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-z0-9']+", text.lower())
    return [t for t in tokens if t and t not in STOP_WORDS]


def build_char_ngrams(text: str, n: int = 3) -> List[str]:
    s = re.sub(r"\s+", " ", text.lower()).strip()
    if len(s) < n:
        return [s] if s else []
    return [s[i : i + n] for i in range(len(s) - n + 1)]


def jaccard(set_a: Sequence[str], set_b: Sequence[str]) -> float:
    a = set(set_a)
    b = set(set_b)
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    if union == 0:
        return 0.0
    return inter / union


@dataclass
class RouteTrace:
    route_id: str
    keyword_score: float
    vector_score: float
    combined_score: float
    reasons: Dict[str, Any]
    manifest: str | None = None


class MockPolicyRouter:
    """
    Demonstrates common harness routing layers:
    - candidate recall via keywords
    - lightweight semantic matching via ngram-Jaccard
    - ambiguity/confidence policy
    """

    def __init__(self, registry_path: str = "mock_route_registry.yaml"):
        registry = read_yaml(registry_path)
        self.routes: List[Dict[str, Any]] = registry["routes"]
        self.router_cfg = registry["router"]
        self.trace: List[Dict[str, Any]] = []
        self._route_profiles: Dict[str, Dict[str, Any]] = {}
        self._build_route_profiles()

    def _build_route_profiles(self) -> None:
        n = self.router_cfg.get("vector_ngram_n", 3)
        for route in self.routes:
            bag = {
                "description": route.get("description", ""),
                "intents": " ".join(route.get("intents", [])),
                "examples": " ".join(route.get("examples", [])),
                "intent_terms": " ".join(route.get("intent_terms", [])),
                "entity_terms": " ".join(route.get("entity_terms", [])),
                "optional_terms": " ".join(route.get("optional_terms", [])),
            }
            merged = " ".join(v for v in bag.values())
            self._route_profiles[route["id"]] = {
                "ngram": build_char_ngrams(merged, n=n),
                "required_bundle_count": len(route.get("required_bundles", [])),
            }

    def _score_keyword_signals(self, route: Dict[str, Any], text: str, tokens: List[str]) -> Tuple[float, Dict[str, Any]]:
        token_set = set(tokens)
        reasons: Dict[str, Any] = {
            "matched_phrases": [],
            "matched_tokens": [],
            "matched_entities": [],
            "required_bundle_hits": [],
            "negatives": [],
        }
        score = 0.0

        phrase_weight = float(route.get("phrase_weight", 1.0))
        term_weight = float(route.get("term_weight", 1.0))
        optional_weight = float(route.get("optional_weight", 0.5))
        required_weight = float(route.get("required_weight", 1.0))

        # Exact phrase matches (best precision for routing)
        for phrase in route.get("phrase_terms", []):
            phrase_norm = phrase.lower()
            if phrase_norm in text:
                score += phrase_weight
                reasons["matched_phrases"].append(phrase_norm)

        # Token-level matches
        for term in route.get("intent_terms", []):
            if term in token_set:
                score += term_weight
                reasons["matched_tokens"].append(term)

        for ent in route.get("entity_terms", []):
            if ent in token_set:
                score += 0.8
                reasons["matched_entities"].append(ent)

        for opt in route.get("optional_terms", []):
            if opt in token_set:
                score += optional_weight

        # Required term bundles indicate route intent clarity
        bundle_hits = 0
        missing_required = False
        bundles = route.get("required_bundles", [])
        for raw_bundle in bundles:
            bundle = [b.lower() for b in raw_bundle]
            if all(part in token_set for part in bundle):
                score += required_weight
                reasons["required_bundle_hits"].append(bundle)
                bundle_hits += 1
        if bundles and bundle_hits == 0:
            missing_required = True
            score -= float(route.get("penalty_for_missing_required_bundle", 1.0))

        # Negatives reduce confidence when conflicting intent words are present
        for neg in route.get("negative_terms", []):
            if neg in token_set:
                score -= 1.2
                reasons["negatives"].append(neg)

        reasons["missing_required_bundle"] = missing_required
        reasons["bundle_hits"] = bundle_hits
        reasons["bundle_total"] = len(bundles)
        return score, reasons

    def _vector_similarity(self, query_text: str, route_id: str) -> float:
        n = self.router_cfg.get("vector_ngram_n", 3)
        query_signature = build_char_ngrams(query_text, n=n)
        route_profile = self._route_profiles[route_id]
        route_signature = route_profile["ngram"]
        return jaccard(query_signature, route_signature)

    def _mock_llm_rerank(self, query: str, candidates: List[RouteTrace]) -> Dict[str, Any]:
        # In a real harness this is where a small or full LLM would be invoked.
        # We keep it deterministic here and return what the harness policy layer would do.
        prompt = (
            "You are a strict policy router. Return one route and confidence among: "
            + ", ".join(route.route_id for route in candidates)
            + ". If ambiguous, return reason."
        )
        self.trace.append(
            {
                "phase": "route_llm_fallback",
                "actor": "MockLLM",
                "action": "policy classification rerank",
                "input": {
                    "query": query,
                    "prompt": prompt,
                    "candidate_routes": [route.route_id for route in candidates],
                    "candidate_scores": [route.combined_score for route in candidates],
                },
                "output": {
                    "selected_route": candidates[0].route_id if candidates else None,
                    "confidence": candidates[0].combined_score if candidates else 0.0,
                    "reason": "deterministic mock fallback used",
                },
            }
        )

        if not candidates:
            return {"selected_route": None, "confidence": 0.0, "reason": "no candidates"}
        return {
            "selected_route": candidates[0].route_id,
            "confidence": candidates[0].combined_score,
            "reason": "rerank from mock model on ambiguity",
        }

    def route(self, user_query: str) -> Dict[str, Any]:
        text = normalize_text(user_query)
        tokens = tokenize(text)

        # ----------------------------
        # Stage 1: keyword/regex recall
        # ----------------------------
        keyword_scores: List[Tuple[Dict[str, Any], float, Dict[str, Any]]] = []
        for route in self.routes:
            raw, reasons = self._score_keyword_signals(route, text, tokens)
            keyword_scores.append((route, raw, reasons))

        max_keyword = max(score for _, score, _ in keyword_scores)
        if max_keyword <= 0:
            max_keyword = 1.0

        recall_ratio = float(self.router_cfg.get("candidate_threshold_ratio", 0.25))
        candidate_cap = int(self.router_cfg.get("candidate_top_k", 3))
        candidates_raw = [
            (route, score, reasons)
            for route, score, reasons in keyword_scores
            if score >= max_keyword * recall_ratio
        ]
        if not candidates_raw:
            candidates_raw = sorted(keyword_scores, key=lambda x: x[1], reverse=True)[:candidate_cap]

        # ----------------------------
        # Stage 2: vector-like similarity re-rank
        # ----------------------------
        route_candidates: List[RouteTrace] = []
        for route, kw_score, reasons in candidates_raw:
            vector_score = self._vector_similarity(text, route["id"])
            kw_norm = kw_score / max_keyword
            combined = 0.7 * kw_norm + 0.3 * vector_score
            route_candidates.append(
                RouteTrace(
                    route_id=route["id"],
                    keyword_score=round(float(kw_norm), 3),
                    vector_score=round(float(vector_score), 3),
                    combined_score=round(float(combined), 3),
                    reasons=reasons,
                    manifest=route.get("manifest"),
                )
            )

        route_candidates = sorted(route_candidates, key=lambda it: it.combined_score, reverse=True)
        route_candidates = route_candidates[: self.router_cfg.get("combined_top_k", 3)]

        # ----------------------------
        # Stage 3: final policy decision + fallback conditions
        # ----------------------------
        selected = route_candidates[0] if route_candidates else None
        direct_confidence_min = float(self.router_cfg.get("direct_confidence_min", 0.58))
        ambiguity_margin = float(self.router_cfg.get("ambiguity_margin", 0.12))
        second_score = route_candidates[1].combined_score if len(route_candidates) > 1 else 0.0

        if not selected:
            decision = "blocked"
            final_route = None
            confidence = 0.0
            fallback_needed = True
            fallback_reason = "no candidates produced"
        else:
            margin = selected.combined_score - second_score
            fallback_needed = (
                selected.combined_score < direct_confidence_min
                or margin < ambiguity_margin
            )
            if fallback_needed and self.router_cfg.get("use_mock_llm_on_ambiguous", True):
                rerank = self._mock_llm_rerank(text, route_candidates)
                final_route = rerank["selected_route"]
                confidence = rerank["confidence"]
                decision = "mock_llm_fallback"
                fallback_reason = rerank["reason"]
            else:
                final_route = selected.route_id
                confidence = selected.combined_score
                decision = "direct"
                fallback_reason = None

        sorted_trace = [
            {
                "route": r.route_id,
                "keyword_score": r.keyword_score,
                "vector_score": r.vector_score,
                "combined_score": r.combined_score,
                "manifest": r.manifest,
            }
            for r in route_candidates
        ]

        result = {
            "query": user_query,
            "normalized_query": text,
            "tokens": tokens,
            "decision": decision,
            "selected_route": final_route,
            "confidence": confidence,
            "fallback_needed": fallback_needed,
            "fallback_reason": fallback_reason,
            "candidate_ranking": sorted_trace,
            "route_signals": {
                "selected_reasons": selected.reasons if selected else {},
                "top2_margin": round(selected.combined_score - second_score, 3) if selected else 0.0,
                "direct_confidence_min": direct_confidence_min,
                "ambiguity_margin": ambiguity_margin,
            },
            "trace": self.trace,
        }
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run mock policy router.")
    parser.add_argument("--query", required=True, help="User query to classify")
    parser.add_argument(
        "--route-registry",
        default="mock_route_registry.yaml",
        help="Path to route registry YAML",
    )
    args = parser.parse_args()

    router = MockPolicyRouter(args.route_registry)
    decision = router.route(args.query)
    print(json.dumps(decision, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

