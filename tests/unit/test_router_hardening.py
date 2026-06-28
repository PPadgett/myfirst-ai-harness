from __future__ import annotations

from harness.router import (
    _extract_evidence_fields,
    _extract_hard_fail_fields,
    classify_route,
)


def test_router_prefers_validator_before_manifests_backward_compat() -> None:
    modern = {
        "validator": {"required_evidence_fields": ["answer"], "hard_fail_errors": ["tool_return_nonzero"]},
        "manifests": {"validator_fields": ["summary"], "hard_fail_errors": ["missing_required_field:summary"]},
    }
    legacy = {
        "manifests": {"validator_fields": ["summary"], "hard_fail_errors": ["missing_required_field:summary"]},
    }
    assert _extract_evidence_fields(modern) == ["answer"]
    assert _extract_hard_fail_fields(modern) == ["tool_return_nonzero"]
    assert _extract_evidence_fields(legacy) == ["summary"]
    assert _extract_hard_fail_fields(legacy) == ["missing_required_field:summary"]


def test_router_manifest_thresholds_are_honored_with_advanced_router() -> None:
    route_manifest = {
        "tool_required": {
            "id": "tool_required",
            "policy": {
                "thresholds": {"ask_confidence": 0.95, "ask_gap": 0.3, "min_confidence": 0.95},
            },
        }
    }
    route = classify_route(
        messages=[{"role": "user", "content": "Please run tests for all modules."}],
        route_manifest=route_manifest,
        feature_level="hardening",
        advanced_router_enabled=True,
    )
    assert route.route.value == "tool_required"
    assert route.next_action == "ask_clarification"


def test_router_applies_route_override_manifest() -> None:
    route_manifest = {
        "direct": {"id": "direct", "policy": {"thresholds": {"min_confidence": 0.85}}},
        "side_effecting_action": {"id": "side_effecting_action", "policy": {"thresholds": {"ask_confidence": 0.50}}},
    }
    route_overrides = {"direct": {"policy": {"thresholds": {"min_confidence": 1.0}}}}
    route = classify_route(
        messages=[{"role": "user", "content": "What is the weather today?"}],
        route_manifest=route_manifest,
        route_overrides=route_overrides,
        feature_level="basic",
        advanced_router_enabled=True,
    )
    assert route.route.value == "direct"
    assert route.confidence >= 1.0
    assert route.next_action == "proceed"
