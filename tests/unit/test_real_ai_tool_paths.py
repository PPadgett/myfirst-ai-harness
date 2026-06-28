from __future__ import annotations

from real_ai_harness import RealHarnessEngine
from unittest import mock


def _make_engine() -> RealHarnessEngine:
    return RealHarnessEngine(
        manifest_path="real_harness_routes.yaml",
        force_disable_llm=True,
        no_network=True,
        max_tool_calls_override=2,
    )


def test_real_engine_unknown_tool_returns_unknown_tool_error() -> None:
    engine = _make_engine()
    result = engine.phase_execute_tools(
        route_id="run_tests",
        plan=[{"name": "not_a_real_tool", "arguments": {"x": 1}}],
    )
    tool_outputs = result["tool_outputs"]
    assert tool_outputs
    assert tool_outputs[0]["error_code"] == "unknown_tool"


def test_real_engine_tool_budget_guard_emits_branch_metadata() -> None:
    engine = _make_engine()
    result = engine.phase_execute_tools(
        route_id="run_tests",
        plan=[
            {"name": "run_tests", "arguments": {}},
            {"name": "new_uuid", "arguments": {}},
            {"name": "calculator", "arguments": {"expression": "1 + 1"}},
        ],
    )
    outputs = result["tool_outputs"]
    assert outputs
    assert any(output.get("tool") == "tool_budget_guard" for output in outputs)
    assert any(output.get("branch") == "tool_budget_guard" for output in outputs)
    assert any(output.get("error_code") == "tool_calls_budget_exceeded" for output in outputs)


def test_real_engine_sandbox_requested_tool_marks_docker_metadata() -> None:
    engine = _make_engine()
    engine.tool_sandbox_mode = "docker"
    with mock.patch("real_ai_harness.shutil.which", return_value=None):
        output = engine._tool_run_sandbox("run_tests", {})
    assert output["tool"] == "run_tests"
    assert output["sandbox"] == "docker"
    assert output["error_code"] == "tool_sandbox_unavailable"


def test_real_engine_sandbox_execution_returns_docker_success_payload() -> None:
    engine = _make_engine()
    engine.tool_sandbox_mode = "docker"

    fake_proc = mock.Mock()
    fake_proc.returncode = 0
    fake_proc.stdout = "ok"
    fake_proc.stderr = ""

    with (
        mock.patch("real_ai_harness.shutil.which", return_value="docker"),
        mock.patch("real_ai_harness.subprocess.run", return_value=fake_proc),
    ):
        output = engine._tool_run_sandbox("run_tests", {})

    assert output["tool"] == "run_tests"
    assert output["sandbox"] == "docker"
    assert output["success"] is True
    assert output["error_code"] is None


def test_real_engine_require_evidence_mode_blocks_validation_on_missing_fields() -> None:
    engine = _make_engine()
    engine.require_evidence = True

    validation = engine.phase_validate(
        route_id="llm_answer",
        final_response="The sky is blue.",
        tool_outputs=[],
    )

    assert validation["ok"] is False
    assert validation["next_action"] == "ask_clarification"
    assert "answer" in validation["missing_fields"]
    assert validation["failed_route_ids"] == ["llm_answer"]
