from __future__ import annotations

from harness.tools_registry import run_tool, validate_args


def test_unknown_tool_returns_unknown_error_code() -> None:
    result = run_tool("ghost_tool", {})
    assert result.success is False
    assert result.error_code == "unknown_tool"
    assert result.error == "Unknown tool: ghost_tool"


def test_validate_args_rejects_unknown_tool() -> None:
    valid, reason = validate_args(name="ghost_tool", args={})
    assert valid is False
    assert reason == "unknown_tool"


def test_calculator_rejects_nonnumeric_expression() -> None:
    result = run_tool("calculator", {"expression": "__import__('os')"})
    assert result.success is False
    assert result.error_code == "tool_expression_invalid"


def test_calculator_evaluates_arithmetic_expression() -> None:
    result = run_tool("calculator", {"expression": "10 / 2 + 5"})
    assert result.success is True
    assert result.output == 10.0
    assert result.error is None
