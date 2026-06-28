"""Tool execution primitives for local tool calling."""

from __future__ import annotations

from typing import Any

from harness.tools_registry import ToolCallResult, ToolSpec, list_tool_specs as _list_tool_specs
from harness.tools_registry import run_tool
from harness.tools_registry import normalize_args as _normalize_args
from harness.tools_registry import validate_args as _validate_args
from harness.tools_registry import specs_as_openai_tools as _specs_as_openai_tools


def list_tool_specs() -> list[ToolSpec]:
    return _list_tool_specs()


def execute_tool(name: str, args: dict[str, Any]) -> ToolCallResult:
    normalized = _normalize_args(args)
    return run_tool(name=name, args=normalized)


def validate_tool_args(name: str, args: dict[str, Any]) -> tuple[bool, str | None]:
    return _validate_args(name=name, args=_normalize_args(args))


def specs_as_openai_tools(only: tuple[str, ...] | list[str] | None = None) -> list[dict[str, Any]]:
    return _specs_as_openai_tools(only=only)
