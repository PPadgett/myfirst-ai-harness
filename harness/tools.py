from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from dateutil import tz
from typing import Any
import ast
import operator
import uuid


_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
}


@dataclass
class ToolSpec:
    name: str
    description: str
    json_schema: dict[str, Any]


@dataclass
class ToolCallResult:
    name: str
    args: dict[str, Any]
    output: Any
    success: bool
    error: str | None = None


def _safe_eval_expr(expr: str) -> Any:
    tree = ast.parse(expr, mode="eval")

    def _eval(node: ast.AST) -> Any:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("Only numeric constants are supported.")
        if isinstance(node, ast.BinOp):
            op = _OPS.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported operator {type(node.op).__name__}")
            return op(_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            op = _OPS.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported unary operator {type(node.op).__name__}")
            return op(_eval(node.operand))
        raise ValueError(f"Unsupported expression {type(node).__name__}")

    return _eval(tree.body)


def list_tool_specs() -> list[ToolSpec]:
    return [
        ToolSpec(
            name="calculator",
            description="Evaluate a single arithmetic expression. Use only for deterministic math.",
            json_schema={
                "name": "calculator",
                "description": "Evaluate arithmetic expression",
                "parameters": {
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                    "required": ["expression"],
                },
            },
        ),
        ToolSpec(
            name="time_now",
            description="Current UTC date/time and timezone-aware local conversion.",
            json_schema={
                "name": "time_now",
                "description": "Return current timestamp",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tz": {"type": "string"},
                    },
                    "required": [],
                },
            },
        ),
        ToolSpec(
            name="new_uuid",
            description="Generate a new UUID4.",
            json_schema={
                "name": "new_uuid",
                "description": "Generate random uuid",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        ),
    ]


def execute_tool(name: str, args: dict[str, Any]) -> ToolCallResult:
    if name == "calculator":
        expr = str(args.get("expression", "")).strip()
        if not expr:
            return ToolCallResult(name=name, args=args, output=None, success=False, error="Missing expression")
        try:
            value = _safe_eval_expr(expr)
        except Exception as exc:
            return ToolCallResult(name=name, args=args, output=None, success=False, error=str(exc))
        return ToolCallResult(name=name, args=args, output=value, success=True)
    if name == "time_now":
        tz_name = str(args.get("tz", "")).strip()
        zone = tz.gettz(tz_name) if tz_name else None
        now = datetime.now(timezone.utc)
        if zone:
            now = now.astimezone(zone)
        return ToolCallResult(name=name, args=args, output=now.isoformat(), success=True)
    if name == "new_uuid":
        return ToolCallResult(name=name, args=args, output=str(uuid.uuid4()), success=True)
    return ToolCallResult(name=name, args=args, output=None, success=False, error="Unknown tool")


def specs_as_openai_tools(only: tuple[str, ...] | list[str] | None = None) -> list[dict[str, Any]]:
    specs = list_tool_specs()
    if only:
        normalized = set(only)
        specs = [s for s in specs if s.name in normalized]
    return [
        {
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.json_schema["parameters"],
            },
        }
        for spec in specs
    ]

