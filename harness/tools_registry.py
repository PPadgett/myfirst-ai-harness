"""Shared tool registry and execution with strict argument validation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
import ast
import json
import operator
import subprocess
import uuid

from dateutil import tz as dateutil_tz


@dataclass
class ToolSpec:
    name: str
    description: str
    json_schema: dict[str, Any]
    sandbox_required: bool = False


@dataclass
class ToolCallResult:
    name: str
    args: dict[str, Any]
    output: Any
    success: bool
    error: str | None = None
    error_code: str | None = None
    sandbox: str | None = None
    started_at: str = ""
    finished_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "success": self.success,
            "output": self.output,
            "arguments": self.args,
            "error": self.error,
            "error_code": self.error_code,
            "sandbox": self.sandbox,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


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


def _normalize_output(payload: Any) -> Any:
    if isinstance(payload, float) and (payload != payload):
        return "nan"
    if isinstance(payload, (tuple, set)):
        return list(payload)
    if isinstance(payload, (dict, list, str, int, float, bool)) or payload is None:
        return payload
    return str(payload)


def list_tool_specs() -> list[ToolSpec]:
    return [
        ToolSpec(
            name="calculator",
            description="Evaluate a single arithmetic expression.",
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
            description="Current timestamp in ISO-8601 format.",
            json_schema={
                "name": "time_now",
                "description": "Return current timestamp",
                "parameters": {"type": "object", "properties": {"tz": {"type": "string"}}, "required": []},
            },
        ),
        ToolSpec(
            name="new_uuid",
            description="Generate a random UUID4.",
            json_schema={
                "name": "new_uuid",
                "description": "Generate random uuid",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        ),
        ToolSpec(
            name="run_tests",
            description="Execute pytest in current repo path.",
            sandbox_required=True,
            json_schema={
                "name": "run_tests",
                "description": "Run pytest command.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "scope": {"type": ["string", "null"]},
                    },
                    "required": [],
                },
            },
        ),
    ]


_SPECS_BY_NAME = {s.name: s for s in list_tool_specs()}


def specs_as_openai_tools(only: tuple[str, ...] | list[str] | None = None) -> list[dict[str, Any]]:
    specs = list_tool_specs()
    if only:
        allowed = set(only)
        specs = [spec for spec in specs if spec.name in allowed]
    fallback_names = []
    if only:
        allowed = set(only)
        fallback_names = sorted(name for name in allowed if name not in {spec.name for spec in specs})
        if fallback_names:
            specs.extend(
                ToolSpec(
                    name=name,
                    description=f"Unknown tool passthrough: {name}",
                    json_schema={
                        "name": name,
                        "description": f"Unknown tool passthrough: {name}",
                        "parameters": {"type": "object", "properties": {}, "required": []},
                    },
                )
                for name in fallback_names
            )
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


def _validate_args(name: str, args: dict[str, Any]) -> tuple[bool, str | None]:
    spec = _SPECS_BY_NAME.get(name)
    if spec is None:
        return False, "unknown_tool"
    schema = spec.json_schema.get("parameters", {})
    required = schema.get("required", [])
    props = schema.get("properties", {})

    for key in required:
        if key not in args:
            return False, f"missing_required_argument:{key}"
    for key, value in args.items():
        prop = props.get(key)
        if not isinstance(prop, dict):
            continue
        expected = prop.get("type")
        if isinstance(expected, list):
            if "null" in expected and value is None:
                continue
            if "string" in expected and isinstance(value, str):
                continue
            if "number" in expected and isinstance(value, (int, float)):
                continue
            if "boolean" in expected and isinstance(value, bool):
                continue
            if "array" in expected and isinstance(value, list):
                continue
            if "object" in expected and isinstance(value, dict):
                continue
            return False, f"type_mismatch:{key}"
        if expected == "string" and not isinstance(value, str):
            return False, f"type_mismatch:{key}"
        if expected == "number" and not isinstance(value, (int, float)):
            return False, f"type_mismatch:{key}"
        if expected == "boolean" and not isinstance(value, bool):
            return False, f"type_mismatch:{key}"
        if expected == "array" and not isinstance(value, list):
            return False, f"type_mismatch:{key}"
        if expected == "object" and not isinstance(value, dict):
            return False, f"type_mismatch:{key}"
    return True, None


def _stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_tool(name: str, args: dict[str, Any]) -> ToolCallResult:
    payload = dict(args) if isinstance(args, dict) else {}
    started = _stamp()
    valid, reason = _validate_args(name, payload)
    if not valid:
        if reason == "unknown_tool":
            error = f"Unknown tool: {name}"
            error_code = "unknown_tool"
        else:
            error = reason
            error_code = reason
        return ToolCallResult(
            name=name,
            args=payload,
            output=None,
            success=False,
            error=error,
            error_code=error_code,
            sandbox="local",
            started_at=started,
            finished_at=_stamp(),
        )

    try:
        if name == "calculator":
            expr = str(payload.get("expression", "")).strip()
            if not expr:
                return ToolCallResult(
                    name=name,
                    args=payload,
                    output=None,
                    success=False,
                    error="Missing expression",
                    error_code="missing_required_argument:expression",
                    started_at=started,
                    finished_at=_stamp(),
                )
            try:
                value = _safe_eval_expr(expr)
            except ValueError as exc:
                return ToolCallResult(
                    name=name,
                    args=payload,
                    output=None,
                    success=False,
                    error=f"tool_expression_invalid:{exc}",
                    error_code="tool_expression_invalid",
                    sandbox="local",
                    started_at=started,
                    finished_at=_stamp(),
                )
            return ToolCallResult(
                name=name,
                args=payload,
                output=_normalize_output(value),
                success=True,
                sandbox="local",
                started_at=started,
                finished_at=_stamp(),
            )
        if name == "time_now":
            tz_name = str(payload.get("tz", "")).strip()
            zone = dateutil_tz.gettz(tz_name) if tz_name else None
            now = datetime.now(timezone.utc)
            if zone:
                now = now.astimezone(zone)
            return ToolCallResult(
                name=name,
                args=payload,
                output=_normalize_output(now.isoformat()),
                success=True,
                sandbox="local",
                started_at=started,
                finished_at=_stamp(),
            )
        if name == "new_uuid":
            return ToolCallResult(
                name=name,
                args=payload,
                output=_normalize_output(str(uuid.uuid4())),
                success=True,
                sandbox="local",
                started_at=started,
                finished_at=_stamp(),
            )
        if name == "run_tests":
            scope = payload.get("scope")
            command = ["pytest"]
            if scope:
                command.append(str(scope))
            proc = subprocess.run(command, cwd=".", capture_output=True, text=True, timeout=600)
            return ToolCallResult(
                name=name,
                args=payload,
                output={
                    "returncode": proc.returncode,
                    "stdout": proc.stdout.strip(),
                    "stderr": proc.stderr.strip(),
                },
                success=proc.returncode == 0,
                error=None if proc.returncode == 0 else "tool_returned_non_zero",
                error_code="tool_return_nonzero" if proc.returncode != 0 else None,
                sandbox="local",
                started_at=started,
                finished_at=_stamp(),
            )
    except Exception as exc:
        return ToolCallResult(
            name=name,
            args=payload,
            output=None,
            success=False,
            error=str(exc),
            error_code=f"tool_execution_exception:{type(exc).__name__}",
            sandbox="local",
            started_at=started,
            finished_at=_stamp(),
        )

    return ToolCallResult(
        name=name,
        args=payload,
        output=None,
        success=False,
        error=f"Unknown tool: {name}",
        error_code="unknown_tool",
        sandbox="local",
        started_at=started,
        finished_at=_stamp(),
    )


def validate_args(name: str, args: dict[str, Any]) -> tuple[bool, str | None]:
    return _validate_args(name, args)


def sandbox_required(name: str) -> bool:
    spec = _SPECS_BY_NAME.get(name)
    if spec is None:
        return False
    return bool(spec.sandbox_required)


def normalize_args(args: Any) -> dict[str, Any]:
    return dict(args) if isinstance(args, dict) else {}


def run_tool_safe(name: str, args: Any) -> ToolCallResult:
    normalized = args if isinstance(args, dict) else {}
    return run_tool(name=name, args=normalized)


def registry_names() -> list[str]:
    return [spec.name for spec in list_tool_specs()]
