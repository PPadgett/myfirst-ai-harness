"""Manifest schema helpers for route validation and policy-driven runtime controls."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError
import yaml


class RouteValidatorSpec(BaseModel):
    required_evidence_fields: list[str] = Field(default_factory=list)
    hard_fail_errors: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="allow")


class RouteManifest(BaseModel):
    id: str
    title: str | None = None
    version: str | None = None
    validator: RouteValidatorSpec | None = None
    manifests: dict[str, Any] | None = None
    policy: dict[str, Any] = Field(default_factory=dict)
    required_slots: dict[str, Any] = Field(default_factory=dict)
    prompts: dict[str, Any] = Field(default_factory=dict)
    route_metadata: dict[str, Any] = Field(default_factory=dict)
    allowlist: list[str] | None = None
    tools: list[str] | None = None
    is_universal_fallback: bool = False

    model_config = ConfigDict(extra="allow")

    def normalized_validator(self) -> RouteValidatorSpec:
        if self.validator is not None:
            return self.validator
        if isinstance(self.manifests, dict):
            fields = self.manifests.get("validator_fields")
            hard_fail = self.manifests.get("hard_fail_errors", [])
            if isinstance(fields, list):
                return RouteValidatorSpec(
                    required_evidence_fields=[str(x) for x in fields],
                    hard_fail_errors=[str(x) for x in hard_fail] if isinstance(hard_fail, list) else [],
                )
        return RouteValidatorSpec()


class ManifestFile(BaseModel):
    version: str | None = None
    routes: list[RouteManifest] = Field(default_factory=list)
    model_config = ConfigDict(extra="allow")


def _read_yaml(path: str | Path) -> dict[str, Any]:
    raw = Path(path).read_text(encoding="utf-8")
    loaded = yaml.safe_load(raw)
    if loaded is None:
        return {}
    if isinstance(loaded, dict):
        return loaded
    raise TypeError(f"Manifest '{path}' must be a YAML mapping.")


def load_route_manifest(path: str | None, *, strict: bool = False) -> dict[str, RouteManifest]:
    """Load route manifests with permissive fallback by default."""
    if not path:
        return {}
    payload = _read_yaml(path)
    routes = payload.get("routes") if isinstance(payload, dict) else None
    if not isinstance(routes, list):
        if strict:
            raise ValueError(f"Manifest '{path}' missing routes list.")
        return {}

    manifest: dict[str, RouteManifest] = {}
    errors: list[str] = []
    for item in routes:
        if not isinstance(item, dict):
            if strict:
                raise ValueError("Each route entry must be a mapping.")
            errors.append("route_not_mapping")
            continue
        if not item.get("id"):
            if strict:
                raise ValueError("Every route entry must include id.")
            errors.append("route_missing_id")
            continue
        try:
            model = RouteManifest.model_validate(item)
            manifest[item["id"]] = model
        except ValidationError as exc:
            if strict:
                raise
            errors.append(str(exc))
    if not strict and errors:
        # Keep permissive behavior: best-effort load with a permissive projection.
        for item in routes:
            if not isinstance(item, dict):
                continue
            rid = str(item.get("id", "")).strip()
            if not rid or rid in manifest:
                continue
            fallback = RouteManifest(id=rid, policy={"_validation_error": "; ".join(errors[-1:])})
            manifest[rid] = fallback
    return manifest

