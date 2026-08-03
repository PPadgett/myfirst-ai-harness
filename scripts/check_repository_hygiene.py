"""Reject non-reviewable or generated files from the Git index."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


MAX_FILE_BYTES = 1_048_576
ALLOWED_EXTENSIONS = {
    ".csv",
    ".example",
    ".json",
    ".md",
    ".ps1",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
ALLOWED_FILENAMES = {
    ".dockerignore",
    ".editorconfig",
    ".gitattributes",
    ".gitignore",
    "Dockerfile",
}
FORBIDDEN_PARTS = {
    ".cache",
    ".models",
    ".venv",
    "__pycache__",
    "build",
    "checkpoints",
    "dist",
    "models",
    "node_modules",
    "state",
    "traces",
    "weights",
}


def tracked_paths(repo_root: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    return [repo_root / item.decode("utf-8") for item in result.stdout.split(b"\0") if item]


def validate_file(repo_root: Path, path: Path) -> list[str]:
    relative = path.relative_to(repo_root)
    errors: list[str] = []

    if any(part.lower() in FORBIDDEN_PARTS for part in relative.parts[:-1]):
        errors.append(f"generated or protected path is tracked: {relative}")
    if relative.parts and relative.parts[0].lower() == "benchmarks":
        errors.append(f"generated benchmark output is tracked: {relative}")

    if path.name not in ALLOWED_FILENAMES and path.suffix.lower() not in ALLOWED_EXTENSIONS:
        errors.append(f"file type is not allowlisted: {relative}")

    if not path.is_file():
        errors.append(f"tracked path is not a regular file: {relative}")
        return errors

    size = path.stat().st_size
    if size > MAX_FILE_BYTES:
        errors.append(f"file exceeds {MAX_FILE_BYTES} bytes: {relative} ({size} bytes)")

    data = path.read_bytes()
    if b"\0" in data:
        errors.append(f"NUL byte indicates binary content: {relative}")
    try:
        data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        errors.append(f"file is not UTF-8 text: {relative} ({exc})")

    return errors


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    errors = [
        error
        for path in tracked_paths(repo_root)
        for error in validate_file(repo_root, path)
    ]
    if errors:
        print("Repository hygiene check failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print("Repository hygiene check passed: tracked files are reviewable UTF-8 text.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
