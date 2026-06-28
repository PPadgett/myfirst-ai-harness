from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
import socket
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Generator
import threading

import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _powershell_command(command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["pwsh", "-NoProfile", "-Command", command],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


SCRIPT_PATH = _repo_root() / "scripts" / "Invoke-HarnessOneShot.ps1"
STATE_SESSIONS_FILE = _repo_root() / "state" / "oneshot-backend-sessions.json"


def _find_free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]
    finally:
        sock.close()


@contextmanager
def _run_mock_model_api(
    *,
    status: int = 500,
    body: dict[str, Any] | None = None,
    models_body: dict[str, Any] | None = None,
    port: int | None = None,
) -> Generator[int, None, None]:
    mock_port = port or _find_free_port()
    payload = json.dumps(body or {}).encode("utf-8")
    models_payload = json.dumps(models_body or {"data": []}).encode("utf-8")

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:  # pragma: no cover - noise suppression
            return

        def do_GET(self) -> None:
            if self.path == "/v1/models":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(models_payload)))
                self.end_headers()
                self.wfile.write(models_payload)
                return
            self.send_response(404)
            self.end_headers()

        def do_POST(self) -> None:
            if self.path != "/v1/chat/completions":
                self.send_response(404)
                self.end_headers()
                return
            _ = self.headers
            _ = self.rfile.read(int(self.headers.get("Content-Length", "0") or 0))
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = HTTPServer(("127.0.0.1", mock_port), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield mock_port
    finally:
        server.shutdown()
        server.server_close()


@contextmanager
def _run_mock_chat_provider(
    *,
    provider_port: int,
    models_body: dict[str, Any],
    chat_response: dict[str, Any],
    chat_status: int = 200,
) -> None:
    payload = json.dumps(chat_response).encode("utf-8")

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:  # pragma: no cover - noise suppression
            return

        def do_GET(self) -> None:
            if self.path == "/v1/models":
                models_payload = json.dumps(models_body).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(models_payload)))
                self.end_headers()
                self.wfile.write(models_payload)
                return
            self.send_response(404)
            self.end_headers()

        def do_POST(self) -> None:
            if self.path != "/v1/chat/completions":
                self.send_response(404)
                self.end_headers()
                return
            _ = self.headers
            _ = self.rfile.read(int(self.headers.get("Content-Length", "0") or 0))
            self.send_response(chat_status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = HTTPServer(("127.0.0.1", provider_port), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield provider_port
    finally:
        server.shutdown()
        server.server_close()


@contextmanager
def _run_mock_runtime_status_api(
    *,
    health_body: dict[str, Any],
    models_body: dict[str, Any],
    port: int | None = None,
    health_status: int = 200,
    models_status: int = 200,
) -> Generator[int, None, None]:
    mock_port = port or _find_free_port()
    health_payload = json.dumps(health_body).encode("utf-8")
    models_payload = json.dumps(models_body).encode("utf-8")

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:  # pragma: no cover - noise suppression
            return

        def do_GET(self) -> None:
            if self.path == "/health":
                self.send_response(health_status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(health_payload)))
                self.end_headers()
                self.wfile.write(health_payload)
                return
            if self.path == "/v1/models":
                self.send_response(models_status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(models_payload)))
                self.end_headers()
                self.wfile.write(models_payload)
                return
            self.send_response(404)
            self.end_headers()

    server = HTTPServer(("127.0.0.1", mock_port), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield mock_port
    finally:
        server.shutdown()
        server.server_close()


@contextmanager
def _preserve_session_file() -> Generator[None, None, None]:
    STATE_SESSIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    original = STATE_SESSIONS_FILE.read_bytes() if STATE_SESSIONS_FILE.exists() else None
    try:
        yield
    finally:
        if original is None:
            STATE_SESSIONS_FILE.unlink(missing_ok=True)
        else:
            STATE_SESSIONS_FILE.write_bytes(original)


@contextmanager
def _run_background_process(*, python_code: str) -> Generator[subprocess.Popen[bytes], None, None]:
    process = subprocess.Popen(
        ["python", "-c", python_code],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        yield process
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass


@pytest.mark.skipif(shutil.which("pwsh") is None, reason="pwsh is required to run PowerShell script tests")
def test_oneshot_dry_run_uses_config_model_and_routable_env_overrides(tmp_path: Path) -> None:
    config = tmp_path / "harness.yaml"
    provider_port = _find_free_port()
    with _run_mock_model_api(models_body={"data": [{"id": "qwen2.5:7b"}]}, port=provider_port) as _:
        config.write_text(
            f"backend:\n  name: openai\n  base_url: \"http://127.0.0.1:{provider_port}/v1\"\n  api_key: null\n  timeout_seconds: 120\nmodel: qwen2.5:7b\n",
            encoding="utf-8",
        )

        command = (
            f". '{SCRIPT_PATH}' ; "
            f"$result = Invoke-HarnessOneShot -Mode runtime -Question 'What are top 3 games?' "
            f"-Config '{config}' -DryRun -FeatureLevel hardening -ToolSandbox docker -EnableAdvancedRouter -RequireEvidence ; "
            "$result | ConvertTo-Json -Depth 20 -Compress"
        )
        completed = _powershell_command(command)

    assert completed.returncode == 0, completed.stderr

    payload = json.loads(completed.stdout.strip())
    assert payload["payload"]["messages"] == [{"role": "user", "content": "What are top 3 games?"}]
    assert "model" not in payload["payload"]
    assert payload["resolved_model"] == "qwen2.5:7b"
    assert payload["runtime_env"]["HARNESS_FEATURE_LEVEL"] == "hardening"
    assert payload["runtime_env"]["HARNESS_TOOL_SANDBOX"] == "docker"
    assert payload["runtime_env"]["HARNESS_REQUIRE_EVIDENCE"] == "1"
    assert payload["runtime_env"]["HARNESS_ENABLE_ADVANCED_ROUTER"] == "1"


@pytest.mark.skipif(shutil.which("pwsh") is None, reason="pwsh is required to run PowerShell script tests")
def test_oneshot_dry_run_rejects_known_bad_override_model(tmp_path: Path) -> None:
    config = tmp_path / "harness.yaml"
    config.write_text("backend:\n  name: openai\nmodel: qwen2.5:7b\n", encoding="utf-8")

    command = (
        f". '{SCRIPT_PATH}' ; "
        f"$result = Invoke-HarnessOneShot -Mode runtime -Question 'What are top 3 games?' "
        f"-Config '{config}' -Model 'hf://meta-llama/Llama-3.1-8B-Instruct' -SkipBackendCheck -DryRun ; "
        "$result | ConvertTo-Json -Depth 20 -Compress"
    )
    completed = _powershell_command(command)
    assert completed.returncode != 0
    assert "hardcoded model override" in completed.stderr or "known hardcoded model" in completed.stdout


@pytest.mark.skipif(shutil.which("pwsh") is None, reason="pwsh is required to run PowerShell script tests")
def test_oneshot_dry_run_explicit_model_override_is_included_in_payload(tmp_path: Path) -> None:
    config = tmp_path / "harness.yaml"
    provider_port = _find_free_port()
    with _run_mock_model_api(models_body={"data": [{"id": "custom-override"}]}, port=provider_port) as _:
        config.write_text(
            f"backend:\n  name: openai\n  base_url: \"http://127.0.0.1:{provider_port}/v1\"\n  api_key: null\n  timeout_seconds: 120\nmodel: qwen2.5:7b\n",
            encoding="utf-8",
        )

        command = (
            f". '{SCRIPT_PATH}' ; "
            f"$result = Invoke-HarnessOneShot -Mode runtime -Question 'What are top 3 games?' "
            f"-Config '{config}' -Model custom-override -DryRun ; "
            "$result | ConvertTo-Json -Depth 20 -Compress"
        )
        completed = _powershell_command(command)

    assert completed.returncode == 0, completed.stderr

    payload = json.loads(completed.stdout.strip())
    assert payload["payload"]["model"] == "custom-override"
    assert payload["resolved_model"] == "custom-override"
    assert payload["explicit_model"] == "custom-override"


@pytest.mark.skipif(shutil.which("pwsh") is None, reason="pwsh is required to run PowerShell script tests")
def test_oneshot_runtime_hard_500_is_reported_as_http_failure(tmp_path: Path) -> None:
    config = tmp_path / "harness.yaml"
    server_port = _find_free_port()
    config.write_text(
        """
backend:
  name: openai
""".strip()
        + "\n",
        encoding="utf-8",
    )

    with _run_mock_model_api(
        models_body={"data": [{"id": "qwen2.5:7b"}]},
        status=500,
        body={"error": {"message": "mock backend failure", "code": "backend_failure"}},
    ) as port:
        config.write_text(
            f"backend:\n  name: openai\n  base_url: \"http://127.0.0.1:{port}/v1\"\n  api_key: null\n  timeout_seconds: 120\nmodel: \"qwen2.5:7b\"\n",
            encoding="utf-8",
        )
        command = (
            f". '{SCRIPT_PATH}' ; "
            f"$ErrorActionPreference='Stop'; "
            f"Invoke-HarnessOneShot -Mode runtime -Question 'What are top 3 games?' "
            f"-Config '{config}' -Port {server_port} -StartupTimeoutSeconds 2 -RequestTimeoutSeconds 2"
            " | ConvertTo-Json -Depth 20 -Compress"
        )
        completed = _powershell_command(command)

    assert completed.returncode == 0, completed.stderr + completed.stdout
    marker_match = re.search(r"(\{[\s\S]*\}\s*)$", completed.stdout)
    assert marker_match is not None, completed.stdout
    response = json.loads(marker_match.group(1))
    assert response["status"] == "validation_block"
    assert response["error_code"] == "model_request_failed"
    assert response["validation"]["ok"] is False


@pytest.mark.skipif(shutil.which("pwsh") is None, reason="pwsh is required to run PowerShell script tests")
def test_oneshot_runtime_preflights_backend_and_fails_quickly_when_unreachable(tmp_path: Path) -> None:
    config = tmp_path / "harness.yaml"
    server_port = _find_free_port()
    config.write_text("backend:\n  name: openai\n", encoding="utf-8")

    unavailable_port = _find_free_port()
    config.write_text(
        f"backend:\n  name: openai\n  base_url: \"http://127.0.0.1:{unavailable_port}/v1\"\n  api_key: null\n  timeout_seconds: 120\nmodel: \"qwen2.5:7b\"\n",
        encoding="utf-8",
    )
    command = (
        f". '{SCRIPT_PATH}' ; "
        f"Invoke-HarnessOneShot -Mode runtime -Question 'What are top 3 games?' "
        f"-Config '{config}' -Port {server_port} -RequestTimeoutSeconds 2 -StartupTimeoutSeconds 2"
    )
    completed = _powershell_command(command)

    assert completed.returncode != 0
    combined = completed.stderr + completed.stdout
    assert "backend is unavailable" in combined


@pytest.mark.skipif(shutil.which("pwsh") is None, reason="pwsh is required to run PowerShell script tests")
def test_oneshot_runtime_preflight_rejects_missing_catalog_model(tmp_path: Path) -> None:
    config = tmp_path / "harness.yaml"
    server_port = _find_free_port()
    config.write_text("backend:\n  name: openai\nmodel: qwen2.5:7b\n", encoding="utf-8")

    with _run_mock_model_api(
        status=200,
        models_body={"data": [{"id": "other-model"}]},
    ) as port:
        config.write_text(
            f"backend:\n  name: openai\n  base_url: \"http://127.0.0.1:{port}/v1\"\n  api_key: null\n  timeout_seconds: 120\nmodel: \"qwen2.5:7b\"\n",
            encoding="utf-8",
        )
        command = (
            f". '{SCRIPT_PATH}' ; "
            f"Invoke-HarnessOneShot -Mode runtime -Question 'What are top 3 games?' "
            f"-Config '{config}' -Port {server_port} -RequestTimeoutSeconds 2 -StartupTimeoutSeconds 2"
        )
        completed = _powershell_command(command)

    assert completed.returncode != 0
    combined = completed.stderr + completed.stdout
    assert "is not available in catalog" in combined or "model 'qwen2.5:7b' is not available in catalog" in combined


@pytest.mark.skipif(shutil.which("pwsh") is None, reason="pwsh is required to run PowerShell script tests")
def test_oneshot_runtime_dry_run_skips_backend_preflight(tmp_path: Path) -> None:
    config = tmp_path / "harness.yaml"
    unavailable_port = _find_free_port()
    config.write_text(
        f"backend:\n  name: openai\n  base_url: \"http://127.0.0.1:{unavailable_port}/v1\"\n  api_key: null\n  timeout_seconds: 120\nmodel: \"qwen2.5:7b\"\n",
        encoding="utf-8",
    )

    command = (
        f". '{SCRIPT_PATH}' ; "
        f"$result = Invoke-HarnessOneShot -Mode runtime -Question 'What is 2+2?' "
        f"-Config '{config}' -SkipBackendCheck -DryRun;"
        "$result | ConvertTo-Json -Depth 20 -Compress"
    )
    completed = _powershell_command(command)

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout.strip())
    assert payload["backend_url"] == f"http://127.0.0.1:{unavailable_port}/v1"
    assert payload["resolved_model"] == "qwen2.5:7b"


@pytest.mark.skipif(shutil.which("pwsh") is None, reason="pwsh is required to run PowerShell script tests")
def test_oneshot_runtime_successful_roundtrip_with_mock_provider(tmp_path: Path) -> None:
    config = tmp_path / "harness.yaml"
    server_port = _find_free_port()
    config.write_text("backend:\n  name: openai\nmodel: qwen2.5:7b\n", encoding="utf-8")

    with _run_mock_chat_provider(
        provider_port=_find_free_port(),
        models_body={"data": [{"id": "qwen2.5:7b"}]},
        chat_response={
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "I can answer that."}}
            ],
            "usage": {"prompt_tokens": 8, "completion_tokens": 4},
        },
        chat_status=200,
    ) as port:
        config.write_text(
            f"backend:\n  name: openai\n  base_url: \"http://127.0.0.1:{port}/v1\"\n  api_key: null\n  timeout_seconds: 120\nmodel: \"qwen2.5:7b\"\n",
            encoding="utf-8",
        )
        command = (
            f". '{SCRIPT_PATH}' ; "
            f"$result = Invoke-HarnessOneShot -Mode runtime -Question 'What is 2+2?' "
            f"-Config '{config}' -Port {server_port} -RequestTimeoutSeconds 2 -StartupTimeoutSeconds 2;"
            "$result | ConvertTo-Json -Depth 20 -Compress"
        )
        completed = _powershell_command(command)

    assert completed.returncode == 0, completed.stderr + completed.stdout
    marker_match = re.search(r"(\{[\s\S]*\}\s*)$", completed.stdout)
    assert marker_match is not None, completed.stdout
    response = json.loads(marker_match.group(1))
    assert response["status"] == "ok"
    assert response["run_id"], response["run_id"]


@pytest.mark.skipif(shutil.which("pwsh") is None, reason="pwsh is required to run PowerShell script tests")
def test_oneshot_runtime_use_existing_server_handles_missing_model_backend_without_preflight(tmp_path: Path) -> None:
    config = tmp_path / "harness.yaml"
    config.write_text("backend:\n  name: openai\nmodel: qwen2.5:7b\n", encoding="utf-8")

    server_port = _find_free_port()
    unavailable_backend_port = _find_free_port()
    config.write_text(
        f"backend:\n  name: openai\n  base_url: \"http://127.0.0.1:{unavailable_backend_port}/v1\"\n  api_key: null\n  timeout_seconds: 120\nmodel: \"qwen2.5:7b\"\n",
        encoding="utf-8",
    )

    command = (
        f". '{SCRIPT_PATH}' ; "
        f"$server = Start-HarnessBackend -ExecutionMode local -Config '{config}' "
        f"-ServerHost '127.0.0.1' -Port {server_port} -WaitSeconds 4 ; "
        "try { "
        f"$result = Invoke-HarnessOneShot -Mode runtime -Question 'What is the most played music video ever?' "
        f"-Config '{config}' -ServerHost '127.0.0.1' -Port {server_port} -UseExistingServer "
        "-StartupTimeoutSeconds 2 -RequestTimeoutSeconds 10; "
        "$result | ConvertTo-Json -Depth 20 -Compress"
        " } finally { "
        f"Stop-HarnessBackend -ExecutionMode local -Config '{config}' -ServerHost '127.0.0.1' -Port {server_port} | Out-Null "
        "}"
    )
    completed = _powershell_command(command)
    assert completed.returncode != 0
    combined = completed.stderr + completed.stdout
    assert "model backend is unavailable" in combined.lower()
    assert "model_backend_unavailable" in combined
    assert "expected_model=qwen2.5:7b" in combined
    assert f"backend_url=http://127.0.0.1:{unavailable_backend_port}/v1/models" in combined


@pytest.mark.skipif(shutil.which("pwsh") is None, reason="pwsh is required to run PowerShell script tests")
def test_oneshot_backend_status_reports_runtime_and_model_backend_health(tmp_path: Path) -> None:
    config = tmp_path / "harness.yaml"
    config.write_text("backend:\n  name: openai\n", encoding="utf-8")
    runtime_port = _find_free_port()
    config.write_text(
        f"backend:\n  name: openai\n  base_url: \"http://127.0.0.1:{runtime_port}/v1\"\n  api_key: null\n  timeout_seconds: 120\nmodel: \"qwen2.5:7b\"\n",
        encoding="utf-8",
    )

    with _run_mock_runtime_status_api(
        health_body={"ok": "true", "backend": "openai", "model": "qwen2.5:7b"},
        models_body={"data": [{"id": "qwen2.5:7b"}, {"id": "other-model"}]},
        port=runtime_port,
    ) as _:
        command = (
            f". '{SCRIPT_PATH}' ; "
            f"$result = Get-HarnessBackendStatus -Config '{config}' -ServerHost '127.0.0.1' -Port {runtime_port} "
            "-RequestTimeoutSeconds 3; "
            "$result | ConvertTo-Json -Depth 20 -Compress"
        )
        completed = _powershell_command(command)

    assert completed.returncode == 0, completed.stderr + completed.stdout
    payload = json.loads(completed.stdout.strip())
    assert payload["server"]["reachable"] is True
    assert payload["server"]["status_code"] == 200
    assert payload["server"]["payload"]["backend"] == "openai"
    assert payload["backend"]["reachable"] is True
    assert payload["backend"]["status_code"] == 200
    assert payload["backend"]["model_present_in_catalog"] is True


@pytest.mark.skipif(shutil.which("pwsh") is None, reason="pwsh is required to run PowerShell script tests")
def test_oneshot_backend_status_prefer_provider_only(tmp_path: Path) -> None:
    config = tmp_path / "harness.yaml"
    runtime_port = _find_free_port()
    backend_port = _find_free_port()
    config.write_text(
        f"backend:\n  name: openai\n  base_url: \"http://127.0.0.1:{backend_port}/v1\"\n  api_key: null\n  timeout_seconds: 120\nmodel: \"qwen2.5:7b\"\n",
        encoding="utf-8",
    )

    with _run_mock_runtime_status_api(
        health_body={"ok": "true", "backend": "openai", "model": "qwen2.5:7b"},
        models_body={"data": [{"id": "qwen2.5:7b"}]},
        port=backend_port,
    ) as _:
        command = (
            f". '{SCRIPT_PATH}' ; "
            f"$result = Get-HarnessBackendStatus -Config '{config}' -ServerHost '127.0.0.1' -Port {runtime_port} -PreferProviderOnly "
            "-RequestTimeoutSeconds 3; "
            "$result | ConvertTo-Json -Depth 20 -Compress"
        )
        completed = _powershell_command(command)

    assert completed.returncode == 0, completed.stderr + completed.stdout
    payload = json.loads(completed.stdout.strip())
    assert payload["server"]["reachable"] is False
    assert "Runtime health probe skipped" in payload["server"]["error"]
    assert payload["provider_plane"]["reachable"] is True
    assert payload["provider_plane"]["status_code"] == 200
    assert payload["provider_plane"]["model_present_in_catalog"] is True


@pytest.mark.skipif(shutil.which("pwsh") is None, reason="pwsh is required to run PowerShell script tests")
def test_oneshot_backend_status_with_missing_runtime_does_not_throw(tmp_path: Path) -> None:
    config = tmp_path / "harness.yaml"
    runtime_port = _find_free_port()
    backend_port = _find_free_port()
    config.write_text(
        f"backend:\n  name: openai\n  base_url: \"http://127.0.0.1:{backend_port}/v1\"\n  api_key: null\n  timeout_seconds: 120\nmodel: \"qwen2.5:7b\"\n",
        encoding="utf-8",
    )

    command = (
        f". '{SCRIPT_PATH}' ; "
        f"$result = Get-HarnessBackendStatus -Config '{config}' -ServerHost '127.0.0.1' -Port {runtime_port} -RequestTimeoutSeconds 2; "
        "$result | ConvertTo-Json -Depth 20 -Compress"
    )
    completed = _powershell_command(command)
    combined = completed.stdout + completed.stderr

    assert completed.returncode == 0, combined
    assert "PropertyNotFoundException" not in combined
    payload = json.loads(completed.stdout.strip())
    assert payload["server"]["reachable"] is False
    assert payload["server"]["error"] != ""
    assert payload["backend"]["reachable"] is False
    assert payload["backend"]["error"] != ""


@pytest.mark.skipif(shutil.which("pwsh") is None, reason="pwsh is required to run PowerShell script tests")
def test_oneshot_backend_status_reports_catalog_mismatch(tmp_path: Path) -> None:
    config = tmp_path / "harness.yaml"
    runtime_port = _find_free_port()
    config.write_text("backend:\n  name: openai\n", encoding="utf-8")
    config.write_text(
        f"backend:\n  name: openai\n  base_url: \"http://127.0.0.1:{runtime_port}/v1\"\n  api_key: null\n  timeout_seconds: 120\nmodel: \"qwen2.5:7b\"\n",
        encoding="utf-8",
    )

    with _run_mock_runtime_status_api(
        health_body={"ok": "true", "backend": "openai", "model": "qwen2.5:7b"},
        models_body={"data": [{"id": "different-model"}]},
        port=runtime_port,
    ) as _:
        command = (
            f". '{SCRIPT_PATH}' ; "
            f"$result = Get-HarnessBackendStatus -Config '{config}' -ServerHost '127.0.0.1' -Port {runtime_port} "
            "-RequestTimeoutSeconds 3; "
            "$result | ConvertTo-Json -Depth 20 -Compress"
        )
        completed = _powershell_command(command)

    assert completed.returncode == 0, completed.stderr + completed.stdout
    payload = json.loads(completed.stdout.strip())
    assert payload["backend"]["reachable"] is True
    assert payload["backend"]["model_present_in_catalog"] is False


@pytest.mark.skipif(shutil.which("pwsh") is None, reason="pwsh is required to run PowerShell script tests")
def test_oneshot_backend_status_with_mock_runtime_and_unreachable_backend_catalog(tmp_path: Path) -> None:
    runtime_port = _find_free_port()
    unreachable_backend_port = _find_free_port()

    config = tmp_path / "harness.yaml"
    config.write_text(
        f"backend:\n  name: openai\n  base_url: \"http://127.0.0.1:{unreachable_backend_port}/v1\"\n  api_key: null\n  timeout_seconds: 120\nmodel: \"qwen2.5:7b\"\n",
        encoding="utf-8",
    )

    with _run_mock_runtime_status_api(
        health_body={"ok": "true", "backend": "openai", "model": "qwen2.5:7b"},
        models_body={"data": []},
        port=runtime_port,
    ) as _:
        command = (
            f". '{SCRIPT_PATH}' ; "
            f"$result = Get-HarnessBackendStatus -Config '{config}' -ServerHost '127.0.0.1' -Port {runtime_port} "
            "-RequestTimeoutSeconds 3; "
            "$result | ConvertTo-Json -Depth 20 -Compress"
        )
        completed = _powershell_command(command)

    combined = completed.stdout + completed.stderr
    assert completed.returncode == 0, combined
    assert "PropertyNotFoundException" not in combined
    payload = json.loads(completed.stdout.strip())
    assert payload["server"]["reachable"] is True
    assert payload["backend"]["reachable"] is False
    assert payload["backend"]["status_code"] in (None, 0, "") or payload["backend"]["status_code"] != 200


@pytest.mark.skipif(shutil.which("pwsh") is None, reason="pwsh is required to run PowerShell script tests")
def test_oneshot_backend_start_local_dry_run(tmp_path: Path) -> None:
    config = tmp_path / "harness.yaml"
    config.write_text(
        """
backend:
  name: openai
model: qwen2.5:7b
""".strip()
        + "\n",
        encoding="utf-8",
    )

    command = (
        f". '{SCRIPT_PATH}' ; "
        f"$result = Start-HarnessBackend -ExecutionMode local -Config '{config}' "
        "-ServerHost '127.0.0.1' -Port 9099 -DryRun; "
        "$result | ConvertTo-Json -Depth 20 -Compress"
    )
    completed = _powershell_command(command)
    assert completed.returncode == 0, completed.stderr

    payload = json.loads(completed.stdout.strip())
    assert payload["mode"] == "local"
    assert payload["started"] is False
    assert payload["host"] == "127.0.0.1"
    assert payload["port"] == 9099


@pytest.mark.skipif(shutil.which("pwsh") is None, reason="pwsh is required to run PowerShell script tests")
def test_oneshot_backend_start_local_reuses_alive_session_without_spawning_second_process(tmp_path: Path) -> None:
    config = tmp_path / "harness.yaml"
    config.write_text(
        """
backend:
  name: openai
model: qwen2.5:7b
""".strip()
        + "\n",
        encoding="utf-8",
    )
    host = "127.0.0.1"
    port = _find_free_port()

    with _run_background_process(
        python_code="import time; time.sleep(9999)"
    ) as process:
        assert process.pid is not None
        with _preserve_session_file():
            STATE_SESSIONS_FILE.write_text(
                json.dumps(
                    [
                        {
                            "key": f"local|{str(config.resolve())}|{host}|{port}",
                            "mode": "local",
                            "process_id": process.pid,
                            "config": str(config.resolve()),
                            "host": host,
                            "port": port,
                            "command": f"python -c",
                            "health_url": f"http://{host}:{port}/health",
                            "started_utc": "2026-01-01T00:00:00Z",
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            command = (
                f". '{SCRIPT_PATH}' ; "
                f"$result = Start-HarnessBackend -ExecutionMode local -Config '{config.resolve()}' "
                f"-ServerHost '{host}' -Port {port}; "
                "$result | ConvertTo-Json -Depth 20 -Compress"
            )
            completed = _powershell_command(command)
            assert completed.returncode == 0, completed.stderr

    payload = json.loads(completed.stdout.strip())
    assert payload["mode"] == "local"
    assert payload["started"] is False
    assert payload["action"] == "already_running"
    assert payload["process_id"] == process.pid


@pytest.mark.skipif(shutil.which("pwsh") is None, reason="pwsh is required to run PowerShell script tests")
def test_oneshot_backend_start_local_reports_external_port_conflict(tmp_path: Path) -> None:
    config = tmp_path / "harness.yaml"
    config.write_text(
        """
backend:
  name: openai
model: qwen2.5:7b
""".strip()
        + "\n",
        encoding="utf-8",
    )
    listener_port = _find_free_port()

    bind_code = (
        "import socket, time; "
        f"s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); "
        f"s.bind(('127.0.0.1',{listener_port})); s.listen(1); time.sleep(9999)"
    )
    with _run_background_process(python_code=bind_code) as process:
        command = (
            f". '{SCRIPT_PATH}' ; "
            f"Start-HarnessBackend -ExecutionMode local -Config '{config}' -ServerHost '127.0.0.1' -Port {listener_port}"
        )
        completed = _powershell_command(command)

    assert completed.returncode != 0
    combined = completed.stderr + completed.stdout
    assert "Cannot bind to port" in combined
    assert f"process {process.pid}" in combined


@pytest.mark.skipif(shutil.which("pwsh") is None, reason="pwsh is required to run PowerShell script tests")
def test_oneshot_backend_stop_local_cleans_stale_session_entries(tmp_path: Path) -> None:
    config = tmp_path / "harness.yaml"
    config.write_text(
        """
backend:
  name: openai
model: qwen2.5:7b
""".strip()
        + "\n",
        encoding="utf-8",
    )
    port = _find_free_port()

    dead_process = subprocess.Popen(
        ["python", "-c", "import time; time.sleep(0.1)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    dead_process.wait(timeout=2)

    with _preserve_session_file():
        STATE_SESSIONS_FILE.write_text(
            json.dumps(
                [
                    {
                        "key": f"local|{str(config.resolve())}|127.0.0.1|{port}",
                        "mode": "local",
                        "process_id": dead_process.pid,
                        "config": str(config.resolve()),
                        "host": "127.0.0.1",
                        "port": port,
                        "command": "python -c",
                        "health_url": "http://127.0.0.1:9099/health",
                        "started_utc": "2026-01-01T00:00:00Z",
                    }
                ],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        command = (
            f". '{SCRIPT_PATH}' ; "
            f"$result = Stop-HarnessBackend -ExecutionMode local -Config '{config}' -ServerHost '127.0.0.1' -Port {port}; "
            "$result | ConvertTo-Json -Depth 20 -Compress"
        )
        completed = _powershell_command(command)

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout.strip())
    assert payload["mode"] == "local"
    assert payload["action"] == "stopped"
    assert payload["removed_count"] == 1
    state_content = STATE_SESSIONS_FILE.read_text(encoding="utf-8").strip()
    assert state_content == "[]"
    assert json.loads(state_content) == []


@pytest.mark.skipif(shutil.which("pwsh") is None, reason="pwsh is required to run PowerShell script tests")
def test_oneshot_backend_start_container_inferred_profile_requires_env_file(tmp_path: Path) -> None:
    config = tmp_path / "harness.yaml"
    config.write_text(
        """
backend:
  name: nvidia_nim
  model: hf://meta-llama/Llama-3.1-8B-Instruct
""".strip()
        + "\n",
        encoding="utf-8",
    )
    env_file = tmp_path / ".env.nvidia"
    env_file.write_text("NIM_LLM_IMAGE=test\n", encoding="utf-8")

    command = (
        f". '{SCRIPT_PATH}' ; "
        f"$result = Start-HarnessBackend -ExecutionMode containerized -Config '{config}' "
        f"-EnvFile '{env_file}' -DryRun; "
        "$result | ConvertTo-Json -Depth 20 -Compress"
    )
    completed = _powershell_command(command)
    assert completed.returncode == 0, completed.stderr

    payload = json.loads(completed.stdout.strip())
    assert payload["mode"] == "containerized"
    assert payload["started"] is False
    assert payload["profile"] == "nvidia"


@pytest.mark.skipif(shutil.which("pwsh") is None, reason="pwsh is required to run PowerShell script tests")
def test_oneshot_backend_stop_local_dry_run(tmp_path: Path) -> None:
    config = tmp_path / "harness.yaml"
    config.write_text(
        """
backend:
  name: openai
model: qwen2.5:7b
""".strip()
        + "\n",
        encoding="utf-8",
    )

    command = (
        f". '{SCRIPT_PATH}' ; "
        f"$result = Stop-HarnessBackend -ExecutionMode local -Config '{config}' -DryRun; "
        "$result | ConvertTo-Json -Depth 20 -Compress"
    )
    completed = _powershell_command(command)
    assert completed.returncode == 0, completed.stderr

    payload = json.loads(completed.stdout.strip())
    assert payload["mode"] == "local"
    assert payload["action"] == "stopped"


@pytest.mark.skipif(shutil.which("pwsh") is None, reason="pwsh is required to run PowerShell script tests")
def test_oneshot_backend_start_container_mode_requires_profile_or_inference_for_openai_error(tmp_path: Path) -> None:
    config = tmp_path / "harness.yaml"
    config.write_text("backend:\n  name: openai\nmodel: qwen2.5:7b\n", encoding="utf-8")

    command = (
        f". '{SCRIPT_PATH}' ; "
        f"Start-HarnessBackend -ExecutionMode containerized -Config '{config}' -DryRun"
    )
    completed = _powershell_command(command)

    assert completed.returncode != 0
    assert "Cannot infer container profile from backend 'openai'" in completed.stderr
