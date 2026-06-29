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
MODEL_STATE_SESSIONS_FILE = _repo_root() / "state" / "oneshot-model-backend-sessions.json"


def _find_free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]
    finally:
        sock.close()


def _write_fake_ollama_store(root: Path, *, model: str = "qwen3", tag: str = "4b") -> Path:
    manifest = root / "manifests" / "registry.ollama.ai" / "library" / model / tag
    blob = root / "blobs" / "sha256-deadbeef"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    blob.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "layers": [
                    {
                        "mediaType": "application/vnd.ollama.image.model",
                        "digest": "sha256:deadbeef",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    blob.write_bytes(b"GGUFfake")
    return blob


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
def _preserve_model_session_file() -> Generator[None, None, None]:
    MODEL_STATE_SESSIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    original = MODEL_STATE_SESSIONS_FILE.read_bytes() if MODEL_STATE_SESSIONS_FILE.exists() else None
    try:
        yield
    finally:
        if original is None:
            MODEL_STATE_SESSIONS_FILE.unlink(missing_ok=True)
        else:
            MODEL_STATE_SESSIONS_FILE.write_bytes(original)


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

    payload = json.loads(completed.stdout.strip().splitlines()[-1])
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

    payload = json.loads(completed.stdout.strip().splitlines()[-1])
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
    assert "error_code=model_backend_unavailable" in combined
    assert f"backend_url=http://127.0.0.1:{unavailable_port}/v1/models" in combined


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
    assert "error_code=model_not_available" in combined or "error_code=model_backend_unavailable" in combined
    assert "expected_model=qwen2.5:7b" in combined


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
def test_oneshot_output_options_select_expand_and_json(tmp_path: Path) -> None:
    config = tmp_path / "harness.yaml"
    unavailable_port = _find_free_port()
    config.write_text(
        f"backend:\n  name: openai\n  base_url: \"http://127.0.0.1:{unavailable_port}/v1\"\n  api_key: null\n  timeout_seconds: 120\nmodel: \"qwen2.5:7b\"\n",
        encoding="utf-8",
    )

    command = (
        f". '{SCRIPT_PATH}' ; "
        f"$selected = Invoke-HarnessOneShot -Mode runtime -Question 'What is 2+2?' "
        f"-Config '{config}' -SkipBackendCheck -DryRun -Property resolved_model,mode; "
        f"$aliasJson = Invoke-HarnessOneShot -Mode runtime -Question 'What is 2+2?' "
        f"-Config '{config}' -SkipBackendCheck -DryRun -Properties resolved_model,backend_name -AsJson; "
        f"$expanded = Invoke-HarnessOneShot -Mode runtime -Question 'What is 2+2?' "
        f"-Config '{config}' -SkipBackendCheck -DryRun -ExpandProperty resolved_model; "
        "[PSCustomObject]@{selected=$selected; alias=($aliasJson | ConvertFrom-Json); expanded=$expanded} "
        "| ConvertTo-Json -Depth 20 -Compress"
    )
    completed = _powershell_command(command)

    assert completed.returncode == 0, completed.stderr + completed.stdout
    payload = json.loads(completed.stdout.strip())
    assert payload["selected"]["resolved_model"] == "qwen2.5:7b"
    assert payload["selected"]["mode"] == "runtime"
    assert payload["alias"]["resolved_model"] == "qwen2.5:7b"
    assert payload["alias"]["backend_name"] == "openai"
    assert payload["expanded"] == "qwen2.5:7b"


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
    assert response["answer"] == "I can answer that."


@pytest.mark.skipif(shutil.which("pwsh") is None, reason="pwsh is required to run PowerShell script tests")
def test_oneshot_runtime_answer_only_with_mock_provider(tmp_path: Path) -> None:
    config = tmp_path / "harness.yaml"
    server_port = _find_free_port()

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
            f"Invoke-HarnessOneShot -Mode runtime -Question 'What is 2+2?' "
            f"-Config '{config}' -Port {server_port} -RequestTimeoutSeconds 2 -StartupTimeoutSeconds 2 -AnswerOnly"
        )
        completed = _powershell_command(command)

    assert completed.returncode == 0, completed.stderr + completed.stdout
    assert completed.stdout.strip().endswith("I can answer that.")


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
    assert "error_code=model_backend_unavailable" in combined
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
    assert payload["backend_name"] == "openai"
    assert payload["selected_model"] == "qwen2.5:7b"
    assert payload["runtime_reachable"] is True
    assert payload["runtime_status_code"] == 200
    assert payload["provider_reachable"] is True
    assert payload["provider_status_code"] == 200
    assert payload["model_present_in_catalog"] is True
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
def test_oneshot_backend_start_own_llm_alias_maps_to_model_backend(tmp_path: Path) -> None:
    port = _find_free_port()

    command = (
        f". '{SCRIPT_PATH}' ; "
        f"$result = Start-HarnessOwnLLMBackend -ModelBackendHost '127.0.0.1' -ModelBackendPort {port} -Model 'missing-local-foundation:v1' -AllowFallback -DryRun; "
        "$result | ConvertTo-Json -Depth 20 -Compress"
    )
    completed = _powershell_command(command)

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout.strip())
    assert payload["mode"] == "model_backend"
    assert payload["started"] is False
    assert payload["model"] == "missing-local-foundation:v1"
    assert payload["python_path"].endswith("python.exe") or payload["python_path"] == "python"
    assert f"-m harness.local_model_provider --host 127.0.0.1 --port {port} --model missing-local-foundation:v1" in payload["command"]
    assert "--backend auto --device cpu --max-new-tokens 256" in payload["command"]
    assert "--llama-cpp-n-ctx 4096 --llama-cpp-n-gpu-layers 0 --llama-cpp-n-threads 4" in payload["command"]
    assert "--allow-fallback" in payload["command"]
    assert payload["configured_backend"] == "auto"
    assert payload["generation_backend"] == "fallback"
    assert payload["model_source_type"] is None
    assert payload["model_artifact_format"] is None
    assert payload["fallback_active"] is True
    assert payload["allow_fallback"] is True
    assert payload["provider_warning"]


@pytest.mark.skipif(shutil.which("pwsh") is None, reason="pwsh is required to run PowerShell script tests")
def test_oneshot_backend_start_whatif_reports_python_path_and_logs(tmp_path: Path) -> None:
    port = _find_free_port()
    fake_python = tmp_path / "custom-python.exe"

    command = (
        f". '{SCRIPT_PATH}' ; "
        f"$result = Start-HarnessOwnLLMBackend -ModelBackendHost '127.0.0.1' -ModelBackendPort {port} "
        f"-Model 'missing-local-foundation:v1' -AllowFallback -PythonPath '{fake_python}' -WhatIf 6>$null; "
        "$result | ConvertTo-Json -Depth 20 -Compress"
    )
    completed = _powershell_command(command)

    assert completed.returncode == 0, completed.stderr + completed.stdout
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload["mode"] == "model_backend"
    assert payload["started"] is False
    assert payload["action"] == "whatif"
    assert payload["python_path"] == str(fake_python)
    assert payload["command"].startswith(str(fake_python))
    assert Path(payload["stdout_log"]).parent.name == "logs"
    assert Path(payload["stderr_log"]).parent.name == "logs"
    assert payload["model_load_attempted"] is False
    assert payload["model_load_succeeded"] is False


@pytest.mark.skipif(shutil.which("pwsh") is None, reason="pwsh is required to run PowerShell script tests")
def test_oneshot_backend_start_auto_without_source_requires_explicit_fallback_for_plain_name(tmp_path: Path) -> None:
    port = _find_free_port()

    command = (
        f". '{SCRIPT_PATH}' ; "
        f"Start-HarnessOwnLLMBackend -ModelBackendHost '127.0.0.1' -ModelBackendPort {port} -Model 'not-a-local-source' -Verbose -Debug -DryRun"
    )
    completed = _powershell_command(command)

    assert completed.returncode != 0
    combined = completed.stderr + completed.stdout
    assert "VERBOSE: Start-HarnessModelBackend resolving startup" in combined
    assert "DEBUG: Model source detection" in combined
    assert "No process was started" in combined
    assert "not-a-local-source" in combined
    assert "-Backend auto requires a real local model source" in combined
    assert "-ModelPath" in combined
    assert "-ModelsRoot" in combined
    assert "-Backend fallback or -AllowFallback" in combined


@pytest.mark.skipif(shutil.which("pwsh") is None, reason="pwsh is required to run PowerShell script tests")
def test_oneshot_backend_start_tag_model_uses_harness_owned_local_store_resolution(tmp_path: Path) -> None:
    port = _find_free_port()
    models_root = tmp_path / "ollama-models"
    blob = _write_fake_ollama_store(models_root)

    command = (
        f"$env:OLLAMA_MODELS = '{models_root}' ; . '{SCRIPT_PATH}' ; "
        f"$result = Start-HarnessOwnLLMBackend -ModelBackendHost '127.0.0.1' -ModelBackendPort {port} -Model 'qwen3:4b' -DryRun; "
        "$result | ConvertTo-Json -Depth 20 -Compress"
    )
    completed = _powershell_command(command)

    assert completed.returncode == 0, completed.stderr + completed.stdout
    payload = json.loads(completed.stdout.strip())
    assert payload["mode"] == "model_backend"
    assert payload["started"] is False
    assert payload["model"] == "qwen3:4b"
    assert payload["configured_backend"] == "auto"
    assert payload["generation_backend"] == "llamacpp"
    assert payload["model_source"] == str(blob)
    assert payload["model_source_type"] == "ollama_store"
    assert payload["model_artifact_format"] == "gguf"
    assert payload["provider_store"] == "ollama"
    assert payload["fallback_active"] is False
    assert "--backend auto" in payload["command"]
    assert "ollama serve" not in payload["command"].lower()


@pytest.mark.skipif(shutil.which("pwsh") is None, reason="pwsh is required to run PowerShell script tests")
def test_oneshot_backend_start_explicit_gguf_uses_llamacpp(tmp_path: Path) -> None:
    port = _find_free_port()
    gguf = tmp_path / "tiny.gguf"
    gguf.write_bytes(b"GGUFfake")

    command = (
        f". '{SCRIPT_PATH}' ; "
        f"$result = Start-HarnessOwnLLMBackend -ModelBackendHost '127.0.0.1' -ModelBackendPort {port} "
        f"-Model 'tiny:v1' -ModelPath '{gguf}' -LlamaCppContext 8192 -LlamaCppGpuLayers 1 -LlamaCppThreads 2 -DryRun; "
        "$result | ConvertTo-Json -Depth 20 -Compress"
    )
    completed = _powershell_command(command)

    assert completed.returncode == 0, completed.stderr + completed.stdout
    payload = json.loads(completed.stdout.strip())
    assert payload["generation_backend"] == "llamacpp"
    assert payload["model_source"] == str(gguf)
    assert payload["model_source_type"] == "filesystem"
    assert payload["model_artifact_format"] == "gguf"
    assert "--model-path" in payload["command"]
    assert "--llama-cpp-n-ctx 8192 --llama-cpp-n-gpu-layers 1 --llama-cpp-n-threads 2" in payload["command"]


@pytest.mark.skipif(shutil.which("pwsh") is None, reason="pwsh is required to run PowerShell script tests")
def test_oneshot_backend_start_hf_model_id_uses_local_transformers_cache(tmp_path: Path) -> None:
    port = _find_free_port()

    command = (
        f". '{SCRIPT_PATH}' ; "
        f"$result = Start-HarnessOwnLLMBackend -ModelBackendHost '127.0.0.1' -ModelBackendPort {port} "
        "-Model 'hf://meta-llama/Llama-3.1-8B-Instruct' -DryRun; "
        "$result | ConvertTo-Json -Depth 20 -Compress"
    )
    completed = _powershell_command(command)

    assert completed.returncode == 0, completed.stderr + completed.stdout
    payload = json.loads(completed.stdout.strip())
    assert payload["configured_backend"] == "auto"
    assert payload["generation_backend"] == "transformers"
    assert payload["model_source"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert payload["model_source_type"] == "huggingface_cache"
    assert payload["model_artifact_format"] == "transformers"
    assert payload["provider_store"] == "huggingface"
    assert payload["fallback_active"] is False
    assert "nvidia" not in payload["command"].lower()


@pytest.mark.skipif(shutil.which("pwsh") is None, reason="pwsh is required to run PowerShell script tests")
def test_oneshot_backend_status_own_llm_alias(tmp_path: Path) -> None:
    port = _find_free_port()
    command = (
        f". '{SCRIPT_PATH}' ; "
        f"$result = Get-HarnessOwnLLMBackendStatus -ModelBackendHost '127.0.0.1' -ModelBackendPort {port} -RequestTimeoutSeconds 1; "
        "$result | ConvertTo-Json -Depth 20 -Compress"
    )
    completed = _powershell_command(command)

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout.strip())
    assert payload["host"] == "127.0.0.1"
    assert payload["port"] == port
    assert payload["health"]["reachable"] is False


@pytest.mark.skipif(shutil.which("pwsh") is None, reason="pwsh is required to run PowerShell script tests")
def test_oneshot_model_backend_status_reports_flat_and_nested_fields(tmp_path: Path) -> None:
    port = _find_free_port()

    with _preserve_model_session_file():
        MODEL_STATE_SESSIONS_FILE.write_text(
            json.dumps(
                [
                    {
                        "key": f"model-backend|127.0.0.1|{port}",
                        "mode": "model_backend",
                        "host": "127.0.0.1",
                        "port": port,
                        "process_id": None,
                        "model": "local-foundation:v1",
                        "python_path": "C:/repo/.venv/Scripts/python.exe",
                        "stdout_log": "C:/repo/state/logs/model.stdout.log",
                        "stderr_log": "C:/repo/state/logs/model.stderr.log",
                    }
                ],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        with _run_mock_runtime_status_api(
            health_body={
            "ok": "true",
            "model": "local-foundation:v1",
            "configured_backend": "auto",
            "generation_backend": "fallback",
            "fallback_active": True,
            "local_model_loaded": False,
            "model_source_type": "ollama_store",
            "model_artifact_format": "gguf",
            "provider_store": "ollama",
            "manifest_path": "C:/Users/test/.ollama/models/manifests/registry.ollama.ai/library/local-foundation/v1",
            "runtime_dependency": "llama_cpp",
            "runtime_dependency_available": False,
            "model_source_present": True,
            "model_load_attempted": True,
            "model_load_succeeded": False,
            "last_load_error": "load failed",
            "last_generation_error": "generation failed",
            "template_applied": True,
            "finish_reason": "length",
            "truncated": True,
            "reasoning_extracted": True,
            "provider_warning": "diagnostic fallback",
        },
        models_body={
            "data": [
                {
                    "id": "local-foundation:v1",
                    "configured_backend": "auto",
                    "generation_backend": "fallback",
                    "fallback_active": True,
                    "local_model_loaded": False,
                    "model_source_type": "ollama_store",
                    "model_artifact_format": "gguf",
                    "provider_store": "ollama",
                    "manifest_path": "C:/Users/test/.ollama/models/manifests/registry.ollama.ai/library/local-foundation/v1",
                    "runtime_dependency": "llama_cpp",
                    "runtime_dependency_available": False,
                    "model_source_present": True,
                    "model_load_attempted": True,
                    "model_load_succeeded": False,
                    "last_load_error": "load failed",
                    "last_generation_error": "generation failed",
                    "template_applied": True,
                    "finish_reason": "length",
                    "truncated": True,
                    "reasoning_extracted": True,
                    "provider_warning": "diagnostic fallback",
                }
            ]
        },
            port=port,
        ) as _:
            command = (
                f". '{SCRIPT_PATH}' ; "
                f"Get-HarnessOwnLLMBackendStatus -ModelBackendHost '127.0.0.1' -ModelBackendPort {port} "
                "-RequestTimeoutSeconds 2 -IncludeSession -AsJson"
            )
            completed = _powershell_command(command)

    assert completed.returncode == 0, completed.stderr + completed.stdout
    payload = json.loads(completed.stdout.strip())
    assert payload["host"] == "127.0.0.1"
    assert payload["port"] == port
    assert payload["model"] == "local-foundation:v1"
    assert payload["health_reachable"] is True
    assert payload["health_status_code"] == 200
    assert payload["models_reachable"] is True
    assert payload["models_status_code"] == 200
    assert payload["model_catalog_present"] is True
    assert payload["configured_backend"] == "auto"
    assert payload["generation_backend"] == "fallback"
    assert payload["fallback_active"] is True
    assert payload["local_model_loaded"] is False
    assert payload["model_source_type"] == "ollama_store"
    assert payload["model_artifact_format"] == "gguf"
    assert payload["provider_store"] == "ollama"
    assert payload["manifest_path"].endswith("/local-foundation/v1")
    assert payload["runtime_dependency"] == "llama_cpp"
    assert payload["runtime_dependency_available"] is False
    assert payload["model_source_present"] is True
    assert payload["model_load_attempted"] is True
    assert payload["model_load_succeeded"] is False
    assert payload["last_load_error"] == "load failed"
    assert payload["last_generation_error"] == "generation failed"
    assert payload["template_applied"] is True
    assert payload["finish_reason"] == "length"
    assert payload["truncated"] is True
    assert payload["reasoning_extracted"] is True
    assert payload["provider_warning"] == "diagnostic fallback"
    assert payload["python_path"].endswith("python.exe")
    assert payload["stdout_log"].endswith("model.stdout.log")
    assert payload["stderr_log"].endswith("model.stderr.log")
    assert payload["session"][0]["stdout_log"].endswith("model.stdout.log")
    assert payload["health"]["payload"]["ok"] == "true"
    assert payload["models"]["models"] == ["local-foundation:v1"]
    assert payload["models"]["model_entry"]["generation_backend"] == "fallback"
    assert payload["models"]["model_entry"]["provider_store"] == "ollama"


@pytest.mark.skipif(shutil.which("pwsh") is None, reason="pwsh is required to run PowerShell script tests")
def test_oneshot_backend_stop_own_llm_alias_dry_run(tmp_path: Path) -> None:
    command = (
        f". '{SCRIPT_PATH}' ; "
        "$result = Stop-HarnessOwnLLMBackend -DryRun; "
        "$result | ConvertTo-Json -Depth 20 -Compress"
    )
    completed = _powershell_command(command)

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout.strip())
    assert payload["mode"] == "model_backend"
    assert payload["action"] == "stopped"


@pytest.mark.skipif(shutil.which("pwsh") is None, reason="pwsh is required to run PowerShell script tests")
def test_oneshot_stack_stop_dry_run_targets_runtime_and_model_backend(tmp_path: Path) -> None:
    config = tmp_path / "harness.yaml"
    config.write_text("backend:\n  name: openai\nmodel: qwen2.5:7b\n", encoding="utf-8")
    host = "127.0.0.1"
    runtime_port = _find_free_port()
    model_port = _find_free_port()

    with _run_background_process(python_code="import time; time.sleep(9999)") as process:
        with _preserve_session_file(), _preserve_model_session_file():
            STATE_SESSIONS_FILE.write_text(
                json.dumps(
                    [
                        {
                            "key": f"local|{str(config.resolve())}|{host}|{runtime_port}",
                            "mode": "local",
                            "process_id": process.pid,
                            "config": str(config.resolve()),
                            "host": host,
                            "port": runtime_port,
                            "command": "python -m harness.server",
                            "health_url": f"http://{host}:{runtime_port}/health",
                            "started_utc": "2026-01-01T00:00:00Z",
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            MODEL_STATE_SESSIONS_FILE.write_text(
                json.dumps(
                    [
                        {
                            "key": f"model-backend|{host}|{model_port}",
                            "mode": "model_backend",
                            "process_id": process.pid,
                            "host": host,
                            "port": model_port,
                            "model": "local-foundation:v1",
                            "command": "python -m harness.local_model_provider",
                            "health_url": f"http://{host}:{model_port}/health",
                            "started_utc": "2026-01-01T00:00:00Z",
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            command = (
                f". '{SCRIPT_PATH}' ; "
                f"Stop-HarnessStack -Config '{config}' -ServerHost '{host}' -Port {runtime_port} "
                f"-ModelBackendHost '{host}' -ModelBackendPort {model_port} -DryRun -AsJson"
            )
            completed = _powershell_command(command)

    assert completed.returncode == 0, completed.stderr + completed.stdout
    payload = json.loads(completed.stdout.strip())
    assert payload["action"] == "stopped"
    assert payload["ok"] is True
    assert payload["dry_run"] is True
    assert payload["runtime_removed_count"] == 1
    assert payload["model_removed_count"] == 1
    assert payload["runtime_backend"]["removed_count"] == 1
    assert payload["model_backend"]["removed_count"] == 1


@pytest.mark.skipif(shutil.which("pwsh") is None, reason="pwsh is required to run PowerShell script tests")
def test_oneshot_stack_stop_alias_supports_output_options(tmp_path: Path) -> None:
    port = _find_free_port()
    command = (
        f". '{SCRIPT_PATH}' ; "
        f"Stop-HarnessAll -SkipRuntimeBackend -ModelBackendPort {port} -DryRun "
        "-Property action,ok,model_removed_count | ConvertTo-Json -Depth 20 -Compress"
    )
    completed = _powershell_command(command)

    assert completed.returncode == 0, completed.stderr + completed.stdout
    payload = json.loads(completed.stdout.strip())
    assert payload["action"] == "stopped"
    assert payload["ok"] is True
    assert payload["model_removed_count"] == 0


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
        stale_key = f"local|{str(config.resolve())}|127.0.0.1|{port}"
        existing_sessions = []
        if STATE_SESSIONS_FILE.exists():
            try:
                existing_sessions = json.loads(STATE_SESSIONS_FILE.read_text(encoding="utf-8") or "[]")
            except json.JSONDecodeError:
                existing_sessions = []
        STATE_SESSIONS_FILE.write_text(
            json.dumps(
                existing_sessions
                + [
                    {
                        "key": stale_key,
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
        assert payload["removed_count"] in {0, 1}
        state_entries = json.loads(STATE_SESSIONS_FILE.read_text(encoding="utf-8").strip() or "[]")
        assert all(entry.get("key") != stale_key for entry in state_entries)
        assert all(entry.get("process_id") != dead_process.pid for entry in state_entries)


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


@pytest.mark.skipif(shutil.which("pwsh") is None, reason="pwsh is required to run PowerShell script tests")
def test_oneshot_backend_start_container_ollama_requires_explicit_profile(tmp_path: Path) -> None:
    config = tmp_path / "harness.yaml"
    config.write_text("backend:\n  name: ollama\nmodel: qwen2.5:7b\n", encoding="utf-8")

    command = (
        f". '{SCRIPT_PATH}' ; "
        f"Start-HarnessBackend -ExecutionMode containerized -Config '{config}' -DryRun"
    )
    completed = _powershell_command(command)

    assert completed.returncode != 0
    assert "Container profile for backend 'ollama' is opt-in only" in completed.stderr
