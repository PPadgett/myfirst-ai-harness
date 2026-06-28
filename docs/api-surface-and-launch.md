API Surface and Launch
======================

This section covers request interface and how the harness is started:

- `harness/server.py`
- `harness/cli.py`
- `harness/config.py`
- `harness.yaml`
- `README.md` (operational runbook)

## 1) `harness/server.py` — HTTP surface + lifecycle

What it does:

- Builds FastAPI app and injects runtime dependencies.
- Resolves backend:
  - `llamacpp`
  - `nvidia_nim`
  - `ollama`
  - `openai`
  - `auto` fallback
- Performs startup readiness checks against provider model catalog for OpenAI-compatible providers.
- Exposes endpoints:
  - `GET /health`
  - `POST /v1/chat/completions`
  - `POST /v1/answer`

Contract details:

- Request body defaults:
  - model fallback to config model
  - temperature, token limits, safety profile, and route override support
  - response schema passthrough
- Response is fully assembled by runtime (`harness/runtime.py`) and returned as OpenAI-like object.

How to evolve for cutting-edge API:

- Add OpenAI streaming endpoint (`/v1/chat/completions` SSE).
- Add SSE/WebSocket transport for long tasks/tool loops.
- Add auth middleware (JWT/OAuth/API key introspection).
- Add quotas/rate-limits and per-tenant throttling.
- Add deterministic request replay endpoint for debugging.

---

## 2) `harness/cli.py` — launch entrypoint

What it does:

- Small wrapper around `harness.server.main`.
- Keeps CLI launch behavior simple and shell-agnostic.

How to evolve:

- Add subcommands for config validation, dry-run, and health probe.
- Add environment check command (`harness inspect`) for config/provider preflight.

---

## 3) `harness/config.py` in launch path

What role it plays in launch:

- `server.py` calls `load_runtime_config()` by default.
- Config path can be overridden via CLI args (`--config`).
- Environment values can override file values for backend/model/timeouts/key settings.

Launch integration notes:

- Keep `HARNESS_*` environment vars explicit in compose for reproducible infra.
- Prefer config-per-profile (`harness-nvidia.yaml`, `harness-ollama.yaml`) for GPU/local provider consistency.

---

## 4) `harness.yaml` and runtime profiles

What it does:

- Defines the default execution contract for non-container and local provider runs.
- Includes backend, retrieval, cache, policy/version, and tool settings.

Recommended layout:

- Keep base `harness.yaml` with shared defaults.
- Use provider-specific overlays for local deployments:
  - `harness-nvidia.yaml`
  - `harness-ollama.yaml`
- Keep `.env.nvidia.example` and `.env.ollama.example` for secrets/paths.

How to evolve config UX:

- Add config inheritance (`base` + `overlay`) support.
- Add validation mode (`harness validate`) with machine-readable error output.
- Add model/capability schema checks (`max_tokens`, tool support, context).

---

## 5) `README.md` and operational start

This is the user-facing operational contract:

- local non-container startup (`python -m harness.server`)
- containerized profiles (NIM/Ollama)
- endpoint usage patterns
- health checks and troubleshooting basics

How to evolve docs-driven operations:

- Add a compatibility matrix (NVIDIA/NIM/Ollama/provider versions).
- Add benchmark scripts and reproducible load profiles.
- Add incident playbook for backend cold-start/model-missing/capacity failures.

---

Windows 11 + container quick flow:

```powershell
Copy-Item .env.nvidia.example .env.nvidia
docker compose --profile nvidia --env-file .env.nvidia -f docker-compose.nvidia.yaml up --build

Copy-Item .env.ollama.example .env.ollama
docker compose --profile ollama --env-file .env.ollama -f docker-compose.nvidia.yaml up --build
```

