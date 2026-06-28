# Local LLM Harness (thinking + non-thinking)

This repository is a local control-plane harness that routes requests through policy-aware runtime stages:

- Route classification (`direct`, `grounded_qa`, `tool_required`, `structured_extraction`, `side_effecting_action`, `high_risk`)
- Thinking-budget controls (`low`, `medium`, `high`, `premium`)
- Optional retrieval over a local corpus
- Guardrails for input/output safety
- Safe tool execution (`calculator`, `time_now`, `new_uuid`)
- Schema extraction and repair for structured output
- Trace logging and optional response cache
- OpenAI-compatible request/response shape

The core goal is to keep the model from owning the entire control loop.

## Supported backends

Current backends are:

- `openai` (default) – any OpenAI-compatible HTTP endpoint
- `llamacpp` – direct local GGUF via `llama-cpp-python`
- `nvidia_nim` (`nim`, `nvidia`, `nvidia-nim`) – NVIDIA NIM serving endpoints (local GPU or remote), including containerized local setups
- `ollama` – Ollama OpenAI-compatible endpoint (local container or remote host)

## Install

```bash
python -m venv .venv
. .venv/Scripts/Activate.ps1
pip install -e .
```

## Quick local start (no container)

Default backend points to an OpenAI-compatible service:

```bash
python -m harness.server --config harness.yaml --host 127.0.0.1 --port 8080
```

## NVIDIA local GPU + containerized harness setup

The repo includes:

- `Dockerfile` – builds the harness container
- `docker-compose.nvidia.yaml` – runs NIM and/or Ollama and the harness via compose profiles
- `.env.nvidia.example` – copy and fill this before starting
- `harness-nvidia.yaml` – harness defaults for NIM backend
- `.env.ollama.example` – copy and fill this before starting Ollama

## Containerized local backends (Windows 11 / PowerShell)

### Option A: NVIDIA NIM

#### 1) Configure environment

```bash
Copy-Item .env.nvidia.example .env.nvidia
```

Set these at minimum:

- `NIM_LLM_IMAGE` = your NVIDIA NIM container image (exact model image from NVIDIA)
- `NIM_MODEL_PATH` = model identifier or local path used by the NIM service
- `NIM_REQUEST_MODEL` = model name that your NIM image expects in requests (if different from `NIM_MODEL_PATH`)

Optional:

- `NIM_API_KEY` for NVIDIA-hosted endpoints
- `HF_TOKEN` / `NGC_API_KEY` for private Hugging Face or NGC model access

#### 2) Start the stack

```bash
docker compose --profile nvidia --env-file .env.nvidia -f docker-compose.nvidia.yaml up --build
```

This brings up:

- `nvidia-nim` on `http://127.0.0.1:8000/v1`
- `harness` on `http://127.0.0.1:8080`

### Option B: Ollama

#### 1) Configure environment

```powershell
Copy-Item .env.ollama.example .env.ollama
```

Set at minimum:

- `OLLAMA_IMAGE` to the image you want (default is `ollama/ollama:latest`)
- `OLLAMA_MODEL` to a local model tag (for example `llama3.1:latest` or `qwen2.5:7b`)
- `HARNESS_MODEL` to match your model

#### 2) Start the stack

```bash
docker compose --profile ollama --env-file .env.ollama -f docker-compose.nvidia.yaml up --build
```

This brings up:

- `ollama` on `http://127.0.0.1:11434/v1`
- `harness` on `http://127.0.0.1:8080`

### 3) Readiness and runtime checks (both stacks)

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8080/health" -Method Get
```

Expected: `backend` in the response matches either `ollama` or `nvidia_nim`, and `model` is your selected model.

Harness startup behavior:

- It retries provider model catalog checks at startup (`/v1/models`) before serving.
- It fails fast with a clear error if the selected model is not present in the provider catalog.
- Health checks in compose now probe provider `/v1/models` endpoints with `curl`/`wget` fallback.

## API usage

### OpenAI-compatible route

```bash
curl "http://127.0.0.1:8080/v1/chat/completions" -H "Content-Type: application/json" `
-d "{\"model\":\"hf://meta-llama/Llama-3.1-8B-Instruct\",\"messages\":[{\"role\":\"user\",\"content\":\"What is the capital of France?\"}]}"
```

### Custom answer route

```bash
curl "http://127.0.0.1:8080/v1/answer" -H "Content-Type: application/json" `
-d "{\"input\":\"Return strict JSON: {\\\"answer\\\": string}\",\"response_schema\":{\"type\":\"object\",\"properties\":{\"answer\":{\"type\":\"string\"}},\"required\":[\"answer\"]}}"
```

If you prefer a PowerShell-native request, use:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8080/v1/chat/completions" -Method Post -ContentType "application/json" -Body '{"model":"hf://meta-llama/Llama-3.1-8B-Instruct","messages":[{"role":"user","content":"What is the capital of France?"}]}'
```

## Runtime config (`harness.yaml`)

```yaml
backend:
  name: openai      # openai | llamacpp | nvidia_nim | ollama | auto
  # aliases also accepted: nvidia, nim, nvidia-nim
  base_url: "http://127.0.0.1:11434/v1"
  api_key: null
  timeout_seconds: 120
  max_tokens: 768
  extra_headers: {}   # best-practice passthrough for provider-specific headers
  extra_body: {}      # best-practice passthrough for provider-specific request payloads

model: "qwen2.5:7b"
corpus_dir: "corpus"
trace_dir: "traces"
cache_dir: ".cache"
enable_cache: true
max_cache_entries: 2000

tool_allowlist:
  - "calculator"
  - "time_now"
  - "new_uuid"
```

NVIDIA NIM example:

```yaml
backend:
  name: nvidia_nim
  base_url: "http://127.0.0.1:8000/v1"
  api_key: null
  timeout_seconds: 120
  max_tokens: 1536
  extra_headers: {}
  extra_body: {}

model: "hf://meta-llama/Llama-3.1-8B-Instruct"
```

For containerized local NIM runs, also use `harness-nvidia.yaml` (already included) and
compose wiring from `docker-compose.nvidia.yaml`.

Ollama example:

```yaml
backend:
  name: ollama
  base_url: "http://ollama:11434/v1"
  api_key: null
  timeout_seconds: 120
  max_tokens: 1024
  extra_headers: {}
  extra_body: {}

model: "llama3.1:latest"
```

### Windows 11 + Docker Desktop note

- Use Docker Desktop with WSL 2 backend and NVIDIA GPU support enabled.
- The compose file now uses `device_requests` to request GPU on Windows.
- If your Linux environment requires legacy runtime mode, swap to `runtime: nvidia` and remove `device_requests`.

## Notes

- `llamacpp` still supports direct local GGUF.
- For production, add stronger policy models, per-tool authorization, and more robust verification gates.
- NVIDIA NIM is treated as an OpenAI-compatible transport, so route behavior, retrieval, guards, tracing, and caching remain in this harness regardless of model provider.

## Part-specific documentation

- [readme.me](/readme.me) — project-level part index and roadmap.
- [docs/core-pipeline-and-policy-control.md](/docs/core-pipeline-and-policy-control.md)
- [docs/local-model-backend-abstraction.md](/docs/local-model-backend-abstraction.md)
- [docs/tooling-validation-safety-traces.md](/docs/tooling-validation-safety-traces.md)
- [docs/api-surface-and-launch.md](/docs/api-surface-and-launch.md)
- [harness/readme.me](/harness/readme.me) — legacy component index.
- [harness/adapters/readme.me](/harness/adapters/readme.me) — legacy adapter index.
- [deployment/readme.me](/deployment/readme.me) — container and deployment runbook.
