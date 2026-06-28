Deployment & Containerization Part
=================================

This section documents how to run the harness locally with containerized model backends.

Included parts
-------------

- `Dockerfile`:
  - Builds a non-root harness image
  - Installs project in editable-safe runtime layout
  - Exposes `8080`
  - Adds `/health` endpoint healthcheck.

- `docker-compose.nvidia.yaml`:
  - Supports profile-based startup for:
    - `--profile nvidia` (NVIDIA NIM + harness)
    - `--profile ollama` (Ollama + harness)
  - Includes provider-level healthchecks and startup dependency controls.

- `.env.nvidia.example` and `.env.ollama.example`
  - Environment overrides for models, provider images, and harness config inputs.

Recommended launch flows (Windows 11 / PowerShell)
-------------------------------------------------

NVIDIA NIM:

```powershell
Copy-Item .env.nvidia.example .env.nvidia
docker compose --profile nvidia --env-file .env.nvidia -f docker-compose.nvidia.yaml up --build
```

Ollama:

```powershell
Copy-Item .env.ollama.example .env.ollama
docker compose --profile ollama --env-file .env.ollama -f docker-compose.nvidia.yaml up --build
```

How it works
------------

- Harness reads runtime config from mounted `HARNESS_CONFIG`.
- Backend service starts first and exposes OpenAI-compatible API (`/v1/models`, `/v1/chat/completions`).
- Harness performs startup readiness validation against model catalog.
- If model is missing or endpoint not ready, startup fails fast with explicit diagnostics.
- Health endpoint can be used for external orchestrators.

Current limitations to know
---------------------------

- NIM and Ollama images may or may not include `curl`; compose health checks include `curl` fallback to `wget`.
- Provider model catalogs can be slow to populate; this is handled with startup retries.
- Cache and trace directories are mounted as named/host paths for persistence and cleanup.

Cutting-edge platform upgrades
------------------------------

- Add dedicated Docker network names and service dependencies for multi-node setups.
- Add GPU scheduling constraints for mixed compute nodes (A100 vs consumer GPUs).
- Add reverse proxy (Traefik/nginx) with TLS and basic auth.
- Add secret manager integration instead of env-only API keys.
- Add sidecar metrics exporter and Prometheus scrape annotations.
- Add optional model download warmup step before harness becomes healthy.
- Add CI/CD integration for rebuilding/attestation with SBOM.
- Add container security hardening:
  - image signing,
  - distroless Python runtime,
  - seccomp/apparmor defaults.

Validation commands
------------------

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8080/health" -Method Get
Invoke-RestMethod -Uri "http://127.0.0.1:8080/v1/chat/completions" -Method Post -ContentType "application/json" -Body '{"model":"llama3:8b","messages":[{"role":"user","content":"Hello"}]}'
```

