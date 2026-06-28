Core Pipeline and Policy Control
===============================

This section documents the control and policy layer:

- `harness/config.py`
- `harness/router.py`
- `harness/runtime.py`

## 0) Dual-track baseline and parity strategy

This repo keeps two execution tracks:

- `real_ai_harness.py` (demo engine): transparent phase trace and manifest-first routing.
- `harness.runtime` (production engine): API-first policy runtime with persistence and checkpoints.

Baseline tracking is tracked in:

- `tests/fixtures/baseline/real_ai/queries.json`
- `tests/fixtures/baseline/runtime/queries.json`

Parity tooling:

- `scripts/run_parity_sanity.py` checks route category/route confidence parity and mismatch signals across selected fixtures.
- `tests/benchmarks/run_trace_regression.py` computes route-category and status/permission match rates.

Current mismatch notes to preserve:

- Real engine outputs include `trace_phase_count`; production uses `trace_checkpoint_count`.
- Demo engine checkpoints are written under `state/` and include route-level checkpoint metadata.
- Runtime checkpoints are written under `state/` and include normalized stage payloads.
- Runtime output adds `tool_calls` and `feature_level` fields.

## Hardened security mode and rollout controls

- `HARNESS_FEATURE_LEVEL=basic|hardening` controls policy strictness.
- `HARNESS_REQUIRE_EVIDENCE=0|1` gates fail-closed evidence enforcement.
- `HARNESS_TOOL_SANDBOX=off|docker` controls isolated heavy-tool execution.
- `HARNESS_ENABLE_ADVANCED_ROUTER=0|1` enables manifest-level threshold overrides and evidence requirements in router.
- `HARNESS_ROUTE_OVERRIDES` (JSON object) / `runtime_config.route_overrides` can override route metadata for testing or staged rollout.
- Route-specific sandbox and validator policy is resolved in this order:
  1. manifest `validator` field (preferred),
  2. legacy `manifests.validator_fields` compatibility fallback,
  3. environment and override defaults.
- The route override branch is currently loaded from:
  `harness/config.py` (`load_runtime_config`), passed through `runtime.process()` as `route_overrides`, then merged in `harness/router.py` in `_resolve_route_metadata`.

Compatibility branch note:

- `validator` is always preferred when present.
- `manifests.validator_fields` and `manifests.hard_fail_errors` are applied only if `validator` is absent.
- In permissive mode invalid manifest entries are downgraded into best-effort route defaults.

Rollback guidance:

- If routing behavior regresses, set `HARNESS_ENABLE_ADVANCED_ROUTER=0`.
- If evidence blocking is too strict, set `HARNESS_REQUIRE_EVIDENCE=0`.
- If sandbox is unstable, set `HARNESS_TOOL_SANDBOX=off` and keep local execution policy.

## 1) `harness/config.py` — config resolution and backend normalization

What it does:

- Loads YAML config from `harness.yaml` by default, or a custom path passed to the loader.
- Normalizes backend aliases:
  - `nvidia`, `nim`, `nvidia-nim` → `nvidia_nim`
  - `llama_cpp`, `llama-cpp` → `llamacpp`
- Enforces known backend names and required fields:
  - requires non-empty `backend.base_url`
  - requires non-empty `model` for OpenAI-compatible providers such as `ollama` and `nvidia_nim`
- Resolves `backend.api_key` with env precedence:
  - `HARNESS_API_KEY`
  - `NVIDIA_API_KEY` / `NIM_API_KEY`
  - `OLLAMA_API_KEY`
- Parses runtime settings:
  - corpus/trace/cache directories
  - caching policy
  - tool allowlist
  - route overrides and prompt/policy versions

How it works in practice:

- This config object is loaded once during server startup.
- It becomes the single source of truth passed to `HarnessRuntime`.
- Alias handling keeps cross-platform config compatibility while preventing brittle backend naming.

How to evolve this part (cutting-edge):

- Add typed schema validation (Pydantic/dataclass validation).
- Add config hot-reload with immutable snapshot and diff checks.
- Add secrets provider integration (Vault, Azure Key Vault, AWS Secrets Manager).
- Add policy/routing versions in config with runtime compatibility checks.
- Add environment-based profile expansion (`dev`, `qa`, `prod`) with merge order.

---

## 2) `harness/router.py` — route policy decision engine

What it does:

- Defines `Route` and `RoutePolicy` objects.
- Classifies inbound request text into policy lanes.
- Controls runtime knobs:
  - retrieval on/off
  - tools on/off
  - strict schema
  - reasoning allowance
  - route-specific temperature / token budgets
  - tool allowlist and verification requirements

How it works:

- `classify_route(messages, response_schema=None, route_override=None)` is the current entry.
- A regex/heuristic stack applies in priority order:
  1. explicit route override
  2. schema-required route
  3. high-risk detection
  4. side-effecting action detection
  5. tool-needed detection
  6. grounded QA detection
  7. reasoning/data tasks
  8. low-risk tasks
  9. default direct route
- For each route it returns a `RoutePolicy` with strict execution controls.
- Route definitions prefer `validator` first and then fallback to `manifests.validator_fields` for compatibility.

How to evolve this part (cutting-edge):

- Replace regex heuristics with a small intent classifier model.
- Add confidence and “uncertain route” mode with safe fallback policy.
- Add retrieval- or tool-aware routing using query embeddings.
- Add policy override source from user tenancy / tenant-specific policy packs.
- Add per-route evaluation metrics and auto-tuning.

---

## 3) `harness/runtime.py` — orchestration runtime

What it does:

- Coordinates the full request lifecycle:
  1. route + policy selection
  2. input guard check
  3. cache lookup
  4. retrieval and reranking (if enabled)
  5. optional tool planning and execution
  6. final generation with optional schema/repair
  7. output guard check
  8. cache write, trace write, response assembly

What it changes from plain model forwarding:

- Adds policy gates per route.
- Adds tool-augmented responses.
- Adds schema repair loop.
- Adds persistent artifacts:
  - request traces
  - cached responses

How it works in sequence:

- `process(request)` creates `trace_id` and `request_id`.
- Determines user message and route policy.
- Performs model name detection for reasoning-capable models using `thinking_model_prefixes`.
- Enforces requested tool constraints against route allowlist.
- Uses deterministic retrieval if `route.use_retrieval`.
- If tools are enabled:
  - prompts model for `tool_calls`
  - executes allowed tools
  - injects tool outputs into final context
- Executes final response generation.
- If schema exists: validates and optionally triggers one repair generation.
- Persists optional trace and cache records.

How to evolve this part (cutting-edge):

- Add tool-output schema validation and canonical result coercion.
- Add multi-step tool loops with bounded reasoning graph.
- Add planner/verifier architecture (separate policy model for tool execution).
- Add explicit idempotency keys + request dedupe.
- Add distributed cache and tracing (Redis + OpenTelemetry + Jaeger).
- Add circuit-breaker around model calls + fallback providers.

---

Related integration:

- API layer uses `runtime.process(...)` as the main execution primitive (`harness/server.py`).
- Adapters are injected into runtime and remain pluggable (`harness/adapters`).
- Validation and safety logic is enforced through `validation.py` and `guards.py`.

## Hardened security mode and rollout controls

The dual-track architecture keeps one external API while allowing policy hardening to be staged:

- `HARNESS_FEATURE_LEVEL=basic` (default) keeps permissive behavior while preserving compatibility.
- `HARNESS_FEATURE_LEVEL=hardening` enables stricter confidence gating and evidence/fail-closed policy behavior.
- `HARNESS_REQUIRE_EVIDENCE=1` makes grounded/tool routes fail-closed when required evidence is missing.
- `HARNESS_TOOL_SANDBOX=off|docker` controls heavy-tool execution isolation.
- `HARNESS_ENABLE_ADVANCED_ROUTER=1` enables manifest-driven threshold and route overrides.

Rollback matrix:

- routing drift: set `HARNESS_ENABLE_ADVANCED_ROUTER=0`
- evidence regressions: set `HARNESS_REQUIRE_EVIDENCE=0`
- sandbox instability: set `HARNESS_TOOL_SANDBOX=off`

Parity checks now include:

- route-category parity
- `next_action` parity
- permission-block behavior
- evidence presence parity
- runtime/infra parity checks for backend availability and provider catalog checks (`Get-HarnessBackendStatus` runtime plane + provider plane output)
