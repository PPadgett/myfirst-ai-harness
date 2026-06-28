Tooling, Validation, Safety, and Traces/Caching
===============================================

This section documents:

- `harness/tools.py`
- `harness/guards.py`
- `harness/validation.py`
- `harness/trace_cache.py`

## 1) `harness/tools.py` — tooling plane

What it does:

- Defines tool specs for model/tool-call planning (`calculator`, `time_now`, `new_uuid`).
- Converts internal tool contracts into OpenAI function tool schemas.
- Executes approved tools in-process and returns structured results.
- Tool execution returns `ToolCallResult` envelopes with route-policy aware error codes.

Current tool design details:

- `ToolSpec` and `ToolCallResult` dataclasses define stable execution contracts.
- `list_tool_specs()` is authoritative for available tools.
- `execute_tool(name, args)` is currently deterministic and local.
- `calculator` uses AST-safe expression evaluation (no Python eval).

How to evolve:

- Add tool authorization policies by route and tenant.
- Add async/tool worker isolation (worker pool or subprocess) for untrusted tools.
- Add strict argument validation with JSON schema enforcement before execution.
- Add timeout/cancellation and error envelopes per tool call.

---

2) `harness/guards.py` — safety guardrails

What it does:

- Runs content-based checks on input/output.
- Returns allow/block decision + reason code.
- `check_input_text` currently uses phrase/blocklist matching for jailbreak-style and risky patterns.
- `check_output_text` applies lightweight unsafe-content filtering.
- `check_tool_request`, `check_tool_output`, and `split_trusted_untrusted` are part of the hardening and evidence/redaction path.

How it evolves:

- Add semantic safety model (classifier).
- Add PII detection + redaction.
- Add policy-driven severity scoring and escalation path.
- Add allowlist/denylist per deployment environment.

---

3) `harness/validation.py` — structure validation

What it does:

- Extracts first JSON object from model output, including raw/fenced recoveries.
- Validates output against simple schema shape (`type`, `properties`, `required`).
- Returns parse result plus diagnostics.

Current limits:

- Type validation is intentionally lightweight.
- Nested schema validation and regex constraints are not fully modeled yet.

How to evolve:

- Upgrade to JSON Schema validator (`jsonschema` / `fastjsonschema`).
- Add strict mode with retry budget and structured error hints.
- Add output normalization and schema-version pinning.

---

4) `harness/trace_cache.py` — operational memory

What it does:

- `TraceStore`: writes per-request trace event JSON to disk.
- `ResponseCache`: sha1-keyed response cache with LRU-like retention.
- `TraceEvent` includes route, status, latency, stage history, and result summary.
- Runtime checkpoint JSON files are written at each phase in `state/` and referenced via `checkpoint_id`.

How it evolves:

- Move traces to structured sink (OpenTelemetry + centralized store).
- Add query/filter APIs for audit and eval workflows.
- Add distributed cache (Redis) and configurable cache TTL.
- Add cache invalidation tied to policy version changes.
- Add anomaly detection over traces (latency spikes, schema failures).

---

Integrated safety stack summary
------------------------------

- Input safety (`guards`) runs before expensive compute.
- Route policy (`router`) controls whether tool/retrieval/schema are enabled.
- Validation (`validation`) constrains structured outputs.
- Tool outputs and runtime traces (`tools` + `trace_cache`) create auditable, reproducible outcomes.

Cutting-edge stack ideas:

- Add policy compiler that enforces guard + schema + route constraints at startup.
- Add eval harness for safety false positives/false negatives and route accuracy.
- Add automated rollback on repeated failure patterns per policy version.

## 5) Hardened security mode (runtime + demo parity)

Hardening is enabled through the same environment controls used by launch and parity flows:

- `HARNESS_FEATURE_LEVEL=basic|hardening`
- `HARNESS_REQUIRE_EVIDENCE=0|1`
- `HARNESS_TOOL_SANDBOX=off|docker`
- `HARNESS_ENABLE_ADVANCED_ROUTER=0|1`
- `HARNESS_ROUTE_OVERRIDES`

Behavior under hardened mode:

- Inputs are split and sanitized from trusted/untrusted boundaries.
- Tool requests and outputs are gate-checked with explicit reasons (`tool_request_blocked`, `tool_output_blocked`, etc.).
- Evidence and claims are redacted before being returned in traces/checkpoints.
- Unknown/forbidden tools return `error_code=unknown_tool` in the tool plane.
- Sandbox failures are surfaced with deterministic tool error codes:
  - `tool_sandbox_unavailable`
  - `tool_sandbox_timeout`
  - `tool_sandbox_exec_error`

Roll-forward / rollback notes:

- Enable hardened mode first in non-production traffic.
- If parity checks regress, retain API compatibility and disable one flag at a time to isolate behavior.
- Always keep the old transport payload fields; hardening is additive.

## Hardening flow notes (current behavior)

- Runtime and demo both apply deterministic pre/post tool guards in each execution path.
- Pre-tool guard:
  - `check_tool_request_with_tool` validates tool-specific argument schemas before dispatch.
  - Unknown tools are blocked with `unknown_tool` (never silently ignored).
- Post-tool guard:
  - `check_tool_output` scans outputs and can emit `tool_output_blocked`.
- Evidence and trace payloads are sanitized with `redact_sensitive_args` before checkpoint persistence.
- Sandbox path:
  - `HARNESS_TOOL_SANDBOX=docker` routes `run_tests` and manifest-marked `tool_sandbox_required` tools through docker-first execution.
  - Deterministic failure envelopes are `tool_sandbox_unavailable`, `tool_sandbox_timeout`, and `tool_sandbox_exec_error`.
- Manifest compatibility:
  - Runtime uses `validator` first and falls back to `manifests.validator_fields` + `manifests.hard_fail_errors` for legacy manifests.
  - Route-level manifest overrides from `runtime_config.route_overrides` (and `HARNESS_ROUTE_OVERRIDES`) are merged before routing validation.
