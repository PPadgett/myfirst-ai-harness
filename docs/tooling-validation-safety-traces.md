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

