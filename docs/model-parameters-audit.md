# Model Parameter Audit — Multi-LLM per Processor Tests

Branch: `codex/multi-engine-routing` (unified) · Date: 2026-07-01
Purpose: every generation parameter that affects the CPU / GPU / NPU / Hybrid comparison tests — configured **and** hard-coded — plus the issues they cause.

---

## 0. Fix status (2026-07-01)

Issues §7.1–§7.9 were addressed on this branch after the audit:

| # | Issue | Fix |
|---|---|---|
| 1 | No seeds | `seed`/`top_p`/`top_k`/`repeat_penalty`/`stop` plumbed end-to-end: HTTP API → runtime (all stages incl. tool-plan, repair, agentic workers) → adapter payload → both provider engines. `Benchmark-MultiDeviceHarness.ps1` sends a fixed `-Seed 42` (MLPerf-style; `-Seed 0` disables). A fixed seed is necessary but not sufficient — batch composition still perturbs results (arXiv:2601.06118), so benchmark at concurrency 1. |
| 2 | Fake NPU device | `--device npu` no longer silently maps to CUDA 0; it runs CPU with warning `npu_unsupported_by_transformers_backend_running_cpu`. Real NPU execution remains Lemonade-only. |
| 3 | Chat-template mismatch | transformers path uses `tokenizer.apply_chat_template` when available (flat join only as last resort, with warning); llama.cpp raw fallback derives ChatML-aware stops instead of `["</s>", "\n\n\n"]`. |
| 4 | Per-engine sampler drift | Explicit request sampling now overrides Ollama Modelfile params and engine defaults on every path. |
| 5 | Route temps silently applied | `/v1/chat/completions` no longer injects `temperature=0.3`; generation knobs are forwarded only when the client sends them (both endpoints consistent). Route defaults still apply when omitted — by design, now documented. |
| 6 | Misleading telemetry | `tokens_per_second` now uses final-generation time (`generation_ms`, new field); `ttft_ms` is `null` (no streaming exists). |
| 7 | Stub masquerade | Any canned-answer substitution now emits `fallback_content_substituted` in provider warnings. Canned texts still exist for the diagnostic backend — do not benchmark against prompts they match. |
| 8 | CPU context halved | Unchanged (config choice) — equalize `max_context` in the loadout yaml before context-sensitive comparisons. |
| 9 | temperature=0 semantics | Unchanged per engine (greedy on transformers vs sampler elsewhere) — pin full sampling explicitly; now possible via #1. |

Also fixed alongside (found in the deep-research review): response cache could never hit (volatile `run_id` + live `route_metadata` reference in the key); concurrent requests corrupted each other's checkpoints (per-request context object now); schema repair never ran on non-tool routes; all traces shared one import-time timestamp; thinking-model detection overrode strict-schema policy; `high_risk` keyword lockout now applies only in `hardening` mode.

## 1. What your benchmark tests send

| Script | temperature | max_tokens | seed | notes |
|---|---|---|---|---|
| `Benchmark-MultiDeviceHarness.ps1` | `0.2` (param default, line 17) | `256` (line 18) | **none** | `execution_profile` set per case; `agentic_parallel` case at line 139 |
| `Invoke-HarnessFullPromptBenchmark.ps1` | `0` for generation probes (line 715) | `2048` default (line 29) | **none** | phase-level `max_tokens` overrides |
| `Invoke-HarnessOneShot` (ad-hoc) | **not sent unless `-Temperature` given** → route policy decides (0.0–0.3, §3) | route default (§3) | **none** | ad-hoc runs are NOT parameter-comparable to benchmark runs |

**There is no `seed` parameter anywhere in the repository** — not in request types, configs, adapters, or scripts. Every sampled run is unrepeatable.

## 2. Per-backend settings (yaml)

### `harness-multi-device.yaml` — model `qwen3:4b`
| backend | port | max_context | max_output_tokens | max_concurrency | timeout_s | capabilities |
|---|---|---|---|---|---|---|
| npu | 11433 | 4096 | 768 | 1 | 120 | npu, hybrid |
| gpu | 11434 | 4096 | 1024 | 2 | 120 | gpu, hybrid |
| cpu | 11435 | **2048** | 1024 | 4 | 120 | cpu |

### `harness-qwen-hybrid-live.yaml` — model `qwen2.5-7b-instruct`
| backend | port | runtime | max_context | max_output_tokens | conc | timeout_s |
|---|---|---|---|---|---|---|
| npu | 11433 | local_provider | 4096 | 768 | 1 | 180 |
| gpu | 11434 | local_provider | 4096 | 1024 | 2 | 180 |
| cpu | 11435 | local_provider | **2048** | 1024 | 4 | 180 |
| hybrid | 13305 | **lemonade_ryzenai** (`amd/Qwen2.5-7B-Instruct_rai_1.7.1_hybrid`) | 4096 | 1024 | 1 | 300 |

Neither yaml configures `extra_body`, so nothing supplements the wire payload below.
Note: `max_output_tokens`/`max_context` are used for backend *selection metadata* — they are not enforced as generation caps on the request path.

## 3. Hard-coded route policy (`harness/router.py`)

Applied when the request doesn't specify temperature/max_tokens (e.g. `Invoke-HarnessOneShot` without flags):

| route | temperature | max_new_tokens | thinking budget | retrieval | tools |
|---|---|---|---|---|---|
| direct | 0.3 | 700 | low | – | – |
| grounded_qa | 0.2 | 900 | medium | yes | – |
| tool_required | 0.0 | 900 | low | – | yes (3) |
| structured_extraction | 0.0 | 512 | low | – | – |
| code_or_data | 0.1 | 1200 | high | yes | yes (4) |
| side_effecting_action | 0.0 | 900 | high | yes | yes (2) |
| high_risk | 0.0 | 1200 | high | yes | yes (2) |

Thinking budgets (tokens): low 256 · medium 512 · high 1024 · premium 2048 (`runtime.py _budget_tokens`).

## 4. Hard-coded runtime transforms (`harness/runtime.py`)

- temperature clamped to [0.0, 2.0]; max_tokens floored at 64
- `strict_schema` routes: temperature forced to `min(0.2, requested)`
- tool-plan call: temperature `min(0.2, requested)`, tokens `min(512, max(128, requested//4))`, reasoning off
- schema-repair call: temperature `0.0`, tokens `256`
- agentic-parallel workers: tokens `max(128, min(512, requested//2))`, requested temperature, reasoning off; controller gets full budget
- model-name match against `thinking_model_prefixes` silently flips `allow_reasoning=True`
- `tokens_per_second` in telemetry = output_tokens ÷ **whole-pipeline latency** (includes retrieval/tool/checkpoint time); `ttft_ms` = total latency (no streaming — not a real TTFT)

## 5. What actually crosses the wire (`harness/adapters/openai_compatible_client.py`)

Exactly four generation-relevant fields: `model`, `messages`, `temperature`, `max_tokens` (+ `response_format: json_object` when schema; + `tools` when planning).
**Not sent: seed, top_p, top_k, repeat_penalty, stop.** Those come from whatever each backend server defaults to (§6) — a different sampler per device.

## 6. Hidden per-engine defaults (`harness/local_model_provider.py`)

| engine path | temperature default | top_p / top_k / penalty | chat template | device handling |
|---|---|---|---|---|
| transformers (lines ~440–500) | temp ≤ 0 → `do_sample=False` (greedy); temp > 0 → sample with **model's generation_config.json** defaults for top_p/top_k | from model file, invisible to harness | **`_build_prompt` = flat string join (line 331) — no chat template at all** | `--device npu` → **torch `device=0` (CUDA GPU 0)**; only "cpu" → CPU. **No NPU engine exists in this provider** |
| llama.cpp, chat API (lines ~540–585) | `0.2` if request omits it | from **Ollama Modelfile params** if model came from the Ollama store, else **llama-cpp-python library defaults** (top_p 0.95, top_k 40, repeat_penalty 1.1) | `chatml` only when an Ollama template is detected | n_gpu_layers from launch flags (`-LlamaCppGpuLayers`, default 0 = CPU) |
| llama.cpp raw fallback (lines ~592–600, on chat-API exception) | inherited | dropped entirely | `_build_prompt` flat join + hard-coded `stop=["</s>", "\n\n\n"]` — **wrong stops for ChatML/Qwen** | — |
| deterministic fallback stub | — | — | — | **canned answers**: "capital of France is Paris", "most played music video → Gangnam Style", etc. (`_fallback_answer`, line ~337). If `fallback_active` is true, "model" output is fake |
| Lemonade (hybrid, external) | receives only temp + max_tokens; all other sampling = Lemonade server defaults | server-side | proper template applied by Lemonade | real Ryzen AI NPU+iGPU |

Provider constructor default `max_new_tokens=256` (line 306) — yet another layer's default, distinct from yaml (768/1024) and route (512–1200).

## 7. Issues identified

1. **No seeds → nothing is reproducible.** Any A/B between devices is one sample from four different distributions.
2. **The "NPU" backend never ran on the NPU.** `local_model_provider` has no NPU/ONNX/Ryzen path; `--device npu` maps to torch `cuda:0` (or fails toward CPU). The historical baseline in `multi-device-benchmark-baseline-2026-07-02.md` shows NPU at 15.14 tok/s ≈ CPU's 16.64 — CPU-class throughput. The only true NPU execution in the repo is the Lemonade **hybrid** backend.
3. **Prompt templates differ per device.** transformers path (and llama.cpp fallback) flatten messages into a raw string — no ChatML. That reliably degrades instruct-tuned Qwen: your CPU row's refusal ("I'm sorry, but I can't assist") and GPU row's degenerate list ("Dreamcast 2000/2001/2002...") are classic template-mismatch artifacts. The **hybrid (Lemonade) row is the only one with a proper server-side template — and the only coherent answer.** You weren't comparing processors; you were comparing prompt formats.
4. **Sampler config is split across 4 layers** (script → route policy → adapter → engine defaults), and the engine layer differs per device. top_p/top_k on one device come from an Ollama Modelfile, on another from generation_config.json, on another from llama-cpp-python defaults.
5. **Omitting `-Temperature` changes the experiment**: route classification silently picks 0.0–0.3.
6. **Telemetry can't be used for device speed**: tokens_per_second includes pipeline overhead; ttft_ms is fake.
7. **Canned-answer stub can masquerade as a model** whenever fallback engages — several README demo questions ("most played music video") are literally in the stub's answer list. Check `fallback_active`/`provider_warning` in traces before trusting any output.
8. **CPU context is halved** (2048 vs 4096) — long-prompt tests truncate differently per device.
9. `temperature=0` means greedy on transformers but sampler-with-defaults elsewhere — "deterministic" probes aren't.

## 8. Recommended fixes (in order)

1. Add `seed`, `top_p`, `top_k`, `repeat_penalty`, `stop` to `ModelGenerateRequest` → adapter payload → provider paths; pin a fixed seed + explicit sampler in every benchmark payload.
2. Use `tokenizer.apply_chat_template` in the transformers path; fix llama.cpp fallback stop tokens; assert template parity across devices before benchmarking.
3. Rename/gate the "npu" local_provider backend until a real NPU engine exists (or route NPU-only through Lemonade too); verify with `requested_device` vs `actual_device` in provider traces — the fields already exist.
4. Benchmarks: always pass temperature/max_tokens/seed explicitly; equalize max_context; take N≥5 samples per cell; record `fallback_active`, `finish_reason`, `truncated` per row.
5. Treat `multi-device-benchmark-baseline-2026-07-02.md` results as invalid except the Hybrid row's qualitative sanity.
