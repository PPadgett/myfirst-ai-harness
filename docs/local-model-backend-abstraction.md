Local Model Backend Abstraction
==============================

This section covers the backend transport layer:

- `harness/adapters/base.py`
- `harness/adapters/openai_compatible_client.py`
- `harness/adapters/llama_cpp_client.py`
- `harness/adapters/__init__.py`

## 1) `harness/adapters/base.py` — contract boundary

What it does:

- Defines `BaseModelClient` abstraction with a single async method:
  - `generate(req: ModelGenerateRequest) -> ModelGenerateResult`
- Enforces a stable runtime interface independent of provider shape.

How to evolve this:

- Add optional methods for:
  - `ping()/models()/supports_tools()/supports_streaming()`
  - `tokenize` and `context_window` introspection.
- Add typed cancellation and retry contract (exception taxonomy).

---

## 2) `harness/adapters/openai_compatible_client.py` — OpenAI-style transport

What it does:

- Implements `generate(...)` for OpenAI-compatible endpoints:
  - builds chat payload (`model`, `messages`, `temperature`, `max_tokens`)
  - forwards `tools` and `response_format` for schema mode
  - injects optional reasoning payload when requested
  - applies `extra_headers` and `extra_body` passthrough
- Maps provider response to:
  - `text`
  - extracted `reasoning`
  - `usage` metrics

Where it is used:

- `harness/server.py` for OpenAI-compatible backends, including custom local providers.
- `harness/adapters/nvidia_nim_client.py` via subclass reuse.

How to evolve this:

- Add robust provider-specific response adapters.
- Add request retries with backoff and timeout budgets by endpoint.
- Add request tracing headers and request IDs.
- Add JSON schema strict mode (`json_schema` aware retries, stricter parser behavior).

---

## 3) `harness/adapters/llama_cpp_client.py` — direct local GGUF path

What it does:

- Loads GGUF via `llama_cpp`.
- Generates with `create_chat_completion` when available.
- Falls back to raw prompt inference path for compatibility with older `llama_cpp` APIs.
- Uses shared message conversion utility for compatibility.

How it evolves toward cutting-edge:

- Add explicit context truncation strategy per model.
- Add quantization-specific tuning params and async batch scheduling.
- Add GPU layer auto-scaling and memory-aware model loading.
- Add per-request streaming if supported by backend version.

---

## 4) `harness/adapters/__init__.py` — export surface

What it does:

- Re-exports concrete adapters:
  - `BaseModelClient`
  - `LlamaCppClient`
  - `OpenAICompatibleClient`
  - `NvidiaNimClient`
- Keeps import ergonomics and stable public API for server/runtime selection.

How to evolve this:

- Add explicit `__all__` policy for plugin-like registration.
- Add adapter metadata registry (name, capabilities, provider tags).

---

NIM-specific note:

- `NvidiaNimClient` is a thin specialization of OpenAI-compatible path with NIM defaults for reasoning field naming.
- Startup validation in `server.py` can verify that configured model exists in `/v1/models` before accepting traffic.

How to add a new backend:

1. add subclass in `harness/adapters`.
2. implement stable `generate` translation.
3. register in server backend switch.
4. document provider model and auth assumptions in config.
