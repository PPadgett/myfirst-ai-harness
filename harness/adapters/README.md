Adapters Part (Model Transport Layer)
====================================

The adapter layer defines how the harness speaks to model backends.
Each backend maps the harness `ModelGenerateRequest` into provider-specific API calls and returns
standard `ModelGenerateResult`.

What it does
------------

- Define a common protocol with `BaseModelClient`.
- Convert runtime calls to backend-compatible payload shape.
- Handle response normalization (`content`, `reasoning`, `usage`).

Files and behavior
------------------

- `base.py`
  - Abstract base class with required `generate(request) -> ModelGenerateResult`.

- `openai_compatible_client.py`
  - Generic OpenAI-compatible client.
  - Sends `POST /chat/completions`.
  - Supports:
    - JSON mode for schema-guided output
    - tool specifications forwarding
    - optional reasoning payload
    - extra headers/body passthrough.

- `nvidia_nim_client.py`
  - Small subclass of OpenAI-compatible client.
  - Keeps NIM-specific defaults while preserving OpenAI transport compatibility.

- `llama_cpp_client.py`
  - Direct local llama.cpp GGUF execution via `llama_cpp`.
  - Converts message list to prompt when API path is not available.

How to add a new provider
--------------------------

1. Create a new `XxxClient` subclass of `BaseModelClient`.
2. Implement `generate(self, req)` to:
   - map `req` fields (`model`, `messages`, `temperature`, `max_new_tokens`, schema/tool hints),
   - call provider endpoint or SDK,
   - return `ModelGenerateResult(text, reasoning, raw, usage)`.
3. Register selection logic in `harness/server.py` backend branch.
4. Add config docs + env vars in deployment docs.

How it works in runtime
-----------------------

- Backend is selected during app construction.
- A single `BaseModelClient` instance is injected into `HarnessRuntime`.
- Runtime stays backend-agnostic; only protocol-level expectations are shared.

Cutting-edge upgrades
---------------------

- Add streaming callback support (`chunked` generator from model APIs) and map into API streams.
- Implement provider error taxonomy and automatic retry budgets/backoff.
- Add per-backend response normalizer adapters (e.g., extraction of tool-like blocks).
- Add model discovery and capability probing:
  - context window size
  - tool/function support
  - max output tokens
  - multimodal/image support flags.
- Add adapter-level circuit breaker and budget enforcement.
- Add async connection pooling + request tracing headers (trace ids, tenant ids).

Security and reliability notes
------------------------------

- Never log model API keys or raw Authorization headers.
- Validate provider URLs and allow only https/http schemes from allowed env values in strict environments.
- Add per-request timeout + jittered retry for transient transport failures.

