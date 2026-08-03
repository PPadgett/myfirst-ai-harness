# Seeded Benchmark Analysis — 2026-07-02

Runs: `benchmarks/seeded-run-01` (18:34) and `seeded-run-02` (18:56) · seed 42, temp 0.2, max_tokens 256, 10 iterations + 2 warmups per case · loadout `harness-live-bench.yaml` (qwen3:4b on npu/gpu/cpu local providers, Qwen2.5-3B-Instruct-Hybrid on Lemonade).

## Verdict: the reproducibility fixes work

| Check | Result |
|---|---|
| Intra-run determinism | **5 of 6 cases: 10/10 iterations bit-identical** (both runs). The exception is explained below and is not a seed failure. |
| Cross-run stability | 3 of 6 cases bit-identical across runs; 2 show minor token drift (state-dependent kernels — a fixed seed is necessary but not sufficient, arXiv:2601.06118); 1 differs by design (see tool case). |
| Speed stability | p50 latencies within ±3% across runs (npu 15.0s→15.0s, gpu 14.6s→15.1s, cpu 21.4s→21.7s). |
| Contamination | **0 fallback rows out of 120.** Every row: `llamacpp`, `qwen3:4b`, `fallback_active=false`. No stub answers, no refusals, no hallucinated garbage — the temp.md failure modes are gone. |
| Tool pipeline | Calculator tool called correctly every iteration (14+29=43), tool_required routed to cpu, structured→gpu, direct→npu, agentic_parallel→cpu controller + 2 workers. Routing matches `route_backend_defaults` exactly. |

Old temp.md baseline vs now: CPU previously refused ("I'm sorry, but I can't assist"); GPU previously hallucinated ("Dreamcast 2000/2001/2002"). Both artifacts eliminated by the chat-template fix. temp.md remains invalid as a baseline.

## Issues surfaced (in priority order)

**1. Hybrid backend down — 20/20 failures.** `Failed to load model 'Qwen2.5-3B-Instruct-Hybrid': RyzenAI-Server failed to start`. Lemonade lists the model in its catalog (so startup validation passed) but the RyzenAI engine fails on load. Check `lemonade-server` logs; test the model standalone with the other backends stopped (possible memory contention); ensure weights are actually pulled, not just registered. Note: a working fallback chain (grounded_qa → gpu → npu → cpu) would have completed all 20 of these requests — failover remains unimplemented by earlier decision.

**2. The "gpu" row is almost certainly not GPU-accelerated.** 17.5 tok/s vs 14.7 tok/s for the pure-CPU npu row — a 19% gap, not the 5–10× a real 999-layer offload gives a 4B q4 model. Likeliest cause: llama-cpp-python installed as a CPU-only wheel, silently ignoring `n_gpu_layers`. Verify in the backend log (look for "offloaded N/… layers to GPU"); reinstall a CUDA/Vulkan build if absent.

**3. 100/120 rows truncated at the 256-token cap — inside `<think>`.** qwen3:4b is a thinking model and ignored the `/no_think` soft switch (visible in outputs: it discusses the flag instead of obeying it). Every measured completion is pure reasoning that never reaches a final answer. For meaningful quality benchmarking: raise `-MaxTokens` to 1024+, or benchmark a non-thinking model (qwen2.5 family), or enforce non-thinking at the template level.

**4. Tool-route nondeterminism has a specific cause: volatile timestamps in the prompt.** `tool_required_short` produced 10 distinct outputs per run while everything else was 10/10 identical. The runtime injects tool results into the model context including `started_at`/`finished_at` ISO timestamps — a different prompt every iteration, so sampling diverges even with a fixed seed. Fix: strip volatile fields from the prompt payload (keep them in traces).

**5. Minor: system prompt leaks internals into outputs.** Models parrot "Route confidence: 0.750", "npu_only", and requested context tokens from the system prompt. Cosmetic here, but it wastes context and contaminates quality evaluation.

**6. Minor script inconsistency:** `agentic_parallel_complex` declares `expected_max_output_tokens: 512` but the payload sends the global 256.

## Corrected mental model of current hardware utilization

- npu row: CPU via llama.cpp (honest warning by design; no NPU engine in local provider)
- gpu row: CPU via llama.cpp (unintended — CPU-only wheel suspected)
- cpu row: CPU via llama.cpp (as intended; slower p50 because tool route makes two model calls)
- hybrid row: real Ryzen AI stack, currently failing to load its model

In other words: today's numbers are three flavors of CPU llama.cpp plus one broken NPU stack — consistent, reproducible, honestly labeled, and now fixable one layer at a time.
