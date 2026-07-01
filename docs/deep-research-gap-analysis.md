# Deep-Research Gap Analysis

Branch: `review/deep-research-gap-analysis` (from `main` @ `346c34c8`)
Date: 2026-07-01
Scope: code on `main` only (`harness/` production runtime, `real_ai_harness.py` demo engine, docs, tests). The newer `codex/multi-engine-routing` work is intentionally excluded.

This report compares the harness against the published state of the art for deep-research agents: frontier-lab system cards and engineering posts, academic surveys, open-source harnesses, and harness/observability/security standards. Sources are cited inline and listed at the end.

---

## 1. What the harness is today

A local control-plane harness that routes one request through one policy-gated pipeline:

```
classify_route (regex heuristics, 7 routes)
  -> input guard (regex blocklist)
  -> cache lookup
  -> retrieval (term-overlap over local corpus dir, k=24 -> rerank 6 -> pack ~1500 words)
  -> tool plan (ONE model call returning JSON tool_calls)
  -> tool execution (calculator / time_now / new_uuid / run_tests, optional docker sandbox)
  -> final generation (single call; thinking budget 256-2048 tokens)
  -> schema validate (+ at most one repair call)
  -> evidence rows + single claim -> output guard -> trace/checkpoint/cache write
```

Genuine strengths, worth keeping as differentiators: policy-owned control loop (the model never owns routing, budgets, or tool authorization), fail-closed evidence gating, per-phase checkpoints with redaction, deterministic tool error taxonomy, dual-track parity testing, and rollout flags with a documented rollback matrix. These align well with the "harness engineering" thesis that reliability comes from what surrounds the model ([LangChain: Anatomy of an Agent Harness](https://www.langchain.com/blog/the-anatomy-of-an-agent-harness)).

The core limitation: the pipeline is **static and single-pass**. Deep-research systems are defined by the opposite — dynamic reasoning, adaptive long-horizon planning, multi-hop retrieval, iterative tool use, and structured report synthesis ([Deep Research Agents: A Systematic Examination and Roadmap, arXiv:2506.18096](https://arxiv.org/abs/2506.18096)).

---

## 2. Code review findings on `main` (fix before building anything else)

**B1. Response cache can never hit.** `runtime.py` `process()` builds `cache_key_payload` containing `"run_id": run_id`, and `run_id = str(uuid.uuid4())` is fresh per request. The SHA1 key is therefore unique every time; `cache.get()` always misses. The whole cache subsystem (config, eviction, `cached=True` branch) is dead code in practice. Remove `run_id` (and any other per-request volatile) from the key.

**B2. Shared-instance concurrency corruption.** `server.py` creates one `HarnessRuntime` and both endpoints `await runtime.process(...)` with no lock. `process()` mutates instance state (`self.checkpoint_refs`, `self.trace_id`, `self.last_run_id`, `self._last_checkpoint_state`, `self.feature_level`). Any two concurrent requests interleave at the `await model_client.generate(...)` suspension points and cross-contaminate checkpoints, trace ids, and `trace_checkpoint_count`. Move per-request state into a local context object (or guard with a lock, at the cost of serializing throughput).

**B3. Schema repair never runs for non-tool routes.** The repair retry is gated by `if tool_plan_answer:` (`runtime.py`, schema-validation block), but `tool_plan_answer` is only set inside the tools branch. `structured_extraction` — the route where strict schema matters most — has `use_tools=False`, so a malformed first response is returned as `validation_block` with no repair attempt. The gate looks like a leftover; repair should key on `final_schema is not None`.

**B4. All traces share one timestamp.** `trace_cache.py`: `TraceEvent.created_at: str = datetime.now(timezone.utc).isoformat()` is a plain dataclass default, evaluated once at import. Every trace written by a given process gets the same `created_at`. Use `field(default_factory=...)`.

**B5. Route policy mutated after classification.** `process()` flips `route.allow_reasoning = True` whenever the model name matches `thinking_model_prefixes` — including on routes whose policy deliberately sets `allow_reasoning=False` (e.g. `structured_extraction`). Model naming silently overrides declared policy; precedence should be policy first.

**B6. `high_risk` always blocks.** `_resolve_next_action` unconditionally returns `ask_clarification` for `HIGH_RISK`, and `_HIGH_RISK_RE` matches the bare words `legal|medical|financial`. Any prompt containing "financial" can never proceed. Fail-closed is defensible; keyword-triggered fail-closed with no calibrated classifier is a false-positive machine (see G2, G8).

Minor: variable shadowing in `_validate_route_evidence` (`for result in tool_results` shadows the `result` dict — latent, not currently a bug); dead latency computation in `_backend_failure_response` (always 0); typo `insufficent_evidence` in the system prompt (`runtime.py:1097`) — if anything downstream ever string-matches the sentinel, the typo becomes load-bearing; tool results and planner instruction injected as `role: "assistant"` messages rather than proper tool/system roles; `readme.me` filename (twice) alongside `README.md`; `WIP` commit messages at branch tips; large `*.log` files tracked in the branch diff.

---

## 3. Gap analysis vs. deep research

Surveys converge on four core stages for deep research: **planning, question developing, web exploration, and report generation** ([Deep Research: A Survey of Autonomous Research Agents, arXiv:2508.12752](https://arxiv.org/abs/2508.12752)). The harness currently implements none of the four in iterative form. Gaps in priority order:

### G1. No agentic loop (the defining gap)
One tool-plan call, one batch of tool executions, one final generation. No observe→re-plan cycle: a tool result can never trigger another tool call, a retrieval can never trigger a follow-up query. OpenAI's Deep Research is an RL-trained single agent that "dynamically adjusts its research strategy based on information encountered" ([Deep Research System Card, OpenAI, Feb 2025](https://cdn.openai.com/deep-research-system-card.pdf)); every OSS harness (GPT-Researcher, LangChain `open_deep_research`, smolagents) runs a bounded multi-step loop. The harness already has the pieces (budgets via `max_tool_calls_per_turn`/`max_model_calls`, checkpoints, tool gates) — they're just wired to a straight line instead of a loop. Recommendation: a bounded loop `while budget and not done: generate → (tool_calls? execute, append observations : break)`, with the existing per-call guards inside the loop body. Policy-owned stop conditions, exactly as the harness philosophy prescribes.

### G2. No planning or question decomposition
`classify_route` picks a lane; nothing decomposes the task into sub-questions or a research outline. DR systems separate plan from execution — STORM generates a question outline before writing; LangChain's implementation plans sections then researches each; the surveys treat "question developing" as its own stage. A planner step that emits a typed plan artifact (checkpointable, like everything else here) is the natural fit.

### G3. Retrieval is a toy tier
`DirectoryCorpusRetriever` does substring term-overlap over files loaded into memory; `SimpleReranker` re-scores with the *same* term-overlap signal (adds no information — it is not a reranker); `pack_context` counts whitespace words as tokens. No BM25, no embeddings, no hybrid fusion, no multi-hop, and no web tier at all (no search API, no fetch tool, no browser). DR agents distinguish API-based retrieval from browser-based exploration and generally need both (arXiv:2506.18096). Even keeping the local-first stance: add BM25 (e.g. rank_bm25) + optional local embeddings (sentence-transformers / llama.cpp embeddings), a real cross-encoder or LLM reranker, tokenizer-based packing, and a `web_search`/`fetch_url` tool behind the existing allowlist+sandbox machinery.

### G4. Citation and claim verification is cosmetic
`_build_claims` emits exactly one claim: the first 220 characters of the response, marked "verified" if no evidence fields were missing. Evidence validation checks *key-name presence* in tool/parsed outputs, not content. Report-bearing systems bind individual claims to individual sources and are measured on it — DeepResearch Bench scores citation count and accuracy per report, and no frontier system exceeded ~30% on its overall metric ([arXiv:2506.11763](https://arxiv.org/abs/2506.11763)) — this is where the field differentiates. The `Claim`/`EvidenceRecord` dataclasses are already the right shape; what's missing is claim extraction (sentence-level), per-claim evidence binding, and an entailment/support check (even a second model call as verifier).

### G5. No sub-agent orchestration or parallelism
Everything is sequential in one context. Anthropic's Research system uses an orchestrator that spawns parallel subagents, reporting a 90.2% improvement over single-agent Opus on their internal eval, with a four-part subagent contract: objective, output format, tool guidance, task boundaries ([How we built our multi-agent research system, Anthropic](https://www.anthropic.com/engineering/multi-agent-research-system)). The surveys' taxonomy (single vs. multi-agent, static vs. dynamic workflow) puts this harness in the static/single quadrant. Multi-agent is a P2 concern — it only pays off after G1–G3 exist — but the manifest schema should anticipate it (sub-run ids, parent run linkage; checkpoints already carry `run_id`).

### G6. No context management or long-horizon state
No compaction, no summarization of accumulated context, no cross-session memory or progress artifacts. Anthropic's long-running-agent guidance is that compaction alone is insufficient — durable progress files, feature/task lists in structured form (JSON over Markdown), and clean-state discipline per session are what let agents span context windows ([Effective harnesses for long-running agents, Anthropic](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)). The checkpoint system is 80% of the substrate; a "resume from checkpoints + compacted summary" path would close it.

### G7. Evaluation measures routing, not research
`run_parity_sanity.py` and `run_trace_regression.py` compare route categories, statuses, and permission blocks between the two tracks — control-plane regression, useful but content-blind. Nothing measures answer quality, groundedness, citation accuracy, or route-classification accuracy against labeled data. The DR field runs GAIA, BrowseComp, Humanity's Last Exam, and rubric-based report benchmarks (DeepResearch Bench, [ResearchRubrics, arXiv:2511.07685](https://arxiv.org/abs/2511.07685)); harness best practice is to eval the *harness* (injection resistance, budget enforcement, tool misuse) not just the model. Concretely: a small local golden set (queries → expected route + rubric-graded answers), an LLM-as-judge grader behind a fixture, and a guard false-positive/false-negative suite (B6 would have been caught by this).

### G8. Router confidence is fabricated
Confidences like `0.72 + 0.03 * regex_match_count` are hand-tuned constants, not calibrated probabilities, yet they gate `ask_clarification` via thresholds tuned to two decimal places. The docs already name the fix (small intent classifier); until then, an eval set (G7) should at least measure how often the regex router is right, and the thresholds should be honest about their provenance.

### G9. Observability is nonstandard
Per-request JSON files with a bespoke schema. The ecosystem has converged on OpenTelemetry GenAI semantic conventions for LLM/agent/tool spans ([OTel GenAI spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/), [OTel AI-agent observability](https://opentelemetry.io/blog/2025/ai-agent-observability/)). Emitting OTel spans alongside (not instead of) the existing JSON traces keeps parity tooling intact and buys dashboards, latency percentiles, and cross-request queries for free. Also: `TraceStore.write` is a blocking file write on the async request path.

### G10. Safety guards below current threat-model floor
Regex blocklists for input/output, no semantic layer, no PII detection (docs acknowledge both). Measured against OWASP LLM Top 10 v2 ([genai.owasp.org](https://genai.owasp.org/llm-top-10/)): indirect prompt injection via retrieved corpus documents or tool outputs is the #1 risk and is only regex-checked here, though `split_trusted_untrusted` + post-tool gates are the right skeleton; excessive-agency controls are genuinely good (allowlists, budgets, confirmation flags); unbounded-consumption controls exist per-request but there is no auth, rate limiting, or per-tenant quota on the server at all. No `pip-audit`/dependency scanning in CI.

### G11. Streaming and API surface
No SSE streaming on `/v1/chat/completions`, which breaks OpenAI-client compatibility expectations and matters for long research runs (docs already list it as future work). Long-horizon DR runs also want a task/job API (submit → poll/stream → artifact) rather than one blocking HTTP request.

### G12. Code style and hygiene
No lint/format/type-check configuration anywhere (`pyproject.toml` has no ruff/black/mypy config, CI has no lint step) — adopting ruff + mypy --strict on `harness/` is cheap and would have flagged B4 and the shadowing. `real_ai_harness.py` is a 2,086-line monolith with mock engines at repo root; move demo/mock engines under a `demo/` or `examples/` package. Four `readme.me` files (root, `harness/`, `harness/adapters/`, `deployment/`) and three tracked log files (`text.log`, `text02.log`, `test03.log`) are repo-hygiene noise. Commit messages on feature branches ("WIP") defeat the traceability the harness otherwise works hard for.

---

## 4. Prioritized roadmap

| Priority | Item | Effort |
|---|---|---|
| P0 | Fix B1–B6 (cache key, concurrency, repair gate, trace timestamp, policy mutation, high-risk lockout) | S |
| P0 | Add ruff + mypy + pip-audit to CI (G12, G10) | S |
| P1 | Bounded agentic tool loop with policy-owned stop conditions (G1) | M |
| P1 | Real retrieval tier: BM25 + embeddings + tokenizer packing; `web_search`/`fetch_url` tools behind existing gates (G3) | M |
| P1 | Golden-set eval harness: route accuracy, rubric-graded answers, guard FP/FN suite (G7, G8) | M |
| P2 | Planner stage emitting typed, checkpointed research plans (G2) | M |
| P2 | Claim extraction + per-claim evidence binding + verifier pass (G4) | M |
| P2 | OTel GenAI spans alongside JSON traces; async trace writes (G9) | S–M |
| P2 | SSE streaming + job-style API for long runs (G11) | M |
| P3 | Orchestrator/sub-agent runs with parent-child run linkage (G5) | L |
| P3 | Compaction + resume-from-checkpoint for long-horizon sessions (G6) | M |
| P3 | Semantic guard layer + PII redaction upgrade (G10) | M |

The through-line: the harness's control-plane discipline (budgets, gates, checkpoints, fail-closed validation) is exactly what the deep-research literature says agents need around them — it is currently wrapped around a pipeline that only takes one step. Closing G1 converts every existing safety asset into an agent asset.

---

## 5. Sources

Frontier labs: [OpenAI Deep Research System Card (PDF)](https://cdn.openai.com/deep-research-system-card.pdf) · [Anthropic: How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) · [Anthropic: Effective harnesses for long-running agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)

Academic: [Deep Research Agents: A Systematic Examination and Roadmap (arXiv:2506.18096)](https://arxiv.org/abs/2506.18096) · [Deep Research: A Survey of Autonomous Research Agents (arXiv:2508.12752)](https://arxiv.org/abs/2508.12752) · [A Comprehensive Survey of Deep Research (arXiv:2506.12594)](https://arxiv.org/pdf/2506.12594) · [DeepResearch Bench (arXiv:2506.11763)](https://arxiv.org/pdf/2506.11763) · [ResearchRubrics (arXiv:2511.07685)](https://arxiv.org/pdf/2511.07685) · [Tongyi DeepResearch Technical Report (arXiv:2510.24701)](https://arxiv.org/html/2510.24701) · [DeepScholar-Bench (arXiv:2508.20033)](https://arxiv.org/pdf/2508.20033)

Open-source harnesses: [LangChain open_deep_research / Anatomy of an Agent Harness](https://www.langchain.com/blog/the-anatomy-of-an-agent-harness) · [Together: Open Deep Research](https://www.together.ai/blog/open-deep-research) · [awesome-deep-research-agent (curated repo)](https://github.com/ai-agents-2030/awesome-deep-research-agent) · GPT-Researcher, STORM, HuggingFace smolagents open-deep-research

Standards & practices: [OTel GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/) · [OTel: AI agent observability](https://opentelemetry.io/blog/2025/ai-agent-observability/) · [OWASP Top 10 for LLM Applications](https://genai.owasp.org/llm-top-10/)
