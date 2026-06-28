from __future__ import annotations

import asyncio
import traceback
from typing import Any

from harness.adapters.base import BaseModelClient
from harness.types import ModelGenerateRequest, ModelGenerateResult


class LlamaCppClient(BaseModelClient):
    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_gpu_layers: int = 0,
        n_threads: int = 4,
        verbose: bool = False,
    ) -> None:
        try:
            from llama_cpp import Llama
        except Exception as exc:
            raise RuntimeError("llama_cpp is not installed. Install it to use backend='llamacpp'.") from exc

        self.model = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            verbose=verbose,
        )

    async def generate(self, req: ModelGenerateRequest) -> ModelGenerateResult:
        def _run() -> ModelGenerateResult:
            payload: dict[str, Any] = {
                "messages": req.messages,
                "temperature": req.temperature,
                "max_tokens": req.max_new_tokens,
            }
            payload["stop"] = ["</s>", "\n\n\n"]
            try:
                result = self.model.create_chat_completion(**payload)
            except Exception:
                # compatibility for older llama-cpp-python signatures
                chat_prompt = self._messages_to_prompt(req.messages)
                result = self.model(
                    chat_prompt,
                    max_tokens=req.max_new_tokens,
                    temperature=req.temperature,
                    stop=payload["stop"],
                )
                return ModelGenerateResult(
                    text=str(result.get("choices", [{}])[0].get("text", "")),
                    reasoning=None,
                    raw=result,
                    usage={"input_tokens": 0, "output_tokens": 0},
                )
            choice = result.get("choices", [{}])[0]
            msg = choice.get("message", {})
            return ModelGenerateResult(
                text=msg.get("content", ""),
                reasoning=msg.get("reasoning") or None,
                raw=result,
                usage=result.get("usage", {"prompt_tokens": 0, "completion_tokens": 0}),
            )

        return await asyncio.to_thread(_run)

    @staticmethod
    def _messages_to_prompt(messages: list[dict[str, str]]) -> str:
        out: list[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if not content:
                continue
            out.append(f"{role.upper()}: {content}")
        return "\n\n".join(out)

