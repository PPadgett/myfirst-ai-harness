from __future__ import annotations

from abc import ABC, abstractmethod

from harness.types import ModelGenerateRequest, ModelGenerateResult


class BaseModelClient(ABC):
    @abstractmethod
    async def generate(self, req: ModelGenerateRequest) -> ModelGenerateResult:
        raise NotImplementedError

