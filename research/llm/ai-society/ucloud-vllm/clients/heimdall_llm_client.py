import os
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI


@dataclass(frozen=True)
class HeimdallLLMConfig:
    base_url: str = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
    api_key: str = os.getenv("OPENAI_API_KEY", "heimdall-local")
    model: str = os.getenv("HEIMDALL_LLM_MODEL", os.getenv("HEIMDALL_MODEL", "Qwen/Qwen3-0.6B"))


class HeimdallLLMClient:
    def __init__(self, config: HeimdallLLMConfig | None = None) -> None:
        self.config = config or HeimdallLLMConfig()
        self.client = AsyncOpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
        )

    async def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        resp = await self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            **kwargs,
        )
        return resp.choices[0].message.content or ""

    async def chat_completion(self, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        return await self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            **kwargs,
        )
