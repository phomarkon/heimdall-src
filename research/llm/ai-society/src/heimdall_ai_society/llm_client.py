from __future__ import annotations

import asyncio
import json
import math
from dataclasses import dataclass, field
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from heimdall_ai_society.schemas import LLMBidDecision


class LLMClientError(RuntimeError):
    pass


@dataclass(frozen=True)
class OpenAICompatibleLLMClient:
    base_url: str
    api_key: str
    model: str
    temperature: float
    max_tokens: int
    timeout_seconds: float
    provider: str = "vllm"
    http_referer: str | None = None
    app_title: str | None = None
    supports_response_format: bool = True
    base_urls: list[str] | None = None
    max_concurrency: int = 4
    per_endpoint_max_concurrency: int | None = None
    _endpoint_urls: tuple[str, ...] = field(init=False, repr=False)
    _total_semaphore: asyncio.Semaphore = field(init=False, repr=False)
    _endpoint_semaphores: tuple[asyncio.Semaphore, ...] = field(init=False, repr=False)
    _endpoint_lock: asyncio.Lock = field(init=False, repr=False)
    _next_endpoint: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        endpoints = tuple((self.base_urls or [self.base_url]))
        if not endpoints:
            raise ValueError("OpenAICompatibleLLMClient requires at least one endpoint")
        per_endpoint = self.per_endpoint_max_concurrency or max(1, math.ceil(self.max_concurrency / len(endpoints)))
        object.__setattr__(self, "_endpoint_urls", endpoints)
        object.__setattr__(self, "_total_semaphore", asyncio.Semaphore(self.max_concurrency))
        object.__setattr__(self, "_endpoint_semaphores", tuple(asyncio.Semaphore(per_endpoint) for _ in endpoints))
        object.__setattr__(self, "_endpoint_lock", asyncio.Lock())

    async def decide(self, messages: list[dict[str, str]]) -> LLMBidDecision:
        endpoint_index, endpoint = await self._next_endpoint_slot()
        async with self._total_semaphore, self._endpoint_semaphores[endpoint_index]:
            return await asyncio.to_thread(self._decide_sync, endpoint, messages)

    async def tool_round(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        tool_choice: str | dict[str, Any] = "auto",
    ) -> dict[str, Any]:
        endpoint_index, endpoint = await self._next_endpoint_slot()
        async with self._total_semaphore, self._endpoint_semaphores[endpoint_index]:
            return await asyncio.to_thread(self._tool_round_sync, endpoint, messages, tools, tool_choice)

    @property
    def endpoint_urls(self) -> list[str]:
        return list(self._endpoint_urls)

    async def _next_endpoint_slot(self) -> tuple[int, str]:
        async with self._endpoint_lock:
            index = self._next_endpoint
            object.__setattr__(self, "_next_endpoint", (index + 1) % len(self._endpoint_urls))
        return index, self._endpoint_urls[index]

    def _decide_sync(self, endpoint: str, messages: list[dict[str, str]]) -> LLMBidDecision:
        if not self.supports_response_format:
            return self._decide_plain_json_sync(endpoint, messages)
        try:
            return self._decide_schema_sync(endpoint, messages)
        except LLMClientError as exc:
            if self.provider != "ollama" or "returned HTTP" not in str(exc):
                raise
            return self._decide_plain_json_sync(endpoint, messages)

    def _decide_schema_sync(self, endpoint: str, messages: list[dict[str, str]]) -> LLMBidDecision:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "heimdall_bid_decision",
                    "schema": LLMBidDecision.model_json_schema(),
                    "strict": True,
                },
            },
        }
        body = json.dumps(payload).encode("utf-8")
        request = Request(
            f"{endpoint.rstrip('/')}/chat/completions",
            data=body,
            headers=self._headers(),
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                raw: dict[str, Any] = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise LLMClientError(f"{self.provider} endpoint returned HTTP {exc.code}: {detail}") from exc
        except (URLError, TimeoutError) as exc:
            raise LLMClientError(f"could not reach {self.provider} endpoint: {exc}") from exc

        return self._decision_from_response(raw)

    def _decide_plain_json_sync(self, endpoint: str, messages: list[dict[str, str]]) -> LLMBidDecision:
        payload = {
            "model": self.model,
            "messages": self._plain_json_messages(messages),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        body = json.dumps(payload).encode("utf-8")
        request = Request(
            f"{endpoint.rstrip('/')}/chat/completions",
            data=body,
            headers=self._headers(),
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                raw: dict[str, Any] = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise LLMClientError(f"{self.provider} endpoint returned HTTP {exc.code}: {detail}") from exc
        except (URLError, TimeoutError) as exc:
            raise LLMClientError(f"could not reach {self.provider} endpoint: {exc}") from exc

        return self._decision_from_response(raw)

    def _decision_from_response(self, raw: dict[str, Any]) -> LLMBidDecision:
        try:
            content = raw["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMClientError(f"unexpected OpenAI-compatible response shape: {raw}") from exc
        try:
            return LLMBidDecision.model_validate_json(content)
        except ValueError as exc:
            raise LLMClientError(f"model returned invalid bid JSON: {content}") from exc

    def _plain_json_messages(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        instruction = (
            "Return only a valid JSON object matching this schema, with no markdown or extra text: "
            f"{json.dumps(LLMBidDecision.model_json_schema(), separators=(',', ':'))}"
        )
        return [{"role": "system", "content": instruction}, *messages]

    def _tool_round_sync(
        self,
        endpoint: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: str | dict[str, Any],
    ) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "tools": tools,
            "tool_choice": tool_choice,
        }
        body = json.dumps(payload).encode("utf-8")
        request = Request(
            f"{endpoint.rstrip('/')}/chat/completions",
            data=body,
            headers=self._headers(),
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                raw: dict[str, Any] = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise LLMClientError(f"{self.provider} endpoint returned HTTP {exc.code}: {detail}") from exc
        except (URLError, TimeoutError) as exc:
            raise LLMClientError(f"could not reach {self.provider} endpoint: {exc}") from exc
        try:
            return raw["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMClientError(f"unexpected OpenAI-compatible response shape: {raw}") from exc

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.http_referer:
            headers["HTTP-Referer"] = self.http_referer
        if self.app_title:
            headers["X-Title"] = self.app_title
        return headers
