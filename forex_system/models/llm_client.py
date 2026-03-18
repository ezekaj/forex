"""Unified LLM client for vLLM API endpoints."""

import json
import logging
from dataclasses import dataclass
from typing import Optional

import httpx

log = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    content: str
    tokens_used: int
    model: str
    raw: dict


class LLMClient:
    """
    Client for vLLM OpenAI-compatible API.
    Works with any model served via vLLM on any port.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        model: str = "auto",
        timeout: float = 120.0,
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._resolved_model = None

    async def _get_model_name(self) -> str:
        """Auto-detect model name from vLLM endpoint."""
        if self._resolved_model:
            return self._resolved_model
        if self.model != "auto":
            self._resolved_model = self.model
            return self.model
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(f"{self.base_url}/v1/models")
                data = resp.json()
                self._resolved_model = data["data"][0]["id"]
                return self._resolved_model
        except Exception as e:
            log.warning(f"Could not auto-detect model: {e}")
            return "unknown"

    async def chat(
        self,
        messages: list[dict],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Send a chat completion request."""
        model = await self._get_model_name()

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
        }

        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        choice = data["choices"][0]
        usage = data.get("usage", {})

        return LLMResponse(
            content=choice["message"]["content"],
            tokens_used=usage.get("completion_tokens", 0) + usage.get("prompt_tokens", 0),
            model=model,
            raw=data,
        )

    async def ask(
        self,
        prompt: str,
        system: str = "",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        json_mode: bool = False,
    ) -> str:
        """Simple ask: returns just the response text."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = await self.chat(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            json_mode=json_mode,
        )
        return response.content

    async def ask_json(
        self,
        prompt: str,
        system: str = "",
        max_tokens: Optional[int] = None,
    ) -> dict:
        """Ask and parse JSON response."""
        text = await self.ask(
            prompt, system, max_tokens=max_tokens, temperature=0.1, json_mode=True
        )
        # Try to extract JSON from response
        text = text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON in the response
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
            log.error(f"Could not parse JSON from response: {text[:200]}")
            return {"error": "parse_failed", "raw": text}

    async def is_healthy(self) -> bool:
        """Check if the vLLM endpoint is responding."""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{self.base_url}/v1/models")
                return resp.status_code == 200
        except Exception:
            return False

    def sync_ask(self, prompt: str, system: str = "", max_tokens: int = None) -> str:
        """Synchronous version of ask() for non-async contexts."""
        import asyncio
        return asyncio.run(self.ask(prompt, system, max_tokens))
