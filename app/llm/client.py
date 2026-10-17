"""LLM access behind one tiny interface.

Everything in Layer 2 talks to `LLMClient.generate_json(...)`. Which model
sits behind it is a config choice:

    LLM_PROVIDER=ollama   (default)  local model, e.g. qwen2.5:3b
    LLM_PROVIDER=rules               no LLM at all (rules only, runs anywhere)

Why a JSON *schema* on every call: Ollama can constrain the model's output to
the schema (grammar-constrained decoding). A 3B model then physically cannot
return anything but a valid enum value / valid object, which removes a whole
class of "the model rambled instead of answering" failures.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Callable, Protocol


class LLMError(Exception):
    """The model was unreachable or returned unusable output."""


class LLMClient(Protocol):
    model: str

    def generate_json(self, system: str, prompt: str, schema: dict) -> dict: ...


class OllamaClient:
    def __init__(self, model: str | None = None, host: str | None = None,
                 timeout: float = 120.0, retries: int = 1):
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
        self.host = (host or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.timeout = timeout
        self.retries = retries

    def generate_json(self, system: str, prompt: str, schema: dict) -> dict:
        body = {
            "model": self.model,
            "stream": False,
            "format": schema,                       # constrained decoding
            "options": {"temperature": 0, "num_ctx": 4096},  # deterministic, small context
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": prompt}],
        }
        last: Exception | None = None
        for _ in range(self.retries + 1):
            try:
                req = urllib.request.Request(
                    f"{self.host}/api/chat", data=json.dumps(body).encode(),
                    headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    payload = json.load(r)
                return json.loads(payload["message"]["content"])
            except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as e:
                last = e
        raise LLMError(f"ollama call failed: {last}")


class NullLLM:
    """Rules-only mode: every LLM call fails, so the pipeline uses rules + flags review."""
    model = "none (rules only)"

    def generate_json(self, system: str, prompt: str, schema: dict) -> dict:
        raise LLMError("LLM disabled")


class FakeLLM:
    """Scripted model for tests: handler(system, prompt, schema) -> dict."""
    model = "fake"

    def __init__(self, handler: Callable[[str, str, dict], dict]):
        self.handler = handler
        self.calls: list[tuple[str, str, dict]] = []

    def generate_json(self, system: str, prompt: str, schema: dict) -> dict:
        self.calls.append((system, prompt, schema))
        return self.handler(system, prompt, schema)


def get_client(provider: str | None = None) -> LLMClient:
    provider = (provider or os.getenv("LLM_PROVIDER", "ollama")).lower()
    if provider == "ollama":
        return OllamaClient()
    if provider in ("rules", "none"):
        return NullLLM()
    raise ValueError(f"unknown LLM provider: {provider}")
