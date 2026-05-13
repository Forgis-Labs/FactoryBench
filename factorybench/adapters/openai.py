"""OpenAI / OpenAI-compatible (DeepSeek, etc.) chat-completions adapter."""
from __future__ import annotations

import os
from typing import Any


class OpenAIAdapter:
    """Calls an OpenAI-compatible chat completions endpoint.

    Args:
        model: Provider model id (e.g. ``"gpt-5.1"``, ``"deepseek-chat"``).
        api_key: Explicit API key. Falls back to ``$OPENAI_API_KEY`` (or the
            ``api_key_env`` you pass).
        base_url: Optional override (used to point at DeepSeek's endpoint).
        api_key_env: Which environment variable to consult if ``api_key`` is
            not given. Defaults to ``"OPENAI_API_KEY"``.
        max_tokens: Max output tokens per call.
        timeout: Per-call timeout in seconds.
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        max_tokens: int = 1024,
        timeout: float = 60.0,
    ):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "OpenAI provider requested but the 'openai' package is not installed. "
                'Install with: pip install "factorybench[openai]"'
            ) from e

        key = api_key or os.environ.get(api_key_env)
        if not key:
            raise RuntimeError(
                f"Missing API key. Set ${api_key_env} or pass api_key= explicitly."
            )

        kwargs: dict[str, Any] = {"api_key": key, "timeout": timeout}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)
        self._model = model
        self._max_tokens = max_tokens

    def predict(self, prompt: str) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self._max_tokens,
        )
        return resp.choices[0].message.content or ""
