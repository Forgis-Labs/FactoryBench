"""Anthropic Messages API adapter.

Module file is ``anthropic_adapter.py`` (not ``anthropic.py``) so the import
``from anthropic import Anthropic`` resolves to the SDK and not this module.
"""
from __future__ import annotations

import os


class AnthropicAdapter:
    """Calls the Anthropic Messages API.

    Args:
        model: Claude model id (e.g. ``"claude-opus-4-7"``,
            ``"claude-sonnet-4-6"``). Users may pass dotted names like
            ``"claude-sonnet-4.6"``; callers should normalize before passing.
        api_key: Explicit API key. Falls back to ``$ANTHROPIC_API_KEY``.
        max_tokens: Max output tokens per call.
        timeout: Per-call timeout in seconds.
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        max_tokens: int = 1024,
        timeout: float = 60.0,
    ):
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "Anthropic provider requested but the 'anthropic' package is not installed. "
                'Install with: pip install "factorybench[anthropic]"'
            ) from e

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("Missing API key. Set $ANTHROPIC_API_KEY.")

        self._anthropic = anthropic
        self._client = anthropic.Anthropic(api_key=key, timeout=timeout)
        self._model = model
        self._max_tokens = max_tokens

    def predict(self, prompt: str) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        for block in response.content:
            if block.type == "text":
                return block.text
        return ""
