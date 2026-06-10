"""Built-in model adapters and the string-spec resolver."""
from __future__ import annotations

from typing import Any

from .mock import MockAdapter

# Each entry: (predicate(spec) -> bool, factory(spec) -> adapter)
_BUILTIN_PROVIDERS: list[tuple] = []


def _register_builtin(predicate, factory) -> None:
    _BUILTIN_PROVIDERS.append((predicate, factory))


# Built-in registrations are below as functions so the optional SDKs are
# imported lazily only when a matching spec is resolved.

def _make_openai(spec: str):
    from .openai import OpenAIAdapter
    return OpenAIAdapter(model=spec)


def _make_anthropic(spec: str):
    from .anthropic_adapter import AnthropicAdapter
    return AnthropicAdapter(model=spec.replace(".", "-"))


def _make_deepseek(spec: str):
    from .openai import OpenAIAdapter
    return OpenAIAdapter(
        model=spec,
        base_url="https://api.deepseek.com",
        api_key_env="DEEPSEEK_API_KEY",
    )


_register_builtin(lambda s: s == "mock", lambda s: MockAdapter())
_register_builtin(lambda s: s.startswith("gpt-") or s.startswith("gpt5"), _make_openai)
_register_builtin(lambda s: s.startswith("claude-"), _make_anthropic)
_register_builtin(lambda s: s.startswith("deepseek"), _make_deepseek)


def resolve_builtin(spec: str) -> Any | None:
    """Return an adapter instance for ``spec`` if a built-in provider matches,
    else ``None``.
    """
    for predicate, factory in _BUILTIN_PROVIDERS:
        if predicate(spec):
            return factory(spec)
    return None


__all__ = ["MockAdapter", "resolve_builtin"]
