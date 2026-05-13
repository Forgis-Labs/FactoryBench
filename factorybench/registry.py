"""Registry of model classes the runner can look up by name."""
from __future__ import annotations

from typing import Any, Callable, TypeVar

_REGISTRY: dict[str, type] = {}

T = TypeVar("T", bound=type)


def register_model(name: str) -> Callable[[T], T]:
    """Decorator: register a class as a FactoryBench-evaluable model.

    The class must implement ``predict(self, prompt: str) -> str``. It may
    optionally implement ``predict_batch(self, prompts: list[str]) -> list[str]``
    for backends that support batched inference.
    """
    if not isinstance(name, str) or not name:
        raise ValueError("register_model requires a non-empty name string")

    def decorator(cls: T) -> T:
        if not hasattr(cls, "predict"):
            raise TypeError(f"{cls.__name__} must define predict(self, prompt) -> str")
        if name in _REGISTRY and _REGISTRY[name] is not cls:
            raise ValueError(f"model name {name!r} is already registered")
        _REGISTRY[name] = cls
        return cls

    return decorator


def get_model(name_or_instance: str | Any) -> Any:
    """Resolve a model. Resolution order:

    1. User-registered name (from ``@register_model``).
    2. Built-in provider spec (``mock``, ``gpt-*``, ``claude-*``, ``deepseek-*``).
    3. A passed-in instance with a ``predict`` method (used as-is).
    """
    if isinstance(name_or_instance, str):
        if name_or_instance in _REGISTRY:
            return _REGISTRY[name_or_instance]()
        from .adapters import resolve_builtin
        builtin = resolve_builtin(name_or_instance)
        if builtin is not None:
            return builtin
        available = ", ".join(sorted(_REGISTRY)) or "(none registered)"
        raise KeyError(
            f"unknown model {name_or_instance!r}. "
            f"Registered: {available}. "
            "Built-in providers: mock, gpt-*, claude-*, deepseek-*."
        )
    if not hasattr(name_or_instance, "predict"):
        raise TypeError("model object must define a predict(prompt) -> str method")
    return name_or_instance


def list_models() -> list[str]:
    """Return the names of all registered models."""
    return sorted(_REGISTRY)
