"""Token counting via ``tiktoken``.

``tiktoken`` is a runtime dependency, so by default this module counts tokens
with the model-appropriate encoding (or ``cl100k_base`` as a universal
fallback). The ``len(text) / 4`` heuristic still lives here as defensive
insurance for two edge cases: (1) air-gapped / first-run environments where
tiktoken can't lazy-download its BPE files from
``openaipublic.blob.core.windows.net``, and (2) niche platforms where the
``tiktoken`` import itself fails. Use :func:`has_tiktoken` to detect the
fallback at runtime.

Note: ``tiktoken`` is OpenAI's tokenizer. For Claude / DeepSeek prompts we
use ``cl100k_base`` as a universal proxy -- close (within ~5-10%) but not
literally the provider's tokenizer. We intentionally do not call Anthropic's
``messages.count_tokens()`` endpoint here: it's an authenticated API call per
prompt, which would make ``--dry-run`` chatty, slow, and dependent on an API key.
"""
from __future__ import annotations

import functools

_HEURISTIC_CHARS_PER_TOKEN = 4


def has_tiktoken() -> bool:
    """True iff a real tokenizer is available."""
    return _maybe_import_tiktoken() is not None


def count_tokens(text: str | None, model: str | None = None) -> int:
    """Count tokens in ``text``. Uses tiktoken when available, heuristic otherwise."""
    if not text:
        return 0
    enc = _get_encoder(model)
    if enc is not None:
        try:
            return len(enc.encode(text))
        except Exception:
            pass
    return max(1, len(text) // _HEURISTIC_CHARS_PER_TOKEN)


def is_precise(model: str | None = None) -> bool:
    """True iff :func:`count_tokens` will use a precise encoder for ``model``."""
    return _get_encoder(model) is not None


@functools.lru_cache(maxsize=16)
def _get_encoder(model: str | None):
    tiktoken = _maybe_import_tiktoken()
    if tiktoken is None:
        return None
    if model:
        try:
            return tiktoken.encoding_for_model(model)
        except Exception:
            pass
        # Try a normalized form (dots -> dashes, common in our adapters).
        try:
            return tiktoken.encoding_for_model(model.replace(".", "-"))
        except Exception:
            pass
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


@functools.lru_cache(maxsize=1)
def _maybe_import_tiktoken():
    try:
        import tiktoken  # type: ignore
        return tiktoken
    except ImportError:
        return None
