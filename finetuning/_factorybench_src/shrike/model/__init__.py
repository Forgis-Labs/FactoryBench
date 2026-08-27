"""Shrike and BearingModel wrappers. Vendored; see ``shrike/__init__.py``."""

# Lazy imports, avoids pulling in peft/transformers when only
# the tokenizer subpackage is needed (e.g., tokenizer training on SageMaker).
def __getattr__(name):
    if name in ("Shrike", "ShrikeConfig"):
        from .shrike import Shrike, ShrikeConfig
        return Shrike if name == "Shrike" else ShrikeConfig
    if name == "BearingModel":
        from .bearing import BearingModel
        return BearingModel
    if name == "TOTEMTokenizer":
        # Upstream keeps totem.py beside this file; this copy took the
        # tokenizers as one subtree, so it lives in shrike.tokenizer.
        from ..tokenizer.totem import TOTEMTokenizer
        return TOTEMTokenizer
    raise AttributeError(f"module 'shrike.model' has no attribute {name!r}")

__all__ = ["Shrike", "ShrikeConfig", "BearingModel", "TOTEMTokenizer"]
