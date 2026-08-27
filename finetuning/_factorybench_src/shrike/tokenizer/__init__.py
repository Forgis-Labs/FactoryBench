"""Time series tokenizers for Shrike. Vendored; see ``shrike/__init__.py``.

Converts raw time series into discrete token sequences that an LLM can process.
All tokenizers share the same interface:
    tokenize(signal) -> Tensor[int]   (encode to discrete codes)
    decode(codes) -> Tensor            (reconstruct signal)

Available tokenizers:
    - TOTEM: VQ-VAE with learned 256-entry codebook, 4:1 compression.
        Trained on generic time series (UCR archive).
        Located at shrike/tokenizer/totem.py

    - FSQ: Finite Scalar Quantization with CNN encoder.
        Grid quantization (no learned codebook), configurable levels.
        Located at shrike/tokenizer/fsq.py

    - FSQ Transformer: FSQ with Transformer encoder for global context.
        Each code position sees the entire signal via self-attention before
        quantization. Produces structured code sequences (4-10% self-transitions)
        vs CNN FSQ (~0.2%). Inspired by Archetype.
        Located at shrike/tokenizer/fsq_transformer.py

    - FSQ Transformer RoPE: as above with rotary position embeddings.
        Located at shrike/tokenizer/fsq_transformer_rope.py

Tokenizer *training* lives upstream (shrike.tokenizer.train_fsq_transformer)
and was not copied into FactoryBench: the four checkpoints being evaluated
carry their tokenizer weights with them, staged onto S3 as the `totem_ckpt` /
`fsq_ckpt` input channel. Nothing here trains a tokenizer.
"""

from .fsq import FSQTokenizer, FSQConfig
from .fsq_transformer import FSQTransformerTokenizer, FSQTransformerConfig
from .totem import TOTEMTokenizer

__all__ = [
    "TOTEMTokenizer",
    "FSQTokenizer", "FSQConfig",
    "FSQTransformerTokenizer", "FSQTransformerConfig",
]
