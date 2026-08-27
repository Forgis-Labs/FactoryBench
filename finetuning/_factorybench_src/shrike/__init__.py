"""Shrike, Time Series Understanding via Discrete Tokenization.

A backbone-agnostic framework that turns any decoder-only LLM into a
time-series reasoner through discrete VQ-VAE tokenization.

VENDORED CODE, DO NOT EDIT TO ADD FEATURES
===========================================
This is not FactoryBench code. It is a **partial copy** of the internal the authors
Shrike/TSLM repository, taken at the state that produced the checkpoints
FactoryBench evaluates, and committed here wholesale by `31e5488`
("finetuning: SageMaker training infra with DoRA on Shrike/BearingModel",
2026-05-28, 3795 insertions across 13 files). It therefore has no history in
this repository: there is one commit that adds every line, and no upstream
commit id was recorded at the time. The upstream repository is the authority
on where it came from.

**Partial** means the `model` and `tokenizer` subtrees only. The upstream
package's data, eval, training-loop and tokenizer-training modules were not
copied, because nothing in FactoryBench calls them. See "Submodules" below for
what is actually here.

**Frozen** is deliberate. Its only job is to deserialize four pretrained
checkpoints: `Shrike.from_pretrained` and `BearingModel.from_pretrained` must
reconstruct the exact module tree, vocabulary extension and DoRA r=32 adapter
those `.pt` files were saved from. Changing a layer name or a config default
here does not improve anything, it makes a checkpoint fail to load, so bring
fixes in from upstream rather than making them here.

It is copied into the SageMaker `source_dir` at build time and imported from
there (see `train_factorybench.py`, "Vendored shrike package"), which is why
it sits inside `_factorybench_src/` rather than at the repository root.

Quick start::

    import shrike as hy

    # Load a pretrained model
    model = hy.Shrike.from_pretrained("checkpoints/best_model.pt",
                                        totem_ckpt="checkpoints/totem.pt",
                                        llm_id="Qwen/Qwen3-4B")

    # Analyze a signal
    result = model.analyze(signal, question="What is the trend?")
    print(result)

    # Forecast (Chameleon-style: generates codes, decodes to values)
    forecast = model.forecast(signal, horizon=64)
    print(forecast.values)

Submodules present in this copy:
    shrike.model, Shrike and BearingModel wrappers
    shrike.tokenizer, TOTEM, FSQ, FSQ-Transformer and FSQ-Transformer-RoPE

Upstream submodules NOT copied: shrike.data, shrike.eval, shrike.train.
"""

__version__ = "0.1.0"

# Convenience re-exports so users can write:
#   from shrike import Shrike, ShrikeConfig
#   or: import shrike as hy; hy.Shrike(...)
#
# Lazy imports: avoids pulling in peft/transformers when only
# the tokenizer subpackage is needed (e.g., tokenizer training on SageMaker).
def __getattr__(name):
    if name == "Shrike" or name == "ShrikeConfig":
        from shrike.model.shrike import Shrike, ShrikeConfig
        return Shrike if name == "Shrike" else ShrikeConfig
    if name == "TOTEMTokenizer":
        # Upstream keeps TOTEM under shrike/model/; in this copy the tokenizers
        # were taken as one subtree, so it lives in shrike/tokenizer/.
        from shrike.tokenizer.totem import TOTEMTokenizer
        return TOTEMTokenizer
    if name == "BearingModel":
        from shrike.model.bearing import BearingModel
        return BearingModel
    raise AttributeError(f"module 'shrike' has no attribute {name!r}")
