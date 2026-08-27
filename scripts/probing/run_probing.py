#!/usr/bin/env python
"""FactoryBench linear-probing experiment.

Self-contained end-to-end runner. Given the faithful eval prompts and the
question JSONs, it:

  1. Runs a frozen open-weight LLM over each prompt and caches, for every
     transformer layer, two residual-stream read-outs:
       - last prompt token
       - mean-pooled over the time-series token span
  2. Builds ground-truth *concept* labels from episode provenance.
  3. Trains per-(concept, layer, read-out) probes:
       - linear (logistic regression, C selected by grouped CV)
       - MLP reference (1 hidden layer) -> "linearity gap"
     with three controls:
       - selectivity  (probe on shuffled labels -> control task)
       - random-init model (same arch, untrained weights)
       - raw-numeric-input baseline (probe on per-channel signal statistics)
     under episode-disjoint train/test splits, with bootstrap CIs.
  4. Measures the model's own *behavioural* accuracy on the same concept via
     short direct-question generation -> the representation-vs-readout diagnostic.
  5. Writes results.json + figures.

Everything is driven off `output/test_eval/{prompts,questions}/levelN`.

Usage (on the GPU instance):
    python run_probing.py \
        --repo   /opt/factorybench \
        --model  Qwen/Qwen3-4B \
        --out    /opt/factorybench/output/probing/qwen3_4b \
        --n-per-concept 600 \
        --random-init --behavioural

"""
from __future__ import annotations

import os
# Pin BLAS/OpenMP to 1 thread BEFORE numpy/scipy import their backends, so the
# hundreds of tiny per-layer sklearn fits don't each spawn a thread pool and
# thrash. (Env alone is unreliable under systemd; we also use threadpool_limits.)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------
# Concept definitions.  Each maps FactoryBench provenance -> a discrete label.
# A concept is drawn from exactly one level so the prompt at probe time matches
# the prompt at eval time.
# ----------------------------------------------------------------------------

FAULT_NAMES = {
    0: "no fault",
}  # extended at load time from data/labelling if available; id is enough for probing


@dataclass
class Concept:
    key: str
    level: int
    kind: str            # "binary" | "multiclass"
    describe: str        # human summary
    # returns a hashable label or None (=> item excluded from this concept)
    label_fn: object
    behavioural_q: str | None = None   # question appended before the model answers
    behavioural_parse: object | None = None  # text -> label, for the model's committed answer
    # behavioural_choices: list of (label, answer_token_str). Used for the parse-free
    # forced fallback: when the model reasons without committing, we read its choice
    # from the logits over these option tokens after a "the answer is" cue.
    behavioural_choices: object | None = None
    # gen_tokens: chain-of-thought budget for the behavioural elicitation. A 6-way
    # fault-type ask needs far more room to self-terminate than a yes/no.
    gen_tokens: int = 160
    # behavioural_q_neg: negated-wording elicitation used as a yes-bias control
    # (a model that answers "yes" to both "is there a fault?" and "is it healthy?"
    # is just yes-biased, not reading the concept).
    behavioural_q_neg: str | None = None
    # mcq_pool: closed set of (label, description) for a multiple-choice behavioural
    # question whose A-F option order is randomised per item (see _build_fault_item).
    mcq_pool: object | None = None
    min_per_class: int = 20
    top_k: int | None = None           # collapse multiclass to top-k most frequent


def _rel(q: dict) -> dict:
    return ((q.get("provenance") or {}).get("relevance") or {})


def _fault_id(q: dict):
    fid = _rel(q).get("fault_id", None)
    try:
        return int(fid)
    except (TypeError, ValueError):
        return None


def _anomaly_label(q: dict):
    fid = _fault_id(q)
    if fid is None:
        return None
    return 1 if fid > 0 else 0


def _faultfamily_label(q: dict):
    fid = _fault_id(q)
    if fid is None or fid == 0:
        return None
    return fid


def _phase_label(q: dict):
    ph = (q.get("provenance") or {}).get("phase_name", None)
    if ph in (None, "NA", ""):
        return None
    try:
        return int(ph)
    except (TypeError, ValueError):
        return str(ph)


def _dataset_label(q: dict):
    ds = (q.get("provenance") or {}).get("dataset", None)
    if ds in (None, "NA", ""):
        return None
    return str(ds).lower()


def _parse_yesno(txt: str):
    t = txt.strip().lower()
    # take first yes/no token
    m = re.search(r"\b(yes|no)\b", t)
    if not m:
        return None
    return 1 if m.group(1) == "yes" else 0


# Closed-set options for the fault_family behavioural question. These are the six
# most-frequent L4 fault types and match the fault_family probe's classes (names
# from data/labelling/rca/root_causes.json). The A..F letter assignment is
# RANDOMISED per item (see _build_fault_item) so no fault type is pinned to a
# fixed position -- otherwise the model's positional bias (it favoured the
# middle-of-list letters) is confounded with its fault knowledge.
FAULT_CHOICES = [
    (10, "extra weight attached to one robot axis (added axis payload)"),
    (11, "collision with a soft foam object"),
    (15, "unstable mounting platform (loose base / released brakes)"),
    (23, "payload misconfiguration (payload mass left at 0 kg)"),
    (25, "external disturbance continuously pulling/pushing the arm"),
    (30, "collision with a cardboard box"),
]
_FAULT_LETTERS = "ABCDEF"


def _fault_perm(item_id: str, seed: int) -> list[int]:
    """Deterministic per-item permutation of the FAULT_CHOICES indices. Keyed by
    the item id (and the run seed) so the letter order is fixed for a given item
    but decorrelated from the fault type across items."""
    key = (abs(hash((str(item_id), int(seed)))) % (2 ** 32))
    return [int(p) for p in np.random.default_rng(key).permutation(len(FAULT_CHOICES))]


def _build_fault_item(item_id: str, seed: int):
    """Return (question_text, letter2fid, forced_labels, perm) for one item, with
    a fresh A-F ordering of the six fault options.

    forced_labels is the list of fault_ids in letter order (A..F), used to invert
    the model's logit choice; perm is stored in the reply record so scoring can
    be reproduced.
    """
    perm = _fault_perm(item_id, seed)
    lines, letter2fid, forced = [], {}, []
    for pos, src in enumerate(perm):
        letter = _FAULT_LETTERS[pos]
        fid, desc = FAULT_CHOICES[src]
        lines.append(f"{letter}. {desc}")
        letter2fid[letter] = fid
        forced.append(fid)
    q = ("\n\nThis episode contains exactly one injected fault. Which type is it?\n"
         + "\n".join(lines) + "\nAnswer with a single letter (A-F).")
    return q, letter2fid, forced, perm


def _parse_fault_letter(txt: str, letter2fid: dict):
    """Extract the chosen letter from a possibly-verbose answer and map it back to
    a fault_id via this item's letter->fault_id map.

    Qwen3-4B reasons before answering ("The answer is **E**", "\\boxed{E}",
    "Answer: E"), so we look for explicit answer markers first and fall back to
    the first standalone A-F letter. Returns the mapped fault_id, or None.
    """
    U = txt.strip().upper()
    L = r"([A-F])(?![A-Za-z])"  # a standalone letter (not inside a word like ANSWER)
    for pat in (r"\\BOXED\{\s*" + L, r"\*\*\s*" + L + r"\s*\*\*",
                r"ANSWER\s*(?:IS|:|=)\s*\*{0,2}\s*" + L,
                r"FINAL ANSWER[:\s]*\*{0,2}\s*" + L,
                r"OPTION\s*" + L, r"^\s*\*{0,2}\s*" + L):
        m = re.search(pat, U)
        if m and m.group(1) in letter2fid:
            return letter2fid[m.group(1)]
    return None  # no clear letter -> unparsed (counts as a non-answer)


CONCEPTS: list[Concept] = [
    Concept(
        key="anomaly",
        level=4,
        kind="binary",
        describe="Episode contains an injected fault (fault_id>0) vs healthy (0).",
        label_fn=_anomaly_label,
        behavioural_q=(
            "\n\nBased only on the sensor stream above, is there anomalous "
            "behaviour in this episode? Answer with exactly one word: yes or no."
        ),
        behavioural_parse=_parse_yesno,
        behavioural_choices=[(0, "no"), (1, "yes")],
        behavioural_q_neg=(
            "\n\nBased only on the sensor stream above, is this episode healthy "
            "(i.e. no anomalous behaviour)? Answer with exactly one word: yes or no."
        ),
        gen_tokens=160,
        min_per_class=30,
    ),
    Concept(
        key="fault_family",
        level=4,
        kind="multiclass",
        describe="Which injected fault type (fault_id) among faulty episodes.",
        label_fn=_faultfamily_label,
        # The question, letter map and forced-choice tokens are built PER ITEM
        # (randomised A-F order); see _build_fault_item / behavioural_accuracy.
        mcq_pool=FAULT_CHOICES,
        gen_tokens=1024,
        top_k=6,
        min_per_class=20,
    ),
    Concept(
        key="task_phase",
        level=1,
        kind="multiclass",
        describe="Task-phase index at the probed sub-series (phase_name).",
        label_fn=_phase_label,
        top_k=6,
        min_per_class=20,
    ),
    Concept(
        key="source_dataset",
        level=4,
        kind="multiclass",
        describe="Source dataset (positive control; should be trivially decodable).",
        label_fn=_dataset_label,
        min_per_class=30,
    ),
]


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------

@dataclass
class Item:
    id: str
    level: int
    episode: str
    prompt: str
    labels: dict = field(default_factory=dict)
    ts_rows: list = field(default_factory=list)


def load_items(repo: Path, level: int, wanted_concepts: list[Concept]) -> list[Item]:
    qdir = repo / "output" / "test_eval" / "questions" / f"level{level}"
    pdir = repo / "output" / "test_eval" / "prompts" / f"level{level}"
    items: list[Item] = []
    for qp in sorted(qdir.glob("*.json")):
        try:
            q = json.load(open(qp, encoding="utf-8"))
        except Exception:
            continue
        qid = q.get("id", qp.stem)
        # faithful prompt file is named level{L}_{id}.json
        pp = pdir / f"level{level}_{qid}.json"
        if not pp.exists():
            continue
        try:
            prompt = json.load(open(pp, encoding="utf-8")).get("prompt", "")
        except Exception:
            continue
        if not prompt:
            continue
        prov = q.get("provenance") or {}
        episode = str(prov.get("episode", qid))
        labels = {}
        for c in wanted_concepts:
            if c.level != level:
                continue
            labels[c.key] = c.label_fn(q)
        ts_rows = (q.get("context") or {}).get("time_series") or []
        items.append(Item(id=qid, level=level, episode=episode,
                          prompt=prompt, labels=labels, ts_rows=ts_rows))
    return items


# ----------------------------------------------------------------------------
# Activation extraction
# ----------------------------------------------------------------------------

def ts_token_span(offset_mapping, prompt: str):
    """Char span of the time-series block -> (tok_start, tok_end) inclusive.

    The block runs from the first '\nt=' row to the 'Question:' marker. Falls
    back to the whole prompt if markers are absent.
    """
    m0 = prompt.find("\nt=")
    if m0 < 0:
        m0 = prompt.find("t=")
    qmark = prompt.find("Question:")
    if m0 < 0:
        return 0, len(offset_mapping) - 1
    c_start = m0
    c_end = qmark if qmark > m0 else len(prompt)
    tok_start, tok_end = None, None
    for i, (a, b) in enumerate(offset_mapping):
        if a == b:  # special token
            continue
        if tok_start is None and b > c_start:
            tok_start = i
        if a < c_end:
            tok_end = i
    if tok_start is None:
        tok_start = 0
    if tok_end is None or tok_end < tok_start:
        tok_end = len(offset_mapping) - 1
    return tok_start, tok_end


def _patch_sdpa_expand_gqa():
    """Make Qwen3's grouped-query attention run on the Turing (T4) mem-efficient
    SDPA kernel.

    Qwen3 has 32 query heads but 8 KV heads and relies on SDPA's `enable_gqa`
    to broadcast. On Turing, the fused flash/mem-efficient kernels REFUSE unequal
    head counts ("both fused kernels require ... the same num_heads"), so SDPA
    falls back to the MATH kernel, which materialises a seq x seq score matrix
    (>16 GB at 16k tokens) and OOMs. We wrap F.scaled_dot_product_attention to
    repeat_interleave K/V up to the query head count and drop enable_gqa, so the
    O(seq) mem-efficient kernel (supported on sm_70+) can run the long prompts.
    Numerically identical to GQA broadcasting.
    """
    import torch
    import torch.nn.functional as F
    if getattr(F, "_gqa_expand_patched", False):
        return
    orig = F.scaled_dot_product_attention

    def wrapped(query, key, value, attn_mask=None, dropout_p=0.0,
                is_causal=False, scale=None, enable_gqa=False, **kw):
        if key.shape[-3] != query.shape[-3] and query.shape[-3] % key.shape[-3] == 0:
            n_rep = query.shape[-3] // key.shape[-3]
            key = key.repeat_interleave(n_rep, dim=-3)
            value = value.repeat_interleave(n_rep, dim=-3)
        return orig(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                    is_causal=is_causal, scale=scale)

    F.scaled_dot_product_attention = wrapped
    F._gqa_expand_patched = True
    print("  [gqa-expand] patched SDPA to repeat KV heads (T4 mem-efficient path)", flush=True)


def _mem_efficient_sdpa():
    """Context manager forcing PyTorch's memory-efficient attention backend.

    FactoryBench prompts reach ~16k tokens; the default SDPA math backend
    materialises a seq x seq score matrix (>16 GB) and OOMs a T4. The
    memory-efficient (cutlass) backend is O(seq) and runs on Turing GPUs.
    """
    import contextlib
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend
        # Prefer flash/efficient (O(seq)); keep math only as last resort. Once
        # the causal mask is None (see _force_is_causal), the dispatcher picks
        # the efficient kernel via the is_causal fast path.
        return sdpa_kernel([SDPBackend.FLASH_ATTENTION,
                            SDPBackend.EFFICIENT_ATTENTION,
                            SDPBackend.MATH])
    except Exception:
        return contextlib.nullcontext()


def _force_is_causal(model):
    """Make the model emit a None attention mask so SDPA uses its native
    is_causal path (mask=None => O(seq) efficient/flash kernel instead of the
    seq x seq score matrix). Correct for our single, un-padded, causal prompts.
    Patches whichever masking hook the installed transformers version uses.
    """
    import sys
    base = getattr(model, "model", model)
    patched = []
    if hasattr(base, "_update_causal_mask"):
        base._update_causal_mask = (lambda *a, **k: None).__get__(base, type(base))
        patched.append("_update_causal_mask")
    # v5: the model imports create_causal_mask INTO ITS OWN module namespace, so
    # we must patch the name where the model actually looks it up.
    mod = sys.modules.get(type(base).__module__)
    for name in ("create_causal_mask", "create_sliding_window_causal_mask"):
        if mod is not None and hasattr(mod, name):
            setattr(mod, name, lambda *a, **k: None)
            patched.append(f"{type(base).__module__}.{name}")
    try:
        import transformers.masking_utils as mu
        for name in ("create_causal_mask", "create_sliding_window_causal_mask"):
            if hasattr(mu, name):
                setattr(mu, name, lambda *a, **k: None)
    except Exception:
        pass
    print(f"  [is_causal] patched: {patched}", flush=True)


def verify_causal(model, tok, device):
    """Model-agnostic guard that attention is still CAUSAL after _force_is_causal.

    Under causal attention the hidden state / logits at position k depend only on
    tokens <= k, so a full forward and a forward truncated to the first k+1 tokens
    must agree at position k. Under (accidentally) bidirectional attention they
    diverge. This catches the failure mode where mask=None means "no masking"
    instead of "use the is_causal fast path". Returns (ok, max_abs_diff).
    """
    import torch
    text = ("The robot arm moved to position twelve and then paused before "
            "lifting the payload off the table surface slowly.")
    enc = tok(text, return_tensors="pt")
    ids = enc["input_ids"].to(device)
    k = min(8, ids.shape[1] - 2)
    with torch.no_grad(), _mem_efficient_sdpa():
        full = model(input_ids=ids, use_cache=False).logits[0, k].float()
        trunc = model(input_ids=ids[:, :k + 1], use_cache=False).logits[0, k].float()
    diff = float((full - trunc).abs().max())
    # Compare predictions, not raw logits: in bf16 the two forwards tile
    # differently and the max abs logit gap is ~0.1 even when perfectly causal.
    # Bidirectional attention, by contrast, shifts the argmax/top-k. So the
    # robust causal signal is top-k token agreement, with `diff` kept for logs.
    argmax_ok = bool(full.argmax() == trunc.argmax())
    top5_ok = bool((full.topk(5).indices == trunc.topk(5).indices).all())
    ok = argmax_ok and top5_ok
    return ok, diff


def _find_decoder_layers(model):
    """Locate the embedding module and the list of decoder layers (Qwen/LLaMA style)."""
    base = getattr(model, "model", model)
    layers = base.layers
    embed = base.embed_tokens
    return embed, layers


def extract_activations(model, tok, items, device, max_length, log_every=25):
    """Return dict: readout -> np.ndarray [n_items, n_layers+1, hidden].

    Uses forward hooks that offload each layer's read-outs to CPU *during* the
    forward pass, so the GPU never holds all layers at once. This keeps peak
    GPU memory ~= weights + one layer, which fits a 16 GB T4 even for the
    ~16k-token FactoryBench prompts.
    """
    import torch
    n = len(items)
    embed, layers = _find_decoder_layers(model)
    n_layers = len(layers) + 1  # + embedding read-out (index 0)

    # per-item scratch, filled by hooks
    cur = {"ts0": 0, "ts1": 0, "last": None, "mean": None, "last_idx": 0}

    def _grab(idx):
        def hook(_module, _inp, out):
            h = out[0] if isinstance(out, tuple) else out  # (1, seq, hidden)
            h = h[0]
            cur["last"][idx] = h[cur["last_idx"]].float().cpu().numpy()
            cur["mean"][idx] = h[cur["ts0"]:cur["ts1"] + 1].float().mean(0).cpu().numpy()
        return hook

    handles = [embed.register_forward_hook(_grab(0))]
    for li, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(_grab(li + 1)))

    last_all = np.zeros((n, n_layers, model.config.hidden_size), dtype=np.float32)
    mean_all = np.zeros((n, n_layers, model.config.hidden_size), dtype=np.float32)
    n_truncated = 0
    t0 = time.time()
    try:
        with torch.no_grad():
            for i, it in enumerate(items):
                enc = tok(it.prompt, return_tensors="pt", truncation=True,
                          max_length=max_length, return_offsets_mapping=True)
                offsets = enc.pop("offset_mapping")[0].tolist()
                input_ids = enc["input_ids"].to(device)
                seq_len = input_ids.shape[1]
                # Truncation guard (fix 9): the offsets index the full prompt but a
                # left-truncated sequence may have dropped the '\nt=' marker, which
                # would silently collapse the mean-pool span onto nearly the whole
                # retained prompt. Detect truncation and invalidate the mean-pool
                # read-out for such items (the last-token read-out is unaffected,
                # since the prompt END is always retained under left truncation).
                full_len = len(tok(it.prompt, add_special_tokens=True,
                                   truncation=False).input_ids)
                truncated = full_len > seq_len
                ts0, ts1 = ts_token_span(offsets[:seq_len], it.prompt)
                cur["ts0"], cur["ts1"] = ts0, min(ts1, seq_len - 1)
                cur["last_idx"] = seq_len - 1
                cur["last"] = last_all[i]
                cur["mean"] = mean_all[i]
                # force the O(seq) mem-efficient attention backend (see helper)
                with _mem_efficient_sdpa():
                    model(input_ids=input_ids, use_cache=False)
                if truncated:
                    mean_all[i, :, :] = np.nan  # mean-pool span unreliable
                    n_truncated += 1
                del input_ids
                if device == "cuda" and (i + 1) % 16 == 0:
                    torch.cuda.empty_cache()
                if (i + 1) % log_every == 0:
                    rate = (i + 1) / (time.time() - t0)
                    eta = (n - i - 1) / rate / 60
                    print(f"    activations {i+1}/{n}  ({rate:.2f} it/s, ETA {eta:.1f} min)",
                          flush=True)
    finally:
        for h in handles:
            h.remove()
    if n_truncated:
        print(f"    [truncation] {n_truncated}/{n} items exceeded max_length={max_length}; "
              f"mean-pool read-out invalidated for those (last-token kept)", flush=True)
    return {"last": last_all, "mean": mean_all}


def raw_input_features(items) -> np.ndarray:
    """Per-channel signal statistics (mean/std/min/max/last) as a raw baseline."""
    def parse_row(s: str) -> dict:
        s = s.strip()
        if s.startswith("t=") and ":" in s:
            s = s.split(":", 1)[1]
        out = {}
        for kv in s.split(","):
            if "=" in kv:
                k, v = kv.split("=", 1)
                try:
                    out[k.strip()] = float(v.strip())
                except ValueError:
                    pass
        return out
    # global channel vocabulary
    vocab = {}
    parsed_items = []
    for it in items:
        rows = [parse_row(r) for r in it.ts_rows]
        parsed_items.append(rows)
        for r in rows:
            for k in r:
                vocab.setdefault(k, len(vocab))
    keys = list(vocab)
    feats = []
    for rows in parsed_items:
        vec = []
        for k in keys:
            col = np.array([r.get(k, np.nan) for r in rows], dtype=np.float64)
            col = col[~np.isnan(col)]
            if col.size == 0:
                vec += [0.0, 0.0, 0.0, 0.0, 0.0]
            else:
                vec += [col.mean(), col.std(), col.min(), col.max(), col[-1]]
        feats.append(vec)
    X = np.array(feats, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


# ----------------------------------------------------------------------------
# Probing
# ----------------------------------------------------------------------------

def _select_items_for_concept(items, concept: Concept, n_max, rng):
    labelled = [(i, it.labels.get(concept.key)) for i, it in enumerate(items)]
    labelled = [(i, y) for i, y in labelled if y is not None]
    if not labelled:
        return [], []
    from collections import Counter
    cnt = Counter(y for _, y in labelled)
    if concept.top_k:
        keep = {y for y, _ in cnt.most_common(concept.top_k)}
        labelled = [(i, y) for i, y in labelled if y in keep]
        cnt = Counter(y for _, y in labelled)
    keep = {y for y, c in cnt.items() if c >= concept.min_per_class}
    labelled = [(i, y) for i, y in labelled if y in keep]
    if len(set(y for _, y in labelled)) < 2:
        return [], []
    # balance-ish cap: cap total at n_max preserving class ratio
    idx = [i for i, _ in labelled]
    ys = [y for _, y in labelled]
    if len(idx) > n_max:
        sel = rng.choice(len(idx), size=n_max, replace=False)
        idx = [idx[s] for s in sel]
        ys = [ys[s] for s in sel]
    return idx, ys


def _make_reducer(Xtr, seed, k=256):
    """Fit StandardScaler + PCA(k) on Xtr ONLY; return a transform closure.

    With only a few hundred training samples a full 2560-dim probe is
    underdetermined; projecting onto the top-k PCs is faster and removes
    null-space overfitting. Returning a fitted transform (rather than the
    reduced arrays) lets callers apply the SAME train-fitted basis to any
    held-out matrix, which is what keeps PCA out of the CV validation folds.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    sc = StandardScaler().fit(Xtr)
    Xs = sc.transform(Xtr)
    k = int(min(k, Xtr.shape[0] - 2, Xtr.shape[1]))
    pca = None
    if k >= 2:
        # randomized solver: robust + fast, avoids the LAPACK gesdd segfault
        # seen with the full solver on this instance's BLAS build.
        pca = PCA(n_components=k, svd_solver="randomized",
                  random_state=seed).fit(Xs)

    def transform(X):
        Xs = sc.transform(X)
        return pca.transform(Xs) if pca is not None else Xs

    return transform


def _lr_cv(Xtr, ytr, Xte, yte, C_grid, groups_tr, seed, want_pred=False):
    """LogisticRegression with grouped-CV C selection, on RAW (un-reduced) X.

    The StandardScaler+PCA reducer is fit *inside* each CV fold on that fold's
    train rows only, so no validation fold's features are ever shaped by a PCA
    basis that included them (previously the reducer was fit once on the whole
    train set before CV, leaking val-fold variance into the basis and inflating
    both the chosen C and the CV-selected layer). After C is picked, the reducer
    is refit on the full train set and applied to the test set for scoring.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    ytr = np.asarray(ytr)
    best_C, best_cv = C_grid[0], -1.0
    n_groups = len(set(groups_tr))
    if n_groups >= 3 and len(C_grid) > 1:
        gkf = GroupKFold(n_splits=min(3, n_groups))
        cv_scores = {C: [] for C in C_grid}
        for tr, va in gkf.split(Xtr, ytr, groups_tr):
            red = _make_reducer(Xtr[tr], seed)          # fit on fold-train ONLY
            Xtr_f, Xva_f = red(Xtr[tr]), red(Xtr[va])
            for C in C_grid:
                clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
                clf.fit(Xtr_f, ytr[tr])
                cv_scores[C].append(clf.score(Xva_f, ytr[va]))
        best_C, best_cv = max(((C, float(np.mean(s))) for C, s in cv_scores.items()),
                              key=lambda z: z[1])
    # Refit the reducer on the FULL train set, then transform test and score.
    red = _make_reducer(Xtr, seed)
    Xtr_r, Xte_r = red(Xtr), red(Xte)
    clf = LogisticRegression(C=best_C, max_iter=2000, class_weight="balanced")
    clf.fit(Xtr_r, ytr)
    acc = float(clf.score(Xte_r, yte))
    if want_pred:
        return acc, best_C, clf.predict(Xte_r), float(max(best_cv, 0.0))
    return acc, best_C


def _probe_one_layer(L, X, ys, tr, te, groups_tr, groups_te, C_grid, do_mlp, seed):
    """All probes for a single layer. X is RAW; reduction happens inside _lr_cv
    (per CV fold) and once more on the full train for test scoring."""
    from sklearn.neural_network import MLPClassifier
    rng = np.random.default_rng(seed + L)
    ys = np.asarray(ys)
    ytr, yte = ys[tr], ys[te]
    Xtr, Xte = X[tr], X[te]
    lin_acc, bestC, ypred, cv_acc = _lr_cv(Xtr, ytr, Xte, yte, C_grid, groups_tr,
                                           seed, want_pred=True)
    ci = _bootstrap_ci(yte, ypred, groups_te, n_boot=500, seed=seed)
    # selectivity control: identical probe on shuffled train labels. Keep its
    # predictions so the selectivity gap gets an episode-paired bootstrap CI.
    ctrl_acc, _, ctrl_pred, _ = _lr_cv(Xtr, rng.permutation(ytr), Xte, yte,
                                       [bestC], groups_tr, seed, want_pred=True)
    sel_ci = _bootstrap_diff_ci(yte, ypred, ctrl_pred, groups_te, n_boot=500, seed=seed)
    # cv_acc is the grouped-CV score on TRAIN; use it (not the test acc) to pick
    # the best layer, so layer selection never peeks at the test set.
    entry = {"layer": int(L), "linear_acc": lin_acc, "cv_acc": cv_acc,
             "linear_ci95": list(ci),
             "control_acc": ctrl_acc, "selectivity": lin_acc - ctrl_acc,
             "selectivity_ci95": list(sel_ci)}
    if do_mlp:
        red = _make_reducer(Xtr, seed)          # train-fit reducer for the MLP too
        Xtr_r, Xte_r = red(Xtr), red(Xte)
        mlp = MLPClassifier(hidden_layer_sizes=(128,), activation="relu",
                            alpha=1e-3, max_iter=300, early_stopping=True,
                            n_iter_no_change=10, random_state=seed)
        mlp.fit(Xtr_r, ytr)
        entry["mlp_acc"] = float(mlp.score(Xte_r, yte))
        entry["linearity_gap"] = entry["mlp_acc"] - lin_acc
    return entry


def _episode_boot_indices(groups_te, n_boot, seed):
    """Yield row-index arrays for `n_boot` episode-level resamples: draw episodes
    with replacement and pool every item belonging to each drawn episode."""
    rng = np.random.default_rng(seed)
    groups_te = np.asarray(groups_te)
    uniq = np.unique(groups_te)
    members = {g: np.where(groups_te == g)[0] for g in uniq}
    G = len(uniq)
    for _ in range(n_boot):
        pick = uniq[rng.integers(0, G, G)]
        yield np.concatenate([members[g] for g in pick])


def _bootstrap_ci(yte, ypred, groups_te, n_boot=1000, seed=0):
    """Episode-grouped percentile bootstrap CI for accuracy.

    Items sharing an episode_id are correlated (avg ~3.3 items/episode on L1),
    so resampling items i.i.d. pretends independence and shrinks the interval by
    ~sqrt(items-per-episode). We resample whole EPISODES with replacement instead.
    """
    yte = np.asarray(yte); ypred = np.asarray(ypred)
    correct = (yte == ypred).astype(np.float64)
    accs = [correct[idx].mean() for idx in _episode_boot_indices(groups_te, n_boot, seed)]
    lo, hi = np.percentile(accs, [2.5, 97.5])
    return float(lo), float(hi)


def _bootstrap_diff_ci(yte, ypred_a, ypred_b, groups_te, n_boot=500, seed=0):
    """Episode-grouped CI for the accuracy DIFFERENCE (probe minus control),
    paired over the same resampled episodes each draw."""
    yte = np.asarray(yte)
    ca = (yte == np.asarray(ypred_a)).astype(np.float64)
    cb = (yte == np.asarray(ypred_b)).astype(np.float64)
    diffs = [ca[idx].mean() - cb[idx].mean()
             for idx in _episode_boot_indices(groups_te, n_boot, seed)]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(lo), float(hi)


def probe_concept(acts, raw_X, items, concept, idx, ys, args, seed):
    """Run the full probe suite for one concept at a given random `seed`.

    Behavioural accuracy is scored separately by the caller (from a single,
    seed-independent generation pass) so the GPU work is not repeated per seed;
    this function returns the episode-disjoint test item IDs so the caller can
    match `a` to `p` on the identical test split for each seed.
    """
    from sklearn.model_selection import GroupShuffleSplit
    ys_report = np.array(ys)  # original labels, for readable class_counts
    # Encode labels as integer codes. String labels (e.g. source_dataset) crash
    # sklearn's MLP early-stopping score path, which calls np.isnan on the string
    # predictions. Codes are equivalent for every probe and fix that.
    _classes, ys = np.unique(ys_report, return_inverse=True)
    groups = np.array([items[i].episode for i in idx])
    # episode-disjoint split (last-token read-out uses the full item set)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    tr, te = next(gss.split(np.zeros(len(idx)), ys, groups))
    C_grid = [1e-3, 1e-2, 1e-1, 1.0]
    from collections import Counter
    cnt = Counter(ys_report.tolist())
    chance = max(cnt.values()) / len(ys)  # majority-class chance (full set)
    cnt_te = Counter(ys_report[te].tolist())
    chance_test = max(cnt_te.values()) / len(te)  # majority chance on the test split

    res = {
        "concept": concept.key, "level": concept.level, "kind": concept.kind,
        "describe": concept.describe, "seed": int(seed),
        "n_items": int(len(idx)), "n_classes": int(len(cnt)),
        "class_counts": {str(k): int(v) for k, v in cnt.items()},
        "chance_majority": float(chance),
        "chance_majority_test": float(chance_test),
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "layers": {},
    }

    try:
        from threadpoolctl import threadpool_limits
    except Exception:
        import contextlib
        threadpool_limits = lambda limits=None: contextlib.nullcontext()
    n_layers = acts["last"].shape[1]
    with threadpool_limits(limits=1):
        for readout in ("last", "mean"):
            A = acts[readout][idx]  # [n, n_layers, hidden]
            # Truncation guard (fix 9): items whose time-series block was cut by
            # left-truncation have an unreliable mean-pool span and are stored as
            # NaN by extract_activations. Drop them from the mean read-out only
            # (the last prompt token is always retained), re-splitting on the
            # valid subset so train/test stay episode-disjoint.
            valid = np.isfinite(A.reshape(A.shape[0], -1)).all(axis=1)
            if valid.all():
                A_ro, ys_ro, groups_ro, tr_ro, te_ro = A, ys, groups, tr, te
            else:
                vidx = np.where(valid)[0]
                A_ro, ys_ro, groups_ro = A[vidx], ys[vidx], groups[vidx]
                gss2 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
                tr_ro, te_ro = next(gss2.split(np.zeros(len(vidx)), ys_ro, groups_ro))
                res.setdefault("readout_n", {})[readout] = int(len(vidx))
            g_tr, g_te = groups_ro[tr_ro], groups_ro[te_ro]
            per_layer = [
                _probe_one_layer(L, np.ascontiguousarray(A_ro[:, L, :]), ys_ro,
                                 tr_ro, te_ro, g_tr, g_te, C_grid, args.mlp, seed)
                for L in range(n_layers)]
            res["layers"][readout] = per_layer

    # raw-numeric-input baseline (single, layer-agnostic), episode-grouped CI.
    Xr = raw_X[idx]
    raw_acc, _, raw_pred, _ = _lr_cv(Xr[tr], ys[tr], Xr[te], ys[te], C_grid,
                                     groups[tr], seed, want_pred=True)
    res["raw_input_acc"] = raw_acc
    res["raw_input_ci95"] = list(_bootstrap_ci(ys[te], raw_pred, groups[te],
                                               n_boot=1000, seed=seed))

    # best linear layer summary. Select the layer by TRAIN grouped-CV accuracy
    # (cv_acc), then report that layer's held-out TEST accuracy. Selecting by
    # test acc over ~37 layers x 2 read-outs would be an optimistic multiple-
    # comparisons peek; cv-based selection avoids it.
    best = {"last": None, "mean": None}
    for ro in ("last", "mean"):
        arr = res["layers"][ro]
        b = max(arr, key=lambda e: e.get("cv_acc", e["linear_acc"]))
        ceil = max(arr, key=lambda e: e["linear_acc"])
        best[ro] = {"layer": b["layer"], "linear_acc": b["linear_acc"],
                    "cv_acc": b.get("cv_acc"), "selectivity": b["selectivity"],
                    "selectivity_ci95": b.get("selectivity_ci95"),
                    "ci95": b["linear_ci95"],
                    # Test-set ceiling: the layer that maximises TEST accuracy.
                    # Kept for reference ONLY (it is an argmax over layers x
                    # read-outs on the test set) and never quoted as a headline.
                    "test_ceiling_layer": ceil["layer"],
                    "test_ceiling_acc": ceil["linear_acc"]}
    res["best_linear"] = best
    res["_split"] = {"train_ids": [items[idx[i]].id for i in tr],
                     "test_ids": [items[idx[i]].id for i in te]}
    return res


def _concept_test_ids(sub, local_idx, ys, seed):
    """The episode-disjoint TEST item IDs for a concept at a given seed. Mirrors
    probe_concept's split exactly, so behavioural generation can be restricted to
    the union of the seeds' test items (the only items `a` is ever scored on)."""
    from sklearn.model_selection import GroupShuffleSplit
    ys_enc = np.unique(np.array(ys), return_inverse=True)[1]
    groups = np.array([sub[i].episode for i in local_idx])
    _, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
                 .split(np.zeros(len(local_idx)), ys_enc, groups))
    return {sub[local_idx[i]].id for i in te}


# ----------------------------------------------------------------------------
# Behavioural accuracy (representation-vs-readout diagnostic)
# ----------------------------------------------------------------------------

def run_behavioural(model, tok, items, concept, item_idxs, device, max_length,
                    beh_seed=0, negated=False, checkpoint_path=None):
    """Greedy behavioural elicitation over `item_idxs`, returning a PER-ITEM
    prediction record plus global answer-behaviour stats.

    The model is let reason (chain-of-thought). If it commits to a parseable
    answer we take it; if it is still reasoning at the token budget we don't drop
    the item -- we append a "therefore the answer is" cue and read its choice
    parse-free from the logits over the option tokens. Every item yields a
    prediction. For the multiple-choice concept the A-F option order is randomised
    per item (fix: positional-bias control) and the permutation is recorded.

    Predictions are seed-independent (greedy decoding, fixed per-item option
    order via `beh_seed`), so this runs ONCE and the caller scores accuracy on
    each probe seed's test split via `score_behavioural_on_split`.

    negated=True uses the concept's negated-wording question (yes-bias control):
    a "healthy: yes" answer maps to label 0 (no fault).
    """
    import torch
    from collections import Counter
    q_fixed = concept.behavioural_q_neg if negated else concept.behavioural_q
    if concept.mcq_pool is None and not q_fixed:
        return None
    if negated and not q_fixed:
        return None

    def first_tok_ids(s: str):
        ids = set()
        for form in {s, " " + s, s.upper(), " " + s.upper()}:
            t = tok(form, add_special_tokens=False).input_ids
            if t:
                ids.add(t[0])
        return ids

    letter_ids = {ch: first_tok_ids(ch) for ch in _FAULT_LETTERS}
    yesno_cue = "\n\nTherefore, the answer (yes or no) is"
    mcq_cue = "\n\nTherefore, of the options above, the single correct one is option"

    # Resume from an incremental checkpoint if one exists at the same gen budget,
    # so a kill / spot-preemption mid-pass only loses the last few generations.
    records: dict[str, dict] = {}
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        try:
            ck = json.load(open(checkpoint_path))
            if ck.get("gen_tokens") == int(concept.gen_tokens):
                records = ck.get("records", {})
                if records:
                    print(f"    resuming behavioural from checkpoint "
                          f"({len(records)} items done)", flush=True)
        except Exception:
            records = {}

    def _flush():
        if checkpoint_path is None:
            return
        tmp = Path(str(checkpoint_path) + ".tmp")
        json.dump({"gen_tokens": int(concept.gen_tokens), "negated": bool(negated),
                   "records": records}, open(tmp, "w"))
        tmp.replace(checkpoint_path)  # atomic

    since_ckpt = 0
    with torch.no_grad():
        for i in item_idxs:
            it = items[i]
            gold = it.labels.get(concept.key)
            if gold is None:
                continue
            if it.id in records:      # already generated (resume) -> skip
                continue
            if concept.mcq_pool is not None:
                q, letter2fid, forced_fids, perm = _build_fault_item(it.id, beh_seed)
                parse = lambda txt, m=letter2fid: _parse_fault_letter(txt, m)
                forced = [(fid, letter_ids[_FAULT_LETTERS[pos]])
                          for pos, fid in enumerate(forced_fids)]
                cue = mcq_cue
            else:
                q, perm = q_fixed, None
                if negated:
                    # invert raw yes/no: "healthy: yes" -> label 0 (no fault)
                    parse = lambda txt: (None if _parse_yesno(txt) is None
                                         else 1 - _parse_yesno(txt))
                    forced = [(1, first_tok_ids("no")), (0, first_tok_ids("yes"))]
                else:
                    parse = concept.behavioural_parse
                    forced = [(lab, first_tok_ids(ans))
                              for lab, ans in concept.behavioural_choices]
                cue = mcq_cue if len(forced) > 2 else yesno_cue
            base = it.prompt.split("\nQuestion:")[0].rstrip()
            enc = tok(base + q, return_tensors="pt", truncation=True, max_length=max_length)
            input_ids = enc["input_ids"].to(device)
            with _mem_efficient_sdpa():
                gen = model.generate(input_ids=input_ids, max_new_tokens=concept.gen_tokens,
                                     do_sample=False, pad_token_id=tok.eos_token_id)
            txt = tok.decode(gen[0][input_ids.shape[1]:], skip_special_tokens=True)
            pred = parse(txt)
            how = "committed"
            if pred is None:
                cue_ids = tok(cue, add_special_tokens=False,
                              return_tensors="pt")["input_ids"].to(device)
                seq = torch.cat([gen, cue_ids], dim=1)
                if seq.shape[1] > max_length:
                    seq = seq[:, -max_length:]
                with _mem_efficient_sdpa():
                    lg = model(input_ids=seq, use_cache=False).logits[0, -1].float()
                pred = max(((lab, max(float(lg[t]) for t in tset)) for lab, tset in forced),
                           key=lambda z: z[1])[0]
                how = "forced"
            records[it.id] = {"pred": int(pred), "gold": int(gold),
                              "episode": it.episode, "how": how,
                              "perm": perm}
            since_ckpt += 1
            if since_ckpt >= 20:
                _flush(); since_ckpt = 0
    _flush()
    if not records:
        return None
    # Stats derived from the full record set (correct after a resume too).
    n = len(records)
    hows = [r["how"] for r in records.values()]
    pred_dist = Counter(r["pred"] for r in records.values())
    return {"records": records, "n_items": int(n),
            "method": "cot+forced_fallback", "gen_tokens": int(concept.gen_tokens),
            "committed_rate": sum(h == "committed" for h in hows) / n,
            "forced_rate": sum(h == "forced" for h in hows) / n,
            "pred_dist": {str(k): int(v) for k, v in sorted(pred_dist.items())},
            "negated": bool(negated)}


def score_behavioural_on_split(beh, test_ids, seed):
    """Accuracy + episode-grouped CI + prediction spread of precomputed
    behavioural predictions, restricted to a test split (matched to the probe's
    test items for this seed). Returns None if no scored items fall in the split.
    """
    from collections import Counter
    recs = beh["records"]
    rows = [recs[i] for i in test_ids if i in recs]
    if not rows:
        return None
    preds = np.array([r["pred"] for r in rows])
    golds = np.array([r["gold"] for r in rows])
    groups = np.array([r["episode"] for r in rows])
    acc = float((preds == golds).mean())
    ci = _bootstrap_ci(golds, preds, groups, n_boot=1000, seed=seed)
    pd = Counter(preds.tolist())
    out = {"acc": acc, "n": len(rows), "ci95": list(ci), "parse_rate": 1.0,
           "method": beh["method"], "gen_tokens": beh["gen_tokens"],
           "committed_rate": beh["committed_rate"], "forced_rate": beh["forced_rate"],
           "pred_dist": {str(k): int(v) for k, v in sorted(pd.items())}}
    if set(golds.tolist()) <= {0, 1}:  # binary: the always-positive signature
        out["pred_pos_rate"] = float((preds == 1).mean())
    return out


def _aggregate_seeds(per_seed):
    """Combine per-seed probe results. The seed-0 result is kept in full (its
    per-layer arrays feed the figures); a `multiseed` block adds mean +/- std of
    the headline numbers across seeds, which is what we report."""
    base = dict(per_seed[0])

    def collect(getter):
        vals = [getter(r) for r in per_seed]
        vals = [float(v) for v in vals if v is not None]
        if not vals:
            return None
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                "vals": vals, "n_seeds": len(vals)}

    base["multiseed"] = {
        "seeds": [r.get("seed") for r in per_seed],
        # p: best CV-selected linear-probe accuracy, last-token and mean-pool
        "p_last": collect(lambda r: r["best_linear"]["last"]["linear_acc"]),
        "p_mean": collect(lambda r: r["best_linear"]["mean"]["linear_acc"]),
        "cv_layer_last": [r["best_linear"]["last"]["layer"] for r in per_seed],
        "cv_layer_mean": [r["best_linear"]["mean"]["layer"] for r in per_seed],
        # a: behavioural accuracy on each seed's matched test split
        "a": collect(lambda r: (r.get("behavioural") or {}).get("acc")),
        "raw_input_acc": collect(lambda r: r.get("raw_input_acc")),
    }
    return base


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, type=Path)
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n-per-concept", type=int, default=600)
    ap.add_argument("--dtype", default="bfloat16")
    # Qwen3-4B natively supports 32k; 16384 covers all but a handful of the
    # longest FactoryBench prompts (up to 16.6k). Items still over budget get
    # their mean-pool read-out invalidated (see extract_activations).
    ap.add_argument("--max-length", type=int, default=16384)
    ap.add_argument("--seed", type=int, default=0,
                    help="base seed; also the fixed generation seed for the "
                         "per-item behavioural option order")
    ap.add_argument("--seeds", default="",
                    help="comma list of probe seeds to aggregate (mean +/- std "
                         "over splits/PCA/C-selection). Defaults to --seed alone.")
    ap.add_argument("--mlp", action="store_true", help="also fit MLP reference probe")
    ap.add_argument("--randomize-weights", action="store_true",
                    help="load the model with RANDOM weights (separate random-init "
                         "control run; merge via plot_probes --results-random)")
    ap.add_argument("--behavioural", action="store_true", help="measure behavioural a")
    ap.add_argument("--concepts", default="", help="comma list to restrict concepts")
    args = ap.parse_args()
    seeds = [int(s) for s in str(args.seeds).split(",") if s.strip() != ""] or [args.seed]

    import warnings
    warnings.filterwarnings("ignore")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  model={args.model}", flush=True)
    # Enable the O(seq) mem-efficient attention path for Qwen3 GQA on Turing GPUs.
    if device == "cuda":
        _patch_sdpa_expand_gqa()
    args.out.mkdir(parents=True, exist_ok=True)

    wanted = CONCEPTS
    if args.concepts:
        keep = set(args.concepts.split(","))
        wanted = [c for c in CONCEPTS if c.key in keep]
    levels = sorted({c.level for c in wanted})

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    # Truncate from the LEFT so the prompt END (the "Question:" and last token,
    # our read-out position) is always preserved on the few >max_length prompts.
    tok.truncation_side = "left"
    torch_dtype = getattr(torch, args.dtype)
    if args.randomize_weights:
        # Memory-safe random-init control: only ONE model ever in memory.
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch_dtype)
        model.to(device)
        args.behavioural = False  # meaningless for an untrained net
        print("  [randomize-weights] using RANDOM-INIT model as control", flush=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch_dtype, trust_remote_code=True,
            low_cpu_mem_usage=True, device_map=device, output_hidden_states=True,
            attn_implementation="sdpa")
    model.eval()
    # NOTE: do NOT null the causal mask. Nulling create_causal_mask makes SDPA run
    # BIDIRECTIONAL (verified: causal self-test fails), which invalidates the
    # representations. We instead keep HF's proper causal mask and rely on the
    # memory-efficient SDPA backend (_mem_efficient_sdpa) to stay O(seq) on long
    # prompts. verify_causal() below is the guard that this stays correct.
    if not args.randomize_weights:
        ok, diff = verify_causal(model, tok, device)
        print(f"  [causal self-test] {'PASS' if ok else 'FAIL'} (max|Δ|={diff:.2e})", flush=True)
        if not ok:
            print("  ERROR: attention is NOT causal (top-k predictions shift when "
                  "future tokens are removed) - activations would be invalid. "
                  "Aborting.", file=sys.stderr, flush=True)
            return 2

    rng = np.random.default_rng(args.seed)
    all_results = {"model": args.model, "device": device,
                   "n_per_concept": args.n_per_concept, "seeds": seeds,
                   "max_length": args.max_length, "concepts": {}}

    # process level by level so we only load one activation tensor at a time
    for level in levels:
        lvl_concepts = [c for c in wanted if c.level == level]
        print(f"\n=== LEVEL {level}: {[c.key for c in lvl_concepts]} ===", flush=True)
        items = load_items(args.repo, level, lvl_concepts)
        print(f"  loaded {len(items)} items with faithful prompts", flush=True)

        # union of items needed across this level's concepts (item SELECTION is
        # fixed by args.seed; only the train/test split varies across probe seeds)
        needed = set()
        sel_per_concept = {}
        for c in lvl_concepts:
            idx, ys = _select_items_for_concept(items, c, args.n_per_concept, rng)
            sel_per_concept[c.key] = (idx, ys)
            needed.update(idx)
            print(f"  concept {c.key}: {len(idx)} items, classes={sorted(set(ys))}", flush=True)
        needed = sorted(needed)
        if not needed:
            continue
        sub = [items[i] for i in needed]
        remap = {orig: k for k, orig in enumerate(needed)}

        tag = "rand" if args.randomize_weights else "real"
        cache_dir = args.out.parent / "acts_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        # cache key includes max_length so activations from a different truncation
        # regime are never silently reused (fix 9).
        cache = cache_dir / f"level{level}_{tag}_n{len(sub)}_ml{args.max_length}.npz"
        ids_now = [it.id for it in sub]
        acts = None
        if cache.exists():
            z = np.load(cache, allow_pickle=True)
            if list(z["ids"]) == ids_now:
                acts = {"last": z["last"], "mean": z["mean"]}
                print(f"  loaded cached activations {cache.name}", flush=True)
        if acts is None:
            print(f"  extracting activations for {len(sub)} unique items ...", flush=True)
            acts = extract_activations(model, tok, sub, device, args.max_length)
            np.savez_compressed(cache, ids=np.array(ids_now), last=acts["last"],
                                mean=acts["mean"])
        raw_X = raw_input_features(sub)

        # Behavioural generation runs ONCE per concept over ALL selected items
        # (greedy => seed-independent; the per-item MCQ order is fixed by
        # args.seed). Each probe seed then scores `a` on its own test split.
        beh_cache = {}
        if args.behavioural and not args.randomize_weights:
            for c in lvl_concepts:
                idx, ys = sel_per_concept[c.key]
                if not idx or not (c.behavioural_q or c.mcq_pool):
                    continue
                ridx = [remap[i] for i in idx]
                # Resume: reuse cached behavioural predictions if a prior run
                # already generated them at this gen budget (spot-preemption safe).
                bpath = args.out / f"behavioural_{c.key}.json"
                if bpath.exists():
                    try:
                        cached = json.load(open(bpath))
                        if (cached.get("main") or {}).get("gen_tokens") == c.gen_tokens:
                            beh_cache[c.key] = cached
                            print(f"  loaded cached behavioural {bpath.name} "
                                  f"({len((cached['main'] or {}).get('records', {}))} items)",
                                  flush=True)
                            continue
                    except Exception:
                        pass
                # Behavioural a is measured on a SINGLE seed's test split (the
                # first probe seed). a is deterministic greedy generation, so it
                # is essentially seed-invariant -- only the test membership would
                # change -- and generating for one split (rather than the union
                # of all probe seeds') is the dominant cost saver. Probe p still
                # runs every seed; only a is single-seed.
                beh_seed = seeds[0]
                beh_split = _concept_test_ids(sub, ridx, ys, beh_seed)
                gen_local = [l for l in ridx if sub[l].id in beh_split]
                print(f"  behavioural generation: {c.key} over {len(gen_local)} test-split "
                      f"items (of {len(ridx)} selected; gen_tokens={c.gen_tokens}) ...", flush=True)
                ck_main = args.out / f"behavioural_{c.key}_main.ckpt.json"
                ck_neg = args.out / f"behavioural_{c.key}_neg.ckpt.json"
                entry = {"main": run_behavioural(model, tok, sub, c, gen_local, device,
                                                 args.max_length, beh_seed=args.seed,
                                                 checkpoint_path=ck_main)}
                if c.behavioural_q_neg:
                    print(f"  negated-wording yes-bias control: {c.key} ...", flush=True)
                    entry["neg"] = run_behavioural(model, tok, sub, c, gen_local, device,
                                                   args.max_length, beh_seed=args.seed,
                                                   negated=True, checkpoint_path=ck_neg)
                beh_cache[c.key] = entry
                with open(args.out / f"behavioural_{c.key}.json", "w") as f:
                    json.dump(entry, f, indent=2)
                for ck in (ck_main, ck_neg):  # concept done -> drop checkpoints
                    try:
                        ck.unlink()
                    except OSError:
                        pass
                if entry["main"]:
                    print(f"    committed_rate={entry['main']['committed_rate']:.2f} "
                          f"forced_rate={entry['main']['forced_rate']:.2f} "
                          f"pred_dist={entry['main']['pred_dist']}", flush=True)

        for c in lvl_concepts:
            idx, ys = sel_per_concept[c.key]
            if not idx:
                continue
            ridx = [remap[i] for i in idx]
            print(f"  probing concept: {c.key}  (seeds={seeds})", flush=True)
            per_seed = []
            for sd in seeds:
                try:
                    r = probe_concept(acts, raw_X, sub, c, ridx, ys, args, sd)
                    bentry = beh_cache.get(c.key)
                    # a is measured on the first seed's split only (see above),
                    # so attach it to that seed's probe result; other seeds carry
                    # p only. _aggregate_seeds then reports a as that single value.
                    if sd == seeds[0] and bentry and bentry.get("main"):
                        b = score_behavioural_on_split(bentry["main"],
                                                       r["_split"]["test_ids"], sd)
                        if b is not None and bentry.get("neg"):
                            bn = score_behavioural_on_split(bentry["neg"],
                                                            r["_split"]["test_ids"], sd)
                            if bn is not None:
                                # neg pred==0 means the model answered "yes, healthy"
                                b["negated_control"] = {
                                    "acc": bn["acc"], "n": bn["n"], "ci95": bn["ci95"],
                                    "healthy_yes_rate": bn["pred_dist"].get("0", 0) / max(1, bn["n"]),
                                    "pred_dist": bn["pred_dist"]}
                        r["behavioural"] = b
                    r.pop("_split", None)
                    per_seed.append(r)
                except Exception as e:
                    import traceback
                    print(f"  [WARN] concept {c.key} seed {sd} failed: {e}", flush=True)
                    traceback.print_exc()
            if not per_seed:
                all_results["concepts"][c.key] = {"error": "all seeds failed",
                                                  "concept": c.key}
            else:
                all_results["concepts"][c.key] = _aggregate_seeds(per_seed)
            with open(args.out / "results.json", "w") as f:
                json.dump(all_results, f, indent=2)

        # free activations before next level
        del acts, raw_X

    with open(args.out / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nwrote {args.out/'results.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
