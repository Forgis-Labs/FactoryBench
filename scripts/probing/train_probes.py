"""Train per-layer linear (and optional MLP) probes on dumped LLM activations.

Consumes the Parquet activations produced by `dump_activations.py` and the
per-item concept-label CSV produced by `build_labels.py`, then fits a
logistic-regression probe per (layer, concept) with 5-fold CV. Reports
train/CV/test accuracy in JSON.

Usage:
    python scripts/probing/train_probes.py \\
        --activations output/probing/qwen3_4b/level1_activations.parquet \\
        --labels output/probing/qwen3_4b/level1_labels.csv \\
        --label-col phase_index \\
        --out output/probing/qwen3_4b/level1_phase_probes.json

The mapping from FactoryBench level to concept label is defined by the
concept definitions used by the probing pipeline.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

try:
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold, train_test_split
    from sklearn.preprocessing import StandardScaler
except ImportError:
    print("ERROR: install pandas + scikit-learn.", file=sys.stderr)
    sys.exit(1)


def _load_activations(path: Path) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    """Return meta DataFrame plus {layer_idx: (n, d) float32 matrix}."""
    tbl = pq.read_table(path)
    meta = tbl.select(["id", "level", "template_id", "prompt_len"]).to_pandas()
    layers: dict[int, np.ndarray] = {}
    layer_cols = [c for c in tbl.column_names if c.startswith("layer_")]
    for c in layer_cols:
        ell = int(c.split("_")[1])
        raw = tbl.column(c).to_pylist()  # list[bytes]
        mat = np.stack([np.frombuffer(b, dtype=np.float32) for b in raw], axis=0)
        layers[ell] = mat
    return meta, layers


def _fit_one_layer(X: np.ndarray, y: np.ndarray, k: int = 5) -> dict:
    """CV + held-out test accuracy for one layer's activations."""
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=17, stratify=y)
    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    Xte_s = scaler.transform(Xte)

    cv = KFold(n_splits=k, shuffle=True, random_state=17)
    cv_scores: list[float] = []
    for tr_idx, va_idx in cv.split(Xtr_s):
        clf = LogisticRegression(C=1e-2, max_iter=2000, n_jobs=-1)
        clf.fit(Xtr_s[tr_idx], ytr[tr_idx])
        cv_scores.append(float(clf.score(Xtr_s[va_idx], ytr[va_idx])))

    clf = LogisticRegression(C=1e-2, max_iter=2000, n_jobs=-1)
    clf.fit(Xtr_s, ytr)
    return {
        "train_acc": float(clf.score(Xtr_s, ytr)),
        "cv_acc_mean": float(np.mean(cv_scores)),
        "cv_acc_std": float(np.std(cv_scores)),
        "test_acc": float(clf.score(Xte_s, yte)),
        "n_train": int(len(ytr)),
        "n_test": int(len(yte)),
        "n_classes": int(len(np.unique(y))),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--activations", required=True)
    ap.add_argument("--labels", required=True, help="CSV with 'id' + label column")
    ap.add_argument("--label-col", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    meta, layers = _load_activations(Path(args.activations))
    labels = pd.read_csv(args.labels)
    if "id" not in labels.columns or args.label_col not in labels.columns:
        print(f"ERROR: labels CSV must have 'id' and '{args.label_col}' columns", file=sys.stderr)
        return 1
    merged = meta.merge(labels[["id", args.label_col]], on="id", how="inner")
    y = merged[args.label_col].to_numpy()
    valid_ids = merged["id"].tolist()
    id_to_idx = {i: k for k, i in enumerate(meta["id"].tolist())}
    keep_idx = np.array([id_to_idx[i] for i in valid_ids], dtype=np.int64)

    per_layer_results: dict[int, dict] = {}
    for ell, mat in sorted(layers.items()):
        try:
            X = mat[keep_idx]
            per_layer_results[ell] = _fit_one_layer(X, y)
        except Exception as e:
            per_layer_results[ell] = {"error": str(e)}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "activations_file": args.activations,
                "labels_file": args.labels,
                "label_col": args.label_col,
                "n_items": int(len(valid_ids)),
                "per_layer": per_layer_results,
            },
            fh,
            indent=2,
        )

    best_ell = max(per_layer_results.items(), key=lambda kv: kv[1].get("test_acc", -1))
    print(
        f"best layer for concept '{args.label_col}': "
        f"layer_{best_ell[0]} test_acc={best_ell[1].get('test_acc'):.3f} "
        f"(n_items={len(valid_ids)}, n_classes={best_ell[1].get('n_classes')})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
