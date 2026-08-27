"""
Fix pick_and_place episode metadata in the FactoryWave episode.parquet file.

Corrections applied (task == 'pick_and_place' only):
  - weight 0.3  → position_of_box = [0.056, 0.122, 0.035]   (light)
  - weight 0.5  → weight 0.6, position_of_box = [0.056, 0.122, 0.035]   (reclassify as medium)
  - weight 0.6  → position_of_box = [0.056, 0.122, 0.035]   (medium)
  - weight 1.2  → position_of_box = [0.08, 0.139, 0.044]    (heavy)
  - weight NaN  → weight 1.2,  position_of_box = [0.08, 0.139, 0.044]   (heavy)

Usage:
    python -m src.data.data_installation.fix_factorywave_metadata
    python -m src.data.data_installation.fix_factorywave_metadata --path data/factorywave/data/episode.parquet
"""
from __future__ import annotations

import argparse
import ast
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

LIGHT_MEDIUM_DIMS = [0.056, 0.122, 0.035]
HEAVY_DIMS = [0.08, 0.139, 0.044]


def _correct_metadata(meta: dict) -> dict:
    if meta.get("task") != "pick_and_place":
        return meta

    weight = meta.get("weight_of_box")

    if weight is None:
        meta["weight_of_box"] = 1.2
        meta["position_of_box"] = str(HEAVY_DIMS)
    elif weight == 0.5:
        meta["weight_of_box"] = 0.6
        meta["position_of_box"] = str(LIGHT_MEDIUM_DIMS)
    elif weight in (0.3, 0.6):
        meta["position_of_box"] = str(LIGHT_MEDIUM_DIMS)
    elif weight == 1.2:
        meta["position_of_box"] = str(HEAVY_DIMS)

    return meta


def fix_episode_parquet(path: Path) -> None:
    import pandas as pd

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_parquet(path)

    def update(raw):
        if not isinstance(raw, str):
            return raw
        try:
            meta = ast.literal_eval(raw)
        except (SyntaxError, ValueError):
            return raw
        if not isinstance(meta, dict):
            return raw
        return str(_correct_metadata(meta))

    before = df["episode_metadata"].copy()
    df["episode_metadata"] = df["episode_metadata"].apply(update)
    changed = (before != df["episode_metadata"]).sum()

    df.to_parquet(path, index=False)
    logger.info(f"Updated {changed} rows in {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    repo_root = Path(__file__).resolve().parents[3]
    parser.add_argument(
        "--path",
        type=Path,
        default=repo_root / "data" / "factorywave" / "data" / "episode.parquet",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    fix_episode_parquet(args.path)


if __name__ == "__main__":
    main()
