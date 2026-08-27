"""CLI helpers for question generation across levels."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Question generation utilities for FactoryBench")
    sub = parser.add_subparsers(dest="cmd")

    # Level 1
    parser_l1 = sub.add_parser("level1", help="Generate Level 1 questions")
    parser_l1.add_argument("--input", type=Path, required=True, help="Normalized episode JSON file")
    parser_l1.add_argument("--output", type=Path, required=True, help="Output questions JSON file")
    parser_l1.add_argument("--n", type=int, default=100)
    parser_l1.add_argument("--min-dt-ms", type=int, default=100)
    parser_l1.add_argument("--max-dt-ms", type=int, default=2000)
    parser_l1.add_argument("--eps-q", type=float, default=1e-3)
    parser_l1.add_argument("--eps-1", type=float, default=None, help="Threshold for Q1 (same position check)")
    parser_l1.add_argument("--eps-2", type=float, default=None, help="Threshold for Q2 (friction check, percent)")
    parser_l1.add_argument("--delta-1", type=int, default=None, help="Time window for Q2 (ms)")
    parser_l1.add_argument("--eps-3", type=float, default=None, help="Threshold for Q4 external force detection")
    parser_l1.add_argument("--seed", type=int, default=None)

    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s")

    if args.cmd == "level1":
        from src.question_generation.level1 import generate_level1_questions

        generate_level1_questions(
            episode_json=args.input,
            out_json=args.output,
            n_questions=args.n,
            min_dt_ms=args.min_dt_ms,
            max_dt_ms=args.max_dt_ms,
            eps_q=args.eps_q,
            seed=args.seed,
            eps_1=args.eps_1,
            eps_2=args.eps_2,
            delta_1=args.delta_1,
            eps_3=args.eps_3,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
