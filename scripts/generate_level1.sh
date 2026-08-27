#!/usr/bin/env bash
# FactoryBench, Generate Level 1 (State Understanding) Q&A pairs.
#
# Usage:
#   bash scripts/generate_level1.sh               # default: 600 questions
#   bash scripts/generate_level1.sh 2000           # custom count
#   bash scripts/generate_level1.sh 2000 --seed 0  # custom count + seed

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

N="${1:-600}"
shift 2>/dev/null || true

echo "=== Level 1: State Understanding ==="
echo "Generating $N questions..."

python -m src.question_generation.level1.level1 \
    --datasets-dir data \
    --output output/questions/level1 \
    -n "$N" \
    --seed 42 \
    "$@" \
    -v

echo ""
echo "Done. Output: output/questions/level1/"
echo "Files: $(ls output/questions/level1/level1_*.json 2>/dev/null | wc -l) questions generated"
