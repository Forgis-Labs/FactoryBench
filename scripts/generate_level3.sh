#!/usr/bin/env bash
# FactoryBench, Generate Level 3 (Counterfactual Reasoning) Q&A pairs.
#
# Usage:
#   bash scripts/generate_level3.sh               # default: 500 questions
#   bash scripts/generate_level3.sh 2000           # custom count
#   bash scripts/generate_level3.sh 2000 --seed 0  # custom count + seed

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

N="${1:-500}"
shift 2>/dev/null || true

echo "=== Level 3: Counterfactual Reasoning ==="
echo "Generating $N questions..."

python -m src.question_generation.level3.level3 \
    --datasets-dir data \
    --output output/questions/level3 \
    -n "$N" \
    --seed 42 \
    "$@" \
    -v

echo ""
echo "Done. Output: output/questions/level3/"
echo "Files: $(ls output/questions/level3/level3_*.json 2>/dev/null | wc -l) questions generated"
