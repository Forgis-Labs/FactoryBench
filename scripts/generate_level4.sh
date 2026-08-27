#!/usr/bin/env bash
# FactoryBench, Generate Level 4 (Troubleshooting & Optimization) Q&A pairs.
#
# Usage:
#   bash scripts/generate_level4.sh               # default: 300 questions
#   bash scripts/generate_level4.sh 1000           # custom count
#   bash scripts/generate_level4.sh 1000 --seed 0  # custom count + seed

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

N="${1:-300}"
shift 2>/dev/null || true

echo "=== Level 4: Troubleshooting & Optimization ==="
echo "Generating $N questions..."

python -m src.question_generation.level4.level4 \
    --datasets-dir data \
    --output output/questions/level4 \
    -n "$N" \
    --seed 42 \
    "$@" \
    -v

echo ""
echo "Done. Output: output/questions/level4/"
echo "Files: $(ls output/questions/level4/level4_*.json 2>/dev/null | wc -l) questions generated"
