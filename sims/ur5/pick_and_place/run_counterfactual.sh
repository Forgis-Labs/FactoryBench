#!/usr/bin/env bash
# run_counterfactual.sh, Run paired baseline + event-injection episodes
# with full sensor logging, using multiple workers.
#
# Runs baselines first, then counterfactuals, to avoid GPU memory contention.
#
# Usage:
#   cd /opt/IsaacSim
#   bash FactoryBench/ur5/pick_and_place/run_counterfactual.sh \
#       --episodes 5000 --workers 4 [--seed 0]

set -euo pipefail

EPISODES=5000
WORKERS=4
SEED=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --episodes)  EPISODES="$2"; shift 2 ;;
        --workers)   WORKERS="$2";  shift 2 ;;
        --seed)      SEED="$2";     shift 2 ;;
        *)           echo "Unknown arg: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_SH="$(cd "$SCRIPT_DIR/../../.." && pwd)/python.sh"
RUN_PY="$SCRIPT_DIR/run.py"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_BASE="$SCRIPT_DIR/logs/counterfactual_${TIMESTAMP}"

PER_WORKER=$(( (EPISODES + WORKERS - 1) / WORKERS ))
SEED_STRIDE=10000

mkdir -p "$LOG_BASE"

echo "=== Counterfactual Run ==="
echo "  Episodes: $EPISODES ($PER_WORKER per worker)"
echo "  Workers:  $WORKERS"
echo "  Seed:     $SEED"
echo "  Output:   $LOG_BASE"
echo ""

# --- Phase 1: Baselines ---
echo "--- Phase 1: Launching $WORKERS baseline workers ---"
PIDS=()
for (( w=0; w<WORKERS; w++ )); do
    W_SEED=$(( SEED + w * SEED_STRIDE ))
    W_LOG="$LOG_BASE/worker_${w}"
    echo "  Worker $w: seed=$W_SEED  episodes=$PER_WORKER"
    "$PYTHON_SH" "$RUN_PY" \
        --episodes "$PER_WORKER" \
        --seed "$W_SEED" \
        --headless \
        --run_type baseline \
        --log_dir "$W_LOG" \
        > "$LOG_BASE/baseline_w${w}.log" 2>&1 &
    PIDS+=($!)
done

echo "Waiting for baselines to finish..."
FAIL=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || FAIL=$((FAIL + 1))
done

if [ "$FAIL" -gt 0 ]; then
    echo "WARNING: $FAIL baseline worker(s) failed."
else
    echo "All baselines finished."
fi
echo ""

# --- Phase 2: Counterfactuals ---
echo "--- Phase 2: Launching $WORKERS counterfactual workers ---"
PIDS=()
for (( w=0; w<WORKERS; w++ )); do
    W_SEED=$(( SEED + w * SEED_STRIDE ))
    W_LOG="$LOG_BASE/worker_${w}"
    echo "  Worker $w: seed=$W_SEED  episodes=$PER_WORKER"
    "$PYTHON_SH" "$RUN_PY" \
        --episodes "$PER_WORKER" \
        --seed "$W_SEED" \
        --events \
        --headless \
        --run_type counterfactual \
        --log_dir "$W_LOG" \
        > "$LOG_BASE/counterfactual_w${w}.log" 2>&1 &
    PIDS+=($!)
done

echo "Waiting for counterfactuals to finish..."
FAIL=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || FAIL=$((FAIL + 1))
done

if [ "$FAIL" -gt 0 ]; then
    echo "WARNING: $FAIL counterfactual worker(s) failed."
else
    echo "All counterfactuals finished."
fi

echo ""
echo "Done. Output: $LOG_BASE/"
for (( w=0; w<WORKERS; w++ )); do
    echo "  worker_${w}/baseline/, steps.csv, episodes.csv"
    echo "  worker_${w}/counterfactual/, steps.csv, episodes.csv"
done
