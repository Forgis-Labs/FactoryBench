#!/usr/bin/env bash
# FactoryBench, Full data installation and normalization pipeline.
#
# Downloads open datasets, normalizes all sources to UR3e schema JSON,
# and injects synthetic events.
#
# Usage:
#   bash scripts/install_data.sh                  # full pipeline
#   bash scripts/install_data.sh --skip-download   # skip downloads, only normalize
#   bash scripts/install_data.sh --max-timestamps 5000  # limit rows per experiment
#
# Prerequisites:
#   pip install pandas h5py requests tqdm pyarrow scipy openpyxl huggingface-hub
#   Set HF_TOKEN in .env or environment for FactoryWave download

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

SKIP_DOWNLOAD=false
MAX_TS_FLAG=""
SEED=42

for arg in "$@"; do
    case "$arg" in
        --skip-download) SKIP_DOWNLOAD=true ;;
        --max-timestamps=*) MAX_TS_FLAG="--max-timestamps ${arg#*=}" ;;
        --max-timestamps) shift; MAX_TS_FLAG="--max-timestamps $1" ;;
    esac
done

echo "============================================"
echo " FactoryBench, Data Installation Pipeline"
echo "============================================"
echo ""

# ------------------------------------------------------------------
# Step 1: Download open datasets
# ------------------------------------------------------------------
if [ "$SKIP_DOWNLOAD" = false ]; then
    # Load .env if present (for HF_TOKEN)
    if [ -f ".env" ]; then
        set -a
        source .env
        set +a
    fi

    echo "[1/7] Downloading FactoryWave dataset from HuggingFace..."
    python -m src.data.data_installation.install_factorywave \
        --out-dir data \
        -v

    echo ""
    echo "[2/7] Downloading AURSAD dataset from Zenodo..."
    python -m src.data.data_installation.install_aursad \
        --out-dir data/open_datasets/aursad \
        $MAX_TS_FLAG \
        --skip-md5

    echo ""
    echo "[3/7] Downloading vorausAD dataset (100 Hz)..."
    python -m src.data.data_installation.install_vaursad \
        --out-dir data/open_datasets/vorausad \
        --variant 100 \
        $MAX_TS_FLAG

    echo ""
    echo "[4/7] Downloading CNC mill dataset from Kaggle..."
    python -m src.data.data_installation.install_cnc \
        --out data/open_datasets/cnc
else
    echo "[1-4/7] Skipping downloads (--skip-download)"
fi

# ------------------------------------------------------------------
# Step 2: Normalize open datasets
# ------------------------------------------------------------------
echo ""
echo "[5/7] Normalizing datasets to UR3e schema..."

# AURSAD (Excel → JSON)
if [ -d "data/open_datasets/aursad" ]; then
    for f in data/open_datasets/aursad/experiment_*.csv; do
        [ -f "$f" ] || continue
        episode_id="$(basename "$f" .csv)"
        python -m src.data.data_normalization.mapped_dataset_normalizer \
            --dataset aursad \
            --input "$f" \
            --output data/normalized_episodes \
            --episode-column sample_nr \
            -v 2>&1 | tail -1
    done
    echo "  ✓ AURSAD normalized"
fi

# VorausAD (CSV → JSON via mapped normalizer)
if [ -d "data/open_datasets/vorausad/vorausad" ]; then
    for f in data/open_datasets/vorausad/vorausad/experiment_*.csv; do
        [ -f "$f" ] || continue
        python -m src.data.data_normalization.mapped_dataset_normalizer \
            --dataset vorausad \
            --input "$f" \
            --output data/normalized_episodes \
            -v 2>&1 | tail -1
    done
    echo "  ✓ VorausAD normalized"
fi

# Simulations (HuggingFace download → JSON)
if [ -d "data/open_datasets/simulations" ]; then
    python -m src.data.simulations_normalizer \
        --input data/open_datasets/simulations \
        --output data/normalized_episodes \
        -v
    echo "  ✓ Simulations normalized"
fi

# ------------------------------------------------------------------
# Step 3: Normalize real robot data (if present)
# ------------------------------------------------------------------
echo ""
echo "[6/7] Normalizing real robot data (if available)..."

# UR3 real data
if [ -d "data/raw/ur3_real" ]; then
    python -m src.data.ur3_real_normalizer \
        --input data/raw/ur3_real \
        --output data/normalized_episodes/ur3_real \
        --source-hz 500 --target-hz 50 \
        -v
    echo "  ✓ UR3 real data normalized (500 Hz → 50 Hz)"
else
    echo "  ⊘ No UR3 real data found at data/raw/ur3_real"
fi

# KUKA KR10 real data
if [ -d "data/raw/kuka_real" ]; then
    python -m src.data.kuka_real_normalizer \
        --input data/raw/kuka_real \
        --output data/normalized_episodes/kuka_real \
        --source-hz 250 --target-hz 50 \
        -v
    echo "  ✓ KUKA KR10 real data normalized (250 Hz → 50 Hz)"
else
    echo "  ⊘ No KUKA real data found at data/raw/kuka_real"
fi

# ------------------------------------------------------------------
# Step 4: Inject synthetic events
# ------------------------------------------------------------------
echo ""
echo "[7/7] Injecting synthetic events..."
python -m src.data.data_normalization.inject_events \
    --normalized-dir data/normalized_episodes \
    --events data/labelling/events.json \
    --datasets aursad vorausad \
    --seed "$SEED" \
    -v

echo ""
echo "============================================"
echo " Data installation complete."
echo " Normalized episodes: data/normalized_episodes/"
echo "============================================"
