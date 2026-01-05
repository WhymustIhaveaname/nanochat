#!/bin/bash

# =============================================================================
# Scaling Law Experiment Script
# =============================================================================
#
# Trains models of different depths and data:param ratios to study scaling.
# Iterates over DEPTHS × RATIOS combinations.
#
# Model sizes (n_embd = depth * 64):
#   d4:  ~2.8M params     d10: ~28M params      d16: ~115M params
#   d6:  ~7.9M params     d12: ~50M params
#   d8:  ~15.7M params    d14: ~80M params
#
# Usage:
#   bash scaling.sh                    # Run all depths
#   WANDB_RUN=scaling bash scaling.sh  # With wandb logging
#
# Training logs saved to: $NANOCHAT_BASE_DIR/training_logs/
# =============================================================================

set -e  # Exit on error

# -----------------------------------------------------------------------------
# Environment setup

source .venv/bin/activate
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="/mnt/data/youran/repos/nanochat/workspace"

# wandb setup (use "dummy" to skip logging)
WANDB_RUN=${WANDB_RUN:-adamw}

# Experiment grid
DEPTHS=(4 6 8 10 12 14 16)
RATIOS=(5 10)

# Auto-detect GPUs
NPROC_PER_NODE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "============================================"
echo "Scaling Law Experiment"
echo "============================================"
echo "GPUs detected: $NPROC_PER_NODE"
echo "WANDB_RUN: $WANDB_RUN"
echo "DEPTHS: ${DEPTHS[*]}"
echo "RATIOS: ${RATIOS[*]}"
echo "NANOCHAT_BASE_DIR: $NANOCHAT_BASE_DIR"
echo "============================================"

# -----------------------------------------------------------------------------
# Train models at different scales and data:param ratios

for RATIO in "${RATIOS[@]}"; do
    for DEPTH in "${DEPTHS[@]}"; do
        echo ""
        echo ">>> Training d${DEPTH} model with ratio=${RATIO}..."
        torchrun --standalone --nproc_per_node=$NPROC_PER_NODE \
            -m scripts.base_train -- \
            --depth=$DEPTH \
            --run=$WANDB_RUN \
            --target_param_data_ratio=$RATIO \
            --use_muon=False \
            --lr_schedule=cosine \
            --warmup_ratio=0.1 \
            --warmdown_ratio=0.9 \
            --final_lr_frac=0.1
    done
done

echo ""
echo "============================================"
echo "Scaling experiment complete!"
echo "Logs: $NANOCHAT_BASE_DIR/training_logs/"
echo "============================================"
