#!/bin/bash

# =============================================================================
# Scaling Law Experiment Script
# =============================================================================
#
# Trains models of different depths to study scaling behavior.
# Each model uses Chinchilla-optimal data:param ratio (20:1).
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
WANDB_RUN=${WANDB_RUN:-dummy}

# Auto-detect GPUs
NPROC_PER_NODE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "============================================"
echo "Scaling Law Experiment"
echo "============================================"
echo "GPUs detected: $NPROC_PER_NODE"
echo "WANDB_RUN: $WANDB_RUN"
echo "NANOCHAT_BASE_DIR: $NANOCHAT_BASE_DIR"
echo "============================================"

# -----------------------------------------------------------------------------
# Train models at different scales

DEPTHS=(4 6 8 10 12 14 16)

for DEPTH in "${DEPTHS[@]}"; do
    echo ""
    echo ">>> Training d${DEPTH} model..."
    torchrun --standalone --nproc_per_node=$NPROC_PER_NODE \
        -m scripts.base_train -- \
        --depth=$DEPTH \
        --run=$WANDB_RUN
done

echo ""
echo "============================================"
echo "Scaling experiment complete!"
echo "Logs: $NANOCHAT_BASE_DIR/training_logs/"
echo "============================================"
