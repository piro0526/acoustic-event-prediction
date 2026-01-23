#!/usr/bin/env python3
"""Train laughter event prediction model.

This script trains a binary classifier to predict laughter events in podcast
audio from pre-extracted features and labels.

Usage:
    # Single GPU (default shift=1)
    uv run train_laughter_predictor.py \
        --features_dir output/laughter/features \
        --shift_frames 1 \
        --output_dir output/laughter_prediction \
        --batch_size 512 \
        --learning_rate 1e-4 \
        --epochs 50

    # Multi-GPU (e.g., 4 GPUs) with custom shift value
    uv run torchrun --nproc_per_node=4 train_laughter_predictor.py \
        --features_dir output/laughter/features \
        --shift_frames 5 \
        --output_dir output/laughter_prediction \
        --batch_size 512 \
        --learning_rate 2e-4 \
        --epochs 50 \
        --num_workers 1
"""

import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from laughter_prediction.train import main

if __name__ == '__main__':
    main()
