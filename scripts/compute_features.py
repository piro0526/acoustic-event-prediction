#!/usr/bin/env python3
"""Main script for computing features and labels on PodcastFillers dataset.

Usage:
    # Single GPU (default shift: 1 frame)
    python compute_features.py --dataset_root data/PodcastFillers --output_root output/laughter/features

    # Multi-GPU (8 GPUs) with custom shift value
    torchrun --nproc_per_node=8 compute_features.py \
        --dataset_root data/PodcastFillers \
        --output_root output/laughter/features \
        --shift_frames 5
"""

import argparse
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from podcast_processing.distributed_orchestrator import DistributedOrchestrator


def main():
    parser = argparse.ArgumentParser(
        description='Compute features and labels for PodcastFillers episodes'
    )
    parser.add_argument(
        '--dataset_root',
        type=str,
        default='data/PodcastFillers',
        help='Root directory of PodcastFillers dataset'
    )
    parser.add_argument(
        '--output_root',
        type=str,
        default='output/laughter/features',
        help='Root directory for output files'
    )
    parser.add_argument(
        '--hf_repo',
        type=str,
        default='kyutai/moshiko-pytorch-bf16',
        help='HuggingFace repository for models'
    )
    parser.add_argument(
        '--shift_frames',
        type=int,
        default=1,
        help='Prediction shift value in frames (e.g., 1, 5, 10, 25)'
    )

    args = parser.parse_args()

    # Create orchestrator
    orchestrator = DistributedOrchestrator(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        hf_repo=args.hf_repo,
        shift_frames=args.shift_frames
    )

    # Run processing
    orchestrator.run()


if __name__ == '__main__':
    main()
