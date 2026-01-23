#!/usr/bin/env python3
"""Main script for computing transformer outputs on PodcastFillers dataset.

Usage:
    # Single GPU
    python compute_transformer_outs.py --dataset_root data/PodcastFillers --output_root output/transformer_outs

    # Multi-GPU (8 GPUs)
    torchrun --nproc_per_node=8 compute_transformer_outs.py --dataset_root data/PodcastFillers --output_root output/transformer_outs
"""

import argparse
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from podcast_processing.distributed_orchestrator import DistributedOrchestrator


def main():
    parser = argparse.ArgumentParser(
        description='Compute transformer outputs for PodcastFillers episodes'
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
        default='output/transformer_outs',
        help='Root directory for output files'
    )
    parser.add_argument(
        '--hf_repo',
        type=str,
        default='kyutai/moshiko-pytorch-bf16',
        help='HuggingFace repository for models'
    )

    args = parser.parse_args()

    # Create orchestrator
    orchestrator = DistributedOrchestrator(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        hf_repo=args.hf_repo
    )

    # Run processing
    orchestrator.run()


if __name__ == '__main__':
    main()
