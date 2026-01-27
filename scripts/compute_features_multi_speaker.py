#!/usr/bin/env python
"""Compute features using multi-speaker processing.

This script processes podcast episodes using original mixed audio and diarization
metadata to create flexible user/system speaker assignments.

Usage:
    # Single GPU (for testing)
    python scripts/compute_features_multi_speaker.py \
        --dataset_root data/PodcastFillers \
        --output_root outputs/features_multi_speaker \
        --min_speaker_share 0.20 \
        --shift_frames 1 \
        --mask_laughter \
        --splits test \
        --limit 1

    # Multi-GPU (production)
    torchrun --nproc_per_node=8 scripts/compute_features_multi_speaker.py \
        --dataset_root data/PodcastFillers \
        --output_root outputs/features_multi_speaker \
        --min_speaker_share 0.20 \
        --shift_frames 1 \
        --mask_laughter
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from podcast_processing.multi_speaker_orchestrator import MultiSpeakerOrchestrator


def main():
    parser = argparse.ArgumentParser(
        description='Compute features using multi-speaker processing',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--dataset_root',
        type=str,
        required=True,
        help='Root directory of PodcastFillers dataset'
    )

    parser.add_argument(
        '--output_root',
        type=str,
        required=True,
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
        help='Prediction shift in frames (default: 1)'
    )

    parser.add_argument(
        '--min_speaker_share',
        type=float,
        default=0.20,
        help='Minimum speaking ratio for user speaker (default: 0.20 = 20%%)'
    )

    parser.add_argument(
        '--mask_laughter',
        action='store_true',
        help='Mask laughter in both user and system audio'
    )

    parser.add_argument(
        '--no_mask_laughter',
        action='store_true',
        help='Do not mask laughter (overrides --mask_laughter)'
    )

    parser.add_argument(
        '--splits',
        type=str,
        nargs='+',
        default=['train', 'validation', 'test'],
        help='Splits to process (default: train validation test)'
    )

    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of episodes to process (for testing)'
    )

    args = parser.parse_args()

    # Determine mask_laughter setting
    mask_laughter = True  # Default to True
    if args.no_mask_laughter:
        mask_laughter = False
    elif args.mask_laughter:
        mask_laughter = True

    # Print configuration
    print("=" * 60)
    print("Multi-Speaker Feature Extraction")
    print("=" * 60)
    print(f"Dataset root: {args.dataset_root}")
    print(f"Output root: {args.output_root}")
    print(f"HF repo: {args.hf_repo}")
    print(f"Shift frames: {args.shift_frames}")
    print(f"Min speaker share: {args.min_speaker_share*100:.0f}%")
    print(f"Mask laughter: {mask_laughter}")
    print(f"Splits: {args.splits}")
    if args.limit:
        print(f"Limit: {args.limit} episodes")
    print("=" * 60)

    # Create and run orchestrator
    orchestrator = MultiSpeakerOrchestrator(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        hf_repo=args.hf_repo,
        shift_frames=args.shift_frames,
        min_speaker_share=args.min_speaker_share,
        mask_laughter=mask_laughter,
        splits=args.splits
    )

    orchestrator.run(limit=args.limit)

    print("Done!")


if __name__ == '__main__':
    main()
