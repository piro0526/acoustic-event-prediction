#!/usr/bin/env python3
"""
Extract features for laughter detection/prediction.

This script extracts features from transformer outputs without creating labels.
Labels can be created separately using create_labels.py.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from podcast_processing.utils import setup_logging

logger = logging.getLogger(__name__)

FRAME_RATE = 12.5  # Hz


def extract_features_from_episode(
    pt_path: Path
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Extract features from a single episode.

    Args:
        pt_path: Path to .pt file containing transformer outputs

    Returns:
        features: Array of shape [2*T, 4096] (2 assignments concatenated)
        metadata: List of metadata dicts for each frame
    """
    # Load transformer output
    data = torch.load(pt_path)
    transformer_out = data['transformer_out']  # [2, T, 4096]
    metadata = data['metadata']

    episode_name = pt_path.stem
    num_frames = transformer_out.shape[1]

    all_features = []
    all_metadata = []

    # Process both assignments
    for assign_idx in range(2):
        assignment = metadata['assignments'][assign_idx]
        user_speaker = assignment['user_id']

        # Extract features (convert bfloat16 to float32)
        features = transformer_out[assign_idx].to(torch.float32).numpy()  # [T, 4096]

        # Create metadata for each frame
        for frame_idx in range(num_frames):
            all_features.append(features[frame_idx])
            all_metadata.append({
                'episode_name': episode_name,
                'user_id': user_speaker,
                'frame_idx': frame_idx,
                'assignment_idx': assign_idx
            })

    features_array = np.array(all_features, dtype=np.float32)

    return features_array, all_metadata


def extract_features_for_split(
    transformer_dir: Path,
    split: str,
    output_dir: Path
) -> None:
    """Extract features for a dataset split.

    Args:
        transformer_dir: Base directory containing transformer_outs/{split}/*.pt
        split: Dataset split ('train', 'validation', or 'test')
        output_dir: Directory to save extracted features
    """
    logger.info(f"Extracting features for {split} split")

    # Setup paths
    pt_dir = transformer_dir / split
    output_split_dir = output_dir / split
    output_split_dir.mkdir(parents=True, exist_ok=True)

    if not pt_dir.exists():
        raise FileNotFoundError(f"Transformer directory not found: {pt_dir}")

    # Find all episodes
    pt_files = sorted(list(pt_dir.glob('*.pt')))
    logger.info(f"Found {len(pt_files)} episodes in {split} split")

    # Process episodes
    all_features = []
    all_metadata = []

    for pt_path in tqdm(pt_files, desc=f"Processing {split} episodes"):
        try:
            features, metadata = extract_features_from_episode(pt_path)

            all_features.append(features)
            all_metadata.extend(metadata)

            logger.info(f"  {pt_path.stem}: {len(features)} frames")

        except Exception as e:
            logger.error(f"Error processing episode {pt_path.stem}: {e}")
            continue

    # Concatenate all features
    features_array = np.concatenate(all_features, axis=0)

    logger.info(f"Extracted features: {features_array.shape}")

    # Save features
    features_path = output_split_dir / 'features.npy'
    np.save(features_path, features_array)
    logger.info(f"Saved features to {features_path}")

    # Save metadata
    metadata_path = output_split_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(all_metadata, f)
    logger.info(f"Saved metadata to {metadata_path}")

    # Save extraction info
    info = {
        'split': split,
        'num_episodes': len(pt_files),
        'num_frames': len(features_array),
        'feature_dim': features_array.shape[1],
        'episodes': [pt.stem for pt in pt_files]
    }
    info_path = output_split_dir / 'extraction_info.json'
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    logger.info(f"Saved extraction info to {info_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract features from transformer outputs'
    )
    parser.add_argument(
        '--transformer_dir',
        type=Path,
        default=Path('output/transformer_outs'),
        help='Directory containing transformer_outs/{split}/*.pt'
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        default=Path('output/laughter/features'),
        help='Output directory for features'
    )
    parser.add_argument(
        '--split',
        type=str,
        required=True,
        choices=['train', 'validation', 'test'],
        help='Dataset split to process'
    )

    args = parser.parse_args()

    # Setup logging
    log_dir = args.output_dir / args.split
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_dir / 'extract_features.log')

    logger.info("Starting feature extraction")
    logger.info(f"Arguments: {vars(args)}")

    extract_features_for_split(
        transformer_dir=args.transformer_dir,
        split=args.split,
        output_dir=args.output_dir
    )

    logger.info("Feature extraction completed!")


if __name__ == '__main__':
    main()
