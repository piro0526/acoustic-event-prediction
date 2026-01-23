"""Preprocess features and labels by concatenating all episodes into single files per split.

This module concatenates all episode-level feature and label files into single
large files for each split, which dramatically speeds up dataset loading during training.

Usage:
    python -m laughter_prediction.preprocess_concat \
        --features_dir outputs/features_masked \
        --output_dir outputs/features_masked_concat \
        --shift_frames 1
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeaturesConcatenator:
    """Concatenate episode-level features and labels into single files per split."""

    def __init__(
        self,
        features_dir: Path,
        output_dir: Path,
        shift_frames: int = 1
    ):
        """Initialize concatenator.

        Args:
            features_dir: Input directory with episode-level structure
            output_dir: Output directory for concatenated files
            shift_frames: Which shift value to use for labels
        """
        self.features_dir = Path(features_dir)
        self.output_dir = Path(output_dir)
        self.shift_frames = shift_frames

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_split(self, split: str):
        """Process one split (train/validation/test).

        Creates:
            {output_dir}/{split}_features.npy - [N, 4096] all features
            {output_dir}/{split}_labels.npy - [N] all labels
            {output_dir}/{split}_metadata.json - episode boundaries and stats

        Args:
            split: One of 'train', 'validation', 'test'
        """
        logger.info(f"Processing {split} split...")

        split_dir = self.features_dir / split

        if not split_dir.exists():
            logger.warning(f"Split directory not found: {split_dir}, skipping")
            return

        # Find all episode directories
        episode_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])

        if not episode_dirs:
            logger.warning(f"No episodes found in {split_dir}, skipping")
            return

        logger.info(f"Found {len(episode_dirs)} episodes")

        # Accumulate features and labels
        all_features = []
        all_labels = []
        episode_metadata = []

        total_frames = 0
        total_positive = 0

        for episode_dir in tqdm(episode_dirs, desc=f"Concatenating {split}"):
            episode_name = episode_dir.name

            # Load episode metadata
            metadata_path = episode_dir / f'metadata_shift_{self.shift_frames}.json'
            if not metadata_path.exists():
                logger.warning(f"Metadata not found for {episode_name}, skipping")
                continue

            with open(metadata_path, 'r') as f:
                ep_metadata = json.load(f)

            # Process both assignments
            for assignment_idx in [0, 1]:
                features_path = episode_dir / f'features_assignment_{assignment_idx}.npy'
                labels_path = episode_dir / f'labels_assignment_{assignment_idx}_shift_{self.shift_frames}.npy'

                if not features_path.exists():
                    logger.warning(f"Features not found: {features_path}, skipping")
                    continue
                if not labels_path.exists():
                    logger.warning(f"Labels not found: {labels_path}, skipping")
                    continue

                # Load features and labels
                features = np.load(features_path)  # [T, 4096]
                labels = np.load(labels_path)      # [T]

                # Validate shapes
                if features.shape[0] != labels.shape[0]:
                    logger.warning(
                        f"Shape mismatch in {episode_name} assignment {assignment_idx}, skipping"
                    )
                    continue

                # Record metadata for this episode chunk
                start_frame = total_frames
                num_frames = features.shape[0]
                end_frame = start_frame + num_frames

                assignment_info = ep_metadata['assignments'][assignment_idx]
                num_positive = int(labels.sum())

                episode_metadata.append({
                    'episode_name': episode_name,
                    'assignment_idx': assignment_idx,
                    'user_speaker_id': assignment_info['user_speaker_id'],
                    'system_speaker_id': assignment_info['system_speaker_id'],
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'num_frames': num_frames,
                    'num_positive_frames': num_positive,
                    'positive_rate': float(num_positive / num_frames) if num_frames > 0 else 0.0
                })

                # Accumulate
                all_features.append(features)
                all_labels.append(labels)

                total_frames += num_frames
                total_positive += num_positive

        if not all_features:
            logger.error(f"No valid data found for {split} split")
            return

        logger.info(f"Concatenating {len(all_features)} feature arrays...")

        # Concatenate all features and labels
        concat_features = np.concatenate(all_features, axis=0)
        concat_labels = np.concatenate(all_labels, axis=0)

        logger.info(f"Final shape: features={concat_features.shape}, labels={concat_labels.shape}")
        logger.info(f"Positive frames: {total_positive}/{total_frames} ({100.0*total_positive/total_frames:.2f}%)")

        # Save concatenated arrays
        features_output_path = self.output_dir / f'{split}_features.npy'
        labels_output_path = self.output_dir / f'{split}_labels.npy'
        metadata_output_path = self.output_dir / f'{split}_metadata.json'

        logger.info(f"Saving features to {features_output_path}...")
        np.save(features_output_path, concat_features)

        logger.info(f"Saving labels to {labels_output_path}...")
        np.save(labels_output_path, concat_labels)

        # Save metadata
        metadata = {
            'split': split,
            'shift_frames': self.shift_frames,
            'num_episodes': len(set(ep['episode_name'] for ep in episode_metadata)),
            'num_frames': total_frames,
            'num_positive_frames': total_positive,
            'positive_rate': float(total_positive / total_frames) if total_frames > 0 else 0.0,
            'episodes': episode_metadata
        }

        logger.info(f"Saving metadata to {metadata_output_path}...")
        with open(metadata_output_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"✓ {split} split completed")
        logger.info(f"  Features: {features_output_path} ({concat_features.nbytes / 1e9:.2f} GB)")
        logger.info(f"  Labels: {labels_output_path} ({concat_labels.nbytes / 1e6:.2f} MB)")

    def process_all_splits(self):
        """Process all available splits (train, validation, test)."""
        splits = ['train', 'validation', 'test']

        for split in splits:
            self.process_split(split)

        logger.info("All splits processed successfully!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Concatenate episode-level features and labels into single files per split'
    )
    parser.add_argument(
        '--features_dir',
        type=str,
        required=True,
        help='Input directory with episode-level structure (e.g., outputs/features_masked)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for concatenated files (e.g., outputs/features_masked_concat)'
    )
    parser.add_argument(
        '--shift_frames',
        type=int,
        default=1,
        help='Which shift value to use for labels (default: 1)'
    )
    parser.add_argument(
        '--splits',
        type=str,
        nargs='+',
        default=['train', 'validation', 'test'],
        help='Which splits to process (default: all)'
    )

    args = parser.parse_args()

    # Create concatenator
    concatenator = FeaturesConcatenator(
        features_dir=args.features_dir,
        output_dir=args.output_dir,
        shift_frames=args.shift_frames
    )

    # Process specified splits
    for split in args.splits:
        concatenator.process_split(split)

    logger.info("Done!")


if __name__ == '__main__':
    main()
