"""Regenerate labels using AnnotationLabelGenerator (all speakers, no distinction).

This script creates new label files from episode_annotations CSV without
speaker filtering. All laughter events are included regardless of speaker.

Usage:
    python scripts/regenerate_labels_all_speakers.py \
        --features_dir outputs/features_masked \
        --annotation_dir data/PodcastFillers/metadata/episode_annotations \
        --output_dir outputs/features_masked_all_speakers \
        --shift_frames 1
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm

from podcast_processing.label_generator import AnnotationLabelGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AllSpeakersLabelRegenerator:
    """Regenerate labels without speaker distinction."""

    def __init__(
        self,
        features_dir: Path,
        annotation_dir: Path,
        output_dir: Path,
        shift_frames: int = 1,
        frame_rate: float = 12.5
    ):
        """Initialize regenerator.

        Args:
            features_dir: Directory with episode-level features
            annotation_dir: Directory with episode_annotations CSVs
            output_dir: Output directory for new concatenated files
            shift_frames: Prediction shift in frames
            frame_rate: Frame rate in Hz
        """
        self.features_dir = Path(features_dir)
        self.annotation_dir = Path(annotation_dir)
        self.output_dir = Path(output_dir)
        self.shift_frames = shift_frames

        self.label_generator = AnnotationLabelGenerator(frame_rate=frame_rate)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_split(self, split: str):
        """Process one split (train/validation/test).

        Args:
            split: One of 'train', 'validation', 'test'
        """
        logger.info(f"Processing {split} split...")

        features_split_dir = self.features_dir / split
        annotation_split_dir = self.annotation_dir / split

        if not features_split_dir.exists():
            logger.warning(f"Features directory not found: {features_split_dir}, skipping")
            return

        if not annotation_split_dir.exists():
            logger.warning(f"Annotation directory not found: {annotation_split_dir}, skipping")
            return

        # Find all episode directories
        episode_dirs = sorted([d for d in features_split_dir.iterdir() if d.is_dir()])

        if not episode_dirs:
            logger.warning(f"No episodes found in {features_split_dir}, skipping")
            return

        logger.info(f"Found {len(episode_dirs)} episodes")

        # Accumulate features and labels
        all_features = []
        all_labels = []
        episode_metadata = []

        total_frames = 0
        total_positive = 0

        for episode_dir in tqdm(episode_dirs, desc=f"Processing {split}"):
            episode_name = episode_dir.name

            # Find annotation CSV
            annotation_path = annotation_split_dir / f"{episode_name}.csv"
            if not annotation_path.exists():
                logger.warning(f"Annotation not found: {annotation_path}, skipping")
                continue

            # Load features from assignment_0 (same as assignment_1)
            features_path = episode_dir / 'features_assignment_0.npy'
            if not features_path.exists():
                logger.warning(f"Features not found: {features_path}, skipping")
                continue

            features = np.load(features_path)  # [T, 4096]
            num_frames = features.shape[0]

            # Load laughter events and create labels
            laughter_events = self.label_generator.load_laughter_events(annotation_path)
            labels = self.label_generator.create_labels_prediction(
                laughter_events=laughter_events,
                num_frames=num_frames,
                shift_frames=self.shift_frames
            )
            labels_np = labels.numpy()

            # Record metadata
            start_frame = total_frames
            end_frame = start_frame + num_frames
            num_positive = int(labels_np.sum())

            episode_metadata.append({
                'episode_name': episode_name,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'num_frames': num_frames,
                'num_laughter_events': len(laughter_events),
                'num_positive_frames': num_positive,
                'positive_rate': float(num_positive / num_frames) if num_frames > 0 else 0.0
            })

            # Accumulate
            all_features.append(features)
            all_labels.append(labels_np)

            total_frames += num_frames
            total_positive += num_positive

        if not all_features:
            logger.error(f"No valid data found for {split} split")
            return

        logger.info(f"Concatenating {len(all_features)} episodes...")

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
            'speaker_filtering': False,
            'num_episodes': len(episode_metadata),
            'num_frames': total_frames,
            'num_positive_frames': total_positive,
            'positive_rate': float(total_positive / total_frames) if total_frames > 0 else 0.0,
            'episodes': episode_metadata
        }

        logger.info(f"Saving metadata to {metadata_output_path}...")
        with open(metadata_output_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Done: {split} split")
        logger.info(f"  Features: {features_output_path} ({concat_features.nbytes / 1e9:.2f} GB)")
        logger.info(f"  Labels: {labels_output_path} ({concat_labels.nbytes / 1e6:.2f} MB)")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Regenerate labels using all speakers (no speaker filtering)'
    )
    parser.add_argument(
        '--features_dir',
        type=str,
        required=True,
        help='Input directory with episode-level features'
    )
    parser.add_argument(
        '--annotation_dir',
        type=str,
        required=True,
        help='Directory with episode_annotations CSVs'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for concatenated files'
    )
    parser.add_argument(
        '--shift_frames',
        type=int,
        default=1,
        help='Prediction shift in frames (default: 1)'
    )
    parser.add_argument(
        '--splits',
        type=str,
        nargs='+',
        default=['train', 'validation', 'test'],
        help='Which splits to process (default: all)'
    )

    args = parser.parse_args()

    regenerator = AllSpeakersLabelRegenerator(
        features_dir=args.features_dir,
        annotation_dir=args.annotation_dir,
        output_dir=args.output_dir,
        shift_frames=args.shift_frames
    )

    for split in args.splits:
        regenerator.process_split(split)

    logger.info("All done!")


if __name__ == '__main__':
    main()
