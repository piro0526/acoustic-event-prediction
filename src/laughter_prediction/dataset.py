"""Dataset for laughter event prediction from pre-extracted features."""

import json
import logging
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class LaughterDataset(Dataset):
    """Dataset for laughter event prediction.

    Loads pre-extracted features and labels from episode-level structure.
    Features and labels are extracted using compute_features.py.
    """

    def __init__(
        self,
        features_dir: Path,
        split: str = 'train',
        shift_frames: int = 1,
        shuffle: bool = True
    ):
        """Initialize dataset.

        Args:
            features_dir: Base directory containing {split}/{episode_name}/ subdirectories
            split: One of 'train', 'test', 'validation'
            shift_frames: Which shift value to use for labels (e.g., 1, 5, 10, 25)
            shuffle: Whether to shuffle frames
        """
        self.features_dir = Path(features_dir) / split
        self.split = split
        self.shift_frames = shift_frames
        self.shuffle = shuffle

        # Validate directory exists
        if not self.features_dir.exists():
            raise FileNotFoundError(f"Features directory not found: {self.features_dir}")

        # Load pre-extracted data
        self._load_data()
        logger.info(f"Loaded {len(self.features)} frames from {split} split")

    def _load_data(self):
        """Load pre-extracted features, labels, and metadata from episode-level structure.

        Loads from:
            {split}/{episode_name}/features_assignment_{0,1}.npy
            {split}/{episode_name}/labels_assignment_{0,1}_shift_{N}.npy
            {split}/{episode_name}/metadata_shift_{N}.json
        """
        logger.info(f"Loading episodes from {self.features_dir}")

        # Find all episode directories
        episode_dirs = sorted([d for d in self.features_dir.iterdir() if d.is_dir()])

        if not episode_dirs:
            raise FileNotFoundError(f"No episode directories found in {self.features_dir}")

        logger.info(f"Found {len(episode_dirs)} episodes")

        # Accumulate features, labels, and metadata
        all_features = []
        all_labels = []
        all_metadata = []

        for episode_dir in episode_dirs:
            episode_name = episode_dir.name

            # Load metadata for this episode
            metadata_path = episode_dir / f'metadata_shift_{self.shift_frames}.json'
            if not metadata_path.exists():
                logger.warning(
                    f"Metadata not found for episode {episode_name} with shift={self.shift_frames}, skipping"
                )
                continue

            with open(metadata_path, 'r') as f:
                episode_metadata = json.load(f)

            # Process both assignments (0, 1)
            for assignment_idx in [0, 1]:
                features_path = episode_dir / f'features_assignment_{assignment_idx}.npy'
                labels_path = episode_dir / f'labels_assignment_{assignment_idx}_shift_{self.shift_frames}.npy'

                # Check if files exist
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
                        f"Shape mismatch in {episode_name} assignment {assignment_idx}: "
                        f"features={features.shape[0]}, labels={labels.shape[0]}, skipping"
                    )
                    continue

                # Get assignment info from metadata
                assignment_info = episode_metadata['assignments'][assignment_idx]

                # Create per-frame metadata
                frame_metadata = {
                    'episode_name': episode_name,
                    'split': self.split,
                    'assignment_idx': assignment_idx,
                    'user_speaker_id': assignment_info['user_speaker_id'],
                    'system_speaker_id': assignment_info['system_speaker_id']
                }

                # Accumulate
                all_features.append(features)
                all_labels.append(labels)
                all_metadata.extend([frame_metadata] * len(features))

                logger.debug(
                    f"Loaded {episode_name} assignment {assignment_idx}: "
                    f"{features.shape[0]} frames, "
                    f"{assignment_info['num_positive_frames']} positive"
                )

        # Concatenate all episodes
        if not all_features:
            raise ValueError(f"No valid episode data found in {self.features_dir}")

        self.features = np.concatenate(all_features, axis=0)
        self.labels = np.concatenate(all_labels, axis=0)
        self.metadata = all_metadata

        # Store label config
        self.label_config = {
            'shift_frames': self.shift_frames,
            'mode': 'prediction'
        }

        # Validate shapes match
        if len(self.features) != len(self.labels):
            raise ValueError(
                f"Features and labels length mismatch: "
                f"{len(self.features)} vs {len(self.labels)}"
            )
        if len(self.features) != len(self.metadata):
            raise ValueError(
                f"Features and metadata length mismatch: "
                f"{len(self.features)} vs {len(self.metadata)}"
            )

        logger.info(
            f"Loaded {len(self.features)} total frames from {len(episode_dirs)} episodes "
            f"({np.sum(self.labels)} positive, {len(self.labels) - np.sum(self.labels)} negative)"
        )

        # Create shuffled indices if requested
        if self.shuffle:
            self.indices = np.random.permutation(len(self.features))
        else:
            self.indices = np.arange(len(self.features))

    def __len__(self) -> int:
        """Return total number of frames."""
        return len(self.features)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample.

        Args:
            idx: Frame index

        Returns:
            Dictionary with 'features' and 'label' tensors
        """
        # Use shuffled index if shuffle=True
        actual_idx = self.indices[idx]

        # Get features and label
        features = torch.from_numpy(self.features[actual_idx].astype(np.float32))
        label = torch.tensor([float(self.labels[actual_idx])], dtype=torch.float32)

        return {
            'features': features,
            'label': label,
        }

    def compute_class_weights(self) -> torch.Tensor:
        """Compute positive class weight for BCE loss from loaded labels.

        Returns:
            pos_weight tensor for BCEWithLogitsLoss
        """
        logger.info("Computing class weights from labels...")

        num_positive = np.sum(self.labels)
        num_negative = len(self.labels) - num_positive

        # Calculate pos_weight
        pos_weight = num_negative / num_positive if num_positive > 0 else 1.0

        logger.info(
            f"Class distribution: "
            f"{num_positive} positive ({num_positive/len(self.labels)*100:.2f}%), "
            f"{num_negative} negative ({num_negative/len(self.labels)*100:.2f}%)"
        )
        logger.info(f"Computed pos_weight: {pos_weight:.4f}")

        return torch.tensor([pos_weight], dtype=torch.float32)

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics from loaded data.

        Returns:
            Dictionary with dataset statistics
        """
        num_positive = int(np.sum(self.labels))
        num_negative = len(self.labels) - num_positive
        pos_ratio = num_positive / len(self.labels) if len(self.labels) > 0 else 0.0

        # Get unique episodes from metadata
        unique_episodes = set(meta['episode_name'] for meta in self.metadata)

        return {
            'num_episodes': len(unique_episodes),
            'num_frames': len(self.labels),
            'num_positive': num_positive,
            'num_negative': num_negative,
            'pos_ratio': float(pos_ratio),
            'label_config': self.label_config,
        }
