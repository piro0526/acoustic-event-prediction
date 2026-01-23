"""Dataset for laughter event prediction from pre-extracted features."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
from functools import lru_cache

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class LaughterDataset(Dataset):
    """Dataset for laughter event prediction.

    Loads pre-extracted features and labels from numpy arrays.
    Features are extracted using extract_features.py and labels are
    created using create_labels.py.
    """

    def __init__(
        self,
        features_dir: Path,
        split: str = 'train',
        shuffle: bool = True
    ):
        """Initialize dataset.

        Args:
            features_dir: Directory containing features/[split]/ with
                         features.npy, labels.npy, metadata.json
            split: One of 'train', 'test', 'validation'
            shuffle: Whether to shuffle frames
        """
        self.features_dir = Path(features_dir) / split
        self.split = split
        self.shuffle = shuffle

        # Validate directory exists
        if not self.features_dir.exists():
            raise FileNotFoundError(f"Features directory not found: {self.features_dir}")

        # Load pre-extracted data
        self._load_data()
        logger.info(f"Loaded {len(self.features)} frames from {split} split")

    def _load_data(self):
        """Load pre-extracted features, labels, and metadata from disk.

        Loads:
            - features.npy: [N, 4096] float32 array
            - labels.npy: [N] int32 array
            - metadata.json: List of metadata dicts for each frame
            - label_config.json: Label generation configuration
        """
        features_path = self.features_dir / 'features.npy'
        labels_path = self.features_dir / 'labels.npy'
        metadata_path = self.features_dir / 'metadata.json'
        label_config_path = self.features_dir / 'label_config.json'

        # Validate files exist
        if not features_path.exists():
            raise FileNotFoundError(f"Features file not found: {features_path}")
        if not labels_path.exists():
            raise FileNotFoundError(f"Labels file not found: {labels_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        logger.info(f"Loading features from {features_path}")
        self.features = np.load(features_path)  # Memory-map for large files

        logger.info(f"Loading labels from {labels_path}")
        self.labels = np.load(labels_path)

        logger.info(f"Loading metadata from {metadata_path}")
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        # Load label config if available
        if label_config_path.exists():
            with open(label_config_path, 'r') as f:
                self.label_config = json.load(f)
        else:
            self.label_config = None

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
