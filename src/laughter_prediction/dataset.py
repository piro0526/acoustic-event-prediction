"""Dataset for laughter event prediction from pre-concatenated features."""

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

    Loads pre-concatenated features and labels created by preprocess_concat.py.
    This avoids the memory overhead of concatenating during initialization.
    """

    def __init__(
        self,
        concat_dir: Path,
        split: str = 'train',
        shuffle: bool = True,
        mmap_mode: str = 'r'
    ):
        """Initialize dataset.

        Args:
            concat_dir: Directory containing pre-concatenated {split}_features.npy files
            split: One of 'train', 'test', 'validation'
            shuffle: Whether to shuffle frames
            mmap_mode: Memory-map mode ('r' for read-only, None to load in memory)
        """
        self.concat_dir = Path(concat_dir)
        self.split = split
        self.shuffle = shuffle
        self.mmap_mode = mmap_mode

        # Load concatenated data
        self._load_concat_data()

        logger.info(f"Loaded {len(self.features)} frames from {split} split")

    def _load_concat_data(self):
        """Load pre-concatenated features and labels.

        Loads from:
            {concat_dir}/{split}_features.npy
            {concat_dir}/{split}_labels.npy
            {concat_dir}/{split}_metadata.json
        """
        # Paths to concatenated files
        features_path = self.concat_dir / f'{self.split}_features.npy'
        labels_path = self.concat_dir / f'{self.split}_labels.npy'
        metadata_path = self.concat_dir / f'{self.split}_metadata.json'

        # Validate files exist
        if not features_path.exists():
            raise FileNotFoundError(f"Features file not found: {features_path}")
        if not labels_path.exists():
            raise FileNotFoundError(f"Labels file not found: {labels_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        # Load metadata
        logger.info(f"Loading metadata from {metadata_path}")
        with open(metadata_path, 'r') as f:
            self.concat_metadata = json.load(f)

        # Load features and labels using memory mapping
        if self.mmap_mode is not None:
            logger.info(f"Loading features with memory-mapping (mode={self.mmap_mode})...")
            self.features = np.load(features_path, mmap_mode=self.mmap_mode)
            self.labels = np.load(labels_path, mmap_mode=self.mmap_mode)
            logger.info(f"Memory-mapped {len(self.features)} frames (not loaded into RAM)")
        else:
            logger.info(f"Loading features into memory...")
            self.features = np.load(features_path)
            self.labels = np.load(labels_path)
            logger.info(f"Loaded {len(self.features)} frames into memory")

        # Validate shapes
        if len(self.features) != len(self.labels):
            raise ValueError(
                f"Features and labels length mismatch: "
                f"{len(self.features)} vs {len(self.labels)}"
            )

        num_positive = int(self.concat_metadata['num_positive_frames'])
        num_negative = self.concat_metadata['num_frames'] - num_positive

        logger.info(
            f"Loaded {self.concat_metadata['num_frames']} frames from "
            f"{self.concat_metadata['num_episodes']} episodes "
            f"({num_positive} positive [{100.0*self.concat_metadata['positive_rate']:.2f}%], "
            f"{num_negative} negative)"
        )

        # Create shuffled indices if requested
        if self.shuffle:
            logger.info("Creating shuffled indices...")
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
        """Compute positive class weight for BCE loss.

        Returns:
            pos_weight tensor for BCEWithLogitsLoss
        """
        logger.info("Computing class weights...")

        num_positive = self.concat_metadata['num_positive_frames']
        num_negative = self.concat_metadata['num_frames'] - num_positive

        # Calculate pos_weight
        pos_weight = num_negative / num_positive if num_positive > 0 else 1.0

        logger.info(
            f"Class distribution: "
            f"{num_positive} positive ({100.0*self.concat_metadata['positive_rate']:.2f}%), "
            f"{num_negative} negative"
        )
        logger.info(f"Computed pos_weight: {pos_weight:.4f}")

        return torch.tensor([pos_weight], dtype=torch.float32)

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics.

        Returns:
            Dictionary with dataset statistics
        """
        return {
            'num_episodes': self.concat_metadata['num_episodes'],
            'num_frames': self.concat_metadata['num_frames'],
            'num_positive': self.concat_metadata['num_positive_frames'],
            'num_negative': self.concat_metadata['num_frames'] - self.concat_metadata['num_positive_frames'],
            'pos_ratio': self.concat_metadata['positive_rate'],
            'shift_frames': self.concat_metadata['shift_frames'],
        }
