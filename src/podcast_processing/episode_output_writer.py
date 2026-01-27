"""Episode output writer for structured feature and label storage."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


class EpisodeOutputWriter:
    """Handles writing episode-level outputs to disk.

    Manages structured output of features, labels, and metadata for
    a single episode with consistent naming conventions.
    """

    def __init__(self, output_root: Path, episode_name: str, split: str, shift_frames: int):
        """Initialize output writer.

        Args:
            output_root: Base output directory
            episode_name: Episode identifier
            split: Dataset split (train/validation/test)
            shift_frames: Prediction shift value for labels
        """
        self.output_root = Path(output_root)
        self.episode_name = episode_name
        self.split = split
        self.shift_frames = shift_frames
        self.episode_dir = self.output_root / split / episode_name

        # Accumulate assignment statistics
        self.assignment_stats = []

    def prepare_directory(self):
        """Create episode output directory if it doesn't exist."""
        self.episode_dir.mkdir(parents=True, exist_ok=True)

    def save_features(self, assignment_idx: int, features: torch.Tensor):
        """Save features as numpy array.

        Args:
            assignment_idx: Assignment index (0 or 1)
            features: Feature tensor [T, 4096]
        """
        path = self.episode_dir / f"features_assignment_{assignment_idx}.npy"
        np.save(path, features.cpu().float().numpy())
        logger.debug(f"Saved features to {path}")

    def save_labels(self, assignment_idx: int, labels: torch.Tensor):
        """Save labels as numpy array.

        Args:
            assignment_idx: Assignment index (0 or 1)
            labels: Label tensor [T]
        """
        path = self.episode_dir / f"labels_assignment_{assignment_idx}_shift_{self.shift_frames}.npy"
        np.save(path, labels.cpu().numpy().astype(np.int32))
        logger.debug(f"Saved labels to {path}")

    def add_assignment_stats(
        self,
        assignment_idx: int,
        user_speaker_id: str,
        system_speaker_id: str,
        stats: Dict
    ):
        """Add assignment statistics to metadata buffer.

        Args:
            assignment_idx: Assignment index (0 or 1)
            user_speaker_id: User speaker ID (e.g., "SPEAKER_00")
            system_speaker_id: System speaker ID
            stats: Dictionary with num_positive_frames, positive_rate
        """
        self.assignment_stats.append({
            'assignment_idx': assignment_idx,
            'user_speaker_id': user_speaker_id,
            'system_speaker_id': system_speaker_id,
            'num_positive_frames': stats['num_positive_frames'],
            'positive_rate': stats['positive_rate']
        })

    def save_metadata(
        self,
        episode_info: Dict,
        num_frames: int,
        duration: float
    ):
        """Save metadata JSON for this shift value.

        Args:
            episode_info: Episode information dict with 'name', 'split'
            num_frames: Total number of frames
            duration: Episode duration in seconds
        """
        metadata = {
            'episode_name': episode_info['name'],
            'split': episode_info['split'],
            'duration_seconds': float(duration),
            'frame_rate': 12.5,
            'num_frames': num_frames,
            'shift_frames': self.shift_frames,
            'assignments': self.assignment_stats
        }

        path = self.episode_dir / f'metadata_shift_{self.shift_frames}.json'
        with open(path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.debug(f"Saved metadata to {path}")


class MultiSpeakerOutputWriter:
    """Handles writing episode-level outputs for multi-speaker processing.

    Extended version of EpisodeOutputWriter that supports variable numbers of
    assignments and multiple system speakers per assignment.
    """

    def __init__(self, output_root: Path, episode_name: str, split: str, shift_frames: int):
        """Initialize output writer.

        Args:
            output_root: Base output directory
            episode_name: Episode identifier
            split: Dataset split (train/validation/test)
            shift_frames: Prediction shift value for labels
        """
        self.output_root = Path(output_root)
        self.episode_name = episode_name
        self.split = split
        self.shift_frames = shift_frames
        self.episode_dir = self.output_root / split / episode_name

        # Accumulate assignment statistics
        self.assignment_stats = []

    def prepare_directory(self):
        """Create episode output directory if it doesn't exist."""
        self.episode_dir.mkdir(parents=True, exist_ok=True)

    def save_features(self, assignment_idx: int, features: torch.Tensor):
        """Save features as numpy array.

        Args:
            assignment_idx: Assignment index (0, 1, 2, ...)
            features: Feature tensor [T, 4096]
        """
        path = self.episode_dir / f"features_assignment_{assignment_idx}.npy"
        np.save(path, features.cpu().float().numpy())
        logger.debug(f"Saved features to {path}")

    def save_labels(self, assignment_idx: int, labels: torch.Tensor):
        """Save labels as numpy array.

        Args:
            assignment_idx: Assignment index (0, 1, 2, ...)
            labels: Label tensor [T]
        """
        path = self.episode_dir / f"labels_assignment_{assignment_idx}_shift_{self.shift_frames}.npy"
        np.save(path, labels.cpu().numpy().astype(np.int32))
        logger.debug(f"Saved labels to {path}")

    def add_assignment_stats(
        self,
        assignment_idx: int,
        user_speaker_id: str,
        user_speaking_ratio: float,
        system_speaker_ids: List[str],
        stats: Dict
    ):
        """Add assignment statistics to metadata buffer.

        Args:
            assignment_idx: Assignment index (0, 1, 2, ...)
            user_speaker_id: User speaker ID (e.g., "SPEAKER_00")
            user_speaking_ratio: User speaker's speaking ratio (0.0-1.0)
            system_speaker_ids: List of system speaker IDs
            stats: Dictionary with num_positive_frames, positive_rate
        """
        self.assignment_stats.append({
            'assignment_idx': assignment_idx,
            'user_speaker_id': user_speaker_id,
            'user_speaking_ratio': user_speaking_ratio,
            'system_speaker_ids': system_speaker_ids,
            'num_positive_frames': stats['num_positive_frames'],
            'positive_rate': stats['positive_rate']
        })

    def save_metadata(
        self,
        episode_info: Dict,
        num_frames: int,
        duration: float,
        all_speakers: List[str],
        min_speaker_share: float,
        mask_laughter: bool
    ):
        """Save metadata JSON for this shift value.

        Args:
            episode_info: Episode information dict with 'name', 'split'
            num_frames: Total number of frames
            duration: Episode duration in seconds
            all_speakers: List of all speaker IDs in episode
            min_speaker_share: Minimum speaking ratio threshold used
            mask_laughter: Whether laughter masking was applied
        """
        metadata = {
            'episode_name': episode_info['name'],
            'split': episode_info['split'],
            'duration_seconds': float(duration),
            'frame_rate': 12.5,
            'num_frames': num_frames,
            'shift_frames': self.shift_frames,
            'min_speaker_share': min_speaker_share,
            'mask_laughter': mask_laughter,
            'all_speakers': all_speakers,
            'assignments': self.assignment_stats
        }

        path = self.episode_dir / f'metadata_shift_{self.shift_frames}.json'
        with open(path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.debug(f"Saved metadata to {path}")
