"""Iterable dataset for laughter event prediction from pre-computed transformer outputs."""

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset, get_worker_info

from .utils import time_to_frame_index, match_speaker_to_assignment

logger = logging.getLogger(__name__)


class LaughterRegressionIterableDataset(IterableDataset):
    """Iterable dataset for laughter event prediction.

    This dataset loads episodes sequentially to avoid random access bottlenecks.
    Each episode is loaded once, and all frames within that episode are yielded
    before moving to the next episode.

    Supports DDP and DataLoader workers for efficient parallel processing.
    """

    def __init__(
        self,
        transformer_dir: Path,
        labels_dir: Path,
        split: str = 'train',
        shuffle: bool = True,
        balance_classes: bool = False
    ):
        """Initialize iterable dataset.

        Args:
            transformer_dir: Directory containing transformer_outs/[split]/*.pt files
            labels_dir: Directory containing laughter_intervals/[split]/*.json files
            split: One of 'train', 'test', 'validation'
            shuffle: Whether to shuffle episodes and frames within episodes
            balance_classes: Whether to balance positive/negative samples (downsample negatives to match positives)
        """
        self.transformer_dir = Path(transformer_dir) / split
        self.labels_dir = Path(labels_dir) / split
        self.split = split
        self.shuffle = shuffle
        self.balance_classes = balance_classes

        # Validate directories exist
        if not self.transformer_dir.exists():
            raise FileNotFoundError(f"Transformer directory not found: {self.transformer_dir}")
        if not self.labels_dir.exists():
            raise FileNotFoundError(f"Labels directory not found: {self.labels_dir}")

        # Find all episodes
        self.episodes = self._find_episodes()
        logger.info(f"Found {len(self.episodes)} episodes in {split} split")

    def _find_episodes(self) -> List[Dict[str, Path]]:
        """Find all episodes with matching transformer and label files.

        Returns:
            List of dicts with 'name', 'transformer_path', 'label_path'
        """
        episodes = []
        pt_files = list(self.transformer_dir.glob('*.pt'))

        for pt_file in pt_files:
            episode_name = pt_file.stem
            json_file = self.labels_dir / f"{episode_name}.json"

            if json_file.exists():
                episodes.append({
                    'name': episode_name,
                    'transformer_path': pt_file,
                    'label_path': json_file
                })
            else:
                logger.warning(f"No label file found for episode: {episode_name}")

        return episodes

    def _load_episode(self, episode: Dict[str, Path]) -> List[Dict[str, torch.Tensor]]:
        """Load episode file and return all frames with labels.

        Args:
            episode: Episode metadata dict

        Returns:
            List of sample dicts with 'features' and 'label'
        """
        # 1. Load data
        data = torch.load(episode['transformer_path'], map_location='cpu')
        transformer_out = data['transformer_out']  # [2, T, 4096]
        metadata = data['metadata']

        with open(episode['label_path'], 'r') as f:
            laughter_data = json.load(f)

        num_frames = transformer_out.shape[1]
        positive_samples = []
        negative_samples = []

        # 2. Generate labels and create samples for both assignments
        for assign_idx in range(2):
            assignment_meta = metadata['assignments'][assign_idx]
            matched = match_speaker_to_assignment(
                laughter_data['intervals'],
                assignment_meta
            )

            # Generate frame-level labels with regression targets
            # Label = 1/(t+1) where t is time in seconds until event['prediction_interval_end']
            # Labels are set for frames up to 10 seconds before the event
            labels = np.zeros(num_frames, dtype=np.float32)
            for event in matched:
                event_end_time = event['prediction_interval_end']
                event_end_frame = time_to_frame_index(event_end_time)

                # Start from 10 seconds before the event end
                lookback_time = 10.0  # seconds
                start_time = max(0, event_end_time - lookback_time)
                start_frame = max(0, time_to_frame_index(start_time))
                end_frame = min(num_frames - 1, event_end_frame)

                # Set labels for each frame: 1/(t+1) where t is time until event end
                for frame_idx in range(start_frame, end_frame + 1):
                    frame_time = frame_idx / 12.5  # Convert frame index to time
                    time_until_event = event_end_time - frame_time
                    if time_until_event >= 0:
                        labels[frame_idx] = 1.0 / (time_until_event + 1.0)

            # Extract features for this assignment
            feats = transformer_out[assign_idx]  # [T, 4096]

            # Create samples for all frames in this assignment
            for t in range(num_frames):
                sample = {
                    'features': feats[t],  # Keep as view for memory efficiency
                    'label': torch.tensor([labels[t]], dtype=torch.float32),
                }

                # Separate positive and negative samples
                if labels[t] == 0.0:
                    negative_samples.append(sample)
                else:
                    positive_samples.append(sample)

        # 3. Balance classes if requested
        if self.balance_classes:
            num_positive = len(positive_samples)
            num_negative = len(negative_samples)

            if num_negative > num_positive:
                # Downsample negatives to match positives
                negative_samples = random.sample(negative_samples, num_positive)

            samples = positive_samples + negative_samples
        else:
            samples = positive_samples + negative_samples

        return samples

    def __iter__(self):
        """Iterate over dataset with DDP and DataLoader worker support."""

        # 1. Determine which episodes this worker is responsible for
        worker_info = get_worker_info()

        # Get process rank information for DDP
        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1

        # Create episode index list
        indices = list(range(len(self.episodes)))

        # Shuffle episodes if requested
        if self.shuffle:
            # Use different seed per rank for different episode ordering
            g = torch.Generator()
            g.manual_seed(torch.initial_seed() + rank)
            indices = torch.randperm(len(indices), generator=g).tolist()

        # 2. DDP: Split episodes across GPUs
        # Each GPU handles 1/world_size of all episodes
        my_indices = indices[rank::world_size]

        # 3. DataLoader workers: Further split within each GPU
        if worker_info is not None:
            my_indices = my_indices[worker_info.id::worker_info.num_workers]

        # 4. Data yielding loop
        for idx in my_indices:
            episode = self.episodes[idx]
            samples = self._load_episode(episode)

            # Shuffle frames within episode for local randomness
            if self.shuffle:
                random.shuffle(samples)

            yield from samples

    def __len__(self):
        """Return approximate dataset length.

        Note: Exact length requires loading all episodes, so we return an estimate
        based on average frames per episode.
        """
        if self.balance_classes:
            # When balanced, dataset size is ~2x positive samples
            stats = self.get_statistics()
            return stats['num_positive'] * 2
        else:
            # Estimate: ~30000 frames per episode * 2 assignments
            return len(self.episodes) * 30000 * 2

    def compute_class_weights(self) -> torch.Tensor:
        """Compute positive class weight for BCE loss from metadata only.

        This method estimates the positive class ratio without loading episode tensors.
        For each episode:
        1. Estimate episode duration from max event_end_inepisode in labels
        2. Sum total laughter prediction interval durations
        3. Calculate positive ratio as total_laughter_duration / episode_duration
        4. Average across all episodes to get dataset-level ratio

        Returns:
            pos_weight tensor for BCEWithLogitsLoss
        """
        logger.info("Computing class weights from label metadata only...")

        episode_ratios = []

        for episode in self.episodes:
            # Load laughter intervals (lightweight JSON only)
            with open(episode['label_path'], 'r') as f:
                laughter_data = json.load(f)

            # Calculate ratio for this episode
            episode_ratio = sum([ stat['coverage_percent'] for _, stat in laughter_data['speaker_statistics'].items()]) / len(laughter_data['speaker_statistics']) / 100
            episode_ratios.append(episode_ratio)

        # Average across all episodes to get dataset-level positive ratio
        avg_positive_ratio = sum(episode_ratios) / len(episode_ratios) if episode_ratios else 0.0
        avg_negative_ratio = 1.0 - avg_positive_ratio

        # Calculate pos_weight
        pos_weight = avg_negative_ratio / avg_positive_ratio if avg_positive_ratio > 0 else 1.0

        logger.info(
            f"Class distribution: "
            f"{avg_positive_ratio*100:.2f}% positive, "
            f"{avg_negative_ratio*100:.2f}% negative"
        )
        logger.info(f"Computed pos_weight: {pos_weight:.4f}")

        return torch.tensor([pos_weight], dtype=torch.float32)

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics estimated from label metadata only.

        This method estimates statistics without loading episode tensors.
        Uses label metadata to approximate durations and frame counts.

        Returns:
            Dictionary with estimated dataset statistics
        """
        episode_durations = []
        episode_ratios = []

        for episode in self.episodes:
            # Load laughter intervals (lightweight JSON only)
            with open(episode['label_path'], 'r') as f:
                laughter_data = json.load(f)

            episode_durations.append(laughter_data['episode_duration'])

            # Calculate ratio for this episode
            episode_ratio = sum([ stat['coverage_percent'] for _, stat in laughter_data['speaker_statistics'].items()]) / len(laughter_data['speaker_statistics']) / 100
            episode_ratios.append(episode_ratio)

        # Calculate statistics
        # Estimate frames from duration (frame_rate = 12.5 Hz)
        total_frames = sum(int(duration * 12.5) for duration in episode_durations)
        # Multiply by 2 for two speaker assignments per episode
        total_frames_with_assignments = total_frames * 2

        # Average positive ratio across episodes
        avg_positive_ratio = sum(episode_ratios) / len(episode_ratios) if episode_ratios else 0.0

        # Estimate counts
        num_positive = int(total_frames_with_assignments * avg_positive_ratio)
        num_negative = total_frames_with_assignments - num_positive

        # Frame statistics per episode
        frames_per_episode = [int(duration * 12.5) for duration in episode_durations]

        # Adjust for class balancing
        if self.balance_classes:
            # After balancing, negative samples are downsampled to match positive
            num_negative_balanced = num_positive
            total_frames_balanced = num_positive + num_negative_balanced
            pos_ratio_balanced = 0.5  # 50/50 split after balancing

            return {
                'num_episodes': len(self.episodes),
                'num_frames': total_frames_balanced,
                'num_positive': num_positive,
                'num_negative': num_negative_balanced,
                'pos_ratio': pos_ratio_balanced,
                'avg_frames_per_episode': float(np.mean(frames_per_episode)) if frames_per_episode else 0.0,
                'min_frames_per_episode': int(np.min(frames_per_episode)) if frames_per_episode else 0,
                'max_frames_per_episode': int(np.max(frames_per_episode)) if frames_per_episode else 0,
                'balanced': True,
                'original_num_negative': num_negative,
                'original_pos_ratio': float(avg_positive_ratio),
            }
        else:
            return {
                'num_episodes': len(self.episodes),
                'num_frames': total_frames_with_assignments,
                'num_positive': num_positive,
                'num_negative': num_negative,
                'pos_ratio': float(avg_positive_ratio),
                'avg_frames_per_episode': float(np.mean(frames_per_episode)) if frames_per_episode else 0.0,
                'min_frames_per_episode': int(np.min(frames_per_episode)) if frames_per_episode else 0,
                'max_frames_per_episode': int(np.max(frames_per_episode)) if frames_per_episode else 0,
                'balanced': False,
            }
