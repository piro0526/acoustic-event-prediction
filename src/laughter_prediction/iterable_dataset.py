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

from .utils import time_to_frame_index, match_speaker_to_assignment, frame_index_to_time

logger = logging.getLogger(__name__)


class LaughterIterableDataset(IterableDataset):
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
        turns_dir: Path,
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
        self.turns_dir = Path(turns_dir) / split
        self.split = split
        self.shuffle = shuffle
        self.balance_classes = balance_classes

        # Validate directories exist
        if not self.transformer_dir.exists():
            raise FileNotFoundError(f"Transformer directory not found: {self.transformer_dir}")
        if not self.labels_dir.exists():
            raise FileNotFoundError(f"Labels directory not found: {self.labels_dir}")
        if not self.turns_dir.exists():
            raise FileNotFoundError(f"Turns directory not found: {self.turns_dir}")

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
            labels_file = self.labels_dir / f"{episode_name}.json"
            turns_file = self.turns_dir / f"{episode_name}.json"

            if labels_file.exists() and turns_file.exists():
                episodes.append({
                    'name': episode_name,
                    'transformer_path': pt_file,
                    'label_path': labels_file,
                    'turns_path': turns_file
                })
            else:
                logger.warning(f"No label or turns file found for episode: {episode_name}")

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

        with open(episode['turns_path'], 'r') as f:
            turns_data = json.load(f)

        change_points = [time_to_frame_index(turn['end']) for turn in turns_data['turns'][:-1]]

        num_frames = transformer_out.shape[1]

        samples = []

        # 2. Generate labels and create samples for both assignments
        for assign_idx in range(2):
            assignment_meta = metadata['assignments'][assign_idx]
            matched = match_speaker_to_assignment(
                laughter_data['intervals'],
                assignment_meta
            )

            # Generate frame-level labels efficiently with NumPy
            labels = np.zeros(num_frames, dtype=np.float32)
            for event in matched:
                s = max(0, time_to_frame_index(event['prediction_interval_end'] - 10))
                e = min(num_frames - 1, time_to_frame_index(event['prediction_interval_end']))
                if event['event_start_inepisode'] - event['current_turn_start'] < 10:
                    labels[s:e + 1] = 1.0
            
            feats = transformer_out[assign_idx]  # [T, 4096]

            # Extract features for this assignment
            turn_idx = 0
            # Create samples for all frames in this assignment
            for t in range(3000, num_frames):
                # Advance turn_idx while change point is before the start of the 3000-frame window
                while turn_idx < len(change_points) and change_points[turn_idx] < (t - 3000):
                    turn_idx += 1

                # Emit sample only if current change point falls within (t-3000, t)
                if turn_idx < len(change_points) and change_points[turn_idx] < t:
                    sample = {
                        'features': feats[t],  # Keep as view for memory efficiency
                        'label': torch.tensor([labels[t]], dtype=torch.float32),
                        'metadata': {
                            'episode_name': metadata['episode_name'],
                            'user_id': assignment_meta['user'],
                            'frame_idx': t,
                        }
                    }
                    samples.append(sample)

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

        """
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
        """


        samples = []

        for episode in self.episodes:
            # Load laughter intervals (lightweight JSON only)
            with open(episode['label_path'], 'r') as f:
                laughter_data = json.load(f)
            with open(episode['turns_path'], 'r') as f:
                turns_data = json.load(f)

            # Compute episode duration and positive ratio
            num_frames = int(laughter_data['episode_duration'] * 12.5)
            change_points = [time_to_frame_index(turn['end']) for turn in turns_data['turns'][:-1]]
            

            speakers = [speaker_id for speaker_id in laughter_data['speaker_statistics'].keys()]

            for assign_idx in range(2):
                speaker = speakers[assign_idx]
                matched = match_speaker_to_assignment(
                    laughter_data['intervals'],
                    {'user': speakers[assign_idx], 'system': ''}
                )

                labels = np.zeros(num_frames, dtype=np.float32)
                for event in matched:
                    s = max(0, time_to_frame_index(event['prediction_interval_end'] - 10))
                    e = min(num_frames - 1, time_to_frame_index(event['prediction_interval_end']))
                    if event['event_start_inepisode'] - event['current_turn_start'] < 10:
                        labels[s:e + 1] = 1.0

                turn_idx = 0
                
                for t in range(3000, num_frames):
                    # Advance turn_idx while change point is before the start of the 3000-frame window
                    while turn_idx < len(change_points) and change_points[turn_idx] < (t - 3000):
                        turn_idx += 1

                    # Emit sample only if current change point falls within (t-3000, t)
                    if turn_idx < len(change_points) and change_points[turn_idx] < t:
                        samples.append(int(labels[t]))

        positive_ratio = sum(samples) / len(samples)
        negative_ratio = 1.0 - positive_ratio

        pos_weight = negative_ratio / positive_ratio

        logger.info(
            f"Class distribution: "
            f"{positive_ratio*100:.2f}% positive, "
            f"{negative_ratio*100:.2f}% negative"
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
        samples = []
        frames_per_episode = []

        for episode in self.episodes:
            # Load laughter intervals (lightweight JSON only)
            with open(episode['label_path'], 'r') as f:
                laughter_data = json.load(f)
            with open(episode['turns_path'], 'r') as f:
                turns_data = json.load(f)

            # Compute episode duration and positive ratio
            num_frames = int(laughter_data['episode_duration'] * 12.5) 
            change_points = [time_to_frame_index(turn['end']) for turn in turns_data['turns'][:-1]]
            

            speakers = [speaker_id for speaker_id in laughter_data['speaker_statistics'].keys()]

            for assign_idx in range(2):
                speaker = speakers[assign_idx]
                matched = match_speaker_to_assignment(
                    laughter_data['intervals'],
                    {'user': speakers[assign_idx], 'system': ''}
                )

                labels = np.zeros(num_frames, dtype=np.float32)
                for event in matched:
                    s = max(0, time_to_frame_index(event['prediction_interval_end'] - 10))
                    e = min(num_frames - 1, time_to_frame_index(event['prediction_interval_end']))
                    if event['event_start_inepisode'] - event['current_turn_start'] < 10:
                        labels[s:e + 1] = 1.0
                turn_idx = 0

                for t in range(3000, num_frames):
                    # Advance turn_idx while change point is before the start of the 3000-frame window
                    while turn_idx < len(change_points) and change_points[turn_idx] < (t - 3000):
                        turn_idx += 1

                    # Emit sample only if current change point falls within (t-3000, t)
                    if turn_idx < len(change_points) and change_points[turn_idx] < t:
                        samples.append(int(labels[t]))

        # Calculate statistics
        # Multiply by 2 for two speaker assignments per episode
        total_frames_with_assignments = len(samples)

        # Average positive ratio across episodes
        positive_ratio = sum(samples) / len(samples)
        negative_ratio = 1.0 - positive_ratio

        # Estimate counts
        num_positive = sum(samples)
        num_negative = len(samples) - sum(samples)


        
        return {
            'num_episodes': len(self.episodes),
            'num_frames': total_frames_with_assignments,
            'num_positive': num_positive,
            'num_negative': num_negative,
            'pos_ratio': float(positive_ratio),
            'avg_frames_per_episode': float(np.mean(frames_per_episode)) if frames_per_episode else 0.0,
            'min_frames_per_episode': int(np.min(frames_per_episode)) if frames_per_episode else 0,
            'max_frames_per_episode': int(np.max(frames_per_episode)) if frames_per_episode else 0,
            'balanced': False,
        }
