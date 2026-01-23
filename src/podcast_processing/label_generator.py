"""Label generation for laughter prediction."""

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List

import torch

logger = logging.getLogger(__name__)


class LabelGenerator:
    """Generate frame-level labels from laughter annotations.

    Creates binary labels for laughter prediction task, where frames
    before laughter events are labeled as positive.
    """

    def __init__(self, frame_rate: float = 12.5):
        """Initialize label generator.

        Args:
            frame_rate: Frame rate in Hz (default: 12.5 for Mimi encoder)
        """
        self.frame_rate = frame_rate

    def load_laughter_events(self, annotation_path: Path) -> List[Dict]:
        """Load laughter events from episode_event_speaker_mapping JSON.

        Args:
            annotation_path: Path to annotation JSON file

        Returns:
            List of laughter event dictionaries with speaker_id, start/end times
        """
        with open(annotation_path, 'r') as f:
            data = json.load(f)

        laughter_events = [
            event for event in data['events']
            if event.get('label_consolidated_vocab') == 'Laughter'
        ]

        logger.debug(f"Loaded {len(laughter_events)} laughter events from {annotation_path}")
        return laughter_events

    def create_labels_prediction(
        self,
        laughter_events: List[Dict],
        num_frames: int,
        user_speaker_id: str,
        shift_frames: int = 1
    ) -> torch.Tensor:
        """Create prediction labels (frames before laughter).

        Labels frames that are N frames BEFORE laughter events.
        If frame t has label 1, it means frame t+shift_frames will have laughter.

        Args:
            laughter_events: List of event dicts from load_laughter_events()
            num_frames: Total number of frames T
            user_speaker_id: Which speaker is 'user' (e.g., 'SPEAKER_00')
            shift_frames: How many frames ahead to predict

        Returns:
            Binary labels tensor [T] (int64)
        """
        labels = torch.zeros(num_frames, dtype=torch.int64)

        # Filter events for user speaker
        user_events = [
            e for e in laughter_events
            if e.get('speaker_id') == user_speaker_id
        ]

        logger.debug(f"Processing {len(user_events)} laughter events for {user_speaker_id}")

        for event in user_events:
            start_time = event['event_start_inepisode']
            end_time = event['event_end_inepisode']

            start_frame = int(start_time * self.frame_rate)
            end_frame = int(end_time * self.frame_rate)

            # Shift labels earlier (predict future frames)
            start_frame = start_frame - shift_frames
            end_frame = end_frame - shift_frames

            # Clamp to valid range
            start_frame = max(0, start_frame)
            end_frame = min(num_frames, end_frame)

            if start_frame < end_frame:
                labels[start_frame:end_frame] = 1

        return labels

    def compute_label_statistics(self, labels: torch.Tensor) -> Dict:
        """Compute statistics for label tensor.

        Args:
            labels: Binary label tensor [T]

        Returns:
            Dictionary with num_frames, num_positive_frames, positive_rate
        """
        num_positive = int(labels.sum().item())
        num_frames = len(labels)

        return {
            'num_frames': num_frames,
            'num_positive_frames': num_positive,
            'positive_rate': float(num_positive / num_frames) if num_frames > 0 else 0.0
        }


class AnnotationLabelGenerator:
    """Generate frame-level labels from episode_annotations CSV (no speaker assignment).

    Creates binary labels for event prediction task without distinguishing speakers.
    All laughter events are included regardless of speaker.
    """

    def __init__(self, frame_rate: float = 12.5):
        """Initialize annotation label generator.

        Args:
            frame_rate: Frame rate in Hz (default: 12.5 for Mimi encoder)
        """
        self.frame_rate = frame_rate

    def load_laughter_events(self, annotation_path: Path) -> List[Dict]:
        """Load laughter events from episode_annotations CSV.

        Args:
            annotation_path: Path to annotation CSV file

        Returns:
            List of laughter event dictionaries with start/end times (no speaker_id)
        """
        laughter_events = []

        with open(annotation_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('label_consolidated_vocab') == 'Laughter':
                    laughter_events.append({
                        'event_start_inepisode': float(row['event_start_inepisode']),
                        'event_end_inepisode': float(row['event_end_inepisode']),
                        'label_consolidated_vocab': row['label_consolidated_vocab'],
                        'clip_name': row.get('clip_name'),
                        'pfID': row.get('pfID'),
                    })

        logger.debug(f"Loaded {len(laughter_events)} laughter events from {annotation_path}")
        return laughter_events

    def create_labels_prediction(
        self,
        laughter_events: List[Dict],
        num_frames: int,
        shift_frames: int = 1
    ) -> torch.Tensor:
        """Create prediction labels (frames before laughter).

        Labels frames that are N frames BEFORE laughter events.
        If frame t has label 1, it means frame t+shift_frames will have laughter.
        Unlike LabelGenerator, this does not filter by speaker.

        Args:
            laughter_events: List of event dicts from load_laughter_events()
            num_frames: Total number of frames T
            shift_frames: How many frames ahead to predict

        Returns:
            Binary labels tensor [T] (int64)
        """
        labels = torch.zeros(num_frames, dtype=torch.int64)

        logger.debug(f"Processing {len(laughter_events)} laughter events (all speakers)")

        for event in laughter_events:
            start_time = event['event_start_inepisode']
            end_time = event['event_end_inepisode']

            start_frame = int(start_time * self.frame_rate)
            end_frame = int(end_time * self.frame_rate)

            # Shift labels earlier (predict future frames)
            start_frame = start_frame - shift_frames
            end_frame = end_frame - shift_frames

            # Clamp to valid range
            start_frame = max(0, start_frame)
            end_frame = min(num_frames, end_frame)

            if start_frame < end_frame:
                labels[start_frame:end_frame] = 1

        return labels

    def compute_label_statistics(self, labels: torch.Tensor) -> Dict:
        """Compute statistics for label tensor.

        Args:
            labels: Binary label tensor [T]

        Returns:
            Dictionary with num_frames, num_positive_frames, positive_rate
        """
        num_positive = int(labels.sum().item())
        num_frames = len(labels)

        return {
            'num_frames': num_frames,
            'num_positive_frames': num_positive,
            'positive_rate': float(num_positive / num_frames) if num_frames > 0 else 0.0
        }
