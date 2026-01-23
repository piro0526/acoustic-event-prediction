#!/usr/bin/env python3
"""
Create or recreate labels for laughter detection/prediction.

This script generates labels.npy from annotations without re-extracting features.
Useful for experimenting with different labeling strategies.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from tqdm import tqdm

from podcast_processing.utils import setup_logging

logger = logging.getLogger(__name__)

FRAME_RATE = 12.5  # Hz


def time_to_frame_index(time_seconds: float, frame_rate: float = FRAME_RATE) -> int:
    """Convert time in seconds to frame index."""
    return int(time_seconds * frame_rate)


def create_frame_labels_detection(
    laughter_events: List[Dict[str, Any]],
    num_frames: int,
    user_speaker_id: str,
    frame_rate: float = FRAME_RATE
) -> np.ndarray:
    """Create labels for current-frame detection.

    Labels frames that OVERLAP with laughter events.

    Returns:
        Binary array [T] where 1 = laughter frame, 0 = non-laughter
    """
    labels = np.zeros(num_frames, dtype=np.int32)

    user_events = [
        event for event in laughter_events
        if event.get('speaker_id') == user_speaker_id
    ]

    for event in user_events:
        start_time = event['event_start_inepisode']
        end_time = event['event_end_inepisode']

        start_frame = int(start_time * frame_rate)
        end_frame = int(end_time * frame_rate)

        start_frame = max(0, start_frame)
        end_frame = min(num_frames, end_frame)

        if start_frame < end_frame:
            labels[start_frame:end_frame] = 1

    return labels


def create_frame_labels_prediction(
    laughter_events: List[Dict[str, Any]],
    num_frames: int,
    user_speaker_id: str,
    frame_rate: float = FRAME_RATE,
    shift_frames: int = 1
) -> np.ndarray:
    """Create labels for next-frame prediction.

    Labels frames that are N frames BEFORE laughter events.
    If frame t has label 1, it means frame t+N will have laughter.

    Args:
        shift_frames: Number of frames to shift labels earlier (default: 1)

    Returns:
        Binary array [T] where 1 = next frame will have laughter
    """
    labels = np.zeros(num_frames, dtype=np.int32)

    user_events = [
        event for event in laughter_events
        if event.get('speaker_id') == user_speaker_id
    ]

    for event in user_events:
        start_time = event['event_start_inepisode']
        end_time = event['event_end_inepisode']

        start_frame = int(start_time * frame_rate)
        end_frame = int(end_time * frame_rate)

        # Shift labels earlier (predict future frames)
        start_frame = start_frame - shift_frames
        end_frame = end_frame - shift_frames

        start_frame = max(0, start_frame)
        end_frame = min(num_frames, end_frame)

        if start_frame < end_frame:
            labels[start_frame:end_frame] = 1

    return labels


def load_laughter_events(json_path: Path) -> List[Dict[str, Any]]:
    """Load laughter events from annotation JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Filter for laughter events only
    laughter_events = [
        event for event in data['events']
        if event.get('label_consolidated_vocab') == 'Laughter'
    ]

    return laughter_events


def create_labels_for_split(
    transformer_dir: Path,
    annotations_dir: Path,
    metadata_path: Path,
    split: str,
    output_dir: Path,
    mode: str = 'detection',
    shift_frames: int = 1
) -> None:
    """Create labels for a dataset split.

    Args:
        transformer_dir: Directory containing transformer_outs/{split}/*.pt
        annotations_dir: Directory containing episode_event_speaker_mapping/{split}/*.json
        metadata_path: Path to existing metadata.json (contains episode ordering)
        split: Dataset split ('train', 'validation', or 'test')
        output_dir: Directory to save labels
        mode: 'detection' (overlap) or 'prediction' (shift forward)
        shift_frames: Number of frames to shift for prediction mode
    """
    logger.info(f"Creating {mode} labels for {split} split")

    pt_dir = transformer_dir / split
    json_dir = annotations_dir / split
    output_split_dir = output_dir / split
    output_split_dir.mkdir(parents=True, exist_ok=True)

    if not pt_dir.exists():
        raise FileNotFoundError(f"Transformer directory not found: {pt_dir}")
    if not json_dir.exists():
        raise FileNotFoundError(f"Annotations directory not found: {json_dir}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    # Load existing metadata to get episode ordering
    logger.info(f"Loading metadata from {metadata_path}")
    with open(metadata_path, 'r') as f:
        existing_metadata = json.load(f)

    # Get unique episodes in original order
    episode_names = []
    seen = set()
    for meta in existing_metadata:
        ep = meta['episode_name']
        if ep not in seen:
            episode_names.append(ep)
            seen.add(ep)

    logger.info(f"Found {len(episode_names)} episodes in metadata")

    # Process episodes in same order
    all_labels = []

    for episode_name in tqdm(episode_names, desc=f"Creating {mode} labels for {split}"):
        pt_path = pt_dir / f"{episode_name}.pt"
        json_path = json_dir / f"{episode_name}.json"

        if not pt_path.exists():
            logger.warning(f"Transformer output not found: {pt_path}")
            continue
        if not json_path.exists():
            logger.warning(f"Annotation not found: {json_path}")
            continue

        try:
            # Load transformer output to get shape
            data = torch.load(pt_path)
            transformer_out = data['transformer_out']  # [2, T, 4096]
            metadata = data['metadata']
            num_frames = transformer_out.shape[1]

            # Load laughter events
            laughter_events = load_laughter_events(json_path)

            # Create labels for both assignments
            for assign_idx in range(2):
                assignment = metadata['assignments'][assign_idx]
                user_speaker = assignment['user']

                if mode == 'detection':
                    labels = create_frame_labels_detection(
                        laughter_events, num_frames, user_speaker
                    )
                elif mode == 'prediction':
                    labels = create_frame_labels_prediction(
                        laughter_events, num_frames, user_speaker, shift_frames=shift_frames
                    )
                else:
                    raise ValueError(f"Unknown mode: {mode}")

                all_labels.append(labels)

        except Exception as e:
            logger.error(f"Error processing episode {episode_name}: {e}")
            continue

    # Concatenate all labels
    labels_array = np.concatenate(all_labels, axis=0)

    logger.info(f"Created labels: {labels_array.shape}")
    logger.info(f"Positive: {np.sum(labels_array)} ({np.sum(labels_array)/len(labels_array)*100:.2f}%)")

    # Save labels
    output_path = output_split_dir / 'labels.npy'
    np.save(output_path, labels_array)
    logger.info(f"Saved labels to {output_path}")

    # Save label config
    config = {
        'mode': mode,
        'shift_frames': shift_frames if mode == 'prediction' else None,
        'split': split,
        'num_frames': len(labels_array),
        'num_positive': int(np.sum(labels_array)),
        'positive_rate': float(np.sum(labels_array) / len(labels_array))
    }
    config_path = output_split_dir / 'label_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"Saved label config to {config_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Create labels for laughter detection/prediction'
    )
    parser.add_argument(
        '--transformer_dir',
        type=Path,
        default=Path('output/transformer_outs'),
        help='Directory containing transformer outputs'
    )
    parser.add_argument(
        '--annotations_dir',
        type=Path,
        default=Path('data/PodcastFillers/metadata/episode_event_speaker_mapping'),
        help='Directory containing annotations'
    )
    parser.add_argument(
        '--features_dir',
        type=Path,
        default=Path('output/laughter/features'),
        help='Directory containing existing features (for metadata)'
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        default=Path('output/laughter/features'),
        help='Output directory for labels'
    )
    parser.add_argument(
        '--split',
        type=str,
        required=True,
        choices=['train', 'validation', 'test'],
        help='Dataset split'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='detection',
        choices=['detection', 'prediction'],
        help='Label mode: detection (overlap) or prediction (shift)'
    )
    parser.add_argument(
        '--shift_frames',
        type=int,
        default=1,
        help='Number of frames to shift for prediction mode (default: 1)'
    )

    args = parser.parse_args()

    # Setup logging
    log_dir = args.output_dir / args.split
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_dir / f'create_labels_{args.mode}.log')

    logger.info("Starting label creation")
    logger.info(f"Arguments: {vars(args)}")

    # Get metadata path
    metadata_path = args.features_dir / args.split / 'metadata.json'

    create_labels_for_split(
        transformer_dir=args.transformer_dir,
        annotations_dir=args.annotations_dir,
        metadata_path=metadata_path,
        split=args.split,
        output_dir=args.output_dir,
        mode=args.mode,
        shift_frames=args.shift_frames
    )

    logger.info("Label creation completed!")


if __name__ == '__main__':
    main()
