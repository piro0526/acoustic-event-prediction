"""Audio masking utilities for system speaker processing."""

import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def create_laughter_mask(
    laughter_events: List[Dict],
    duration: float,
    system_speaker_id: str,
    sample_rate: int = 24000,
    num_samples: Optional[int] = None
) -> Optional[np.ndarray]:
    """Create audio mask from laughter events for system speaker.

    Args:
        laughter_events: List of laughter event dictionaries
        duration: Episode duration in seconds
        system_speaker_id: Speaker ID to mask (e.g., 'SPEAKER_00')
        sample_rate: Audio sample rate in Hz (default: 24000)
        num_samples: Exact number of samples (if None, computed from duration)

    Returns:
        Boolean mask [num_samples] where True = mask (zero out), or None if no events
    """
    if num_samples is None:
        num_samples = int(duration * sample_rate)
    mask = np.zeros(num_samples, dtype=bool)

    # Filter events for system speaker
    system_events = [
        e for e in laughter_events
        if e.get('speaker_id') == system_speaker_id
    ]

    if not system_events:
        logger.debug(f"No laughter events found for {system_speaker_id}")
        return None

    logger.debug(f"Creating mask from {len(system_events)} laughter events for {system_speaker_id}")

    for event in system_events:
        start_time = event['event_start_inepisode']
        end_time = event['event_end_inepisode']

        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)

        # Clamp to valid range
        start_sample = max(0, start_sample)
        end_sample = min(num_samples, end_sample)

        if start_sample < end_sample:
            mask[start_sample:end_sample] = True

    masked_count = mask.sum()
    logger.debug(f"Mask covers {masked_count}/{num_samples} samples ({100.0 * masked_count / num_samples:.2f}%)")

    return mask


def create_random_mask(
    duration: float,
    mask_ratio: float = 0.1,
    sample_rate: int = 24000,
    seed: Optional[int] = None
) -> np.ndarray:
    """Create random audio mask.

    Args:
        duration: Episode duration in seconds
        mask_ratio: Fraction of samples to mask (default: 0.1 = 10%)
        sample_rate: Audio sample rate in Hz (default: 24000)
        seed: Random seed for reproducibility

    Returns:
        Boolean mask [num_samples] where True = mask (zero out)
    """
    if seed is not None:
        np.random.seed(seed)

    num_samples = int(duration * sample_rate)
    num_masked = int(num_samples * mask_ratio)

    # Create mask with specified ratio
    mask = np.zeros(num_samples, dtype=bool)
    masked_indices = np.random.choice(num_samples, size=num_masked, replace=False)
    mask[masked_indices] = True

    logger.debug(f"Random mask: {num_masked}/{num_samples} samples ({100.0 * mask_ratio:.2f}%)")

    return mask


def create_interval_mask(
    duration: float,
    intervals: List[tuple],
    sample_rate: int = 24000
) -> np.ndarray:
    """Create audio mask from time intervals.

    Args:
        duration: Episode duration in seconds
        intervals: List of (start_time, end_time) tuples in seconds
        sample_rate: Audio sample rate in Hz (default: 24000)

    Returns:
        Boolean mask [num_samples] where True = mask (zero out)
    """
    num_samples = int(duration * sample_rate)
    mask = np.zeros(num_samples, dtype=bool)

    for start_time, end_time in intervals:
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)

        # Clamp to valid range
        start_sample = max(0, start_sample)
        end_sample = min(num_samples, end_sample)

        if start_sample < end_sample:
            mask[start_sample:end_sample] = True

    masked_count = mask.sum()
    logger.debug(f"Interval mask: {masked_count}/{num_samples} samples ({100.0 * masked_count / num_samples:.2f}%)")

    return mask
