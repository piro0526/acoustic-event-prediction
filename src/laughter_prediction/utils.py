"""Utility functions for laughter event prediction."""

import logging
from pathlib import Path
from typing import List, Dict, Any
import torch

# Frame rate from Mimi compression (12.5 Hz)
FRAME_RATE = 12.5


def time_to_frame_index(time_seconds: float) -> int:
    """Convert time in seconds to frame index.

    Args:
        time_seconds: Time in seconds

    Returns:
        Frame index (0-based)
    """
    return int(time_seconds * FRAME_RATE)


def frame_index_to_time(frame_idx: int) -> float:
    """Convert frame index to time in seconds.

    Args:
        frame_idx: Frame index (0-based)

    Returns:
        Time in seconds
    """
    return frame_idx / FRAME_RATE


def match_speaker_to_assignment(
    laughter_intervals: List[Dict[str, Any]],
    assignment_metadata: Dict[str, str]
) -> List[Dict[str, Any]]:
    """Filter laughter events for the user speaker in this assignment.

    Args:
        laughter_intervals: List of laughter event dicts from JSON
        assignment_metadata: Dict with 'user' and 'system' speaker IDs

    Returns:
        Filtered list of events for user speaker
    """
    user_speaker = assignment_metadata['user']
    return [
        event for event in laughter_intervals
        if event['speaker_id'] == user_speaker
    ]


def setup_logging(log_dir: Path, rank: int = 0) -> logging.Logger:
    """Setup logging configuration.

    Args:
        log_dir: Directory to save log files
        rank: Process rank (for distributed training)

    Returns:
        Logger instance
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create logger
    logger = logging.getLogger('laughter_prediction')
    logger.setLevel(logging.DEBUG if rank == 0 else logging.INFO)

    # Remove existing handlers
    logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler (only for master process)
    if rank == 0:
        file_handler = logging.FileHandler(log_dir / 'training.log')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    path: Path,
    scheduler: torch.optim.lr_scheduler._LRScheduler = None
) -> None:
    """Save model checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state to save
        epoch: Current epoch
        metrics: Training metrics
        path: Path to save checkpoint
        scheduler: Optional learning rate scheduler
    """
    # Handle DDP wrapped models
    model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }

    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler = None,
    device: str = 'cuda'
) -> Dict[str, Any]:
    """Load model checkpoint.

    Args:
        path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        device: Device to load checkpoint to

    Returns:
        Dictionary with epoch and metrics
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    # Handle DDP wrapped models
    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return {
        'epoch': checkpoint.get('epoch', 0),
        'metrics': checkpoint.get('metrics', {})
    }
