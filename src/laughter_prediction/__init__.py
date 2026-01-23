"""Laughter event prediction module for acoustic event detection in podcasts."""

from .dataset import LaughterDataset
from .iterable_dataset import LaughterIterableDataset
from .model import LaughterPredictor
from .metrics import compute_metrics, compute_confusion_matrix
from .utils import time_to_frame_index, frame_index_to_time

__all__ = [
    'LaughterDataset',
    'LaughterIterableDataset',
    'LaughterPredictor',
    'compute_metrics',
    'compute_confusion_matrix',
    'time_to_frame_index',
    'frame_index_to_time',
]
