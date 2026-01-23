"""Metrics computation for laughter event prediction."""

import numpy as np
from typing import Dict, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    balanced_accuracy_score,
    average_precision_score,
    matthews_corrcoef
)


def compute_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """Compute classification metrics.

    Args:
        predictions: Predicted probabilities [N]
        labels: Ground truth binary labels [N]
        threshold: Decision threshold for binary classification

    Returns:
        Dictionary with metrics: accuracy, precision, recall, f1, auroc, auprc,
        balanced_accuracy, detection_rate, youden_j_index, mcc
    """
    # Convert probabilities to binary predictions
    binary_preds = (predictions >= threshold).astype(int)

    # Compute confusion matrix for Youden's J Index
    tn, fp, fn, tp = confusion_matrix(labels, binary_preds, labels=[0, 1]).ravel()

    # Calculate recall and specificity for Youden's J Index
    recall = recall_score(labels, binary_preds, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Compute metrics
    metrics = {
        'accuracy': accuracy_score(labels, binary_preds),
        'precision': precision_score(labels, binary_preds, zero_division=0),
        'recall': recall,
        'f1': f1_score(labels, binary_preds, zero_division=0),
        'balanced_accuracy': balanced_accuracy_score(labels, binary_preds),
        'detection_rate': recall,  # Alias for recall
        'youden_j_index': recall + specificity - 1.0,
        'mcc': matthews_corrcoef(labels, binary_preds),
    }

    # Compute AUROC (requires probabilities, not binary predictions)
    try:
        metrics['auroc'] = roc_auc_score(labels, predictions)
    except ValueError:
        # Can fail if only one class is present in labels
        metrics['auroc'] = 0.0

    # Compute AUPRC (requires probabilities, not binary predictions)
    try:
        metrics['auprc'] = average_precision_score(labels, predictions)
    except ValueError:
        # Can fail if only one class is present in labels
        metrics['auprc'] = 0.0

    return metrics


def compute_confusion_matrix(
    predictions: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5
) -> Tuple[int, int, int, int]:
    """Compute confusion matrix.

    Args:
        predictions: Predicted probabilities [N]
        labels: Ground truth binary labels [N]
        threshold: Decision threshold

    Returns:
        Tuple of (TN, FP, FN, TP)
    """
    binary_preds = (predictions >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, binary_preds, labels=[0, 1]).ravel()
    return int(tn), int(fp), int(fn), int(tp)


def find_optimal_threshold(
    predictions: np.ndarray,
    labels: np.ndarray,
    metric: str = 'f1',
    num_thresholds: int = 100
) -> Tuple[float, float]:
    """Find optimal decision threshold by grid search.

    Args:
        predictions: Predicted probabilities [N]
        labels: Ground truth binary labels [N]
        metric: Metric to optimize ('f1', 'precision', 'recall', 'accuracy', 'youden_j')
        num_thresholds: Number of thresholds to try

    Returns:
        Tuple of (best_threshold, best_metric_value)
    """
    thresholds = np.linspace(0, 1, num_thresholds)
    best_threshold = 0.5
    best_score = -1.0 if metric == 'youden_j' else 0.0  # Youden's J can be negative

    for threshold in thresholds:
        binary_preds = (predictions >= threshold).astype(int)

        if metric == 'f1':
            score = f1_score(labels, binary_preds, zero_division=0)
        elif metric == 'precision':
            score = precision_score(labels, binary_preds, zero_division=0)
        elif metric == 'recall':
            score = recall_score(labels, binary_preds, zero_division=0)
        elif metric == 'accuracy':
            score = accuracy_score(labels, binary_preds)
        elif metric == 'youden_j':
            # Youden's J Index = Sensitivity + Specificity - 1
            tn, fp, fn, tp = confusion_matrix(labels, binary_preds, labels=[0, 1]).ravel()
            sensitivity = recall_score(labels, binary_preds, zero_division=0)
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            score = sensitivity + specificity - 1.0
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold, best_score


def print_metrics_report(metrics: Dict[str, float], prefix: str = "") -> None:
    """Print formatted metrics report.

    Args:
        metrics: Dictionary of metrics
        prefix: Prefix for metric names (e.g., "train_", "val_")
    """
    print(f"\n{prefix}Metrics Report:")
    print("-" * 50)
    for key, value in metrics.items():
        print(f"{key:>12s}: {value:.4f}")
    print("-" * 50)
