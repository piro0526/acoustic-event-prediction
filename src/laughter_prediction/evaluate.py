"""Evaluation script for laughter event prediction."""

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import DataLoader

from laughter_prediction.dataset import LaughterDataset
from laughter_prediction.model import LaughterPredictor
from laughter_prediction.metrics import (
    compute_metrics,
    compute_confusion_matrix,
    find_optimal_threshold,
    print_metrics_report
)
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from laughter_prediction.utils import load_checkpoint, setup_logging

logger = logging.getLogger(__name__)


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
    dataset: LaughterDataset = None
) -> tuple:
    """Evaluate model on test set.

    Args:
        model: Model to evaluate
        dataloader: Test dataloader
        device: Device to evaluate on
        dataset: Optional dataset for metadata access

    Returns:
        Tuple of (predictions, labels, metadata)
    """
    model.eval()
    all_preds = []
    all_labels = []

    # Get model's dtype
    model_dtype = next(model.parameters()).dtype

    with torch.no_grad():
        for batch in dataloader:
            features = batch['features'].to(device, dtype=model_dtype)
            labels = batch['label'].cpu().numpy()

            logits = model(features)
            probs = torch.sigmoid(logits).cpu().numpy()

            all_preds.append(probs)
            all_labels.append(labels)

    predictions = np.concatenate(all_preds, axis=0).squeeze()
    labels = np.concatenate(all_labels, axis=0).squeeze()

    # Get metadata from dataset if available
    metadata = dataset.metadata if dataset is not None else None

    return predictions, labels, metadata


def plot_confusion_matrix(
    tn: int,
    fp: int,
    fn: int,
    tp: int,
    output_path: Path
) -> None:
    """Plot and save confusion matrix.

    Args:
        tn: True negatives
        fp: False positives
        fn: False negatives
        tp: True positives
        output_path: Path to save plot
    """
    cm = np.array([[tn, fp], [fn, tp]])

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Negative', 'Positive'],
        yticklabels=['Negative', 'Positive']
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved confusion matrix to {output_path}")


def plot_roc_curve(
    predictions: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    optimal_threshold: float = None
) -> None:
    """Plot and save ROC curve.

    Args:
        predictions: Predicted probabilities
        labels: Ground truth binary labels
        output_path: Path to save plot
        optimal_threshold: Optional threshold to mark on the curve
    """
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(labels, predictions)
    roc_auc = auc(fpr, tpr)

    # Create plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')

    # Mark optimal threshold point if provided
    if optimal_threshold is not None:
        # Find the point on ROC curve closest to the optimal threshold
        idx = np.argmin(np.abs(thresholds - optimal_threshold))
        plt.scatter(fpr[idx], tpr[idx], color='red', s=100, zorder=5,
                   label=f'Optimal threshold = {optimal_threshold:.4f}')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved ROC curve to {output_path}")


def plot_pr_curve(
    predictions: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    optimal_threshold: float = None
) -> None:
    """Plot and save Precision-Recall curve.

    Args:
        predictions: Predicted probabilities
        labels: Ground truth binary labels
        output_path: Path to save plot
        optimal_threshold: Optional threshold to mark on the curve
    """
    # Compute PR curve
    precision, recall, thresholds = precision_recall_curve(labels, predictions)
    average_precision = average_precision_score(labels, predictions)

    # Create plot
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2,
             label=f'PR curve (AP = {average_precision:.4f})')

    # Baseline (random classifier): precision = positive_rate
    positive_rate = np.mean(labels)
    plt.axhline(y=positive_rate, color='navy', lw=2, linestyle='--',
                label=f'Random classifier (AP = {positive_rate:.4f})')

    # Mark optimal threshold point if provided
    if optimal_threshold is not None:
        # Find the point on PR curve closest to the optimal threshold
        # Note: thresholds array is one element shorter than precision/recall
        idx = np.argmin(np.abs(thresholds - optimal_threshold))
        plt.scatter(recall[idx], precision[idx], color='red', s=100, zorder=5,
                   label=f'Optimal threshold = {optimal_threshold:.4f}')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='best')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved PR curve to {output_path}")


def save_predictions(
    predictions: np.ndarray,
    labels: np.ndarray,
    metadata: list,
    output_dir: Path
) -> None:
    """Save predictions to separate CSV files for each episode.

    Args:
        predictions: Predicted probabilities
        labels: Ground truth labels
        metadata: List of metadata dicts (one per frame)
        output_dir: Directory to save CSV files
    """
    # Extract metadata fields
    episodes = [m['episode_name'] for m in metadata]
    frames = [m['frame_idx'] for m in metadata]
    user_speakers = [m['user_id'] for m in metadata]

    # Create dataframe
    df = pd.DataFrame({
        'episode': episodes,
        'frame': frames,
        'time_ms': np.array(frames) * 80,  # 80ms per frame at 12.5Hz
        'user_speaker': user_speakers,
        'prediction': predictions,
        'label': labels
    })

    # Sort by episode, then by user_speaker, then by frame
    df = df.sort_values(['episode', 'user_speaker', 'frame']).reset_index(drop=True)

    # Create predictions subdirectory
    predictions_dir = output_dir / 'predictions'
    predictions_dir.mkdir(exist_ok=True)

    # Save each episode to a separate CSV file
    episode_groups = df.groupby('episode')
    num_episodes = len(episode_groups)

    for episode_name, episode_df in episode_groups:
        # Drop the episode column since it's redundant in per-episode files
        episode_df = episode_df.drop(columns=['episode'])

        # Save to file
        episode_file = predictions_dir / f'{episode_name}.csv'
        episode_df.to_csv(episode_file, index=False)

    logger.info(f"Saved predictions for {num_episodes} episodes to {predictions_dir}/")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description='Evaluate laughter event prediction model'
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--features_dir',
        type=str,
        default='output/laughter/features',
        help='Directory containing features/[split]/ with features.npy and labels.npy'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='output/laughter_prediction/results',
        help='Directory to save results'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=512,
        help='Batch size for evaluation'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to evaluate on'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['train', 'test', 'validation'],
        help='Which split to evaluate'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=None,
        help='Use a fixed threshold instead of optimizing on validation set (e.g., 0.65)'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=0,
        help='Number of data loading workers (default: 0 for main process only)'
    )

    args = parser.parse_args()

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(output_dir)
    logger.info(f"Evaluating checkpoint: {args.checkpoint}")
    logger.info(f"Arguments: {vars(args)}")

    # Load test dataset
    logger.info(f"Loading {args.split} dataset...")
    test_dataset = LaughterDataset(
        Path(args.features_dir),
        split=args.split,
        shuffle=False
    )
    logger.info(f"Test dataset loaded with {len(test_dataset)} frames")

    # Print test dataset statistics
    test_stats = test_dataset.get_statistics()
    logger.info(f"Test dataset statistics: {test_stats}")

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,  # Set to 0 to avoid shared memory issues with mmap
        pin_memory=True
    )

    # Load model
    logger.info("\nLoading model...")
    model = LaughterPredictor().to(args.device)

    checkpoint_info = load_checkpoint(
        Path(args.checkpoint),
        model,
        device=args.device
    )

    logger.info(f"Loaded checkpoint from epoch {checkpoint_info['epoch']}")
    logger.info(f"Checkpoint metrics: {checkpoint_info['metrics']}")

    # Determine optimal threshold
    if args.threshold is not None:
        # Use user-specified threshold
        logger.info("\n" + "="*60)
        logger.info(f"Using user-specified threshold: {args.threshold:.4f}")
        logger.info("="*60)
        optimal_threshold = args.threshold
        use_validation = False
        val_metrics_optimal = None
        optimal_f1 = None
    else:
        # Load validation dataset for threshold optimization
        logger.info("\nLoading validation dataset for threshold optimization...")
        try:
            val_dataset = LaughterDataset(
                Path(args.features_dir),
                split='validation',
                shuffle=False
            )
            logger.info(f"Validation dataset loaded with {len(val_dataset)} frames")

            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,  # Set to 0 to avoid shared memory issues with mmap
                pin_memory=True
            )
            use_validation = True
        except FileNotFoundError:
            logger.warning("Validation dataset not found. Will use test set for threshold optimization (not recommended).")
            use_validation = False

        # Find optimal threshold on validation set
        if use_validation:
            logger.info("\n" + "="*60)
            logger.info("VALIDATION SET: Finding optimal threshold")
            logger.info("="*60)
            val_predictions, val_labels, _ = evaluate_model(
                model, val_loader, args.device, val_dataset
            )

            logger.info("Finding optimal threshold (maximizing F1 score on validation set)...")
            optimal_threshold, optimal_f1 = find_optimal_threshold(
                val_predictions, val_labels, metric='f1'
            )
            logger.info(f"Optimal threshold (from validation): {optimal_threshold:.4f}")
            logger.info(f"Validation F1 score at optimal threshold: {optimal_f1:.4f}")

            # Show validation metrics at this threshold
            val_metrics_optimal = compute_metrics(val_predictions, val_labels, threshold=optimal_threshold)
            logger.info("\nValidation metrics at optimal threshold:")
            print_metrics_report(val_metrics_optimal, prefix="Validation ")
        else:
            # Fallback: use test set (not recommended)
            logger.warning("Using test set for threshold optimization (not best practice)")
            test_predictions_for_threshold, test_labels_for_threshold, _ = evaluate_model(
                model, test_loader, args.device, test_dataset
            )
            optimal_threshold, optimal_f1 = find_optimal_threshold(
                test_predictions_for_threshold, test_labels_for_threshold, metric='f1'
            )
            logger.info(f"Optimal threshold: {optimal_threshold:.4f}")
            logger.info(f"F1 score at optimal threshold: {optimal_f1:.4f}")
            val_metrics_optimal = None

    # Evaluate on test set
    logger.info("\n" + "="*60)
    if args.threshold is not None:
        logger.info(f"TEST SET: Evaluation with user-specified threshold")
    else:
        logger.info(f"TEST SET: Evaluation with validation-optimized threshold")
    logger.info("="*60)
    predictions, labels, metadata = evaluate_model(
        model, test_loader, args.device, test_dataset
    )

    # Compute metrics with default threshold
    logger.info("\nTest metrics with threshold=0.5:")
    metrics = compute_metrics(predictions, labels, threshold=0.5)
    print_metrics_report(metrics, prefix="Test ")

    # Compute metrics with optimal threshold
    if args.threshold is not None:
        logger.info(f"\nTest metrics with user-specified threshold={optimal_threshold:.4f}:")
    else:
        logger.info(f"\nTest metrics with optimal threshold={optimal_threshold:.4f} (from validation):")
    optimal_metrics = compute_metrics(predictions, labels, threshold=optimal_threshold)
    print_metrics_report(optimal_metrics, prefix="Test ")

    logger.info(f"\nnum_preds:{len(predictions)}")
    logger.info(f"\nnum_labels:{len(labels)}")

    # Compute confusion matrix (with both thresholds)
    tn_05, fp_05, fn_05, tp_05 = compute_confusion_matrix(predictions, labels, threshold=0.5)
    tn_opt, fp_opt, fn_opt, tp_opt = compute_confusion_matrix(predictions, labels, threshold=optimal_threshold)

    logger.info(f"\nConfusion Matrix (threshold=0.5):")
    logger.info(f"TN: {tn_05}, FP: {fp_05}, FN: {fn_05}, TP: {tp_05}")

    logger.info(f"\nConfusion Matrix (threshold={optimal_threshold:.4f}):")
    logger.info(f"TN: {tn_opt}, FP: {fp_opt}, FN: {fn_opt}, TP: {tp_opt}")

    # Plot confusion matrices
    cm_path_05 = output_dir / 'confusion_matrix_threshold_0.5.png'
    plot_confusion_matrix(tn_05, fp_05, fn_05, tp_05, cm_path_05)

    cm_path_opt = output_dir / f'confusion_matrix_threshold_{optimal_threshold:.4f}.png'
    plot_confusion_matrix(tn_opt, fp_opt, fn_opt, tp_opt, cm_path_opt)

    # Plot ROC curve
    roc_path = output_dir / 'roc_curve.png'
    plot_roc_curve(predictions, labels, roc_path, optimal_threshold=optimal_threshold)

    # Plot PR curve
    pr_path = output_dir / 'pr_curve.png'
    plot_pr_curve(predictions, labels, pr_path, optimal_threshold=optimal_threshold)

    # Save results
    if args.threshold is not None:
        threshold_source = 'user_specified'
    elif use_validation:
        threshold_source = 'validation'
    else:
        threshold_source = 'test'

    results = {
        'checkpoint': str(args.checkpoint),
        'split': args.split,
        'test_dataset_statistics': test_stats,
        'optimal_threshold_source': threshold_source,
        'optimal_threshold': float(optimal_threshold),
        'optimal_threshold_metric': 'f1' if args.threshold is None else 'user_specified',
        'test_metrics_threshold_0.5': metrics,
        'test_metrics_optimal_threshold': optimal_metrics,
        'confusion_matrix_threshold_0.5': {
            'tn': int(tn_05),
            'fp': int(fp_05),
            'fn': int(fn_05),
            'tp': int(tp_05)
        },
        'confusion_matrix_optimal_threshold': {
            'tn': int(tn_opt),
            'fp': int(fp_opt),
            'fn': int(fn_opt),
            'tp': int(tp_opt)
        }
    }

    # Add validation metrics if available
    if use_validation and val_metrics_optimal is not None:
        results['validation_metrics_optimal_threshold'] = val_metrics_optimal
        results['validation_f1_at_optimal_threshold'] = float(optimal_f1)

    results_path = output_dir / f'{args.split}_metrics.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results to {results_path}")

    # Save predictions (only if metadata is available)
    if metadata is not None:
        save_predictions(predictions, labels, metadata, output_dir)
    else:
        logger.info("Skipping detailed predictions CSV (metadata not available)")

    logger.info("\nEvaluation completed!")


if __name__ == '__main__':
    main()
