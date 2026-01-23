"""Training script for laughter event prediction with multi-GPU support."""

import argparse
import logging
import os
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from laughter_prediction.dataset import LaughterDataset
from laughter_prediction.model import LaughterPredictor
from laughter_prediction.metrics import compute_metrics, print_metrics_report
from laughter_prediction.utils import setup_logging, save_checkpoint, load_checkpoint
from laughter_prediction.focal_loss import FocalLoss, AdaptiveFocalLoss


logger = logging.getLogger(__name__)


def initialize_bias_for_prior(model: nn.Module, prior_prob: float):
    """Initialize final layer bias to output prior probability.

    For binary classification with sigmoid activation, to get output
    probability p, we need bias b such that sigmoid(b) = p.
    This gives: b = log(p / (1 - p))

    Args:
        model: Model (possibly wrapped in DDP)
        prior_prob: Prior probability of positive class (e.g., 0.002)
    """
    import math

    # Handle DDP wrapped model
    if isinstance(model, DDP):
        actual_model = model.module
    else:
        actual_model = model

    # Calculate bias value for prior probability
    # sigmoid(bias) = prior_prob
    # bias = log(prior_prob / (1 - prior_prob))
    bias_value = math.log(prior_prob / (1 - prior_prob))

    # Find the final linear layer
    if hasattr(actual_model, 'classifier'):
        classifier = actual_model.classifier

        # Handle both simple Linear and Sequential cases
        if isinstance(classifier, nn.Linear):
            # Simple linear classifier
            final_layer = classifier
        elif isinstance(classifier, nn.Sequential):
            # MLP classifier - get last linear layer
            final_layer = classifier[-1]
        else:
            logger.warning(f"Unknown classifier type: {type(classifier)}")
            return

        # Initialize bias
        if isinstance(final_layer, nn.Linear) and final_layer.bias is not None:
            nn.init.constant_(final_layer.bias, bias_value)
            logger.info(f"Initialized final layer bias to {bias_value:.4f} "
                       f"(for prior probability {prior_prob:.6f})")
        else:
            logger.warning("Could not initialize bias: final layer has no bias parameter")


def setup_distributed():
    """Initialize distributed training if using torchrun.

    Returns:
        Tuple of (rank, local_rank, world_size, is_master)
    """
    # Check if running under torchrun
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # Initialize process group if not already initialized
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl')

        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))

        # Set GPU device
        torch.cuda.set_device(local_rank)

        is_master = (rank == 0)
        return rank, local_rank, world_size, is_master
    else:
        # Single GPU or CPU
        return 0, 0, 1, True


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    world_size: int,
    rank: int = 0,
    writer: SummaryWriter = None,
    epoch: int = 0
) -> float:
    """Train for one epoch.

    Args:
        model: Model to train
        dataloader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        world_size: Number of GPUs
        rank: Process rank (for progress bar display)
        writer: TensorBoard writer
        epoch: Current epoch number

    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    # Create progress bar only for rank 0
    pbar = tqdm(dataloader, desc="Training", disable=(rank != 0))

    for batch_idx, batch in enumerate(pbar):
        features = batch['features'].to(device, dtype=torch.float32, non_blocking=True)
        labels = batch['label'].to(device, dtype=torch.float32, non_blocking=True)

        optimizer.zero_grad()

        logits = model(features)
        loss = criterion(logits, labels)

        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        # Log to TensorBoard
        if writer is not None and rank == 0:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('train/batch_loss', loss.item(), global_step)
            writer.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], global_step)

        # Update progress bar with current loss
        if rank == 0:
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    # Aggregate loss across GPUs
    if world_size > 1:
        loss_tensor = torch.tensor(avg_loss, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = loss_tensor.item() / world_size

    return avg_loss


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
    world_size: int,
    rank: int = 0,
    writer: SummaryWriter = None,
    epoch: int = 0
) -> dict:
    """Validate model.

    Args:
        model: Model to validate
        dataloader: Validation dataloader
        criterion: Loss function
        device: Device to validate on
        world_size: Number of GPUs
        rank: Process rank (for progress bar display)
        writer: TensorBoard writer
        epoch: Current epoch number

    Returns:
        Dictionary with loss and metrics
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_preds = []
    all_labels = []

    # Create progress bar only for rank 0
    pbar = tqdm(dataloader, desc="Validation", disable=(rank != 0))

    with torch.no_grad():
        for batch in pbar:
            features = batch['features'].to(device, dtype=torch.float32, non_blocking=True)
            labels = batch['label'].to(device, dtype=torch.float32, non_blocking=True)

            logits = model(features)
            loss = criterion(logits, labels)

            # Get probabilities
            probs = torch.sigmoid(logits)

            # Collect predictions and labels
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            total_loss += loss.item()
            num_batches += 1

            # Update progress bar with current loss
            if rank == 0:
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    # Aggregate loss across GPUs
    if world_size > 1:
        loss_tensor = torch.tensor(avg_loss, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = loss_tensor.item() / world_size

    # Concatenate all predictions and labels
    all_preds = np.concatenate(all_preds, axis=0).squeeze()
    all_labels = np.concatenate(all_labels, axis=0).squeeze()

    # Compute metrics
    metrics = compute_metrics(all_preds, all_labels)
    metrics['loss'] = avg_loss

    # Log to TensorBoard
    if writer is not None and rank == 0:
        writer.add_scalar('val/loss', avg_loss, epoch)
        writer.add_scalar('val/accuracy', metrics['accuracy'], epoch)
        writer.add_scalar('val/precision', metrics['precision'], epoch)
        writer.add_scalar('val/recall', metrics['recall'], epoch)
        writer.add_scalar('val/f1', metrics['f1'], epoch)
        # Use 'auroc' key (not 'auc') as returned by compute_metrics
        if 'auroc' in metrics:
            writer.add_scalar('val/auroc', metrics['auroc'], epoch)

    return metrics


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description='Train laughter event prediction model'
    )

    # Data arguments
    parser.add_argument(
        '--features_dir',
        type=str,
        default='output/laughter/features_concat',
        help='Directory containing pre-concatenated features and labels (*_features.npy, *_labels.npy)'
    )
    parser.add_argument(
        '--shift_frames',
        type=int,
        default=1,
        help='[DEPRECATED] Shift value is now read from metadata'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='output/laughter_prediction',
        help='Directory to save checkpoints and logs'
    )

    # Model arguments
    parser.add_argument(
        '--input_dim',
        type=int,
        default=4096,
        help='Input dimension'
    )
    parser.add_argument(
        '--hidden_dim',
        type=int,
        default=None,
        help='Hidden dimension for MLP variant (None for simple linear)'
    )

    # Training arguments
    parser.add_argument(
        '--batch_size',
        type=int,
        default=512,
        help='Batch size per GPU'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0.01,
        help='Weight decay'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of epochs'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of dataloader workers'
    )
    parser.add_argument(
        '--early_stopping_patience',
        type=int,
        default=10,
        help='Early stopping patience (epochs)'
    )

    # Device arguments
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to train on'
    )

    # Loss function arguments
    parser.add_argument(
        '--loss_type',
        type=str,
        default='bce',
        choices=['bce', 'focal', 'adaptive_focal'],
        help='Loss function to use: bce (BCEWithLogitsLoss), focal (FocalLoss), adaptive_focal (AdaptiveFocalLoss)'
    )
    parser.add_argument(
        '--focal_alpha',
        type=float,
        default=0.25,
        help='Alpha parameter for Focal Loss (ignored for adaptive_focal)'
    )
    parser.add_argument(
        '--focal_gamma',
        type=float,
        default=2.0,
        help='Gamma parameter for Focal Loss (focusing parameter)'
    )

    args = parser.parse_args()

    # Setup distributed training
    rank, local_rank, world_size, is_master = setup_distributed()

    # Setup device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cpu')

    # Setup logging
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_dir = output_dir / 'logs'
    checkpoint_dir = output_dir / 'checkpoints'
    tensorboard_dir = output_dir / 'tensorboard'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(log_dir, rank=rank)

    # Initialize TensorBoard writer (only for rank 0)
    writer = None
    if is_master:
        writer = SummaryWriter(log_dir=str(tensorboard_dir))
        logger.info(f"TensorBoard logs will be saved to: {tensorboard_dir}")
        logger.info(f"To view logs, run: tensorboard --logdir={tensorboard_dir}")

    if is_master:
        logger.info(f"Starting training with {world_size} GPU(s)")
        logger.info(f"Arguments: {vars(args)}")

    # Create datasets
    if is_master:
        logger.info("Loading datasets...")

    train_dataset = LaughterDataset(
        concat_dir=Path(args.features_dir),
        split='train',
        shuffle=True,
        mmap_mode=None  # Load into memory
    )

    # Try to load validation set, fall back to using train if not available
    try:
        val_dataset = LaughterDataset(
            concat_dir=Path(args.features_dir),
            split='validation',
            shuffle=False,
            mmap_mode=None  # Load into memory
        )
    except FileNotFoundError:
        if is_master:
            logger.warning("Validation set not found, using train set for validation")
        val_dataset = train_dataset

    if is_master:
        logger.info(f"Train dataset: ~{len(train_dataset)} frames")
        logger.info(f"Validation dataset: ~{len(val_dataset)} frames")

        # Print dataset statistics
        train_stats = train_dataset.get_statistics()
        logger.info(f"Train statistics: {train_stats}")

    # Compute class weights
    if is_master:
        pos_weight = train_dataset.compute_class_weights()
        logger.info(f"Positive class weight: {pos_weight.item():.4f}")
    else:
        pos_weight = torch.tensor([1.0])

    # Broadcast pos_weight to all processes
    if world_size > 1:
        pos_weight = pos_weight.to(device)
        dist.broadcast(pos_weight, src=0)

    # Create dataloaders with DistributedSampler for DDP
    train_sampler = None
    val_sampler = None

    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),  # Shuffle only if not using sampler
        sampler=train_sampler,
        num_workers=args.num_workers,  # Set to 0 to avoid shared memory issues with mmap
        pin_memory=True,
        persistent_workers=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,  # Set to 0 to avoid shared memory issues with mmap
        pin_memory=True,
        persistent_workers=False
    )

    # Create model
    model = LaughterPredictor(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim
    ).to(device)

    if is_master:
        logger.info(f"Model: {model}")
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Number of parameters: {num_params:,}")

    # Wrap model with DDP for multi-GPU
    if world_size > 1:
        if is_master:
            logger.info("Initializing DDP...")
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        if is_master:
            logger.info("DDP initialized successfully")

    # Create optimizer and loss
    if is_master:
        logger.info("Creating optimizer and loss...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # Select loss function based on arguments
    if args.loss_type == 'bce':
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
        if is_master:
            logger.info(f"Using BCEWithLogitsLoss with pos_weight={pos_weight.item():.4f}")
    elif args.loss_type == 'focal':
        criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma, reduction='sum')
        if is_master:
            logger.info(f"Using FocalLoss with alpha={args.focal_alpha}, gamma={args.focal_gamma}")

        # Initialize bias for prior probability when using focal loss
        # Compute prior probability and broadcast to all processes
        if is_master:
            train_stats = train_dataset.get_statistics()
            prior_prob = torch.tensor([train_stats['pos_ratio']], device=device)
        else:
            prior_prob = torch.zeros(1, device=device)

        # Broadcast prior_prob to all processes if using DDP
        if world_size > 1:
            dist.broadcast(prior_prob, src=0)

        prior_prob_value = prior_prob.item()
        if is_master:
            logger.info(f"Initializing bias for prior probability: {prior_prob_value:.6f}")

        # Initialize bias on all processes
        initialize_bias_for_prior(model, prior_prob_value)

    elif args.loss_type == 'adaptive_focal':
        criterion = AdaptiveFocalLoss(gamma=args.focal_gamma, pos_weight=pos_weight.to(device))
        if is_master:
            computed_alpha = (pos_weight / (1 + pos_weight)).item()
            logger.info(f"Using AdaptiveFocalLoss with gamma={args.focal_gamma}, computed alpha={computed_alpha:.4f}")

    if is_master:
        logger.info("Optimizer and loss created")

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',  # Maximize F1 score
        factor=0.5,
        patience=3
    )


    # Training loop
    best_loss = float('inf')
    patience_counter = 0

    if is_master:
        logger.info("Starting training loop...")

    for epoch in range(args.epochs):
        if is_master:
            logger.info(f"Epoch {epoch+1}/{args.epochs} starting...")

        # Set epoch for DistributedSampler to ensure different shuffling per epoch
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, world_size, rank,
            writer=writer, epoch=epoch
        )

        # Validate
        val_metrics = validate(
            model, val_loader, criterion, device, world_size, rank,
            writer=writer, epoch=epoch
        )

        # Update learning rate scheduler
        scheduler.step(val_metrics['f1'])

        # Log metrics
        if is_master:
            logger.info(
                f"Epoch {epoch+1}/{args.epochs} - "
                f"Train Loss: {train_loss:.4f} - "
                f"Val Loss: {val_metrics['loss']:.4f} - "
                f"Val F1: {val_metrics['f1']:.4f} - "
                f"Val Precision: {val_metrics['precision']:.4f} - "
                f"Val Recall: {val_metrics['recall']:.4f}"
            )

            # Log epoch-level metrics to TensorBoard
            if writer is not None:
                writer.add_scalar('train/epoch_loss', train_loss, epoch)

            # Save checkpoint
            save_checkpoint(
                model, optimizer, epoch, val_metrics,
                checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pt',
                scheduler
            )

            # Save best model
            if val_metrics['loss'] < best_loss:
                best_loss = val_metrics['loss']
                save_checkpoint(
                    model, optimizer, epoch, val_metrics,
                    checkpoint_dir / 'best_model.pt',
                    scheduler
                )
                logger.info(f"New best Loss: {best_loss:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= args.early_stopping_patience:
                logger.info(
                    f"Early stopping triggered after {epoch+1} epochs "
                    f"(patience: {args.early_stopping_patience})"
                )
                break

    # Final evaluation
    if is_master:
        logger.info("\nTraining completed!")
        logger.info(f"Best validation Loss: {best_loss:.4f}")

        # Load best model and evaluate
        logger.info("\nEvaluating best model...")
        load_checkpoint(
            checkpoint_dir / 'best_model.pt',
            model,
            device=device
        )

        val_metrics = validate(
            model, val_loader, criterion, device, world_size, rank
        )

        print_metrics_report(val_metrics, prefix="Final Validation ")

    # Close TensorBoard writer
    if writer is not None:
        writer.close()
        logger.info("TensorBoard writer closed")

    # Cleanup distributed
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
