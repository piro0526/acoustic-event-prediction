"""Training script for laughter event prediction with IterableDataset and multi-GPU support."""

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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from laughter_prediction.iterable_dataset import LaughterIterableDataset
from laughter_prediction.model import LaughterPredictor
from laughter_prediction.metrics import compute_metrics, print_metrics_report
from laughter_prediction.utils import setup_logging, save_checkpoint, load_checkpoint
from laughter_prediction.focal_loss import FocalLoss, AdaptiveFocalLoss


logger = logging.getLogger(__name__)


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
        description='Train laughter event prediction model with IterableDataset'
    )

    # Data arguments
    parser.add_argument(
        '--transformer_dir',
        type=str,
        required=True,
        help='Directory containing transformer_outs/[split]/*.pt files'
    )
    parser.add_argument(
        '--labels_dir',
        type=str,
        required=True,
        help='Directory containing laughter_intervals/[split]/*.json files'
    )
    parser.add_argument(
        '--turns_dir',
        type=str,
        required=True,
        help='Directory containing laughter_turns/[split]/*.json files'
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
        default=20,
        help='Early stopping patience (epochs)'
    )
    parser.add_argument(
        '--balance_classes',
        action='store_true',
        help='Balance positive/negative samples by downsampling negatives to match positives'
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

    # Create datasets - Use IterableDataset for sequential access
    if is_master:
        logger.info("Loading datasets...")
        if args.balance_classes:
            logger.info("Class balancing enabled: negative samples will be downsampled to match positive samples")

    train_dataset = LaughterIterableDataset(
        Path(args.transformer_dir),
        Path(args.labels_dir),
        Path(args.turns_dir),
        split='train',
        shuffle=True,
        balance_classes=args.balance_classes
    )

    # Try to load validation set, fall back to using train if not available
    try:
        val_dataset = LaughterIterableDataset(
            Path(args.transformer_dir),
            Path(args.labels_dir),
            Path(args.turns_dir),
            split='validation',
            shuffle=False,
            balance_classes=False  # Don't balance validation set
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
        if args.balance_classes:
            # When classes are balanced, use pos_weight=1.0 (no weighting needed)
            pos_weight = torch.tensor([1.0])
            logger.info("Class balancing enabled: using pos_weight=1.0 (no weighting)")
        else:
            pos_weight = train_dataset.compute_class_weights()
            logger.info(f"Positive class weight: {pos_weight.item():.4f}")
    else:
        pos_weight = torch.tensor([1.0])

    # Broadcast pos_weight to all processes
    if world_size > 1:
        pos_weight = pos_weight.to(device)
        dist.broadcast(pos_weight, src=0)

    # Create dataloaders
    # Note: IterableDataset handles DDP splitting internally, so no sampler needed
    # shuffle must be False for IterableDataset (shuffling is done in the dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # IMPORTANT: Must be False for IterableDataset
        sampler=None,   # IMPORTANT: No sampler for IterableDataset
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # IMPORTANT: Must be False for IterableDataset
        sampler=None,   # IMPORTANT: No sampler for IterableDataset
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False
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
        criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
        if is_master:
            logger.info(f"Using FocalLoss with alpha={args.focal_alpha}, gamma={args.focal_gamma}")
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
    best_loss = 100
    patience_counter = 0

    if is_master:
        logger.info("Starting training loop...")

    for epoch in range(args.epochs):
        if is_master:
            logger.info(f"Epoch {epoch+1}/{args.epochs} starting...")

        # Note: No need to set epoch for IterableDataset (no sampler used)
        # Shuffling is handled internally by the dataset

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
                f"Val Acc: {val_metrics['accuracy']:.4f}"
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
                logger.info(f"New best loss: {best_loss:.4f}")
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
