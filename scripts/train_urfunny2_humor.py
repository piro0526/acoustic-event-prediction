#!/usr/bin/env python
"""Train humor predictor on UR-Funny2 using Moshi frame-level features.

Architecture:
    1. Mean pooling across frames: [T, 4096] -> [4096]
    2. Linear binary classifier: [4096] -> [1]

Usage:
    python scripts/train_urfunny2_humor.py \
        --features_dir output/urfunny2/features \
        --metadata_dir data/urfunny2/metadata \
        --output_dir output/urfunny2/humor_prediction

    # Multi-GPU
    torchrun --nproc_per_node=4 scripts/train_urfunny2_humor.py \
        --features_dir output/urfunny2/features \
        --metadata_dir data/urfunny2/metadata \
        --output_dir output/urfunny2/humor_prediction
"""

import argparse
import logging
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------- Model ----------

class MeanPoolingClassifier(nn.Module):
    """Mean pooling over frames + linear binary classifier.

    [B, T, 4096] -> mean pool -> [B, 4096] -> Linear -> [B, 1]
    """

    def __init__(self, input_dim: int = 4096):
        super().__init__()
        self.classifier = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: [B, T, input_dim] padded frame features.
            mask: [B, T] bool, True for valid frames.

        Returns:
            logits: [B, 1]
        """
        if mask is not None:
            x = x * mask.unsqueeze(-1).float()
            pooled = x.sum(dim=1) / mask.sum(dim=1, keepdim=True).float().clamp(min=1)
        else:
            pooled = x.mean(dim=1)

        return self.classifier(pooled)  # [B, 1]


# ---------- Dataset ----------

class URFunny2Dataset(Dataset):
    """Dataset for UR-Funny2 humor prediction.

    Loads pre-computed [T, 4096] features and binary humor labels.
    """

    def __init__(self, sample_ids: list[int], features_dir: Path, labels: dict[int, int]):
        self.features_dir = features_dir
        self.samples = []
        missing = 0
        for sid in sample_ids:
            fpath = features_dir / f"{sid}.npy"
            if fpath.exists() and sid in labels:
                self.samples.append((sid, fpath, labels[sid]))
            else:
                missing += 1
        if missing > 0:
            logger.warning(f"Skipped {missing} samples (missing features or labels)")
        logger.info(f"Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sid, fpath, label = self.samples[idx]
        features = np.load(fpath, mmap_mode="r")  # [T, 4096] memory-mapped
        features = np.array(features)  # copy to writable array for torch
        return torch.from_numpy(features), torch.tensor(label, dtype=torch.float32), sid


def collate_fn(batch):
    """Pad variable-length frame sequences."""
    features_list, labels, sids = zip(*batch)
    lengths = [f.shape[0] for f in features_list]
    max_len = max(lengths)
    dim = features_list[0].shape[1]

    padded = torch.zeros(len(batch), max_len, dim)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for i, (f, l) in enumerate(zip(features_list, lengths)):
        padded[i, :l] = f
        mask[i, :l] = True

    labels = torch.stack(labels)
    return padded, mask, labels, list(sids)


# ---------- Distributed ----------

def setup_distributed():
    if "RANK" not in os.environ:
        return 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    return rank, world_size, device


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


# ---------- Evaluation ----------

@torch.no_grad()
def evaluate(model, dataloader, device, criterion):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    total_loss = 0.0
    n = 0

    for features, mask, labels, _ in dataloader:
        features, mask, labels = features.to(device), mask.to(device), labels.to(device)
        logits = model(features, mask).squeeze(-1)
        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        n += labels.size(0)

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    metrics = {
        "loss": total_loss / max(n, 1),
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
    }
    try:
        metrics["auc"] = roc_auc_score(all_labels, all_probs)
    except ValueError:
        metrics["auc"] = 0.0

    return metrics


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(description="Train UR-Funny2 humor predictor")
    parser.add_argument("--features_dir", type=str, required=True)
    parser.add_argument("--metadata_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="output/urfunny2/humor_prediction")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    args = parser.parse_args()

    rank, world_size, device = setup_distributed()
    is_main = rank == 0

    logging.basicConfig(
        level=logging.INFO if is_main else logging.WARNING,
        format=f"[Rank {rank}] %(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    features_dir = Path(args.features_dir)
    metadata_dir = Path(args.metadata_dir)
    output_dir = Path(args.output_dir)

    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    with open(metadata_dir / "humor_label_sdk.pkl", "rb") as f:
        labels = pickle.load(f)
    with open(metadata_dir / "data_folds.pkl", "rb") as f:
        folds = pickle.load(f)

    logger.info(f"Labels: {len(labels)}, Train: {len(folds['train'])}, Dev: {len(folds['dev'])}, Test: {len(folds['test'])}")

    # Datasets
    train_ds = URFunny2Dataset(folds["train"], features_dir, labels)
    dev_ds = URFunny2Dataset(folds["dev"], features_dir, labels)
    test_ds = URFunny2Dataset(folds["test"], features_dir, labels)

    # DataLoaders
    train_sampler = DistributedSampler(train_ds, shuffle=True) if world_size > 1 else None
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None),
        sampler=train_sampler, num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
    )
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)

    # Model
    model = MeanPoolingClassifier(input_dim=4096).to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[int(os.environ["LOCAL_RANK"])])

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    writer = SummaryWriter(output_dir / "logs") if is_main else None

    if is_main:
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model: MeanPoolingClassifier, params={param_count:,}")

    best_f1 = 0.0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        train_loss = 0.0
        train_n = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", disable=not is_main)
        for features, mask, batch_labels, _ in pbar:
            features, mask, batch_labels = features.to(device), mask.to(device), batch_labels.to(device)

            logits = model(features, mask).squeeze(-1)
            loss = criterion(logits, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * batch_labels.size(0)
            train_n += batch_labels.size(0)
            pbar.set_postfix(loss=f"{train_loss / train_n:.4f}")

        # Evaluate
        if is_main:
            avg_train_loss = train_loss / max(train_n, 1)
            dev_metrics = evaluate(model, dev_loader, device, criterion)
            scheduler.step(dev_metrics["f1"])

            logger.info(
                f"Epoch {epoch}: train_loss={avg_train_loss:.4f} | "
                f"dev_loss={dev_metrics['loss']:.4f} acc={dev_metrics['accuracy']:.4f} "
                f"f1={dev_metrics['f1']:.4f} auc={dev_metrics['auc']:.4f}"
            )

            if writer:
                writer.add_scalar("train/loss", avg_train_loss, epoch)
                for k, v in dev_metrics.items():
                    writer.add_scalar(f"dev/{k}", v, epoch)
                writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

            # Save best model
            if dev_metrics["f1"] > best_f1:
                best_f1 = dev_metrics["f1"]
                patience_counter = 0
                ckpt = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": ckpt,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "dev_metrics": dev_metrics,
                    "args": vars(args),
                }, output_dir / "best_model.pt")
                logger.info(f"  -> New best F1: {best_f1:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    logger.info(f"Early stopping at epoch {epoch} (patience={args.patience})")
                    break

    # Final test evaluation
    if is_main:
        ckpt = torch.load(output_dir / "best_model.pt", map_location=device, weights_only=True)
        raw_model = model.module if isinstance(model, DDP) else model
        raw_model.load_state_dict(ckpt["model_state_dict"])
        test_metrics = evaluate(model, test_loader, device, criterion)
        logger.info("=" * 60)
        logger.info("Test Results:")
        for k, v in test_metrics.items():
            logger.info(f"  {k}: {v:.4f}")
        logger.info(f"  (best dev F1 at epoch {ckpt['epoch']})")
        logger.info("=" * 60)

        if writer:
            for k, v in test_metrics.items():
                writer.add_scalar(f"test/{k}", v, 0)
            writer.close()

    cleanup_distributed()


if __name__ == "__main__":
    main()
