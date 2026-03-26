"""Train a DeBERTa-v3-large text-only humor detector on the UR-Funny2 dataset."""

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup


class URFunnyTextDataset(Dataset):
    def __init__(self, ids: list[int], language: dict, labels: dict, tokenizer, max_length: int = 512):
        self.ids = ids
        self.language = language
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sample_id = self.ids[idx]
        lang = self.language[sample_id]

        # Concatenate context sentences and punchline
        context = " ".join(lang["context_sentences"])
        punchline = lang["punchline_sentence"]
        # Use sentence-pair format: context [SEP] punchline
        encoding = self.tokenizer(
            context,
            punchline,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        label = self.labels[sample_id]
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding["token_type_ids"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


def load_data(metadata_dir: str):
    metadata_dir = Path(metadata_dir)
    with open(metadata_dir / "data_folds.pkl", "rb") as f:
        folds = pickle.load(f)
    with open(metadata_dir / "humor_label_sdk.pkl", "rb") as f:
        labels = pickle.load(f)
    with open(metadata_dir / "language_sdk.pkl", "rb") as f:
        language = pickle.load(f)
    return folds, labels, language


def setup_distributed():
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
            total_loss += outputs.loss.item() * labels.size(0)
            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # Per-class metrics
    tp = ((all_preds == 1) & (all_labels == 1)).sum().item()
    fp = ((all_preds == 1) & (all_labels == 0)).sum().item()
    fn = ((all_preds == 0) & (all_labels == 1)).sum().item()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    parser = argparse.ArgumentParser(description="Train DeBERTa-v3-large humor detector on UR-Funny2")
    parser.add_argument("--metadata_dir", type=str, default="data/urfunny2/metadata", help="Path to UR-Funny2 metadata")
    parser.add_argument("--output_dir", type=str, default="output/humor_text", help="Output directory")
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-v3-large", help="Pretrained model name")
    parser.add_argument("--max_length", type=int, default=256, help="Max token sequence length")
    parser.add_argument("--batch_size", type=int, default=16, help="Per-GPU batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio of total steps")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader num workers")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping")
    parser.add_argument("--eval_steps", type=int, default=0, help="Evaluate every N steps (0=end of epoch only)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_main = rank == 0

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Loading data from {args.metadata_dir}")

    folds, labels, language = load_data(args.metadata_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
    model.to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    train_dataset = URFunnyTextDataset(folds["train"], language, labels, tokenizer, args.max_length)
    dev_dataset = URFunnyTextDataset(folds["dev"], language, labels, tokenizer, args.max_length)
    test_dataset = URFunnyTextDataset(folds["test"], language, labels, tokenizer, args.max_length)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size * 2, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size * 2, num_workers=args.num_workers, pin_memory=True)

    total_steps = (len(train_loader) // args.gradient_accumulation_steps) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "logs")) if is_main else None

    if is_main:
        print(f"Train: {len(train_dataset)}, Dev: {len(dev_dataset)}, Test: {len(test_dataset)}")
        print(f"Total steps: {total_steps}, Warmup: {warmup_steps}")
        print(f"World size: {world_size}, Per-GPU batch: {args.batch_size}, Effective batch: {args.batch_size * world_size * args.gradient_accumulation_steps}")

    best_dev_f1 = 0.0
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        epoch_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            batch_labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=batch_labels)
            loss = outputs.loss / args.gradient_accumulation_steps
            loss.backward()

            epoch_loss += outputs.loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if is_main and global_step % 50 == 0:
                    avg_loss = epoch_loss / (step + 1)
                    lr = scheduler.get_last_lr()[0]
                    print(f"Epoch {epoch+1}/{args.epochs} Step {global_step} Loss: {avg_loss:.4f} LR: {lr:.2e}")
                    writer.add_scalar("train/loss", avg_loss, global_step)
                    writer.add_scalar("train/lr", lr, global_step)

                if args.eval_steps > 0 and global_step % args.eval_steps == 0 and is_main:
                    dev_metrics = evaluate(model, dev_loader, device)
                    print(f"  [Step {global_step}] Dev - Loss: {dev_metrics['loss']:.4f} Acc: {dev_metrics['accuracy']:.4f} F1: {dev_metrics['f1']:.4f}")
                    for k, v in dev_metrics.items():
                        writer.add_scalar(f"dev/{k}", v, global_step)
                    if dev_metrics["f1"] > best_dev_f1:
                        best_dev_f1 = dev_metrics["f1"]
                        save_model = model.module if isinstance(model, DDP) else model
                        save_model.save_pretrained(os.path.join(args.output_dir, "best_model"))
                        tokenizer.save_pretrained(os.path.join(args.output_dir, "best_model"))
                        print(f"  Saved best model (F1={best_dev_f1:.4f})")
                    model.train()

        # End-of-epoch evaluation
        if is_main:
            avg_epoch_loss = epoch_loss / len(train_loader)
            dev_metrics = evaluate(model, dev_loader, device)
            print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {avg_epoch_loss:.4f}")
            print(f"  Dev - Loss: {dev_metrics['loss']:.4f} Acc: {dev_metrics['accuracy']:.4f} P: {dev_metrics['precision']:.4f} R: {dev_metrics['recall']:.4f} F1: {dev_metrics['f1']:.4f}")

            writer.add_scalar("epoch/train_loss", avg_epoch_loss, epoch + 1)
            for k, v in dev_metrics.items():
                writer.add_scalar(f"epoch/dev_{k}", v, epoch + 1)

            if dev_metrics["f1"] > best_dev_f1:
                best_dev_f1 = dev_metrics["f1"]
                save_model = model.module if isinstance(model, DDP) else model
                save_model.save_pretrained(os.path.join(args.output_dir, "best_model"))
                tokenizer.save_pretrained(os.path.join(args.output_dir, "best_model"))
                print(f"  Saved best model (F1={best_dev_f1:.4f})")

    # Final test evaluation
    if is_main:
        print("\n--- Loading best model for test evaluation ---")
        best_model = AutoModelForSequenceClassification.from_pretrained(os.path.join(args.output_dir, "best_model"))
        best_model.to(device)
        test_metrics = evaluate(best_model, test_loader, device)
        print(f"Test - Loss: {test_metrics['loss']:.4f} Acc: {test_metrics['accuracy']:.4f} P: {test_metrics['precision']:.4f} R: {test_metrics['recall']:.4f} F1: {test_metrics['f1']:.4f}")

        for k, v in test_metrics.items():
            writer.add_scalar(f"test/{k}", v, 0)
        writer.close()

        # Save test results
        import json
        with open(os.path.join(args.output_dir, "test_results.json"), "w") as f:
            json.dump(test_metrics, f, indent=2)
        print(f"\nResults saved to {args.output_dir}/test_results.json")

    cleanup_distributed()


if __name__ == "__main__":
    main()
