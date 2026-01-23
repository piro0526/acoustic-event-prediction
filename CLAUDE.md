# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Acoustic event prediction system for detecting laughter in podcast audio. Uses Moshi/Mimi (Kyutai) transformer models to extract 4096-dimensional audio features from the PodcastFillers dataset, then trains binary classifiers to predict laughter events.

**Tech Stack:** Python 3.12+, PyTorch, torchaudio, HuggingFace, scikit-learn, multi-GPU training via DDP

## Development Commands

### Environment Setup
```bash
# Install dependencies (uses uv package manager)
uv sync

# Install with dev dependencies
uv sync --extra dev
```

### Feature Extraction
```bash
# Single GPU - compute transformer features from audio
python scripts/compute_transformer_outs.py \
    --dataset_root data/PodcastFillers \
    --output_root output/transformer_outs

# Multi-GPU (8 GPUs)
torchrun --nproc_per_node=8 scripts/compute_transformer_outs.py \
    --dataset_root data/PodcastFillers \
    --output_root output/transformer_outs \
    --hf_repo kyutai/moshiko-pytorch-bf16
```

### Model Training
```bash
# Single GPU
python scripts/train_laughter_predictor.py \
    --transformer_dir output/transformer_outs \
    --labels_dir data/PodcastFillers/metadata/episode_laughter_prediction_intervals \
    --output_dir output/laughter_prediction \
    --batch_size 512 \
    --learning_rate 1e-4 \
    --epochs 50

# Multi-GPU (4 GPUs)
torchrun --nproc_per_node=4 scripts/train_laughter_predictor.py \
    --transformer_dir output/transformer_outs \
    --labels_dir data/PodcastFillers/metadata/episode_laughter_prediction_intervals \
    --turns_dir data/PodcastFillers/metadata/episode_laughter_turns \
    --output_dir output/laughter_prediction \
    --batch_size 512 \
    --learning_rate 3e-4 \
    --epochs 100 \
    --num_workers 0 \
    --loss_type bce

# Or use the provided script
./run_train.sh
```

### Code Quality
```bash
# Format code (line length: 120)
uv run black src/ scripts/

# Lint
uv run ruff check src/ scripts/
uv run pyright src/ scripts/
uv run flake8 src/ scripts/

# Run tests
uv run pytest
```

### Monitoring Training
```bash
# View TensorBoard logs
tensorboard --logdir output/laughter_prediction
```

## Architecture

### Module Structure

```
src/
├── moshi/                      # Kyutai Moshi inference codebase
│   ├── models/                 # LM, compression, TTS models
│   ├── modules/                # Neural network modules (transformer, conv, etc.)
│   ├── conditioners/           # Conditioning mechanisms
│   ├── quantization/           # Model quantization utilities
│   └── run_inference.py        # Inference entry point
│
├── podcast_processing/         # Data preprocessing pipeline
│   ├── distributed_orchestrator.py  # Multi-GPU feature extraction coordinator
│   ├── episode_processor.py         # Process individual episodes
│   ├── extract_features.py          # Extract 4096-dim transformer features
│   └── create_labels.py             # Generate binary laughter labels
│
└── laughter_prediction/        # Core prediction module
    ├── model.py                # LaughterPredictor classifier
    ├── train.py                # Multi-GPU training with DDP
    ├── dataset.py              # LaughterDataset (standard)
    ├── iterable_dataset.py     # IterableLaughterDataset (streaming)
    ├── evaluate.py             # Evaluation utilities
    ├── metrics.py              # Precision, recall, F1, AUC, confusion matrix
    └── focal_loss.py           # Focal and adaptive focal loss
```

### Data Pipeline

1. **Feature Extraction** ([podcast_processing/](src/podcast_processing/)):
   - Loads PodcastFillers episodes (audio + metadata)
   - Passes audio through Moshi transformer encoder
   - Produces 4096-dimensional features per frame
   - Multi-GPU distributed processing via `DistributedOrchestrator`

2. **Label Creation** ([podcast_processing/create_labels.py](src/podcast_processing/create_labels.py)):
   - Reads laughter event annotations from metadata
   - Generates frame-level binary labels
   - Supports prediction intervals (look-ahead windows)

3. **Training** ([laughter_prediction/train.py](src/laughter_prediction/train.py)):
   - Loads features and labels via `LaughterDataset`
   - Trains `LaughterPredictor` (linear or MLP classifier)
   - Multi-GPU training with DistributedDataParallel
   - Supports BCE, Focal Loss, Adaptive Focal Loss
   - Saves checkpoints and logs to TensorBoard

4. **Evaluation** ([laughter_prediction/evaluate.py](src/laughter_prediction/evaluate.py)):
   - Computes precision, recall, F1, AUC
   - Generates confusion matrices
   - Optimizes decision thresholds

### Key Components

**LaughterPredictor** ([laughter_prediction/model.py](src/laughter_prediction/model.py)):
- Simple binary classifier on 4096-dim features
- Options: linear layer only (recommended) or MLP with hidden layer
- Outputs logits → apply sigmoid for probabilities

**Training Variants**:
- `train.py`: Standard training with `LaughterDataset` (loads all data)
- `iterable_train.py`: Streaming training with `IterableLaughterDataset` (memory-efficient)
- `regression_iterable_dataset.py`: Experimental regression variant

**Loss Functions**:
- BCE: Standard binary cross-entropy
- Focal Loss: Handles class imbalance
- Adaptive Focal Loss: Dynamically adjusts focal weight

### Distributed Training

Multi-GPU training uses PyTorch Distributed Data Parallel:
- Launch with `torchrun --nproc_per_node=N`
- Training script detects rank automatically via environment variables
- Metrics aggregated across GPUs in [laughter_prediction/train.py](src/laughter_prediction/train.py)
- Feature extraction parallelized across episodes in [podcast_processing/distributed_orchestrator.py](src/podcast_processing/distributed_orchestrator.py)

## Data Organization

```
data/PodcastFillers/           # 50GB dataset
├── audio/                     # Episode audio files
└── metadata/                  # Annotations
    ├── episode_laughter_prediction_intervals/  # Binary labels for prediction
    └── episode_laughter_turns/                 # Original laughter annotations

output/
├── transformer_outs/          # Extracted features (4096-dim per frame)
└── laughter_prediction/       # Training outputs (checkpoints, logs, metrics)

models/                        # Pre-trained Moshi models (15GB, downloaded from HF)
```

## Important Patterns

### Adding Training Arguments

Training arguments are defined in [laughter_prediction/train.py](src/laughter_prediction/train.py) `main()`. To add new arguments:
1. Add `parser.add_argument()` in the argument parser section
2. Access via `args.your_argument` in training logic
3. Update [scripts/train_laughter_predictor.py](scripts/train_laughter_predictor.py) if changing interface

### Creating New Loss Functions

1. Implement in [laughter_prediction/focal_loss.py](src/laughter_prediction/focal_loss.py) or new file
2. Register in [laughter_prediction/train.py](src/laughter_prediction/train.py) loss selection logic
3. Use `--loss_type your_loss` flag

### Custom Metrics

Add metrics in [laughter_prediction/metrics.py](src/laughter_prediction/metrics.py) and integrate in [laughter_prediction/evaluate.py](src/laughter_prediction/evaluate.py) evaluation loop.

## Dependencies Note

- **bitsandbytes**: Linux-only (for quantization). Not available on macOS/Windows.
- **CUDA**: Required for GPU training. Multi-GPU requires NCCL backend.
- **uv**: Modern Python package manager (replaces pip/poetry). Lock file: [uv.lock](uv.lock)
