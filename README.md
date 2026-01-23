# Acoustic Event Prediction

Acoustic event prediction system for detecting laughter in podcast audio using Moshi/Mimi models. This project processes the PodcastFillers dataset to extract audio features and trains binary classifiers to predict laughter events.

## Features

- **Audio Feature Extraction**: Extract 4096-dimensional features from audio using Moshi/Mimi encoder
- **Multi-GPU Support**: Distributed processing and training with PyTorch DDP
- **Binary Classification**: Predict laughter events with configurable prediction shifts
- **Multiple Loss Functions**: BCE, Focal Loss, and Adaptive Focal Loss
- **Comprehensive Evaluation**: Precision, recall, F1-score, AUC-ROC, confusion matrices
- **Episode-Level Storage**: Efficient per-episode storage format
- **TensorBoard Integration**: Real-time training monitoring and visualization

## Requirements

- Python 3.12+
- CUDA-capable GPU (multi-GPU recommended)
- 50GB+ disk space for PodcastFillers dataset
- 15GB+ disk space for pre-trained models

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management:

```bash
# Install dependencies
uv sync

# Install with development tools
uv sync --extra dev
```

## Quick Start

### 1. Feature Extraction and Label Generation

Extract features and generate labels from podcast audio:

```bash
# Single GPU (default shift: 1 frame)
python scripts/compute_features.py \
    --dataset_root data/PodcastFillers \
    --output_root output/laughter/features

# Multi-GPU (8 GPUs recommended) with custom shift value
torchrun --nproc_per_node=8 scripts/compute_features.py \
    --dataset_root data/PodcastFillers \
    --output_root output/laughter/features \
    --shift_frames 5 \
    --hf_repo kyutai/moshiko-pytorch-bf16

# Generate labels for additional shift values (features will be skipped)
python scripts/compute_features.py \
    --dataset_root data/PodcastFillers \
    --output_root output/laughter/features \
    --shift_frames 10
```

### 2. Model Training

Train the laughter prediction classifier:

```bash
# Single GPU (default shift=1)
python scripts/train_laughter_predictor.py \
    --features_dir output/laughter/features \
    --shift_frames 1 \
    --output_dir output/laughter_prediction \
    --batch_size 512 \
    --learning_rate 1e-4 \
    --epochs 50

# Multi-GPU (4 GPUs) with custom shift value
torchrun --nproc_per_node=4 scripts/train_laughter_predictor.py \
    --features_dir output/laughter/features \
    --shift_frames 5 \
    --output_dir output/laughter_prediction \
    --batch_size 512 \
    --learning_rate 3e-4 \
    --epochs 100 \
    --num_workers 0 \
    --loss_type bce
```

### 3. Monitor Training

```bash
tensorboard --logdir output/laughter_prediction
```

## Project Structure

```
acoustic-event-prediction/
├── src/
│   ├── moshi/                      # Kyutai Moshi inference codebase
│   ├── podcast_processing/         # Data preprocessing pipeline
│   └── laughter_prediction/        # Core prediction module
├── scripts/
│   ├── compute_features.py # Feature extraction and label generation
│   └── train_laughter_predictor.py # Model training
├── data/                           # PodcastFillers dataset (50GB)
├── models/                         # Pre-trained models (15GB)
├── outputs/                        # Training outputs
└── pyproject.toml                  # Project configuration
```

See [CLAUDE.md](CLAUDE.md) for detailed architecture documentation.

## Data Pipeline

1. **Feature Extraction and Label Generation**:
   - Load podcast episodes from PodcastFillers dataset
   - Pass audio through Moshi encoder → 4096-dim features per frame
   - Generate binary labels from laughter annotations with configurable prediction shift
   - Save episode-level outputs (features, labels, metadata)
2. **Training**: Train `LaughterPredictor` classifier with multi-GPU DDP
3. **Evaluation**: Compute metrics and optimize decision thresholds

## Training Arguments

Key arguments for `train_laughter_predictor.py`:

- `--features_dir`: Directory containing episode-level features and labels
- `--shift_frames`: Prediction shift value to use for labels (default: 1)
- `--output_dir`: Output directory for checkpoints and logs
- `--batch_size`: Batch size per GPU (default: 512)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--epochs`: Number of training epochs (default: 50)
- `--loss_type`: Loss function - `bce`, `focal`, or `adaptive_focal` (default: bce)
- `--num_workers`: DataLoader workers (default: 0 for multi-GPU)

## Model Architecture

**LaughterPredictor**: Binary classifier that takes 4096-dimensional features and predicts whether a laughter event will occur in the prediction interval.

- **Simple variant** (recommended): Single linear layer
- **MLP variant**: Linear → ReLU → Dropout → Linear

## Development

```bash
# Format code (line length: 120)
uv run black src/ scripts/

# Lint
uv run ruff check src/ scripts/
uv run pyright src/ scripts/

# Run tests
uv run pytest
```

## License

This project incorporates code from multiple sources:

### Moshi Components (`src/moshi/`)

The Moshi inference codebase is adapted from:
- **[Moshi](https://github.com/kyutai-labs/moshi)** by Kyutai Labs - MIT License
  - Copyright (c) Kyutai, all rights reserved
  - Includes code adapted from Audiocraft (Copyright Meta Platforms, Inc.)

### Fine-tuning Infrastructure (`src/finetune/`)

Fine-tuning utilities are based on:
- **[moshi-finetune](https://github.com/kyutai-labs/moshi-finetune)** by Kyutai Labs - Apache License 2.0
  - Copyright (c) Kyutai Labs
  - Contributors: Laurent Mazare, Hippolyte Pilchen, Alexandre Défossez, Václav Volhejn
  - Uses code from mistral-finetune (Apache License 2.0)

### Model Weights

Moshi model weights are released under **CC-BY 4.0**.

### This Project

Original code for laughter prediction and podcast processing is available under the MIT License.

## Citation

If you use Moshi or Mimi in your work, please cite:

```bibtex
@article{defossez2024moshi,
  title={Moshi: a speech-text foundation model for real-time dialogue},
  author={Défossez, Alexandre and Synnaeve, Gabriel and Adi, Yossi and Copet, Jade and Kharitonov, Eugene and
          Zeghidour, Neil and Usunier, Nicolas and others},
  journal={arXiv preprint arXiv:2410.00037},
  year={2024}
}
```

## Acknowledgments

- **Kyutai Labs** for the Moshi/Mimi models and infrastructure
- **Meta Platforms, Inc.** for Audiocraft (basis for Moshi implementation)
- **PodcastFillers Dataset** for training data

## Links

- [Moshi (Kyutai Labs)](https://github.com/kyutai-labs/moshi)
- [Moshi Fine-tuning](https://github.com/kyutai-labs/moshi-finetune)
- [Moshi Paper (arXiv:2410.00037)](https://arxiv.org/abs/2410.00037)
