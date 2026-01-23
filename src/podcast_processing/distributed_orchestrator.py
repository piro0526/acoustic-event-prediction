"""Distributed orchestrator for multi-GPU processing."""

import os
import logging
from pathlib import Path

# Set PyTorch memory management before importing torch
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import torch.distributed as dist
from tqdm import tqdm

from finetune.distributed import BACKEND, get_rank, get_world_size, is_torchrun, set_device
from moshi.models.loaders import CheckpointInfo, TEXT_TOKENIZER_NAME
import sentencepiece

from podcast_processing.dataset_enumerator import DatasetEnumerator
from podcast_processing.episode_processor import EpisodeProcessor

logger = logging.getLogger(__name__)


class DistributedOrchestrator:
    """Coordinate multi-GPU processing of podcast episodes.

    Handles:
    - Distributed initialization
    - Episode distribution across GPUs
    - Model loading on each GPU
    - Episode processing coordination
    - Error handling and logging
    """

    def __init__(
        self,
        dataset_root: str | Path,
        output_root: str | Path,
        hf_repo: str = 'kyutai/moshiko-pytorch-bf16'
    ):
        """Initialize orchestrator.

        Args:
            dataset_root: Root directory of PodcastFillers dataset
            output_root: Root directory for output files
            hf_repo: HuggingFace repository for models
        """
        self.dataset_root = Path(dataset_root)
        self.output_root = Path(output_root)
        self.hf_repo = hf_repo

        # Create output directories
        self.output_root.mkdir(parents=True, exist_ok=True)

    def run(self):
        """Main execution method for distributed processing."""
        # Initialize distributed if running with torchrun
        if is_torchrun():
            dist.init_process_group(backend=BACKEND)
            # Set CUDA_VISIBLE_DEVICES if not already set (for set_device compatibility)
            if 'CUDA_VISIBLE_DEVICES' not in os.environ:
                # Auto-detect available GPUs
                num_gpus = torch.cuda.device_count()
                os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in range(num_gpus))
            set_device()

        rank = get_rank() if dist.is_initialized() else 0
        world_size = get_world_size() if dist.is_initialized() else 1

        # Set up logging
        self._setup_logger(rank)

        logger.info(f"Rank {rank}/{world_size} starting")
        logger.info(f"Dataset root: {self.dataset_root}")
        logger.info(f"Output root: {self.output_root}")

        # Enumerate all episodes
        logger.info("Enumerating episodes...")
        enumerator = DatasetEnumerator(self.dataset_root)
        all_episodes = enumerator.enumerate_episodes()

        logger.info(f"Total episodes: {len(all_episodes)}")

        # Distribute episodes across GPUs
        episodes_per_gpu = len(all_episodes) // world_size
        start = rank * episodes_per_gpu
        end = start + episodes_per_gpu if rank < world_size - 1 else len(all_episodes)
        my_episodes = all_episodes[start:end]

        logger.info(f"Rank {rank} processing episodes {start}-{end-1} ({len(my_episodes)} total)")

        # Load models
        logger.info("Loading models...")
        device = f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'

        # Load checkpoint info
        checkpoint = CheckpointInfo.from_hf_repo(
            hf_repo=self.hf_repo,
            moshi_weights="custom_weights/moshi/consolidated.safetensors",
            mimi_weights="custom_weights/mimi/consolidated.safetensors",
            config_path="custom_weights/moshi/config.json"
        )

        mimi = checkpoint.get_mimi(device=device)
        mimi.eval()
        # Disable gradients for all parameters to save memory
        for param in mimi.parameters():
            param.requires_grad = False

        lm_model = checkpoint.get_moshi(device=device)
        lm_model.eval()
        # Disable gradients for all parameters to save memory
        for param in lm_model.parameters():
            param.requires_grad = False

        tokenizer = self._load_tokenizer(checkpoint)

        # Clear any initialization cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(f"Models loaded on {device}")

        # Create processor
        processor = EpisodeProcessor(mimi, lm_model, tokenizer, device)

        # Process episodes
        success_count = 0
        error_count = 0

        with tqdm(total=len(my_episodes), desc=f"GPU {rank}", position=rank) as pbar:
            for episode in my_episodes:
                try:
                    # Create output path
                    output_path = self.output_root / episode['split'] / f"{episode['name']}.pt"
                    output_path.parent.mkdir(parents=True, exist_ok=True)

                    # Check if already processed
                    if output_path.exists():
                        logger.info(f"Skipping already processed: {episode['name']}")
                        pbar.update(1)
                        continue

                    # Validate episode
                    self._validate_episode(episode)

                    # Process episode
                    processor.process_episode(episode, output_path)

                    success_count += 1
                    logger.info(f"✓ {episode['name']}")

                except Exception as e:
                    error_count += 1
                    logger.error(f"✗ {episode['name']}: {type(e).__name__}: {e}", exc_info=True)

                finally:
                    # Clear CUDA cache to avoid OOM
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    pbar.update(1)

        logger.info(f"Rank {rank} finished: {success_count} success, {error_count} errors")

        # Cleanup distributed
        if dist.is_initialized():
            dist.destroy_process_group()

    def _load_tokenizer(self, checkpoint: CheckpointInfo):
        """Load SentencePiece tokenizer."""
        logger.info("Loading tokenizer...")
        tokenizer = sentencepiece.SentencePieceProcessor()
        tokenizer.load(str(checkpoint.tokenizer))
        return tokenizer

    def _validate_episode(self, episode_info: dict):
        """Validate that all required files exist for an episode."""
        # Check diarization file
        if not episode_info['diarization'].exists():
            raise FileNotFoundError(f"Diarization missing: {episode_info['diarization']}")

        # Check transcript file
        if not episode_info['transcript'].exists():
            raise FileNotFoundError(f"Transcript missing: {episode_info['transcript']}")

        # Check audio directory
        if not episode_info['audio_dir'].exists():
            raise FileNotFoundError(f"Audio directory missing: {episode_info['audio_dir']}")

        # Check number of speakers
        import json
        with open(episode_info['diarization']) as f:
            diar = json.load(f)

        num_speakers = diar.get('num_speakers', 0)
        if num_speakers < 2:
            raise ValueError(f"Episode has less than 2 speakers: {num_speakers}")

    def _setup_logger(self, rank: int):
        """Set up logging for this rank."""
        # Create logs directory
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)

        # File handler for this rank
        log_file = log_dir / f'worker_rank_{rank}.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            f'[Rank {rank}] %(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
