"""Distributed orchestrator for multi-speaker processing."""

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

from podcast_processing.multi_speaker_processor import MultiSpeakerProcessor

logger = logging.getLogger(__name__)


class MultiSpeakerOrchestrator:
    """Coordinate multi-GPU processing of podcast episodes with multi-speaker support.

    Uses original mixed audio and diarization metadata to create flexible
    user/system speaker assignments.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        output_root: str | Path,
        hf_repo: str = 'kyutai/moshiko-pytorch-bf16',
        shift_frames: int = 1,
        min_speaker_share: float = 0.20,
        mask_laughter: bool = True,
        splits: list[str] = None,
        moshi_weights: str | None = None,
        mimi_weights: str | None = None,
        config_path: str | None = None
    ):
        """Initialize orchestrator.

        Args:
            dataset_root: Root directory of PodcastFillers dataset
            output_root: Root directory for output files
            hf_repo: HuggingFace repository for models
            shift_frames: Prediction shift value for label generation
            min_speaker_share: Minimum speaking ratio for user speaker (default: 0.20)
            mask_laughter: Whether to mask laughter in both audio streams (default: True)
            splits: List of splits to process (default: ['train', 'validation', 'test'])
            moshi_weights: Path to local Moshi weights (default: None, use HF repo)
            mimi_weights: Path to local Mimi weights (default: None, use HF repo)
            config_path: Path to local config file (default: None, use HF repo)
        """
        self.dataset_root = Path(dataset_root)
        self.output_root = Path(output_root)
        self.hf_repo = hf_repo
        self.shift_frames = shift_frames
        self.min_speaker_share = min_speaker_share
        self.mask_laughter = mask_laughter
        self.splits = splits or ['train', 'validation', 'test']
        self.moshi_weights = moshi_weights
        self.mimi_weights = mimi_weights
        self.config_path = config_path

        # Directory paths
        self.audio_dir = self.dataset_root / 'audio' / 'episode_wav'
        self.diarization_dir = self.dataset_root / 'metadata' / 'episode_diarizations'
        self.transcript_dir = self.dataset_root / 'metadata' / 'episode_transcripts'
        self.annotations_dir = self.dataset_root / 'metadata' / 'episode_event_speaker_mapping'

        # Create output directories
        self.output_root.mkdir(parents=True, exist_ok=True)

    def run(self, limit: int = None):
        """Main execution method for distributed processing.

        Args:
            limit: Optional limit on number of episodes to process (for testing)
        """
        # Initialize distributed if running with torchrun
        if is_torchrun():
            dist.init_process_group(backend=BACKEND)
            if 'CUDA_VISIBLE_DEVICES' not in os.environ:
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
        logger.info(f"Min speaker share: {self.min_speaker_share*100:.0f}%")
        logger.info(f"Mask laughter: {self.mask_laughter}")

        # Enumerate all episodes
        logger.info("Enumerating episodes...")
        all_episodes = self._enumerate_episodes()

        if limit:
            all_episodes = all_episodes[:limit]
            logger.info(f"Limited to {limit} episodes")

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

        checkpoint = CheckpointInfo.from_hf_repo(
            hf_repo=self.hf_repo,
            moshi_weights=self.moshi_weights,
            mimi_weights=self.mimi_weights,
            config_path=self.config_path,
        )

        mimi = checkpoint.get_mimi(device=device)
        mimi.eval()
        for param in mimi.parameters():
            param.requires_grad = False

        lm_model = checkpoint.get_moshi(device=device)
        lm_model.eval()
        for param in lm_model.parameters():
            param.requires_grad = False

        tokenizer = self._load_tokenizer(checkpoint)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(f"Models loaded on {device}")

        # Create processor
        processor = MultiSpeakerProcessor(
            mimi=mimi,
            lm_model=lm_model,
            tokenizer=tokenizer,
            device=device,
            shift_frames=self.shift_frames,
            min_speaker_share=self.min_speaker_share,
            mask_laughter=self.mask_laughter
        )

        # Process episodes
        success_count = 0
        skip_count = 0
        error_count = 0

        with tqdm(total=len(my_episodes), desc=f"GPU {rank}", position=rank) as pbar:
            for episode in my_episodes:
                try:
                    # Check if already processed
                    check_path = (
                        self.output_root / episode['split'] / episode['name'] /
                        f"features_assignment_0.npy"
                    )

                    if check_path.exists():
                        logger.info(f"Skipping already processed: {episode['name']}")
                        skip_count += 1
                        pbar.update(1)
                        continue

                    # Validate episode
                    self._validate_episode(episode)

                    # Process episode
                    processor.process_episode(
                        episode_info=episode,
                        output_root=self.output_root,
                        annotations_dir=self.annotations_dir
                    )

                    success_count += 1
                    logger.info(f"✓ {episode['name']}")

                except Exception as e:
                    error_count += 1
                    logger.error(f"✗ {episode['name']}: {type(e).__name__}: {e}", exc_info=True)

                finally:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    pbar.update(1)

        logger.info(
            f"Rank {rank} finished: {success_count} success, {skip_count} skipped, {error_count} errors"
        )

        # Cleanup distributed
        if dist.is_initialized():
            dist.destroy_process_group()

    def _enumerate_episodes(self) -> list[dict]:
        """Enumerate all episodes across splits.

        Returns:
            List of episode info dictionaries
        """
        episodes = []

        for split in self.splits:
            audio_split_dir = self.audio_dir / split
            diarization_split_dir = self.diarization_dir / split
            transcript_split_dir = self.transcript_dir / split

            if not audio_split_dir.exists():
                logger.warning(f"Audio directory not found: {audio_split_dir}")
                continue

            # Find all audio files
            for audio_file in sorted(audio_split_dir.glob('*.wav')):
                episode_name = audio_file.stem
                diarization_path = diarization_split_dir / f"{episode_name}.json"
                transcript_path = transcript_split_dir / f"{episode_name}.json"

                # Skip if diarization doesn't exist
                if not diarization_path.exists():
                    logger.warning(f"Diarization not found for {episode_name}, skipping")
                    continue

                episodes.append({
                    'name': episode_name,
                    'split': split,
                    'audio_path': audio_file,
                    'diarization': diarization_path,
                    'transcript': transcript_path
                })

        return episodes

    def _validate_episode(self, episode_info: dict):
        """Validate that all required files exist for an episode."""
        if not episode_info['audio_path'].exists():
            raise FileNotFoundError(f"Audio file missing: {episode_info['audio_path']}")

        if not episode_info['diarization'].exists():
            raise FileNotFoundError(f"Diarization missing: {episode_info['diarization']}")

        # Transcript is optional but warn if missing
        if not episode_info['transcript'].exists():
            logger.warning(f"Transcript missing: {episode_info['transcript']}")

    def _load_tokenizer(self, checkpoint: CheckpointInfo):
        """Load SentencePiece tokenizer."""
        logger.info("Loading tokenizer...")
        tokenizer = sentencepiece.SentencePieceProcessor()
        tokenizer.load(str(checkpoint.tokenizer))
        return tokenizer

    def _setup_logger(self, rank: int):
        """Set up logging for this rank."""
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)

        log_file = log_dir / f'multi_speaker_rank_{rank}.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            f'[Rank {rank}] %(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
