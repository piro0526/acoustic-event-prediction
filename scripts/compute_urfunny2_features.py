#!/usr/bin/env python
"""Compute Moshi transformer_out features for UR-Funny2 audio files.

Each audio file is treated as system audio (speaker), with silent user audio (listener).
Text tokens are generated from the transcripts in language_sdk.pkl using the Interleaver.

Usage:
    # Single GPU
    python scripts/compute_urfunny2_features.py \
        --audio_dir data/urfunny2/urfunny2_audios \
        --metadata_dir data/urfunny2/metadata \
        --output_dir output/urfunny2/features

    # Multi-GPU
    torchrun --nproc_per_node=4 scripts/compute_urfunny2_features.py \
        --audio_dir data/urfunny2/urfunny2_audios \
        --metadata_dir data/urfunny2/metadata \
        --output_dir output/urfunny2/features
"""

import argparse
import logging
import os
import pickle
import sys
from pathlib import Path

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from finetune.data.interleaver import Interleaver
from moshi.models.loaders import CheckpointInfo
from moshi.models.lm_utils import _delay_sequence

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 24000
FRAME_RATE = 12.5
MAX_CONTEXT_FRAMES = 3000  # Moshi LM context length
MAX_CONTEXT_DURATION = MAX_CONTEXT_FRAMES / FRAME_RATE  # 240 seconds
MAX_CONTEXT_SAMPLES = int(MAX_CONTEXT_DURATION * TARGET_SAMPLE_RATE)
SYSTEM_SPEAKER_LABEL = "SPEAKER_SYSTEM"


def setup_logging(rank: int):
    level = logging.INFO if rank == 0 else logging.WARNING
    logging.basicConfig(
        level=level,
        format=f'[Rank {rank}] %(asctime)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
    )


def load_models(hf_repo: str, device: torch.device, moshi_weights=None, mimi_weights=None, config_path=None):
    """Load Mimi and Moshi LM models."""
    ckpt = CheckpointInfo.from_hf_repo(
        hf_repo,
        moshi_weights=moshi_weights,
        mimi_weights=mimi_weights,
        config_path=config_path,
    )
    mimi = ckpt.get_mimi(device=device)
    mimi.eval()
    lm = ckpt.get_moshi(device=device, dtype=torch.bfloat16)
    lm.eval()
    tokenizer = ckpt.get_text_tokenizer()
    return mimi, lm, tokenizer


def build_alignments_from_metadata(meta: dict) -> list:
    """Convert UR-Funny2 metadata to Interleaver alignment format.

    Returns list of (word, (start, end), speaker_label) tuples sorted by time.
    """
    alignments = []

    # Context sentences
    context_features = meta.get('context_features', [])
    context_intervals = meta.get('context_intervals', [])
    for sent_words, sent_intervals in zip(context_features, context_intervals):
        # sent_intervals is a list of [start, end] for each word
        intervals_list = sent_intervals if isinstance(sent_intervals, list) else sent_intervals.tolist()
        for word, (start, end) in zip(sent_words, intervals_list):
            alignments.append((word, (start, end), SYSTEM_SPEAKER_LABEL))

    # Punchline sentence
    punchline_features = meta.get('punchline_features', [])
    punchline_intervals = meta.get('punchline_intervals', [])
    if isinstance(punchline_intervals, np.ndarray):
        punchline_intervals = punchline_intervals.tolist()
    for word, (start, end) in zip(punchline_features, punchline_intervals):
        alignments.append((word, (start, end), SYSTEM_SPEAKER_LABEL))

    # Sort by start time
    alignments.sort(key=lambda x: x[1][0])
    return alignments


def process_audio_chunk(
    user_audio: torch.Tensor,
    system_audio: torch.Tensor,
    text_tokens: torch.Tensor,
    mimi,
    lm,
    device: torch.device,
) -> torch.Tensor:
    """Process a single chunk through Mimi + LM.

    Args:
        user_audio: [1, samples] at 24kHz (silent)
        system_audio: [1, samples] at 24kHz
        text_tokens: [1, 1, T_frames] text token stream
        mimi: Mimi model
        lm: LM model
        device: CUDA device

    Returns:
        transformer_out: [T, 4096]
    """
    with torch.no_grad():
        # Encode with Mimi -> [1, K=32, T_frames]
        # Mimi requires float32 input
        user_audio_gpu = user_audio[None, :, :].to(device=device, dtype=torch.float32)
        user_codes = mimi.encode(user_audio_gpu)
        del user_audio_gpu

        system_audio_gpu = system_audio[None, :, :].to(device=device, dtype=torch.float32)
        system_codes = mimi.encode(system_audio_gpu)
        del system_audio_gpu

        # Align all to same length
        T_frames = max(user_codes.shape[-1], system_codes.shape[-1], text_tokens.shape[-1])
        user_codes = F.pad(user_codes, (0, T_frames - user_codes.shape[-1]))
        system_codes = F.pad(system_codes, (0, T_frames - system_codes.shape[-1]))
        text_tokens = F.pad(text_tokens, (0, T_frames - text_tokens.shape[-1]))

        # Extract first dep_q codebooks
        dep_q = lm.dep_q  # 8
        user_codes_subset = user_codes[:, :dep_q, :]
        system_codes_subset = system_codes[:, :dep_q, :]

        # Combine: [text(1), user(8), system(8)] -> [1, 17, T]
        codes = torch.cat([text_tokens, user_codes_subset, system_codes_subset], dim=1)

        # Apply causal delays
        B, K, T = codes.shape
        delays = lm.delays
        assert len(delays) == K

        initial = lm._get_initial_token().expand(B, -1, -1)
        delayed_codes = _delay_sequence(delays, codes, initial)
        delayed_codes = torch.cat([initial, delayed_codes], dim=2)

        # LM forward pass -> [1, T, 4096]
        transformer_out, _ = lm.forward_text(delayed_codes[:, :, :-1])

        result = transformer_out.squeeze(0).float().cpu()

        del user_codes, system_codes, user_codes_subset, system_codes_subset
        del text_tokens, codes, delayed_codes, transformer_out

    return result  # [T, 4096]


def process_single_file(
    audio_path: Path,
    output_path: Path,
    meta: dict,
    mimi,
    lm,
    interleaver: Interleaver,
    device: torch.device,
):
    """Process one audio file and save transformer_out."""
    if output_path.exists():
        logger.debug(f"Skipping (exists): {audio_path.name}")
        return

    # Load audio
    waveform, sr = torchaudio.load(audio_path)

    # Resample to 24kHz if needed
    if sr != TARGET_SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, TARGET_SAMPLE_RATE)
        waveform = resampler(waveform)

    # Ensure mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # system_audio = the audio file, user_audio = silent
    system_audio = waveform
    user_audio = torch.zeros_like(waveform)

    duration = waveform.shape[-1] / TARGET_SAMPLE_RATE

    # Truncate to last 240s if audio exceeds Moshi context length
    if waveform.shape[-1] > MAX_CONTEXT_SAMPLES:
        logger.info(f"  Truncating {duration:.1f}s audio to last {MAX_CONTEXT_DURATION:.0f}s")
        system_audio = system_audio[:, -MAX_CONTEXT_SAMPLES:]
        user_audio = user_audio[:, -MAX_CONTEXT_SAMPLES:]
        # Compute time offset for alignment filtering
        time_offset = duration - MAX_CONTEXT_DURATION
        duration = MAX_CONTEXT_DURATION
    else:
        time_offset = 0.0

    # Build alignments from metadata
    alignments = build_alignments_from_metadata(meta)

    # Filter and shift alignments if truncated
    if time_offset > 0:
        alignments = [
            (word, (start - time_offset, end - time_offset), speaker)
            for word, (start, end), speaker in alignments
            if end > time_offset
        ]

    # Create text token stream via Interleaver
    text_tokens = interleaver.prepare_item(
        alignments, duration, SYSTEM_SPEAKER_LABEL
    )  # [1, 1, T_frames]

    result = process_audio_chunk(user_audio, system_audio, text_tokens, mimi, lm, device)

    # Save as numpy
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, result.numpy())
    logger.info(f"Saved {audio_path.name} -> {result.shape}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description='Compute Moshi features for UR-Funny2')
    parser.add_argument('--audio_dir', type=str, required=True, help='Directory with .wav files')
    parser.add_argument('--metadata_dir', type=str, required=True, help='Directory with language_sdk.pkl etc.')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for .npy features')
    parser.add_argument('--hf_repo', type=str, default='kyutai/moshiko-pytorch-bf16')
    parser.add_argument('--moshi_weights', type=str, default=None)
    parser.add_argument('--mimi_weights', type=str, default=None)
    parser.add_argument('--config_path', type=str, default=None)
    parser.add_argument('--limit', type=int, default=None, help='Process only first N files')
    args = parser.parse_args()

    # Distributed setup
    is_distributed = 'RANK' in os.environ
    if is_distributed:
        import torch.distributed as dist
        dist.init_process_group(backend='nccl')
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(device)
    else:
        rank = 0
        world_size = 1
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    setup_logging(rank)

    audio_dir = Path(args.audio_dir)
    metadata_dir = Path(args.metadata_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    logger.info("Loading language_sdk.pkl...")
    with open(metadata_dir / 'language_sdk.pkl', 'rb') as f:
        language_sdk = pickle.load(f)
    logger.info(f"Loaded metadata for {len(language_sdk)} samples")

    # Enumerate audio files that have metadata
    audio_files = []
    for audio_path in sorted(audio_dir.glob('*.wav')):
        sample_id = int(audio_path.stem)
        if sample_id in language_sdk:
            audio_files.append((audio_path, sample_id))

    if args.limit:
        audio_files = audio_files[:args.limit]

    logger.info(f"Found {len(audio_files)} audio files with metadata")

    # Distribute across GPUs
    my_files = audio_files[rank::world_size]
    logger.info(f"Rank {rank}/{world_size}: processing {len(my_files)} files")

    # Load models
    logger.info("Loading models...")
    mimi, lm, tokenizer = load_models(
        args.hf_repo, device,
        moshi_weights=args.moshi_weights,
        mimi_weights=args.mimi_weights,
        config_path=args.config_path,
    )
    logger.info("Models loaded")

    # Initialize Interleaver
    interleaver = Interleaver(
        tokenizer=tokenizer,
        audio_frame_rate=FRAME_RATE,
        text_padding=3,
        end_of_text_padding=0,
        zero_padding=2048,
        device=device,
    )

    # Process files
    for i, (audio_path, sample_id) in enumerate(my_files):
        output_path = output_dir / f"{sample_id}.npy"
        meta = language_sdk[sample_id]
        try:
            process_single_file(audio_path, output_path, meta, mimi, lm, interleaver, device)
        except Exception as e:
            logger.error(f"Failed {audio_path.name}: {e}", exc_info=True)
            continue

        if (i + 1) % 100 == 0:
            logger.info(f"Progress: {i+1}/{len(my_files)}")

    logger.info("Done")

    if is_distributed:
        import torch.distributed as dist
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
