"""Multi-speaker episode processing with flexible user/system assignment.

This module processes episodes using original mixed audio and diarization metadata
to create user/system audio separation. It supports variable numbers of user speakers
based on speaking ratio threshold.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

from finetune.data.interleaver import Interleaver
from podcast_processing.alignment_merger import AlignmentMerger, Alignment
from podcast_processing.audio_masking import create_all_laughter_mask
from podcast_processing.episode_output_writer import MultiSpeakerOutputWriter
from podcast_processing.label_generator import LabelGenerator

logger = logging.getLogger(__name__)


class MultiSpeakerProcessor:
    """Process episodes with flexible user/system speaker assignment.

    This processor:
    1. Loads original mixed audio (not speaker-separated)
    2. Uses diarization to create speaker masks
    3. Selects speakers with >= min_speaker_share as potential users
    4. For each user, combines all other speakers as system
    5. Applies laughter masking to both user and system audio
    6. Generates labels from ALL speakers' laughter
    """

    # Mimi encoder expects 24kHz audio
    TARGET_SAMPLE_RATE = 24000
    # Original audio files are at 16kHz
    SOURCE_SAMPLE_RATE = 16000
    FRAME_RATE = 12.5

    def __init__(
        self,
        mimi,
        lm_model,
        tokenizer,
        device,
        shift_frames: int,
        min_speaker_share: float = 0.20,
        mask_laughter: bool = True
    ):
        """Initialize processor with models.

        Args:
            mimi: Mimi compression model for audio encoding
            lm_model: LMModel for transformer processing
            tokenizer: SentencePiece tokenizer
            device: Device to run on (e.g., 'cuda:0')
            shift_frames: Prediction shift value for label generation
            min_speaker_share: Minimum speaking ratio for user speaker (default: 0.20)
            mask_laughter: Whether to mask laughter in both audio streams (default: True)
        """
        self.mimi = mimi
        self.lm = lm_model
        self.tokenizer = tokenizer
        self.device = device
        self.shift_frames = shift_frames
        self.min_speaker_share = min_speaker_share
        self.mask_laughter = mask_laughter

        # Initialize Interleaver for text token stream creation
        self.interleaver = Interleaver(
            tokenizer=tokenizer,
            audio_frame_rate=self.FRAME_RATE,
            text_padding=3,
            end_of_text_padding=0,
            zero_padding=2048,
            device=device
        )

        self.alignment_merger = AlignmentMerger()
        self.label_generator = LabelGenerator(frame_rate=self.FRAME_RATE)

    def process_episode(
        self,
        episode_info: Dict,
        output_root: Path,
        annotations_dir: Path
    ):
        """Process a single episode and save features, labels, and metadata.

        Args:
            episode_info: Dict with:
                - 'split': Dataset split (train/validation/test)
                - 'name': Episode name
                - 'audio_path': Path to original mixed audio file
                - 'diarization': Path to diarization JSON
                - 'transcript': Path to transcript JSON
            output_root: Base output directory
            annotations_dir: Directory containing episode_event_speaker_mapping annotations
        """
        logger.info(f"Processing episode: {episode_info['name']}")

        # Load diarization data
        diarization = self._load_diarization(episode_info['diarization'])

        # Get all speakers and valid user speakers
        all_speakers = self._get_all_speakers(diarization)
        valid_users = self._get_valid_user_speakers(diarization)

        if not valid_users:
            logger.warning(
                f"No speakers with >= {self.min_speaker_share*100:.0f}% share in "
                f"{episode_info['name']}, skipping"
            )
            return

        logger.info(f"All speakers: {all_speakers}")
        logger.info(f"Valid user speakers (>={self.min_speaker_share*100:.0f}%): {valid_users}")

        # Initialize output writer
        writer = MultiSpeakerOutputWriter(
            output_root=output_root,
            episode_name=episode_info['name'],
            split=episode_info['split'],
            shift_frames=self.shift_frames
        )
        writer.prepare_directory()

        # Load laughter annotations
        annotation_path = (
            annotations_dir / episode_info['split'] / f"{episode_info['name']}.json"
        )

        if not annotation_path.exists():
            logger.warning(f"Annotation not found: {annotation_path}, skipping labels")
            laughter_events = []
        else:
            laughter_events = self.label_generator.load_laughter_events(annotation_path)

        # Load original mixed audio
        original_audio = self._load_original_audio(episode_info['audio_path'])
        num_samples = original_audio.shape[-1]
        duration = num_samples / self.TARGET_SAMPLE_RATE

        logger.info(f"Episode duration: {duration:.2f}s, samples: {num_samples}")

        # Create laughter mask (all speakers) if enabled
        laughter_mask = None
        if self.mask_laughter and laughter_events:
            laughter_mask = create_all_laughter_mask(
                laughter_events=laughter_events,
                duration=duration,
                sample_rate=self.TARGET_SAMPLE_RATE,
                num_samples=num_samples
            )
            if laughter_mask is not None:
                masked_samples = laughter_mask.sum()
                logger.info(
                    f"Created laughter mask: {masked_samples}/{num_samples} samples "
                    f"({100.0 * masked_samples / num_samples:.2f}%)"
                )

        # Load and merge alignments for text tokens
        alignments = self.alignment_merger.merge_transcript_with_diarization(
            episode_info['transcript'],
            episode_info['diarization']
        )

        num_frames = None  # Will be set after first assignment processing

        # Process each valid user speaker
        for assign_idx, user_speaker in enumerate(valid_users):
            system_speakers = [s for s in all_speakers if s != user_speaker]
            user_ratio = diarization['speaker_statistics'][user_speaker]['speaking_ratio']

            logger.info(
                f"Processing Assignment {assign_idx}: "
                f"user={user_speaker} ({user_ratio*100:.1f}%), "
                f"system={system_speakers}"
            )

            # Check if features file already exists
            features_path = writer.episode_dir / f"features_assignment_{assign_idx}.npy"

            if not features_path.exists():
                # Create speaker masks from diarization
                user_mask = self._create_speaker_mask_from_diarization(
                    diarization, [user_speaker], num_samples
                )
                system_mask = self._create_speaker_mask_from_diarization(
                    diarization, system_speakers, num_samples
                )

                # Apply speaker masks to original audio
                user_audio = self._apply_speaker_mask(original_audio, user_mask)
                system_audio = self._apply_speaker_mask(original_audio, system_mask)

                # Apply laughter mask to both (if enabled)
                if laughter_mask is not None:
                    user_audio = self._apply_laughter_mask(user_audio, laughter_mask)
                    system_audio = self._apply_laughter_mask(system_audio, laughter_mask)

                # Get text tokens for system speakers
                system_alignments = [
                    a for a in alignments if a[2] in system_speakers
                ]
                system_alignments = sorted(system_alignments, key=lambda x: x[1][0])

                # Process through Mimi/Moshi
                features = self._process_assignment(
                    user_audio=user_audio,
                    system_audio=system_audio,
                    alignments=system_alignments,
                    duration=duration
                )

                num_frames = features.shape[0]

                # Save features
                writer.save_features(assign_idx, features)
                logger.info(f"  Saved features: {num_frames} frames")
            else:
                # Load existing features to get num_frames
                existing_features = np.load(features_path)
                num_frames = existing_features.shape[0]
                logger.info(f"  Features already exist: {num_frames} frames")

            # Generate labels (all speakers, no filtering)
            labels = self.label_generator.create_labels_all_speakers(
                laughter_events=laughter_events,
                num_frames=num_frames,
                shift_frames=self.shift_frames
            )

            stats = self.label_generator.compute_label_statistics(labels)

            # Save labels
            writer.save_labels(assign_idx, labels)

            # Add statistics to metadata buffer
            writer.add_assignment_stats(
                assignment_idx=assign_idx,
                user_speaker_id=user_speaker,
                user_speaking_ratio=user_ratio,
                system_speaker_ids=system_speakers,
                stats=stats
            )

            logger.info(
                f"  Saved labels (shift={self.shift_frames}): "
                f"{stats['num_positive_frames']}/{num_frames} positive "
                f"({stats['positive_rate']*100:.2f}%)"
            )

            # Clear GPU cache after each assignment
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Save metadata after all assignments
        writer.save_metadata(
            episode_info=episode_info,
            num_frames=num_frames,
            duration=duration,
            all_speakers=all_speakers,
            min_speaker_share=self.min_speaker_share,
            mask_laughter=self.mask_laughter
        )

        logger.info(f"Completed episode: {episode_info['name']} ({len(valid_users)} assignments)")

    def _load_diarization(self, diarization_path: Path) -> Dict:
        """Load diarization JSON file.

        Args:
            diarization_path: Path to diarization JSON

        Returns:
            Diarization data dictionary
        """
        with open(diarization_path) as f:
            return json.load(f)

    def _get_all_speakers(self, diarization: Dict) -> List[str]:
        """Get all speaker IDs from diarization.

        Args:
            diarization: Diarization data dictionary

        Returns:
            List of all speaker IDs sorted by total_time descending
        """
        speaker_stats = diarization.get('speaker_statistics', {})
        sorted_speakers = sorted(
            speaker_stats.items(),
            key=lambda x: x[1].get('total_time', 0),
            reverse=True
        )
        return [spk_id for spk_id, _ in sorted_speakers]

    def _get_valid_user_speakers(self, diarization: Dict) -> List[str]:
        """Get speakers with speaking ratio >= min_speaker_share.

        Args:
            diarization: Diarization data dictionary

        Returns:
            List of valid speaker IDs sorted by total_time descending
        """
        speaker_stats = diarization.get('speaker_statistics', {})
        valid_speakers = [
            (spk_id, stats)
            for spk_id, stats in speaker_stats.items()
            if stats.get('speaking_ratio', 0) >= self.min_speaker_share
        ]
        valid_speakers.sort(key=lambda x: x[1].get('total_time', 0), reverse=True)
        return [spk_id for spk_id, _ in valid_speakers]

    def _load_original_audio(self, audio_path: Path) -> torch.Tensor:
        """Load original mixed audio file and resample to target rate.

        Args:
            audio_path: Path to audio file

        Returns:
            Audio tensor [channels, samples] at TARGET_SAMPLE_RATE (24kHz)
        """
        audio, sr = torchaudio.load(audio_path)

        logger.debug(f"Loaded audio: {audio.shape}, {sr}Hz")

        # Resample if needed (original files are 16kHz, Mimi expects 24kHz)
        if sr != self.TARGET_SAMPLE_RATE:
            logger.debug(f"Resampling from {sr}Hz to {self.TARGET_SAMPLE_RATE}Hz")
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr,
                new_freq=self.TARGET_SAMPLE_RATE
            )
            audio = resampler(audio)
            logger.debug(f"Resampled audio: {audio.shape}")

        return audio

    def _create_speaker_mask_from_diarization(
        self,
        diarization: Dict,
        speaker_ids: List[str],
        num_samples: int
    ) -> np.ndarray:
        """Create boolean mask where specified speakers are speaking.

        Args:
            diarization: Diarization data dictionary with 'segments' list
            speaker_ids: List of speaker IDs to include (True in mask)
            num_samples: Total number of audio samples

        Returns:
            Boolean mask [num_samples] where True = speaker is speaking
        """
        mask = np.zeros(num_samples, dtype=bool)

        for segment in diarization['segments']:
            if segment['speaker_id'] in speaker_ids:
                start_sample = int(segment['start'] * self.TARGET_SAMPLE_RATE)
                end_sample = int(segment['end'] * self.TARGET_SAMPLE_RATE)
                start_sample = max(0, start_sample)
                end_sample = min(num_samples, end_sample)
                if start_sample < end_sample:
                    mask[start_sample:end_sample] = True

        return mask

    def _apply_speaker_mask(self, audio: torch.Tensor, mask: np.ndarray) -> torch.Tensor:
        """Zero out audio where speaker is NOT speaking.

        Args:
            audio: Audio tensor [channels, samples]
            mask: Boolean mask [samples] where True = keep, False = zero

        Returns:
            Masked audio tensor [channels, samples]
        """
        masked_audio = audio.clone()
        mask_tensor = torch.from_numpy(~mask)  # Invert: False where we want to keep
        masked_audio[:, mask_tensor] = 0.0
        return masked_audio

    def _apply_laughter_mask(self, audio: torch.Tensor, mask: np.ndarray) -> torch.Tensor:
        """Zero out audio during laughter events.

        Args:
            audio: Audio tensor [channels, samples]
            mask: Boolean mask [samples] where True = mask (zero out)

        Returns:
            Masked audio tensor [channels, samples]
        """
        masked_audio = audio.clone()
        mask_tensor = torch.from_numpy(mask)

        # Resize mask if needed
        num_samples = audio.shape[-1]
        if len(mask_tensor) != num_samples:
            indices = torch.linspace(0, len(mask_tensor) - 1, num_samples).long()
            indices = indices.clamp(0, len(mask_tensor) - 1)
            mask_tensor = mask_tensor[indices]

        masked_audio[:, mask_tensor] = 0.0
        return masked_audio

    def _process_assignment(
        self,
        user_audio: torch.Tensor,
        system_audio: torch.Tensor,
        alignments: List[Alignment],
        duration: float
    ) -> torch.Tensor:
        """Process one speaker assignment to get transformer output.

        Args:
            user_audio: User audio tensor [channels, samples]
            system_audio: System audio tensor [channels, samples]
            alignments: List of (word, (start, end), speaker) tuples for system
            duration: Duration in seconds

        Returns:
            Transformer output tensor [T, 4096]
        """
        # Check if audio needs chunking (>3.5 minutes = 210 seconds)
        max_chunk_duration = 210
        max_chunk_samples = int(max_chunk_duration * self.TARGET_SAMPLE_RATE)

        max_samples = max(user_audio.shape[-1], system_audio.shape[-1])

        if max_samples > max_chunk_samples:
            logger.info(f"Audio is {duration:.1f}s, splitting into chunks")
            return self._process_assignment_chunked(
                user_audio, system_audio, alignments, duration
            )

        with torch.no_grad():
            # Encode audios with Mimi -> [1, K, T_frames]
            user_audio_gpu = user_audio[None, :, :].to(self.device)
            user_codes = self.mimi.encode(user_audio_gpu)
            del user_audio_gpu

            system_audio_gpu = system_audio[None, :, :].to(self.device)
            system_codes = self.mimi.encode(system_audio_gpu)
            del system_audio_gpu

            logger.debug(f"User codes: {user_codes.shape}, System codes: {system_codes.shape}")

            # Create text token stream for system speakers
            # Use a dummy speaker label since we have pre-filtered alignments
            text_tokens = self.interleaver.prepare_item(
                alignments,
                duration,
                "SYSTEM"  # Dummy label, not used for filtering
            )

            logger.debug(f"Text tokens shape: {text_tokens.shape}")

            # Align lengths (pad to max)
            max_t = max(
                user_codes.shape[-1],
                system_codes.shape[-1],
                text_tokens.shape[-1]
            )

            user_codes = F.pad(user_codes, (0, max_t - user_codes.shape[-1]))
            system_codes = F.pad(system_codes, (0, max_t - system_codes.shape[-1]))
            text_tokens = F.pad(text_tokens, (0, max_t - text_tokens.shape[-1]))

            # Extract only the codebooks used by Moshi
            dep_q = self.lm.dep_q
            user_codes_subset = user_codes[:, :dep_q, :]
            system_codes_subset = system_codes[:, :dep_q, :]

            codes = torch.cat([text_tokens, user_codes_subset, system_codes_subset], dim=1)

            logger.debug(f"Combined codes shape: {codes.shape}")

            # Apply acoustic delays
            B, K, T = codes.shape
            delays = self.lm.delays
            assert len(delays) == K, f"Expected {K} delays, got {len(delays)}"

            from moshi.models.lm_utils import _delay_sequence

            initial = self.lm._get_initial_token().expand(B, -1, -1)
            delayed_codes = _delay_sequence(delays, codes, initial)
            delayed_codes = torch.cat([initial, delayed_codes], dim=2)

            # Run through LM
            transformer_out, _ = self.lm.forward_text(delayed_codes[:, :, :-1])

            result = transformer_out.squeeze(0).cpu()

            del user_codes, system_codes, user_codes_subset, system_codes_subset
            del text_tokens, codes, delayed_codes, transformer_out

        return result

    def _process_assignment_chunked(
        self,
        user_audio: torch.Tensor,
        system_audio: torch.Tensor,
        alignments: List[Alignment],
        duration: float
    ) -> torch.Tensor:
        """Process long audio in chunks to avoid OOM.

        Args:
            Same as _process_assignment

        Returns:
            Transformer output tensor [T, 4096]
        """
        chunk_duration = 210
        chunk_samples = int(chunk_duration * self.TARGET_SAMPLE_RATE)

        max_samples = max(user_audio.shape[-1], system_audio.shape[-1])
        num_chunks = (max_samples + chunk_samples - 1) // chunk_samples

        logger.info(f"Processing {duration:.1f}s audio in {num_chunks} chunks")

        all_transformer_outs = []

        for chunk_idx in range(num_chunks):
            start_sample = chunk_idx * chunk_samples
            end_sample = min((chunk_idx + 1) * chunk_samples, max_samples)
            chunk_actual_duration = (end_sample - start_sample) / self.TARGET_SAMPLE_RATE

            logger.info(f"  Chunk {chunk_idx+1}/{num_chunks} ({chunk_actual_duration:.1f}s)")

            # Extract chunk from audios
            user_chunk = user_audio[:, start_sample:end_sample] if start_sample < user_audio.shape[-1] else \
                        torch.zeros((user_audio.shape[0], end_sample - start_sample))
            system_chunk = system_audio[:, start_sample:end_sample] if start_sample < system_audio.shape[-1] else \
                          torch.zeros((system_audio.shape[0], end_sample - start_sample))

            # Filter and adjust alignments for this chunk
            chunk_start_time = start_sample / self.TARGET_SAMPLE_RATE
            chunk_end_time = end_sample / self.TARGET_SAMPLE_RATE
            chunk_alignments = [
                (word, (start - chunk_start_time, end - chunk_start_time), speaker)
                for word, (start, end), speaker in alignments
                if start >= chunk_start_time and start < chunk_end_time
            ]

            chunk_out = self._process_assignment(
                user_chunk, system_chunk, chunk_alignments, chunk_actual_duration
            )

            all_transformer_outs.append(chunk_out)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        result = torch.cat(all_transformer_outs, dim=0)
        logger.info(f"Combined {num_chunks} chunks: {result.shape}")

        return result
