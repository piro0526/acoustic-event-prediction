"""Core episode processing logic for computing transformer outputs."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

from finetune.data.interleaver import Interleaver
from podcast_processing.alignment_merger import AlignmentMerger, Alignment
from podcast_processing.episode_output_writer import EpisodeOutputWriter
from podcast_processing.label_generator import LabelGenerator

logger = logging.getLogger(__name__)


class EpisodeProcessor:
    """Process a single episode to extract transformer outputs.

    This class handles:
    1. Loading diarization and selecting top 2 speakers
    2. Loading and merging transcript/diarization for word alignments
    3. Loading speaker audio files
    4. Processing both speaker assignments (user/system role swaps)
    5. Encoding with Mimi, text tokenization with Interleaver, LM forward pass
    6. Saving results with metadata
    """

    def __init__(self, mimi, lm_model, tokenizer, device, shift_frames: int):
        """Initialize processor with models.

        Args:
            mimi: Mimi compression model for audio encoding
            lm_model: LMModel for transformer processing
            tokenizer: SentencePiece tokenizer
            device: Device to run on (e.g., 'cuda:0')
            shift_frames: Prediction shift value for label generation
        """
        self.mimi = mimi
        self.lm = lm_model
        self.tokenizer = tokenizer
        self.device = device
        self.shift_frames = shift_frames

        # Initialize Interleaver for text token stream creation
        self.interleaver = Interleaver(
            tokenizer=tokenizer,
            audio_frame_rate=12.5,  # Mimi frame rate
            text_padding=3,  # existing_text_padding_id from LM config
            end_of_text_padding=0,  # existing_text_end_padding_id
            zero_padding=2048,  # Special zero padding token (outside audio code range)
            device=device
        )

        self.alignment_merger = AlignmentMerger()
        self.label_generator = LabelGenerator(frame_rate=12.5)

    def process_episode(self, episode_info: Dict, output_root: Path, annotations_dir: Path):
        """Process a single episode and save features, labels, and metadata.

        Args:
            episode_info: Dict with 'split', 'name', 'audio_dir', 'diarization', 'transcript'
            output_root: Base output directory (not episode-specific path)
            annotations_dir: Directory containing episode_event_speaker_mapping annotations
        """
        logger.info(f"Processing episode: {episode_info['name']}")

        # Initialize output writer
        writer = EpisodeOutputWriter(
            output_root=output_root,
            episode_name=episode_info['name'],
            split=episode_info['split'],
            shift_frames=self.shift_frames
        )
        writer.prepare_directory()

        # Load laughter annotations
        annotation_path = (
            annotations_dir / episode_info['split'] /
            f"{episode_info['name']}.json"
        )

        if not annotation_path.exists():
            logger.warning(f"Annotation not found: {annotation_path}, skipping labels")
            laughter_events = []
        else:
            laughter_events = self.label_generator.load_laughter_events(annotation_path)

        # Load diarization and get top 2 speakers by total_time
        speakers = self._get_top_2_speakers(episode_info['diarization'])
        longer_spk, shorter_spk = speakers[0], speakers[1]

        logger.info(f"Top speakers: {longer_spk} (longer), {shorter_spk} (shorter)")

        # Load and merge alignments
        alignments = self.alignment_merger.merge_transcript_with_diarization(
            episode_info['transcript'],
            episode_info['diarization']
        )

        # Load audio files for both speakers
        user_audio_1, system_audio_1 = self._load_audios(
            episode_info['audio_dir'], longer_spk, shorter_spk
        )

        # Get episode duration from audio length
        duration = max(user_audio_1.shape[-1], system_audio_1.shape[-1]) / 24000.0

        logger.info(f"Episode duration: {duration:.2f}s")

        # Process both assignments
        assignments_info = [
            {'user': longer_spk, 'system': shorter_spk, 'user_audio': user_audio_1, 'system_audio': system_audio_1},
            {'user': shorter_spk, 'system': longer_spk, 'user_audio': system_audio_1, 'system_audio': user_audio_1}
        ]

        for assign_idx, assign_info in enumerate(assignments_info):
            logger.info(f"Processing Assignment {assign_idx}")

            # Check if features file exists
            features_path = writer.episode_dir / f"features_assignment_{assign_idx}.npy"

            if not features_path.exists():
                # Extract features
                features = self._process_assignment(
                    user_audio=assign_info['user_audio'],
                    system_audio=assign_info['system_audio'],
                    alignments=alignments,
                    system_speaker_label=assign_info['system'],
                    duration=duration
                )  # Returns [T, 4096]

                num_frames = features.shape[0]

                # Save features
                writer.save_features(assign_idx, features)
                logger.info(f"  Saved features: {num_frames} frames")
            else:
                # Load existing features to get num_frames
                existing_features = np.load(features_path)
                num_frames = existing_features.shape[0]
                logger.info(f"  Features already exist: {num_frames} frames")

            # Generate and save labels
            if laughter_events:
                labels = self.label_generator.create_labels_prediction(
                    laughter_events=laughter_events,
                    num_frames=num_frames,
                    user_speaker_id=assign_info['user'],
                    shift_frames=self.shift_frames
                )

                stats = self.label_generator.compute_label_statistics(labels)

                # Save labels
                writer.save_labels(assign_idx, labels)

                # Add statistics to metadata buffer
                writer.add_assignment_stats(
                    assignment_idx=assign_idx,
                    user_speaker_id=assign_info['user'],
                    system_speaker_id=assign_info['system'],
                    stats=stats
                )

                logger.info(f"  Saved labels (shift={self.shift_frames}): {stats['num_positive_frames']}/{num_frames} positive")

            # Clear GPU cache after each assignment
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Save metadata after all assignments are processed
        writer.save_metadata(episode_info, num_frames, duration)

        logger.info(f"Completed episode: {episode_info['name']}")

    def _process_assignment(
        self,
        user_audio: torch.Tensor,
        system_audio: torch.Tensor,
        alignments: List[Alignment],
        system_speaker_label: str,
        duration: float
    ) -> torch.Tensor:
        """Process one speaker assignment to get transformer output.

        Args:
            user_audio: User audio tensor [channels, samples]
            system_audio: System audio tensor [channels, samples]
            alignments: List of (word, (start, end), speaker) tuples
            system_speaker_label: Which speaker is the system (for text tokens)
            duration: Duration in seconds

        Returns:
            Transformer output tensor [T, 4096]
        """
        # Check if audio needs chunking (>3.5 minutes = 210 seconds)
        max_chunk_duration = 210  # seconds, conservative limit
        sample_rate = 24000
        max_chunk_samples = int(max_chunk_duration * sample_rate)

        user_audio_samples = user_audio.shape[-1]
        system_audio_samples = system_audio.shape[-1]
        max_samples = max(user_audio_samples, system_audio_samples)

        if max_samples > max_chunk_samples:
            logger.info(f"Audio is {duration:.1f}s, splitting into chunks for processing")
            return self._process_assignment_chunked(
                user_audio, system_audio, alignments, system_speaker_label, duration
            )

        with torch.no_grad():
            # Encode audios with Mimi -> [1, K, T_frames]
            # Move audio to device just before encoding to minimize memory footprint
            user_audio_gpu = user_audio[None, :, :].to(self.device)
            user_codes = self.mimi.encode(user_audio_gpu)
            del user_audio_gpu  # Free immediately

            system_audio_gpu = system_audio[None, :, :].to(self.device)
            system_codes = self.mimi.encode(system_audio_gpu)
            del system_audio_gpu  # Free immediately

            logger.debug(f"User codes shape: {user_codes.shape}, System codes shape: {system_codes.shape}")

            # Get text tokens for system speaker
            system_alignments = [
                a for a in alignments if a[2] == system_speaker_label
            ]

            # Sort alignments by time (Interleaver expects sorted)
            system_alignments = sorted(system_alignments, key=lambda x: x[1][0])

            logger.debug(f"System speaker has {len(system_alignments)} words")

            # Create text token stream -> [1, 1, T_frames]
            text_tokens = self.interleaver.prepare_item(
                system_alignments,
                duration,
                system_speaker_label
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

            logger.debug(f"Padded to max_t={max_t}")

            # Combine: [text_tokens, user_codes, system_codes] -> [1, K, T]
            # K = 1 (text) + n_q (user codes) + n_q (system codes)
            # Mimi outputs 32 codebooks, but Moshi LM uses only first n_q=16 codebooks
            # We use dep_q=8 for dependent modeling

            # Extract only the codebooks used by Moshi (first dep_q codebooks)
            dep_q = self.lm.dep_q  # Usually 8
            user_codes_subset = user_codes[:, :dep_q, :]
            system_codes_subset = system_codes[:, :dep_q, :]

            codes = torch.cat([text_tokens, user_codes_subset, system_codes_subset], dim=1)

            logger.debug(f"Combined codes shape: {codes.shape} (1 text + {dep_q} user + {dep_q} system = {1+dep_q*2})")

            # Apply acoustic delays for causal processing
            # This is crucial for acoustic event prediction to maintain causality
            B, K, T = codes.shape

            # Get delays from LM model
            delays = self.lm.delays
            assert len(delays) == K, f"Expected {K} delays, got {len(delays)}"

            # Import delay function
            from moshi.models.lm_utils import _delay_sequence

            # Apply delay sequence (shift each codebook by its delay)
            initial = self.lm._get_initial_token().expand(B, -1, -1)
            delayed_codes = _delay_sequence(delays, codes, initial)

            # Add initial tokens at the beginning and remove last time step
            delayed_codes = torch.cat([initial, delayed_codes], dim=2)

            logger.debug(f"Delayed codes shape: {delayed_codes.shape}")

            # Run through LM with delayed codes (causal, uses only past information)
            # delayed_codes[:, :, :-1] removes the last timestep which is never an input
            transformer_out, _ = self.lm.forward_text(delayed_codes[:, :, :-1])  # [1, T, 4096]

            logger.debug(f"Transformer output shape: {transformer_out.shape}")

            # Move to CPU immediately to free GPU memory
            result = transformer_out.squeeze(0).cpu()

            # Clear intermediate GPU tensors
            del user_codes, system_codes, user_codes_subset, system_codes_subset
            del text_tokens, codes, delayed_codes, transformer_out

        return result  # [T, 4096]

    def _process_assignment_chunked(
        self,
        user_audio: torch.Tensor,
        system_audio: torch.Tensor,
        alignments: List[Alignment],
        system_speaker_label: str,
        duration: float
    ) -> torch.Tensor:
        """Process long audio in chunks to avoid OOM.

        Args:
            Same as _process_assignment

        Returns:
            Transformer output tensor [T, 4096]
        """
        sample_rate = 24000
        chunk_duration = 210  # 3.5 minutes per chunk
        chunk_samples = int(chunk_duration * sample_rate)

        # Determine number of chunks needed
        max_samples = max(user_audio.shape[-1], system_audio.shape[-1])
        num_chunks = (max_samples + chunk_samples - 1) // chunk_samples

        logger.info(f"Processing {duration:.1f}s audio in {num_chunks} chunks of {chunk_duration}s each")

        all_transformer_outs = []

        for chunk_idx in range(num_chunks):
            start_sample = chunk_idx * chunk_samples
            end_sample = min((chunk_idx + 1) * chunk_samples, max_samples)
            chunk_actual_duration = (end_sample - start_sample) / sample_rate

            logger.info(f"  Processing chunk {chunk_idx+1}/{num_chunks} ({chunk_actual_duration:.1f}s)")

            # Extract chunk from audios
            user_chunk = user_audio[:, start_sample:end_sample] if start_sample < user_audio.shape[-1] else \
                        torch.zeros((user_audio.shape[0], end_sample - start_sample), device=user_audio.device)
            system_chunk = system_audio[:, start_sample:end_sample] if start_sample < system_audio.shape[-1] else \
                          torch.zeros((system_audio.shape[0], end_sample - start_sample), device=system_audio.device)

            # Filter alignments for this time range
            chunk_start_time = start_sample / sample_rate
            chunk_end_time = end_sample / sample_rate
            chunk_alignments = [
                a for a in alignments
                if a[1][0] >= chunk_start_time and a[1][0] < chunk_end_time
            ]

            # Adjust alignment times to be relative to chunk start
            chunk_alignments_adjusted = [
                (word, (start - chunk_start_time, end - chunk_start_time), speaker)
                for word, (start, end), speaker in chunk_alignments
            ]

            # Process this chunk (recursive call, but it won't chunk again due to size check)
            chunk_out = self._process_assignment(
                user_chunk, system_chunk, chunk_alignments_adjusted,
                system_speaker_label, chunk_actual_duration
            )

            all_transformer_outs.append(chunk_out)

            # Clear cache between chunks
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Concatenate all chunks
        result = torch.cat(all_transformer_outs, dim=0)
        logger.info(f"Combined {num_chunks} chunks into final output shape: {result.shape}")

        return result

    def _get_top_2_speakers(self, diarization_path: Path) -> List[str]:
        """Get top 2 speakers by total speaking time.

        Args:
            diarization_path: Path to diarization metadata JSON

        Returns:
            List of 2 speaker IDs (e.g., ['SPEAKER_00', 'SPEAKER_01'])
            Ordered by total_time descending
        """
        with open(diarization_path) as f:
            diar = json.load(f)

        speaker_stats = diar.get('speaker_statistics', {})

        if len(speaker_stats) < 2:
            raise ValueError(
                f"Episode has less than 2 speakers: {len(speaker_stats)}"
            )

        # Sort by total_time descending
        sorted_speakers = sorted(
            speaker_stats.items(),
            key=lambda x: x[1].get('total_time', 0),
            reverse=True
        )

        return [sorted_speakers[0][0], sorted_speakers[1][0]]

    def _load_audios(
        self,
        audio_dir: Path,
        speaker1: str,
        speaker2: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load audio files for two speakers.

        Args:
            audio_dir: Directory containing SPEAKER_XX.wav files
            speaker1: First speaker ID (e.g., 'SPEAKER_00')
            speaker2: Second speaker ID

        Returns:
            Tuple of (audio1, audio2) tensors [channels, samples]
        """
        audio1_path = audio_dir / f"{speaker1}.wav"
        audio2_path = audio_dir / f"{speaker2}.wav"

        if not audio1_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio1_path}")
        if not audio2_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio2_path}")

        audio1, sr1 = torchaudio.load(audio1_path)
        audio2, sr2 = torchaudio.load(audio2_path)

        if sr1 != 24000 or sr2 != 24000:
            raise ValueError(
                f"Expected 24kHz audio, got {sr1}Hz and {sr2}Hz"
            )

        logger.debug(f"Loaded {speaker1}: {audio1.shape}, {speaker2}: {audio2.shape}")

        return audio1, audio2
