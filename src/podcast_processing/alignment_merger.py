"""Merge transcript and diarization data for accurate word-level speaker alignment."""

import json
import logging
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)

# Type alias for alignment tuple: (word, (start_sec, end_sec), speaker_label)
Alignment = Tuple[str, Tuple[float, float], str]


class AlignmentMerger:
    """Merge transcript word timings with diarization speaker labels.

    **Critical**: Transcript speaker IDs are inaccurate, so we ONLY use diarization
    segment information to determine speaker labels based on time overlap.
    """

    def merge_transcript_with_diarization(
        self,
        transcript_path: Path,
        diarization_path: Path
    ) -> List[Alignment]:
        """Merge transcript and diarization to create word-level speaker alignments.

        Algorithm:
        1. Extract all words with time intervals from transcript (ignore speaker IDs)
        2. Use diarization segments (time intervals + SPEAKER_XX labels)
        3. For each word, calculate time overlap with diarization segments
        4. Assign the speaker with maximum time overlap

        Args:
            transcript_path: Path to transcript JSON file
            diarization_path: Path to diarization metadata JSON file

        Returns:
            List of (word, (start_sec, end_sec), speaker_label) tuples
        """
        with open(transcript_path) as f:
            transcript = json.load(f)

        with open(diarization_path) as f:
            diarization = json.load(f)

        # Extract diarization segments
        diar_segments = diarization.get('segments', [])

        if not diar_segments:
            logger.warning(f"No diarization segments found in {diarization_path}")
            return []

        # Process transcript segments and assign speakers based on time overlap
        alignments = []

        for segment in transcript.get('segments', []):
            # Get the best hypothesis (nbest[0]) which contains words
            nbest = segment.get('nbest', [])
            if not nbest:
                continue

            words = nbest[0].get('words', [])

            for word_info in words:
                word = word_info.get('text', '')
                if not word:
                    continue

                # Convert offset/duration from microseconds (10^-7 seconds) to seconds
                start_sec = word_info.get('offset', 0) / 1e7
                duration_micro = word_info.get('duration', 0)
                end_sec = (word_info.get('offset', 0) + duration_micro) / 1e7

                # Find speaker from diarization by time overlap
                speaker_label = self._find_speaker_by_time(
                    start_sec, end_sec, diar_segments
                )

                alignments.append((word, (start_sec, end_sec), speaker_label))

        logger.info(f"Created {len(alignments)} word alignments")
        return alignments

    def _find_speaker_by_time(
        self,
        start: float,
        end: float,
        diar_segments: List[dict]
    ) -> str:
        """Find the speaker with maximum time overlap with the given interval.

        Args:
            start: Start time in seconds
            end: End time in seconds
            diar_segments: List of diarization segments with 'start', 'end', 'speaker_id'

        Returns:
            Speaker label (e.g., 'SPEAKER_00') or fallback 'unknown_speaker'
        """
        max_overlap = 0.0
        best_speaker = None

        for seg in diar_segments:
            seg_start = seg.get('start', 0)
            seg_end = seg.get('end', 0)

            # Calculate overlap
            overlap_start = max(start, seg_start)
            overlap_end = min(end, seg_end)
            overlap = max(0.0, overlap_end - overlap_start)

            if overlap > max_overlap:
                max_overlap = overlap
                best_speaker = seg.get('speaker_id')

        # Fallback if no overlap found
        if best_speaker is None:
            logger.warning(f"No speaker found for interval [{start:.2f}, {end:.2f}], using fallback")
            return "unknown_speaker"

        return best_speaker
