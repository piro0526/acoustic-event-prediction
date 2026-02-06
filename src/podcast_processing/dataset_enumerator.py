"""Dataset enumeration for PodcastFillers dataset."""

from pathlib import Path
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class DatasetEnumerator:
    """Enumerate all episodes in the PodcastFillers dataset.

    Args:
        dataset_root: Root directory of the PodcastFillers dataset
    """

    def __init__(self, dataset_root: str | Path):
        self.dataset_root = Path(dataset_root)

        if not self.dataset_root.exists():
            raise ValueError(f"Dataset root does not exist: {self.dataset_root}")

    def enumerate_episodes(self) -> List[Dict]:
        """Enumerate all episodes across train/test/validation splits.

        Returns:
            List of dictionaries containing episode information:
            [{
                'split': str,  # 'train', 'test', or 'validation'
                'name': str,   # episode name
                'audio_dir': Path,  # directory containing SPEAKER_XX.wav files
                'diarization': Path,  # path to diarization metadata.json
                'transcript': Path    # path to transcript .json file
            }]
        """
        episodes = []

        for split in ['train', 'test', 'validation']:
            audio_base_dir = self.dataset_root / 'audio' / 'episode_wav_dial' / split

            if not audio_base_dir.exists():
                logger.warning(f"Split directory does not exist: {audio_base_dir}")
                continue

            for episode_dir in sorted(audio_base_dir.iterdir()):
                if not episode_dir.is_dir():
                    continue

                episode_name = episode_dir.name

                diarization_path = (
                    self.dataset_root / 'metadata' / 'episode_diarizations' /
                    split / f"{episode_name}.json"
                )

                transcript_path = (
                    self.dataset_root / 'metadata' / 'episode_transcripts' /
                    split / f"{episode_name}.json"
                )

                episodes.append({
                    'split': split,
                    'name': episode_name,
                    'audio_dir': episode_dir,
                    'diarization': diarization_path,
                    'transcript': transcript_path
                })

        logger.info(f"Found {len(episodes)} episodes across all splits")
        return episodes
