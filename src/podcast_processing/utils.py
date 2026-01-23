"""Utility functions for podcast processing."""

import logging
from pathlib import Path


def setup_logging(output_dir: Path, log_filename: str = 'run.log') -> logging.Logger:
    """Setup logging configuration.

    Args:
        output_dir: Directory to save log file
        log_filename: Name of log file (default: 'run.log')

    Returns:
        Configured logger
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / log_filename

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging to {log_path}")

    return logger
