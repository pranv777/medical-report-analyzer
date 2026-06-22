"""
src/utils/logger.py
Structured logger using loguru with file rotation.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from loguru import logger as _logger


def setup_logger(
    log_dir: str = "logs",
    log_level: str = "INFO",
    log_file: Optional[str] = None,
) -> None:
    """
    Configure loguru with console + rotating file sink.
    Call once at application startup.
    """
    _logger.remove()

    # Console handler — colourised
    _logger.add(
        sys.stderr,
        level=log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # File handler — JSON structured
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    file_name = log_file or "medical_rag.log"
    _logger.add(
        Path(log_dir) / file_name,
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} — {message}",
        rotation="10 MB",
        retention="14 days",
        compression="zip",
        serialize=False,
    )


def get_logger(name: str):
    """Return a bound logger with the given module name."""
    return _logger.bind(name=name)
