"""Utilities for configuring project-wide logging based on loguru."""

import sys
from pathlib import Path

from loguru import logger

# 直接暴露 loguru logger，无需二次封装
get_logger = logger


def setup_logger(level: str = "INFO", log_file: str | Path | None = None) -> None:
    """Configure loguru handlers for stderr and optional file output."""
    logger.remove()

    log_format = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{line} | {message}"
    logger.add(sys.stderr, level=level, format=log_format)

    if log_file is not None:
        file_path = Path(log_file)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(file_path),
            level=level,
            format=log_format,
            rotation="10 MB",
            retention=5,
        )
