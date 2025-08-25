"""Utilities for LLM citation machine.

This module contains the utilities to help with downloading,
maintaining, and assisting LLMs, embedding models, and other
utilities.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import ClassVar

from huggingface_hub import snapshot_download

from constants import models_dir


def download_model(model_id: str) -> Path:
    """Download a model from Hugging Face Hub into the repo's models_dir."""
    target_dir = models_dir / Path(model_id).name
    if not Path.exists(target_dir):
        target_dir.mkdir(parents=True, exist_ok=True)
        local_dir = snapshot_download(
            repo_id=model_id,
            local_dir=target_dir,
            local_dir_use_symlinks=False,  # ensures actual files are copied
        )
    else:
        local_dir = target_dir
    return Path(local_dir)

class ColoredFormatter(logging.Formatter):
    """Colored formatter for more excitting logs."""

    # ANSI escape sequences for colors
    COLORS: ClassVar[dict[str, str]] = {
        "DEBUG": "\033[94m",    # light blue
        "INFO": "\033[92m",     # green
        "WARNING": "\033[93m",  # yellow
        "ERROR": "\033[91m",    # orange/red
        "CRITICAL": "\033[31m", # bright red
    }
    RESET: ClassVar[str] = "\033[0m"

    def __init__(self, fmt: str | None, *, use_colors: bool = True) -> None:
        """Initialize the color formatter class."""
        super().__init__(fmt)
        self.use_colors = use_colors

    def format(self, record: logging.LogRecord) -> str:
        """Format for the log."""
        message = super().format(record)
        if self.use_colors:
            color = self.COLORS.get(record.levelname, self.RESET)
            return f"{color}{message}{self.RESET}"
        return message


class ColoredLogger(logging.Logger):
    """Colored logger class to make logging more exciting."""

    def __init__(self, name: str, level: logging._Level = logging.DEBUG) -> None:
        """Initialize colored logger."""
        super().__init__(name, level)

        console_handler = logging.StreamHandler(sys.stdout)

        # Enable colors only if stdout is a terminal
        use_colors = sys.stdout.isatty()
        formatter = ColoredFormatter("%(levelname)s: %(message)s",
                                     use_colors=use_colors)
        console_handler.setFormatter(formatter)

        self.addHandler(console_handler)
        self.propagate = False  # prevent duplicate logs

clogger = ColoredLogger("ColoredLogger")


# Example usage
if __name__ == "__main__":
    clogger.debug("This is a debug message (light blue)")
    clogger.info("This is an info message (green)")
    clogger.warning("This is a warning message (yellow)")
    clogger.error("This is an error message (orange-ish red)")
    clogger.critical("This is a critical message (red)")
