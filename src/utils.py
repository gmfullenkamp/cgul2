"""Utilities for LLM citation machine.

This module contains the utilities to help with downloading,
maintaining, and assisting LLMs and embedding models.
"""

from pathlib import Path

from huggingface_hub import snapshot_download

from constants import models_dir


def download_model(model_id: str) -> Path:
    """Download a model from Hugging Face Hub into the repo's models_dir."""
    target_dir = models_dir / Path(model_id).name
    target_dir.mkdir(parents=True, exist_ok=True)

    local_dir = snapshot_download(
        repo_id=model_id,
        local_dir=target_dir,
        local_dir_use_symlinks=False,  # ensures actual files are copied, not symlinks
    )
    return Path(local_dir)
