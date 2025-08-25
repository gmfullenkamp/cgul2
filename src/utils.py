"""Utilities for LLM citation machine.

This module contains the utilities to help with downloading,
maintaining, and assisting LLMs, embedding models, and other
utilities.
"""

from __future__ import annotations

import logging
import sys
import os
import subprocess
from pathlib import Path
from typing import ClassVar, Optional
import tempfile

from huggingface_hub import snapshot_download

from constants import models_dir


def download_model(model_id: str) -> Path:
    """Download a model from Hugging Face Hub into the repo's models_dir."""
    clogger.debug(f"Downloading {model_id}.")
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

def _run(cmd: list[str], cwd: Optional[Path] = None) -> None:
    """Run a command and raise with nice error text on failure."""
    clogger.debug(f"Running subprocess: {cmd}")
    try:
        subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{e}") from e

def _ensure_llama_cpp(llama_cpp_dir: Path) -> None:
    """Clone llama.cpp if not present and sanity‑check the converter."""
    if not llama_cpp_dir.exists():
        # Shallow clone is enough for the converter
        _run(["git", "clone", "--depth", "1", "https://github.com/ggerganov/llama.cpp.git", str(llama_cpp_dir.parent / llama_cpp_dir.name)])
    convert_py = llama_cpp_dir / "convert_hf_to_gguf.py"
    if not convert_py.exists():
        raise FileNotFoundError(f"Could not find converter at {convert_py}")

def download_and_convert_to_gguf(
    model_id: str,
    *,
    outtype: str = "q8_0",
    outfile: Optional[Path] = None,
    llama_cpp_dir: Optional[Path] = None,
    # NEW: automatically make an Ollama model from the GGUF
    register_with_ollama: bool = True,
    ollama_model_name: Optional[str] = None,
    ollama_host: Optional[str] = None,   # if set, exported to OLLAMA_HOST for the create/show calls
) -> Path:
    """
    Download a Hugging Face model to models_dir and convert it to GGUF using llama.cpp.
    Optionally, register the resulting .gguf with Ollama by creating a local model.

    Args:
        model_id: HF repo id, e.g. "lmsys/vicuna-13b-v1.5".
        outtype: Quantization / storage type: "q8_0", "f16", "f32", etc.
        outfile: Explicit path for the final .gguf. If None, uses:
                 <models_dir>/<repo_name>/<repo_name>.<outtype>.gguf
        llama_cpp_dir: Path to a local llama.cpp repo. If None, assumed at: <repo_root>/llama.cpp
        register_with_ollama: If True, create (or reuse) an Ollama model that points to this GGUF.
        ollama_model_name: Name to give the Ollama model. If None and registering, defaults to
                           "<repo_name>-{outtype}-local".
        ollama_host: Optional base URL (e.g., "http://127.0.0.1:11434") for Ollama CLI; if provided,
                     it's set in the environment for the 'ollama' subprocess calls.

    Returns:
        Path to the created (or existing) .gguf file.
    """
    # 1) Download HF model into local cache
    local_model_dir = download_model(model_id)

    # 2) Work out llama.cpp location
    default_llama_path = Path.cwd() / "llama.cpp"
    llama_cpp_dir = Path(llama_cpp_dir) if llama_cpp_dir else default_llama_path
    _ensure_llama_cpp(llama_cpp_dir)

    # 3) Decide output file
    repo_name = Path(model_id).name  # e.g., "vicuna-13b-v1.5"
    if outfile is None:
        outfile = local_model_dir / f"{repo_name}.{outtype}.gguf"
    outfile = Path(outfile)

    # 4) Convert if needed
    if not outfile.exists():
        clogger.debug(f"Converting {model_id} to GGUF file.")
        convert_py = llama_cpp_dir / "convert_hf_to_gguf.py"
        cmd = [
            sys.executable,
            str(convert_py),
            str(local_model_dir),       # pass local HF dir
            "--outfile", str(outfile),
            "--outtype", outtype,
        ]
        _run(cmd)  # reuse your repo's runner (raises on failure)

        if not outfile.exists():
            raise RuntimeError(f"GGUF conversion reported success but file not found: {outfile}")
    else:
        clogger.debug(f"GGUF already exists at {outfile}; skipping conversion.")

    # 5) Optionally register with Ollama
    if register_with_ollama:
        # Default model name if not provided
        model_name = ollama_model_name or f"{repo_name}-{outtype}-local"

        # Quick idempotency check: does model already exist in Ollama?
        try:
            _run([sys.executable, "-m", "ollama", "show", model_name])
            clogger.info(f"Ollama model '{model_name}' already exists; skipping create.")
        except subprocess.CalledProcessError:
            # Create a temporary Modelfile that points to our GGUF
            gguf_posix = Path(outfile).as_posix()  # handles Windows backslashes
            with tempfile.TemporaryDirectory() as td:
                modelfile = Path(td) / "Modelfile"
                modelfile.write_text(f"FROM {gguf_posix}\n", encoding="utf-8")
                clogger.info(f"Creating Ollama model '{model_name}' from {gguf_posix}")
                _run([sys.executable, "-m", "ollama", "create", model_name, "-f", str(modelfile)])

        # Optional: quick sanity run (comment out if you don’t want a test invoke)
        # _run_with_env(["ollama", "run", model_name, "--prompt", "hi"],
        #               extra_env={"OLLAMA_HOST": ollama_host}, check=False)

        clogger.info(f"Ollama model ready: {model_name}")
        outfile = model_name

    return outfile

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
