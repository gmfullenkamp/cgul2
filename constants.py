"""Holds the constants for the repo."""

from pathlib import Path

# Holds the repository root absolute path
repo_root = Path(__file__).resolve().parents[0]  # go up one levels

# Holds the models directory and embeddings directory absolute paths
models_dir = repo_root / "models"
if not Path.exists(models_dir):
    Path.mkdir(models_dir, parents=True)
embeddings_dir = models_dir / "embeddings"
if not Path.exists(embeddings_dir):
    Path.mkdir(embeddings_dir, parents=True)

# Holds the vector store directory absolute path
vector_store_dir = repo_root / "vector_store"
if not Path.exists(vector_store_dir):
    Path.mkdir(vector_store_dir, parents=True)
