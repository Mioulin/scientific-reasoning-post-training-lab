"""Fetch only the immutable, explicitly configured public model revision."""

import json
from pathlib import Path

from huggingface_hub import snapshot_download

cfg = json.loads(Path("configs/model.json").read_text())
snapshot_download(
    cfg["model_id"],
    revision=cfg["revision"],
    cache_dir=cfg["cache_dir"],
    allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "LICENSE", "README.md"],
)
