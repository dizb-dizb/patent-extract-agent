"""Generate stable dataset version hash for reproducibility."""

from __future__ import annotations

import hashlib
from pathlib import Path


def dataset_version(path: Path) -> str:
    """Hash of file content for reproducibility."""
    text = path.read_text(encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:16]


def dataset_version_multi(paths: list[Path]) -> str:
    """Combined hash of multiple files."""
    h = hashlib.sha256()
    for p in sorted(paths, key=lambda x: str(x)):
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()[:16]
