"""Shared generation result."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GenerationResult:
    scene_dirs: tuple[Path, ...]
