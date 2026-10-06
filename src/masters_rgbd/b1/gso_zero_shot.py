"""Evaluate the frozen P512/adaptive system zero-shot on admitted GSO renders."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from masters_rgbd.contracts.geometry import (
    ModelROI,
)


class GSOZeroShotError(RuntimeError):
    """A frozen model, cohort, render, or surface-only invariant differs."""


@dataclass(frozen=True, slots=True)
class GSORenderedSample:
    asset_id: str
    category: str
    rgb: np.ndarray
    depth_m: np.ndarray
    mask: np.ndarray
    partial_points_camera_m: np.ndarray
    intrinsics: np.ndarray
    roi: ModelROI
    target_surface_points_camera_m: np.ndarray
    input_sha256: dict[str, str]
    scene_scale: float
    depth_alignment_p95_m: float
    depth_alignment_max_m: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GSOZeroShotError(f"cannot read {label}: {path}") from error
    if not isinstance(value, dict):
        raise GSOZeroShotError(f"{label} is not a JSON object")
    return value
