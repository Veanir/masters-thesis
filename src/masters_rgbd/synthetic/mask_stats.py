"""Mask-derived quality statistics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class MaskStats:
    pixel_count: int
    image_fraction: float
    bbox_xyxy: tuple[int, int, int, int] | None
    bbox_area: int
    bbox_area_fraction: float
    bbox_fill_fraction: float

    def touches_border(self, *, height: int, width: int, margin: int) -> bool:
        if self.bbox_xyxy is None or margin <= 0:
            return False
        x_min, y_min, x_max, y_max = self.bbox_xyxy
        return (
            x_min < margin or y_min < margin or x_max >= width - margin or y_max >= height - margin
        )

    def center_offset_fraction(self, *, height: int, width: int) -> float:
        if self.bbox_xyxy is None or height <= 0 or width <= 0:
            return 0.0
        x_min, y_min, x_max, y_max = self.bbox_xyxy
        bbox_cx = 0.5 * float(x_min + x_max)
        bbox_cy = 0.5 * float(y_min + y_max)
        image_cx = 0.5 * float(width - 1)
        image_cy = 0.5 * float(height - 1)
        return max(abs(bbox_cx - image_cx) / float(width), abs(bbox_cy - image_cy) / float(height))


def compute_binary_mask_stats(mask: NDArray[np.bool_] | NDArray[np.uint8]) -> MaskStats:
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {binary.shape}")

    image_area = int(binary.size)
    pixel_count = int(np.count_nonzero(binary))
    if pixel_count == 0 or image_area == 0:
        return MaskStats(
            pixel_count=pixel_count,
            image_fraction=0.0,
            bbox_xyxy=None,
            bbox_area=0,
            bbox_area_fraction=0.0,
            bbox_fill_fraction=0.0,
        )

    rows, cols = np.nonzero(binary)
    x_min = int(cols.min())
    x_max = int(cols.max())
    y_min = int(rows.min())
    y_max = int(rows.max())
    bbox_area = int((x_max - x_min + 1) * (y_max - y_min + 1))
    return MaskStats(
        pixel_count=pixel_count,
        image_fraction=float(pixel_count / image_area),
        bbox_xyxy=(x_min, y_min, x_max, y_max),
        bbox_area=bbox_area,
        bbox_area_fraction=float(bbox_area / image_area),
        bbox_fill_fraction=float(pixel_count / bbox_area) if bbox_area else 0.0,
    )


def compute_instance_mask_stats(mask: NDArray[np.integer], instance_id: int) -> MaskStats:
    return compute_binary_mask_stats(np.asarray(mask) == int(instance_id))
