"""Shared conservative depth-support classification for completion points."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from numpy.typing import NDArray

Float64Array = NDArray[np.float64]

BoolArray = NDArray[np.bool_]

Int64Array = NDArray[np.int64]

DEFAULT_DEPTH_MARGIN_M = 0.005


class DepthVisibilityLabel(IntEnum):
    """Classification relative to a fully observed local depth support."""

    UNKNOWN = 0
    SUPPORTED_NOT_HIDDEN = 1
    HIDDEN_BEHIND_OBSERVATION = 2


@dataclass(frozen=True, slots=True)
class DepthVisibilityClassification:
    labels: NDArray[np.uint8]
    rows: Int64Array
    columns: Int64Array
    query_depth_m: Float64Array
    minimum_support_depth_m: Float64Array
    maximum_support_depth_m: Float64Array
    full_support_mask: BoolArray

    @property
    def unknown_mask(self) -> BoolArray:
        return self.labels == int(DepthVisibilityLabel.UNKNOWN)

    @property
    def hidden_mask(self) -> BoolArray:
        return self.labels == int(DepthVisibilityLabel.HIDDEN_BEHIND_OBSERVATION)


def classify_depth_visibility(
    points_camera_m: object,
    *,
    depth_m: object,
    target_mask: object,
    intrinsics: object,
    depth_margin_m: float = DEFAULT_DEPTH_MARGIN_M,
) -> DepthVisibilityClassification:
    """Classify points using full valid 3x3 support and its maximum depth.

    Camera convention is +X right, +Y up and visible Z negative. A point is
    hidden only when its depth is strictly greater than the maximum positive
    depth in the complete 3x3 target support plus ``depth_margin_m``. Invalid,
    non-forward, border and incomplete-support points remain explicitly UNKNOWN.
    """

    points = np.asarray(points_camera_m, dtype=np.float64)
    depth = np.asarray(depth_m, dtype=np.float64)
    mask = np.asarray(target_mask, dtype=np.bool_)
    camera = np.asarray(intrinsics, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,):
        raise ValueError("points_camera_m must have shape [N, 3]")
    if depth.ndim != 2 or mask.shape != depth.shape:
        raise ValueError("depth_m and target_mask must have equal 2D shapes")
    if camera.shape != (4,) or not np.all(np.isfinite(camera)) or np.any(camera[:2] <= 0.0):
        raise ValueError("intrinsics must be finite [fx, fy, cx, cy] with positive focal lengths")
    if not np.isfinite(depth_margin_m) or depth_margin_m <= 0.0:
        raise ValueError("depth_margin_m must be positive and finite")

    count = len(points)
    labels = np.full(count, int(DepthVisibilityLabel.UNKNOWN), dtype=np.uint8)
    rows = np.full(count, -1, dtype=np.int64)
    columns = np.full(count, -1, dtype=np.int64)
    query_depth = -points[:, 2]
    finite_forward = np.all(np.isfinite(points), axis=1) & (query_depth > 0.0)
    valid_indices = np.flatnonzero(finite_forward)
    fx, fy, cx, cy = camera
    if len(valid_indices):
        valid_depth = query_depth[valid_indices]
        valid_points = points[valid_indices]
        columns[valid_indices] = np.floor(cx + fx * valid_points[:, 0] / valid_depth + 0.5).astype(
            np.int64
        )
        rows[valid_indices] = np.floor(cy - fy * valid_points[:, 1] / valid_depth + 0.5).astype(
            np.int64
        )

    height, width = depth.shape
    strict_projection = (
        finite_forward & (rows >= 1) & (rows < height - 1) & (columns >= 1) & (columns < width - 1)
    )
    full_support = np.zeros(count, dtype=np.bool_)
    minimum_depth = np.full(count, np.nan, dtype=np.float64)
    maximum_depth = np.full(count, np.nan, dtype=np.float64)
    spatial_indices = np.flatnonzero(strict_projection)
    if len(spatial_indices):
        supported = np.ones(len(spatial_indices), dtype=np.bool_)
        local_minimum = np.full(len(spatial_indices), np.inf, dtype=np.float64)
        local_maximum = np.full(len(spatial_indices), -np.inf, dtype=np.float64)
        spatial_rows = rows[spatial_indices]
        spatial_columns = columns[spatial_indices]
        for row_offset in (-1, 0, 1):
            for column_offset in (-1, 0, 1):
                local_depth = depth[spatial_rows + row_offset, spatial_columns + column_offset]
                local_mask = mask[spatial_rows + row_offset, spatial_columns + column_offset]
                supported &= local_mask & np.isfinite(local_depth) & (local_depth > 0.0)
                local_minimum = np.minimum(local_minimum, local_depth)
                local_maximum = np.maximum(local_maximum, local_depth)
        supported_indices = spatial_indices[supported]
        full_support[supported_indices] = True
        minimum_depth[supported_indices] = local_minimum[supported]
        maximum_depth[supported_indices] = local_maximum[supported]

    labels[full_support] = int(DepthVisibilityLabel.SUPPORTED_NOT_HIDDEN)
    hidden = full_support & (query_depth > maximum_depth + float(depth_margin_m))
    labels[hidden] = int(DepthVisibilityLabel.HIDDEN_BEHIND_OBSERVATION)
    for array in (
        labels,
        rows,
        columns,
        query_depth,
        minimum_depth,
        maximum_depth,
        full_support,
    ):
        array.setflags(write=False)
    return DepthVisibilityClassification(
        labels=labels,
        rows=rows,
        columns=columns,
        query_depth_m=query_depth,
        minimum_support_depth_m=minimum_depth,
        maximum_support_depth_m=maximum_depth,
        full_support_mask=full_support,
    )
