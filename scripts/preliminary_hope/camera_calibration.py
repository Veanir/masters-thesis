"""Observation-only, calibrated 1920x1080 HOPE to 640x480 RaySt3R transform."""

from __future__ import annotations

import numpy as np

from scripts.common.paths import script_help

script_help(__doc__, __name__)


def letterbox_hope(rgb, raw_depth, visible, intrinsics):
    """Decimate sensor depth at block centres; average RGB within visible blocks.

    No interpolation of depth, GT pose, object crop or reference mesh is used.
    The source ray at (3*u+1, 3*v+1) becomes output pixel (u, v+60).
    """
    rgb, raw_depth, visible = map(np.asarray, (rgb, raw_depth, visible))
    k = np.asarray(intrinsics, dtype=np.float64)
    if rgb.shape != (1080, 1920, 3) or rgb.dtype != np.uint8:
        raise ValueError("Expected native HOPE uint8 1920x1080 RGB")
    if raw_depth.shape != (1080, 1920) or not np.issubdtype(raw_depth.dtype, np.integer):
        raise ValueError("Expected native integer HOPE depth")
    if visible.shape != raw_depth.shape or visible.dtype != bool:
        raise ValueError("Expected native boolean instance mask")
    if (
        k.shape != (3, 3)
        or not np.isfinite(k).all()
        or k[0, 0] <= 0
        or k[1, 1] <= 0
        or not np.array_equal(k[2], [0, 0, 1])
        or k[0, 1] != 0
        or k[1, 0] != 0
    ):
        raise ValueError("Expected valid zero-skew pinhole intrinsics")
    # Require all source RGB pixels in the averaging footprint to belong to
    # the visible target. This prevents background colour leaking into RGB.
    retained = visible.reshape(360, 3, 640, 3).all(axis=(1, 3))
    colors = np.rint(rgb.reshape(360, 3, 640, 3, 3).mean(axis=(1, 3))).astype(np.uint8)
    depth = raw_depth[1::3, 1::3]
    image_out = np.zeros((480, 640, 3), dtype=np.uint8)
    depth_out = np.zeros((480, 640), dtype=raw_depth.dtype)
    mask_out = np.zeros((480, 640), dtype=bool)
    image_out[60:420] = np.where(retained[..., None], colors, 0)
    depth_out[60:420] = np.where(retained, depth, 0)
    mask_out[60:420] = retained
    transform = np.array([[1 / 3, 0, -1 / 3], [0, 1 / 3, 60 - 1 / 3], [0, 0, 1.0]])
    return image_out, depth_out, mask_out, transform @ k
