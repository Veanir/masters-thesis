"""Verify pixel-centre ray identity, masking, and no depth interpolation."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "ray_hope_camera", Path(__file__).parents[1] / "scripts/preliminary_hope/camera_calibration.py"
)
camera = importlib.util.module_from_spec(spec)
spec.loader.exec_module(camera)


def inputs():
    return (
        np.full((1080, 1920, 3), 100, np.uint8),
        np.full((1080, 1920), 837, np.uint16),
        np.ones((1080, 1920), bool),
        np.array([[1390.53, 0, 964.957], [0, 1386.99, 522.586], [0, 0, 1.0]]),
    )


def test_all_retained_sensor_rays_match_original_pixel_centres():
    rgb, depth, mask, k = inputs()
    rgb_out, depth_out, mask_out, k_out = camera.letterbox_hope(rgb, depth, mask, k)
    v, u = np.nonzero(mask_out)
    output_rays = np.column_stack((u, v, np.ones(len(u)))) @ np.linalg.inv(k_out).T
    source_rays = (
        np.column_stack((3 * u + 1, 3 * (v - 60) + 1, np.ones(len(u)))) @ np.linalg.inv(k).T
    )
    assert np.max(np.abs(output_rays - source_rays)) < 5e-16
    assert mask_out.sum() == 360 * 640
    assert np.all(depth_out[mask_out] == 837)
    assert np.all(rgb_out[mask_out] == 100)
    assert not mask_out[:60].any() and not mask_out[420:].any()
    assert not depth_out[~mask_out].any() and not rgb_out[~mask_out].any()
    assert k[0, 0] == 1390.53  # Caller calibration remains unchanged.


def test_mask_footprint_and_exact_depth_sample():
    rgb, depth, mask, k = inputs()
    mask[0, 0] = False
    depth[1, 4] = 943
    depth[0, 3] = 1122
    rgb[:3, 3:6] = 20
    rgb[1, 4] = 29
    out_rgb, out_depth, out_mask, _ = camera.letterbox_hope(rgb, depth, mask, k)
    assert not out_mask[60, 0] and not out_rgb[60, 0].any()
    assert out_depth[60, 1] == 943
    assert np.all(out_rgb[60, 1] == 21)


def test_other_resolution_is_not_silently_warped():
    rgb, depth, mask, k = inputs()
    with pytest.raises(ValueError, match="1920x1080"):
        camera.letterbox_hope(rgb[:480, :640], depth[:480, :640], mask[:480, :640], k)
