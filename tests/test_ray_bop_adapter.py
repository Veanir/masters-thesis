"""Analytic metric geometry and invalid observation checks for real RGB-D."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "ray_bop_adapter", Path(__file__).parents[1] / "scripts/common/bop_adapter.py"
)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


def test_camera_and_millimetres_match_known_plane():
    k = np.array([[100.0, 0, 1.0], [0, 200.0, 1.0], [0, 0, 1.0]])
    measured = adapter.unproject_cv(np.full((3, 3), 2.0), np.ones((3, 3), bool), k)
    assert np.allclose(measured[[0, 4, 8]], [[-0.02, -0.01, 2.0], [0, 0, 2.0], [0.02, 0.01, 2.0]])
    mesh, correction = adapter.camera_mesh([[0, 0, 0], [20, 10, 0]], np.eye(3), [0, 0, 2000])
    assert np.allclose(mesh, measured[[4, 8]])
    assert correction == 0
    assert np.allclose(mesh * adapter.CV_TO_PROJECT, [[0, 0, -2.0], [0.02, -0.01, -2.0]])


def test_observation_removes_invalid_depth_and_matches_model_quantization():
    raw = np.full((5, 5), 2000, np.uint16)
    raw[0, 0], raw[1, 1] = 0, 2500
    visible = np.ones((5, 5), bool)
    visible[2, 2] = False
    rgb = np.full((5, 5, 3), 117, np.uint8)
    data = adapter.prepare_observation(rgb, raw, visible, np.eye(3), 1.0, minimum_pixels=4)
    assert data["mask"].sum() == 22
    assert np.all(data["rgb"][~data["mask"]] == 0)
    assert np.all(data["depth_m"][data["mask"]] == 2.0)
    assert data["partial_points_camera_m"].shape == (4, 3)
    assert len(np.unique(data["partial_points_camera_m"], axis=0)) == 4
    assert np.all(data["partial_points_camera_m"][:, 2] == -2.0)
    assert np.all(rgb == 117) and raw[1, 1] == 2500


@pytest.mark.parametrize("scale", [0, -1, np.nan, 1000])
def test_invalid_or_metre_confused_scale_rejected(scale):
    with pytest.raises(ValueError):
        adapter.prepare_observation(
            np.ones((3, 3, 3), np.uint8),
            np.full((3, 3), 2000, np.uint16),
            np.ones((3, 3), bool),
            np.eye(3),
            scale,
            minimum_pixels=4,
        )


def test_quantization_bound_and_rotation_checks():
    raw = np.full((3, 3), 815, np.uint16)
    data = adapter.prepare_observation(
        np.ones((3, 3, 3), np.uint8), raw, np.ones((3, 3), bool), np.eye(3), 1.0, minimum_pixels=4
    )
    assert data["depth_quantization_max_m"] <= 10 / 65535 / 2
    with pytest.raises(ValueError, match="rotation"):
        adapter.camera_mesh([[0, 0, 0]], np.eye(3) * 1.01, [0, 0, 1000])
    with pytest.raises(ValueError, match="intrinsics"):
        adapter.unproject_cv(np.ones((2, 2)), np.ones((2, 2), bool), np.zeros((3, 3)))
