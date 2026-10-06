"""Analytic adversarial checks, no final-cohort predictions or thresholds fitted."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from scripts.common.paths import script_help
from scripts.eval.surface_metrics import (
    paired_dependence_sensitivity,
    paired_object_summary,
    sample_surface,
    score_cloud,
    select_points,
    validate_cloud,
    visibility,
)

script_help(__doc__, __name__)


def main():
    depth = np.ones((11, 11))
    k = np.array([10.0, 10.0, 5.0, 5.0])
    points = np.array([[0, 0, -1], [0, 0, -1.1], [0, 0, -0.9], [1, 0, -1], [0, 0, 1]], float)
    flags = visibility(points, depth, k)
    assert flags["visible"].tolist() == [True, False, False, False, False]
    assert flags["occluded"].tolist() == [False, True, False, False, False]
    assert flags["known_free"].tolist() == [False, False, True, False, False]
    assert np.all(flags["visible"].astype(int) + flags["occluded"] + flags["uncertain"] == 1)
    # An external occluder counts even outside the target's own input mask.
    external = depth.copy()
    external[4:7, 4:7] = 0.5
    assert visibility(points[:1], external, k)["occluded"][0]
    # A depth discontinuity, missing pixel or NaN cannot certify visibility.
    for bad in [0.0, np.nan, 0.5]:
        invalid = depth.copy()
        invalid[4, 4] = bad
        assert not visibility(points[:1], invalid, k)["supported"][0]
    # Project coordinates y up must address the corresponding upper image row.
    upper = depth.copy()
    upper[1:4, 4:7] = 0.5
    assert visibility(np.array([[0, 0.3, -1]]), upper, k)["occluded"][0]
    # Full-frame edge and huge finite predictions are uncertain, never clamped.
    assert not visibility(np.array([[1e100, 0, -1]]), depth, k)["supported"][0]
    for invalid in [np.array([[np.nan, 0, 0]]), np.empty((3, 0))]:
        try:
            validate_cloud(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid prediction silently accepted")
    truth = np.array([[0, 0, -1], [0.1, 0, -1]], float)
    prediction = truth + np.array([[0, 0, 0.003], [0, 0, 0.006]])
    masks = {
        "visible": np.ones(2, bool),
        "occluded": np.zeros(2, bool),
        "uncertain": np.zeros(2, bool),
    }
    result = score_cloud(prediction, truth, cKDTree(truth), masks, truth, depth, k)
    assert result["surface"]["0.005"] == {"precision": 0.5, "recall": 0.5, "fscore": 0.5}
    assert np.isclose(result["chamfer_mean_m_success_only"], 0.0045)
    empty = score_cloud(np.empty((0, 3)), truth, cKDTree(truth), masks, np.empty((0, 3)), depth, k)
    assert empty["surface"]["0.005"]["fscore"] == 0 and empty["chamfer_capped_100mm_m"] == 0.1
    assert empty["chamfer_mean_m_success_only"] is None and empty["known_free_rate_all"] is None
    assert empty["gt_subsets"]["occluded"]["recall5"] is None
    distant = score_cloud(truth + 100, truth, cKDTree(truth), masks, truth, depth, k)
    assert distant["chamfer_capped_100mm_m"] == 0.1
    assert len(select_points(truth, 512, sample_id="tiny")) == 2
    cloud = np.arange(3000).reshape(1000, 3)
    assert np.array_equal(
        select_points(cloud, 512, sample_id="same"), select_points(cloud, 512, sample_id="same")
    )
    assert len(np.unique(select_points(cloud, 512, sample_id="same"), axis=0)) == 512
    # Unequal view counts must not give one object extra aggregate weight.
    summary = paired_object_summary(np.array([1.0, -1.0, -1.0]), np.array([1, 2, 2]))
    assert summary["difference"] == 0.0
    constant = paired_object_summary(np.full(12, 0.03), np.repeat([1, 2, 3], 4))
    assert np.allclose(constant["object_bootstrap95"], [0.03, 0.03])
    sensitivity = paired_dependence_sensitivity(
        np.full((12, 3), 0.03), np.repeat([1, 2, 3], 4), np.tile([1, 2, 3, 4], 3)
    )
    assert np.allclose(sensitivity["object_and_seed_bootstrap95"], [0.03, 0.03])
    assert np.allclose(sensitivity["scene_cluster_bootstrap95"], [0.03, 0.03])
    # Preserve the historical area sampler exactly, without importing Torch.
    original = Path(__file__).resolve().parents[2] / "src/masters_rgbd/b1/projected_mesh.py"
    node = next(
        n
        for n in ast.parse(original.read_text(encoding="utf-8-sig")).body
        if isinstance(n, ast.FunctionDef) and n.name == "_sample_mesh_surface"
    )
    ns = {"np": np, "RapidMeshError": ValueError}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(original), "exec"), ns)
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 2]], float)
    faces = np.array([[0, 1, 2], [0, 1, 3]])
    assert np.array_equal(
        sample_surface(vertices, faces, count=1000, seed=7),
        ns["_sample_mesh_surface"](vertices, faces, count=1000, seed=7),
    )
    print(
        json.dumps(
            {
                "analytic_checks": "passed",
                "historical_area_sampler_exact_parity": True,
                "heldout_predictions_used": False,
            }
        )
    )


if __name__ == "__main__":
    main()
