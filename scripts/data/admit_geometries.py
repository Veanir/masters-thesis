"""Freeze geometry admission using source physics only, never model scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.common.paths import script_help
from scripts.data.split_geometry_families import OUT, sha

script_help(__doc__, __name__)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--physics", type=Path, required=True)
    p.add_argument("--train-count", type=int, default=320)
    p.add_argument("--validation-count", type=int, default=40)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    source = OUT / "candidate-split-v1.json"
    pools = json.loads(source.read_text())
    physics = json.loads(a.physics.read_text())
    assert physics["status"] == "candidate_physics_admission_complete" and physics[
        "manifest_sha256"
    ] == sha(source)
    candidates = {r["key"]: r for r in pools["rows"]}
    assert len(candidates) == 396
    assert len(physics["rows"]) == 396 and {r["key"] for r in physics["rows"]} == set(candidates)
    passed = []
    failed = []
    for result in physics["rows"]:
        row = candidates[result["key"]]
        assert all(
            row[k] == result[k]
            for k in ["asset_id", "family_id", "split_pool", "admission_priority"]
        )
        assert len(result["orientations"]) == 2
        checks = []
        for orientation in result["orientations"]:
            transform = np.asarray(orientation["world_T_object"])
            assert transform.shape == (4, 4) and np.isfinite(transform).all()
            assert np.allclose(transform[3], [0, 0, 0, 1], atol=1e-6) and np.allclose(
                transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-5
            )
            okay = (
                orientation["max_vertex_movement_m"] <= 0.001
                and orientation["floor_penetration_m"] <= 0.002
            )
            assert okay == orientation["passed"]
            checks.append(okay)
        assert all(checks) == result["passed"]
        (passed if result["passed"] else failed).append(
            {**row, "physics_orientations": result["orientations"]}
        )
    requested = {"train": a.train_count, "synthetic_validation": a.validation_count}
    assert 300 <= a.train_count <= 320 and 30 <= a.validation_count <= 40
    counts = {split: sum(r["split_pool"] == split for r in passed) for split in requested}
    assert all(counts[s] >= n for s, n in requested.items()), (
        "Source QA shortfall; record revised size or expand independent pools before generation",
        counts,
        requested,
    )
    chosen = []
    reserve = []
    for split, count in requested.items():
        subset = sorted(
            (r for r in passed if r["split_pool"] == split), key=lambda r: r["admission_priority"]
        )
        chosen.extend(subset[:count])
        reserve.extend(subset[count:])
    assert len({r["family_id"] for r in chosen}) == len(chosen)
    assert not any(
        r["family_used_in_renderer_pilot"] and r["split_pool"] != "train" for r in chosen
    )
    report = {
        "status": "geometry_admission_frozen_before_full_scene_generation",
        "rows": chosen,
        "passed_reserve": reserve,
        "rejected": failed,
        "counts": requested,
        "available_after_physics": counts,
        "candidate_manifest_sha256": sha(source),
        "physics_results_sha256": sha(a.physics),
        "script_sha256": sha(Path(__file__)),
        "occluder_policy": (
            "Only a member of rows with the same split_pool as target. Never "
            "a reserve or rejected geometry."
        ),
        "scope": (
            "Source geometry admission, not final frame admission. Scene "
            "penetration, RGB-D calibration, sensor perturbation, visibility "
            "and PHOTO QA remain separate."
        ),
        "model_quality_used": False,
    }
    assert not a.output.exists()
    a.output.write_text(json.dumps(report, indent=2))
    print("Admitted", requested, "failed", len(failed), "reserve", len(reserve), flush=True)


if __name__ == "__main__":
    main()
