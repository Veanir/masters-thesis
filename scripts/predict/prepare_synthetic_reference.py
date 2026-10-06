"""Build and audit all720 local synthetic surface references, without predictions."""

from __future__ import annotations

import concurrent.futures
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from scripts.common.paths import ROOT, digest, read, save, script_help
from scripts.eval.surface_metrics import sample_surface, stable_seed, visibility

script_help(__doc__, __name__)


PREP = ROOT / "runs/thesis-supplement-20260910/inference-preparation-v1"
OUT = ROOT / "runs/thesis-supplement-20260910/holdout-reference-v1"


def one(row, rules):
    sid = row["sample_id"]
    source = ROOT / row["observation"]["path"]
    assert digest(source) == row["observation"]["sha256"]
    with np.load(source, allow_pickle=False) as z:
        k = z["K"]
        intrinsics = k[[0, 1, 0, 1], [0, 1, 2, 2]].astype(np.float64)
        vertices = z["mesh_vertices_camera_cv_m"] * np.array([1, -1, -1])
        faces = z["mesh_faces"]
        truth = sample_surface(
            vertices,
            faces,
            count=rules["reference_points"],
            seed=stable_seed(rules["holdout_truth_seed_purpose"], sid),
        )
        mask = z["input_mask"].astype(bool)
        v, u = np.nonzero(mask)
        decoded = z["input_depth_uint16"].astype(np.float64) * (10.0 / 65535)
        d = decoded[v, u]
        fx, fy, cx, cy = intrinsics
        dense = np.column_stack(((u - cx) * d / fx, -(v - cy) * d / fy, -d))
        full_depth = z["depth_sensor_m"].astype(np.float64)
        assert full_depth.shape == decoded.shape == (480, 640)
        clean = z["depth_clean_m"]
        target = z["source_target_mask"].astype(bool)
        valid = target & np.isfinite(clean) & (clean > 0)
        vv, uu = np.nonzero(valid)
        assert len(vv)
        ids = np.floor((np.arange(min(4096, len(vv))) + 0.5) * len(vv) / min(4096, len(vv))).astype(
            int
        )
        vv, uu = vv[ids], uu[ids]
        dd = clean[vv, uu].astype(np.float64)
        clean_points = np.column_stack(((uu - cx) * dd / fx, -(vv - cy) * dd / fy, -dd))
        distance = cKDTree(truth).query(clean_points, workers=1)[0]
        finite = (
            np.isfinite(vertices).all() and np.isfinite(truth).all() and np.isfinite(dense).all()
        )
        assert finite and len(dense) == row["input_pixels"]
        flags = visibility(truth, full_depth, intrinsics, tolerance_m=0.005, max_patch_span_m=0.010)
        uncovered = cKDTree(dense).query(truth, workers=1)[0] > 0.005
        assert np.all(flags["visible"].astype(int) + flags["occluded"] + flags["uncertain"] == 1)
        # The inverse projection must recover original source pixels/depth.
        zz = -dense[:, 2]
        assert np.max(np.abs(cx + fx * dense[:, 0] / zz - u)) < 1e-8
        assert np.max(np.abs(cy - fy * dense[:, 1] / zz - v)) < 1e-8
        assert np.array_equal(zz, d)
    dest = OUT / (sid + ".npz")
    assert not dest.exists()
    np.savez_compressed(
        dest,
        truth_points_camera_m=truth,
        input_dense_points_camera_m=dense,
        full_scene_depth_m=full_depth,
        intrinsics=intrinsics,
        **{
            key: flags[key]
            for key in ("visible", "occluded", "uncertain", "known_free", "supported")
        },
        uncovered_by_dense_input=uncovered,
    )
    p95 = float(np.quantile(distance, 0.95))
    return {
        **row,
        "cache_sha256": digest(dest),
        "bytes": dest.stat().st_size,
        "alignment_clean_target_sample_count": len(clean_points),
        "alignment_clean_target_median_m": float(np.median(distance)),
        "alignment_clean_target_p95_m": p95,
        "alignment_clean_target_max_m": float(distance.max()),
        "alignment_p95_within_5mm": p95 <= 0.005,
        "camera_projection_roundtrip_passed": True,
        "input_median_depth_m": float(np.median(d)),
        "reference_subsets": {
            key: int(flags[key].sum()) for key in ("visible", "occluded", "uncertain")
        },
    }


def main():
    assert not OUT.exists()
    protocol_path = PREP / "protocol.json"
    protocol = read(protocol_path)
    complete = read(PREP / "complete.json")
    assert digest(protocol_path) == complete["protocol"]["sha256"]
    reference = protocol["holdout_reference_index"]
    index_path = ROOT / reference["path"]
    assert digest(index_path) == reference["sha256"]
    index = read(index_path)
    assert index["GT_must_not_be_uploaded_for_inference"] and len(index["rows"]) == 720
    rules = protocol["metric_rules"]
    assert digest(ROOT / rules["core"]["path"]) == rules["core"]["sha256"]
    OUT.mkdir(parents=True)
    tick = time.monotonic()
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(one, row, rules) for row in index["rows"]]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
            if len(results) % 60 == 0:
                save(
                    OUT / "progress.json",
                    {"completed": len(results), "total": 720, "seconds": time.monotonic() - tick},
                )
                print(
                    "HOLDOUT REFERENCE",
                    len(results),
                    "seconds",
                    round(time.monotonic() - tick, 1),
                    flush=True,
                )
    results.sort(key=lambda r: r["sample_id"])
    assert [r["sample_id"] for r in results] == [r["sample_id"] for r in index["rows"]]
    record = {
        "status": "all720_local_synthetic_references_built_before_predictions",
        "protocol_sha256": digest(protocol_path),
        "source_reference_index_sha256": digest(index_path),
        "script_sha256": digest(Path(__file__)),
        "core_sha256": rules["core"]["sha256"],
        "GT_must_not_be_uploaded_for_inference": True,
        "rows": results,
        "seconds": time.monotonic() - tick,
        "audit": {
            "observations": 720,
            "geometries": len({r["key"] for r in results}),
            "projection_roundtrip_all_passed": all(
                r["camera_projection_roundtrip_passed"] for r in results
            ),
            "alignment_p95_within_5mm_count": sum(r["alignment_p95_within_5mm"] for r in results),
            "alignment_p95_max_m": max(r["alignment_clean_target_p95_m"] for r in results),
            "mesh_bbox_diagonal_range_m": [
                min(r["mesh_bbox_diagonal_m"] for r in results),
                max(r["mesh_bbox_diagonal_m"] for r in results),
            ],
        },
        "scope": (
            "All reserved identities retained. Alignment compares clean "
            "visible target samples to a finite mesh-surface sample, not "
            "exact point-to-triangle distance. No prediction scores or "
            "outcome-based filtering."
        ),
    }
    save(OUT / "complete.json", record)
    print(
        json.dumps(
            {"status": record["status"], "audit": record["audit"], "seconds": record["seconds"]}
        )
    )


if __name__ == "__main__":
    main()
