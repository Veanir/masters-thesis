"""Prepare immutable local GT/reference caches under the frozen evaluation rules.

No predictions or method scores are computed. This directory must never be
included in an inference input archive.
"""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from scripts.common.paths import script_help
from scripts.data.select_homebrewed_observations import DATA, ROOT
from scripts.eval.surface_metrics import sample_surface, stable_seed, visibility

script_help(__doc__, __name__)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    base = ROOT / "runs/research-evolution-evaluation-20260907"
    protocol_path = base / "protocol-v1.json"
    protocol = json.loads(protocol_path.read_text())
    for ref in protocol["references"].values():
        assert sha(ROOT / ref["path"]) == ref["sha256"]
    reservation = json.loads((ROOT / protocol["references"]["reservation"]["path"]).read_text())
    manifest_path = ROOT / protocol["references"]["GT_manifest"]["path"]
    manifest = json.loads(manifest_path.read_text())
    observations = manifest_path.parent
    out = base / "HB-reference-cache-v1"
    out.mkdir(exist_ok=False)
    rows = []
    checked = {}
    tick = time.monotonic()
    for reference, row in zip(reservation["observations"], manifest["rows"], strict=True):
        sid = row["sample_id"]
        source = observations / (sid + ".npz")
        assert sha(source) == row["output_sha256"]
        assert all(reference[k] == row[k] for k in ["scene_id", "image_id", "gt_id", "obj_id"])
        files = {key: DATA / reference["files"][key]["path"] for key in ["depth", "scene_camera"]}
        for key, path in files.items():
            if str(path) not in checked:
                checked[str(path)] = sha(path)
            assert checked[str(path)] == reference["files"][key]["sha256"]
        camera = json.loads(files["scene_camera"].read_text())[str(row["image_id"])]
        full_depth = (
            np.asarray(Image.open(files["depth"]), dtype=np.float64)
            * float(camera["depth_scale"])
            * 0.001
        )
        with np.load(source) as z:
            intrinsics = z["intrinsics"]
            k = np.asarray(camera["cam_K"], dtype=np.float32).reshape(3, 3)
            assert np.array_equal(intrinsics, k[[0, 1, 0, 1], [0, 1, 2, 2]])
            truth = sample_surface(
                z["mesh_vertices_camera_m"],
                z["mesh_faces"],
                count=protocol["reference_points"],
                seed=stable_seed(protocol["truth_seed_purpose"], sid),
            )
            flags = visibility(
                truth,
                full_depth,
                intrinsics,
                tolerance_m=protocol["visibility_tolerance_m"],
                max_patch_span_m=protocol["visibility_patch_span_m"],
            )
            v, u = np.nonzero(z["mask"])
            depth = z["depth_m"][v, u]
            fx, fy, cx, cy = intrinsics
            dense = np.column_stack(((u - cx) * depth / fx, -(v - cy) * depth / fy, -depth))
        assert len(dense) == row["input_pixels"]
        uncovered = (
            cKDTree(dense).query(truth, workers=2)[0] > 0.005
            if len(dense)
            else np.ones(len(truth), bool)
        )
        assert np.all(flags["visible"].astype(int) + flags["occluded"] + flags["uncertain"] == 1)
        dest = out / (sid + ".npz")
        np.savez_compressed(
            dest,
            truth_points_camera_m=truth,
            input_dense_points_camera_m=dense,
            full_scene_depth_m=full_depth,
            intrinsics=intrinsics,
            **{
                key: flags[key]
                for key in ["visible", "occluded", "uncertain", "known_free", "supported"]
            },
            uncovered_by_dense_input=uncovered,
        )
        rows.append(
            {
                **row,
                "cache_sha256": sha(dest),
                "source_observation_sha256": sha(source),
                "full_scene_depth_source_sha256": reference["files"]["depth"]["sha256"],
                "bytes": dest.stat().st_size,
            }
        )
        if len(rows) % 33 == 0:
            print(
                "HB GT CACHE", len(rows), "elapsed", round(time.monotonic() - tick, 1), flush=True
            )
    assert [r["sample_id"] for r in rows] == protocol["sample_ids"]
    result = {
        "status": "local_HB_evaluation_cache_complete_no_prediction_scores",
        "protocol_sha256": sha(protocol_path),
        "script_sha256": sha(Path(__file__)),
        "core_sha256": sha(Path(__file__).parents[2].joinpath("scripts/eval/surface_metrics.py")),
        "rows": rows,
        "seconds": time.monotonic() - tick,
        "GT_must_not_be_uploaded_for_inference": True,
        "scope": (
            "Frozen geometry, full raw scene depth, visibility and INPUT "
            "references only. Never use HB quantities for model or "
            "augmentation selection."
        ),
    }
    (out / "complete.json").write_text(json.dumps(result, indent=2))
    print("Prepared all198 local evaluator records without inference", flush=True)


if __name__ == "__main__":
    main()
