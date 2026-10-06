"""Independent artifact, RGB, split and triangle-ray checks before data admission."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion

from scripts.data.audit_rgb_frames import rgb_frame_qa
from scripts.data.audit_triangle_meshes import check_pair, load_mesh
from scripts.data.corrupt_rendered_depth import camera_matrices


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path)
    p.add_argument("--admission", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    manifest = json.loads((a.input / "complete.json").read_text())
    admission = json.loads(a.admission.read_text())
    assert manifest["admission_sha256"] == sha(a.admission)
    allowed = {r["key"]: r for r in admission["rows"]}
    rows = []
    for target in manifest["rows"]:
        assert (
            target["key"] in allowed
            and target["split_pool"] == allowed[target["key"]]["split_pool"]
        )
        candidate_frames = {r["frame_id"] for r in target["frames"]}
        assert len(candidate_frames) == len(target["frames"])
        checked = []
        scene_checks = []
        for reference in target["arrangements"]:
            folder = a.input / target["asset_id"] / reference["folder"]
            scene = json.loads((folder / "scene.json").read_text())
            meshes = []
            for obj in scene["objects"]:
                assert (
                    obj["key"] in allowed
                    and allowed[obj["key"]]["split_pool"]
                    == target["split_pool"]
                    == obj["split_pool"]
                )
                assert obj["family_id"] == allowed[obj["key"]]["family_id"]
                path = folder / obj["file"]
                assert sha(path) == obj["sha256"]
                meshes.append(load_mesh(path))
            assert len({r["family_id"] for r in scene["objects"]}) == 3
            pairs = [
                check_pair(meshes[i], meshes[j]) for i, j in itertools.combinations(range(3), 2)
            ]
            geometry_ok = (
                all(m.is_volume for m in meshes)
                and all(r["passed"] for r in pairs)
                and max(max(0, -m.vertices[:, 2].min()) for m in meshes) <= 0.002
                and max(scene["max_vertex_movement_m"]) <= 0.001
            )
            assert geometry_ok == scene["geometry_passed"]
            scene_checks.append(
                {"folder": reference["folder"], "geometry_passed": geometry_ok, "pairs": pairs}
            )
            for capture in scene["captures"]:
                assert all(sha(folder / n) == h for n, h in capture["files"].items())
                view = capture["view"]
                rgb = np.asarray(Image.open(folder / f"rgb-{view}.png").convert("RGB"))
                qa = rgb_frame_qa(rgb)
                with np.load(folder / f"rgbd-{view}.npz") as data:
                    depth = data["depth_clean_m"]
                    mask = data["target_mask"].astype(bool)
                with np.load(folder / f"camera-{view}.npz") as raw:
                    k, world_to_project = camera_matrices(raw)
                assert rgb.shape == (*depth.shape, 3) and depth.shape == mask.shape
                project_to_world = np.linalg.inv(world_to_project)
                position = project_to_world[:3, 3]
                position_error = float(
                    np.linalg.norm(position - np.array(capture["camera_position_world"]))
                )
                valid = binary_erosion(mask, iterations=2) & np.isfinite(depth) & (depth > 0)
                v, u = np.nonzero(valid)
                if len(u):
                    count = min(256, len(u))
                    choice = np.floor((np.arange(count) + 0.5) * len(u) / count).astype(int)
                    u, v = u[choice], v[choice]
                    rays = np.column_stack(
                        ((u - k[0, 2]) / k[0, 0], -(v - k[1, 2]) / k[1, 1], -np.ones(len(u)))
                    )
                    directions = rays @ project_to_world[:3, :3].T
                    origins = np.broadcast_to(position, directions.shape).copy()
                    hit, ids, _ = meshes[0].ray.intersects_location(
                        origins, directions, multiple_hits=False
                    )
                    coords = hit @ world_to_project[:3, :3].T + world_to_project[:3, 3]
                    errors = np.abs(-coords[:, 2] - depth[v[ids], u[ids]])
                else:
                    count = 0
                    errors = np.array([])
                hit_fraction = len(errors) / count if count else 0
                p95 = float(np.quantile(errors, 0.95)) if len(errors) else None
                median = float(np.median(errors)) if len(errors) else None
                camera_ok = (
                    position_error <= 1e-5
                    and hit_fraction >= 0.99
                    and p95 is not None
                    and p95 <= 0.00025
                    and median <= 0.0001
                )
                passed = bool(
                    geometry_ok
                    and qa["valid"]
                    and camera_ok
                    and mask.sum() >= 1000
                    and (mask & np.isfinite(depth) & (depth > 0)).sum() >= 1000
                )
                checked.append(
                    {
                        "frame_id": capture["frame_id"],
                        "selected_candidate": capture["frame_id"] in candidate_frames,
                        "passed": passed,
                        "rgb_qa": qa,
                        "camera_position_error_m": position_error,
                        "hit_fraction": hit_fraction,
                        "median_depth_error_m": median,
                        "p95_depth_error_m": p95,
                    }
                )
        selected = [r for r in checked if r["selected_candidate"]]
        assert len(selected) == len(candidate_frames)
        record = {
            "key": target["key"],
            "family_id": target["family_id"],
            "split_pool": target["split_pool"],
            "frames": checked,
            "scenes": scene_checks,
            "selected_count": len(selected),
            "passed_20_frames": len(selected) == 20 and all(r["passed"] for r in selected),
        }
        rows.append(record)
        print("BATCH QA", target["asset_id"], record["passed_20_frames"], flush=True)
    result = {
        "status": "independent_scene_batch_QA_complete",
        "rows": rows,
        "all_targets_passed": all(r["passed_20_frames"] for r in rows),
        "source_manifest_sha256": sha(a.input / "complete.json"),
        "admission_sha256": sha(a.admission),
        "script_sha256": sha(Path(__file__)),
        "camera_thresholds": {
            "position_error_m": 1e-5,
            "triangle_ray_hit_fraction": 0.99,
            "median_depth_error_m": 0.0001,
            "p95_depth_error_m": 0.00025,
        },
        "scope": (
            "Source RGB and clean RGB-D/mesh consistency; does not certify "
            "generated PHOTO images or real sensor fidelity."
        ),
    }
    assert not a.output.exists()
    a.output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
