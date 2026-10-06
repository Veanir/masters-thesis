"""Audit the reserved HB cohort without predictions or membership changes."""

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from scripts.common.bop_adapter import camera_mesh, prepare_observation
from scripts.common.paths import script_help
from scripts.data.select_homebrewed_observations import DATA, OUT, sha

script_help(__doc__, __name__)


def main():
    reservation = OUT / "cohort-reservation-v1.json"
    cohort = json.loads(reservation.read_text())
    output = OUT / "calibration-v1"
    assert not output.exists()
    output.mkdir()
    contract = {
        "reservation_sha256": sha(reservation),
        "script_sha256": sha(Path(__file__)),
        "adapter_sha256": sha(Path(__file__).parents[2].joinpath("scripts/common/bop_adapter.py")),
        "minimum_valid_pixels": 512,
        "registration_median_limit_m": 0.002,
        "registration_p95_limit_m": 0.015,
        "threshold_provenance": (
            "Existing exposed YCB/HOPE diagnostics; no HB-based threshold tuning."
        ),
        "distance_sample": (
            "Up to2048 deterministic row-major midpoint strata of all "
            "filtered target sensor points; exact triangle distance."
        ),
        "membership_policy": (
            "Retain every198 reserved observation including failures. This "
            "audit does not select a new cohort."
        ),
        "model_predictions_computed": False,
        "versions": {"numpy": np.__version__, "trimesh": trimesh.__version__},
    }
    (output / "contract.json").write_text(json.dumps(contract, indent=2))
    rows = []
    meshes = {}
    verified = {}
    for ordinal, original in enumerate(cohort["observations"]):
        tick = time.monotonic()
        r = {k: original[k] for k in ["scene_id", "image_id", "gt_id", "obj_id", "visibility_bin"]}
        refs = {k: DATA / v["path"] for k, v in original["files"].items()}
        for key, path in refs.items():
            if str(path) not in verified:
                verified[str(path)] = sha(path)
            assert verified[str(path)] == original["files"][key]["sha256"]
        camera = json.loads(refs["scene_camera"].read_text())[str(r["image_id"])]
        pose = json.loads(refs["scene_gt"].read_text())[str(r["image_id"])][r["gt_id"]]
        assert pose["obj_id"] == r["obj_id"]
        rgb = np.asarray(Image.open(refs["rgb"]).convert("RGB"))
        depth = np.asarray(Image.open(refs["depth"]))
        mask = np.asarray(Image.open(refs["mask_visib"])) > 0
        k = np.asarray(camera["cam_K"]).reshape(3, 3)
        r.update(
            rgb_shape=list(rgb.shape),
            depth_shape=list(depth.shape),
            mask_pixels=int(mask.sum()),
            positive_target_pixels=int((mask & (depth > 0)).sum()),
        )
        assert depth.shape == (480, 640), "Unexpected resolution; no silent resize"
        try:
            data = prepare_observation(rgb, depth, mask, k, camera["depth_scale"], minimum_pixels=1)
        except ValueError as exc:
            r.update(status="input_failure", reason=str(exc))
        else:
            obj = r["obj_id"]
            if obj not in meshes:
                mesh = trimesh.load(refs["mesh"], process=False, force="mesh")
                mesh.vertices *= 0.001
                meshes[obj] = mesh
            mesh = meshes[obj]
            posed, correction = camera_mesh(
                mesh.vertices * 1000, pose["cam_R_m2c"], pose["cam_t_m2c"]
            )
            raw = np.asarray(pose["cam_R_m2c"]).reshape(3, 3)
            u, _, vt = np.linalg.svd(raw)
            rotation = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
            points = data["sensor_points_cv"]
            count = min(2048, len(points))
            idx = np.floor((np.arange(count) + 0.5) * len(points) / count).astype(int)
            object_points = (points[idx] - np.asarray(pose["cam_t_m2c"]) * 0.001) @ rotation
            distances = trimesh.proximity.closest_point(mesh, object_points)[1]
            median, p95 = np.quantile(distances, [0.5, 0.95])
            pixel_ok = len(points) >= 512
            registration = median <= 0.002 and p95 <= 0.015
            v, x = np.nonzero(data["mask"])
            quantized = data["points_cv"]
            uv = quantized[:, :2] / quantized[:, 2, None] * [k[0, 0], k[1, 1]] + [k[0, 2], k[1, 2]]
            ray_error = float(np.max(abs(uv - np.column_stack((x, v)))))
            assert ray_error < 1e-9 and data["depth_quantization_max_m"] <= 10 / 65535 / 2 + 1e-12
            r.update(
                status="passed"
                if pixel_ok and registration
                else "input_failure"
                if not pixel_ok
                else "registration_flag",
                filtered_pixels=len(points),
                sampled_points=count,
                alignment_median_m=float(median),
                alignment_p95_m=float(p95),
                enough_input_pixels=pixel_ok,
                registration_within_reference_limits=bool(registration),
                pose_so3_correction=correction,
                pixel_roundtrip_max_error=ray_error,
                depth_quantization_max_m=data["depth_quantization_max_m"],
                median_depth_m=data["median_depth_m"],
                posed_mesh_z_range_m=[float(posed[:, 2].min()), float(posed[:, 2].max())],
            )
        r["seconds"] = time.monotonic() - tick
        rows.append(r)
        (output / "progress.json").write_text(
            json.dumps({"completed": len(rows), "counts": dict(Counter(x["status"] for x in rows))})
        )
        if (ordinal + 1) % 6 == 0:
            print(
                "HB CALIBRATION", ordinal + 1, dict(Counter(x["status"] for x in rows)), flush=True
            )
    assert len(rows) == 198
    result = {
        "status": "reserved_cohort_coordinate_audit_complete",
        "contract": contract,
        "rows": rows,
        "counts": dict(Counter(r["status"] for r in rows)),
        "membership_unchanged": True,
        "model_predictions_computed": False,
        "interpretation": (
            "Registration flags describe source mesh/sensor agreement, not "
            "model quality. Resolve adapter errors without choosing "
            "better-performing observations."
        ),
    }
    (output / "complete.json").write_text(json.dumps(result, indent=2))
    print(result["counts"], flush=True)


if __name__ == "__main__":
    main()
