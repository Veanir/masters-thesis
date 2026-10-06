"""Observation-only RaySt3R export of the previously exposed YCB-V cohort.

Geometry is exported separately for calibration and subsequent scoring. This
script does not perform model inference or define a new final test cohort.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image

MANIFEST_SHA256 = "20ba21f2e3c4b0f75a5447ad5408aed722a4a9fbd2d4d475a48f8b5c3065e8cd"
CV_TO_PROJECT = np.array([1.0, -1.0, -1.0])


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def unproject_cv(depth_m, mask, k):
    """Backproject integer pixel centres, with BOP/OpenCV +z forward."""
    depth_m, mask, k = np.asarray(depth_m), np.asarray(mask), np.asarray(k)
    if depth_m.ndim != 2 or mask.shape != depth_m.shape or mask.dtype != bool:
        raise ValueError("Depth and boolean mask must have equal HxW shape")
    if k.shape != (3, 3) or not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0:
        raise ValueError("Invalid camera intrinsics")
    if not np.array_equal(k[2], [0, 0, 1]) or k[0, 1] != 0 or k[1, 0] != 0:
        raise ValueError("Only canonical, zero-skew pinhole intrinsics are supported")
    if not np.isfinite(depth_m).all() or np.any(depth_m < 0) or np.any(depth_m[mask] <= 0):
        raise ValueError("Depth must be finite, nonnegative and positive under the mask")
    v, u = np.nonzero(mask)
    z = depth_m[v, u].astype(np.float64)
    return np.column_stack(((u - k[0, 2]) * z / k[0, 0], (v - k[1, 2]) * z / k[1, 1], z))


def prepare_observation(rgb, raw_depth, visible, k, depth_scale_mm, *, minimum_pixels=512):
    """Use sensor observations only; there is no pose or mesh argument."""
    rgb, raw_depth, visible = np.asarray(rgb), np.asarray(raw_depth), np.asarray(visible)
    if rgb.dtype != np.uint8 or rgb.shape != (*raw_depth.shape, 3):
        raise ValueError("Expected uint8 RGB matching depth dimensions")
    if not np.issubdtype(raw_depth.dtype, np.integer) or np.any(raw_depth < 0):
        raise ValueError("Expected nonnegative integer sensor depth")
    if visible.shape != raw_depth.shape or visible.dtype != bool:
        raise ValueError("Expected matching boolean visible mask")
    if not np.isfinite(depth_scale_mm) or depth_scale_mm <= 0:
        raise ValueError("Invalid BOP depth scale")
    sensor = raw_depth.astype(np.float64) * float(depth_scale_mm) * 0.001
    positive = visible & (raw_depth > 0)
    if not positive.any():
        raise ValueError("No valid sensor measurements")
    median = float(np.median(sensor[positive]))
    mask = positive & (np.abs(sensor - median) <= 0.25)
    if mask.sum() < minimum_pixels:
        raise ValueError("Too few valid sensor pixels")
    if np.any(sensor[mask] <= 0.1) or np.any(sensor[mask] > 10):
        raise ValueError("Target depth outside RaySt3R's (0.1, 10] metre range")
    sensor = np.where(mask, sensor, 0.0)
    encoded = np.rint(sensor * 65535 / 10).astype(np.uint16)
    decoded = encoded.astype(np.float64) * (10 / 65535)
    # GenericLoaderSmall removes encoded depths <= the encoded 0.1 m threshold.
    if np.any(encoded[mask] <= int(0.1 / 10 * 65535)):
        raise ValueError("RaySt3R would remove a retained target pixel")
    points_cv = unproject_cv(decoded, mask, k)
    sensor_cv = unproject_cv(sensor, mask, k)
    indices = np.floor((np.arange(minimum_pixels) + 0.5) * len(points_cv) / minimum_pixels).astype(
        int
    )
    partial = points_cv[indices] * CV_TO_PROJECT
    return {
        "rgb": np.where(mask[..., None], rgb, 0).astype(np.uint8),
        "depth_uint16": encoded,
        "depth_m": decoded,
        "depth_sensor_m": sensor,
        "mask": mask,
        "points_cv": points_cv,
        "sensor_points_cv": sensor_cv,
        "partial_points_camera_m": partial,
        "median_depth_m": median,
        "depth_quantization_max_m": float(np.abs(decoded - sensor).max()),
        "backprojection_quantization_max_m": float(
            np.linalg.norm(points_cv - sensor_cv, axis=1).max()
        ),
    }


def camera_mesh(vertices_mm, rotation, translation_mm):
    """Known GT object pose; call only in the scoring/export path."""
    vertices_mm = np.asarray(vertices_mm, np.float64)
    raw = np.asarray(rotation, np.float64).reshape(3, 3)
    t = np.asarray(translation_mm, np.float64).reshape(3)
    if not all(np.isfinite(a).all() for a in (vertices_mm, raw, t)):
        raise ValueError("Nonfinite mesh or pose")
    u, _, vt = np.linalg.svd(raw)
    r = u @ np.diag([1.0, 1.0, np.linalg.det(u @ vt)]) @ vt
    correction = float(np.abs(r - raw).max())
    if correction > 2e-6:
        raise ValueError("BOP rotation requires excessive SO(3) correction")
    vertices_cv = (vertices_mm @ r.T + t) * 0.001
    if np.any(vertices_cv[:, 2] <= 0):
        raise ValueError("GT mesh is not entirely in front of camera")
    return vertices_cv, correction


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    args = parser.parse_args()
    import open3d as o3d
    import torch
    import trimesh

    if digest(args.manifest) != MANIFEST_SHA256:
        raise ValueError("Expected frozen, previously exposed relative42 cohort")
    manifest = json.loads(args.manifest.read_text())
    cohort = manifest["observations"]
    if len(cohort) != 42 or len({r["obj_id"] for r in cohort}) != 21:
        raise ValueError("Unexpected cohort topology")
    root = args.dataset.resolve()
    checked = {}
    for row in cohort:
        for name, relative in row["inputs"].items():
            if name.endswith("_sha256"):
                continue
            path = (root / relative).resolve()
            path.relative_to(root)
            expected = row["inputs"][name + "_sha256"]
            if relative not in checked:
                checked[relative] = digest(path)
            if checked[relative] != expected:
                raise ValueError(f"Changed input: {relative}")
    input_root, obs_root = args.inputs.resolve(), args.observations.resolve()
    if input_root == obs_root or input_root in obs_root.parents or obs_root in input_root.parents:
        raise ValueError("Model inputs and evaluator geometry must be separate directories")
    if input_root.exists() or obs_root.exists():
        raise ValueError("Use fresh export directories")
    input_root.mkdir(parents=True)
    obs_root.mkdir(parents=True)
    contract = {
        "purpose": "previously exposed YCB-V diagnostic, not fresh final evidence",
        "manifest_sha256": digest(args.manifest),
        "script_sha256": digest(__file__),
        "input_resolution": [480, 640],
        "crop_or_resize": False,
        "sensor_filter": "visible mask, positive depth, median depth +/-0.25m",
        "model_depth": "round(sensor_metres/10*65535), uint16 PNG; decode *10/65535",
        "model_camera": "OpenCV; identity cam2world; actual observed intrinsics",
        "scoring_camera": "project camera [x,-y,-z] from OpenCV; metres",
        "input512": "row-major midpoint strata from exact quantized model observation",
        "alignment": "all filtered sensor pixels to posed mesh triangles via Open3D",
        "alignment_tolerance_vs_frozen_m": 1e-6,
        "packages": {
            "numpy": np.__version__,
            "open3d": o3d.__version__,
            "trimesh": trimesh.__version__,
            "torch": torch.__version__,
        },
    }
    save_json(
        obs_root / "binding.json",
        {
            "started_at": datetime.now(UTC).isoformat(),
            "contract": contract,
            "source_sha256": checked,
        },
    )
    inputs, observations = [], []
    try:
        for ordinal, row in enumerate(cohort):
            start = time.perf_counter()
            paths = {k: root / v for k, v in row["inputs"].items() if not k.endswith("_sha256")}
            camera = json.loads(paths["scene_camera"].read_text())[str(row["image_id"])]
            pose = json.loads(paths["scene_gt"].read_text())[str(row["image_id"])][row["gt_id"]]
            if pose["obj_id"] != row["obj_id"]:
                raise ValueError("GT identity differs from frozen manifest")
            k = np.asarray(camera["cam_K"], np.float64).reshape(3, 3)
            rgb = np.asarray(Image.open(paths["rgb"]).convert("RGB"))
            raw_depth = np.asarray(Image.open(paths["depth"]))
            visible = np.asarray(Image.open(paths["mask_visib"])) > 0
            if raw_depth.shape != (480, 640):
                raise ValueError("Unexpected native resolution; no silent resize")
            data = prepare_observation(rgb, raw_depth, visible, k, camera["depth_scale"])
            if (
                int(data["mask"].sum()) != row["filtered_pixels"]
                or int(visible.sum()) != row["visible_pixels"]
                or abs(data["median_depth_m"] - row["median_depth_m"]) > 1e-7
            ):
                raise ValueError("Observation filtering differs from frozen cohort")
            mesh = trimesh.load(paths["mesh"], process=False, force="mesh")
            vertices_cv, correction = camera_mesh(
                mesh.vertices, pose["cam_R_m2c"], pose["cam_t_m2c"]
            )
            if abs(correction - row["pose_so3_max_correction"]) > 1e-12:
                raise ValueError("Pose correction differs from frozen cohort")
            scene = o3d.t.geometry.RaycastingScene(nthreads=2)
            scene.add_triangles(
                o3d.core.Tensor(vertices_cv.astype(np.float32)),
                o3d.core.Tensor(np.asarray(mesh.faces, np.uint32)),
            )
            alignment = scene.compute_distance(
                o3d.core.Tensor(data["sensor_points_cv"].astype(np.float32)), nthreads=2
            ).numpy()
            median, p95 = np.quantile(alignment, [0.5, 0.95]).tolist()
            if (
                abs(median - row["alignment_median_m"]) > 1e-6
                or abs(p95 - row["alignment_p95_m"]) > 1e-6
                or median > 0.002 + 1e-6
                or p95 > 0.015 + 1e-6
                or len(alignment) != row["alignment_point_count"]
            ):
                raise ValueError(f"Calibration parity failed: {row['target_id']} {median} {p95}")
            if data["depth_quantization_max_m"] > 10 / 65535 / 2 + 1e-12:
                raise ValueError("Excessive depth encoding error")
            name = row["target_id"]
            folder = input_root / name
            folder.mkdir()
            Image.fromarray(data["rgb"]).save(folder / "rgb.png")
            Image.fromarray(data["depth_uint16"]).save(folder / "depth.png")
            Image.fromarray(data["mask"].astype(np.uint8) * 255).save(folder / "mask.png")
            torch.save(torch.from_numpy(k.astype(np.float32)), folder / "intrinsics.pt")
            torch.save(torch.eye(4), folder / "cam2world.pt")
            np.save(
                folder / "input512_camera_m.npy", data["partial_points_camera_m"].astype(np.float32)
            )
            if not np.array_equal(
                np.asarray(Image.open(folder / "depth.png")), data["depth_uint16"]
            ):
                raise ValueError("Depth PNG roundtrip changed values")
            identity = {
                "sample_id": name,
                "obj_id": row["obj_id"],
                "scene_id": row["scene_id"],
                "image_id": row["image_id"],
                "gt_id": row["gt_id"],
                "ordinal": ordinal,
                "stratum": row["stratum"],
                "evaluation_role": "exposed_real_diagnostic",
            }
            inputs.append(
                {
                    **identity,
                    "gt_exported": False,
                    "rayst3r_output_to_project_camera": CV_TO_PROJECT.tolist(),
                    "export_sha256": {p.name: digest(p) for p in sorted(folder.iterdir())},
                }
            )
            obs_path = obs_root / (name + ".npz")
            np.savez_compressed(
                obs_path,
                rgb=data["rgb"],
                depth_m=data["depth_m"],
                depth_sensor_m=data["depth_sensor_m"],
                mask=data["mask"],
                intrinsics=k[[0, 1, 0, 1], [0, 1, 2, 2]],
                partial_points_camera_m=data["partial_points_camera_m"],
                mesh_vertices_camera_m=vertices_cv * CV_TO_PROJECT,
                mesh_faces=np.asarray(mesh.faces, np.int64),
            )
            record = {
                **identity,
                "output_sha256": digest(obs_path),
                "alignment_median_m": median,
                "alignment_p95_m": p95,
                "alignment_points": len(alignment),
                "pose_correction": correction,
                "depth_quantization_max_m": data["depth_quantization_max_m"],
                "backprojection_quantization_max_m": data["backprojection_quantization_max_m"],
                "seconds": time.perf_counter() - start,
            }
            observations.append(record)
            save_json(obs_root / (name + ".json"), record)
            print(json.dumps(record), flush=True)
        base = {
            "status": "complete",
            "ended_at": datetime.now(UTC).isoformat(),
            "contract": contract,
            "model_predictions_computed": False,
        }
        save_json(
            obs_root / "complete.json", {**base, "rows": observations, "source_sha256": checked}
        )
        save_json(input_root / "complete.json", {**base, "rows": inputs, "gt_exported": False})
    except Exception:
        save_json(
            obs_root / "failure.json",
            {
                "traceback": traceback.format_exc(),
                "completed": len(observations),
                "contract": contract,
            },
        )
        raise


if __name__ == "__main__":
    main()
