"""Audit every public HOPE validation observation; no predictions or cohort selection."""

from __future__ import annotations

import argparse
import io
import json
import time
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.common.bop_adapter import camera_mesh, digest, prepare_observation, save_json
from scripts.common.paths import script_help
from scripts.preliminary_hope.camera_calibration import letterbox_hope

script_help(__doc__, __name__)


def audit(archives, output):
    import open3d as o3d
    import trimesh

    if output.exists():
        raise ValueError("Use a fresh audit directory")
    metadata = json.loads((archives / "audit.json").read_text())
    assert metadata["status"] == "complete"
    for row in metadata["rows"]:
        assert digest(archives / row["file"]) == row["sha256"]
    output.mkdir(parents=True)
    contract = {
        "status": "defined_before_calibration",
        "source_revision": metadata["revision"],
        "script_sha256": digest(__file__),
        "camera_script_sha256": digest(
            Path(__file__).parents[2].joinpath("scripts/preliminary_hope/camera_calibration.py")
        ),
        "bop_script_sha256": digest(
            Path(__file__).parents[2].joinpath("scripts/common/bop_adapter.py")
        ),
        "minimum_pixels": 512,
        "median_depth_filter_m": 0.25,
        "registration_median_limit_m": 0.002,
        "registration_p95_limit_m": 0.015,
        "threshold_provenance": (
            "same absolute quality thresholds as previously exposed YCB-V diagnostic"
        ),
        "model_meshes": "models_eval; BOP millimetres and BOP camera pose",
        "native_depth_correction": "already included by dataset authors; not applied again",
        "transform": (
            "RGB 3x3 area average, depth block centre, full visible-mask "
            "footprint; 640x360 with 60 rows padding each side; calibrated "
            "block-centre intrinsics"
        ),
        "model_predictions_computed": False,
        "cohort_selected": False,
    }
    save_json(output / "binding.json", contract)
    rows = []
    with (
        zipfile.ZipFile(archives / "hope_val_realsense.zip") as data_zip,
        zipfile.ZipFile(archives / "hope_models.zip") as meshes_zip,
    ):
        meshes = {}
        for gt_path in sorted(n for n in data_zip.namelist() if n.endswith("scene_gt.json")):
            folder = gt_path.rsplit("/", 1)[0]
            gt = json.loads(data_zip.read(gt_path))
            camera = json.loads(data_zip.read(folder + "/scene_camera.json"))
            info = json.loads(data_zip.read(folder + "/scene_gt_info.json"))
            for image_id in sorted(gt, key=int):
                image_stem = f"{int(image_id):06d}"
                rgb = np.asarray(
                    Image.open(
                        io.BytesIO(data_zip.read(folder + f"/rgb/{image_stem}.png"))
                    ).convert("RGB")
                )
                depth = np.asarray(
                    Image.open(io.BytesIO(data_zip.read(folder + f"/depth/{image_stem}.png")))
                )
                k = np.asarray(camera[image_id]["cam_K"]).reshape(3, 3)
                for gt_id, pose in enumerate(gt[image_id]):
                    tick = time.perf_counter()
                    obj_id = pose["obj_id"]
                    record = {
                        "scene_id": int(folder.split("/")[-1]),
                        "image_id": int(image_id),
                        "gt_id": gt_id,
                        "obj_id": obj_id,
                        "visib_fract": info[image_id][gt_id]["visib_fract"],
                    }
                    visible = (
                        np.asarray(
                            Image.open(
                                io.BytesIO(
                                    data_zip.read(
                                        folder + f"/mask_visib/{image_stem}_{gt_id:06d}.png"
                                    )
                                )
                            )
                        )
                        > 0
                    )
                    reduced_rgb, reduced_depth, reduced_mask, reduced_k = letterbox_hope(
                        rgb, depth, visible, k
                    )
                    record.update(
                        native_visible_pixels=int(visible.sum()),
                        reduced_visible_pixels=int(reduced_mask.sum()),
                    )
                    try:
                        obs = prepare_observation(
                            reduced_rgb,
                            reduced_depth,
                            reduced_mask,
                            reduced_k,
                            camera[image_id]["depth_scale"],
                        )
                    except ValueError as error:
                        record.update(status="input_ineligible", reason=str(error))
                    else:
                        if obj_id not in meshes:
                            meshes[obj_id] = trimesh.load(
                                io.BytesIO(meshes_zip.read(f"models_eval/obj_{obj_id:06d}.ply")),
                                file_type="ply",
                                process=False,
                                force="mesh",
                            )
                        mesh = meshes[obj_id]
                        vertices, correction = camera_mesh(
                            mesh.vertices, pose["cam_R_m2c"], pose["cam_t_m2c"]
                        )
                        scene = o3d.t.geometry.RaycastingScene(nthreads=2)
                        scene.add_triangles(
                            o3d.core.Tensor(vertices.astype(np.float32)),
                            o3d.core.Tensor(np.asarray(mesh.faces, np.uint32)),
                        )
                        distances = scene.compute_distance(
                            o3d.core.Tensor(obs["sensor_points_cv"].astype(np.float32)), nthreads=2
                        ).numpy()
                        median, p95 = np.quantile(distances, [0.5, 0.95]).tolist()
                        passed = (
                            median <= contract["registration_median_limit_m"]
                            and p95 <= contract["registration_p95_limit_m"]
                        )
                        record.update(
                            status="passed" if passed else "registration_failed",
                            filtered_pixels=int(obs["mask"].sum()),
                            median_depth_m=obs["median_depth_m"],
                            alignment_median_m=median,
                            alignment_p95_m=p95,
                            pose_so3_correction=correction,
                            depth_quantization_max_m=obs["depth_quantization_max_m"],
                        )
                    record["seconds"] = time.perf_counter() - tick
                    rows.append(record)
            save_json(
                output / "progress.json",
                {"completed": len(rows), "counts": dict(Counter(r["status"] for r in rows))},
            )
            print(
                json.dumps(
                    {
                        "scene": folder,
                        "completed": len(rows),
                        "counts": dict(Counter(r["status"] for r in rows)),
                    }
                ),
                flush=True,
            )
    save_json(
        output / "complete.json",
        {
            "status": "complete",
            "contract": contract,
            "rows": rows,
            "counts": dict(Counter(r["status"] for r in rows)),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.archives, args.output)
