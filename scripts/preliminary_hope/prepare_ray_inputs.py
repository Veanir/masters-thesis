"""Freeze one calibrated view per HOPE object/scene and export disjoint inputs/GT."""

import argparse
import hashlib
import io
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.common.bop_adapter import (
    CV_TO_PROJECT,
    camera_mesh,
    digest,
    prepare_observation,
    save_json,
)
from scripts.preliminary_hope.camera_calibration import letterbox_hope


def select(rows):
    groups = defaultdict(list)
    for row in rows:
        if row["status"] == "passed":
            groups[row["obj_id"], row["scene_id"]].append(row)

    def rank(r):
        return hashlib.sha256(
            f"hope-final-v1:{r['obj_id']}:{r['scene_id']}:{r['image_id']}:{r['gt_id']}".encode()
        ).hexdigest()

    return [min(groups[key], key=rank) for key in sorted(groups)]


def main():
    import trimesh

    p = argparse.ArgumentParser()
    p.add_argument("--archives", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--observations", type=Path, required=True)
    a = p.parse_args()
    import open3d as o3d
    import torch

    inputs, observations = a.inputs.resolve(), a.observations.resolve()
    assert (
        inputs != observations
        and inputs not in observations.parents
        and observations not in inputs.parents
    )
    assert not inputs.exists() and not observations.exists()
    audit = json.loads(a.audit.read_text())
    assert audit["status"] == "complete" and len(audit["rows"]) == 920
    assert (
        digest(Path(__file__).parents[2].joinpath("scripts/preliminary_hope/camera_calibration.py"))
        == audit["contract"]["camera_script_sha256"]
    )
    assert (
        digest(Path(__file__).parents[2].joinpath("scripts/common/bop_adapter.py"))
        == audit["contract"]["bop_script_sha256"]
    )
    sources = json.loads((a.archives / "audit.json").read_text())
    for row in sources["rows"]:
        assert digest(a.archives / row["file"]) == row["sha256"]
    selected = select(audit["rows"])
    assert (
        len(selected) == 146
        and len({r["obj_id"] for r in selected}) == 28
        and len({r["scene_id"] for r in selected}) == 10
    )
    inputs.mkdir(parents=True)
    observations.mkdir(parents=True)
    contract = {
        "purpose": "Project-held-out real HOPE validation cohort for the photorealism study",
        "cohort_rule": (
            "One passed observation per object/scene pair; smallest SHA256 of "
            "hope-final-v1:obj:scene:image:gt, independent of image quality "
            "or predictions"
        ),
        "audit_sha256": digest(a.audit),
        "script_sha256": digest(__file__),
        "source_revision": sources["revision"],
        "n_observations": 146,
        "n_objects": 28,
        "n_scenes": 10,
        "model_predictions_computed": False,
        "global_pretraining_unseen_claim": False,
        "source_split": "BOP HOPE val RealSense",
        "camera_transform": audit["contract"]["transform"],
        "selection_frozen_before_export": True,
        "selection_bias": (
            "Calibration admission uses GT registration. Results conditional "
            "on these quality filters."
        ),
    }
    save_json(observations / "cohort.json", {"contract": contract, "rows": selected})
    input_rows, obs_rows = [], []
    with (
        zipfile.ZipFile(a.archives / "hope_val_realsense.zip") as z,
        zipfile.ZipFile(a.archives / "hope_models.zip") as meshzip,
    ):
        cache = {}
        for ordinal, row in enumerate(selected):
            scene, image_id, gt_id, obj = (
                row["scene_id"],
                row["image_id"],
                row["gt_id"],
                row["obj_id"],
            )
            base = f"val/{scene:06d}"
            cam = json.loads(z.read(base + "/scene_camera.json"))[str(image_id)]
            pose = json.loads(z.read(base + "/scene_gt.json"))[str(image_id)][gt_id]
            assert pose["obj_id"] == obj
            rgb = np.asarray(
                Image.open(io.BytesIO(z.read(base + f"/rgb/{image_id:06d}.png"))).convert("RGB")
            )
            depth = np.asarray(Image.open(io.BytesIO(z.read(base + f"/depth/{image_id:06d}.png"))))
            mask = (
                np.asarray(
                    Image.open(
                        io.BytesIO(z.read(base + f"/mask_visib/{image_id:06d}_{gt_id:06d}.png"))
                    )
                )
                > 0
            )
            rgb, depth, mask, k = letterbox_hope(
                rgb, depth, mask, np.array(cam["cam_K"]).reshape(3, 3)
            )
            data = prepare_observation(rgb, depth, mask, k, cam["depth_scale"])
            assert int(data["mask"].sum()) == row["filtered_pixels"]
            if obj not in cache:
                cache[obj] = trimesh.load(
                    io.BytesIO(meshzip.read(f"models_eval/obj_{obj:06d}.ply")),
                    file_type="ply",
                    process=False,
                    force="mesh",
                )
            mesh = cache[obj]
            vertices, correction = camera_mesh(mesh.vertices, pose["cam_R_m2c"], pose["cam_t_m2c"])
            rays = o3d.t.geometry.RaycastingScene(nthreads=2)
            rays.add_triangles(
                o3d.core.Tensor(vertices.astype(np.float32)),
                o3d.core.Tensor(np.asarray(mesh.faces, np.uint32)),
            )
            distances = rays.compute_distance(
                o3d.core.Tensor(data["sensor_points_cv"].astype(np.float32)), nthreads=2
            ).numpy()
            median, p95 = np.quantile(distances, [0.5, 0.95]).tolist()
            assert (
                abs(median - row["alignment_median_m"]) < 1e-6
                and abs(p95 - row["alignment_p95_m"]) < 1e-6
            )
            sid = f"hope-obj{obj:06d}-scene{scene:06d}-image{image_id:06d}-gt{gt_id:06d}"
            folder = inputs / sid
            folder.mkdir()
            Image.fromarray(data["rgb"]).save(folder / "rgb.png")
            Image.fromarray(data["depth_uint16"]).save(folder / "depth.png")
            Image.fromarray(data["mask"].astype(np.uint8) * 255).save(folder / "mask.png")
            torch.save(torch.from_numpy(k.astype(np.float32)), folder / "intrinsics.pt")
            torch.save(torch.eye(4), folder / "cam2world.pt")
            np.save(
                folder / "input512_camera_m.npy", data["partial_points_camera_m"].astype(np.float32)
            )
            identity = {
                "sample_id": sid,
                "obj_id": obj,
                "scene_id": scene,
                "image_id": image_id,
                "gt_id": gt_id,
                "ordinal": ordinal,
                "evaluation_role": "project_heldout_real",
            }
            input_rows.append(
                {
                    **identity,
                    "gt_exported": False,
                    "rayst3r_output_to_project_camera": CV_TO_PROJECT.tolist(),
                    "export_sha256": {p.name: digest(p) for p in sorted(folder.iterdir())},
                }
            )
            out = observations / (sid + ".npz")
            np.savez_compressed(
                out,
                rgb=data["rgb"],
                depth_m=data["depth_m"],
                depth_sensor_m=data["depth_sensor_m"],
                mask=data["mask"],
                intrinsics=k[[0, 1, 0, 1], [0, 1, 2, 2]],
                partial_points_camera_m=data["partial_points_camera_m"],
                mesh_vertices_camera_m=vertices * CV_TO_PROJECT,
                mesh_faces=np.asarray(mesh.faces, np.int64),
            )
            obs_rows.append(
                {
                    **identity,
                    "output_sha256": digest(out),
                    "alignment_median_m": median,
                    "alignment_p95_m": p95,
                    "pose_correction": correction,
                    "depth_quantization_max_m": data["depth_quantization_max_m"],
                }
            )
            if (ordinal + 1) % 28 == 0:
                print(json.dumps({"exported": ordinal + 1, "total": 146}), flush=True)
    save_json(
        inputs / "complete.json",
        {
            "status": "complete",
            "contract": contract,
            "rows": input_rows,
            "gt_exported": False,
            "model_predictions_computed": False,
        },
    )
    save_json(
        observations / "complete.json",
        {
            "status": "complete",
            "contract": contract,
            "rows": obs_rows,
            "model_predictions_computed": False,
            "cohort_sha256": digest(observations / "cohort.json"),
        },
    )
    print(json.dumps({"status": "complete", "n": len(input_rows)}), flush=True)


if __name__ == "__main__":
    main()
