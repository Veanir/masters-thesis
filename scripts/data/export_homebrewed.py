"""Observation-only HB inputs and disjoint local GT, preserving all198 IDs."""

import json
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from scripts.common.bop_adapter import CV_TO_PROJECT, camera_mesh, prepare_observation
from scripts.common.paths import script_help
from scripts.data.select_homebrewed_observations import DATA, OUT, ROOT, sha

script_help(__doc__, __name__)


def main():
    reservation = OUT / "cohort-reservation-v1.json"
    auditpath = OUT / "calibration-v1/complete.json"
    cohort = json.loads(reservation.read_text())
    audit = json.loads(auditpath.read_text())
    assert audit["membership_unchanged"] and audit["contract"]["reservation_sha256"] == sha(
        reservation
    )
    inputs = ROOT / "runs/research-evolution-hb-inputs-v1"
    observations = ROOT / "runs/research-evolution-hb-observations-v1"
    assert not inputs.exists() and not observations.exists()
    inputs.mkdir()
    observations.mkdir()
    assert (
        inputs not in observations.parents
        and observations not in inputs.parents
        and inputs != observations
    )
    checked = {}
    meshes = {}
    inputrows = []
    obsrows = []
    contract = {
        "reservation_sha256": sha(reservation),
        "calibration_sha256": sha(auditpath),
        "script_sha256": sha(Path(__file__)),
        "adapter_sha256": sha(Path(__file__).parents[2].joinpath("scripts/common/bop_adapter.py")),
        "source_split": "BOP HB val_primesense",
        "resolution": [640, 480],
        "crop_or_resize": False,
        "minimum_input_pixels": 512,
        "sensor_filter": (
            "Visible source mask, positive depth, median +/-0.25m. No GT "
            "pose/mesh used for input filtering."
        ),
        "depth_encoding": "round(sensor_m/10*65535), uint16. No source sensor correction.",
        "rgb": (
            "Full source image retained; selected model context must be "
            "applied explicitly by its frozen wrapper."
        ),
        "camera": (
            "Source BOP K as float32 and identity cam2world; JSON converted "
            "to tensors in inference environment."
        ),
        "failure_policy": (
            "All198 records remain.16 have insufficient input and must be "
            "reported separately, never replaced or silently skipped in "
            "aggregation."
        ),
        "metric_protocol_frozen": False,
        "model_predictions_computed": False,
        "global_pretraining_unseen_claim": False,
    }
    (observations / "binding.json").write_text(json.dumps(contract, indent=2))
    for ordinal, (row, qa) in enumerate(zip(cohort["observations"], audit["rows"], strict=True)):
        assert all(row[k] == qa[k] for k in ["scene_id", "image_id", "gt_id", "obj_id"])
        paths = {k: DATA / v["path"] for k, v in row["files"].items()}
        for key, path in paths.items():
            if str(path) not in checked:
                checked[str(path)] = sha(path)
            assert checked[str(path)] == row["files"][key]["sha256"]
        camera = json.loads(paths["scene_camera"].read_text())[str(row["image_id"])]
        k = np.asarray(camera["cam_K"], dtype=np.float32).reshape(3, 3)
        rgb = np.asarray(Image.open(paths["rgb"]).convert("RGB"))
        depth = np.asarray(Image.open(paths["depth"]))
        mask = np.asarray(Image.open(paths["mask_visib"])) > 0
        data = prepare_observation(rgb, depth, mask, k, camera["depth_scale"], minimum_pixels=1)
        count = int(data["mask"].sum())
        assert count == qa["filtered_pixels"]
        eligible = count >= 512
        sid = (
            f"hb-obj{row['obj_id']:06d}-scene{row['scene_id']:06d}"
            f"-image{row['image_id']:06d}-gt{row['gt_id']:06d}"
        )
        identity = {
            "sample_id": sid,
            "obj_id": row["obj_id"],
            "scene_id": row["scene_id"],
            "image_id": row["image_id"],
            "gt_id": row["gt_id"],
            "ordinal": ordinal,
            "visibility_bin": row["visibility_bin"],
            "evaluation_role": "project_heldout_real",
            "input_eligible": eligible,
            "input_pixels": count,
            "source_registration_status": qa["status"],
        }
        folder = inputs / sid
        folder.mkdir()
        # The serialization block below has no access to pose or mesh.
        Image.fromarray(rgb).save(folder / "rgb.png")
        Image.fromarray(data["depth_uint16"]).save(folder / "depth.png")
        Image.fromarray(data["mask"].astype(np.uint8) * 255).save(folder / "mask.png")
        (folder / "camera.json").write_text(
            json.dumps({"K": k.tolist(), "cam2world": np.eye(4, dtype=np.float32).tolist()})
        )
        assert np.array_equal(np.asarray(Image.open(folder / "depth.png")), data["depth_uint16"])
        inputrows.append(
            {
                **identity,
                "status": "ready_for_frozen_inference"
                if eligible
                else "insufficient_sensor_pixels",
                "gt_exported": False,
                "export_sha256": {f.name: sha(f) for f in sorted(folder.iterdir())},
            }
        )
        # GT starts here and is written exclusively to a disjoint local folder.
        pose = json.loads(paths["scene_gt"].read_text())[str(row["image_id"])][row["gt_id"]]
        assert pose["obj_id"] == row["obj_id"]
        if row["obj_id"] not in meshes:
            meshes[row["obj_id"]] = trimesh.load(paths["mesh"], process=False, force="mesh")
        mesh = meshes[row["obj_id"]]
        vertices, correction = camera_mesh(mesh.vertices, pose["cam_R_m2c"], pose["cam_t_m2c"])
        points = data["points_cv"] * CV_TO_PROJECT
        n = min(512, len(points))
        ids = np.floor((np.arange(n) + 0.5) * len(points) / n).astype(int)
        target = observations / (sid + ".npz")
        np.savez_compressed(
            target,
            rgb=rgb,
            depth_m=data["depth_m"],
            depth_sensor_m=data["depth_sensor_m"],
            mask=data["mask"],
            intrinsics=k[[0, 1, 0, 1], [0, 1, 2, 2]],
            partial_points_camera_m=points[ids],
            mesh_vertices_camera_m=vertices * CV_TO_PROJECT,
            mesh_faces=np.asarray(mesh.faces, dtype=np.int64),
        )
        obsrows.append(
            {
                **identity,
                "output_sha256": sha(target),
                "alignment_median_m": qa["alignment_median_m"],
                "alignment_p95_m": qa["alignment_p95_m"],
                "pose_so3_correction": correction,
                "depth_quantization_max_m": data["depth_quantization_max_m"],
            }
        )
        if (ordinal + 1) % 33 == 0:
            print("HB EXPORT", ordinal + 1, flush=True)
    assert len(inputrows) == 198 and sum(r["input_eligible"] for r in inputrows) == 182
    (inputs / "complete.json").write_text(
        json.dumps(
            {
                "status": "observation_only_HB_export_complete",
                "contract": contract,
                "rows": inputrows,
                "gt_exported": False,
            },
            indent=2,
        )
    )
    (observations / "complete.json").write_text(
        json.dumps(
            {
                "status": "HB_evaluation_geometry_export_complete",
                "contract": contract,
                "rows": obsrows,
            },
            indent=2,
        )
    )
    print("Preserved198, eligible182, insufficient16; no inference", flush=True)


if __name__ == "__main__":
    main()
