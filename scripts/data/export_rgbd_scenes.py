"""Export independently certified scenes with paired appearance/depth inputs."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.common.bop_adapter import prepare_observation
from scripts.data.corrupt_rendered_depth import (
    camera_matrices,
    crop_channel,
    cropped_intrinsics,
    sensor_candidate,
)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--qa", type=Path, required=True)
    p.add_argument("--statistics", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    manifest = json.loads((a.source / "complete.json").read_text())
    qa = json.loads(a.qa.read_text())
    statistics = json.loads(a.statistics.read_text())
    assert qa["all_targets_passed"] and qa["source_manifest_sha256"] == sha(
        a.source / "complete.json"
    )
    approved = {r["key"]: r for r in qa["rows"]}
    assert set(approved) == {r["key"] for r in manifest["rows"]}
    assert not a.output.exists()
    a.output.mkdir()
    rows = []
    for target in manifest["rows"]:
        assert approved[target["key"]]["passed_20_frames"]
        accepted = {
            r["frame_id"]
            for r in approved[target["key"]]["frames"]
            if r["selected_candidate"] and r["passed"]
        }
        assert accepted == {r["frame_id"] for r in target["frames"]}
        for frame in target["frames"]:
            folder = a.source / frame["relative_folder"]
            view = frame["view"]
            sid = frame["frame_id"]
            dest = a.output / sid
            dest.mkdir()
            assert all(sha(folder / n) == h for n, h in frame["files"].items())
            rgb = np.asarray(Image.open(folder / f"rgb-{view}.png").convert("RGB"))
            with np.load(folder / f"rgbd-{view}.npz") as z:
                clean = z["depth_clean_m"]
                mask = z["target_mask"].astype(bool)
            with np.load(folder / f"camera-{view}.npz") as z:
                k, w2gl = camera_matrices(z)
            height, width = clean.shape
            box = (0, 0, width, height)
            rgb = crop_channel(rgb, box, rgb=True)
            clean = crop_channel(clean, box)
            mask = crop_channel(mask, box)
            k = cropped_intrinsics(k, box)
            sensor, noise = sensor_candidate(clean, mask, sid, statistics)
            # The development calibration uses 1mm source quantization. Convert
            # to this common sensor representation before the shared adapter.
            step = noise["depth_step_m"]
            assert np.isclose(step, 0.001, atol=1e-12)
            raw = np.rint(sensor.astype(float) / step).astype(np.uint32)
            data = prepare_observation(rgb, raw, mask, k, step * 1000)
            scene = json.loads((folder / "scene.json").read_text())
            obj = scene["objects"][0]
            assert obj["key"] == target["key"] and sha(folder / obj["file"]) == obj["sha256"]
            with np.load(folder / obj["file"]) as z:
                verts = (z["vertices_world_m"] @ w2gl[:3, :3].T + w2gl[:3, 3]) * [1, -1, -1]
                faces = z["faces"]
            assert verts[:, 2].min() > 0
            # This package is TRAIN/VAL supervision. It is never an inference
            # input bundle, where GT meshes must be kept in a separate path.
            np.savez_compressed(
                dest / "observation.npz",
                rgb=rgb,
                input_depth_uint16=data["depth_uint16"],
                input_mask=data["mask"],
                K=k.astype(np.float32),
                depth_clean_m=clean,
                source_target_mask=mask,
                depth_sensor_m=data["depth_sensor_m"],
                mesh_vertices_camera_cv_m=verts,
                mesh_faces=faces,
            )
            Image.fromarray(rgb).save(dest / "rgb.png")
            Image.fromarray(data["mask"].astype(np.uint8) * 255).save(dest / "mask.png")
            Image.fromarray(data["depth_uint16"]).save(dest / "depth.png")
            assert np.array_equal(np.asarray(Image.open(dest / "depth.png")), data["depth_uint16"])
            row = {
                "sample_id": sid,
                "key": target["key"],
                "family_id": target["family_id"],
                "split": target["split_pool"],
                "source_frame": frame,
                "source_mesh_sha256": obj["sha256"],
                "noise": noise,
                "input_pixels": int(data["mask"].sum()),
                "median_depth_m": data["median_depth_m"],
                "files": {f.name: sha(f) for f in sorted(dest.iterdir())},
            }
            rows.append(row)
        print("EXPORTED", target["asset_id"], 20, flush=True)
    result = {
        "status": "certified_synthetic_observations_exported",
        "rows": rows,
        "source_manifest_sha256": sha(a.source / "complete.json"),
        "qa_sha256": sha(a.qa),
        "statistics_sha256": sha(a.statistics),
        "script_sha256": sha(Path(__file__)),
        "contract_sha256": sha(
            Path(__file__).parents[2].joinpath("scripts/data/corrupt_rendered_depth.py")
        ),
        "adapter_sha256": sha(Path(__file__).parents[2].joinpath("scripts/common/bop_adapter.py")),
        "input_resolution": [640, 480],
        "crop": False,
        "rgb": "Full original RGB; context masking belongs to the explicitly paired model wrapper.",
        "depth": (
            "One deterministic realization per observation reused by "
            "BASE/CLASSIC/PHOTO, 1mm sensor quantization then Ray10m uint16 "
            "encoding."
        ),
        "novel_view_supervision_ready": False,
        "training_ready": False,
    }
    (a.output / "complete.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
