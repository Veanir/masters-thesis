"""CPU triangle supervision for certified phase2 observations; bounded RAM."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--vendor", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    import open3d as o3d
    import torch

    sys.path.insert(0, str(a.vendor.resolve()))
    from eval_wrapper.sample_poses import pointmap_to_poses

    torch.set_num_threads(a.threads)
    manifest = json.loads((a.source / "complete.json").read_text())
    assert manifest["status"] == "certified_synthetic_observations_exported"
    assert not a.output.exists()
    a.output.mkdir()
    rows = []
    binding = {
        "source_manifest_sha256": sha(a.source / "complete.json"),
        "script_sha256": sha(Path(__file__)),
        "pose_sampler_sha256": sha(a.vendor / "eval_wrapper/sample_poses.py"),
        "novel_views": 21,
        "virtual_depth_rule": (
            "finite positive first hit below10m, nonzero uint16 code; input "
            "sensor threshold is unchanged"
        ),
        "versions": {"torch": torch.__version__, "open3d": o3d.__version__},
        "gt_access": "Synthetic TRAIN/VAL mesh supervision only. No real held-out data.",
        "camera": (
            "CV+x right,+y down,+z forward; metric meshes. Unnormalized rays "
            "make t_hit axial depth."
        ),
        "memory": "One observation and its21 target views at a time.",
    }
    (a.output / "binding.json").write_text(json.dumps(binding, indent=2))
    for row in manifest["rows"]:
        tick = time.monotonic()
        path = a.source / row["sample_id"] / "observation.npz"
        assert sha(path) == row["files"]["observation.npz"]
        with np.load(path) as z:
            data = {k: z[k] for k in z.files}
        k = data["K"]
        mask = data["input_mask"]
        depth = data["input_depth_uint16"].astype(np.float32) * np.float32(10 / 65535)
        h, w = mask.shape
        v, u = np.mgrid[:h, :w]
        dirs = np.stack(
            ((u - k[0, 2]) / k[0, 0], (v - k[1, 2]) / k[1, 1], np.ones_like(u)), -1
        ).astype(np.float32)
        observed = (dirs * depth[..., None])[mask]
        assert len(observed) >= 512
        cams = pointmap_to_poses(torch.from_numpy(observed), n_poses=5, device="cpu").astype(
            np.float32
        )
        assert cams.shape == (21, 4, 4) and np.isfinite(cams).all()
        assert np.allclose(
            cams[:, :3, :3].transpose(0, 2, 1) @ cams[:, :3, :3], np.eye(3), atol=2e-6
        )
        scene = o3d.t.geometry.RaycastingScene(nthreads=a.threads)
        scene.add_triangles(
            o3d.core.Tensor(data["mesh_vertices_camera_cv_m"].astype(np.float32)),
            o3d.core.Tensor(data["mesh_faces"].astype(np.uint32)),
        )
        depths = []
        masks = []
        errors = []
        pixels = []
        for camera in cams:
            directions = dirs @ camera[:3, :3].T
            origins = np.broadcast_to(camera[:3, 3], directions.shape)
            rays = np.concatenate((origins, directions), -1).astype(np.float32)
            hit = scene.cast_rays(o3d.core.Tensor(rays), nthreads=a.threads)["t_hit"].numpy()
            valid = np.isfinite(hit) & (hit > 0) & (hit < 10)
            encoded = np.rint(np.where(valid, hit, 0).astype(float) * 65535 / 10).astype(np.uint16)
            valid &= encoded > 0
            encoded[~valid] = 0
            indices = np.flatnonzero(valid)
            assert len(indices) > 100, (row["sample_id"], len(indices))
            chosen = indices[
                np.floor(
                    (np.arange(min(512, len(indices))) + 0.5)
                    * len(indices)
                    / min(512, len(indices))
                ).astype(int)
            ]
            decoded = encoded.ravel()[chosen].astype(np.float32) * np.float32(10 / 65535)
            points = (
                origins.reshape(-1, 3)[chosen]
                + directions.reshape(-1, 3)[chosen] * decoded[:, None]
            )
            distances = scene.compute_distance(o3d.core.Tensor(points), nthreads=a.threads).numpy()
            error = float(distances.max())
            assert np.isfinite(error) and error < 0.00025, (row["sample_id"], error)
            depths.append(encoded)
            masks.append(valid)
            errors.append(error)
            pixels.append(len(indices))
        target = a.output / (row["sample_id"] + ".npz")
        np.savez_compressed(
            target,
            rgb=data["rgb"],
            input_depth_uint16=data["input_depth_uint16"],
            input_mask=mask,
            K=k,
            novel_c2ws=cams,
            novel_depth_uint16=np.stack(depths),
            novel_masks=np.stack(masks),
        )
        result = {
            "sample_id": row["sample_id"],
            "family_id": row["family_id"],
            "key": row["key"],
            "split": row["split"],
            "source_observation_sha256": sha(path),
            "output_sha256": sha(target),
            "novel_surface_error_max_m": max(errors),
            "novel_valid_pixel_counts": pixels,
            "seconds": time.monotonic() - tick,
            "bytes": target.stat().st_size,
        }
        rows.append(result)
        (a.output / "progress.json").write_text(json.dumps(rows, indent=2))
        print("LABELS", len(rows), row["sample_id"], result["seconds"], flush=True)
    (a.output / "complete.json").write_text(
        json.dumps(
            {**binding, "status": "certified_synthetic_ray_labels_complete", "rows": rows}, indent=2
        )
    )


if __name__ == "__main__":
    main()
