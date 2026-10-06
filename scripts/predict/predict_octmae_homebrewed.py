"""Bound OctMAE system evaluation with explicit complete-population failures."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8388608), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-root", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--job", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--campaign", type=Path)
    a = p.parse_args()
    root = a.model_root.resolve()
    inputs = a.inputs.resolve()
    out = a.output.resolve()
    job_path = a.job.resolve()
    job = json.loads(job_path.read_text())
    assert not out.exists() and job["purpose"] in [
        "frozen_final_HB_inference",
        "final_inference_development_canary",
    ]
    assert (
        job["inference_seed"] == 20260906
        and job["checkpoint_step"] is None
        and job["normalization"] == "imagenet"
    )
    assert job["source_commit"] == "98bbd21e1f445b77b131be02fa902940e0836ab0"
    assert (
        job["checkpoint_sha256"]
        == "947c9cb8c9d342807747960d575990c7dc66c9559482cbce22d92b4eead6d049"
    )
    assert (
        sha(root / "checkpoints/octmae.ckpt") == job["checkpoint_sha256"]
        and sha(Path(__file__)) == job["runner_sha256"]
    )
    assert (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        == job["source_commit"]
    )
    subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", "octmae", "configs/default.yaml"],
        cwd=root,
        check=True,
    )
    manifest_path = inputs / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert not manifest["gt_uploaded"]
    assert sha(manifest_path) == job["input_manifest_sha256"] and len(
        {r["sample_id"] for r in manifest["rows"]}
    ) == len(manifest["rows"])
    if job["purpose"] == "frozen_final_HB_inference":
        assert a.campaign is not None and sha(a.campaign) == job["campaign_sha256"]
        campaign = json.loads(a.campaign.read_text())
        assert campaign["status"] == "frozen_evolution_campaign"
        method = next(m for m in campaign["evaluation_methods"] if m["id"] == job["method_id"])
        assert (
            method["model"] == "OctMAE"
            and method["role"] == "pretrained"
            and method["input_variant"] == job["input_variant"]
        )
        assert len(manifest["rows"]) == 198 and all(
            r["cohort"] == "HB198" for r in manifest["rows"]
        )
    else:
        assert all(r["cohort"] == "YCB-exposed" for r in manifest["rows"])
    assert all(
        r["variant"] == job["input_variant"] and r["normalization"] == "imagenet"
        for r in manifest["rows"]
    )
    os.chdir(root)
    sys.path.insert(0, str(root))
    import torch
    from octmae.nets import OctMAE
    from octmae.nets.utils import get_xyz_from_octree
    from octmae.utils.config import parse_config
    from octmae.utils.misc import fetch_data, unnormalize_pts

    assert torch.__version__.split("+")[0] == "2.2.0", (
        "Reuse the screened OctMAE environment, not the Ray venv"
    )
    torch.manual_seed(job["inference_seed"])
    np.random.seed(job["inference_seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    out.mkdir()
    config_source = root / "configs/default.yaml"
    assert sha(config_source) == job["config_sha256"]
    lines = config_source.read_text().splitlines()
    for i, line in enumerate(lines):
        if line.startswith(("train_dataset_url:", "val_dataset_url:")):
            key, value = line.split(":", 1)
            lines[i] = key + ": " + json.dumps(value.strip())
    config_path = out / "config-inference.yaml"
    config_path.write_text("\n".join(lines) + "\n")
    config = parse_config(str(config_path))
    config.update_octree = True
    model = OctMAE(config)
    # Public Lightning checkpoint already loaded in screening; require its exact
    # hash before the intentional legacy loader in the preserved Torch2.2 env.
    checkpoint = torch.load(
        root / "checkpoints/octmae.ckpt", map_location="cpu", weights_only=False
    )
    weights = checkpoint["state_dict"]
    assert all(key.startswith("model.") for key in weights)
    model.load_state_dict(
        {key.removeprefix("model."): value for key, value in weights.items()}, strict=True
    )
    del checkpoint, weights
    model.cuda().eval()
    records = []
    for row in manifest["rows"]:
        sid = row["sample_id"]
        folder = inputs / sid
        assert set(row["files"]) == {"rgb.png", "depth.tiff", "mask.png", "camera.json"}
        assert all(sha(folder / name) == value for name, value in row["files"].items())
        cloud = np.empty((0, 3), np.float32)
        invalid = None
        reason = None
        seconds = 0.0
        peak = 0
        represented = None
        forward_executed = False
        if row["input_eligible"]:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            tick = time.monotonic()
            batch = fetch_data(
                str(folder / "rgb.png"),
                str(folder / "depth.tiff"),
                str(folder / "mask.png"),
                str(folder / "camera.json"),
                config,
                depth_scale=1.0,
            )
            represented = int(batch[3][0].points.shape[0])
            if represented == 0:
                reason = "empty_input_after_official_fixed_grid_clip"
                status = "empty_prediction"
            else:
                with torch.inference_mode():
                    prediction = model(batch)
                    forward_executed = True
                    tree = prediction["octrees_out"]
                    points = get_xyz_from_octree(tree, config.max_lod, True)
                    if len(points):
                        points = unnormalize_pts(
                            points, batch[-2][0], config.grid_size, 1 << config.min_lod
                        )
                        points = (
                            points
                            - tree.normals[config.max_lod] * tree.features[config.max_lod][:, :1]
                        )
                    cloud = points.cpu().numpy() / 1000.0 * np.array([1, -1, -1], np.float32)
                if cloud.ndim != 2 or cloud.shape[1] != 3 or not np.isfinite(cloud).all():
                    bad = out / (sid + "-invalid.npy")
                    np.save(bad, cloud, allow_pickle=False)
                    invalid = {"filename": bad.name, "sha256": sha(bad), "shape": list(cloud.shape)}
                    cloud = np.empty((0, 3), np.float32)
                    status = "numerical_failure"
                else:
                    status = (
                        "empty_prediction"
                        if len(cloud) == 0
                        else "small_prediction"
                        if len(cloud) < 512
                        else "predicted"
                    )
            torch.cuda.synchronize()
            seconds = time.monotonic() - tick
            peak = torch.cuda.max_memory_allocated()
        else:
            status = "insufficient_input"
        target = out / (sid + ".npz")
        np.savez_compressed(target, points_camera_m=cloud)
        records.append(
            {
                "sample_id": sid,
                "cohort": row["cohort"],
                "status": status,
                "count": len(cloud),
                "sha256": sha(target),
                "invalid_artifact": invalid,
                "system_failure_reason": reason,
                "model_forward_executed": forward_executed,
                "points_after_official_grid_clip": represented,
                "seconds": seconds,
                "peak_cuda_bytes": peak,
            }
        )
        save(out / "progress.json", records)
        if len(records) % 20 == 0:
            print("OCTMAE FINAL", job["input_variant"], len(records), status, flush=True)
    save(
        out / "complete.json",
        {
            "status": "bound_observation_evaluation_complete",
            "rows": records,
            "input_manifest_sha256": sha(manifest_path),
            "job_sha256": sha(job_path),
            "job": job,
            "script_sha256": sha(Path(__file__)),
            "gt_uploaded": False,
            "strict_checkpoint_load": True,
            "config_sha256": sha(config_source),
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(),
            "source_files_sha256": {
                str(path.relative_to(root)): sha(path)
                for path in sorted((root / "octmae").rglob("*.py"))
            },
            "precision": (
                "Screened Torch2.2 float32 inference, matmulTF32 off, cuDNNTF32 "
                "on; ImageNet RGB normalization"
            ),
            "scope": (
                "Fixed public system. Source eligibility precedes any crop. "
                "Official fixed-grid clipping may yield an explicit empty system "
                "reconstruction without network forward. Infrastructure "
                "exceptions abort. No depth/pose alignment or output-quality "
                "selection."
            ),
        },
    )


if __name__ == "__main__":
    main()
