"""Observation-only evaluation of a bound pretrained/final-delta Ray model."""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8388608), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--job", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--historical-freeze", type=Path)
    p.add_argument("--campaign", type=Path)
    a = p.parse_args()
    work = a.work.resolve()
    inputs = a.inputs.resolve()
    job_path = a.job.resolve()
    output = a.output.resolve()
    job = json.loads(job_path.read_text())
    if a.checkpoint:
        a.checkpoint = a.checkpoint.resolve()
    assert job["context"] == "masked" and job["inference_seed"] == 20260905
    assert sha(Path(__file__)) == job["runner_sha256"]
    assert job["purpose"] in ["frozen_final_HB_inference", "final_inference_development_canary"]
    if job["purpose"] == "frozen_final_HB_inference":
        assert a.campaign is not None and sha(a.campaign) == job["campaign_sha256"]
        campaign = json.loads(a.campaign.read_text())
        assert campaign["status"] == "frozen_evolution_campaign"
        method = next(m for m in campaign["evaluation_methods"] if m["id"] == job["method_id"])
        assert method["model"] == "RaySt3R" and method["context"] == job["context"]
        if method["role"] == "historical":
            assert (
                method["checkpoint_model"] == job["checkpoint_model"]
                and job["checkpoint_kind"] == "historical"
            )
        elif method["role"] == "adapted":
            assert (
                method["training_job_sha256"] == job["training_job_sha256"]
                and job["checkpoint_kind"] == "evolution"
            )
        else:
            assert method["role"] == "pretrained" and job["checkpoint_sha256"] is None
    manifest_path = inputs / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert not manifest["gt_uploaded"] and sha(manifest_path) == job["input_manifest_sha256"]
    assert len({r["sample_id"] for r in manifest["rows"]}) == len(manifest["rows"])
    if job["purpose"] == "frozen_final_HB_inference":
        assert len(manifest["rows"]) == 198 and all(
            r["cohort"] == "HB198" for r in manifest["rows"]
        )
    else:
        assert all(r["cohort"] in ["YCB-exposed", "synthetic-val"] for r in manifest["rows"])
    code_path = work.parent / "code-binding.json"
    code = json.loads(code_path.read_text())
    assert sha(code_path) == job["code_binding_sha256"]
    assert all(sha(work.parent / n) == h for n, h in code["files"].items())
    assert all(sha(work.parent / n) == r["sha256"] for n, r in code["weights"].items())
    import torch

    vendor = work / "vendor/rayst3r"
    os.chdir(vendor)
    sys.path[:0] = [str(vendor), str(Path(__file__).resolve().parent)]
    from scripts.common.ray_inference import PROFILE
    from scripts.common.ray_inference import install as portable

    portable()
    from scripts.train.rgb_context import install, set_context

    install()
    set_context(job["context"])
    from eval_wrapper.eval import EvalWrapper, eval_scene

    from scripts.common.ray_view_chunks import decoder_view_chunks

    torch.manual_seed(job["inference_seed"])
    np.random.seed(job["inference_seed"])
    wrapper = EvalWrapper(str(work / "weights/rayst3r.pth"))
    if a.checkpoint:
        assert sha(a.checkpoint) == job["checkpoint_sha256"]
        # Our hash-bound checkpoint includes TorchVersion metadata. Its only
        # additional pickle global was inspected locally before any evaluation.
        with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
            state = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
        assert state["step"] == job["checkpoint_step"]
        if job["checkpoint_kind"] == "historical":
            assert (
                a.historical_freeze is not None
                and sha(a.historical_freeze) == job["historical_freeze_sha256"]
            )
            frozen = json.loads(a.historical_freeze.read_text())
            assert frozen["status"] == "frozen" and len(frozen["runs"]) == 9
            ref = next(r for r in frozen["runs"] if r["run"] == job["checkpoint_model"])
            assert (
                ref["arm"] == "BASE"
                and ref["seed"] in [0, 1, 2]
                and ref["checkpoint_sha256"] == sha(a.checkpoint)
            )
            complete_path = a.checkpoint.parent / "complete.json"
            assert sha(complete_path) == ref["complete_sha256"]
            complete = json.loads(complete_path.read_text())
            assert complete["status"] == "complete" and complete["step"] == 2160
            assert (
                state["step"] == 2160
                and not state["binding"]["canary"]
                and state["binding"]["arm"] == "BASE"
                and state["binding"]["seed"] == ref["seed"]
            )
            assert sha(work / "weights/rayst3r.pth") == state["binding"]["base_weights_sha256"]
            assert (
                sha(work / "weights/dinov2_vitl14_reg4_pretrain.pth")
                == state["binding"]["dino_weights_sha256"]
            )
            assert all(
                sha(work / n) == h for n, h in state["binding"]["vendor_sources_sha256"].items()
            )
        else:
            assert (
                job["checkpoint_kind"] == "evolution"
                and state["binding"]["context"] == job["context"]
            )
            if job["purpose"] == "frozen_final_HB_inference":
                assert (
                    state["step"] == 25600
                    and state["binding"]["job_sha256"] == job["training_job_sha256"]
                )
                assert state["binding"]["arm"] == method["arm"]
            assert all(
                sha(work / "weights" / n) == h for n, h in state["binding"]["weights"].items()
            )
            assert all(sha(work / n) == h for n, h in state["binding"]["vendor_sources"].items())
        ids = {
            id(p)
            for module in [
                *list(wrapper.model.decoder_blocks)[-2:],
                wrapper.model.pts_head,
                wrapper.model.classifier_head,
            ]
            for p in module.parameters()
        }
        expected = {n for n, p in wrapper.model.named_parameters() if id(p) in ids}
        assert set(state["delta"]) == expected
        if job["checkpoint_kind"] == "evolution":
            assert set(state["binding"]["trainable"]) == expected
        assert all(torch.isfinite(value).all() for value in state["delta"].values())
        incompatible = wrapper.model.load_state_dict(state["delta"], strict=False)
        assert (
            not incompatible.unexpected_keys
            and set(incompatible.missing_keys) == set(wrapper.model.state_dict()) - expected
        )
        del state
    else:
        assert job["checkpoint_sha256"] is None
    wrapper.eval().requires_grad_(False)
    dino = torch.hub.load(
        str(work / "vendor/dinov2"), "dinov2_vitl14_reg", source="local", pretrained=False
    )
    dino.load_state_dict(
        torch.load(
            work / "weights/dinov2_vitl14_reg4_pretrain.pth", map_location="cpu", weights_only=True
        ),
        strict=True,
    )
    dino.eval().cuda().requires_grad_(False)
    output.mkdir(exist_ok=False)
    records = []
    for row in manifest["rows"]:
        folder = inputs / row["sample_id"]
        assert set(row["files"]) == {"rgb.png", "depth.png", "mask.png", "camera.json"}
        assert all(sha(folder / n) == h for n, h in row["files"].items())
        camera = json.loads((folder / "camera.json").read_text())
        cloud = np.empty((0, 3), np.float32)
        seconds = 0.0
        peak = 0
        invalid_artifact = None
        if row["input_eligible"]:
            torch.save(torch.tensor(camera["K"], dtype=torch.float32), folder / "intrinsics.pt")
            torch.save(
                torch.tensor(camera["cam2world"], dtype=torch.float32), folder / "cam2world.pt"
            )
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            tick = time.monotonic()
            # Infrastructure exceptions intentionally abort the incomplete run;
            # they are not converted to scientific model failures.
            with torch.inference_mode(), decoder_view_chunks(wrapper.model, 3):
                points = eval_scene(
                    wrapper, str(folder), dino_model=dino, set_conf=5, n_pred_views=5
                )
            torch.cuda.synchronize()
            cloud = points.detach().float().cpu().numpy() * np.array([1, -1, -1], np.float32)
            seconds = time.monotonic() - tick
            peak = torch.cuda.max_memory_allocated()
            if cloud.ndim != 2 or cloud.shape[1] != 3 or not np.isfinite(cloud).all():
                invalid_path = output / (row["sample_id"] + "-invalid.npy")
                np.save(invalid_path, cloud, allow_pickle=False)
                invalid_artifact = {
                    "filename": invalid_path.name,
                    "sha256": sha(invalid_path),
                    "shape": list(cloud.shape),
                }
                status = "numerical_failure"
                cloud = np.empty((0, 3), np.float32)
            else:
                status = (
                    "empty_prediction"
                    if len(cloud) == 0
                    else "small_prediction"
                    if len(cloud) < 512
                    else "predicted"
                )
        else:
            status = "insufficient_input"
        path = output / (row["sample_id"] + ".npz")
        np.savez_compressed(path, points_camera_m=cloud)
        records.append(
            {
                "sample_id": row["sample_id"],
                "cohort": row["cohort"],
                "status": status,
                "count": len(cloud),
                "sha256": sha(path),
                "invalid_artifact": invalid_artifact,
                "seconds": seconds,
                "peak_cuda_bytes": peak,
            }
        )
        (output / "progress.json").write_text(json.dumps(records, indent=2))
        if len(records) % 20 == 0:
            print("RAY EVAL", job["context"], len(records), status, flush=True)
    result = {
        "status": "bound_ray_observation_evaluation_complete",
        "rows": records,
        "input_manifest_sha256": sha(manifest_path),
        "job_sha256": sha(job_path),
        "job": job,
        "script_sha256": sha(Path(__file__)),
        "portable_profile": PROFILE,
        "portable_script_sha256": sha(
            Path(__file__).parents[2].joinpath("scripts/common/ray_inference.py")
        ),
        "context_helper_sha256": sha(
            Path(__file__).parents[2].joinpath("scripts/train/rgb_context.py")
        ),
        "view_chunks_sha256": sha(
            Path(__file__).parents[2].joinpath("scripts/common/common/ray_view_chunks.py")
        ),
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "gt_uploaded": False,
    }
    (output / "complete.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
