"""Fixed paired BASE/CLASSIC/PHOTO adaptation with recoverable optimizer/RNG."""

import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--training", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--context", choices=["masked", "full"], required=True)
    p.add_argument("--arm", choices=["BASE", "CLASSIC", "PHOTO"], required=True)
    p.add_argument("--photo", type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    work = a.work.resolve()
    source = a.training.resolve()
    out = a.output.resolve()
    planpath = a.plan.resolve()
    job = json.loads(planpath.read_text())
    assert job["purpose"] == "paired_evolution_matrix_frozen"
    assert job["context"] == a.context and job["seed"] in [0, 1, 2] and job["epochs"] == 4
    assert len(job["sample_ids"]) == 6400 and job["steps"] == 25600
    photo = a.photo.resolve()
    photo_manifest = photo / "complete.json"
    assert sha(photo_manifest) == job["photo_admission_manifest_sha256"]
    admission = json.loads(photo_manifest.read_text())
    assert admission["status"] == "frozen_PHOTO_admission"
    photo_rows = {r["sample_id"]: r for r in admission["rows"]}
    assert len(photo_rows) == len(admission["rows"]) and set(photo_rows) == set(job["sample_ids"])
    accepted = {sid for sid, row in photo_rows.items() if row["accepted"]}
    from PIL import Image

    import scripts.train.training_io as recovery
    from scripts.train.augmentation_schedule import classic_rgb
    from scripts.train.augmentation_schedule import schedule as matched_schedule

    assert job["decoder_blocks"] == 2 and job["lr"] == 1e-5 and job["checkpoint_interval"] == 3200
    assert job["training_manifest_sha256"] == sha(source / "complete.json")
    manifest = json.loads((source / "complete.json").read_text())
    assert manifest["status"] == "certified_synthetic_ray_labels_complete"
    lookup = {r["sample_id"]: r for r in manifest["rows"]}
    selected = [lookup[s] for s in job["sample_ids"]]
    assert len(selected) == len({r["sample_id"] for r in selected}) and all(
        r["split"] == "train" for r in selected
    )
    if a.resume:
        assert (
            out.is_dir()
            and (out / "recovery.pt").is_file()
            and not (out / "complete.json").exists()
        )
    else:
        assert not out.exists()
        out.mkdir(parents=True)
    vendor = work / "vendor/rayst3r"
    sys.path.insert(0, str(vendor))
    os.chdir(vendor)
    from engine import eval_model
    from eval_wrapper.eval import EvalWrapper

    from scripts.train.rgb_context import install, set_context

    install()
    set_context(a.context)
    torch.manual_seed(job["seed"])
    np.random.seed(job["seed"])
    random.seed(job["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = EvalWrapper(str(work / "weights/rayst3r.pth")).model
    model.eval().requires_grad_(False)
    for block in list(model.decoder_blocks)[-job["decoder_blocks"] :]:
        block.train().requires_grad_(True)
    model.pts_head.train().requires_grad_(True)
    model.classifier_head.train().requires_grad_(True)
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
    trainable = {n: p.numel() for n, p in model.named_parameters() if p.requires_grad}
    parameters = [p for p in model.parameters() if p.requires_grad]
    assert 0 < sum(trainable.values()) < sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=job["lr"], weight_decay=1e-4, foreach=False)
    binding = {
        "job_sha256": sha(planpath),
        "context": a.context,
        "arm": a.arm,
        "script_sha256": sha(Path(__file__)),
        "training_design_sha256": sha(
            Path(__file__).parents[2].joinpath("scripts/train/augmentation_schedule.py")
        ),
        "recovery_helper_sha256": sha(Path(recovery.__file__)),
        "photo_admission_manifest_sha256": sha(photo_manifest),
        "context_helper_sha256": sha(
            Path(__file__).parents[2].joinpath("scripts/train/rgb_context.py")
        ),
        "training_manifest_sha256": sha(source / "complete.json"),
        "weights": {
            n: sha(work / "weights" / n) for n in ["rayst3r.pth", "dinov2_vitl14_reg4_pretrain.pth"]
        },
        "vendor_sources": {
            str(p.relative_to(work)): sha(p)
            for name in ["rayst3r", "dinov2"]
            for p in sorted((work / "vendor" / name).rglob("*.py"))
        },
        "trainable": trainable,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "precision": (
            "Official engine BF16 model forward, FP32 weights/DINO; TF32 "
            "disabled. Final inference uses the separate portable FP32 "
            "profile."
        ),
        "determinism": {
            "algorithms": True,
            "cublas_workspace": ":4096:8",
            "cudnn_benchmark": False,
        },
        "checkpoint_selection": "Last fixed update only; no early stopping or test access.",
    }
    if a.resume:
        assert json.loads((out / "binding.json").read_text()) == binding
    else:
        (out / "binding.json").write_text(json.dumps(binding, indent=2))
    cache = OrderedDict()
    verified = set()
    verified_photo = set()

    def load(row):
        sid = row["sample_id"]
        if sid not in cache:
            path = source / (sid + ".npz")
            if sid not in verified:
                assert sha(path) == row["output_sha256"]
                verified.add(sid)
            with np.load(path) as z:
                cache[sid] = {k: z[k] for k in z.files}
            while len(cache) > 4:
                cache.popitem(last=False)
        cache.move_to_end(sid)
        return cache[sid]

    schedule = matched_schedule(job["sample_ids"], accepted, job["seed"], epochs=job["epochs"])
    assert len(schedule) == job["steps"]
    if a.resume:
        assert json.loads((out / "schedule.json").read_text()) == schedule
    else:
        (out / "schedule.json").write_text(json.dumps(schedule, indent=2))
    schedule_sha = sha(out / "schedule.json")
    log_path = out / "steps.jsonl"
    resume_step = (
        recovery.restore(out / "recovery.pt", model, optimizer, binding, schedule_sha, log_path)
        if a.resume
        else 0
    )

    def rgb_for(item, exposure):
        original = item["rgb"]
        sid = exposure["sample_id"]
        if a.arm == "BASE" or not exposure["effective_augmentation"]:
            return original, False
        if a.arm == "CLASSIC":
            rgb = classic_rgb(original, exposure["augmentation_seed"])
        else:
            row = photo_rows[sid]
            path = (photo / row["filename"]).resolve()
            assert path.is_relative_to(photo)
            if sid not in verified_photo:
                assert sha(path) == row["sha256"]
                verified_photo.add(sid)
            assert row["source_rgb_array_sha256"] == hashlib.sha256(original.tobytes()).hexdigest()
            rgb = np.asarray(Image.open(path).convert("RGB"))
        assert rgb.shape == original.shape and rgb.dtype == original.dtype
        return rgb, not np.array_equal(rgb, original)

    started = time.monotonic()
    times = []
    with log_path.open("a" if a.resume else "w", buffering=1) as log:
        for exposure in schedule[resume_step:]:
            tick = time.monotonic()
            item = load(lookup[exposure["sample_id"]])
            views = exposure["views"]
            k = torch.from_numpy(item["K"].copy()).float()
            mask = torch.from_numpy(item["input_mask"].copy()).bool()
            rgb, actual_change = rgb_for(item, exposure)
            batch = {
                "input_cams": {
                    "c2ws": torch.eye(4)[None, None],
                    "Ks": k[None, None],
                    "depths": torch.from_numpy(item["input_depth_uint16"].astype(np.float32))[
                        None, None
                    ],
                    "valid_masks": mask[None, None],
                    "original_valid_masks": mask[None, None].clone(),
                    "imgs": torch.from_numpy(rgb.copy())[None, None],
                },
                "new_cams": {
                    "c2ws": torch.from_numpy(item["novel_c2ws"][views].copy())[None],
                    "Ks": k[None, None].repeat(1, 2, 1, 1),
                    "depths": torch.from_numpy(
                        item["novel_depth_uint16"][views].astype(np.float32)
                    )[None],
                    "valid_masks": torch.from_numpy(item["novel_masks"][views].copy())[None].bool(),
                },
            }
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats()
            loss = eval_model(model, batch, mode="loss", dino_model=dino)["loss"]
            assert torch.isfinite(loss), "Nonfinite loss"
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1, error_if_nonfinite=True)
            assert not any(p.grad is not None for p in model.parameters() if not p.requires_grad)
            assert not any(p.grad is not None for p in dino.parameters())
            optimizer.step()
            torch.cuda.synchronize()
            duration = time.monotonic() - tick
            times.append(duration)
            record = {
                **exposure,
                "arm": a.arm,
                "actual_RGB_changed": actual_change,
                "loss": float(loss.detach()),
                "grad_norm": float(norm),
                "seconds": duration,
                "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
            }
            log.write(json.dumps(record) + "\n")
            if exposure["step"] % 320 == 0:
                (out / "progress.json").write_text(json.dumps(record))
                print("MATRIX", a.arm, a.context, record, flush=True)
            if exposure["step"] % job["checkpoint_interval"] == 0 or exposure["step"] == len(
                schedule
            ):
                log.flush()
                recovery.save(
                    out / "recovery.pt",
                    model,
                    optimizer,
                    exposure["step"],
                    binding,
                    schedule_sha,
                    log_path,
                )
    delta = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
    path = out / "final-delta.pt"
    torch.save({"delta": delta, "step": len(schedule), "binding": binding}, path)
    (out / "complete.json").write_text(
        json.dumps(
            {
                "status": "paired_evolution_matrix_training_complete",
                "binding": binding,
                "steps": len(schedule),
                "resumed_from_step": resume_step,
                "seconds_current_process": time.monotonic() - started,
                "median_step_seconds_current_process": float(np.median(times)) if times else None,
                "checkpoint": path.name,
                "checkpoint_sha256": sha(path),
                "schedule_sha256": sha(out / "schedule.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
