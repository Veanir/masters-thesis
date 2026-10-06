"""Paired final-checkpoint adaptation; mounted inputs contain TRAIN only."""

import argparse
import json
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from scripts.common.ray_training import BASE, load_models, loss_for, make_batch
from scripts.preliminary_hope.prepare_training_plan import classic_rgb, digest, save_json, schedule


def now():
    return datetime.now(UTC).isoformat()


def save_checkpoint(path, model, optimizer, step, binding, final=False):
    # The official base checkpoint plus this exact named subset reconstructs the model.
    delta = {name: p.detach().cpu() for name, p in model.named_parameters() if p.requires_grad}
    state = {
        "delta": delta,
        "step": step,
        "binding": binding,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
    }
    if not final:
        state["optimizer"] = optimizer.state_dict()
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["BASE", "CLASSIC", "PHOTO"], required=True)
    parser.add_argument("--seed", type=int, choices=[0, 1, 2], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training", type=Path, default=BASE / "training")
    parser.add_argument("--photo", type=Path, default=BASE / "photo")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--canary", action="store_true", help="Three optimizer steps, discarded; never a final run"
    )
    args = parser.parse_args()
    manifest_path = args.training / "complete.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "complete" and len(manifest["rows"]) == 36
    assert all(r["split"] == "train" for r in manifest["rows"])
    assert manifest["novel_views"] == 21
    data = []
    for row in manifest["rows"]:
        path = args.training / (row["sample_id"] + ".npz")
        assert digest(path) == row["output_sha256"], path
        with np.load(path, allow_pickle=False) as loaded:
            data.append({key: loaded[key] for key in loaded.files})
    plan = schedule(args.seed)
    photo_manifest = None
    if args.arm == "PHOTO":
        from scripts.preliminary_hope.training_data import load_photo_images

        photo_manifest, photo_images = load_photo_images(args.photo, manifest_path, manifest, data)
    binding = {
        "arm": args.arm,
        "seed": args.seed,
        "steps": 3 if args.canary else 2160,
        "canary": args.canary,
        "optimizer": "AdamW",
        "lr": 1e-5,
        "weight_decay": 1e-4,
        "clip_norm": 1,
        "precision": "bf16 autocast, fp32 parameters",
        "optimizer_updates_per_step": 1,
        "training_sha256": digest(manifest_path),
        "runner_sha256": digest(__file__),
        "plan_script_sha256": digest(
            Path(__file__).parents[2].joinpath("scripts/preliminary_hope/prepare_training_plan.py")
        ),
        "helper_sha256": digest(
            Path(__file__).parents[2].joinpath("scripts/common/ray_training.py")
        ),
        "base_weights_sha256": digest(BASE / "weights/rayst3r.pth"),
        "dino_weights_sha256": digest(BASE / "weights/dinov2_vitl14_reg4_pretrain.pth"),
        "vendor_sources_sha256": {
            str(p.relative_to(BASE)): digest(p)
            for vendor in ("rayst3r", "dinov2")
            for p in sorted((BASE / "vendor" / vendor).rglob("*.py"))
        },
        "photo_sha256": digest(args.photo / "complete.json") if photo_manifest else None,
        "photo_validator_sha256": digest(
            Path(__file__).parents[2].joinpath("scripts/preliminary_hope/training_data.py")
        )
        if photo_manifest
        else None,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "cost_usd": 0,
        "holdout_access": False,
    }
    args.output.mkdir(parents=True, exist_ok=args.resume)
    if (args.output / "complete.json").exists():
        raise RuntimeError("Run already complete; refusing restart")
    if args.resume:
        previous = json.loads((args.output / "binding.json").read_text())
        assert previous["contract"] == binding, "Resume contract changed"
    else:
        save_json(args.output / "binding.json", {"started_at": now(), "contract": binding})
        save_json(args.output / "schedule.json", plan)
    torch.manual_seed(args.seed)
    model, dino, trainable = load_models()
    save_json(args.output / "trainable-parameters.json", trainable)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1e-5, weight_decay=1e-4, foreach=False)
    start_step = 0
    if args.resume:
        state = torch.load(args.output / "resume.pt", map_location="cpu", weights_only=False)
        assert state["binding"] == binding and set(state["delta"]) == set(trainable)
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name in state["delta"]:
                    parameter.copy_(state["delta"][name])
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        start_step = state["step"]
        del state
    start = time.perf_counter()
    durations = []
    try:
        with (args.output / "steps.jsonl").open("a", buffering=1) as log:
            for exposure in plan[start_step : binding["steps"]]:
                torch.cuda.synchronize()
                tick = time.perf_counter()
                item = data[exposure["ordinal"]]
                rgb = item["rgb"]
                if exposure["augment"]:
                    if args.arm == "CLASSIC":
                        rgb = classic_rgb(rgb, item["input_mask"], exposure["augmentation_seed"])
                    elif args.arm == "PHOTO":
                        rgb = photo_images[
                            manifest["rows"][exposure["ordinal"]]["sample_id"], exposure["variant"]
                        ]
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.reset_peak_memory_stats()
                loss = loss_for(model, dino, make_batch(item, exposure["views"], rgb))
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Nonfinite loss at {exposure['step']}")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(parameters, 1, error_if_nonfinite=True)
                if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
                    raise RuntimeError("Frozen RaySt3R parameter received gradients")
                if any(p.grad is not None for p in dino.parameters()):
                    raise RuntimeError("Frozen DINO received gradients")
                optimizer.step()
                torch.cuda.synchronize()
                seconds = time.perf_counter() - tick
                durations.append(seconds)
                step = exposure["step"]
                record = {
                    **exposure,
                    "loss": float(loss.detach()),
                    "grad_norm": float(norm),
                    "seconds": seconds,
                    "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
                    "at": now(),
                    "attempt_start_step": start_step,
                }
                log.write(json.dumps(record) + "\n")
                if step % 36 == 0 or args.canary:
                    save_json(
                        args.output / "progress.json",
                        {
                            **record,
                            "arm": args.arm,
                            "seed": args.seed,
                            "total_steps": binding["steps"],
                            "elapsed_this_attempt": time.perf_counter() - start,
                        },
                    )
                    print(json.dumps(record), flush=True)
                if step in (720, 1440) and not args.canary:
                    save_checkpoint(args.output / "resume.pt", model, optimizer, step, binding)
                    save_json(
                        args.output / f"checkpoint-{step}.json",
                        {"step": step, "sha256": digest(args.output / "resume.pt")},
                    )
        final = None
        if not args.canary:
            final = args.output / "final-delta.pt"
            save_checkpoint(final, model, optimizer, binding["steps"], binding, final=True)
        save_json(
            args.output / "complete.json",
            {
                "status": "complete",
                "contract": binding,
                "ended_at": now(),
                "step": binding["steps"],
                "seconds_this_attempt": time.perf_counter() - start,
                "median_step_seconds": float(np.median(durations)),
                "checkpoint_sha256": digest(final) if final else None,
                "checkpoint": final.name if final else None,
                "canary_weights_discarded": args.canary,
            },
        )
    except Exception:
        save_json(
            args.output / "failure.json",
            {"at": now(), "traceback": traceback.format_exc(), "contract": binding},
        )
        raise


if __name__ == "__main__":
    main()
