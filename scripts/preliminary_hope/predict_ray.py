"""Holdout inference gated by all frozen final checkpoints; no GT mount."""

import argparse
import json
import os
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from scripts.common.paths import ROOT
from scripts.common.ray_view_chunks import decoder_view_chunks
from scripts.preliminary_hope.prepare_training_plan import digest, save_json

BASE = Path(os.environ.get("RGBD_MODEL_WORK", str(ROOT)))
VENDOR = BASE / "vendor/rayst3r"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        choices=["pretrained"]
        + [f"{a}-seed{s}" for s in range(3) for a in ("base", "classic", "photo")],
    )
    parser.add_argument("--cohort", required=True, choices=("hope146", "ycbv42", "abo24"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    os.chdir(VENDOR)
    sys.path.insert(0, str(VENDOR))
    from eval_wrapper.eval import EvalWrapper, eval_scene

    freeze_path = Path("/checkpoints/frozen-checkpoints.json")
    frozen = json.loads(freeze_path.read_text(encoding="utf-8-sig"))
    assert frozen["scripts"]["preliminary_hope/predict_ray.py"] == digest(__file__)
    assert frozen["scripts"]["common/ray_view_chunks.py"] == digest(
        Path(__file__).parents[2].joinpath("scripts/common/common/ray_view_chunks.py")
    )
    assert frozen["status"] == "frozen" and len(frozen["runs"]) == 9
    common = frozen["common_training_contract"]
    assert digest(BASE / "weights/rayst3r.pth") == common["base_weights_sha256"]
    assert digest(BASE / "weights/dinov2_vitl14_reg4_pretrain.pth") == common["dino_weights_sha256"]
    for source, expected in common["vendor_sources_sha256"].items():
        assert digest(BASE / source) == expected
    assert {r["run"] for r in frozen["runs"]} == {
        f"{a}-seed{s}" for s in range(3) for a in ("base", "classic", "photo")
    }
    for row in frozen["runs"]:
        directory = Path("/checkpoints") / row["run"]
        assert digest(directory / "complete.json") == row["complete_sha256"]
        assert digest(directory / "final-delta.pt") == row["checkpoint_sha256"]
        complete = json.loads((directory / "complete.json").read_text())
        assert (
            complete["status"] == "complete"
            and complete["step"] == 2160
            and not complete["contract"]["canary"]
        )
    input_path = Path("/inputs/complete.json")
    manifest = json.loads(input_path.read_text())
    count, role = {
        "hope146": (146, "project_heldout_real"),
        "ycbv42": (42, "exposed_real_diagnostic"),
        "abo24": (24, "adaptation_holdout"),
    }[args.cohort]
    assert manifest["status"] == "complete" and len(manifest["rows"]) == count
    assert frozen["cohorts"][args.cohort]["input_manifest_sha256"] == digest(input_path)
    assert frozen["inference"]["view_chunk"] == 3
    assert frozen["inference"]["confidence"] == 5 and frozen["inference"]["total_views"] == 22
    for row in manifest["rows"]:
        assert row["evaluation_role"] == role and not row["gt_exported"]
        for filename, expected in row["export_sha256"].items():
            assert digest(Path("/inputs") / row["sample_id"] / filename) == expected
    binding = {
        "model": args.model,
        "frozen_checkpoints_sha256": digest(freeze_path),
        "input_manifest_sha256": digest(input_path),
        "runner_sha256": digest(__file__),
        "base_weights_sha256": digest(BASE / "weights/rayst3r.pth"),
        "dino_weights_sha256": digest(BASE / "weights/dinov2_vitl14_reg4_pretrain.pth"),
        "threshold": 5,
        "n_pred_views_axis": 5,
        "total_views": 22,
        "view_chunk": 3,
        "cohort": args.cohort,
        "view_chunk_script_sha256": digest(
            Path(__file__).parents[2].joinpath("scripts/common/common/ray_view_chunks.py")
        ),
        "no_gt_access": True,
        "torch": torch.__version__,
        "cost_usd": 0,
    }
    args.output.mkdir(parents=True, exist_ok=args.resume)
    if (args.output / "complete.json").exists():
        raise RuntimeError("Model evaluation already complete")
    if args.resume:
        assert json.loads((args.output / "binding.json").read_text())["contract"] == binding
    else:
        save_json(
            args.output / "binding.json",
            {"started_at": datetime.now(UTC).isoformat(), "contract": binding},
        )
    try:
        torch.manual_seed(20260905)
        np.random.seed(20260905)
        wrapper = EvalWrapper(str(BASE / "weights/rayst3r.pth"))
        if args.model != "pretrained":
            delta_path = Path("/checkpoints") / args.model / "final-delta.pt"
            state = torch.load(delta_path, map_location="cpu", weights_only=False)
            assert state["step"] == 2160 and not state["binding"]["canary"]
            assert state["binding"]["base_weights_sha256"] == binding["base_weights_sha256"]
            assert state["binding"]["dino_weights_sha256"] == binding["dino_weights_sha256"]
            for source, expected in state["binding"]["vendor_sources_sha256"].items():
                assert digest(BASE / source) == expected
            ids = {
                id(p)
                for module in [
                    *list(wrapper.model.decoder_blocks)[-2:],
                    wrapper.model.pts_head,
                    wrapper.model.classifier_head,
                ]
                for p in module.parameters()
            }
            expected_names = {name for name, p in wrapper.model.named_parameters() if id(p) in ids}
            assert set(state["delta"]) == expected_names
            assert all(torch.isfinite(p).all() for p in state["delta"].values())
            incompatible = wrapper.model.load_state_dict(state["delta"], strict=False)
            assert not incompatible.unexpected_keys
            assert (
                set(incompatible.missing_keys) == set(wrapper.model.state_dict()) - expected_names
            )
            del state
        wrapper.eval().requires_grad_(False)
        dino = torch.hub.load(
            str(BASE / "vendor/dinov2"), "dinov2_vitl14_reg", source="local", pretrained=False
        )
        dino.load_state_dict(
            torch.load(
                BASE / "weights/dinov2_vitl14_reg4_pretrain.pth",
                map_location="cpu",
                weights_only=True,
            ),
            strict=True,
        )
        dino.eval().cuda().requires_grad_(False)
        records = []
        for row in manifest["rows"]:
            name = row["sample_id"]
            output_path = args.output / (name + ".npz")
            record_path = output_path.with_suffix(".json")
            if args.resume and record_path.exists():
                record = json.loads(record_path.read_text())
                assert digest(output_path) == record["output_sha256"]
                records.append(record)
                continue
            tick = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode(), decoder_view_chunks(wrapper.model, 3):
                points = eval_scene(
                    wrapper,
                    str(Path("/inputs") / name),
                    dino_model=dino,
                    set_conf=5,
                    n_pred_views=5,
                )
            torch.cuda.synchronize()
            points = points.detach().float().cpu().numpy() * np.asarray(
                row["rayst3r_output_to_project_camera"], np.float32
            )
            assert points.ndim == 2 and points.shape[1] == 3 and np.isfinite(points).all()
            np.savez_compressed(output_path, points_camera_m=points)
            record = {
                "sample_id": name,
                "count": len(points),
                "seconds": time.perf_counter() - tick,
                "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
                "output_sha256": digest(output_path),
            }
            save_json(record_path, record)
            records.append(record)
            save_json(
                args.output / "progress.json",
                {"model": args.model, "completed": len(records), "total": count},
            )
            print(json.dumps(record), flush=True)
        save_json(
            args.output / "complete.json",
            {
                "status": "complete",
                "contract": binding,
                "records": records,
                "ended_at": datetime.now(UTC).isoformat(),
            },
        )
    except Exception:
        save_json(
            args.output / "failure.json", {"traceback": traceback.format_exc(), "contract": binding}
        )
        raise


if __name__ == "__main__":
    main()
