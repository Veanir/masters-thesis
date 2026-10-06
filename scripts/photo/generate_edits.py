"""Bounded final PHOTO batch, pinned pilot recipe and durable per-image recovery."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import time
from pathlib import Path

from scripts.photo.image_records import atomic, commit, load, sha


def safe(root, name):
    path = (root / name).resolve()
    assert path.is_relative_to(root)
    return path


def validate_review_state(job, recipe, author_path):
    recipe_sha = hashlib.sha256(
        json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if job["purpose"] == "final_PHOTO_generation_pending_author_review":
        assert author_path is None and job.get("author_control_sha256") is None
        assert job["recipe_sha256"] == recipe_sha
        assert job["training_ready"] is False and job["author_review_pending"] is True
    elif job["purpose"] == "frozen_final_PHOTO_generation":
        assert author_path is not None and sha(author_path) == job["author_control_sha256"]
        author = json.loads(author_path.read_text())
        assert author["status"] == "author_PHOTO_control_recorded_and_recipe_confirmed"
        assert author["confirmed_recipe_sha256"] == recipe_sha
        assert (
            author["author_response_recorded"] is True
            and len(author["fixed_sample_assessments"]) == 4
        )
    else:
        assert job["purpose"] == "final_PHOTO_runner_development_canary"
    return recipe_sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--job", type=Path, required=True)
    p.add_argument("--input-plan", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--author-control", type=Path)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    root = a.root.resolve()
    inputs = a.inputs.resolve()
    out = a.output.resolve()
    job_path = a.job.resolve()
    job = json.loads(job_path.read_text())
    assert job["purpose"] in [
        "frozen_final_PHOTO_generation",
        "final_PHOTO_generation_pending_author_review",
        "final_PHOTO_runner_development_canary",
    ]
    assert job["runner_sha256"] == sha(Path(__file__)) and job["records_helper_sha256"] == sha(
        Path(__file__).parents[2].joinpath("scripts/photo/image_records.py")
    )
    assert sha(a.input_plan) == job["input_plan_sha256"]
    plan = json.loads(a.input_plan.read_text())
    recipe = plan["recipe"]
    assert (
        recipe["model"] == "black-forest-labs/FLUX.2-klein-4B"
        and recipe["revision"] == "e7b7dc27f91deacad38e78976d1f2b499d76a294"
    )
    assert (
        recipe["resolution"] == [640, 480]
        and recipe["num_inference_steps"] == 4
        and recipe["guidance_scale"] == 1.0
    )
    recipe_sha = validate_review_state(job, recipe, a.author_control)
    manifest_path = inputs / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert sha(manifest_path) == job["input_manifest_sha256"]
    assert (
        manifest["plan_sha256"] == sha(a.input_plan)
        and manifest["status"] == "PHOTO_source_inputs_prepared_no_generation_or_admission"
    )
    rows = manifest["rows"]
    assert (
        len(rows) == 640
        and len({r["sample_id"] for r in rows}) == 640
        and len({r["seed"] for r in rows}) == 640
    )
    if job["purpose"] == "final_PHOTO_runner_development_canary":
        assert len(job["sample_ids"]) == len(set(job["sample_ids"])) == 3
        lookup = {r["sample_id"]: r for r in rows}
        rows = [lookup[sid] for sid in job["sample_ids"]]
        assert all(r["reuse_pilot_output"] is not None for r in rows), (
            "Canary must reproduce three already generated TRAIN observations"
        )
    else:
        assert [r["sample_id"] for r in rows] == job["sample_ids"]
    for row in rows:
        folder = safe(inputs, row["sample_id"])
        assert set(row["files"]) == {"rgb.png", "mask.png", "depth-guide.png"}
        assert all(sha(folder / name) == digest for name, digest in row["files"].items())
    binding = {
        "job_sha256": sha(job_path),
        "input_manifest_sha256": sha(manifest_path),
        "input_plan_sha256": sha(a.input_plan),
        "runner_sha256": sha(Path(__file__)),
        "records_helper_sha256": sha(
            Path(__file__).parents[2].joinpath("scripts/photo/image_records.py")
        ),
    }
    if a.resume:
        assert out.exists() and json.loads((out / "binding.json").read_text()) == binding
    else:
        out.mkdir(exist_ok=False)
        atomic(out / "binding.json", json.dumps(binding, indent=2).encode())
    records = []
    pending = []
    for row in rows:
        expected = {
            "sample_id": row["sample_id"],
            "condition": "reference",
            "seed": row["seed"],
            "input_rgb_sha256": row["files"]["rgb.png"],
            "source_rgb_array_sha256": row["source_rgb_array_sha256"],
            "job_sha256": sha(job_path),
            "masked_or_composited": False,
            "status": "unreviewed",
        }
        record = load(out, row["sample_id"], expected)
        if record is None:
            pending.append((row, expected))
        else:
            records.append(record)
    # The caller sets an optional wall deadline. This runner refuses a new
    # image after the predeclared batch deadline and never changes it on resume.
    assert (
        time.time() < job.get("deadline_unix", float("inf"))
        and shutil.disk_usage(out).free > 4 * 1024**3
    )
    model_binding_path = safe(root, job["model_binding_file"])
    assert sha(model_binding_path) == job["model_binding_sha256"]
    model_binding = json.loads(model_binding_path.read_text())
    assert (
        model_binding["model"] == recipe["model"]
        and model_binding["revision"] == recipe["revision"]
    )
    model_dir = safe(root, job["model_directory"])
    assert all(
        sha(safe(model_dir, name)) == digest for name, digest in model_binding["files"].items()
    )
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import numpy as np
    import torch
    from diffusers import Flux2KleinPipeline
    from PIL import Image

    pipe = None
    prompt = recipe["prompt"] + recipe["depth_suffix"]
    versions = {
        name: importlib.metadata.version(name)
        for name in ["torch", "diffusers", "transformers", "accelerate", "huggingface-hub"]
    }
    assert versions == job["expected_versions"], (
        "Use the exact pilot environment; do not silently replace dependencies"
    )
    runtime = {
        "gpu": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "versions": versions,
        "cuda": torch.version.cuda,
    }
    runtime_path = out / "runtime.json"
    if runtime_path.exists():
        assert json.loads(runtime_path.read_text()) == runtime, (
            "Do not mix generation environments within a resumed batch"
        )
    else:
        atomic(runtime_path, json.dumps(runtime, indent=2).encode())
    for row, expected in pending:
        assert time.time() + 30 < job.get("deadline_unix", float("inf")), (
            "Batch deadline approaching; completed images retained for explicit resume"
        )
        reuse = (
            row["reuse_pilot_output"]
            if job["purpose"] != "final_PHOTO_runner_development_canary"
            else None
        )
        sid = row["sample_id"]
        if reuse is not None:
            ref = job["pilot_reuse"][sid]
            path = safe(root, ref["file"])
            assert (
                sha(path) == ref["sha256"] == reuse["sha256"] and expected["seed"] == reuse["seed"]
            )
            # Copy source PNG bytes exactly, including its original encoding.
            target = out / (sid + "-reference.png")
            temporary = target.with_name(target.name + ".pending")
            assert not temporary.exists()
            raw = path.read_bytes()
            with temporary.open("xb") as f:
                f.write(raw)
                f.flush()
                os.fsync(f.fileno())
            record = {
                **expected,
                "filename": target.name,
                "sha256": ref["sha256"],
                "seconds": 0.0,
                "peak_cuda_bytes": 0,
                "reused_pilot": True,
            }
            atomic(out / (sid + ".json"), json.dumps(record, indent=2).encode())
            os.replace(temporary, target)
        else:
            if pipe is None:
                pipe = Flux2KleinPipeline.from_pretrained(
                    model_dir, torch_dtype=torch.bfloat16, local_files_only=True
                )
                pipe.to("cuda")
                pipe.set_progress_bar_config(disable=True)
            folder = inputs / sid
            rgb = Image.open(folder / "rgb.png").convert("RGB")
            depth = Image.open(folder / "depth-guide.png").convert("RGB")
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            tick = time.monotonic()
            output = pipe(
                image=[rgb, depth],
                prompt=prompt,
                width=640,
                height=480,
                num_inference_steps=4,
                guidance_scale=1.0,
                generator=torch.Generator(device="cuda").manual_seed(expected["seed"]),
            ).images[0]
            torch.cuda.synchronize()
            seconds = time.monotonic() - tick
            array = np.asarray(output)
            assert array.dtype == np.uint8 and array.std() > 1 and output.size == (640, 480)
            record = commit(
                out,
                sid,
                output,
                expected,
                {
                    "seconds": seconds,
                    "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
                    "reused_pilot": False,
                },
            )
        records.append(record)
        if len(records) % 20 == 0:
            print("FINAL PHOTO", len(records), "of", len(rows), "; images unreviewed", flush=True)
    by_sid = {r["sample_id"]: r for r in records}
    assert set(by_sid) == set(job["sample_ids"])
    ordered = [by_sid[sid] for sid in job["sample_ids"]]
    complete = {
        "status": "all640_PHOTO_images_generated_require_admission"
        if len(rows) == 640
        else "three_PHOTO_runner_canary_images_require_parity_check",
        "rows": ordered,
        **binding,
        "model": recipe["model"],
        "revision": recipe["revision"],
        "model_binding_sha256": sha(model_binding_path),
        "versions": versions,
        "gpu": torch.cuda.get_device_name(),
        "precision": "BF16, no quantization or CPU offload",
        "training_ready": False,
        "generation_complete": True,
        "author_control_sha256": job.get("author_control_sha256"),
        "recipe_sha256": recipe_sha,
        "generation_purpose": job["purpose"],
        "author_review_pending": job["purpose"] == "final_PHOTO_generation_pending_author_review",
    }
    final = out / "complete.json"
    if final.exists():
        assert json.loads(final.read_text()) == complete
    else:
        atomic(final, json.dumps(complete, indent=2).encode())
    print("PHOTO BATCH", len(rows), "COMPLETE; QA and admission still required", flush=True)


if __name__ == "__main__":
    main()
