"""Pinned offline Klein edits, with exact source and output byte provenance."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from scripts.common.paths import ROOT as WORKSPACE_ROOT


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for data in iter(lambda: f.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def save(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["download", "generate"])
    a = p.parse_args()
    root = WORKSPACE_ROOT / "runs/photo-pilot"
    job = json.loads((root / "job/job.json").read_text())
    for rel, sha in job["files"].items():
        path = (root / "job" / rel).resolve()
        path.relative_to((root / "job").resolve())
        assert digest(path) == sha
    out, weights = root / "results", root / "models"
    out.mkdir(exist_ok=True)
    model = job["model"]
    tick = time.perf_counter()
    if a.mode == "download":
        from huggingface_hub import snapshot_download

        token = sys.stdin.read().strip()
        assert token.startswith("hf_")
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        try:
            snapshot_download(
                model["repo_id"],
                revision=model["revision"],
                local_dir=weights,
                token=token,
                allow_patterns=model["allow_patterns"],
                max_workers=4,
            )
        except Exception as e:
            save(out / "failure.json", {"stage": "download", "error_type": type(e).__name__})
            raise SystemExit(1) from None
        token = None
        save(
            out / "download.json",
            {
                "model": model,
                "seconds": time.perf_counter() - tick,
                "files": {
                    f.relative_to(weights).as_posix(): digest(f)
                    for f in weights.rglob("*")
                    if f.is_file() and ".cache" not in f.parts
                },
            },
        )
        return
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import numpy as np
    import torch
    from diffusers import Flux2KleinPipeline
    from PIL import Image

    downloaded = json.loads((out / "download.json").read_text())
    assert downloaded["model"] == model
    for rel, sha in downloaded["files"].items():
        assert digest(weights / rel) == sha
    pipe = Flux2KleinPipeline.from_pretrained(
        weights, torch_dtype=torch.bfloat16, local_files_only=True, low_cpu_mem_usage=True
    )
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)
    binding = {
        "job_sha256": digest(root / "job/job.json"),
        "script_sha256": digest(__file__),
        "download_sha256": digest(out / "download.json"),
        "model": model,
        "gpu": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            text=True,
        ),
        "settings": job["settings"],
        "training_ready": False,
    }
    save(out / "binding.json", binding)
    (out / "pip-freeze.txt").write_text(
        subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    )
    rows = []
    for sample in job["samples"]:
        rgb = Image.open(root / "job" / sample["rgb"]).convert("RGB")
        mask = np.asarray(Image.open(root / "job" / sample["mask"])) > 0
        for variant, prompt in enumerate(job["prompts"]):
            seed = job["seed_base"] + 2 * sample["ordinal"] + variant
            prefix = sample["sample_id"] + f"-v{variant}"
            assert not (out / (prefix + ".json")).exists(), "Never rerun completed variants"
            start = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            image = pipe(
                image=rgb,
                prompt=prompt,
                width=640,
                height=480,
                num_inference_steps=job["settings"]["steps"],
                guidance_scale=job["settings"]["guidance_scale"],
                generator=torch.Generator("cpu").manual_seed(seed),
            ).images[0]
            torch.cuda.synchronize()
            pixels = np.asarray(image)
            assert (
                image.size == (640, 480)
                and pixels.dtype == np.uint8
                and pixels.shape == (480, 640, 3)
            )
            assert pixels.std() > 1
            raw, masked = out / (prefix + "-raw.png"), out / (prefix + "-masked.png")
            image.save(raw)
            Image.fromarray(np.where(mask[..., None], pixels, 0).astype(np.uint8)).save(masked)
            row = {
                "sample_id": sample["sample_id"],
                "variant": variant,
                "seed": seed,
                "raw": raw.name,
                "raw_sha256": digest(raw),
                "masked": masked.name,
                "masked_sha256": digest(masked),
                "seconds": time.perf_counter() - start,
                "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
                "geometry_certified": False,
            }
            save(out / (prefix + ".json"), row)
            rows.append(row)
            print(json.dumps(row), flush=True)
    assert len(rows) == job["expected_images"]
    save(
        out / "complete.json",
        {
            "status": "generated_pending_review",
            "rows": rows,
            "binding": binding,
            "training_ready": False,
            "seconds": time.perf_counter() - tick,
        },
    )


if __name__ == "__main__":
    main()
