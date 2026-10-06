"""Pinned SAM2 masks with the exact screened prompts and inference settings."""

import argparse
import importlib.metadata
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image


def sha(path):
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8388608), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--model-root", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--job", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    bundle = a.bundle.resolve()
    root = a.model_root.resolve()
    job_path = a.job.resolve()
    out = a.output.resolve()
    job = json.loads(job_path.read_text())
    assert job["purpose"] in [
        "SAM_runner_portability_canary",
        "frozen_final_SAM_silhouette_diagnostic",
    ]
    assert sha(Path(__file__)) == job["runner_sha256"] and sha(a.weights) == job["weight_sha256"]
    assert (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        == job["github_commit"]
    )
    subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "sam2"], cwd=root, check=True)
    versions = {name: importlib.metadata.version(name) for name in job["expected_versions"]}
    assert versions == job["expected_versions"]
    assert len({(r["sample_id"], r["condition"]) for r in job["rows"]}) == len(job["rows"])
    assert len(job["rows"]) == (96 if job["purpose"] == "SAM_runner_portability_canary" else 1280)
    sys.path.insert(0, str(root))
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    out.mkdir(exist_ok=False)
    model = build_sam2(
        job["model_config"], str(a.weights.resolve()), device="cuda", apply_postprocessing=False
    )
    predictor = SAM2ImagePredictor(
        model, mask_threshold=0.0, max_hole_area=0.0, max_sprinkle_area=0.0
    )
    rows = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for row in job["rows"]:
            path = (bundle / row["file"]).resolve()
            assert path.is_relative_to(bundle) and sha(path) == row["sha256"]
            rgb = np.asarray(Image.open(path).convert("RGB"))
            assert rgb.shape == (480, 640, 3)
            torch.cuda.synchronize()
            tick = time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            predictor.set_image(rgb)
            masks, scores, _ = predictor.predict(
                point_coords=np.array([row["positive_point_xy"]]),
                point_labels=np.array([1]),
                box=np.array(row["box_xyxy"]),
                multimask_output=False,
            )
            torch.cuda.synchronize()
            assert masks.shape == (1, 480, 640)
            filename = row["sample_id"] + "-" + row["condition"] + "-sam2.png"
            Image.fromarray(masks[0].astype(np.uint8) * 255).save(out / filename)
            rows.append(
                {
                    **row,
                    "filename": filename,
                    "output_sha256": sha(out / filename),
                    "predicted_scores": scores.tolist(),
                    "seconds": time.monotonic() - tick,
                    "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
                }
            )
            (out / "progress.json").write_text(json.dumps(rows, indent=2))
            if len(rows) % 16 == 0:
                print("BOUND SAM", len(rows), "of", len(job["rows"]), flush=True)
    (out / "complete.json").write_text(
        json.dumps(
            {
                "status": "bound_SAM_masks_complete_require_local_QA",
                "rows": rows,
                "job_sha256": sha(job_path),
                "script_sha256": sha(Path(__file__)),
                "versions": versions,
                "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(),
                "source_files_sha256": {
                    p.relative_to(root).as_posix(): sha(p)
                    for p in sorted((root / "sam2").rglob("*.py"))
                },
                "precision": (
                    "Screened BF16, no compilation or CUDA connected-component "
                    "postprocessing; one mask, source box plus interior point."
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
