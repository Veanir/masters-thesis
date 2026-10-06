"""Bind 640 source/edit pairs to the screened SAM diagnostic, without running GPU."""

import argparse
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

from scripts.common.paths import CODE_ROOT
from scripts.photo.prepare_edit_inputs import PHOTO, save, sha
from scripts.photo.verify_generated_edits import verify


def prompt(mask):
    assert mask.shape == (480, 640) and mask.dtype == bool and mask.any()
    v, u = np.nonzero(mask)
    h, w = mask.shape
    dx = (u.max() - u.min() + 1) * 0.1
    dy = (v.max() - v.min() + 1) * 0.1
    box = [
        max(0, float(u.min() - dx)),
        max(0, float(v.min() - dy)),
        min(w - 1, float(u.max() + dx)),
        min(h - 1, float(v.max() + dy)),
    ]
    y, x = np.unravel_index(np.argmax(distance_transform_edt(mask)), mask.shape)
    return box, [int(x), int(y)]


def prepare(inputs, photo, photo_job, out):
    integrity = verify(photo, photo_job, inputs / "manifest.json")
    source = json.loads((inputs / "manifest.json").read_text())
    proof = json.loads((inputs / "verification.json").read_text())
    assert (
        proof["manifest_sha256"] == sha(inputs / "manifest.json")
        and proof["decoded_source_arrays_checked"]
    )
    predictions = json.loads((photo / "complete.json").read_text())
    reference = json.loads((PHOTO / "sam-portability-v1/bundle/job.json").read_text())
    pilot = json.loads((PHOTO / "sam2-photo24-reference-job-v3.json").read_text())
    assert reference["original_job_sha256"] == sha(PHOTO / "sam2-photo24-reference-job-v3.json")
    assert (
        reference["screen_thresholds"]
        == pilot["screen_thresholds"]
        == {
            "source_IoU_min": 0.98,
            "edited_IoU_min": 0.98,
            "max_direction_boundary_p95_pixels_at640width": 3.0,
        }
    )
    out.mkdir(exist_ok=False)
    bundle = out / "bundle"
    bundle.mkdir()
    for name in ["images", "targets", "tools"]:
        (bundle / name).mkdir()
    rows = []
    for original, edited in zip(source["rows"], predictions["rows"], strict=False):
        sid = original["sample_id"]
        assert sid == edited["sample_id"]
        mask_path = inputs / sid / "mask.png"
        assert sha(mask_path) == original["files"]["mask.png"]
        mask = np.asarray(Image.open(mask_path)) > 0
        box, point = prompt(mask)
        shutil.copyfile(mask_path, bundle / "targets" / (sid + ".png"))
        for condition in ["source", "reference"]:
            image = (
                inputs / sid / "rgb.png" if condition == "source" else photo / edited["filename"]
            )
            expected = original["files"]["rgb.png"] if condition == "source" else edited["sha256"]
            assert sha(image) == expected
            filename = sid + "-" + condition + ".png"
            shutil.copyfile(image, bundle / "images" / filename)
            rows.append(
                {
                    "sample_id": sid,
                    "condition": condition,
                    "file": "images/" + filename,
                    "sha256": expected,
                    "source_mask_sha256": original["files"]["mask.png"],
                    "box_xyxy": box,
                    "positive_point_xy": point,
                }
            )
    assert len(rows) == len({(r["sample_id"], r["condition"]) for r in rows}) == 1280
    names = ["photo/predict_silhouettes.py", "photo/silhouette_metrics.py"]
    for name in names:
        destination = bundle / "tools/scripts" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(CODE_ROOT / "scripts" / name, destination)
    shutil.copytree(
        CODE_ROOT / "scripts",
        bundle / "tools/scripts",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    job = {
        "purpose": "frozen_final_SAM_silhouette_diagnostic",
        "rows": rows,
        "runner_sha256": sha(bundle / "tools/scripts/photo/predict_silhouettes.py"),
        **{
            k: reference[k]
            for k in [
                "github_commit",
                "weight_url",
                "weight_sha256",
                "weight_bytes",
                "model_config",
                "screen_thresholds",
                "expected_versions",
            ]
        },
        "source_manifest_sha256": sha(inputs / "manifest.json"),
        "photo_manifest_sha256": sha(photo / "complete.json"),
        "pilot_diagnostic_job_sha256": sha(PHOTO / "sam2-photo24-reference-job-v3.json"),
        "postprocessing": False,
        "multimask_output": False,
        "scope": (
            "Auxiliary silhouette diagnostic. Full source+edit images, "
            "identical source-derived box and interior point. No mask prompt, "
            "real data, geometry certification or PHOTO admission."
        ),
    }
    save(bundle / "job.json", job)
    save(out / "photo-integrity.json", integrity)
    bindings = {
        p.relative_to(bundle).as_posix(): sha(p) for p in sorted(bundle.rglob("*")) if p.is_file()
    }
    archive = out / "sam-inputs.zip"
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_STORED) as z:
        for name in bindings:
            z.write(bundle / name, name)
    with zipfile.ZipFile(archive) as z:
        import hashlib

        assert set(z.namelist()) == set(bindings)
        assert all(
            hashlib.sha256(z.read(name)).hexdigest() == digest for name, digest in bindings.items()
        )
    record = {
        "status": "SAM1280_production_inputs_prepared_no_GPU_or_admission",
        "count": 1280,
        "pair_count": 640,
        "job_sha256": sha(bundle / "job.json"),
        "archive_sha256": sha(archive),
        "archive_bytes": archive.stat().st_size,
        "files_sha256": bindings,
        "photo_integrity_sha256": sha(out / "photo-integrity.json"),
        "source_verification_sha256": sha(inputs / "verification.json"),
        "training_ready": False,
        "script_sha256": sha(Path(__file__)),
    }
    save(out / "complete.json", record)
    return record


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--photo", type=Path, required=True)
    p.add_argument("--photo-job", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(
        json.dumps(
            prepare(
                a.inputs.resolve(), a.photo.resolve(), a.photo_job.resolve(), a.output.resolve()
            ),
            indent=2,
        )
    )
