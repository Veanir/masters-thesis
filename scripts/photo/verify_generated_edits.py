"""Verify full downloaded image records; this is integrity, not visual admission."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.photo.image_records import sha


def verify(result, job_path, input_manifest):
    def read(p):
        return json.loads(p.read_text())

    job = read(job_path)
    manifest = read(input_manifest)
    complete = read(result / "complete.json")
    assert sha(input_manifest) == job["input_manifest_sha256"]
    assert complete["status"] == "all640_PHOTO_images_generated_require_admission"
    assert complete["job_sha256"] == sha(job_path) and complete["input_manifest_sha256"] == sha(
        input_manifest
    )
    assert (
        complete["input_plan_sha256"] == job["input_plan_sha256"]
        and complete["recipe_sha256"] == job["recipe_sha256"]
    )
    assert (
        complete["runner_sha256"] == job["runner_sha256"]
        and complete["records_helper_sha256"] == job["records_helper_sha256"]
    )
    assert complete["training_ready"] is False and complete["author_review_pending"] is True
    assert (
        complete["author_control_sha256"] is None
        and complete["generation_purpose"] == job["purpose"]
    )
    assert (
        complete["versions"] == job["expected_versions"]
        and complete["model_binding_sha256"] == job["model_binding_sha256"]
    )
    rows = complete["rows"]
    assert len(rows) == len({r["sample_id"] for r in rows}) == 640
    assert (
        [r["sample_id"] for r in rows]
        == job["sample_ids"]
        == [r["sample_id"] for r in manifest["rows"]]
    )
    reused = 0
    for row, source in zip(rows, manifest["rows"], strict=False):
        sid = row["sample_id"]
        filename = sid + "-reference.png"
        assert row["filename"] == filename and (result / filename).resolve().is_relative_to(
            result.resolve()
        )
        assert row == read(result / (sid + ".json")) and sha(result / filename) == row["sha256"]
        assert (
            row["condition"] == "reference"
            and row["status"] == "unreviewed"
            and row["masked_or_composited"] is False
        )
        assert row["job_sha256"] == sha(job_path) and row["seed"] == source["seed"]
        assert (
            row["source_rgb_array_sha256"] == source["source_rgb_array_sha256"]
            and row["input_rgb_sha256"] == source["files"]["rgb.png"]
        )
        with Image.open(result / filename) as image:
            assert image.mode == "RGB" and image.size == (640, 480)
            pixels = np.asarray(image)
            assert pixels.dtype == np.uint8 and pixels.std() > 1
        expected = source["reuse_pilot_output"]
        assert type(row["reused_pilot"]) is bool and row["reused_pilot"] == (expected is not None)
        if expected is not None:
            assert row["sha256"] == expected["sha256"] == job["pilot_reuse"][sid]["sha256"]
            reused += 1
    return {
        "status": "all640_PHOTO_downloaded_images_integrity_verified_not_admitted",
        "count": 640,
        "reused_pilot_count": reused,
        "manifest_sha256": sha(result / "complete.json"),
        "job_sha256": sha(job_path),
        "input_manifest_sha256": sha(input_manifest),
        "all_passed": True,
        "visual_review_complete": False,
        "training_ready": False,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--result", type=Path, required=True)
    p.add_argument("--job", type=Path, required=True)
    p.add_argument("--input-manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = verify(a.result, a.job, a.input_manifest)
    with a.output.open("x") as f:
        json.dump(result, f, indent=2)
