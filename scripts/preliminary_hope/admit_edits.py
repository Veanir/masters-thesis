"""Materialize all reviewed variants, preserving every TRAIN identity and its labels."""

import argparse
import hashlib
import io
import zipfile
from collections import Counter

import numpy as np
from PIL import Image

from scripts.common.paths import ROOT, digest, read, save
from scripts.preliminary_hope.training_data import load_photo_images
from scripts.preliminary_hope.verify_generated_edits import verify


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=__import__("pathlib").Path, required=True)
    parser.add_argument("--decisions", type=__import__("pathlib").Path, required=True)
    parser.add_argument("--output", type=__import__("pathlib").Path, required=True)
    a = parser.parse_args()
    assert not a.output.exists(), "Never overwrite reviewed training data"
    assert verify(a.candidates / "results.zip", a.candidates / "job/job.json") == read(
        a.candidates / "verified-download.json"
    )
    job = read(a.candidates / "job/job.json")
    decisions = read(a.decisions)
    assert (
        decisions["status"] == "all_candidates_reviewed"
        and decisions["test_outcomes_used"] is False
    )
    expected = {(r["sample_id"], v) for r in job["samples"] for v in (0, 1)}
    mapped = {(r["sample_id"], r["variant"]): r for r in decisions["rows"]}
    assert len(mapped) == len(decisions["rows"]) == 72 and set(mapped) == expected
    train = ROOT / "runs/ray-adaptation-training-data-v5-20260905"
    training = read(train / "complete.json")
    assert digest(train / "complete.json") == job["training_sha256"]
    observations = []
    source = {}
    for row in training["rows"]:
        path = train / (row["sample_id"] + ".npz")
        assert digest(path) == row["output_sha256"]
        with np.load(path, allow_pickle=False) as z:
            data = {"rgb": z["rgb"], "input_mask": z["input_mask"]}
        observations.append(data)
        source[row["sample_id"]] = data
    import json

    final_output = a.output
    a.output = final_output.with_name(final_output.name + ".pending-validation")
    assert not a.output.exists(), "Inspect any earlier failed staging directory first"
    a.output.mkdir()
    rows, review_rows = [], []
    with zipfile.ZipFile(a.candidates / "results.zip") as z:
        for generated in json.loads(z.read("complete.json"))["rows"]:
            sid, variant = generated["sample_id"], generated["variant"]
            decision = mapped[sid, variant]
            assert decision["raw_sha256"] == generated["raw_sha256"]
            assert (
                decision["appearance_reviewed"]
                and decision["label_compatibility_reviewed"]
                and decision["reason"]
            )
            assert decision["decision"] in ("accepted", "fallback_original")
            pixels = z.read(generated["raw"])
            assert hashlib.sha256(pixels).hexdigest() == generated["raw_sha256"]
            raw = np.array(Image.open(io.BytesIO(pixels)).convert("RGB"))
            mask = source[sid]["input_mask"].astype(bool)
            if decision["decision"] == "accepted":
                assert decision["label_compatibility"] == "pass_visual_review"
                rgb = np.where(mask[..., None], raw, 0).astype(np.uint8)
            else:
                rgb = np.where(mask[..., None], source[sid]["rgb"], 0).astype(np.uint8)
            filename = f"{sid}-v{variant}.png"
            Image.fromarray(rgb).save(a.output / filename)
            sha = digest(a.output / filename)
            rows.append(
                {
                    "sample_id": sid,
                    "variant": variant,
                    "filename": filename,
                    "sha256": sha,
                    "decision": decision["decision"],
                    "source_raw_sha256": generated["raw_sha256"],
                }
            )
            review_rows.append({**decision, "training_rgb_sha256": sha})
    review = {
        **decisions,
        "status": "final_review",
        "rows": review_rows,
        "candidate_job_sha256": digest(a.candidates / "job/job.json"),
        "decision_source_sha256": digest(a.decisions),
        "geometry_certified": False,
        "limitations": (
            "Model-assisted visual QC, not independent human annotation or "
            "measured 3D equivalence. Residual geometry/appearance ambiguity "
            "remains; conclusions concern this specified filtered "
            "augmentation pipeline."
        ),
    }
    save(a.output / "review.json", review)
    accepted = sum(r["decision"] == "accepted" for r in rows)
    categories = {r["sample_id"]: r["category_id"] for r in training["rows"]}
    save(
        a.output / "complete.json",
        {
            "status": "complete",
            "training_ready": True,
            "rows": rows,
            "training_sha256": digest(train / "complete.json"),
            "review_file": "review.json",
            "review_sha256": digest(a.output / "review.json"),
            "accepted_images": accepted,
            "fallback_images": 72 - accepted,
            "accepted_per_category": dict(
                Counter(categories[r["sample_id"]] for r in rows if r["decision"] == "accepted")
            ),
            "objects_with_accepted_augmentation": len(
                {r["sample_id"] for r in rows if r["decision"] == "accepted"}
            ),
            "effective_photo_update_fraction": accepted * 15 / 2160,
            "effective_exposure_rule": (
                "Each sample/variant is selected in 15 of 2160 updates per seed; "
                "rejected variants use original RGB."
            ),
            "candidate_archive_sha256": digest(a.candidates / "results.zip"),
            "script_sha256": digest(__file__),
            "label_arrays_modified": False,
            "test_outcomes_used": False,
        },
    )
    try:
        load_photo_images(a.output, train / "complete.json", training, observations)
    except Exception:
        failed = read(a.output / "complete.json")
        failed.update(status="validation_failed", training_ready=False)
        save(a.output / "complete.json", failed)
        raise
    a.output.rename(final_output)
    print({"accepted": accepted, "fallback": 72 - accepted, "training_ready": True})


if __name__ == "__main__":
    main()
