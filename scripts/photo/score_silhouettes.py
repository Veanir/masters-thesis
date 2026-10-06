"""Recompute every source/edit silhouette diagnostic; no PHOTO acceptance labels."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.photo.image_records import sha
from scripts.photo.silhouette_metrics import metrics


def score(job_path, predictions, targets, pilot_regression=False):
    job = json.loads(job_path.read_text())
    result = json.loads((predictions / "complete.json").read_text())
    expected_n = 96 if pilot_regression else 1280
    assert result["job_sha256"] == sha(job_path)
    if pilot_regression:
        assert job["status"] == "silhouette_diagnostic_frozen_before_SAM_predictions"
        assert result["status"] == "96_SAM2_masks_complete_requires_local_QA"
    else:
        assert job["purpose"] == "frozen_final_SAM_silhouette_diagnostic"
        assert result["status"] == "bound_SAM_masks_complete_require_local_QA"
        assert (
            result["script_sha256"] == job["runner_sha256"]
            and result["versions"] == job["expected_versions"]
        )
    assert job["screen_thresholds"] == {
        "source_IoU_min": 0.98,
        "edited_IoU_min": 0.98,
        "max_direction_boundary_p95_pixels_at640width": 3.0,
    }
    expected = {(r["sample_id"], r["condition"]): r for r in job["rows"]}
    assert len(expected) == len(job["rows"]) == len(result["rows"]) == expected_n
    assert [(r["sample_id"], r["condition"]) for r in result["rows"]] == list(expected)
    rows = []
    for row in result["rows"]:
        sid = row["sample_id"]
        condition = row["condition"]
        assert condition in ["source", "reference"]
        assert all(row[k] == v for k, v in expected[sid, condition].items())
        target = targets / sid / "mask.png" if pilot_regression else targets / (sid + ".png")
        prediction = (predictions / row["filename"]).resolve()
        assert target.resolve().is_relative_to(targets.resolve()) and prediction.is_relative_to(
            predictions.resolve()
        )
        assert sha(target) == row["source_mask_sha256"] and sha(prediction) == row["output_sha256"]
        with Image.open(target) as image:
            gt = np.asarray(image) > 0
        with Image.open(prediction) as image:
            assert image.mode == "L" and image.size == (640, 480)
            array = np.asarray(image)
            assert set(np.unique(array)).issubset({0, 255})
            predicted = array > 0
        rows.append(
            {
                "sample_id": sid,
                "condition": condition,
                **metrics(gt, predicted, job["screen_thresholds"], condition),
                "prediction_sha256": sha(prediction),
            }
        )
    source = {r["sample_id"]: r for r in rows if r["condition"] == "source"}
    edited = {r["sample_id"]: r for r in rows if r["condition"] == "reference"}
    assert set(source) == set(edited) and len(source) == expected_n // 2
    pairs = [
        {
            "sample_id": sid,
            "source_calibrated": source[sid]["raw_screen_pass"],
            "edited_raw_screen_pass": edited[sid]["raw_screen_pass"],
            "calibrated_silhouette_pass": source[sid]["raw_screen_pass"]
            and edited[sid]["raw_screen_pass"],
        }
        for sid in source
    ]
    for row in rows:
        row["source_calibrated"] = source[row["sample_id"]]["raw_screen_pass"]
    summary = {
        c: {
            "n": sum(r["condition"] == c for r in rows),
            "raw_pass": sum(r["condition"] == c and r["raw_screen_pass"] for r in rows),
            "calibrated_pass": sum(
                r["condition"] == c and r["raw_screen_pass"] and r["source_calibrated"]
                for r in rows
            ),
        }
        for c in ["source", "reference"]
    }
    return {
        "status": "SAM_pilot_metric_regression_only"
        if pilot_regression
        else "all640_source_edit_silhouette_diagnostics_recomputed",
        "count": expected_n,
        "pair_count": expected_n // 2,
        "rows": rows,
        "pairs": pairs,
        "summary": summary,
        "job_sha256": sha(job_path),
        "SAM_manifest_sha256": sha(predictions / "complete.json"),
        "script_sha256": sha(Path(__file__)),
        "training_ready": False,
        "PHOTO_admission_frozen": False,
        "scope": (
            "Calibrated silhouette evidence only. Failed source calibration "
            "prevents automatic interpretation; internal shape, occlusion/depth "
            "consistency and photorealism still require review."
        ),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--job", type=Path, required=True)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--targets", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--pilot-regression", action="store_true")
    a = p.parse_args()
    result = score(a.job, a.predictions, a.targets, a.pilot_regression)
    with a.output.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result["summary"], indent=2))
