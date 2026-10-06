"""Count actual frame/model-mask RGB changes under the exact paired schedule."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.train.augmentation_schedule import classic_rgb, schedule


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8388608), b""):
            h.update(block)
    return h.hexdigest()


def changes(source, edited, mask, context):
    assert (
        source.dtype == edited.dtype == np.uint8
        and source.shape == edited.shape
        and source.ndim == 3
        and source.shape[2] == 3
    )
    assert mask.dtype == bool and mask.shape == source.shape[:2] and context in ["masked", "full"]
    difference = np.abs(source.astype(np.int16) - edited.astype(np.int16))
    pixels = np.any(difference != 0, axis=2)
    visible = mask.copy() if context == "masked" else np.ones(mask.shape, bool)
    # The pinned DINO loader masks first, then CenterCrop to multiples of14.
    # Its subsequent RGB normalization is affine and preserves pixel changes.
    h, w = mask.shape
    crop_h = h // 14 * 14
    crop_w = w // 14 * 14
    assert min(crop_h, crop_w) > 0
    top = int(round((h - crop_h) / 2))
    left = int(round((w - crop_w) / 2))
    retained = np.zeros(mask.shape, bool)
    retained[top : top + crop_h, left : left + crop_w] = True
    input_mask_pixels = int(visible.sum())
    visible &= retained
    count = int(visible.sum())
    changed = int(pixels[visible].sum())
    return {
        "frame_changed": bool(pixels.any()),
        "frame_changed_pixels": int(pixels.sum()),
        "frame_pixels": pixels.size,
        "model_RGB_changed": changed > 0,
        "model_changed_pixels": changed,
        "model_pixels": count,
        "pixels_removed_by_DINO_crop": input_mask_pixels - count,
        "DINO_crop_ltrb": [left, top, left + crop_w, top + crop_h],
        "model_changed_pixel_fraction": changed / count if count else None,
        "model_mean_abs_channel_change_0_255": float(difference[visible].mean()) if count else None,
    }


def audit(training, photo, plan, output, software_canary=False):
    assert not output.exists()
    job = json.loads(plan.read_text())
    labels = json.loads((training / "complete.json").read_text())
    admission = json.loads((photo / "complete.json").read_text())
    assert (
        labels["status"] == "certified_synthetic_ray_labels_complete"
        and admission["status"] == "frozen_PHOTO_admission"
    )
    assert job["training_manifest_sha256"] == sha(training / "complete.json") and job[
        "photo_admission_manifest_sha256"
    ] == sha(photo / "complete.json")
    assert job["epochs"] == 4 and job["context"] in ["masked", "full"]
    ids = job["sample_ids"]
    assert len(ids) == len(set(ids)) and job["steps"] == 4 * len(ids)
    if software_canary:
        assert job["purpose"] == "paired_evolution_matrix_recovery_canary"
    else:
        assert job["purpose"] == "paired_evolution_matrix_frozen" and len(ids) == 6400
    lookup = {r["sample_id"]: r for r in labels["rows"]}
    photo_lookup = {r["sample_id"]: r for r in admission["rows"]}
    assert (
        len(lookup) == len(labels["rows"])
        and len(photo_lookup) == len(admission["rows"])
        and set(photo_lookup) == set(ids)
    )
    assert all(lookup[sid]["split"] == "train" for sid in ids)
    accepted = {sid for sid in ids if photo_lookup[sid]["accepted"]}
    plans = {seed: schedule(ids, accepted, seed, epochs=4) for seed in [0, 1, 2]}
    by_sid = {seed: {sid: [] for sid in ids} for seed in plans}
    for seed, exposures in plans.items():
        for exposure in exposures:
            by_sid[seed][exposure["sample_id"]].append(exposure)
    output.mkdir()
    records = []
    totals = {str(seed): {arm: Counter() for arm in ["BASE", "CLASSIC", "PHOTO"]} for seed in plans}
    for ordinal, sid in enumerate(ids):
        ref = lookup[sid]
        path = training / (sid + ".npz")
        assert sha(path) == ref["output_sha256"]
        with np.load(path) as z:
            rgb = z["rgb"].copy()
            mask = z["input_mask"].copy()
        photo_row = photo_lookup[sid]
        edited_path = (photo / photo_row["filename"]).resolve()
        edited_path.relative_to(photo.resolve())
        assert (
            sha(edited_path) == photo_row["sha256"]
            and hashlib.sha256(rgb.tobytes()).hexdigest() == photo_row["source_rgb_array_sha256"]
        )
        edited = np.asarray(Image.open(edited_path).convert("RGB"))
        photo_change = changes(rgb, edited, mask, job["context"])
        unchanged = changes(rgb, rgb, mask, job["context"])
        sample = {
            "sample_id": sid,
            "accepted": sid in accepted,
            "source_npz_sha256": ref["output_sha256"],
            "photo_sha256": photo_row["sha256"],
            "input_mask_array_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
            "raw_PHOTO_change": photo_change,
            "exposures": [],
        }
        for seed in plans:
            for exposure in by_sid[seed][sid]:
                effective = exposure["effective_augmentation"]
                actual = {
                    "BASE": unchanged,
                    "PHOTO": photo_change if effective else unchanged,
                    "CLASSIC": changes(
                        rgb, classic_rgb(rgb, exposure["augmentation_seed"]), mask, job["context"]
                    )
                    if effective
                    else unchanged,
                }
                sample["exposures"].append({"seed": seed, **exposure, "actual": actual})
                for arm, change in actual.items():
                    t = totals[str(seed)][arm]
                    t["exposures"] += 1
                    t["augmentation_attempts"] += int(
                        exposure["augmentation_attempt"] and arm != "BASE"
                    )
                    t["admitted_augmentation_exposures"] += int(effective and arm != "BASE")
                    for key in [
                        "frame_changed",
                        "frame_changed_pixels",
                        "frame_pixels",
                        "model_RGB_changed",
                        "model_changed_pixels",
                        "model_pixels",
                    ]:
                        t[key] += int(change[key])
        records.append(sample)
        if (ordinal + 1) % 320 == 0:
            print("MODEL RGB EXPOSURE AUDIT", ordinal + 1, flush=True)
    for arms in totals.values():
        assert arms["BASE"]["model_RGB_changed"] == arms["BASE"]["frame_changed"] == 0
        for arm in ["CLASSIC", "PHOTO"]:
            assert arms[arm]["augmentation_attempts"] == 2 * len(ids) and arms[arm][
                "admitted_augmentation_exposures"
            ] == 2 * len(accepted)
        for arm, counts in arms.items():
            assert (
                counts["model_RGB_changed"]
                <= counts["frame_changed"]
                <= counts["admitted_augmentation_exposures"]
            )
            counts = dict(counts)
            counts["frame_changed_exposure_fraction"] = (
                counts["frame_changed"] / counts["exposures"]
            )
            counts["model_changed_exposure_fraction"] = (
                counts["model_RGB_changed"] / counts["exposures"]
            )
            arms[arm] = counts
    result = {
        "status": "software_canary_model_RGB_exposure_audit_complete"
        if software_canary
        else "full_TRAIN_model_RGB_exposure_audit_complete",
        "software_canary": software_canary,
        "count": len(ids),
        "seeds": [0, 1, 2],
        "context": job["context"],
        "accepted_count": len(accepted),
        "totals": totals,
        "rows": records,
        "plan_sha256": sha(plan),
        "training_manifest_sha256": sha(training / "complete.json"),
        "photo_admission_manifest_sha256": sha(photo / "complete.json"),
        "script_sha256": sha(Path(__file__)),
        "training_design_sha256": sha(
            Path(__file__).parents[2].joinpath("scripts/train/augmentation_schedule.py")
        ),
        "scope": (
            "Pixel changes only, not a photorealism or geometry certificate. "
            "Exact RGB and sensor input mask from training labels, followed "
            "by the pinned DINO CenterCrop to multiples of14. Subsequent "
            "affine normalization preserves changes; DINO feature differences "
            "are not measured. All three seeds use fixed schedules; CLASSIC "
            "uses exact per-step transforms. Rejected PHOTO and matched "
            "CLASSIC fallbacks count as unchanged."
        ),
    }
    (output / "complete.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(
        json.dumps(
            {k: result[k] for k in ["status", "count", "context", "accepted_count", "totals"]}
        )
    )
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--training", type=Path, required=True)
    p.add_argument("--photo", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--software-canary", action="store_true")
    a = p.parse_args()
    audit(a.training, a.photo, a.plan, a.output, a.software_canary)


if __name__ == "__main__":
    main()
