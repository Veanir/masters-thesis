"""Reject unreviewed, tampered, or falsely labelled PHOTO data before training."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from scripts.preliminary_hope.prepare_training_plan import digest  # noqa: E402
from scripts.preliminary_hope.training_data import load_photo_images  # noqa: E402


def bundle(tmp_path):
    training = {"rows": [{"sample_id": "train-a"}]}
    training_path = tmp_path / "training.json"
    training_path.write_text(json.dumps(training))
    root = tmp_path / "photo"
    root.mkdir()
    mask = np.zeros((480, 640), bool)
    mask[20:70, 30:90] = True
    rgb = np.where(mask[..., None], np.full((480, 640, 3), 80, np.uint8), 0).astype(np.uint8)
    records, reviews = [], []
    for variant in (0, 1):
        path = root / f"v{variant}.png"
        augmented = rgb.copy()
        augmented[mask] += 20 + variant
        Image.fromarray(augmented).save(path)
        row = {
            "sample_id": "train-a",
            "variant": variant,
            "filename": path.name,
            "sha256": digest(path),
            "decision": "accepted",
        }
        records.append(row)
        reviews.append(
            {
                "sample_id": "train-a",
                "variant": variant,
                "decision": "accepted",
                "training_rgb_sha256": row["sha256"],
                "label_compatibility_reviewed": True,
                "appearance_reviewed": True,
            }
        )
    review = {"status": "final_review", "test_outcomes_used": False, "rows": reviews}
    review_path = root / "review.json"
    review_path.write_text(json.dumps(review))
    manifest = {
        "status": "complete",
        "training_ready": True,
        "training_sha256": digest(training_path),
        "review_file": "review.json",
        "review_sha256": digest(review_path),
        "rows": records,
        "accepted_images": 2,
    }
    (root / "complete.json").write_text(json.dumps(manifest))
    return root, training_path, training, [{"rgb": rgb, "input_mask": mask}]


def test_reviewed_pair_loads(tmp_path):
    manifest, images = load_photo_images(*bundle(tmp_path))
    assert manifest["accepted_images"] == 2
    assert set(images) == {("train-a", 0), ("train-a", 1)}


def test_unreviewed_and_tampered_images_fail(tmp_path):
    args = bundle(tmp_path)
    root = args[0]
    path = root / "complete.json"
    manifest = json.loads(path.read_text())
    manifest["training_ready"] = False
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="quality review"):
        load_photo_images(*args)
    manifest["training_ready"] = True
    path.write_text(json.dumps(manifest))
    Image.new("RGB", (640, 480)).save(root / "v0.png")
    with pytest.raises(ValueError, match="bytes changed"):
        load_photo_images(*args)


def test_forged_fallback_is_not_accepted_as_original(tmp_path):
    args = bundle(tmp_path)
    root = args[0]
    path = root / "complete.json"
    manifest = json.loads(path.read_text())
    review_path = root / "review.json"
    review = json.loads(review_path.read_text())
    manifest["rows"][0]["decision"] = "fallback_original"
    review["rows"][0].update(decision="fallback_original", reason="rejected appearance")
    review_path.write_text(json.dumps(review))
    manifest["review_sha256"] = digest(review_path)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="not the original"):
        load_photo_images(*args)
