"""Validate reviewed PHOTO pixels against the exact immutable TRAIN observations."""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.common.paths import script_help
from scripts.preliminary_hope.prepare_training_plan import digest

script_help(__doc__, __name__)


def load_photo_images(root, training_path, training, observations):
    root = Path(root).resolve()
    manifest = json.loads((root / "complete.json").read_text())
    if manifest.get("status") != "complete" or manifest.get("training_ready") is not True:
        raise ValueError("PHOTO data has not passed its bound quality review")
    if manifest.get("training_sha256") != digest(training_path):
        raise ValueError("PHOTO data belongs to another training manifest")
    review_path = (root / manifest["review_file"]).resolve()
    review_path.relative_to(root)
    if digest(review_path) != manifest["review_sha256"]:
        raise ValueError("PHOTO review changed")
    review = json.loads(review_path.read_text())
    if review.get("status") != "final_review" or review.get("test_outcomes_used") is not False:
        raise ValueError("PHOTO review is not independent of test outcomes")
    expected = {(r["sample_id"], v) for r in training["rows"] for v in (0, 1)}

    def index(rows):
        mapped = {(r["sample_id"], r["variant"]): r for r in rows}
        if len(mapped) != len(rows) or set(mapped) != expected:
            raise ValueError("PHOTO sample/variant coverage is incomplete or duplicated")
        return mapped

    records, decisions = index(manifest["rows"]), index(review["rows"])
    observed = {
        r["sample_id"]: data for r, data in zip(training["rows"], observations, strict=True)
    }
    result = {}
    accepted = 0
    for key, row in records.items():
        decision = decisions[key]
        if (
            row["decision"] != decision["decision"]
            or row["sha256"] != decision["training_rgb_sha256"]
        ):
            raise ValueError("PHOTO image differs from the reviewed decision")
        path = (root / row["filename"]).resolve()
        path.relative_to(root)
        if digest(path) != row["sha256"]:
            raise ValueError("PHOTO image bytes changed")
        with Image.open(path) as image:
            if image.mode != "RGB" or image.size != (640, 480):
                raise ValueError("PHOTO image format changed")
            rgb = np.array(image)
        source = observed[key[0]]
        mask = source["input_mask"].astype(bool)
        if rgb[~mask].any():
            raise ValueError("PHOTO includes pixels outside the frozen observation mask")
        if row["decision"] == "accepted":
            if not decision.get("label_compatibility_reviewed") or not decision.get(
                "appearance_reviewed"
            ):
                raise ValueError("PHOTO review omitted geometry or appearance")
            if np.array_equal(rgb[mask], source["rgb"][mask]):
                raise ValueError("An unchanged image cannot count as accepted PHOTO augmentation")
            accepted += 1
        elif row["decision"] == "fallback_original":
            if not decision.get("reason") or not np.array_equal(rgb[mask], source["rgb"][mask]):
                raise ValueError("PHOTO fallback is not the original observation")
        else:
            raise ValueError("Unsupported PHOTO decision")
        result[key] = rgb
    if not accepted or accepted != manifest["accepted_images"]:
        raise ValueError("PHOTO contains no accepted augmentation or misreports its count")
    return manifest, result
