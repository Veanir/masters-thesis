"""Freeze real-cohort inputs using source annotations, never predictor quality."""

import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.common.paths import DATASET_ROOT, script_help
from scripts.common.paths import ROOT as WORKSPACE_ROOT

script_help(__doc__, __name__)


ROOT = WORKSPACE_ROOT
DATA = DATASET_ROOT / "bop-hb-evolution-v1/hb"
OUT = ROOT / "runs/research-evolution-hb-20260906"
SEED = "masters-evolution-hb-reserve-v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    final = OUT / "cohort-reservation-v1.json"
    assert not final.exists(), "Reservation is immutable; create an explicit new version if needed"
    candidates = defaultdict(list)
    binding = {}
    for scene in sorted((DATA / "val_primesense").iterdir()):
        if not scene.is_dir():
            continue
        gt = json.loads((scene / "scene_gt.json").read_text())
        info = json.loads((scene / "scene_gt_info.json").read_text())
        for file in ("scene_gt.json", "scene_gt_info.json", "scene_camera.json"):
            binding[str((scene / file).relative_to(DATA))] = sha(scene / file)
        for image_id, instances in gt.items():
            for gt_id, instance in enumerate(instances):
                meta = info[image_id][gt_id]
                if meta["px_count_visib"] < 512 or meta["visib_fract"] < 0.1:
                    continue
                vis = meta["visib_fract"]
                row = {
                    "scene_id": int(scene.name),
                    "image_id": int(image_id),
                    "gt_id": gt_id,
                    "obj_id": instance["obj_id"],
                    "visibility_fraction": vis,
                    "source_visible_pixels": meta["px_count_visib"],
                    "visibility_bin": "low" if vis < 0.4 else "medium" if vis < 0.7 else "high",
                }
                row["rank"] = hashlib.sha256(
                    (
                        SEED
                        + "/"
                        + "/".join(str(row[k]) for k in ("scene_id", "image_id", "gt_id", "obj_id"))
                    ).encode()
                ).hexdigest()
                candidates[instance["obj_id"]].append(row)
    selected = []
    for object_id in sorted(candidates):
        chosen = []

        def admit(row, chosen=chosen):
            return all(
                row["scene_id"] != prev["scene_id"] or abs(row["image_id"] - prev["image_id"]) >= 15
                for prev in chosen
            )

        for visibility in ("low", "medium", "high"):
            rows = sorted(
                (r for r in candidates[object_id] if r["visibility_bin"] == visibility),
                key=lambda r: r["rank"],
            )
            for row in rows:
                if admit(row):
                    chosen.append(row)
                if sum(r["visibility_bin"] == visibility for r in chosen) == 2:
                    break
        for row in sorted(candidates[object_id], key=lambda r: r["rank"]):
            if len(chosen) >= 6:
                break
            if admit(row):
                chosen.append(row)
        assert len(chosen) == 6, (
            f"Object {object_id} has fewer than six eligible spaced observations"
        )
        selected.extend(chosen)
    assert len(candidates) == 33 and len(selected) == 198
    failures = []
    for row in selected:
        scene = DATA / "val_primesense" / f"{row['scene_id']:06d}"
        frame = f"{row['image_id']:06d}"
        refs = {
            "rgb": scene / "rgb" / f"{frame}.png",
            "depth": scene / "depth" / f"{frame}.png",
            "mask_visib": scene / "mask_visib" / f"{frame}_{row['gt_id']:06d}.png",
            "scene_camera": scene / "scene_camera.json",
            "scene_gt": scene / "scene_gt.json",
            "scene_gt_info": scene / "scene_gt_info.json",
            "mesh": DATA / "models" / f"obj_{row['obj_id']:06d}.ply",
        }
        row["files"] = {
            key: {"path": str(path.relative_to(DATA)), "sha256": sha(path)}
            for key, path in refs.items()
        }
        rgb = np.asarray(Image.open(refs["rgb"]).convert("RGB"))
        depth = np.asarray(Image.open(refs["depth"]))
        mask = np.asarray(Image.open(refs["mask_visib"])) > 0
        camera = json.loads(refs["scene_camera"].read_text())[str(row["image_id"])]
        k = np.array(camera["cam_K"]).reshape(3, 3)
        valid = mask & (depth > 0)
        z = depth[valid] * camera["depth_scale"] * 0.001
        row["input_audit"] = {
            "rgb_shape": list(rgb.shape),
            "depth_shape": list(depth.shape),
            "mask_pixels": int(mask.sum()),
            "valid_depth_pixels": int(valid.sum()),
            "median_depth_m": float(np.median(z)) if len(z) else None,
            "depth_scale_mm": camera["depth_scale"],
            "intrinsics": k.tolist(),
        }
        if rgb.shape[:2] != depth.shape or depth.shape != mask.shape or valid.sum() < 512:
            failures.append(
                {
                    "scene_id": row["scene_id"],
                    "image_id": row["image_id"],
                    "obj_id": row["obj_id"],
                    "reason": (
                        "Invalid RGB-D alignment dimensions or fewer than 512 positive "
                        "target depth pixels"
                    ),
                }
            )
    protocol = {
        "version": 1,
        "created_unix": time.time(),
        "seed": SEED,
        "role": "Reserved final cohort for phase 2; source validation split, not official BOP test",
        "scope": (
            "All 33 objects; six views each; no HB input used for training, "
            "model screening, augmentation or checkpoint selection"
        ),
        "selection": (
            "At least 512 annotated visible pixels and visibility >=0.1; "
            "hash-ranked two per available low [0.1,0.4), medium [0.4,0.7), "
            "high [0.7,1] bin; hash-fill unavailable bins; same-object "
            "same-scene frames separated by >=15 IDs"
        ),
        "source_annotations_used_for": (
            "Identity, visibility stratification, and input mask only; GT "
            "geometry stays in evaluation path"
        ),
        "failed_input_policy": (
            "Preserve selected membership and report input/adapter failures; "
            "no quality-based replacements"
        ),
        "adaptation_data_exclusion": (
            "HB models, RGB, depths and replicas excluded from synthetic data and occluders"
        ),
        "dependence": (
            "Objects and scenes overlap across views; primary object "
            "bootstrap plus scene-cluster sensitivity; never count 198 views "
            "as independent objects"
        ),
        "pretraining_limitation": (
            "No claim of absence from public pretraining; audit each selected "
            "model before interpretation"
        ),
        "source_revision": json.loads((OUT / "source.json").read_text())["repository_revision"],
        "script_sha256": sha(Path(__file__)),
        "object_count": 33,
        "observation_count": 198,
        "visibility_counts": dict(Counter(r["visibility_bin"] for r in selected)),
        "scene_counts": dict(Counter(str(r["scene_id"]) for r in selected)),
        "annotation_bindings": binding,
        "input_failures": failures,
        "observations": selected,
        "predictions_computed": False,
        "remaining_gate": (
            "Validate coordinate/GT conventions without choosing cohorts by "
            "model scores; freeze all metric and adaptation settings before "
            "inference"
        ),
    }
    final.write_text(json.dumps(protocol, indent=2))
    (OUT / "cohort-reservation-v1.sha256").write_text(sha(final) + "  " + final.name + "\n")
    print(
        json.dumps(
            {
                key: protocol[key]
                for key in (
                    "object_count",
                    "observation_count",
                    "visibility_counts",
                    "scene_counts",
                    "input_failures",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
