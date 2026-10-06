"""Prepare common observation-only Ray/OctMAE inputs; no heldout inference."""

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.common.paths import ROOT as WORKSPACE_ROOT
from scripts.data.corrupt_rendered_depth import crop_box, crop_channel, cropped_intrinsics

ROOT = WORKSPACE_ROOT


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p, obj):
    p.write_text(json.dumps(obj, indent=2, allow_nan=False))


def convert(rgb, depth_codes, mask, k, variant, eligible):
    assert (
        rgb.shape == (480, 640, 3)
        and rgb.dtype == np.uint8
        and depth_codes.shape == mask.shape == (480, 640)
    )
    assert (
        depth_codes.dtype == np.uint16
        and mask.dtype == bool
        and np.isfinite(k).all()
        and k.shape == (3, 3)
    )
    assert np.all(depth_codes[mask] > 0)
    # Same quantized sensor depths as Ray, converted to the official OctMAE mm units.
    depth = depth_codes.astype(np.float32) * np.float32(10000 / 65535)
    masked = rgb.copy()
    masked[~mask] = 0
    box = (0, 0, 640, 480)
    if variant == "crop1p4" and eligible:
        box = crop_box(mask, 1.4)
        masked = crop_channel(masked, box, rgb=True)
        depth = crop_channel(depth, box)
        mask = crop_channel(mask.astype(np.uint8), box) > 0
        k = cropped_intrinsics(k, box).astype(np.float32)
    else:
        assert variant in ["original", "crop1p4"]
    return masked, depth, mask, k, box


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists()
    protocol_path = ROOT / "runs/research-evolution-evaluation-20260907/protocol-v1.json"
    protocol = json.loads(protocol_path.read_text())
    ref = protocol["references"]["input_manifest"]
    source_path = ROOT / ref["path"]
    assert sha(source_path) == ref["sha256"]
    source = json.loads(source_path.read_text())
    assert (
        not source["gt_exported"]
        and len(source["rows"]) == 198
        and [r["sample_id"] for r in source["rows"]] == protocol["sample_ids"]
    )
    a.output.mkdir()
    folders = {name: a.output / name for name in ["ray", "octmae-original", "octmae-crop1p4"]}
    rows = {name: [] for name in folders}
    for folder in folders.values():
        folder.mkdir()
    measurements = []
    for row in source["rows"]:
        sid = row["sample_id"]
        original = source_path.parent / sid
        assert not row["gt_exported"]
        assert set(row["export_sha256"]) == {"rgb.png", "depth.png", "mask.png", "camera.json"}
        assert all(sha(original / name) == digest for name, digest in row["export_sha256"].items())
        rgb = np.asarray(Image.open(original / "rgb.png").convert("RGB"))
        depth_codes = np.asarray(Image.open(original / "depth.png"), dtype=np.uint16)
        mask = np.asarray(Image.open(original / "mask.png")) > 0
        camera = json.loads((original / "camera.json").read_text())
        k = np.asarray(camera["K"], np.float32)
        assert int(mask.sum()) == row["input_pixels"] and row["input_eligible"] == (
            int(mask.sum()) >= 512
        )
        for name, folder in folders.items():
            target = folder / sid
            target.mkdir()
            meta = {
                "sample_id": sid,
                "cohort": "HB198",
                "input_eligible": row["input_eligible"],
                "input_pixels": row["input_pixels"],
            }
            if name == "ray":
                for file in row["export_sha256"]:
                    shutil.copyfile(original / file, target / file)
                files = dict(row["export_sha256"])
            else:
                variant = name.removeprefix("octmae-")
                image, depth, valid, new_k, box = convert(
                    rgb, depth_codes, mask, k, variant, row["input_eligible"]
                )
                Image.fromarray(image).save(target / "rgb.png")
                Image.fromarray(depth).save(target / "depth.tiff")
                Image.fromarray(valid.astype(np.uint8) * 255).save(target / "mask.png")
                save(
                    target / "camera.json",
                    {"cam_K": new_k.reshape(-1).astype(float).tolist(), "depth_scale": 1.0},
                )
                files = {
                    file: sha(target / file)
                    for file in ["rgb.png", "depth.tiff", "mask.png", "camera.json"]
                }
                meta.update(
                    variant=variant,
                    normalization="imagenet",
                    crop_box=list(box),
                    resampled_mask_pixels=int(valid.sum()),
                    original_source_eligibility_retained=True,
                )
                if variant == "original":
                    assert np.array_equal(valid, mask) and np.array_equal(new_k, k)
                    assert (
                        np.max(
                            abs(
                                depth.astype(np.float64) * 0.001
                                - depth_codes.astype(np.float64) * 10 / 65535
                            )
                        )
                        < 1e-6
                    )
                measurements.append(
                    {
                        "sample_id": sid,
                        "variant": variant,
                        "source_mask_pixels": int(mask.sum()),
                        "resampled_mask_pixels": int(valid.sum()),
                        "input_eligible_before_resampling": row["input_eligible"],
                        "crop_box": list(box),
                    }
                )
            rows[name].append({**meta, "files": files})
    bindings = {}
    for name, folder in folders.items():
        manifest = {
            "status": "observation_only_final_HB_inputs_prepared",
            "rows": rows[name],
            "gt_uploaded": False,
            "source_input_manifest_sha256": sha(source_path),
            "evaluation_protocol_sha256": sha(protocol_path),
            "script_sha256": sha(Path(__file__)),
            "schema": "ray-uint16-depth" if name == "ray" else "octmae-float32-mm-depth",
            "scope": (
                "All198 unchanged IDs and original512pixel eligibility. No "
                "meshes, GT pose or test metrics. Ray files byte-identical. "
                "OctMAE sees the same decoded quantized depth; original or "
                "fixed1.4crop with sensor mask and corrected pixel-center K."
            ),
        }
        save(folder / "manifest.json", manifest)
        archive = a.output / (name + ".zip")
        with zipfile.ZipFile(archive, "x", zipfile.ZIP_STORED) as z:
            z.write(folder / "manifest.json", "manifest.json")
            for row in rows[name]:
                for file in row["files"]:
                    z.write(folder / row["sample_id"] / file, row["sample_id"] + "/" + file)
        bindings[name] = {
            "manifest_sha256": sha(folder / "manifest.json"),
            "archive_sha256": sha(archive),
            "archive_bytes": archive.stat().st_size,
        }
    save(
        a.output / "complete.json",
        {
            "status": "three_complete_observation_only_HB_bundles_prepared_no_inference",
            "bindings": bindings,
            "measurements": measurements,
            "observations": 198,
            "eligible": 182,
            "ineligible": 16,
            "protocol_sha256": sha(protocol_path),
            "source_input_manifest_sha256": sha(source_path),
            "script_sha256": sha(Path(__file__)),
            "crop_helper_sha256": sha(
                Path(__file__).parents[2].joinpath("scripts/data/corrupt_rendered_depth.py")
            ),
            "requires_independent_input_verification_before_inference": True,
            "model_predictions_computed": False,
        },
    )
    print(
        json.dumps({"observations": 198, "eligible": 182, "ineligible": 16, "bindings": bindings})
    )


if __name__ == "__main__":
    main()
