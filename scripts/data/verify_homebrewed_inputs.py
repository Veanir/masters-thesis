"""Independently verify all final input archives and crop ray/depth mappings."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.common.paths import ROOT as WORKSPACE_ROOT
from scripts.common.paths import script_help

script_help(__doc__, __name__)


ROOT = WORKSPACE_ROOT


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("prepared", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists()
    complete = json.loads((a.prepared / "complete.json").read_text())
    assert complete["status"] == "three_complete_observation_only_HB_bundles_prepared_no_inference"
    source_path = ROOT / "runs/research-evolution-hb-inputs-v2/complete.json"
    assert sha(source_path) == complete["source_input_manifest_sha256"]
    source = json.loads(source_path.read_text())
    expected = {r["sample_id"]: r for r in source["rows"]}
    checks = []
    max_ray_error = 0.0
    max_depth_error = 0.0
    tie_pixels = 0
    source_x = np.tile(np.arange(640, dtype=np.int32), (480, 1))
    source_y = np.tile(np.arange(480, dtype=np.int32)[:, None], (1, 640))
    for name, binding in complete["bindings"].items():
        folder = a.prepared / name
        manifest_path = folder / "manifest.json"
        assert sha(manifest_path) == binding["manifest_sha256"]
        manifest = json.loads(manifest_path.read_text())
        assert not manifest["gt_uploaded"] and {r["sample_id"] for r in manifest["rows"]} == set(
            expected
        )
        assert len(manifest["rows"]) == 198
        allowed = {"manifest.json"}
        for row in manifest["rows"]:
            sid = row["sample_id"]
            original = expected[sid]
            inp = source_path.parent / sid
            out = folder / sid
            assert (
                row["input_eligible"] == original["input_eligible"]
                and row["input_pixels"] == original["input_pixels"]
            )
            filenames = (
                {"rgb.png", "depth.png", "mask.png", "camera.json"}
                if name == "ray"
                else {"rgb.png", "depth.tiff", "mask.png", "camera.json"}
            )
            assert set(row["files"]) == filenames
            assert all(
                sha(inp / file) == value for file, value in original["export_sha256"].items()
            )
            assert all(sha(out / file) == value for file, value in row["files"].items())
            allowed.update(sid + "/" + file for file in filenames)
            if name == "ray":
                assert row["files"] == original["export_sha256"]
                continue
            rgb = np.asarray(Image.open(inp / "rgb.png").convert("RGB"))
            mask = np.asarray(Image.open(inp / "mask.png")) > 0
            codes = np.asarray(Image.open(inp / "depth.png"), dtype=np.uint16)
            k = np.asarray(json.loads((inp / "camera.json").read_text())["K"], float)
            got_rgb = np.asarray(Image.open(out / "rgb.png").convert("RGB"))
            got_mask = np.asarray(Image.open(out / "mask.png")) > 0
            got_depth = np.asarray(Image.open(out / "depth.tiff"))
            got_k = np.asarray(
                json.loads((out / "camera.json").read_text())["cam_K"], float
            ).reshape(3, 3)
            assert (
                got_depth.dtype == np.float32
                and got_depth.shape == got_mask.shape == (480, 640)
                and got_rgb.shape == (480, 640, 3)
            )
            left, top, right, bottom = row["crop_box"]
            assert (
                0 <= left < right <= 640
                and 0 <= top < bottom <= 480
                and (right - left) * 3 == (bottom - top) * 4
            )
            if not row["input_eligible"] or name == "octmae-original":
                assert row["crop_box"] == [0, 0, 640, 480]
            v, u = np.mgrid[:480, :640]
            x = left + (u + 0.5) * (right - left) / 640
            y = top + (v + 0.5) * (bottom - top) / 480
            # Read the discrete library mapping from coordinate images, then
            # independently prove it is nearest to the ideal calibrated ray.
            # At exact half-pixel ties either adjacent source centre is valid.
            ix = np.asarray(
                Image.fromarray(source_x)
                .crop((left, top, right, bottom))
                .resize((640, 480), Image.Resampling.NEAREST)
            )
            iy = np.asarray(
                Image.fromarray(source_y)
                .crop((left, top, right, bottom))
                .resize((640, 480), Image.Resampling.NEAREST)
            )
            assert (
                np.max(abs(ix + 0.5 - x)) <= 0.500000001
                and np.max(abs(iy + 0.5 - y)) <= 0.500000001
            )
            changed_x = ix != np.floor(x).astype(int)
            changed_y = iy != np.floor(y).astype(int)
            assert np.all(np.abs(x[changed_x] - np.rint(x[changed_x])) < 1e-9)
            assert np.all(np.abs(y[changed_y] - np.rint(y[changed_y])) < 1e-9)
            tie_pixels += int(np.sum(changed_x | changed_y))
            assert np.array_equal(got_mask, mask[iy, ix])
            expected_mm = codes.astype(np.float32) * np.float32(10000 / 65535)
            assert np.array_equal(got_depth, expected_mm[iy, ix])
            assert int(got_mask.sum()) == row["resampled_mask_pixels"]
            expected_rgb = rgb.copy()
            expected_rgb[~mask] = 0
            expected_rgb = np.asarray(
                Image.fromarray(expected_rgb)
                .crop((left, top, right, bottom))
                .resize((640, 480), Image.Resampling.BILINEAR)
            )
            assert np.array_equal(got_rgb, expected_rgb)
            ids = np.linspace(0, 480 * 640 - 1, 600, dtype=int)
            uv = np.stack([u.ravel()[ids], v.ravel()[ids], np.ones(len(ids))], 1)
            mapped = np.stack([x.ravel()[ids] - 0.5, y.ravel()[ids] - 0.5, np.ones(len(ids))], 1)
            error = float(np.max(np.abs(uv @ np.linalg.inv(got_k).T - mapped @ np.linalg.inv(k).T)))
            assert error < 1e-6
            max_ray_error = max(max_ray_error, error)
            quant = float(
                np.max(
                    abs(got_depth.astype(float) * 0.001 - codes[iy, ix].astype(float) * 10 / 65535)
                )
            )
            assert quant < 1e-6
            max_depth_error = max(max_depth_error, quant)
            checks.append(
                {
                    "sample_id": sid,
                    "variant": row["variant"],
                    "ray_error": error,
                    "mm_float32_conversion_error_m": quant,
                    "original_input_eligible": row["input_eligible"],
                }
            )
        archive = a.prepared / (name + ".zip")
        assert sha(archive) == binding["archive_sha256"]
        with zipfile.ZipFile(archive) as z:
            assert len(z.namelist()) == len(allowed) and set(z.namelist()) == allowed
            for file in allowed:
                assert hashlib.sha256(z.read(file)).hexdigest() == sha(folder / file)
    result = {
        "status": "all198_three_inference_bundles_independently_verified",
        "all_passed": True,
        "source_observations": 198,
        "model_bundles": 3,
        "eligible_per_bundle": 182,
        "ineligible_per_bundle": 16,
        "octmae_geometry_checks": len(checks),
        "maximum_crop_ray_error": max_ray_error,
        "maximum_float32_mm_conversion_error_m": max_depth_error,
        "rows": checks,
        "prepared_manifest_sha256": sha(a.prepared / "complete.json"),
        "script_sha256": sha(Path(__file__)),
        "GT_included": False,
        "model_predictions_computed": False,
        "output_pixels_with_alternate_exact_half_pixel_tie": tie_pixels,
        "scope": (
            "All source/export/file/archive hashes and eligibility checked; "
            "Ray byte-identical; coordinate images expose actual nearest "
            "mapping, independently bounded within0.5 source pixels with only "
            "exact-half ties differing from floor. Exact shared depth/mask "
            "mapping plus600 calibrated pixel-center rays per OctMAE "
            "observation/variant. TIFF millimeters checked against common "
            "uint16 depth. No GT geometry or model scores read."
        ),
    }
    a.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in ["rows", "scope"]}))


if __name__ == "__main__":
    main()
