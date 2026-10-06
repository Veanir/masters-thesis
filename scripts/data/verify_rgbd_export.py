"""Independent array and encoding validation of a completed synthetic export."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.common.paths import script_help

script_help(__doc__, __name__)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("source", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    manifest = json.loads((a.source / "complete.json").read_text())
    rows = []
    assert len({r["sample_id"] for r in manifest["rows"]}) == len(manifest["rows"])
    for r in manifest["rows"]:
        folder = a.source / r["sample_id"]
        assert all(sha(folder / n) == v for n, v in r["files"].items())
        with np.load(folder / "observation.npz") as z:
            rgb = z["rgb"]
            mask = z["input_mask"]
            encoded = z["input_depth_uint16"]
            sensor = z["depth_sensor_m"]
            k = z["K"]
            assert (
                rgb.shape == (480, 640, 3)
                and rgb.dtype == np.uint8
                and mask.dtype == bool
                and encoded.dtype == np.uint16
            )
            assert np.array_equal(rgb, np.asarray(Image.open(folder / "rgb.png")))
            assert np.array_equal(encoded, np.asarray(Image.open(folder / "depth.png")))
            assert np.array_equal(mask, np.asarray(Image.open(folder / "mask.png")) > 0)
            assert mask.sum() == r["input_pixels"] and mask.sum() >= 512
            assert not np.any(mask & ~z["source_target_mask"])
            assert np.array_equal(mask, encoded > 0) and np.all(sensor[~mask] == 0)
            assert np.max(np.abs(sensor[mask] * 1000 - np.rint(sensor[mask] * 1000))) < 1e-7
            assert np.array_equal(encoded, np.rint(sensor * 65535 / 10).astype(np.uint16))
            assert (
                np.allclose(k[2], [0, 0, 1])
                and np.allclose(k[:2, 2], [319.5, 239.5])
                and k[0, 0] > 0
                and k[1, 1] > 0
            )
            assert (
                np.isfinite(z["mesh_vertices_camera_cv_m"]).all()
                and z["mesh_vertices_camera_cv_m"][:, 2].min() > 0
            )
            rows.append(
                {
                    "sample_id": r["sample_id"],
                    "input_pixels": int(mask.sum()),
                    "fx": float(k[0, 0]),
                    "passed": True,
                }
            )
    report = {
        "status": "synthetic_export_arrays_verified",
        "source_manifest_sha256": sha(a.source / "complete.json"),
        "script_sha256": sha(Path(__file__)),
        "rows": rows,
        "all_passed": True,
        "count": len(rows),
        "minimum_input_pixels": min(r["input_pixels"] for r in rows),
        "focal_lengths": sorted({r["fx"] for r in rows}),
    }
    assert not a.output.exists()
    a.output.write_text(json.dumps(report, indent=2))
    print({k: v for k, v in report.items() if k != "rows"})


if __name__ == "__main__":
    main()
