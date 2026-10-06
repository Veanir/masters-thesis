"""Verify B1 source observations against the deterministic query cache."""

import json
import time

import numpy as np

from masters_rgbd.b1.dev_cohort import load_dev_sample
from masters_rgbd.b1.training import _cache_path, _manifest_path, _sha256_file


def preflight(args, specs):
    """Verify current source geometry/observations against old cache, without edits."""
    started = time.monotonic()
    records = []
    manifest = _manifest_path(args.dataset_root)
    for index, spec in enumerate(specs):
        prepared = load_dev_sample(args.dataset_root, manifest, spec, partial_point_count=512)
        observation = prepared.sample.observation
        path = _cache_path(args.cache_root, spec)
        with np.load(path, allow_pickle=False) as data:
            for name, actual in (
                ("rgb", observation.rgb),
                ("depth_m", observation.depth_m),
                ("mask", observation.mask),
                ("partial_points_camera_m", observation.partial_points_camera_m),
            ):
                if data[name].shape != actual.shape or not np.allclose(
                    data[name], actual, rtol=1e-6, atol=1e-7
                ):
                    raise ValueError(f"cache observation mismatch: {spec.asset.asset_id}/{name}")
            camera = observation.intrinsics
            if not np.allclose(data["intrinsics"], (camera.fx, camera.fy, camera.cx, camera.cy)):
                raise ValueError("cache intrinsics mismatch")
            for name in data.files:
                if not np.isfinite(data[name]).all():
                    raise ValueError(f"nonfinite cache array {name}")
            for prefix, count in (("train", 512), ("evaluation", 1024)):
                if data[f"{prefix}_points"].shape != (count, 3):
                    raise ValueError("query shape mismatch")
                if data[f"{prefix}_udf_m"].shape != (count,) or np.any(data[f"{prefix}_udf_m"] < 0):
                    raise ValueError("query distance mismatch")
        records.append(
            {
                "asset_id": spec.asset.asset_id,
                "path": str(path.resolve()),
                "sha256": _sha256_file(path),
            }
        )
        if (index + 1) % 10 == 0:
            print(json.dumps({"preflight": index + 1, "total": len(specs)}), flush=True)
    result = {
        "status": "passed",
        "scope": (
            "source geometry fingerprints, full observations, finite query arrays; "
            "labels not recomputed"
        ),
        "cache_spec_sha256": _sha256_file(args.cache_root / "cache-spec.json"),
        "inventory_sha256": _sha256_file(args.inventory),
        "manifest_sha256": _sha256_file(manifest),
        "wall_time_s": time.monotonic() - started,
        "cache": records,
    }
    args.output_root.mkdir(parents=True, exist_ok=False)
    (args.output_root / "preflight.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "cache"}), flush=True)
