"""Measure phenomenological depth statistics on exposed YCB development only."""

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from scripts.common.paths import DATASET_ROOT, script_help
from scripts.common.paths import ROOT as WORKSPACE_ROOT

script_help(__doc__, __name__)


ROOT = WORKSPACE_ROOT
SOURCE = ROOT / "runs/ray-ycbv-observations-20260906"
OUT = ROOT / "runs/research-evolution-depth-model-20260907"


def main():
    OUT.mkdir(exist_ok=True)
    rows = []
    bindings = json.loads((SOURCE / "binding.json").read_text())["source_sha256"]
    data = DATASET_ROOT / "bop-ycbv-real-v1/ycbv"
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=float)
    for path in sorted(SOURCE.glob("obj*.npz")):
        meta = json.loads(path.with_suffix(".json").read_text())
        assert hashlib.sha256(path.read_bytes()).hexdigest() == meta["output_sha256"]
        prefix = f"test/{meta['scene_id']:06d}"
        depth_path = f"{prefix}/depth/{meta['image_id']:06d}.png"
        mask_path = f"{prefix}/mask_visib/{meta['image_id']:06d}_{meta['gt_id']:06d}.png"
        camera_path = f"{prefix}/scene_camera.json"
        for relative in (depth_path, mask_path, camera_path):
            assert hashlib.sha256((data / relative).read_bytes()).hexdigest() == bindings[relative]
        scale = (
            json.loads((data / camera_path).read_text())[str(meta["image_id"])]["depth_scale"]
            * 0.001
        )
        depth = np.asarray(Image.open(data / depth_path), dtype=np.float64) * scale
        mask = np.asarray(Image.open(data / mask_path)) > 0
        valid = np.isfinite(depth) & (depth > 0)
        interior = ndimage.binary_erosion(mask, iterations=3)
        boundary = mask & ~interior
        patch = ndimage.binary_erosion(interior & valid, iterations=2)
        safe = np.where(valid, depth, 0.0)
        variation = ndimage.maximum_filter(safe, size=5) - ndimage.minimum_filter(safe, size=5)
        patch &= variation < 0.01
        inverse = np.divide(1.0, depth, out=np.zeros_like(depth), where=valid)
        residual = ndimage.convolve(inverse, kernel, mode="nearest")[patch]
        mad = float(np.median(abs(residual - np.median(residual)))) if len(residual) else None
        sigma_inverse = None if mad is None else mad / (0.6744897501960817 * np.sqrt(20))
        values = np.unique(depth[mask & valid])
        steps = np.diff(values)
        steps = steps[steps > 1e-7]
        quantum = (
            float(np.median(steps[steps <= np.quantile(steps, 0.1) * 1.5])) if len(steps) else None
        )
        rows.append(
            {
                "sample_id": meta["sample_id"],
                "obj_id": meta["obj_id"],
                "input_sha256": meta["output_sha256"],
                "raw_depth_sha256": bindings[depth_path],
                "raw_visible_mask_sha256": bindings[mask_path],
                "source_depth_step_m": scale,
                "valid_target_pixels": int((mask & valid).sum()),
                "mask_pixels": int(mask.sum()),
                "median_depth_m": float(np.median(depth[mask & valid])),
                "interior_invalid_rate": float((interior & ~valid).sum() / interior.sum())
                if interior.any()
                else None,
                "boundary_invalid_rate": float((boundary & ~valid).sum() / boundary.sum())
                if boundary.any()
                else None,
                "inverse_depth_noise_coefficient_per_m": sigma_inverse,
                "plane_patch_pixels": int(patch.sum()),
                "observed_lower_step_m": quantum,
            }
        )
    assert len(rows) == 42
    keys = [
        "interior_invalid_rate",
        "boundary_invalid_rate",
        "inverse_depth_noise_coefficient_per_m",
        "observed_lower_step_m",
    ]
    medians = {key: float(np.median([r[key] for r in rows if r[key] is not None])) for key in keys}
    report = {
        "status": "exposed_development_depth_statistics_complete",
        "rows": rows,
        "medians_across_observations": medians,
        "method": (
            "Inverse-depth 5-point Laplacian; 3-pixel target erosion and "
            "2-pixel validity erosion; 5x5 depth range <1 cm; MAD / (0.67449 "
            "sqrt(20)). Small independent inverse-depth noise corresponds "
            "approximately to sigma_z = coefficient * z^2."
        ),
        "limitations": (
            "Single-frame high-frequency estimator includes quantization, "
            "local curvature and spatial correlation. It is not a sensor "
            "calibration or a full structured-light simulator. Missing-value "
            "rates omit other sensor artifacts. No HB data or reconstruction "
            "scores used."
        ),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "protocol_status": (
            "Statistics for choosing a paired sensor-depth augmentation; "
            "parameters not yet frozen for final data."
        ),
    }
    (OUT / "development-statistics-v2.json").write_text(json.dumps(report, indent=2))
    print(medians, flush=True)


if __name__ == "__main__":
    main()
