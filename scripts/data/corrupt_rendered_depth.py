"""Camera/crop conversion validated against the Isaac triangle-ray oracle.

Sensor perturbation is an explicit development candidate, not a frozen final
dataset protocol. RGB variants must reuse the same generated depth artifact.
"""

import hashlib

import numpy as np
from PIL import Image
from scipy import ndimage

from scripts.common.paths import script_help

script_help(__doc__, __name__)


def camera_matrices(params):
    width, height = np.asarray(params["renderProductResolution"], dtype=int)
    projection = np.asarray(params["cameraProjection"]).reshape(4, 4).T
    assert abs(projection[0, 2]) < 1e-8 and abs(projection[1, 2]) < 1e-8
    k = np.array(
        [
            [projection[0, 0] * width / 2, 0, width / 2 - 0.5],
            [0, projection[1, 1] * height / 2, height / 2 - 0.5],
            [0, 0, 1],
        ],
        dtype=float,
    )
    world_to_project = np.asarray(params["cameraViewTransform"]).reshape(4, 4).T
    assert abs(float(params["metersPerSceneUnit"]) - 1) < 1e-8
    return k, world_to_project


def crop_box(mask, margin=1.4):
    v, u = np.nonzero(mask)
    if not len(u):
        raise ValueError("Target is invisible; preserve failure, no crop")
    height, width = mask.shape
    desired = max((u.max() - u.min() + 1) * margin, (v.max() - v.min() + 1) * margin * 4 / 3)
    crop_width = min(width, 4 * int(np.ceil(desired / 4)))
    crop_height = crop_width * 3 // 4
    assert crop_height <= height
    left = int(np.clip(np.floor((u.min() + u.max() + 1 - crop_width) / 2), 0, width - crop_width))
    top = int(np.clip(np.floor((v.min() + v.max() + 1 - crop_height) / 2), 0, height - crop_height))
    return left, top, left + crop_width, top + crop_height


def cropped_intrinsics(k, box, size=(640, 480)):
    left, top, right, bottom = box
    sx, sy = size[0] / (right - left), size[1] / (bottom - top)
    result = np.asarray(k, dtype=float).copy()
    result[0, :] *= sx
    result[1, :] *= sy
    result[0, 2] = sx * (k[0, 2] - left + 0.5) - 0.5
    result[1, 2] = sy * (k[1, 2] - top + 0.5) - 0.5
    return result


def crop_channel(array, box, size=(640, 480), rgb=False):
    mode = Image.Resampling.BILINEAR if rgb else Image.Resampling.NEAREST
    return np.asarray(Image.fromarray(array).crop(box).resize(size, resample=mode)).copy()


def sensor_candidate(clean, mask, sample_id, statistics):
    medians = statistics["medians_across_observations"]
    seed = int(
        hashlib.sha256(("evolution-sensor-candidate-v1/" + sample_id).encode()).hexdigest()[:16], 16
    )
    rng = np.random.default_rng(seed)
    height, width = clean.shape
    valid = np.isfinite(clean) & (clean > 0)
    base = np.where(valid, clean, 0.0).astype(float)
    coefficient = medians["inverse_depth_noise_coefficient_per_m"]
    noisy = base + rng.normal(size=base.shape) * coefficient * base**2
    # The storage unit (0.1mm in YCB) differs from the observed sensor step
    # (1mm). Use the measured positive difference, not the encoding unit.
    step = float(medians["observed_lower_step_m"])
    noisy = np.rint(noisy / step) * step
    # Correlated dropout is a simple approximation; this field has no RGB input.
    field = ndimage.gaussian_filter(rng.normal(size=base.shape), sigma=1.5 * height / 480)
    radius = max(1, round(3 * height / 480))
    boundary = mask & ~ndimage.binary_erosion(mask, iterations=radius)
    thresholds = {
        name: float(np.quantile(field, medians[name + "_invalid_rate"]))
        for name in ["interior", "boundary"]
    }
    missing = field < np.where(boundary, thresholds["boundary"], thresholds["interior"])
    noisy[~valid | missing] = 0.0
    return noisy.astype(np.float32), {
        "seed": seed,
        "depth_step_m": step,
        "noise_coefficient_per_m": coefficient,
        "correlation_sigma_pixels": 1.5 * height / 480,
        "boundary_radius_pixels": radius,
        "target_invalid_rate": float((mask & (noisy <= 0)).sum() / mask.sum())
        if mask.any()
        else None,
        "status": (
            "development sensor candidate; not a physical sensor simulator or final frozen protocol"
        ),
    }


def geometry_selfcheck():
    k = np.array([[720.0, 0, 479.5], [0, 720.0, 359.5], [0, 0, 1.0]])
    box = (104, 81, 584, 441)
    size = (640, 480)
    new = cropped_intrinsics(k, box, size)
    rng = np.random.default_rng(20260907)
    uv = rng.uniform([0, 0], size, (500, 2))
    scale = np.array(size) / np.array([box[2] - box[0], box[3] - box[1]])
    source = (uv + 0.5) / scale + np.array(box[:2]) - 0.5
    old_rays = np.column_stack((source, np.ones(len(source)))) @ np.linalg.inv(k).T
    new_rays = np.column_stack((uv, np.ones(len(uv)))) @ np.linalg.inv(new).T
    assert np.max(abs(old_rays - new_rays)) < 1e-12
    print("Crop pixel-centre ray invariance passed", flush=True)


if __name__ == "__main__":
    geometry_selfcheck()
