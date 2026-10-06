"""Pure NumPy/SciPy final evaluation primitives; no inference or GT alignment."""

import hashlib

import numpy as np
from scipy.spatial import cKDTree

from scripts.common.paths import script_help

script_help(__doc__, __name__)


def stable_seed(purpose, sample_id):
    return int(hashlib.sha256((purpose + "/" + sample_id).encode()).hexdigest()[:16], 16)


def validate_cloud(points):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError(
            "Nonfinite or malformed cloud; preserve as an explicit prediction failure, do not trim"
        )
    return points


def sample_surface(vertices, faces, *, count, seed):
    vertices = validate_cloud(vertices)
    faces = np.asarray(faces)
    if faces.ndim != 2 or faces.shape[1] != 3 or not np.issubdtype(faces.dtype, np.integer):
        raise ValueError("Invalid triangular faces")
    if np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise ValueError("Face index out of range")
    triangles = vertices[faces]
    areas = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
    )
    valid = areas > 1e-14
    triangles = triangles[valid]
    areas = areas[valid]
    if not len(areas):
        raise ValueError("No nondegenerate target triangles")
    rng = np.random.default_rng(seed)
    selected = triangles[rng.choice(len(triangles), size=count, p=areas / areas.sum())]
    root = np.sqrt(rng.random(count))
    second = rng.random(count)
    barycentric = np.column_stack((1 - root, root * (1 - second), root * second))
    return np.sum(selected * barycentric[:, :, None], axis=1)


def select_points(points, count, *, sample_id):
    points = validate_cloud(points)
    if count is None or len(points) <= count:
        return points
    if count <= 0:
        raise ValueError("Positive point cap required")
    # Identical ID-based rule for every method, arm and training seed. No GT input.
    rng = np.random.default_rng(stable_seed("evolution-eval-point-selection-v1", sample_id))
    return points[rng.choice(len(points), count, replace=False)]


def visibility(points, depth_m, intrinsics, *, tolerance_m=0.005, max_patch_span_m=0.010):
    """Metric project camera [x right,y up,z backwards], full-scene sensor depth.

    Support requires all9 finite positive (0.1,10]m depths and no >10mm
    patch discontinuity. Missing/out-of-frame/front-inconsistent GT is uncertain.
    """
    points = validate_cloud(points)
    depth = np.asarray(depth_m, dtype=float)
    fx, fy, cx, cy = np.asarray(intrinsics, dtype=float)
    if depth.ndim != 2 or min(fx, fy) <= 0 or not np.isfinite([fx, fy, cx, cy]).all():
        raise ValueError("Invalid pinhole depth camera")
    z = -points[:, 2]
    u = np.full(len(points), -10, dtype=np.int64)
    v = u.copy()
    positive = z > 0
    # Avoid invalid float-to-int casts for far/behind-camera points.
    px = np.zeros(len(points))
    py = px.copy()
    px[positive] = cx + fx * points[positive, 0] / z[positive]
    py[positive] = cy - fy * points[positive, 1] / z[positive]
    h, w = depth.shape
    spatial = positive & (px >= 0.5) & (px < w - 1.5) & (py >= 0.5) & (py < h - 1.5)
    ix = np.flatnonzero(spatial)
    u[ix] = np.floor(px[ix] + 0.5).astype(np.int64)
    v[ix] = np.floor(py[ix] + 0.5).astype(np.int64)
    patches = np.stack(
        [depth[v[ix] + dy, u[ix] + dx] for dy in [-1, 0, 1] for dx in [-1, 0, 1]], axis=1
    )
    valid = np.isfinite(patches).all(1) & (patches > 0.1).all(1) & (patches <= 10).all(1)
    valid &= (patches.max(1) - patches.min(1)) <= max_patch_span_m
    supported = np.zeros(len(points), bool)
    supported[ix] = valid
    reference = np.zeros(len(points))
    reference[ix] = np.median(patches, axis=1)
    visible = supported & (np.abs(z - reference) <= tolerance_m)
    occluded = supported & (z > reference + tolerance_m)
    free = supported & (z < reference - tolerance_m)
    return {
        "visible": visible,
        "occluded": occluded,
        "uncertain": ~(visible | occluded),
        "known_free": free,
        "supported": supported,
    }


def score_cloud(
    points,
    truth,
    truth_tree,
    truth_masks,
    input_dense,
    scene_depth,
    intrinsics,
    *,
    thresholds=(0.002, 0.005, 0.010),
    chamfer_cap_m=0.1,
):
    points = validate_cloud(points)
    truth = validate_cloud(truth)
    input_dense = validate_cloud(input_dense)
    if not len(truth):
        raise ValueError("Empty GT is an evaluation error, never a model failure")
    if len(points):
        forward = truth_tree.query(points, workers=2)[0]
        backward = cKDTree(points).query(truth, workers=2)[0]
        added = (
            cKDTree(input_dense).query(points, workers=2)[0] > 0.005
            if len(input_dense)
            else np.ones(len(points), bool)
        )
        masks = visibility(points, scene_depth, intrinsics)
        chamfer = float((forward.mean() + backward.mean()) / 2)
        capped = float(
            (np.minimum(forward, chamfer_cap_m).mean() + np.minimum(backward, chamfer_cap_m).mean())
            / 2
        )
    else:
        forward = np.empty(0)
        backward = np.full(len(truth), np.inf)
        added = np.zeros(0, bool)
        masks = {key: np.zeros(0, bool) for key in ["known_free", "supported"]}
        chamfer = None
        capped = chamfer_cap_m
    surface = {}
    for threshold in thresholds:
        precision = float((forward <= threshold).mean()) if len(points) else 0.0
        recall = float((backward <= threshold).mean())
        surface[f"{threshold:.3f}"] = {
            "precision": precision,
            "recall": recall,
            "fscore": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        }
    subsets = {
        name: {
            "count": int(mask.sum()),
            "recall5": float((backward[mask] <= 0.005).mean()) if mask.any() else None,
        }
        for name, mask in truth_masks.items()
    }
    supported = masks["supported"]
    free = masks["known_free"]
    return {
        "count": len(points),
        "surface": surface,
        "chamfer_mean_m_success_only": chamfer,
        "chamfer_capped_100mm_m": capped,
        "gt_subsets": subsets,
        "added_count": int(added.sum()),
        "added_precision5": float((forward[added] <= 0.005).mean()) if added.any() else None,
        "known_free_count": int(free.sum()),
        "known_free_rate_all": float(free.mean()) if len(points) else None,
        "known_free_rate_supported": float(free[supported].mean()) if supported.any() else None,
        "supported_prediction_count": int(supported.sum()),
    }


def paired_object_summary(differences, object_ids, *, seed=2026090702, replicates=10000):
    """Input is one paired difference per observation after averaging fixed seeds."""
    differences = np.asarray(differences, dtype=float)
    object_ids = np.asarray(object_ids)
    if (
        differences.ndim != 1
        or len(differences) != len(object_ids)
        or not np.isfinite(differences).all()
    ):
        raise ValueError("Invalid paired differences")
    objects = np.unique(object_ids)
    values = np.array([differences[object_ids == obj].mean() for obj in objects])
    if not len(values):
        raise ValueError("Empty analysis population")
    indices = np.random.default_rng(seed).integers(len(values), size=(replicates, len(values)))
    interval = np.quantile(values[indices].mean(1), [0.025, 0.975])
    return {
        "difference": float(values.mean()),
        "object_bootstrap95": interval.tolist(),
        "objects": len(objects),
        "observations": len(differences),
        "object_differences": dict(zip(map(str, objects), map(float, values), strict=False)),
    }


def paired_dependence_sensitivity(
    differences_by_seed, object_ids, scene_ids, *, seed=2026090703, replicates=10000
):
    """Scene-cluster and crossed object/training-seed sensitivity; no view CI."""
    values = np.asarray(differences_by_seed, dtype=float)
    objects = np.asarray(object_ids)
    scenes = np.asarray(scene_ids)
    if (
        values.ndim != 2
        or values.shape[0] != len(objects)
        or len(scenes) != len(objects)
        or not np.isfinite(values).all()
    ):
        raise ValueError("Invalid paired matrix")
    unique_objects = np.unique(objects)
    unique_scenes = np.unique(scenes)
    if not len(unique_objects) or values.shape[1] < 1:
        raise ValueError("Empty paired matrix")
    matrix = np.stack([values[objects == obj].mean(0) for obj in unique_objects])
    rng = np.random.default_rng(seed)
    object_draws = rng.integers(len(unique_objects), size=(replicates, len(unique_objects)))
    seed_draws = rng.integers(values.shape[1], size=(replicates, values.shape[1]))
    crossed = matrix[object_draws[:, :, None], seed_draws[:, None, :]].mean((1, 2))
    # Fixed inverse view-count weights give each original object equal weight.
    # Scene resampling intentionally perturbs the observed scene/object mixture.
    weights = np.array([1 / np.sum(objects == obj) for obj in objects])
    mean = values.mean(1)
    numerators = np.array([np.sum(weights[scenes == s] * mean[scenes == s]) for s in unique_scenes])
    denominators = np.array([np.sum(weights[scenes == s]) for s in unique_scenes])
    scene_draws = rng.integers(len(unique_scenes), size=(replicates, len(unique_scenes)))
    estimates = numerators[scene_draws].sum(1) / denominators[scene_draws].sum(1)
    return {
        "per_seed_object_mean": matrix.mean(0).tolist(),
        "object_and_seed_bootstrap95": np.quantile(crossed, [0.025, 0.975]).tolist(),
        "scene_cluster_bootstrap95": np.quantile(estimates, [0.025, 0.975]).tolist(),
        "scenes": len(unique_scenes),
        "training_seeds": values.shape[1],
        "scope": (
            "Sensitivity intervals, not independent confirmation. Only three "
            "training seeds in the final matrix; scene resampling changes "
            "object mix."
        ),
    }
