"""Surface metrics and historical visibility rules used by the HOPE pilot."""

import numpy as np
from scipy.spatial import cKDTree

from scripts.common.paths import script_help
from scripts.common.surface_distances import metrics

script_help(__doc__, __name__)


def select(points, count):
    if len(points) <= count:
        return points
    return points[np.random.default_rng(20260904).choice(len(points), count, replace=False)]


def visibility(points, data):
    """Same conservative 3x3 minimum-depth support as historical diagnostics."""
    z = -points[:, 2]
    fx, fy, cx, cy = data["intrinsics"]
    u = np.floor(cx + fx * points[:, 0] / np.maximum(z, 1e-15) + 0.5).astype(np.int64)
    v = np.floor(cy - fy * points[:, 1] / np.maximum(z, 1e-15) + 0.5).astype(np.int64)
    depth, mask = data["depth_m"], data["mask"].astype(bool)
    height, width = depth.shape
    spatial = (z > 0) & (u >= 1) & (u < width - 1) & (v >= 1) & (v < height - 1)
    supported = np.zeros(len(points), bool)
    minimum = np.full(len(points), np.inf)
    ix = np.flatnonzero(spatial)
    valid = np.ones(len(ix), bool)
    local_min = np.full(len(ix), np.inf)
    for dv in (-1, 0, 1):
        for du in (-1, 0, 1):
            d = depth[v[ix] + dv, u[ix] + du]
            valid &= mask[v[ix] + dv, u[ix] + du] & (d > 0)
            local_min = np.minimum(local_min, d)
    supported[ix], minimum[ix] = valid, local_min
    visible = supported & (np.abs(z - minimum) <= 0.005)
    occluded = supported & (z > minimum + 0.005)
    free = supported & (z < minimum - 0.005)
    uncertain = ~(visible | occluded)
    return {
        "visible": visible,
        "occluded": occluded,
        "uncertain": uncertain,
        "supported": supported,
        "known_free": free,
    }


def score(points, truth, truth_tree, truth_masks, input_tree, data):
    count = len(points)
    if count:
        forward = truth_tree.query(points, workers=2)[0]
        backward = cKDTree(points).query(truth, workers=2)[0]
        surface = metrics(forward, backward)
        hits = backward <= 0.005
        added = input_tree.query(points, workers=2)[0] > 0.005
        point_masks = visibility(points, data)
        free = point_masks["known_free"]
        supported = point_masks["supported"]
    else:
        surface = {
            f"{t:.3f}": {"precision": 0.0, "recall": 0.0, "fscore": 0.0, "chamfer_mean_m": None}
            for t in (0.002, 0.005, 0.010)
        }
        forward = np.empty(0)
        added = free = supported = np.zeros(0, bool)
        hits = np.zeros(len(truth), bool)
    subsets = {
        name: {
            "count": int(mask.sum()),
            "recall5": float(hits[mask].mean()) if mask.any() else None,
        }
        for name, mask in truth_masks.items()
    }
    return {
        "count": count,
        "surface": surface,
        "gt_subsets": subsets,
        "added_count": int(added.sum()),
        "added_precision5": float((forward[added] <= 0.005).mean()) if added.any() else None,
        "known_free_count": int(free.sum()),
        "known_free_rate_all": float(free.mean()) if count else 0.0,
        "known_free_rate_supported": float(free[supported].mean()) if supported.any() else None,
        "supported_prediction_count": int(supported.sum()),
    }


def check():
    truth = np.array([[0.0, 0.0, -1.0], [0.01, 0.0, -1.0]])
    data = {
        "intrinsics": np.array([10.0, 10.0, 5.0, 5.0]),
        "depth_m": np.ones((11, 11)),
        "mask": np.ones((11, 11), bool),
    }
    test = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, -1.1], [0.0, 0.0, -0.9], [10.0, 0.0, -1.0]])
    flags = visibility(test, data)
    assert flags["visible"].tolist() == [True, False, False, False]
    assert flags["occluded"].tolist() == [False, True, False, False]
    assert flags["known_free"].tolist() == [False, False, True, False]
    assert flags["uncertain"].tolist() == [False, False, True, True]
    tree = cKDTree(truth)
    masks = {"all": np.ones(2, bool)}
    assert score(truth, truth, tree, masks, tree, data)["surface"]["0.005"]["fscore"] == 1
    assert (
        score(np.empty((0, 3)), truth, tree, masks, tree, data)["surface"]["0.005"]["fscore"] == 0
    )
    assert len(np.unique(select(np.arange(300).reshape(100, 3), 10), axis=0)) == 10
