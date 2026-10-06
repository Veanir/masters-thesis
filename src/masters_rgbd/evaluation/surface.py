"""Metric surface evaluation used by the analytic P0 fixture."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


def _surface(value: object, *, name: str) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError(f"{name} must be a non-empty [N, 3] array")
    if not np.isfinite(points).all():
        raise ValueError(f"{name} contains non-finite values")
    return points


@dataclass(frozen=True)
class SurfaceMetrics:
    chamfer_mean_m: float
    chamfer_p95_m: float
    accuracy_p95_m: float
    completeness_p95_m: float
    fscore: float
    precision: float
    recall: float
    threshold_m: float


def evaluate_surface_points(
    predicted_m: object,
    target_m: object,
    *,
    threshold_m: float,
) -> SurfaceMetrics:
    if not np.isfinite(threshold_m) or threshold_m <= 0.0:
        raise ValueError("threshold_m must be positive and finite")
    predicted = _surface(predicted_m, name="predicted_m")
    target = _surface(target_m, name="target_m")
    accuracy = cKDTree(target).query(predicted, k=1, workers=-1)[0]
    completeness = cKDTree(predicted).query(target, k=1, workers=-1)[0]
    precision = float(np.mean(accuracy <= threshold_m))
    recall = float(np.mean(completeness <= threshold_m))
    fscore = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
    accuracy_p95 = float(np.quantile(accuracy, 0.95))
    completeness_p95 = float(np.quantile(completeness, 0.95))
    return SurfaceMetrics(
        chamfer_mean_m=float(0.5 * (accuracy.mean() + completeness.mean())),
        chamfer_p95_m=0.5 * (accuracy_p95 + completeness_p95),
        accuracy_p95_m=accuracy_p95,
        completeness_p95_m=completeness_p95,
        fscore=fscore,
        precision=precision,
        recall=recall,
        threshold_m=threshold_m,
    )
