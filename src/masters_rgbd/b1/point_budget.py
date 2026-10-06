"""Deterministic count control for reconstructed vertices."""

import numpy as np


def budget512(points: np.ndarray) -> np.ndarray:
    """Uniform seeded subset of existing output vertices; never sees GT."""
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1:] != (3,) or not np.isfinite(points).all():
        raise ValueError("expected finite [N,3] points")
    if len(points) <= 512:
        return points.copy()
    return points[np.random.default_rng(20260904).choice(len(points), 512, replace=False)]
