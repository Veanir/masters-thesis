"""Paired family and training resampling for the B1 comparison."""

import numpy as np


def estimate(deltas, clusters):
    grouped = np.stack([deltas[:, indices].mean(axis=1) for indices in clusters], axis=1)
    if not np.isfinite(grouped).all():
        raise ValueError("nonfinite paired metric")
    rng = np.random.default_rng(20260904)
    seeds = rng.integers(0, grouped.shape[0], (20000, grouped.shape[0]))
    families = rng.integers(0, grouped.shape[1], (20000, grouped.shape[1]))
    samples = grouped[seeds[:, :, None], families[:, None, :]].mean(axis=(1, 2))
    return {
        "object_mean_delta": float(deltas.mean()),
        "family_mean_delta": float(grouped.mean()),
        "family_ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
        "per_seed_mean_delta": deltas.mean(axis=1).tolist(),
    }


METRICS = {
    "fscore5": lambda r: r["surface"]["0.005"]["fscore"],
    "precision5": lambda r: r["surface"]["0.005"]["precision"],
    "recall5": lambda r: r["surface"]["0.005"]["recall"],
    "hidden_recall5": lambda r: r["visibility"]["hidden_recall_5mm"],
}
