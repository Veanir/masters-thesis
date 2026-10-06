"""Object-balanced paired inference; scene clustering is a sensitivity analysis."""

import numpy as np

from scripts.common.paths import script_help

script_help(__doc__, __name__)


BOOTSTRAP_SEED = 20260906
REPLICATES = 10000


def object_ids(rows):
    return [str(r.get("obj_id", r.get("asset_id", r["sample_id"]))) for r in rows]


def object_means(values, rows):
    values = np.asarray(values, dtype=float)
    if values.shape != (len(rows),) or not len(rows) or not np.isfinite(values).all():
        raise ValueError("Paired values must be finite and match every observation")
    ids = np.asarray(object_ids(rows))
    labels = sorted(set(ids))
    return labels, np.array([values[ids == label].mean() for label in labels])


def paired_interval(values, rows):
    labels, means = object_means(values, rows)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap = means[rng.integers(0, len(means), size=(REPLICATES, len(means)))].mean(axis=1)
    result = {
        "mean": float(means.mean()),
        "ci95": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        "objects": len(labels),
        "observations": len(rows),
        "objects_positive": int((means > 0).sum()),
        "objects_negative": int((means < 0).sum()),
        "object_deltas": dict(zip(labels, means.tolist(), strict=True)),
        "unit": "object: average observations and paired training seeds before resampling objects",
        "bootstrap_replicates": REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "scope": (
            "conditional on selected scenes, admitted observations and the "
            "three fixed training seeds"
        ),
    }
    if all("scene_id" in r for r in rows):
        values = np.asarray(values, dtype=float)
        ids = np.asarray(object_ids(rows))
        scenes = np.asarray([r["scene_id"] for r in rows])
        unique = np.unique(scenes)
        weights = np.array([1 / np.count_nonzero(ids == key) for key in ids])
        sums = np.array([np.sum(values[scenes == s] * weights[scenes == s]) for s in unique])
        totals = np.array([weights[scenes == s].sum() for s in unique])
        draws = np.random.default_rng(BOOTSTRAP_SEED + 1).integers(
            0, len(unique), size=(REPLICATES, len(unique))
        )
        boot = sums[draws].sum(axis=1) / totals[draws].sum(axis=1)
        leave_one = {}
        for scene in unique:
            indices = np.flatnonzero(scenes != scene)
            _, remaining = object_means(values[indices], [rows[i] for i in indices])
            leave_one[str(int(scene))] = {
                "mean": float(remaining.mean()),
                "objects": len(remaining),
            }
        result["scene_sensitivity"] = {
            "clusters": len(unique),
            "ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
            "method": (
                "resample whole scenes with replacement; fixed inverse "
                "within-object observation-count weights; ratio of weighted sums"
            ),
            "scope": (
                "scene-cluster sensitivity, not a two-way object-and-scene "
                "population confidence interval"
            ),
            "bootstrap_seed": BOOTSTRAP_SEED + 1,
            "leave_one_scene_out": leave_one,
        }
    return result


def balanced_mean(values, rows):
    """Missing subset metrics stay missing; report coverage separately at call site."""
    indices = [i for i, value in enumerate(values) if value is not None]
    if not indices:
        return None
    _, means = object_means([values[i] for i in indices], [rows[i] for i in indices])
    return float(means.mean())
