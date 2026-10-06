"""Surface distance metrics shared by the pilot evaluators."""

import numpy as np

from scripts.common.paths import script_help

script_help(__doc__, __name__)


def metrics(forward, backward):
    result = {}
    for threshold in (0.002, 0.005, 0.010):
        p, r = float(np.mean(forward <= threshold)), float(np.mean(backward <= threshold))
        result[f"{threshold:.3f}"] = {
            "precision": p,
            "recall": r,
            "fscore": 2 * p * r / (p + r) if p + r else 0.0,
            "chamfer_mean_m": float((np.mean(forward) + np.mean(backward)) / 2),
        }
    return result
