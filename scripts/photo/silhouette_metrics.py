"""The screened silhouette metric, including explicit empty-mask behavior."""

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

from scripts.common.paths import script_help

script_help(__doc__, __name__)


def metrics(source, prediction, thresholds, condition):
    assert (
        source.shape == prediction.shape == (480, 640)
        and source.dtype == prediction.dtype == bool
        and source.any()
    )
    union = source | prediction
    iou = float((source & prediction).sum() / union.sum())
    a = source & ~binary_erosion(source)
    b = prediction & ~binary_erosion(prediction)
    ab = float(np.quantile(distance_transform_edt(~b)[a], 0.95)) if b.any() else None
    ba = float(np.quantile(distance_transform_edt(~a)[b], 0.95)) if b.any() else None
    threshold = thresholds["source_IoU_min" if condition == "source" else "edited_IoU_min"]
    return {
        "IoU": iou,
        "GT_to_SAM_boundary_p95_at640": ab,
        "SAM_to_GT_boundary_p95_at640": ba,
        "raw_screen_pass": bool(
            iou >= threshold
            and ab is not None
            and max(ab, ba) <= thresholds["max_direction_boundary_p95_pixels_at640width"]
        ),
    }
