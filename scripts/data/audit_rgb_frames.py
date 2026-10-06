"""Reject empty or effectively blank RGB; this is not a photorealism metric."""

import numpy as np

from scripts.common.paths import script_help

script_help(__doc__, __name__)


def rgb_frame_qa(rgb):
    rgb = np.asarray(rgb)
    valid_shape = rgb.ndim == 3 and rgb.shape[-1] == 3 and rgb.dtype == np.uint8
    if not valid_shape:
        return {"valid": False, "reason": "wrong_shape_or_dtype"}
    maximum = int(rgb.max())
    std = float(rgb.std())
    nonblack = float((rgb.max(axis=-1) > 3).mean())
    return {
        "valid": bool(maximum > 3 and std > 1),
        "max": maximum,
        "std": std,
        "fraction_pixels_above3": nonblack,
        "rule": "uint8 HxWx3 AND global max>3 AND global std>1; reject empty renderer buffers",
    }
