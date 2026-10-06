"""Shared rendered scene arrays."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class RenderedScene:
    rgb: NDArray[np.uint8]
    depth: NDArray[np.float32]
    instance_mask: NDArray[np.uint16]
