"""Frozen procedural solids for the M1 overfit-readiness source gate.

This module only materializes the predeclared source cohort.  It deliberately
does not call topology, solidification, raster, phase, resolution, manual-QC,
or certificate code.  Those outcomes belong to a later, independently recorded
audit so the source selection is visibly frozen before its results are known.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

GENERATOR_ALGORITHM = "masters-rgbd-deterministic-procedural-triangle-mesh-v1"

FloatArray = NDArray[np.float64]

IntArray = NDArray[np.int64]


class ProceduralSourceError(ValueError):
    """The source freeze or a generated artifact violates its declared contract."""


@dataclass(frozen=True, slots=True)
class TriangleMesh:
    vertices_m: FloatArray
    faces: IntArray

    def __post_init__(self) -> None:
        vertices = np.asarray(self.vertices_m, dtype=np.float64)
        faces = np.asarray(self.faces, dtype=np.int64)
        if vertices.ndim != 2 or vertices.shape[1:] != (3,) or len(vertices) < 4:
            raise ProceduralSourceError("vertices_m must have shape [N>=4, 3]")
        if faces.ndim != 2 or faces.shape[1:] != (3,) or len(faces) < 4:
            raise ProceduralSourceError("faces must have triangulated shape [M>=4, 3]")
        if not np.isfinite(vertices).all():
            raise ProceduralSourceError("vertices_m must be finite")
        if faces.min() < 0 or faces.max() >= len(vertices):
            raise ProceduralSourceError("faces reference vertices outside the mesh")
        vertices = np.array(vertices, dtype=np.float64, copy=True, order="C")
        faces = np.array(faces, dtype=np.int64, copy=True, order="C")
        vertices.setflags(write=False)
        faces.setflags(write=False)
        object.__setattr__(self, "vertices_m", vertices)
        object.__setattr__(self, "faces", faces)
