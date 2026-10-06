"""One-command, read-only freeze of the legacy M1 diagnostic asset list."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

IntArray = NDArray[np.int64]


class LegacyManifestRejectionCode(StrEnum):
    MANIFEST_INTEGRITY = "manifest_integrity"
    MALFORMED_ROW = "malformed_row"
    INCONSISTENT_ASSET = "inconsistent_asset"
    UNSAFE_MESH_PATH = "unsafe_mesh_path"
    MISSING_MESH = "missing_mesh"
    DIRTY_SOURCE = "dirty_source"


class LegacyManifestRejected(ValueError):
    def __init__(self, code: LegacyManifestRejectionCode, detail: str) -> None:
        super().__init__(f"{code.value}: {detail}")
        self.code = code
        self.detail = detail


class MeshLoadRejected(ValueError):
    """A mesh could not be reduced to one finite triangle array."""


def _reject(code: LegacyManifestRejectionCode, detail: str) -> None:
    raise LegacyManifestRejected(code, detail)


def load_trimesh_geometry(path: Path) -> tuple[FloatArray, IntArray, str]:
    """Load one GLB scene with node transforms baked into a triangle mesh."""

    try:
        import trimesh

        scene = trimesh.load_scene(path, process=False)
        mesh = scene.to_geometry()
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)
    except Exception as error:  # external parser exceptions are not stable across releases
        raise MeshLoadRejected(f"{path.name}: {type(error).__name__}: {error}") from error
    if (
        vertices.ndim != 2
        or vertices.shape[1:] != (3,)
        or len(vertices) == 0
        or not np.isfinite(vertices).all()
        or faces.ndim != 2
        or faces.shape[1:] != (3,)
        or len(faces) == 0
    ):
        raise MeshLoadRejected(f"{path.name}: loader returned invalid triangle arrays")
    return vertices, faces, f"trimesh-{trimesh.__version__}-process-false-to-geometry"
