"""Outcome-free geometry diagnostics for a legacy M1 smoke inventory.

This module deliberately does not read the legacy dataset.  Callers provide one
triangulated asset at a time, including the independently known uniform conversion
from source units to metres.  The resulting 80/10/10 split is a deterministic smoke
partition only; it is not the leakage-safe primary v2 research split.

The five audit scores are explicit, quantized geometry proxies.  They are useful for
freezing a diverse diagnostic sample, but they are not semantic ground truth and do
not establish that an asset actually is concave, hollow, slender, or detailed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from numbers import Integral, Real

import numpy as np
from numpy.typing import ArrayLike, NDArray

from masters_rgbd.contracts.manifests import SplitName
from masters_rgbd.data.geometry_fingerprint import (
    DEFAULT_QUANTIZATION_M,
    GeometryFingerprint,
    fingerprint_processed_triangle_geometry,
)
from masters_rgbd.data.gt_audit_selection import AuditCandidate

LEGACY_DIAGNOSTIC_INVENTORY_VERSION = "legacy-m1-diagnostic-inventory-v2"

LEGACY_DIAGNOSTIC_SPLIT_STRATEGY = "geometry-hash-seeded-80-10-10-smoke-v1"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_SCORE_SCALE = 1_000

_MAX_SCORE = 2_000_000_000

_THIN_SLENDER_THRESHOLD = 4.0

_CONCAVE_HOLLOW_THRESHOLD = 0.20

_FINE_DETAIL_THRESHOLD = 32.0

FloatArray = NDArray[np.float64]

IntArray = NDArray[np.int64]


class LegacyDiagnosticRejectionCode(StrEnum):
    """Stable failure reasons at the diagnostic inventory boundary."""

    INVALID_IDENTITY = "invalid_identity"
    INVALID_SCALE = "invalid_scale"
    MALFORMED_VERTICES = "malformed_vertices"
    NONFINITE_VERTICES = "nonfinite_vertices"
    MALFORMED_FACES = "malformed_faces"
    INVALID_FACE_INDEX = "invalid_face_index"
    DEGENERATE_GEOMETRY = "degenerate_geometry"
    INVALID_SEED = "invalid_seed"
    INVALID_SOURCE_FACTS = "invalid_source_facts"
    SOURCE_FACT_MISMATCH = "source_fact_mismatch"


class LegacyDiagnosticRejected(ValueError):
    """Typed rejection that prevents a malformed row from disappearing silently."""

    def __init__(self, code: LegacyDiagnosticRejectionCode, detail: str) -> None:
        super().__init__(f"{code.value}: {detail}")
        self.code = code
        self.detail = detail


def _reject(code: LegacyDiagnosticRejectionCode, detail: str) -> None:
    raise LegacyDiagnosticRejected(code, detail)


@dataclass(frozen=True, slots=True)
class LegacySourceRowFacts:
    """Optional outcome-free counters copied from a legacy inventory row.

    Supplied values are checked against the geometry; they never override measured
    values or scores.  ``vertex_count`` and ``face_count`` refer to the raw supplied
    arrays.  The other counters refer to quantization-welded, duplicate-free triangle
    topology used by this module.
    """

    vertex_count: int | None = None
    face_count: int | None = None
    boundary_edge_count: int | None = None
    component_count: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "vertex_count",
            "face_count",
            "boundary_edge_count",
            "component_count",
        ):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, Integral) or value < 0
            ):
                _reject(
                    LegacyDiagnosticRejectionCode.INVALID_SOURCE_FACTS,
                    f"{name} must be a non-negative integer or None",
                )


@dataclass(frozen=True, slots=True)
class LegacyGeometryProxies:
    """Measured quantities from which the integer audit scores are derived."""

    welded_vertex_count: int
    triangle_count: int
    removed_degenerate_triangle_count: int
    unique_edge_count: int
    boundary_edge_count: int
    nonmanifold_edge_count: int
    component_count: int
    nested_component_pairs: int
    surface_area_m2: float
    bounding_box_volume_m3: float
    enclosed_volume_proxy_m3: float
    boundary_length_fraction: float
    thin_slender_ratio: float
    concave_hollow_deficit: float
    fine_detail_density: float


@dataclass(frozen=True, slots=True)
class LegacyDiagnosticInventoryItem:
    """One validated diagnostic row and its audit-selector representation."""

    audit_candidate: AuditCandidate
    geometry_fingerprint: GeometryFingerprint
    proxies: LegacyGeometryProxies
    uniform_scale_m_per_source_unit: float
    split_seed: int
    source_row_facts: LegacySourceRowFacts | None
    inventory_version: str = LEGACY_DIAGNOSTIC_INVENTORY_VERSION
    split_strategy: str = LEGACY_DIAGNOSTIC_SPLIT_STRATEGY

    @property
    def diagnostic_split(self) -> SplitName:
        return self.audit_candidate.split


@dataclass(frozen=True, slots=True)
class ProcessedLegacyDiagnosticGeometry:
    """Metric cleaned geometry shared by inventory and diagnostic GT smoke."""

    vertices_mesh_local_m: FloatArray
    faces: IntArray
    fingerprint: GeometryFingerprint
    removed_degenerate_triangle_count: int

    def __post_init__(self) -> None:
        vertices = np.array(self.vertices_mesh_local_m, dtype=np.float64, copy=True)
        faces = np.array(self.faces, dtype=np.int64, copy=True)
        vertices.setflags(write=False)
        faces.setflags(write=False)
        object.__setattr__(self, "vertices_mesh_local_m", vertices)
        object.__setattr__(self, "faces", faces)


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, first: int, second: int) -> None:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root == second_root:
            return
        if self.rank[first_root] < self.rank[second_root]:
            first_root, second_root = second_root, first_root
        self.parent[second_root] = first_root
        if self.rank[first_root] == self.rank[second_root]:
            self.rank[first_root] += 1


def _validated_scale(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        _reject(
            LegacyDiagnosticRejectionCode.INVALID_SCALE,
            "uniform scale must be a real scalar",
        )
    scale = float(value)
    if not np.isfinite(scale) or scale <= 0.0:
        _reject(
            LegacyDiagnosticRejectionCode.INVALID_SCALE,
            "uniform scale must be finite and positive",
        )
    return scale


def _validated_mesh(
    vertices: ArrayLike, faces: ArrayLike, *, scale: float
) -> tuple[FloatArray, IntArray]:
    try:
        vertex_array = np.asarray(vertices)
    except (TypeError, ValueError) as error:
        _reject(
            LegacyDiagnosticRejectionCode.MALFORMED_VERTICES,
            f"vertices must be a rectangular numeric array: {error}",
        )
    if vertex_array.ndim != 2 or vertex_array.shape[1:] != (3,) or len(vertex_array) == 0:
        _reject(
            LegacyDiagnosticRejectionCode.MALFORMED_VERTICES,
            f"vertices must have non-empty shape [vertex, xyz], got {vertex_array.shape}",
        )
    if not np.issubdtype(vertex_array.dtype, np.number) or np.issubdtype(
        vertex_array.dtype, np.complexfloating
    ):
        _reject(
            LegacyDiagnosticRejectionCode.MALFORMED_VERTICES,
            "vertices must contain real numeric values",
        )
    vertices_m = np.asarray(vertex_array, dtype=np.float64) * scale
    if not np.isfinite(vertices_m).all():
        _reject(
            LegacyDiagnosticRejectionCode.NONFINITE_VERTICES,
            "scaled vertices contain non-finite values",
        )

    try:
        face_array = np.asarray(faces)
    except (TypeError, ValueError) as error:
        _reject(
            LegacyDiagnosticRejectionCode.MALFORMED_FACES,
            f"faces must be a rectangular integer array: {error}",
        )
    if face_array.ndim != 2 or face_array.shape[1:] != (3,) or len(face_array) == 0:
        _reject(
            LegacyDiagnosticRejectionCode.MALFORMED_FACES,
            f"faces must have non-empty shape [face, 3], got {face_array.shape}",
        )
    if not np.issubdtype(face_array.dtype, np.integer):
        _reject(
            LegacyDiagnosticRejectionCode.MALFORMED_FACES,
            "faces must contain integer vertex indices",
        )
    face_indices = np.asarray(face_array, dtype=np.int64)
    if face_indices.min() < 0 or face_indices.max() >= len(vertices_m):
        _reject(
            LegacyDiagnosticRejectionCode.INVALID_FACE_INDEX,
            "faces contain an index outside vertices",
        )
    return vertices_m, face_indices


def _canonical_topology(
    vertices_m: FloatArray,
    faces: IntArray,
) -> tuple[FloatArray, IntArray, int]:
    scaled = vertices_m / DEFAULT_QUANTIZATION_M
    if not np.isfinite(scaled).all() or float(np.max(np.abs(scaled))) >= float(2**63):
        _reject(
            LegacyDiagnosticRejectionCode.DEGENERATE_GEOMETRY,
            "vertices exceed the diagnostic quantization range",
        )
    quantized = np.asarray(np.rint(scaled), dtype=np.int64)
    welded_quantized, inverse = np.unique(quantized, axis=0, return_inverse=True)
    welded_vertices = welded_quantized.astype(np.float64) * DEFAULT_QUANTIZATION_M
    triangles = inverse[faces]
    ordered = np.sort(triangles, axis=1)
    collapsed = (ordered[:, 0] == ordered[:, 1]) | (ordered[:, 1] == ordered[:, 2])
    removed_count = int(np.count_nonzero(collapsed))
    ordered = ordered[~collapsed]
    if len(ordered) == 0:
        _reject(
            LegacyDiagnosticRejectionCode.DEGENERATE_GEOMETRY,
            "all triangles collapse after metric quantization",
        )
    canonical_triangles = np.unique(ordered, axis=0)
    coordinates = welded_vertices[canonical_triangles]
    twice_area = np.linalg.norm(
        np.cross(coordinates[:, 1] - coordinates[:, 0], coordinates[:, 2] - coordinates[:, 0]),
        axis=1,
    )
    zero_area = twice_area <= 0.0
    removed_count += int(np.count_nonzero(zero_area))
    canonical_triangles = canonical_triangles[~zero_area]
    if len(canonical_triangles) == 0:
        _reject(
            LegacyDiagnosticRejectionCode.DEGENERATE_GEOMETRY,
            "all triangles have zero metric area after quantization",
        )
    used_indices = np.unique(canonical_triangles)
    compact_vertices = welded_vertices[used_indices]
    remap = np.full(len(welded_vertices), -1, dtype=np.int64)
    remap[used_indices] = np.arange(len(used_indices))
    return compact_vertices, remap[canonical_triangles], removed_count


def preprocess_legacy_diagnostic_geometry(
    *,
    vertices: ArrayLike,
    faces: ArrayLike,
    uniform_scale_m_per_source_unit: float,
) -> ProcessedLegacyDiagnosticGeometry:
    """Apply the frozen metric cleanup before fingerprints, proxies, or GT."""

    scale = _validated_scale(uniform_scale_m_per_source_unit)
    vertices_m, face_indices = _validated_mesh(vertices, faces, scale=scale)
    canonical_vertices, canonical_triangles, removed_count = _canonical_topology(
        vertices_m, face_indices
    )
    fingerprint = fingerprint_processed_triangle_geometry(canonical_vertices, canonical_triangles)
    return ProcessedLegacyDiagnosticGeometry(
        vertices_mesh_local_m=canonical_vertices,
        faces=canonical_triangles,
        fingerprint=fingerprint,
        removed_degenerate_triangle_count=removed_count,
    )
