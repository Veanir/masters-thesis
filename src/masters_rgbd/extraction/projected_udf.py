"""Fixed-budget surface extraction from an unsigned distance field.

The unsigned ``udf == tau`` level set contains two offset sheets.  This
extractor triangulates those sheets on a dense lattice and then takes one
deterministic projection step toward the zero set using gradients derived from
the same lattice.  It is intended for surface-distance metrics only: without a
valid sign or occupancy field the resulting mesh is not a solid and its
winding and watertightness have no volumetric meaning.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import numpy as np

from masters_rgbd.contracts.fields import (
    ExtractionStats,
    FieldModel,
    Mesh,
    QueryBatch,
    QueryBudget,
    ROIBounds,
)
from masters_rgbd.extraction.dense_cells import (
    _CUBE_CORNERS,
    _TETRAHEDRA,
    _triangulate_tetrahedron,
)


@dataclass(frozen=True)
class ProjectedUDFExtractionResult:
    """A metric surface mesh plus diagnostics specific to UDF projection."""

    mesh: Mesh
    surface_points_m: np.ndarray
    stats: ExtractionStats
    tau_m: float
    projection_mode: str
    projection_valid_fraction: float
    roi_boundary_contact_fraction: float

    def __post_init__(self) -> None:
        points = np.array(self.surface_points_m, dtype=np.float64, copy=True)
        if points.ndim != 2 or points.shape[1:] != (3,):
            raise ValueError("surface_points_m must have shape [N, 3]")
        if not np.isfinite(points).all():
            raise ValueError("surface_points_m contains non-finite values")
        if not np.isfinite(self.tau_m) or self.tau_m <= 0.0:
            raise ValueError("tau_m must be positive and finite")
        if self.projection_mode not in {"unit_tau", "newton_tau_capped"}:
            raise ValueError("projection_mode is invalid")
        for name, value in (
            ("projection_valid_fraction", self.projection_valid_fraction),
            ("roi_boundary_contact_fraction", self.roi_boundary_contact_fraction),
        ):
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        points.setflags(write=False)
        object.__setattr__(self, "surface_points_m", points)


def _interpolate_grid_vectors(
    vectors: np.ndarray,
    points: np.ndarray,
    minimum: np.ndarray,
    cell_size: np.ndarray,
    resolution: int,
) -> np.ndarray:
    """Trilinearly interpolate a vector grid at metric-space points."""

    coordinates = np.clip((points - minimum) / cell_size, 0.0, float(resolution))
    lower = np.floor(coordinates).astype(np.int64)
    lower = np.minimum(lower, resolution - 1)
    fraction = coordinates - lower
    result = np.zeros((len(points), 3), dtype=np.float64)
    for x_offset in (0, 1):
        x_weight = fraction[:, 0] if x_offset else 1.0 - fraction[:, 0]
        for y_offset in (0, 1):
            y_weight = fraction[:, 1] if y_offset else 1.0 - fraction[:, 1]
            for z_offset in (0, 1):
                z_weight = fraction[:, 2] if z_offset else 1.0 - fraction[:, 2]
                weight = (x_weight * y_weight * z_weight)[:, None]
                indices = lower + np.asarray((x_offset, y_offset, z_offset))
                result += weight * vectors[indices[:, 0], indices[:, 1], indices[:, 2]]
    return result


def _clean_projected_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    weld_tolerance_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Weld projected sheets and remove degenerate or duplicate triangles."""

    if len(faces) == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.int64)

    quantized = np.rint(vertices / weld_tolerance_m).astype(np.int64)
    _, inverse = np.unique(quantized, axis=0, return_inverse=True)
    group_count = int(inverse.max()) + 1
    welded = np.zeros((group_count, 3), dtype=np.float64)
    np.add.at(welded, inverse, vertices)
    counts = np.bincount(inverse, minlength=group_count)
    welded /= counts[:, None]

    remapped = inverse[faces]
    distinct = (
        (remapped[:, 0] != remapped[:, 1])
        & (remapped[:, 1] != remapped[:, 2])
        & (remapped[:, 0] != remapped[:, 2])
    )
    remapped = remapped[distinct]
    if len(remapped) == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.int64)

    triangles = welded[remapped]
    doubled_area = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    remapped = remapped[doubled_area > 1e-14]
    if len(remapped) == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.int64)

    _, unique_face_indices = np.unique(np.sort(remapped, axis=1), axis=0, return_index=True)
    remapped = remapped[np.sort(unique_face_indices)]
    used, compact_faces = np.unique(remapped.reshape(-1), return_inverse=True)
    compact_vertices = welded[used]
    return compact_vertices, compact_faces.reshape(-1, 3).astype(np.int64, copy=False)


@dataclass(frozen=True)
class DenseProjectedUDFExtractor:
    """Extract a projected UDF surface with one fixed dense query budget."""

    resolution: int = 96
    minimum_gradient_norm: float = 1e-6
    tau_scale: float = 1.0
    projection_mode: Literal["unit_tau", "newton_tau_capped"] = "unit_tau"

    def __post_init__(self) -> None:
        if (
            isinstance(self.resolution, bool)
            or not isinstance(self.resolution, (int, np.integer))
            or self.resolution < 2
        ):
            raise ValueError("resolution must be an integer of at least 2")
        if not np.isfinite(self.minimum_gradient_norm) or self.minimum_gradient_norm <= 0.0:
            raise ValueError("minimum_gradient_norm must be positive and finite")
        if not np.isfinite(self.tau_scale) or self.tau_scale <= 0.0:
            raise ValueError("tau_scale must be positive and finite")
        if self.projection_mode not in {"unit_tau", "newton_tau_capped"}:
            raise ValueError("projection_mode is invalid")

    @property
    def required_query_count(self) -> int:
        return (int(self.resolution) + 1) ** 3

    def extract(
        self,
        field: FieldModel,
        roi: ROIBounds,
        budget: QueryBudget,
    ) -> ProjectedUDFExtractionResult:
        if self.required_query_count > budget.max_field_evaluations:
            raise ValueError(
                f"query budget {budget.max_field_evaluations} is smaller than "
                f"required dense count {self.required_query_count}"
            )

        started = perf_counter()
        minimum = np.asarray(roi.minimum, dtype=np.float64)
        maximum = np.asarray(roi.maximum, dtype=np.float64)
        cell_size = (maximum - minimum) / float(self.resolution)
        cell_diagonal_m = float(np.linalg.norm(cell_size))
        tau_m = self.tau_scale * cell_diagonal_m
        axes = [np.linspace(minimum[axis], maximum[axis], self.resolution + 1) for axis in range(3)]
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
        prediction = field.query(QueryBatch(grid.reshape(-1, 3), roi.frame))
        if len(prediction.udf_m) != self.required_query_count:
            raise ValueError("field prediction count differs from the dense query count")
        udf = prediction.udf_m.reshape(grid.shape[:3])
        level = udf - tau_m

        inside = level <= 0.0
        corners = (
            inside[:-1, :-1, :-1],
            inside[1:, :-1, :-1],
            inside[1:, 1:, :-1],
            inside[:-1, 1:, :-1],
            inside[:-1, :-1, 1:],
            inside[1:, :-1, 1:],
            inside[1:, 1:, 1:],
            inside[:-1, 1:, 1:],
        )
        active = np.logical_or.reduce(corners) & np.logical_or.reduce(
            tuple(~corner for corner in corners)
        )
        active_indices = np.argwhere(active)

        raw_vertices: list[np.ndarray] = []
        raw_faces: list[tuple[int, int, int]] = []
        for index in active_indices:
            cell_minimum = minimum + index * cell_size
            corner_points = cell_minimum + _CUBE_CORNERS * cell_size
            x, y, z = (int(value) for value in index)
            corner_values = np.asarray(
                (
                    level[x, y, z],
                    level[x + 1, y, z],
                    level[x + 1, y + 1, z],
                    level[x, y + 1, z],
                    level[x, y, z + 1],
                    level[x + 1, y, z + 1],
                    level[x + 1, y + 1, z + 1],
                    level[x, y + 1, z + 1],
                ),
                dtype=np.float64,
            )
            for tetrahedron in _TETRAHEDRA:
                for triangle in _triangulate_tetrahedron(corner_points, corner_values, tetrahedron):
                    offset = len(raw_vertices)
                    raw_vertices.extend(triangle)
                    raw_faces.append((offset, offset + 1, offset + 2))

        if raw_vertices:
            vertex_array = np.stack(raw_vertices)
            face_array = np.asarray(raw_faces, dtype=np.int64)
            gradients = np.stack(np.gradient(udf, *cell_size, edge_order=1), axis=-1)
            interpolated = _interpolate_grid_vectors(
                gradients, vertex_array, minimum, cell_size, int(self.resolution)
            )
            gradient_norm = np.linalg.norm(interpolated, axis=1)
            projection_valid = np.isfinite(gradient_norm) & (
                gradient_norm >= self.minimum_gradient_norm
            )
            projection_valid_fraction = float(np.mean(projection_valid))
            projected = np.array(vertex_array, copy=True)
            if self.projection_mode == "unit_tau":
                displacement = (
                    tau_m * interpolated[projection_valid] / gradient_norm[projection_valid, None]
                )
            else:
                displacement = (
                    tau_m
                    * interpolated[projection_valid]
                    / np.square(gradient_norm[projection_valid, None])
                )
                displacement_norm = np.linalg.norm(displacement, axis=1)
                over_cap = displacement_norm > cell_diagonal_m
                displacement[over_cap] *= (cell_diagonal_m / displacement_norm[over_cap])[:, None]
            projected[projection_valid] -= displacement
            projected = np.clip(projected, minimum, maximum)
            face_array = face_array[np.all(projection_valid[face_array], axis=1)]
            weld_tolerance = max(float(np.min(cell_size)) / 8.0, np.finfo(np.float64).eps)
            mesh_vertices, mesh_faces = _clean_projected_mesh(
                projected, face_array, weld_tolerance_m=weld_tolerance
            )
        else:
            projection_valid_fraction = 0.0
            mesh_vertices = np.empty((0, 3), dtype=np.float64)
            mesh_faces = np.empty((0, 3), dtype=np.int64)

        if len(mesh_vertices):
            boundary_band = float(np.min(cell_size))
            boundary_contact = np.any(
                (mesh_vertices - minimum <= boundary_band)
                | (maximum - mesh_vertices <= boundary_band),
                axis=1,
            )
            roi_boundary_contact_fraction = float(np.mean(boundary_contact))
        else:
            roi_boundary_contact_fraction = 0.0

        stats = ExtractionStats(
            requested_point_count=self.required_query_count,
            field_evaluation_count=self.required_query_count,
            cache_hit_count=0,
            active_leaf_count=len(active_indices),
            wall_time_s=perf_counter() - started,
        )
        mesh = Mesh(mesh_vertices, mesh_faces, roi.frame)
        return ProjectedUDFExtractionResult(
            mesh=mesh,
            surface_points_m=mesh.vertices_m,
            stats=stats,
            tau_m=tau_m,
            projection_mode=self.projection_mode,
            projection_valid_fraction=projection_valid_fraction,
            roi_boundary_contact_fraction=roi_boundary_contact_fraction,
        )
