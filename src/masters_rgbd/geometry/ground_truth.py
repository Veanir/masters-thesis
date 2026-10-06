"""Production dual ground-truth queries for mesh surfaces and solid grids."""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from masters_rgbd.contracts.fields import GroundTruthValues, QueryBatch
from masters_rgbd.contracts.geometry import AffineTransform, CoordinateFrame

FloatArray = NDArray[np.float64]

BoolArray = NDArray[np.bool_]

IntArray = NDArray[np.int64]


def _points3(value: object, *, name: str, allow_empty: bool = False) -> FloatArray:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{name} must have shape [N, 3], got {points.shape}")
    if not allow_empty and len(points) == 0:
        raise ValueError(f"{name} cannot be empty")
    if not np.isfinite(points).all():
        raise ValueError(f"{name} contains non-finite values")
    return points


def _segment_distance_squared(
    points: FloatArray,
    start: FloatArray,
    end: FloatArray,
) -> FloatArray:
    segment = end - start
    denominator = np.sum(segment * segment, axis=-1)
    numerator = np.sum((points - start) * segment, axis=-1)
    fraction = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0.0,
    )
    fraction = np.clip(fraction, 0.0, 1.0)
    nearest = start + fraction[..., None] * segment
    difference = points - nearest
    return np.sum(difference * difference, axis=-1)


def point_triangle_distance_squared(
    points_m: object,
    triangles_m: object,
    *,
    relative_degeneracy_factor: float = 64.0,
) -> FloatArray:
    """Return all pairwise squared distances for points and triangles.

    The result has shape ``[point_count, triangle_count]``. Degenerate triangles
    fall back to their three edges, so broken source meshes cannot introduce
    NaNs into UDF labels.
    """

    points = _points3(points_m, name="points_m", allow_empty=True)
    triangles = np.asarray(triangles_m, dtype=np.float64)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
        raise ValueError(f"triangles_m must have shape [T, 3, 3], got {triangles.shape}")
    if not np.isfinite(triangles).all():
        raise ValueError("triangles_m contains non-finite values")
    if not np.isfinite(relative_degeneracy_factor) or relative_degeneracy_factor <= 0.0:
        raise ValueError("relative_degeneracy_factor must be positive and finite")
    if len(points) == 0 or len(triangles) == 0:
        return np.empty((len(points), len(triangles)), dtype=np.float64)

    point_grid = points[:, None, :]
    a = triangles[None, :, 0, :]
    b = triangles[None, :, 1, :]
    c = triangles[None, :, 2, :]
    ab = b - a
    ac = c - a
    bc = c - b
    relative = point_grid - a

    d00 = np.sum(ab * ab, axis=-1)
    d11 = np.sum(ac * ac, axis=-1)
    normal = np.cross(ab, ac)
    normal_squared = np.sum(normal * normal, axis=-1)
    maximum_edge_squared = np.maximum.reduce((d00, d11, np.sum(bc * bc, axis=-1)))
    relative_cross_tolerance = (
        relative_degeneracy_factor * np.finfo(np.float64).eps * maximum_edge_squared
    )
    valid_triangle = normal_squared > relative_cross_tolerance * relative_cross_tolerance
    inverse_denominator = np.divide(
        1.0,
        normal_squared,
        out=np.zeros_like(normal_squared),
        where=valid_triangle,
    )
    barycentric_b = np.sum(np.cross(relative, ac) * normal, axis=-1) * inverse_denominator
    barycentric_c = np.sum(np.cross(ab, relative) * normal, axis=-1) * inverse_denominator
    barycentric_a = 1.0 - barycentric_b - barycentric_c
    projection_inside = (
        valid_triangle & (barycentric_a >= 0.0) & (barycentric_b >= 0.0) & (barycentric_c >= 0.0)
    )

    signed_plane_numerator = np.sum(relative * normal, axis=-1)
    plane_distance_squared = np.divide(
        signed_plane_numerator * signed_plane_numerator,
        normal_squared,
        out=np.full_like(signed_plane_numerator, np.inf),
        where=valid_triangle,
    )
    edge_distance_squared = np.minimum.reduce(
        (
            _segment_distance_squared(point_grid, a, b),
            _segment_distance_squared(point_grid, b, c),
            _segment_distance_squared(point_grid, c, a),
        )
    )
    return np.where(projection_inside, plane_distance_squared, edge_distance_squared)


def _aabb_distance_squared(point: FloatArray, minimum: FloatArray, maximum: FloatArray) -> float:
    offset = np.maximum(np.maximum(minimum - point, point - maximum), 0.0)
    return float(np.dot(offset, offset))


@dataclass(frozen=True)
class _BvhNode:
    minimum: FloatArray
    maximum: FloatArray
    left: int = -1
    right: int = -1
    triangle_indices: IntArray | None = None


class _TriangleAabbBvh:
    """Deterministic exact nearest-triangle index; no approximate pruning."""

    def __init__(self, triangles: FloatArray, *, leaf_size: int) -> None:
        if leaf_size <= 0:
            raise ValueError("BVH leaf_size must be positive")
        self._triangles = triangles
        self._triangle_minimum = triangles.min(axis=1)
        self._triangle_maximum = triangles.max(axis=1)
        self._centroids = (self._triangle_minimum + self._triangle_maximum) * 0.5
        self._leaf_size = leaf_size
        self._nodes: list[_BvhNode] = []
        self._build(np.arange(len(triangles), dtype=np.int64))

    def _build(self, triangle_indices: IntArray) -> int:
        node_index = len(self._nodes)
        minimum = np.nextafter(
            self._triangle_minimum[triangle_indices].min(axis=0),
            -np.inf,
        )
        maximum = np.nextafter(
            self._triangle_maximum[triangle_indices].max(axis=0),
            np.inf,
        )
        self._nodes.append(_BvhNode(minimum, maximum))
        if len(triangle_indices) <= self._leaf_size:
            frozen_indices = np.array(triangle_indices, copy=True)
            frozen_indices.setflags(write=False)
            self._nodes[node_index] = _BvhNode(
                minimum,
                maximum,
                triangle_indices=frozen_indices,
            )
            return node_index

        centroid_extent = np.ptp(self._centroids[triangle_indices], axis=0)
        axis = int(np.argmax(centroid_extent))
        order = np.argsort(self._centroids[triangle_indices, axis], kind="stable")
        ordered_indices = triangle_indices[order]
        midpoint = len(ordered_indices) // 2
        left = self._build(ordered_indices[:midpoint])
        right = self._build(ordered_indices[midpoint:])
        self._nodes[node_index] = _BvhNode(minimum, maximum, left=left, right=right)
        return node_index

    def distances_squared(self, points: FloatArray) -> FloatArray:
        result = np.empty((len(points),), dtype=np.float64)
        for point_index, point in enumerate(points):
            best = np.inf
            queue: list[tuple[float, int]] = [
                (_aabb_distance_squared(point, self._nodes[0].minimum, self._nodes[0].maximum), 0)
            ]
            while queue:
                lower_bound, node_index = heapq.heappop(queue)
                if lower_bound >= best:
                    break
                node = self._nodes[node_index]
                if node.triangle_indices is not None:
                    leaf_distances = point_triangle_distance_squared(
                        point[None, :],
                        self._triangles[node.triangle_indices],
                    )
                    best = min(best, float(leaf_distances.min()))
                    continue
                for child_index in (node.left, node.right):
                    child = self._nodes[child_index]
                    child_bound = _aabb_distance_squared(point, child.minimum, child.maximum)
                    if child_bound < best:
                        heapq.heappush(queue, (child_bound, child_index))
            result[point_index] = best
        return result


@dataclass(frozen=True)
class TriangleSurface:
    vertices_mesh_local_m: FloatArray
    faces: IntArray
    bvh_leaf_size: int = 16
    _bvh: _TriangleAabbBvh = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        vertices = np.array(self.vertices_mesh_local_m, dtype=np.float64, copy=True)
        raw_faces = np.asarray(self.faces)
        if not np.issubdtype(raw_faces.dtype, np.integer) or np.issubdtype(
            raw_faces.dtype, np.bool_
        ):
            raise ValueError("surface faces must contain integer vertex indices")
        faces = np.array(raw_faces, dtype=np.int64, copy=True)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
            raise ValueError("surface vertices must have shape [N, 3] and be non-empty")
        if not np.isfinite(vertices).all():
            raise ValueError("surface vertices contain non-finite values")
        if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
            raise ValueError("surface faces must have shape [T, 3] and be non-empty")
        if np.any(faces < 0) or np.any(faces >= len(vertices)):
            raise ValueError("surface faces contain invalid vertex indices")
        if (
            isinstance(self.bvh_leaf_size, bool)
            or not isinstance(self.bvh_leaf_size, (int, np.integer))
            or self.bvh_leaf_size <= 0
        ):
            raise ValueError("bvh_leaf_size must be a positive integer")
        vertices.setflags(write=False)
        faces.setflags(write=False)
        object.__setattr__(self, "vertices_mesh_local_m", vertices)
        object.__setattr__(self, "faces", faces)
        object.__setattr__(
            self, "_bvh", _TriangleAabbBvh(vertices[faces], leaf_size=self.bvh_leaf_size)
        )

    @property
    def triangles_mesh_local_m(self) -> FloatArray:
        return self.vertices_mesh_local_m[self.faces]

    def udf_m(
        self,
        points_mesh_local_m: object,
    ) -> FloatArray:
        points = _points3(points_mesh_local_m, name="points_mesh_local_m", allow_empty=True)
        if len(points) == 0:
            return np.empty((0,), dtype=np.float64)
        return np.sqrt(np.maximum(self._bvh.distances_squared(points), 0.0))

    def udf_m_oracle(
        self,
        points_mesh_local_m: object,
        *,
        point_chunk_size: int = 512,
        triangle_chunk_size: int = 2048,
    ) -> FloatArray:
        points = _points3(points_mesh_local_m, name="points_mesh_local_m", allow_empty=True)
        if point_chunk_size <= 0 or triangle_chunk_size <= 0:
            raise ValueError("UDF chunk sizes must be positive")
        if len(points) == 0:
            return np.empty((0,), dtype=np.float64)
        triangles = self.triangles_mesh_local_m
        result = np.full((len(points),), np.inf, dtype=np.float64)
        for point_start in range(0, len(points), point_chunk_size):
            point_stop = min(point_start + point_chunk_size, len(points))
            point_chunk = points[point_start:point_stop]
            chunk_minimum = np.full((len(point_chunk),), np.inf, dtype=np.float64)
            for triangle_start in range(0, len(triangles), triangle_chunk_size):
                triangle_stop = min(triangle_start + triangle_chunk_size, len(triangles))
                distances_squared = point_triangle_distance_squared(
                    point_chunk,
                    triangles[triangle_start:triangle_stop],
                )
                chunk_minimum = np.minimum(chunk_minimum, distances_squared.min(axis=1))
            result[point_start:point_stop] = np.sqrt(chunk_minimum)
        return result


class GroundTruthDomainError(ValueError):
    """Raised when a query falls outside the frozen solid-label domain."""


@dataclass(frozen=True)
class SolidOccupancyGrid:
    occupancy_xyz: BoolArray
    origin_mesh_local_m: tuple[float, float, float]
    voxel_size_m: float
    outside_domain_policy: Literal["free", "raise"] = "free"

    def __post_init__(self) -> None:
        occupancy = np.array(self.occupancy_xyz, dtype=np.bool_, copy=True)
        if occupancy.ndim != 3 or any(size <= 0 for size in occupancy.shape):
            raise ValueError("solid occupancy must be a non-empty [X, Y, Z] array")
        origin = np.asarray(self.origin_mesh_local_m, dtype=np.float64)
        if origin.shape != (3,) or not np.isfinite(origin).all():
            raise ValueError("solid origin must be a finite vector3")
        if not np.isfinite(self.voxel_size_m) or self.voxel_size_m <= 0.0:
            raise ValueError("solid voxel_size_m must be positive and finite")
        if self.outside_domain_policy not in ("free", "raise"):
            raise ValueError("outside_domain_policy must be 'free' or 'raise'")
        occupancy.setflags(write=False)
        object.__setattr__(self, "occupancy_xyz", occupancy)
        object.__setattr__(self, "origin_mesh_local_m", tuple(float(value) for value in origin))
        object.__setattr__(self, "voxel_size_m", float(self.voxel_size_m))

    @property
    def maximum_mesh_local_m(self) -> FloatArray:
        return np.asarray(self.origin_mesh_local_m) + self.voxel_size_m * np.asarray(
            self.occupancy_xyz.shape
        )

    def query(self, points_mesh_local_m: object) -> BoolArray:
        points = _points3(points_mesh_local_m, name="points_mesh_local_m", allow_empty=True)
        if len(points) == 0:
            return np.empty((0,), dtype=np.bool_)
        origin = np.asarray(self.origin_mesh_local_m, dtype=np.float64)
        indices = np.floor((points - origin) / self.voxel_size_m).astype(np.int64)
        shape = np.asarray(self.occupancy_xyz.shape, dtype=np.int64)
        valid = np.all((indices >= 0) & (indices < shape), axis=1)
        if not np.all(valid) and self.outside_domain_policy == "raise":
            bad_indices = np.flatnonzero(~valid)
            preview = ", ".join(str(int(value)) for value in bad_indices[:5])
            raise GroundTruthDomainError(
                f"{len(bad_indices)} solid queries fall outside the half-open grid domain; "
                f"query indices: {preview}"
            )
        result = np.zeros((len(points),), dtype=np.bool_)
        valid_indices = indices[valid]
        result[valid] = self.occupancy_xyz[
            valid_indices[:, 0], valid_indices[:, 1], valid_indices[:, 2]
        ]
        result.setflags(write=False)
        return result

    def require_surface_coverage(
        self,
        vertices_mesh_local_m: object,
        *,
        minimum_padding_voxels: int = 1,
    ) -> None:
        vertices = _points3(vertices_mesh_local_m, name="vertices_mesh_local_m")
        if minimum_padding_voxels < 1:
            raise ValueError("minimum_padding_voxels must be at least one")
        padding_m = minimum_padding_voxels * self.voxel_size_m
        valid_minimum = np.asarray(self.origin_mesh_local_m) + padding_m
        valid_maximum = self.maximum_mesh_local_m - padding_m
        if np.any(vertices.min(axis=0) < valid_minimum) or np.any(
            vertices.max(axis=0) > valid_maximum
        ):
            raise ValueError("solid lattice does not contain the surface with required padding")


@dataclass(frozen=True)
class DualMeshGroundTruth:
    surface: TriangleSurface
    solid: SolidOccupancyGrid | None
    camera_T_mesh: AffineTransform
    occupancy_valid: bool
    _mesh_T_camera: AffineTransform = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if (
            self.camera_T_mesh.source != CoordinateFrame.MESH_LOCAL
            or self.camera_T_mesh.target != CoordinateFrame.CAMERA
            or not self.camera_T_mesh.is_rigid()
        ):
            raise ValueError("dual GT requires a rigid camera_T_mesh")
        if not isinstance(self.occupancy_valid, (bool, np.bool_)):
            raise ValueError("occupancy_valid must be a boolean certificate state")
        if self.occupancy_valid and self.solid is None:
            raise ValueError("occupancy-valid dual GT requires a solid grid")
        if not self.occupancy_valid and self.solid is not None:
            raise ValueError("occupancy-invalid dual GT cannot carry a solid grid")
        if self.occupancy_valid:
            assert self.solid is not None
            self.solid.require_surface_coverage(self.surface.vertices_mesh_local_m)
        object.__setattr__(self, "occupancy_valid", bool(self.occupancy_valid))
        object.__setattr__(self, "_mesh_T_camera", self.camera_T_mesh.inverse())

    def query(self, queries: QueryBatch) -> GroundTruthValues:
        if queries.frame != CoordinateFrame.CAMERA:
            raise ValueError("dual GT public queries must use camera frame")
        points_mesh = self._mesh_T_camera.transform_points(queries.points)
        udf = self.surface.udf_m(points_mesh)
        if self.occupancy_valid:
            assert self.solid is not None
            occupied = self.solid.query(points_mesh)
        else:
            occupied = np.zeros(len(points_mesh), dtype=np.bool_)
        return GroundTruthValues(
            udf_m=udf,
            occupied=occupied,
            occupancy_valid=np.full(len(points_mesh), self.occupancy_valid, dtype=np.bool_),
        )
