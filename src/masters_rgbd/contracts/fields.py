"""Model-independent field, extraction, and prediction contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from masters_rgbd.contracts.geometry import CoordinateFrame

FloatArray = NDArray[np.float64]

BoolArray = NDArray[np.bool_]

IntArray = NDArray[np.int64]


def _readonly_array(value: object, *, dtype: object, shape_tail: tuple[int, ...], name: str):
    array = np.array(value, dtype=dtype, copy=True)
    if array.ndim != len(shape_tail) + 1 or array.shape[1:] != shape_tail:
        suffix = ", ".join(str(value) for value in shape_tail)
        raise ValueError(f"{name} must have shape [N, {suffix}], got {array.shape}")
    if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    array.setflags(write=False)
    return array


def _readonly_vector(value: object, *, dtype: object, length: int, name: str):
    array = np.array(value, dtype=dtype, copy=True)
    if array.shape != (length,):
        raise ValueError(f"{name} must have shape [{length}], got {array.shape}")
    if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class QueryBatch:
    points: FloatArray
    frame: CoordinateFrame

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "points",
            _readonly_array(self.points, dtype=np.float64, shape_tail=(3,), name="points"),
        )

    def __len__(self) -> int:
        return len(self.points)


@dataclass(frozen=True)
class GroundTruthValues:
    udf_m: FloatArray
    occupied: BoolArray
    occupancy_valid: BoolArray

    def __post_init__(self) -> None:
        count = len(np.asarray(self.udf_m))
        udf = _readonly_vector(self.udf_m, dtype=np.float64, length=count, name="udf_m")
        occupied = _readonly_vector(
            self.occupied,
            dtype=np.bool_,
            length=count,
            name="occupied",
        )
        occupancy_valid = _readonly_vector(
            self.occupancy_valid,
            dtype=np.bool_,
            length=count,
            name="occupancy_valid",
        )
        if np.any(udf < 0.0):
            raise ValueError("UDF values cannot be negative")
        occupied = np.asarray(occupied & occupancy_valid, dtype=np.bool_)
        occupied.setflags(write=False)
        object.__setattr__(self, "udf_m", udf)
        object.__setattr__(self, "occupied", occupied)
        object.__setattr__(self, "occupancy_valid", occupancy_valid)


@dataclass(frozen=True)
class FieldPrediction:
    udf_m: FloatArray
    occupancy_logit: FloatArray

    def __post_init__(self) -> None:
        count = len(np.asarray(self.udf_m))
        udf = _readonly_vector(self.udf_m, dtype=np.float64, length=count, name="udf_m")
        logits = _readonly_vector(
            self.occupancy_logit,
            dtype=np.float64,
            length=count,
            name="occupancy_logit",
        )
        if np.any(udf < 0.0):
            raise ValueError("predicted UDF values cannot be negative")
        object.__setattr__(self, "udf_m", udf)
        object.__setattr__(self, "occupancy_logit", logits)


@runtime_checkable
class FieldModel(Protocol):
    def query(self, queries: QueryBatch) -> FieldPrediction: ...


@dataclass(frozen=True)
class ROIBounds:
    minimum: tuple[float, float, float]
    maximum: tuple[float, float, float]
    frame: CoordinateFrame = CoordinateFrame.CAMERA

    def __post_init__(self) -> None:
        minimum = np.asarray(self.minimum, dtype=np.float64)
        maximum = np.asarray(self.maximum, dtype=np.float64)
        if minimum.shape != (3,) or maximum.shape != (3,):
            raise ValueError("ROI bounds must be vector3 values")
        if not np.isfinite(minimum).all() or not np.isfinite(maximum).all():
            raise ValueError("ROI bounds must be finite")
        if np.any(maximum <= minimum):
            raise ValueError("ROI maximum must be greater than minimum on every axis")


@dataclass(frozen=True)
class QueryBudget:
    max_field_evaluations: int

    def __post_init__(self) -> None:
        if self.max_field_evaluations <= 0:
            raise ValueError("query budget must be positive")


@dataclass(frozen=True)
class Mesh:
    vertices_m: FloatArray
    faces: IntArray
    frame: CoordinateFrame

    def __post_init__(self) -> None:
        vertices = _readonly_array(
            self.vertices_m,
            dtype=np.float64,
            shape_tail=(3,),
            name="vertices_m",
        )
        faces = _readonly_array(self.faces, dtype=np.int64, shape_tail=(3,), name="faces")
        if len(faces) and (np.any(faces < 0) or np.any(faces >= len(vertices))):
            raise ValueError("mesh faces contain invalid vertex indices")
        object.__setattr__(self, "vertices_m", vertices)
        object.__setattr__(self, "faces", faces)


@dataclass(frozen=True)
class AdaptiveLeaf:
    minimum: tuple[float, float, float]
    maximum: tuple[float, float, float]
    level: int | None
    frame: CoordinateFrame

    def __post_init__(self) -> None:
        if self.level is not None and self.level < 0:
            raise ValueError("leaf level cannot be negative")
        ROIBounds(self.minimum, self.maximum, self.frame)


@dataclass(frozen=True)
class ExtractionStats:
    requested_point_count: int
    field_evaluation_count: int
    cache_hit_count: int
    active_leaf_count: int
    wall_time_s: float

    def __post_init__(self) -> None:
        if (
            self.requested_point_count < 0
            or self.field_evaluation_count < 0
            or self.cache_hit_count < 0
            or self.active_leaf_count < 0
            or self.wall_time_s < 0.0
        ):
            raise ValueError("extraction statistics cannot be negative")
        if self.field_evaluation_count > self.requested_point_count:
            raise ValueError("field evaluations cannot exceed requested points")
        if self.requested_point_count - self.field_evaluation_count != self.cache_hit_count:
            raise ValueError("requested points must equal field evaluations plus cache hits")

    @property
    def query_count(self) -> int:
        """Canonical scientific query count: actual field evaluations."""

        return self.field_evaluation_count


@dataclass(frozen=True)
class ExtractionResult:
    mesh: Mesh
    surface_points_m: FloatArray
    leaves: tuple[AdaptiveLeaf, ...]
    stats: ExtractionStats

    def __post_init__(self) -> None:
        points = _readonly_array(
            self.surface_points_m,
            dtype=np.float64,
            shape_tail=(3,),
            name="surface_points_m",
        )
        if self.mesh.frame != (self.leaves[0].frame if self.leaves else self.mesh.frame):
            raise ValueError("mesh and leaves must use the same frame")
        object.__setattr__(self, "surface_points_m", points)
