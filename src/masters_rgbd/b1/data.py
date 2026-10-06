"""Deterministic RGB-D and field-query data for the B1 vertical slice.

The renderer deliberately has no scene or background input: it rasterizes only
the target mesh into the frozen P0 camera convention.  Query randomness is a
function of the frozen sampling seed and shard index only; asset identifiers do
not enter this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from numpy.typing import NDArray

from masters_rgbd.contracts.fields import GroundTruthValues, QueryBatch
from masters_rgbd.contracts.geometry import (
    AffineTransform,
    CameraIntrinsics,
    CoordinateFrame,
    ModelROI,
)
from masters_rgbd.geometry.ground_truth import (
    DualMeshGroundTruth,
    SolidOccupancyGrid,
    TriangleSurface,
)

BoolArray = NDArray[np.bool_]

Float32Array = NDArray[np.float32]

Float64Array = NDArray[np.float64]


def _readonly_array(
    value: object,
    *,
    dtype: object,
    shape: tuple[int | None, ...],
    name: str,
) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True, order="C")
    if array.ndim != len(shape) or any(
        expected is not None and observed != expected
        for observed, expected in zip(array.shape, shape, strict=True)
    ):
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class QuerySamplingConfig:
    """Frozen near-surface/uniform mixture for independently replayable shards."""

    near_surface_count: int = 2_048
    uniform_count: int = 2_048
    base_seed: int = 20_260_826
    near_surface_sigma_voxels: float = 2.0
    near_surface_sigma_m: float | None = None

    def __post_init__(self) -> None:
        for name in ("near_surface_count", "uniform_count", "base_seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer")
        if self.near_surface_count <= 0 or self.uniform_count <= 0:
            raise ValueError("both query mixture counts must be positive")
        if self.base_seed < 0 or self.base_seed > np.iinfo(np.uint32).max:
            raise ValueError("base_seed must fit uint32")
        if not np.isfinite(self.near_surface_sigma_voxels) or self.near_surface_sigma_voxels <= 0.0:
            raise ValueError("near_surface_sigma_voxels must be positive and finite")
        if self.near_surface_sigma_m is not None and (
            not np.isfinite(self.near_surface_sigma_m) or self.near_surface_sigma_m <= 0.0
        ):
            raise ValueError("near_surface_sigma_m must be positive and finite or None")
        object.__setattr__(self, "near_surface_count", int(self.near_surface_count))
        object.__setattr__(self, "uniform_count", int(self.uniform_count))
        object.__setattr__(self, "base_seed", int(self.base_seed))


@dataclass(frozen=True, slots=True)
class RGBDObservation:
    """Target-only fixed-view tensors and their visible-point-derived ROI."""

    rgb: Float32Array
    depth_m: Float32Array
    mask: BoolArray
    intrinsics: CameraIntrinsics
    partial_pixels_uv: Float64Array
    partial_points_camera_m: Float64Array
    model_roi: ModelROI
    camera_T_mesh: AffineTransform

    def __post_init__(self) -> None:
        if not isinstance(self.intrinsics, CameraIntrinsics):
            raise TypeError("intrinsics must be CameraIntrinsics")
        height, width = self.intrinsics.height, self.intrinsics.width
        rgb = _readonly_array(
            self.rgb,
            dtype=np.float32,
            shape=(height, width, 3),
            name="rgb",
        )
        depth = _readonly_array(
            self.depth_m,
            dtype=np.float32,
            shape=(height, width),
            name="depth_m",
        )
        mask = _readonly_array(
            self.mask,
            dtype=np.bool_,
            shape=(height, width),
            name="mask",
        )
        points = _readonly_array(
            self.partial_points_camera_m,
            dtype=np.float64,
            shape=(None, 3),
            name="partial_points_camera_m",
        )
        pixels = _readonly_array(
            self.partial_pixels_uv,
            dtype=np.float64,
            shape=(len(points), 2),
            name="partial_pixels_uv",
        )
        if not np.any(mask):
            raise ValueError("rendered target mask cannot be empty")
        if np.any(depth[mask] <= 0.0) or np.any(depth[~mask] != 0.0):
            raise ValueError("depth must be positive on target and zero outside")
        if np.any(rgb < 0.0) or np.any(rgb > 1.0) or np.any(rgb[~mask] != 0.0):
            raise ValueError("RGB must be in [0, 1] and zero outside the target")
        if len(points) == 0:
            raise ValueError("partial point cloud cannot be empty")
        integer_pixels = pixels.astype(np.int64)
        if not np.array_equal(pixels, integer_pixels.astype(np.float64)):
            raise ValueError("partial pixels must use integer target-pixel centers")
        columns = integer_pixels[:, 0]
        rows = integer_pixels[:, 1]
        if (
            np.any(columns < 0)
            or np.any(columns >= width)
            or np.any(rows < 0)
            or np.any(rows >= height)
        ):
            raise ValueError("partial pixels fall outside the RGB-D image")
        flat_indices = rows * width + columns
        if np.any(np.diff(flat_indices) <= 0):
            raise ValueError("partial pixels must be a unique row-major ordered subset")
        if not np.all(mask[rows, columns]):
            raise ValueError("partial pixels must be an exact subset of the target mask")
        expected_points = self.intrinsics.unproject(pixels, depth[rows, columns])
        if not np.allclose(points, expected_points, rtol=0.0, atol=2e-7):
            raise ValueError("partial points differ from RGB-D unprojection")
        if not np.allclose(points[:, 2], -depth[rows, columns], rtol=0.0, atol=2e-7):
            raise ValueError("P0 camera depth must equal -z_camera")
        if not isinstance(self.model_roi, ModelROI):
            raise TypeError("model_roi must be ModelROI")
        if (
            self.camera_T_mesh.source != CoordinateFrame.MESH_LOCAL
            or self.camera_T_mesh.target != CoordinateFrame.CAMERA
            or not self.camera_T_mesh.is_rigid()
        ):
            raise ValueError("camera_T_mesh must be a rigid mesh-local to camera transform")
        object.__setattr__(self, "rgb", rgb)
        object.__setattr__(self, "depth_m", depth)
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "partial_pixels_uv", pixels)
        object.__setattr__(self, "partial_points_camera_m", points)


@dataclass(frozen=True, slots=True)
class B1FixedViewSample:
    observation: RGBDObservation
    surface: TriangleSurface
    solid: SolidOccupancyGrid | None
    ground_truth: DualMeshGroundTruth

    def __post_init__(self) -> None:
        if not isinstance(self.observation, RGBDObservation):
            raise TypeError("observation must be RGBDObservation")
        if not isinstance(self.surface, TriangleSurface):
            raise TypeError("surface must be TriangleSurface")
        if self.solid is not None and not isinstance(self.solid, SolidOccupancyGrid):
            raise TypeError("solid must be SolidOccupancyGrid or None")
        if not isinstance(self.ground_truth, DualMeshGroundTruth):
            raise TypeError("ground_truth must be DualMeshGroundTruth")
        if self.ground_truth.occupancy_valid != (self.solid is not None):
            raise ValueError("solid presence must equal the ground-truth occupancy certificate")
        if self.ground_truth.solid is not self.solid:
            raise ValueError("sample and ground truth must reference the same solid grid")
        if not np.array_equal(
            self.ground_truth.camera_T_mesh.matrix,
            self.observation.camera_T_mesh.matrix,
        ):
            raise ValueError("observation and ground truth camera transforms differ")


@dataclass(frozen=True, slots=True)
class B1QueryShard:
    shard_index: int
    queries_camera: QueryBatch
    points_model_roi: Float64Array
    values: GroundTruthValues
    near_surface_mask: BoolArray

    def __post_init__(self) -> None:
        if (
            isinstance(self.shard_index, bool)
            or not isinstance(self.shard_index, Integral)
            or self.shard_index < 0
        ):
            raise ValueError("shard_index must be a non-negative integer")
        if self.queries_camera.frame != CoordinateFrame.CAMERA:
            raise ValueError("B1 query shards must use the camera frame")
        count = len(self.queries_camera)
        model_points = _readonly_array(
            self.points_model_roi,
            dtype=np.float64,
            shape=(count, 3),
            name="points_model_roi",
        )
        near_mask = _readonly_array(
            self.near_surface_mask,
            dtype=np.bool_,
            shape=(count,),
            name="near_surface_mask",
        )
        if len(self.values.udf_m) != count:
            raise ValueError("query and ground-truth counts differ")
        if np.any(np.abs(model_points) > 1.0 + 1e-12):
            raise ValueError("query points must stay inside the visible-only model ROI")
        if not np.any(near_mask) or np.all(near_mask):
            raise ValueError("query shard must contain both near-surface and uniform points")
        object.__setattr__(self, "shard_index", int(self.shard_index))
        object.__setattr__(self, "points_model_roi", model_points)
        object.__setattr__(self, "near_surface_mask", near_mask)

    @property
    def near_surface_count(self) -> int:
        return int(np.count_nonzero(self.near_surface_mask))

    @property
    def uniform_count(self) -> int:
        return len(self.near_surface_mask) - self.near_surface_count


def _masked_pixels_uv(mask: BoolArray) -> Float64Array:
    rows, columns = np.nonzero(mask)
    return np.column_stack((columns, rows)).astype(np.float64, copy=False)


def _uniform_visible_pixel_subset(
    pixels_uv: Float64Array,
    *,
    requested_count: int | None,
) -> Float64Array:
    """Select ordered midpoint strata over the row-major visible pixel list."""

    available = len(pixels_uv)
    if requested_count is None or requested_count >= available:
        return np.array(pixels_uv, dtype=np.float64, copy=True)
    indices = np.floor(
        (np.arange(requested_count, dtype=np.float64) + 0.5) * available / requested_count
    ).astype(np.int64)
    if len(np.unique(indices)) != requested_count:  # pragma: no cover - mathematical guard
        raise RuntimeError("uniform visible-pixel strata produced duplicate indices")
    return np.array(pixels_uv[indices], dtype=np.float64, copy=True)


def _sample_surface_near_roi(
    sample: B1FixedViewSample,
    *,
    count: int,
    sigma_m: float,
    rng: np.random.Generator,
) -> Float64Array:
    vertices = sample.observation.camera_T_mesh.transform_points(
        sample.surface.vertices_mesh_local_m
    )
    triangles = vertices[sample.surface.faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    twice_areas = np.linalg.norm(cross, axis=1)
    valid_faces = twice_areas > 0.0
    if not np.any(valid_faces):
        raise ValueError("near-surface sampling requires non-degenerate triangles")
    triangles = triangles[valid_faces]
    normals = cross[valid_faces] / twice_areas[valid_faces, None]
    probabilities = twice_areas[valid_faces] / np.sum(twice_areas[valid_faces])
    model_T_camera = sample.observation.model_roi.model_T_camera
    accepted: list[Float64Array] = []
    accepted_count = 0
    maximum_draws = max(64 * count, 4_096)
    draws = 0
    while accepted_count < count and draws < maximum_draws:
        draw_count = min(max(2 * (count - accepted_count), 64), maximum_draws - draws)
        face_indices = rng.choice(len(triangles), size=draw_count, p=probabilities)
        selected = triangles[face_indices]
        root = np.sqrt(rng.random(draw_count))
        second = rng.random(draw_count)
        barycentric = np.column_stack((1.0 - root, root * (1.0 - second), root * second))
        surface_points = np.sum(selected * barycentric[:, :, None], axis=1)
        offsets = rng.normal(0.0, sigma_m, size=draw_count)
        candidates = surface_points + normals[face_indices] * offsets[:, None]
        model_candidates = model_T_camera.transform_points(candidates)
        keep = np.all(np.abs(model_candidates) <= 1.0, axis=1)
        if np.any(keep):
            retained = candidates[keep][: count - accepted_count]
            accepted.append(retained)
            accepted_count += len(retained)
        draws += draw_count
    if accepted_count != count:
        raise ValueError("visible-only ROI cannot supply the requested near-surface queries")
    return np.concatenate(accepted, axis=0)


def sample_query_shard(
    sample: B1FixedViewSample,
    *,
    shard_index: int,
    config: QuerySamplingConfig | None = None,
) -> B1QueryShard:
    """Replay one labeled near-surface/uniform shard without asset-ID seeding."""

    if not isinstance(sample, B1FixedViewSample):
        raise TypeError("sample must be B1FixedViewSample")
    if isinstance(shard_index, bool) or not isinstance(shard_index, Integral) or shard_index < 0:
        raise ValueError("shard_index must be a non-negative integer")
    sampling = QuerySamplingConfig() if config is None else config
    if not isinstance(sampling, QuerySamplingConfig):
        raise TypeError("config must be QuerySamplingConfig")
    seed = np.random.SeedSequence((sampling.base_seed, int(shard_index)))
    rng = np.random.default_rng(seed)
    if sampling.near_surface_sigma_m is not None:
        sigma_m = sampling.near_surface_sigma_m
    elif sample.solid is not None:
        sigma_m = sampling.near_surface_sigma_voxels * sample.solid.voxel_size_m
    else:
        raise ValueError("surface-only samples require an explicit near_surface_sigma_m")
    near = _sample_surface_near_roi(
        sample,
        count=sampling.near_surface_count,
        sigma_m=sigma_m,
        rng=rng,
    )
    uniform_model = rng.uniform(-1.0, 1.0, size=(sampling.uniform_count, 3))
    uniform_camera = sample.observation.model_roi.camera_T_model.transform_points(uniform_model)
    points_camera = np.concatenate((near, uniform_camera), axis=0)
    near_mask = np.concatenate(
        (
            np.ones(sampling.near_surface_count, dtype=np.bool_),
            np.zeros(sampling.uniform_count, dtype=np.bool_),
        )
    )
    permutation = rng.permutation(len(points_camera))
    points_camera = points_camera[permutation]
    near_mask = near_mask[permutation]
    queries = QueryBatch(points_camera, CoordinateFrame.CAMERA)
    points_model = sample.observation.model_roi.model_T_camera.transform_points(points_camera)
    values = sample.ground_truth.query(queries)
    return B1QueryShard(
        shard_index=int(shard_index),
        queries_camera=queries,
        points_model_roi=points_model,
        values=values,
        near_surface_mask=near_mask,
    )
