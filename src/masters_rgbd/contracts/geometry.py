"""Coordinate, camera, and transform contracts.

The public convention is a right-handed metric camera frame: +X points image
right, +Y points image up, and visible points have Z < 0. Depth is therefore
``-z_camera`` and is always positive for a valid observation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class CoordinateFrame(StrEnum):
    CAMERA = "camera"
    MODEL_ROI = "model_roi"
    MESH_LOCAL = "mesh_local"


def _points3(value: object, *, name: str) -> FloatArray:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{name} must have shape [N, 3], got {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError(f"{name} contains non-finite values")
    return points


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("camera width and height must be positive")
        values = np.asarray((self.fx, self.fy, self.cx, self.cy), dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("camera intrinsics must be finite")
        if self.fx <= 0.0 or self.fy <= 0.0:
            raise ValueError("fx and fy must be positive")

    def project(self, points_camera_m: object) -> FloatArray:
        points = _points3(points_camera_m, name="points_camera_m")
        depth_m = -points[:, 2]
        if np.any(depth_m <= 0.0):
            raise ValueError("all projected camera points must have z < 0")
        pixels = np.empty((len(points), 2), dtype=np.float64)
        pixels[:, 0] = self.cx + self.fx * points[:, 0] / depth_m
        pixels[:, 1] = self.cy - self.fy * points[:, 1] / depth_m
        return pixels

    def unproject(self, pixels_uv: object, depth_m: object) -> FloatArray:
        pixels = np.asarray(pixels_uv, dtype=np.float64)
        depth = np.asarray(depth_m, dtype=np.float64)
        if pixels.ndim != 2 or pixels.shape[1] != 2:
            raise ValueError(f"pixels_uv must have shape [N, 2], got {pixels.shape}")
        if depth.shape != (len(pixels),):
            raise ValueError(f"depth_m must have shape [{len(pixels)}], got {depth.shape}")
        if not np.isfinite(pixels).all() or not np.isfinite(depth).all():
            raise ValueError("pixels and depth must be finite")
        if np.any(depth <= 0.0):
            raise ValueError("depth_m must be positive")
        points = np.empty((len(pixels), 3), dtype=np.float64)
        points[:, 0] = (pixels[:, 0] - self.cx) * depth / self.fx
        points[:, 1] = -(pixels[:, 1] - self.cy) * depth / self.fy
        points[:, 2] = -depth
        return points


@dataclass(frozen=True)
class AffineTransform:
    """Column-vector transform named as ``target_T_source``."""

    source: CoordinateFrame
    target: CoordinateFrame
    matrix: FloatArray

    def __post_init__(self) -> None:
        matrix = np.array(self.matrix, dtype=np.float64, copy=True)
        if matrix.shape != (4, 4):
            raise ValueError(f"transform matrix must have shape [4, 4], got {matrix.shape}")
        if not np.isfinite(matrix).all():
            raise ValueError("transform matrix contains non-finite values")
        if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-12):
            raise ValueError("transform matrix must use affine homogeneous coordinates")
        if abs(float(np.linalg.det(matrix[:3, :3]))) < 1e-12:
            raise ValueError("transform matrix is singular")
        matrix.setflags(write=False)
        object.__setattr__(self, "matrix", matrix)

    def transform_points(self, points_source: object) -> FloatArray:
        points = _points3(points_source, name="points_source")
        homogeneous = np.concatenate((points, np.ones((len(points), 1))), axis=1)
        return (self.matrix @ homogeneous.T).T[:, :3]

    def inverse(self) -> AffineTransform:
        return AffineTransform(
            source=self.target,
            target=self.source,
            matrix=np.linalg.inv(self.matrix),
        )

    def is_rigid(self, *, atol: float = 1e-9) -> bool:
        """Return whether the linear part is a proper 3D rotation."""

        linear = self.matrix[:3, :3]
        return bool(
            np.allclose(linear.T @ linear, np.eye(3), atol=atol, rtol=0.0)
            and np.isclose(np.linalg.det(linear), 1.0, atol=atol, rtol=0.0)
        )

    def then(self, next_transform: AffineTransform) -> AffineTransform:
        if self.target != next_transform.source:
            raise ValueError(
                f"cannot compose {self.source}->{self.target} with "
                f"{next_transform.source}->{next_transform.target}"
            )
        return AffineTransform(
            source=self.source,
            target=next_transform.target,
            matrix=next_transform.matrix @ self.matrix,
        )

    def to_json(self) -> list[list[float]]:
        return self.matrix.tolist()


@dataclass(frozen=True)
class ModelROI:
    """Inference ROI derived only from the visible target point cloud.

    ``MODEL_ROI`` is a dimensionless cube in [-1, 1]^3. Public predictions are
    transformed back to metric camera coordinates before evaluation.
    """

    center_camera_m: tuple[float, float, float]
    half_extent_m: float

    def __post_init__(self) -> None:
        center = np.asarray(self.center_camera_m, dtype=np.float64)
        if center.shape != (3,) or not np.isfinite(center).all():
            raise ValueError("center_camera_m must be a finite vector3")
        if not np.isfinite(self.half_extent_m) or self.half_extent_m <= 0.0:
            raise ValueError("half_extent_m must be positive and finite")

    @classmethod
    def from_visible_points(
        cls,
        points_camera_m: object,
        *,
        minimum_half_extent_m: float,
        padding_fraction: float = 0.25,
    ) -> ModelROI:
        points = _points3(points_camera_m, name="points_camera_m")
        if len(points) == 0:
            raise ValueError("visible point cloud cannot be empty")
        if minimum_half_extent_m <= 0.0:
            raise ValueError("minimum_half_extent_m must be positive")
        if padding_fraction < 0.0:
            raise ValueError("padding_fraction cannot be negative")
        minimum = points.min(axis=0)
        maximum = points.max(axis=0)
        center = 0.5 * (minimum + maximum)
        half_extent = max(
            minimum_half_extent_m,
            0.5 * float(np.max(maximum - minimum)) * (1.0 + padding_fraction),
        )
        return cls(tuple(float(value) for value in center), half_extent)

    @property
    def model_T_camera(self) -> AffineTransform:
        scale = 1.0 / self.half_extent_m
        center = np.asarray(self.center_camera_m, dtype=np.float64)
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] *= scale
        matrix[:3, 3] = -center * scale
        return AffineTransform(CoordinateFrame.CAMERA, CoordinateFrame.MODEL_ROI, matrix)

    @property
    def camera_T_model(self) -> AffineTransform:
        return self.model_T_camera.inverse()
