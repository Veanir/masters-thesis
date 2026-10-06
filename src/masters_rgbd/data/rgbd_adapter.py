"""Fail-closed admission boundary for legacy v1 dataset records.

The legacy representation embedded a uniform mesh-unit-to-metre scale in
``camera_T_mesh``.  V2 requires metric mesh-local vertices and a rigid proper
``camera_T_mesh``.  This module performs only that lossless boundary rewrite;
it does not infer or repair missing provenance.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

from masters_rgbd.contracts.geometry import AffineTransform, CoordinateFrame

FloatArray = NDArray[np.float64]


class LegacyTransformRejectionCode(StrEnum):
    """Stable reasons why a legacy transform cannot cross the v2 boundary."""

    MALFORMED_TRANSFORM = "malformed_transform"
    NONFINITE_TRANSFORM = "nonfinite_transform"
    REFLECTION = "reflection"
    SHEAR = "shear"
    NONUNIFORM_SCALE = "nonuniform_scale"
    SCALE_DRIFT = "scale_drift"
    EQUIVALENCE_FAILURE = "equivalence_failure"


class LegacyTransformRejected(ValueError):
    """Typed rejection; callers must not silently drop the failed record."""

    def __init__(self, code: LegacyTransformRejectionCode, detail: str) -> None:
        super().__init__(f"{code.value}: {detail}")
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class AdaptedLegacyTransform:
    """V2-equivalent geometry plus a numerical equivalence witness."""

    vertices_mesh_local_m: FloatArray
    camera_T_mesh: AffineTransform
    baked_uniform_scale_m_per_legacy_unit: float
    equivalence_max_abs_error_m: float

    def __post_init__(self) -> None:
        vertices = np.array(self.vertices_mesh_local_m, dtype=np.float64, copy=True)
        vertices.setflags(write=False)
        object.__setattr__(self, "vertices_mesh_local_m", vertices)


def _reject(code: LegacyTransformRejectionCode, detail: str) -> None:
    raise LegacyTransformRejected(code, detail)


def adapt_legacy_v1_transform(
    vertices_mesh_local_legacy: object,
    legacy_camera_T_mesh: object,
    *,
    expected_uniform_scale_m_per_legacy_unit: float,
    atol: float = 1e-9,
    rtol: float = 1e-9,
) -> AdaptedLegacyTransform:
    """Bake a declared uniform legacy scale into vertices.

    The accepted mapping is exactly ``camera = (scale * R) @ vertex + t``.
    Its v2 form is ``camera = R @ (scale * vertex) + t``.  Any reflection,
    shear, non-uniform scale, or drift from the independently declared scale
    is rejected with a stable typed reason.
    """

    vertices = np.asarray(vertices_mesh_local_legacy, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        _reject(
            LegacyTransformRejectionCode.MALFORMED_TRANSFORM,
            f"vertices must be a non-empty [N, 3] array, got {vertices.shape}",
        )
    if not np.isfinite(vertices).all():
        _reject(
            LegacyTransformRejectionCode.NONFINITE_TRANSFORM,
            "vertices contain non-finite values",
        )

    matrix = np.asarray(legacy_camera_T_mesh, dtype=np.float64)
    if matrix.shape != (4, 4):
        _reject(
            LegacyTransformRejectionCode.MALFORMED_TRANSFORM,
            f"camera_T_mesh must have shape [4, 4], got {matrix.shape}",
        )
    if not np.isfinite(matrix).all():
        _reject(
            LegacyTransformRejectionCode.NONFINITE_TRANSFORM,
            "camera_T_mesh contains non-finite values",
        )
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=atol, rtol=0.0):
        _reject(
            LegacyTransformRejectionCode.MALFORMED_TRANSFORM,
            "camera_T_mesh must use affine homogeneous coordinates",
        )
    if (
        not np.isfinite(expected_uniform_scale_m_per_legacy_unit)
        or expected_uniform_scale_m_per_legacy_unit <= 0.0
    ):
        _reject(
            LegacyTransformRejectionCode.SCALE_DRIFT,
            "expected uniform scale must be positive and finite",
        )
    if not np.isfinite(atol) or not np.isfinite(rtol) or atol < 0.0 or rtol < 0.0:
        _reject(
            LegacyTransformRejectionCode.MALFORMED_TRANSFORM,
            "atol and rtol must be non-negative and finite",
        )

    linear = matrix[:3, :3]
    determinant = float(np.linalg.det(linear))
    if determinant < 0.0:
        _reject(
            LegacyTransformRejectionCode.REFLECTION,
            f"linear determinant must be positive, got {determinant}",
        )

    gram = linear.T @ linear
    diagonal = np.diag(gram)
    scale_squared = float(np.mean(diagonal))
    tolerance = max(atol**2, rtol * abs(scale_squared))
    off_diagonal = gram - np.diag(diagonal)
    if float(np.max(np.abs(off_diagonal))) > tolerance:
        _reject(
            LegacyTransformRejectionCode.SHEAR,
            "linear columns are not orthogonal",
        )
    if float(np.max(np.abs(diagonal - scale_squared))) > tolerance:
        _reject(
            LegacyTransformRejectionCode.NONUNIFORM_SCALE,
            "orthogonal axes have different scales",
        )
    if scale_squared <= 0.0:
        _reject(
            LegacyTransformRejectionCode.NONUNIFORM_SCALE,
            "linear scale is singular",
        )

    measured_scale = float(np.sqrt(scale_squared))
    if not np.isclose(
        measured_scale,
        expected_uniform_scale_m_per_legacy_unit,
        atol=atol,
        rtol=rtol,
    ):
        _reject(
            LegacyTransformRejectionCode.SCALE_DRIFT,
            "measured scale "
            f"{measured_scale} differs from declared "
            f"{expected_uniform_scale_m_per_legacy_unit}",
        )

    # SVD removes only accepted floating-point noise; the equivalence witness
    # below ensures it cannot conceal a material rewrite.
    left, _, right_t = np.linalg.svd(linear)
    rotation = left @ right_t
    rigid_matrix = np.eye(4, dtype=np.float64)
    rigid_matrix[:3, :3] = rotation
    rigid_matrix[:3, 3] = matrix[:3, 3]
    camera_T_mesh = AffineTransform(
        source=CoordinateFrame.MESH_LOCAL,
        target=CoordinateFrame.CAMERA,
        matrix=rigid_matrix,
    )
    baked_vertices = vertices * measured_scale

    legacy_homogeneous = np.concatenate((vertices, np.ones((len(vertices), 1))), axis=1)
    legacy_points_camera = (matrix @ legacy_homogeneous.T).T[:, :3]
    adapted_points_camera = camera_T_mesh.transform_points(baked_vertices)
    residual = (
        float(np.max(np.abs(legacy_points_camera - adapted_points_camera)))
        if len(vertices)
        else 0.0
    )
    equivalence_tolerance = atol + rtol * max(
        1.0,
        float(np.max(np.abs(legacy_points_camera))) if len(vertices) else 1.0,
    )
    if residual > equivalence_tolerance:
        _reject(
            LegacyTransformRejectionCode.EQUIVALENCE_FAILURE,
            f"maximum camera-space residual {residual} exceeds {equivalence_tolerance}",
        )

    return AdaptedLegacyTransform(
        vertices_mesh_local_m=baked_vertices,
        camera_T_mesh=camera_T_mesh,
        baked_uniform_scale_m_per_legacy_unit=measured_scale,
        equivalence_max_abs_error_m=residual,
    )
