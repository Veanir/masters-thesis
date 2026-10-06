from __future__ import annotations

import numpy as np
import pytest

from masters_rgbd.contracts.fields import (
    FieldPrediction,
    QueryBatch,
    QueryBudget,
    ROIBounds,
)
from masters_rgbd.contracts.geometry import CoordinateFrame
from masters_rgbd.extraction.projected_udf import DenseProjectedUDFExtractor


class _SphereUDFField:
    def __init__(
        self,
        *,
        center: tuple[float, float, float],
        radius: float,
        occupancy_logit: float,
    ) -> None:
        self.center = np.asarray(center, dtype=np.float64)
        self.radius = radius
        self.occupancy_logit = occupancy_logit
        self.queried_count = 0

    def query(self, queries: QueryBatch) -> FieldPrediction:
        self.queried_count += len(queries)
        udf = np.abs(np.linalg.norm(queries.points - self.center, axis=1) - self.radius)
        logits = np.full(len(queries), self.occupancy_logit, dtype=np.float64)
        return FieldPrediction(udf, logits)


class _FlatUDFField:
    def query(self, queries: QueryBatch) -> FieldPrediction:
        return FieldPrediction(
            np.ones(len(queries), dtype=np.float64),
            np.zeros(len(queries), dtype=np.float64),
        )


def _extract(field: object, *, resolution: int = 24):
    extractor = DenseProjectedUDFExtractor(
        resolution=resolution,
        minimum_gradient_norm=1e-8,
    )
    roi = ROIBounds(
        (-0.3, -0.3, -1.3),
        (0.3, 0.3, -0.7),
        CoordinateFrame.CAMERA,
    )
    return (
        extractor,
        roi,
        extractor.extract(
            field,  # type: ignore[arg-type]
            roi,
            QueryBudget(extractor.required_query_count),
        ),
    )


def test_projected_oracle_sphere_recovers_central_surface() -> None:
    center = np.asarray((0.0, 0.0, -1.0))
    radius = 0.18
    field = _SphereUDFField(
        center=tuple(center),
        radius=radius,
        occupancy_logit=123.0,
    )

    extractor, _, result = _extract(field)

    radial_error = np.abs(np.linalg.norm(result.mesh.vertices_m - center, axis=1) - radius)
    assert field.queried_count == extractor.required_query_count
    assert len(result.mesh.vertices_m) > 0
    assert len(result.mesh.faces) > 0
    assert float(np.mean(radial_error)) < result.tau_m / 4.0
    assert result.projection_valid_fraction == 1.0
    assert result.roi_boundary_contact_fraction == 0.0


def test_projected_udf_enforces_exact_fixed_budget() -> None:
    field = _SphereUDFField(center=(0.0, 0.0, -1.0), radius=0.18, occupancy_logit=0.0)
    extractor = DenseProjectedUDFExtractor(resolution=8, minimum_gradient_norm=1e-8)
    roi = ROIBounds((-0.3, -0.3, -1.3), (0.3, 0.3, -0.7))

    with pytest.raises(ValueError, match="smaller than required dense count"):
        extractor.extract(field, roi, QueryBudget(extractor.required_query_count - 1))

    assert field.queried_count == 0
    result = extractor.extract(field, roi, QueryBudget(extractor.required_query_count))
    assert field.queried_count == extractor.required_query_count
    assert result.stats.requested_point_count == extractor.required_query_count
    assert result.stats.field_evaluation_count == extractor.required_query_count
    assert result.stats.cache_hit_count == 0


def test_projected_mesh_ignores_occupancy_and_is_deterministic_and_valid() -> None:
    positive = _SphereUDFField(center=(0.0, 0.0, -1.0), radius=0.18, occupancy_logit=1e6)
    negative = _SphereUDFField(center=(0.0, 0.0, -1.0), radius=0.18, occupancy_logit=-1e6)

    _, _, first = _extract(positive, resolution=18)
    _, _, second = _extract(negative, resolution=18)
    _, _, repeated = _extract(positive, resolution=18)

    np.testing.assert_array_equal(first.mesh.vertices_m, second.mesh.vertices_m)
    np.testing.assert_array_equal(first.mesh.faces, second.mesh.faces)
    np.testing.assert_array_equal(first.mesh.vertices_m, repeated.mesh.vertices_m)
    np.testing.assert_array_equal(first.mesh.faces, repeated.mesh.faces)
    assert np.isfinite(first.mesh.vertices_m).all()
    assert np.all(first.mesh.faces >= 0)
    assert np.all(first.mesh.faces < len(first.mesh.vertices_m))
    triangles = first.mesh.vertices_m[first.mesh.faces]
    doubled_area = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    assert np.all(doubled_area > 1e-14)
    assert len(np.unique(np.sort(first.mesh.faces, axis=1), axis=0)) == len(first.mesh.faces)


def test_flat_udf_returns_a_typed_empty_surface() -> None:
    extractor, _, result = _extract(_FlatUDFField(), resolution=8)

    assert result.mesh.vertices_m.shape == (0, 3)
    assert result.mesh.faces.shape == (0, 3)
    assert result.surface_points_m.shape == (0, 3)
    assert result.stats.query_count == extractor.required_query_count
    assert result.stats.active_leaf_count == 0
    assert result.projection_valid_fraction == 0.0
    assert result.roi_boundary_contact_fraction == 0.0


@pytest.mark.parametrize("tau_scale", (0.5, 1.0, 1.25))
def test_tau_scale_changes_level_without_changing_query_budget(tau_scale: float) -> None:
    field = _SphereUDFField(center=(0.0, 0.0, -1.0), radius=0.18, occupancy_logit=0.0)
    extractor = DenseProjectedUDFExtractor(
        resolution=12,
        minimum_gradient_norm=1e-8,
        tau_scale=tau_scale,
    )
    roi = ROIBounds((-0.3, -0.3, -1.3), (0.3, 0.3, -0.7))

    result = extractor.extract(field, roi, QueryBudget(extractor.required_query_count))

    expected_cell_diagonal = np.sqrt(3.0) * 0.6 / 12.0
    assert result.tau_m == pytest.approx(tau_scale * expected_cell_diagonal)
    assert result.stats.field_evaluation_count == extractor.required_query_count
    assert result.projection_mode == "unit_tau"


@pytest.mark.parametrize("value", (0.0, -1.0, float("inf"), float("nan")))
def test_invalid_tau_scale_is_rejected(value: float) -> None:
    with pytest.raises(ValueError, match="tau_scale"):
        DenseProjectedUDFExtractor(tau_scale=value)


def test_invalid_projection_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="projection_mode"):
        DenseProjectedUDFExtractor(projection_mode="unknown")  # type: ignore[arg-type]


class _ScaledSphereUDFField(_SphereUDFField):
    def __init__(self, *, scale: float) -> None:
        super().__init__(center=(0.0, 0.0, -1.0), radius=0.18, occupancy_logit=0.0)
        self.scale = scale

    def query(self, queries: QueryBatch) -> FieldPrediction:
        prediction = super().query(queries)
        return FieldPrediction(prediction.udf_m * self.scale, prediction.occupancy_logit)


def test_capped_newton_projection_improves_non_unit_gradient_field() -> None:
    center = np.asarray((0.0, 0.0, -1.0))
    roi = ROIBounds((-0.3, -0.3, -1.3), (0.3, 0.3, -0.7))
    field = _ScaledSphereUDFField(scale=2.0)
    legacy = DenseProjectedUDFExtractor(
        resolution=24,
        minimum_gradient_norm=1e-8,
        projection_mode="unit_tau",
    )
    newton = DenseProjectedUDFExtractor(
        resolution=24,
        minimum_gradient_norm=1e-8,
        projection_mode="newton_tau_capped",
    )

    legacy_result = legacy.extract(field, roi, QueryBudget(legacy.required_query_count))
    newton_result = newton.extract(field, roi, QueryBudget(newton.required_query_count))
    legacy_error = np.mean(
        np.abs(np.linalg.norm(legacy_result.mesh.vertices_m - center, axis=1) - 0.18)
    )
    newton_error = np.mean(
        np.abs(np.linalg.norm(newton_result.mesh.vertices_m - center, axis=1) - 0.18)
    )

    assert newton_error < legacy_error
    assert newton_result.projection_mode == "newton_tau_capped"
