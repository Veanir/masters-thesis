from __future__ import annotations

import inspect

import pytest

torch = pytest.importorskip("torch")

from masters_rgbd.b1.model import B1FieldModel  # noqa: E402


def _inputs() -> tuple[torch.Tensor, ...]:
    height, width = 32, 40
    row = torch.linspace(0.0, 1.0, height).view(1, 1, height, 1)
    column = torch.linspace(0.0, 1.0, width).view(1, 1, 1, width)
    rgb = torch.cat(
        (
            column.expand(1, 1, height, width),
            row.expand(1, 1, height, width),
            (0.5 * row + 0.5 * column).expand(1, 1, height, width),
        ),
        dim=1,
    )
    mask = torch.ones((1, 1, height, width))
    depth_m = torch.ones((1, 1, height, width))
    partial_points_camera_m = torch.tensor(
        [
            [
                [-0.15, -0.10, -1.00],
                [-0.05, -0.10, -0.96],
                [0.05, -0.10, -0.96],
                [0.15, -0.10, -1.00],
                [-0.15, 0.00, -0.98],
                [-0.05, 0.00, -0.94],
                [0.05, 0.00, -0.94],
                [0.15, 0.00, -0.98],
                [-0.15, 0.10, -1.00],
                [-0.05, 0.10, -0.96],
                [0.05, 0.10, -0.96],
                [0.15, 0.10, -1.00],
            ]
        ]
    )
    query_points_camera_m = torch.tensor(
        [
            [
                [-0.10, -0.08, -1.03],
                [-0.05, 0.00, -1.00],
                [0.00, 0.00, -0.96],
                [0.05, 0.00, -1.00],
                [0.10, 0.08, -1.03],
                [0.00, 0.12, -1.08],
                [0.00, -0.12, -1.08],
            ]
        ]
    )
    intrinsics = torch.tensor([[36.0, 36.0, 19.5, 15.5]])
    return (
        rgb,
        depth_m,
        mask,
        partial_points_camera_m,
        query_points_camera_m,
        intrinsics,
    )


def _combined_output(model: B1FieldModel, inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
    prediction = model(*inputs)
    return torch.cat((prediction.udf_m, prediction.occupancy_logits), dim=-1)


@pytest.mark.parametrize("conditioning", ("pixel_knn", "global"))
def test_b1_model_shapes_gradients_and_positive_udf(conditioning: str) -> None:
    torch.manual_seed(7)
    model = B1FieldModel(conditioning=conditioning, knn_neighbors=4)
    inputs = _inputs()

    prediction = model(*inputs)

    assert prediction.udf_m.shape == (1, 7)
    assert prediction.occupancy_logits.shape == (1, 7)
    assert torch.isfinite(prediction.udf_m).all()
    assert torch.isfinite(prediction.occupancy_logits).all()
    assert torch.all(prediction.udf_m > 0.0)

    loss = prediction.udf_m.mean() + prediction.occupancy_logits.square().mean()
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert all(gradient is not None for gradient in gradients)
    assert sum(float(gradient.abs().sum()) for gradient in gradients if gradient is not None) > 0.0


def test_global_conditioning_has_no_local_geometry_and_ignores_intrinsics() -> None:
    torch.manual_seed(23)
    model = B1FieldModel(conditioning="global", knn_neighbors=4).eval()
    rgb, depth_m, mask, partial_points, queries, intrinsics = _inputs()

    assert not hasattr(model, "local_geometry")
    with torch.no_grad():
        encoding = model.encode_observation(rgb, depth_m, mask, partial_points)
        baseline = model.decode_queries(encoding, queries, intrinsics)
        changed = model.decode_queries(
            encoding,
            queries,
            intrinsics + torch.tensor([[15.0, -10.0, 7.0, -6.0]]),
        )

    assert torch.equal(baseline.udf_m, changed.udf_m)
    assert torch.equal(baseline.occupancy_logits, changed.occupancy_logits)


def test_unknown_conditioning_is_rejected() -> None:
    with pytest.raises(ValueError, match="conditioning"):
        B1FieldModel(conditioning="unknown")  # type: ignore[arg-type]


def test_unknown_rgb_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="rgb_mode"):
        B1FieldModel(rgb_mode="unknown")  # type: ignore[arg-type]


def test_depth_only_mode_is_invariant_to_rgb() -> None:
    torch.manual_seed(19)
    model = B1FieldModel(knn_neighbors=4, rgb_mode="depth_only").eval()
    rgb, depth, mask, partial, query, intrinsics = _inputs()

    with torch.no_grad():
        first = model(rgb, depth, mask, partial, query, intrinsics)
        second = model(torch.rand_like(rgb), depth, mask, partial, query, intrinsics)

    torch.testing.assert_close(first.udf_m, second.udf_m)
    torch.testing.assert_close(first.occupancy_logits, second.occupancy_logits)


def test_cached_encoding_is_query_chunk_equivalent() -> None:
    torch.manual_seed(11)
    model = B1FieldModel(knn_neighbors=4).eval()
    rgb, depth_m, mask, partial_points, queries, intrinsics = _inputs()

    with torch.no_grad():
        whole = model(rgb, depth_m, mask, partial_points, queries, intrinsics)
        encoding = model.encode_observation(rgb, depth_m, mask, partial_points)
        first = model.decode_queries(encoding, queries[:, :3], intrinsics)
        second = model.decode_queries(encoding, queries[:, 3:], intrinsics)

    assert torch.allclose(
        whole.udf_m, torch.cat((first.udf_m, second.udf_m), dim=1), atol=1e-7, rtol=1e-6
    )
    assert torch.allclose(
        whole.occupancy_logits,
        torch.cat((first.occupancy_logits, second.occupancy_logits), dim=1),
        atol=1e-7,
        rtol=1e-6,
    )


def test_ray_descriptor_exposes_query_to_observed_depth_delta() -> None:
    model = B1FieldModel(knn_neighbors=4, ray_descriptor=True).eval()
    rgb, depth_m, mask, partial_points, _, intrinsics = _inputs()
    queries = torch.tensor([[[0.0, 0.0, -0.8], [0.0, 0.0, -1.2]]])

    with torch.no_grad():
        encoding = model.encode_observation(rgb, depth_m, mask, partial_points)
        descriptor = model._sample_ray_descriptor(encoding, queries, intrinsics)

    assert descriptor.shape == (1, 2, 6)
    assert torch.isfinite(descriptor).all()
    assert descriptor[0, 0, 4] < 0.0
    assert descriptor[0, 1, 4] > 0.0


def test_prediction_depends_on_rgbd_partial_geometry_and_pixel_alignment() -> None:
    torch.manual_seed(19)
    model = B1FieldModel(knn_neighbors=4).eval()
    inputs = _inputs()

    with torch.no_grad():
        baseline = _combined_output(model, inputs)

        rgbd_changed = list(inputs)
        rgbd_changed[0] = 1.0 - rgbd_changed[0]
        rgbd_changed[1] = rgbd_changed[1] + 0.25
        changed_rgbd = _combined_output(model, tuple(rgbd_changed))

        geometry_changed = list(inputs)
        geometry_changed[3] = geometry_changed[3] + torch.tensor([0.08, -0.04, -0.12])
        changed_geometry = _combined_output(model, tuple(geometry_changed))

        intrinsics_changed = list(inputs)
        intrinsics_changed[5] = intrinsics_changed[5] + torch.tensor([[0.0, 0.0, 5.0, -4.0]])
        changed_intrinsics = _combined_output(model, tuple(intrinsics_changed))

    assert not torch.allclose(baseline, changed_rgbd, atol=1e-7, rtol=1e-6)
    assert not torch.allclose(baseline, changed_geometry, atol=1e-7, rtol=1e-6)
    assert not torch.allclose(baseline, changed_intrinsics, atol=1e-7, rtol=1e-6)


def test_public_interface_has_no_asset_or_ground_truth_identity() -> None:
    parameter_names = set(inspect.signature(B1FieldModel.forward).parameters)
    forbidden = {"asset_id", "sample_id", "camera_T_mesh", "mesh", "ground_truth", "bounds"}

    assert forbidden.isdisjoint(parameter_names)
    with pytest.raises(TypeError):
        B1FieldModel()(*_inputs(), asset_id="proc_box_carton_001")
