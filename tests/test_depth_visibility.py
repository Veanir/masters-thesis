from __future__ import annotations

import numpy as np

from masters_rgbd.b1.depth_visibility import (
    DepthVisibilityLabel,
    classify_depth_visibility,
)


def _point(depth_m: float) -> list[float]:
    return [0.0, 0.0, -depth_m]


def test_hidden_requires_strictly_more_than_full_support_maximum_plus_5mm() -> None:
    depth = np.full((7, 7), 1.0, dtype=np.float64)
    depth[2, 2] = 1.002
    points = np.asarray(
        [
            _point(1.001),
            _point(1.007),
            _point(np.nextafter(1.007, np.inf)),
        ]
    )

    result = classify_depth_visibility(
        points,
        depth_m=depth,
        target_mask=np.ones_like(depth, dtype=np.bool_),
        intrinsics=np.asarray((100.0, 100.0, 3.0, 3.0)),
    )

    assert result.minimum_support_depth_m.tolist() == [1.0, 1.0, 1.0]
    assert result.maximum_support_depth_m.tolist() == [1.002, 1.002, 1.002]
    assert result.labels.tolist() == [
        int(DepthVisibilityLabel.SUPPORTED_NOT_HIDDEN),
        int(DepthVisibilityLabel.SUPPORTED_NOT_HIDDEN),
        int(DepthVisibilityLabel.HIDDEN_BEHIND_OBSERVATION),
    ]


def test_incomplete_or_invalid_support_is_explicitly_unknown() -> None:
    depth = np.full((7, 7), 1.0, dtype=np.float64)
    mask = np.ones_like(depth, dtype=np.bool_)
    mask[2, 2] = False
    points = np.asarray([_point(1.1), [0.0, 0.0, 0.1], [1.0, 0.0, -1.0]])

    result = classify_depth_visibility(
        points,
        depth_m=depth,
        target_mask=mask,
        intrinsics=np.asarray((100.0, 100.0, 3.0, 3.0)),
    )

    assert result.labels.tolist() == [
        int(DepthVisibilityLabel.UNKNOWN),
        int(DepthVisibilityLabel.UNKNOWN),
        int(DepthVisibilityLabel.UNKNOWN),
    ]
    assert result.unknown_mask.tolist() == [True, True, True]


def test_non_finite_local_depth_makes_full_support_unknown() -> None:
    depth = np.full((7, 7), 1.0, dtype=np.float64)
    depth[4, 4] = np.nan

    result = classify_depth_visibility(
        np.asarray([_point(1.1)]),
        depth_m=depth,
        target_mask=np.ones_like(depth, dtype=np.bool_),
        intrinsics=np.asarray((100.0, 100.0, 3.0, 3.0)),
    )

    assert result.labels.item() == int(DepthVisibilityLabel.UNKNOWN)
