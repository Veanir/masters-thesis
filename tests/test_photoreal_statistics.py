"""Unequal object frequencies must not masquerade as independent objects."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from scripts.preliminary_hope.bootstrap_objects import balanced_mean, paired_interval


def test_frequent_object_does_not_dominate_primary_estimate():
    rows = [{"sample_id": str(i), "obj_id": 1, "scene_id": i} for i in range(9)]
    rows += [{"sample_id": "last", "obj_id": 2, "scene_id": 0}]
    result = paired_interval([1] * 9 + [-1], rows)
    assert result["mean"] == 0
    assert result["objects"] == 2 and result["observations"] == 10
    assert result["scene_sensitivity"]["clusters"] == 9
    assert result["scene_sensitivity"]["leave_one_scene_out"]["0"]["objects"] == 1


def test_identical_paired_difference_has_zero_width_intervals():
    rows = [{"sample_id": str(i), "obj_id": i % 3, "scene_id": i // 3} for i in range(12)]
    result = paired_interval([0.25] * 12, rows)
    assert result["ci95"] == pytest.approx([0.25, 0.25])
    assert result["scene_sensitivity"]["ci95"] == pytest.approx([0.25, 0.25])


def test_missing_subset_is_not_a_zero_score():
    rows = [{"sample_id": str(i), "obj_id": i} for i in range(3)]
    assert balanced_mean([None, 0.4, 0.8], rows) == pytest.approx(0.6)
    assert balanced_mean([None] * 3, rows) is None
    with pytest.raises(ValueError):
        paired_interval([0, float("nan"), 1], rows)
