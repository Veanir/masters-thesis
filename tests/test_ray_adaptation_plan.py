"""Check paired exposure counts, resume slicing and mask preservation."""

import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "ray_adaptation_plan",
    Path(__file__).parents[1] / "scripts/preliminary_hope/prepare_training_plan.py",
)
plan = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plan)


def test_paired_counts_and_resume():
    plans = []
    for seed in range(3):
        rows = plan.schedule(seed)
        assert len(rows) == 2160
        assert rows == plan.schedule(seed)
        assert rows[:720] + plan.schedule(seed)[720:] == rows
        for ordinal in range(36):
            exposures = [r for r in rows if r["ordinal"] == ordinal]
            assert len(exposures) == 60
            assert sum(r["augment"] for r in exposures) == 30
            for variant in range(2):
                assert sum(r["augment"] and r["variant"] == variant for r in exposures) == 15
        for epoch in range(60):
            assert sorted(r["ordinal"] for r in rows if r["epoch"] == epoch) == list(range(36))
        assert all(len(set(r["views"])) == 2 and all(0 <= v < 21 for v in r["views"]) for r in rows)
        plans.append(rows)
    assert plans[0] != plans[1] != plans[2]


def test_classic_preserves_mask_and_input():
    rgb = np.full((16, 16, 3), 127, dtype=np.uint8)
    mask = np.zeros((16, 16), bool)
    mask[4:12, 4:12] = True
    before = rgb.copy()
    augmented = plan.classic_rgb(rgb, mask, 17)
    assert np.array_equal(rgb, before)
    assert np.array_equal(augmented, plan.classic_rgb(rgb, mask, 17))
    assert augmented.dtype == np.uint8 and augmented.shape == rgb.shape
    assert not augmented[~mask].any()
    assert np.any(augmented[mask] != rgb[mask])
