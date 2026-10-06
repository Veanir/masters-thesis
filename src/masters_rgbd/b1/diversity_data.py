"""Category-matched 83/140 generated-geometry study; reuses the B1 trainer."""

from __future__ import annotations

from masters_rgbd.b1.dev_cohort import resolve_dev_cohort
from masters_rgbd.b1.training import (
    _cache_spec,
    _validate_cache_spec,
)


def _specs(args):
    specs = resolve_dev_cohort(
        args.dataset_root, args.inventory, train_count=140, validation_count=25, test_count=23
    )
    _validate_cache_spec(
        args.cache_root,
        _cache_spec(
            specs, data_seed=0, train_queries=512, evaluation_queries=1024, partial_points=512
        ),
    )
    return specs
