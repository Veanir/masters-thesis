"""Create and validate the fixed 140/25/23 B1 cohort query cache."""

import argparse
from pathlib import Path

from masters_rgbd.b1.cache_preflight import preflight
from masters_rgbd.b1.dev_cohort import resolve_dev_cohort
from masters_rgbd.b1.query_cache import _materialize_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    specs = resolve_dev_cohort(
        args.dataset_root, args.inventory, train_count=140, validation_count=25, test_count=23
    )
    _materialize_cache(
        specs,
        dataset_root=args.dataset_root,
        cache_root=args.cache_root,
        data_seed=0,
        train_queries=512,
        evaluation_queries=1024,
        partial_points=512,
    )
    preflight(args, specs)


if __name__ == "__main__":
    main()
