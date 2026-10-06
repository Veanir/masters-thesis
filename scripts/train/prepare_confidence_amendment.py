"""Bind the existing confidence derivative amendment to a local training launch."""

import argparse
from pathlib import Path

from scripts.common.paths import CODE_ROOT, digest, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    trainer = CODE_ROOT / "scripts/train/train_ray_variants.py"
    save(
        args.output / "launch.json",
        {
            "trainer": str(trainer),
            "trainer_sha256": digest(trainer),
            "training_wrapper_sha256": digest(
                CODE_ROOT / "scripts/train/train_ray_with_stable_confidence.py"
            ),
            "amendment_sha256": digest(CODE_ROOT / "scripts/train/confidence_exponential.py"),
            "result_dir_absolute": str(args.results.resolve()),
            "seed": args.seed,
        },
    )


if __name__ == "__main__":
    main()
