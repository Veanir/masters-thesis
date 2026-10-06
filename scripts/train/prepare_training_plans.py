"""Bind the unchanged matched training protocol to newly prepared local inputs."""

import argparse
from pathlib import Path

from scripts.common.paths import CODE_ROOT, digest, read, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--photo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    training = read(args.training / "complete.json")
    photo = read(args.photo / "complete.json")
    assert training["status"] == "certified_synthetic_ray_labels_complete"
    assert photo["status"] == "frozen_PHOTO_admission"
    ids = [row["sample_id"] for row in training["rows"]]
    assert len(ids) == len(set(ids)) == 6400
    assert all(row["split"] == "train" for row in training["rows"])
    assert set(ids) == {row["sample_id"] for row in photo["rows"]}
    config = read(CODE_ROOT / "configs/training.json")
    args.output.mkdir(parents=True, exist_ok=False)
    for seed in config["seeds"]:
        job = {
            key: config[key]
            for key in (
                "purpose",
                "context",
                "epochs",
                "steps",
                "decoder_blocks",
                "lr",
                "checkpoint_interval",
            )
        }
        job.update(
            seed=seed,
            sample_ids=ids,
            training_manifest_sha256=digest(args.training / "complete.json"),
            photo_admission_manifest_sha256=digest(args.photo / "complete.json"),
            sample_order="Exact certified TRAIN6400 assembly manifest row order",
        )
        save(args.output / f"seed{seed}-job.json", job)


if __name__ == "__main__":
    main()
