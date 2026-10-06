"""Secondary density controls for all nine frozen diversity models."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.trellis_b1.family_bootstrap import METRICS, estimate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.input.read_text())
    assert source["status"] == "complete"
    arms = source["arms"]
    ids = [r["asset_id"] for r in arms["input512"]["per_object"]]
    assert len(set(ids)) == 25
    assert all([r["asset_id"] for r in a["per_object"]] == ids for a in arms.values())
    matrices, means = {}, {}
    for arm in ("base_u", "mix_u", "mix_2u"):
        for mode in ("natural", "vertex512", "area512"):
            key = f"{arm}/{mode}"
            matrices[key], means[key] = {}, {}
            for metric, get in METRICS.items():
                values = []
                for seed in range(3):
                    variants = (
                        [f"{arm}_seed{seed}/{mode}"]
                        if mode != "area512"
                        else [
                            f"{arm}_seed{seed}/area512-{s}" for s in (20260904, 20260905, 20260906)
                        ]
                    )
                    values.append(
                        np.mean([[get(r) for r in arms[v]["per_object"]] for v in variants], axis=0)
                    )
                matrix = np.asarray(values, dtype=float)
                assert matrix.shape == (3, 25) and np.isfinite(matrix).all()
                matrices[key][metric] = matrix
                means[key][metric] = {
                    "mean": float(matrix.mean()),
                    "per_seed": matrix.mean(axis=1).tolist(),
                }
    comparisons = {}
    for mode in ("natural", "vertex512", "area512"):
        for right, left in (("mix_u", "base_u"), ("mix_2u", "base_u"), ("mix_2u", "mix_u")):
            key = f"{right}-minus-{left}/{mode}"
            comparisons[key] = {}
            for metric in METRICS:
                delta = matrices[f"{right}/{mode}"][metric] - matrices[f"{left}/{mode}"][metric]
                estimated = estimate(delta, [[i] for i in range(25)])
                comparisons[key][metric] = {
                    "delta": float(delta.mean()),
                    "per_seed": delta.mean(axis=1).tolist(),
                    "seed_object_ci95": estimated["family_ci95"],
                }
    result = {
        "scope": "secondary exploratory validation25 density analysis",
        "source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "supplement_sha256": hashlib.sha256(
            (
                Path(__file__).resolve().parents[2] / "configs/trellis_point_budgets.json"
            ).read_bytes()
        ).hexdigest(),
        "sampling_repetitions_averaged_within_object": True,
        "bootstrap": "20000 crossed seed/object resamples, seed20260904; 3 training seeds",
        "arm_summaries": means,
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({key: value["fscore5"] for key, value in comparisons.items()}, indent=2))


if __name__ == "__main__":
    main()
