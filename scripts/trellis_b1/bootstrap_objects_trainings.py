"""Audit all nine runs and report the three predeclared paired comparisons."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from masters_rgbd.b1.training import _sha256_file
from scripts.trellis_b1.train_predict_comparison import MANIFEST


def bootstrap(delta, seed_indices, object_indices):
    draws = delta[seed_indices[:, :, None], object_indices[:, None, :]].mean(axis=(1, 2))
    return np.quantile(draws, [0.025, 0.975]).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surfaces", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    folder = args.surfaces
    complete_path = folder / "complete.json"
    complete = json.loads(complete_path.read_text())
    plan = json.loads(MANIFEST.read_text())
    assert complete["binding"]["campaign_sha256"] == _sha256_file(MANIFEST)
    ids = [x["asset_id"] for x in complete["binding"]["inputs"]]
    assert len(ids) == len(set(ids)) == 25
    inputs = {x["asset_id"]: x for x in complete["binding"]["inputs"]}
    metrics = (
        "fscore",
        "precision",
        "recall",
        "chamfer_mean_m",
        "hidden_recall_5mm",
        "visible_recall_5mm",
    )
    arrays = {arm: {metric: [] for metric in metrics} for arm in ("base-u", "mix-u", "mix-2u")}
    training, categories, mesh_count, empties = [], None, 0, []
    curves_by_run = {}
    for run in plan["runs"]:
        name = f"{run['arm']}-s{run['seed']}"
        train_root = args.runs_root / name
        result = json.loads((train_root / "complete.json").read_text())
        curves = json.loads((train_root / "curves.json").read_text())
        resolved = json.loads((train_root / "resolved-config.json").read_text())
        assert result["updates"] == run["updates"] and result["test_opened"] is False
        assert resolved["training_asset_schedule"] == run["schedule"]
        assert len(curves) == run["validation_count"]
        assert curves[-1]["update"] == run["updates"]
        assert all(
            np.isfinite(x["validation_mae_m"]) and np.isfinite(x["training_loss"]) for x in curves
        )
        best = min(curves, key=lambda x: x["validation_mae_m"])
        assert (
            result["best_update"] == best["update"]
            and result["best_validation_mae_m"] == best["validation_mae_m"]
        )
        checkpoint = train_root / "best-validation.pt"
        assert result["checkpoint_sha256"] == _sha256_file(checkpoint)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        assert all(torch.isfinite(x).all() for x in state.values())
        training.append(result)
        curves_by_run[name] = curves
        aggregate_path = folder / name / "aggregate.json"
        arm = json.loads(aggregate_path.read_text())
        assert arm == complete["arms"][name]
        records = arm["surface"]["per_object"]
        assert [x["asset_id"] for x in records] == ids
        current_categories = [x["category"] for x in records]
        if categories is not None:
            assert current_categories == categories
        categories = current_categories
        for record in records:
            asset = record["asset_id"]
            assert record["cache_sha256"] == inputs[asset]["cache_sha256"]
            assert record["gt_points_sha256"] == inputs[asset]["gt_points_sha256"]
            path = folder / name / f"{asset}.npz"
            assert _sha256_file(path) == record["mesh_sha256"]
            with np.load(path, allow_pickle=False) as mesh:
                assert all(np.isfinite(mesh[key]).all() for key in mesh.files)
                assert len(mesh["surface_points_camera_m"]) == record["point_count"]
            mesh_count += 1
            if record["point_count"] == 0:
                empties.append(f"{name}/{asset}")
        for metric in metrics:
            values = [
                x["visibility"][metric]
                if metric.endswith("recall_5mm")
                else x["surface"]["0.005"][metric]
                for x in records
            ]
            arrays[run["arm"]][metric].append(values)
    assert mesh_count == 225 and len(training) == 9
    arrays = {
        arm: {key: np.asarray(value, dtype=float) for key, value in metrics.items()}
        for arm, metrics in arrays.items()
    }
    rng = np.random.default_rng(20260905)
    seed_indices = rng.integers(0, 3, size=(20000, 3))
    object_indices = rng.integers(0, 25, size=(20000, 25))
    assert bootstrap(np.zeros((3, 25)), seed_indices, object_indices) == [0.0, 0.0]
    assert bootstrap(np.ones((3, 25)), seed_indices, object_indices) == [1.0, 1.0]
    summaries = {}
    for arm, values in arrays.items():
        summaries[arm] = {
            key: dict(mean=float(a.mean()), per_seed=a.mean(axis=1).tolist())
            if np.isfinite(a).all()
            else dict(status="missing metric; no silent subset mean")
            for key, a in values.items()
        }
    comparisons = {}
    for right, left in (("mix-u", "base-u"), ("mix-2u", "base-u"), ("mix-2u", "mix-u")):
        rows = {}
        for metric in metrics:
            delta = arrays[right][metric] - arrays[left][metric]
            if not np.isfinite(delta).all():
                rows[metric] = dict(status="missing metric; no paired subset")
                continue
            category_deltas = [
                float(delta[:, np.asarray(categories) == category].mean())
                for category in sorted(set(categories))
            ]
            rows[metric] = dict(
                delta=float(delta.mean()),
                per_seed=delta.mean(axis=1).tolist(),
                bootstrap95=bootstrap(delta, seed_indices, object_indices),
                positive_pairs=int((delta > 0).sum()),
                negative_pairs=int((delta < 0).sum()),
                ties=int((delta == 0).sum()),
                category_macro_delta=float(np.mean(category_deltas)),
            )
        comparisons[f"{right}_minus_{left}"] = rows
    strata_path = MANIFEST.parent / "strata.json"
    strata_spec = json.loads(strata_path.read_text(encoding="utf-8-sig"))
    assert strata_spec["campaign_sha256"] == _sha256_file(MANIFEST)
    strata = {}
    for label, key, count in (
        ("category_in_training", "known_category_ids", 17),
        ("category_not_in_training", "untrained_category_ids", 8),
    ):
        mask = np.asarray([asset in strata_spec[key] for asset in ids])
        assert mask.sum() == count
        arm_values = {
            arm: {
                metric: dict(
                    mean=float(a[:, mask].mean()), per_seed=a[:, mask].mean(axis=1).tolist()
                )
                for metric, a in values.items()
                if np.isfinite(a[:, mask]).all()
            }
            for arm, values in arrays.items()
        }
        differences = {}
        for right, left in (("mix-u", "base-u"), ("mix-2u", "base-u"), ("mix-2u", "mix-u")):
            differences[f"{right}_minus_{left}"] = {
                metric: dict(
                    delta=float(
                        (arrays[right][metric][:, mask] - arrays[left][metric][:, mask]).mean()
                    ),
                    per_seed=(arrays[right][metric][:, mask] - arrays[left][metric][:, mask])
                    .mean(axis=1)
                    .tolist(),
                )
                for metric in metrics
                if np.isfinite(arrays[right][metric][:, mask]).all()
                and np.isfinite(arrays[left][metric][:, mask]).all()
            }
        strata[label] = dict(
            count=count,
            asset_ids=strata_spec[key],
            arm_summaries=arm_values,
            comparisons=differences,
        )
    prefix_audit = []
    for seed in (0, 1, 2):
        short = curves_by_run[f"mix-u-s{seed}"]
        long = curves_by_run[f"mix-2u-s{seed}"][: len(short)]
        delta = np.asarray(
            [
                abs(a["validation_mae_m"] - b["validation_mae_m"])
                for a, b in zip(short, long, strict=True)
            ]
        )
        prefix_audit.append(
            dict(
                seed=seed,
                maximum_validation_mae_difference_m=float(delta.max()),
                mean_validation_mae_difference_m=float(delta.mean()),
                bitwise_identical_validation_curve=bool((delta == 0).all()),
            )
        )
    result = dict(
        scope="exploratory validation25; no final test",
        status="audit passed",
        mesh_count=225,
        run_count=9,
        empty_reconstructions=empties,
        category_counts=dict(Counter(categories)),
        campaign_sha256=_sha256_file(MANIFEST),
        surface_complete_sha256=_sha256_file(complete_path),
        training=training,
        matched_input_prefix_reproducibility=prefix_audit,
        metadata_strata_supplement_sha256=_sha256_file(strata_path),
        metadata_strata=strata,
        arm_summaries=summaries,
        comparisons=comparisons,
    )
    destination = args.output
    destination.mkdir(exist_ok=False)
    (destination / "summary.json").write_text(json.dumps(result, indent=2))
    lines = [
        "# Balanced diversity: completed validation campaign",
        "",
        "Audit passed: nine training runs and225surface reconstructions; no test outcomes opened.",
        "",
        "|Arm|F5 mean|F5 seed0/1/2|Precision|Recall|Hidden recall|Chamfer mm|",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for arm, values in summaries.items():
        chamfer = values["chamfer_mean_m"].get("mean")
        chamfer_text = (
            "undefined (empty reconstruction)" if chamfer is None else f"{1000 * chamfer:.6f}"
        )
        lines.append(
            f"|{arm}|{values['fscore']['mean']:.6f}|{values['fscore']['per_seed']}|{values['precision']['mean']:.6f}|{values['recall']['mean']:.6f}|{values['hidden_recall_5mm']['mean']:.6f}|{chamfer_text}|"
        )
    lines += [
        "",
        "|Comparison|F5 delta|Exploratory95% CI|Category macro delta|",
        "|---|---:|---|---:|",
    ]
    for name, values in comparisons.items():
        f = values["fscore"]
        lines.append(
            f"|{name}|{f['delta']:.6f}|{f['bootstrap95']}|{f['category_macro_delta']:.6f}|"
        )
    lines += [
        "",
        (
            "These are reused validation data and three seeds. Category "
            "balance and shared-object query streams are controlled; MIX/2U "
            "has twice the compute and checkpoint-selection opportunities. "
            "Finite query pools, a single membership draw and incomplete "
            "near-family control remain limitations. Surface results must be "
            "reported even when negative or inconsistent across seeds."
        ),
        (
            "Matched MIX/U input prefixes showed observed nondeterminism of "
            "the training trajectory; the cause was not established. This "
            "limits trajectory pairing, not the identity of data and query "
            "schedules. All prefix discrepancies are retained in "
            "summary.json; no selective reruns were made."
        ),
    ]
    lines += [
        "",
        "## Metadata-based descriptive strata (primary all25 unchanged)",
        "",
        "|Stratum|Objects|BASE/U F5|MIX/U F5|MIX/2U F5|",
        "|---|---:|---:|---:|---:|",
    ]
    for label, stratum in strata.items():
        means = [
            stratum["arm_summaries"][arm]["fscore"]["mean"] for arm in ("base-u", "mix-u", "mix-2u")
        ]
        lines.append(f"|{label}|{stratum['count']}|{means[0]:.6f}|{means[1]:.6f}|{means[2]:.6f}|")
    lines += [
        "",
        (
            "Per-seed strata and all paired deltas are retained in "
            "summary.json. MIX114 versus old full140 is not an isolated "
            "object-count comparison because category coverage differs."
        ),
    ]
    (destination / "summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(dict(status="audit passed", runs=9, surfaces=225, comparisons=comparisons)))


if __name__ == "__main__":
    main()
