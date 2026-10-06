"""Paired object/seed summaries of the complete fixed supplement, without tuning."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import numpy as np

from scripts.common.paths import CODE_ROOT, digest, read, save, script_help
from scripts.eval.score_controls import BASE, BUDGETS, bound, ref
from scripts.eval.score_controls import OUT as SCORES
from scripts.eval.score_homebrewed import object_mean, scalars
from scripts.eval.surface_metrics import paired_dependence_sensitivity, paired_object_summary

script_help(__doc__, __name__)


OUT = BASE / "analysis-v1"
SEED = 2026091002
REPLICATES = 10000
ARMS = ("BASE", "CLASSIC", "PHOTO")
SEEDS = (0, 1, 2)


def paired(matrix, objects, scenes):
    matrix = np.asarray(matrix, float)
    assert matrix.ndim == 2 and matrix.shape[0] == len(objects) and np.isfinite(matrix).all()
    result = {
        "paired_objects": paired_object_summary(
            matrix.mean(1), objects, seed=SEED, replicates=REPLICATES
        )
    }
    if matrix.shape[1] > 1:
        sensitivity = paired_dependence_sensitivity(
            matrix, objects, scenes, seed=SEED, replicates=REPLICATES
        )
        # Only the preregistered object-plus-training-seed result is interpreted
        # for these supplements; no synthetic scene bootstrap claim is made.
        result["per_seed_object_mean"] = sensitivity["per_seed_object_mean"]
        result["object_and_seed_bootstrap95"] = sensitivity["object_and_seed_bootstrap95"]
        result["training_seeds"] = matrix.shape[1]
    return result


def population(rows_by_method, ids, budget):
    first = rows_by_method[next(iter(rows_by_method))]
    objects = [first[sid]["object_id"] for sid in ids]
    scenes = [first[sid]["scene_id"] for sid in ids]
    methods = {}
    for mid, records in rows_by_method.items():
        selected = [records[sid] for sid in ids]
        flat = [scalars(r["method"]["scores"][budget]) for r in selected]
        methods[mid] = {
            "metrics": {key: object_mean([r[key] for r in flat], objects) for key in flat[0]},
            "status_counts": dict(Counter(r["method"]["status"] for r in selected)),
            "under512": sum(r["method"]["native_count"] < 512 for r in selected),
            "empty": sum(r["method"]["native_count"] == 0 for r in selected),
        }
    values = {
        arm: np.array(
            [
                [
                    rows_by_method[f"ray-{arm}-seed{seed}"][sid]["method"]["scores"][budget][
                        "surface"
                    ]["0.005"]["fscore"]
                    for seed in SEEDS
                ]
                for sid in ids
            ]
        )
        for arm in ARMS
    }
    arms = {
        arm: {
            "F5": object_mean(matrix.mean(1), objects),
            "per_seed_F5": [object_mean(matrix[:, i], objects)["mean"] for i in SEEDS],
        }
        for arm, matrix in values.items()
    }
    contrasts = {
        left + "-" + right: paired(values[left] - values[right], objects, scenes)
        for left, right in (("PHOTO", "BASE"), ("CLASSIC", "BASE"), ("PHOTO", "CLASSIC"))
    }
    for reference in ("ray-pretrained", "INPUT"):
        if reference in rows_by_method:
            scores = np.array(
                [
                    rows_by_method[reference][sid]["method"]["scores"][budget]["surface"]["0.005"][
                        "fscore"
                    ]
                    for sid in ids
                ]
            )[:, None]
            for arm in ARMS:
                contrasts[arm + "-" + reference] = paired(values[arm] - scores, objects, scenes)
    return {
        "observations": len(ids),
        "objects": len(set(objects)),
        "methods": methods,
        "arms": arms,
        "F5_contrasts": contrasts,
    }


def rgb_paired(neutral, original, ids, budget):
    first = neutral[next(iter(neutral))]
    objects = np.array([first[sid]["object_id"] for sid in ids])
    scenes = [first[sid]["scene_id"] for sid in ids]
    method_effects, matrices = {}, {}
    for mid, by_id in neutral.items():
        neutral_values = [scalars(by_id[sid]["method"]["scores"][budget]) for sid in ids]
        original_values = [scalars(original[sid]["methods"][mid]["scores"][budget]) for sid in ids]
        metrics = {}
        for key in neutral_values[0]:
            diff = [
                n[key] - o[key] if n[key] is not None and o[key] is not None else None
                for n, o in zip(neutral_values, original_values, strict=True)
            ]
            summary = object_mean(diff, objects)
            good = np.array([v is not None for v in diff])
            if key in ("fscore@0.005", "precision@0.005", "recall@0.005"):
                assert good.all()
                summary["paired_objects"] = paired_object_summary(
                    diff, objects, seed=SEED, replicates=REPLICATES
                )
            metrics[key] = summary
        matrices[mid] = np.array(
            [
                n["fscore@0.005"] - o["fscore@0.005"]
                for n, o in zip(neutral_values, original_values, strict=True)
            ]
        )
        method_effects[mid] = {
            "neutral_minus_original": metrics,
            "original_F5": object_mean([o["fscore@0.005"] for o in original_values], objects),
            "neutral_F5": object_mean([n["fscore@0.005"] for n in neutral_values], objects),
        }
    by_arm = {
        arm: np.stack([matrices[f"ray-{arm}-seed{seed}"] for seed in SEEDS], axis=1) for arm in ARMS
    }
    arms = {arm: paired(values, objects, scenes) for arm, values in by_arm.items()}
    difference_of_effects = paired(by_arm["PHOTO"] - by_arm["BASE"], objects, scenes)
    return {
        "observations": len(ids),
        "objects": len(set(objects)),
        "methods": method_effects,
        "arm_neutral_minus_original_F5": arms,
        "change_in_PHOTO_minus_BASE_F5": difference_of_effects,
        "conditional_metric_rule": (
            "Other metric differences use only pairs with both values "
            "defined, with explicit paired denominators; no imputation."
        ),
    }


def native_changes(neutral, ids):
    result = {}
    for mid, rows in neutral.items():
        selected = [rows[sid] for sid in ids]
        objects = [r["object_id"] for r in selected]
        changes = [r["RGB_neutral_minus_original_cloud"] for r in selected]
        metrics = {}
        for key in (
            "native_count_difference",
            "symmetric_mean_distance_m",
            "bidirectional_p95_m",
            "bidirectional_max_m",
        ):
            values = [r[key] for r in changes]
            valid = [v for v in values if v is not None]
            metrics[key] = {
                "object_balanced": object_mean(values, objects),
                "raw_median": float(np.median(valid)) if valid else None,
                "raw_quantile05_95": np.quantile(valid, [0.05, 0.95]).tolist() if valid else None,
            }
        result[mid] = {
            "observations": len(ids),
            "exact_array_equal_count": sum(r["exact_array_equal"] for r in changes),
            "both_nonempty_count": sum(r["both_nonempty"] for r in changes),
            "metrics": metrics,
        }
    return result


def main():
    assert not OUT.exists()
    complete = read(SCORES / "complete.json")
    assert complete["status"] == "all20_supplement_jobs_scored_locally"
    assert complete["scoring_job_sha256"] == digest(SCORES / "job.json")
    job = read(SCORES / "job.json")
    protocol = read(bound(job["protocol"]))
    assert SEED == 2026091002 and str(SEED) in protocol["metric_rules"]["intervals"]
    assert list(BUDGETS.values()) == list(protocol["metric_rules"]["point_budgets"].values())
    cloud_proofs = []
    all_parity = []
    for value in job["parity_records"]:
        proof = read(bound(value))
        parity = proof["independently_recomputed_local_parity"]
        assert all(p["passed"] for p in parity)
        all_parity.extend(parity)
        cloud_proofs.append(value)
    assert len(all_parity) == 30
    cohorts = {"HB198-neutral128": {}, "synthetic-holdout720": {}}
    source_jobs = []
    for jid, source in complete["jobs"].items():
        path = bound(source)
        scored = read(path)
        assert (
            scored["status"] == "all_fixed_job_observations_scored"
            and scored["scoring_job_sha256"] == complete["scoring_job_sha256"]
        )
        entry = scored["entry"]
        assert entry["id"] == jid and len(scored["rows"]) == entry["observations"]
        rows = {}
        for r in scored["rows"]:
            row_path = path.parent / "rows" / (r["sample_id"] + ".json")
            assert digest(row_path) == r["sha256"]
            row = read(row_path)
            assert (
                row["sample_id"] == r["sample_id"]
                and row["method_id"] == entry["method_id"]
                and row["condition"] == entry["condition"]
            )
            rows[row["sample_id"]] = row
        assert entry["method_id"] not in cohorts[entry["condition"]]
        cohorts[entry["condition"]][entry["method_id"]] = rows
        source_jobs.append(source)
    assert len(cohorts["HB198-neutral128"]) == 9 and len(cohorts["synthetic-holdout720"]) == 11
    old = {r["sample_id"]: r for r in read(bound(job["old_scores"]["per-observation.json"]))}
    old_summary = read(bound(job["old_scores"]["summaries.json"]))
    summaries = {}
    for condition, records in cohorts.items():
        ids = [
            r["sample_id"] for r in read(bound(protocol["inputs"][condition]["manifest"]))["rows"]
        ]
        assert all(set(rows) == set(ids) for rows in records.values())
        first = records[next(iter(records))]
        selected = {"all": ids, "eligible": [sid for sid in ids if first[sid]["input_eligible"]]}
        if condition == "synthetic-holdout720":
            assert len(ids) == len(selected["eligible"]) == 720
        else:
            assert len(ids) == 198 and len(selected["eligible"]) == 182 and set(ids) == set(old)
        condition_summary = {}
        for label, sample_ids in selected.items():
            values = {budget: population(records, sample_ids, budget) for budget in BUDGETS}
            if condition == "HB198-neutral128":
                comparisons = {
                    budget: rgb_paired(records, old, sample_ids, budget) for budget in BUDGETS
                }
                for budget, comparison in comparisons.items():
                    for mid, effect in comparison["methods"].items():
                        expected = old_summary[label]["budgets"][budget]["methods"][mid]["metrics"][
                            "fscore@0.005"
                        ]["mean"]
                        assert abs(effect["original_F5"]["mean"] - expected) < 1e-12
                condition_summary[label] = {
                    "scores": values,
                    "paired_RGB": comparisons,
                    "native_changes": native_changes(records, sample_ids),
                }
            else:
                condition_summary[label] = {"scores": values}
        summaries[condition] = condition_summary
    OUT.mkdir()
    save(OUT / "summaries.json", summaries)
    with (OUT / "F5-per-observation.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "condition",
                "method_id",
                "sample_id",
                "object_id",
                "status",
                "native_count",
                "F5_cap16384",
                "F5_cap512",
            ]
        )
        for condition, methods in cohorts.items():
            for mid, rows in methods.items():
                for sid, row in rows.items():
                    writer.writerow(
                        [
                            condition,
                            mid,
                            sid,
                            row["object_id"],
                            row["method"]["status"],
                            row["method"]["native_count"],
                            *[
                                row["method"]["scores"][b]["surface"]["0.005"]["fscore"]
                                for b in BUDGETS
                            ],
                        ]
                    )
    record = {
        "status": "complete_fixed_supplement_summaries_no_test_driven_selection",
        "scoring_complete": ref(SCORES / "complete.json"),
        "scoring_job": ref(SCORES / "job.json"),
        "protocol": job["protocol"],
        "source_jobs": source_jobs,
        "parity_records": cloud_proofs,
        "all30_parity_checks": all_parity,
        "summaries": ref(OUT / "summaries.json"),
        "F5_csv": ref(OUT / "F5-per-observation.csv"),
        "source_sha256": {
            name: digest(CODE_ROOT / "scripts" / name)
            for name in (
                Path(__file__).relative_to(CODE_ROOT / "scripts").as_posix(),
                "eval/surface_metrics.py",
                "eval/score_homebrewed.py",
            )
        },
        "bootstrap_seed": SEED,
        "bootstrap_replicates": REPLICATES,
        "old_HB_primary_effect_unchanged": old_summary["all"]["budgets"]["cap16384"][
            "F5_contrasts"
        ]["PHOTO-BASE"],
        "scope": (
            "Post-study diagnostics. Same nine final checkpoints; all198HB "
            "and all720synthetic observations retained. Primary HB remains "
            "historical. Finite three-seed uncertainty and uneven training "
            "repair remain. No independent confirmation of a causal mechanism."
        ),
    }
    save(OUT / "complete.json", record)
    print("SUPPLEMENT AGGREGATION COMPLETE", OUT)


if __name__ == "__main__":
    main()
