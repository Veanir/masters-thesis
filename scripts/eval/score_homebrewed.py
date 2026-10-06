"""Score a complete frozen method list; HB is read only after campaign freeze."""

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from scripts.common.paths import CODE_ROOT
from scripts.common.paths import ROOT as WORKSPACE_ROOT
from scripts.eval.surface_metrics import (
    paired_dependence_sensitivity,
    paired_object_summary,
    score_cloud,
    select_points,
    validate_cloud,
)

ROOT = WORKSPACE_ROOT
BUDGETS = {"cap16384": 16384, "cap512": 512, "shared_nonfailed_cap": None, "native": None}
FAILURES = {"empty_prediction", "numerical_failure", "insufficient_input"}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8388608), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def bound(ref):
    path = (ROOT / ref["path"]).resolve()
    path.relative_to(ROOT.resolve())
    assert sha(path) == ref["sha256"], "Artifact binding failed: " + str(path)
    return path


def check_prediction(points, row, input_eligible):
    points = validate_cloud(points)
    count = len(points)
    status = row["status"]
    assert count == row["count"] and status in FAILURES | {"small_prediction", "predicted"}
    if not input_eligible:
        assert status == "insufficient_input" and count == 0
    elif status in FAILURES:
        assert status != "insufficient_input" and count == 0
    elif status == "small_prediction":
        assert 0 < count < 512
    else:
        assert count >= 512
    return points


def score_observation(sid, cache, clouds, statuses):
    truth = validate_cloud(cache["truth_points_camera_m"])
    dense = validate_cloud(cache["input_dense_points_camera_m"])
    masks = {
        key: np.asarray(cache[key], dtype=bool)
        for key in ["visible", "occluded", "uncertain", "uncovered_by_dense_input"]
    }
    assert all(mask.shape == (len(truth),) for mask in masks.values())
    assert np.all(masks["visible"].astype(int) + masks["occluded"] + masks["uncertain"] == 1)
    clouds = {key: validate_cloud(value) for key, value in clouds.items()}
    assert set(clouds) == set(statuses)
    eligible = [len(value) for value in clouds.values() if len(value) >= 512]
    shared = min(16384, min(eligible)) if eligible else 512
    tree = cKDTree(truth)
    result = {}
    for method, cloud in clouds.items():
        scores = {}
        by_count = {}
        for budget, limit in BUDGETS.items():
            if budget == "shared_nonfailed_cap":
                limit = shared
            count = min(len(cloud), limit) if limit is not None else len(cloud)
            if count not in by_count:
                points = select_points(cloud, limit, sample_id=sid)
                by_count[count] = score_cloud(
                    points,
                    truth,
                    tree,
                    masks,
                    dense,
                    cache["full_scene_depth_m"],
                    cache["intrinsics"],
                )
            scores[budget] = by_count[count]
        result[method] = {
            "status": statuses[method],
            "native_count": len(cloud),
            "under512": len(cloud) < 512,
            "shared_count_equal": len(cloud) >= 512,
            "scores": scores,
        }
    return {"shared_nonfailed_point_cap": shared, "methods": result}


def scalars(score):
    result = {
        key: score[key]
        for key in [
            "count",
            "chamfer_mean_m_success_only",
            "chamfer_capped_100mm_m",
            "added_count",
            "added_precision5",
            "known_free_count",
            "known_free_rate_all",
            "known_free_rate_supported",
            "supported_prediction_count",
        ]
    }
    for threshold, metrics in score["surface"].items():
        for name, value in metrics.items():
            result[name + "@" + threshold] = value
    for group, metrics in score["gt_subsets"].items():
        for name, value in metrics.items():
            result["GT_" + group + "_" + name] = value
    return result


def object_mean(values, objects):
    values = np.asarray([np.nan if value is None else value for value in values], float)
    objects = np.asarray(objects)
    assert not np.isinf(values).any()
    means = []
    for obj in np.unique(objects):
        selected = values[objects == obj]
        selected = selected[np.isfinite(selected)]
        if len(selected):
            means.append(float(selected.mean()))
    return {
        "mean": float(np.mean(means)) if means else None,
        "defined_observations": int(np.isfinite(values).sum()),
        "total_observations": len(values),
        "defined_objects": len(means),
        "total_objects": len(np.unique(objects)),
    }


def summarize(records, methods, protocol):
    populations = {
        "all": records,
        "eligible": [r for r in records if r["input_eligible"]],
        "registration_passed": [r for r in records if r["source_registration_status"] == "passed"],
    }
    populations.update(
        {
            name: [r for r in records if r["visibility_bin"] == name]
            for name in ["low", "medium", "high"]
        }
    )
    matrix = {(m["arm"], m["seed"]): m["id"] for m in methods if m["role"] == "adapted"}
    output = {}
    for population, rows in populations.items():
        if not rows:
            output[population] = {"observations": 0}
            continue
        objects = np.array([r["obj_id"] for r in rows])
        scenes = np.array([r["scene_id"] for r in rows])
        budgets = {}
        for budget in BUDGETS:
            summaries = {}
            for method in methods:
                mid = method["id"]
                values = [scalars(r["methods"][mid]["scores"][budget]) for r in rows]
                summaries[mid] = {
                    "metrics": {
                        key: object_mean([v[key] for v in values], objects) for key in values[0]
                    },
                    "status_counts": dict(Counter(r["methods"][mid]["status"] for r in rows)),
                    "native_under512": sum(r["methods"][mid]["under512"] for r in rows),
                }
            arm_means = {}
            for arm in ["BASE", "CLASSIC", "PHOTO"]:
                per_seed = np.array(
                    [
                        [
                            r["methods"][matrix[(arm, seed)]]["scores"][budget]["surface"]["0.005"][
                                "fscore"
                            ]
                            for seed in [0, 1, 2]
                        ]
                        for r in rows
                    ]
                )
                arm_means[arm] = object_mean(per_seed.mean(1), objects)
            contrasts = {}
            for left, right in [("PHOTO", "BASE"), ("CLASSIC", "BASE"), ("PHOTO", "CLASSIC")]:
                values = np.array(
                    [
                        [
                            r["methods"][matrix[(left, seed)]]["scores"][budget]["surface"][
                                "0.005"
                            ]["fscore"]
                            - r["methods"][matrix[(right, seed)]]["scores"][budget]["surface"][
                                "0.005"
                            ]["fscore"]
                            for seed in [0, 1, 2]
                        ]
                        for r in rows
                    ]
                )
                contrasts[left + "-" + right] = {
                    "paired_objects": paired_object_summary(
                        values.mean(1),
                        objects,
                        seed=protocol["object_bootstrap_seed"],
                        replicates=protocol["bootstrap_replicates"],
                    ),
                    "dependence_sensitivity": paired_dependence_sensitivity(
                        values,
                        objects,
                        scenes,
                        seed=protocol["sensitivity_seed"],
                        replicates=protocol["bootstrap_replicates"],
                    ),
                }
            budgets[budget] = {
                "methods": summaries,
                "arm_F5_seed_averaged": arm_means,
                "F5_contrasts": contrasts,
            }
        output[population] = {
            "observations": len(rows),
            "objects": len(np.unique(objects)),
            "scenes": len(np.unique(scenes)),
            "budgets": budgets,
        }
    return output


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--job", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    job = read(a.job)
    assert job["status"] == "frozen_final_HB_scoring_job" and not a.output.exists()
    protocol_path = bound(job["protocol"])
    protocol = read(protocol_path)
    campaign = read(bound(job["campaign"]))
    assert campaign["status"] == "frozen_evolution_campaign" and campaign[
        "evaluation_protocol_sha256"
    ] == sha(protocol_path)
    assert job["campaign_freeze_sha256"] == sha(bound(job["campaign"]))
    assert job["source_files_sha256"] == {
        name: sha(CODE_ROOT / "scripts" / name)
        for name in ["eval/score_homebrewed.py", "eval/surface_metrics.py"]
    }
    assert (
        sha(ROOT / "scripts/eval/surface_metrics.py")
        == protocol["references"]["metric_core"]["sha256"]
    )
    methods = job["methods"]
    assert methods == campaign["evaluation_methods"]
    assert (
        len({m["id"] for m in methods}) == len(methods)
        and sum(m["role"] == "input" for m in methods) == 1
    )
    assert {"pretrained", "historical"}.issubset({m["role"] for m in methods})
    adapted = [m for m in methods if m["role"] == "adapted"]
    assert len(adapted) == 9 and {(m["arm"], m["seed"]) for m in adapted} == {
        (arm, seed) for arm in ["BASE", "CLASSIC", "PHOTO"] for seed in [0, 1, 2]
    }
    cache_path = bound(job["cache"])
    cache = read(cache_path)
    assert cache["protocol_sha256"] == sha(protocol_path)
    assert [r["sample_id"] for r in cache["rows"]] == protocol["sample_ids"] and len(
        cache["rows"]
    ) == 198
    expected = set(protocol["sample_ids"])
    predictions = {}
    for method in methods:
        if method["role"] == "input":
            continue
        path = bound(job["prediction_manifests"][method["id"]])
        manifest = read(path)
        assert (
            manifest["status"]
            in [
                "bound_ray_observation_evaluation_complete",
                "bound_observation_evaluation_complete",
            ]
            and not manifest["gt_uploaded"]
        )
        inference_path = bound(job["prediction_jobs"][method["id"]])
        inference = read(inference_path)
        assert manifest["job_sha256"] == sha(inference_path) and manifest["job"] == inference
        assert (
            inference["method_id"] == method["id"]
            and inference["campaign_sha256"] == job["campaign_freeze_sha256"]
        )
        if method["role"] == "adapted":
            assert (
                inference["checkpoint_step"] == 25600 and inference["checkpoint_sha256"] is not None
            )
        assert (
            len(manifest["rows"]) == 198 and {r["sample_id"] for r in manifest["rows"]} == expected
        )
        predictions[method["id"]] = {
            "folder": path.parent,
            "rows": {r["sample_id"]: r for r in manifest["rows"]},
        }
    a.output.mkdir()
    records = []
    tick = time.monotonic()
    for reference in cache["rows"]:
        sid = reference["sample_id"]
        path = cache_path.parent / (sid + ".npz")
        assert sha(path) == reference["cache_sha256"]
        with np.load(path) as z:
            arrays = {key: z[key] for key in z.files}
        assert len(arrays["truth_points_camera_m"]) == protocol["reference_points"]
        assert len(arrays["input_dense_points_camera_m"]) == reference["input_pixels"]
        clouds = {}
        statuses = {}
        for method in methods:
            mid = method["id"]
            if method["role"] == "input":
                clouds[mid] = arrays["input_dense_points_camera_m"]
                statuses[mid] = "observed_input" if len(clouds[mid]) else "empty_input"
                continue
            source = predictions[mid]
            row = source["rows"][sid]
            pred = source["folder"] / (sid + ".npz")
            assert sha(pred) == row["sha256"]
            with np.load(pred) as z:
                clouds[mid] = check_prediction(
                    z["points_camera_m"], row, reference["input_eligible"]
                )
            statuses[mid] = row["status"]
            if row["status"] == "numerical_failure":
                invalid = row["invalid_artifact"]
                bad = (source["folder"] / invalid["filename"]).resolve()
                bad.relative_to(source["folder"].resolve())
                assert sha(bad) == invalid["sha256"]
                try:
                    validate_cloud(np.load(bad, allow_pickle=False))
                except ValueError:
                    pass
                else:
                    raise AssertionError("Numerical failure artifact is a valid cloud")
        record = {
            key: reference[key]
            for key in [
                "sample_id",
                "obj_id",
                "scene_id",
                "input_eligible",
                "source_registration_status",
                "visibility_bin",
            ]
        }
        record.update(score_observation(sid, arrays, clouds, statuses))
        records.append(record)
        if len(records) % 11 == 0:
            save(
                a.output / "progress.json",
                {"observations": len(records), "total": 198, "seconds": time.monotonic() - tick},
            )
            print("FINAL SCORE", len(records), flush=True)
    save(a.output / "per-observation.json", records)
    summaries = summarize(records, methods, protocol)
    save(a.output / "summaries.json", summaries)
    save(
        a.output / "complete.json",
        {
            "status": "all198_final_HB_observations_scored",
            "job_sha256": sha(a.job),
            "campaign_sha256": job["campaign_freeze_sha256"],
            "per_observation_sha256": sha(a.output / "per-observation.json"),
            "summaries_sha256": sha(a.output / "summaries.json"),
            "observations": len(records),
            "method_count": len(methods),
            "seconds": time.monotonic() - tick,
            "primary": summaries["all"]["budgets"]["cap16384"]["F5_contrasts"]["PHOTO-BASE"],
            "scope": (
                "All198 retained; all fixed methods/seeds determine shared count. "
                "Undefined Chamfer/free-space/subset quantities retain explicit "
                "denominators. No test-driven selection."
            ),
        },
    )


if __name__ == "__main__":
    main()
