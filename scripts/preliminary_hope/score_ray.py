"""Evaluate all frozen runs with common point budgets and object-paired uncertainty."""

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

import masters_rgbd
from masters_rgbd.b1.projected_mesh import _sample_mesh_surface
from scripts.common.sampled_surface_metrics import check, score, select, visibility
from scripts.preliminary_hope.bootstrap_objects import balanced_mean, paired_interval
from scripts.preliminary_hope.prepare_training_plan import digest, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cohort", choices=("hope146", "ycbv42", "abo24"), required=True)
    args = parser.parse_args()
    check()
    arms = ("base", "classic", "photo")
    models = ["pretrained"] + [f"{a}-seed{s}" for s in range(3) for a in arms]
    observation_manifest = args.observations / "complete.json"
    observations = json.loads(observation_manifest.read_text())
    count, role = {
        "hope146": (146, "project_heldout_real"),
        "ycbv42": (42, "exposed_real_diagnostic"),
        "abo24": (24, "adaptation_holdout"),
    }[args.cohort]
    assert observations["status"] == "complete" and len(observations["rows"]) == count
    assert all(r["evaluation_role"] == role for r in observations["rows"])
    freeze = args.checkpoints / "frozen-checkpoints.json"
    frozen = json.loads(freeze.read_text(encoding="utf-8-sig"))
    assert frozen["status"] == "frozen" and len(frozen["runs"]) == 9
    assert {r["run"] for r in frozen["runs"]} == set(models) - {"pretrained"}
    assert frozen["cohorts"][args.cohort]["observation_manifest_sha256"] == digest(
        observation_manifest
    )
    assert frozen["analysis"]["script_sha256"] == digest(__file__)
    assert frozen["analysis"]["statistics_sha256"] == digest(
        Path(__file__).parents[2].joinpath("scripts/preliminary_hope/bootstrap_objects.py")
    )
    for filename, sha in frozen["scripts"].items():
        assert digest(Path(__file__).parent / filename) == sha
    package_root = Path(masters_rgbd.__file__).resolve().parents[1]
    for filename, sha in frozen["scoring_sources"].items():
        assert digest(package_root / filename) == sha
    predictions = {}
    bindings = {}
    expected_ids = {r["sample_id"] for r in observations["rows"]}
    for name in models:
        path = args.predictions / name / "complete.json"
        manifest = json.loads(path.read_text())
        assert manifest["status"] == "complete" and manifest["contract"]["model"] == name
        assert manifest["contract"]["frozen_checkpoints_sha256"] == digest(freeze)
        assert manifest["contract"]["cohort"] == args.cohort
        assert (
            manifest["contract"]["runner_sha256"]
            == frozen["scripts"]["preliminary_hope/predict_ray.py"]
        )
        assert (
            manifest["contract"]["view_chunk_script_sha256"]
            == frozen["scripts"]["common/ray_view_chunks.py"]
        )
        assert manifest["contract"]["view_chunk"] == 3 and manifest["contract"]["threshold"] == 5
        assert manifest["contract"]["total_views"] == 22
        assert (
            manifest["contract"]["input_manifest_sha256"]
            == frozen["cohorts"][args.cohort]["input_manifest_sha256"]
        )
        assert len(manifest["records"]) == count
        assert {r["sample_id"] for r in manifest["records"]} == expected_ids
        predictions[name] = {r["sample_id"]: r for r in manifest["records"]}
        bindings[name] = digest(path)
    args.output.mkdir(parents=True, exist_ok=False)
    binding = {
        "started_at": datetime.now(UTC).isoformat(),
        "script_sha256": digest(__file__),
        "scorer_sha256": digest(
            Path(__file__).parents[2].joinpath("scripts/common/sampled_surface_metrics.py")
        ),
        "observation_manifest_sha256": digest(observation_manifest),
        "prediction_manifests": bindings,
        "frozen_checkpoints_sha256": digest(freeze),
        "reference_count": 262144,
        "primary_count": 512,
        "bootstrap": {"unit": "object", "replicates": 10000, "seed": 20260906},
        "cost_usd": 0,
        "cohort": args.cohort,
        "evaluation_role": role,
        "training_seeds": [0, 1, 2],
        "statistics_sha256": digest(
            Path(__file__).parents[2].joinpath("scripts/preliminary_hope/bootstrap_objects.py")
        ),
        "primary_comparison": (
            "photo-seed-mean minus base-seed-mean, F5 at 512 points, object-balanced"
        ),
        "secondary_comparison": "photo-seed-mean minus classic-seed-mean",
    }
    save_json(args.output / "binding.json", binding)
    records = []
    for row in observations["rows"]:
        tick = time.perf_counter()
        name = row["sample_id"]
        path = args.observations / (name + ".npz")
        assert digest(path) == row["output_sha256"]
        with np.load(path, allow_pickle=False) as stored:
            data = {k: stored[k] for k in stored.files}
        truth = _sample_mesh_surface(
            data["mesh_vertices_camera_m"],
            data["mesh_faces"],
            count=262144,
            seed=20260905 + row["ordinal"],
        )
        truth_tree = cKDTree(truth)
        partial = data["partial_points_camera_m"]
        input_tree = cKDTree(partial)
        flags = visibility(truth, data)
        masks = {k: flags[k] for k in ("visible", "occluded", "uncertain")}
        masks["historical_hidden"] = ~flags["supported"] | flags["occluded"]
        masks["uncovered_by_input512"] = input_tree.query(truth, workers=2)[0] > 0.005
        v, u = np.where(data["mask"].astype(bool) & (data["depth_m"] > 0))
        z = data["depth_m"][v, u]
        fx, fy, cx, cy = data["intrinsics"]
        dense = np.column_stack(((u - cx) * z / fx, -(v - cy) * z / fy, -z))
        clouds = {"input512": partial, "input_dense": dense}
        source_hashes = {}
        for model in models:
            path = args.predictions / model / (name + ".npz")
            source_hashes[model] = digest(path)
            assert source_hashes[model] == predictions[model][name]["output_sha256"]
            with np.load(path, allow_pickle=False) as stored:
                clouds[model] = stored["points_camera_m"]
            assert np.isfinite(clouds[model]).all() and clouds[model].shape[1:] == (3,)
        result = {}
        visual = {"truth": select(truth, 4096), "rgb": data["rgb"]}
        for model, cloud in clouds.items():
            result[model] = {}
            for budget in (512, 16384, "native"):
                points = cloud if budget == "native" else select(cloud, budget)
                result[model][str(budget)] = score(
                    points, truth, truth_tree, masks, input_tree, data
                )
            visual[model] = select(cloud, 4096)
        record = {
            **row,
            "models": result,
            "source_hashes": source_hashes,
            "seconds": time.perf_counter() - tick,
        }
        save_json(args.output / (name + ".json"), record)
        np.savez_compressed(args.output / (name + "-visual.npz"), **visual)
        records.append(record)
        print(
            json.dumps(
                {
                    "sample_id": name,
                    "f5": {m: r["512"]["surface"]["0.005"]["fscore"] for m, r in result.items()},
                    "seconds": record["seconds"],
                }
            ),
            flush=True,
        )
    summary = {}
    for model in ("input512", "input_dense", *models):
        summary[model] = {}
        for budget in ("512", "16384", "native"):
            cells = [r["models"][model][budget] for r in records]
            summary[model][budget] = {
                "mean_f5": balanced_mean([c["surface"]["0.005"]["fscore"] for c in cells], records),
                "mean_p5": balanced_mean(
                    [c["surface"]["0.005"]["precision"] for c in cells], records
                ),
                "mean_r5": balanced_mean([c["surface"]["0.005"]["recall"] for c in cells], records),
                "empty": sum(c["count"] == 0 for c in cells),
                "count_min": min(c["count"] for c in cells),
                "count_max": max(c["count"] for c in cells),
                "occluded_recall5": balanced_mean(
                    [c["gt_subsets"]["occluded"]["recall5"] for c in cells], records
                ),
                "occluded_recall_eligible_observations": sum(
                    c["gt_subsets"]["occluded"]["recall5"] is not None for c in cells
                ),
                "mean_known_free_rate": balanced_mean(
                    [c["known_free_rate_all"] for c in cells], records
                ),
            }
    f5 = {
        m: np.array([r["models"][m]["512"]["surface"]["0.005"]["fscore"] for r in records])
        for m in summary
    }
    for arm in arms:
        f5[f"{arm}-seed-mean"] = np.mean([f5[f"{arm}-seed{s}"] for s in range(3)], axis=0)
    pairs = [
        (f"{arm}-seed{s}", reference)
        for s in range(3)
        for arm in arms
        for reference in ("pretrained", "input512")
    ]
    pairs += [
        (f"{a}-seed{s}", f"{b}-seed{s}")
        for s in range(3)
        for a, b in (("classic", "base"), ("photo", "base"), ("photo", "classic"))
    ]
    pairs += [
        (f"{arm}-seed-mean", reference) for arm in arms for reference in ("pretrained", "input512")
    ]
    pairs += [
        (f"{a}-seed-mean", f"{b}-seed-mean")
        for a, b in (("classic", "base"), ("photo", "base"), ("photo", "classic"))
    ]
    comparisons = {f"{a} minus {b}": paired_interval(f5[a] - f5[b], records) for a, b in pairs}
    save_json(
        args.output / "complete.json",
        {
            **binding,
            "status": "complete",
            "ended_at": datetime.now(UTC).isoformat(),
            "rows": records,
            "summary": summary,
            "paired_f5_comparisons": comparisons,
            "photo_status": "trained_on_reviewed_bound_data",
            "evaluation_scope": (
                "project-held-out HOPE primary; exposed YCB-V diagnostic; "
                "synthetic ABO secondary; not proven unseen pretraining objects"
            ),
            "inference_limits": (
                "Intervals condition on the three fixed training seeds. Secondary "
                "comparisons are descriptive; no multiplicity-adjusted "
                "significance claims."
            ),
        },
    )


if __name__ == "__main__":
    main()
