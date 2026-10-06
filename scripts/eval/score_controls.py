"""Local scoring of fixed supplementary jobs as verified downloads become ready."""

from __future__ import annotations

import argparse
import concurrent.futures
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

import scripts.eval.score_homebrewed as frozen
from scripts.common.paths import ROOT, digest, read, save

BASE = ROOT / "runs/thesis-supplement-20260910"
OUT = BASE / "scores-v1"
PREP = BASE / "inference-preparation-v1"
OLD_SCORES = ROOT / "runs/research-evolution-evaluation-20260907/final-scores-20260909-v6"
HB_CACHE = ROOT / "runs/research-evolution-evaluation-20260907/HB-reference-cache-v1"
SOURCES = ("eval/score_controls.py", "eval/score_homebrewed.py", "eval/surface_metrics.py")
BUDGETS = {"cap16384": 16384, "cap512": 512}
frozen.BUDGETS = BUDGETS


def ref(path):
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": digest(path)}


def bound(value):
    path = ROOT / value["path"]
    assert digest(path) == value["sha256"], path
    return path


def compare_clouds(new, old):
    result = {
        "new_native_count": len(new),
        "old_native_count": len(old),
        "native_count_difference": len(new) - len(old),
        "exact_array_equal": bool(np.array_equal(new, old)),
        "both_nonempty": bool(len(new) and len(old)),
        "symmetric_mean_distance_m": None,
        "bidirectional_p95_m": None,
        "bidirectional_max_m": None,
    }
    if len(new) and len(old):
        ab = cKDTree(old).query(new, workers=2)[0]
        ba = cKDTree(new).query(old, workers=2)[0]
        result.update(
            symmetric_mean_distance_m=float((ab.mean() + ba.mean()) / 2),
            bidirectional_p95_m=float(max(np.quantile(ab, 0.95), np.quantile(ba, 0.95))),
            bidirectional_max_m=float(max(ab.max(), ba.max())),
        )
    return result


def one(reference, cache_folder, entry, prediction, original, output):
    sid = reference["sample_id"]
    path = Path(cache_folder) / (sid + ".npz")
    assert digest(path) == reference["cache_sha256"]
    with np.load(path, allow_pickle=False) as z:
        arrays = {key: z[key] for key in z.files}
    assert len(arrays["truth_points_camera_m"]) == 262144
    assert len(arrays["input_dense_points_camera_m"]) == reference["input_pixels"]
    if prediction is None:
        points = arrays["input_dense_points_camera_m"]
        status = "observed_input" if len(points) else "empty_input"
    else:
        pred_path, pred_row = prediction
        assert digest(Path(pred_path)) == pred_row["sha256"]
        with np.load(pred_path, allow_pickle=False) as z:
            assert z.files == ["points_camera_m"]
            points = frozen.check_prediction(
                z["points_camera_m"], pred_row, reference["input_eligible"]
            )
        status = pred_row["status"]
    result = frozen.score_observation(
        sid, arrays, {entry["method_id"]: points}, {entry["method_id"]: status}
    )
    # The original helper computes a shared-cap field, but only the two fixed
    # budget scores are evaluated and reported by this supplement.
    method = result["methods"][entry["method_id"]]
    assert list(method["scores"]) == list(BUDGETS)
    record = {
        "sample_id": sid,
        "condition": entry["condition"],
        "method_id": entry["method_id"],
        "input_eligible": reference["input_eligible"],
        "object_id": str(reference.get("obj_id", reference.get("key"))),
        "scene_id": str(reference["scene_id"])
        if "scene_id" in reference
        else sid.rsplit("-v", 1)[0],
        "source_registration_status": reference.get("source_registration_status"),
        "reference_sha256": reference["cache_sha256"],
        "prediction_sha256": prediction[1]["sha256"] if prediction else None,
        "method": method,
    }
    if original:
        old_path, old_row = original
        assert digest(Path(old_path)) == old_row["sha256"]
        with np.load(old_path, allow_pickle=False) as z:
            old = frozen.check_prediction(
                z["points_camera_m"], old_row, reference["input_eligible"]
            )
        record["RGB_neutral_minus_original_cloud"] = compare_clouds(points, old)
        record["original_prediction_sha256"] = old_row["sha256"]
    destination = Path(output) / (sid + ".json")
    assert not destination.exists()
    save(destination, record)
    return {"sample_id": sid, "sha256": digest(destination), "status": status}


def score_job(job, entry, folder=None, verification=None):
    reference_path = bound(job["references"][entry["condition"]])
    references = read(reference_path)["rows"]
    assert len(references) == entry["observations"]
    predictions, originals = {}, {}
    if folder:
        manifest_path = folder / "complete.json"
        assert ref(manifest_path) == verification["prediction_manifest"]
        assert verification["all_prediction_bytes_and_shapes_verified"]
        manifest = read(manifest_path)
        assert (
            manifest["status"] == "bound_ray_observation_evaluation_complete"
            and manifest["gt_uploaded"] is False
        )
        assert manifest["job_sha256"] == entry["job_sha256"]
        assert (
            manifest["job"]["method_id"] == entry["method_id"]
            and manifest["job"]["condition"] == entry["condition"]
        )
        assert manifest["job"]["supplement_protocol_sha256"] == job["protocol"]["sha256"]
        predictions = {
            r["sample_id"]: (str(folder / (r["sample_id"] + ".npz")), r) for r in manifest["rows"]
        }
        assert set(predictions) == {r["sample_id"] for r in references}
        if entry["condition"] == "HB198-neutral128":
            original_path = bound(job["original_prediction_manifests"][entry["method_id"]])
            original = read(original_path)
            assert original["job"]["checkpoint_sha256"] == manifest["job"]["checkpoint_sha256"]
            originals = {
                r["sample_id"]: (str(original_path.parent / (r["sample_id"] + ".npz")), r)
                for r in original["rows"]
            }
            assert set(originals) == set(predictions)
    out = OUT / "jobs" / entry["id"]
    out.mkdir(parents=True, exist_ok=False)
    (out / "rows").mkdir()
    tick = time.monotonic()
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=job["workers"]) as pool:
        futures = [
            pool.submit(
                one,
                row,
                str(reference_path.parent),
                entry,
                predictions.get(row["sample_id"]),
                originals.get(row["sample_id"]),
                str(out / "rows"),
            )
            for row in references
        ]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
            if len(results) % 60 == 0:
                save(
                    out / "progress.json",
                    {
                        "completed": len(results),
                        "total": len(references),
                        "seconds": time.monotonic() - tick,
                    },
                )
    results.sort(key=lambda r: r["sample_id"])
    assert len(results) == len(references)
    record = {
        "status": "all_fixed_job_observations_scored",
        "entry": entry,
        "scoring_job_sha256": digest(OUT / "job.json"),
        "prediction_manifest": ref(folder / "complete.json") if folder else None,
        "reference_manifest": ref(reference_path),
        "rows": results,
        "status_counts": dict(Counter(r["status"] for r in results)),
        "seconds": time.monotonic() - tick,
        "budgets": BUDGETS,
    }
    save(out / "complete.json", record)
    print("SCORE COMPLETE", entry["id"], len(results), round(record["seconds"], 1), flush=True)
    return ref(out / "complete.json")


def main():
    parser = argparse.ArgumentParser(description="Score completed local supplementary predictions.")
    parser.add_argument(
        "--job",
        type=Path,
        required=True,
        help="Bound scoring job with references, protocol and entries.",
    )
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    global_output = args.output.resolve()
    globals()["OUT"] = global_output
    job = read(args.job)
    bound(job["protocol"])
    global_output.mkdir(parents=True, exist_ok=False)
    save(global_output / "job.json", job)
    done = {}
    for entry in job["entries"]:
        folder = ROOT / entry["prediction_directory"] if entry.get("prediction_directory") else None
        verification = read(ROOT / entry["verification"]) if folder else None
        done[entry["id"]] = score_job(job, entry, folder, verification)
    save(
        global_output / "complete.json",
        {
            "status": "all20_supplement_jobs_scored_locally",
            "jobs": done,
            "scoring_job_sha256": digest(global_output / "job.json"),
            "model_observation_pairs": sum(
                e["observations"] for e in job["entries"] if e["method_id"] != "INPUT"
            ),
            "INPUT_observations": sum(
                e["observations"] for e in job["entries"] if e["method_id"] == "INPUT"
            ),
            "budgets": BUDGETS,
            "GT_uploaded": False,
        },
    )


if __name__ == "__main__":
    main()
