"""Freeze observation-only supplementary inputs and bound final checkpoints.

No model execution. Synthetic meshes stay in the original local observations;
their references are recorded separately and never enter inference packages.
"""

from __future__ import annotations

import json
import shutil
import zipfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.common.paths import ROOT, digest, read, save, script_help
from scripts.predict.prepare_prediction_inputs import completed_matrix

script_help(__doc__, __name__)


OUT = ROOT / "runs/thesis-supplement-20260910/inference-preparation-v1"
MAIN = ROOT / "runs/research-evolution-campaign-20260907/MAIN-independent-owner-v1"
CAMPAIGN = ROOT / "runs/research-evolution-campaign-20260907/frozen-campaign-v3/campaign.json"
EVAL = ROOT / "runs/research-evolution-evaluation-20260907"


def ref(path):
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": digest(path)}


def bound(reference):
    path = ROOT / reference["path"]
    assert digest(path) == reference["sha256"], path
    return path


def copy(source, destination, expected):
    assert digest(source) == expected, source
    destination.parent.mkdir(parents=True, exist_ok=True)
    assert not destination.exists(), destination
    # Independent copies avoid changing historical files through future writes.
    shutil.copyfile(source, destination)
    assert digest(destination) == expected


def make_manifest(folder, rows, scope, sources):
    manifest = {
        "status": "poststudy_observation_only_inputs_frozen",
        "rows": rows,
        "gt_uploaded": False,
        "scope": scope,
        "source_refs": sources,
        "schema": "ray-uint16-depth",
        "script_sha256": digest(Path(__file__)),
    }
    save(folder / "manifest.json", manifest)
    assert all(set(r["files"]) == {"rgb.png", "depth.png", "mask.png", "camera.json"} for r in rows)
    archive = folder.with_suffix(".zip")
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_STORED) as z:
        z.write(folder / "manifest.json", "manifest.json")
        for row in rows:
            for name, expected in row["files"].items():
                path = folder / row["sample_id"] / name
                assert digest(path) == expected
                z.write(path, row["sample_id"] + "/" + name)
    return {
        "manifest": ref(folder / "manifest.json"),
        "archive": ref(archive),
        "archive_bytes": archive.stat().st_size,
        "count": len(rows),
        "eligible": sum(r["input_eligible"] for r in rows),
    }


def main():
    assert not OUT.exists()
    campaign = read(CAMPAIGN)
    assert digest(CAMPAIGN) == "12d972125cc9674f33af613d32dbf9d80197a858227aa62900fb3fb27544e5bf"
    checkpoints = completed_matrix(MAIN / "complete.json", CAMPAIGN)
    source_complete = bound(campaign["references"]["source_complete"])
    descriptive = bound(campaign["references"]["source_descriptive_report"])
    descriptors = {r["sample_id"]: r for r in read(descriptive)["rows"]}
    roles = {
        role: [r for r in descriptors.values() if r["role"] == role]
        for role in ("TRAIN", "VAL_development", "VAL_holdout")
    }
    assert [len(roles[k]) for k in roles] == [6400, 80, 720]
    role_keys = {role: {r["key"] for r in rows} for role, rows in roles.items()}
    role_families = {role: {r["family_id"] for r in rows} for role, rows in roles.items()}
    assert [len(role_keys[k]) for k in roles] == [320, 4, 36]
    for left, right in (
        ("TRAIN", "VAL_development"),
        ("TRAIN", "VAL_holdout"),
        ("VAL_development", "VAL_holdout"),
    ):
        assert not role_keys[left] & role_keys[right]
        assert not role_families[left] & role_families[right]
    index = {}
    source_refs = []
    for entry in read(source_complete)["sources"]:
        folder = ROOT / entry["source"]
        path = folder / "complete.json"
        assert digest(path) == entry["manifest_sha256"]
        source_refs.append(ref(path))
        for row in read(path)["rows"]:
            assert row["sample_id"] not in index
            index[row["sample_id"]] = (folder, row)
    assert set(index) == set(descriptors)
    OUT.mkdir(parents=True)
    inputs = OUT / "inputs"
    inputs.mkdir()
    holdout = inputs / "synthetic-holdout720"
    neutral = inputs / "HB198-neutral128"
    parity = inputs / "HB-original-parity3"
    for path in (holdout, neutral, parity):
        path.mkdir()

    hb_path = EVAL / "HB-inference-inputs-v1/ray/manifest.json"
    hb = read(hb_path)
    old_inputs = read(EVAL / "HB-inference-inputs-v1/complete.json")
    assert digest(hb_path) == old_inputs["bindings"]["ray"]["manifest_sha256"]
    assert len(hb["rows"]) == 198 and not hb["gt_uploaded"]
    neutral_rows, parity_rows, rgb_checks = [], [], []
    seen_objects, selected_parity = set(), []
    for row in sorted(hb["rows"], key=lambda r: r["sample_id"]):
        obj = row["sample_id"].split("-")[1]
        if row["input_eligible"] and obj not in seen_objects and len(selected_parity) < 3:
            selected_parity.append(row["sample_id"])
            seen_objects.add(obj)
    assert len(selected_parity) == 3
    for row in hb["rows"]:
        sid = row["sample_id"]
        src = hb_path.parent / sid
        dst = neutral / sid
        dst.mkdir()
        for name, expected in row["files"].items():
            assert digest(src / name) == expected
            if name != "rgb.png":
                copy(src / name, dst / name, expected)
        mask = np.asarray(Image.open(src / "mask.png")) > 0
        original = np.asarray(Image.open(src / "rgb.png").convert("RGB"))
        gray = np.zeros_like(original)
        gray[mask] = 128
        Image.fromarray(gray).save(dst / "rgb.png")
        check = np.asarray(Image.open(dst / "rgb.png"))
        assert np.all(check[mask] == 128) and np.all(check[~mask] == 0)
        assert int(mask.sum()) == row["input_pixels"]
        files = {name: digest(dst / name) for name in row["files"]}
        assert all(
            files[name] == row["files"][name] for name in ("depth.png", "mask.png", "camera.json")
        )
        neutral_rows.append({**row, "cohort": "HB198-neutral128", "files": files})
        rgb_checks.append(
            {
                "sample_id": sid,
                "unchanged_depth_mask_camera": True,
                "neutral_rgb_verified": True,
                "input_pixels": int(mask.sum()),
                "original_rgb_sha256": row["files"]["rgb.png"],
                "neutral_rgb_sha256": files["rgb.png"],
            }
        )
        if sid in selected_parity:
            for name, expected in row["files"].items():
                copy(src / name, parity / sid / name, expected)
            parity_rows.append({**row, "cohort": "HB-original-parity3"})
    hb_ref = ref(hb_path)
    input_records = {
        "HB198-neutral128": make_manifest(
            neutral,
            neutral_rows,
            (
                "All 198 HB identities and original eligibility. Only RGB changed "
                "to uint8 128 inside historical mask, 0 outside. No new training."
            ),
            [hb_ref],
        ),
        "HB-original-parity3": make_manifest(
            parity,
            parity_rows,
            (
                "Execution parity only. First eligible ID for each of first three "
                "object IDs in sorted frozen HB list; chosen without scores. "
                "Original files byte-identical."
            ),
            [hb_ref],
        ),
    }
    holdout_rows, local_references = [], []
    for desc in sorted(roles["VAL_holdout"], key=lambda r: r["sample_id"]):
        sid = desc["sample_id"]
        folder, row = index[sid]
        src = folder / sid
        observation = src / "observation.npz"
        assert digest(observation) == row["files"]["observation.npz"]
        with np.load(observation, allow_pickle=False) as z:
            rgb, codes, mask, k = z["rgb"], z["input_depth_uint16"], z["input_mask"], z["K"]
            vertices, faces = z["mesh_vertices_camera_cv_m"], z["mesh_faces"]
            assert vertices.ndim == 2 and vertices.shape[1] == 3 and np.isfinite(vertices).all()
            assert (
                faces.ndim == 2
                and faces.shape[1] == 3
                and faces.min() >= 0
                and faces.max() < len(vertices)
            )
            assert rgb.dtype == np.uint8 and codes.dtype == np.uint16 and mask.dtype == bool
            assert rgb.shape == (480, 640, 3) and codes.shape == mask.shape == (480, 640)
            assert k.shape == (3, 3) and k.dtype == np.float32 and np.isfinite(k).all()
            assert np.all(codes[mask] > 0)
            diameter = float(np.linalg.norm(np.ptp(vertices, axis=0)))
        dst = holdout / sid
        for name in ("rgb.png", "depth.png", "mask.png"):
            copy(src / name, dst / name, row["files"][name])
        assert np.array_equal(np.asarray(Image.open(dst / "rgb.png").convert("RGB")), rgb)
        assert np.array_equal(np.asarray(Image.open(dst / "depth.png"), dtype=np.uint16), codes)
        assert np.array_equal(np.asarray(Image.open(dst / "mask.png")) > 0, mask)
        save(
            dst / "camera.json",
            {"K": k.tolist(), "cam2world": np.eye(4, dtype=np.float32).tolist()},
        )
        pixels = int(mask.sum())
        assert pixels == row["input_pixels"] == desc["input_pixels"]
        meta = {
            "sample_id": sid,
            "cohort": "synthetic-holdout720",
            "input_pixels": pixels,
            "input_eligible": pixels >= 512,
            "files": {
                name: digest(dst / name)
                for name in ("rgb.png", "depth.png", "mask.png", "camera.json")
            },
        }
        holdout_rows.append(meta)
        local_references.append(
            {
                **desc,
                "observation": ref(observation),
                "input_eligible": pixels >= 512,
                "source_mesh_sha256": row["source_mesh_sha256"],
                "mesh_bbox_diagonal_m": diameter,
                "camera_axes": (
                    "source: x right, y down, z forward; scorer: x right, y up, z backward"
                ),
                "conversion_to_scorer": [1, -1, -1],
            }
        )
        if len(holdout_rows) % 120 == 0:
            print("HOLDOUT INPUT", len(holdout_rows), flush=True)
    assert len(holdout_rows) == 720
    assert set(Counter(r["key"] for r in local_references).values()) == {20}
    input_records["synthetic-holdout720"] = make_manifest(
        holdout,
        holdout_rows,
        (
            "All 720 observations from 36 pre-existing reserved geometries. "
            "No PHOTO. Original RGB/depth/mask and K. No meshes, pose labels "
            "or scores included."
        ),
        [ref(source_complete), ref(descriptive), *source_refs],
    )
    save(
        OUT / "holdout-local-reference-index.json",
        {
            "GT_must_not_be_uploaded_for_inference": True,
            "role_counts": {k: len(v) for k, v in roles.items()},
            "disjoint_geometry_and_family_ids": True,
            "rows": local_references,
            "source_refs": [ref(source_complete), ref(descriptive), *source_refs],
        },
    )
    save(
        OUT / "RGB-input-pair-checks.json",
        {"count": len(rgb_checks), "all_passed": True, "rows": rgb_checks},
    )

    scoring_job_path = EVAL / "final-scoring-job-20260909-v6/job.json"
    old_scoring = read(scoring_job_path)
    methods = [
        m
        for m in campaign["evaluation_methods"]
        if m["role"] == "adapted" or m["id"] == "ray-pretrained"
    ]
    assert len(methods) == 10
    parity_references = {}
    for method in methods:
        mid = method["id"]
        manifest_path = bound(old_scoring["prediction_manifests"][mid])
        manifest = read(manifest_path)
        assert manifest["job"]["method_id"] == mid and len(manifest["rows"]) == 198
        refs = []
        for sid in selected_parity:
            row = next(r for r in manifest["rows"] if r["sample_id"] == sid)
            path = manifest_path.parent / (sid + ".npz")
            assert digest(path) == row["sha256"]
            target = OUT / "parity-references" / mid / path.name
            copy(path, target, row["sha256"])
            refs.append(
                {
                    "sample_id": sid,
                    "prediction": ref(target),
                    "original_prediction": ref(path),
                    "status": row["status"],
                    "count": row["count"],
                }
            )
        parity_references[mid] = {"original_manifest": ref(manifest_path), "rows": refs}
    save(
        OUT / "parity-references.json",
        {
            "purpose": "Numerical execution comparison; predictions are not GT",
            "methods": parity_references,
        },
    )
    protocol = {
        "status": "frozen_poststudy_inference_protocol_before_new_predictions",
        "created_utc": datetime.now(UTC).isoformat(),
        "reproduction_protocol": ref(ROOT / "configs/supplement.json"),
        "historical_campaign": ref(CAMPAIGN),
        "historical_matrix": ref(MAIN / "complete.json"),
        "historical_evaluation_protocol": campaign["references"]["evaluation_protocol"],
        "methods": methods,
        "checkpoints": checkpoints,
        "inputs": input_records,
        "conditions": {
            "HB198-neutral128": [m["id"] for m in methods if m["role"] == "adapted"],
            "synthetic-holdout720": [m["id"] for m in methods],
            "HB-original-parity3": [m["id"] for m in methods],
        },
        "scope": (
            "Post-study diagnostics only. Main HB result remains unchanged. "
            "No training, checkpoint selection, new method selection or "
            "exclusion based on outcomes."
        ),
        "parity_limits": read(EVAL / "adapter-canary-v1/bundle/plan.json")["limits"],
        "parity_rule": (
            "Every method on its execution host must pass original-HB "
            "reproduction before new inference. Same previously accepted "
            "cloud-distance/count limits; report exact equality separately."
        ),
        "parity_reference_manifest": ref(OUT / "parity-references.json"),
        "holdout_reference_index": ref(OUT / "holdout-local-reference-index.json"),
        "GT_inference_separation": True,
        "metric_rules": {
            "core": ref(ROOT / "scripts/eval/surface_metrics.py"),
            "reference_points": 262144,
            "holdout_truth_seed_purpose": "thesis-supplement-holdout-reference-v1",
            "coordinate_system": "metric project camera [x right, y up, z backward]",
            "thresholds_m": [0.002, 0.005, 0.010],
            "primary_threshold_m": 0.005,
            "point_budgets": {"primary": 16384, "sensitivity": 512},
            "point_selection": "Unchanged ID-based selection in evolution_eval_core.select_points",
            "neutral_comparison": (
                "Paired neutral minus original within each of all 9 final models; "
                "object-level means and all per-seed effects. Supplementary "
                "PHOTO-BASE contrast under neutral RGB."
            ),
            "holdout_comparison": (
                "Object-level means over 36 geometries, all 3 training seeds; "
                "BASE/CLASSIC/PHOTO plus public Ray and INPUT. No pooling with HB."
            ),
            "intervals": (
                "Object bootstrap and object-plus-training-seed bootstrap using "
                "existing core; 10000 replicates, seed 2026091002. Conditional "
                "interpretations; three seeds remain limited."
            ),
            "source_quality_audit": (
                "Before predictions, inspect mesh scale and clean visible "
                "target-to-mesh alignment for all720. Preserve every reserved "
                "observation and report failures; no outcome-based filtering."
            ),
            "undefined_metrics": (
                "Keep null with explicit denominator; empty predictions remain in "
                "F-score population."
            ),
        },
        "script_sha256": digest(Path(__file__)),
    }
    save(OUT / "protocol.json", protocol)
    complete = {
        "status": "supplementary_inputs_and_final_checkpoints_bound_no_GPU_execution",
        "protocol": ref(OUT / "protocol.json"),
        "inputs": input_records,
        "real_HB_observations": 198,
        "synthetic_geometries": 36,
        "synthetic_observations": 720,
        "neutral_jobs": 9,
        "holdout_jobs": 10,
        "parity_jobs": 10,
        "new_observation_method_pairs": 1782 + 7200 + 30,
        "input_pair_checks": ref(OUT / "RGB-input-pair-checks.json"),
        "GT_uploaded": False,
        "source_pretraining_independence_claim": False,
    }
    save(OUT / "complete.json", complete)
    print(json.dumps(complete))


if __name__ == "__main__":
    main()
