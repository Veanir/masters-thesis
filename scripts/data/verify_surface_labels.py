"""Verify every source-bound label and independent foreground/background rays."""

import argparse
import ast
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import trimesh

from scripts.common.paths import ROOT as WORKSPACE_ROOT
from scripts.common.paths import script_help
from scripts.data.check_label_first_hits import verify_view

script_help(__doc__, __name__)


ROOT = WORKSPACE_ROOT


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--canary-count", type=int)
    a = p.parse_args()
    assert not a.output.exists()
    manifest = json.loads((a.labels / "complete.json").read_text())
    source = json.loads((a.source / "complete.json").read_text())
    certified = json.loads((a.source / "verification-v1.json").read_text())
    assert (
        manifest["status"] == "certified_synthetic_ray_labels_complete" and certified["all_passed"]
    )
    assert (
        manifest["source_manifest_sha256"]
        == certified["source_manifest_sha256"]
        == sha(a.source / "complete.json")
    )
    assert manifest["script_sha256"] == sha(ROOT / "scripts/data/generate_surface_labels.py")
    references = {r["sample_id"]: r for r in source["rows"]}
    assert len(references) == len(source["rows"])
    assert len(manifest["rows"]) == len(references) and {
        r["sample_id"] for r in manifest["rows"]
    } == set(references)
    with zipfile.ZipFile(
        ROOT / "runs/research-evolution-ray-context-20260907/ray-code-v1.zip"
    ) as z:
        sampler = z.read(
            next(n for n in z.namelist() if n.endswith("eval_wrapper/sample_poses.py"))
        )
    assert hashlib.sha256(sampler).hexdigest() == manifest["pose_sampler_sha256"]
    namespace = {"np": np}
    for node in ast.parse(sampler.decode()).body:
        if isinstance(node, ast.FunctionDef) and node.name in ["look_at", "sample_camera_poses"]:
            exec(
                compile(ast.Module(body=[node], type_ignores=[]), "pinned_sampler", "exec"),
                namespace,
            )
    selection = list(enumerate(manifest["rows"]))
    if a.canary_count is not None:
        assert 0 < a.canary_count < len(selection)
        by_key = {}
        for ordinal, row in selection:
            by_key.setdefault(row["key"], []).append((ordinal, row))
        keys = sorted(
            by_key,
            key=lambda key: hashlib.sha256(("label-checker-canary-v1/" + key).encode()).hexdigest(),
        )[: a.canary_count]
        selection = [
            min(
                by_key[key],
                key=lambda item: hashlib.sha256(item[1]["sample_id"].encode()).hexdigest(),
            )
            for key in keys
        ]
        special = next(
            (
                item
                for item in enumerate(manifest["rows"])
                if item[1]["sample_id"] == "Perricone_MD_Face_Finishing_Moisturizer_4_oz-a00-v0"
            ),
            None,
        )
        if special and special not in selection:
            same = next(
                (i for i, item in enumerate(selection) if item[1]["key"] == special[1]["key"]),
                len(selection) - 1,
            )
            selection[same] = special
        assert len(selection) == a.canary_count
    records = []
    total_rays = 0
    background_rays = 0
    maximum = 0.0
    for ordinal, row in selection:
        sid = row["sample_id"]
        ref = references[sid]
        path = a.labels / (sid + ".npz")
        original = a.source / sid / "observation.npz"
        assert all(row[n] == ref[n] for n in ["key", "family_id", "split"])
        assert (
            sha(path) == row["output_sha256"]
            and sha(original) == row["source_observation_sha256"] == ref["files"]["observation.npz"]
        )
        with np.load(path) as z, np.load(original) as x:
            for key in ["rgb", "input_depth_uint16", "input_mask", "K"]:
                assert np.array_equal(z[key], x[key])
            encoded = z["novel_depth_uint16"]
            masks = z["novel_masks"]
            cams = z["novel_c2ws"]
            k = z["K"]
            assert (
                encoded.dtype == np.uint16
                and masks.dtype == bool
                and encoded.shape == masks.shape == (21, 480, 640)
            )
            assert (
                cams.shape == (21, 4, 4)
                and np.isfinite(cams).all()
                and np.array_equal(masks, encoded > 0)
            )
            counts = masks.sum((1, 2))
            assert counts.min() > 100 and np.array_equal(counts, row["novel_valid_pixel_counts"])
            assert np.allclose(
                cams[:, :3, :3].transpose(0, 2, 1) @ cams[:, :3, :3], np.eye(3), atol=2e-6
            )
            v, u = np.mgrid[:480, :640]
            dirs = np.stack(
                ((u - k[0, 2]) / k[0, 0], (v - k[1, 2]) / k[1, 1], np.ones_like(u)), -1
            ).astype(np.float32)
            observed = (
                dirs
                * (z["input_depth_uint16"].astype(np.float32) * np.float32(10 / 65535))[..., None]
            )[z["input_mask"]]
            lo, hi = observed.min(0), observed.max(0)
            center = (lo + hi) / 2
            radius = max(0.7 * np.linalg.norm(center), 1.3 * np.linalg.norm(hi - lo) / 2)
            expected = namespace["sample_camera_poses"](center, radius, radius, 5).astype(
                np.float32
            )
            distances = np.max(np.abs(cams[:, None] - expected[None, :]), axis=(2, 3))
            mapping = distances.argmin(1)
            assert len(set(mapping.tolist())) == 21 and distances.min(1).max() < 1e-6
            minimum_codes = np.where(masks, encoded, np.iinfo(np.uint16).max).min((1, 2))
            views = [ordinal % 21]
            near_view = int(minimum_codes.argmin())
            if float(minimum_codes[near_view]) * 10 / 65535 < 0.1 and near_view not in views:
                views.append(near_view)
            mesh = trimesh.Trimesh(x["mesh_vertices_camera_cv_m"], x["mesh_faces"], process=False)
            checks = []
            for view in views:
                checked = verify_view(
                    mesh,
                    cams[view],
                    k,
                    encoded[view],
                    "evolution-label-independent-v1/" + sid + "/" + str(view),
                )
                checks.append({"view": view, **checked})
                total_rays += checked["rays"]
                background_rays += checked["background_rays"]
                maximum = max(maximum, checked["max_axial_error_m"])
            records.append(
                {
                    "sample_id": sid,
                    "minimum_novel_pixels": int(counts.min()),
                    "pose_set_max_error": float(distances.min(1).max()),
                    "independent_views": checks,
                }
            )
        if (ordinal + 1) % 80 == 0:
            print("LABEL BATCH VERIFIED", ordinal + 1, flush=True)
    result = {
        "status": "all_batch_labels_verified_with_independent_positive_and_background_rays"
        if a.canary_count is None
        else "partial_canary_labels_verified_with_independent_positive_and_background_rays",
        "rows": records,
        "count": len(records),
        "source_count": len(manifest["rows"]),
        "full_batch_verified": a.canary_count is None,
        "independent_rays": total_rays,
        "background_rays": background_rays,
        "max_axial_error_m": maximum,
        "source_manifest_sha256": sha(a.source / "complete.json"),
        "labels_manifest_sha256": sha(a.labels / "complete.json"),
        "script_sha256": sha(Path(__file__)),
        "ray_checker_sha256": sha(
            Path(__file__).parents[2] / "scripts/data/check_label_first_hits.py"
        ),
        "boundary_footprint_checks": [
            {"sample_id": r["sample_id"], "view": view["view"], **check}
            for r in records
            for view in r["independent_views"]
            for check in view["boundary_footprint_checks"]
        ],
        "scope": (
            "Every artifact, source array,21camera sets,masks and counts. "
            "All128 deterministic rays retained as in v1 except uniform "
            "one-pixel mask boundary exclusion. Strict center first-hit250um "
            "on interior and background. Positive silhouette center "
            "disagreements require BOTH independent mesh surface "
            "distance<250um AND first-hit interval of9rays in the same "
            "half-pixel footprint. All such cases and original axial errors "
            "are recorded. No labels changed. A canary is explicitly partial; "
            "full admission requires all rows."
        ),
    }
    a.output.write_text(json.dumps(result, indent=2))
    print({k: v for k, v in result.items() if k not in ["rows", "scope"]}, flush=True)


if __name__ == "__main__":
    main()
