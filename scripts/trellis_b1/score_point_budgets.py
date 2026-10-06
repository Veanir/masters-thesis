"""CPU-only output-budget controls on hash-bound validation meshes."""

import argparse
import json
from pathlib import Path

import numpy as np

from masters_rgbd.b1.dev_cohort import load_dev_sample, resolve_dev_cohort
from masters_rgbd.b1.gso_zero_shot import _sha256
from masters_rgbd.b1.point_budget import budget512
from masters_rgbd.b1.projected_mesh import (
    _aggregate,
    _sample_mesh_surface,
    _surface_metrics,
    _visibility_recall_metrics,
)
from masters_rgbd.b1.training import _cache_path, _manifest_path
from masters_rgbd.contracts.manifests import SplitName
from scripts.common.paths import CODE_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True, help="label=surface folder")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = CODE_ROOT
    config_path = args.config
    config = json.loads(config_path.read_text())
    dataset, cache = Path(config["dataset_root"]), Path(config["cache_root"])
    sources = {}
    bindings = {}
    for item in args.source:
        label, folder = item.split("=", 1)
        assert label not in sources
        folder = Path(folder).resolve(strict=True)
        aggregate = json.loads((folder / "aggregate.json").read_text())["surface"]
        sources[label] = (folder, {r["asset_id"]: r for r in aggregate["per_object"]})
        bindings[label] = {
            "folder": str(folder),
            "aggregate_sha256": _sha256(folder / "aggregate.json"),
        }
    cohort = resolve_dev_cohort(
        dataset, Path(config["inventory"]), train_count=140, validation_count=25, test_count=23
    )
    selected = [s for s in cohort if s.asset.split == SplitName.VALIDATION]
    ids = {s.asset.asset_id for s in selected}
    assert len(ids) == 25 and all(set(rows) == ids for _, rows in sources.values())
    fixture = np.arange(1800, dtype=float).reshape(600, 3)
    assert budget512(fixture).shape == (512, 3)
    assert np.array_equal(budget512(fixture), budget512(fixture))
    args.output.mkdir(parents=True, exist_ok=False)
    binding = {
        "scope": "validation25 only",
        "sources": bindings,
        "sampling_seeds": [20260904, 20260905, 20260906],
        "script_sha256": _sha256(Path(__file__)),
        "config_sha256": _sha256(config_path),
        "protocol_sha256": _sha256(root / "configs/trellis_point_budgets.json"),
        "metric_source_sha256": _sha256(root / "src/masters_rgbd/b1/projected_mesh.py"),
    }
    (args.output / "binding.json").write_text(json.dumps(binding, indent=2))
    records = {}
    for index, spec in enumerate(selected):
        prepared = load_dev_sample(dataset, _manifest_path(dataset), spec, partial_point_count=512)
        obs = prepared.sample.observation
        cache_path = _cache_path(cache, spec)
        cache_hash = _sha256(cache_path)
        with np.load(cache_path, allow_pickle=False) as stored:
            for key, actual in [
                ("rgb", obs.rgb),
                ("depth_m", obs.depth_m),
                ("mask", obs.mask),
                ("partial_points_camera_m", obs.partial_points_camera_m),
            ]:
                assert stored[key].shape == actual.shape and np.allclose(
                    stored[key], actual, rtol=1e-6, atol=1e-7
                )
            intrinsics = stored["intrinsics"].copy()
        vertices = obs.camera_T_mesh.transform_points(prepared.sample.surface.vertices_mesh_local_m)
        truth = _sample_mesh_surface(
            vertices, prepared.sample.surface.faces, count=16384, seed=5000 + index
        )
        points = obs.partial_points_camera_m
        assert points.shape == (512, 3)
        predictions = {"input512": points}
        mesh_bindings = {}
        for axis, axis_name in enumerate("xyz"):
            mirrored = points.copy()
            mirrored[:, axis] = points[:, axis].min() + points[:, axis].max() - points[:, axis]
            predictions[f"reflect_{axis_name}_512"] = budget512(np.concatenate([points, mirrored]))
        for label, (folder, source_rows) in sources.items():
            historical = source_rows[spec.asset.asset_id]
            assert historical["cache_sha256"] == cache_hash
            path = folder / f"{spec.asset.asset_id}.npz"
            assert _sha256(path) == historical["mesh_sha256"]
            mesh_bindings[label] = historical["mesh_sha256"]
            with np.load(path, allow_pickle=False) as mesh:
                natural = mesh["surface_points_camera_m"].copy()
                predictions[f"{label}/natural"] = natural
                predictions[f"{label}/vertex512"] = budget512(natural)
                for seed in binding["sampling_seeds"]:
                    predictions[f"{label}/area512-{seed}"] = (
                        _sample_mesh_surface(
                            mesh["mesh_vertices_camera_m"], mesh["mesh_faces"], count=512, seed=seed
                        )
                        if len(mesh["mesh_faces"])
                        else np.empty((0, 3))
                    )
            assert (
                abs(
                    _surface_metrics(natural, truth)["0.005"]["fscore"]
                    - historical["surface"]["0.005"]["fscore"]
                )
                <= 1e-4
            )
        object_rows = {}
        for name, predicted in predictions.items():
            assert predicted.ndim == 2 and predicted.shape[1] == 3 and np.isfinite(predicted).all()
            row = {
                "asset_id": spec.asset.asset_id,
                "category": spec.asset.category_id,
                "point_count": len(predicted),
                "surface": _surface_metrics(predicted, truth),
                "visibility": _visibility_recall_metrics(
                    predicted, truth, depth_m=obs.depth_m, mask=obs.mask, intrinsics=intrinsics
                ),
            }
            records.setdefault(name, []).append(row)
            object_rows[name] = row
        (args.output / f"{spec.asset.asset_id}.json").write_text(
            json.dumps(
                {"cache_sha256": cache_hash, "mesh_sha256": mesh_bindings, "arms": object_rows},
                indent=2,
            )
        )
        print(f"{index + 1}/25 {spec.asset.asset_id}", flush=True)
    result = {
        "binding": binding,
        "status": "complete",
        "arms": {name: _aggregate(rows, expected_count=25) for name, rows in records.items()},
    }
    (args.output / "complete.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
