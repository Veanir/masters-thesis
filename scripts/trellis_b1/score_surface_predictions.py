"""Frozen validation25 surfaces for all nine balanced-diversity runs."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from masters_rgbd.b1.dense_control import CONFIG
from masters_rgbd.b1.dev_cohort import load_dev_sample
from masters_rgbd.b1.diversity_surface import score
from masters_rgbd.b1.model import B1FieldModel
from masters_rgbd.b1.projected_mesh import _aggregate, _sample_mesh_surface
from masters_rgbd.b1.training import _cache_path, _manifest_path, _sha256_file
from masters_rgbd.contracts.manifests import SplitName
from scripts.common.paths import CODE_ROOT
from scripts.trellis_b1.train_predict_comparison import (
    MANIFEST,
    _specs,
    inputs,
    verify_inputs,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--preflight-report", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    cli = parser.parse_args()
    args = inputs(cli)
    specs = _specs(args)
    verify_inputs(args, specs)
    plan = json.loads(MANIFEST.read_text())
    completed = []
    for run in plan["runs"]:
        name = f"{run['arm']}-s{run['seed']}"
        folder = cli.runs_root / name
        complete = json.loads((folder / "complete.json").read_text())
        resolved = json.loads((folder / "resolved-config.json").read_text())
        assert complete["updates"] == run["updates"] and complete["seed"] == run["seed"]
        assert complete["data_arm"] == run["arm"] and complete["test_opened"] is False
        assert resolved["training_asset_schedule"] == run["schedule"]
        assert resolved["campaign_sha256"] == _sha256_file(MANIFEST)
        assert _sha256_file(folder / "best-validation.pt") == complete["checkpoint_sha256"]
        completed.append((name, folder, complete))
    assert len(completed) == 9
    selected = [s for s in specs if s.asset.split == SplitName.VALIDATION]
    assert len(selected) == 25
    samples, identities = [], []
    for index, spec in enumerate(selected):
        prepared = load_dev_sample(
            args.dataset_root, _manifest_path(args.dataset_root), spec, partial_point_count=512
        )
        obs = prepared.sample.observation
        cache = _cache_path(args.cache_root, spec)
        assert not cache.name.startswith("test-")
        with np.load(cache, allow_pickle=False) as stored:
            data = {
                key: stored[key].copy()
                for key in ("rgb", "depth_m", "mask", "partial_points_camera_m", "intrinsics")
            }
        for key, actual in (
            ("rgb", obs.rgb),
            ("depth_m", obs.depth_m),
            ("mask", obs.mask),
            ("partial_points_camera_m", obs.partial_points_camera_m),
        ):
            assert data[key].shape == actual.shape and np.allclose(
                data[key], actual, rtol=1e-6, atol=1e-7
            )
        intrinsics = obs.intrinsics
        assert np.allclose(
            data["intrinsics"], (intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy)
        )
        vertices = obs.camera_T_mesh.transform_points(prepared.sample.surface.vertices_mesh_local_m)
        truth = _sample_mesh_surface(
            vertices, prepared.sample.surface.faces, count=16384, seed=5000 + index
        )
        identity = dict(
            asset_id=spec.asset.asset_id,
            cache_sha256=_sha256_file(cache),
            gt_points_sha256=hashlib.sha256(truth.tobytes()).hexdigest(),
            gt_seed=5000 + index,
        )
        identities.append(identity)
        samples.append((spec, data, obs.model_roi, truth, identity))
    output = cli.output
    output.mkdir(exist_ok=False)
    binding = dict(
        scope="validation25 only; all nine arms",
        extractor=CONFIG,
        campaign_sha256=_sha256_file(MANIFEST),
        script_sha256=_sha256_file(Path(__file__)),
        protocol_sha256=_sha256_file(CODE_ROOT / "configs/trellis_surface.json"),
        model_sha256=_sha256_file(CODE_ROOT / "src/masters_rgbd/b1/model.py"),
        score_source_sha256=_sha256_file(CODE_ROOT / "src/masters_rgbd/b1/diversity_surface.py"),
        inputs=identities,
    )
    (output / "binding.json").write_text(json.dumps(binding, indent=2))
    torch.set_num_threads(2)
    device = torch.device("cuda")
    aggregates = {}
    for name, training_folder, complete in completed:
        model = B1FieldModel(
            conditioning="pixel_knn",
            decoder_channels=128,
            knn_neighbors=8,
            minimum_roi_half_extent_m=0.001,
            roi_padding_fraction=0.25,
        ).to(device)
        state = torch.load(
            training_folder / "best-validation.pt", map_location=device, weights_only=True
        )
        assert all(torch.isfinite(tensor).all() for tensor in state.values())
        model.load_state_dict(state, strict=True)
        model.eval()
        folder = output / name
        folder.mkdir()
        records = []
        for index, (spec, data, roi, truth, identity) in enumerate(samples):
            record = score(
                model, data, roi, truth, folder, spec.asset.asset_id, spec.asset.category_id, device
            )
            for metrics in record["surface"].values():
                for key, value in metrics.items():
                    if value is None:
                        assert record["point_count"] == 0 and key.startswith("chamfer")
                    else:
                        assert np.isfinite(value), "nonfinite surface metric"
            record.update(identity)
            records.append(record)
            print(f"{name} {index + 1}/25", flush=True)
        aggregates[name] = dict(training=complete, surface=_aggregate(records, expected_count=25))
        (folder / "aggregate.json").write_text(json.dumps(aggregates[name], indent=2))
        del model, state
        torch.cuda.empty_cache()
    (output / "complete.json").write_text(
        json.dumps(dict(binding=binding, arms=aggregates), indent=2)
    )


if __name__ == "__main__":
    main()
