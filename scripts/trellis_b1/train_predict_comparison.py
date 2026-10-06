"""Prepare or explicitly run the frozen BASE57/MIX114 campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from masters_rgbd.b1.diversity_data import _specs
from masters_rgbd.b1.model import B1FieldModel
from masters_rgbd.b1.training import (
    _cache_path,
    _git_sha,
    _manifest_path,
    _observation,
    _predict_udf,
    _sha256_file,
    _training_query_indices,
)
from masters_rgbd.contracts.manifests import SplitName
from scripts.common.paths import CODE_ROOT, DATASET_ROOT
from scripts.common.paths import ROOT as WORKSPACE_ROOT

REPO = WORKSPACE_ROOT
MANIFEST = REPO / "runs/trellis-b1/plan.json"
U = 6688
INTERVAL = 176


def inputs(cli):
    return SimpleNamespace(
        dataset_root=cli.dataset_root,
        inventory=cli.inventory,
        cache_root=cli.cache_root,
        preflight_report=cli.preflight_report,
    )


def verify_inputs(args, specs):
    fingerprints = json.loads((CODE_ROOT / "configs/code_fingerprints.json").read_text())
    for relative, expected in fingerprints.items():
        if _sha256_file(CODE_ROOT / relative) != expected:
            raise ValueError(f"frozen training source changed: {relative}")
    report = json.loads(args.preflight_report.read_text(encoding="utf-8"))
    if report["status"] != "passed" or len(report["cache"]) != 188:
        raise ValueError("incomplete original cache preflight")
    for path, expected in (
        (args.inventory, report["inventory_sha256"]),
        (_manifest_path(args.dataset_root), report["manifest_sha256"]),
        (args.cache_root / "cache-spec.json", report["cache_spec_sha256"]),
    ):
        if _sha256_file(path) != expected:
            raise ValueError(f"preflight binding changed: {path}")
    expected = {str(_cache_path(args.cache_root, s).resolve()) for s in specs}
    if {r["path"] for r in report["cache"]} != expected:
        raise ValueError("cache membership mismatch")
    for row in report["cache"]:
        if _sha256_file(Path(row["path"])) != row["sha256"]:
            raise ValueError(f"cache bytes changed: {row['path']}")


def build(specs):
    groups = defaultdict(list)
    for spec in specs:
        if spec.asset.split == SplitName.TRAIN:
            groups[spec.asset.category_id].append(spec.asset.asset_id)
    pools = {}
    excluded = []
    for cat, ids in sorted(groups.items()):
        ranked = sorted(
            ids,
            key=lambda x: (hashlib.sha256(f"balanced-diversity-v1/{x}".encode()).hexdigest(), x),
        )
        half = len(ranked) // 2
        if half:
            pools[cat] = {"base": ranked[:half], "mix": ranked[: 2 * half]}
        excluded.extend(ranked[2 * half :])
    assert len(pools) == 44
    assert sum(len(p["base"]) for p in pools.values()) == 57
    assert sum(len(p["mix"]) for p in pools.values()) == 114
    assert len(excluded) == 26
    by_id = {s.asset.asset_id: s.asset.category_id for s in specs}
    ordinals = {s.asset.asset_id: i for i, s in enumerate(specs)}
    train_hashes = {s.asset.geometry_sha256 for s in specs if s.asset.split == SplitName.TRAIN}
    val_hashes = {s.asset.geometry_sha256 for s in specs if s.asset.split == SplitName.VALIDATION}
    assert not train_hashes & val_hashes
    runs = []
    for seed in (0, 1, 2):
        rng = random.Random(seed)
        categories = []
        for _ in range(304):
            cycle = sorted(pools)
            rng.shuffle(cycle)
            categories.extend(cycle)
        schedules = {}
        for pool in ("base", "mix"):
            rngs = {
                cat: random.Random(
                    int(
                        hashlib.sha256(
                            f"balanced-diversity-v1/{seed}/{pool}/{cat}".encode()
                        ).hexdigest(),
                        16,
                    )
                )
                for cat in pools
            }
            queues = {cat: [] for cat in pools}
            schedule = []
            for cat in categories:
                if not queues[cat]:
                    queues[cat] = list(pools[cat][pool])
                    rngs[cat].shuffle(queues[cat])
                schedule.append(queues[cat].pop())
            schedules[pool] = schedule
        for arm, pool, updates in (
            ("base-u", "base", U),
            ("mix-u", "mix", U),
            ("mix-2u", "mix", 2 * U),
        ):
            schedule = schedules[pool][:updates]
            count = Counter(schedule)
            assert len(schedule) == updates
            assert set(count) == {x for p in pools.values() for x in p[pool]}
            for p in pools.values():
                exposure = updates // 44 // len(p[pool])
                assert all(count[x] == exposure for x in p[pool])
            for step in range(INTERVAL, updates + 1, INTERVAL):
                category_count = Counter(by_id[x] for x in schedule[:step])
                assert set(category_count.values()) == {step // 44}
            runs.append(
                dict(
                    arm=arm,
                    pool=pool,
                    seed=seed,
                    updates=updates,
                    validation_count=updates // INTERVAL,
                    schedule=schedule,
                    asset_exposures=dict(sorted(count.items())),
                )
            )
        base, mix, double = runs[-3:]
        assert mix["schedule"] == double["schedule"][:U]
        assert [by_id[x] for x in base["schedule"]] == [by_id[x] for x in mix["schedule"]]
        assert all(
            base["asset_exposures"][x] == double["asset_exposures"][x]
            for x in base["asset_exposures"]
        )
        # Same per-object RNG stream despite different global update positions.
        traces = []
        for run in (base, double):
            seen = Counter()
            trace = defaultdict(list)
            for asset in run["schedule"]:
                if asset in base["asset_exposures"]:
                    trace[asset].append((seed, seen[asset], ordinals[asset]))
                seen[asset] += 1
            traces.append(dict(trace))
        assert traces[0] == traces[1]
    assert sum(r["updates"] for r in runs) == 80256
    return dict(
        schema_version="balanced-generated-diversity-v1",
        pools=pools,
        excluded_training_ids=sorted(excluded),
        seeds=[0, 1, 2],
        U=U,
        validation_interval_updates=INTERVAL,
        runs=runs,
        training=dict(
            conditioning="pixel_knn",
            learning_rate=0.0015,
            train_query_pool=512,
            queries_per_update=256,
            decoder_channels=128,
            partial_points=512,
            checkpoint_selection="minimum validation25 object-mean UDF MAE; strict improvement",
        ),
        query_seed_rule=(
            "SeedSequence(model_seed, zero_based_per_object_exposure, original_cohort_ordinal)"
        ),
        exact_geometry_train_validation_overlap_count=0,
        validation_ids=sorted(
            s.asset.asset_id for s in specs if s.asset.split == SplitName.VALIDATION
        ),
        historical_test_ids=sorted(
            s.asset.asset_id for s in specs if s.asset.split == SplitName.TEST
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--arm", choices=("base-u", "mix-u", "mix-2u"))
    parser.add_argument("--seed", type=int, choices=(0, 1, 2))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--preflight-report", type=Path, required=True)
    cli = parser.parse_args()
    args = inputs(cli)
    specs = _specs(args)
    verify_inputs(args, specs)
    plan = build(specs)
    archived = json.loads((CODE_ROOT / "manifests/trellis_b1.json").read_text())
    archived_runs = archived.pop("run_files")
    archived["runs"] = []
    for ref in archived_runs:
        path = CODE_ROOT / "manifests" / ref["path"]
        assert _sha256_file(path) == ref["sha256"]
        archived["runs"].append(json.loads(path.read_text()))
    assert all(archived[key] == value for key, value in plan.items()), (
        "Historical B1 schedule changed"
    )
    if cli.prepare:
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        plan["preflight_sha256"] = _sha256_file(args.preflight_report)
        plan["inventory_sha256"] = _sha256_file(args.inventory)
        with MANIFEST.open("x", encoding="utf-8") as f:
            json.dump(plan, f, indent=2)
        strata = json.loads((CODE_ROOT / "manifests/trellis_b1_strata.json").read_text())
        strata["campaign_sha256"] = _sha256_file(MANIFEST)
        (MANIFEST.parent / "strata.json").write_text(json.dumps(strata, indent=2))
        print(
            json.dumps(
                dict(
                    status="prepared; no training started",
                    runs=9,
                    updates=80256,
                    query_draws=80256 * 256,
                    manifest_sha256=_sha256_file(MANIFEST),
                )
            )
        )
        return
    if cli.arm is None or cli.seed is None or cli.output_root is None:
        parser.error("explicit training requires --arm, --seed and --output-root")
    frozen = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if any(frozen[k] != v for k, v in plan.items()):
        raise ValueError("current schedule differs from frozen campaign")
    if frozen["preflight_sha256"] != _sha256_file(args.preflight_report):
        raise ValueError("preflight identity changed")
    row = next(r for r in plan["runs"] if r["arm"] == cli.arm and r["seed"] == cli.seed)
    sources = {
        name: _sha256_file(Path(module.__file__))
        for name, module in tuple(sys.modules.items())
        if name.startswith("masters_rgbd") and getattr(module, "__file__", None)
    }
    resolved = dict(
        schema_version="rapid-dev-b0-b1-v1",
        study="balanced-generated-diversity-v1",
        arm="pixel_knn",
        conditioning="pixel_knn",
        data_arm=cli.arm,
        source_commit=_git_sha(REPO),
        loaded_source_sha256=sources,
        runner_sha256=_sha256_file(Path(__file__)),
        campaign_sha256=_sha256_file(MANIFEST),
        seed=cli.seed,
        data_seed=0,
        epochs=1,
        updates=row["updates"],
        train_count=140,
        validation_count=25,
        test_count=23,
        train_queries=512,
        evaluation_queries=1024,
        queries_per_step=256,
        partial_points=512,
        decoder_channels=128,
        certain_free_queries_per_step=0,
        active_train_asset_ids=sorted(row["asset_exposures"]),
        dataset_root=str(args.dataset_root.resolve()),
        inventory=str(args.inventory.resolve()),
        cache_root=str(args.cache_root.resolve()),
        cohort=[asdict(s) for s in specs],
        training_asset_schedule=row["schedule"],
        validation_interval_updates=INTERVAL,
    )
    cli.output_root.mkdir(parents=True, exist_ok=False)
    (cli.output_root / "resolved-config.json").write_text(
        json.dumps(resolved, default=str, indent=2)
    )
    started = time.monotonic()
    train, validation = {}, []
    for ordinal, spec in enumerate(specs):
        if spec.asset.split not in (SplitName.TRAIN, SplitName.VALIDATION):
            continue
        if (
            spec.asset.split == SplitName.TRAIN
            and spec.asset.asset_id not in row["asset_exposures"]
        ):
            continue
        path = _cache_path(args.cache_root, spec)
        assert not path.name.startswith("test-")
        with np.load(path, allow_pickle=False) as data:
            loaded = {key: data[key].copy() for key in data.files}
        if spec.asset.split == SplitName.TRAIN:
            train[spec.asset.asset_id] = (ordinal, loaded)
        else:
            validation.append(loaded)
    assert len(validation) == 25 and len(train) == len(row["asset_exposures"])
    torch.set_num_threads(2)
    torch.manual_seed(cli.seed)
    torch.cuda.manual_seed_all(cli.seed)
    device = torch.device("cuda")
    model = B1FieldModel(
        conditioning="pixel_knn",
        decoder_channels=128,
        knn_neighbors=8,
        minimum_roi_half_extent_m=0.001,
        roi_padding_fraction=0.25,
    ).to(device)
    assert sum(p.numel() for p in model.parameters()) == 156738
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0015, weight_decay=1e-6)
    exposures, curves, losses = Counter(), [], []
    best = float("inf")
    for update, asset in enumerate(row["schedule"], 1):
        ordinal, data = train[asset]
        rng = np.random.default_rng(np.random.SeedSequence((cli.seed, exposures[asset], ordinal)))
        exposures[asset] += 1
        indices = _training_query_indices(
            data["train_points"],
            data["train_udf_m"],
            data["depth_m"],
            data["mask"],
            data["intrinsics"],
            total_count=256,
            certain_free_count=0,
            near_surface_count=128,
            near_surface_mask=data.get("train_near_surface_mask"),
            rng=rng,
        )
        obs = _observation(data, device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pred = model.decode_queries(
            model.encode_observation(*obs[:4]),
            torch.from_numpy(data["train_points"][indices])[None].to(device),
            obs[4],
        ).udf_m
        loss = 20 * F.smooth_l1_loss(
            pred, torch.from_numpy(data["train_udf_m"][indices])[None].to(device), beta=0.005
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach()))
        if update % INTERVAL == 0:
            model.eval()
            errors = [
                float(
                    abs(
                        _predict_udf(model, v, v["evaluation_points"], device)
                        - v["evaluation_udf_m"]
                    ).mean()
                )
                for v in validation
            ]
            value = float(np.mean(errors))
            assert np.isfinite(value) and np.isfinite(errors).all(), "nonfinite validation MAE"
            curves.append(
                dict(
                    update=update,
                    validation_mae_m=value,
                    validation_per_object_mae_m=errors,
                    training_loss=float(np.mean(losses)),
                )
            )
            losses.clear()
            if value < best:
                best = value
                torch.save(model.state_dict(), cli.output_root / "best-validation.pt")
            (cli.output_root / "curves.json").write_text(json.dumps(curves, indent=2))
            print(
                json.dumps(dict(arm=cli.arm, seed=cli.seed, update=update, validation_mae_m=value)),
                flush=True,
            )
    assert dict(exposures) == row["asset_exposures"]
    torch.save(model.state_dict(), cli.output_root / "last.pt")
    summary = dict(
        data_arm=cli.arm,
        seed=cli.seed,
        updates=row["updates"],
        best_validation_mae_m=best,
        best_update=min(curves, key=lambda x: x["validation_mae_m"])["update"],
        end_to_end_run_s=time.monotonic() - started,
        checkpoint_sha256=_sha256_file(cli.output_root / "best-validation.pt"),
        test_opened=False,
    )
    (cli.output_root / "complete.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
