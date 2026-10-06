"""Export TRAIN36 image-edit contexts after split-safe re-rendering; no model selection."""

from __future__ import annotations

import shutil

import numpy as np
from PIL import Image

from scripts.common.paths import ROOT, digest, read, save, script_help

script_help(__doc__, __name__)


def main():
    train = ROOT / "runs/ray-adaptation-training-data-v5-20260905"
    source = ROOT.parent / "output/abo-discovery-360-v1"
    repaired = ROOT / "runs/photoreal-clean-context-20260906"
    output = ROOT / "runs/photoreal-training-contexts-20260906"
    assert not output.exists()
    repairs = {r["sample_id"]: r for r in read(repaired / "complete.json")["rows"]}
    audit = {
        r["sample_id"]: r
        for r in read(ROOT / "runs/photoreal-model-screen-20260906/context-split-audit.json")[
            "rows"
        ]
    }
    pilot = {
        r["sample_id"]: r
        for r in read(ROOT / "runs/photoreal-model-screen-20260906/job/job.json")["samples"]
    }
    rows = read(train / "complete.json")["rows"]
    assert len(rows) == 36 and len(repairs) == 14
    output.mkdir()
    shutil.copyfile(__file__, output / "executed-script.py")
    results = []
    for row in rows:
        assert row["split"] == "train"
        sid = row["sample_id"]
        item = audit[sid]
        if item["excluded_assets"]:
            patch = repairs[sid]
            assert patch["removed_asset_ids"] == item["excluded_assets"]
            assert not (set(patch["remaining_asset_ids"]) & set(item["excluded_assets"]))
            context = repaired / sid
            for name, sha in patch["files"].items():
                assert digest(context / name) == sha
        else:
            context = source / "renders/train" / sid
            sample = read(context / "sample.json")
            for name in ("rgb_clean", "depth_clean_m"):
                assert (
                    digest(source / sample["artifacts"][name]["uri"])
                    == sample["artifacts"][name]["sha256"]
                )
        path = train / (sid + ".npz")
        assert digest(path) == row["output_sha256"]
        with np.load(path, allow_pickle=False) as data:
            mask = data["input_mask"].astype(bool)
            target = data["rgb"]
        rgb = np.asarray(
            Image.open(context / "rgb_clean.png")
            .convert("RGB")
            .resize((640, 480), Image.Resampling.BILINEAR)
        ).copy()
        rgb[mask] = target[mask]
        depth = np.load(context / "depth_clean_m.npy", allow_pickle=False)
        depth = depth[(np.arange(480) * 360 // 480)[:, None], np.arange(640)[None, :]]
        valid = np.isfinite(depth) & (depth > 0)
        inv = np.zeros_like(depth)
        inv[valid] = 1 / depth[valid]
        low, high = np.percentile(inv[valid], [1, 99])
        control = np.where(
            valid, np.rint(255 * np.clip((inv - low) / (high - low), 0, 1)), 0
        ).astype(np.uint8)
        dest = output / sid
        dest.mkdir()
        for name, pixels in [
            ("rgb", rgb),
            ("mask", mask.astype(np.uint8) * 255),
            ("depth_control", control),
        ]:
            Image.fromarray(pixels).save(dest / (name + ".png"))
        if sid in pilot:
            original = ROOT / "runs/photoreal-model-screen-20260906/job"
            for name in ("rgb", "mask", "depth_control"):
                assert digest(dest / (name + ".png")) == digest(original / pilot[sid][name])
        assert np.array_equal(rgb[mask], target[mask])
        results.append(
            {k: row[k] for k in ("sample_id", "ordinal", "category_id", "asset_id", "split")}
        )
        results[-1].update(
            context_repaired=bool(item["excluded_assets"]),
            removed_distractor_ids=item["excluded_assets"],
            context_visible_asset_ids=[
                i for i in item["visible_asset_ids"] if i not in item["excluded_assets"]
            ],
            target_rgb_exact=True,
            training_npz_sha256=digest(path),
            inverse_depth_percentiles=[float(low), float(high)],
            files={f.name: digest(f) for f in dest.iterdir()},
        )
    save(
        output / "complete.json",
        {
            "status": "complete",
            "rows": results,
            "training_ready": False,
            "training_sha256": digest(train / "complete.json"),
            "source_audit_sha256": digest(
                ROOT / "runs/photoreal-model-screen-20260906/context-split-audit.json"
            ),
            "repairs_sha256": digest(repaired / "complete.json"),
            "script_sha256": digest(__file__),
            "training_labels_modified": False,
            "generator_context_only": True,
            "context_policy": (
                "22 existing full scenes plus 14 exact-pose re-renders without "
                "held-out distractors. Retain original target RGB inside original "
                "training mask; after editing retain only that original mask."
            ),
        },
    )
    print({"inputs": len(results), "repaired": len(repairs), "original_pilot_exact": len(pilot)})


if __name__ == "__main__":
    main()
