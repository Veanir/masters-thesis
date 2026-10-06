"""Re-render TRAIN context after removing held-out distractors; labels stay frozen."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.common.paths import ROOT, digest, read, save, script_help

script_help(__doc__, __name__)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=1)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    assert a.limit > 0 and not a.output.exists()
    provider = ROOT.parent / "masters-rgbd-data-diversity/src"
    sys.path.insert(0, str(provider))
    import sapien

    from masters_rgbd.contracts.manifests import SplitName
    from masters_rgbd.data.abo_render_360 import (
        RenderAsset,
        RenderSpec,
        _camera_pose,
        _matrix_to_quaternion,
        _ObjectScene,
    )

    assert importlib.metadata.version("sapien") == "3.0.3"
    source = ROOT.parent / "output/abo-discovery-360-v1"
    audit = read(ROOT / "runs/photoreal-model-screen-20260906/context-split-audit.json")
    candidates = [r for r in audit["rows"] if r["excluded_assets"]][: a.limit]
    assets = {r["asset_id"]: r for r in read(source / "package/inventory.json")["assets"]}
    config = read(source / "package/dataset.json")["renderer"]["config"]
    spec = RenderSpec(**{k: config[k] for k in RenderSpec.__dataclass_fields__})
    a.output.mkdir(parents=True)
    shutil.copyfile(__file__, a.output / "executed-script.py")
    save(
        a.output / "binding.json",
        {
            "source_renderer_sha256": digest(provider / "masters_rgbd/data/abo_render_360.py"),
            "script_sha256": digest(__file__),
            "selection_audit_sha256": digest(
                ROOT / "runs/photoreal-model-screen-20260906/context-split-audit.json"
            ),
            "purpose": (
                "Remove held-out distractors from generator context only; do not "
                "change training depth/mask/GT"
            ),
            "source_reproduction_thresholds": {
                "mask_iou_min": 0.999,
                "target_depth_p95_max_m": 0.00001,
            },
            "original_observation_pixels_must_remain_target": True,
            "sapien": "3.0.3",
        },
    )
    records = []
    for item in candidates:
        sid = item["sample_id"]
        original = source / "renders/train" / sid
        scene_data = read(original / "sample.json")

        def asset(asset_id):
            row = assets[asset_id]
            mesh = source / row["original_mesh"]["uri"]
            assert digest(mesh) == row["original_mesh"]["sha256"]
            surface = read(source / row["processed_mesh"]["uri"])
            return RenderAsset(
                asset_id,
                row["category_id"],
                mesh,
                source / surface["vertices"]["uri"],
                source / surface["faces"]["uri"],
                SplitName.TRAIN,
            )

        scene = _ObjectScene(
            asset(scene_data["asset_id"]),
            tuple(asset(i) for i in scene_data["distractor_asset_ids"]),
            spec,
        )
        try:
            target_world = scene._set_actor(0, np.zeros(2), 0.0)
            world_camera = target_world @ np.linalg.inv(np.asarray(scene_data["camera_T_mesh"]))
            assert (
                np.max(np.abs(world_camera[:3, 3] - scene_data["camera_position_world_m"])) < 1e-7
            )
            scene.camera.entity.set_pose(
                _camera_pose(sapien, world_camera[:3, 3], world_camera[:3, :3])
            )
            for actor, pose in zip(
                scene.actors[1:], scene_data["distractor_world_transforms"], strict=True
            ):
                pose = np.asarray(pose)
                actor.set_pose(sapien.Pose(pose[:3, 3], _matrix_to_quaternion(pose[:3, :3])))
            if scene_data["controlled_occluder"]:
                occluder = scene_data["controlled_occluder"]
                assert np.allclose(
                    occluder["size_m"], [scene.occluder_width, 0.008, scene.occluder_height]
                )
                pose = np.asarray(occluder["world_transform"])
                scene.occluder.set_pose(
                    sapien.Pose(pose[:3, 3], _matrix_to_quaternion(pose[:3, :3]))
                )
                scene.occluder_body.enable()
            rgb, depth, ids = scene._render()
            mask = ids == scene.actors[0].get_per_scene_id()
            old_mask = np.asarray(Image.open(original / "target_mask.png")) > 0
            old_depth = np.load(original / "depth_clean_m.npy", allow_pickle=False)
            iou = float((mask & old_mask).sum() / (mask | old_mask).sum())
            error = float(
                np.percentile(np.abs(depth[mask & old_mask] - old_depth[mask & old_mask]), 95)
            )
            assert iou >= 0.999 and error <= 0.00001, (sid, iou, error)
            for index, asset_id in enumerate(scene_data["distractor_asset_ids"], start=1):
                if asset_id in item["excluded_assets"]:
                    scene.bodies[index].disable()
            rgb, clean_depth, clean_ids = scene._render()
            clean_mask = clean_ids == scene.actors[0].get_per_scene_id()
            # Removing occluders may reveal more target, but every original target
            # pixel must remain the same visible surface. The model later sees
            # only the unchanged original mask, after RGB editing.
            assert np.all(clean_mask[old_mask])
            assert np.max(np.abs(clean_depth[old_mask] - old_depth[old_mask])) < 0.0001
            folder = a.output / sid
            folder.mkdir()
            Image.fromarray(rgb).save(folder / "rgb_clean.png")
            np.save(folder / "depth_clean_m.npy", clean_depth)
            Image.fromarray(clean_mask.astype(np.uint8) * 255).save(
                folder / "context_target_mask.png"
            )
            record = {
                "sample_id": sid,
                "removed_asset_ids": item["excluded_assets"],
                "remaining_asset_ids": [
                    i for i in item["visible_asset_ids"] if i not in item["excluded_assets"]
                ],
                "original_scene_sha256": digest(original / "sample.json"),
                "original_mask_iou": iou,
                "original_target_depth_p95_m": error,
                "original_target_pixels": int(old_mask.sum()),
                "context_target_pixels": int(clean_mask.sum()),
                "training_labels_modified": False,
                "files": {f.name: digest(f) for f in folder.iterdir()},
            }
            save(folder / "complete.json", record)
            records.append(record)
            print(json.dumps(record), flush=True)
        finally:
            scene.close()
    save(
        a.output / "complete.json", {"status": "complete", "rows": records, "training_ready": False}
    )


if __name__ == "__main__":
    main()
