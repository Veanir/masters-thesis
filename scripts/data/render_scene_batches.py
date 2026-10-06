"""Single owner of sequential fresh renderer processes and retained attempts."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.common.bop_adapter import prepare_observation
from scripts.data.corrupt_rendered_depth import (
    camera_matrices,
    crop_channel,
    cropped_intrinsics,
    sensor_candidate,
)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--admission", type=Path, required=True)
    p.add_argument("--statistics", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--count", type=int, required=True)
    a = p.parse_args()
    statistics = json.loads(a.statistics.read_text())
    export_checks = []
    all_rows = sorted(
        json.loads(a.admission.read_text())["rows"],
        key=lambda r: (r["split_pool"] != "train", r["admission_priority"]),
    )
    selected = all_rows[a.start : a.start + a.count]
    assert len(selected) == a.count and a.count > 0
    assert not a.output.exists()
    a.output.mkdir()
    attempts = a.output / "attempts"
    attempts.mkdir()
    records = []
    script = Path(__file__).parents[2].joinpath("scripts/data/render_isaac_scenes.py")
    for ordinal, row in enumerate(selected, a.start):
        started = time.monotonic()
        target = a.output / row["asset_id"]
        target.mkdir()
        frames = []
        arrangements = []
        for arrangement in range(8):
            if len(frames) >= 20:
                break
            part = attempts / f"target-{ordinal:04d}-arrangement-{arrangement:02d}"
            log = part.with_suffix(".log")
            command = [
                sys.executable,
                str(script),
                "--assets",
                str(a.assets),
                "--admission",
                str(a.admission),
                "--output",
                str(part),
                "--start",
                str(ordinal),
                "--count",
                "1",
                "--arrangement",
                str(arrangement),
                "--allow-root",
            ]
            tick = time.monotonic()
            with log.open("wb") as stream:
                result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
            assert result.returncode == 0, f"Isolated renderer failed; retained {log}"
            child = json.loads((part / "complete.json").read_text())
            assert child["admission_sha256"] == sha(a.admission) and len(child["rows"]) == 1
            item = child["rows"][0]
            assert item["key"] == row["key"] and len(item["arrangements"]) == 1
            ref = item["arrangements"][0]
            scene_folder = part / row["asset_id"] / ref["folder"]
            dest = target / ref["folder"]
            assert not dest.exists()
            shutil.move(str(scene_folder), str(dest))
            # relative_folder stays asset_id/arrangement-NN in the final package.
            eligible = []
            for frame in item["frames"]:
                view = frame["view"]
                sid = frame["frame_id"]
                rgb = np.asarray(Image.open(dest / f"rgb-{view}.png").convert("RGB"))
                with np.load(dest / f"rgbd-{view}.npz") as z:
                    clean = z["depth_clean_m"]
                    mask = z["target_mask"].astype(bool)
                with np.load(dest / f"camera-{view}.npz") as z:
                    k, _ = camera_matrices(z)
                box = (0, 0, clean.shape[1], clean.shape[0])
                rgb = crop_channel(rgb, box, rgb=True)
                clean = crop_channel(clean, box)
                mask = crop_channel(mask, box)
                k = cropped_intrinsics(k, box)
                sensor, noise = sensor_candidate(clean, mask, sid, statistics)
                step = noise["depth_step_m"]
                assert np.isclose(step, 0.001, atol=1e-12)
                raw = np.rint(sensor.astype(float) / step).astype(np.uint32)
                try:
                    data = prepare_observation(rgb, raw, mask, k, step * 1000)
                    check = {
                        "frame_id": sid,
                        "passed": True,
                        "input_pixels": int(data["mask"].sum()),
                        "sensor_seed": noise["seed"],
                    }
                    eligible.append({**frame, "model_input_admission": check})
                except ValueError as error:
                    check = {
                        "frame_id": sid,
                        "passed": False,
                        "reason": str(error),
                        "sensor_seed": noise["seed"],
                    }
                export_checks.append(check)
            frames.extend(eligible[: 20 - len(frames)])
            arrangements.append(
                {
                    **ref,
                    "process_wall_seconds": time.monotonic() - tick,
                    "isolated_manifest": str((part / "complete.json").relative_to(a.output)),
                    "isolated_manifest_sha256": sha(part / "complete.json"),
                }
            )
            print(
                "ISOLATED SCENES",
                ordinal + 1,
                arrangement,
                len(frames),
                arrangements[-1]["process_wall_seconds"],
                flush=True,
            )
        assert len(frames) <= 20
        record = {
            "key": row["key"],
            "asset_id": row["asset_id"],
            "family_id": row["family_id"],
            "split_pool": row["split_pool"],
            "frames": frames,
            "arrangements": arrangements,
            "required_frames": 20,
            "complete_20_frames": len(frames) == 20,
            "seconds": time.monotonic() - started,
        }
        (target / "complete.json").write_text(json.dumps(record, indent=2))
        records.append(record)
        (a.output / "progress.json").write_text(
            json.dumps(
                [
                    {
                        "asset_id": r["asset_id"],
                        "valid_frames": len(r["frames"]),
                        "seconds": r["seconds"],
                    }
                    for r in records
                ],
                indent=2,
            )
        )
    result = {
        "status": "bounded_scene_batch_complete_requires_external_QA",
        "rows": records,
        "admission_sha256": sha(a.admission),
        "script_sha256": sha(Path(__file__)),
        "generator_sha256": sha(script),
        "source_resolution": [960, 720],
        "execution": (
            "One fresh SimulationApp process per physical arrangement, after "
            "same-process stage changes failed RGB capture and three isolated "
            "scenes passed."
        ),
        "model_input_checks": export_checks,
        "statistics_sha256": sha(a.statistics),
        "contract_sha256": sha(
            Path(__file__).parents[2].joinpath("scripts/data/corrupt_rendered_depth.py")
        ),
        "adapter_sha256": sha(Path(__file__).parents[2].joinpath("scripts/common/bop_adapter.py")),
        "target_frames": 20,
        "max_arrangements": 8,
        "physics_dt": 1 / 120,
        "physics_steps": 660,
        "training_ready": False,
    }
    (a.output / "complete.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
