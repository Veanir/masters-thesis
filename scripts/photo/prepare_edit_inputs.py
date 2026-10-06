"""Prepare source-certified PHOTO inputs while the author control stays pending."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.common.paths import ROOT as WORKSPACE_ROOT

ROOT = WORKSPACE_ROOT
DATA = ROOT / "runs/research-evolution-data-20260907"
PHOTO = ROOT / "runs/research-evolution-photo-20260907"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8388608), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def prepare_group(folder, group, out, plan, pilot, predictions):
    path = folder / "complete.json"
    verified_path = folder / "verification-v1.json"
    source = json.loads(path.read_text())
    qa = json.loads(verified_path.read_text())
    assert (
        qa["all_passed"]
        and qa["source_manifest_sha256"] == sha(path)
        and qa["count"] == len(source["rows"])
    )
    rows = sorted(
        [r for r in source["rows"] if r["split"] == "train"], key=lambda r: r["sample_id"]
    )
    assert (
        len(rows) == 640
        and len({r["sample_id"] for r in rows}) == 640
        and len({r["key"] for r in rows}) == 32
    )
    assert {r["key"] for r in rows} == set(plan["groups"][group]["geometry_keys"])
    out.mkdir()
    records = []
    for local_ordinal, row in enumerate(rows):
        sid = row["sample_id"]
        original = folder / sid / "observation.npz"
        assert sha(original) == row["files"]["observation.npz"]
        with np.load(original, allow_pickle=False) as z:
            rgb = z["rgb"]
            mask = z["source_target_mask"]
            depth = z["depth_clean_m"]
            sensor = z["input_mask"]
        assert (
            rgb.shape == (480, 640, 3)
            and rgb.dtype == np.uint8
            and mask.shape == depth.shape == sensor.shape == (480, 640)
        )
        mask = mask.astype(bool)
        assert mask.any() and int(sensor.sum()) >= 512
        gray = np.where(
            np.isfinite(depth) & (depth > 0), np.clip((2 - depth) / 1.9, 0, 1) * 255, 0
        ).astype(np.uint8)
        target = out / sid
        target.mkdir()
        Image.fromarray(rgb).save(target / "rgb.png")
        Image.fromarray(mask.astype(np.uint8) * 255).save(target / "mask.png")
        Image.fromarray(np.repeat(gray[:, :, None], 3, axis=2)).save(target / "depth-guide.png")
        files = {n: sha(target / n) for n in ["rgb.png", "mask.png", "depth-guide.png"]}
        ordinal = group * 640 + local_ordinal
        seed = plan["seed_base"] + 48 + ordinal
        reuse = None
        if sid in pilot:
            old = pilot[sid]
            ref = predictions[sid]
            assert files == old["files"] and sha(original) == old["source_observation_sha256"]
            seed = plan["seed_base"] + old["ordinal"]
            assert (
                ref["seed"] == seed
                and ref["condition"] == "reference"
                and ref["input_rgb_sha256"] == files["rgb.png"]
            )
            generated = (
                DATA
                / "photo24-reference-v3/unpacked/results/photo24-reference-v3"
                / ref["filename"]
            )
            assert sha(generated) == ref["sha256"]
            reuse = {
                "path": str(generated.relative_to(ROOT)),
                "sha256": ref["sha256"],
                "seed": seed,
                "reason": (
                    "Exact prior source, recipe and seed. Reuse all48 including "
                    "rejected/unreviewed outputs; no quality-based regeneration."
                ),
            }
        records.append(
            {
                "sample_id": sid,
                "ordinal": ordinal,
                "seed": seed,
                "asset_key": row["key"],
                "family_id": row["family_id"],
                "source": row["key"].split("/")[0],
                "source_observation_sha256": sha(original),
                "source_rgb_array_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                "files": files,
                "target_mask_pixels": int(mask.sum()),
                "sensor_input_pixels": int(sensor.sum()),
                "reuse_pilot_output": reuse,
            }
        )
    manifest = {
        "status": "PHOTO_source_inputs_prepared_no_generation_or_admission",
        "rows": records,
        "source_manifest_sha256": sha(path),
        "source_verification_sha256": sha(verified_path),
        "plan_sha256": sha(out.parent / "plan.json"),
        "source_export": str(folder.relative_to(ROOT)),
        "group": group,
        "script_sha256": sha(Path(__file__)),
        "resolution": [640, 480],
        "GT_uploaded": False,
        "scope": (
            "RGB, clean-scene depth guide and clean target mask from "
            "certified TRAIN observations. No novel-view supervision or real "
            "data. RGB matches future label source exactly. Mask is for QA "
            "only; generator sees RGB plus depth guide."
        ),
    }
    save(out / "manifest.json", manifest)
    archive = out.with_suffix(".zip")
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_STORED) as z:
        z.write(out / "manifest.json", "manifest.json")
        for row in records:
            for filename in row["files"]:
                z.write(out / row["sample_id"] / filename, row["sample_id"] + "/" + filename)
    # Independent decode and source-array comparison also catches wrong PNG mode
    # or encoding; checksums alone would only certify the produced bytes.
    for row in records:
        sid = row["sample_id"]
        with np.load(folder / sid / "observation.npz", allow_pickle=False) as original:
            assert np.array_equal(np.asarray(Image.open(out / sid / "rgb.png")), original["rgb"])
            assert np.array_equal(
                np.asarray(Image.open(out / sid / "mask.png")) > 0,
                original["source_target_mask"].astype(bool),
            )
            d = original["depth_clean_m"]
            expected = np.zeros(d.shape, np.uint8)
            valid = np.isfinite(d) & (d > 0)
            expected[valid] = (np.clip((2 - d[valid]) / 1.9, 0, 1) * 255).astype(np.uint8)
            assert np.array_equal(
                np.asarray(Image.open(out / sid / "depth-guide.png")),
                np.repeat(expected[..., None], 3, axis=2),
            )
    with zipfile.ZipFile(archive) as z:
        expected = {"manifest.json"} | {
            r["sample_id"] + "/" + n for r in records for n in r["files"]
        }
        assert set(z.namelist()) == expected
        assert hashlib.sha256(z.read("manifest.json")).hexdigest() == sha(out / "manifest.json")
        for row in records:
            for n, digest in row["files"].items():
                assert hashlib.sha256(z.read(row["sample_id"] + "/" + n)).hexdigest() == digest
    result = {
        "status": "all640_PHOTO_inputs_verified_no_generation_or_admission",
        "group": group,
        "manifest_sha256": sha(out / "manifest.json"),
        "archive_sha256": sha(archive),
        "archive_bytes": archive.stat().st_size,
        "count": 640,
        "reused_pilot_count": sum(r["reuse_pilot_output"] is not None for r in records),
        "decoded_source_arrays_checked": True,
        "archive_entries_and_hashes_checked": True,
        "source_manifest_sha256": sha(path),
        "script_sha256": sha(Path(__file__)),
    }
    save(out / "verification.json", result)
    print(json.dumps(result), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--prepare-available", action="store_true")
    a = p.parse_args()
    out = a.output.resolve()
    source_plan_path = DATA / "source-final-plan-v1/plan.json"
    source_plan = json.loads(source_plan_path.read_text())
    pilot_path = PHOTO / "photo24-reference-job-v3.json"
    pilot_job = json.loads(pilot_path.read_text())
    pilot = {r["sample_id"]: r for r in pilot_job["rows"]}
    pred_path = DATA / "photo24-reference-v3/unpacked/results/photo24-reference-v3/complete.json"
    pred = json.loads(pred_path.read_text())
    assert pred["job_sha256"] == sha(pilot_path) and len(pilot) == 48 and len(pred["rows"]) == 48
    predictions = {r["sample_id"]: r for r in pred["rows"]}
    assert set(predictions) == set(pilot)
    source_refs = [
        {
            "source_export": source_plan["initial_source"],
            "expected_manifest_sha256": source_plan["initial_source_manifest_sha256"],
            "geometry_keys": sorted(
                {
                    r["key"]
                    for r in json.loads(
                        (ROOT / source_plan["initial_source"] / "complete.json").read_text()
                    )["rows"]
                    if r["split"] == "train"
                }
            ),
        }
    ]
    for batch in source_plan["batches"][:9]:
        job_path = ROOT / batch["job"]
        assert sha(job_path) == batch["job_sha256"]
        job = json.loads(job_path.read_text())
        source_refs.append(
            {
                "source_export": batch["export"],
                "geometry_keys": sorted(job["keys"]),
                "required_source_proof": str(
                    (
                        DATA / "source-final-v1" / batch["name"] / "local-QA-complete.json"
                    ).relative_to(ROOT)
                ),
            }
        )
    assert (
        len(source_refs) == 10
        and len({key for g in source_refs for key in g["geometry_keys"]}) == 320
    )
    recipe = {
        key: pilot_job[key]
        for key in [
            "model",
            "revision",
            "prompt",
            "depth_suffix",
            "num_inference_steps",
            "guidance_scale",
            "depth_guide_mapping",
            "resolution",
            "profile_choice_sha256",
        ]
    }
    if not out.exists():
        out.mkdir()
        save(
            out / "plan.json",
            {
                "status": "source_only_preparation_bound_while_author_PHOTO_control_pending",
                "groups": source_refs,
                "expected_observations": 6400,
                "source_plan_sha256": sha(source_plan_path),
                "pilot_job_sha256": sha(pilot_path),
                "pilot_result_sha256": sha(pred_path),
                "recipe": recipe,
                "seed_base": pilot_job["seed_base"],
                "seed_rule": (
                    "All48 pilot IDs keep original seed_base+pilot ordinal and output "
                    "including failures. Remaining IDs use seed_base+48+global "
                    "ordinal. Global ordinal=640*group+lexicographic SID index within "
                    "group. Groups=initial32TRAIN then fixed source batch order."
                ),
                "script_sha256": sha(Path(__file__)),
                "generation_authorized_by_this_file": False,
                "PHOTO_admission_frozen": False,
                "scope": (
                    "Preparation of source inputs only. Final generator approval, "
                    "author control, full-image and target-part review, SAM "
                    "calibration, admission and matched CLASSIC exposure remain "
                    "required before training."
                ),
            },
        )
    plan = json.loads((out / "plan.json").read_text())
    assert plan["source_plan_sha256"] == sha(source_plan_path) and plan["script_sha256"] == sha(
        Path(__file__)
    )
    assert (
        plan["pilot_job_sha256"] == sha(pilot_path)
        and plan["pilot_result_sha256"] == sha(pred_path)
        and plan["groups"] == source_refs
        and plan["recipe"] == recipe
    )
    if a.prepare_available:
        for group, ref in enumerate(source_refs):
            folder = ROOT / ref["source_export"]
            dest = out / ("group-" + str(group).zfill(2))
            if dest.exists():
                assert (dest / "verification.json").exists(), (
                    "Partial preparation must be diagnosed, not overwritten"
                )
                continue
            if "required_source_proof" in ref:
                proof_path = ROOT / ref["required_source_proof"]
                if not proof_path.exists():
                    continue
                proof = json.loads(proof_path.read_text())
                assert proof["source"] == ref["source_export"] and proof["count"] == 640
                assert (
                    sha(folder / "complete.json") == proof["manifest_sha256"]
                    and sha(folder / "verification-v1.json") == proof["verification_sha256"]
                )
            else:
                assert sha(folder / "complete.json") == ref["expected_manifest_sha256"]
            prepare_group(folder, group, dest, plan, pilot, predictions)
    verified = [json.loads(p.read_text()) for p in sorted(out.glob("group-*/verification.json"))]
    records = [
        r
        for path in sorted(out.glob("group-*/manifest.json"))
        for r in json.loads(path.read_text())["rows"]
    ]
    assert len({r["sample_id"] for r in records}) == len(records) and len(
        {r["seed"] for r in records}
    ) == len(records)
    if len(verified) == 10:
        assert (
            len(records) == 6400 and sum(r["reuse_pilot_output"] is not None for r in records) == 48
        )
    save(
        out / "progress.json",
        {
            "prepared_observations": len(records),
            "expected_observations": 6400,
            "groups": verified,
            "generation_started": False,
            "admission_frozen": False,
            "plan_sha256": sha(out / "plan.json"),
        },
    )
    print(
        "PHOTO INPUTS PREPARED", len(records), "of6400; author control remains pending", flush=True
    )


if __name__ == "__main__":
    main()
