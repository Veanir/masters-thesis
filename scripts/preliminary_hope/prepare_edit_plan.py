"""Freeze two edits for every TRAIN observation before reviewing any new output."""

import shutil
import zipfile

from scripts.common.paths import ROOT, digest, read, save, script_help

script_help(__doc__, __name__)


OUT = ROOT / "runs/klein-candidates-20260906"


def main():
    bundle = OUT / "job"
    if bundle.exists():
        raise RuntimeError("Candidate job already exists; do not overwrite its contract")
    contexts = ROOT / "runs/photoreal-training-contexts-20260906"
    manifest = read(contexts / "complete.json")
    assert manifest["status"] == "complete" and len(manifest["rows"]) == 36
    assert manifest["training_labels_modified"] is False
    screen = read(ROOT / "runs/photoreal-model-screen-20260906/job/job.json")
    model = next(m for m in screen["models"] if m["name"] == "klein")
    prompts = [
        screen["prompt"],
        screen["prompt"].replace("natural soft daylight", "soft warm indoor illumination"),
    ]
    pilot = read(ROOT / "configs/sim2real-photoreal-pilot-inputs-20260906.json")
    pilot_ids = {r["sample_id"] for r in pilot["rows"]}
    bundle.mkdir(parents=True)
    files, samples = {}, []
    for row in sorted(
        manifest["rows"], key=lambda r: (r["sample_id"] not in pilot_ids, r["sample_id"])
    ):
        sid = row["sample_id"]
        assert row["split"] == "train" and row["target_rgb_exact"]
        (bundle / sid).mkdir()
        for name in ("rgb.png", "mask.png"):
            source = contexts / sid / name
            assert digest(source) == row["files"][name]
            shutil.copyfile(source, bundle / sid / name)
            files[sid + "/" + name] = digest(source)
        samples.append(
            {
                "sample_id": sid,
                "category_id": row["category_id"],
                "ordinal": row["ordinal"],
                "rgb": sid + "/rgb.png",
                "mask": sid + "/mask.png",
                "source_npz_sha256": row["training_npz_sha256"],
                "pilot12": sid in pilot_ids,
                "context_repaired": row["context_repaired"],
            }
        )
    for source in (
        ROOT / "scripts/preliminary_hope/generate_edits.py",
        ROOT / "environments/photo/requirements.txt",
    ):
        shutil.copyfile(source, bundle / source.name)
        files[source.name] = digest(source)
    save(
        bundle / "job.json",
        {
            "purpose": (
                "All TRAIN36 x two fixed appearance variants; candidate outputs "
                "require individual review"
            ),
            "model": model,
            "samples": samples,
            "files": files,
            "prompts": prompts,
            "seed_base": 2026090700,
            "expected_images": 72,
            "settings": {
                "steps": 4,
                "guidance_scale": 1.0,
                "precision": "bfloat16",
                "offload": "model_cpu",
            },
            "training_sha256": manifest["training_sha256"],
            "contexts_sha256": digest(contexts / "complete.json"),
            "training_ready": False,
            "test_outcomes_used": False,
            "review_policy": (
                "Review every raw target crop against original RGB and geometry. "
                "Reject movement, reshaping, new/missing parts or unreadable "
                "appearance. Rejected variant falls back to exact original TRAIN "
                "RGB; no generated variant is accepted by masking alone. No "
                "retries or best-of selection."
            ),
            "selection": (
                "All 36 TRAIN observations; preselected TRAIN12 generated first. "
                "No validation or test images uploaded."
            ),
        },
    )
    with zipfile.ZipFile(OUT / "job.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(bundle.rglob("*")):
            if path.is_file():
                z.write(path, path.relative_to(bundle).as_posix())
    print({"samples": len(samples), "images": 72, "job_sha256": digest(bundle / "job.json")})


if __name__ == "__main__":
    main()
