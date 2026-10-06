"""Generate surface supervision from a verified local source bundle."""

import argparse
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from scripts.common.paths import CODE_ROOT, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--vendor", type=Path, required=True)
    args = parser.parse_args()
    work = args.work.resolve()
    job_path = work / "label-batch-job.json"
    job = json.loads(job_path.read_text())
    assert all(
        digest(CODE_ROOT / "scripts" / name) == sha
        for name, sha in job["source_files_sha256"].items()
    )
    source = (work / job["source_directory"]).resolve()
    out = (work / "results" / job["output"]).resolve()
    assert shutil.disk_usage(work).free > 10 * 1024**3
    assert source.is_relative_to(work) and out.is_relative_to(work / "results")
    assert not source.exists()
    out.mkdir(parents=True, exist_ok=False)
    (out / "job.json").write_bytes(job_path.read_bytes())
    archive = work / job["source_archive"]
    assert digest(archive) == job["source_archive_sha256"]
    with zipfile.ZipFile(archive) as bundle:
        assert all((source / name).resolve().is_relative_to(source) for name in bundle.namelist())
        bundle.extractall(source)
    assert digest(source / "complete.json") == job["source_manifest_sha256"]
    manifest = json.loads((source / "complete.json").read_text())
    assert len(manifest["rows"]) == job["observation_count"]
    assert all(row["split"] == "train" for row in manifest["rows"])
    subprocess.run(
        [
            sys.executable,
            str(CODE_ROOT / "scripts/data/generate_surface_labels.py"),
            "--source",
            str(source),
            "--vendor",
            str(args.vendor),
            "--output",
            str(out / "labels"),
            "--threads",
            "4",
        ],
        check=True,
    )
    labels = json.loads((out / "labels/complete.json").read_text())
    assert len(labels["rows"]) == job["observation_count"]
    (out / "complete.json").write_text(
        json.dumps(
            {
                "status": "label_batch_generated_requires_independent_local_QA",
                "job_sha256": digest(job_path),
                "labels_manifest_sha256": digest(out / "labels/complete.json"),
                "source_manifest_sha256": job["source_manifest_sha256"],
                "observation_count": job["observation_count"],
                "script_sha256": digest(Path(__file__)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
