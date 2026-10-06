"""Freeze an observation-only upload for one fully verified TRAIN source batch."""

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path

from scripts.common.paths import CODE_ROOT, script_help
from scripts.common.paths import ROOT as WORKSPACE_ROOT

script_help(__doc__, __name__)

ROOT = WORKSPACE_ROOT


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def prepare(source, out, name):
    source = source.resolve()
    out = out.resolve()
    source.relative_to(ROOT.resolve())
    out.relative_to(ROOT.resolve())
    assert re.fullmatch("[a-z0-9_-]{1,80}", name) and not out.exists()
    path = source / "complete.json"
    manifest = json.loads(path.read_text())
    verification = json.loads((source / "verification-v1.json").read_text())
    assert (
        manifest["status"] == "certified_synthetic_observations_exported"
        and verification["all_passed"]
    )
    assert (
        verification["source_manifest_sha256"] == sha(path)
        and len(manifest["rows"]) == verification["count"]
    )
    assert 0 < len(manifest["rows"]) <= 640 and all(r["split"] == "train" for r in manifest["rows"])
    assert len({r["sample_id"] for r in manifest["rows"]}) == len(manifest["rows"])
    out.mkdir(parents=True)
    archive = out / (name + "-source.zip")
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_STORED) as z:
        z.write(path, "complete.json")
        for row in manifest["rows"]:
            sample = source / row["sample_id"] / "observation.npz"
            assert sha(sample) == row["files"]["observation.npz"]
            z.write(sample, row["sample_id"] + "/observation.npz")
    job = {
        "purpose": "Frozen first-hit supervision for a verified TRAIN source batch",
        "source_archive": archive.name,
        "source_archive_sha256": sha(archive),
        "source_directory": "source-" + name,
        "output": name,
        "source_manifest_sha256": sha(path),
        "source_verification_sha256": sha(source / "verification-v1.json"),
        "observation_count": len(manifest["rows"]),
        "source_export": str(source.relative_to(ROOT)),
        "source_files_sha256": {
            n: sha(CODE_ROOT / "scripts" / n)
            for n in ["data/generate_surface_labels.py", "data/run_label_batch.py"]
        },
        "script_sha256": sha(Path(__file__)),
        "gt_scope": "Synthetic TRAIN supervision; no validation or real heldout observations.",
    }
    job_path = out / "label-batch-job.json"
    job_path.write_text(json.dumps(job, indent=2))
    bootstrap = out / "label-batch.sh"
    bootstrap.write_text(
        "#!/usr/bin/env bash\nset -Eeuo pipefail\n"
        ': "${RGBD_CODE_ROOT:?Set checkout path}"\n'
        ': "${RGBD_LABEL_WORK:?Set extracted job bundle directory}"\n'
        ': "${RGBD_RAY_VENDOR:?Set RaySt3R checkout path}"\n'
        'python "$RGBD_CODE_ROOT/scripts/data/run_label_batch.py" '
        '--work "$RGBD_LABEL_WORK" --vendor "$RGBD_RAY_VENDOR"\n',
        newline="\n",
    )
    result = {
        "status": "verified_TRAIN_label_input_bundle_prepared",
        "name": name,
        "source_export": str(source.relative_to(ROOT)),
        "job": str(job_path.relative_to(ROOT)),
        "job_sha256": sha(job_path),
        "bootstrap": str(bootstrap.relative_to(ROOT)),
        "bootstrap_sha256": sha(bootstrap),
        "archive": str(archive.relative_to(ROOT)),
        "archive_bytes": archive.stat().st_size,
    }
    (out / "complete.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--name", required=True)
    a = p.parse_args()
    print(json.dumps(prepare(a.source, a.output, a.name)), flush=True)


if __name__ == "__main__":
    main()
