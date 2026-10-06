"""Validate pilot generation archives without any infrastructure access."""

import hashlib
import json
import zipfile

from scripts.common.paths import (
    digest,
    read,
    script_help,
)

script_help(__doc__, __name__)


def verify(archive, job_path):
    job = read(job_path)
    with zipfile.ZipFile(archive) as z:
        result = json.loads(z.read("complete.json"))
        binding = json.loads(z.read("binding.json"))
        assert result["binding"] == binding and binding["job_sha256"] == digest(job_path)
        assert binding["script_sha256"] == job["files"]["preliminary_hope/generate_edits.py"]
        assert binding["settings"] == job["settings"] and binding["model"] == job["model"]
        rows = result["rows"]
        expected = {(s["sample_id"], v) for s in job["samples"] for v in (0, 1)}
        assert (
            len(rows) == job["expected_images"]
            and {(r["sample_id"], r["variant"]) for r in rows} == expected
        )
        for row in rows:
            for key in ("raw", "masked"):
                assert hashlib.sha256(z.read(row[key])).hexdigest() == row[key + "_sha256"]
    return {"archive_sha256": digest(archive), "images": len(rows), "training_ready": False}
