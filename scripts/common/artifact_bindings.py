"""Validate local artifact identity and observation membership."""

import hashlib
import json

from scripts.common.paths import ROOT


def sha(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def canonical(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def local(path):
    path = path.resolve()
    path.relative_to(ROOT.resolve())
    return path


def reference(path):
    path = local(path)
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path)}


def bound(ref):
    path = local(ROOT / ref["path"])
    assert sha(path) == ref["sha256"], "Changed bound artifact: " + str(path)
    return path


def save(path, value):
    # Exclusive writes prevent replacing a previous freeze or partial attempt.
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def indexed(rows):
    result = {row["sample_id"]: row for row in rows}
    assert len(result) == len(rows), "Duplicate observation IDs"
    return result
