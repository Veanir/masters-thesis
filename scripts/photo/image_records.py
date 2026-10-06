"""Durable per-image PHOTO records with hash-checked interrupted commit recovery."""

import hashlib
import io
import json
import os

from PIL import Image

from scripts.common.paths import script_help

script_help(__doc__, __name__)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8388608), b""):
            h.update(block)
    return h.hexdigest()


def atomic(path, raw):
    pending = path.with_name(path.name + ".writing")
    with pending.open("xb") as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    os.replace(pending, path)


def load(folder, sid, expected):
    path = folder / (sid + ".json")
    image = folder / (sid + "-reference.png")
    pending = image.with_name(image.name + ".pending")
    if not path.exists():
        assert not image.exists() and not pending.exists(), (
            "Uncommitted image requires diagnosis; never overwrite or silently regenerate"
        )
        return None
    record = json.loads(path.read_text())
    assert all(record.get(key) == value for key, value in expected.items()), (
        "PHOTO record binding mismatch"
    )
    assert record["filename"] == image.name
    if not image.exists():
        assert pending.exists() and sha(pending) == record["sha256"], (
            "Missing committed image; preserve evidence"
        )
        os.replace(pending, image)
    assert not pending.exists() and sha(image) == record["sha256"], "Existing PHOTO pixels changed"
    with Image.open(image) as value:
        assert value.size == (640, 480) and value.mode == "RGB"
    return record


def commit(folder, sid, image, expected, measurements):
    target = folder / (sid + "-reference.png")
    pending = target.with_name(target.name + ".pending")
    record_path = folder / (sid + ".json")
    assert not target.exists() and not record_path.exists() and not pending.exists()
    assert image.size == (640, 480) and image.mode == "RGB"
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    raw = stream.getvalue()
    with pending.open("xb") as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    record = {
        **measurements,
        **expected,
        "filename": target.name,
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    atomic(record_path, json.dumps(record, indent=2, allow_nan=False).encode())
    os.replace(pending, target)
    return record
