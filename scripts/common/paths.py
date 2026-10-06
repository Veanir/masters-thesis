"""Portable paths and JSON helpers for reproduction workflows."""

import hashlib
import json
import os
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("RGBD_WORKSPACE", str(CODE_ROOT))).resolve()
DATASET_ROOT = Path(os.environ.get("RGBD_DATASETS", str(ROOT / "data"))).resolve()
OUTPUT_ROOT = Path(os.environ.get("RGBD_FIGURES", str(ROOT / "artifacts/figures"))).resolve()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def script_help(description, module_name):
    if module_name == "__main__" and any(arg in {"--help", "-h"} for arg in sys.argv[1:]):
        # Legacy batch scripts have no argument parser; show their real interface.
        print(description)
        print("\nBatch entry point. Set RGBD_WORKSPACE, RGBD_DATASETS and RGBD_FIGURES.")
        print("Inputs and outputs follow the relative paths declared in this script.")
        raise SystemExit(0)
