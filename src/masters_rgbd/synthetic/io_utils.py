"""Filesystem and JSON helpers."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)
        file.write("\n")


def ensure_clean_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Directory already exists: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=False)


def make_relative_path(path: Path, root: Path) -> str:
    return Path(os.path.relpath(path.resolve(), root.resolve())).as_posix()


def path_from_dataset_root(dataset_root: Path, relative_path: str) -> Path:
    if Path(relative_path).is_absolute():
        raise ValueError(f"Dataset paths must be relative, got: {relative_path}")
    return dataset_root / relative_path


def safe_copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def remove_if_exists(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()
