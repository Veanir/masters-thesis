"""Materialize the original deterministic B1 query cache."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any
from uuid import uuid4
from zipfile import BadZipFile

import numpy as np

from masters_rgbd.b1.dev_cohort import (
    DevViewSpec,
    _safe_path,
    load_dev_sample,
    prepare_dev_queries,
)
from masters_rgbd.b1.residual_labels import (
    ResidualLabelSpec,
    build_residual_label_cache,
    load_residual_label_cache,
    validate_residual_label_cache_sources,
)
from masters_rgbd.b1.training import _cache_path, _cache_spec, _manifest_path, _sha256_file

_CACHE_ENTRY_SCHEMA_VERSION = "rapid-dev-cache-entry-v2"

_CACHE_METADATA_MEMBER = "cache_metadata_json"

_CACHE_ARRAY_DTYPES = {
    "rgb": "float32",
    "depth_m": "float32",
    "mask": "bool",
    "partial_points_camera_m": "float64",
    "intrinsics": "float32",
    "train_points": "float32",
    "train_udf_m": "float32",
    "train_near_surface_mask": "bool",
    "evaluation_points": "float32",
    "evaluation_udf_m": "float32",
    "evaluation_near_surface_mask": "bool",
}


def _residual_cache_path(cache_root: Path, spec: DevViewSpec) -> Path:
    digest = hashlib.sha256(spec.sample_id.encode("utf-8")).hexdigest()
    return cache_root / f"{spec.asset.split.value}-{spec.asset.asset_id}-{digest}"


def _ensure_cache_spec(cache_root: Path, expected: dict[str, Any]) -> None:
    cache_root.mkdir(parents=True, exist_ok=True)
    spec_path = cache_root / "cache-spec.json"
    if spec_path.is_file():
        try:
            actual = json.loads(spec_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError(f"cannot validate rapid Dev cache spec: {spec_path}") from error
        if actual != expected:
            raise RuntimeError(
                "rapid Dev cache spec mismatch; use a new output root instead of reusing cache"
            )
        return
    if any(cache_root.iterdir()):
        raise RuntimeError(
            "rapid Dev cache contains data without cache-spec.json; refusing unverified reuse"
        )
    spec_path.write_text(
        json.dumps(expected, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _validate_cache_spec(cache_root: Path, expected: dict[str, Any]) -> None:
    """Validate a cache root without creating or changing anything in it."""

    spec_path = cache_root / "cache-spec.json"
    try:
        actual = json.loads(spec_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"cannot validate read-only rapid Dev cache spec: {spec_path}"
        ) from error
    if actual != expected:
        raise RuntimeError("read-only rapid Dev cache spec mismatch")


@dataclass(frozen=True, slots=True)
class _CacheMaterializationTask:
    spec: DevViewSpec
    dataset_root: Path
    manifest: Path
    cache_root: Path
    data_seed: int
    train_queries: int
    evaluation_queries: int
    partial_points: int
    cache_path_resolver: Callable[[Path, DevViewSpec], Path]
    query_ordinal: int
    mesh_loader: Callable[[Path], tuple[np.ndarray, np.ndarray, str]] | None
    residual_cache_root: Path | None
    residual_label_spec: ResidualLabelSpec
    residual_cache_read_only: bool


def _array_binding(value: np.ndarray) -> dict[str, object]:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return {"dtype": str(array.dtype), "shape": list(array.shape), "sha256": digest.hexdigest()}


def _source_binding(task: _CacheMaterializationTask) -> dict[str, object]:
    root = task.dataset_root.resolve(strict=True)
    manifest_parent = task.manifest.resolve(strict=True).parent
    paths = {
        "rgb": _safe_path(root, task.spec.rgb_uri, base=manifest_parent),
        "depth": _safe_path(root, task.spec.depth_uri, base=manifest_parent),
        "mask": _safe_path(root, task.spec.mask_uri, base=manifest_parent),
        "mesh": _safe_path(root, task.spec.mesh_uri, base=manifest_parent),
    }
    spec_payload = json.dumps(asdict(task.spec), default=str, sort_keys=True, separators=(",", ":"))
    return {
        "spec_sha256": hashlib.sha256(spec_payload.encode("utf-8")).hexdigest(),
        "files_sha256": {name: _sha256_file(path) for name, path in sorted(paths.items())},
    }


def _expected_cache_shapes(task: _CacheMaterializationTask) -> dict[str, list[int]]:
    camera = task.spec.camera
    height = int(camera["height"])
    width = int(camera["width"])
    partial_count = min(task.partial_points, task.spec.visible_pixel_count)
    return {
        "rgb": [height, width, 3],
        "depth_m": [height, width],
        "mask": [height, width],
        "partial_points_camera_m": [partial_count, 3],
        "intrinsics": [4],
        "train_points": [task.train_queries, 3],
        "train_udf_m": [task.train_queries],
        "train_near_surface_mask": [task.train_queries],
        "evaluation_points": [task.evaluation_queries, 3],
        "evaluation_udf_m": [task.evaluation_queries],
        "evaluation_near_surface_mask": [task.evaluation_queries],
    }


def _cache_entry_is_valid(
    path: Path, task: _CacheMaterializationTask, source_binding: dict[str, object]
) -> bool:
    expected_members = set(_CACHE_ARRAY_DTYPES) | {_CACHE_METADATA_MEMBER}
    expected_shapes = _expected_cache_shapes(task)
    try:
        with np.load(path, allow_pickle=False) as cached:
            if set(cached.files) != expected_members:
                return False
            metadata_value = cached[_CACHE_METADATA_MEMBER]
            if metadata_value.shape != () or metadata_value.dtype.kind != "U":
                return False
            metadata = json.loads(str(metadata_value.item()))
            arrays = {name: np.asarray(cached[name]) for name in _CACHE_ARRAY_DTYPES}
    except (OSError, ValueError, KeyError, EOFError, BadZipFile, json.JSONDecodeError):
        return False
    observed_bindings = {name: _array_binding(value) for name, value in arrays.items()}
    if any(
        str(arrays[name].dtype) != dtype or list(arrays[name].shape) != expected_shapes[name]
        for name, dtype in _CACHE_ARRAY_DTYPES.items()
    ):
        return False
    return metadata == {
        "schema_version": _CACHE_ENTRY_SCHEMA_VERSION,
        "source_binding": source_binding,
        "arrays": observed_bindings,
    }


def _materialize_cache_item(task: _CacheMaterializationTask) -> None:
    """Materialize one view; module scope keeps Windows spawn picklable."""

    spec = task.spec
    path = task.cache_path_resolver(task.cache_root, spec)
    source_binding = _source_binding(task)
    main_cache_valid = False
    if path.is_file():
        main_cache_valid = _cache_entry_is_valid(path, task, source_binding)
    if main_cache_valid and task.residual_cache_root is None:
        return
    load_kwargs: dict[str, Any] = {"partial_point_count": task.partial_points}
    if task.mesh_loader is not None:
        load_kwargs["mesh_loader"] = task.mesh_loader
    prepared = load_dev_sample(
        task.dataset_root,
        task.manifest,
        spec,
        **load_kwargs,
    )
    observation = prepared.sample.observation
    intrinsics = np.asarray(
        (
            observation.intrinsics.fx,
            observation.intrinsics.fy,
            observation.intrinsics.cx,
            observation.intrinsics.cy,
        ),
        dtype=np.float64,
    )
    if task.residual_cache_root is not None:
        residual_path = _residual_cache_path(task.residual_cache_root, spec)
        if residual_path.exists():
            residual_cache = load_residual_label_cache(
                residual_path, expected_spec=task.residual_label_spec
            )
            validate_residual_label_cache_sources(
                residual_cache,
                surface=prepared.sample.surface,
                camera_T_mesh=observation.camera_T_mesh,
                depth_m=observation.depth_m,
                target_mask=observation.mask,
                intrinsics=intrinsics,
            )
        else:
            if task.residual_cache_read_only:
                raise RuntimeError(f"read-only residual cache entry is missing: {residual_path}")
            build_residual_label_cache(
                residual_path,
                surface=prepared.sample.surface,
                camera_T_mesh=observation.camera_T_mesh,
                depth_m=observation.depth_m,
                target_mask=observation.mask,
                intrinsics=intrinsics,
                spec=task.residual_label_spec,
            )
    if main_cache_valid:
        return
    shards = prepare_dev_queries(
        prepared,
        asset_ordinal=task.query_ordinal,
        seed=task.data_seed,
        train_near_surface_count=task.train_queries // 2,
        train_uniform_count=task.train_queries - task.train_queries // 2,
        evaluation_near_surface_count=task.evaluation_queries // 2,
        evaluation_uniform_count=task.evaluation_queries - task.evaluation_queries // 2,
        near_surface_sigma_m=0.003,
    )
    arrays = {
        "rgb": observation.rgb,
        "depth_m": observation.depth_m,
        "mask": observation.mask,
        "partial_points_camera_m": observation.partial_points_camera_m,
        "intrinsics": intrinsics.astype(np.float32),
        "train_points": shards.train.queries_camera.points.astype(np.float32),
        "train_udf_m": shards.train.values.udf_m.astype(np.float32),
        "train_near_surface_mask": shards.train.near_surface_mask,
        "evaluation_points": shards.evaluation.queries_camera.points.astype(np.float32),
        "evaluation_udf_m": shards.evaluation.values.udf_m.astype(np.float32),
        "evaluation_near_surface_mask": shards.evaluation.near_surface_mask,
    }
    metadata = {
        "schema_version": _CACHE_ENTRY_SCHEMA_VERSION,
        "source_binding": source_binding,
        "arrays": {name: _array_binding(value) for name, value in arrays.items()},
    }
    temporary = path.with_name(f".{path.name}.staging-{uuid4().hex}.npz")
    try:
        np.savez_compressed(
            temporary,
            **arrays,
            **{
                _CACHE_METADATA_MEMBER: np.asarray(
                    json.dumps(metadata, sort_keys=True, separators=(",", ":"))
                )
            },
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _materialize_cache(
    specs: tuple[DevViewSpec, ...],
    *,
    dataset_root: Path,
    cache_root: Path,
    data_seed: int,
    train_queries: int,
    evaluation_queries: int,
    partial_points: int,
    cache_path_resolver: Callable[[Path, DevViewSpec], Path] = _cache_path,
    asset_ordinal_by_id: dict[str, int] | None = None,
    mesh_loader: Callable[[Path], tuple[np.ndarray, np.ndarray, str]] | None = None,
    workers: int = 1,
    require_membership_labels: bool = False,
    residual_cache_root: Path | None = None,
    residual_label_spec: ResidualLabelSpec | None = None,
    residual_cache_read_only: bool = False,
) -> None:
    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        raise ValueError("cache workers must be a positive integer")
    if not isinstance(require_membership_labels, bool):
        raise TypeError("require_membership_labels must be a boolean")
    _ensure_cache_spec(
        cache_root,
        _cache_spec(
            specs,
            data_seed=data_seed,
            train_queries=train_queries,
            evaluation_queries=evaluation_queries,
            partial_points=partial_points,
            asset_ordinal_by_id=asset_ordinal_by_id,
        ),
    )
    frozen_residual_spec = (
        ResidualLabelSpec() if residual_label_spec is None else residual_label_spec
    )
    if residual_cache_root is not None:
        expected_residual_root = {
            "schema_version": "rapid-dev-residual-cache-root-v1",
            "residual_label_spec": frozen_residual_spec.to_mapping(),
            "cohort": json.loads(json.dumps([asdict(spec) for spec in specs], default=str)),
        }
        if residual_cache_read_only:
            _validate_cache_spec(residual_cache_root, expected_residual_root)
        else:
            _ensure_cache_spec(residual_cache_root, expected_residual_root)
    manifest = _manifest_path(dataset_root)
    tasks = tuple(
        _CacheMaterializationTask(
            spec=spec,
            dataset_root=dataset_root,
            manifest=manifest,
            cache_root=cache_root,
            data_seed=data_seed,
            train_queries=train_queries,
            evaluation_queries=evaluation_queries,
            partial_points=partial_points,
            cache_path_resolver=cache_path_resolver,
            query_ordinal=(
                ordinal if asset_ordinal_by_id is None else asset_ordinal_by_id[spec.asset.asset_id]
            ),
            mesh_loader=mesh_loader,
            residual_cache_root=residual_cache_root,
            residual_label_spec=frozen_residual_spec,
            residual_cache_read_only=residual_cache_read_only,
        )
        for ordinal, spec in enumerate(specs)
    )
    if workers == 1:
        for task in tasks:
            _materialize_cache_item(task)
        return
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=get_context("spawn"),
    ) as executor:
        tuple(executor.map(_materialize_cache_item, tasks))
