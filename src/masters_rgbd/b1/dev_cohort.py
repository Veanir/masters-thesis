"""Read-only Dev cohort adapter for the existing legacy RGB-D pool.

This module deliberately keeps the legacy inventory's diagnostic status.  It
uses its outcome-free geometry fingerprints to make a fast, geometry-disjoint
development cohort; it does not promote the legacy pool to thesis evidence.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from PIL import Image

from masters_rgbd.b1.data import (
    B1FixedViewSample,
    B1QueryShard,
    QuerySamplingConfig,
    RGBDObservation,
    _masked_pixels_uv,
    _uniform_visible_pixel_subset,
    sample_query_shard,
)
from masters_rgbd.contracts.geometry import CameraIntrinsics, ModelROI
from masters_rgbd.contracts.manifests import SplitName
from masters_rgbd.data.audit_rgbd import load_trimesh_geometry
from masters_rgbd.data.rgbd_adapter import adapt_legacy_v1_transform
from masters_rgbd.data.rgbd_inventory import (
    preprocess_legacy_diagnostic_geometry,
)
from masters_rgbd.geometry.ground_truth import DualMeshGroundTruth, TriangleSurface

MeshLoader = Callable[[Path], tuple[np.ndarray, np.ndarray, str]]

_DEFAULT_MANIFEST = PurePosixPath(
    "generated_sapien_trellis_train_5k/training_manifest_target.jsonl"
)

_DEFAULT_COUNTS = {
    SplitName.TRAIN: 48,
    SplitName.VALIDATION: 8,
    SplitName.TEST: 8,
}


class DevCohortError(ValueError):
    """The legacy inputs cannot support the requested read-only Dev cohort."""


@dataclass(frozen=True, slots=True)
class DevAssetSpec:
    asset_id: str
    category_id: str
    split: SplitName
    geometry_sha256: str
    triangle_count: int
    uniform_scale_m_per_source_unit: float


@dataclass(frozen=True, slots=True)
class DevViewSpec:
    asset: DevAssetSpec
    sample_id: str
    scene_id: str
    instance_id: int
    rgb_uri: str
    depth_uri: str
    mask_uri: str
    mesh_uri: str
    camera: dict[str, Any]
    legacy_camera_T_mesh: tuple[tuple[float, ...], ...]
    visible_pixel_count: int


@dataclass(frozen=True, slots=True)
class PreparedDevAsset:
    spec: DevViewSpec
    sample: B1FixedViewSample


@dataclass(frozen=True, slots=True)
class DevQueryShards:
    train: B1QueryShard
    evaluation: B1QueryShard


def prepare_dev_queries(
    prepared: PreparedDevAsset,
    *,
    asset_ordinal: int,
    seed: int,
    train_near_surface_count: int,
    train_uniform_count: int,
    evaluation_near_surface_count: int,
    evaluation_uniform_count: int,
    near_surface_sigma_m: float,
) -> DevQueryShards:
    """Create deterministic surface-valid shards without a solid voxel size."""

    if isinstance(asset_ordinal, bool) or not isinstance(asset_ordinal, int) or asset_ordinal < 0:
        raise DevCohortError("asset_ordinal must be a non-negative integer")
    train = sample_query_shard(
        prepared.sample,
        shard_index=2 * asset_ordinal,
        config=QuerySamplingConfig(
            near_surface_count=train_near_surface_count,
            uniform_count=train_uniform_count,
            base_seed=seed,
            near_surface_sigma_m=near_surface_sigma_m,
        ),
    )
    evaluation = sample_query_shard(
        prepared.sample,
        shard_index=2 * asset_ordinal + 1,
        config=QuerySamplingConfig(
            near_surface_count=evaluation_near_surface_count,
            uniform_count=evaluation_uniform_count,
            base_seed=seed,
            near_surface_sigma_m=near_surface_sigma_m,
        ),
    )
    return DevQueryShards(train=train, evaluation=evaluation)


def _safe_path(root: Path, uri: str, *, base: Path | None = None) -> Path:
    pure = PurePosixPath(uri.replace("\\", "/"))
    if pure.is_absolute():
        raise DevCohortError(f"absolute legacy URI is forbidden: {uri}")
    anchor = root if base is None else base
    path = (anchor / Path(*pure.parts)).resolve(strict=True)
    if path != root and root not in path.parents:
        raise DevCohortError(f"legacy URI escapes dataset root: {uri}")
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_dev_assets(
    inventory_path: Path,
    *,
    train_count: int = 48,
    validation_count: int = 8,
    test_count: int = 8,
) -> tuple[DevAssetSpec, ...]:
    """Choose the lightest unique geometries per frozen diagnostic split.

    Triangle count is an outcome-independent runtime proxy.  SHA-256 and asset
    ID provide stable tie breakers.  The default result is exactly 48/8/8.
    """

    counts = {
        SplitName.TRAIN: train_count,
        SplitName.VALIDATION: validation_count,
        SplitName.TEST: test_count,
    }
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in counts.values()
    ):
        raise DevCohortError("Dev split counts must be positive integers")
    try:
        raw = json.loads(Path(inventory_path).read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise DevCohortError(f"cannot read Dev inventory: {inventory_path}") from error
    if not isinstance(raw, dict) or not isinstance(raw.get("items"), list):
        raise DevCohortError("Dev inventory must contain an items array")
    if raw.get("rejections"):
        raise DevCohortError("Dev inventory contains rejected geometries")
    selected: list[DevAssetSpec] = []
    seen_geometry: set[str] = set()
    for split, required_count in counts.items():
        candidates: list[DevAssetSpec] = []
        for item in raw["items"]:
            try:
                audit = item["audit_candidate"]
                fingerprint = item["geometry_fingerprint"]
                if SplitName(str(audit["split"])) != split:
                    continue
                candidate = DevAssetSpec(
                    asset_id=str(audit["asset_id"]),
                    category_id=str(audit["category"]),
                    split=split,
                    geometry_sha256=str(fingerprint["sha256"]),
                    triangle_count=int(fingerprint["triangle_count"]),
                    uniform_scale_m_per_source_unit=float(item["uniform_scale_m_per_source_unit"]),
                )
            except (KeyError, TypeError, ValueError) as error:
                raise DevCohortError("Dev inventory contains a malformed item") from error
            if (
                len(candidate.geometry_sha256) != 64
                or candidate.triangle_count <= 0
                or not np.isfinite(candidate.uniform_scale_m_per_source_unit)
                or candidate.uniform_scale_m_per_source_unit <= 0.0
            ):
                raise DevCohortError(f"invalid Dev geometry record: {candidate.asset_id}")
            candidates.append(candidate)
        candidates.sort(
            key=lambda value: (
                value.triangle_count,
                value.geometry_sha256,
                value.asset_id,
            )
        )
        for candidate in candidates:
            if candidate.geometry_sha256 in seen_geometry:
                continue
            selected.append(candidate)
            seen_geometry.add(candidate.geometry_sha256)
            if sum(item.split == split for item in selected) == required_count:
                break
        if sum(item.split == split for item in selected) != required_count:
            raise DevCohortError(
                f"split {split.value} has fewer than {required_count} unique geometries"
            )
    return tuple(selected)


def _view_spec(asset: DevAssetSpec, row: dict[str, Any]) -> DevViewSpec:
    try:
        label = row["label"]
        scale = np.asarray(label["scale"], dtype=np.float64)
        if not np.allclose(
            scale,
            asset.uniform_scale_m_per_source_unit,
            atol=1e-12,
            rtol=1e-12,
        ):
            raise DevCohortError(f"view scale differs for {asset.asset_id}")
        return DevViewSpec(
            asset=asset,
            sample_id=str(row["sample_id"]),
            scene_id=str(row["scene_id"]),
            instance_id=int(row["instance_id"]),
            rgb_uri=str(row["paths"]["rgb"]),
            depth_uri=str(row["paths"]["depth"]),
            mask_uri=str(row["paths"]["instance_mask"]),
            mesh_uri=str(label["mesh"]),
            camera=dict(row["camera"]),
            legacy_camera_T_mesh=tuple(
                tuple(float(value) for value in values) for values in label["camera_T_mesh"]
            ),
            visible_pixel_count=int(row["visibility"]["visible_pixel_count"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        if isinstance(error, DevCohortError):
            raise
        raise DevCohortError(f"malformed selected view: {asset.asset_id}") from error


def select_top_views(
    dataset_root: Path,
    inventory_path: Path,
    assets: tuple[DevAssetSpec, ...],
    *,
    manifest_path: Path | None = None,
    max_views_per_asset: int = 1,
) -> dict[str, tuple[DevViewSpec, ...]]:
    """Select up to N most-visible target views per object, deterministically."""

    if (
        isinstance(max_views_per_asset, bool)
        or not isinstance(max_views_per_asset, int)
        or max_views_per_asset <= 0
    ):
        raise DevCohortError("max views per asset must be a positive integer")

    root = Path(dataset_root).resolve(strict=True)
    manifest = (
        _safe_path(root, _DEFAULT_MANIFEST.as_posix())
        if manifest_path is None
        else Path(manifest_path).resolve(strict=True)
    )
    if manifest != root and root not in manifest.parents:
        raise DevCohortError("training manifest must stay inside the legacy dataset root")
    try:
        inventory = json.loads(Path(inventory_path).read_bytes())
        expected_manifest_sha256 = str(inventory["source_manifest"]["manifest_sha256"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise DevCohortError("inventory has no source-manifest binding") from error
    if _sha256_file(manifest) != expected_manifest_sha256:
        raise DevCohortError("legacy target manifest differs from the inventory binding")

    by_id = {asset.asset_id: asset for asset in assets}
    if len(by_id) != len(assets):
        raise DevCohortError("selected Dev asset IDs must be unique")
    candidates: dict[str, list[dict[str, Any]]] = {asset_id: [] for asset_id in by_id}
    seen_sample_ids: set[str] = set()
    try:
        with manifest.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                row = json.loads(line)
                asset_id = row.get("object_id")
                if asset_id not in by_id:
                    continue
                if row.get("is_target") is not True:
                    raise DevCohortError(f"line {line_number} is not a target sample")
                int(row["visibility"]["visible_pixel_count"])
                sample_id = str(row["sample_id"])
                if sample_id in seen_sample_ids:
                    raise DevCohortError(f"duplicate target sample ID: {sample_id}")
                seen_sample_ids.add(sample_id)
                candidates[asset_id].append(row)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        if isinstance(error, DevCohortError):
            raise
        raise DevCohortError("legacy target manifest contains a malformed row") from error

    missing = sorted(asset_id for asset_id, rows in candidates.items() if not rows)
    if missing:
        raise DevCohortError(f"selected objects have no target view: {missing[:3]}")
    result: dict[str, tuple[DevViewSpec, ...]] = {}
    for asset in assets:
        ranked = sorted(
            candidates[asset.asset_id],
            key=lambda row: (
                -int(row["visibility"]["visible_pixel_count"]),
                str(row["sample_id"]),
            ),
        )[:max_views_per_asset]
        result[asset.asset_id] = tuple(_view_spec(asset, row) for row in ranked)
    return result


def select_best_views(
    dataset_root: Path,
    inventory_path: Path,
    assets: tuple[DevAssetSpec, ...],
    *,
    manifest_path: Path | None = None,
) -> tuple[DevViewSpec, ...]:
    """Select the most visible target view per object without writing legacy data."""

    selected = select_top_views(
        dataset_root,
        inventory_path,
        assets,
        manifest_path=manifest_path,
        max_views_per_asset=1,
    )
    return tuple(selected[asset.asset_id][0] for asset in assets)


def load_dev_sample(
    dataset_root: Path,
    manifest_path: Path,
    spec: DevViewSpec,
    *,
    partial_point_count: int | None = 512,
    minimum_roi_half_extent_m: float = 0.001,
    roi_padding_fraction: float = 0.25,
    mesh_loader: MeshLoader = load_trimesh_geometry,
) -> PreparedDevAsset:
    """Load one selected legacy view as surface-valid, occupancy-invalid B1 data."""

    root = Path(dataset_root).resolve(strict=True)
    manifest_parent = Path(manifest_path).resolve(strict=True).parent
    mesh_path = _safe_path(root, spec.mesh_uri, base=manifest_parent)
    vertices, faces, _ = mesh_loader(mesh_path)
    processed = preprocess_legacy_diagnostic_geometry(
        vertices=vertices,
        faces=faces,
        uniform_scale_m_per_source_unit=spec.asset.uniform_scale_m_per_source_unit,
    )
    if processed.fingerprint.sha256 != spec.asset.geometry_sha256:
        raise DevCohortError(f"processed geometry differs for {spec.asset.asset_id}")
    adapted = adapt_legacy_v1_transform(
        vertices,
        spec.legacy_camera_T_mesh,
        expected_uniform_scale_m_per_legacy_unit=(spec.asset.uniform_scale_m_per_source_unit),
    )
    surface = TriangleSurface(processed.vertices_mesh_local_m, processed.faces)

    rgb_path = _safe_path(root, spec.rgb_uri, base=manifest_parent)
    depth_path = _safe_path(root, spec.depth_uri, base=manifest_parent)
    mask_path = _safe_path(root, spec.mask_uri, base=manifest_parent)
    try:
        with Image.open(rgb_path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / np.float32(255.0)
        with Image.open(mask_path) as image:
            instance_mask = np.asarray(image)
        depth = np.load(depth_path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise DevCohortError(f"cannot decode selected RGB-D view: {spec.sample_id}") from error
    try:
        camera = spec.camera
        intrinsics_raw = camera["intrinsics"]
        intrinsics = CameraIntrinsics(
            width=int(camera["width"]),
            height=int(camera["height"]),
            fx=float(intrinsics_raw["fx"]),
            fy=float(intrinsics_raw["fy"]),
            cx=float(intrinsics_raw["cx"]),
            cy=float(intrinsics_raw["cy"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise DevCohortError(f"invalid camera record: {spec.sample_id}") from error
    if rgb.shape[:2] != (intrinsics.height, intrinsics.width):
        raise DevCohortError(f"RGB shape differs from camera: {spec.sample_id}")
    if depth.shape != (intrinsics.height, intrinsics.width):
        raise DevCohortError(f"depth shape differs from camera: {spec.sample_id}")
    if instance_mask.shape != (intrinsics.height, intrinsics.width):
        raise DevCohortError(f"mask shape differs from camera: {spec.sample_id}")
    target_mask = np.asarray(instance_mask == spec.instance_id, dtype=np.bool_)
    rgb = np.asarray(rgb * target_mask[..., None], dtype=np.float32)
    depth = np.asarray(depth, dtype=np.float32)
    depth = np.where(target_mask, depth, np.float32(0.0)).astype(np.float32, copy=False)
    pixels = _uniform_visible_pixel_subset(
        _masked_pixels_uv(target_mask),
        requested_count=partial_point_count,
    )
    columns = pixels[:, 0].astype(np.int64)
    rows = pixels[:, 1].astype(np.int64)
    points = intrinsics.unproject(pixels, depth[rows, columns])
    roi = ModelROI.from_visible_points(
        points,
        minimum_half_extent_m=minimum_roi_half_extent_m,
        padding_fraction=roi_padding_fraction,
    )
    observation = RGBDObservation(
        rgb=rgb,
        depth_m=depth,
        mask=target_mask,
        intrinsics=intrinsics,
        partial_pixels_uv=pixels,
        partial_points_camera_m=points,
        model_roi=roi,
        camera_T_mesh=adapted.camera_T_mesh,
    )
    ground_truth = DualMeshGroundTruth(
        surface=surface,
        solid=None,
        camera_T_mesh=adapted.camera_T_mesh,
        occupancy_valid=False,
    )
    return PreparedDevAsset(
        spec=spec,
        sample=B1FixedViewSample(observation, surface, None, ground_truth),
    )


def resolve_dev_cohort(
    dataset_root: Path,
    inventory_path: Path,
    *,
    manifest_path: Path | None = None,
    train_count: int = _DEFAULT_COUNTS[SplitName.TRAIN],
    validation_count: int = _DEFAULT_COUNTS[SplitName.VALIDATION],
    test_count: int = _DEFAULT_COUNTS[SplitName.TEST],
) -> tuple[DevViewSpec, ...]:
    """Resolve the default 64-object cohort without materializing heavy meshes."""

    root = Path(dataset_root).resolve(strict=True)
    manifest = (
        _safe_path(root, _DEFAULT_MANIFEST.as_posix())
        if manifest_path is None
        else Path(manifest_path).resolve(strict=True)
    )
    assets = select_dev_assets(
        inventory_path,
        train_count=train_count,
        validation_count=validation_count,
        test_count=test_count,
    )
    return select_best_views(root, inventory_path, assets, manifest_path=manifest)
