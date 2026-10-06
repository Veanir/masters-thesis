"""Fast geometry-disjoint B0/B1 development comparison on legacy RGB-D views."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from masters_rgbd.b1.dev_cohort import (
    DevViewSpec,
)
from masters_rgbd.b1.model import B1FieldModel

_FREE_SPACE_SURFACE_THRESHOLD_M = 0.005


def _git_sha(workspace: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=workspace, text=True).strip()


def _manifest_path(dataset_root: Path) -> Path:
    return dataset_root / "generated_sapien_trellis_train_5k" / "training_manifest_target.jsonl"


def _cache_path(cache_root: Path, spec: DevViewSpec) -> Path:
    return cache_root / f"{spec.asset.split.value}-{spec.asset.asset_id}.npz"


def _cache_spec(
    specs: tuple[DevViewSpec, ...],
    *,
    data_seed: int,
    train_queries: int,
    evaluation_queries: int,
    partial_points: int,
    asset_ordinal_by_id: dict[str, int] | None = None,
) -> dict[str, Any]:
    cohort = json.loads(json.dumps([asdict(spec) for spec in specs], default=str))
    result = {
        "schema_version": "rapid-dev-cache-v1",
        "data_seed": data_seed,
        "train_queries": train_queries,
        "evaluation_queries": evaluation_queries,
        "partial_points": partial_points,
        "cohort": cohort,
    }
    if asset_ordinal_by_id is not None:
        result["query_ordinal_by_asset"] = dict(sorted(asset_ordinal_by_id.items()))
        result["query_ordinal_policy"] = "shared-asset-ordinal-v1"
    return result


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _observation(data: Any, device: torch.device) -> tuple[torch.Tensor, ...]:
    return (
        torch.from_numpy(data["rgb"].transpose(2, 0, 1).copy())[None].to(device),
        torch.from_numpy(data["depth_m"].copy())[None, None].to(device),
        torch.from_numpy(data["mask"].astype(np.float32))[None, None].to(device),
        torch.from_numpy(data["partial_points_camera_m"].copy())[None].to(device),
        torch.from_numpy(data["intrinsics"].copy())[None].to(device),
    )


def _depth_visibility_masks(
    points_camera_m: np.ndarray,
    target_udf_m: np.ndarray,
    depth_m: np.ndarray,
    mask: np.ndarray,
    intrinsics: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Classify exact-GT queries using conservative target-depth visibility."""

    points = np.asarray(points_camera_m, dtype=np.float64)
    target = np.asarray(target_udf_m, dtype=np.float64)
    depth = np.asarray(depth_m, dtype=np.float64)
    target_mask = np.asarray(mask, dtype=np.bool_)
    camera = np.asarray(intrinsics, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or target.shape != (len(points),):
        raise ValueError("query points and target UDF have incompatible shapes")
    if depth.ndim != 2 or target_mask.shape != depth.shape or camera.shape != (4,):
        raise ValueError("depth, mask, or intrinsics have incompatible shapes")
    if not all(np.isfinite(value).all() for value in (points, target, depth, camera)):
        raise ValueError("visibility inputs must be finite")

    fx, fy, cx, cy = camera
    query_depth = -points[:, 2]
    safe_depth = np.maximum(query_depth, np.finfo(np.float64).eps)
    columns = np.floor(cx + fx * points[:, 0] / safe_depth + 0.5).astype(np.int64)
    rows = np.floor(cy - fy * points[:, 1] / safe_depth + 0.5).astype(np.int64)
    height, width = depth.shape
    spatial = (
        (query_depth > 0.0)
        & (rows >= 1)
        & (rows < height - 1)
        & (columns >= 1)
        & (columns < width - 1)
    )
    valid = np.zeros(len(points), dtype=np.bool_)
    observed_depth = np.full(len(points), np.inf, dtype=np.float64)
    spatial_indices = np.flatnonzero(spatial)
    if len(spatial_indices):
        local_valid = np.ones(len(spatial_indices), dtype=np.bool_)
        local_minimum = np.full(len(spatial_indices), np.inf, dtype=np.float64)
        local_rows = rows[spatial_indices]
        local_columns = columns[spatial_indices]
        for row_offset in (-1, 0, 1):
            for column_offset in (-1, 0, 1):
                local_depth = depth[local_rows + row_offset, local_columns + column_offset]
                local_mask = target_mask[local_rows + row_offset, local_columns + column_offset]
                local_valid &= local_mask & (local_depth > 0.0)
                local_minimum = np.minimum(local_minimum, local_depth)
        valid[spatial_indices] = local_valid
        observed_depth[spatial_indices] = local_minimum

    threshold = _FREE_SPACE_SURFACE_THRESHOLD_M
    certain_free = valid & (query_depth < observed_depth - threshold) & (target > threshold)
    target_surface = target <= threshold
    hidden_surface = target_surface & (~valid | (query_depth > observed_depth + threshold))
    return certain_free, hidden_surface


def _training_query_indices(
    points_camera_m: np.ndarray,
    target_udf_m: np.ndarray,
    depth_m: np.ndarray,
    mask: np.ndarray,
    intrinsics: np.ndarray,
    *,
    total_count: int,
    certain_free_count: int,
    rng: np.random.Generator,
    hidden_surface_count: int = 0,
    near_surface_count: int | None = None,
    near_surface_mask: np.ndarray | None = None,
) -> np.ndarray:
    if not 0 <= certain_free_count < total_count <= len(points_camera_m):
        raise ValueError("training query counts are incompatible with the cached query pool")
    if not 0 <= hidden_surface_count < total_count:
        raise ValueError("hidden-surface query count must be in [0, total_count)")
    if certain_free_count and hidden_surface_count:
        raise ValueError("certain-free and hidden-surface sampling cannot be mixed")
    if hidden_surface_count:
        if (
            near_surface_count is None
            or not hidden_surface_count < near_surface_count < total_count
        ):
            raise ValueError("hidden sampling requires hidden < near-surface < total counts")
        if near_surface_mask is None:
            raise ValueError("hidden sampling requires the cached near-surface mask")
        near_mask = np.asarray(near_surface_mask)
        if near_mask.dtype != np.bool_ or near_mask.shape != (len(points_camera_m),):
            raise ValueError("cached near-surface mask must be a matching boolean vector")
        _, hidden = _depth_visibility_masks(
            points_camera_m, target_udf_m, depth_m, mask, intrinsics
        )
        hidden_near_indices = np.flatnonzero(hidden & near_mask)
        general_near_indices = np.flatnonzero((~hidden) & near_mask)
        uniform_indices = np.flatnonzero(~near_mask)
        general_near_count = near_surface_count - hidden_surface_count
        uniform_count = total_count - near_surface_count
        if (
            len(hidden_near_indices) < hidden_surface_count
            or len(general_near_indices) < general_near_count
            or len(uniform_indices) < uniform_count
        ):
            raise ValueError("cached query pool cannot supply the declared hidden-surface mix")
        selected = np.concatenate(
            (
                rng.choice(hidden_near_indices, size=hidden_surface_count, replace=False),
                rng.choice(general_near_indices, size=general_near_count, replace=False),
                rng.choice(uniform_indices, size=uniform_count, replace=False),
            )
        )
        return selected[rng.permutation(len(selected))]
    if certain_free_count == 0:
        return rng.choice(len(points_camera_m), size=total_count, replace=False)
    certain_free, _ = _depth_visibility_masks(
        points_camera_m, target_udf_m, depth_m, mask, intrinsics
    )
    free_indices = np.flatnonzero(certain_free)
    general_indices = np.flatnonzero(~certain_free)
    general_count = total_count - certain_free_count
    if len(free_indices) == 0 or len(general_indices) < general_count:
        raise ValueError("cached query pool cannot supply the declared certain-free mix")
    selected_general = rng.choice(general_indices, size=general_count, replace=False)
    selected_free = rng.choice(
        free_indices,
        size=certain_free_count,
        replace=certain_free_count > len(free_indices),
    )
    combined = np.concatenate((selected_general, selected_free))
    return combined[rng.permutation(len(combined))]


def _predict_udf(
    model: B1FieldModel,
    data: Any,
    points: np.ndarray,
    device: torch.device,
    *,
    chunk_size: int = 4096,
) -> np.ndarray:
    rgb, depth, mask, partial, intrinsics = _observation(data, device)
    with torch.no_grad():
        encoding = model.encode_observation(rgb, depth, mask, partial)
        predictions = []
        for start in range(0, len(points), chunk_size):
            query = torch.from_numpy(points[start : start + chunk_size].copy())[None].to(device)
            predictions.append(
                model.decode_queries(encoding, query, intrinsics).udf_m[0].float().cpu().numpy()
            )
    return np.concatenate(predictions)
