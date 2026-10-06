"""Versioned residual-surface label cache for B1-P512-RESIDUAL512-S0."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray

from masters_rgbd.b1.depth_visibility import (
    DEFAULT_DEPTH_MARGIN_M,
    DepthVisibilityLabel,
    classify_depth_visibility,
)
from masters_rgbd.contracts.geometry import AffineTransform, CoordinateFrame
from masters_rgbd.contracts.manifests import canonical_json_bytes
from masters_rgbd.geometry.ground_truth import TriangleSurface

Float64Array = NDArray[np.float64]

UInt8Array = NDArray[np.uint8]

RESIDUAL_LABEL_SCHEMA_VERSION = "b1-residual-label-cache-v1"

DEFAULT_SURFACE_POINT_COUNT = 65_536


@dataclass(frozen=True, slots=True)
class ResidualLabelCacheError(ValueError):
    """Residual-label inputs or an existing cache violate the frozen contract."""


@dataclass(frozen=True, slots=True)
class ResidualLabelSpec:
    schema_version: str = RESIDUAL_LABEL_SCHEMA_VERSION
    surface_sampling: str = "triangle-area-weighted-barycentric-v1"
    surface_sampling_seed: int = 0
    surface_point_count: int = DEFAULT_SURFACE_POINT_COUNT
    visibility_rule: str = "full-valid-3x3-max-positive-depth-v1"
    depth_margin_m: float = DEFAULT_DEPTH_MARGIN_M
    unknown_policy: str = "exclude-from-residual-supervision"
    knn_implementation: str = "scipy.spatial.cKDTree.query-eps0-v1"
    knn_k: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != RESIDUAL_LABEL_SCHEMA_VERSION:
            raise ResidualLabelCacheError("unsupported residual-label schema version")
        if self.surface_sampling != "triangle-area-weighted-barycentric-v1":
            raise ResidualLabelCacheError("unsupported residual surface sampler")
        if (
            isinstance(self.surface_sampling_seed, bool)
            or not isinstance(self.surface_sampling_seed, int)
            or self.surface_sampling_seed < 0
        ):
            raise ResidualLabelCacheError("surface sampling seed must be non-negative")
        if (
            isinstance(self.surface_point_count, bool)
            or not isinstance(self.surface_point_count, int)
            or self.surface_point_count <= 0
        ):
            raise ResidualLabelCacheError("surface point count must be positive")
        if not np.isfinite(self.depth_margin_m) or self.depth_margin_m <= 0.0:
            raise ResidualLabelCacheError("depth margin must be positive and finite")
        if self.visibility_rule != "full-valid-3x3-max-positive-depth-v1":
            raise ResidualLabelCacheError("unsupported residual visibility rule")
        if self.unknown_policy != "exclude-from-residual-supervision":
            raise ResidualLabelCacheError("unsupported residual UNKNOWN policy")
        if self.knn_implementation != "scipy.spatial.cKDTree.query-eps0-v1":
            raise ResidualLabelCacheError("unsupported residual kNN implementation")
        if self.knn_k != 1:
            raise ResidualLabelCacheError("residual distance contract currently requires kNN k=1")

    def to_mapping(self) -> dict[str, object]:
        return asdict(self)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.to_mapping())).hexdigest()


class ResidualLabelCache:
    root: Path
    spec: ResidualLabelSpec
    surface_points_camera_m: Float64Array
    visibility_labels: UInt8Array
    manifest: dict[str, object]

    @property
    def hidden_mask(self) -> NDArray[np.bool_]:
        return self.visibility_labels == int(DepthVisibilityLabel.HIDDEN_BEHIND_OBSERVATION)

    @property
    def unknown_mask(self) -> NDArray[np.bool_]:
        return self.visibility_labels == int(DepthVisibilityLabel.UNKNOWN)


def _array_sha256(value: NDArray[np.generic]) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(canonical_json_bytes(list(array.shape)))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _geometry_sha256(surface: TriangleSurface) -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "vertices_sha256": _array_sha256(surface.vertices_mesh_local_m),
                "faces_sha256": _array_sha256(surface.faces),
            }
        )
    ).hexdigest()


def _observation_sha256(
    *, depth_m: NDArray[np.float64], target_mask: NDArray[np.bool_], intrinsics: NDArray[np.float64]
) -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "depth_m_sha256": _array_sha256(depth_m),
                "target_mask_sha256": _array_sha256(target_mask),
                "intrinsics_sha256": _array_sha256(intrinsics),
            }
        )
    ).hexdigest()


def residual_observation_sha256(*, depth_m: object, target_mask: object, intrinsics: object) -> str:
    """Return the cache-compatible binding for an RGB-D observation."""

    return _observation_sha256(
        depth_m=np.asarray(depth_m, dtype=np.float64),
        target_mask=np.asarray(target_mask, dtype=np.bool_),
        intrinsics=np.asarray(intrinsics, dtype=np.float64),
    )


def validate_residual_label_cache_observation(
    cache: ResidualLabelCache,
    *,
    depth_m: object,
    target_mask: object,
    intrinsics: object,
) -> None:
    """Fail closed unless the cache belongs to this exact clean observation."""

    expected = residual_observation_sha256(
        depth_m=depth_m,
        target_mask=target_mask,
        intrinsics=intrinsics,
    )
    validate_residual_label_cache_observation_binding(cache, observation_sha256=expected)


def validate_residual_label_cache_observation_binding(
    cache: ResidualLabelCache, *, observation_sha256: str
) -> None:
    """Fail closed against an already captured clean-observation binding."""

    if len(observation_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in observation_sha256
    ):
        raise ResidualLabelCacheError("observation_sha256 must be a lowercase SHA-256 digest")
    bindings = cache.manifest.get("bindings")
    if not isinstance(bindings, dict) or bindings.get("observation_sha256") != observation_sha256:
        raise ResidualLabelCacheError("residual-label cache source bindings differ: observation")


def sample_complete_surface_points(
    surface: TriangleSurface, *, count: int, seed: int
) -> Float64Array:
    """Deterministically sample complete-mesh points proportional to triangle area."""

    spec = ResidualLabelSpec(surface_sampling_seed=seed, surface_point_count=count)
    del spec
    triangles = surface.triangles_mesh_local_m
    twice_areas = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
    )
    valid = twice_areas > 0.0
    if not np.any(valid):
        raise ResidualLabelCacheError("surface has no non-degenerate triangles")
    triangles = triangles[valid]
    probabilities = twice_areas[valid] / np.sum(twice_areas[valid])
    rng = np.random.default_rng(seed)
    face_indices = rng.choice(len(triangles), size=count, p=probabilities)
    selected = triangles[face_indices]
    root = np.sqrt(rng.random(count))
    second = rng.random(count)
    barycentric = np.column_stack((1.0 - root, root * (1.0 - second), root * second))
    result = np.sum(selected * barycentric[:, :, None], axis=1)
    result.setflags(write=False)
    return result


def build_residual_label_cache(
    root: Path,
    *,
    surface: TriangleSurface,
    camera_T_mesh: AffineTransform,
    depth_m: object,
    target_mask: object,
    intrinsics: object,
    spec: ResidualLabelSpec | None = None,
) -> ResidualLabelCache:
    """Materialize a new immutable residual-label cache and verify its readback."""

    frozen = ResidualLabelSpec() if spec is None else spec
    if not isinstance(frozen, ResidualLabelSpec):
        raise TypeError("spec must be ResidualLabelSpec")
    if (
        camera_T_mesh.source != CoordinateFrame.MESH_LOCAL
        or camera_T_mesh.target != CoordinateFrame.CAMERA
    ):
        raise ResidualLabelCacheError("camera_T_mesh must transform mesh-local points to camera")
    destination = Path(root)
    if destination.exists():
        raise ResidualLabelCacheError("residual-label cache destination already exists")
    depth = np.asarray(depth_m, dtype=np.float64)
    mask = np.asarray(target_mask, dtype=np.bool_)
    camera = np.asarray(intrinsics, dtype=np.float64)
    mesh_points = sample_complete_surface_points(
        surface,
        count=frozen.surface_point_count,
        seed=frozen.surface_sampling_seed,
    )
    camera_points = camera_T_mesh.transform_points(mesh_points)
    classification = classify_depth_visibility(
        camera_points,
        depth_m=depth,
        target_mask=mask,
        intrinsics=camera,
        depth_margin_m=frozen.depth_margin_m,
    )
    points = np.asarray(camera_points, dtype=np.float64)
    labels = np.asarray(classification.labels, dtype=np.uint8)
    counts = {
        "complete_surface_point_count": len(points),
        "hidden_residual_point_count": int(np.count_nonzero(classification.hidden_mask)),
        "supported_not_hidden_point_count": int(
            np.count_nonzero(labels == int(DepthVisibilityLabel.SUPPORTED_NOT_HIDDEN))
        ),
        "unknown_point_count": int(np.count_nonzero(classification.unknown_mask)),
    }
    manifest: dict[str, object] = {
        "schema_version": RESIDUAL_LABEL_SCHEMA_VERSION,
        "spec": frozen.to_mapping(),
        "spec_sha256": frozen.sha256,
        "bindings": {
            "geometry_sha256": _geometry_sha256(surface),
            "camera_T_mesh_sha256": _array_sha256(camera_T_mesh.matrix),
            "observation_sha256": _observation_sha256(
                depth_m=depth, target_mask=mask, intrinsics=camera
            ),
        },
        "arrays": {
            "surface_points_camera_m": {
                "dtype": "float64",
                "shape": [len(points), 3],
                "sha256": _array_sha256(points),
            },
            "visibility_labels": {
                "dtype": "uint8",
                "shape": [len(labels)],
                "sha256": _array_sha256(labels),
            },
        },
        "label_values": {label.name: int(label) for label in DepthVisibilityLabel},
        "counts": counts,
    }
    staging = destination.with_name(f".{destination.name}.staging-{uuid4().hex}")
    published = False
    try:
        staging.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(
            staging / "labels.npz",
            surface_points_camera_m=points,
            visibility_labels=labels,
        )
        (staging / "manifest.json").write_bytes(canonical_json_bytes(manifest))
        try:
            staging.replace(destination)
        except OSError:
            if not destination.exists():
                raise
            concurrent = load_residual_label_cache(destination, expected_spec=frozen)
            validate_residual_label_cache_sources(
                concurrent,
                surface=surface,
                camera_T_mesh=camera_T_mesh,
                depth_m=depth,
                target_mask=mask,
                intrinsics=camera,
            )
            shutil.rmtree(staging, ignore_errors=True)
            return concurrent
        published = True
        return load_residual_label_cache(destination, expected_spec=frozen)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        if published:
            shutil.rmtree(destination, ignore_errors=True)
        raise


def validate_residual_label_cache_sources(
    cache: ResidualLabelCache,
    *,
    surface: TriangleSurface,
    camera_T_mesh: AffineTransform,
    depth_m: object,
    target_mask: object,
    intrinsics: object,
) -> None:
    """Fail closed unless a loaded cache is bound to these exact source inputs."""

    depth = np.asarray(depth_m, dtype=np.float64)
    mask = np.asarray(target_mask, dtype=np.bool_)
    camera = np.asarray(intrinsics, dtype=np.float64)
    validate_residual_label_cache_observation(
        cache,
        depth_m=depth,
        target_mask=mask,
        intrinsics=camera,
    )
    expected = {
        "geometry_sha256": _geometry_sha256(surface),
        "camera_T_mesh_sha256": _array_sha256(camera_T_mesh.matrix),
        "observation_sha256": _observation_sha256(
            depth_m=depth, target_mask=mask, intrinsics=camera
        ),
    }
    if cache.manifest.get("bindings") != expected:
        raise ResidualLabelCacheError("residual-label cache source bindings differ")


def load_residual_label_cache(
    root: Path, *, expected_spec: ResidualLabelSpec | None = None
) -> ResidualLabelCache:
    """Load and fail closed on any schema, binding or array-hash mismatch."""

    cache_root = Path(root).resolve(strict=True)
    try:
        payload = (cache_root / "manifest.json").read_bytes()
        manifest = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ResidualLabelCacheError("cannot read residual-label manifest") from error
    if not isinstance(manifest, dict) or canonical_json_bytes(manifest) != payload:
        raise ResidualLabelCacheError("residual-label manifest is not canonical JSON")
    try:
        spec = ResidualLabelSpec(**manifest["spec"])
    except (KeyError, TypeError, ResidualLabelCacheError) as error:
        raise ResidualLabelCacheError("residual-label manifest has an invalid spec") from error
    if manifest.get("schema_version") != RESIDUAL_LABEL_SCHEMA_VERSION:
        raise ResidualLabelCacheError("residual-label manifest schema differs")
    if manifest.get("spec_sha256") != spec.sha256:
        raise ResidualLabelCacheError("residual-label spec hash differs")
    if expected_spec is not None and spec != expected_spec:
        raise ResidualLabelCacheError("residual-label cache does not match the expected spec")
    try:
        with np.load(cache_root / "labels.npz", allow_pickle=False) as archive:
            if set(archive.files) != {"surface_points_camera_m", "visibility_labels"}:
                raise ResidualLabelCacheError("residual-label archive members differ")
            points = np.asarray(archive["surface_points_camera_m"], dtype=np.float64)
            labels = np.asarray(archive["visibility_labels"], dtype=np.uint8)
    except ResidualLabelCacheError:
        raise
    except (OSError, ValueError, KeyError) as error:
        raise ResidualLabelCacheError("cannot read residual-label arrays") from error
    arrays = manifest.get("arrays")
    if not isinstance(arrays, dict):
        raise ResidualLabelCacheError("residual-label array bindings are missing")
    observed = {
        "surface_points_camera_m": {
            "dtype": "float64",
            "shape": list(points.shape),
            "sha256": _array_sha256(points),
        },
        "visibility_labels": {
            "dtype": "uint8",
            "shape": list(labels.shape),
            "sha256": _array_sha256(labels),
        },
    }
    if (
        arrays != observed
        or points.shape != (spec.surface_point_count, 3)
        or labels.shape != (spec.surface_point_count,)
    ):
        raise ResidualLabelCacheError("residual-label arrays do not match their bindings")
    valid_labels = {int(label) for label in DepthVisibilityLabel}
    if not set(int(value) for value in np.unique(labels)).issubset(valid_labels):
        raise ResidualLabelCacheError("residual-label cache contains unknown label values")
    expected_label_values = {label.name: int(label) for label in DepthVisibilityLabel}
    if manifest.get("label_values") != expected_label_values:
        raise ResidualLabelCacheError("residual-label manifest label values differ")
    expected_counts = {
        "complete_surface_point_count": len(points),
        "hidden_residual_point_count": int(
            np.count_nonzero(labels == int(DepthVisibilityLabel.HIDDEN_BEHIND_OBSERVATION))
        ),
        "supported_not_hidden_point_count": int(
            np.count_nonzero(labels == int(DepthVisibilityLabel.SUPPORTED_NOT_HIDDEN))
        ),
        "unknown_point_count": int(np.count_nonzero(labels == int(DepthVisibilityLabel.UNKNOWN))),
    }
    if manifest.get("counts") != expected_counts:
        raise ResidualLabelCacheError("residual-label manifest counts differ")
    bindings = manifest.get("bindings")
    if (
        not isinstance(bindings, dict)
        or set(bindings)
        != {
            "geometry_sha256",
            "camera_T_mesh_sha256",
            "observation_sha256",
        }
        or any(not isinstance(value, str) or len(value) != 64 for value in bindings.values())
    ):
        raise ResidualLabelCacheError("residual-label source bindings are invalid")
    points.setflags(write=False)
    labels.setflags(write=False)
    return ResidualLabelCache(cache_root, spec, points, labels, manifest)
