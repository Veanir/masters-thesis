"""Random scene sampling."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from masters_rgbd.synthetic.config import GeneratorConfig
from masters_rgbd.synthetic.schemas import ObjectAsset
from masters_rgbd.synthetic.transforms import look_at_camera, random_quat_wxyz


@dataclass(frozen=True)
class SampledObject:
    instance_id: int
    asset: ObjectAsset
    position: tuple[float, float, float]
    quat_wxyz: tuple[float, float, float, float]
    scale: tuple[float, float, float]

    @property
    def body_name(self) -> str:
        return f"object_{self.instance_id:03d}"

    @property
    def visual_geom_name(self) -> str:
        return f"{self.body_name}_visual"


@dataclass(frozen=True)
class SampledCamera:
    position: tuple[float, float, float]
    target: tuple[float, float, float]
    xyaxes: tuple[float, float, float, float, float, float]
    world_R_camera: tuple[
        tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]
    ]


@dataclass(frozen=True)
class SampledScene:
    scene_id: str
    objects: tuple[SampledObject, ...]
    camera: SampledCamera
    target_instance_id: int = 1


class SceneSampler:
    def __init__(self, config: GeneratorConfig, seed: int) -> None:
        self._config = config
        self._rng = np.random.default_rng(seed)

    def sample(self, scene_id: str, assets: list[ObjectAsset]) -> SampledScene:
        if not assets:
            raise ValueError("Cannot sample a scene without assets")

        sampling = self._config.sampling
        object_count = int(
            self._rng.integers(sampling.object_count_min, sampling.object_count_max + 1)
        )
        replace = object_count > len(assets)
        selected_indices = self._rng.choice(len(assets), size=object_count, replace=replace)
        xy_positions = sample_spawn_xy_positions(self._config, self._rng, object_count)
        objects = tuple(
            self._sample_object(
                instance_id=index + 1,
                asset=assets[int(asset_index)],
                stack_index=index,
                xy=xy_positions[index],
            )
            for index, asset_index in enumerate(selected_indices)
        )
        target_instance_id = int(self._rng.integers(1, object_count + 1))
        camera = self.sample_camera_for_objects(objects, target_instance_id=target_instance_id)
        return SampledScene(
            scene_id=scene_id,
            objects=objects,
            camera=camera,
            target_instance_id=target_instance_id,
        )

    def sample_camera_from_positions(self, positions: np.ndarray) -> SampledCamera:
        return sample_camera_from_positions(self._config, self._rng, positions)

    def sample_camera_for_objects(
        self,
        objects: tuple[SampledObject, ...],
        *,
        target_instance_id: int,
        positions: dict[int, np.ndarray] | None = None,
    ) -> SampledCamera:
        return sample_camera_for_objects(
            self._config,
            self._rng,
            objects,
            target_instance_id=target_instance_id,
            positions=positions,
        )

    def _sample_object(
        self,
        *,
        instance_id: int,
        asset: ObjectAsset,
        stack_index: int,
        xy: np.ndarray,
    ) -> SampledObject:
        sampling = self._config.sampling
        scale_value = float(self._rng.uniform(sampling.object_scale_min, sampling.object_scale_max))
        if (
            sampling.target_object_extent_min is not None
            and sampling.target_object_extent_max is not None
        ):
            max_extent = max(float(value) for value in asset.extents)
            if max_extent <= 0.0:
                raise ValueError(f"Object {asset.object_id} has invalid extents: {asset.extents}")
            target_extent = float(
                self._rng.uniform(
                    sampling.target_object_extent_min,
                    sampling.target_object_extent_max,
                )
            )
            scale_value *= target_extent / max_extent
        x = float(xy[0])
        y = float(xy[1])
        bbox_min_z = float(asset.bbox_min[2]) * scale_value
        z = (
            self._config.table.top_z
            - bbox_min_z
            + sampling.initial_drop_height
            + stack_index * sampling.per_object_stack_height
        )
        quat = random_quat_wxyz(self._rng)
        return SampledObject(
            instance_id=instance_id,
            asset=asset,
            position=(x, y, float(z)),
            quat_wxyz=tuple(float(value) for value in quat),  # type: ignore[arg-type]
            scale=(scale_value, scale_value, scale_value),
        )


def sample_spawn_xy_positions(
    config: GeneratorConfig,
    rng: np.random.Generator,
    object_count: int,
) -> np.ndarray:
    sampling = config.sampling
    if object_count <= 0:
        raise ValueError("object_count must be positive")

    xy_min = np.asarray(sampling.spawn_xy_min, dtype=np.float64)
    xy_max = np.asarray(sampling.spawn_xy_max, dtype=np.float64)
    positions: list[np.ndarray] = []
    max_attempts = max(200, object_count * 100)
    for _attempt in range(max_attempts):
        candidate = rng.uniform(xy_min, xy_max)
        if all(
            float(np.linalg.norm(candidate - existing)) >= sampling.spawn_xy_min_distance
            for existing in positions
        ):
            positions.append(candidate)
            if len(positions) == object_count:
                return np.asarray(positions, dtype=np.float64)

    # Avoid falling back to a collapsed pile when the requested object density is high.
    grid_side = int(np.ceil(np.sqrt(object_count)))
    xs = np.linspace(xy_min[0], xy_max[0], grid_side)
    ys = np.linspace(xy_min[1], xy_max[1], grid_side)
    grid = np.asarray([(x, y) for x in xs for y in ys], dtype=np.float64)
    rng.shuffle(grid)
    jitter_radius = min(0.015, 0.25 * sampling.spawn_xy_min_distance)
    jitter = rng.uniform(-jitter_radius, jitter_radius, size=(object_count, 2))
    return np.clip(grid[:object_count] + jitter, xy_min, xy_max)


def sample_camera_from_positions(
    config: GeneratorConfig,
    rng: np.random.Generator,
    positions: np.ndarray,
    *,
    focus_extent: float | None = None,
    margin: float | None = None,
) -> SampledCamera:
    sampling = config.sampling
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] == 0:
        raise ValueError(f"Expected non-empty positions with shape (N, 3), got {positions.shape}")

    mins = positions.min(axis=0)
    maxs = positions.max(axis=0)
    center = 0.5 * (mins + maxs)
    xy_radius = float(np.linalg.norm(maxs[:2] - mins[:2]) * 0.5)
    if focus_extent is None:
        object_padding = 0.1
        if sampling.target_object_extent_max is not None:
            object_padding = max(object_padding, 0.5 * float(sampling.target_object_extent_max))
    else:
        object_padding = max(0.04, 0.5 * float(focus_extent))
    camera_margin = sampling.camera_cluster_margin if margin is None else margin
    view_radius = max(0.08, xy_radius + object_padding) * camera_margin
    dynamic_radius = view_radius / math.tan(math.radians(config.render.fovy_degrees) * 0.5)
    radius = float(np.clip(dynamic_radius, sampling.camera_radius_min, sampling.camera_radius_max))
    if sampling.camera_distance_jitter_fraction > 0.0:
        radius *= float(
            rng.uniform(
                1.0 - sampling.camera_distance_jitter_fraction,
                1.0 + sampling.camera_distance_jitter_fraction,
            )
        )
        radius = float(np.clip(radius, sampling.camera_radius_min, sampling.camera_radius_max))

    azimuth = math.radians(
        float(
            rng.uniform(
                sampling.camera_azimuth_min_degrees,
                sampling.camera_azimuth_max_degrees,
            )
        )
    )
    elevation = math.radians(
        float(
            rng.uniform(
                sampling.camera_elevation_min_degrees,
                sampling.camera_elevation_max_degrees,
            )
        )
    )
    target = np.asarray(sampling.camera_target, dtype=np.float64)
    target[:2] = center[:2]
    target[2] = float(np.clip(center[2], config.table.top_z + 0.04, config.table.top_z + 0.35))
    jitter = rng.uniform(-1.0, 1.0, size=3) * np.asarray(
        sampling.camera_target_jitter,
        dtype=np.float64,
    )
    target = target + jitter
    position = target + np.array(
        [
            radius * math.cos(elevation) * math.cos(azimuth),
            radius * math.cos(elevation) * math.sin(azimuth),
            radius * math.sin(elevation),
        ],
        dtype=np.float64,
    )
    rotation, xyaxes = look_at_camera(
        position,
        target,
        np.array([0.0, 0.0, 1.0], dtype=np.float64),
    )
    return SampledCamera(
        position=tuple(float(value) for value in position),  # type: ignore[arg-type]
        target=tuple(float(value) for value in target),  # type: ignore[arg-type]
        xyaxes=tuple(float(value) for value in xyaxes),  # type: ignore[arg-type]
        world_R_camera=tuple(tuple(float(value) for value in row) for row in rotation),  # type: ignore[arg-type]
    )


def sample_camera_for_objects(
    config: GeneratorConfig,
    rng: np.random.Generator,
    objects: tuple[SampledObject, ...],
    *,
    target_instance_id: int,
    positions: dict[int, np.ndarray] | None = None,
) -> SampledCamera:
    if positions is None:
        positions = {
            item.instance_id: np.asarray(item.position, dtype=np.float64) for item in objects
        }

    if config.sampling.camera_focus == "scene":
        scene_positions = np.asarray(
            [positions[item.instance_id] for item in objects], dtype=np.float64
        )
        return sample_camera_from_positions(config, rng, scene_positions)

    target = next((item for item in objects if item.instance_id == target_instance_id), None)
    if target is None:
        raise ValueError(
            f"target_instance_id {target_instance_id} is not present in sampled objects"
        )

    target_position = np.asarray(positions[target_instance_id], dtype=np.float64).reshape(1, 3)
    target_extent = max(
        float(extent) * float(scale)
        for extent, scale in zip(target.asset.extents, target.scale, strict=True)
    )
    return sample_camera_from_positions(
        config,
        rng,
        target_position,
        focus_extent=target_extent,
        margin=config.sampling.camera_target_margin,
    )
