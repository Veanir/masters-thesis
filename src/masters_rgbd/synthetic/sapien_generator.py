"""SAPIEN-backed v2 spike generator.

This backend keeps a single SAPIEN scene alive, loads prepared object meshes once,
and rapidly reuses the actors across generated scenes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from masters_rgbd.synthetic.assets import load_prepared_assets
from masters_rgbd.synthetic.backgrounds import (
    MeshBackgroundAsset,
    SurfaceAsset,
    load_mesh_background_assets,
    load_surface_assets,
)
from masters_rgbd.synthetic.config import GeneratorConfig, ValidationConfig
from masters_rgbd.synthetic.errors import AssetError, DependencyMissingError, GenerationError
from masters_rgbd.synthetic.exporter import write_rendered_scene
from masters_rgbd.synthetic.io_utils import ensure_clean_dir, make_relative_path, remove_if_exists
from masters_rgbd.synthetic.mask_stats import compute_instance_mask_stats
from masters_rgbd.synthetic.renderer import RenderedScene
from masters_rgbd.synthetic.scene_sampler import (
    SampledCamera,
    sample_camera_for_objects,
    sample_spawn_xy_positions,
)
from masters_rgbd.synthetic.schemas import (
    CameraMetadata,
    ObjectAsset,
    SceneMetadata,
    SceneObjectMetadata,
)
from masters_rgbd.synthetic.transforms import (
    invert_transform,
    make_transform,
    matrix_to_json,
    matrix_to_quat_wxyz,
    transform_from_pos_quat,
)
from masters_rgbd.synthetic.validate import summarize_validation_report, validate_scene

if TYPE_CHECKING:
    from masters_rgbd.synthetic.generator import GenerationResult

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class _ObjectSlot:
    asset: ObjectAsset
    entity: object
    render_body: object
    rigid_body: object
    render_material: object | None
    scale: tuple[float, float, float]


@dataclass(frozen=True)
class _ActiveObject:
    instance_id: int
    slot: _ObjectSlot
    initial_position: tuple[float, float, float]
    initial_quat_wxyz: tuple[float, float, float, float]


@dataclass(frozen=True)
class _BackgroundMeshSlot:
    asset: MeshBackgroundAsset
    entity: object
    render_body: object
    scale: tuple[float, float, float]
    rotation: tuple[float, float, float, float]
    position: tuple[float, float, float]


@dataclass(frozen=True)
class _SapienContactQuality:
    contact_count: int
    min_separation_m: float
    max_penetration_m: float

    def to_json(self) -> dict[str, float | int]:
        return {
            "contact_count": int(self.contact_count),
            "min_separation_m": float(self.min_separation_m),
            "max_penetration_m": float(self.max_penetration_m),
        }


def generate_dataset_sapien(
    dataset_root: Path,
    *,
    scene_count: int,
    seed: int,
    config: GeneratorConfig | None = None,
    overwrite: bool = False,
    asset_limit: int | None = None,
) -> GenerationResult:
    """Generate scenes with the SAPIEN v2 spike backend."""

    from masters_rgbd.synthetic.generator import GenerationResult

    if scene_count <= 0:
        raise ValueError("scene_count must be positive")

    config = config or GeneratorConfig()
    objects_dir = dataset_root / "objects"
    scenes_dir = dataset_root / "scenes"
    assets = load_prepared_assets(objects_dir, limit=asset_limit)
    if overwrite:
        remove_if_exists(scenes_dir)
    scenes_dir.mkdir(parents=True, exist_ok=True)

    if config.sampling.object_count_max > len(assets):
        raise GenerationError(
            "SAPIEN v2 spike uses a preloaded actor pool and currently requires "
            f"object_count_max <= loaded assets ({config.sampling.object_count_max} > "
            f"{len(assets)}). "
            "Increase --asset-limit or lower --max-objects."
        )

    rng = np.random.default_rng(seed)
    generated: list[Path] = []
    scene_index = 0
    with _SapienScenePool(dataset_root=dataset_root, assets=assets, config=config, rng=rng) as pool:
        while len(generated) < scene_count:
            scene_id = f"scene_{scene_index:06d}"
            scene_index += 1
            scene_dir = scenes_dir / scene_id
            if scene_dir.exists() and not overwrite:
                raise FileExistsError(f"Scene directory already exists: {scene_dir}")

            summary = "no validation attempted"
            for attempt in range(1, config.validation.max_attempts_per_scene + 1):
                remove_if_exists(scene_dir)
                ensure_clean_dir(scene_dir, overwrite=False)
                try:
                    pool.generate_one_scene(scene_dir=scene_dir, scene_id=scene_id)
                    report = validate_scene(scene_dir, dataset_root, config.validation)
                    if report.ok:
                        generated.append(scene_dir)
                        break
                    summary = summarize_validation_report(report, dataset_root)
                except Exception:
                    remove_if_exists(scene_dir)
                    raise
                remove_if_exists(scene_dir)
                if attempt == config.validation.max_attempts_per_scene:
                    raise GenerationError(
                        f"Could not generate a valid SAPIEN scene {scene_id} after "
                        f"{attempt} attempts. "
                        f"Last validation failure: {summary}"
                    )

    return GenerationResult(scene_dirs=tuple(generated))


class _SapienScenePool:
    def __init__(
        self,
        *,
        dataset_root: Path,
        assets: list[ObjectAsset],
        config: GeneratorConfig,
        rng: np.random.Generator,
    ) -> None:
        try:
            import sapien
        except ImportError as exc:  # pragma: no cover - depends on optional extra.
            raise DependencyMissingError(
                "sapien is required for the v2 backend. Run with "
                "`uv run --with sapien generate-dataset-v2-sapien ...` or install "
                "the `sapien` extra."
            ) from exc

        self._sapien = sapien
        self._dataset_root = dataset_root
        self._assets = assets
        self._config = config
        self._rng = rng
        self._texture_cache: dict[Path, object] = {}
        self._environment_map_path = _resolve_render_asset_path(
            config.render.environment_map,
            dataset_root=dataset_root,
        )
        self._mesh_background_assets = self._load_mesh_background_assets()
        self._surface_assets = self._load_surface_assets()
        self._active_surface: SurfaceAsset | None = None
        self._active_mesh_background: _BackgroundMeshSlot | None = None
        self._ground_material: object | None = None
        self._scene = self._build_scene()
        self._background_mesh_slots = self._build_background_mesh_slots()
        self._slots = self._build_object_slots()
        self._robot_entity = (
            self._build_robot_camera_mount() if config.robot_camera.visible else None
        )
        self._camera = self._scene.add_camera(
            config.render.camera_name,
            config.render.width,
            config.render.height,
            math.radians(config.render.fovy_degrees),
            config.render.znear,
            config.render.zfar,
        )

    def _load_mesh_background_assets(self) -> list[MeshBackgroundAsset]:
        background = self._config.background
        if not background.enabled:
            return []
        background_dir = self._resolve_background_dir()
        try:
            return load_mesh_background_assets(background_dir, background_id=background.surface_id)
        except AssetError:
            if background.surface_id is not None:
                return []
            raise

    def _load_surface_assets(self) -> list[SurfaceAsset]:
        background = self._config.background
        if not background.enabled or self._mesh_background_assets:
            return []
        background_dir = self._resolve_background_dir()
        return load_surface_assets(background_dir, surface_id=background.surface_id)

    def _resolve_background_dir(self) -> Path:
        candidate = Path(self._config.background.directory)
        if candidate.is_absolute():
            return candidate
        for base in (self._dataset_root, Path.cwd(), _PROJECT_ROOT):
            resolved = (base / candidate).resolve()
            if resolved.is_dir():
                return resolved
        return (self._dataset_root / candidate).resolve()

    def __enter__(self) -> _SapienScenePool:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        del exc_type, exc_value, traceback
        self.close()

    def close(self) -> None:
        self._scene.clear()

    def generate_one_scene(self, *, scene_dir: Path, scene_id: str) -> None:
        active_objects = self._sample_active_objects()
        self._reset_scene(active_objects)
        for _ in range(self._config.physics.settle_steps):
            self._scene.step()

        positions = {item.instance_id: self._visual_focus_position(item) for item in active_objects}
        final_camera = self._sample_camera(active_objects, positions=positions)
        self._set_camera_pose(final_camera)
        rendered = self._render_with_camera(active_objects, final_camera)
        visible_target_id = self._select_visible_target(rendered, active_objects)
        if visible_target_id is not None and self._should_refocus(rendered, visible_target_id):
            final_camera = self._sample_camera(
                active_objects,
                positions=positions,
                target_instance_id=visible_target_id,
            )
            self._set_camera_pose(final_camera)
            rendered = self._render_with_camera(active_objects, final_camera)
            visible_target_id = self._select_visible_target(rendered, active_objects)
        if visible_target_id is not None:
            self._target_instance_id = visible_target_id
        camera_metadata = self._build_camera_metadata(final_camera)
        scene_metadata = self._build_scene_metadata(
            scene_id=scene_id,
            rendered=rendered,
            active_objects=active_objects,
            final_camera=final_camera,
            contact_quality=self._measure_contact_quality(),
        )
        write_rendered_scene(scene_dir, rendered, camera_metadata, scene_metadata)

    def _build_scene(self) -> object:
        scene = self._sapien.Scene()
        scene.set_timestep(self._config.physics.timestep)
        self._ground_material = self._sapien.render.RenderMaterial(
            base_color=list(self._config.table.rgba),
            roughness=0.88,
            metallic=0.0,
            specular=0.12,
        )
        scene.add_ground(
            altitude=self._config.table.top_z,
            render=not self._mesh_background_assets,
            render_material=self._ground_material,
            render_half_size=[
                self._config.table.size[0] * 0.5,
                self._config.table.size[1] * 0.5,
            ],
        )
        scene.set_ambient_light(
            [0.12, 0.12, 0.12] if self._environment_map_path else [0.28, 0.28, 0.28]
        )
        if self._environment_map_path is not None:
            scene.set_environment_map(str(self._environment_map_path))
        scene.add_directional_light(
            [0.2, 0.4, -1.0],
            [0.55, 0.53, 0.50],
            shadow=self._config.render.enable_shadows,
            position=[0.0, 0.0, 1.2],
            shadow_scale=1.2,
            shadow_near=-0.2,
            shadow_far=2.0,
            shadow_map_size=min(8192, self._config.render.shadow_map_size),
        )
        scene.add_point_light([0.15, -0.35, 0.65], [0.10, 0.10, 0.10])
        return scene

    def _apply_random_surface(self) -> None:
        self._active_mesh_background = None
        for slot in self._background_mesh_slots:
            slot.render_body.disable()
            slot.entity.set_pose(self._sapien.Pose([0.0, 0.0, -5.0]))
        if self._background_mesh_slots:
            slot_index = int(self._rng.integers(0, len(self._background_mesh_slots)))
            slot = self._background_mesh_slots[slot_index]
            slot.render_body.enable()
            slot.entity.set_pose(self._sapien.Pose(list(slot.position), list(slot.rotation)))
            self._active_mesh_background = slot
            self._active_surface = None
            return

        if not self._surface_assets:
            self._active_surface = None
            return
        surface_index = int(self._rng.integers(0, len(self._surface_assets)))
        self._active_surface = self._surface_assets[surface_index]
        self._apply_ground_surface(self._active_surface)

    def _apply_ground_surface(self, surface: SurfaceAsset) -> None:
        if self._ground_material is None:
            return
        material = self._ground_material
        material.set_base_color(list(surface.base_color))
        material.set_roughness(surface.roughness)
        material.set_metallic(surface.metallic)
        material.set_specular(surface.specular)
        material.set_base_color_texture(self._load_texture(surface.base_color_file, srgb=True))
        material.set_roughness_texture(self._load_texture(surface.roughness_file, srgb=False))
        material.set_metallic_texture(self._load_texture(surface.metallic_file, srgb=False))
        material.set_normal_texture(self._load_texture(surface.normal_file, srgb=False))

    def _build_background_mesh_slots(self) -> list[_BackgroundMeshSlot]:
        slots: list[_BackgroundMeshSlot] = []
        for asset_index, asset in enumerate(self._mesh_background_assets):
            scale, rotation, position = _background_mesh_transform(
                asset,
                visual_size_m=self._config.background.visual_size_m,
                top_z=self._config.table.top_z,
                z_offset_m=self._config.background.visual_z_offset_m,
            )
            builder = self._scene.create_actor_builder()
            builder.set_name(f"background_{asset_index:03d}_{asset.background_id}")
            builder.add_visual_from_file(
                str(asset.mesh_file),
                scale=scale,
                material=None,
                name=f"background_visual_{asset_index:03d}",
            )
            entity = builder.build_static(
                name=f"background_{asset_index:03d}_{asset.background_id}"
            )
            render_body = entity.find_component_by_type(self._sapien.render.RenderBodyComponent)
            if render_body is None:
                raise GenerationError(
                    f"Background mesh {asset.background_id!r} is missing render component"
                )
            slot = _BackgroundMeshSlot(
                asset=asset,
                entity=entity,
                render_body=render_body,
                scale=scale,
                rotation=rotation,
                position=position,
            )
            render_body.disable()
            entity.set_pose(self._sapien.Pose([0.0, 0.0, -5.0]))
            slots.append(slot)
        return slots

    def _build_object_slots(self) -> list[_ObjectSlot]:
        slots: list[_ObjectSlot] = []
        material = self._scene.create_physical_material(
            self._config.physics.friction[0],
            self._config.physics.friction[0],
            0.0,
        )
        for asset_index, asset in enumerate(self._assets):
            scale = self._object_scale(asset)
            builder = self._scene.create_actor_builder()
            builder.set_name(f"pool_object_{asset_index:03d}_{asset.object_id}")
            render_material = self._build_render_material(asset, asset_index)
            builder.add_visual_from_file(
                str(asset.mesh_file),
                scale=scale,
                material=render_material,
                name=f"visual_{asset_index:03d}",
            )
            collision_scale = tuple(
                float(value) * self._config.physics.collision_scale_multiplier for value in scale
            )
            if asset.collision_mesh_files:
                for collision_mesh in asset.collision_mesh_files:
                    builder.add_convex_collision_from_file(
                        str(collision_mesh),
                        scale=collision_scale,
                        material=material,
                        density=self._config.physics.object_density,
                    )
            else:
                half_size = tuple(
                    float(extent) * scale[axis] * 0.5 for axis, extent in enumerate(asset.extents)
                )
                center = tuple(
                    float(value) * scale[axis] for axis, value in enumerate(asset.center)
                )
                builder.add_box_collision(
                    pose=self._sapien.Pose(center),
                    half_size=half_size,
                    material=material,
                    density=self._config.physics.object_density,
                )

            entity = builder.build(name=f"pool_object_{asset_index:03d}_{asset.object_id}")
            render_body, rigid_body = self._find_object_components(entity)
            rigid_body.solver_position_iterations = min(
                255, max(1, self._config.physics.solver_iterations)
            )
            rigid_body.solver_velocity_iterations = 8
            slots.append(
                _ObjectSlot(
                    asset=asset,
                    entity=entity,
                    render_body=render_body,
                    rigid_body=rigid_body,
                    render_material=render_material,
                    scale=scale,
                )
            )
        for slot in slots:
            self._deactivate_slot(slot)
        return slots

    def _find_object_components(self, entity: object) -> tuple[object, object]:
        render_body = entity.find_component_by_type(self._sapien.render.RenderBodyComponent)
        rigid_body = entity.find_component_by_type(self._sapien.physx.PhysxRigidDynamicComponent)
        if render_body is None or rigid_body is None:
            raise GenerationError(
                f"SAPIEN actor {entity.name!r} is missing render or dynamic body components"
            )
        return render_body, rigid_body

    def _object_scale(self, asset: ObjectAsset) -> tuple[float, float, float]:
        base_scale = 1.0
        sampling = self._config.sampling
        if (
            sampling.target_object_extent_min is not None
            and sampling.target_object_extent_max is not None
        ):
            max_extent = max(float(value) for value in asset.extents)
            if max_extent <= 0.0:
                raise GenerationError(
                    f"Object {asset.object_id} has invalid extents: {asset.extents}"
                )
            target_extent = 0.5 * (
                float(sampling.target_object_extent_min) + float(sampling.target_object_extent_max)
            )
            base_scale = target_extent / max_extent
        return (base_scale, base_scale, base_scale)

    def _build_render_material(self, asset: ObjectAsset, asset_index: int) -> object | None:
        if asset.mesh_file.suffix.lower() in {".glb", ".gltf"}:
            return None

        material = self._sapien.render.RenderMaterial(
            base_color=list(self._fallback_object_color(asset_index)),
            roughness=0.72,
            metallic=0.0,
            specular=0.28,
        )
        base_color_texture = self._find_texture_file(
            asset,
            (
                "basecolor",
                "base_color",
                "albedo",
                "diffuse",
                "color",
            ),
            fallback=asset.texture_file,
        )
        if base_color_texture is not None:
            texture = self._load_texture(base_color_texture, srgb=True)
            material.set_base_color([1.0, 1.0, 1.0, 1.0])
            material.set_base_color_texture(texture)

        roughness_texture = self._find_texture_file(asset, ("roughness", "rough"))
        if roughness_texture is not None:
            material.set_roughness_texture(self._load_texture(roughness_texture, srgb=False))

        metallic_texture = self._find_texture_file(asset, ("metallic", "metalness", "metal"))
        if metallic_texture is not None:
            material.set_metallic_texture(self._load_texture(metallic_texture, srgb=False))

        normal_texture = self._find_texture_file(asset, ("normal", "normalgl", "normaldx", "nrm"))
        if normal_texture is not None:
            material.set_normal_texture(self._load_texture(normal_texture, srgb=False))

        return material

    def _load_texture(self, path: Path, *, srgb: bool) -> object:
        resolved = path.resolve()
        cached = self._texture_cache.get(resolved)
        if cached is not None:
            return cached
        texture = self._sapien.render.RenderTexture2D(
            str(resolved),
            mipmap_levels=4,
            filter_mode="linear",
            address_mode="repeat",
            srgb=srgb,
        )
        self._texture_cache[resolved] = texture
        return texture

    def _find_texture_file(
        self,
        asset: ObjectAsset,
        keywords: tuple[str, ...],
        *,
        fallback: Path | None = None,
    ) -> Path | None:
        candidates: list[Path] = []
        for pattern in ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.tga", "*.bmp"):
            candidates.extend(asset.object_dir.glob(pattern))
        for candidate in sorted(candidates):
            compact_name = _compact_texture_name(candidate.stem)
            if any(keyword in compact_name for keyword in keywords):
                return candidate
        return fallback if fallback is not None and fallback.is_file() else None

    def _fallback_object_color(self, asset_index: int) -> tuple[float, float, float, float]:
        hue = (0.61803398875 * float(asset_index + 1)) % 1.0
        r, g, b = _hsv_to_rgb(hue, 0.42, 0.86)
        return (r, g, b, 1.0)

    def _build_robot_camera_mount(self) -> object:
        builder = self._scene.create_actor_builder()
        dark = (0.08, 0.085, 0.09, 1.0)
        metal = (0.45, 0.48, 0.50, 1.0)
        builder.add_box_visual(
            pose=self._sapien.Pose([0.0, -0.090, 0.140]),
            half_size=[0.055, 0.025, 0.030],
            material=dark,
        )
        builder.add_box_visual(
            pose=self._sapien.Pose([0.085, -0.115, 0.080]),
            half_size=[0.012, 0.014, 0.040],
            material=metal,
        )
        builder.add_box_visual(
            pose=self._sapien.Pose([-0.085, -0.115, 0.080]),
            half_size=[0.012, 0.014, 0.040],
            material=metal,
        )
        entity = builder.build_kinematic(name=self._config.robot_camera.body_name)
        return entity

    def _sample_active_objects(self) -> tuple[_ActiveObject, ...]:
        sampling = self._config.sampling
        object_count = int(
            self._rng.integers(sampling.object_count_min, sampling.object_count_max + 1)
        )
        slot_indices = self._rng.choice(len(self._slots), size=object_count, replace=False)
        xy_positions = sample_spawn_xy_positions(self._config, self._rng, object_count)
        active: list[_ActiveObject] = []
        for index, slot_index in enumerate(slot_indices):
            slot = self._slots[int(slot_index)]
            if index == object_count - 1:
                x = float(self._rng.uniform(-0.035, 0.035))
                y = float(self._rng.uniform(-0.035, 0.035))
            else:
                x = float(xy_positions[index, 0])
                y = float(xy_positions[index, 1])
            bbox_min_z = float(slot.asset.bbox_min[2]) * float(slot.scale[2])
            max_extent = max(
                float(extent) * float(scale)
                for extent, scale in zip(slot.asset.extents, slot.scale, strict=True)
            )
            drop_height = max(self._config.sampling.initial_drop_height, 0.14)
            stack_height = max(
                self._config.sampling.per_object_stack_height, 0.32 * max_extent, 0.045
            )
            z = self._config.table.top_z - bbox_min_z + drop_height + index * stack_height
            quat = _random_yaw_quat_wxyz(self._rng)
            active.append(
                _ActiveObject(
                    instance_id=index + 1,
                    slot=slot,
                    initial_position=(x, y, float(z)),
                    initial_quat_wxyz=tuple(float(value) for value in quat),
                )
            )
        return tuple(active)

    def _reset_scene(self, active_objects: tuple[_ActiveObject, ...]) -> None:
        self._apply_random_surface()
        for slot in self._slots:
            self._deactivate_slot(slot)
        staged_steps = 0
        if self._config.physics.settle_steps > 0:
            staged_steps = min(
                120, max(1, self._config.physics.settle_steps // len(active_objects))
            )
        for item in active_objects:
            self._activate_slot(item.slot, item.initial_position, item.initial_quat_wxyz)
            for _ in range(staged_steps):
                self._scene.step()
        if self._robot_entity is not None:
            self._robot_entity.set_pose(self._sapien.Pose([0.0, 0.0, -5.0]))
        self._scene.update_render()

    def _activate_slot(
        self,
        slot: _ObjectSlot,
        position: tuple[float, float, float],
        quat_wxyz: tuple[float, float, float, float],
    ) -> None:
        slot.render_body.enable()
        slot.rigid_body.enable()
        pose = self._sapien.Pose(position, quat_wxyz)
        slot.entity.set_pose(pose)
        slot.rigid_body.set_pose(pose)
        slot.rigid_body.set_linear_velocity(np.zeros(3, dtype=np.float32))
        slot.rigid_body.set_angular_velocity(np.zeros(3, dtype=np.float32))
        slot.rigid_body.wake_up()

    def _deactivate_slot(self, slot: _ObjectSlot) -> None:
        slot.render_body.disable()
        slot.rigid_body.disable()
        pose = self._sapien.Pose([0.0, 0.0, -5.0])
        slot.entity.set_pose(pose)
        slot.rigid_body.set_pose(pose)
        slot.rigid_body.set_linear_velocity(np.zeros(3, dtype=np.float32))
        slot.rigid_body.set_angular_velocity(np.zeros(3, dtype=np.float32))

    def _sample_camera(
        self,
        active_objects: tuple[_ActiveObject, ...],
        *,
        positions: dict[int, np.ndarray],
        target_instance_id: int | None = None,
    ) -> SampledCamera:
        proxy_objects = tuple(
            _SampleProxy(
                instance_id=item.instance_id,
                asset=item.slot.asset,
                position=tuple(float(value) for value in positions[item.instance_id]),
                scale=item.slot.scale,
            )
            for item in active_objects
        )
        if target_instance_id is None:
            candidates = [item for item in positions.items() if self._is_focus_candidate(item[1])]
            if not candidates:
                candidates = list(positions.items())
            target_instance_id = max(candidates, key=lambda item: float(item[1][2]))[0]
        self._target_instance_id = target_instance_id
        return sample_camera_for_objects(
            self._config,
            self._rng,
            proxy_objects,  # type: ignore[arg-type]
            target_instance_id=target_instance_id,
            positions=positions,
        )

    def _visual_focus_position(self, item: _ActiveObject) -> np.ndarray:
        origin = np.asarray(item.slot.entity.pose.p, dtype=np.float64).copy()
        max_extent = max(
            float(extent) * float(scale)
            for extent, scale in zip(item.slot.asset.extents, item.slot.scale, strict=True)
        )
        origin[2] += 0.5 * max_extent
        return origin

    def _is_focus_candidate(self, position: np.ndarray) -> bool:
        x, y, z = (float(value) for value in position)
        return (
            -0.45 <= x <= 0.45
            and -0.45 <= y <= 0.45
            and self._config.table.top_z - 0.02 <= z <= self._config.table.top_z + 0.65
        )

    def _render_with_camera(
        self,
        active_objects: tuple[_ActiveObject, ...],
        final_camera: SampledCamera,
    ) -> RenderedScene:
        if self._robot_entity is not None:
            self._robot_entity.set_pose(self._robot_mount_pose_for_camera(final_camera))
        self._scene.update_render()
        self._camera.take_picture()
        return self._render(active_objects)

    def _select_visible_target(
        self,
        rendered: RenderedScene,
        active_objects: tuple[_ActiveObject, ...],
    ) -> int | None:
        best_instance_id: int | None = None
        best_score = -1.0
        for item in active_objects:
            stats = compute_instance_mask_stats(rendered.instance_mask, item.instance_id)
            score = stats.bbox_area_fraction
            if stats.pixel_count > 0 and score > best_score:
                best_instance_id = item.instance_id
                best_score = score
        return best_instance_id

    def _should_refocus(self, rendered: RenderedScene, visible_target_id: int) -> bool:
        current_target_id = getattr(self, "_target_instance_id", None)
        if current_target_id != visible_target_id:
            return True
        return _target_needs_refocus(
            rendered.instance_mask,
            visible_target_id,
            self._config.validation,
        )

    def _set_camera_pose(self, sampled_camera: SampledCamera) -> None:
        self._camera.entity.set_pose(self._sapien_pose_for_camera(sampled_camera))

    def _sapien_pose_for_camera(self, sampled_camera: SampledCamera) -> object:
        world_R_camera = np.asarray(sampled_camera.world_R_camera, dtype=np.float64)
        right = world_R_camera[:, 0]
        up = world_R_camera[:, 1]
        back = world_R_camera[:, 2]
        sapien_rotation = np.column_stack([-back, -right, up])
        quat = matrix_to_quat_wxyz(sapien_rotation)
        return self._sapien.Pose(list(sampled_camera.position), quat.tolist())

    def _robot_mount_pose_for_camera(self, sampled_camera: SampledCamera) -> object:
        world_R_camera = np.asarray(sampled_camera.world_R_camera, dtype=np.float64)
        quat = matrix_to_quat_wxyz(world_R_camera)
        return self._sapien.Pose(list(sampled_camera.position), quat.tolist())

    def _render(self, active_objects: tuple[_ActiveObject, ...]) -> RenderedScene:
        color = self._camera.get_picture("Color")
        rgb = _to_uint8_rgb(
            color[..., :3],
            exposure=self._config.render.rgb_exposure,
            gamma=self._config.render.rgb_gamma,
            tone_mapping=self._config.render.tone_mapping,
        )

        position = self._camera.get_picture("Position")
        depth = -np.asarray(position[..., 2], dtype=np.float32)
        depth[~np.isfinite(depth)] = 0.0
        depth[depth < 0.0] = 0.0

        segmentation = self._camera.get_picture("Segmentation")
        entity_id_to_instance_id = {
            int(item.slot.entity.get_per_scene_id()): int(item.instance_id)
            for item in active_objects
        }
        instance_mask = np.zeros(segmentation.shape[:2], dtype=np.uint16)
        visual_ids = _extract_entity_segmentation_ids(
            segmentation,
            entity_id_to_instance_id.keys(),
        )
        for entity_id, instance_id in entity_id_to_instance_id.items():
            instance_mask[visual_ids == entity_id] = np.uint16(instance_id)
        return RenderedScene(rgb=rgb, depth=depth, instance_mask=instance_mask)

    def _build_camera_metadata(self, final_camera: SampledCamera) -> CameraMetadata:
        world_R_camera = np.asarray(final_camera.world_R_camera, dtype=np.float64)
        world_t_camera = np.asarray(final_camera.position, dtype=np.float64)
        world_T_camera = make_transform(world_R_camera, world_t_camera)
        camera_T_world = invert_transform(world_T_camera)
        intrinsic = self._camera.get_intrinsic_matrix()
        intrinsics = {
            "fx": float(intrinsic[0, 0]),
            "fy": float(intrinsic[1, 1]),
            "cx": float(intrinsic[0, 2]),
            "cy": float(intrinsic[1, 2]),
        }
        return CameraMetadata(
            width=self._config.render.width,
            height=self._config.render.height,
            fovy_degrees=self._config.render.fovy_degrees,
            intrinsics=intrinsics,
            world_T_camera=matrix_to_json(world_T_camera),
            camera_T_world=matrix_to_json(camera_T_world),
        )

    def _build_scene_metadata(
        self,
        *,
        scene_id: str,
        rendered: RenderedScene,
        active_objects: tuple[_ActiveObject, ...],
        final_camera: SampledCamera,
        contact_quality: _SapienContactQuality,
    ) -> SceneMetadata:
        world_R_camera = np.asarray(final_camera.world_R_camera, dtype=np.float64)
        world_t_camera = np.asarray(final_camera.position, dtype=np.float64)
        camera_T_world = invert_transform(make_transform(world_R_camera, world_t_camera))
        image_area = float(self._config.render.width * self._config.render.height)
        objects_metadata: list[SceneObjectMetadata] = []
        for item in active_objects:
            position = np.asarray(item.slot.entity.pose.p, dtype=np.float64).copy()
            quat = np.asarray(item.slot.entity.pose.q, dtype=np.float64).copy()
            world_T_object = transform_from_pos_quat(position, quat)
            camera_T_object = camera_T_world @ world_T_object
            mask_stats = compute_instance_mask_stats(rendered.instance_mask, item.instance_id)
            objects_metadata.append(
                SceneObjectMetadata(
                    instance_id=item.instance_id,
                    object_id=item.slot.asset.object_id,
                    mesh_path=make_relative_path(item.slot.asset.mesh_file, self._dataset_root),
                    scale=item.slot.scale,
                    world_T_object=matrix_to_json(world_T_object),
                    camera_T_object=matrix_to_json(camera_T_object),
                    visible_pixel_count=mask_stats.pixel_count,
                    visible_image_fraction=float(mask_stats.pixel_count / image_area),
                    visible_bbox_xyxy=mask_stats.bbox_xyxy,
                    visible_bbox_area_fraction=mask_stats.bbox_area_fraction,
                    visible_bbox_fill_fraction=mask_stats.bbox_fill_fraction,
                )
            )

        return SceneMetadata(
            scene_id=scene_id,
            units="meters",
            workspace_frame=self._config.workspace.frame,
            workspace_bounds=self._config.workspace.to_json(),
            objects=tuple(objects_metadata),
            simulator={
                "name": "sapien",
                "version": str(getattr(self._sapien, "__version__", "unknown")),
                "backend": "v2_spike_actor_pool",
                "timestep": self._config.physics.timestep,
                "settle_steps": self._config.physics.settle_steps,
                "preloaded_actor_count": len(self._slots),
                "environment_map": (
                    make_relative_path(self._environment_map_path, self._dataset_root)
                    if self._environment_map_path is not None
                    else None
                ),
                "contact_quality": contact_quality.to_json(),
                "camera_mount": {
                    "type": "eye_in_hand",
                    "visible": self._config.robot_camera.visible,
                    "body_name": self._config.robot_camera.body_name,
                },
            },
            target_instance_id=getattr(self, "_target_instance_id", None),
            background=self._background_metadata(),
        )

    def _background_metadata(self) -> dict[str, object]:
        mesh_slot = self._active_mesh_background
        if mesh_slot is not None:
            asset = mesh_slot.asset
            return {
                "background_id": asset.background_id,
                "kind": "trellis_mesh",
                "source": asset.source,
                "description": asset.description,
                "mesh_path": make_relative_path(asset.mesh_file, self._dataset_root),
                "scale": [float(value) for value in mesh_slot.scale],
                "world_position": [float(value) for value in mesh_slot.position],
                "world_quat_wxyz": [float(value) for value in mesh_slot.rotation],
            }

        surface = self._active_surface
        if surface is None:
            return {
                "surface_id": "fallback_table",
                "source": "table_config",
                "base_color": [float(value) for value in self._config.table.rgba],
            }
        return {
            "surface_id": surface.surface_id,
            "source": surface.source,
            "description": surface.description,
            "base_color_path": make_relative_path(surface.base_color_file, self._dataset_root),
            "roughness_path": make_relative_path(surface.roughness_file, self._dataset_root),
            "metallic_path": make_relative_path(surface.metallic_file, self._dataset_root),
            "normal_path": make_relative_path(surface.normal_file, self._dataset_root),
            "roughness": float(surface.roughness),
            "metallic": float(surface.metallic),
            "specular": float(surface.specular),
        }

    def _measure_contact_quality(self) -> _SapienContactQuality:
        min_separation = 0.0
        contact_count = 0
        for contact in self._scene.get_contacts():
            for point in contact.points:
                contact_count += 1
                min_separation = min(min_separation, float(point.separation))
        return _SapienContactQuality(
            contact_count=contact_count,
            min_separation_m=min_separation,
            max_penetration_m=max(0.0, -min_separation),
        )


@dataclass(frozen=True)
class _SampleProxy:
    instance_id: int
    asset: ObjectAsset
    position: tuple[float, float, float]
    scale: tuple[float, float, float]


def _resolve_render_asset_path(path: Path | str | None, *, dataset_root: Path) -> Path | None:
    if path is None:
        return None
    candidate = Path(path)
    if candidate.is_absolute():
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        return candidate

    for base in (Path.cwd(), _PROJECT_ROOT, dataset_root):
        resolved = (base / candidate).resolve()
        if resolved.is_file():
            return resolved

    raise FileNotFoundError(candidate)


def _background_mesh_transform(
    asset: MeshBackgroundAsset,
    *,
    visual_size_m: float,
    top_z: float,
    z_offset_m: float,
) -> tuple[
    tuple[float, float, float], tuple[float, float, float, float], tuple[float, float, float]
]:
    extents = np.asarray(asset.extents, dtype=np.float64)
    center = np.asarray(asset.center, dtype=np.float64)
    if extents.shape != (3,) or not np.isfinite(extents).all() or float(np.max(extents)) <= 0.0:
        raise GenerationError(
            f"Background mesh {asset.background_id!r} has invalid extents: {asset.extents}"
        )

    thin_axis = int(np.argmin(extents))
    rotation = _rotation_matrix_for_background_axis(thin_axis)
    rotated_center = rotation @ center
    rotated_extents = np.abs(rotation) @ extents
    horizontal_extent = float(max(rotated_extents[0], rotated_extents[1]))
    if horizontal_extent <= 0.0:
        raise GenerationError(
            f"Background mesh {asset.background_id!r} has invalid horizontal size"
        )
    uniform_scale = float(visual_size_m / horizontal_extent)
    scaled_center = rotated_center * uniform_scale
    scaled_vertical_half_extent = 0.5 * float(rotated_extents[2]) * uniform_scale
    position = np.array(
        [
            -scaled_center[0],
            -scaled_center[1],
            float(top_z) + float(z_offset_m) - scaled_center[2] - scaled_vertical_half_extent,
        ],
        dtype=np.float64,
    )
    quat = matrix_to_quat_wxyz(rotation)
    return (
        (uniform_scale, uniform_scale, uniform_scale),
        tuple(float(value) for value in quat),  # type: ignore[return-value]
        tuple(float(value) for value in position),  # type: ignore[return-value]
    )


def _rotation_matrix_for_background_axis(thin_axis: int) -> np.ndarray:
    if thin_axis == 0:
        theta = -0.5 * math.pi
        return np.array(
            [
                [math.cos(theta), 0.0, math.sin(theta)],
                [0.0, 1.0, 0.0],
                [-math.sin(theta), 0.0, math.cos(theta)],
            ],
            dtype=np.float64,
        )
    if thin_axis == 1:
        theta = 0.5 * math.pi
        return np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, math.cos(theta), -math.sin(theta)],
                [0.0, math.sin(theta), math.cos(theta)],
            ],
            dtype=np.float64,
        )
    return np.eye(3, dtype=np.float64)


def _to_uint8_rgb(
    color: np.ndarray,
    *,
    exposure: float,
    gamma: float,
    tone_mapping: str,
) -> np.ndarray:
    mapped = np.asarray(color[..., :3], dtype=np.float32)
    mapped = np.nan_to_num(mapped, nan=0.0, posinf=0.0, neginf=0.0)
    mapped = np.maximum(mapped * float(exposure), 0.0)
    if tone_mapping == "reinhard":
        mapped = mapped / (1.0 + mapped)
    elif tone_mapping == "aces":
        mapped = _aces_filmic(mapped)
    else:
        mapped = np.clip(mapped, 0.0, 1.0)
    mapped = np.clip(mapped, 0.0, 1.0)
    mapped = np.power(mapped, 1.0 / float(gamma))
    return (mapped * 255.0 + 0.5).astype(np.uint8)


def _target_needs_refocus(
    instance_mask: np.ndarray,
    target_instance_id: int,
    validation: ValidationConfig,
) -> bool:
    stats = compute_instance_mask_stats(instance_mask, target_instance_id)
    height, width = instance_mask.shape
    if stats.image_fraction < validation.min_target_image_fraction:
        return True
    if stats.bbox_area_fraction < validation.min_target_bbox_fraction:
        return True
    if stats.bbox_area_fraction > validation.max_target_bbox_fraction:
        return True
    if (
        stats.center_offset_fraction(height=height, width=width)
        > validation.max_target_center_offset_fraction
    ):
        return True
    return stats.touches_border(
        height=height,
        width=width,
        margin=validation.target_border_margin_pixels,
    )


def _aces_filmic(color: np.ndarray) -> np.ndarray:
    a = 2.51
    b = 0.03
    c = 2.43
    d = 0.59
    e = 0.14
    return (color * (a * color + b)) / (color * (c * color + d) + e)


def _hsv_to_rgb(hue: float, saturation: float, value: float) -> tuple[float, float, float]:
    hue = hue % 1.0
    sector = int(hue * 6.0)
    fraction = hue * 6.0 - sector
    p = value * (1.0 - saturation)
    q = value * (1.0 - fraction * saturation)
    t = value * (1.0 - (1.0 - fraction) * saturation)
    sector %= 6
    if sector == 0:
        return value, t, p
    if sector == 1:
        return q, value, p
    if sector == 2:
        return p, value, t
    if sector == 3:
        return p, q, value
    if sector == 4:
        return t, p, value
    return value, p, q


def _random_yaw_quat_wxyz(rng: np.random.Generator) -> tuple[float, float, float, float]:
    yaw = float(rng.uniform(-math.pi, math.pi))
    half_yaw = 0.5 * yaw
    return (math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw))


def _extract_entity_segmentation_ids(
    segmentation: np.ndarray,
    entity_ids: object,
) -> np.ndarray:
    entity_id_values = tuple(int(value) for value in entity_ids)
    entity_channel = 1 if segmentation.shape[-1] > 1 else 0
    ids = np.asarray(segmentation[..., entity_channel], dtype=np.int64)
    if entity_channel != 0 and not any(np.any(ids == entity_id) for entity_id in entity_id_values):
        ids = np.asarray(segmentation[..., 0], dtype=np.int64)
    return ids


def _compact_texture_name(value: str) -> str:
    return "".join(char.lower() for char in value if char.isalnum())
