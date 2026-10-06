"""Configuration objects for dataset generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class RenderConfig:
    width: int = 1920
    height: int = 1080
    fovy_degrees: float = 52.0
    znear: float = 0.01
    zfar: float = 10.0
    environment_map: Path | str | None = None
    rgb_exposure: float = 1.0
    rgb_gamma: float = 2.2
    tone_mapping: str = "aces"
    enable_shadows: bool = False
    shadow_map_size: int = 8192
    offscreen_samples: int = 8
    shadow_scale: float = 0.25
    shadow_clip: float = 1.0
    camera_name: str = "render_camera"

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Render width and height must be positive")
        if not 1.0 <= self.fovy_degrees <= 170.0:
            raise ValueError("fovy_degrees must be in [1, 170]")
        if self.znear <= 0.0 or self.zfar <= self.znear:
            raise ValueError("Invalid camera near/far clipping range")
        if self.environment_map is not None and not str(self.environment_map):
            raise ValueError("environment_map cannot be empty")
        if self.rgb_exposure <= 0.0:
            raise ValueError("rgb_exposure must be positive")
        if self.rgb_gamma <= 0.0:
            raise ValueError("rgb_gamma must be positive")
        if self.tone_mapping not in {"clip", "reinhard", "aces"}:
            raise ValueError("tone_mapping must be 'clip', 'reinhard', or 'aces'")
        if self.shadow_map_size <= 0:
            raise ValueError("shadow_map_size must be positive")
        if self.offscreen_samples < 0:
            raise ValueError("offscreen_samples cannot be negative")
        if self.shadow_scale <= 0.0:
            raise ValueError("shadow_scale must be positive")
        if self.shadow_clip <= 0.0:
            raise ValueError("shadow_clip must be positive")


@dataclass(frozen=True)
class TableConfig:
    size: tuple[float, float, float] = (4.0, 4.0, 0.05)
    top_z: float = 0.0
    rgba: tuple[float, float, float, float] = (0.46, 0.48, 0.49, 1.0)

    @property
    def half_extents(self) -> tuple[float, float, float]:
        return (self.size[0] * 0.5, self.size[1] * 0.5, self.size[2] * 0.5)

    @property
    def center_z(self) -> float:
        return self.top_z - self.size[2] * 0.5


@dataclass(frozen=True)
class BackgroundConfig:
    enabled: bool = True
    directory: Path | str = "backgrounds"
    surface_id: str | None = None
    visual_size_m: float = 1.8
    visual_z_offset_m: float = -0.004

    def __post_init__(self) -> None:
        if not str(self.directory):
            raise ValueError("background directory cannot be empty")
        if self.surface_id is not None and not self.surface_id:
            raise ValueError("background surface_id cannot be empty")
        if self.visual_size_m <= 0.0:
            raise ValueError("background visual_size_m must be positive")


@dataclass(frozen=True)
class WorkspaceConfig:
    frame: str = "camera"
    minimum: tuple[float, float, float] = (-1.0, -0.8, 0.05)
    maximum: tuple[float, float, float] = (1.0, 0.8, 2.0)

    def to_json(self) -> dict[str, list[float]]:
        return {
            "min": [float(value) for value in self.minimum],
            "max": [float(value) for value in self.maximum],
        }


@dataclass(frozen=True)
class SamplingConfig:
    object_count_min: int = 5
    object_count_max: int = 6
    object_scale_min: float = 1.0
    object_scale_max: float = 1.0
    target_object_extent_min: float | None = 0.14
    target_object_extent_max: float | None = 0.24
    spawn_xy_min: tuple[float, float] = (-0.15, -0.12)
    spawn_xy_max: tuple[float, float] = (0.15, 0.12)
    spawn_xy_min_distance: float = 0.075
    initial_drop_height: float = 0.10
    per_object_stack_height: float = 0.015
    camera_radius_min: float = 0.26
    camera_radius_max: float = 0.48
    camera_azimuth_min_degrees: float = -145.0
    camera_azimuth_max_degrees: float = -35.0
    camera_elevation_min_degrees: float = 45.0
    camera_elevation_max_degrees: float = 68.0
    camera_target: tuple[float, float, float] = (0.0, 0.0, 0.08)
    camera_target_jitter: tuple[float, float, float] = (0.02, 0.02, 0.025)
    camera_focus: str = "target"
    camera_cluster_margin: float = 1.15
    camera_target_margin: float = 1.25
    camera_distance_jitter_fraction: float = 0.08

    def __post_init__(self) -> None:
        if self.object_count_min <= 0 or self.object_count_max < self.object_count_min:
            raise ValueError("Invalid object count range")
        if self.object_scale_min <= 0.0 or self.object_scale_max < self.object_scale_min:
            raise ValueError("Invalid object scale range")
        if (
            self.spawn_xy_min[0] > self.spawn_xy_max[0]
            or self.spawn_xy_min[1] > self.spawn_xy_max[1]
        ):
            raise ValueError("Invalid spawn xy range")
        if self.spawn_xy_min_distance < 0.0:
            raise ValueError("spawn_xy_min_distance cannot be negative")
        if (self.target_object_extent_min is None) != (self.target_object_extent_max is None):
            raise ValueError(
                "Both target_object_extent_min and target_object_extent_max must be set or None"
            )
        if self.target_object_extent_min is not None and self.target_object_extent_max is not None:
            if self.target_object_extent_min <= 0.0:
                raise ValueError("target_object_extent_min must be positive")
            if self.target_object_extent_max < self.target_object_extent_min:
                raise ValueError("Invalid target object extent range")
        if self.camera_radius_min <= 0.0 or self.camera_radius_max < self.camera_radius_min:
            raise ValueError("Invalid camera radius range")
        if self.camera_focus not in {"scene", "target"}:
            raise ValueError("camera_focus must be 'scene' or 'target'")
        if self.camera_cluster_margin <= 0.0:
            raise ValueError("camera_cluster_margin must be positive")
        if self.camera_target_margin <= 0.0:
            raise ValueError("camera_target_margin must be positive")
        if not 0.0 <= self.camera_distance_jitter_fraction < 1.0:
            raise ValueError("camera_distance_jitter_fraction must be in [0, 1)")


@dataclass(frozen=True)
class PhysicsConfig:
    timestep: float = 0.002
    settle_steps: int = 2500
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    object_density: float = 350.0
    friction: tuple[float, float, float] = (1.0, 0.005, 0.0001)
    solver_iterations: int = 100
    solver_ls_iterations: int = 50
    solver_noslip_iterations: int = 20
    contact_margin: float = 0.004
    contact_gap: float = 0.0
    contact_solref: tuple[float, float] = (0.004, 1.8)
    contact_solimp: tuple[float, float, float, float, float] = (0.95, 0.99, 0.001, 0.5, 2.0)
    collision_scale_multiplier: float = 1.015

    def __post_init__(self) -> None:
        if self.timestep <= 0.0:
            raise ValueError("timestep must be positive")
        if self.settle_steps < 0:
            raise ValueError("settle_steps cannot be negative")
        if self.object_density <= 0.0:
            raise ValueError("object_density must be positive")
        if self.solver_iterations <= 0:
            raise ValueError("solver_iterations must be positive")
        if self.solver_ls_iterations < 0:
            raise ValueError("solver_ls_iterations cannot be negative")
        if self.solver_noslip_iterations < 0:
            raise ValueError("solver_noslip_iterations cannot be negative")
        if self.contact_margin < 0.0:
            raise ValueError("contact_margin cannot be negative")
        if self.contact_gap < 0.0 or self.contact_gap > self.contact_margin:
            raise ValueError("contact_gap must be in [0, contact_margin]")
        if len(self.contact_solref) != 2:
            raise ValueError("contact_solref must have length 2")
        if len(self.contact_solimp) != 5:
            raise ValueError("contact_solimp must have length 5")
        if self.collision_scale_multiplier <= 0.0:
            raise ValueError("collision_scale_multiplier must be positive")


@dataclass(frozen=True)
class ValidationConfig:
    min_instance_pixels: int = 0
    min_instance_image_fraction: float = 0.0
    min_target_image_fraction: float = 0.04
    min_target_bbox_fraction: float = 0.10
    max_target_bbox_fraction: float = 0.72
    max_target_center_offset_fraction: float = 0.28
    target_border_margin_pixels: int = 16
    min_foreground_image_fraction: float = 0.10
    min_foreground_bbox_fraction: float = 0.12
    max_foreground_bbox_fraction: float = 0.82
    min_foreground_bbox_fill_fraction: float = 0.12
    border_margin_pixels: int = 0
    max_contact_penetration_m: float = 0.004
    max_object_xy_radius_m: float | None = None
    max_object_xy_span_m: float | None = None
    require_visible_instances: bool = True
    max_attempts_per_scene: int = 80

    def __post_init__(self) -> None:
        if self.min_instance_pixels < 0:
            raise ValueError("min_instance_pixels cannot be negative")
        if self.min_instance_image_fraction < 0.0:
            raise ValueError("min_instance_image_fraction cannot be negative")
        if self.min_target_image_fraction < 0.0:
            raise ValueError("min_target_image_fraction cannot be negative")
        if self.min_target_bbox_fraction < 0.0:
            raise ValueError("min_target_bbox_fraction cannot be negative")
        if self.max_target_bbox_fraction <= 0.0:
            raise ValueError("max_target_bbox_fraction must be positive")
        if self.max_target_bbox_fraction < self.min_target_bbox_fraction:
            raise ValueError("Invalid target bbox fraction range")
        if self.max_target_center_offset_fraction < 0.0:
            raise ValueError("max_target_center_offset_fraction cannot be negative")
        if self.target_border_margin_pixels < 0:
            raise ValueError("target_border_margin_pixels cannot be negative")
        if self.min_foreground_image_fraction < 0.0:
            raise ValueError("min_foreground_image_fraction cannot be negative")
        if self.min_foreground_bbox_fraction < 0.0:
            raise ValueError("min_foreground_bbox_fraction cannot be negative")
        if self.max_foreground_bbox_fraction <= 0.0:
            raise ValueError("max_foreground_bbox_fraction must be positive")
        if self.max_foreground_bbox_fraction < self.min_foreground_bbox_fraction:
            raise ValueError("Invalid foreground bbox fraction range")
        if self.min_foreground_bbox_fill_fraction < 0.0:
            raise ValueError("min_foreground_bbox_fill_fraction cannot be negative")
        if self.border_margin_pixels < 0:
            raise ValueError("border_margin_pixels cannot be negative")
        if self.max_contact_penetration_m < 0.0:
            raise ValueError("max_contact_penetration_m cannot be negative")
        if self.max_object_xy_radius_m is not None and self.max_object_xy_radius_m <= 0.0:
            raise ValueError("max_object_xy_radius_m must be positive when set")
        if self.max_object_xy_span_m is not None and self.max_object_xy_span_m <= 0.0:
            raise ValueError("max_object_xy_span_m must be positive when set")
        if self.max_attempts_per_scene <= 0:
            raise ValueError("max_attempts_per_scene must be positive")


@dataclass(frozen=True)
class RobotCameraConfig:
    visible: bool = True
    body_name: str = "robot_wrist_camera"
    finger_forward_offset: float = 0.21
    finger_vertical_offset: float = -0.115
    finger_lateral_offset: float = 0.078

    def __post_init__(self) -> None:
        if not self.body_name:
            raise ValueError("robot camera body_name cannot be empty")
        if self.finger_forward_offset <= 0.0:
            raise ValueError("finger_forward_offset must be positive")
        if self.finger_lateral_offset <= 0.0:
            raise ValueError("finger_lateral_offset must be positive")


@dataclass(frozen=True)
class GeneratorConfig:
    render: RenderConfig = field(default_factory=RenderConfig)
    table: TableConfig = field(default_factory=TableConfig)
    background: BackgroundConfig = field(default_factory=BackgroundConfig)
    workspace: WorkspaceConfig = field(default_factory=WorkspaceConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    robot_camera: RobotCameraConfig = field(default_factory=RobotCameraConfig)
