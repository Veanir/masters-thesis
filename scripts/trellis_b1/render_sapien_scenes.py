"""CLI for the SAPIEN v2 spike generator."""

from __future__ import annotations

import argparse
from pathlib import Path

from masters_rgbd.synthetic.config import (
    BackgroundConfig,
    GeneratorConfig,
    PhysicsConfig,
    RenderConfig,
    RobotCameraConfig,
    SamplingConfig,
    ValidationConfig,
)
from masters_rgbd.synthetic.sapien_generator import generate_dataset_sapien

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_ENVIRONMENT_MAP = Path("assets/ferndale_studio_07_4k.exr")


def _default_environment_map() -> Path | None:
    if (_PROJECT_ROOT / _DEFAULT_ENVIRONMENT_MAP).is_file():
        return _DEFAULT_ENVIRONMENT_MAP
    return None


def build_parser() -> argparse.ArgumentParser:
    render_defaults = RenderConfig()
    background_defaults = BackgroundConfig()
    sampling_defaults = SamplingConfig()
    physics_defaults = PhysicsConfig()
    validation_defaults = ValidationConfig()
    v2_rgb_exposure = 0.45
    v2_target_object_extent_min = 0.13
    v2_target_object_extent_max = 0.21
    v2_camera_radius_min = 0.22
    v2_camera_radius_max = 0.46
    v2_camera_target_margin = 1.20
    v2_settle_steps = 420
    v2_target_border_margin_pixels = 20
    v2_enable_shadows = True
    v2_spawn_x_half_range = 0.070
    v2_spawn_y_half_range = 0.055
    v2_spawn_min_distance = 0.020
    v2_initial_drop_height = 0.16
    v2_per_object_stack_height = 0.045
    v2_max_object_xy_radius = 0.55
    v2_max_object_xy_span = 0.80
    parser = argparse.ArgumentParser(
        description="Generate synthetic RGBD scenes with the SAPIEN v2 spike backend."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("generated_sapien_spike"),
        help="Dataset root containing prepared objects/ and output scenes/.",
    )
    parser.add_argument("--scenes", type=int, default=4, help="Number of valid scenes to generate.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--width", type=int, default=640, help="Render width.")
    parser.add_argument("--height", type=int, default=360, help="Render height.")
    parser.add_argument(
        "--fovy",
        type=float,
        default=render_defaults.fovy_degrees,
        help="Vertical field of view in degrees.",
    )
    parser.add_argument(
        "--environment-map",
        type=Path,
        default=_default_environment_map(),
        help="Optional HDRI/cubemap file for SAPIEN image-based lighting.",
    )
    parser.add_argument(
        "--rgb-exposure",
        type=float,
        default=v2_rgb_exposure,
        help="Exposure multiplier applied before SAPIEN RGB tone mapping.",
    )
    parser.add_argument(
        "--rgb-gamma",
        type=float,
        default=render_defaults.rgb_gamma,
        help="Gamma applied after SAPIEN RGB tone mapping.",
    )
    parser.add_argument(
        "--tone-mapping",
        choices=("clip", "reinhard", "aces"),
        default=render_defaults.tone_mapping,
        help="Tone mapping operator for SAPIEN RGB PNG export.",
    )
    parser.add_argument(
        "--disable-shadows",
        action="store_true",
        default=not v2_enable_shadows,
        help="Disable SAPIEN directional-light shadows.",
    )
    parser.add_argument(
        "--shadow-map-size",
        type=int,
        default=render_defaults.shadow_map_size,
        help="SAPIEN shadow map resolution.",
    )
    parser.add_argument(
        "--background-dir",
        type=Path,
        default=background_defaults.directory,
        help="Directory with prepared table/background surface materials.",
    )
    parser.add_argument(
        "--background-id",
        default=background_defaults.surface_id,
        help="Use one fixed background surface id instead of randomizing per scene.",
    )
    parser.add_argument(
        "--disable-background-surfaces",
        action="store_true",
        help="Use the plain TableConfig material instead of prepared background surfaces.",
    )
    parser.add_argument(
        "--background-visual-size",
        type=float,
        default=background_defaults.visual_size_m,
        help="Largest horizontal size for TRELLIS background meshes, in meters.",
    )
    parser.add_argument(
        "--background-visual-z-offset",
        type=float,
        default=background_defaults.visual_z_offset_m,
        help="Vertical offset for render-only background meshes relative to the collision plane.",
    )
    parser.add_argument(
        "--min-objects",
        type=int,
        default=sampling_defaults.object_count_min,
        help="Minimum active objects per scene.",
    )
    parser.add_argument(
        "--max-objects",
        type=int,
        default=sampling_defaults.object_count_max,
        help="Maximum active objects per scene.",
    )
    parser.add_argument(
        "--target-object-extent-min",
        type=float,
        default=v2_target_object_extent_min,
        help="Minimum normalized object largest extent in meters.",
    )
    parser.add_argument(
        "--target-object-extent-max",
        type=float,
        default=v2_target_object_extent_max,
        help="Maximum normalized object largest extent in meters.",
    )
    parser.add_argument(
        "--camera-radius-min",
        type=float,
        default=v2_camera_radius_min,
        help="Minimum camera radius before dynamic framing clamps.",
    )
    parser.add_argument(
        "--camera-radius-max",
        type=float,
        default=v2_camera_radius_max,
        help="Maximum camera radius before dynamic framing clamps.",
    )
    parser.add_argument(
        "--camera-focus",
        choices=("target", "scene"),
        default=sampling_defaults.camera_focus,
        help="Frame the target object or the whole visible object group.",
    )
    parser.add_argument(
        "--camera-target-margin",
        type=float,
        default=v2_camera_target_margin,
        help="Multiplicative framing margin around the target object.",
    )
    parser.add_argument(
        "--spawn-x-half-range",
        type=float,
        default=v2_spawn_x_half_range,
        help="Half-width of the SAPIEN object drop area around the table center, in meters.",
    )
    parser.add_argument(
        "--spawn-y-half-range",
        type=float,
        default=v2_spawn_y_half_range,
        help="Half-depth of the SAPIEN object drop area around the table center, in meters.",
    )
    parser.add_argument(
        "--spawn-min-distance",
        type=float,
        default=v2_spawn_min_distance,
        help="Minimum initial XY spacing between dropped objects, in meters.",
    )
    parser.add_argument(
        "--initial-drop-height",
        type=float,
        default=v2_initial_drop_height,
        help="Initial SAPIEN drop height above each object's table contact point, in meters.",
    )
    parser.add_argument(
        "--per-object-stack-height",
        type=float,
        default=v2_per_object_stack_height,
        help="Additional vertical offset per sequentially dropped object, in meters.",
    )
    parser.add_argument(
        "--settle-steps",
        type=int,
        default=v2_settle_steps,
        help="SAPIEN physics settling steps. Lower than v1 by default for spike iteration speed.",
    )
    parser.add_argument(
        "--collision-scale-multiplier",
        type=float,
        default=physics_defaults.collision_scale_multiplier,
        help="Uniform scale multiplier for collision meshes only.",
    )
    parser.add_argument(
        "--solver-iterations",
        type=int,
        default=physics_defaults.solver_iterations,
        help="PhysX position solver iterations per active actor.",
    )
    parser.add_argument(
        "--min-target-fraction",
        type=float,
        default=validation_defaults.min_target_image_fraction,
        help="Minimum image fraction occupied by the target instance mask.",
    )
    parser.add_argument(
        "--min-target-bbox-fraction",
        type=float,
        default=validation_defaults.min_target_bbox_fraction,
        help="Minimum image fraction covered by the target instance bounding box.",
    )
    parser.add_argument(
        "--max-target-bbox-fraction",
        type=float,
        default=validation_defaults.max_target_bbox_fraction,
        help="Maximum image fraction covered by the target instance bounding box.",
    )
    parser.add_argument(
        "--max-target-center-offset-fraction",
        type=float,
        default=validation_defaults.max_target_center_offset_fraction,
        help="Maximum normalized target bbox center offset from image center.",
    )
    parser.add_argument(
        "--target-border-margin-pixels",
        type=int,
        default=v2_target_border_margin_pixels,
        help="Minimum required margin between the target instance and the image border.",
    )
    parser.add_argument(
        "--min-foreground-fraction",
        type=float,
        default=validation_defaults.min_foreground_image_fraction,
        help="Minimum image fraction occupied by all visible object instances.",
    )
    parser.add_argument(
        "--min-foreground-bbox-fraction",
        type=float,
        default=validation_defaults.min_foreground_bbox_fraction,
        help="Minimum image fraction covered by the foreground bounding box.",
    )
    parser.add_argument(
        "--max-foreground-bbox-fraction",
        type=float,
        default=validation_defaults.max_foreground_bbox_fraction,
        help="Maximum image fraction covered by the foreground bounding box.",
    )
    parser.add_argument(
        "--min-foreground-bbox-fill-fraction",
        type=float,
        default=validation_defaults.min_foreground_bbox_fill_fraction,
        help="Minimum foreground-to-foreground-bbox fill ratio.",
    )
    parser.add_argument(
        "--max-contact-penetration",
        type=float,
        default=validation_defaults.max_contact_penetration_m,
        help="Reject scenes with deeper final SAPIEN contact penetration, in meters.",
    )
    parser.add_argument(
        "--max-object-xy-radius",
        type=float,
        default=v2_max_object_xy_radius,
        help="Reject scenes where any object center is farther from table center, in meters.",
    )
    parser.add_argument(
        "--max-object-xy-span",
        type=float,
        default=v2_max_object_xy_span,
        help="Reject scenes where final object XY span exceeds this value, in meters.",
    )
    parser.add_argument(
        "--max-attempts-per-scene",
        type=int,
        default=validation_defaults.max_attempts_per_scene,
        help="Maximum random resampling attempts before a scene id is considered failed.",
    )
    parser.add_argument(
        "--asset-limit",
        type=int,
        default=None,
        help="Optional prepared asset pool limit. Keep this small while benchmarking the spike.",
    )
    parser.add_argument(
        "--show-robot-camera",
        action="store_true",
        help=(
            "Render the experimental eye-in-hand camera/gripper geometry. "
            "Hidden by default in the spike."
        ),
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing scene dirs.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = GeneratorConfig(
        render=RenderConfig(
            width=args.width,
            height=args.height,
            fovy_degrees=args.fovy,
            environment_map=args.environment_map,
            rgb_exposure=args.rgb_exposure,
            rgb_gamma=args.rgb_gamma,
            tone_mapping=args.tone_mapping,
            enable_shadows=not args.disable_shadows,
            shadow_map_size=args.shadow_map_size,
        ),
        background=BackgroundConfig(
            enabled=not args.disable_background_surfaces,
            directory=args.background_dir,
            surface_id=args.background_id,
            visual_size_m=args.background_visual_size,
            visual_z_offset_m=args.background_visual_z_offset,
        ),
        sampling=SamplingConfig(
            object_count_min=args.min_objects,
            object_count_max=args.max_objects,
            target_object_extent_min=args.target_object_extent_min,
            target_object_extent_max=args.target_object_extent_max,
            camera_radius_min=args.camera_radius_min,
            camera_radius_max=args.camera_radius_max,
            camera_focus=args.camera_focus,
            camera_target_margin=args.camera_target_margin,
            spawn_xy_min=(-args.spawn_x_half_range, -args.spawn_y_half_range),
            spawn_xy_max=(args.spawn_x_half_range, args.spawn_y_half_range),
            spawn_xy_min_distance=args.spawn_min_distance,
            initial_drop_height=args.initial_drop_height,
            per_object_stack_height=args.per_object_stack_height,
        ),
        physics=PhysicsConfig(
            settle_steps=args.settle_steps,
            solver_iterations=args.solver_iterations,
            collision_scale_multiplier=args.collision_scale_multiplier,
        ),
        validation=ValidationConfig(
            min_target_image_fraction=args.min_target_fraction,
            min_target_bbox_fraction=args.min_target_bbox_fraction,
            max_target_bbox_fraction=args.max_target_bbox_fraction,
            max_target_center_offset_fraction=args.max_target_center_offset_fraction,
            target_border_margin_pixels=args.target_border_margin_pixels,
            min_foreground_image_fraction=args.min_foreground_fraction,
            min_foreground_bbox_fraction=args.min_foreground_bbox_fraction,
            max_foreground_bbox_fraction=args.max_foreground_bbox_fraction,
            min_foreground_bbox_fill_fraction=args.min_foreground_bbox_fill_fraction,
            max_contact_penetration_m=args.max_contact_penetration,
            max_object_xy_radius_m=args.max_object_xy_radius,
            max_object_xy_span_m=args.max_object_xy_span,
            max_attempts_per_scene=args.max_attempts_per_scene,
        ),
        robot_camera=RobotCameraConfig(visible=args.show_robot_camera),
    )
    result = generate_dataset_sapien(
        args.dataset_root,
        scene_count=args.scenes,
        seed=args.seed,
        config=config,
        overwrite=args.overwrite,
        asset_limit=args.asset_limit,
    )
    print(f"Generated {len(result.scene_dirs)} SAPIEN scenes in {args.dataset_root / 'scenes'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
