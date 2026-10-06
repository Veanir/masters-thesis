"""Generated dataset validation."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from masters_rgbd.synthetic.config import ValidationConfig
from masters_rgbd.synthetic.io_utils import path_from_dataset_root, read_json
from masters_rgbd.synthetic.mask_stats import compute_binary_mask_stats, compute_instance_mask_stats
from masters_rgbd.synthetic.schemas import CameraMetadata, SceneMetadata


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    path: Path
    message: str

    @property
    def is_error(self) -> bool:
        return self.severity == "error"

    def format(self, root: Path | None = None) -> str:
        path = self.path
        if root is not None:
            with suppress(ValueError):
                path = path.relative_to(root)
        return f"[{self.severity}] {path}: {self.message}"


@dataclass(frozen=True)
class ValidationReport:
    issues: tuple[ValidationIssue, ...]

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.is_error)

    @property
    def warning_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "warning")

    @property
    def ok(self) -> bool:
        return self.error_count == 0

    def extend(self, other: ValidationReport) -> ValidationReport:
        return ValidationReport(self.issues + other.issues)


def summarize_validation_report(
    report: ValidationReport,
    root: Path | None = None,
    *,
    limit: int = 5,
) -> str:
    if limit <= 0:
        raise ValueError("limit must be positive")
    issues = [issue.format(root) for issue in report.issues]
    if not issues:
        return "no validation issues reported"
    visible = issues[:limit]
    if len(issues) > limit:
        visible.append(f"... {len(issues) - limit} more")
    return "; ".join(visible)


def validate_scene(
    scene_dir: Path,
    dataset_root: Path,
    config: ValidationConfig | None = None,
) -> ValidationReport:
    config = config or ValidationConfig()
    issues: list[ValidationIssue] = []

    def error(path: Path, message: str) -> None:
        issues.append(ValidationIssue("error", path, message))

    def warning(path: Path, message: str) -> None:
        issues.append(ValidationIssue("warning", path, message))

    required_files = ["rgb.png", "depth.npy", "instance_mask.png", "camera.json", "scene.json"]
    for file_name in required_files:
        path = scene_dir / file_name
        if not path.is_file():
            error(path, "missing required file")
    if any(issue.is_error for issue in issues):
        return ValidationReport(tuple(issues))

    try:
        camera_json = read_json(scene_dir / "camera.json")
        CameraMetadata.validate_json(camera_json)
    except Exception as exc:
        error(scene_dir / "camera.json", str(exc))
        camera_json = None

    try:
        scene_json = read_json(scene_dir / "scene.json")
        SceneMetadata.validate_json(scene_json)
    except Exception as exc:
        error(scene_dir / "scene.json", str(exc))
        scene_json = None

    try:
        rgb = np.asarray(Image.open(scene_dir / "rgb.png"))
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            error(scene_dir / "rgb.png", f"expected RGB image, got shape {rgb.shape}")
    except Exception as exc:
        error(scene_dir / "rgb.png", str(exc))
        rgb = None

    try:
        depth = np.load(scene_dir / "depth.npy")
        if depth.ndim != 2:
            error(scene_dir / "depth.npy", f"expected 2D depth, got shape {depth.shape}")
        elif not np.issubdtype(depth.dtype, np.floating):
            error(scene_dir / "depth.npy", f"expected floating depth, got {depth.dtype}")
        elif not np.isfinite(depth).all():
            error(scene_dir / "depth.npy", "depth contains non-finite values")
        elif not np.any(depth > 0.0):
            error(scene_dir / "depth.npy", "depth does not contain positive values")
    except Exception as exc:
        error(scene_dir / "depth.npy", str(exc))
        depth = None

    try:
        instance_mask = np.asarray(Image.open(scene_dir / "instance_mask.png"))
        if instance_mask.ndim != 2:
            error(
                scene_dir / "instance_mask.png",
                f"expected 2D mask, got shape {instance_mask.shape}",
            )
    except Exception as exc:
        error(scene_dir / "instance_mask.png", str(exc))
        instance_mask = None

    if camera_json is not None and rgb is not None:
        expected_shape = (int(camera_json["height"]), int(camera_json["width"]))
        if rgb.shape[:2] != expected_shape:
            error(
                scene_dir / "rgb.png",
                f"shape {rgb.shape[:2]} does not match camera {expected_shape}",
            )
    if rgb is not None and depth is not None and depth.shape != rgb.shape[:2]:
        error(scene_dir / "depth.npy", "depth shape does not match rgb shape")
    if rgb is not None and instance_mask is not None and instance_mask.shape != rgb.shape[:2]:
        error(scene_dir / "instance_mask.png", "mask shape does not match rgb shape")

    if scene_json is not None:
        objects = scene_json.get("objects", [])
        expected_ids = {int(item["instance_id"]) for item in objects}
        target_instance_id = scene_json.get("target_instance_id")
        target_instance_id = int(target_instance_id) if target_instance_id is not None else None
        contact_quality = scene_json.get("simulator", {}).get("contact_quality")
        if contact_quality is not None:
            max_penetration = float(contact_quality.get("max_penetration_m", 0.0))
            if max_penetration > config.max_contact_penetration_m:
                error(
                    scene_dir / "scene.json",
                    (
                        f"max contact penetration is {max_penetration:.6f} m, "
                        f"maximum is {config.max_contact_penetration_m:.6f} m"
                    ),
                )
        for item in objects:
            mesh_path = str(item["mesh_path"])
            try:
                mesh_file = path_from_dataset_root(dataset_root, mesh_path)
            except ValueError as exc:
                error(scene_dir / "scene.json", str(exc))
                continue
            if not mesh_file.is_file():
                error(scene_dir / "scene.json", f"mesh_path does not exist: {mesh_path}")
        object_xy = _object_xy_positions(objects)
        if object_xy.size:
            if config.max_object_xy_radius_m is not None:
                radii = np.linalg.norm(object_xy, axis=1)
                max_radius = float(np.max(radii))
                if max_radius > config.max_object_xy_radius_m:
                    error(
                        scene_dir / "scene.json",
                        (
                            f"object XY radius is {max_radius:.3f} m, "
                            f"maximum is {config.max_object_xy_radius_m:.3f} m"
                        ),
                    )
            if config.max_object_xy_span_m is not None:
                spans = object_xy.max(axis=0) - object_xy.min(axis=0)
                max_span = float(np.max(spans))
                if max_span > config.max_object_xy_span_m:
                    error(
                        scene_dir / "scene.json",
                        (
                            f"object XY span is {max_span:.3f} m, "
                            f"maximum is {config.max_object_xy_span_m:.3f} m"
                        ),
                    )

        if instance_mask is not None:
            image_area = float(instance_mask.size)
            foreground_stats = compute_binary_mask_stats(instance_mask > 0)
            if foreground_stats.image_fraction < config.min_foreground_image_fraction:
                message = (
                    f"foreground occupies {foreground_stats.image_fraction:.4f} of the image, "
                    f"minimum is {config.min_foreground_image_fraction:.4f}"
                )
                if config.require_visible_instances:
                    error(scene_dir / "instance_mask.png", message)
                else:
                    warning(scene_dir / "instance_mask.png", message)
            if foreground_stats.bbox_area_fraction < config.min_foreground_bbox_fraction:
                message = (
                    f"foreground bbox occupies {foreground_stats.bbox_area_fraction:.4f} of "
                    f"the image, "
                    f"minimum is {config.min_foreground_bbox_fraction:.4f}"
                )
                if config.require_visible_instances:
                    error(scene_dir / "instance_mask.png", message)
                else:
                    warning(scene_dir / "instance_mask.png", message)
            if foreground_stats.bbox_area_fraction > config.max_foreground_bbox_fraction:
                message = (
                    f"foreground bbox occupies {foreground_stats.bbox_area_fraction:.4f} of "
                    f"the image, "
                    f"maximum is {config.max_foreground_bbox_fraction:.4f}"
                )
                if config.require_visible_instances:
                    error(scene_dir / "instance_mask.png", message)
                else:
                    warning(scene_dir / "instance_mask.png", message)
            if foreground_stats.bbox_fill_fraction < config.min_foreground_bbox_fill_fraction:
                message = (
                    f"foreground fills {foreground_stats.bbox_fill_fraction:.4f} of its bbox, "
                    f"minimum is {config.min_foreground_bbox_fill_fraction:.4f}"
                )
                if config.require_visible_instances:
                    error(scene_dir / "instance_mask.png", message)
                else:
                    warning(scene_dir / "instance_mask.png", message)

            observed_ids = {int(value) for value in np.unique(instance_mask)}
            unknown_ids = observed_ids - expected_ids - {0}
            if unknown_ids:
                error(
                    scene_dir / "instance_mask.png",
                    f"mask contains unknown instance ids: {sorted(unknown_ids)}",
                )
            for instance_id in sorted(expected_ids):
                instance_stats = compute_instance_mask_stats(instance_mask, instance_id)
                pixel_count = instance_stats.pixel_count
                pixel_fraction = pixel_count / image_area if image_area else 0.0
                if (
                    pixel_count < config.min_instance_pixels
                    or pixel_fraction < config.min_instance_image_fraction
                ):
                    message = (
                        f"instance {instance_id} has {pixel_count} visible pixels "
                        f"({pixel_fraction:.4f} of image), minimum is "
                        f"{config.min_instance_pixels} pixels and "
                        f"{config.min_instance_image_fraction:.4f} image fraction"
                    )
                    if config.require_visible_instances:
                        error(scene_dir / "instance_mask.png", message)
                    else:
                        warning(scene_dir / "instance_mask.png", message)
                is_target = target_instance_id == instance_id
                if is_target:
                    height, width = instance_mask.shape
                    if pixel_fraction < config.min_target_image_fraction:
                        message = (
                            f"target instance {instance_id} occupies {pixel_fraction:.4f} "
                            f"of image, "
                            f"minimum is {config.min_target_image_fraction:.4f}"
                        )
                        if config.require_visible_instances:
                            error(scene_dir / "instance_mask.png", message)
                        else:
                            warning(scene_dir / "instance_mask.png", message)
                    if instance_stats.bbox_area_fraction < config.min_target_bbox_fraction:
                        message = (
                            f"target instance {instance_id} bbox occupies "
                            f"{instance_stats.bbox_area_fraction:.4f} of image, "
                            f"minimum is {config.min_target_bbox_fraction:.4f}"
                        )
                        if config.require_visible_instances:
                            error(scene_dir / "instance_mask.png", message)
                        else:
                            warning(scene_dir / "instance_mask.png", message)
                    if instance_stats.bbox_area_fraction > config.max_target_bbox_fraction:
                        message = (
                            f"target instance {instance_id} bbox occupies "
                            f"{instance_stats.bbox_area_fraction:.4f} of image, "
                            f"maximum is {config.max_target_bbox_fraction:.4f}"
                        )
                        if config.require_visible_instances:
                            error(scene_dir / "instance_mask.png", message)
                        else:
                            warning(scene_dir / "instance_mask.png", message)
                    center_offset = instance_stats.center_offset_fraction(
                        height=height, width=width
                    )
                    if center_offset > config.max_target_center_offset_fraction:
                        message = (
                            f"target instance {instance_id} bbox center offset is "
                            f"{center_offset:.4f}, "
                            f"maximum is {config.max_target_center_offset_fraction:.4f}"
                        )
                        if config.require_visible_instances:
                            error(scene_dir / "instance_mask.png", message)
                        else:
                            warning(scene_dir / "instance_mask.png", message)

                border_margin = (
                    config.target_border_margin_pixels if is_target else config.border_margin_pixels
                )
                if pixel_count > 0 and border_margin > 0:
                    height, width = instance_mask.shape
                    if instance_stats.touches_border(
                        height=height,
                        width=width,
                        margin=border_margin,
                    ):
                        message = (
                            f"instance {instance_id} is too close to image border; "
                            f"required margin is {border_margin} px"
                        )
                        if config.require_visible_instances:
                            error(scene_dir / "instance_mask.png", message)
                        else:
                            warning(scene_dir / "instance_mask.png", message)

    return ValidationReport(tuple(issues))


def _object_xy_positions(objects: list[dict[str, object]]) -> np.ndarray:
    positions: list[tuple[float, float]] = []
    for item in objects:
        matrix = item.get("world_T_object")
        if not isinstance(matrix, list) or len(matrix) < 2:
            continue
        try:
            positions.append((float(matrix[0][3]), float(matrix[1][3])))  # type: ignore[index]
        except (TypeError, ValueError, IndexError):
            continue
    return np.asarray(positions, dtype=np.float64)
