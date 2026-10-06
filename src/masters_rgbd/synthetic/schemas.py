"""JSON-serializable metadata schemas."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from masters_rgbd.synthetic.io_utils import read_json
from masters_rgbd.synthetic.transforms import validate_matrix4


def _float_list(values: list[float] | tuple[float, ...]) -> list[float]:
    return [float(value) for value in values]


@dataclass(frozen=True)
class ObjectAsset:
    object_id: str
    object_dir: Path
    mesh_path: str
    collision_mesh_paths: tuple[str, ...]
    texture_path: str | None
    bbox_min: tuple[float, float, float]
    bbox_max: tuple[float, float, float]
    center: tuple[float, float, float]
    extents: tuple[float, float, float]
    watertight: bool | None

    @classmethod
    def from_json_file(cls, path: Path) -> ObjectAsset:
        data = read_json(path)
        bbox = data.get("bbox", {})
        return cls(
            object_id=str(data["object_id"]),
            object_dir=path.parent,
            mesh_path=str(data["mesh_path"]),
            collision_mesh_paths=tuple(str(item) for item in data.get("collision_mesh_paths", [])),
            texture_path=str(data["texture_path"]) if data.get("texture_path") else None,
            bbox_min=tuple(float(v) for v in bbox["min"]),  # type: ignore[arg-type]
            bbox_max=tuple(float(v) for v in bbox["max"]),  # type: ignore[arg-type]
            center=tuple(float(v) for v in bbox["center"]),  # type: ignore[arg-type]
            extents=tuple(float(v) for v in bbox["extents"]),  # type: ignore[arg-type]
            watertight=data.get("watertight"),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "mesh_path": self.mesh_path,
            "collision_mesh_paths": list(self.collision_mesh_paths),
            "texture_path": self.texture_path,
            "bbox": {
                "min": _float_list(self.bbox_min),
                "max": _float_list(self.bbox_max),
                "center": _float_list(self.center),
                "extents": _float_list(self.extents),
            },
            "watertight": self.watertight,
        }

    @property
    def mesh_file(self) -> Path:
        return self.object_dir / self.mesh_path

    @property
    def collision_mesh_files(self) -> tuple[Path, ...]:
        return tuple(self.object_dir / path for path in self.collision_mesh_paths)

    @property
    def texture_file(self) -> Path | None:
        if self.texture_path is None:
            return None
        return self.object_dir / self.texture_path


@dataclass(frozen=True)
class CameraMetadata:
    width: int
    height: int
    fovy_degrees: float
    intrinsics: dict[str, float]
    world_T_camera: list[list[float]]
    camera_T_world: list[list[float]]

    def to_json(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "fovy_degrees": self.fovy_degrees,
            "intrinsics": {key: float(value) for key, value in self.intrinsics.items()},
            "world_T_camera": self.world_T_camera,
            "camera_T_world": self.camera_T_world,
        }

    @staticmethod
    def validate_json(data: dict[str, Any]) -> None:
        if int(data["width"]) <= 0 or int(data["height"]) <= 0:
            raise ValueError("Camera width/height must be positive")
        intrinsics = data["intrinsics"]
        for key in ("fx", "fy", "cx", "cy"):
            if float(intrinsics[key]) < 0.0:
                raise ValueError(f"Invalid camera intrinsic {key}")
        validate_matrix4(data["world_T_camera"], name="world_T_camera")
        validate_matrix4(data["camera_T_world"], name="camera_T_world")


@dataclass(frozen=True)
class SceneObjectMetadata:
    instance_id: int
    object_id: str
    mesh_path: str
    scale: tuple[float, float, float]
    world_T_object: list[list[float]]
    camera_T_object: list[list[float]]
    visible_pixel_count: int
    visible_image_fraction: float
    visible_bbox_xyxy: tuple[int, int, int, int] | None = None
    visible_bbox_area_fraction: float = 0.0
    visible_bbox_fill_fraction: float = 0.0

    def to_json(self) -> dict[str, Any]:
        return {
            "instance_id": int(self.instance_id),
            "object_id": self.object_id,
            "mesh_path": self.mesh_path,
            "scale": _float_list(self.scale),
            "world_T_object": self.world_T_object,
            "camera_T_object": self.camera_T_object,
            "visible_pixel_count": int(self.visible_pixel_count),
            "visible_image_fraction": float(self.visible_image_fraction),
            "visible_bbox_xyxy": list(self.visible_bbox_xyxy) if self.visible_bbox_xyxy else None,
            "visible_bbox_area_fraction": float(self.visible_bbox_area_fraction),
            "visible_bbox_fill_fraction": float(self.visible_bbox_fill_fraction),
        }


@dataclass(frozen=True)
class SceneMetadata:
    scene_id: str
    units: str
    workspace_frame: str
    workspace_bounds: dict[str, list[float]]
    objects: tuple[SceneObjectMetadata, ...]
    simulator: dict[str, Any]
    target_instance_id: int | None = None
    background: dict[str, Any] | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "units": self.units,
            "workspace_frame": self.workspace_frame,
            "workspace_bounds": self.workspace_bounds,
            "background": self.background,
            "target_instance_id": self.target_instance_id,
            "objects": [item.to_json() for item in self.objects],
            "simulator": self.simulator,
        }

    @staticmethod
    def validate_json(data: dict[str, Any]) -> None:
        if not data.get("scene_id"):
            raise ValueError("scene_id is required")
        bounds = data["workspace_bounds"]
        if len(bounds["min"]) != 3 or len(bounds["max"]) != 3:
            raise ValueError("workspace_bounds must contain min/max vector3")
        target_instance_id = data.get("target_instance_id")
        object_ids = {int(obj["instance_id"]) for obj in data.get("objects", [])}
        if target_instance_id is not None and int(target_instance_id) not in object_ids:
            raise ValueError("target_instance_id must reference one of the scene objects")
        for obj in data.get("objects", []):
            if int(obj["instance_id"]) <= 0:
                raise ValueError("instance_id must be positive")
            validate_matrix4(obj["world_T_object"], name="world_T_object")
            validate_matrix4(obj["camera_T_object"], name="camera_T_object")
