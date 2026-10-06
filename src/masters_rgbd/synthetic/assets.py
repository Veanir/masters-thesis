"""Asset preparation for object directories used by the dataset generators."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from masters_rgbd.synthetic.errors import AssetError, DependencyMissingError
from masters_rgbd.synthetic.io_utils import ensure_clean_dir, safe_copy_file, write_json
from masters_rgbd.synthetic.schemas import ObjectAsset

_SAFE_ID_PATTERN = re.compile(r"[^a-zA-Z0-9_.-]+")


@dataclass(frozen=True)
class SourceObject:
    object_id: str
    source_dir: Path
    visual_mesh: Path
    collision_meshes: tuple[Path, ...]
    texture: Path | None
    original_xml: Path | None


@dataclass(frozen=True)
class SourceTrellisObject:
    object_id: str
    glb_file: Path


def sanitize_object_id(value: str) -> str:
    sanitized = _SAFE_ID_PATTERN.sub("_", value.strip()).strip("._-")
    if not sanitized:
        raise AssetError(f"Cannot build object_id from {value!r}")
    return sanitized


def scan_source_objects(source_root: Path) -> list[SourceObject]:
    """Find object directories matching the mujoco_scanned_objects layout."""

    if not source_root.is_dir():
        raise AssetError(f"Source root does not exist or is not a directory: {source_root}")

    objects: list[SourceObject] = []
    seen_ids: set[str] = set()
    for directory in sorted(path for path in source_root.iterdir() if path.is_dir()):
        visual_mesh = directory / "model.obj"
        if not visual_mesh.is_file():
            continue
        object_id = sanitize_object_id(directory.name)
        if object_id in seen_ids:
            raise AssetError(f"Duplicate sanitized object_id {object_id!r}")
        seen_ids.add(object_id)
        collision_meshes = tuple(sorted(directory.glob("model_collision_*.obj")))
        texture = directory / "texture.png"
        original_xml = directory / "model.xml"
        objects.append(
            SourceObject(
                object_id=object_id,
                source_dir=directory,
                visual_mesh=visual_mesh,
                collision_meshes=collision_meshes,
                texture=texture if texture.is_file() else None,
                original_xml=original_xml if original_xml.is_file() else None,
            )
        )
    if not objects:
        raise AssetError(f"No objects with model.obj found under {source_root}")
    return objects


def scan_trellis_glb_objects(source_root: Path) -> list[SourceTrellisObject]:
    """Find flat or nested TRELLIS.2 GLB assets."""

    if not source_root.is_dir():
        raise AssetError(f"Source root does not exist or is not a directory: {source_root}")

    objects: list[SourceTrellisObject] = []
    seen_ids: set[str] = set()
    for glb_file in sorted(source_root.rglob("*.glb")):
        object_id = sanitize_object_id(glb_file.stem)
        if object_id in seen_ids:
            raise AssetError(f"Duplicate sanitized object_id {object_id!r}")
        seen_ids.add(object_id)
        objects.append(SourceTrellisObject(object_id=object_id, glb_file=glb_file))
    if not objects:
        raise AssetError(f"No .glb files found under {source_root}")
    return objects


def _load_mesh_bounds(
    mesh_path: Path,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...], tuple[float, ...], bool | None]:
    mesh = _load_mesh_for_geometry(mesh_path)
    bounds = mesh.bounds
    bbox_min = tuple(float(value) for value in bounds[0])
    bbox_max = tuple(float(value) for value in bounds[1])
    extents = tuple(float(value) for value in mesh.extents)
    center = tuple(float(value) for value in mesh.bounding_box.centroid)
    watertight = bool(mesh.is_watertight) if hasattr(mesh, "is_watertight") else None
    return bbox_min, bbox_max, center, extents, watertight


def _load_mesh_for_geometry(mesh_path: Path):
    try:
        import trimesh
    except ImportError as exc:  # pragma: no cover - exercised before uv sync only.
        raise DependencyMissingError("trimesh is required to prepare assets") from exc

    loaded = trimesh.load(mesh_path, process=False)
    mesh = loaded.to_geometry() if isinstance(loaded, trimesh.Scene) else loaded
    if mesh.is_empty:
        raise AssetError(f"Mesh is empty: {mesh_path}")
    return mesh


def prepare_assets(
    source_root: Path,
    output_objects_dir: Path,
    *,
    limit: int | None = None,
    overwrite: bool = False,
) -> list[ObjectAsset]:
    """Copy scanned object assets into the dataset-owned object directory."""

    source_objects = scan_source_objects(source_root)
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        source_objects = source_objects[:limit]

    output_objects_dir.mkdir(parents=True, exist_ok=True)
    prepared: list[ObjectAsset] = []
    for source in source_objects:
        object_dir = output_objects_dir / source.object_id
        ensure_clean_dir(object_dir, overwrite=overwrite)

        safe_copy_file(source.visual_mesh, object_dir / "mesh.obj")
        texture_path = None
        if source.texture is not None:
            safe_copy_file(source.texture, object_dir / "texture.png")
            texture_path = "texture.png"

        collision_paths: list[str] = []
        for index, collision_mesh in enumerate(source.collision_meshes):
            target_name = f"collision_{index:03d}.obj"
            safe_copy_file(collision_mesh, object_dir / target_name)
            collision_paths.append(target_name)

        if source.original_xml is not None:
            safe_copy_file(source.original_xml, object_dir / "source_model.xml")

        bbox_min, bbox_max, center, extents, watertight = _load_mesh_bounds(object_dir / "mesh.obj")
        metadata = ObjectAsset(
            object_id=source.object_id,
            object_dir=object_dir,
            mesh_path="mesh.obj",
            collision_mesh_paths=tuple(collision_paths),
            texture_path=texture_path,
            bbox_min=bbox_min,  # type: ignore[arg-type]
            bbox_max=bbox_max,  # type: ignore[arg-type]
            center=center,  # type: ignore[arg-type]
            extents=extents,  # type: ignore[arg-type]
            watertight=watertight,
        )
        write_json(object_dir / "metadata.json", metadata.to_json())
        prepared.append(metadata)
    return prepared


def prepare_trellis_glb_assets(
    source_root: Path,
    output_objects_dir: Path,
    *,
    limit: int | None = None,
    overwrite: bool = False,
) -> list[ObjectAsset]:
    """Copy TRELLIS.2 GLB assets and prepare lightweight box collision meshes."""

    source_objects = scan_trellis_glb_objects(source_root)
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        source_objects = source_objects[:limit]

    output_objects_dir.mkdir(parents=True, exist_ok=True)
    prepared: list[ObjectAsset] = []
    for source in source_objects:
        object_dir = output_objects_dir / source.object_id
        ensure_clean_dir(object_dir, overwrite=overwrite)

        visual_path = object_dir / "mesh.glb"
        collision_path = object_dir / "collision_000.obj"
        safe_copy_file(source.glb_file, visual_path)

        mesh = _load_mesh_for_geometry(visual_path)
        collision_mesh = mesh.bounding_box.to_mesh()
        collision_mesh.export(collision_path)

        bbox_min, bbox_max, center, extents, watertight = _load_mesh_bounds(visual_path)
        metadata = ObjectAsset(
            object_id=source.object_id,
            object_dir=object_dir,
            mesh_path="mesh.glb",
            collision_mesh_paths=("collision_000.obj",),
            texture_path=None,
            bbox_min=bbox_min,  # type: ignore[arg-type]
            bbox_max=bbox_max,  # type: ignore[arg-type]
            center=center,  # type: ignore[arg-type]
            extents=extents,  # type: ignore[arg-type]
            watertight=watertight,
        )
        json_data = metadata.to_json()
        json_data["source_format"] = "trellis_glb"
        json_data["source_path"] = str(source.glb_file)
        write_json(object_dir / "metadata.json", json_data)
        prepared.append(metadata)
    return prepared


def load_prepared_assets(objects_dir: Path, *, limit: int | None = None) -> list[ObjectAsset]:
    if not objects_dir.is_dir():
        raise AssetError(f"Prepared objects directory does not exist: {objects_dir}")
    metadata_paths = sorted(objects_dir.glob("*/metadata.json"))
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        metadata_paths = metadata_paths[:limit]
    if not metadata_paths:
        raise AssetError(f"No prepared object metadata found in {objects_dir}")

    assets: list[ObjectAsset] = []
    for metadata_path in metadata_paths:
        asset = ObjectAsset.from_json_file(metadata_path)
        if not asset.mesh_file.is_file():
            raise AssetError(f"Missing mesh for object {asset.object_id}: {asset.mesh_file}")
        for collision_mesh in asset.collision_mesh_files:
            if not collision_mesh.is_file():
                raise AssetError(
                    f"Missing collision mesh for object {asset.object_id}: {collision_mesh}"
                )
        if asset.texture_file is not None and not asset.texture_file.is_file():
            raise AssetError(f"Missing texture for object {asset.object_id}: {asset.texture_file}")
        assets.append(asset)
    return assets
