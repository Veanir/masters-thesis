"""Background surface assets for SAPIEN table rendering."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from masters_rgbd.synthetic.assets import sanitize_object_id
from masters_rgbd.synthetic.errors import AssetError, DependencyMissingError
from masters_rgbd.synthetic.io_utils import ensure_clean_dir, read_json, write_json


@dataclass(frozen=True)
class SurfaceAsset:
    surface_id: str
    surface_dir: Path
    base_color_path: str
    roughness_path: str
    metallic_path: str
    normal_path: str
    base_color: tuple[float, float, float, float]
    roughness: float
    metallic: float
    specular: float
    description: str
    source: str

    @classmethod
    def from_json_file(cls, path: Path) -> SurfaceAsset:
        data = read_json(path)
        return cls(
            surface_id=str(data["surface_id"]),
            surface_dir=path.parent,
            base_color_path=str(data["base_color_path"]),
            roughness_path=str(data["roughness_path"]),
            metallic_path=str(data["metallic_path"]),
            normal_path=str(data["normal_path"]),
            base_color=tuple(float(value) for value in data["base_color"]),  # type: ignore[arg-type]
            roughness=float(data["roughness"]),
            metallic=float(data["metallic"]),
            specular=float(data["specular"]),
            description=str(data.get("description", "")),
            source=str(data.get("source", "unknown")),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "surface_id": self.surface_id,
            "base_color_path": self.base_color_path,
            "roughness_path": self.roughness_path,
            "metallic_path": self.metallic_path,
            "normal_path": self.normal_path,
            "base_color": [float(value) for value in self.base_color],
            "roughness": float(self.roughness),
            "metallic": float(self.metallic),
            "specular": float(self.specular),
            "description": self.description,
            "source": self.source,
        }

    @property
    def base_color_file(self) -> Path:
        return self.surface_dir / self.base_color_path

    @property
    def roughness_file(self) -> Path:
        return self.surface_dir / self.roughness_path

    @property
    def metallic_file(self) -> Path:
        return self.surface_dir / self.metallic_path

    @property
    def normal_file(self) -> Path:
        return self.surface_dir / self.normal_path


@dataclass(frozen=True)
class MeshBackgroundAsset:
    background_id: str
    background_dir: Path
    mesh_path: str
    bbox_min: tuple[float, float, float]
    bbox_max: tuple[float, float, float]
    center: tuple[float, float, float]
    extents: tuple[float, float, float]
    watertight: bool | None
    description: str
    source: str
    source_path: str | None

    @classmethod
    def from_json_file(cls, path: Path) -> MeshBackgroundAsset:
        data = read_json(path)
        bbox = data.get("bbox", {})
        return cls(
            background_id=str(data["background_id"]),
            background_dir=path.parent,
            mesh_path=str(data["mesh_path"]),
            bbox_min=tuple(float(v) for v in bbox["min"]),  # type: ignore[arg-type]
            bbox_max=tuple(float(v) for v in bbox["max"]),  # type: ignore[arg-type]
            center=tuple(float(v) for v in bbox["center"]),  # type: ignore[arg-type]
            extents=tuple(float(v) for v in bbox["extents"]),  # type: ignore[arg-type]
            watertight=data.get("watertight"),
            description=str(data.get("description", "")),
            source=str(data.get("source", "unknown")),
            source_path=str(data["source_path"]) if data.get("source_path") else None,
        )

    def to_json(self) -> dict[str, Any]:
        data = {
            "background_id": self.background_id,
            "kind": "trellis_mesh",
            "mesh_path": self.mesh_path,
            "bbox": {
                "min": [float(value) for value in self.bbox_min],
                "max": [float(value) for value in self.bbox_max],
                "center": [float(value) for value in self.center],
                "extents": [float(value) for value in self.extents],
            },
            "watertight": self.watertight,
            "description": self.description,
            "source": self.source,
        }
        if self.source_path is not None:
            data["source_path"] = self.source_path
        return data

    @property
    def mesh_file(self) -> Path:
        return self.background_dir / self.mesh_path


@dataclass(frozen=True)
class _SurfaceSpec:
    surface_id: str
    base_rgb: tuple[int, int, int]
    roughness: float
    metallic: float
    specular: float
    style: str
    description: str


_SURFACE_SPECS: tuple[_SurfaceSpec, ...] = (
    _SurfaceSpec(
        "gray_lab_bench",
        (142, 147, 145),
        0.78,
        0.0,
        0.12,
        "speckled",
        "matte gray robotics lab bench",
    ),
    _SurfaceSpec(
        "white_laminate",
        (205, 207, 199),
        0.66,
        0.0,
        0.16,
        "laminate",
        "slightly worn white laminate tabletop",
    ),
    _SurfaceSpec(
        "brushed_steel", (150, 152, 150), 0.42, 1.0, 0.38, "brushed", "brushed steel work surface"
    ),
    _SurfaceSpec(
        "green_cutting_mat",
        (70, 132, 83),
        0.82,
        0.0,
        0.10,
        "grid_mat",
        "green cutting mat with subtle grid",
    ),
    _SurfaceSpec(
        "black_rubber", (38, 39, 38), 0.88, 0.0, 0.08, "rubber", "dark anti-slip rubber work mat"
    ),
    _SurfaceSpec(
        "wood_workbench", (145, 111, 72), 0.62, 0.0, 0.18, "wood", "worn wooden workbench surface"
    ),
    _SurfaceSpec(
        "cardboard_sheet",
        (166, 138, 95),
        0.90,
        0.0,
        0.05,
        "cardboard",
        "recycled cardboard sheet surface",
    ),
    _SurfaceSpec(
        "blue_plastic_tray",
        (76, 113, 150),
        0.55,
        0.0,
        0.22,
        "plastic",
        "blue molded plastic tray floor",
    ),
    _SurfaceSpec(
        "dark_conveyor_belt",
        (45, 47, 48),
        0.84,
        0.0,
        0.08,
        "belt",
        "dark conveyor belt rubber texture",
    ),
    _SurfaceSpec(
        "concrete_floor",
        (118, 120, 116),
        0.86,
        0.0,
        0.06,
        "concrete",
        "smooth gray concrete floor patch",
    ),
)


def load_surface_assets(
    backgrounds_dir: Path, *, surface_id: str | None = None
) -> list[SurfaceAsset]:
    if not backgrounds_dir.is_dir():
        if surface_id is not None:
            raise AssetError(f"Background surface directory does not exist: {backgrounds_dir}")
        return []

    metadata_paths = sorted(backgrounds_dir.glob("*/metadata.json"))
    if surface_id is not None:
        metadata_paths = [path for path in metadata_paths if path.parent.name == surface_id]
        if not metadata_paths:
            raise AssetError(f"Surface {surface_id!r} was not found under {backgrounds_dir}")

    surfaces: list[SurfaceAsset] = []
    for metadata_path in metadata_paths:
        surface = SurfaceAsset.from_json_file(metadata_path)
        for file in (
            surface.base_color_file,
            surface.roughness_file,
            surface.metallic_file,
            surface.normal_file,
        ):
            if not file.is_file():
                raise AssetError(
                    f"Missing background surface texture for {surface.surface_id}: {file}"
                )
        surfaces.append(surface)
    return surfaces


def load_mesh_background_assets(
    backgrounds_dir: Path,
    *,
    background_id: str | None = None,
) -> list[MeshBackgroundAsset]:
    if not backgrounds_dir.is_dir():
        if background_id is not None:
            raise AssetError(f"Background mesh directory does not exist: {backgrounds_dir}")
        return []

    metadata_paths = sorted(backgrounds_dir.glob("*/metadata.json"))
    if background_id is not None:
        metadata_paths = [path for path in metadata_paths if path.parent.name == background_id]
        if not metadata_paths:
            raise AssetError(
                f"Background mesh {background_id!r} was not found under {backgrounds_dir}"
            )

    assets: list[MeshBackgroundAsset] = []
    for metadata_path in metadata_paths:
        data = read_json(metadata_path)
        if data.get("kind") != "trellis_mesh":
            continue
        asset = MeshBackgroundAsset.from_json_file(metadata_path)
        if not asset.mesh_file.is_file():
            raise AssetError(
                f"Missing background mesh for {asset.background_id}: {asset.mesh_file}"
            )
        assets.append(asset)
    return assets


def prepare_trellis_background_assets(
    source_root: Path,
    output_dir: Path,
    *,
    limit: int | None = None,
    overwrite: bool = False,
) -> list[MeshBackgroundAsset]:
    if not source_root.is_dir():
        raise AssetError(f"Source root does not exist or is not a directory: {source_root}")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")

    glb_files = sorted(source_root.rglob("*.glb"))
    if limit is not None:
        glb_files = glb_files[:limit]
    if not glb_files:
        raise AssetError(f"No .glb background meshes found under {source_root}")

    output_dir.mkdir(parents=True, exist_ok=True)
    prepared: list[MeshBackgroundAsset] = []
    seen_ids: set[str] = set()
    for glb_file in glb_files:
        background_id = sanitize_object_id(glb_file.stem)
        if background_id in seen_ids:
            raise AssetError(f"Duplicate sanitized background_id {background_id!r}")
        seen_ids.add(background_id)
        background_dir = output_dir / background_id
        ensure_clean_dir(background_dir, overwrite=overwrite)
        target_mesh = background_dir / "mesh.glb"
        target_mesh.write_bytes(glb_file.read_bytes())

        bbox_min, bbox_max, center, extents, watertight = _load_mesh_bounds(target_mesh)
        asset = MeshBackgroundAsset(
            background_id=background_id,
            background_dir=background_dir,
            mesh_path="mesh.glb",
            bbox_min=bbox_min,  # type: ignore[arg-type]
            bbox_max=bbox_max,  # type: ignore[arg-type]
            center=center,  # type: ignore[arg-type]
            extents=extents,  # type: ignore[arg-type]
            watertight=watertight,
            description=f"TRELLIS background mesh {background_id}",
            source="trellis_glb_background",
            source_path=str(glb_file),
        )
        write_json(background_dir / "metadata.json", asset.to_json())
        prepared.append(asset)
    return prepared


def prepare_procedural_backgrounds(
    output_dir: Path,
    *,
    seed: int = 0,
    size: int = 512,
    overwrite: bool = False,
    surface_ids: tuple[str, ...] | None = None,
) -> list[SurfaceAsset]:
    if size <= 0:
        raise ValueError("size must be positive")

    specs = _select_specs(surface_ids)
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared: list[SurfaceAsset] = []
    for spec in specs:
        surface_dir = output_dir / spec.surface_id
        ensure_clean_dir(surface_dir, overwrite=overwrite)
        rng = np.random.default_rng(_stable_seed(seed, spec.surface_id))
        maps = _generate_surface_maps(spec, rng=rng, size=size)

        Image.fromarray(maps["base_color"], mode="RGB").save(surface_dir / "base_color.png")
        Image.fromarray(maps["roughness"], mode="L").save(surface_dir / "roughness.png")
        Image.fromarray(maps["metallic"], mode="L").save(surface_dir / "metallic.png")
        Image.fromarray(maps["normal"], mode="RGB").save(surface_dir / "normal.png")

        surface = SurfaceAsset(
            surface_id=spec.surface_id,
            surface_dir=surface_dir,
            base_color_path="base_color.png",
            roughness_path="roughness.png",
            metallic_path="metallic.png",
            normal_path="normal.png",
            base_color=(
                spec.base_rgb[0] / 255.0,
                spec.base_rgb[1] / 255.0,
                spec.base_rgb[2] / 255.0,
                1.0,
            ),
            roughness=spec.roughness,
            metallic=spec.metallic,
            specular=spec.specular,
            description=spec.description,
            source="procedural_v1",
        )
        write_json(surface_dir / "metadata.json", surface.to_json())
        prepared.append(surface)
    return prepared


def available_procedural_surface_ids() -> tuple[str, ...]:
    return tuple(spec.surface_id for spec in _SURFACE_SPECS)


def _select_specs(surface_ids: tuple[str, ...] | None) -> tuple[_SurfaceSpec, ...]:
    if surface_ids is None:
        return _SURFACE_SPECS
    requested = set(surface_ids)
    specs = tuple(spec for spec in _SURFACE_SPECS if spec.surface_id in requested)
    missing = requested - {spec.surface_id for spec in specs}
    if missing:
        raise AssetError(f"Unknown procedural surface ids: {sorted(missing)}")
    return specs


def _load_mesh_bounds(
    mesh_path: Path,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...], tuple[float, ...], bool | None]:
    try:
        import trimesh
    except ImportError as exc:  # pragma: no cover - project dependency in normal env.
        raise DependencyMissingError(
            "trimesh is required to prepare TRELLIS background assets"
        ) from exc

    loaded = trimesh.load(mesh_path, process=False)
    mesh = loaded.to_geometry() if isinstance(loaded, trimesh.Scene) else loaded
    if mesh.is_empty:
        raise AssetError(f"Background mesh is empty: {mesh_path}")
    bounds = mesh.bounds
    return (
        tuple(float(value) for value in bounds[0]),
        tuple(float(value) for value in bounds[1]),
        tuple(float(value) for value in mesh.bounding_box.centroid),
        tuple(float(value) for value in mesh.extents),
        bool(mesh.is_watertight) if hasattr(mesh, "is_watertight") else None,
    )


def _stable_seed(seed: int, surface_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{surface_id}".encode()).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _generate_surface_maps(
    spec: _SurfaceSpec,
    *,
    rng: np.random.Generator,
    size: int,
) -> dict[str, np.ndarray]:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    base = np.full((size, size, 3), spec.base_rgb, dtype=np.float32)
    height = np.zeros((size, size), dtype=np.float32)
    roughness = np.full((size, size), spec.roughness, dtype=np.float32)

    noise = rng.normal(0.0, 1.0, size=(size, size)).astype(np.float32)
    fine_noise = rng.normal(0.0, 1.0, size=(size, size)).astype(np.float32)
    height += 0.012 * noise + 0.004 * fine_noise

    if spec.style == "speckled":
        base += noise[..., None] * 5.0
        _draw_scratches(base, height, rng, count=55, color_delta=18, size=size)
    elif spec.style == "laminate":
        base += noise[..., None] * 3.0
        _draw_scratches(base, height, rng, count=32, color_delta=-10, size=size)
    elif spec.style == "brushed":
        streaks = rng.normal(0.0, 1.0, size=(size, 1)).astype(np.float32)
        base += streaks[..., None] * 10.0
        base += np.sin(yy[..., None] * 0.22) * 4.0
        height += streaks * 0.018
        roughness += rng.normal(0.0, 0.04, size=(size, size)).astype(np.float32)
    elif spec.style == "grid_mat":
        base += noise[..., None] * 5.0
        grid_step = max(32, size // 8)
        line_mask = ((xx % grid_step) < 2.0) | ((yy % grid_step) < 2.0)
        base[line_mask] *= 0.72
        height[line_mask] -= 0.04
    elif spec.style == "rubber":
        dots = np.sin(xx * 0.42) * np.sin(yy * 0.42) > 0.82
        base += noise[..., None] * 4.0
        base[dots] += 9.0
        height[dots] += 0.035
    elif spec.style == "wood":
        grain = np.sin(xx * 0.045 + np.sin(yy * 0.022) * 3.0)
        rings = np.sin(xx * 0.011 + yy * 0.006)
        base += (grain[..., None] * 18.0) + (rings[..., None] * 8.0) + noise[..., None] * 4.0
        height += grain * 0.025
    elif spec.style == "cardboard":
        fibers = np.sin(xx * 0.35 + rng.normal(0.0, 0.2, size=(size, size))).astype(np.float32)
        base += fibers[..., None] * 8.0 + noise[..., None] * 7.0
        height += fibers * 0.02
    elif spec.style == "plastic":
        base += noise[..., None] * 2.5
        waves = np.sin(xx * 0.035) + np.sin(yy * 0.041)
        height += waves * 0.01
        roughness += waves * 0.015
    elif spec.style == "belt":
        ribs = np.sin(yy * 0.36) > 0.78
        base += noise[..., None] * 3.0
        base[ribs] += 7.0
        height[ribs] += 0.032
    elif spec.style == "concrete":
        stains = _soft_value_noise(rng, size=size, coarse=16)
        base += noise[..., None] * 4.5 + stains[..., None] * 18.0
        height += stains * 0.018

    base = np.clip(base, 0.0, 255.0).astype(np.uint8)
    roughness_u8 = np.clip(roughness, 0.02, 1.0)
    roughness_u8 = (roughness_u8 * 255.0 + 0.5).astype(np.uint8)
    metallic_u8 = np.full((size, size), int(round(spec.metallic * 255.0)), dtype=np.uint8)
    normal = _height_to_normal_map(height)
    return {
        "base_color": base,
        "roughness": roughness_u8,
        "metallic": metallic_u8,
        "normal": normal,
    }


def _draw_scratches(
    base: np.ndarray,
    height: np.ndarray,
    rng: np.random.Generator,
    *,
    count: int,
    color_delta: int,
    size: int,
) -> None:
    image = Image.fromarray(np.clip(base, 0.0, 255.0).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image)
    for _ in range(count):
        x = int(rng.integers(0, size))
        y = int(rng.integers(0, size))
        length = int(rng.integers(size // 12, size // 3))
        angle = float(rng.normal(0.0, 0.35))
        dx = int(np.cos(angle) * length)
        dy = int(np.sin(angle) * length)
        shade = int(np.clip(128 + color_delta + rng.normal(0.0, 18.0), 0, 255))
        draw.line((x, y, x + dx, y + dy), fill=(shade, shade, shade), width=1)
        y0 = max(0, min(size - 1, y))
        x0 = max(0, min(size - 1, x))
        height[y0, x0] -= 0.05
    base[:] = np.asarray(image, dtype=np.float32)


def _soft_value_noise(rng: np.random.Generator, *, size: int, coarse: int) -> np.ndarray:
    small = rng.normal(0.0, 1.0, size=(coarse, coarse)).astype(np.float32)
    image = Image.fromarray(
        ((small - small.min()) / max(float(np.ptp(small)), 1e-6) * 255.0).astype(np.uint8)
    )
    image = image.resize((size, size), Image.Resampling.BICUBIC)
    values = np.asarray(image, dtype=np.float32) / 255.0
    return values - float(values.mean())


def _height_to_normal_map(height: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(height.astype(np.float32))
    strength = 6.0
    nx = -gx * strength
    ny = -gy * strength
    nz = np.ones_like(height, dtype=np.float32)
    length = np.sqrt(nx * nx + ny * ny + nz * nz)
    normal = np.stack((nx / length, ny / length, nz / length), axis=-1)
    return np.clip((normal * 0.5 + 0.5) * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
