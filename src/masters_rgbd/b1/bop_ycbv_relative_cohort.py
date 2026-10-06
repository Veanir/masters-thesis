"""Select a frozen, model-blind relative-visibility YCB-V cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from masters_rgbd.b1.data import _masked_pixels_uv
from masters_rgbd.data.audit_rgbd import load_trimesh_geometry
from masters_rgbd.geometry.ground_truth import TriangleSurface

DEFAULT_SPEC = Path("configs/bop_ycbv_relative42_v1.json")
CV_TO_REPO = np.diag((1.0, -1.0, -1.0))
IMPLEMENTATION_FILES = (
    "pyproject.toml",
    "uv.lock",
    "src/masters_rgbd/b1/bop_ycbv_relative_cohort.py",
    "src/masters_rgbd/b1/data.py",
    "src/masters_rgbd/data/audit_rgbd.py",
    "src/masters_rgbd/geometry/ground_truth.py",
)
ARCHIVE_URLS = {
    "ycbv_base.zip": (
        "https://huggingface.co/datasets/bop-benchmark/ycbv/resolve/main/ycbv_base.zip"
    ),
    "ycbv_models.zip": (
        "https://huggingface.co/datasets/bop-benchmark/ycbv/resolve/main/ycbv_models.zip"
    ),
    "ycbv_test_bop19.zip": (
        "https://huggingface.co/datasets/bop-benchmark/ycbv/resolve/main/ycbv_test_bop19.zip"
    ),
}


class BOPYCBVRelativeCohortError(RuntimeError):
    """The frozen model-blind YCB-V selection contract was violated."""


@dataclass(frozen=True, slots=True)
class Candidate:
    obj_id: int
    scene_id: int
    image_id: int
    gt_id: int
    visibility: float
    visible_pixels: int
    filtered_pixels: int
    valid_depth_ratio: float
    metadata_pixel_drift: int
    median_depth_m: float
    pose_so3_max_correction: float
    scene_camera_path: Path
    scene_gt_path: Path
    scene_gt_info_path: Path
    rgb_path: Path
    depth_path: Path
    mask_path: Path
    mesh_path: Path
    filtered_mask: np.ndarray
    depth_m: np.ndarray
    intrinsics: np.ndarray
    transform: np.ndarray


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BOPYCBVRelativeCohortError(f"cannot read JSON: {path}") from error


def _validated_spec(path: Path) -> dict[str, Any]:
    spec = _read_json(path)
    if not isinstance(spec, dict) or spec.get("schema_version") != (
        "bop-ycbv-relative-cohort-spec-v1"
    ):
        raise BOPYCBVRelativeCohortError("unexpected relative-cohort spec")
    object_ids = spec.get("object_ids")
    if (
        not isinstance(object_ids, list)
        or not object_ids
        or any(isinstance(value, bool) or not isinstance(value, int) for value in object_ids)
        or object_ids != sorted(set(object_ids))
    ):
        raise BOPYCBVRelativeCohortError("object IDs must be unique sorted integers")
    expected = {
        "dataset": "ycbv",
        "selection_seed": 0,
        "expected_observation_count": 2 * len(object_ids),
        "strata": ["relative-low", "relative-high"],
        "candidate_source": "test_targets_bop19.json",
        "mesh_root": "models_eval",
        "selection": {
            "relative_partition": "bottom-and-top-ceil-third-by-visibility",
            "tie_break": "sha256-v1",
            "pairing": "different-scene-where-possible-else-different-frame",
        },
        "claims": {"relative_visibility": True, "absolute_occlusion": False},
    }
    if any(spec.get(key) != value for key, value in expected.items()):
        raise BOPYCBVRelativeCohortError("relative-cohort contract differs")
    filters = spec.get("filters")
    if filters != {
        "bbox_margin_px": 1,
        "metadata_pixel_drift_max": 2,
        "median_depth_radius_m": 0.25,
        "effective_pixel_min": 512,
        "so3_projection_max_correction": 0.000002,
        "alignment_median_max_m": 0.002,
        "alignment_p95_max_m": 0.015,
    }:
        raise BOPYCBVRelativeCohortError("relative-cohort filters differ")
    implementation = spec.get("implementation")
    if implementation != {
        "require_clean_source_commit": True,
        "files": list(IMPLEMENTATION_FILES),
    }:
        raise BOPYCBVRelativeCohortError("implementation binding contract differs")
    source = spec.get("dataset_source")
    if not isinstance(source, dict) or source.get("provider") != "BOP Benchmark":
        raise BOPYCBVRelativeCohortError("dataset source provider differs")
    if source.get("dataset_url") != "https://huggingface.co/datasets/bop-benchmark/ycbv":
        raise BOPYCBVRelativeCohortError("dataset source URL differs")
    if source.get("license") != {
        "spdx_id": "MIT",
        "notice_path": "dataset_info.md",
        "copyright": (
            "Copyright (c) 2017 Robotics and State Estimation Lab at The University of Washington"
        ),
    }:
        raise BOPYCBVRelativeCohortError("dataset license binding differs")
    if source.get("object_model_source") != {
        "name": "YCB Object and Model Set",
        "url": "https://www.ycbbenchmarks.com/object-models/",
    }:
        raise BOPYCBVRelativeCohortError("object model source binding differs")
    if source.get("object_model_license") != {
        "spdx_id": "CC-BY-4.0",
        "url": "https://creativecommons.org/licenses/by/4.0/",
    }:
        raise BOPYCBVRelativeCohortError("object model license binding differs")
    files = source.get("dataset_files")
    expected_paths = {
        "dataset_info.md",
        "test_targets_bop19.json",
        "models_eval/models_info.json",
    }
    if not isinstance(files, dict) or set(files) != expected_paths:
        raise BOPYCBVRelativeCohortError("dataset provenance files differ")
    archives = source.get("archives")
    if not isinstance(archives, list) or len(archives) != len(ARCHIVE_URLS):
        raise BOPYCBVRelativeCohortError("dataset archive binding differs")
    for record in [*files.values(), *archives]:
        if (
            not isinstance(record, dict)
            or not isinstance(record.get("sha256"), str)
            or len(record["sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in record["sha256"])
            or isinstance(record.get("size_bytes"), bool)
            or not isinstance(record.get("size_bytes"), int)
            or record["size_bytes"] <= 0
        ):
            raise BOPYCBVRelativeCohortError("invalid dataset provenance digest")
    for record in archives:
        name = record.get("name")
        if name not in ARCHIVE_URLS or record.get("url") != ARCHIVE_URLS[name]:
            raise BOPYCBVRelativeCohortError("dataset archive name or URL differs")
    if [record["name"] for record in archives] != list(ARCHIVE_URLS):
        raise BOPYCBVRelativeCohortError("dataset archives are not canonically ordered")
    return spec


def _bundle_sha256(bindings: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for path, file_sha256 in bindings.items():
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha256.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _source_provenance(source_root: Path) -> dict[str, Any]:
    source_root = source_root.resolve(strict=True)

    def git(*arguments: str) -> bytes:
        completed = subprocess.run(
            ("git", "-C", str(source_root), *arguments),
            check=False,
            capture_output=True,
        )
        if completed.returncode != 0:
            raise BOPYCBVRelativeCohortError("cannot establish Git source provenance")
        return completed.stdout

    top_level = Path(git("rev-parse", "--show-toplevel").decode().strip()).resolve()
    if top_level != source_root:
        raise BOPYCBVRelativeCohortError("source root is not the Git top level")
    source_commit = git("rev-parse", "HEAD").decode("ascii").strip()
    if len(source_commit) != 40 or any(c not in "0123456789abcdef" for c in source_commit):
        raise BOPYCBVRelativeCohortError("source commit is not a full Git SHA")
    dirty = git("status", "--porcelain", "--untracked-files=no", "--", *IMPLEMENTATION_FILES)
    if dirty:
        raise BOPYCBVRelativeCohortError("implementation source files are not clean")
    bindings: dict[str, str] = {}
    for relative in IMPLEMENTATION_FILES:
        path = source_root / relative
        if not path.is_file():
            raise BOPYCBVRelativeCohortError(f"missing implementation file: {relative}")
        current = _sha256(path)
        committed = hashlib.sha256(git("show", f"{source_commit}:{relative}")).hexdigest()
        if current != committed:
            raise BOPYCBVRelativeCohortError(
                f"implementation file differs from source commit: {relative}"
            )
        bindings[relative] = current
    return {
        "source_commit": source_commit,
        "clean": True,
        "files": bindings,
        "bundle_sha256": _bundle_sha256(bindings),
    }


def _dataset_provenance(
    dataset_root: Path, archive_root: Path, source: dict[str, Any]
) -> tuple[dict[str, Any], set[Path]]:
    archive_root = archive_root.resolve(strict=True)
    consumed: set[Path] = set()
    dataset_files: dict[str, Any] = {}
    for relative, expected in source["dataset_files"].items():
        path = dataset_root / relative
        if not path.is_file():
            raise BOPYCBVRelativeCohortError(f"missing dataset provenance file: {relative}")
        actual = {"sha256": _sha256(path), "size_bytes": path.stat().st_size}
        if actual != expected:
            raise BOPYCBVRelativeCohortError(f"dataset provenance differs: {relative}")
        consumed.add(path)
        dataset_files[relative] = actual
    archives = []
    for expected in source["archives"]:
        path = archive_root / expected["name"]
        if not path.is_file():
            raise BOPYCBVRelativeCohortError(f"missing source archive: {expected['name']}")
        actual = {
            "name": expected["name"],
            "url": expected["url"],
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
        if actual != expected:
            raise BOPYCBVRelativeCohortError(
                f"source archive provenance differs: {expected['name']}"
            )
        archives.append(actual)
    return (
        {
            "provider": source["provider"],
            "dataset_url": source["dataset_url"],
            "license": source["license"],
            "object_model_source": source["object_model_source"],
            "object_model_license": source["object_model_license"],
            "dataset_files": dataset_files,
            "archives": archives,
            "archive_bundle_sha256": _bundle_sha256(
                {record["name"]: record["sha256"] for record in archives}
            ),
        },
        consumed,
    )


def _bop_pose(
    rotation: object, translation_mm: object, *, maximum: float
) -> tuple[np.ndarray, float]:
    raw = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    translation = np.asarray(translation_mm, dtype=np.float64).reshape(3) * 0.001
    if not np.isfinite(raw).all() or not np.isfinite(translation).all():
        raise BOPYCBVRelativeCohortError("BOP pose contains non-finite values")
    u, _, vt = np.linalg.svd(raw)
    sign = float(np.linalg.det(u @ vt))
    projected = u @ np.diag((1.0, 1.0, sign)) @ vt
    correction = float(np.max(np.abs(projected - raw)))
    if correction > maximum:
        raise BOPYCBVRelativeCohortError("BOP pose SO(3) correction exceeds contract")
    if not np.allclose(projected.T @ projected, np.eye(3), atol=1e-12, rtol=0.0) or not np.isclose(
        np.linalg.det(projected), 1.0, atol=1e-12, rtol=0.0
    ):
        raise BOPYCBVRelativeCohortError("projected BOP rotation is not strict SO(3)")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = CV_TO_REPO @ projected
    transform[:3, 3] = CV_TO_REPO @ translation
    return transform, correction


def _unproject_all(depth_m: np.ndarray, mask: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    pixels = _masked_pixels_uv(mask)
    columns = pixels[:, 0].astype(np.int64)
    rows = pixels[:, 1].astype(np.int64)
    values = depth_m[rows, columns]
    fx, fy, cx, cy = intrinsics
    return np.column_stack(((columns - cx) / fx * values, -(rows - cy) / fy * values, -values))


def _target_map(path: Path, object_ids: set[int]) -> dict[tuple[int, int, int], int]:
    rows = _read_json(path)
    if not isinstance(rows, list):
        raise BOPYCBVRelativeCohortError("BOP19 targets must be a list")
    result: dict[tuple[int, int, int], int] = {}
    for row in rows:
        try:
            key = (int(row["scene_id"]), int(row["im_id"]), int(row["obj_id"]))
            count = int(row["inst_count"])
        except (KeyError, TypeError, ValueError) as error:
            raise BOPYCBVRelativeCohortError("malformed BOP19 target") from error
        if key[2] in object_ids:
            if key in result or count <= 0:
                raise BOPYCBVRelativeCohortError("duplicate or empty BOP19 target")
            result[key] = count
    if {key[2] for key in result} != object_ids:
        raise BOPYCBVRelativeCohortError("BOP19 targets do not cover every object")
    return result


def _candidate_rows(
    dataset_root: Path, spec: dict[str, Any]
) -> tuple[dict[int, list[Candidate]], set[Path], dict[str, int]]:
    limits = spec["filters"]
    target_path = dataset_root / spec["candidate_source"]
    targets = _target_map(target_path, set(spec["object_ids"]))
    consumed = {target_path}
    candidates = {obj_id: [] for obj_id in spec["object_ids"]}
    rejected = {"bbox": 0, "metadata_pixel_drift": 0, "effective_pixels": 0}
    frame_keys = sorted({key[:2] for key in targets})
    for scene_id, image_id in frame_keys:
        scene_root = dataset_root / "test" / f"{scene_id:06d}"
        paths = {
            "camera": scene_root / "scene_camera.json",
            "gt": scene_root / "scene_gt.json",
            "info": scene_root / "scene_gt_info.json",
            "rgb": scene_root / "rgb" / f"{image_id:06d}.png",
            "depth": scene_root / "depth" / f"{image_id:06d}.png",
        }
        consumed.update((paths["camera"], paths["gt"], paths["info"], paths["depth"]))
        camera_rows = _read_json(paths["camera"])
        gt_rows = _read_json(paths["gt"])
        info_rows = _read_json(paths["info"])
        try:
            camera = camera_rows[str(image_id)]
            ground_truth = gt_rows[str(image_id)]
            information = info_rows[str(image_id)]
            with Image.open(paths["depth"]) as image:
                raw_depth = np.asarray(image)
        except (KeyError, TypeError, OSError) as error:
            raise BOPYCBVRelativeCohortError("cannot decode BOP frame") from error
        if raw_depth.ndim != 2 or len(ground_truth) != len(information):
            raise BOPYCBVRelativeCohortError("BOP frame arrays or annotations differ")
        intrinsics_matrix = np.asarray(camera["cam_K"], dtype=np.float64).reshape(3, 3)
        intrinsics = np.asarray(
            (
                intrinsics_matrix[0, 0],
                intrinsics_matrix[1, 1],
                intrinsics_matrix[0, 2],
                intrinsics_matrix[1, 2],
            )
        )
        if not np.isfinite(intrinsics).all() or np.any(intrinsics[:2] <= 0.0):
            raise BOPYCBVRelativeCohortError("invalid per-frame camera intrinsics")
        depth_m = raw_depth.astype(np.float64) * float(camera["depth_scale"]) * 0.001
        height, width = raw_depth.shape
        frame_objects = sorted(
            obj for scene, image, obj in targets if (scene, image) == (scene_id, image_id)
        )
        for obj_id in frame_objects:
            gt_ids = [
                index for index, row in enumerate(ground_truth) if int(row["obj_id"]) == obj_id
            ]
            if len(gt_ids) != targets[(scene_id, image_id, obj_id)]:
                raise BOPYCBVRelativeCohortError("BOP19 inst_count differs from scene GT")
            mesh_path = dataset_root / spec["mesh_root"] / f"obj_{obj_id:06d}.ply"
            consumed.add(mesh_path)
            for gt_id in gt_ids:
                info = information[gt_id]
                left, top, box_width, box_height = (int(value) for value in info["bbox_obj"])
                margin = limits["bbox_margin_px"]
                if not (
                    left >= margin
                    and top >= margin
                    and left + box_width <= width - margin
                    and top + box_height <= height - margin
                ):
                    rejected["bbox"] += 1
                    continue
                mask_path = scene_root / "mask_visib" / f"{image_id:06d}_{gt_id:06d}.png"
                consumed.add(mask_path)
                try:
                    with Image.open(mask_path) as image:
                        visible = np.asarray(image) > 0
                except OSError as error:
                    raise BOPYCBVRelativeCohortError("cannot decode BOP visible mask") from error
                if visible.shape != raw_depth.shape:
                    raise BOPYCBVRelativeCohortError("visible mask and depth shapes differ")
                visible_count = int(np.count_nonzero(visible))
                drift = visible_count - int(info["px_count_visib"])
                if abs(drift) > limits["metadata_pixel_drift_max"]:
                    rejected["metadata_pixel_drift"] += 1
                    continue
                positive_visible = visible & (raw_depth > 0)
                if not np.any(positive_visible):
                    rejected["effective_pixels"] += 1
                    continue
                median_depth = float(np.median(depth_m[positive_visible]))
                filtered = positive_visible & (
                    np.abs(depth_m - median_depth) <= limits["median_depth_radius_m"]
                )
                filtered_count = int(np.count_nonzero(filtered))
                if filtered_count < limits["effective_pixel_min"]:
                    rejected["effective_pixels"] += 1
                    continue
                pose = ground_truth[gt_id]
                transform, correction = _bop_pose(
                    pose["cam_R_m2c"],
                    pose["cam_t_m2c"],
                    maximum=limits["so3_projection_max_correction"],
                )
                candidates[obj_id].append(
                    Candidate(
                        obj_id=obj_id,
                        scene_id=scene_id,
                        image_id=image_id,
                        gt_id=gt_id,
                        visibility=float(info["visib_fract"]),
                        visible_pixels=visible_count,
                        filtered_pixels=filtered_count,
                        valid_depth_ratio=filtered_count / visible_count,
                        metadata_pixel_drift=drift,
                        median_depth_m=median_depth,
                        pose_so3_max_correction=correction,
                        scene_camera_path=paths["camera"],
                        scene_gt_path=paths["gt"],
                        scene_gt_info_path=paths["info"],
                        rgb_path=paths["rgb"],
                        depth_path=paths["depth"],
                        mask_path=mask_path,
                        mesh_path=mesh_path,
                        filtered_mask=filtered,
                        depth_m=depth_m,
                        intrinsics=intrinsics,
                        transform=transform,
                    )
                )
    return candidates, consumed, rejected


def _selection_digest(candidate: Candidate, stratum: str, seed: int) -> str:
    payload = "\0".join(
        (
            "bop-ycbv-real-sensor-relative-v1",
            f"seed={seed}",
            str(candidate.obj_id),
            stratum,
            str(candidate.scene_id),
            str(candidate.image_id),
            str(candidate.gt_id),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _input_tree_sha256(paths: set[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda value: value.relative_to(root).as_posix()):
        if not path.is_file():
            raise BOPYCBVRelativeCohortError(f"missing consumed input: {path}")
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _publish_exclusive_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise BOPYCBVRelativeCohortError("output manifest already exists") from error
    finally:
        temporary.unlink(missing_ok=True)


def select_ycbv_relative_cohort(
    dataset_root: Path,
    spec_path: Path,
    output_path: Path,
    *,
    archive_root: Path,
    source_root: Path,
) -> dict[str, Any]:
    dataset_root = Path(dataset_root).resolve(strict=True)
    spec_path = Path(spec_path).resolve(strict=True)
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise BOPYCBVRelativeCohortError("output manifest already exists")
    spec = _validated_spec(spec_path)
    implementation = _source_provenance(Path(source_root))
    dataset_source, provenance_inputs = _dataset_provenance(
        dataset_root, Path(archive_root), spec["dataset_source"]
    )
    candidates, consumed, cheap_rejections = _candidate_rows(dataset_root, spec)
    consumed.update(provenance_inputs)
    surfaces: dict[int, TriangleSurface] = {}
    alignment_cache: dict[tuple[int, int, int], tuple[float, float, int]] = {}
    alignment_rejections = 0

    def alignment(candidate: Candidate) -> tuple[float, float, int]:
        nonlocal alignment_rejections
        key = (candidate.scene_id, candidate.image_id, candidate.gt_id)
        if key not in alignment_cache:
            if candidate.obj_id not in surfaces:
                vertices_mm, faces, _ = load_trimesh_geometry(candidate.mesh_path)
                surfaces[candidate.obj_id] = TriangleSurface(vertices_mm * 0.001, faces)
            points = _unproject_all(
                candidate.depth_m, candidate.filtered_mask, candidate.intrinsics
            )
            local = (points - candidate.transform[:3, 3]) @ candidate.transform[:3, :3]
            distances = surfaces[candidate.obj_id].udf_m(local)
            alignment_cache[key] = (
                float(np.median(distances)),
                float(np.quantile(distances, 0.95)),
                len(points),
            )
        return alignment_cache[key]

    selected: list[tuple[str, Candidate, str, float, float, int]] = []
    object_summary: dict[str, Any] = {}
    limits = spec["filters"]
    for obj_id in spec["object_ids"]:
        ordered = sorted(
            candidates[obj_id],
            key=lambda row: (row.visibility, row.scene_id, row.image_id, row.gt_id),
        )
        if len(ordered) < 2:
            raise BOPYCBVRelativeCohortError(f"object {obj_id} has fewer than two candidates")
        third = math.ceil(len(ordered) / 3)
        pools = {
            "relative-low": ordered[:third],
            "relative-high": ordered[-third:],
        }
        picks: dict[str, tuple[str, Candidate, float, float, int]] = {}
        for stratum in spec["strata"]:
            ranked = sorted(
                (_selection_digest(row, stratum, spec["selection_seed"]), row)
                for row in pools[stratum]
            )
            if stratum == "relative-high":
                low = picks["relative-low"][1]
                different = [item for item in ranked if item[1].scene_id != low.scene_id]
                fallback = [
                    item
                    for item in ranked
                    if item[1].scene_id == low.scene_id and item[1].image_id != low.image_id
                ]
                ranked = different + fallback
            pick = None
            for digest, candidate in ranked:
                median, p95, count = alignment(candidate)
                if (
                    median <= limits["alignment_median_max_m"]
                    and p95 <= limits["alignment_p95_max_m"]
                ):
                    pick = (digest, candidate, median, p95, count)
                    break
                alignment_rejections += 1
            if pick is None:
                raise BOPYCBVRelativeCohortError(
                    f"object {obj_id} {stratum} has no alignment-passing candidate"
                )
            picks[stratum] = pick
            selected.append((stratum, *pick))
        low = picks["relative-low"][1]
        high = picks["relative-high"][1]
        object_summary[str(obj_id)] = {
            "candidate_count": len(ordered),
            "relative_third_size": third,
            "different_scene": low.scene_id != high.scene_id,
        }
    if len(selected) != spec["expected_observation_count"]:
        raise BOPYCBVRelativeCohortError("selected observation count differs")
    observations = []
    for stratum, digest, candidate, median, p95, count in selected:
        try:
            with Image.open(candidate.rgb_path) as image:
                if image.size != (candidate.depth_m.shape[1], candidate.depth_m.shape[0]):
                    raise BOPYCBVRelativeCohortError("selected RGB and depth shapes differ")
                image.convert("RGB").load()
        except OSError as error:
            raise BOPYCBVRelativeCohortError("cannot decode selected RGB frame") from error
        consumed.add(candidate.rgb_path)
        observations.append(
            {
                "target_id": (
                    f"obj{candidate.obj_id:06d}-scene{candidate.scene_id:06d}-"
                    f"image{candidate.image_id:06d}-gt{candidate.gt_id:06d}"
                ),
                "obj_id": candidate.obj_id,
                "scene_id": candidate.scene_id,
                "image_id": candidate.image_id,
                "gt_id": candidate.gt_id,
                "stratum": stratum,
                "relative_visibility_only": True,
                "selection_sha256": digest,
                "visib_fract": candidate.visibility,
                "visible_pixels": candidate.visible_pixels,
                "filtered_pixels": candidate.filtered_pixels,
                "valid_depth_ratio": candidate.valid_depth_ratio,
                "metadata_pixel_drift": candidate.metadata_pixel_drift,
                "median_depth_m": candidate.median_depth_m,
                "pose_so3_max_correction": candidate.pose_so3_max_correction,
                "alignment_median_m": median,
                "alignment_p95_m": p95,
                "alignment_point_count": count,
                "inputs": {
                    "scene_camera": candidate.scene_camera_path.relative_to(
                        dataset_root
                    ).as_posix(),
                    "scene_camera_sha256": _sha256(candidate.scene_camera_path),
                    "scene_gt": candidate.scene_gt_path.relative_to(dataset_root).as_posix(),
                    "scene_gt_sha256": _sha256(candidate.scene_gt_path),
                    "scene_gt_info": candidate.scene_gt_info_path.relative_to(
                        dataset_root
                    ).as_posix(),
                    "scene_gt_info_sha256": _sha256(candidate.scene_gt_info_path),
                    "rgb": candidate.rgb_path.relative_to(dataset_root).as_posix(),
                    "rgb_sha256": _sha256(candidate.rgb_path),
                    "depth": candidate.depth_path.relative_to(dataset_root).as_posix(),
                    "depth_sha256": _sha256(candidate.depth_path),
                    "mask_visib": candidate.mask_path.relative_to(dataset_root).as_posix(),
                    "mask_visib_sha256": _sha256(candidate.mask_path),
                    "mesh": candidate.mesh_path.relative_to(dataset_root).as_posix(),
                    "mesh_sha256": _sha256(candidate.mesh_path),
                },
            }
        )
    manifest = {
        "schema_version": "bop-ycbv-relative-cohort-manifest-v1",
        "status": "frozen-model-blind-relative-visibility-cohort",
        "dataset": "ycbv",
        "implementation": implementation,
        "dataset_source": dataset_source,
        "selector_source_sha256": _sha256(Path(__file__)),
        "spec_sha256": _sha256(spec_path),
        "selection_input_tree_sha256": _input_tree_sha256(consumed, dataset_root),
        "selection_seed": spec["selection_seed"],
        "observation_count": len(observations),
        "object_count": len(spec["object_ids"]),
        "strata": spec["strata"],
        "claims": spec["claims"],
        "filters": spec["filters"],
        "selection": spec["selection"],
        "cheap_rejections": cheap_rejections,
        "alignment_candidate_count": len(alignment_cache),
        "alignment_rejection_count": alignment_rejections,
        "objects": object_summary,
        "observations": observations,
    }
    _publish_exclusive_json(output_path, manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path("."))
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = select_ycbv_relative_cohort(
        args.dataset_root,
        args.spec,
        args.output,
        archive_root=args.archive_root,
        source_root=args.source_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
