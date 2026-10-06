"""Extract and score meshes for the eight held-out rapid-Dev objects."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial import cKDTree

from masters_rgbd.b1.model import B1FieldModel, ObservationEncoding
from masters_rgbd.b1.training import (
    _depth_visibility_masks,
)
from masters_rgbd.contracts.fields import FieldPrediction, QueryBatch
from masters_rgbd.contracts.geometry import CoordinateFrame
from masters_rgbd.evaluation.surface import evaluate_surface_points

_THRESHOLDS_M = (0.002, 0.005, 0.010)


class RapidMeshError(RuntimeError):
    """The existing rapid-Dev run cannot support mesh evaluation."""


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RapidMeshError(f"cannot read {label}: {path}") from error
    if not isinstance(value, dict):
        raise RapidMeshError(f"{label} must be a JSON object: {path}")
    return value


class _UDFObservationField:
    """Expose only model UDF values to the projected-UDF extractor."""

    def __init__(
        self,
        model: B1FieldModel,
        encoding: ObservationEncoding,
        intrinsics: torch.Tensor,
        *,
        chunk_size: int = 4096,
    ) -> None:
        self._model = model
        self._encoding = encoding
        self._intrinsics = intrinsics
        self._device = intrinsics.device
        self._chunk_size = chunk_size

    def query(self, queries: QueryBatch) -> FieldPrediction:
        if queries.frame != CoordinateFrame.CAMERA:
            raise ValueError("rapid mesh field accepts camera-frame queries only")
        chunks: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(queries), self._chunk_size):
                points = torch.from_numpy(
                    np.array(queries.points[start : start + self._chunk_size], dtype=np.float32)
                )[None].to(self._device)
                prediction = self._model.decode_queries(
                    self._encoding,
                    points,
                    self._intrinsics,
                )
                chunks.append(prediction.udf_m[0].detach().float().cpu().numpy())
        udf_m = np.concatenate(chunks).astype(np.float64, copy=False)
        # Occupancy is intentionally neither read from the model nor used by the extractor.
        return FieldPrediction(udf_m=udf_m, occupancy_logit=np.zeros_like(udf_m))


def _sample_mesh_surface(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    count: int,
    seed: int,
) -> np.ndarray:
    triangles = np.asarray(vertices, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    areas = np.linalg.norm(cross, axis=1)
    valid = areas > 1e-14
    triangles = triangles[valid]
    areas = areas[valid]
    if len(triangles) == 0:
        raise RapidMeshError("target mesh has no non-degenerate triangles")
    rng = np.random.default_rng(seed)
    selected = triangles[rng.choice(len(triangles), size=count, p=areas / areas.sum())]
    root = np.sqrt(rng.random(count))
    second = rng.random(count)
    barycentric = np.column_stack((1.0 - root, root * (1.0 - second), root * second))
    return np.sum(selected * barycentric[:, :, None], axis=1)


def _surface_metrics(predicted: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    if len(predicted) == 0:
        return {
            f"{threshold:.3f}": {
                "fscore": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "chamfer_mean_m": None,
                "chamfer_p95_m": None,
            }
            for threshold in _THRESHOLDS_M
        }
    return {
        f"{threshold:.3f}": asdict(
            evaluate_surface_points(predicted, target, threshold_m=threshold)
        )
        for threshold in _THRESHOLDS_M
    }


def _visibility_recall_metrics(
    predicted: np.ndarray,
    target: np.ndarray,
    *,
    depth_m: np.ndarray,
    mask: np.ndarray,
    intrinsics: np.ndarray,
    threshold_m: float = 0.005,
) -> dict[str, float | int | None]:
    _, hidden = _depth_visibility_masks(
        target,
        np.zeros(len(target), dtype=np.float64),
        depth_m,
        mask,
        intrinsics,
    )
    visible = ~hidden
    if len(predicted):
        distances, _ = cKDTree(predicted).query(target, k=1, workers=1)
        hits = distances <= threshold_m
    else:
        hits = np.zeros(len(target), dtype=np.bool_)
    hidden_count = int(np.count_nonzero(hidden))
    visible_count = int(np.count_nonzero(visible))
    hidden_hits = int(np.count_nonzero(hits & hidden))
    visible_hits = int(np.count_nonzero(hits & visible))
    return {
        "hidden_target_count": hidden_count,
        "hidden_hit_count_5mm": hidden_hits,
        "hidden_recall_5mm": None if hidden_count == 0 else hidden_hits / hidden_count,
        "visible_target_count": visible_count,
        "visible_hit_count_5mm": visible_hits,
        "visible_recall_5mm": None if visible_count == 0 else visible_hits / visible_count,
    }


def _stats_dict(stats: object) -> dict[str, Any]:
    if is_dataclass(stats) and not isinstance(stats, type):
        return asdict(stats)
    if hasattr(stats, "__dict__"):
        return dict(vars(stats))
    raise RapidMeshError("extractor stats are not serializable")


def _aggregate(records: list[dict[str, Any]], *, expected_count: int) -> dict[str, Any]:
    if expected_count <= 0 or len(records) != expected_count:
        raise RapidMeshError("aggregate record count differs from the declared test cohort")
    surface: dict[str, Any] = {}
    for threshold in _THRESHOLDS_M:
        key = f"{threshold:.3f}"
        surface[key] = {}
        for metric in ("fscore", "precision", "recall", "chamfer_mean_m", "chamfer_p95_m"):
            values = [record["surface"][key][metric] for record in records]
            surface[key][f"object_mean_{metric}"] = (
                None if any(value is None for value in values) else float(np.mean(values))
            )
    return {
        "schema_version": "rapid-mesh-evaluation-v1",
        "object_count": len(records),
        "occupancy_used": False,
        "surface": surface,
        "visibility": {
            key: (
                float(np.mean(values))
                if (
                    values := [
                        record["visibility"][key]
                        for record in records
                        if record["visibility"][key] is not None
                    ]
                )
                else None
            )
            for key in ("hidden_recall_5mm", "visible_recall_5mm")
        },
        "per_object": records,
    }


def _new_extractor(
    resolution: int,
    *,
    tau_scale: float = 1.0,
    projection_mode: str = "unit_tau",
):
    from masters_rgbd.extraction.projected_udf import DenseProjectedUDFExtractor

    return DenseProjectedUDFExtractor(
        resolution=resolution,
        tau_scale=tau_scale,
        projection_mode=projection_mode,
    )
