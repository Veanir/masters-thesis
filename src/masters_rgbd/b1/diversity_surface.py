"""Hash-bound dense96 test/GSO evaluation for the submission diversity runs."""

from __future__ import annotations

import json

import numpy as np
import torch

from masters_rgbd.b1.dense_control import CONFIG
from masters_rgbd.b1.gso_zero_shot import (
    _sha256,
)
from masters_rgbd.b1.projected_mesh import (
    _new_extractor,
    _stats_dict,
    _surface_metrics,
    _UDFObservationField,
    _visibility_recall_metrics,
)
from masters_rgbd.b1.training import _observation
from masters_rgbd.contracts.fields import QueryBudget, ROIBounds
from masters_rgbd.contracts.geometry import CoordinateFrame


def score(model, data, roi, truth, output, asset_id, category, device):
    rgb, depth, mask, partial, camera = _observation(data, device)
    with torch.no_grad():
        encoding = model.encode_observation(rgb, depth, mask, partial)
    extractor = _new_extractor(**CONFIG)
    center, half = np.asarray(roi.center_camera_m), roi.half_extent_m
    extraction = extractor.extract(
        _UDFObservationField(model, encoding, camera),
        ROIBounds(tuple(center - half), tuple(center + half), CoordinateFrame.CAMERA),
        QueryBudget(extractor.required_query_count),
    )
    points = extraction.surface_points_m
    mesh_path = output / f"{asset_id}.npz"
    np.savez_compressed(
        mesh_path,
        surface_points_camera_m=points,
        mesh_vertices_camera_m=extraction.mesh.vertices_m,
        mesh_faces=extraction.mesh.faces,
    )
    result = {
        "asset_id": asset_id,
        "category": category,
        "point_count": len(points),
        "mesh_sha256": _sha256(mesh_path),
        "extractor_stats": _stats_dict(extraction.stats),
        "surface": _surface_metrics(points, truth),
        "visibility": _visibility_recall_metrics(
            points, truth, depth_m=data["depth_m"], mask=data["mask"], intrinsics=data["intrinsics"]
        ),
    }
    (output / f"{asset_id}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
