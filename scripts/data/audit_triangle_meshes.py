"""Audit actual posed triangle meshes, independently of PhysX colliders."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import trimesh


def _load_cleaned_mesh(path):
    with np.load(path) as data:
        mesh = trimesh.Trimesh(vertices=data["vertices_world_m"], faces=data["faces"], process=True)
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals(multibody=True)
    return mesh


def load_mesh(path):
    legacy = _load_cleaned_mesh(path)
    if legacy.is_volume:
        return legacy
    with np.load(path, allow_pickle=False) as data:
        vertices = data["vertices_world_m"]
        faces = data["faces"]
    if not np.isfinite(vertices).all():
        return legacy
    unique, inverse = np.unique(vertices, axis=0, return_inverse=True)
    exact = trimesh.Trimesh(vertices=unique, faces=inverse[faces], process=False)
    if not (
        np.isfinite(exact.area_faces).all() and (exact.area_faces > 0).all() and exact.is_volume
    ):
        return legacy
    assert np.array_equal(exact.triangles, vertices[faces]), "Fallback changed a rendered triangle"
    exact.metadata["certification_loader"] = "exact_positive_area_triangles_after_legacy_opening"
    exact.metadata["legacy_face_count"] = len(legacy.faces)
    exact.metadata["preserved_face_count"] = len(faces)
    return exact


def check_pair(left, right):
    if not left.is_volume or not right.is_volume:
        return {"status": "uncertifiable_open_or_inconsistent_surface", "passed": False}
    if np.any(
        np.minimum(left.bounds[1], right.bounds[1]) <= np.maximum(left.bounds[0], right.bounds[0])
    ):
        return {"status": "disjoint_aabbs", "overlap_volume_m3": 0.0, "passed": True}
    intersection = trimesh.boolean.intersection([left, right], engine="manifold", check_volume=True)
    overlap = float(abs(intersection.volume)) if not intersection.is_empty else 0.0
    fraction = float(overlap / min(abs(left.volume), abs(right.volume)))
    # Allow numerical/contact slivers up to 1 mm^3 or 0.1% of the smaller body.
    limit = max(1e-9, 0.001 * min(abs(left.volume), abs(right.volume)))
    return {
        "status": "actual_triangle_boolean",
        "overlap_volume_m3": overlap,
        "smaller_body_overlap_fraction": fraction,
        "allowed_volume_m3": float(limit),
        "passed": bool(overlap <= limit),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    rows = []
    for folder in sorted(a.input.iterdir()):
        if not (folder / "scene.json").exists():
            continue
        scene = json.loads((folder / "scene.json").read_text())
        meshes = [load_mesh(folder / f"object-{j}.npz") for j in range(len(scene["objects"]))]
        pairs = []
        for i, j in itertools.combinations(range(len(meshes)), 2):
            try:
                result = check_pair(meshes[i], meshes[j])
            except Exception as exc:
                result = {"status": "audit_error", "error": repr(exc), "passed": False}
            pairs.append({"left": i, "right": j, **result})
        floor = [float(max(0, -m.vertices[:, 2].min())) for m in meshes]
        row = {
            "asset_id": folder.name,
            "pairs": pairs,
            "floor_penetration_m": floor,
            "mesh_volume_certifiable": [bool(m.is_volume) for m in meshes],
            "physics_settled": scene["settled_and_above_floor"],
            "passed": scene["settled_and_above_floor"]
            and max(floor) <= 0.002
            and all(r["passed"] for r in pairs),
        }
        rows.append(row)
        print(folder.name, row["passed"], flush=True)
    assert rows, "No captured scenes to audit"
    result = {
        "status": "actual_mesh_physics_audit_complete",
        "scene_count": len(rows),
        "passed_count": sum(r["passed"] for r in rows),
        "rows": rows,
        "method": (
            "Trimesh welded actual render triangles; manifold3d Boolean "
            "intersection of positive closed volumes; open meshes cannot "
            "pass. Floor <=2 mm, saved last-0.5s motion <=1 mm."
        ),
        "overlap_tolerance": "max(1 mm^3, 0.1% of smaller body volume)",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "versions": {"trimesh": trimesh.__version__},
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
