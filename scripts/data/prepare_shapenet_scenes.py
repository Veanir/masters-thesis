"""Prepare eligible ShapeNet family representatives with explicit metric priors."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import trimesh

from scripts.common.paths import DATASET_ROOT, script_help
from scripts.common.paths import ROOT as WORKSPACE_ROOT

script_help(__doc__, __name__)


ROOT = WORKSPACE_ROOT
BASE = ROOT / "runs/research-evolution-assets-20260906"
DATA = DATASET_ROOT / "shapenet-evolution-v1"
PRIORS = {
    "bottle": ("height", 0.22),
    "bowl": ("diameter", 0.16),
    "can": ("height", 0.12),
    "mug": ("height", 0.10),
}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def prepare(row):
    key = row["asset_key"]
    folder = DATA / "prepared" / ("shapenet-" + key)
    path = DATA / "models" / row["path"]
    assert sha(path) == row["obj_sha256"]
    scene = trimesh.load(path, force="scene", process=False, allow_remote=False)
    mesh = scene.to_mesh()
    assert mesh.visual.kind == "texture"
    if mesh.visual.uv is None:
        mesh.visual.uv = np.full((len(mesh.vertices), 2), 0.5)
    material = mesh.visual.material
    if hasattr(material, "to_simple"):
        material = material.to_simple()
    image = getattr(material, "image", None)
    if image is None:
        from PIL import Image

        diffuse = np.asarray(material.diffuse, dtype=np.uint8)
        image = Image.fromarray(np.tile(diffuse, (4, 4, 1)))
    # Atlas padding may be transparent. The renderer explicitly authors opaque
    # diffuse materials, including source glass; source opacity is not simulated.
    role, nominal = PRIORS[row["category"]]
    unit = (
        int(hashlib.sha256(("evolution-metric-prior-v1/" + key).encode()).hexdigest()[:16], 16)
        / 2**64
    )
    dimension = nominal * (0.8 + 0.4 * unit)
    denominator = mesh.extents[1] if role == "height" else max(mesh.extents[0], mesh.extents[2])
    scale = dimension / denominator
    center = mesh.bounds.mean(axis=0)
    vertices = (mesh.vertices - center) * scale
    vertices = vertices[:, [0, 2, 1]] * np.array([1, -1, 1])
    assert 0.03 <= np.ptp(vertices, axis=0).max() <= 0.6
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        folder / "mesh.npz",
        vertices=vertices.astype(np.float32),
        faces=mesh.faces.astype(np.int32),
        uv=np.asarray(mesh.visual.uv, dtype=np.float32),
    )
    physical = trimesh.Trimesh(vertices=vertices, faces=mesh.faces, process=True)
    physical.update_faces(physical.nondegenerate_faces())
    physical.update_faces(physical.unique_faces())
    physical.remove_unreferenced_vertices()
    physical.fix_normals(multibody=True)
    assert physical.is_volume
    np.savez_compressed(
        folder / "physical-mesh.npz",
        vertices=physical.vertices.astype(np.float32),
        faces=physical.faces.astype(np.int32),
    )
    image.convert("RGB").save(folder / "texture.png")
    meta = {
        "asset_id": folder.name,
        "source": "ShapeNetCore v2",
        "source_id": row["source_id"],
        "synset": row["synset"],
        "category": row["category"],
        "source_obj_sha256": row["obj_sha256"],
        "mesh_npz_sha256": sha(folder / "mesh.npz"),
        "physical_mesh_sha256": sha(folder / "physical-mesh.npz"),
        "texture_sha256": sha(folder / "texture.png"),
        "scale_applied": float(scale),
        "centering_offset_source_units": center.tolist(),
        "metric_prior": {
            "role": role,
            "nominal_m": nominal,
            "assigned_m": dimension,
            "range_factor": [0.8, 1.2],
            "seed": "evolution-metric-prior-v1",
            "basis": "Explicit design prior, not estimated from held-out HB or model scores",
        },
        "axes": "source Y-up to physics Z-up: [x,-z,y]",
        "appearance": (
            "Source diffuse textures/material colors packed by trimesh into "
            "one atlas; renderer authors opacity=1 and roughness=0.5, "
            "including source glass. Not an optical reproduction of "
            "transparency."
        ),
        "texture_size": list(image.size),
        "extents_m": np.ptp(vertices, axis=0).tolist(),
        "vertices": len(vertices),
        "triangles": len(mesh.faces),
        "welded_watertight": True,
        "welded_winding_consistent": True,
        "welded_volume_m3": float(physical.volume),
        "license": (
            "ShapeNet research/education noncommercial restrictions; no "
            "general source redistribution"
        ),
        "status": "prepared candidate; render/physics QA and final split pending",
    }
    (folder / "asset.json").write_text(json.dumps(meta, indent=2))
    return meta


def main():
    families = json.loads((BASE / "cross-source-families-v1/families.json").read_text())
    source = json.loads((BASE / "shapenet-physical-audit-v1/physical-summary.json").read_text())
    lookup = {r["asset_key"]: r for r in source["rows"]}
    chosen = []
    for group in families["groups"]:
        if not group["eligible_candidates"]:
            continue
        key = sorted(group["eligible_candidates"], key=lambda k: (not k.startswith("gso/"), k))[0]
        if key.startswith("shapenet/"):
            chosen.append(lookup[key.split("/", 1)[1]])
    results = []
    failures = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [(r, pool.submit(prepare, r)) for r in chosen]
        for i, (r, f) in enumerate(futures):
            try:
                results.append(f.result())
            except Exception as exc:
                failures.append({"asset_key": r["asset_key"], "error": repr(exc)})
            if (i + 1) % 20 == 0:
                print(
                    "ShapeNet render prep",
                    i + 1,
                    "/",
                    len(chosen),
                    "failures",
                    len(failures),
                    flush=True,
                )
    report = {
        "status": "candidate_preparation_complete",
        "selected": len(chosen),
        "prepared": results,
        "failures": failures,
        "physical_priors": PRIORS,
        "family_manifest_sha256": sha(BASE / "cross-source-families-v1/families.json"),
        "script_sha256": sha(Path(__file__)),
    }
    (BASE / "shapenet-render-candidates-v1.json").write_text(json.dumps(report, indent=2))
    print("prepared", len(results), "failures", len(failures), flush=True)


if __name__ == "__main__":
    main()
