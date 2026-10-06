"""Prepare 24 textured physical objects for renderer QA; no final split claim."""

import hashlib
import json
import shutil
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import trimesh
from PIL import Image

from scripts.common.paths import DATASET_ROOT, script_help
from scripts.common.paths import ROOT as WORKSPACE_ROOT

script_help(__doc__, __name__)


ROOT = WORKSPACE_ROOT
OUT = ROOT / "runs/research-evolution-assets-20260906"
DATA = DATASET_ROOT / "gso-evolution-v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(item):
    name = item["name"]
    archive = DATA / "archives" / (name + ".zip")
    encoded = urllib.parse.quote(name, safe="")
    url = f"https://fuel.gazebosim.org/1.0/GoogleResearch/models/{encoded}/1/{encoded}.zip"
    if not archive.exists():
        partial = archive.with_suffix(".partial")
        with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as f:
            shutil.copyfileobj(response, f)
        assert 0 < partial.stat().st_size < 100_000_000
        partial.rename(archive)
    model_dir = DATA / "models" / name
    model_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            assert (model_dir / member.filename).resolve().is_relative_to(model_dir.resolve())
        z.extractall(model_dir)
    obj = next(model_dir.rglob("model.obj"))
    metadata = next(model_dir.rglob("metadata.pbtxt"))
    text = metadata.read_text()
    assert f'name: "{name}"' in text and "Creative Commons Attribution 4.0 International" in text
    sdf = next(model_dir.rglob("model.sdf")).read_text()
    assert "<scale>" not in sdf, "Resolve explicit mesh scaling before admission"
    mesh = trimesh.load(obj, force="mesh", process=False)
    assert isinstance(mesh, trimesh.Trimesh) and np.isfinite(mesh.vertices).all()
    assert 0.03 <= mesh.extents.max() <= 0.6, f"Outside tabletop pilot size: {mesh.extents}"
    assert len(mesh.faces) > 20 and mesh.visual.kind == "texture"
    assert mesh.visual.uv is not None and len(mesh.visual.uv) == len(mesh.vertices)
    prepared = DATA / "prepared" / name
    prepared.mkdir(parents=True, exist_ok=True)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    offset = (vertices.max(axis=0) + vertices.min(axis=0)) / 2
    np.savez_compressed(
        prepared / "mesh.npz",
        vertices=vertices - offset,
        faces=mesh.faces.astype(np.int32),
        uv=np.asarray(mesh.visual.uv, dtype=np.float32),
    )
    physical = trimesh.Trimesh(vertices=vertices - offset, faces=mesh.faces, process=True)
    np.savez_compressed(
        prepared / "physical-mesh.npz",
        vertices=physical.vertices.astype(np.float32),
        faces=physical.faces.astype(np.int32),
    )
    # Fuel resolves the basename through SDF resource paths. A generic OBJ loader
    # does not search materials/textures, so bind that declared texture explicitly.
    mtl = next(model_dir.rglob("model.mtl")).read_text()
    assert "map_Kd texture.png" in mtl
    textures = list(model_dir.rglob("texture.png"))
    assert len(textures) == 1
    Image.open(textures[0]).convert("RGB").save(prepared / "texture.png")
    geometry_hash = hashlib.sha256(
        vertices.tobytes() + mesh.faces.astype(np.int32).tobytes()
    ).hexdigest()
    result = {
        "asset_id": name,
        "source_url": url,
        "version": 1,
        "license": item["license_name"],
        "catalog_categories": item.get("categories", ["unknown"]),
        "archive_sha256": sha(archive),
        "original_obj_sha256": sha(obj),
        "geometry_sha256": geometry_hash,
        "metadata_sha256": sha(metadata),
        "mesh_npz_sha256": sha(prepared / "mesh.npz"),
        "texture_sha256": sha(prepared / "texture.png"),
        "source_units": "m (SDFormat SI interpretation; no scale)",
        "scale_applied": 1.0,
        "centering_offset_m": offset.tolist(),
        "extents_m": mesh.extents.tolist(),
        "vertices": len(mesh.vertices),
        "triangles": len(mesh.faces),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "physical_mesh_sha256": sha(prepared / "physical-mesh.npz"),
        "welded_watertight": bool(physical.is_watertight),
        "welded_winding_consistent": bool(physical.is_winding_consistent),
        "welded_volume_m3": float(physical.volume),
        "physical_preprocessing": (
            "trimesh process=True merges UV seam vertices for topology audit; "
            "textured render mesh unchanged"
        ),
        "status": "renderer_pilot_only; collision and split-family admission pending",
    }
    (prepared / "asset.json").write_text(json.dumps(result, indent=2))
    print("prepared", name, flush=True)
    return result


def main():
    (DATA / "archives").mkdir(parents=True, exist_ok=True)
    catalog = json.loads((OUT / "gso-catalog.json").read_text())
    historical = {
        p.name for p in (DATASET_ROOT / "gso_evidence_candidate_v1/models").iterdir() if p.is_dir()
    }
    groups = ("Bottles and Cans and Cups", "Consumer Goods", "Toys", "Shoe", "Bag", "unknown")
    selected = []
    for category in groups:
        candidates = [
            r
            for r in catalog
            if r.get("categories", ["unknown"]) == [category]
            and r["name"] not in historical
            and r["license_name"] == "Creative Commons Attribution 4.0 International"
        ]
        candidates.sort(
            key=lambda r: hashlib.sha256(("evolution-24-v1/" + r["name"]).encode()).hexdigest()
        )
        selected.extend(candidates[:4])
    assert len(selected) == 24
    (OUT / "pilot-selection.json").write_text(
        json.dumps(
            {
                "purpose": "24 source-stratified renderer inputs, not final data",
                "seed": "evolution-24-v1",
                "excluded_historical_names": sorted(historical),
                "selected": selected,
            },
            indent=2,
        )
    )
    results, failures = [], []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [(item, pool.submit(prepare, item)) for item in selected]
        for item, future in futures:
            try:
                results.append(future.result())
            except Exception as exc:
                failures.append({"asset_id": item["name"], "error": str(exc)})
    (OUT / "pilot-assets.json").write_text(
        json.dumps({"prepared": results, "failures": failures}, indent=2)
    )
    print(json.dumps({"prepared": len(results), "failures": failures}), flush=True)


if __name__ == "__main__":
    main()
