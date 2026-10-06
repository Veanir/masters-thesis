"""Audit source surfaces and conservatively group near replicas before splitting."""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from scripts.common.paths import DATASET_ROOT, script_help
from scripts.common.paths import ROOT as WORKSPACE_ROOT

script_help(__doc__, __name__)


ROOT = WORKSPACE_ROOT
OUT = ROOT / "runs/research-evolution-assets-20260906/shapenet-physical-audit-v1"
DATA = DATASET_ROOT / "shapenet-evolution-v1/models"
VERSION = "shapenet-physical-source-audit-v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(row):
    name = row["synset"] + "-" + row["source_id"]
    result_path = OUT / (name + ".json")
    if result_path.exists():
        result = json.loads(result_path.read_text())
        assert result["version"] == VERSION and result["obj_sha256"] == row["obj_sha256"]
        return result
    path = DATA / row["path"]
    assert sha(path) == row["obj_sha256"]
    scene = trimesh.load(path, force="scene", process=False, allow_remote=False)
    material_rows = []
    for key, part in scene.geometry.items():
        material = getattr(part.visual, "material", None)
        image = getattr(material, "image", None)
        diffuse = getattr(material, "diffuse", None)
        material_rows.append(
            {
                "name": key,
                "faces": len(part.faces),
                "visual_kind": part.visual.kind,
                "has_texture_image": image is not None,
                "diffuse_rgba": None if diffuse is None else np.asarray(diffuse).tolist(),
            }
        )
    merged = scene.to_mesh()
    # Build a geometry-only mesh so seam vertices merge independently of UVs.
    physical = trimesh.Trimesh(vertices=merged.vertices, faces=merged.faces, process=True)
    before = len(physical.faces)
    physical.update_faces(physical.nondegenerate_faces())
    physical.update_faces(physical.unique_faces())
    physical.remove_unreferenced_vertices()
    physical.fix_normals(multibody=True)
    extents = physical.extents
    reasons = []
    if not physical.is_watertight:
        reasons.append("open_surface")
    if not physical.is_winding_consistent or not physical.is_volume:
        reasons.append("not_consistent_positive_volume")
    if extents.min() / extents.max() < 0.03:
        reasons.append("near_planar_for_rigid_tabletop_role")
    if len(physical.faces) > 200_000:
        reasons.append("exceeds_pilot_triangle_limit")
    texture_refs = []
    for mtl in path.parent.glob("*.mtl"):
        for line in mtl.read_text(errors="replace").splitlines():
            if line.startswith("map_Kd "):
                ref = line[7:].strip()
                target = (mtl.parent / ref).resolve()
                exists = target.is_file() and target.is_relative_to(DATA.resolve())
                texture_refs.append({"reference": ref, "exists_inside_dataset": exists})
    if any(not r["exists_inside_dataset"] for r in texture_refs):
        reasons.append("unresolved_declared_texture")
    points, _ = trimesh.sample.sample_surface(physical, 4096, seed=20260907)
    points = (points - physical.bounds.mean(axis=0)) / extents.max()
    np.savez_compressed(
        OUT / (name + "-geometry.npz"),
        points=points.astype(np.float32),
        vertices=(physical.vertices - physical.bounds.mean(axis=0)).astype(np.float32),
        faces=physical.faces.astype(np.int32),
    )
    result = {
        "version": VERSION,
        **row,
        "asset_key": name,
        "parts": material_rows,
        "texture_references": texture_refs,
        "watertight_after_seam_weld": bool(physical.is_watertight),
        "consistent_positive_volume": bool(physical.is_volume),
        "normalized_volume": float(physical.volume),
        "cleaned_faces": len(physical.faces),
        "removed_degenerate_or_duplicate_faces": before - len(physical.faces),
        "physical_candidate": not reasons,
        "rejection_reasons": reasons,
        "geometry_npz_sha256": sha(OUT / (name + "-geometry.npz")),
    }
    result_path.write_text(json.dumps(result, indent=2))
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inventory = json.loads((OUT.parent / "shapenet-geometry-inventory.json").read_text())
    started = time.monotonic()
    rows = []
    failures = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [(row, pool.submit(audit, row)) for row in inventory["rows"]]
        for index, (source, future) in enumerate(futures):
            try:
                rows.append(future.result())
            except Exception as exc:
                failures.append({"source_id": source["source_id"], "error": str(exc)})
            if (index + 1) % 100 == 0:
                print(
                    "source audit",
                    index + 1,
                    "/",
                    len(futures),
                    "failures",
                    len(failures),
                    flush=True,
                )
    summary = {
        "version": VERSION,
        "source_count": len(inventory["rows"]),
        "audited": len(rows),
        "failures": failures,
        "physical_candidates": sum(r["physical_candidate"] for r in rows),
        "candidate_categories": dict(
            Counter(r["category"] for r in rows if r["physical_candidate"])
        ),
        "rejection_reasons": dict(
            Counter(reason for r in rows for reason in r["rejection_reasons"])
        ),
        "script_sha256": sha(Path(__file__)),
        "seconds": time.monotonic() - started,
        "rows": rows,
    }
    (OUT / "physical-summary.json").write_text(json.dumps(summary, indent=2))
    print(
        {
            k: summary[k]
            for k in ("physical_candidates", "candidate_categories", "rejection_reasons")
        },
        flush=True,
    )
    assert not failures, (
        f"{len(failures)} source audit errors; inspect summary and resume cached successful "
        f"assets before grouping"
    )
    # Group across categories, using all audited geometries, so rejection does not
    # hide links between a candidate and its replicas. This is a conservative
    # geometric family heuristic, not a certificate of semantic independence.
    points = [np.load(OUT / (r["asset_key"] + "-geometry.npz"))["points"] for r in rows]
    extents = np.array([np.ptp(p, axis=0) for p in points])
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(j)] = find(i)

    exact = {}
    for i, row in enumerate(rows):
        key = row["scale_translation_invariant_vertex_hash"]
        if key in exact:
            union(exact[key], i)
        else:
            exact[key] = i
    trees = [cKDTree(p) for p in points]
    links = []
    tested = 0
    for i in range(len(rows)):
        for j in range(i):
            if find(i) == find(j):
                continue
            # Source +Y is upright. Test four yaw orientations and mirror cases.
            for swap in (False, True):
                order = [2, 1, 0] if swap else [0, 1, 2]
                if np.max(np.abs(extents[i] - extents[j, order])) > 0.04:
                    continue
                for sx, sz in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
                    transformed = points[j][:, order] * np.array([sx, 1, sz])
                    forward = trees[i].query(transformed, workers=1)[0]
                    tested += 1
                    if np.mean(forward) > 0.015 or np.quantile(forward, 0.95) > 0.03:
                        continue
                    backward = cKDTree(transformed).query(points[i], workers=1)[0]
                    if np.mean(backward) <= 0.015 and np.quantile(backward, 0.95) <= 0.03:
                        union(i, j)
                        links.append(
                            {
                                "left": rows[i]["asset_key"],
                                "right": rows[j]["asset_key"],
                                "mean_forward": float(np.mean(forward)),
                                "mean_backward": float(np.mean(backward)),
                                "p95_forward": float(np.quantile(forward, 0.95)),
                                "p95_backward": float(np.quantile(backward, 0.95)),
                                "swap_xz": swap,
                                "sign_x": sx,
                                "sign_z": sz,
                            }
                        )
                        break
                if find(i) == find(j):
                    break
        if (i + 1) % 100 == 0:
            print(
                "family audit", i + 1, "tested transforms", tested, "links", len(links), flush=True
            )
    members = {}
    for i, row in enumerate(rows):
        members.setdefault(find(i), []).append(row["asset_key"])
    groups = []
    for names in members.values():
        names = sorted(names)
        groups.append(
            {
                "family_id": "shapenet-family-"
                + hashlib.sha256("\n".join(names).encode()).hexdigest()[:20],
                "members": names,
            }
        )
    result = {
        "version": VERSION,
        "status": "source_family_heuristic_complete",
        "groups": groups,
        "near_links": links,
        "method": (
            "Connected components of exact normalized vertex-set matches or "
            "bidirectional nearest-surface-sample mean <=0.015 and p95 <=0.03 "
            "normalized max extent; 4096 uniform surface samples, source-up "
            "yaw and horizontal mirrors; bbox tolerance 0.04"
        ),
        "limitations": (
            "Finite sampling can miss local details; transitive components "
            "may overgroup distinct shapes. Review boundary examples and "
            "cross-source replicas before final split."
        ),
        "tested_transforms": tested,
        "seconds": time.monotonic() - started,
        "script_sha256": sha(Path(__file__)),
    }
    (OUT / "families.json").write_text(json.dumps(result, indent=2))
    print("family groups", len(groups), "completed in", result["seconds"], "seconds", flush=True)


if __name__ == "__main__":
    main()
