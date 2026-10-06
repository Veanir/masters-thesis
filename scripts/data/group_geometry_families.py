"""Pre-split replica screen across source candidates and reserved HB geometry.

Geometry-only screening; no predictions, sensor frames, or model scores read.
PCA plus axis permutations/reflections handles arbitrary source orientations.
Finite surface sampling remains a heuristic, not an independence certificate.
"""

import hashlib
import itertools
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from scripts.common.paths import DATASET_ROOT, script_help
from scripts.common.paths import ROOT as WORKSPACE_ROOT

script_help(__doc__, __name__)


ROOT = WORKSPACE_ROOT
BASE = ROOT / "runs/research-evolution-assets-20260906"
OUT = BASE / "cross-source-families-v1"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def candidates():
    rows = []
    for r in json.loads((BASE / "gso-source-pool-prepared-v1.json").read_text())["prepared"]:
        rows.append(
            {
                "key": "gso/" + r["asset_id"],
                "source": "gso",
                "candidate": bool(
                    r["welded_watertight"]
                    and r["welded_winding_consistent"]
                    and r["welded_volume_m3"] > 0
                ),
                "path": str(
                    DATASET_ROOT / "gso-evolution-v1/prepared" / r["asset_id"] / "physical-mesh.npz"
                ),
            }
        )
    summary = json.loads((BASE / "shapenet-physical-audit-v1/physical-summary.json").read_text())
    assert summary["audited"] == 1006 and not summary["failures"]
    for r in summary["rows"]:
        if r["physical_candidate"]:
            rows.append(
                {
                    "key": "shapenet/" + r["asset_key"],
                    "source": "shapenet",
                    "candidate": True,
                    "path": str(
                        BASE / "shapenet-physical-audit-v1" / (r["asset_key"] + "-geometry.npz")
                    ),
                }
            )
    for p in sorted((DATASET_ROOT / "bop-hb-evolution-v1/hb/models").glob("obj_*.ply")):
        rows.append(
            {"key": "hb/" + p.stem, "source": "hb_reserved", "candidate": False, "path": str(p)}
        )
    for p in sorted((DATASET_ROOT / "gso_evidence_candidate_v1/models").rglob("model.obj")):
        rows.append(
            {
                "key": "historical/" + p.parent.parent.name,
                "source": "historical_gso",
                "candidate": False,
                "path": str(p),
            }
        )
    assert sum(r["source"] == "hb_reserved" for r in rows) == 33
    assert len({r["key"] for r in rows}) == len(rows)
    return rows


def features(row):
    path = Path(row["path"])
    digest = sha(path)
    dest = OUT / (hashlib.sha256(row["key"].encode()).hexdigest()[:24] + ".npz")
    if dest.exists():
        z = np.load(dest)
        assert str(z["source_sha256"]) == digest
        return z["points"]
    if path.suffix == ".npz":
        z = np.load(path)
        mesh = trimesh.Trimesh(vertices=z["vertices"], faces=z["faces"], process=False)
    else:
        mesh = trimesh.load(path, force="mesh", process=False, allow_remote=False)
    points, _ = trimesh.sample.sample_surface(mesh, 4096, seed=20260907)
    centered = points - points.mean(axis=0)
    _, basis = np.linalg.eigh(centered.T @ centered)
    points = centered @ basis[:, ::-1]
    points = (points - (points.max(axis=0) + points.min(axis=0)) / 2) / np.ptp(points, axis=0).max()
    points = points.astype(np.float32)
    np.savez_compressed(dest, points=points, source_sha256=digest)
    return points


def main():
    OUT.mkdir(exist_ok=True)
    start = time.monotonic()
    rows = candidates()
    points = [features(r) for r in rows]
    extents = np.array([np.ptp(p, axis=0) for p in points])
    trees = [cKDTree(p) for p in points]
    parent = list(range(len(rows)))
    links = []

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(j)] = find(i)

    lookup = {r["key"]: i for i, r in enumerate(rows)}
    previous = json.loads((BASE / "shapenet-physical-audit-v1/families.json").read_text())
    assert sum(len(g["members"]) for g in previous["groups"]) == 1006
    for group in previous["groups"]:
        ids = [lookup["shapenet/" + k] for k in group["members"] if "shapenet/" + k in lookup]
        for i in ids[1:]:
            union(ids[0], i)
    tested = 0
    for i in range(len(rows)):
        for j in range(i):
            if find(i) == find(j):
                continue
            if np.max(np.abs(np.sort(extents[i]) - np.sort(extents[j]))) > 0.05:
                continue
            for order in itertools.permutations(range(3)):
                if np.max(np.abs(extents[i] - extents[j, list(order)])) > 0.05:
                    continue
                for signs in itertools.product((-1, 1), repeat=3):
                    moved = points[j][:, order] * np.array(signs)
                    coarse = trees[i].query(moved[::8], workers=1)[0]
                    tested += 1
                    if coarse.mean() > 0.022 or np.quantile(coarse, 0.95) > 0.045:
                        continue
                    forward = trees[i].query(moved, workers=1)[0]
                    if forward.mean() > 0.015 or np.quantile(forward, 0.95) > 0.03:
                        continue
                    backward = cKDTree(moved).query(points[i], workers=1)[0]
                    if backward.mean() <= 0.015 and np.quantile(backward, 0.95) <= 0.03:
                        union(i, j)
                        links.append(
                            {
                                "left": rows[i]["key"],
                                "right": rows[j]["key"],
                                "mean_forward": float(forward.mean()),
                                "mean_backward": float(backward.mean()),
                                "p95_forward": float(np.quantile(forward, 0.95)),
                                "p95_backward": float(np.quantile(backward, 0.95)),
                                "order": order,
                                "signs": signs,
                            }
                        )
                        break
                if find(i) == find(j):
                    break
        if (i + 1) % 50 == 0:
            print(
                "cross-source",
                i + 1,
                "/",
                len(rows),
                "transforms",
                tested,
                "links",
                len(links),
                flush=True,
            )
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(find(i), []).append(row)
    result = []
    for members in groups.values():
        names = sorted(r["key"] for r in members)
        forbidden = any(r["source"] in ("hb_reserved", "historical_gso") for r in members)
        result.append(
            {
                "family_id": hashlib.sha256("\n".join(names).encode()).hexdigest()[:20],
                "members": names,
                "excluded_from_new_training": forbidden,
                "eligible_candidates": []
                if forbidden
                else sorted(r["key"] for r in members if r["candidate"]),
            }
        )
    report = {
        "status": "pre_split_cross_source_heuristic_complete",
        "rows": rows,
        "source_counts": dict(Counter(r["source"] for r in rows)),
        "groups": result,
        "links": links,
        "eligible_families": sum(bool(g["eligible_candidates"]) for g in result),
        "excluded_candidate_count": sum(
            sum(rows[lookup[k]]["candidate"] for k in g["members"])
            for g in result
            if g["excluded_from_new_training"]
        ),
        "seconds": time.monotonic() - start,
        "script_sha256": sha(Path(__file__)),
        "method": (
            "4096 deterministic surface samples; PCA orientation; 48 axis "
            "permutations/reflections; normalized max extent; bidirectional "
            "mean <=0.015 and p95 <=0.03; transitive components; inherit all "
            "ShapeNet source-family links."
        ),
        "limitations": (
            "Finite samples, PCA instability and limited transformations can "
            "miss replicas; conservative links can overmerge. Not a "
            "certificate against pretraining overlap. No model scores used."
        ),
    }
    (OUT / "families.json").write_text(json.dumps(report, indent=2))
    print(
        {
            k: report[k]
            for k in ("source_counts", "eligible_families", "excluded_candidate_count", "seconds")
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
