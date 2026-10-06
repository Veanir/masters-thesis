"""Source geometry inventory only; physical scale and split admission stay pending."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from scripts.common.paths import DATASET_ROOT, script_help
from scripts.common.paths import ROOT as WORKSPACE_ROOT

script_help(__doc__, __name__)


ROOT = WORKSPACE_ROOT
OUT = ROOT / "runs/research-evolution-assets-20260906"
DATA = DATASET_ROOT / "shapenet-evolution-v1/models"


def main():
    downloaded = json.loads((OUT / "shapenet-download.json").read_text())
    rows = []
    for category in downloaded["rows"]:
        for name in category["models"]:
            path = DATA / name
            raw = path.read_bytes()
            text = raw.decode("utf-8", errors="replace")
            points = np.array(
                [
                    [float(c) for c in line.split()[1:4]]
                    for line in text.splitlines()
                    if line.startswith("v ")
                ]
            )
            faces = sum(line.startswith("f ") for line in text.splitlines())
            assert points.ndim == 2 and points.shape[1] == 3 and np.isfinite(points).all()
            extents = np.ptp(points, axis=0)
            center = (points.max(axis=0) + points.min(axis=0)) / 2
            normalized = np.unique(
                np.round((points - center) / extents.max(), decimals=5), axis=0
            ).astype("<f4")
            row = {
                "source_id": Path(name).parts[1],
                "synset": category["synset"],
                "category": category["category"],
                "path": name,
                "obj_sha256": hashlib.sha256(raw).hexdigest(),
                "vertices": len(points),
                "faces": faces,
                "normalized_extents": extents.tolist(),
                "bbox_diagonal": float(np.linalg.norm(extents)),
                "scale_translation_invariant_vertex_hash": hashlib.sha256(
                    normalized.tobytes()
                ).hexdigest(),
                "material_libraries": [
                    line[7:].strip() for line in text.splitlines() if line.startswith("mtllib ")
                ],
                "has_uv": any(line.startswith("vt ") for line in text.splitlines()),
                "eligibility": (
                    "source candidate only; metric rescaling, topology and "
                    "near-duplicate audit pending"
                ),
            }
            rows.append(row)
    hashes = Counter(r["scale_translation_invariant_vertex_hash"] for r in rows)
    report = {
        "status": "source_inventory_complete",
        "source_revision": downloaded["revision"],
        "model_count": len(rows),
        "category_counts": dict(Counter(r["category"] for r in rows)),
        "exact_normalized_vertex_groups": len(hashes),
        "duplicate_excess": sum(n - 1 for n in hashes.values()),
        "grouping_limitation": (
            "Identical vertex-set hashes group exact candidates only; not "
            "proof against near duplicates or re-triangulated replicas"
        ),
        "units": (
            "Normalized geometry, not assumed metres; physical target "
            "dimensions must be assigned explicitly before rendering"
        ),
        "source_axes": "ShapeNet v2 +Y up, -Z front; convert consistently for Z-up physics",
        "rows": rows,
    }
    (OUT / "shapenet-geometry-inventory.json").write_text(json.dumps(report, indent=2))
    print(
        {
            k: report[k]
            for k in (
                "model_count",
                "category_counts",
                "exact_normalized_vertex_groups",
                "duplicate_excess",
            )
        }
    )


if __name__ == "__main__":
    main()
