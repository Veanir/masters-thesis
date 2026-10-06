"""Bind available family representatives; deliberately does not assign final splits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.common.paths import DATASET_ROOT, script_help
from scripts.common.paths import ROOT as WORKSPACE_ROOT

script_help(__doc__, __name__)

ROOT = WORKSPACE_ROOT
BASE = ROOT / "runs/research-evolution-assets-20260906"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = BASE / "cross-source-families-v1/families.json"
    families = json.loads(source.read_text())
    rows = []
    failures = []
    pilots = {
        r["asset_id"] for r in json.loads((BASE / "pilot-assets.json").read_text())["prepared"]
    }
    for group in families["groups"]:
        if not group["eligible_candidates"]:
            continue
        key = sorted(group["eligible_candidates"], key=lambda k: (not k.startswith("gso/"), k))[0]
        origin, name = key.split("/", 1)
        path = (
            DATASET_ROOT
            / f"{origin}-evolution-v1"
            / "prepared"
            / (name if origin == "gso" else "shapenet-" + name)
        )
        if not (path / "asset.json").exists():
            failures.append(
                {
                    "key": key,
                    "reason": "Representative preparation did not pass; no silent replacement",
                }
            )
            continue
        meta = json.loads((path / "asset.json").read_text())
        for filename, field in [
            ("mesh.npz", "mesh_npz_sha256"),
            ("physical-mesh.npz", "physical_mesh_sha256"),
            ("texture.png", "texture_sha256"),
        ]:
            assert sha(path / filename) == meta[field]
        rows.append(
            {
                "key": key,
                "asset_id": meta["asset_id"],
                "family_id": group["family_id"],
                "family_members": group["members"],
                "path": str(path.resolve()),
                "asset_json_sha256": sha(path / "asset.json"),
                "source": origin,
                "category": meta.get("category", meta.get("catalog_categories", ["unknown"])[0]),
                "extents_m": meta["extents_m"],
                "vertices": meta["vertices"],
                "triangles": meta["triangles"],
                "split": None,
                "family_used_in_renderer_pilot": any(
                    k.startswith("gso/") and k.split("/", 1)[1] in pilots for k in group["members"]
                ),
            }
        )
    report = {
        "status": "available_source_representatives_before_renderer_gate",
        "rows": rows,
        "preparation_failures": failures,
        "family_manifest_sha256": sha(source),
        "script_sha256": sha(Path(__file__)),
        "proposed_size_only": {
            "train_families": 320,
            "synthetic_validation_families": 40,
            "observations_per_family": 20,
        },
        "constraints": (
            "Final size, splitting and role assignment require pilot "
            "physics/calibration/timing QA. Every family stays in one split, "
            "including all occluders. Families inspected in renderer "
            "development may not become an untouched synthetic test. HB is "
            "already reserved and excluded."
        ),
    }
    (BASE / "available-candidates-v1.json").write_text(json.dumps(report, indent=2))
    print(
        "Bound available representatives",
        len(rows),
        "preparation failures",
        len(failures),
        flush=True,
    )


if __name__ == "__main__":
    main()
