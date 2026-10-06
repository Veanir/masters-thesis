"""Freeze disjoint candidate pools before full physics admission or training."""

from __future__ import annotations

import hashlib
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from scripts.common.paths import ROOT as WORKSPACE_ROOT
from scripts.common.paths import script_help

script_help(__doc__, __name__)

ROOT = WORKSPACE_ROOT
BASE = ROOT / "runs/research-evolution-assets-20260906"
OUT = ROOT / "runs/research-evolution-data-20260907"
SEED = "evolution-candidate-family-split-v1"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def rank(key):
    return hashlib.sha256((SEED + "/" + key).encode()).hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    path = OUT / "candidate-split-v1.json"
    assert not path.exists()
    inputs = BASE / "available-candidates-v1.json"
    rows = json.loads(inputs.read_text())["rows"]
    mixed = BASE / "mixed24-pilot-selection-v1.json"
    exposed = {r["family_id"] for r in json.loads(mixed.read_text())["rows"]}
    excluded = []
    eligible = []
    for row in rows:
        if row["asset_id"] == "Jansport_School_Backpack_Blue_Streak":
            excluded.append(
                {
                    "key": row["key"],
                    "family_id": row["family_id"],
                    "reason": (
                        "With the selected fine collider recipe, repeated actual-mesh "
                        "floor penetration about10.44mm; reject this family at the source "
                        "pilot gate, retaining the2mm tolerance."
                    ),
                }
            )
            continue
        row = {
            **row,
            "family_used_in_renderer_pilot": row["family_used_in_renderer_pilot"]
            or row["family_id"] in exposed,
        }
        eligible.append(row)
    strata = defaultdict(list)
    for row in eligible:
        if not row["family_used_in_renderer_pilot"]:
            strata[row["source"] + "/" + row["category"]].append(row)
    n = sum(map(len, strata.values()))
    desired = {k: 46 * len(v) / n for k, v in strata.items()}
    allocation = {k: int(v) for k, v in desired.items()}
    for k in sorted(strata, key=lambda k: (-(desired[k] - allocation[k]), rank(k)))[
        : 46 - sum(allocation.values())
    ]:
        allocation[k] += 1
    validation = set()
    for key, values in strata.items():
        validation.update(
            r["family_id"] for r in sorted(values, key=lambda r: rank(r["key"]))[: allocation[key]]
        )
    assert len(validation) == 46
    for row in eligible:
        row["split_pool"] = "synthetic_validation" if row["family_id"] in validation else "train"
    for split in ["train", "synthetic_validation"]:
        subset = sorted(
            (r for r in eligible if r["split_pool"] == split), key=lambda r: rank(r["key"])
        )
        for index, row in enumerate(subset):
            row["admission_priority"] = index
    assert not any(
        r["family_used_in_renderer_pilot"] and r["split_pool"] != "train" for r in eligible
    )
    report = {
        "status": "candidate_family_pools_frozen_before_full_physics_admission",
        "seed": SEED,
        "rows": eligible,
        "excluded": excluded,
        "source_manifest_sha256": sha(inputs),
        "mixed_pilot_manifest_sha256": sha(mixed),
        "counts": dict(Counter(r["split_pool"] for r in eligible)),
        "planned_final_counts": {"train": 320, "synthetic_validation": 40},
        "physics_admission": (
            "Two deterministic initial orientations per candidate. Both "
            "require last0.5s max vertex movement<=1mm and actual mesh floor "
            "penetration<=2mm, positive closed source volume. Admit in stored "
            "priority order within each pool; reserve candidates never cross "
            "splits. Final scene pairs separately require actual mesh Boolean "
            "QA."
        ),
        "shortfall_policy": (
            "If fewer than the planned counts pass, stop before full "
            "generation/training and record a revised size based on source QA "
            "and budget; never move families across pools or choose by model "
            "performance."
        ),
        "occluder_policy": (
            "After final admission, all target and occluder geometries must "
            "come only from the same finalized split. Reserve candidates not "
            "admitted may not appear as occluders."
        ),
        "script_sha256": sha(Path(__file__)),
    }
    path.write_text(json.dumps(report, indent=2))
    archive = ROOT / "build/research-evolution/isaac-candidates396-v1.zip"
    assert not archive.exists()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(path, "candidate-split.json")
        for index, row in enumerate(eligible):
            directory = Path(row["path"])
            assert sha(directory / "asset.json") == row["asset_json_sha256"]
            for name in ["asset.json", "mesh.npz", "physical-mesh.npz", "texture.png"]:
                z.write(directory / name, row["asset_id"] + "/" + name)
            if (index + 1) % 50 == 0:
                print("bundle candidates", index + 1, "/", len(eligible), flush=True)
    (OUT / "candidate-bundle-v1.json").write_text(
        json.dumps(
            {
                "sha256": sha(archive),
                "size_bytes": archive.stat().st_size,
                "manifest_sha256": sha(path),
            },
            indent=2,
        )
    )
    print("Candidate pools", report["counts"], "bundle bytes", archive.stat().st_size, flush=True)


if __name__ == "__main__":
    main()
