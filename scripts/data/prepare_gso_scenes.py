"""Acquire a bounded candidate pool; final TRAIN/validation remain unassigned."""

import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from scripts.common.paths import script_help
from scripts.data.prepare_gso_assets import DATA, OUT, prepare, sha

script_help(__doc__, __name__)


QUOTAS = {
    "unknown": 100,
    "Consumer Goods": 100,
    "Toys": 80,
    "Bottles and Cans and Cups": 30,
    "Shoe": 20,
    "Bag": 12,
    "Action Figures": 12,
    "Legos": 6,
    "Board Games": 10,
    "Keyboard": 2,
    "Mouse": 2,
    "Headphones": 2,
}
SEED = "evolution-gso-source-pool-v1"


def cached_prepare(item):
    folder = DATA / "prepared" / item["name"]
    if (folder / "asset.json").exists():
        result = json.loads((folder / "asset.json").read_text())
        if all(
            sha(folder / name) == result[key]
            for name, key in [
                ("mesh.npz", "mesh_npz_sha256"),
                ("physical-mesh.npz", "physical_mesh_sha256"),
                ("texture.png", "texture_sha256"),
            ]
        ):
            return result
    return prepare(item)


def main():
    catalog = json.loads((OUT / "gso-catalog.json").read_text())
    previous = json.loads((OUT / "pilot-selection.json").read_text())
    excluded = set(previous["excluded_historical_names"])
    pilots = {r["name"] for r in previous["selected"]}
    selected = []
    for category, quota in QUOTAS.items():
        candidates = [
            r
            for r in catalog
            if r.get("categories", ["unknown"]) == [category]
            and r["name"] not in excluded
            and r["license_name"] == "Creative Commons Attribution 4.0 International"
        ]
        candidates.sort(
            key=lambda r: (
                r["name"] not in pilots,
                hashlib.sha256((SEED + "/" + r["name"]).encode()).hexdigest(),
            )
        )
        selected.extend(candidates[:quota])
    assert pilots <= set(r["name"] for r in selected)
    size = sum(r["filesize"] for r in selected)
    assert size < 6_000_000_000
    selection = {
        "version": 1,
        "seed": SEED,
        "purpose": (
            "Source candidate pool only; final splits and full rendering size "
            "require physical/family/timing gates"
        ),
        "catalog_sha256": sha(OUT / "gso-catalog.json"),
        "quotas": QUOTAS,
        "selected_count": len(selected),
        "expected_catalog_bytes": size,
        "excluded_historical_names": sorted(excluded),
        "selected": selected,
    }
    (OUT / "gso-source-pool-v1.json").write_text(json.dumps(selection, indent=2))
    results = []
    failures = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(cached_prepare, item): item for item in selected}
        for future in as_completed(futures):
            item = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                failures.append({"asset_id": item["name"], "error": str(exc)})
            if (len(results) + len(failures)) % 25 == 0:
                print(
                    "source pool",
                    len(results),
                    "prepared",
                    len(failures),
                    "failed of",
                    len(selected),
                    flush=True,
                )
    summary = {
        "status": "source_pool_preparation_complete",
        "selection_sha256": sha(OUT / "gso-source-pool-v1.json"),
        "prepared_count": len(results),
        "failures": failures,
        "prepared": sorted(results, key=lambda r: r["asset_id"]),
        "welded_watertight_count": sum(r["welded_watertight"] for r in results),
        "category_counts": dict(Counter(k for r in results for k in r["catalog_categories"])),
    }
    (OUT / "gso-source-pool-prepared-v1.json").write_text(json.dumps(summary, indent=2))
    print(
        {
            k: summary[k]
            for k in ("status", "prepared_count", "welded_watertight_count", "category_counts")
        },
        flush=True,
    )
    print("failure_count", len(failures), flush=True)


if __name__ == "__main__":
    main()
