"""Snapshot official GSO metadata, independent of reconstruction scores."""

import json
import time
import urllib.parse
import urllib.request

from scripts.common.paths import ROOT as WORKSPACE_ROOT
from scripts.common.paths import script_help

script_help(__doc__, __name__)

ROOT = WORKSPACE_ROOT
OUT = ROOT / "runs/research-evolution-assets-20260906"
API = "https://fuel.gazebosim.org/1.0"


def get(url):
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return json.load(response)
        except Exception:
            if attempt == 2:
                raise
            time.sleep(3)


def main():
    OUT.mkdir(exist_ok=True)
    items = []
    for page in range(1, 31):
        path = OUT / f"gso-page-{page:02d}.json"
        if path.exists():
            rows = json.loads(path.read_text())
        else:
            query = urllib.parse.urlencode(
                {
                    "page": page,
                    "per_page": 100,
                    "q": "collections:Scanned Objects by Google Research",
                }
            )
            rows = get(API + "/models?" + query)
            assert isinstance(rows, list)
            path.write_text(json.dumps(rows, indent=2))
        if not rows:
            break
        items.extend(rows)
        print("page", page, "count", len(rows), flush=True)
    else:
        raise RuntimeError("GSO pagination bound exceeded")
    unique = {item["name"]: item for item in items if item["owner"] == "GoogleResearch"}
    (OUT / "gso-catalog.json").write_text(json.dumps(list(unique.values()), indent=2))
    print("unique official objects", len(unique), flush=True)
    print("first metadata", json.dumps(next(iter(unique.values()))), flush=True)


if __name__ == "__main__":
    main()
