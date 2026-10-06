"""Fetch pinned public HOPE archives and audit annotation availability, without inference."""

from __future__ import annotations

import argparse
import urllib.request
import zipfile
from pathlib import Path

from scripts.common.paths import digest, read, save, script_help

script_help(__doc__, __name__)


def audit(manifest, output):
    source = read(manifest)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in source["files"]:
        filename = item["rfilename"]
        assert filename in (
            "hope_base.zip",
            "hope_models.zip",
            "hope_test_realsense.zip",
            "hope_val_realsense.zip",
        )
        target = output / filename
        url = (
            f"https://huggingface.co/datasets/bop-benchmark/hope/resolve/{source['sha']}/{filename}"
        )
        if not target.exists():
            part = target.with_suffix(".zip.part")
            with urllib.request.urlopen(url, timeout=120) as response, part.open("wb") as handle:
                while block := response.read(1024 * 1024):
                    handle.write(block)
            assert part.stat().st_size == item["size"] and digest(part) == item["lfs"]["sha256"]
            part.replace(target)
        assert target.stat().st_size == item["size"] and digest(target) == item["lfs"]["sha256"]
        with zipfile.ZipFile(target) as archive:
            names = archive.namelist()
            annotation = [
                n
                for n in names
                if n.endswith(("scene_gt.json", "scene_gt_info.json", "scene_camera.json"))
            ]
            row = {
                "file": filename,
                "sha256": digest(target),
                "members": len(names),
                "annotations": annotation,
                "rgb_images": sum("/rgb/" in n and n.endswith((".png", ".jpg")) for n in names),
                "depth_images": sum("/depth/" in n and n.endswith(".png") for n in names),
                "visible_masks": sum("/mask_visib/" in n and n.endswith(".png") for n in names),
            }
            (output / (filename + ".members.txt")).write_text("\n".join(names), encoding="utf-8")
        rows.append(row)
        save(
            output / "audit.json",
            {"revision": source["sha"], "status": "in_progress", "rows": rows},
        )
        print(f"{filename}: {len(annotation)} camera/GT files", flush=True)
    save(
        output / "audit.json",
        {
            "revision": source["sha"],
            "status": "complete",
            "rows": rows,
            "predictions_computed": False,
            "cohort_frozen": False,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.manifest, args.output)
