"""CLI for preparing object assets."""

from __future__ import annotations

import argparse
from pathlib import Path

from masters_rgbd.synthetic.assets import prepare_assets, prepare_trellis_glb_assets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare object assets for dataset generation.")
    parser.add_argument("--source", type=Path, required=True, help="Path to source object assets.")
    parser.add_argument(
        "--source-format",
        choices=("mso", "trellis-glb"),
        default="mso",
        help="Input layout: mujoco_scanned_objects directories or TRELLIS.2 GLB files.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("generated"),
        help="Dataset root containing objects/ and scenes/.",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Optional maximum number of objects."
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite prepared object dirs.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.source_format == "trellis-glb":
        prepared = prepare_trellis_glb_assets(
            args.source,
            args.dataset_root / "objects",
            limit=args.limit,
            overwrite=args.overwrite,
        )
    else:
        prepared = prepare_assets(
            args.source,
            args.dataset_root / "objects",
            limit=args.limit,
            overwrite=args.overwrite,
        )
    print(f"Prepared {len(prepared)} objects in {args.dataset_root / 'objects'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
