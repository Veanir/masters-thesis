"""CLI for preparing procedural background surface materials."""

from __future__ import annotations

import argparse
from pathlib import Path

from masters_rgbd.synthetic.backgrounds import (
    available_procedural_surface_ids,
    prepare_procedural_backgrounds,
    prepare_trellis_background_assets,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare procedural PBR-like surface materials for SAPIEN backgrounds."
    )
    parser.add_argument(
        "--source-format",
        choices=("procedural", "trellis-glb"),
        default="procedural",
        help="Background source format to prepare.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Source directory for --source-format trellis-glb.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("generated_sapien_trellis"),
        help="Dataset root where backgrounds/ should be created.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Explicit output directory. Defaults to <dataset-root>/backgrounds.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Procedural texture seed.")
    parser.add_argument("--size", type=int, default=512, help="Texture width and height in pixels.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit prepared TRELLIS GLB backgrounds.",
    )
    parser.add_argument(
        "--surface-id",
        action="append",
        choices=available_procedural_surface_ids(),
        help="Prepare only this surface id. Can be passed multiple times.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing surface dirs.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = args.output if args.output is not None else args.dataset_root / "backgrounds"
    if args.source_format == "trellis-glb":
        if args.source is None:
            raise ValueError("--source is required when --source-format trellis-glb")
        assets = prepare_trellis_background_assets(
            args.source,
            output_dir,
            limit=args.limit,
            overwrite=args.overwrite,
        )
        asset_ids = ", ".join(asset.background_id for asset in assets)
        print(f"Prepared {len(assets)} TRELLIS background meshes in {output_dir}: {asset_ids}")
        return 0

    surfaces = prepare_procedural_backgrounds(
        output_dir,
        seed=args.seed,
        size=args.size,
        overwrite=args.overwrite,
        surface_ids=tuple(args.surface_id) if args.surface_id else None,
    )
    surface_ids = ", ".join(surface.surface_id for surface in surfaces)
    print(f"Prepared {len(surfaces)} procedural background surfaces in {output_dir}: {surface_ids}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
