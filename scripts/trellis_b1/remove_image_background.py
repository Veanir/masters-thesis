"""Remove backgrounds from FLUX references and emit RGBA PNGs for TRELLIS.2."""

from __future__ import annotations

import argparse
import json
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

AlphaFn = Callable[[Image.Image], Image.Image]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images", type=Path, required=True, help="Directory with raw RGB references."
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Output directory for RGBA PNGs."
    )
    parser.add_argument(
        "--method",
        choices=("rmbg2", "border"),
        default="rmbg2",
        help=(
            "Background remover. rmbg2 uses briaai/RMBG-2.0; border is a "
            "cheap solid-background fallback."
        ),
    )
    parser.add_argument(
        "--model",
        default="briaai/RMBG-2.0",
        help="Hugging Face segmentation model id used by --method rmbg2.",
    )
    parser.add_argument("--device", default="cuda", help="Torch device for --method rmbg2.")
    parser.add_argument("--image-size", type=int, default=1024, help="RMBG input resolution.")
    parser.add_argument(
        "--alpha-threshold", type=int, default=16, help="Alpha threshold used for QC."
    )
    parser.add_argument("--min-foreground-fraction", type=float, default=0.025)
    parser.add_argument("--max-foreground-fraction", type=float, default=0.85)
    parser.add_argument("--border-margin", type=int, default=2)
    parser.add_argument("--contact-sheet-limit", type=int, default=120)
    parser.add_argument("--contact-sheet-rows", type=int, default=8)
    parser.add_argument("--contact-thumb-size", type=int, default=192)
    parser.add_argument("--fail-on-warning", action="store_true")
    parser.add_argument("--no-skip-existing", action="store_true")
    return parser


def _image_paths(root: Path) -> list[Path]:
    paths: list[Path] = []
    for pattern in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        paths.extend(root.glob(pattern))
    return sorted(paths)


def _load_rmbg2(
    model_id: str,
    device: str,
    image_size: int,
) -> AlphaFn:
    try:
        import torch
        from torchvision import transforms
        from transformers import AutoModelForImageSegmentation
    except ImportError as exc:
        raise RuntimeError(
            "RMBG-2.0 needs torch, torchvision, transformers and kornia. "
            "Install the optional TRELLIS asset environment dependencies."
        ) from exc

    if device == "cuda" and not torch.cuda.is_available():
        print("[background] cuda requested but unavailable; falling back to cpu")
        device = "cpu"

    model = AutoModelForImageSegmentation.from_pretrained(
        model_id,
        trust_remote_code=True,
    )
    model.eval().to(device)
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    def remove_background(image: Image.Image) -> Image.Image:
        rgb = image.convert("RGB")
        input_tensor = transform(rgb).unsqueeze(0).to(device)
        with torch.inference_mode():
            prediction = model(input_tensor)[-1].sigmoid().cpu()[0].squeeze()
        mask = transforms.ToPILImage()(prediction).resize(rgb.size, Image.Resampling.BILINEAR)
        rgba = rgb.convert("RGBA")
        rgba.putalpha(mask)
        return rgba

    return remove_background


def _remove_by_border_color(image: Image.Image) -> Image.Image:
    """Fallback for nearly solid backgrounds; not a semantic segmenter."""

    rgb = np.array(image.convert("RGB"))
    height, width = rgb.shape[:2]
    border = max(4, min(height, width) // 32)
    samples = np.concatenate(
        [
            rgb[:border, :, :].reshape(-1, 3),
            rgb[-border:, :, :].reshape(-1, 3),
            rgb[:, :border, :].reshape(-1, 3),
            rgb[:, -border:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    background = np.median(samples, axis=0)
    distance = np.linalg.norm(rgb.astype(np.float32) - background.astype(np.float32), axis=2)
    mask = (distance > 26).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep = 1 + int(np.argmax(areas))
        mask = np.where(labels == keep, 255, 0).astype(np.uint8)

    return Image.fromarray(np.dstack([rgb, mask]), mode="RGBA")


def _component_stats(mask: np.ndarray) -> tuple[int, float]:
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )
    if count <= 1:
        return 0, 0.0
    areas = stats[1:, cv2.CC_STAT_AREA]
    total = float(np.sum(areas))
    if total <= 0:
        return 0, 0.0
    return int(count - 1), float(np.max(areas) / total)


def _qc_alpha(
    image: Image.Image,
    *,
    alpha_threshold: int,
    min_foreground_fraction: float,
    max_foreground_fraction: float,
    border_margin: int,
) -> dict[str, Any]:
    rgba = image.convert("RGBA")
    alpha = np.array(rgba)[:, :, 3]
    mask = alpha > alpha_threshold
    height, width = mask.shape
    area = int(np.count_nonzero(mask))
    pixel_count = int(width * height)
    warnings: list[str] = []

    if area == 0:
        return {
            "foreground_fraction": 0.0,
            "bbox": None,
            "touches_border": False,
            "component_count": 0,
            "largest_component_fraction": 0.0,
            "warnings": ["empty_foreground"],
        }

    ys, xs = np.nonzero(mask)
    x0 = int(xs.min())
    y0 = int(ys.min())
    x1 = int(xs.max()) + 1
    y1 = int(ys.max()) + 1
    foreground_fraction = float(area / pixel_count)
    touches_border = (
        x0 <= border_margin
        or y0 <= border_margin
        or x1 >= width - border_margin
        or y1 >= height - border_margin
    )
    component_count, largest_component_fraction = _component_stats(mask)

    if foreground_fraction < min_foreground_fraction:
        warnings.append("foreground_too_small")
    if foreground_fraction > max_foreground_fraction:
        warnings.append("foreground_too_large")
    if touches_border:
        warnings.append("foreground_touches_border")
    if component_count > 4 and largest_component_fraction < 0.9:
        warnings.append("fragmented_foreground")

    lower_band = mask[int(height * 0.72) :, :]
    if foreground_fraction > 0.18 and np.max(np.mean(lower_band, axis=1), initial=0.0) > 0.65:
        warnings.append("large_horizontal_bottom_band")

    return {
        "foreground_fraction": foreground_fraction,
        "bbox": [x0, y0, x1, y1],
        "touches_border": touches_border,
        "component_count": component_count,
        "largest_component_fraction": largest_component_fraction,
        "warnings": warnings,
    }


def _checkerboard(size: tuple[int, int], cell: int = 16) -> Image.Image:
    width, height = size
    yy, xx = np.indices((height, width))
    board = ((xx // cell + yy // cell) % 2).astype(np.uint8)
    light = np.array([220, 220, 220], dtype=np.uint8)
    dark = np.array([165, 165, 165], dtype=np.uint8)
    rgb = np.where(board[:, :, None] == 0, light, dark)
    return Image.fromarray(rgb, mode="RGB")


def _thumbnail(image: Image.Image, size: int, background: str = "white") -> Image.Image:
    thumb = Image.new("RGB", (size, size), background)
    copy = image.copy()
    copy.thumbnail((size, size), Image.Resampling.LANCZOS)
    x = (size - copy.width) // 2
    y = (size - copy.height) // 2
    if copy.mode == "RGBA":
        thumb.paste(copy.convert("RGB"), (x, y), copy.getchannel("A"))
    else:
        thumb.paste(copy.convert("RGB"), (x, y))
    return thumb


def _alpha_preview(image: Image.Image) -> Image.Image:
    alpha = image.convert("RGBA").getchannel("A")
    return Image.merge("RGB", [alpha, alpha, alpha])


def _cutout_preview(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    board = _checkerboard(rgba.size)
    board.paste(rgba.convert("RGB"), (0, 0), rgba.getchannel("A"))
    return board


def _write_contact_sheets(
    rows: list[dict[str, Any]],
    output_dir: Path,
    *,
    limit: int,
    rows_per_sheet: int,
    thumb_size: int,
) -> None:
    if limit <= 0 or rows_per_sheet <= 0 or not rows:
        return

    selected = rows[:limit]
    label_height = 24
    columns = 3
    cell_width = thumb_size
    cell_height = thumb_size + label_height
    per_sheet = rows_per_sheet

    for sheet_index, start in enumerate(range(0, len(selected), per_sheet)):
        batch = selected[start : start + per_sheet]
        sheet = Image.new("RGB", (columns * cell_width, len(batch) * cell_height), "white")
        draw = ImageDraw.Draw(sheet)
        for row_index, row in enumerate(batch):
            raw = Image.open(row["source"]).convert("RGB")
            rgba = Image.open(row["output"]).convert("RGBA")
            previews = [
                _thumbnail(raw, thumb_size),
                _thumbnail(_alpha_preview(rgba), thumb_size),
                _thumbnail(_cutout_preview(rgba), thumb_size),
            ]
            y = row_index * cell_height
            for column, preview in enumerate(previews):
                sheet.paste(preview, (column * cell_width, y))
            label = f"{row['id']} | {','.join(row['qc']['warnings']) or 'ok'}"
            draw.text((4, y + thumb_size + 4), label[:90], fill=(0, 0, 0))
        sheet.save(output_dir / f"qc_contact_sheet_{sheet_index:03d}.png")


def _metadata_path(output_path: Path) -> Path:
    return output_path.with_suffix(".background.json")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    global cv2
    import cv2

    args.output.mkdir(parents=True, exist_ok=True)
    images = _image_paths(args.images)
    if not images:
        raise ValueError(f"No reference images found in {args.images}")

    if args.method == "rmbg2":
        remove_background = _load_rmbg2(args.model, args.device, args.image_size)
    else:
        remove_background = _remove_by_border_color

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    failure_path = args.output / "background_removal_failures.jsonl"
    failure_path.unlink(missing_ok=True)

    for image_path in images:
        output_path = args.output / f"{image_path.stem}.png"
        metadata_path = _metadata_path(output_path)
        if output_path.exists() and output_path.stat().st_size > 0 and not args.no_skip_existing:
            rgba = Image.open(output_path).convert("RGBA")
            qc = _qc_alpha(
                rgba,
                alpha_threshold=args.alpha_threshold,
                min_foreground_fraction=args.min_foreground_fraction,
                max_foreground_fraction=args.max_foreground_fraction,
                border_margin=args.border_margin,
            )
            rows.append(
                {
                    "id": image_path.stem,
                    "source": str(image_path),
                    "output": str(output_path),
                    "qc": qc,
                }
            )
            print(f"skipped existing {image_path.stem}")
            continue

        try:
            output_path.unlink(missing_ok=True)
            metadata_path.unlink(missing_ok=True)
            rgba = remove_background(Image.open(image_path))
            qc = _qc_alpha(
                rgba,
                alpha_threshold=args.alpha_threshold,
                min_foreground_fraction=args.min_foreground_fraction,
                max_foreground_fraction=args.max_foreground_fraction,
                border_margin=args.border_margin,
            )
            rgba.save(output_path)
            row = {
                "id": image_path.stem,
                "source": str(image_path),
                "output": str(output_path),
                "qc": qc,
            }
            metadata_path.write_text(
                json.dumps(
                    {
                        "id": image_path.stem,
                        "source": str(image_path),
                        "output": str(output_path),
                        "method": args.method,
                        "model": args.model if args.method == "rmbg2" else None,
                        "qc": qc,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            rows.append(row)
            warning_text = f" warnings={','.join(qc['warnings'])}" if qc["warnings"] else ""
            print(f"generated {output_path.name}{warning_text}")
        except Exception as exc:  # noqa: BLE001 - keep long production batches moving.
            output_path.unlink(missing_ok=True)
            metadata_path.unlink(missing_ok=True)
            failure = {
                "id": image_path.stem,
                "image": str(image_path),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            failures.append(failure)
            with failure_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(failure, sort_keys=True) + "\n")
            print(f"failed {image_path.stem}: {type(exc).__name__}: {exc}", flush=True)

    _write_contact_sheets(
        rows,
        args.output,
        limit=args.contact_sheet_limit,
        rows_per_sheet=args.contact_sheet_rows,
        thumb_size=args.contact_thumb_size,
    )

    warning_count = sum(1 for row in rows if row["qc"]["warnings"])
    report = {
        "input_dir": str(args.images),
        "output_dir": str(args.output),
        "method": args.method,
        "model": args.model if args.method == "rmbg2" else None,
        "image_count": len(images),
        "output_count": len(rows),
        "failure_count": len(failures),
        "warning_count": warning_count,
        "items": rows,
    }
    (args.output / "background_removal_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if not rows:
        return 1
    if args.fail_on_warning and warning_count:
        return 2
    if len(failures) == len(images):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
