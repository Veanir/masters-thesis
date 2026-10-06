"""Batch-convert reference images to PBR GLB assets with TRELLIS.2."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

trellis_root = Path(os.environ.get("RGBD_TRELLIS_ROOT", "vendor/TRELLIS.2")).resolve()
if trellis_root.exists():
    sys.path.insert(0, str(trellis_root))


class PassthroughRembg:
    """Avoid gated RMBG downloads when the input already carries an alpha mask."""

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass

    def to(self, _device: str) -> PassthroughRembg:
        return self

    def cuda(self) -> PassthroughRembg:
        return self

    def cpu(self) -> PassthroughRembg:
        return self

    def __call__(self, image: Image.Image) -> Image.Image:
        return image.convert("RGBA")


def _dino_v3_extract_features_compat(self: object, image: torch.Tensor) -> torch.Tensor:
    model = self.model  # type: ignore[attr-defined]
    image = image.to(model.embeddings.patch_embeddings.weight.dtype)
    hidden_states = model.embeddings(image, bool_masked_pos=None)
    position_embeddings = model.rope_embeddings(image)
    layers = getattr(model, "layer", None)
    if layers is None and hasattr(model, "model"):
        layers = getattr(model.model, "layer", None)
    if layers is None:
        raise AttributeError(
            "Could not find DINOv3 transformer layers on model.layer or model.model.layer"
        )

    for layer_module in layers:
        hidden_states = layer_module(
            hidden_states,
            position_embeddings=position_embeddings,
        )
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]

    return F.layer_norm(hidden_states, hidden_states.shape[-1:])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images", type=Path, required=True, help="Directory with reference PNG/JPG images."
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Output directory for GLB assets."
    )
    parser.add_argument("--model", default="microsoft/TRELLIS.2-4B", help="Hugging Face model id.")
    parser.add_argument("--texture-size", type=int, default=4096)
    parser.add_argument("--decimation-target", type=int, default=1_000_000)
    parser.add_argument(
        "--make-video", action="store_true", help="Also export PBR preview MP4 files."
    )
    parser.add_argument(
        "--no-skip-existing", action="store_true", help="Regenerate GLBs even when output exists."
    )
    parser.add_argument(
        "--allow-border-alpha",
        action="store_true",
        help="Legacy fallback: estimate alpha from a solid border color when input has no alpha.",
    )
    parser.add_argument(
        "--envmap",
        type=Path,
        default=Path("TRELLIS.2/assets/hdri/forest.exr"),
        help="HDRI env map path relative to the working directory.",
    )
    return parser


def _image_paths(root: Path) -> list[Path]:
    paths: list[Path] = []
    for pattern in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        paths.extend(root.glob(pattern))
    return sorted(path for path in paths if not path.stem.startswith("qc_contact_sheet_"))


def _has_real_alpha(image: Image.Image) -> bool:
    image = image.convert("RGBA")
    alpha = np.array(image)[:, :, 3]
    return bool(np.any(alpha != 255))


def _with_border_alpha(image: Image.Image) -> Image.Image:
    image = image.convert("RGBA")
    rgb = np.array(image.convert("RGB"))
    h, w = rgb.shape[:2]
    border = max(4, min(h, w) // 32)
    samples = np.concatenate(
        [
            rgb[:border, :, :].reshape(-1, 3),
            rgb[-border:, :, :].reshape(-1, 3),
            rgb[:, :border, :].reshape(-1, 3),
            rgb[:, -border:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    bg = np.median(samples, axis=0)
    distance = np.linalg.norm(rgb.astype(np.float32) - bg.astype(np.float32), axis=2)
    mask = (distance > 28).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep = 1 + int(np.argmax(areas))
        mask = np.where(labels == keep, 255, 0).astype(np.uint8)

    output = np.dstack([rgb, mask])
    return Image.fromarray(output, mode="RGBA")


def _require_alpha(
    image: Image.Image, image_path: Path, *, allow_border_alpha: bool
) -> Image.Image:
    image = image.convert("RGBA")
    if _has_real_alpha(image):
        return image
    if allow_border_alpha:
        return _with_border_alpha(image)
    raise ValueError(
        f"{image_path} has no transparent alpha channel. "
        "Run trellis_b1/remove_image_background.py first and pass --images references_rgba, "
        "or use --allow-border-alpha for the old solid-background heuristic."
    )


def _load_runtime():
    global \
        cv2, \
        imageio, \
        np, \
        o_voxel, \
        torch, \
        F, \
        Image, \
        image_feature_extractor, \
        Trellis2ImageTo3DPipeline, \
        trellis_rembg, \
        EnvMap, \
        render_utils
    import cv2
    import imageio
    import numpy as np
    import o_voxel
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from trellis2.modules import image_feature_extractor
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    from trellis2.pipelines import rembg as trellis_rembg
    from trellis2.renderers import EnvMap
    from trellis2.utils import render_utils

    trellis_rembg.BiRefNet = PassthroughRembg
    image_feature_extractor.DinoV3FeatureExtractor.extract_features = (
        _dino_v3_extract_features_compat
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _load_runtime()
    args.output.mkdir(parents=True, exist_ok=True)
    images = _image_paths(args.images)
    if not images:
        raise ValueError(f"No images found in {args.images}")
    print(f"found {len(images)} input images in {args.images}", flush=True)

    print(f"loading envmap {args.envmap}", flush=True)
    envmap = EnvMap(
        torch.tensor(
            cv2.cvtColor(cv2.imread(str(args.envmap), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
            dtype=torch.float32,
            device="cuda",
        )
    )
    print(f"loading TRELLIS model {args.model}", flush=True)
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(args.model)
    pipeline.cuda()
    print("TRELLIS model loaded on cuda", flush=True)
    failures: list[dict[str, str]] = []
    failures_path = args.output / "failures.jsonl"
    failures_path.unlink(missing_ok=True)

    for image_path in images:
        stem = image_path.stem
        output_path = args.output / f"{stem}.glb"
        if output_path.exists() and output_path.stat().st_size > 0 and not args.no_skip_existing:
            print(f"skipped existing {stem}.glb")
            continue
        try:
            print(f"processing {stem}", flush=True)
            output_path.unlink(missing_ok=True)
            image = _require_alpha(
                Image.open(image_path),
                image_path,
                allow_border_alpha=args.allow_border_alpha,
            )
            mesh = pipeline.run(image)[0]
            mesh.simplify(16_777_216)
            if args.make_video:
                video = render_utils.make_pbr_vis_frames(
                    render_utils.render_video(mesh, envmap=envmap)
                )
                imageio.mimsave(args.output / f"{stem}.mp4", video, fps=15)
            glb = o_voxel.postprocess.to_glb(
                vertices=mesh.vertices,
                faces=mesh.faces,
                attr_volume=mesh.attrs,
                coords=mesh.coords,
                attr_layout=mesh.layout,
                voxel_size=mesh.voxel_size,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                decimation_target=args.decimation_target,
                texture_size=args.texture_size,
                remesh=True,
                remesh_band=1,
                remesh_project=0,
                verbose=True,
            )
            glb.export(output_path, extension_webp=False)
            print(f"generated {stem}.glb")
        except Exception as exc:  # noqa: BLE001 - keep long production batches moving.
            output_path.unlink(missing_ok=True)
            failure = {
                "id": stem,
                "image": str(image_path),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            failures.append(failure)
            with failures_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(failure, sort_keys=True) + "\n")
            print(f"failed {stem}: {type(exc).__name__}: {exc}", flush=True)
            torch.cuda.empty_cache()
    if failures:
        print(f"completed with {len(failures)} failed objects; see {failures_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
